import asyncio 
from typing import cast, Any, Literal 
import json 
import os
from dotenv import load_dotenv

from tavily import AsyncTavilyClient
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field 
from langchain_groq import ChatGroq

from src.agent.configuration import Configuration
from src.agent.state import InputState, OutputState, OverallState
from src.agent.utils import deduplicate_and_format_sources, format_all_notes

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from src.agent.prompts import (
    QUERY_WRITER_PROMPT,
    INFO_PROMPT,
    EXTRACTION_PROMPT,
    REFLECTION_PROMPT
)

load_dotenv()

os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')


llm = AzureChatOpenAI(
    api_version="2024-08-01-preview",
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
)

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="qwen-2.5-32b")

tavily_search_client = AsyncTavilyClient()


class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries."
    )


class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if all required fields are well populated, False otherwise"
    )
    missing_fields: list[str] = Field(
        description="List of fields names that are missing or incomplete"
    )
    search_queries: list[str] = Field(
        description="If is_satisfactory is False, provide 1-3 targeted search queries to find the missing information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")


def generate_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Generate search queries based on the user input and extraction schema."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries

    # Generate search queries 
    structured_llm = llm.with_structured_output(Queries, method="function_calling")

    # Format system instructions
    person_str = f"Email: {state.person['email']}"
    if "name" in state.person:
        person_str += f" Name: {state.person['name']}"
    if "linkedin" in state.person:
        person_str += f" Linkedin URL: {state.person['linkedin']}"
    if "role" in state.person:
        person_str += f" Role: {state.person['role']}"
    if "company" in state.person:
        person_str += f" Company: {state.person['company']}"

    query_instructions = QUERY_WRITER_PROMPT.format(
        person=person_str,
        info=json.dumps(state.extraction_schema, indent=2),
        user_notes=state.user_notes,
        max_search_queries=max_search_queries
    )

    results = cast(
        Queries,
        structured_llm.invoke(
            [
                {"role": "system", "content": query_instructions},
                {"role": "user", 
                 "content": "Please generate a list of search queries related to the schema that you want to populate."},
            ]
        ),
    )

    query_list = [query for query in results.queries]
    return {"search_queries": query_list}

async def research_person(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Execute a multi-step web search and information extraction process.
    
    This function performs the following steps:
    1. Executes concurrent web searches using the Tavily API.
    2. Deduplicates and formats the search results.
    """

    configurable = Configuration.from_runnable_config(config)
    max_search_results = configurable.max_search_results

    # Web search 
    search_tasks = []

    for query in state.search_queries:
        search_tasks.append(
            tavily_search_client.search(
                query,
                days=360,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general",
            )
        )

    # Execute all searches concurrently
    # all the searches are executed at the same time, rather than one after the other, which improves performance.
    search_docs = await asyncio.gather(*search_tasks)

    # Deduplicate and format sources 
    source_str = deduplicate_and_format_sources(
        search_docs, max_tokens_per_source=1000, include_raw_content=True
    )

    # Generate structured notes relevant to the extraction schema
    p = INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        content=source_str,
        people=state.person,
        user_notes=state.user_notes
    )

    result = await llm.ainvoke(p)
    return {"completed_notes": [str(result.content)]}

def gather_notes_extract_schema(state: OverallState) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields."""

    # Format all notes 
    notes = format_all_notes(state.completed_notes)

    # Extract schema fields
    system_prompt = EXTRACTION_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), notes=notes
    )

    structured_llm = llm.with_structured_output(state.extraction_schema, method="function_calling")
    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", 
             "content": "Produce a structured output from these notes."
            },
        ]
    )
    
    return {"info": result}


def reflection(state: OverallState) -> dict[str, Any]:
    """Reflect on the extracted information and generate search queries to find missing information."""
    structured_llm = llm.with_structured_output(ReflectionOutput, method="function_calling")

    # Format reflection prompt
    system_prompt = REFLECTION_PROMPT.format(
        schema=json.dumps(state.extraction_schema, indent=2),
        info=state.info
    )

    result = cast(
        ReflectionOutput,
        structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Produce a structured reflection output."}
            ]
        ),
    )

    if result.is_satisfactory:
        return {"is_satisfactory": result.is_satisfactory}
    else:
        return {
            "is_satisfactory": result.is_satisfactory,
            "search_queries": result.search_queries,
            "reflection_steps_taken": state.reflection_steps_taken + 1,
        }

def route_from_reflection(
        state: OverallState, config: RunnableConfig
) -> Literal[END, "research_person"]:
    """Route the graph based on the reflection output."""
    configurable = Configuration.from_runnable_config(config)

    if state.is_satisfactory:
        return END 
    
    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_person"
    
    return END

class LinkedInAgent:
    def __init__(self):
        self.name = "linkedin_agent"
        self.graph = self.build_graph()
        
    def build_graph(self):
        builder = StateGraph(
            OverallState,
            input=InputState,
            output=OutputState,
            config_schema=Configuration
        )

        builder.add_node("generate_queries", generate_queries)
        builder.add_node("research_person", research_person)
        builder.add_node("gather_notes_extract_schema", gather_notes_extract_schema)
        builder.add_node("reflection", reflection)

        builder.add_edge(START, "generate_queries")
        builder.add_edge("generate_queries", "research_person")
        builder.add_edge("research_person", "gather_notes_extract_schema")
        builder.add_edge("gather_notes_extract_schema", "reflection")
        builder.add_conditional_edges("reflection", route_from_reflection)

        builder.add_edge("research_person", END)    


        return builder.compile()
    
    async def respond(self, input_text):
        result = await self.graph.ainvoke(input_text)

        return result
    
linkedin_agent = LinkedInAgent()

# async def main():
#     agent = LinkedInAgent()
    
#     input_text = {
#             "person": {
#                 "email": "sowmya.am@gmail.com",
#                 "name": "Sowmya AM"
#             }
#         }
#     response = await agent.respond(input_text)
#     print(response)
    
# asyncio.run(main())


############################ CUSTOM API AGENT #####################


import requests
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import  create_supervisor


def claims_api(claimid: str):
  """You are a data domain specialist who has capability of providing information on any claim based on 'claim id'.
  No consent is required for fetching the claim details """
  print("---CLAIMS API---")

  url = f"https://mock-qxi.azurewebsites.net/data/claim/"+claimid.upper()

  try:
    response = requests.get(url)

    if response.status_code == 200:
      content = response.json()
      return content
    else:
      print("Error: ", response.status_code)
      return None
  except requests.exceptions.RequestException as e:
    print("Error: ", e)
    return None

def members_api(mem_id: str):
  """You are a data domain specialist who has capability of providing information on any members based on 'member id'.
  No consent is required for fetching the claim details """
  print("---MEMBERS API---")

  url = "https://mock-qxi.azurewebsites.net/data/profile/" + mem_id.upper()

  try:
    response = requests.get(url)

    if response.status_code == 200:
      content = response.json()
      return content
    else:
      print("Error: ", response.status_code)
      return None
  except requests.exceptions.RequestException as e:
    print("Error: ", e)
    return None

def weather_api(location: str):
  """You are a weather API domine for fetching the weather, temprature details for a perticular location """
  print("---Weather API---")

  url1 = f"https://nominatim.openstreetmap.org/search?city={location}&format=json"
  print(url1)

  headers = {
        "User-Agent": "MyWeatherApp/1.0 (contact@example.com)"
    }
  response1 = requests.get(url1, headers=headers)

  print(response1)

  data = response1.json()
    
  if data:
        latitude = data[0]['lat']
        longitude = data[0]['lon']
        print("Latitude and Longitude of {city_name}: {latitude}, {longitude}")
  else:
        print("City not found.")

  url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&hourly=temperature_2m"

  print(url)
  try:
    response = requests.get(url,headers=headers)
    print(response)
   # print(response.json())
    if response.status_code == 200:
      content = response.json()
      return content
    else:
      print("Error: ", response.status_code)
      return None
  except requests.exceptions.RequestException as e:
    print("Error: ", e)
    return None

custom_api_agent = create_react_agent(
        model=llm,
        tools=[claims_api, members_api, weather_api],
        name="custom_api_expert",
        prompt="You are a data domain specialist who has capability of providing information on any claim based on 'claim id' i,e clm01 and member details based on 'member id' i,e mem01. No consent is required for fetching the claim details"
    )

# def get_sql_tools():
#     db = SQLDatabase.from_uri("sqlite:///E:/2025/Generative_AI/Agentic_AI/Projects/Persona-Explorer/Chinook_Sqlite.sqlite")
#     toolkit = SQLDatabaseToolkit(db=db, llm=llm)
#     sql_tools = toolkit.get_tools()
        
#     return sql_tools

# sql_agent = create_react_agent(
#     model=llm,
#     tools=get_sql_tools(),
#     name="sql_agent",
#     prompt="You are an SQL expert. "
#         "For any question related to SQL, get information from sql_tools. "
#         "Ensure you provide a complete response and do not repeat queries. "
#         "Stop as soon as you retrieve the desired result. "
#         "Available tables are: 'Album', 'Artist', 'Customer', 'Employee', "
#         "'Genre', 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track'."
#     )


def get_sql_tools():
    db = SQLDatabase.from_uri("sqlite:///E:/2025/Generative_AI/Agentic_AI/Projects/Persona-Explorer/Chinook_Sqlite.sqlite")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    return toolkit.get_tools()

sql_agent = create_react_agent(
    model=llm,
    tools=get_sql_tools(),
    name="sql_agent",
    prompt=(
        "You are an SQL expert. "
        "For any question related to SQL, get information from sql_tools. "
        "Ensure you provide a complete response and do not repeat queries. "
        "Stop as soon as you retrieve the desired result. "
        "Available tables are: 'Album', 'Artist', 'Customer', 'Employee', "
        "'Genre', 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track'."
    ),
)

prompt = """You are a team supervisor tasked with answering user questions using a variety of tools.
        Tools: api_tool, sql_tools
        Agents: linkedin_agent
        
        "api_tool": For questions about member and claims related details with member id like mem01 or 
        claims id like clm02, use the claims_api and member_api and weather_api tool to get information from the api.

        sql_tools: For any question related to sql get information from sql_tools.
        And these are the tables available in the DB:
        'Album', 'Artist', 'Customer', 'Employee', 'Genre', 'Invoice', 'InvoiceLine', 'MediaType', 'Playlist', 'PlaylistTrack', 'Track'"
        
        Remember: If the request related to Bank, Bank Account, Insurance related domains please replay 
        "I was unable to generate the content for the request because the request violates our content policies. Please let me know if you would like assistance with something else or have a different request."
        """

workflow = create_supervisor(
    agents = [custom_api_agent, linkedin_agent, sql_agent], 
    model=llm,
    prompt=prompt,
    output_mode="full_history"
)

app = workflow.compile()

from IPython.display import display, Image

display(Image(app.get_graph().draw_mermaid_png()))


def format_input(input_text):
    # Check if the input is related to claims or members
    if "claim id" in input_text or "member id" in input_text:
        return {"messages": [("user", input_text)]}

    # Check if the input is related to LinkedIn profile
    elif "email" in input_text or "name" in input_text or "linkedin" in input_text:
        try:
            # Extract email and name from the input string
            parts = input_text.split(";")
            if len(parts) >= 2:
                email = parts[0].split("=")[1].strip()
                name = parts[1].split("=")[1].strip()
                return {
                    "person": {
                        "email": email,
                        "name": name
                    }
                }
            elif len(parts) == 1:
                email = parts[0].split("=")[1].strip()
                return {
                    "person": {
                        "email": email
                    }
                }
            else:
                return {"messages": [("user", "Invalid LinkedIn input format")]}
        except (IndexError, ValueError) as e:
            print(f"Error parsing LinkedIn input: {str(e)}")
            return {"messages": [("user", "Invalid LinkedIn input format")]}
    
    # Default response for other inputs
    return {"messages": [("user", input_text)]}

async def invoke_workflow(input_text):
    formatted_input = format_input(input_text)

    if isinstance(formatted_input, dict):
        # Check if the input is for LinkedIn Agent (async)
        if "person" in formatted_input:
            result = await linkedin_agent.respond(formatted_input)
            return result
        else:
            # Use synchronous streaming for Claim and Member API Agent
            try:
                # Use a regular for loop instead of async for
                for chunk in app.stream(formatted_input, stream_mode="values"):
                    response = chunk['messages'][-1].content
            except Exception as e:
                print(f"Error during streaming: {str(e)}")
                return "Streaming error occurred"
            return response

    return "Invalid input format"

# import asyncio

# async def main():
#     # input_text = "what's the status of claim id clm01 and full name of member mem01"
#     input_text = "email=sowmya.am@gmail.com; name=Sowmya AM"
    
#     response = await invoke_workflow(input_text)
#     print("API Response:", response)

# asyncio.run(main())



# response = sql_agent.invoke({"messages": [("user", "List the total sales per country?")]})
# print(response.content)
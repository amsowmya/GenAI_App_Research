from src.rag_agent.store_index import get_retriever
from typing import List, TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
import os


llm = AzureChatOpenAI(
    api_version="2024-08-01-preview",
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
)

# llm = ChatGroq(groq_api_key=groq_api_key, model_name="qwen-2.5-32b")

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def retrieve_documents(query: str) -> List: 
    """Function will retrieve the documents stored in the vector store.
    param: query
    return: List of documents
    """
    # faiss_vectorstore = get_faiss_vector_store()
    retriever = get_retriever()
    return retriever.invoke(query)


class grade(BaseModel):
  binary_score: str = Field(description="Relevance score 'yes' or 'no'")

def grade_documents(state: State) -> Literal["Output_Generator", "Query_Rewriter"]:
  print("---GRADE DOCUMENT---")
  llm_with_structured_op = llm.with_structured_output(grade)

  prompt = PromptTemplate(
      template = """You are a grader deciding if a document is relevant to a user's question.
      Here is the document: {context}
      Here is the user's question: {question}
      If the document talks about or contains information related to the user's question, mark it as relevant.
      Give a 'yes' or 'no' answer to show if the document is relevant to the question.
      """,
      input_variables = ["context", "question"]
  )
  chain = prompt | llm_with_structured_op

  messages = state['messages']
  last_message = messages[-1]
  question = messages[0].content
  docs = last_message.content

  # chroma_vectorstore = get_chroma_vector_store
  # retriever = chroma_vectorstore.as_retriever()
  # docs = retriever.invoke(question)

  print("DOC *****************************************: ", docs)

  scored_result = chain.invoke({"question": question, "context": docs})
  score = scored_result.binary_score

  if score == 'yes':
    print("---DECISION: DOCS RELEVANT---")
    # return "generator"
    return "Output_Generator"
  else:
    print("---DECISION: DOCS NOT RELEVANT---")
    # return "rewrite"
    return "Query_Rewriter"
  
def generate(state: State):
  print("---GENERATE---")
  messages = state['messages']

  question = messages[0].content

  last_message = messages[-1]
  docs = last_message.content

  prompt = hub.pull("rlm/rag-prompt")

  rag_chain = prompt | llm

  response = rag_chain.invoke({"context": docs, "question": question})
  print(f"this is my response: {response}")

  return {"messages": [response]}


def rewrite(state: State):
  print("---TRANSFORM QUERY---")
  messages = state['messages']
  question = messages[0].content

  message = [HumanMessage(content=f"""Look at the input and try to reason about the underlying semantic intent or meaning.
                  Here is the initial question: {question}
                  Formulate an improved question just in single line quiestion.""")
  ]

  response = llm.invoke(message)
  # response = "content='Why we need agent framework?' additional_kwargs={} response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 79, 'total_tokens': 100, 'completion_time': 0.076363636, 'prompt_time': 0.012099538, 'queue_time': 0.157481545, 'total_time': 0.088463174}, 'model_name': 'llama-3.3-70b-versatile', 'system_fingerprint': 'fp_4196e754db', 'finish_reason': 'stop', 'logprobs': None} id='run-10df125a-a327-4629-902a-7c25b54e5331-0' usage_metadata={'input_tokens': 79, 'output_tokens': 21, 'total_tokens': 100}"
  print("*********** NEW ques: ", response)
  return {"messages": [response]}


builder = StateGraph(State)

builder.add_node("retrieve_tools", ToolNode([retrieve_documents]))
builder.add_node("Output_Generator", generate)
builder.add_node("Query_Rewriter", rewrite)

builder.add_edge(START, "retrieve_tools")
builder.add_conditional_edges(
    "retrieve_tools",
    grade_documents,
    ["Output_Generator", "Query_Rewriter"]
)

builder.add_edge("Query_Rewriter", "assistant")

builder.add_edge("Output_Generator", END)

app = builder.compile()


result = app.invoke({"messages": [("user", "what is self attention")]})

print(result)
import streamlit as st 
import asyncio
from langchain_core.messages import AIMessage, HumanMessage
from src.agent.graph import invoke_workflow
from src.utils.helper import text_to_speech


async def stream_output(user_input: str):
    response = await invoke_workflow(user_input)
    response = flatten_response(response)
    print("RESPONSE: ", response)
    # Ensure the response is a plain string before returning
    if asyncio.iscoroutine(response):
        response = await response
    return str(response)

def flatten_response(response):
    if isinstance(response, dict):
        # Join key-value pairs into a single string
        return "; ".join(f"{k}: {v}" for k, v in response.get("info", {}).items())
    return str(response)

# response = await invoke_workflow(user_input)
# flattened_response = flatten_response(response)
# print("Flattened Response:", flattened_response)

def get_audio_bytes(response):
    try:
        asyncio.run(text_to_speech(response))
        with open("speech.mp3", "rb") as audio_file:
            audio_bytes = audio_file.read()
        return audio_bytes
    except Exception as e:
        print(f"Error fetching audio bytes: {str(e)}")
        return None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello!.. I am a multi-agent AI. How can I help you?") 
    ]
    
st.title("Multi Agents AI App with Voice Assistant 🤖")

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
            
user_query = st.chat_input("Type a message...", key="user_input")

print("USER_INPUT: ",user_query)

# if user_query is not None and user_query.strip() != "":
if user_query is not None and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        response = asyncio.run(stream_output(user_query))
        response = flatten_response(response)  # Ensure response is a plain string
        st.markdown(response)

        # Convert response to audio
        audio_bytes = get_audio_bytes(response)
        if audio_bytes:
            st.audio(audio_bytes)
        
        # Append AI response to chat history
        st.session_state.chat_history.append(AIMessage(content=response))
    
    
    

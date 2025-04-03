import streamlit as st 
import asyncio
from langchain_core.messages import AIMessage, HumanMessage
from src.agent.graph import invoke_workflow
from src.utils.helper import text_to_speech


async def stream_output(user_input: str):
    response = await invoke_workflow(user_input)
    print("RESPONSE: ", response)
    return response

# def get_audio_bytes(response):
#     text_to_speech(response)
#     audio_file = open("speech.mp3", "rb")
#     audio_bytes = audio_file.read()
#     return audio_bytes

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
        response = stream_output(user_query)
        st.markdown(response)

        audio_bytes = get_audio_bytes(response)
        st.audio(audio_bytes)
        
        # st.download_button(
        #     label="Download Speech",
        #     data = audio_bytes,
        #     file_name="speech.mp3",
        #     mime="audio/mp3"
        # )

    st.session_state.chat_history.append(AIMessage(content=response))
    
    
    

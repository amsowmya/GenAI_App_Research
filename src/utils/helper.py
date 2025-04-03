import os
import asyncio
from gtts import gTTS
import google.generativeai as genai
import speech_recognition as sr

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

def voice_input():
    r = sr.Recognizer() 
    
    with sr.Microphone() as source:
        print("Listening..")
        audio = r.listen(source)
        
        try:
            text = r.recognize_google(audio)
            print("You said: ", text)
            return text
        except sr.UnknownValueError:
            print("Sorry, could not understand the audio")
        except sr.RequestError as e:
            print("Could not request result from google speech recognition service: {0}".format(e))

def llm_model(user_text):
    genai.configure(api_key=GOOGLE_API_KEY)
    
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(user_text)
    result = response.text
    return result

# def text_to_speech(text):
#     tts = gTTS(text=text, lang="en")
    
#     # save the speech from the given text in the mp3 format
#     tts.save("speech.mp3")
    
# from gtts import gTTS
# import asyncio
        
async def text_to_speech(text):
    # Handle async coroutines and convert them to strings
    if asyncio.iscoroutine(text):
        text = await text

    # Check if the response is a dictionary or object
    if isinstance(text, dict):
        # Convert dictionary to a human-readable string
        text = str(text)

    try:
        # Convert text to speech
        tts = gTTS(text=text, lang='en')
        tts.save("speech.mp3")
        print("Audio saved as speech.mp3")
    except Exception as e:
        print(f"Error during text-to-speech conversion: {str(e)}")
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader

from typing import List

import os

FAISS_PERSIST_DIRECTORY = "faiss_index"
DATA_FILE_PATH = "E:\\2025\\Generative_AI\\Agentic_AI\\Projects\\GenAI_App_Research\\data"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

hugging_face_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

def load_pdf_file(data):
    loader = PyPDFDirectoryLoader(data)
    
    documents = loader.load()
    return documents

def text_split(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def get_text_chunks():
    extracted_data = load_pdf_file(DATA_FILE_PATH)
    text_chunks = text_split(extracted_data)
    return text_chunks

# FAISS for bedrock embeddings
def create_faiss_vector_store():
    print("Creating FAISS Vector Store...")
    text_chunks = get_text_chunks()
    vectorstore_faiss = FAISS.from_documents(documents=text_chunks, 
                                             embedding=hugging_face_embeddings)
    vectorstore_faiss.save_local(FAISS_PERSIST_DIRECTORY)
    
def get_faiss_vector_store():
    faiss_index = FAISS.load_local(FAISS_PERSIST_DIRECTORY, 
                                   embeddings=hugging_face_embeddings, 
                                   allow_dangerous_deserialization=True)
    return faiss_index

if not os.path.exists(FAISS_PERSIST_DIRECTORY):
    os.makedirs(FAISS_PERSIST_DIRECTORY)

if not os.listdir(FAISS_PERSIST_DIRECTORY):
    print("Creating FAISS Vector Store...")
    create_faiss_vector_store()
else:
    print("FAISS Vector store is already exist...") 
    
    
def get_retriever():
    # chroma_vectorstore = get_chroma_vector_store()
    # retriever = chroma_vectorstore.as_retriever()

    faiss_vectorstore = get_faiss_vector_store()
    retriever = faiss_vectorstore.as_retriever()
    return retriever
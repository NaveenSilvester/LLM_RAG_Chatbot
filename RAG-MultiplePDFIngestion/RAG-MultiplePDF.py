""" Langchain based LLM RAG Chatbot based on Ollma's tinyllama. 
The RAG pipeline has should have the ability to digest multiple PDF ingestion
 from multiple folders and embeddings are done using FastEmbedEmbeddings and 
 a FAISS vector database
"""
# Requirements
# pip install langchain faiss-cpu fastembed pypdf ollama
# pip install -U langchain-community
# pip install -U langchain-ollama


## Importing Credentials and Secrets from venv
##################################################################
## This code snippet sets up environment variables to enable LangSmith 
## tracing for LangChain applications.
##################################################################
# os: Standard Python library for interacting with the operating system.
# load_dotenv(): Loads variables from a .env file into your environment 
# so you can access them via os.getenv().
# load_dotenv() Activates the .env loader so anything defined in the .env file 
# (like your API keys or config flags) gets pushed into the environment.
# os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# Reads LANGCHAIN_API_KEY from the environment (or .env file), and explicitly sets 
# it in os.environ. 
# This makes sure the LangChain or LangSmith client can access your API key reliably.
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# Enables advanced tracing for LangSmith V2, allowing you to monitor and debug your 
# LangChain app behavior (like LLM calls, retrieval steps, etc.) via LangSmith's UI.

print("#################################################################################")
print        ("Preparation and Setting up Credentials and Environment variables")
print("#################################################################################")
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

print("#################################################################################")
print(         "Preparation and Setting up Credentials and Environment variables")
print("                                 COMPLETE-STAGE-0")
print("#################################################################################")


print("#################################################################################")
print("                                    STAGE-1")
print("                             Loading PDF files from Folder")
print("#################################################################################")
from langchain_community.document_loaders import PyPDFLoader
#from langchain.document_loaders import PyPDFLoader
from pathlib import Path

def load_pdfs_from_folders(folder_paths):
    docs = []
    for folder in folder_paths:
        pdf_paths = Path(folder).glob("*.pdf")
        for path in pdf_paths:
           # print ("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$", path)
            loader = PyPDFLoader(str(path))
            docs.extend(loader.load())
    return docs

print("#################################################################################")
print("                                 COMPLETE-STAGE-1")
print("#################################################################################")

print("#################################################################################")
print("                                    STAGE-2")
print("                             Embeddings with FastEmbed")
print("#################################################################################")

from langchain_community.embeddings import FastEmbedEmbeddings
#from langchain.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
#from langchain.vectorstores import FAISS

def create_vector_db(docs):
    embeddings = FastEmbedEmbeddings()
    vector_db = FAISS.from_documents(docs, embedding=embeddings)
    return vector_db

print("#################################################################################")
print("                                 COMPLETE-STAGE-2")
print("#################################################################################")


print("#################################################################################")
print("                                    STAGE-3")
print("                         Setting Up TinyLlama via Ollama")
print("#################################################################################")
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="tinyllama")

# from langchain.llms import Ollama
# llm = Ollama(model="tinyllama")
print("#################################################################################")
print("                                 COMPLETE-STAGE-3")
print("#################################################################################")

print("#################################################################################")
print("                                    STAGE-4")
print("                         Creating a Retriever")
print("#################################################################################")
from langchain.chains import RetrievalQA

def create_rag_chain(llm, vector_db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_db.as_retriever(),
        chain_type="stuff"
    )
print("#################################################################################")
print("                                 COMPLETE-STAGE-4")
print("#################################################################################")

print("#################################################################################")
print("                                    STAGE-5")
print("                         Query and Interact with ChatBot")
print("#################################################################################")
from langchain.chains import RetrievalQA

def run_chatbot(folders):
    docs = load_pdfs_from_folders(folders)
    print("Done Loading of ", docs)
    db = create_vector_db(docs)
    print("Done DB")
    qa_chain = create_rag_chain(llm, db)
    print("Query DB with LLM")

    while True:
        query = input("Ask your question: ")
        if query.lower() in ["exit", "quit"]:
            break
        response = qa_chain.invoke(query)
        print(f"Answer: {response}")

run_chatbot("./")
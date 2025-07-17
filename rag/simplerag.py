## RAG Pipeline with Vector Database FAISS DB


## Importing Credentials and Secrets from venv
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

"""
## Data Ingestion  - Tetxt File
from langchain_community.document_loaders import TextLoader
loader=TextLoader("speech.txt") 
docs= loader.load()
print("###########################################\n")
print("Data Ingestion output from Text")
print("###########################################\n")

print(docs)

print("###########################################\n")
"""

"""
## Data Ingestion  - Website
from langchain_community.document_loaders import WebBaseLoader
import bs4
loader=WebBaseLoader(web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
                     bs_kwargs=dict(parse_only=bs4.SoupStrainer(
                         class_=("post-title", "post-content", "post-header")
                     )))

docsw = loader.load()
print("###########################################\n")
print("Data Ingestion output from WebSite")
print("###########################################\n")
print(docsw)

"""

#"""
## Data Ingestion  - PDFs
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('attention.pdf')
docs=loader.load()
print("###########################################\n")
print("STEP-1: Data Ingestion output from PDF")
print("###########################################\n")
#print(docs)
print("STEP-1: Ingestion Complete")
print("###########################################\n")

#"""

## Data Chunking using RecursiveCharacterTextSplitter
print("###########################################\n")
print("STEP-2: Data Chunking from PDF ingestion documents")
print("###########################################\n")

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunk_documents=text_splitter.split_documents(docs)
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
print(f"Number of Chunks:  {len(chunk_documents)}")
#print(chunk_documents)
print("STEP-2: Chunking Complete (ChunkSize = 1000 and overlap = 200 characters)")
print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

## Vector Embedding and Vector Storing
print("###########################################\n")
print("STEP-3: Embedding Chunked Documents using langchain_community.embeddings")

# OllamaEmbeddings
#from langchain_community.embeddings import OllamaEmbeddings
#db=FAISS.from_documents(chunk_documents,OllamaEmbeddings())

# FastEmbedEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
print ("STEP-3: Completed Embedding")
print("###########################################\n")

print("###########################################\n")
print("STEP-4: Creation of Vectorstore using FAISS (Facebook AI Similarity Search)")
from langchain_community.vectorstores import FAISS
db=FAISS.from_documents(chunk_documents,FastEmbedEmbeddings())
print ("STEP-4: Complete creation of vectorstore using FAISS")
print("###########################################\n")


## Query DB
print("###########################################\n")
print("STEP-5: Query Vectorstore")
print("###########################################\n")
query="The encoder is composed of a stack of N = 6 identical layers"
retrieved_result=db.similarity_search(query)
print("Here is the Result from RAG Query:\n")
print(retrieved_result[0].page_content)
print("###########################################\n")

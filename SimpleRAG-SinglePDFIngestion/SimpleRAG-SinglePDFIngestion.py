""" Write an Langchain based LLM RAG Chatbot based on Ollma's tinyllama. 
The RAG pipeline has a PDF ingestion and embeddings are done using FastEmbedEmbeddings and a FAISS vector database
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
print("                             Loading the LLM Model")
print("#################################################################################")

# Step 1: Load the LLM (TinyLlama via Ollama)
from langchain_ollama import OllamaLLM
llm = OllamaLLM(model="tinyllama")

print("#################################################################################")
print("                                 COMPLETE-STAGE-1")
print("#################################################################################")


print("#################################################################################")
print("                                    STAGE-2")
print("                             Loading and Splitting PDF Document")
print("#################################################################################")
# Step 2: Load and Split PDF Document
# Loads Naveen.pdf into a list of Document objects.
# loader = PyPDFLoader("Naveen.pdf")
# Each document typically contains text and metadata (e.g., page number).
# splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# docs = splitter.split_documents(documents)
# Uses RecursiveCharacterTextSplitter to split each document into smaller chunks:
# chunk_size=500: max characters per chunk.
# chunk_overlap=50: 50-character overlap between chunks to preserve context across boundaries.
# This process makes your text more manageable for embeddings and search indexing.

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
loader = PyPDFLoader("Naveen.pdf")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)
print("#################################################################################")
print("                                 COMPLETE-STAGE-2")
print("#################################################################################")


print("#################################################################################")
print("                                    STAGE-3")
print("                        Generating Embedding using FastEmbed")
print("#################################################################################")
# Step 3: Generate Embeddings Using FastEmbed
# The snippet sets up a vector embedding pipeline using LangChain’s FastEmbedEmbeddings, 
# which is a lightweight, efficient model for turning text into high-dimensional 
# vectors—ideal for semantic search and RAG workflows.
## from langchain_community.embeddings import FastEmbedEmbeddings
# Pulls in the FastEmbedEmbeddings class from LangChain’s community module.
# It’s optimized for speed and local execution, often backed by fast vector libraries.
## embedding_model = FastEmbedEmbeddings()
# Instantiates the embedding model with default settings.
# You can configure it with parameters (e.g. model name or device) if needed—though it 
# works well out-of-the-box for most use cases.
## doc_embeddings = embedding_model.embed_documents([doc.page_content for doc in docs])
# Extracts the raw text (page_content) from each chunk in your docs list.
# Passes the list of texts to embed_documents, which returns a list of embeddings - 
# each one is a dense vector (e.g. 384-dimensional).
# These vectors are now ready to be indexed into a vector store like FAISS or 
# Chroma for similarity search.

#from langchain.embeddings import FastEmbedEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings

embedding_model = FastEmbedEmbeddings()
doc_embeddings = embedding_model.embed_documents([doc.page_content for doc in docs])
print("#################################################################################")
print("                                 COMPLETE-STAGE-3")
print("#################################################################################")



print("#################################################################################")
print("                                    STAGE-4")
print("                    Store Embeddings in FAISS Vector Database")
print("#################################################################################")

# Step 4: Store Embeddings in FAISS Vector Database
#from langchain.vectorstores import FAISS
# The below code snippet wires up your docs into FAISS, turning plain text chunks into 
# a searchable vector index
## from langchain_community.vectorstores import FAISS
# Brings in the FAISS integration from LangChain’s community module.
# FAISS is a high-performance, open-source library for efficient similarity search over 
# dense vectors.
## faiss_index = FAISS.from_texts([doc.page_content for doc in docs], embedding_model)
# Here’s what’s happening:
# [doc.page_content for doc in docs]: Extracts the raw text from each chunk. These are your inputs for embedding.
# embedding_model: Should be an instance like FastEmbedEmbeddings() or similar, already configured.
# FAISS.from_texts(...):
# Embeds each text chunk using the provided model.
# Creates a FAISS index internally using those embeddings.
# Returns a LangChain-compatible FAISS object (faiss_index) that you can query with semantic similarity.

from langchain_community.vectorstores import FAISS
faiss_index = FAISS.from_texts([doc.page_content for doc in docs], embedding_model)
print("#################################################################################")
print("                                 COMPLETE-STAGE-4")
print("#################################################################################")


print("#################################################################################")
print("                                    STAGE-5")
print("                             Creating RetrievalQA Chain")
print("#################################################################################")
# Step 5: Create RetrievalQA Chain
# This code snippet is the culmination of a retrieval-based question-answering pipeline—your 
# RAG system is officially coming alive.
## from langchain.chains import RetrievalQA
# Imports LangChain’s high-level wrapper for retrieval-augmented question answering.
# RetrievalQA orchestrates interaction between a language model (LLM) and retriever 
# (typically backed by a vector store like FAISS).
""" rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=faiss_index.as_retriever()
)
"""
# This sets up the actual RAG chain, with three major components:
## llm=llm
# Refers to your configured large language model (could be OpenAI, Anthropic, HuggingFace,Ollama etc.).
# The LLM is responsible for generating natural language answers.
## chain_type="stuff"
# LangChain supports different ways of passing retrieved documents to the LLM.
## "stuff": Simple—concatenates all retrieved docs and "stuffs" them into the prompt.
# Other options: "map_reduce", "refine" or "map_rerank" for more structured processing.
# "stuff" is best when your chunks are compact (e.g., ≤ 500 tokens) and your prompt budget allows.
## retriever=faiss_index.as_retriever()
# Converts your FAISS vector store into a retriever interface.
# Enables semantic lookup: when a query is made, it fetches the most similar document chunks.


from langchain.chains import RetrievalQA
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=faiss_index.as_retriever()
)
print("#################################################################################")
print("                                 COMPLETE-STAGE-5")
print("#################################################################################")


print("#################################################################################")
print("                                    STAGE-6")
print("                             Run Chatbot with Query")
print("#################################################################################")
# Step 6: Run Chatbot
# This snippet is the final act of your Retrieval-Augmented Generation (RAG) pipeline.
# it triggers the system to fetch a context-rich, LLM-generated answer using your embedded 
# document knowledge base. Here's the breakdown:
# query = "Can you tell me something about Naveen Silvester's Industry Experience."
# This is a natural language question you're feeding into the system.
# The model will search the FAISS index to find semantically relevant chunks of your 
# PDF documents.
# In your setup, likely aimed at extracting insights from a document like a CV or 
# technical paper that references Naveen Silvester.
# Invoke the RAG Chain
## response = rag_chain.invoke(query)
# This sends the query to your RetrievalQA chain or ConversationalRetrievalChain 
# (depending on how you wired it).
# invoke triggers:
# Retriever → Semantic search in FAISS for top-k relevant chunks.
# LLM → Constructs a well-formed answer using those retrieved chunks.
# Display the Response
# print(response)
# Finally, you print the answer generated by the LLM.

query = "Can you tell me something about Naveen Silvester's Industry Experience."
response = rag_chain.invoke(query)
print(response)
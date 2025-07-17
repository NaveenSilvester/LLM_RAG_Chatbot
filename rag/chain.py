#######################################################################
##      A RAG Chatbot using LLM 
##    (End-to-End Implementation)
#######################################################################

## Retriever and Chain with Langchain


## Importing Credentials and Secrets from venv
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

#"""
## Data Ingestion  - PDFs
print("###########################################")
print("STEP-1: Data Ingestion from PDF")
print("###########################################\n")
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader('attention.pdf')
docs=loader.load()
#print(docs)
print("###########################################")
print("STEP-1: Ingestion Complete")
print(f"Total Number of Documents {len(docs)}")
print("###########################################\n")

#"""


## Data Chunking using RecursiveCharacterTextSplitter
print("###################################################")
print("STEP-2: Data Chunking from PDF ingestion documents")
print("###################################################\n")
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunk_documents=text_splitter.split_documents(docs)
print("###################################################")
print("STEP-2: Chunking Complete \n(ChunkSize = 1000 and overlap = 200 characters)")
print(f"Number of Chunks:  {len(chunk_documents)}")
#print(chunk_documents)
print("###################################################\n")





## Vector Embedding and Vector Storing
print("#######################################################")
print("STEP-3: Embedding Chunked Documents using \n(FastEmbeddings) from langchain_community.embeddings")
print("#######################################################\n")
# OllamaEmbeddings
#from langchain_community.embeddings import OllamaEmbeddings
#db=FAISS.from_documents(chunk_documents,OllamaEmbeddings())

# FastEmbedEmbeddings
from langchain_community.embeddings import FastEmbedEmbeddings
print("#######################################################")
print ("STEP-3: Completed Embedding")
print("#######################################################\n")


print("###########################################################")
print("STEP-4: Creation of Vectorstore using FAISS \n(Facebook AI Similarity Search)")
from langchain_community.vectorstores import FAISS
db=FAISS.from_documents(chunk_documents,FastEmbedEmbeddings())
print("###########################################################\n")

print("###########################################################")
print ("STEP-4: Complete creation of vectorstore using FAISS")
print("###########################################################\n")


## Query DB
print("###############################################################")
print("STEP-5: Query Vectorstore")
print("###############################################################\n")
query="The encoder is composed of a stack of N = 6 identical layers"
retrieved_result=db.similarity_search(query)
print("Here is the Result from RAG Query:\n")
print(retrieved_result[0].page_content)
print("###############################################################\n")

## Designing Template
print("###################################################################")
print("STEP-6: Designing Prompt Template")
print("###################################################################\n")
from langchain_core.prompts import ChatPromptTemplate
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer. 
I will tip you $1000 if the user finds the answer helpful.
<context>
{context}
</context>

Question:{input}
                                          
""")
print("###################################################################")
print("STEP-6: Comlpeted Designing Prompt Template")
print("###################################################################\n")

## Using LMM model
print("###################################################################")
print("STEP-7: Using LLM Model")
print("###################################################################\n")
from langchain_community.llms import Ollama
llm=Ollama(model="tinyllama")

#llm=Ollama(model="gemma:2b")
#llm=Ollama(model="DistilGPT-2")
#output_parser=StrOutputParser()
print("###################################################################")
print("STEP-7: Complete setting up LLM Model")
print("###################################################################\n")


## Chain
print("###################################################################")
print("STEP-8: Creating Document Chain with Model and Prompt")
print("###################################################################\n")
from langchain.chains.combine_documents import create_stuff_documents_chain
documents_chain=create_stuff_documents_chain(llm,prompt)
print("###################################################################")
print("STEP-8: Complete Chain with Model and Prompt")
print("###################################################################\n")

## Retriever
print("###################################################################")
print("STEP-9: Creating a Retriever")
print("###################################################################\n")

"""
Retrievers: A retriever is an inerface that returns documents given an unstructured query.
It is more generatl than a vector store. A retriever does not need to be able to store
documents, nly o return (or retrieve) them. Vector stores can be sued as a backbone 
of a retriever, but herea re oter types of retrievers as well.
https://python.langchain.com/docs/modules/data_connection/retrievers/
"""
retriever=db.as_retriever()
print("###################################################################")
print("STEP-9: Complted creating a Retriever")
print("###################################################################\n")


# Retrieval Chain
print("###################################################################")
print("STEP-10: Creating a Retrieval Chain")
print("###################################################################\n")
"""
Retrieval chain: This chain takes in a user inquiry, which is then passed to the
retriever to fetch relevant documents. Those documents (and original inputs) are 
then passed to an LLM to generate a response
https://python.langchain.com/docs/modules/chains/
"""
from langchain.chains import create_retrieval_chain
retrieval_chain=create_retrieval_chain(retriever,documents_chain)
print("###################################################################")
print("STEP-10: Completed creating a Retrieval Chain")
print("###################################################################\n")


# Invoke response to the input
print("###################################################################")
print("FINAL STEP: Invoke the Chatbot response with user input")
print("###################################################################\n")
print("INPUT: Describe Attention Function")
#response=retrieval_chain.invoke({"input":"Write a Python code to read text file"})
response=retrieval_chain.invoke({"input":"Describe Attention Function"})
response['answer']
print (response['answer'])
print("###################################################################")
print("-------------------Complete---------------------")
print("###################################################################\n")
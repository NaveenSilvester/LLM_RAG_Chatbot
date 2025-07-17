#LANGCHAIN_API_KEY="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"
#OPENAI_API_KEY=""
#LANGCHAIN_PROJECT="Tutorial"

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.output_parsers import StrOutParser

from langchain_core.output_parsers.string import StrOutputParser
#from langchain_core.output_parsers.string import StrOutputParser
from langchain_community.llms import Ollama 

import streamlit as st
import os 
from dotenv import load_dotenv
load_dotenv()

## environmental variable call

#os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

## Langsmit tracking

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

## Creating Chatbot
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful asistant. Please provide response to the user prompt"),
        ("user", "Question:{question}")
    ]
)

# streamlit framework

st.title("Langchain Demo with LLama2-Gemma:2b API")
input_text=st.text_input("Search the topic you want")

# Open AI LLM Call
llm=Ollama(model="gemma:2b")
output_parser=StrOutputParser()

## Chain
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
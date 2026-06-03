from langchain_community.document_loaders import TextLoader
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSequence
from pydantic import BaseModel, Field
from typing import Literal
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
mistral_key = os.getenv("MISTRAL_API_KEY")

# Initialize models
model = ChatMistralAI(
    api_key=mistral_key,
    model_name="mistral-tiny",
    temperature=0.3, 
    max_tokens=100       
)

post_prompt = PromptTemplate(
    template="Generate the summary of the document:\n {topic}",
    input_variables=["topic"],
)

parser = StrOutputParser()

# Option 1: Relative path (recommended)
loader = TextLoader("textloaderfile.txt", encoding="utf-8")

# Option 2: Use absolute path (more reliable)
# loader = TextLoader("/home/xostar/Desktop/LangChain Learning Flow/DocumentLoader/textloaderfile.txt", encoding="utf-8")

documents = loader.load()

chain =  post_prompt |  model | parser


print(f"Successfully loaded {len(documents)} document(s)\n")
# print(documents[0].page_content[:100])
print(type(documents))

chain =  post_prompt |  model | parser
result = chain.invoke({"topic": documents[0].page_content})
print("Summary of the document:\n", result)
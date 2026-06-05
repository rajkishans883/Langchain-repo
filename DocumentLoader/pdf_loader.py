from langchain_community.document_loaders import PyPDFLoader
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

loader= PyPDFLoader("output.pdf")

doc=loader.load()
print(len(doc))

print(doc[0].page_content)
print(doc[0].metadata)


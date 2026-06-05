from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,WebBaseLoader
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSequence
from pydantic import BaseModel, Field
from typing import Literal
from bs4 import BeautifulSoup
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

url='https://www.amazon.in/s?k=sneakers+for+man'
loader= WebBaseLoader(url)

doc=loader.load()

print(len(doc))


print(doc[0].page_content)

prompt = PromptTemplate(
    template="give me you opinio on your overervation of the data that which shoose in best in budget:\n {data_amazon}",
    input_variables=["data_amazon"],
)
parser = StrOutputParser()

chain= prompt | model | parser

result = chain.invoke({"data_amazon": doc[0].page_content})
print("Summary of the document:\n", result)
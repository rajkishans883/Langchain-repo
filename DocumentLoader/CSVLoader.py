from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,CSVLoader
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSequence
from pydantic import BaseModel, Field
from typing import Literal
import os
from dotenv import load_dotenv

data = CSVLoader(file_path="csvfile_.csv").load()

print(len(data))

print(data[0].page_content)

prompt = PromptTemplate(
    template="give me you opinio on your overervation of the data that which shoose in best in budget:\n {data_amazon}",
    input_variables=["data_amazon"],
)
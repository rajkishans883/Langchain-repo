from langchain_mistralai import ChatMistralAI
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from typing import Literal
import os
from dotenv import load_dotenv

load_dotenv()

mistral_key=os.getenv("MISTRAL_API_KEY")

model=ChatMistralAI(
    api_key=mistral_key,    
    model_name="mistral-tiny",
    temperature=0.9,        
)
prompt = PromptTemplate(
    template="create a funny joke about{topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="explain the joke {funny_joke}",
    input_variables=["funny_joke"],
)
parser = StrOutputParser()

chain = RunnableSequence(prompt, model , parser , prompt2 , model , parser)

print(chain.invoke({"topic": "cats"}))

from langchain_mistralai import ChatMistralAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from pydantic import BaseModel, Field
from typing import Literal
import os
from dotenv import load_dotenv

load_dotenv()

mistral_key=os.getenv("MISTRAL_API_KEY")

model=ChatMistralAI(
    api_key=mistral_key,    
    model_name="mistral-tiny",
    temperature=0.3,        
)

prompt = PromptTemplate(
    template="generate a tweet about {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="generate a linkedin post about this {topic}",
    input_variables=["topic"],
)
parser1 = StrOutputParser()
parser2 = StrOutputParser()

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash",
    task="text-generation"
)

model2 = ChatHuggingFace(
    llm=llm ,
    max_tokens=1000,
)

parallel_chain = RunnableParallel(
   {
       "generate_tweet": RunnableSequence(prompt, model , parser1 ),
       "generate_post": RunnableSequence(prompt2, model2, parser2)
   }
)

result=(parallel_chain.invoke({"topic": "Ai is dangours or not"}))

print(result.get("generate_tweet"))
print(result.get("generate_post"))
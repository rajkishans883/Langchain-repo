from langchain_mistralai import ChatMistralAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel,RunnableBranch
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
    template="generate a detailed report about {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template="summerize the topic under 300 words topic:{topic}",
    input_variables=["topic"],
)   
parser1 = StrOutputParser()
parser2 = StrOutputParser()

report_chain = RunnableSequence(prompt, model , parser1 )
summary_chain = RunnableSequence(prompt2, model , parser2 )



branch_chain = RunnableBranch(
    (lambda x:len(x.split())>500,summary_chain),
    RunnablePassthrough()
)
final_chain=RunnableSequence(report_chain, branch_chain)

result=final_chain.invoke({"topic": "AI is dangerous or not"})
print(result)
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

mistral_key=os.getenv("MISTRAL_API_KEY")

model=ChatMistralAI(
    api_key=mistral_key,    
    model_name="mistral-tiny",
    temperature=0.5,
    streaming=True,         
)

template= PromptTemplate(
    template="provice use five intresting and less known facts about {topic}",
    input_variables=["topic"],
)


parser= StrOutputParser()

chain = template | model| parser

result_chain = chain.invoke({"topic":"conner macgregor"})

print(result_chain)
print(type(result_chain))

print(chain.get_graph().draw_ascii())
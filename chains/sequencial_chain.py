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

prompt1 = PromptTemplate(
    template="provide me details fact and information about {topic}",
    input_variables=["topic"],
)

prompt2= PromptTemplate(
    template="summarize the following information in a concise manner: {information}",
    input_variables=["information"]
)

parser=StrOutputParser()

chain =prompt1 | model | parser |prompt2 | model|parser

chain_result=chain.invoke({"topic":"conner macgregor"})

print(chain_result)

print(chain.get_graph().draw_ascii())
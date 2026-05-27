from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
import os
from pydantic import BaseModel, Field

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Pro",
    task="text-generation"
)

model = ChatHuggingFace(
    llm=llm ,
    streaming=True,
    max_tokens=1000,
)

class Person(BaseModel):
    name:str=Field(description="The name of the person ")
    age:int=Field(gt=10,lt=100,description="The age of the person")
    city:str=Field(description="The city of the person")
    country:str=Field(description="The country of the person")

parser=PydanticOutputParser(pydantic_object=Person)
prompt=PromptTemplate(
    template="Generate a random person's name,age details where he belong to {city} \n{format_instructions}",
    input_variables=["city"],
    partial_variables={'format_instructions':parser.get_format_instructions()}

)

# prompt_result=prompt.format(city="delhi")

# result=model.invoke(prompt_result)  
# final_result=parser.parse(result.content)

# chain=prompt|model|parserer       
# chain_result=chain.invoke({"city": "delhi"})
# for chunks in model.stream(chain_result):
#     print(chunks.content,end="",flush=True)  #PydanticOutputParser is not work well with the streaming data normaly

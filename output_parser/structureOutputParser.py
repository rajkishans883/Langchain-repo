from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain.output_parsers.structured import (
    ResponseSchema,
    StructuredOutputParser
)
import os
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

schema=[
    ResponseSchema(name="key_themes", description="Write down all the key themes discussed in the review in a list", type="array"),
    ResponseSchema(name="summary", description="A brief summary of the review", type="string"),
    ResponseSchema(name="sentiment", description="Return sentiment of the review either negative, positive or neutral", type="string"),
    ResponseSchema(name="pros", description="Write down all the pros inside a list", type="array"),
    ResponseSchema(name="cons", description="Write down all the cons inside a list", type="array"),
    ResponseSchema(name="name", description="Write the name of the reviewer", type="string")
]
    

parser=StructuredOutputParser.from_response_schemas(schema) 
prompt =PromptTemplate(
    template="explain this {topic} in simple words \n{format_instructions}",
    input_variables=["topic"],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

prompt_result=prompt.format(topic="virat kohli")

result=model.invoke(prompt_result)
final_result=parser.parse(result.content)
print(final_result)

# chain= prompt | model | parser

# chain_result=chain.invoke({"topic": "blockchain"})
# print(chain_result)
# print(type(chain_result))
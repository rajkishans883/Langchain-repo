from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
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

parser=JsonOutputParser()
prompt =PromptTemplate(
    template="explain this {topic} in simple words \n{format_instructions}",
    input_variables=["topic"],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# prompt_result=prompt.format(topic="blockchain")

# result=model.invoke(prompt_result)
# final_result=parser.parse(result.content)
# print(final_result)

chain= prompt | model | parser

chain_result=chain.invoke({"topic": "blockchain"})
print(chain_result)
print(type(chain_result))
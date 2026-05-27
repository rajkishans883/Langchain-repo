from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
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


template1 = PromptTemplate(
    template="""
    Generate a detailed report on {topic} with:

    1. Introduction
    2. Key Concepts
    3. Advantages
    4. Disadvantages
    5. Conclusion

    Use proper headings and bullet points.
    """,
    input_variables=["topic"]
)

template2= PromptTemplate(
    template="give me summary of 5 lines on the {text}",
    input_variables=["text"]
)

prompt1=template1.format(topic="politics")
prompt2=template2.format(text="politics")

result=model.invoke(prompt1)

print(result.content)
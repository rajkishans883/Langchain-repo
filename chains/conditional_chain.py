from google.auth import default
from langchain_mistralai import ChatMistralAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch,RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
import os
from dotenv import load_dotenv

load_dotenv()

mistral_key=os.getenv("MISTRAL_API_KEY")

model=ChatMistralAI(
    api_key=mistral_key,    
    model_name="mistral-tiny",
    temperature=0.5,        
)
# llm = HuggingFaceEndpoint(
#     repo_id="nvidia/Nemotron-Labs-Diffusion-14B                                                                                                                                                                                                                                                                                                                                                                                             ",
#     task="text-generation"
# )                                                                                                               

# model2 = ChatHuggingFace(
#     llm=llm ,
#     max_tokens=1000,
# )

parser=StrOutputParser()

class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative"]=Field(description="The sentiment of the feedback")

parser2=PydanticOutputParser(pydantic_object=Sentiment)

template1 = PromptTemplate(
    template="classify the sentiment of the following text into three categories: positive, negative. Text: {feedback} \n{format_instructions}",
    input_variables=["feedback"],
    partial_variables={'format_instructions':parser2.get_format_instructions()}
)

classifier_chain=template1 | model | parser2

print(classifier_chain.invoke({"feedback": '''I'm really happy with the new features. The UI is so much better and the documentation is clearer. I'm also really happy with the performance of the app. It's fast and responsive, and I love how it's able to handle a lot of traffic without crashing.''' }))

prompt2=PromptTemplate(
    template="create a posititve feed back {feedbacl}",
    input_variables=["feedbacl"]
)

prompt3=PromptTemplate(
    template="create a negative feed back {feedbacl}",
    input_variables=["feedbacl"]
)

branch_chain= RunnableBranch(
    (lambda x:x.sentiment=="positive",prompt2|model|parser),
    (lambda x:x.sentiment=="negative",prompt3|model|parser),
    RunnableLambda(lambda x: "Invalid sentiment")
)

chain = classifier_chain | branch_chain


feedback={
    '''I'm really happy with the new features. The UI is so much better and the documentation is clearer. I'm also really happy with the performance of the app. It's fast and responsive, and I love how it's able to handle a lot of traffic without crashing.'''
}

result=chain.invoke({"feedback": feedback})
print(result)


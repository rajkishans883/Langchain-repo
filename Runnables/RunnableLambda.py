from langchain_mistralai import ChatMistralAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from langchain_core.runnables import RunnableParallel, RunnableBranch,RunnableLambda,RunnablePassthrough,RunnableSequence
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

def word_count(input: str) -> int:
    """Counts the number of words in the input string."""
    return len(input.split())

runnable_word_count= RunnableLambda(word_count)

prompt = PromptTemplate(
    template="create a joke  in the given topic: {text}",
    input_variables=["text"],
)
    
parser = StrOutputParser()

joke_generate = RunnableSequence(prompt, model , parser )

parralel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "word_count": RunnableLambda(lambda input: len(input.split())),
}
)

final_chain = RunnableSequence(joke_generate,parralel_chain)\

final_result = final_chain.invoke({"text": "santa banta jokes"})

final= '''{} \n word count  of the joke is {}'''.format(final_result.get("joke"),final_result.get("word_count"))

print(final) 
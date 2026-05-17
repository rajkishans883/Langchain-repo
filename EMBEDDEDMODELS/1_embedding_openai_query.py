from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv
import os
load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

model= MistralAIEmbeddings(
    api_key=mistral_api_key,
    model_name="mistral-small",
    temperature=0.7,
    streaming=True,         # look at this or chunk in result:
    
)

embeddings= MistralAIEmbeddings(model_name="mistral-embed")

result=embeddings.embed_query("What is the capital of France?")
print(str(result))
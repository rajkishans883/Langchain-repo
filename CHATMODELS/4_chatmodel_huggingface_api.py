from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/flan-t5-large",
    task="text2text-generation",
    max_new_tokens=100,
    temperature=0.7
)

response = llm.invoke("Who is Virat Kohli?")

print(response)
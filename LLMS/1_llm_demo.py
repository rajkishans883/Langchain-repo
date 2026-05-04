from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import os
load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

llm = ChatMistralAI(
    api_key=mistral_api_key,
    model_name="mistral-tiny",
    temperature=0.7,
)

response=llm.invoke("What is the capital of France?")

def format_response(response):
    return f"""
    ===== Mistral AI Response =====
    Content: {response.content}
    Model: {response.response_metadata.get('model_name', 'unknown')}
    Finish Reason: {response.response_metadata.get('finish_reason', 'unknown')}
    Token Usage:
        - Prompt Tokens: {response.response_metadata.get('token_usage', {}).get('prompt_tokens', 0)}
        - Completion Tokens: {response.response_metadata.get('token_usage', {}).get('completion_tokens', 0)}
        - Total Tokens: {response.response_metadata.get('token_usage', {}).get('total_tokens', 0)}
    ================================
    """

# Print the formatted response
print(format_response(response))
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os 
load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")
model= ChatMistralAI(
    api_key=mistral_api_key,
    model_name="mistral-tiny",
    temperature=0.7,
    max_tokens=1000,
    streaming=True,         
)
chat_history=[]

while True:
 
    user_input = input("You: ")
    chat_history.append((HumanMessage(content=user_input)))
    # prompt = ChatPromptTemplate.from_messages(chat_history)
    prompt = f"""
You are a helpful AI assistant.

Give short and direct answers.
Maximum 10 lines.
Do not explain too much.

Question: {chat_history}
"""
    if user_input.lower() == "exit":
        break
    result=model.invoke(prompt)      #this is the way to get the full response after the model finishes generating
    print(f"Mistral AI Response:${result.content}")
    chat_history.append((AIMessage(content=result.content)))
    # print(format_response(result))
    # for chunks in  model.stream(prompt):
    #     print(f"{chunks.content}", end="", flush=True)
        
    print("\n")

print(chat_history)
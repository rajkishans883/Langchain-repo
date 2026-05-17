from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

mistral_api_key = os.getenv("MISTRAL_API_KEY")

model = ChatMistralAI(
    api_key=mistral_api_key,
    model_name="mistral-tiny",
    temperature=0.7,
)

st.header("Mistral AI Chat Interface")

paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention Is All You Need",
        "BERT",
        "GPT-3",
        "Diffusion Models"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    [
        "Beginner-Friendly",
        "Technical",
        "Code-Oriented"
    ]
)

length_input = st.selectbox(
    "Select Explanation Length",
    [
        "Short",
        "Medium",
        "Long"
    ]
)

if st.button("Summarize"):

    prompt = f"""
    Explain the paper: {paper_input}

    Style: {style_input}

    Length: {length_input}
    """

    result = model.invoke(prompt)

    st.write(result.content)
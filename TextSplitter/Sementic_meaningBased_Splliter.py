from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableSequence
from pydantic import BaseModel, Field
from typing import Literal
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
mistral_key = os.getenv("MISTRAL_API_KEY")

# Initialize models
model = ChatMistralAI(
    api_key=mistral_key,
    model_name="mistral-tiny",
    temperature=0.3,       
)   
embeddings = MistralAIEmbeddings(model="mistral-embed")
loader = PyPDFLoader("../DocumentLoader/output.pdf")

loaded_docs = loader.load()

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile",  # The strategy (percentile, standard_deviation, etc.)
    breakpoint_threshold_amount=95           # The threshold value (e.g., 95th percentile)
)

sample="""The Indian Premier League (IPL) has transformed domestic cricket into a high-stakes, multi-billion-dollar global entertainment phenomenon. Beyond the explosive boundary-hitting and elite athletic talent on display, the tournament is a masterclass in tactical strategy and data analytics. Franchise management teams spend months analyzing player metrics, pitch variations, and historical match-ups to optimize their squads during the auction, making every single delivery on the field a battle of calculated risks and split-second decision-making.
Building modern web applications requires a tight integration between client-side user experiences and server-side data management, a balance perfectly struck by the MERN stack. By utilizing MongoDB, Express.js, React, and Node.js, developers can maintain a unified JavaScript environment across the entire development pipeline. React handles the dynamic, component-driven user interface, while Node.js and Express manage the asynchronous backend routing, seamlessly
Unlike traditional AI systems that simply respond to static prompts, Agentic AI represents a shift toward autonomous execution. These advanced systems use Large Language Models as a central reasoning engine, allowing them to break down complex goals into sequential tasks, select and use external tools, and self-correct when errors occur. By maintaining state and evaluating their own performance over multi-step workflows, AI agents can handle open-ended projects—such as automated software debugging or real-time market analysis—with minimal human intervention."""


result = splitter.create_documents([sample])

print(len(result))
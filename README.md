# 🦜 LangChain Repository

<div align="center">

![LangChain](https://img.shields.io/badge/LangChain-Latest-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8%2B-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

**A comprehensive guide to LangChain - Building AI-powered applications with Language Models**

[🚀 Quick Start](#-quick-start) • [📚 Topics](#-topics-covered) • [💻 Examples](#-code-examples) • [🤝 Contributing](#-contributing)

</div>

---

## 📖 About This Repository

This repository contains **complete LangChain tutorials, examples, and projects** designed to help you master building applications with Large Language Models (LLMs).

Whether you're a beginner or experienced developer, you'll find practical examples and best practices for:
- 🤖 Building AI chatbots
- 🔗 Chaining LLM operations
- 📊 RAG (Retrieval Augmented Generation) systems
- 🧠 Memory management
- 🛠️ Agent development
- 📈 Production-ready applications

---

## 🎯 What is LangChain?

**LangChain** is a framework for developing applications powered by language models. It enables you to:

- 🔗 **Chain operations** - Combine multiple LLM calls and tools
- 📚 **Access data** - Integrate with external data sources
- 🤖 **Create agents** - Build intelligent autonomous agents
- 💾 **Manage memory** - Maintain conversation context
- 🚀 **Deploy at scale** - Build production-ready applications

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- API keys for LLMs (OpenAI, Mistral, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/rajkishans883/Langchain-repo.git
cd Langchain-repo

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Example

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Initialize LLM
llm = ChatOpenAI(api_key="your-api-key", model_name="gpt-3.5-turbo")

# Create prompt template
prompt = ChatPromptTemplate.from_template(
    "Tell me a short story about {topic}"
)

# Create chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run chain
result = chain.run(topic="artificial intelligence")
print(result)
```

---

## 📚 Topics Covered

### 1. **LangChain Basics**
- LLM Setup & Configuration
- Chat Models vs LLMs
- Prompt Engineering
- Output Parsing

### 2. **Chains**
- Sequential Chains
- LLMChain
- Custom Chains
- Chain Debugging

### 3. **Memory**
- Buffer Memory
- Summary Memory
- Entity Memory
- Token Buffer Memory

### 4. **Agents**
- ReAct Agents
- Tool Integration
- Agent Executor
- Custom Tools

### 5. **RAG (Retrieval Augmented Generation)**
- Vector Databases
- Document Loaders
- Text Splitters
- Similarity Search
- Complete RAG Pipeline

### 6. **Vector Databases**
- Pinecone Integration
- Weaviate Integration
- FAISS Integration
- Embedding Models

### 7. **LLM Models**
- OpenAI API
- Mistral Models
- Google Gemini
- Anthropic Claude
- Local Models (LLaMA, Ollama)

### 8. **Advanced Topics**
- Streaming Responses
- Async Operations
- Error Handling
- Monitoring & Logging

### 9. **Real-World Projects**
- AI Chatbot
- Q&A System
- Content Generator
- Document Analyzer
- Code Assistant

---

## 💻 Code Examples

### Example 1: Simple Chatbot

```python
from langchain.chat_models import ChatMistralAI
from langchain.prompts import ChatPromptTemplate

# Initialize model
model = ChatMistralAI(
    api_key="your-mistral-api-key",
    model_name="mistral-small",
    temperature=0.7,
    streaming=True,
)

# Create prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant"),
    ("human", "{question}")
])

# Run
response = model.invoke(prompt.format_messages(question="What is LangChain?"))
print(response.content)
```

### Example 2: RAG System

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# Load documents
loader = TextLoader("document.txt")
documents = loader.load()

# Split text
splitter = CharacterTextSplitter(chunk_size=1000)
texts = splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore = FAISS.from_documents(texts, embeddings)

# Create retrieval QA
llm = ChatOpenAI(api_key="your-key")
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Query
answer = qa.run("What is the document about?")
print(answer)
```

### Example 3: Agent with Tools

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
import math

# Define tools
def calculate(expression):
    return str(eval(expression))

tools = [
    Tool(
        name="Calculator",
        func=calculate,
        description="Useful for math calculations"
    )
]

# Create agent
llm = ChatOpenAI(api_key="your-key")
agent = create_react_agent(llm, tools)

# Execute
executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

result = executor.invoke({"input": "What is 25 * 4?"})
print(result)
```

---

## 📂 Repository Structure

```
langchain-repo/
├── 1_basics/
│   ├── 01_llm_setup.py
│   ├── 02_prompt_templates.py
│   ├── 03_chains.py
│   └── 04_output_parsing.py
│
├── 2_memory/
│   ├── buffer_memory.py
│   ├── summary_memory.py
│   ├── entity_memory.py
│   └── conversation_memory.py
│
├── 3_agents/
│   ├── simple_agent.py
│   ├── agent_with_tools.py
│   ├── custom_tools.py
│   └── react_agent.py
│
├── 4_rag/
│   ├── simple_rag.py
│   ├── advanced_rag.py
│   ├── vector_databases.py
│   └── embeddings.py
│
├── 5_models/
│   ├── openai_models.py
│   ├── mistral_models.py
│   ├── claude_models.py
│   ├── gemini_models.py
│   └── local_models.py
│
├── 6_projects/
│   ├── ai_chatbot.py
│   ├── qa_system.py
│   ├── content_generator.py
│   ├── document_analyzer.py
│   └── code_assistant.py
│
├── 7_advanced/
│   ├── streaming.py
│   ├── async_operations.py
│   ├── error_handling.py
│   └── monitoring.py
│
├── requirements.txt
├── .env.example
├── README.md
└── LICENSE
```

---

## 🛠️ Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/rajkishans883/Langchain-repo.git
cd Langchain-repo
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Setup Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your API keys
# OPENAI_API_KEY=your-key
# MISTRAL_API_KEY=your-key
# PINECONE_API_KEY=your-key
```

### Step 5: Run Examples

```bash
# Run basic example
python 1_basics/01_llm_setup.py

# Run RAG example
python 4_rag/simple_rag.py

# Run chatbot project
python 6_projects/ai_chatbot.py
```

---

## 📋 Requirements

```
langchain==0.1.x
openai==1.x
mistralai==0.0.x
pinecone-client==2.2.x
faiss-cpu==1.7.x
python-dotenv==1.0.x
requests==2.31.x
```

See `requirements.txt` for complete list.

---

## 🎓 Learning Path

**Recommended order to learn:**

1. ✅ Basics (LLMs, Prompts, Chains)
2. ✅ Memory Management
3. ✅ Agents & Tools
4. ✅ RAG Systems
5. ✅ Different Models
6. ✅ Real-World Projects
7. ✅ Advanced Topics

---

## 💡 Common Use Cases

### 1. **AI Chatbot**
```python
# Conversational AI with memory
chatbot.chat("What's your name?")
chatbot.chat("Remember my name for later")
```

### 2. **Document Q&A**
```python
# Ask questions about documents
qa_system.query("Summarize the document")
```

### 3. **Content Generation**
```python
# Generate content automatically
generator.create_blog_post(topic="AI")
```

### 4. **Code Analysis**
```python
# Analyze and explain code
assistant.explain_code(code_snippet)
```

---

## 🔑 API Keys Needed

Get your API keys from:

- **OpenAI**: https://platform.openai.com/api-keys
- **Mistral**: https://console.mistral.ai/api-keys/
- **Google Gemini**: https://makersuite.google.com/
- **Anthropic Claude**: https://console.anthropic.com/
- **Pinecone**: https://www.pinecone.io/
- **Cohere**: https://dashboard.cohere.com/

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📚 Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Mistral Docs](https://docs.mistral.ai/)
- [RAG Explained](https://arxiv.org/abs/2005.11401)

---

## 🎥 Learning Resources

### Video Tutorials
- Complete LangChain Playlist (YouTube)
- RAG Systems Deep Dive
- Building Production Apps

### Courses
- LangChain Masterclass
- Advanced RAG Techniques
- Deploying LLM Apps

---

## 📞 Support & Contact

- 💼 **Professional Profile**: [Upgrad Hyderabad](https://www.upgrad.com/)
- 📧 **Email**: rajkishans883@gmail.com
- 🔗 **LinkedIn**: [Raj Kishan](https://linkedin.com/in/rajkishan388/)
- 🐙 **GitHub**: [@rajkishans883](https://github.com/rajkishans883)
- 🎥 **YouTube**: [Channel](https://youtube.com/@yourChannel)

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Show Your Support

If this repository helped you, please:
- ⭐ Star this repository
- 🔗 Share with others
- 💬 Give feedback
- 🐛 Report issues

---

<div align="center">

**Made with ❤️ by Raj Kishan**

*Full Stack Developer & Tech Trainer | Upgrad Hyderabad*

**Happy Learning! 🚀**

</div>

---

## 🗺️ Roadmap

- [ ] Add more LLM provider examples
- [ ] Advanced RAG techniques
- [ ] Production deployment guide
- [ ] Performance optimization
- [ ] Cost optimization strategies
- [ ] Multi-modal LangChain examples
- [ ] Real-time streaming examples

---

## ❓ FAQ

**Q: Which model should I use?**
A: Start with Mistral-small or GPT-3.5-turbo. They offer great balance of quality and cost.

**Q: How do I reduce API costs?**
A: Use smaller models, batch requests, implement caching, and use local embeddings.

**Q: Can I run this offline?**
A: Yes! Use Ollama for local LLMs and FAISS for local embeddings.

**Q: Is this suitable for production?**
A: Yes, with proper error handling, monitoring, and optimization.

---

## 🔐 Security

- Never commit API keys
- Use environment variables
- Use .env files (add to .gitignore)
- Implement rate limiting
- Validate user inputs

---

**Last Updated**: May 2024
**Python Version**: 3.8+
**LangChain Version**: 0.1.x+

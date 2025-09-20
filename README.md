# Plan-Execute App

## Overview
The **Plan-Execute App** is an AI-powered research assistant that transforms a user query into **actionable research tasks**.  
Inspired by [Manus AI](https://manus.ai/), it leverages **LangGraph** for orchestration, a **web browsing tool** for data collection, and integrates both **Ollama GPT OSS models** and a **GPT Vision model**.

## Features
- 📝 **Automatic Planning** – Turns a natural language query into a step-by-step research plan.  
- 🌐 **Web Browsing** – Uses a vision-enabled LLM to explore and extract information from web pages.  
- 🤖 **Multi-Model Support** –  
  - Ollama GPT-OSS for reasoning, planning, and text tasks.  
  - GPT Vision for analyzing and interpreting web content.  
- 📊 **Report Generation** – Summarizes past steps and results into a clean Markdown report.  

## How It Works
1. **Input a query** (e.g., “Summarize the latest AI research on protein folding”).  
2. The app **plans tasks** using Ollama GPT.  
3. The app **executes tasks**, browsing the web and extracting insights.  
4. A **final report** is generated in `report.md`.  

## Tech Stack
- [LangGraph](https://www.langchain.com/langgraph) – Workflow orchestration  
- [Streamlit](https://streamlit.io) – Interactive UI  
- [Ollama](https://ollama.ai) – Local LLM execution  
- GPT Vision Model – Web content analysis  

## Quick Start
```bash
# Clone the repository
git clone https://github.com/your-username/cgai.git
cd cgai-demo

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

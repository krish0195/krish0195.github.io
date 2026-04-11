
---
title: "Modern AI Development: From Setup to Agentic Systems"
date: 2026-04-06
lastmod: 2026-04-06
draft: false
tags: ["AI", "LLMOps", "LangChain"]
---

### 🔹 Overview

This blog explains the modern AI stack—from environment setup using `uv` to building intelligent agentic systems.

---

### 🔹 Environment Setup (`uv`)

I used `uv` as a fast and lightweight alternative to traditional tools like conda.

**Key commands:**
- Initialize project → `uv init`
- Install dependencies → `uv sync`
- Add package → `uv add numpy`
- Run script → `uv run main.py`

---

### 🔹 AI Basics

- **AI** → Simulates human intelligence  
- **ML** → Learns patterns from data  
- **DL** → Uses neural networks  
- **GenAI** → Generates new content (text, images, code)

---

### 🔹 Types of AI

- **Discriminative AI** → Classification (predicts output)  
- **Generative AI** → Creates new data  
- **Agentic AI** → Takes actions and makes decisions  

---

### 🔹 Transformers Revolution

Key milestones:
- VAE (2013)  
- GAN (2014)  
- Transformers (2017)  

Transformers enabled powerful models like BERT and GPT.

---

### 🔹 Building AI Agents (LangChain)

To move beyond chatbots, I explored tools in LangChain:

- Tool Decorator → Simple tools  
- Custom Tools → Advanced logic  
- Structured Tools → Multiple inputs  
- Built-in Tools → Search, APIs  

---

### 🔹 Outcome

- Clear understanding of modern AI stack  
- Ability to move from ML → GenAI → Agents  
- Hands-on knowledge of tools like LangChain  

---

### 🔹 Learning

AI is evolving from:
- Prediction → Generation → Action  

Understanding this shift is key to building real-world AI applications.

---

## 🧩 LLM & GenAI Development Ecosystem

The modern AI stack can be broken into multiple layers:

---

### 🔹 1. Foundations & Fine-Tuning

- **Frameworks:** Hugging Face, PyTorch  
- **Fine-tuning Tools:** Unsloth, LLaMA Factory  

---

### 🔹 2. RAG Backbone (Vector DB & Search)

**Open Source / Local:**
- FAISS  
- Chroma  
- Qdrant  
- Milvus  

**Managed / Cloud:**
- Supabase  
- Pinecone  
- AWS OpenSearch  
- Azure AI Search  

**SQL Extensions:**
- pgvector  

---

### 🔹 3. Application Frameworks & Agents

- **Frameworks:** LangChain, LlamaIndex, Haystack, LangSmith  
- **Agentic AI:** LangGraph, AutoGen, DeepAgent, n8n  
- **Local Runtime:** Ollama  

---

### 🔹 4. Data Modeling & Parsing

- **Validation:** Pydantic  
- **Document AI:**  
  - LlamaParse  
  - Docling  
  - Google Document AI  
  - Amazon Textract  
  - Azure Document Intelligence  
  - Unstructured.io  

- **Python Libraries:**  
  - python-docx  
  - PyMuPDF  
  - pdfplumber  

---

### 🚀 5. Infrastructure & Observability

**Databases & Storage:**
- Neo4j  
- MongoDB  
- PostgreSQL  
- Redis  

**Evaluation:**
- RAGAS  
- TruLens  
- DeepEval  
- OpenAI Evals  

**Prompt Engineering:**
- PromptLayer  
- LangChain Hub  
- DSPy  
- OpenAI Cookbook  

**Deployment & CI/CD:**
- AWS EC2 / ECS / Fargate  
- AWS SageMaker  
- GitHub Actions  
- Azure Kubernetes Service (AKS)  

**Orchestration & Testing:**
- Apache Airflow  
- pytest  

---

## 🔹 Final Takeaway

Modern AI is not just about models — it’s about **ecosystems** combining:

➡️ Data + Models + Agents + Infrastructure  

Mastering this stack is key to building **real-world AI applications**.
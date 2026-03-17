# 🦊 GitLab Handbook AI Chatbot – Technical Blueprint

**Author:** Khushboo Chaurasiya
**Project Type:** Generative AI (RAG-based Chatbot)
**Date:** March 2026

---

## 🔍 Key Highlights

* Built a **RAG-based AI chatbot** using **Gemini + ChromaDB**
* Implemented full pipeline: **Scraping → Processing → Chunking → Embedding → Retrieval → Generation**
* Achieved **accurate answers with source citations**
* Designed modular system with **FastAPI + Streamlit**
* Reduced hallucination using **context grounding + prompt design**

---

## 📌 Project Overview

GitLab’s handbook contains thousands of pages, making it difficult to search efficiently.

This project solves that by building a **conversational AI chatbot** that:

* Accepts natural language queries
* Retrieves relevant handbook content
* Generates accurate answers using LLM
* Provides source references

---

## 🏗️ System Architecture

```
User → Embedding → Vector Search → Top-K Chunks → LLM → Answer + Sources
```

### Components:

* **Frontend:** Streamlit chat UI
* **Backend:** FastAPI (RAG pipeline)
* **Vector DB:** ChromaDB
* **LLM:** Google Gemini
* **Embedding Model:** text-embedding-004

---

## 🧠 RAG Architecture (Core Idea)

**RAG = Retrieval + Generation**

1. Retrieve relevant document chunks
2. Pass them to LLM
3. Generate grounded answer

### Why RAG?

* No need for model training
* Reduces hallucination
* Always uses updated data
* Provides source traceability

---

## ⚙️ Data Pipeline

```
GitLab Data → Clean → Chunk → Embed → Store in ChromaDB
```

### Steps:

1. **Data Collection** – Scraped/Cloned GitLab handbook
2. **Preprocessing** – Removed noise & formatting
3. **Chunking** – 1000 chars with 200 overlap
4. **Embedding** – Converted text to vectors
5. **Storage** – Stored in ChromaDB

---

## 🔄 Query Flow

1. User asks question
2. Convert question → embedding
3. Retrieve top 5 similar chunks
4. Build prompt (context + question)
5. Gemini generates answer
6. Return answer + sources

---

## 📁 Project Structure

```
Helpe-Bot/
├── backend/
├── frontend/
├── scripts/
├── config/
├── tests/
├── .env.example
├── requirements.txt
├── README.md
└── TECHNICAL_BLUEPRINT.md
```

---

## 🚀 Implementation Summary

* Used **LangChain** for chunking
* Used **ChromaDB** for vector search
* Used **Gemini API** for generation
* Built API endpoints using **FastAPI**
* Created UI using **Streamlit**

---

## 🚧 Challenges & Learnings

* Faced GitHub upload issues due to large files → resolved by cleaning repo
* Learned importance of **vector similarity vs keyword search**
* Improved prompt design to reduce hallucinations
* Understood trade-offs between **RAG vs fine-tuning**

---

## 🎯 Key Design Decisions

* **ChromaDB:** Easy local setup
* **Gemini:** Free tier + strong performance
* **Chunk size (1000):** Balance between context & precision
* **Top-K retrieval (5):** Improves answer quality

---

## 💬 Interview Explanation (Short)

> Built a RAG-based chatbot using Gemini and ChromaDB.
> Scraped GitLab handbook, chunked data, generated embeddings, and stored them.
> At query time, relevant chunks are retrieved using cosine similarity and passed to LLM to generate accurate answers with sources.

---

## 📈 Future Improvements

* Hybrid search (BM25 + vector)
* Conversation memory
* Deployment on cloud
* Feedback system

---

## ⚠️ Note

Vector database and raw data are excluded due to size constraints.
Run scripts to regenerate locally.

---

**End of Document**

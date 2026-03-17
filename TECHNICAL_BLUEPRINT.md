```markdown
# GitLab Handbook AI Chatbot – Complete Implementation Blueprint & Technical Design Document

> **Author:** [Khushboo Chaurasiya]
> **Date:** March 2026
> **Project Type:** Internship Assignment – Generative AI Chatbot using RAG
> **Difficulty Level:** Beginner-Friendly

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [RAG Architecture Explanation](#4-rag-architecture-explanation)
5. [Data Pipeline Design](#5-data-pipeline-design)
6. [Query Processing Flow](#6-query-processing-flow)
7. [Project Folder Structure](#7-project-folder-structure)
8. [Step-by-Step Implementation Roadmap](#8-step-by-step-implementation-roadmap)
9. [Deployment Plan](#9-deployment-plan)
10. [Bonus Features](#10-bonus-features-for-higher-evaluation-score)
11. [Interview Explanation](#11-interview-explanation)
12. [Implementation Prompts](#12-implementation-prompts)

---

## 1. Project Overview

### 1.1 Problem Statement

GitLab maintains an extensive public handbook (https://handbook.gitlab.com) and direction pages
(https://about.gitlab.com/direction/) that contain thousands of pages covering company culture,
engineering practices, product strategy, HR policies, and more. Finding specific information across
this massive knowledge base is time-consuming and inefficient using traditional keyword search.

**The core problem:** Users need a fast, conversational way to ask natural-language questions and
receive accurate, context-aware answers sourced directly from GitLab's official documentation.

### 1.2 Goal of the Chatbot

Build an AI-powered chatbot that:
- Accepts natural language questions from users.
- Retrieves the most relevant sections from GitLab's handbook and direction pages.
- Uses a Large Language Model (LLM) to generate accurate, human-readable answers.
- Cites the sources so users can verify the information.

### 1.3 Target Users

| User Type | Use Case |
|-----------|----------|
| GitLab employees | Quickly find internal policies, processes, and guidelines |
| Job candidates | Research GitLab's culture and values before interviews |
| Open-source contributors | Understand GitLab's engineering practices |
| Students & researchers | Study GitLab's remote-work and DevOps methodologies |

### 1.4 Key Features

- **Conversational Q&A** – Ask questions in plain English and get direct answers.
- **Source Citations** – Every answer links back to the original handbook page.
- **Context-Aware Retrieval** – Uses vector similarity search (not keyword matching) for better relevance.
- **Low Hallucination** – RAG architecture grounds the LLM in real documents, reducing made-up answers.
- **Simple UI** – Clean chat interface anyone can use.

---

## 2. System Architecture

### 2.1 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                               │
│                  (Streamlit / React Frontend)                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │  User sends a question
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BACKEND SERVER                                 │
│                  (FastAPI / Flask / Express)                         │
│                                                                     │
│  1. Receive user question                                           │
│  2. Convert question → embedding vector                             │
│  3. Search vector database for similar chunks                       │
│  4. Retrieve top-K relevant document chunks                         │
│  5. Build prompt = retrieved context + user question                │
│  6. Send prompt to LLM (Gemini / OpenAI)                           │
│  7. Return LLM's answer + source citations to frontend             │
└──────────────┬──────────────────────┬───────────────────────────────┘
               │                      │
               ▼                      ▼
┌──────────────────────┐  ┌───────────────────────────┐
│   VECTOR DATABASE    │  │    LLM API SERVICE        │
│  (FAISS / ChromaDB)  │  │  (Google Gemini / OpenAI) │
│                      │  │                           │
│  Stores document     │  │  Generates natural        │
│  embeddings for      │  │  language answers from     │
│  similarity search   │  │  context + question        │
└──────────────────────┘  └───────────────────────────┘
         ▲
         │  (Pre-built offline)
┌──────────────────────────────────────┐
│         DATA PIPELINE                │
│                                      │
│  GitLab Handbook → Scrape/Download   │
│  → Clean Text → Chunk Documents     │
│  → Generate Embeddings → Store in   │
│    Vector Database                   │
└──────────────────────────────────────┘
```

### 2.2 Components Explained

| Component | Role |
|-----------|------|
| **Frontend** | Chat UI where users type questions and see answers |
| **Backend Server** | Orchestrates the entire RAG pipeline: embedding, retrieval, LLM call |
| **Embedding Model** | Converts text (questions & documents) into numerical vectors |
| **Vector Database** | Stores document embeddings; performs fast similarity search |
| **LLM (Large Language Model)** | Reads retrieved context and generates a human-readable answer |
| **Data Pipeline** | Offline process that scrapes, cleans, chunks, and embeds GitLab data |

### 2.3 How the Chatbot Interacts with the Knowledge Base and LLM

1. **Offline (one-time setup):** The data pipeline scrapes GitLab's handbook, splits it into small
   chunks, converts each chunk into a vector embedding, and stores them in a vector database.

2. **Online (every user query):**
   - The user's question is converted into an embedding using the same embedding model.
   - The vector database finds the chunks whose embeddings are most similar to the question's
     embedding (cosine similarity).
   - The top 3–5 most relevant chunks are retrieved.
   - These chunks, along with the user's original question, are sent to the LLM as a prompt.
   - The LLM reads the context and generates an answer grounded in the retrieved documents.
   - The answer (with source links) is returned to the user.

---

## 3. Technology Stack

### 3.1 Recommended Stack (Beginner-Friendly)

| Layer | Tool | Why This Tool? |
|-------|------|----------------|
| **LLM** | Google Gemini (gemini-1.5-flash) | Free tier available, excellent quality, easy API, great for beginners |
| **Embedding Model** | Google's `text-embedding-004` or `all-MiniLM-L6-v2` (HuggingFace) | Free, high quality, widely used in tutorials |
| **Vector Database** | ChromaDB | Easiest setup (just `pip install`), no server needed, perfect for prototypes |
| **Backend Framework** | FastAPI (Python) | Modern, fast, auto-generates API docs, beginner-friendly with Python |
| **Frontend** | Streamlit | Build a chat UI in ~50 lines of Python, no HTML/CSS/JS needed |
| **Web Scraping** | BeautifulSoup + Requests | Simple Python libraries for scraping web pages |
| **Text Splitting** | LangChain's `RecursiveCharacterTextSplitter` | Smart chunking that respects sentence boundaries |
| **Orchestration** | LangChain | Connects all RAG components with minimal boilerplate |
| **Deployment** | Streamlit Community Cloud / HuggingFace Spaces | Free hosting for Python apps |

### 3.2 Alternative Stack (If You Want to Use MERN Skills)

| Layer | Tool | Why? |
|-------|------|------|
| **Backend** | Express.js + LangChain.js | Use your existing Node.js experience |
| **Frontend** | React.js | Use your existing React experience |
| **Vector DB** | Supabase (pgvector) | Free tier, managed PostgreSQL with vector search |
| **LLM** | OpenAI GPT-4o-mini | Affordable, excellent quality |
| **Deployment** | Vercel (frontend) + Railway (backend) | Free tiers available |

### 3.3 Why Each Tool Matters

- **LLM (Gemini):** The "brain" that understands your question and generates answers. Gemini is
  chosen because Google offers a generous free tier (15 RPM, 1M tokens/day) and the API is simple.

- **Embedding Model:** Converts text into numbers (vectors). Think of it as translating words into
  a coordinate system where similar words are close together. This is what makes semantic search
  possible.

- **Vector Database (ChromaDB):** A special database optimized for storing and searching vectors.
  Unlike SQL databases that search by exact matches, vector databases find similar items. ChromaDB
  runs locally with zero configuration—just install and use.

- **FastAPI:** A Python web framework that creates REST APIs. It's faster than Flask and
  auto-generates interactive documentation at `/docs`.

- **Streamlit:** A Python library that turns scripts into web apps. You write Python, and it
  renders a UI. Perfect for ML/AI demos.

- **LangChain:** A framework that provides pre-built components for RAG pipelines. Instead of
  writing everything from scratch, you plug together modules.

---

## 4. RAG Architecture Explanation

### 4.1 What is RAG?

**RAG = Retrieval Augmented Generation**

RAG is an AI architecture pattern that combines two capabilities:

1. **Retrieval:** Finding relevant information from a knowledge base (like a smart search engine).
2. **Generation:** Using an LLM to generate a natural language answer based on the retrieved
   information.

**Simple analogy:** Imagine you're taking an open-book exam. Instead of memorizing everything,
you search your textbook for relevant pages (retrieval), read them, and then write your answer
(generation). RAG works the same way for AI.

### 4.2 Why RAG is Required for This Project

| Problem Without RAG | How RAG Solves It |
|----------------------|-------------------|
| LLMs don't know GitLab's specific handbook content | RAG feeds the exact handbook text to the LLM at query time |
| LLMs have a knowledge cutoff date | RAG uses live/updated documents, so answers are always current |
| LLMs hallucinate (make up facts) | RAG grounds answers in real documents, reducing hallucination |
| Fine-tuning an LLM on GitLab data is expensive | RAG requires no fine-tuning—just store documents and retrieve them |
| LLMs have token limits | RAG sends only the relevant chunks (not the entire handbook) |

### 4.3 How RAG Improves Chatbot Accuracy

```
WITHOUT RAG:
User: "What is GitLab's vacation policy?"
LLM: *Makes up a generic answer* ← WRONG, may hallucinate

WITH RAG:
User: "What is GitLab's vacation policy?"
System: [Retrieves actual handbook section about PTO policy]
LLM: "According to GitLab's handbook, team members have unlimited PTO..."
     ← ACCURATE, grounded in real data
```

**Key accuracy improvements:**
- **Factual grounding:** The LLM can only reference information that exists in the retrieved context.
- **Source traceability:** Every answer can point to the exact handbook section it came from.
- **Reduced hallucination:** The system prompt instructs the LLM to answer ONLY from the provided
  context and say "I don't know" if the context doesn't contain the answer.

### 4.4 RAG vs Other Approaches

| Approach | Pros | Cons | Best For |
|----------|------|------|----------|
| **RAG** | No training needed, always up-to-date, cheap | Depends on retrieval quality | This project ✅ |
| **Fine-tuning** | Deep domain knowledge | Expensive, needs retraining when data changes | Static datasets |
| **Prompt stuffing** | Simple | Token limit reached quickly | Very small datasets |

---

## 5. Data Pipeline Design

### 5.1 Pipeline Overview

```
GitLab Website → Scrape Pages → Clean HTML → Extract Text
→ Split into Chunks → Generate Embeddings → Store in ChromaDB
```

### 5.2 Step 1: Collect GitLab Handbook Data

**Source URLs:**
- Handbook: `https://handbook.gitlab.com/handbook/`
- Direction: `https://about.gitlab.com/direction/`

**Methods (choose one):**

**Method A – Web Scraping (Recommended for learning)**
```python
import requests
from bs4 import BeautifulSoup

def scrape_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Remove navigation, footer, scripts
    for tag in soup(['nav', 'footer', 'script', 'style', 'header']):
        tag.decompose()

    # Extract main content
    content = soup.find('main') or soup.find('article') or soup.find('body')
    text = content.get_text(separator='\n', strip=True)
    return text
```

**Method B – Use GitLab's Public API / Git Repository**
GitLab's handbook is stored as Markdown files in a public Git repository. You can clone it:
```bash
git clone https://gitlab.com/gitlab-com/content-sites/handbook.git
```
This gives you all handbook pages as `.md` files—no scraping needed!

**Method C – Use Pre-built Datasets**
Search HuggingFace Datasets for "GitLab handbook" to find pre-scraped datasets.

### 5.3 Step 2: Clean and Preprocess the Text

Raw scraped data contains noise. Clean it:

```python
import re

def clean_text(text):
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s.,;:!?\'\"()\-/]', '', text)

    # Remove very short lines (navigation artifacts)
    lines = text.split('\n')
    lines = [line.strip() for line in lines if len(line.strip()) > 30]

    return '\n'.join(lines)
```

**What to clean:**
- ❌ HTML tags and attributes
- ❌ Navigation menus, footers, sidebars
- ❌ JavaScript code
- ❌ Duplicate content
- ❌ Very short fragments (< 30 characters)
- ✅ Keep headings, paragraphs, lists, tables

### 5.4 Step 3: Split Documents into Chunks

**Why chunk?** LLMs have token limits. The entire handbook is millions of tokens. We need to split
it into small, meaningful pieces that can be individually retrieved.

**Chunking strategy:**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # Each chunk is ~1000 characters (~250 words)
    chunk_overlap=200,      # 200 chars overlap between consecutive chunks
    separators=["\n\n", "\n", ". ", " "]  # Split at paragraph > line > sentence > word
)

chunks = splitter.split_text(cleaned_text)
```

**Key parameters explained:**
- **chunk_size = 1000:** Each chunk contains ~250 words. Small enough for precise retrieval, large
  enough to contain meaningful information.
- **chunk_overlap = 200:** Consecutive chunks share 200 characters. This prevents information from
  being cut in the middle of a sentence.
- **separators:** The splitter first tries to split at double newlines (paragraphs), then single
  newlines, then sentences, then words. This preserves the logical structure of the text.

### 5.5 Step 4: Create Embeddings

**What is an embedding?** An embedding converts text into a list of numbers (a vector). Texts with
similar meaning have vectors that are close together in this number space.

```python
# Using Google's Embedding API
import google.generativeai as genai

genai.configure(api_key="YOUR_GEMINI_API_KEY")

def get_embedding(text):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return result['embedding']  # Returns a list of 768 numbers

# Example
embedding = get_embedding("What is GitLab's remote work policy?")
print(len(embedding))  # 768
```

**Or using a free local model (no API key needed):**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text).tolist()  # Returns a list of 384 numbers
```

### 5.6 Step 5: Store Embeddings in Vector Database

```python
import chromadb

# Create a persistent ChromaDB instance (saves to disk)
client = chromadb.PersistentClient(path="./vector_db")

# Create a collection (like a table in SQL)
collection = client.get_or_create_collection(
    name="gitlab_handbook",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)

# Add documents to the collection
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[f"chunk_{i}"],
        metadatas=[{
            "source": "handbook.gitlab.com/handbook/values",
            "chunk_index": i
        }]
    )

print(f"Stored {collection.count()} chunks in ChromaDB")
```

**What happens here:**
1. ChromaDB automatically generates embeddings for each document (or you can provide your own).
2. Each chunk is stored with its text, embedding, and metadata (source URL, chunk index).
3. The data is persisted to disk in the `./vector_db` folder.

---

## 6. Query Processing Flow

### 6.1 End-to-End Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                     USER ASKS A QUESTION                         │
│         "What is GitLab's approach to remote work?"              │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│              STEP 1: EMBED THE QUESTION                          │
│                                                                  │
│  Convert question text → vector [0.023, -0.156, 0.891, ...]     │
│  Using the SAME embedding model used for documents               │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│              STEP 2: VECTOR SIMILARITY SEARCH                    │
│                                                                  │
│  Search ChromaDB for the top 5 chunks most similar to the        │
│  question vector (using cosine similarity)                       │
│                                                                  │
│  Result: 5 text chunks about remote work from the handbook       │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│              STEP 3: BUILD THE PROMPT                            │
│                                                                  │
│  System: "You are a helpful assistant that answers questions      │
│  about GitLab using ONLY the provided context. If the context    │
│  doesn't contain the answer, say 'I don't have that info.'"     │
│                                                                  │
│  Context: [chunk_1] [chunk_2] [chunk_3] [chunk_4] [chunk_5]      │
│                                                                  │
│  Question: "What is GitLab's approach to remote work?"           │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│              STEP 4: LLM GENERATES ANSWER                        │
│                                                                  │
│  Gemini reads the context + question and generates:              │
│                                                                  │
│  "GitLab is one of the world's largest all-remote companies.     │
│   According to the handbook, they believe remote work allows     │
│   hiring the best talent globally..."                            │
│                                                                  │
│  Sources: [handbook.gitlab.com/handbook/company/culture/...]     │
└──────────────────────────┬───────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│              STEP 5: RETURN ANSWER TO USER                       │
│                                                                  │
│  Display the answer in the chat UI with source links             │
└──────────────────────────────────────────────────────────────────┘
```

### 6.2 Code Implementation of the Query Flow

```python
import google.generativeai as genai
import chromadb

# Initialize
genai.configure(api_key="YOUR_API_KEY")
client = chromadb.PersistentClient(path="./vector_db")
collection = client.get_collection("gitlab_handbook")

def answer_question(user_question: str) -> dict:
    # Step 1 & 2: Query ChromaDB (it handles embedding + search automatically)
    results = collection.query(
        query_texts=[user_question],
        n_results=5  # Retrieve top 5 relevant chunks
    )

    # Step 3: Build the prompt
    context = "\n\n---\n\n".join(results['documents'][0])
    sources = [meta['source'] for meta in results['metadatas'][0]]

    prompt = f"""You are a helpful AI assistant that answers questions about GitLab
using ONLY the provided context from GitLab's official handbook.

RULES:
- Answer ONLY based on the context provided below.
- If the context doesn't contain enough information, say "I don't have enough
  information from the handbook to answer this question."
- Cite which section the information comes from.
- Be concise but thorough.

CONTEXT:
{context}

QUESTION: {user_question}

ANSWER:"""

    # Step 4: Call the LLM
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)

    # Step 5: Return answer + sources
    return {
        "answer": response.text,
        "sources": list(set(sources)),
        "num_chunks_used": len(results['documents'][0])
    }
```

---

## 7. Project Folder Structure

```
gitlab-handbook-chatbot/
│
├── backend/                    # Backend server code
│   ├── app.py                  # Main FastAPI / Flask application
│   ├── rag_engine.py           # Core RAG logic (retrieve + generate)
│   ├── embeddings.py           # Embedding generation utilities
│   ├── config.py               # Configuration settings (API keys, model names)
│   └── requirements.txt        # Python dependencies
│
├── frontend/                   # Frontend code
│   ├── streamlit_app.py        # Streamlit chat UI (if using Streamlit)
│   └── requirements.txt       # Frontend-specific dependencies
│   │
│   └── react-app/              # (Alternative: React frontend)
│       ├── src/
│       │   ├── App.jsx
│       │   ├── components/
│       │   │   ├── ChatWindow.jsx
│       │   │   ├── MessageBubble.jsx
│       │   │   └── SourceCard.jsx
│       │   └── services/
│       │       └── api.js
│       └── package.json
│
├── data/                       # Raw and processed data
│   ├── raw/                    # Raw scraped HTML / Markdown files
│   ├── processed/              # Cleaned text files
│   └── chunks/                 # Chunked documents (JSON)
│
├── vector_db/                  # ChromaDB persistent storage
│   └── chroma.sqlite3          # Auto-generated by ChromaDB
│
├── scripts/                    # Data pipeline scripts
│   ├── scraper.py              # Web scraping script
│   ├── preprocessor.py         # Text cleaning script
│   ├── chunker.py              # Document chunking script
│   └── build_vectordb.py       # Embedding + vector DB creation script
│
├── config/                     # Configuration files
│   ├── .env                    # Environment variables (API keys) – DO NOT COMMIT
│   └── settings.yaml           # App settings (chunk size, model name, etc.)
│
├── tests/                      # Unit tests
│   ├── test_rag_engine.py
│   └── test_embeddings.py
│
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── docker-compose.yml          # (Optional) Docker setup
└── Makefile                    # (Optional) Common commands
```

### Folder Explanation

| Folder | Purpose |
|--------|---------|
| `backend/` | Core server code — the API that handles questions and returns answers |
| `frontend/` | User interface — where users interact with the chatbot |
| `data/` | All data files — raw downloads, cleaned text, and pre-chunked documents |
| `vector_db/` | ChromaDB storage — auto-created when you run the pipeline |
| `scripts/` | One-time pipeline scripts — run these to build the knowledge base |
| `config/` | Settings and secrets — API keys, model choices, chunk parameters |
| `tests/` | Automated tests — verify your RAG engine works correctly |

---

## 8. Step-by-Step Implementation Roadmap

### Phase 1 – Data Collection (Day 1–2)

**Goal:** Download GitLab handbook content and save it locally.

**Tasks:**
1. Clone the GitLab handbook repository:
   ```bash
   git clone https://gitlab.com/gitlab-com/content-sites/handbook.git data/raw/handbook
   ```
2. Or write a scraper for key handbook pages using BeautifulSoup.
3. Save all content as `.md` or `.txt` files in `data/raw/`.
4. Create a manifest file listing all collected pages with their URLs.

**Output:** `data/raw/` folder with 100+ markdown/text files.

**Key file to create:** `scripts/scraper.py`

---

### Phase 2 – Data Processing (Day 2–3)

**Goal:** Clean raw data and prepare it for chunking.

**Tasks:**
1. Remove HTML tags, navigation elements, and boilerplate.
2. Fix encoding issues and special characters.
3. Remove duplicate content.
4. Standardize formatting (consistent line endings, whitespace).
5. Save cleaned files to `data/processed/`.

**Output:** `data/processed/` folder with clean text files.

**Key file to create:** `scripts/preprocessor.py`

---

### Phase 3 – Embedding Generation (Day 3–4)

**Goal:** Split documents into chunks and generate vector embeddings.

**Tasks:**
1. Install LangChain: `pip install langchain langchain-text-splitters`
2. Use `RecursiveCharacterTextSplitter` with `chunk_size=1000` and `chunk_overlap=200`.
3. Store chunks as JSON with metadata (source URL, chunk index, title).
4. Generate embeddings for each chunk using your chosen embedding model.

**Output:** `data/chunks/` folder with JSON files containing chunks + embeddings.

**Key file to create:** `scripts/chunker.py`

---

### Phase 4 – Vector Database Setup (Day 4–5)

**Goal:** Store all embeddings in ChromaDB for fast similarity search.

**Tasks:**
1. Install ChromaDB: `pip install chromadb`
2. Create a persistent ChromaDB collection called `gitlab_handbook`.
3. Insert all chunks with their embeddings and metadata.
4. Test the database by running sample queries.
5. Verify that search results are relevant.

**Output:** `vector_db/` folder with ChromaDB data.

**Key file to create:** `scripts/build_vectordb.py`

**Test command:**
```python
results = collection.query(query_texts=["What are GitLab's values?"], n_results=3)
print(results['documents'])
```

---

### Phase 5 – Chatbot Backend Development (Day 5–7)

**Goal:** Create the API that connects the vector database to the LLM.

**Tasks:**
1. Install FastAPI: `pip install fastapi uvicorn`
2. Create the RAG engine (`backend/rag_engine.py`) with the query function from Section 6.
3. Create FastAPI endpoints:
   - `POST /ask` – accepts a question, returns an answer with sources.
   - `GET /health` – health check endpoint.
4. Add error handling and input validation.
5. Set up environment variable loading for API keys.

**API contract:**
```json
// POST /ask
// Request:
{ "question": "What is GitLab's approach to async communication?" }

// Response:
{
  "answer": "GitLab practices async-first communication...",
  "sources": ["handbook.gitlab.com/handbook/communication/"],
  "confidence": 0.89
}
```

**Key files to create:** `backend/app.py`, `backend/rag_engine.py`, `backend/config.py`

---

### Phase 6 – Frontend Chat UI (Day 7–9)

**Goal:** Build a user-friendly chat interface.

**Option A: Streamlit (Fastest)**
```python
import streamlit as st
import requests

st.set_page_config(page_title="GitLab Handbook AI", page_icon="🦊")
st.title("🦊 GitLab Handbook Chatbot")
st.caption("Ask me anything about GitLab's handbook!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about GitLab..."):
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from backend
    with st.spinner("Searching the handbook..."):
        response = requests.post(
            "http://localhost:8000/ask",
            json={"question": prompt}
        )
        data = response.json()

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(data["answer"])
        if data.get("sources"):
            st.markdown("**📚 Sources:**")
            for source in data["sources"]:
                st.markdown(f"- [{source}]({source})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": data["answer"]
    })
```

**Option B: React (if you prefer using your MERN skills)**
- Create a React app with a `ChatWindow` component.
- Use `fetch` or `axios` to call the FastAPI backend.
- Display messages in a scrollable chat container.

**Key file to create:** `frontend/streamlit_app.py` or `frontend/react-app/`

---

### Phase 7 – Testing & Deployment (Day 9–10)

**Goal:** Test the complete system and deploy it.

**Tasks:**
1. Test with 20+ questions covering different handbook topics.
2. Verify source citations are correct.
3. Test edge cases: empty questions, very long questions, off-topic questions.
4. Deploy to a free platform (see Section 9).
5. Write documentation (README.md).

---

## 9. Deployment Plan

### Option 1: Streamlit Community Cloud (Easiest – Recommended)

**Best for:** Python-only projects using Streamlit frontend.

**Steps:**
1. Push your code to a GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account.
4. Select your repository and the entry file (`frontend/streamlit_app.py`).
5. Add API keys in the "Secrets" section (equivalent to `.env`).
6. Click "Deploy."

**Limitations:** 1GB memory, sleeps after inactivity, Python only.

**Architecture for Streamlit Cloud:** Combine backend + frontend into one Streamlit app (no
separate FastAPI server needed).

```python
# Combined app: streamlit_app.py
import streamlit as st
import chromadb
import google.generativeai as genai

# All RAG logic runs directly inside the Streamlit app
# No need for a separate backend server
```

### Option 2: HuggingFace Spaces (Good Alternative)

**Best for:** ML/AI demos with more memory requirements.

**Steps:**
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose "Streamlit" or "Gradio" as the SDK.
3. Push your code to the Space's Git repository.
4. Add API keys as "Secrets" in the Space settings.
5. The app auto-deploys on every push.

**Advantages:** 2GB free memory, GPU options available, great for AI projects.

### Option 3: Vercel + Railway (For MERN Stack Approach)

**Best for:** React frontend + Express/FastAPI backend.

**Steps:**
1. **Frontend (Vercel):**
   - Push React app to GitHub.
   - Import project on [vercel.com](https://vercel.com).
   - Set `REACT_APP_API_URL` environment variable.
   - Deploy automatically.

2. **Backend (Railway):**
   - Push FastAPI backend to GitHub.
   - Connect to [railway.app](https://railway.app).
   - Add environment variables (API keys).
   - Deploy automatically.

### Option 4: Docker (For Advanced Users)

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Include pre-built vector database
COPY vector_db/ ./vector_db/

EXPOSE 8501
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501"]
```

---

## 10. Bonus Features (For Higher Evaluation Score)

### 10.1 Source Citations with Links
Every answer should display the exact handbook page it came from, with clickable links.

```python
# In your prompt template, add:
"Always cite the source handbook page for each fact you mention."
```

### 10.2 Guardrails (Prevent Off-Topic / Harmful Responses)

```python
SYSTEM_PROMPT = """You are a GitLab Handbook assistant.

IMPORTANT RULES:
1. ONLY answer questions related to GitLab's handbook and direction pages.
2. If a question is unrelated to GitLab, politely redirect: "I can only answer
   questions about GitLab's handbook. Try asking about GitLab's values, processes,
   or policies."
3. Never generate harmful, offensive, or misleading content.
4. Never make up information. If unsure, say "I don't have enough information."
"""
```

### 10.3 Context Transparency
Show users which chunks were used to generate the answer. This builds trust.

```python
# In the UI, add an expandable section:
with st.expander("📄 View Retrieved Context"):
    for i, chunk in enumerate(retrieved_chunks):
        st.markdown(f"**Chunk {i+1}:**\n{chunk}")
```

### 10.4 Conversation Memory
Remember previous messages in the conversation for follow-up questions.

```python
# Store conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Include history in the prompt
history_text = "\n".join([
    f"User: {msg['user']}\nAssistant: {msg['assistant']}"
    for msg in st.session_state.chat_history[-3:]  # Last 3 exchanges
])

prompt = f"""Previous conversation:
{history_text}

Current context: {context}
Current question: {user_question}
"""
```

### 10.5 Better UX Features
- **Suggested questions:** Show 3–4 example questions on first load.
- **Loading animations:** Show "Searching handbook..." while processing.
- **Feedback buttons:** Add 👍/👎 buttons for answer quality feedback.
- **Dark mode:** Streamlit supports it natively with `st.set_page_config(layout="wide")`.
- **Export chat:** Allow users to download the conversation as a text file.

### 10.6 Advanced: Hybrid Search
Combine vector search with keyword search (BM25) for better retrieval:

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Combine vector + keyword search
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # 70% semantic, 30% keyword
)
```

### 10.7 Query Analytics Dashboard
Track the most asked questions, average response time, and user satisfaction:

```python
import json
from datetime import datetime

def log_query(question, answer, sources, response_time):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer_length": len(answer),
        "sources_count": len(sources),
        "response_time_ms": response_time
    }
    with open("logs/query_log.jsonl", "a") as f:
        f.write(json.dumps(log_entry) + "\n")
```

---

## 11. Interview Explanation

### Short Version (30 seconds)

> "I built a Generative AI chatbot that answers questions about GitLab's handbook using RAG—
> Retrieval Augmented Generation. I scraped GitLab's handbook pages, split them into chunks,
> generated vector embeddings using Google's embedding model, and stored them in ChromaDB. When a
> user asks a question, the system converts it to a vector, finds the most similar handbook chunks
> using cosine similarity, and sends those chunks along with the question to Gemini, which generates
> a grounded, accurate answer with source citations."

### Detailed Version (2 minutes)

> "The project is a domain-specific AI chatbot built using the RAG architecture. Let me walk you
> through the key components:
>
> **Data Pipeline:** I collected data from GitLab's public handbook by cloning their content
> repository. The raw Markdown files were cleaned, and then split into overlapping chunks of ~1000
> characters using LangChain's RecursiveCharacterTextSplitter. Each chunk was converted into a
> 768-dimensional vector embedding using Google's text-embedding-004 model, and stored in ChromaDB,
> a lightweight vector database.
>
> **Query Pipeline:** When a user asks a question, it's embedded using the same model, and ChromaDB
> performs a cosine similarity search to find the top 5 most relevant chunks. These chunks are
> injected into a carefully designed prompt template with a system instruction that constrains the
> LLM to only use the provided context. The prompt is sent to Google Gemini (gemini-1.5-flash),
> which generates a natural language answer.
>
> **Architecture Decisions:** I chose RAG over fine-tuning because the handbook content changes
> frequently, RAG requires no GPU training, and it's more cost-effective. I picked ChromaDB for
> zero-configuration setup, Gemini for its generous free tier, and Streamlit for rapid UI
> prototyping. The system includes guardrails to prevent off-topic responses and source citations
> for transparency.
>
> **Results:** The chatbot accurately answers questions about GitLab's values, remote work policies,
> engineering practices, and more, with clear source citations that users can verify."

### Common Interview Questions & Answers

**Q: Why did you choose RAG instead of fine-tuning?**
> RAG is more suitable here because: (1) GitLab's handbook is updated frequently, and RAG uses live
> documents—no retraining needed. (2) Fine-tuning requires GPUs and significant compute resources.
> (3) RAG provides source traceability, which fine-tuning doesn't.

**Q: How do you handle hallucination?**
> Three strategies: (1) The system prompt explicitly instructs the LLM to only use the provided
> context. (2) If the context doesn't contain relevant information, the LLM is instructed to say
> "I don't know." (3) Source citations allow users to verify the answer.

**Q: What is cosine similarity and why use it?**
> Cosine similarity measures the angle between two vectors. It ranges from -1 to 1, where 1 means
> identical direction (similar meaning). It's preferred over Euclidean distance for text because it
> focuses on the direction of vectors (semantic meaning) rather than magnitude.

**Q: How do you decide the chunk size?**
> 1000 characters (~250 words) with 200-character overlap. Too small = chunks lack context. Too
> large = chunks may contain irrelevant information that dilutes the answer. The overlap ensures no
> information is lost at chunk boundaries. These are widely accepted default values in RAG systems.

**Q: How would you scale this system?**
> For production: (1) Replace ChromaDB with Pinecone or Weaviate for managed hosting. (2) Add
> caching for frequently asked questions. (3) Use async processing for concurrent requests.
> (4) Implement a feedback loop to improve retrieval quality. (5) Add monitoring and analytics.

---

## 12. Implementation Prompts

Use these prompts with AI coding assistants (Claude, Gemini, Copilot) to implement each phase.

### Phase 1 – Data Collection Prompt

```
I am building a RAG chatbot that answers questions from GitLab's handbook.

Write a Python script (scripts/scraper.py) that:
1. Clones the GitLab handbook repository from
   https://gitlab.com/gitlab-com/content-sites/handbook.git
2. Recursively finds all .md files in the cloned repository
3. Reads each file and saves the content with metadata (file path, title extracted
   from the first heading) as a JSON file in data/raw/
4. Creates a manifest.json listing all files with their paths and titles
5. Handles errors gracefully and logs progress

Use Python standard libraries + gitpython for cloning.
Add proper logging and error handling.
```

### Phase 2 – Data Processing Prompt

```
I have raw Markdown files from GitLab's handbook in data/raw/.

Write a Python script (scripts/preprocessor.py) that:
1. Reads all .md files from data/raw/
2. Removes YAML frontmatter (content between --- markers)
3. Removes HTML tags
4. Removes image references and link URLs (keep link text)
5. Removes navigation elements and repetitive boilerplate
6. Normalizes whitespace and fixes encoding issues
7. Filters out files with less than 100 characters of content
8. Saves cleaned text files to data/processed/ with the same directory structure
9. Generates a processing report showing: files processed, files skipped, total characters

Use BeautifulSoup for HTML cleaning and regex for text normalization.
```

### Phase 3 – Embedding Generation Prompt

```
I have cleaned text files in data/processed/.

Write a Python script (scripts/chunker.py) that:
1. Reads all text files from data/processed/
2. Uses LangChain's RecursiveCharacterTextSplitter with:
   - chunk_size=1000
   - chunk_overlap=200
3. For each chunk, stores:
   - chunk_id (unique identifier)
   - text (the chunk content)
   - source_file (original file path)
   - source_url (constructed GitLab handbook URL)
   - chunk_index (position in the original document)
4. Saves all chunks as a JSON file in data/chunks/all_chunks.json
5. Prints summary: total documents, total chunks, average chunk size

Use LangChain text splitters. Include proper error handling.
```

### Phase 4 – Vector Database Setup Prompt

```
I have document chunks in data/chunks/all_chunks.json.

Write a Python script (scripts/build_vectordb.py) that:
1. Reads chunks from the JSON file
2. Creates a ChromaDB persistent client at ./vector_db
3. Creates a collection called "gitlab_handbook" with cosine similarity
4. Adds all chunks in batches of 100 (ChromaDB has batch limits)
5. Each document should have:
   - document text
   - metadata: source_url, source_file, chunk_index
   - unique ID
6. After insertion, runs 3 test queries and prints results to verify
7. Prints: total documents stored, collection name, storage path

Use chromadb PersistentClient. Use ChromaDB's built-in embedding function
(default-embedding-function) OR Google's text-embedding-004.
Add progress bars using tqdm.
```

### Phase 5 – Backend Development Prompt

```
Build a FastAPI backend for my RAG chatbot.

Create backend/app.py with:
1. POST /ask endpoint that accepts {"question": "string"}
2. GET /health endpoint
3. The /ask endpoint should:
   - Query ChromaDB for top 5 relevant chunks
   - Build a prompt with system instruction + context + question
   - Call Google Gemini (gemini-1.5-flash) API
   - Return {"answer": "...", "sources": [...], "chunks_used": 5}
4. Use environment variables for API keys (python-dotenv)
5. Add CORS middleware for frontend access
6. Add input validation and error handling
7. Add request/response logging

Create backend/rag_engine.py with:
1. RAGEngine class that initializes ChromaDB and Gemini
2. retrieve(question, n_results=5) method
3. generate(question, context) method
4. ask(question) method that combines retrieve + generate

Create backend/config.py with:
1. Settings class using pydantic-settings
2. Load from .env file
3. Fields: GEMINI_API_KEY, CHROMA_DB_PATH, COLLECTION_NAME, TOP_K, CHUNK_SIZE

Use FastAPI, chromadb, google-generativeai, python-dotenv, pydantic-settings.
```

### Phase 6 – Frontend Development Prompt

```
Build a Streamlit chat UI for my RAG chatbot.

Create frontend/streamlit_app.py with:
1. Page config: title="GitLab Handbook AI", icon="🦊", layout="wide"
2. Sidebar with:
   - Project description
   - Example questions (clickable buttons)
   - "How it works" expandable section
3. Main chat area with:
   - Chat message history using st.chat_message
   - User input using st.chat_input
   - Loading spinner while generating answers
   - Source citations as expandable sections
   - Retrieved context in a separate expandable section
4. Session state for chat history
5. Direct integration with RAG engine (no API calls needed for Streamlit deployment)
6. Error handling with user-friendly messages
7. Custom CSS for better appearance

Use streamlit, chromadb, google-generativeai.
The app should directly import and use the RAG engine, not call an API.
```

### Phase 7 – Deployment Prompt

```
Help me deploy my Streamlit RAG chatbot to Streamlit Community Cloud.

I need:
1. A requirements.txt with all dependencies and pinned versions
2. A .streamlit/config.toml with theme configuration
3. A .streamlit/secrets.toml template (for local testing)
4. Instructions for:
   - Setting up the GitHub repository
   - Configuring secrets on Streamlit Cloud
   - Handling the vector database (should be pre-built and included in the repo)
   - Troubleshooting common deployment issues
5. A GitHub Actions workflow (.github/workflows/deploy.yml) that:
   - Runs on push to main
   - Runs basic tests
   - Validates the app can import without errors

Provide step-by-step deployment instructions.
```

### Full Project Setup Prompt

```
Set up the complete project structure for a GitLab Handbook RAG Chatbot.

Create:
1. All directories: backend/, frontend/, data/raw/, data/processed/, data/chunks/,
   vector_db/, scripts/, config/, tests/, .streamlit/
2. requirements.txt with: streamlit, chromadb, google-generativeai, langchain,
   langchain-text-splitters, beautifulsoup4, requests, python-dotenv, fastapi,
   uvicorn, tqdm
3. .env.example with placeholder API keys
4. .gitignore that ignores: .env, vector_db/, data/raw/, __pycache__/,
   *.pyc, .streamlit/secrets.toml
5. README.md with project overview, setup instructions, and usage guide
6. config/settings.yaml with default configuration values

Use Python 3.11+. Include comments explaining each configuration option.
```

---

## Appendix: Quick Reference Commands

```bash
# 1. Clone the project and set up environment
git clone <your-repo-url>
cd gitlab-handbook-chatbot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# 3. Run the data pipeline
python scripts/scraper.py          # Collect data
python scripts/preprocessor.py     # Clean data
python scripts/chunker.py          # Chunk documents
python scripts/build_vectordb.py   # Build vector database

# 4. Run the chatbot locally
streamlit run frontend/streamlit_app.py

# 5. (Optional) Run the FastAPI backend
uvicorn backend.app:app --reload --port 8000

# 6. Run tests
pytest tests/
```

---

> **💡 Final Tip:** Build this project incrementally. Get each phase working before moving to the
> next. Start with Phase 4 (if using the Git clone approach for data) and get a working command-line
> chatbot before adding the UI. A simple working chatbot is more impressive than a complex broken one.

---

*Document generated for internship assignment – GitLab Handbook AI Chatbot*
*Architecture: Retrieval Augmented Generation (RAG)*
*Last updated: March 2026*
```

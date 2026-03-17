# 🦊 GitLab Handbook AI Chatbot

An AI-powered chatbot that answers questions about GitLab's Handbook and Direction pages using **Retrieval Augmented Generation (RAG)**.

## 🏗️ Architecture

```
User Question → Embedding → Vector Search (ChromaDB)
→ Retrieve Context → Gemini LLM → Answer + Sources
```

**Tech Stack:** Python · Google Gemini · ChromaDB · FastAPI · Streamlit · LangChain

## 📁 Project Structure

```
Joveo/
├── backend/              # FastAPI server + RAG engine
│   ├── app.py            # API endpoints (/ask, /health)
│   ├── rag_engine.py     # Core RAG pipeline
│   ├── embeddings.py     # Embedding utilities
│   └── config.py         # Settings management
├── frontend/             # Streamlit chat UI
│   └── streamlit_app.py
├── scripts/              # Data pipeline scripts
│   ├── scraper.py        # Phase 1: Data collection
│   ├── preprocessor.py   # Phase 2: Data cleaning
│   ├── chunker.py        # Phase 3: Document chunking
│   └── build_vectordb.py # Phase 4: Vector DB creation
├── data/                 # Raw, processed, and chunked data
├── vector_db/            # ChromaDB storage (auto-generated)
├── config/               # Configuration files
│   └── settings.yaml     # App settings
├── tests/                # Unit tests
├── .env.example          # Environment variable template
├── requirements.txt      # Python dependencies
└── TECHNICAL_BLUEPRINT.md # Full design document
```

## 🚀 Quick Start

### 1. Set up the environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure API keys
```bash
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY
```
Get your free API key at: https://aistudio.google.com/apikey

### 3. Run the data pipeline
```bash
python scripts/scraper.py          # Collect GitLab handbook data
python scripts/preprocessor.py     # Clean the data
python scripts/chunker.py          # Split into chunks
python scripts/build_vectordb.py   # Build vector database
```

### 4. Launch the chatbot
```bash
python -m streamlit run frontend/streamlit_app.py
```

### 5. (Optional) Run the API server
```bash
uvicorn backend.app:app --reload --port 8000
```
API docs available at: http://localhost:8000/docs

## 📖 Documentation

See [TECHNICAL_BLUEPRINT.md](TECHNICAL_BLUEPRINT.md) for the complete technical design document.

## 📝 License

This project is built for educational / internship purposes.

## ⚠️ Note: 

Vector database and raw data are excluded due to size constraints. Run scripts to regenerate locally.

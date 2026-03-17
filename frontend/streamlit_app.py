"""
streamlit_app.py -- Streamlit Chat UI (Phase 6)
=================================================
A polished chat interface for the GitLab Handbook AI Chatbot.
Integrates directly with the RAG engine (no separate API server needed).

Run locally with:
    streamlit run frontend/streamlit_app.py

Features:
    - Conversational chat interface with message history
    - Loading spinner while generating answers
    - Source citations with expandable details
    - Retrieved context transparency
    - Sidebar with example questions and info
    - Session-based conversation memory
"""

import sys
import time
from pathlib import Path

import streamlit as st

# -- Add project root to path so we can import backend modules --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ================================================================
# Page Configuration (MUST be the first Streamlit command)
# ================================================================
st.set_page_config(
    page_title="GitLab Handbook AI",
    page_icon="https://about.gitlab.com/nuxt-images/ico/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ================================================================
# Custom CSS for a polished look
# ================================================================
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #FC6D26 0%, #E24329 50%, #FCA326 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
    }

    /* Source card styling */
    .source-card {
        background-color: #1E1E2E;
        border: 1px solid #313244;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 0.85rem;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0D1117;
    }

    /* Example question buttons */
    .stButton > button {
        border: 1px solid #30363D;
        border-radius: 8px;
        background-color: #161B22;
        color: #C9D1D9;
        text-align: left;
        padding: 8px 12px;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #FC6D26;
        background-color: #1C2128;
    }

    /* Footer */
    .footer-text {
        color: #6E7681;
        font-size: 0.75rem;
        text-align: center;
        padding-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ================================================================
# RAG Engine Initialization (cached so it loads only once)
# ================================================================
@st.cache_resource(show_spinner="Loading RAG Engine...")
def load_rag_engine():
    """Load the RAG engine once and cache it across all reruns."""
    try:
        from backend.rag_engine import RAGEngine
        engine = RAGEngine()
        return engine
    except Exception as e:
        return None


def get_engine_error_message():
    """Return a helpful message when the RAG engine fails to load."""
    return (
        "The RAG Engine could not be initialized. This usually means:\n\n"
        "1. **Vector database not built yet** -- Run the data pipeline first:\n"
        "   ```\n"
        "   python scripts/scraper.py\n"
        "   python scripts/preprocessor.py\n"
        "   python scripts/chunker.py\n"
        "   python scripts/build_vectordb.py\n"
        "   ```\n"
        "2. **Missing API key** -- Make sure `GEMINI_API_KEY` is set in your `.env` file.\n"
        "3. **Dependencies not installed** -- Run `pip install -r requirements.txt`"
    )


# ================================================================
# Sidebar
# ================================================================
with st.sidebar:
    # Logo and title
    st.markdown(
        '<p style="text-align:center; font-size: 3rem; margin-bottom: 0;">🦊</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h2 style="text-align:center; margin-top:0;">GitLab Handbook AI</h2>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Description
    st.markdown(
        "Ask questions about GitLab's **handbook**, **values**, "
        "**engineering practices**, and **company policies**."
    )
    st.markdown("")

    # Example questions
    st.markdown("### Example Questions")
    example_questions = [
        "What are GitLab's CREDIT values?",
        "How does GitLab approach remote work?",
        "What is GitLab's hiring process?",
        "How does async communication work at GitLab?",
        "What is GitLab's approach to transparency?",
        "How are performance reviews done at GitLab?",
    ]
    for q in example_questions:
        if st.button(q, key=f"ex_{hash(q)}", use_container_width=True):
            st.session_state["pending_question"] = q

    st.markdown("---")

    # How it works
    with st.expander("How It Works", expanded=False):
        st.markdown("""
        **RAG (Retrieval Augmented Generation)**

        1. Your question is converted into a vector embedding
        2. ChromaDB searches for similar handbook sections
        3. The top 5 relevant chunks are retrieved
        4. Context + question are sent to Google Gemini
        5. Gemini generates an answer grounded in the handbook
        """)

    # Settings
    with st.expander("Settings", expanded=False):
        show_context = st.toggle("Show retrieved context", value=False)
        show_chunk_count = st.toggle("Show chunk count", value=True)

    st.markdown("---")

    # Clear chat button
    if st.button("Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()

    # Footer
    st.markdown(
        '<p class="footer-text">Powered by Google Gemini + ChromaDB<br>'
        "Built with Streamlit</p>",
        unsafe_allow_html=True,
    )


# ================================================================
# Main Chat Area
# ================================================================

# Header
st.markdown('<p class="main-header">GitLab Handbook Chatbot</p>', unsafe_allow_html=True)
st.caption("Ask me anything about GitLab's handbook and direction pages!")
st.markdown("")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- [{src}]({src})")

        # Show context if enabled
        if msg["role"] == "assistant" and msg.get("context") and show_context:
            with st.expander("📄 Retrieved Context", expanded=False):
                for idx, chunk in enumerate(msg["context"]):
                    st.markdown(f"**Chunk {idx + 1}:**")
                    st.text(chunk[:500] + ("..." if len(chunk) > 500 else ""))
                    if idx < len(msg["context"]) - 1:
                        st.markdown("---")


# ================================================================
# Handle User Input
# ================================================================

# Check for pending question from sidebar example buttons
pending = st.session_state.pop("pending_question", None)

# Chat input box
user_input = st.chat_input("Ask a question about GitLab...")

# Use whichever input is available
question = pending or user_input

if question:
    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    # Generate the answer
    with st.chat_message("assistant"):
        # Loading spinner
        with st.spinner("Searching the handbook..."):
            engine = load_rag_engine()

            if engine is None:
                error_msg = get_engine_error_message()
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
            else:
                try:
                    # Call the RAG engine
                    start_time = time.time()
                    result = engine.ask(question)
                    elapsed = time.time() - start_time

                    answer = result.get("answer", "Sorry, I could not generate an answer.")
                    sources = result.get("sources", [])
                    context = result.get("context", [])
                    num_chunks = result.get("num_chunks_used", 0)

                    # Display the answer
                    st.markdown(answer)

                    # Metadata line
                    if show_chunk_count:
                        st.caption(
                            f"Used {num_chunks} chunks | "
                            f"Response time: {elapsed:.1f}s"
                        )

                    # Source citations
                    if sources:
                        with st.expander("📚 Sources", expanded=False):
                            for src in sources:
                                st.markdown(f"- [{src}]({src})")

                    # Retrieved context (if setting is enabled)
                    if context and show_context:
                        with st.expander("📄 Retrieved Context", expanded=False):
                            for idx, chunk in enumerate(context):
                                st.markdown(f"**Chunk {idx + 1}:**")
                                st.text(
                                    chunk[:500]
                                    + ("..." if len(chunk) > 500 else "")
                                )
                                if idx < len(context) - 1:
                                    st.markdown("---")

                    # Save assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "context": context,
                    })

                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                    })

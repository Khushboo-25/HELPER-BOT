"""
rag_engine.py – Core RAG (Retrieval Augmented Generation) Engine
=================================================================
This is the heart of the chatbot. It:
  1. Takes a user's question
  2. Retrieves relevant document chunks from ChromaDB
  3. Builds a prompt with the retrieved context
  4. Sends the prompt to Google Gemini
  5. Returns the generated answer with source citations

Usage:
    from backend.rag_engine import RAGEngine
    engine = RAGEngine()
    result = engine.ask("What are GitLab's values?")
    print(result["answer"])
"""

import google.generativeai as genai
import chromadb
from backend.config import settings


class RAGEngine:
    """
    RAG Engine that connects ChromaDB (retrieval) with Gemini (generation).

    Attributes:
        collection: ChromaDB collection containing document embeddings.
        llm_model: Google Gemini GenerativeModel instance.
    """

    def __init__(self):
        """Initialize the RAG engine by connecting to ChromaDB and Gemini."""
        # Configure Gemini API
        genai.configure(api_key=settings.GEMINI_API_KEY)

        # Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        # Initialize Gemini LLM
        self.llm_model = genai.GenerativeModel(settings.LLM_MODEL)

    def retrieve(self, question: str, n_results: int = None) -> dict:
        """
        Retrieve relevant document chunks from the vector database.

        We compute the query embedding using the same model that was used to
        populate the collection.  ChromaDB will otherwise try to embed the
        text itself using its own default, which may have a different
        dimensionality and result in an `InvalidArgumentError`.

        Args:
            question: The user's question.
            n_results: Number of chunks to retrieve (default: settings.TOP_K).

        Returns:
            ChromaDB query results containing documents, metadatas, and distances.
        """
        if n_results is None:
            n_results = settings.TOP_K

        # generate embedding for the question
        embedding_resp = genai.embed_content(
            settings.EMBEDDING_MODEL,
            question,
        )
        query_emb = embedding_resp.get("embedding")

        results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
        )
        return results

    def build_prompt(self, question: str, context_chunks: list[str]) -> str:
        """
        Build the prompt that will be sent to the LLM.

        The prompt includes:
        - A system instruction telling the LLM to only use provided context
        - The retrieved context chunks
        - The user's question

        Args:
            question: The user's question.
            context_chunks: List of relevant text chunks from the handbook.

        Returns:
            The formatted prompt string.
        """
        context = "\n\n---\n\n".join(context_chunks)

        prompt = f"""You are a helpful AI assistant that answers questions about GitLab
using ONLY the provided context from GitLab's official handbook.

RULES:
- Answer ONLY based on the context provided below.
- If the context doesn't contain enough information, say "I don't have enough
  information from the handbook to answer this question."
- Cite which section the information comes from.
- Be concise but thorough.
- Use bullet points for lists.
- If the question is not related to GitLab, politely redirect the user.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
        return prompt

    def generate(self, prompt: str) -> str:
        """
        Send the prompt to Google Gemini and get the generated answer.

        Args:
            prompt: The complete prompt with context and question.

        Returns:
            The LLM's generated answer as a string.
        """
        response = self.llm_model.generate_content(prompt)
        return response.text

    def ask(self, question: str) -> dict:
        """
        Full RAG pipeline: retrieve context → build prompt → generate answer.

        This is the main method you call to get an answer to a question.

        Args:
            question: The user's natural language question.

        Returns:
            A dictionary with:
            - answer (str): The generated answer.
            - sources (list[str]): Source URLs for the retrieved chunks.
            - num_chunks_used (int): Number of chunks used as context.
            - context (list[str]): The actual retrieved text chunks.
        """
        # Step 1: Retrieve relevant chunks
        results = self.retrieve(question)

        # Step 2: Extract documents and metadata
        documents = results["documents"][0] if results["documents"] else []
        metadatas = results["metadatas"][0] if results["metadatas"] else []

        if not documents:
            return {
                "answer": "I couldn't find any relevant information in the handbook.",
                "sources": [],
                "num_chunks_used": 0,
                "context": []
            }

        # Step 3: Build prompt with retrieved context
        prompt = self.build_prompt(question, documents)

        # Step 4: Generate answer using LLM
        answer = self.generate(prompt)

        # Step 5: Extract unique source URLs
        sources = list(set(
            meta.get("source", "Unknown source")
            for meta in metadatas
        ))

        return {
            "answer": answer,
            "sources": sources,
            "num_chunks_used": len(documents),
            "context": documents
        }

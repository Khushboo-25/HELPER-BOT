"""
embeddings.py – Embedding Generation Utilities
================================================
Provides functions to generate text embeddings using Google's
text-embedding-004 model via the Gemini API.

These embeddings convert text into numerical vectors (lists of numbers)
that capture semantic meaning. Similar texts produce similar vectors.
"""

import google.generativeai as genai
from backend.config import settings


def configure_genai():
    """Configure the Google Generative AI SDK with the API key."""
    genai.configure(api_key=settings.GEMINI_API_KEY)


def get_embedding(text: str) -> list[float]:
    """
    Generate an embedding vector for a given text.

    Args:
        text: The text string to embed.

    Returns:
        A list of floats representing the embedding vector (768 dimensions).

    Example:
        >>> embedding = get_embedding("What is GitLab's remote work policy?")
        >>> len(embedding)
        768
    """
    configure_genai()
    result = genai.embed_content(
        model=settings.EMBEDDING_MODEL,
        content=text
    )
    return result["embedding"]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.

    Args:
        texts: A list of text strings to embed.

    Returns:
        A list of embedding vectors (one per input text).
    """
    configure_genai()
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=settings.EMBEDDING_MODEL,
            content=text
        )
        embeddings.append(result["embedding"])
    return embeddings

"""
build_vectordb.py – Vector Database Builder
=============================================
Reads chunked documents and inserts them into Supabase (pgvector)
with embeddings and metadata.

Usage:
python scripts/build_vectordb.py

Input: data/chunks/all_chunks.json
Output: Supabase table "documents"
"""
import time
import json
import logging
import sys
import os
from pathlib import Path

# make sure we can import backend modules when running from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import google.generativeai as genai
import chromadb
from backend.config import settings


# -- Setup logging (Windows-safe) --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler("logs/build_vectordb.log", encoding="utf-8")],
)
logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_FILE = PROJECT_ROOT / "data" / "chunks" / "all_chunks.json"

BATCH_SIZE = 100  # Number of chunks to add per batch


def load_chunks() -> list:
    """Load all chunk entries from the JSON file."""
    if not CHUNKS_FILE.exists():
        logger.error("Chunks file not found: %s", CHUNKS_FILE)
        sys.exit(1)

    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks = data.get("chunks", [])
    logger.info("Loaded %d chunks from disk", len(chunks))
    return chunks


def init_chroma() -> chromadb.api.models.Collection:
    """Create or open the persistent ChromaDB collection.

    If the collection already contains data, it will be cleared to avoid
    duplicates. This makes the operation idempotent.
    """
    client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
    collection = client.get_or_create_collection(
        name=settings.CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # clear existing contents so we can rebuild from scratch
    try:
        collection.delete()  # deletes all records
        logger.info("Existing collection cleared")
    except Exception:
        # some versions of chromadb raise if collection is empty
        pass

    return collection


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert a list of strings into embeddings via Gemini.

    Google’s `generativeai` client provides `embed_content` which accepts an
    iterable of text strings and returns a dictionary with a single key
    "embedding" containing the list of vectors.
    """
    result = genai.embed_content(
        settings.EMBEDDING_MODEL,
        texts,
    )
    # result may look like {"embedding": [[...], [...], ...]}
    embeddings = result.get("embedding")
    if embeddings is None:
        raise RuntimeError("Embedding API returned no embeddings")
    return embeddings


def add_chunks_to_collection(collection, chunks: list):
    """Insert chunks into the ChromaDB collection in batches."""
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        ids = [c["chunk_id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = [
            {
                "source": c.get("source_url", ""),
                "source_file": c.get("source_file", ""),
                "chunk_index": c.get("chunk_index", 0),
            }
            for c in batch
        ]
        try:
            embeddings = embed_texts(texts)
        except Exception as e:
            logger.error("Embedding failed on batch %d: %s", i + 1, e)
            logger.error("Quota may have been exceeded or an API error occurred.\n" \
                         "You can re-run this script later after obtaining more quota or\n" \
                         "adjusting your embedding model.")
            return i

        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        logger.info("Inserted batch %d - %d", i + 1, min(i + BATCH_SIZE, total))
        time.sleep(2)   # prevents API rate limit
    logger.info("Finished inserting %d chunks", total)
    return total

def run_tests(collection):
    """Perform a few sanity-check queries against the newly populated DB."""
    test_queries = [
        "What are GitLab's values?",
        "How does GitLab handle remote work?",
        "Explain GitLab's hiring process."
    ]
    for q in test_queries:
        results = collection.query(query_texts=[q], n_results=3)
        docs = results.get("documents", [[]])[0]
        logger.info("Query: %s -> %d results", q, len(docs))
        for d in docs:
            logger.info("  %s", (d[:80] + "...") if len(d) > 80 else d)


def main():
    logger.info("%s", "=" * 60)
    logger.info("Phase 4: Vector Database Setup")
    logger.info("%s", "=" * 60)

    # Step 0: configure Gemini (for embeddings)
    genai.configure(api_key=settings.GEMINI_API_KEY)

    # Load chunks
    chunks = load_chunks()
    chunks=chunks[:3000]
    if not chunks:
        logger.error("No chunks available to insert into vector DB.")
        sys.exit(1)

    # Initialize / clear chroma collection
    collection = init_chroma()

    # Insert all chunks
    num_inserted = add_chunks_to_collection(collection, chunks)

    # Run simple test queries only if we actually managed to insert at least one
    if num_inserted > 0:
        run_tests(collection)
    else:
        logger.warning("No chunks inserted into vector DB; skipping test queries.")

    logger.info("Phase 4 complete. Vector database available at %s", settings.CHROMA_DB_PATH)


if __name__ == "__main__":
    main()


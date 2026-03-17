"""
chunker.py -- Document Chunking (Phase 3)
==========================================
Reads cleaned text files from data/processed/, splits them into
overlapping chunks using LangChain's RecursiveCharacterTextSplitter,
and saves all chunks with metadata to data/chunks/all_chunks.json.

Usage:
    python scripts/chunker.py

Input:  data/processed/  (cleaned .txt files from Phase 2)
Output: data/chunks/all_chunks.json
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter

# -- Setup Logging (Windows-safe: no emojis) --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/chunker.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# -- Configuration --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
LOGS_DIR = PROJECT_ROOT / "logs"

CHUNK_SIZE = 1000       # Characters per chunk (~250 words)
CHUNK_OVERLAP = 200     # Overlap between consecutive chunks
MIN_CHUNK_LENGTH = 50   # Discard chunks shorter than this


def ensure_directories():
    """Create required output directories if they don't exist."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Input directory:  %s", PROCESSED_DIR)
    logger.info("Output directory: %s", CHUNKS_DIR)


def create_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create and return a configured text splitter.

    The splitter tries to split at natural boundaries in this order:
      1. Double newline (paragraph breaks)
      2. Single newline (line breaks)
      3. Period + space (sentence boundaries)
      4. Space (word boundaries)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
        is_separator_regex=False,
    )
    logger.info("Text splitter configured: chunk_size=%d, chunk_overlap=%d",
                CHUNK_SIZE, CHUNK_OVERLAP)
    return splitter


def build_source_url(relative_path: str) -> str:
    """
    Convert a relative file path into an approximate GitLab handbook URL.

    Example:
        content/handbook/values/_index.txt
        -> https://handbook.gitlab.com/handbook/values/

    Args:
        relative_path: Relative path from data/processed/.

    Returns:
        Approximate source URL string.
    """
    # Normalize path separators
    path = relative_path.replace("\\", "/")

    # Remove file extension
    for ext in [".txt", ".md", ".mdx"]:
        if path.endswith(ext):
            path = path[: -len(ext)]
            break

    # Remove _index or index suffix (common in Hugo-based sites)
    if path.endswith("/_index") or path.endswith("/index"):
        path = path.rsplit("/", 1)[0]

    # Remove leading "content/" if present (Hugo content directory)
    if path.startswith("content/"):
        path = path[len("content/"):]

    # Build URL
    base_url = "https://handbook.gitlab.com"
    if not path.startswith("/"):
        path = "/" + path

    return base_url + path + "/"


def chunk_all_files(splitter: RecursiveCharacterTextSplitter) -> list:
    """
    Read all processed text files, split them into chunks, and
    return a list of chunk dicts with metadata.

    Args:
        splitter: Configured RecursiveCharacterTextSplitter.

    Returns:
        List of chunk dicts with keys:
            chunk_id, text, source_file, source_url, chunk_index
    """
    # Find all text files in processed directory
    text_files = sorted(PROCESSED_DIR.rglob("*.txt"))
    # Exclude the processing report
    text_files = [f for f in text_files if f.name != "processing_report.json"]

    logger.info("Found %d text files to chunk", len(text_files))

    all_chunks = []
    global_chunk_id = 0
    files_with_chunks = 0
    files_skipped = 0

    for i, text_file in enumerate(text_files):
        try:
            # Read the cleaned content
            content = text_file.read_text(encoding="utf-8", errors="replace")

            # Skip files that are too short
            if len(content.strip()) < MIN_CHUNK_LENGTH:
                files_skipped += 1
                continue

            # Split into chunks
            chunks = splitter.split_text(content)

            if not chunks:
                files_skipped += 1
                continue

            # Build metadata
            relative_path = str(text_file.relative_to(PROCESSED_DIR)).replace("\\", "/")
            source_url = build_source_url(relative_path)

            files_with_chunks += 1

            for chunk_index, chunk_text in enumerate(chunks):
                # Skip very short chunks
                if len(chunk_text.strip()) < MIN_CHUNK_LENGTH:
                    continue

                chunk_entry = {
                    "chunk_id": f"chunk_{global_chunk_id}",
                    "text": chunk_text,
                    "source_file": relative_path,
                    "source_url": source_url,
                    "chunk_index": chunk_index,
                }
                all_chunks.append(chunk_entry)
                global_chunk_id += 1

            # Log progress every 500 files
            if (i + 1) % 500 == 0:
                logger.info("  Processed %d / %d files (%d chunks so far)...",
                            i + 1, len(text_files), len(all_chunks))

        except Exception as e:
            logger.warning("Error processing %s: %s", text_file.name, e)

    logger.info("[OK] Chunking complete: %d chunks from %d files",
                len(all_chunks), files_with_chunks)
    if files_skipped > 0:
        logger.info("[SKIP] Skipped %d files (too short or empty)", files_skipped)

    return all_chunks


def save_chunks(chunks: list):
    """
    Save all chunks to data/chunks/all_chunks.json.

    Args:
        chunks: List of chunk dicts.
    """
    output_path = CHUNKS_DIR / "all_chunks.json"

    output_data = {
        "metadata": {
            "total_chunks": len(chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "created_at": datetime.now().isoformat(),
        },
        "chunks": chunks,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("Chunks saved: %s (%.1f MB)", output_path, file_size_mb)


def compute_stats(chunks: list) -> dict:
    """Compute summary statistics about the chunks."""
    if not chunks:
        return {"total_chunks": 0}

    lengths = [len(c["text"]) for c in chunks]
    unique_sources = set(c["source_file"] for c in chunks)

    return {
        "total_chunks": len(chunks),
        "unique_source_files": len(unique_sources),
        "avg_chunk_length": round(sum(lengths) / len(lengths), 1),
        "min_chunk_length": min(lengths),
        "max_chunk_length": max(lengths),
        "total_characters": sum(lengths),
    }


def main():
    """Run the full document chunking pipeline."""
    logger.info("=" * 60)
    logger.info("Phase 3: Document Chunking")
    logger.info("=" * 60)

    # Step 1: Ensure directories exist
    ensure_directories()

    # Check that processed data exists
    if not PROCESSED_DIR.exists() or not any(PROCESSED_DIR.iterdir()):
        logger.error("[FAILED] No processed data found in %s", PROCESSED_DIR)
        logger.error("Run 'python scripts/preprocessor.py' first (Phase 2).")
        sys.exit(1)

    # Step 2: Create text splitter
    splitter = create_splitter()

    # Step 3: Chunk all files
    chunks = chunk_all_files(splitter)

    if not chunks:
        logger.error("[FAILED] No chunks generated!")
        sys.exit(1)

    # Step 4: Save chunks to JSON
    save_chunks(chunks)

    # Step 5: Print summary statistics
    stats = compute_stats(chunks)

    logger.info("=" * 60)
    logger.info("[OK] Phase 3 Complete: Document Chunking Finished!")
    logger.info("   Total chunks:        %d", stats["total_chunks"])
    logger.info("   Unique source files:  %d", stats["unique_source_files"])
    logger.info("   Avg chunk length:     %s chars", stats["avg_chunk_length"])
    logger.info("   Min chunk length:     %d chars", stats["min_chunk_length"])
    logger.info("   Max chunk length:     %d chars", stats["max_chunk_length"])
    logger.info("   Total characters:     %s", f"{stats['total_characters']:,}")
    logger.info("   Output: data/chunks/all_chunks.json")
    logger.info("=" * 60)
    logger.info("Next step: Run 'python scripts/build_vectordb.py' (Phase 4)")


if __name__ == "__main__":
    main()

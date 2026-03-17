"""
preprocessor.py -- Data Cleaning & Preprocessing (Phase 2)
===========================================================
Reads raw Markdown files from data/raw/, cleans them by removing
YAML frontmatter, HTML tags, links, images, and normalizing whitespace,
then saves clean text files to data/processed/.

Usage:
    python scripts/preprocessor.py

Input:  data/raw/        (raw Markdown files from Phase 1)
Output: data/processed/  (cleaned plain-text files)
"""

import os
import re
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# -- Setup Logging (Windows-safe: no emojis) --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/preprocessor.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# -- Configuration --
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOGS_DIR = PROJECT_ROOT / "logs"

# Minimum character count for a file to be kept after cleaning
MIN_CONTENT_LENGTH = 100


def ensure_directories():
    """Create required output directories if they don't exist."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Input directory:  %s", RAW_DIR)
    logger.info("Output directory: %s", PROCESSED_DIR)


def remove_yaml_frontmatter(text: str) -> str:
    """
    Remove YAML frontmatter enclosed between --- markers at the
    start of the file.

    Example:
        ---
        title: Some Page
        description: A description
        ---
        Actual content here...

    Returns the text without the frontmatter block.
    """
    # Match --- at the very start, then any content, then closing ---
    pattern = r"^---\s*\n.*?\n---\s*\n"
    cleaned = re.sub(pattern, "", text, count=1, flags=re.DOTALL)
    return cleaned


def remove_html_tags(text: str) -> str:
    """
    Remove all HTML tags from the text.
    Handles self-closing tags, opening/closing tags, and HTML comments.
    """
    # Remove HTML comments <!-- ... -->
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)

    # Remove HTML tags like <div>, </div>, <br/>, <img ... />
    text = re.sub(r"<[^>]+>", "", text)

    return text


def remove_markdown_links(text: str) -> str:
    """
    Remove Markdown link syntax but keep the link text.

    [link text](url)        -> link text
    [link text][reference]  -> link text
    """
    # Inline links: [text](url) or [text](url "title")
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)

    # Reference links: [text][ref]
    text = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", text)

    # Bare URLs (http/https)
    text = re.sub(r"https?://\S+", "", text)

    return text


def remove_markdown_images(text: str) -> str:
    """
    Remove Markdown image syntax entirely.

    ![alt text](url)  -> (removed)
    """
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", "", text)
    return text


def remove_reference_definitions(text: str) -> str:
    """
    Remove Markdown reference-style link definitions.

    [ref]: url "title"
    """
    text = re.sub(r"^\[[^\]]+\]:\s+.*$", "", text, flags=re.MULTILINE)
    return text


def clean_markdown_formatting(text: str) -> str:
    """
    Simplify Markdown formatting while preserving readable structure.
    - Keep headings as plain text with a newline prefix
    - Remove bold/italic markers
    - Remove code block fences
    - Remove horizontal rules
    """
    # Remove code block fences (``` or ~~~) but keep the content inside
    text = re.sub(r"^```\w*\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^~~~\w*\s*$", "", text, flags=re.MULTILINE)

    # Remove bold/italic markers: **text**, __text__, *text*, _text_
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    # Avoid removing underscores in words like variable_name
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)

    # Remove strikethrough: ~~text~~
    text = re.sub(r"~~(.+?)~~", r"\1", text)

    # Remove horizontal rules (---, ***, ___)
    text = re.sub(r"^[\-\*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # Simplify headings: ## Heading -> Heading
    text = re.sub(r"^#+\s+", "", text, flags=re.MULTILINE)

    # Remove inline code backticks but keep content
    text = re.sub(r"`([^`]+)`", r"\1", text)

    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace throughout the text:
    - Convert tabs to spaces
    - Remove trailing whitespace on each line
    - Collapse 3+ consecutive blank lines into 2
    - Strip leading/trailing whitespace from the whole text
    """
    # Convert tabs to spaces
    text = text.replace("\t", "    ")

    # Remove trailing whitespace on each line
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)

    # Collapse 3+ consecutive newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def remove_table_formatting(text: str) -> str:
    """
    Simplify Markdown table separator rows (|---|---|).
    Keep the header and data rows as plain text.
    """
    # Remove table separator rows: | --- | --- | or |:---:|:---:|
    text = re.sub(r"^\|[\s\-:]+\|[\s\-:|]*$", "", text, flags=re.MULTILINE)

    # Remove leading/trailing pipe characters from table rows
    text = re.sub(r"^\|(.+)\|$", r"\1", text, flags=re.MULTILINE)

    return text


def clean_file(content: str) -> str:
    """
    Apply the full cleaning pipeline to a single file's content.

    Pipeline order:
        1. Remove YAML frontmatter
        2. Remove HTML tags
        3. Remove Markdown images
        4. Remove Markdown links (keep text)
        5. Remove reference definitions
        6. Simplify table formatting
        7. Clean Markdown formatting (bold, italic, code fences)
        8. Normalize whitespace

    Args:
        content: Raw Markdown file content.

    Returns:
        Cleaned plain text.
    """
    text = content

    # Step 1: Remove YAML frontmatter
    text = remove_yaml_frontmatter(text)

    # Step 2: Remove HTML tags and comments
    text = remove_html_tags(text)

    # Step 3: Remove images
    text = remove_markdown_images(text)

    # Step 4: Remove links (keep link text)
    text = remove_markdown_links(text)

    # Step 5: Remove reference-style link definitions
    text = remove_reference_definitions(text)

    # Step 6: Simplify table formatting
    text = remove_table_formatting(text)

    # Step 7: Clean Markdown formatting
    text = clean_markdown_formatting(text)

    # Step 8: Normalize whitespace
    text = normalize_whitespace(text)

    return text


def process_all_files():
    """
    Process all Markdown files in data/raw/ and save cleaned
    versions to data/processed/.

    Returns:
        A report dict with processing statistics.
    """
    # Find all markdown files
    extensions = {".md", ".mdx", ".markdown"}
    raw_files = [
        f for f in RAW_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in extensions
    ]

    logger.info("Found %d Markdown files to process", len(raw_files))

    processed_count = 0
    skipped_count = 0
    error_count = 0
    total_chars_before = 0
    total_chars_after = 0

    for i, raw_file in enumerate(raw_files):
        try:
            # Read raw content
            content = raw_file.read_text(encoding="utf-8", errors="replace")
            total_chars_before += len(content)

            # Clean the content
            cleaned = clean_file(content)

            # Skip files that are too short after cleaning
            if len(cleaned) < MIN_CONTENT_LENGTH:
                skipped_count += 1
                continue

            total_chars_after += len(cleaned)

            # Preserve relative directory structure
            relative_path = raw_file.relative_to(RAW_DIR)

            # Change extension to .txt for cleaned files
            dest_path = PROCESSED_DIR / relative_path.with_suffix(".txt")
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Write cleaned content
            dest_path.write_text(cleaned, encoding="utf-8")
            processed_count += 1

            # Log progress every 500 files
            if (i + 1) % 500 == 0:
                logger.info("  Processed %d / %d files...", i + 1, len(raw_files))

        except Exception as e:
            logger.warning("Error processing %s: %s", raw_file.name, e)
            error_count += 1

    report = {
        "total_raw_files": len(raw_files),
        "processed_files": processed_count,
        "skipped_files": skipped_count,
        "error_files": error_count,
        "total_chars_before": total_chars_before,
        "total_chars_after": total_chars_after,
        "reduction_percent": round(
            (1 - total_chars_after / max(total_chars_before, 1)) * 100, 1
        ),
        "processing_date": datetime.now().isoformat(),
    }

    return report


def save_report(report: dict):
    """Save the processing report as JSON."""
    report_path = PROCESSED_DIR / "processing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Report saved: %s", report_path)


def main():
    """Run the full data preprocessing pipeline."""
    logger.info("=" * 60)
    logger.info("Phase 2: Data Processing -- Preprocessor")
    logger.info("=" * 60)

    # Step 1: Ensure directories exist
    ensure_directories()

    # Check that raw data exists
    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        logger.error("[FAILED] No raw data found in %s", RAW_DIR)
        logger.error("Run 'python scripts/scraper.py' first (Phase 1).")
        sys.exit(1)

    # Step 2: Process all files
    report = process_all_files()

    # Step 3: Save report
    save_report(report)

    # Summary
    logger.info("=" * 60)
    logger.info("[OK] Phase 2 Complete: Data Processing Finished!")
    logger.info("   Raw files found:     %d", report["total_raw_files"])
    logger.info("   Files processed:     %d", report["processed_files"])
    logger.info("   Files skipped:       %d (too short after cleaning)", report["skipped_files"])
    logger.info("   Files with errors:   %d", report["error_files"])
    logger.info("   Chars before:        %s", f"{report['total_chars_before']:,}")
    logger.info("   Chars after:         %s", f"{report['total_chars_after']:,}")
    logger.info("   Size reduction:      %s%%", report["reduction_percent"])
    logger.info("   Output directory:    %s", PROCESSED_DIR)
    logger.info("=" * 60)
    logger.info("Next step: Run 'python scripts/chunker.py' (Phase 3)")


if __name__ == "__main__":
    main()

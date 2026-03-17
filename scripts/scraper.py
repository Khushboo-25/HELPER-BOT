"""
scraper.py -- GitLab Handbook Data Collection (Phase 1)
=======================================================
Clones the GitLab handbook repository and extracts all Markdown files
into the data/raw/ directory for further processing.

Usage:
    python scripts/scraper.py

Output:
    data/raw/           -> All .md files copied here (preserving folder structure)
    data/raw/manifest.json -> Manifest listing all collected files with metadata
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

# -- Setup Logging (Windows-safe: no emojis in log messages) --
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/scraper.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# -- Configuration --
# The real GitLab handbook repo containing Markdown content
REPO_URL = "https://gitlab.com/gitlab-com/content-sites/handbook.git"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLONE_DIR = PROJECT_ROOT / "data" / "_clone_temp"   # Temporary clone location
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LOGS_DIR = PROJECT_ROOT / "logs"

# File extensions to collect
EXTENSIONS = {".md", ".mdx", ".markdown"}


def ensure_directories():
    """Create required directories if they don't exist."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", RAW_DIR)


def clone_repository():
    """
    Clone the GitLab handbook repository using GitPython.
    Uses a shallow clone (depth=1) to save bandwidth and time.

    Returns:
        git.Repo: The cloned repository object.
    """
    try:
        from git import Repo
    except ImportError:
        logger.error(
            "GitPython is not installed. Install it with: pip install gitpython"
        )
        sys.exit(1)

    # If clone directory already exists, remove it for a fresh clone
    if CLONE_DIR.exists():
        logger.info("Removing existing clone directory: %s", CLONE_DIR)
        shutil.rmtree(CLONE_DIR, ignore_errors=True)

    logger.info("Cloning repository: %s", REPO_URL)
    logger.info("Clone destination: %s", CLONE_DIR)
    logger.info("This may take a few minutes depending on repo size...")

    try:
        repo = Repo.clone_from(
            REPO_URL,
            str(CLONE_DIR),
            depth=1,                # Shallow clone (only latest commit)
            single_branch=True,     # Only clone the default branch
        )
        logger.info("[OK] Repository cloned successfully!")
        return repo
    except Exception as e:
        logger.error("[FAILED] Failed to clone repository: %s", e)
        sys.exit(1)


def find_markdown_files():
    """
    Recursively find all Markdown files in the cloned repository.

    Returns:
        list[Path]: List of paths to all markdown files found.
    """
    md_files = []
    for ext in EXTENSIONS:
        md_files.extend(CLONE_DIR.rglob(f"*{ext}"))

    # Exclude files inside .git/ or node_modules/
    md_files = [
        f for f in md_files
        if ".git" not in f.parts and "node_modules" not in f.parts
    ]

    logger.info("Found %d Markdown files in the repository", len(md_files))
    return md_files


def extract_title(content: str, filepath: Path) -> str:
    """
    Extract the title from a Markdown file.
    Looks for the first H1 heading (# Title) in the content.

    Args:
        content:  The raw markdown text.
        filepath: Path to the file (used as fallback title).

    Returns:
        The extracted title string.
    """
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
        # Also check for YAML frontmatter title
        if stripped.startswith("title:"):
            title = stripped[6:].strip().strip("'\"")
            if title:
                return title
    # Fallback: use the filename without extension
    return filepath.stem.replace("-", " ").replace("_", " ").title()


def copy_markdown_files(md_files: list) -> list:
    """
    Copy all Markdown files from the clone to data/raw/, preserving
    the relative folder structure.

    Args:
        md_files: List of Path objects pointing to .md files.

    Returns:
        List of manifest entries (dicts) for each copied file.
    """
    manifest_entries = []
    copied_count = 0
    skipped_count = 0

    for md_file in md_files:
        try:
            # Read file content
            content = md_file.read_text(encoding="utf-8", errors="replace")

            # Skip very small files (likely empty or just frontmatter)
            if len(content.strip()) < 50:
                skipped_count += 1
                continue

            # Calculate relative path from clone directory
            relative_path = md_file.relative_to(CLONE_DIR)

            # Create destination path in data/raw/
            dest_path = RAW_DIR / relative_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy the file
            shutil.copy2(md_file, dest_path)
            copied_count += 1

            # Extract metadata for manifest
            title = extract_title(content, md_file)
            entry = {
                "id": copied_count,
                "filename": md_file.name,
                "relative_path": str(relative_path).replace("\\", "/"),
                "title": title,
                "size_bytes": len(content.encode("utf-8")),
                "char_count": len(content),
                "line_count": content.count("\n") + 1,
            }
            manifest_entries.append(entry)

            if copied_count % 100 == 0:
                logger.info("  Copied %d files so far...", copied_count)

        except Exception as e:
            logger.warning("Error processing %s: %s", md_file.name, e)
            skipped_count += 1

    logger.info("[OK] Copied %d files to %s", copied_count, RAW_DIR)
    if skipped_count > 0:
        logger.info("[SKIP] Skipped %d files (too short or errors)", skipped_count)

    return manifest_entries


def save_manifest(entries: list):
    """
    Save a manifest.json file listing all collected documents with metadata.

    Args:
        entries: List of dicts containing file metadata.
    """
    manifest = {
        "source_repo": REPO_URL,
        "collection_date": datetime.now().isoformat(),
        "total_files": len(entries),
        "total_characters": sum(e["char_count"] for e in entries),
        "files": entries,
    }

    manifest_path = RAW_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Manifest saved: %s", manifest_path)
    logger.info("   Total files:      %d", manifest["total_files"])
    logger.info("   Total characters: %s", f"{manifest['total_characters']:,}")


def cleanup_clone():
    """Remove the temporary clone directory to save disk space."""
    if CLONE_DIR.exists():
        logger.info("Cleaning up temporary clone: %s", CLONE_DIR)
        shutil.rmtree(CLONE_DIR, ignore_errors=True)
        logger.info("[OK] Temporary clone removed")


def main():
    """Run the full data collection pipeline."""
    logger.info("=" * 60)
    logger.info("Phase 1: Data Collection -- GitLab Handbook Scraper")
    logger.info("=" * 60)

    # Step 1: Ensure directories exist
    ensure_directories()

    # Step 2: Clone the repository
    clone_repository()

    # Step 3: Find all Markdown files
    md_files = find_markdown_files()

    if not md_files:
        logger.error("[FAILED] No Markdown files found in the repository!")
        cleanup_clone()
        sys.exit(1)

    # Step 4: Copy Markdown files to data/raw/
    manifest_entries = copy_markdown_files(md_files)

    # Step 5: Save manifest
    save_manifest(manifest_entries)

    # Step 6: Cleanup temporary clone
    cleanup_clone()

    # Summary
    logger.info("=" * 60)
    logger.info("[OK] Phase 1 Complete: Data Collection Finished!")
    logger.info("   Files collected: %d", len(manifest_entries))
    logger.info("   Output directory: %s", RAW_DIR)
    logger.info("   Manifest: %s", RAW_DIR / "manifest.json")
    logger.info("=" * 60)
    logger.info("Next step: Run 'python scripts/preprocessor.py' (Phase 2)")


if __name__ == "__main__":
    main()

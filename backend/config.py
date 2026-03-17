"""
config.py -- Application Configuration
======================================
Loads settings from multiple sources with this priority:
  1. Streamlit Cloud secrets (for deployment)
  2. Environment variables
  3. .env file (for local development)
  4. Default values

Uses pydantic-settings for type-safe configuration management.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from pydantic import Field
import yaml

# Load environment variables from .env; override existing values so edits take
# effect immediately when the module is imported.
try:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=True)
except Exception:
    # fall back to default behavior if something goes wrong
    load_dotenv()


# -- Project root (two levels up from backend/config.py) --
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_streamlit_secrets():
    """
    Try to load secrets from Streamlit Cloud.
    On Streamlit Community Cloud, secrets are set in the app dashboard
    and accessed via st.secrets. This injects them into os.environ
    so pydantic-settings can pick them up.

    We guard against the default placeholder value (`your_gemini_api_key_here`)
    which may appear in `st.secrets` when running locally (Streamlit reads the
    `.env.example` file).  Overwriting a real key with the placeholder would
    break the app, so we only set the env var if the secret looks valid.
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key in ["GEMINI_API_KEY"]:
                if key in st.secrets:
                    val = st.secrets[key]
                    if val and val != "your_gemini_api_key_here":
                        os.environ[key] = val
    except Exception:
        pass  # Not running in Streamlit, skip


def load_yaml_config() -> dict:
    """Load settings from config/settings.yaml."""
    yaml_path = PROJECT_ROOT / "config" / "settings.yaml"
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


# Inject Streamlit secrets into environment before Settings loads, but only
# if there isn’t already a plausible API key set.  Accessing `st.secrets`
# indirectly mutates `os.environ`, and the local `.env.example` placeholder
# ends up overriding a real key when we import Streamlit.  We avoid this by
# skipping Streamlit entirely when the environment already looks valid.
if not os.environ.get("GEMINI_API_KEY", "").startswith("AIza"):
    load_streamlit_secrets()

class Settings(BaseSettings):
    """
    Application settings.
    Values are loaded in this priority order:
      1. Streamlit Cloud secrets (injected into env above)
      2. Environment variables
      3. .env file
      4. Default values defined here
    """

    # -- API Keys --
    GEMINI_API_KEY: str = Field(
        default="",
        description="Google Gemini API key from https://aistudio.google.com/apikey"
    )

    # -- ChromaDB --
    CHROMA_DB_PATH: str = Field(
        default="./vector_db",
        description="Path to ChromaDB persistent storage"
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="gitlab_handbook",
        description="Name of the ChromaDB collection"
    )

    # -- RAG Settings --
    TOP_K: int = Field(
        default=5,
        description="Number of document chunks to retrieve per query"
    )
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Size of each text chunk in characters"
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        description="Overlap between consecutive chunks in characters"
    )

    # -- LLM Settings --
    LLM_MODEL: str = Field(
        default="models/gemini-2.5-flash",
        description="Gemini model to use for answer generation"
    )
    EMBEDDING_MODEL: str = Field(
        default="models/gemini-embedding-001",
        description="Embedding model for vector generation (Gemini embeddings)"
    )

    # -- Server Settings --
    BACKEND_HOST: str = Field(default="0.0.0.0")
    BACKEND_PORT: int = Field(default=8000)

    class Config:
        # We already loaded .env above using python-dotenv (override=False), and
        # letting pydantic re-read it tends to overwrite existing values.  By
        # disabling the built-in env_file behaviour we avoid that problem.
        env_file = None
        env_file_encoding = "utf-8"
        extra = "ignore"


# -- Singleton instance --
# Construct the Settings object.  We previously loaded the .env file with
# python-dotenv (override=False) above, so the environment should already have
# the correct values and we don't need any additional juggling here.
settings = Settings()


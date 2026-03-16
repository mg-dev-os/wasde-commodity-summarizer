"""Load application configuration from config.yaml and environment."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
_cached_config = None


def get_config():
    """Load and return config dict. Uses cached config after first load."""
    global _cached_config
    if _cached_config is not None:
        return _cached_config
    if not _CONFIG_PATH.exists():
        _cached_config = _default_config()
        return _cached_config
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        _cached_config = yaml.safe_load(f) or _default_config()
    return _cached_config


def _default_config():
    return {
        "llm": {
            "provider": "groq",
            "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.1-8b-instant"},
            "gemini": {"model": "gemini-1.5-flash"},
        },
        "openai": {
            "model": "gpt-4o-mini",
            "max_tokens": 2048,
            "temperature": 0.3,
            "max_commentary_chars": 12000,
        },
        "pdf": {
            "commodity_headers": [
                "Wheat",
                "Coarse Grains",
                "Rice",
                "Oilseeds",
                "Cotton",
                "Sugar",
                "Livestock",
                "Poultry",
                "Dairy",
            ],
        },
        "ui": {"page_title": "WASDE Commodity Summarizer", "max_file_size_mb": 50},
        "retrieval": {
            "chunking": {"strategy": "commodity_aware", "chunk_size": 600, "overlap": 100},
            "embedding": {"provider": "sentence-transformers", "openai_model": "text-embedding-3-small"},
            "vector_store": {"provider": "lancedb", "path": ".lancedb", "top_k": 5},
        },
        "use_maf_retrieval": False,
    }


def get_openai_api_key():
    """Return OpenAI API key from environment."""
    return os.environ.get("OPENAI_API_KEY", "").strip()


def get_groq_api_key():
    """Return Groq API key from environment."""
    return os.environ.get("GROQ_API_KEY", "").strip()


def get_gemini_api_key():
    """Return Google Gemini API key from environment."""
    return os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("GOOGLE_API_KEY", "").strip()

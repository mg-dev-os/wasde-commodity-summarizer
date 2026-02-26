"""Tests for PDF extraction and summarization (with mocks)."""

import pytest

from src.config_loader import get_config, get_openai_api_key
from src.pdf_extractor import _split_by_headers, extract_text_by_commodity
from src.summarizer import summarize_commentary, summarize_all_commodities


# ----- Config -----


def test_get_config_returns_dict():
    cfg = get_config()
    assert isinstance(cfg, dict)
    assert "openai" in cfg
    assert "pdf" in cfg


def test_get_config_has_commodity_headers():
    cfg = get_config()
    headers = cfg.get("pdf", {}).get("commodity_headers", [])
    assert "Wheat" in headers
    assert "Coarse Grains" in headers


def test_get_openai_api_key_returns_string():
    key = get_openai_api_key()
    assert isinstance(key, str)


# ----- PDF split logic (no real PDF) -----


def test_split_by_headers_empty_text():
    assert _split_by_headers("", ["Wheat", "Rice"]) == {}
    assert _split_by_headers("   \n\n  ", ["Wheat"]) == {}


def test_split_by_headers_no_headers_in_text():
    text = "Some intro.\nThen more text.\nNo section headers here."
    result = _split_by_headers(text, ["Wheat", "Rice"])
    # Falls back to first header as key, whole text as value
    assert len(result) == 1
    assert "Wheat" in result or "Commentary" in result
    assert "Some intro" in list(result.values())[0]


def test_split_by_headers_two_sections():
    text = """
    Intro line.
    Wheat
    Wheat production is up. Exports higher.
    Rice
    Rice stocks lower. Demand strong.
    """
    result = _split_by_headers(text, ["Wheat", "Rice"])
    assert "Wheat" in result
    assert "Rice" in result
    assert "Wheat production" in result["Wheat"]
    assert "Rice stocks" in result["Rice"]


def test_split_by_headers_case_insensitive():
    text = "wHEAT\nSection A.\nCOARSE GRAINS\nSection B."
    result = _split_by_headers(text, ["Wheat", "Coarse Grains"])
    assert "Wheat" in result
    assert "Coarse Grains" in result
    assert "Section A" in result["Wheat"]
    assert "Section B" in result["Coarse Grains"]


# ----- Summarizer with mocked OpenAI -----


def test_summarize_all_commodities_empty_sections():
    result = summarize_all_commodities({})
    assert result == {}


def test_summarize_all_commodities_skips_empty_text(monkeypatch):
    """When OpenAI is selected but API key is missing, we get error for non-empty; empty sections get placeholder."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    base_cfg = get_config()
    monkeypatch.setattr("src.summarizer.get_config", lambda: {**base_cfg, "llm": {"provider": "openai"}})
    sections = {"Wheat": "Some text", "Rice": ""}
    result = summarize_all_commodities(sections)
    assert "Wheat" in result
    assert "Rice" in result
    assert "No commentary extracted" in result["Rice"] or "(No commentary" in result["Rice"]


def test_summarize_commentary_no_api_key(monkeypatch):
    import unittest.mock as mock
    monkeypatch.setenv("OPENAI_API_KEY", "")
    base_cfg = get_config()
    monkeypatch.setattr("src.summarizer.get_config", lambda: {**base_cfg, "llm": {"provider": "openai"}})
    out = summarize_commentary("Wheat", "Production up.")
    assert "OPENAI_API_KEY" in out or "not set" in out.lower()


def test_summarize_commentary_mock_openai(monkeypatch):
    """Mock OpenAI client to return a fixed summary without calling the API."""
    import unittest.mock as mock
    mock_summary = "Mock summary: wheat production increased; exports higher."

    def fake_create(**kwargs):
        class Choice:
            class Message:
                content = mock_summary
            message = Message()
        class Resp:
            choices = [Choice()]
        return Resp()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    base_cfg = get_config()
    monkeypatch.setattr("src.summarizer.get_config", lambda: {**base_cfg, "llm": {"provider": "openai"}})
    with mock.patch("src.summarizer.OpenAI") as mock_client:
        mock_client.return_value.chat.completions.create = fake_create
        out = summarize_commentary("Wheat", "Production up. Exports higher.")
    assert mock_summary in out or "Mock summary" in out


# ----- PDF extractor: file not found -----


def test_extract_text_by_commodity_file_not_found():
    with pytest.raises(FileNotFoundError, match="not found"):
        extract_text_by_commodity("nonexistent.pdf")


def test_extract_text_by_commodity_not_pdf():
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
        f.write(b"not a pdf")
        path = f.name
    try:
        with pytest.raises(ValueError, match="must be a PDF"):
            extract_text_by_commodity(path)
    finally:
        import os
        os.unlink(path)

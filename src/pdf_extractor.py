"""Extract text from PDF and split by commodity section headers."""

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

from src.config_loader import get_config


def extract_text_by_commodity(pdf_path: str | Path) -> dict[str, str]:
    """
    Extract full text from PDF and split into sections by commodity headers.
    Returns dict mapping commodity name -> section text (commentary).
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError("File must be a PDF")

    config = get_config()
    headers = config.get("pdf", {}).get("commodity_headers", [])

    doc = fitz.open(path)
    try:
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
    finally:
        doc.close()

    return _split_by_headers(full_text, headers)


def _split_by_headers(text: str, headers: list[str]) -> dict[str, str]:
    """
    Split document text by commodity header lines.
    Headers are matched case-insensitively; section text is from header to next header or end.
    """
    if not text.strip():
        return {}

    # Normalize: one newline between lines, strip
    lines = [ln.strip() for ln in text.replace("\r\n", "\n").split("\n") if ln.strip()]
    if not lines:
        return {}

    # Build list of (header_index, header_name) for headers found in order
    header_names_lower = [h.strip().lower() for h in headers if h.strip()]
    found: list[tuple[int, str]] = []  # (line_index, original header name from config)
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for h in headers:
            if not h.strip():
                continue
            # Match whole line or line starting with header (e.g. "Wheat" or "Wheat:")
            if line_lower == h.strip().lower() or line_lower.startswith(h.strip().lower()):
                found.append((i, h.strip()))
                break

    # Sort by index and deduplicate by keeping first occurrence of each header name
    seen: set[str] = set()
    ordered: list[tuple[int, str]] = []
    for i, name in sorted(found, key=lambda x: x[0]):
        if name not in seen:
            seen.add(name)
            ordered.append((i, name))

    if not ordered:
        # No headers found: treat whole text as one "General" section or first header
        first = headers[0] if headers else "Commentary"
        return {first: text.strip()}

    result: dict[str, str] = {}
    for idx, (start, name) in enumerate(ordered):
        end = ordered[idx + 1][0] if idx + 1 < len(ordered) else len(lines)
        section_lines = lines[start + 1 : end]  # exclude header line itself
        section_text = "\n".join(section_lines).strip()
        if section_text:
            result[name] = section_text

    return result


def extract_raw_text(pdf_path: str | Path) -> str:
    """Extract raw text from entire PDF (for fallback or debugging)."""
    path = Path(pdf_path)
    doc = fitz.open(path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()

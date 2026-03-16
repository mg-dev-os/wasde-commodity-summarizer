"""Modular Chunking strategies for the retrieval pipeline."""

import abc

from src.config_loader import get_config


class Chunker(abc.ABC):
    """Abstract base class for chunking strategies."""
    
    @abc.abstractmethod
    def chunk(self, text_or_sections: str | dict[str, str]) -> list[dict]:
        """
        Produce chunks from input.
        Returns a list of dicts: {"text": "chunk_str", "metadata": {"commodity": "Wheat", ...}}
        """
        pass

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """Helper to split a single string into overlapping chunks."""
        text = (text or "").strip()
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if not chunk.strip():
                start = end - overlap
                continue
            chunks.append(chunk)
            start = end - overlap
        return chunks


class StandardChunker(Chunker):
    """Splits raw text into standard chunks with no specific metadata."""
    
    def __init__(self, chunk_size: int = 600, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text_or_sections: str | dict[str, str]) -> list[dict]:
        if isinstance(text_or_sections, dict):
            # Fallback if given sections instead of raw text
            text = "\n\n".join(text_or_sections.values())
        else:
            text = text_or_sections
            
        chunks = self._split_text(text, self.chunk_size, self.overlap)
        return [{"text": c, "metadata": {}} for c in chunks]


class CommodityAwareChunker(Chunker):
    """
    Expects a dictionary of sections (commodity_name -> text).
    Chunks each commodity's text and attaches {"commodity": commodity_name} as metadata.
    """
    
    def __init__(self, chunk_size: int = 600, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text_or_sections: str | dict[str, str]) -> list[dict]:
        if isinstance(text_or_sections, str):
            # Fallback if given raw text instead of sections; treat as "General" commodity
            sections = {"General": text_or_sections}
        else:
            sections = text_or_sections
            
        result = []
        for commodity_name, text in sections.items():
            chunks = self._split_text(text, self.chunk_size, self.overlap)
            for c in chunks:
                result.append({
                    "text": c,
                    "metadata": {"commodity": commodity_name}
                })
        return result


def get_chunker() -> Chunker:
    """Factory method to get the configured chunker."""
    config = get_config()
    retrieval_cfg = config.get("retrieval", {})
    chunking_cfg = retrieval_cfg.get("chunking", {})
    
    strategy = chunking_cfg.get("strategy", "commodity_aware")
    # Maintain backwards compatibility if chunking config is missing entirely
    chunk_size = chunking_cfg.get("chunk_size", 600)
    overlap = chunking_cfg.get("overlap", 100)
    
    if strategy == "commodity_aware":
        return CommodityAwareChunker(chunk_size=chunk_size, overlap=overlap)
    else:
        return StandardChunker(chunk_size=chunk_size, overlap=overlap)

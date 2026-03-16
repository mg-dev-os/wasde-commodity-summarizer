"""Modular Embedding strategies for the retrieval pipeline."""

import abc

from src.config_loader import get_config, get_openai_api_key


class Embedder(abc.ABC):
    """Abstract base class for embedding models."""
    
    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Returns the dimension of the embeddings."""
        pass

    @abc.abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        pass

    @abc.abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string."""
        pass


class SentenceTransformerEmbedder(Embedder):
    """Local embeddings using SentenceTransformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Import lazily to avoid slowing down startup if not used
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(query)
        return embedding.tolist()


class OpenAIEmbedder(Embedder):
    """Cloud embeddings using OpenAI API."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        from openai import OpenAI
        
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Cannot initialize OpenAIEmbedder.")
            
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        
        # Statically define dimensions for common models
        if "small" in model_name:
            self._dim = 1536
        elif "large" in model_name:
            self._dim = 3072
        else:
            self._dim = 1536  # Default fallback

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(input=texts, model=self.model_name)
        # OpenAI returns embeddings in the same order
        return [data.embedding for data in sorted(response.data, key=lambda x: x.index)]

    def embed_query(self, query: str) -> list[float]:
        response = self.client.embeddings.create(input=query, model=self.model_name)
        return response.data[0].embedding


def get_embedder() -> Embedder:
    """Factory method to get the configured embedder."""
    config = get_config()
    retrieval_cfg = config.get("retrieval", {})
    embedding_cfg = retrieval_cfg.get("embedding", {})
    
    provider = embedding_cfg.get("provider", "sentence-transformers")
    
    if provider == "openai":
        model_name = embedding_cfg.get("openai_model", "text-embedding-3-small")
        return OpenAIEmbedder(model_name=model_name)
    else:
        # Default to sentence-transformers
        model_name = "all-MiniLM-L6-v2"  # Using minilm for speed by default
        return SentenceTransformerEmbedder(model_name=model_name)

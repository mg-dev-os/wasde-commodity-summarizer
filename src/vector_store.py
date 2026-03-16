"""Modular Vector store for document chunks and retrieval."""

import abc
from pathlib import Path

from src.config_loader import get_config
from src.embeddings import Embedder, get_embedder
from src.chunking import get_chunker


class VectorStore(abc.ABC):
    """Abstract base class for vector storage."""
    
    @abc.abstractmethod
    def index_documents(self, doc_id: str, chunks: list[dict]) -> None:
        """Index chunks. Chunks is a list of {"text": "...", "metadata": {"commodity": "..."}}."""
        pass
        
    @abc.abstractmethod
    def search(self, query: str, doc_id: str, top_k: int = 5, filters: dict = None) -> list[str]:
        """Search and return chunk texts, optionally filtered by metadata."""
        pass


class LanceDBStore(VectorStore):
    """LanceDB implementation with metadata support."""
    
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        config = get_config()
        path = config.get("retrieval", {}).get("vector_store", {}).get("path", ".lancedb")
        if not path:
            # Fallback
            path = config.get("vector_store", {}).get("path", ".lancedb") 
        self.db_path = Path(path).resolve()
        
    def _get_schema(self, dim: int):
        from lancedb.pydantic import LanceModel, Vector
        class DocChunk(LanceModel):
            doc_id: str
            text: str
            vector: Vector(dim)
            commodity: str = "" # metadata
        return DocChunk

    def index_documents(self, doc_id: str, chunks: list[dict]) -> None:
        import lancedb
        
        if not chunks:
            return
            
        texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        
        records = []
        for chunk, emb in zip(chunks, embeddings):
            commodity = chunk.get("metadata", {}).get("commodity", "")
            records.append({
                "doc_id": doc_id,
                "text": chunk["text"],
                "vector": emb,
                "commodity": commodity
            })
            
        self.db_path.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(self.db_path))
        table_name = "wasde_chunks"
        
        DocChunkSchema = self._get_schema(self.embedder.dim)
        
        if table_name in db.table_names():
            table = db.open_table(table_name)
            try:
                safe_id = doc_id.replace("'", "''")
                table.delete("doc_id = '{}'".format(safe_id))
            except Exception:
                pass
        else:
            table = db.create_table(table_name, schema=DocChunkSchema, mode="overwrite")
            
        table.add(records)

    def search(self, query: str, doc_id: str, top_k: int = 5, filters: dict = None) -> list[str]:
        import lancedb
        
        if not self.db_path.exists():
            return []
            
        db = lancedb.connect(str(self.db_path))
        table_name = "wasde_chunks"
        if table_name not in db.table_names():
            return []
            
        query_vector = self.embedder.embed_query(query)
        table = db.open_table(table_name)
        
        search_query = table.search(query_vector)
        
        # Build SQL filters
        sql_filters = []
        safe_doc_id = doc_id.replace("'", "''")
        sql_filters.append(f"doc_id = '{safe_doc_id}'")
        
        if filters and "commodity" in filters and filters["commodity"] and filters["commodity"] != "All":
            val = filters["commodity"].replace("'", "''")
            sql_filters.append(f"commodity = '{val}'")
            
        where_clause = " AND ".join(sql_filters)
        try:
            search_query = search_query.where(where_clause, prefilter=True)
        except Exception:
            search_query = search_query.where(where_clause)
            
        results = search_query.limit(top_k).to_list()
        
        filtered = [r.get("text", "") for r in results if r.get("text")]
        return filtered


def get_vector_store() -> VectorStore:
    """Factory method to get the configured vector store."""
    config = get_config()
    provider = config.get("retrieval", {}).get("vector_store", {}).get("provider", "lancedb")
    
    embedder = get_embedder()
    
    if provider == "lancedb":
        return LanceDBStore(embedder=embedder)
    else:
        return LanceDBStore(embedder=embedder)


# Backward compatible helper functions used by app.py and retrieval_agent.py
def index_document(doc_id: str, full_text_or_sections: str | dict[str, str]) -> None:
    """Helper to chunk and index a document."""
    chunker = get_chunker()
    chunks = chunker.chunk(full_text_or_sections)
    vs = get_vector_store()
    vs.index_documents(doc_id, chunks)


def search(query: str, doc_id: str, top_k: int = 5, filters: dict = None) -> list[str]:
    """Helper to search the vector store."""
    vs = get_vector_store()
    return vs.search(query, doc_id, top_k=top_k, filters=filters)

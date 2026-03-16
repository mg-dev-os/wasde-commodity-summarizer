"""Tests for the new modular retrieval pipeline components."""

import pytest
from unittest.mock import patch, MagicMock

from src.chunking import StandardChunker, CommodityAwareChunker
from src.embeddings import SentenceTransformerEmbedder
from src.vector_store import VectorStore, LanceDBStore

def test_standard_chunker():
    """Test that StandardChunker splits text and adds empty metadata"""
    chunker = StandardChunker(chunk_size=10, overlap=2)
    text = "1234567890abcdefghij"
    
    chunks = chunker.chunk(text)
    
    assert len(chunks) == 3
    assert chunks[0]["text"] == "1234567890"
    assert chunks[1]["text"] == "90abcdefgh"
    assert chunks[2]["text"] == "ghij"
    
    for c in chunks:
        assert c["metadata"] == {}

def test_commodity_aware_chunker():
    """Test that CommodityAwareChunker attaches the section name as metadata"""
    chunker = CommodityAwareChunker(chunk_size=50, overlap=0)
    
    sections = {
        "Wheat": "Wheat text here.",
        "Corn": "Corn text here."
    }
    
    chunks = chunker.chunk(sections)
    
    assert len(chunks) == 2
    assert chunks[0]["text"] == "Wheat text here."
    assert chunks[0]["metadata"]["commodity"] == "Wheat"
    
    assert chunks[1]["text"] == "Corn text here."
    assert chunks[1]["metadata"]["commodity"] == "Corn"

@patch.dict('sys.modules', {'sentence_transformers': MagicMock()})
def test_sentence_transformer_embedder():
    """Test the embedding abstraction interface"""
    import sys
    mock_sentence_transformer = MagicMock()
    sys.modules['sentence_transformers'].SentenceTransformer = mock_sentence_transformer
    
    # Setup mock
    mock_instance = MagicMock()
    mock_instance.get_sentence_embedding_dimension.return_value = 384
    
    # Mock encode to return a numpy-like array with a tolist() method, or just lists
    class MockArray:
        def __init__(self, data):
            self.data = data
        def tolist(self):
            return self.data
            
    mock_instance.encode.side_effect = lambda texts: MockArray([[0.1, 0.2] for _ in (texts if isinstance(texts, list) else [texts])])
    mock_sentence_transformer.return_value = mock_instance
    
    embedder = SentenceTransformerEmbedder(model_name="test-model")
    
    assert embedder.dim == 384
    
    docs = embedder.embed_documents(["doc1", "doc2"])
    assert len(docs) == 2
    assert docs[0] == [0.1, 0.2]
    
    query = embedder.embed_query("test query")
    assert query == [[0.1, 0.2]] # Note: encode on single string mocked to return array of 1

@patch.dict('sys.modules', {'lancedb': MagicMock(), 'lancedb.pydantic': MagicMock()})
@patch('src.vector_store.get_config')
def test_lancedb_vector_store_schema_and_filters(mock_get_config):
    """Test the LanceDB schema and filtering logic construction"""
    import sys
    mock_connect = sys.modules['lancedb'].connect
    # Setup mocks
    mock_get_config.return_value = {
        "retrieval": {
            "vector_store": {"path": ".testdb"}
        }
    }
    
    mock_embedder = MagicMock()
    mock_embedder.dim = 384
    mock_embedder.embed_query.return_value = [0.1] * 384
    
    mock_db = MagicMock()
    mock_table = MagicMock()
    mock_search = MagicMock()
    
    mock_connect.return_value = mock_db
    mock_db.table_names.return_value = ["wasde_chunks"]
    mock_db.open_table.return_value = mock_table
    
    mock_table.search.return_value = mock_search
    mock_search.where.return_value = mock_search
    mock_search.limit.return_value = mock_search
    mock_search.to_list.return_value = [
        {"text": "result 1", "commodity": "Wheat"}
    ]
    
    store = LanceDBStore(embedder=mock_embedder)
    
    # Needs to exist
    with patch('pathlib.Path.exists', return_value=True):
        results = store.search("Query", doc_id="doc1", top_k=5, filters={"commodity": "Wheat"})
        
        assert len(results) == 1
        assert results[0] == "result 1"
        
        # Verify filtering logic was applied
        mock_table.search.assert_called_once()
        # Verify the where clause contains our filters
        args, kwargs = mock_search.where.call_args
        where_clause = args[0]
        assert "doc_id = 'doc1'" in where_clause
        assert "commodity = 'Wheat'" in where_clause

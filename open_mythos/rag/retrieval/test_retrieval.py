"""
Tests for RAG Retrieval Module
"""

import pytest
import numpy as np
from typing import List, Dict, Any


# Sample test documents
SAMPLE_DOCS = [
    {"id": "doc1", "content": "Python is a programming language", "metadata": {"lang": "en"}},
    {"id": "doc2", "content": "JavaScript is for web development", "metadata": {"lang": "en"}},
    {"id": "doc3", "content": "Machine learning uses algorithms", "metadata": {"field": "ML"}},
    {"id": "doc4", "content": "Deep learning is a subset of ML", "metadata": {"field": "DL"}},
    {"id": "doc5", "content": "Natural language processing is NLP", "metadata": {"field": "NLP"}},
]


class MockEmbedder:
    """Mock embedder for testing."""
    
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
    
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            hash_val = hash(text.lower())
            np.random.seed(hash_val % (2**31))
            emb = np.random.randn(self.dimension)
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        return np.array(embeddings)


class TestBM25:
    """Tests for BM25 retrieval."""
    
    def test_bm25_indexing(self):
        from open_mythos.rag.retrieval.bm25 import BM25Retriever
        
        bm25 = BM25Retriever()
        bm25.index(SAMPLE_DOCS)
        
        assert bm25._is_indexed
        assert len(bm25.bm25.documents) == 5
    
    def test_bm25_retrieval(self):
        from open_mythos.rag.retrieval.bm25 import BM25Retriever
        
        bm25 = BM25Retriever()
        bm25.index(SAMPLE_DOCS)
        
        results = bm25.retrieve("Python programming", top_k=3)
        
        assert len(results) <= 3
        assert all("doc_id" in r for r in results)
        assert all("score" in r for r in results)
        # First result should be doc1 about Python
        assert results[0]["doc_id"] == "doc1"
    
    def test_bm25_scores(self):
        from open_mythos.rag.retrieval.bm25 import BM25Retriever
        
        bm25 = BM25Retriever()
        bm25.index(SAMPLE_DOCS)
        
        results = bm25.retrieve("machine learning", top_k=2)
        
        # Scores should be positive and descending
        for i in range(len(results) - 1):
            assert results[i]["score"] >= results[i + 1]["score"]


class TestVectorRetriever:
    """Tests for vector retrieval."""
    
    def test_vector_indexing(self):
        from open_mythos.rag.retrieval.vector_retriever import VectorRetriever
        
        embedder = MockEmbedder()
        vector = VectorRetriever(embedder=embedder)
        vector.index(SAMPLE_DOCS)
        
        assert vector._is_indexed
        assert len(vector._documents) == 5
    
    def test_vector_retrieval(self):
        from open_mythos.rag.retrieval.vector_retriever import VectorRetriever
        
        embedder = MockEmbedder()
        vector = VectorRetriever(embedder=embedder)
        vector.index(SAMPLE_DOCS)
        
        results = vector.retrieve("deep neural networks", top_k=3)
        
        assert len(results) <= 3
        assert all("doc_id" in r for r in results)
        assert all("score" in r for r in results)


class TestHybridRetriever:
    """Tests for hybrid retrieval with RRF."""
    
    def test_hybrid_creation(self):
        from open_mythos.rag.retrieval.bm25 import BM25Retriever
        from open_mythos.rag.retrieval.vector_retriever import VectorRetriever
        from open_mythos.rag.retrieval.hybrid_retriever import HybridRetriever
        
        bm25 = BM25Retriever()
        bm25.index(SAMPLE_DOCS)
        
        embedder = MockEmbedder()
        vector = VectorRetriever(embedder=embedder)
        vector.index(SAMPLE_DOCS)
        
        hybrid = HybridRetriever(bm25=bm25, vector=vector)
        
        assert hybrid._has_bm25
        assert hybrid._has_vector
    
    def test_hybrid_retrieval(self):
        from open_mythos.rag.retrieval.bm25 import BM25Retriever
        from open_mythos.rag.retrieval.vector_retriever import VectorRetriever
        from open_mythos.rag.retrieval.hybrid_retriever import HybridRetriever
        
        bm25 = BM25Retriever()
        bm25.index(SAMPLE_DOCS)
        
        embedder = MockEmbedder()
        vector = VectorRetriever(embedder=embedder)
        vector.index(SAMPLE_DOCS)
        
        hybrid = HybridRetriever(bm25=bm25, vector=vector)
        
        results = hybrid.retrieve("Python machine learning", top_k=3)
        
        assert len(results) <= 3
        assert all("doc_id" in r for r in results)
        assert all("score" in r for r in results)
        assert all("retriever_type" in r for r in results)
    
    def test_hybrid_with_different_weights(self):
        from open_mythos.rag.retrieval.bm25 import BM25Retriever
        from open_mythos.rag.retrieval.vector_retriever import VectorRetriever
        from open_mythos.rag.retrieval.hybrid_retriever import HybridRetriever
        
        bm25 = BM25Retriever()
        bm25.index(SAMPLE_DOCS)
        
        embedder = MockEmbedder()
        vector = VectorRetriever(embedder=embedder)
        vector.index(SAMPLE_DOCS)
        
        hybrid = HybridRetriever(bm25=bm25, vector=vector)
        hybrid.set_weights(bm25_weight=3.0, vector_weight=1.0)
        
        assert hybrid.rrf_config.weight_bm25 == 3.0
        assert hybrid.rrf_config.weight_vector == 1.0


class TestRRFScorer:
    """Tests for RRF scoring."""
    
    def test_rrf_score_calculation(self):
        from open_mythos.rag.retrieval.hybrid_retriever import RRFScorer, RRFConfig
        
        scorer = RRFScorer(RRFConfig(k=60))
        
        # Rank 1 should have highest score
        score1 = scorer._get_rrf_score(1)
        score2 = scorer._get_rrf_score(2)
        score10 = scorer._get_rrf_score(10)
        
        assert score1 > score2 > score10
        assert 0 < score1 <= 1/60
    
    def test_rrf_fusion(self):
        from open_mythos.rag.retrieval.hybrid_retriever import RRFScorer, RRFConfig, RetrievalResult
        
        scorer = RRFScorer(RRFConfig(k=60))
        
        results_a = [
            RetrievalResult(doc_id="d1", score=1.0, content="a", rank=1),
            RetrievalResult(doc_id="d2", score=0.8, content="b", rank=2),
        ]
        results_b = [
            RetrievalResult(doc_id="d2", score=1.0, content="b", rank=1),
            RetrievalResult(doc_id="d3", score=0.9, content="c", rank=2),
        ]
        
        fused = scorer.fuse_results({"retriever_a": results_a, "retriever_b": results_b})
        
        # d2 appears in both, should rank higher
        doc_scores = {r.doc_id: r.score for r in fused}
        assert doc_scores["d2"] > doc_scores["d1"]
        assert doc_scores["d2"] > doc_scores["d3"]


class TestHyDE:
    """Tests for HyDE query expansion."""
    
    def test_hyde_generation(self):
        from open_mythos.rag.retrieval.hyde import HyDEQueryExpander
        
        class MockLLM:
            def generate(self, prompt, **kwargs):
                return "This is a hypothetical answer about " + prompt.split("Question:")[1].split("Passage:")[0]
        
        hyde = HyDEQueryExpander(llm_client=MockLLM())
        
        doc = hyde.generate("What is Python?")
        
        assert doc.query == "What is Python?"
        assert len(doc.content) > 0
        assert "Python" in doc.content or "hypothetical" in doc.content.lower()
    
    def test_hyde_expansion(self):
        from open_mythos.rag.retrieval.hyde import HyDEQueryExpander
        
        class MockLLM:
            def generate(self, prompt, **kwargs):
                return "Generated hypothetical document"
        
        class MockEmbedder:
            def embed(self, text):
                return [0.1] * 128
        
        hyde = HyDEQueryExpander(
            llm_client=MockLLM(),
            embedder=MockEmbedder(),
        )
        
        result = hyde.expand_query("What is Python?", return_embedding=True)
        
        assert result["original_query"] == "What is Python?"
        assert "hypothetical_document" in result
        assert "embedding" in result
        assert len(result["embedding"]) == 128


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

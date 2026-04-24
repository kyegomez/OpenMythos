"""
Retrieval Module
================

Retrieval components:
- embedding_models: BGE-large, E5 embedding models
- cross_encoder_reranker: CrossEncoder reranking
- bm25: BM25+ keyword-based retrieval
- vector_retriever: Vector similarity retrieval
- hyde: HyDE query expansion
- hybrid_retriever: BM25+Vector hybrid retrieval with RRF
- config: Configuration presets

Usage:
    from open_mythos.rag.retrieval import BGELargeEmbedder, CrossEncoderReranker

    # Embedding
    embedder = BGELargeEmbedder()
    embeddings = embedder.encode(["hello world"])

    # Reranking
    reranker = CrossEncoderReranker()
    results = reranker.rerank("query", ["doc1", "doc2"], top_k=5)

    # Hybrid Retrieval with RRF
    from open_mythos.rag.retrieval import HybridRetriever, BM25Retriever, VectorRetriever
    from open_mythos.rag.retrieval import HyDEQueryExpander
    
    bm25 = BM25Retriever()
    bm25.index(documents)
    
    vector = VectorRetriever(embedder=embedder)
    vector.index(documents)
    
    hybrid = HybridRetriever(bm25=bm25, vector=vector)
    results = hybrid.retrieve("query", top_k=10)
    
    # HyDE Query Expansion
    hyde = HyDEQueryExpander(llm_client=llm, embedder=embedder)
    expanded = hyde.expand_query("What is Python?")
"""

from open_mythos.rag.retrieval.embedding_models import (
    EmbeddingModel,
    BGELargeEmbedder,
    E5Embedder,
    create_embedding_model,
    cosine_sim,
)
from open_mythos.rag.retrieval.cross_encoder_reranker import (
    RerankResult,
    CrossEncoderReranker,
    HybridReranker,
    RerankerPool,
)

# BM25 and hybrid retrieval imports
from open_mythos.rag.retrieval.bm25 import BM25Retriever, BM25plus
from open_mythos.rag.retrieval.vector_retriever import VectorRetriever
from open_mythos.rag.retrieval.hyde import HyDEQueryExpander, HyDEPipeline
from open_mythos.rag.retrieval.hybrid_retriever import (
    HybridRetriever,
    RRFScorer,
    RRFConfig,
    RetrievalResult,
    normalize_scores,
    combine_weighted_scores,
)
from open_mythos.rag.retrieval.config import (
    RetrievalConfig,
    BM25Config,
    VectorConfig,
    HyDEConfig,
    RRFConfig as RRFConfigType,
    get_config,
    create_from_config,
)

__all__ = [
    # Embedding models
    "EmbeddingModel",
    "BGELargeEmbedder",
    "E5Embedder",
    "create_embedding_model",
    "cosine_sim",
    # Reranker
    "RerankResult",
    "CrossEncoderReranker",
    "HybridReranker",
    "RerankerPool",
    # BM25 retrieval
    "BM25Retriever",
    "BM25plus",
    # Vector retrieval
    "VectorRetriever",
    # HyDE query expansion
    "HyDEQueryExpander",
    "HyDEPipeline",
    # Hybrid retrieval with RRF
    "HybridRetriever",
    "RRFScorer",
    "RRFConfig",
    "RetrievalResult",
    "normalize_scores",
    "combine_weighted_scores",
    # Configuration
    "RetrievalConfig",
    "BM25Config",
    "VectorConfig",
    "HyDEConfig",
    "get_config",
    "create_from_config",
]
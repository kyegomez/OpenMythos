"""
Configuration for RAG Retrieval Module
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class BM25Config:
    """BM25 retrieval configuration."""
    k1: float = 1.5  # Term frequency saturation
    b: float = 0.75  # Length normalization
    delta: float = 1.0  # BM25+ constant
    min_doc_freq: int = 1  # Minimum document frequency


@dataclass
class VectorConfig:
    """Vector retrieval configuration."""
    model: str = "text-embedding-ada-002"
    dimension: int = 1536
    normalize: bool = True
    batch_size: int = 32
    similarity_metric: str = "cosine"  # "cosine" or "euclidean"


@dataclass
class HyDEConfig:
    """HyDE query expansion configuration."""
    embedding_model: str = "text-embedding-ada-002"
    max_tokens: int = 256
    temperature: float = 0.7
    use_hyde: bool = True


@dataclass
class RRFConfig:
    """Reciprocal Rank Fusion configuration."""
    k: float = 60  # RRF constant
    normalize_scores: bool = True
    weight_bm25: float = 1.0
    weight_vector: float = 1.0


@dataclass
class RetrievalConfig:
    """Main retrieval configuration."""
    bm25: BM25Config = field(default_factory=BM25Config)
    vector: VectorConfig = field(default_factory=VectorConfig)
    hyde: HyDEConfig = field(default_factory=HyDEConfig)
    rrf: RRFConfig = field(default_factory=RRFConfig)
    default_top_k: int = 10
    score_threshold: float = 0.0


@dataclass
class LLMConfig:
    """LLM configuration for HyDE."""
    model: str = "gpt-4"
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.7


# Default configuration instance
DEFAULT_CONFIG = RetrievalConfig()

# Configuration presets

PRESETS = {
    "balanced": RetrievalConfig(
        bm25=BM25Config(k1=1.5, b=0.75, delta=1.0),
        vector=VectorConfig(similarity_metric="cosine"),
        rrf=RRFConfig(k=60, weight_bm25=1.0, weight_vector=1.0),
    ),
    "keyword_heavy": RetrievalConfig(
        bm25=BM25Config(k1=1.8, b=0.8, delta=1.0),
        vector=VectorConfig(similarity_metric="cosine"),
        rrf=RRFConfig(k=60, weight_bm25=2.0, weight_vector=0.5),
    ),
    "semantic_heavy": RetrievalConfig(
        bm25=BM25Config(k1=1.2, b=0.6, delta=1.0),
        vector=VectorConfig(similarity_metric="cosine"),
        rrf=RRFConfig(k=60, weight_bm25=0.5, weight_vector=2.0),
    ),
    "hyde_enhanced": RetrievalConfig(
        bm25=BM25Config(k1=1.5, b=0.75, delta=1.0),
        vector=VectorConfig(similarity_metric="cosine"),
        hyde=HyDEConfig(use_hyde=True),
        rrf=RRFConfig(k=60, weight_bm25=1.0, weight_vector=1.0),
    ),
}


def get_config(name: str = "balanced") -> RetrievalConfig:
    """
    Get a named configuration preset.
    
    Args:
        name: Preset name ('balanced', 'keyword_heavy', 'semantic_heavy', 'hyde_enhanced')
        
    Returns:
        RetrievalConfig instance
    """
    return PRESETS.get(name, DEFAULT_CONFIG)


def create_from_config(config: RetrievalConfig) -> Dict[str, Any]:
    """
    Create retriever instances from configuration.
    
    Args:
        config: RetrievalConfig instance
        
    Returns:
        Dict with 'bm25', 'vector', 'hyde', 'hybrid' keys
    """
    from .bm25 import BM25Retriever
    from .vector_retriever import VectorRetriever, EmbedderWrapper
    from .hyde import HyDEQueryExpander
    from .hybrid_retriever import HybridRetriever, RRFConfig as RRFConfigType
    
    # Create BM25 retriever
    bm25 = BM25Retriever(
        k1=config.bm25.k1,
        b=config.bm25.b,
        delta=config.bm25.delta,
        min_doc_freq=config.bm25.min_doc_freq,
    )
    
    # Create embedder
    embedder = EmbedderWrapper(
        model=config.vector.model,
        dimension=config.vector.dimension,
    )
    
    # Create vector retriever
    vector = VectorRetriever(
        embedder=embedder,
        normalize_embeddings=config.vector.normalize,
        batch_size=config.vector.batch_size,
    )
    
    # Create RRF config
    rrf_config = RRFConfigType(
        k=config.rrf.k,
        normalize_scores=config.rrf.normalize_scores,
        weight_bm25=config.rrf.weight_bm25,
        weight_vector=config.rrf.weight_vector,
    )
    
    # Create HyDE expander
    hyde = None
    if config.hyde.use_hyde:
        hyde = HyDEQueryExpander(
            embedder=embedder,
            embedding_model=config.hyde.embedding_model,
            max_tokens=config.hyde.max_tokens,
            temperature=config.hyde.temperature,
        )
    
    # Create hybrid retriever
    hybrid = HybridRetriever(
        bm25=bm25,
        vector=vector,
        hyde=hyde,
        rrf_config=rrf_config,
        default_top_k=config.default_top_k,
    )
    
    return {
        "bm25": bm25,
        "vector": vector,
        "hyde": hyde,
        "hybrid": hybrid,
    }

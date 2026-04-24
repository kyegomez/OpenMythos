"""
Search Module

Advanced search capabilities for OpenMythos:
- Hybrid Search (BM25 + Vector + RRF)
- Query Expansion (Synonym, Generalization, HyDE)
- Reranking (Cross-encoder, LLM, Diversity)
- Memory-Optimized Search
"""

# Hybrid Search
from .hybrid_search import (
    SearchResult,
    SearchQuery,
    SearchConfig,
    BM25,
    SimpleVectorSearch,
    ReciprocalRankFusion,
    QueryExpander,
    HyDEGenerator,
    CrossEncoderReranker,
    HybridSearchEngine,
    MemorySearch,
)

# Query Expansion
from .query_expander import (
    QueryExpansionStrategy,
    SynonymExpansion,
    GeneralizationExpansion,
    SpecializationExpansion,
    PhrasingExpansion,
    DecompositionExpansion,
    MultiStrategyQueryExpander,
    HyDEQueryExpander,
    AdaptiveQueryExpander,
)

# Reranking
from .reranker import (
    RankedResult,
    BaseReranker,
    CrossEncoderReranker as CrossEncoderRanker,
    LLMReranker,
    LearningToRankReranker,
    DiversityReranker,
    EnsembleReranker,
)

__all__ = [
    # Search
    "SearchResult",
    "SearchQuery",
    "SearchConfig",
    "BM25",
    "SimpleVectorSearch",
    "ReciprocalRankFusion",
    "QueryExpander",
    "HyDEGenerator",
    "CrossEncoderReranker",
    "HybridSearchEngine",
    "MemorySearch",
    # Query Expansion
    "QueryExpansionStrategy",
    "SynonymExpansion",
    "GeneralizationExpansion",
    "SpecializationExpansion",
    "PhrasingExpansion",
    "DecompositionExpansion",
    "MultiStrategyQueryExpander",
    "HyDEQueryExpander",
    "AdaptiveQueryExpander",
    # Reranking
    "RankedResult",
    "BaseReranker",
    "CrossEncoderRanker",
    "LLMReranker",
    "LearningToRankReranker",
    "DiversityReranker",
    "EnsembleReranker",
]

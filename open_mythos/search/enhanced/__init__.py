"""
Enhanced Search Module

Advanced search features:
- Tag System for memory management
- Semantic Chunking for content segmentation
- Importance Scoring (WMR) for retrieval ranking
"""

from .tag_system import (
    Tag,
    TaggedItem,
    TagExtractor,
    TagRepository,
    TagFilter,
)

from .semantic_chunking import (
    Chunk,
    ChunkingConfig,
    FixedSizeChunking,
    SentenceChunking,
    ParagraphChunking,
    RecursiveChunking,
    SemanticChunker,
)

from .importance_scorer import (
    ImportanceScore,
    ScorerConfig,
    ImportanceScorer,
    WMRScorer,
    AdaptiveScorer,
    MultiFactorScorer,
)

__all__ = [
    # Tag System
    "Tag",
    "TaggedItem",
    "TagExtractor",
    "TagRepository",
    "TagFilter",
    # Semantic Chunking
    "Chunk",
    "ChunkingConfig",
    "FixedSizeChunking",
    "SentenceChunking",
    "ParagraphChunking",
    "RecursiveChunking",
    "SemanticChunker",
    # Importance Scoring
    "ImportanceScore",
    "ScorerConfig",
    "ImportanceScorer",
    "WMRScorer",
    "AdaptiveScorer",
    "MultiFactorScorer",
]

"""
Hybrid Retriever with RRF (Reciprocal Rank Fusion)

Combines BM25 keyword-based retrieval with vector semantic retrieval
using Reciprocal Rank Fusion for improved results.
"""

from typing import List, Optional, Dict, Any, Callable, Union
from dataclasses import dataclass, field
import logging
import numpy as np

from .bm25 import BM25Retriever, BM25plus
from .vector_retriever import VectorRetriever
from .hyde import HyDEQueryExpander, HyDEPipeline

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieval result from any retriever."""
    doc_id: str
    score: float
    content: str
    metadata: Optional[Dict[str, Any]] = None
    retriever_type: Optional[str] = None  # 'bm25', 'vector', 'hybrid'
    rank: Optional[int] = None


@dataclass
class RRFConfig:
    """Configuration for Reciprocal Rank Fusion."""
    k: float = 60  # RRF constant, typical values: 60, 100
    normalize_scores: bool = True  # Whether to normalize scores before fusion
    weight_bm25: float = 1.0  # Weight for BM25 scores
    weight_vector: float = 1.0  # Weight for vector scores


class RRFScorer:
    """
    Reciprocal Rank Fusion scorer.
    
    RRF combines rankings from multiple retrieval methods using the formula:
        RRF(d) = sum over retrievers of: 1 / (k + rank(d))
    
    where rank(d) is the rank of document d in retriever r, and k is a constant.
    
    This provides a simple yet effective way to combine multiple retrieval
    methods without requiring score normalization or learning weights.
    
    Reference:
        Reciprocal Rank Fusion outperforms Condorcet and Individual Ranking
        Methods in Multi-Stage Retrieval (2019)
    """
    
    def __init__(self, config: Optional[RRFConfig] = None):
        """
        Initialize RRF scorer.
        
        Args:
            config: RRF configuration
        """
        self.config = config or RRFConfig()
    
    def _get_rrf_score(self, rank: int) -> float:
        """
        Calculate RRF score for a given rank.
        
        Args:
            rank: Document rank (1-indexed)
            
        Returns:
            RRF score
        """
        return 1.0 / (self.config.k + rank)
    
    def fuse_results(
        self,
        results_by_retriever: Dict[str, List[RetrievalResult]],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Fuse results from multiple retrievers using RRF.
        
        Args:
            results_by_retriever: Dict mapping retriever name to list of results
            top_k: Optional limit on number of results to return
            
        Returns:
            Fused list of results sorted by RRF score
        """
        # Track document scores and metadata
        doc_scores: Dict[str, float] = {}
        doc_metadata: Dict[str, Dict[str, Any]] = {}
        
        for retriever_name, results in results_by_retriever.items():
            weight = 1.0
            if retriever_name == "bm25":
                weight = self.config.weight_bm25
            elif retriever_name == "vector":
                weight = self.config.weight_vector
            
            for rank, result in enumerate(results, start=1):
                rrf_score = self._get_rrf_score(rank) * weight
                
                if result.doc_id in doc_scores:
                    doc_scores[result.doc_id] += rrf_score
                else:
                    doc_scores[result.doc_id] = rrf_score
                    doc_metadata[result.doc_id] = {
                        "content": result.content,
                        "metadata": result.metadata,
                        "retriever_type": result.retriever_type,
                    }
        
        # Sort by fused score
        sorted_doc_ids = sorted(
            doc_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Build final results
        final_results = []
        for rank, (doc_id, score) in enumerate(sorted_doc_ids, start=1):
            metadata = doc_metadata[doc_id]
            final_results.append(RetrievalResult(
                doc_id=doc_id,
                score=score,
                content=metadata["content"],
                metadata=metadata["metadata"],
                retriever_type="hybrid",
                rank=rank,
            ))
        
        if top_k is not None:
            final_results = final_results[:top_k]
        
        return final_results
    
    def fuse_rankings(
        self,
        rankings: Dict[str, List[str]],
        scores: Optional[Dict[str, Dict[str, float]]] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fuse rankings from multiple retrievers.
        
        This is a simplified version that works with just document IDs and ranks.
        
        Args:
            rankings: Dict mapping retriever name to ordered list of doc_ids
            scores: Optional dict mapping retriever name to doc_id -> score
            top_k: Optional limit on results
            
        Returns:
            List of dicts with doc_id and fused score
        """
        doc_scores: Dict[str, float] = {}
        doc_info: Dict[str, Dict[str, Any]] = {}
        
        for retriever_name, doc_ids in rankings.items():
            weight = 1.0
            if retriever_name == "bm25":
                weight = self.config.weight_bm25
            elif retriever_name == "vector":
                weight = self.config.weight_vector
            
            for rank, doc_id in enumerate(doc_ids, start=1):
                rrf_score = self._get_rrf_score(rank) * weight
                
                if doc_id in doc_scores:
                    doc_scores[doc_id] += rrf_score
                else:
                    doc_scores[doc_id] = rrf_score
                    if scores and retriever_name in scores:
                        doc_info[doc_id] = {"score": scores[retriever_name].get(doc_id)}
                    else:
                        doc_info[doc_id] = {}
        
        # Sort and return
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, start=1):
            result = {"doc_id": doc_id, "score": score, "rank": rank}
            result.update(doc_info.get(doc_id, {}))
            results.append(result)
        
        if top_k is not None:
            results = results[:top_k]
        
        return results


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and vector retrieval with RRF.
    
    This class provides a unified interface for combining keyword-based
    (BM25) and semantic (vector) retrieval methods using Reciprocal
    Rank Fusion for improved overall retrieval performance.
    
    Features:
    - BM25 keyword-based retrieval with BM25+ algorithm
    - Vector retrieval with embedding similarity
    - Reciprocal Rank Fusion for combining results
    - Optional HyDE query expansion support
    - Configurable weights and fusion parameters
    
    Example:
        >>> hybrid = HybridRetriever(bm25=bm25_retriever, vector=vector_retriever)
        >>> results = hybrid.retrieve("What is Python?", top_k=10)
    """
    
    def __init__(
        self,
        bm25: Optional[BM25Retriever] = None,
        vector: Optional[VectorRetriever] = None,
        hyde: Optional[HyDEQueryExpander] = None,
        rrf_config: Optional[RRFConfig] = None,
        default_top_k: int = 10,
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            bm25: BM25 retriever instance
            vector: Vector retriever instance
            hyde: Optional HyDE query expander
            rrf_config: RRF configuration
            default_top_k: Default number of results to retrieve
        """
        self.bm25 = bm25
        self.vector = vector
        self.hyde = hyde
        self.rrf_config = rrf_config or RRFConfig()
        self.rrf_scorer = RRFScorer(self.rrf_config)
        self.default_top_k = default_top_k
        
        # Check what retrievers are available
        self._has_bm25 = bm25 is not None
        self._has_vector = vector is not None
        self._has_hyde = hyde is not None
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_bm25: bool = True,
        use_vector: bool = True,
        use_hyde: bool = False,
        score_threshold: Optional[float] = None,
        hyde_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid search.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            use_bm25: Whether to use BM25 retrieval
            use_vector: Whether to use vector retrieval
            use_hyde: Whether to use HyDE query expansion (requires vector)
            score_threshold: Minimum score threshold for results
            hyde_kwargs: Additional arguments for HyDE expansion
            
        Returns:
            List of retrieval results with scores
        """
        top_k = top_k or self.default_top_k
        
        results_by_retriever: Dict[str, List[RetrievalResult]] = {}
        
        # Process query
        processed_query = query
        query_embedding = None
        
        if use_hyde and self._has_hyde and self._has_vector:
            hyde_kwargs = hyde_kwargs or {}
            expanded = self.hyde.expand_query(query)
            processed_query = expanded.get("hypothetical_document", query)
            query_embedding = expanded.get("embedding")
        
        # BM25 retrieval
        if use_bm25 and self._has_bm25:
            bm25_results = self.bm25.retrieve(
                query if not use_hyde else processed_query,
                top_k=top_k * 2,  # Get more results for better fusion
                score_threshold=score_threshold,
            )
            results_by_retriever["bm25"] = [
                RetrievalResult(
                    doc_id=r["doc_id"],
                    score=r["score"],
                    content=r["content"],
                    metadata=r.get("metadata"),
                    retriever_type="bm25",
                )
                for r in bm25_results
            ]
        
        # Vector retrieval
        if use_vector and self._has_vector:
            vector_results = self.vector.retrieve(
                query if not use_hyde else processed_query,
                query_embedding=query_embedding,
                top_k=top_k * 2,
                score_threshold=score_threshold,
            )
            results_by_retriever["vector"] = [
                RetrievalResult(
                    doc_id=r["doc_id"],
                    score=r["score"],
                    content=r["content"],
                    metadata=r.get("metadata"),
                    retriever_type="vector",
                )
                for r in vector_results
            ]
        
        # Check if we have any results
        if not results_by_retriever:
            logger.warning("No retrievers available or configured")
            return []
        
        # Fuse results using RRF
        fused = self.rrf_scorer.fuse_results(results_by_retriever, top_k=top_k)
        
        # Convert to dict format
        return [
            {
                "doc_id": r.doc_id,
                "score": r.score,
                "content": r.content,
                "metadata": r.metadata,
                "retriever_type": r.retriever_type,
                "rank": r.rank,
            }
            for r in fused
        ]
    
    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_separate: bool = False,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve documents and optionally return separate BM25/vector scores.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            return_separate: Whether to return separate BM25 and vector results
            
        Returns:
            Dict with 'fused' results and optionally 'bm25' and 'vector' results
        """
        top_k = top_k or self.default_top_k
        results = {"fused": self.retrieve(query, top_k=top_k)}
        
        if return_separate:
            if self._has_bm25:
                bm25_results = self.bm25.retrieve(query, top_k=top_k)
                results["bm25"] = bm25_results
            
            if self._has_vector:
                vector_results = self.vector.retrieve(query, top_k=top_k)
                results["vector"] = vector_results
        
        return results
    
    def set_weights(self, bm25_weight: float = 1.0, vector_weight: float = 1.0) -> None:
        """
        Set weights for BM25 and vector retrieval in RRF fusion.
        
        Args:
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector scores
        """
        self.rrf_config.weight_bm25 = bm25_weight
        self.rrf_config.weight_vector = vector_weight
    
    def set_rrf_k(self, k: float) -> None:
        """
        Set the RRF constant k.
        
        Args:
            k: RRF constant (higher values reduce the impact of rank differences)
        """
        self.rrf_config.k = k


class HybridRetrievalPipeline:
    """
    Extended hybrid retrieval pipeline with preprocessing and postprocessing.
    
    This class provides additional functionality for building complete
    retrieval pipelines including:
    - Query preprocessing
    - Result postprocessing
    - Caching mechanisms
    - Multiple fusion strategies
    """
    
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        preprocessors: Optional[List[Callable]] = None,
        postprocessors: Optional[List[Callable]] = None,
    ):
        """
        Initialize hybrid retrieval pipeline.
        
        Args:
            hybrid_retriever: Base hybrid retriever
            preprocessors: Optional list of query preprocessors
            postprocessors: Optional list of result postprocessors
        """
        self.hybrid_retriever = hybrid_retriever
        self.preprocessors = preprocessors or []
        self.postprocessors = postprocessors or []
    
    def _preprocess_query(self, query: str) -> str:
        """Apply all preprocessors to the query."""
        for preprocessor in self.preprocessors:
            query = preprocessor(query)
        return query
    
    def _postprocess_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """Apply all postprocessors to results."""
        for postprocessor in self.postprocessors:
            results = postprocessor(results, query)
        return results
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents with full pipeline processing.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments for hybrid retriever
            
        Returns:
            Processed list of retrieval results
        """
        # Preprocess query
        processed_query = self._preprocess_query(query)
        
        # Retrieve
        results = self.hybrid_retriever.retrieve(
            processed_query,
            top_k=top_k,
            **kwargs
        )
        
        # Postprocess results
        results = self._postprocess_results(results, query)
        
        # Add query info to results
        for result in results:
            result["query"] = query
            result["processed_query"] = processed_query
        
        return results


# Utility functions

def normalize_scores(
    scores: Dict[str, float],
    method: str = "minmax",
) -> Dict[str, float]:
    """
    Normalize scores to a standard range.
    
    Args:
        scores: Dict mapping doc_id to score
        method: Normalization method ('minmax', 'zscore', 'rank')
        
    Returns:
        Dict of normalized scores
    """
    if not scores:
        return {}
    
    values = list(scores.values())
    
    if method == "minmax":
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return {k: 1.0 for k in scores}
        return {
            k: (v - min_val) / (max_val - min_val)
            for k, v in scores.items()
        }
    
    elif method == "zscore":
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        if std == 0:
            return {k: 0.0 for k in scores}
        return {
            k: (v - mean) / std
            for k, v in scores.items()
        }
    
    elif method == "rank":
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return {
            doc_id: 1.0 / (rank + 1)
            for rank, (doc_id, _) in enumerate(sorted_items)
        }
    
    return scores


def combine_weighted_scores(
    score_dicts: Dict[str, Dict[str, float]],
    weights: Optional[Dict[str, float]] = None,
    normalize: bool = True,
) -> Dict[str, float]:
    """
    Combine multiple score dictionaries with optional weights.
    
    Args:
        score_dicts: Dict mapping name to doc_id -> score
        weights: Optional dict mapping name to weight
        normalize: Whether to normalize each score set
        
    Returns:
        Dict mapping doc_id to combined score
    """
    if weights is None:
        weights = {name: 1.0 for name in score_dicts}
    
    # Normalize scores if requested
    if normalize:
        normalized = {
            name: normalize_scores(scores, method="minmax")
            for name, scores in score_dicts.items()
        }
    else:
        normalized = score_dicts
    
    # Get all doc_ids
    all_doc_ids = set()
    for scores in normalized.values():
        all_doc_ids.update(scores.keys())
    
    # Combine scores
    combined = {}
    for doc_id in all_doc_ids:
        total = 0.0
        for name, scores in normalized.items():
            if doc_id in scores:
                total += scores[doc_id] * weights.get(name, 1.0)
        combined[doc_id] = total
    
    return combined

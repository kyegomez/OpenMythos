"""
Reranking Module

Advanced reranking techniques for search results:
- Cross-encoder reranking
- LLM-based reranking
- Learning to rank (LTR)
- Diversity reranking
- MMR (Maximal Marginal Relevance)
"""

import math
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import re


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RankedResult:
    """A reranked result with detailed scores."""
    id: str
    content: str
    original_rank: int
    original_score: float
    rerank_score: float
    relevance_score: float = 0.0
    diversity_score: float = 0.0
    final_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Base Reranker
# ============================================================================

class BaseReranker:
    """Base class for rerankers."""
    
    def rerank(
        self,
        query: str,
        results: List[Tuple[str, str, float]],
        top_k: int = 5
    ) -> List[RankedResult]:
        """
        Rerank search results.
        
        Args:
            query: Original query
            results: List of (doc_id, content, original_score) tuples
            top_k: Number of results to return
        
        Returns:
            List of RankedResult objects
        """
        raise NotImplementedError


# ============================================================================
# Cross-Encoder Reranker
# ============================================================================

class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder based reranking.
    
    Cross-encoders process query-document pairs together,
    providing precise relevance scoring.
    
    In production, use models like:
    - cross-encoder/ms-marco-MiniLM-L-6-v2
    - cross-encoder/ms-marco-Longformer-L-0
    - sentence-transformers/cross-encoder
    """
    
    def __init__(
        self,
        model: Optional[Callable[[str, str], float]] = None,
        batch_size: int = 8
    ):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model: Function that computes relevance(query, document) -> score
            batch_size: Batch size for processing
        """
        self.model = model or self._tfidf_relevance
        self.batch_size = batch_size
    
    def rerank(
        self,
        query: str,
        results: List[Tuple[str, str, float]],
        top_k: int = 5
    ) -> List[RankedResult]:
        """Rerank using cross-encoder relevance scoring."""
        ranked = []
        
        for rank, (doc_id, content, orig_score) in enumerate(results):
            # Compute cross-encoder relevance
            relevance = self.model(query, content)
            
            # Combine original score with relevance
            # Weight: 70% relevance, 30% original score
            rerank_score = 0.7 * relevance + 0.3 * orig_score
            
            ranked.append(RankedResult(
                id=doc_id,
                content=content,
                original_rank=rank + 1,
                original_score=orig_score,
                rerank_score=rerank_score,
                relevance_score=relevance,
                final_score=rerank_score
            ))
        
        # Sort by final score
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign new ranks
        for i, r in enumerate(ranked):
            r.final_score = len(ranked) - i  # Higher is better
        
        return ranked[:top_k]
    
    def _tfidf_relevance(self, query: str, document: str) -> float:
        """
        TF-IDF based relevance (fallback when no model available).
        
        Computes relevance based on:
        - Query term overlap
        - Term frequency in query vs document
        - Position of terms
        """
        query_terms = set(query.lower().split())
        doc_lower = document.lower()
        doc_terms = set(re.findall(r'\w+', doc_lower))
        
        if not query_terms:
            return 0.0
        
        # 1. Exact term overlap
        overlap = query_terms & doc_terms
        overlap_score = len(overlap) / len(query_terms)
        
        # 2. IDF-weighted overlap
        # (Simplified - in production use actual IDF)
        idf_weighted = 0.0
        for term in overlap:
            # Approximate IDF: longer documents = more specific terms
            if len(document) > 1000:
                idf_weighted += 1.5
            else:
                idf_weighted += 1.0
        idf_score = idf_weighted / len(query_terms) if query_terms else 0
        
        # 3. Query term density in document
        density = 0.0
        for term in query_terms:
            count = doc_lower.count(term)
            if count > 0:
                density += min(count, 5) / 5  # Cap at 5 occurrences
        
        density_score = density / len(query_terms) if query_terms else 0
        
        # 4. Early position boost
        position_boost = 0.0
        first_positions = []
        for term in query_terms:
            pos = doc_lower.find(term)
            if pos != -1:
                first_positions.append(pos)
        
        if first_positions:
            avg_first_pos = sum(first_positions) / len(first_positions)
            position_boost = max(0, 1.0 - (avg_first_pos / 500))
        
        # Combine scores
        relevance = (
            overlap_score * 0.35 +
            idf_score * 0.25 +
            density_score * 0.25 +
            position_boost * 0.15
        )
        
        return relevance


# ============================================================================
# LLM Reranker
# ============================================================================

class LLM Reranker(BaseReranker):
    """
    LLM-based reranking.
    
    Uses an LLM to score and reorder results.
    Can also provide reasoning for scores.
    """
    
    def __init__(
        self,
        llm_provider: Callable[[str], str],
        scoring_prompt_template: Optional[str] = None,
        extract_score: Callable[[str], float] = None
    ):
        """
        Initialize LLM reranker.
        
        Args:
            llm_provider: LLM function that takes prompt and returns text
            scoring_prompt_template: Template for scoring prompt
            extract_score: Function to extract numeric score from LLM response
        """
        self.llm_provider = llm_provider
        self.scoring_prompt_template = scoring_prompt_template or self._default_template
        self.extract_score = extract_score or self._default_extract_score
    
    def rerank(
        self,
        query: str,
        results: List[Tuple[str, str, float]],
        top_k: int = 5
    ) -> List[RankedResult]:
        """Rerank using LLM scoring."""
        ranked = []
        
        for rank, (doc_id, content, orig_score) in enumerate(results):
            # Build scoring prompt
            prompt = self.scoring_prompt_template.format(
                query=query,
                document=content[:2000]  # Limit document length
            )
            
            # Get LLM response
            response = self.llm_provider(prompt)
            
            # Extract score
            relevance = self.extract_score(response)
            
            # Combine with original score
            rerank_score = 0.6 * relevance + 0.4 * orig_score
            
            ranked.append(RankedResult(
                id=doc_id,
                content=content,
                original_rank=rank + 1,
                original_score=orig_score,
                rerank_score=rerank_score,
                relevance_score=relevance,
                final_score=rerank_score,
                metadata={"llm_reasoning": response}
            ))
        
        # Sort by final score
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        
        for i, r in enumerate(ranked):
            r.final_score = len(ranked) - i
        
        return ranked[:top_k]
    
    def _default_template(self, query: str, document: str) -> str:
        """Default scoring prompt template."""
        return f"""Rate the relevance of the document to the query on a scale of 0-10.

Query: {query}

Document:
{document}

---

Provide a relevance score (0-10) where:
- 0: Completely irrelevant
- 5: Partially relevant, some useful information
- 10: Perfectly relevant, directly answers the query

Score: """


# ============================================================================
# Learning to Rank (LTR) Reranker
# ============================================================================

class LearningToRankReranker(BaseReranker):
    """
    Learning-to-Rank based reranking.
    
    Uses features to predict relevance scores.
    In production, use models like LightGBM, LambdaMART, or neural LTR.
    """
    
    def __init__(self):
        self.weights: Dict[str, float] = {
            "tfidf_relevance": 0.25,
            "bm25_score": 0.20,
            "semantic_similarity": 0.20,
            "importance": 0.15,
            "recency": 0.10,
            "diversity": 0.10,
        }
    
    def rerank(
        self,
        query: str,
        results: List[Tuple[str, str, float]],
        top_k: int = 5
    ) -> List[RankedResult]:
        """Rerank using learned features."""
        ranked = []
        
        for rank, (doc_id, content, orig_score) in enumerate(results):
            # Extract features
            features = self._extract_features(query, content, orig_score, rank)
            
            # Compute weighted score
            score = sum(
                features.get(key, 0) * weight
                for key, weight in self.weights.items()
            )
            
            ranked.append(RankedResult(
                id=doc_id,
                content=content,
                original_rank=rank + 1,
                original_score=orig_score,
                rerank_score=score,
                relevance_score=features.get("tfidf_relevance", 0),
                final_score=score
            ))
        
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        
        for i, r in enumerate(ranked):
            r.final_score = len(ranked) - i
        
        return ranked[:top_k]
    
    def _extract_features(
        self,
        query: str,
        content: str,
        orig_score: float,
        rank: int
    ) -> Dict[str, float]:
        """Extract features for LTR."""
        features = {}
        
        # TF-IDF relevance
        cross_encoder = CrossEncoderReranker()
        features["tfidf_relevance"] = cross_encoder._tfidf_relevance(query, content)
        
        # BM25-like score (normalized)
        features["bm25_score"] = min(orig_score, 1.0)
        
        # Semantic similarity (placeholder)
        features["semantic_similarity"] = orig_score
        
        # Position/rank feature
        features["importance"] = 1.0 / (rank + 1)
        
        # Recency (placeholder - would use timestamp)
        features["recency"] = 0.5
        
        # Diversity (placeholder)
        features["diversity"] = 0.5
        
        return features


# ============================================================================
# Diversity Reranker (MMR)
# ============================================================================

class DiversityReranker(BaseReranker):
    """
    Maximal Marginal Relevance (MMR) reranking.
    
    Balances relevance with diversity to avoid redundant results.
    
    MMR = argmax(λ * Relevance - (1-λ) * MaxSimilarity)
    """
    
    def __init__(
        self,
        base_reranker: Optional[BaseReranker] = None,
        lambda_param: float = 0.7,
        similarity_func: Optional[Callable[[str, str], float]] = None
    ):
        """
        Initialize diversity reranker.
        
        Args:
            base_reranker: Base reranker for relevance scoring
            lambda_param: Balance parameter (higher = more relevance, lower = more diversity)
            similarity_func: Function to compute document similarity
        """
        self.base_reranker = base_reranker or CrossEncoderReranker()
        self.lambda_param = lambda_param
        self.similarity_func = similarity_func or self._cosine_similarity
    
    def rerank(
        self,
        query: str,
        results: List[Tuple[str, str, float]],
        top_k: int = 5
    ) -> List[RankedResult]:
        """Rerank with diversity (MMR)."""
        if not results:
            return []
        
        # First, get relevance scores from base reranker
        base_ranked = self.base_reranker.rerank(query, results, top_k=len(results))
        
        # MMR selection
        selected: List[RankedResult] = []
        remaining = list(base_ranked)
        
        while len(selected) < top_k and remaining:
            best_mmr = float('-inf')
            best_idx = 0
            best_result: Optional[RankedResult] = None
            
            for i, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate.relevance_score
                
                # Max similarity to already selected
                max_sim = 0.0
                for selected_doc in selected:
                    sim = self.similarity_func(candidate.content, selected_doc.content)
                    max_sim = max(max_sim, sim)
                
                # MMR score
                mmr = (self.lambda_param * relevance -
                       (1 - self.lambda_param) * max_sim)
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
                    best_result = candidate
            
            if best_result:
                selected.append(best_result)
                remaining.pop(best_idx)
        
        # Assign final scores
        for i, r in enumerate(selected):
            r.diversity_score = 1.0 - (i / len(results))
            r.final_score = (
                0.5 * r.relevance_score +
                0.3 * r.diversity_score +
                0.2 * (1.0 / r.original_rank)
            )
        
        selected.sort(key=lambda x: x.final_score, reverse=True)
        
        for i, r in enumerate(selected):
            r.final_score = len(selected) - i
        
        return selected
    
    def _cosine_similarity(self, text1: str, text2: str) -> float:
        """Compute simple cosine similarity based on term overlap."""
        terms1 = set(re.findall(r'\w+', text1.lower()))
        terms2 = set(re.findall(r'\w+', text2.lower()))
        
        if not terms1 or not terms2:
            return 0.0
        
        intersection = terms1 & terms2
        union = terms1 | terms2
        
        return len(intersection) / len(union)


# ============================================================================
# Ensemble Reranker
# ============================================================================

class EnsembleReranker(BaseReranker):
    """
    Ensemble reranker combining multiple rerankers.
    
    Combines:
    - Cross-encoder reranking
    - LLM reranking (if available)
    - Diversity reranking
    """
    
    def __init__(
        self,
        rerankers: Optional[List[BaseReranker]] = None,
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble reranker.
        
        Args:
            rerankers: List of rerankers to combine
            weights: Weights for each reranker (default: equal)
        """
        self.rerankers = rerankers or [
            CrossEncoderReranker(),
            DiversityReranker(lambda_param=0.8),
        ]
        
        self.weights = weights or [1.0] * len(self.rerankers)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def rerank(
        self,
        query: str,
        results: List[Tuple[str, str, float]],
        top_k: int = 5
    ) -> List[RankedResult]:
        """Rerank using ensemble of rerankers."""
        # Get scores from each reranker
        all_scores: Dict[str, List[float]] = {doc_id: [] for doc_id, _, _ in results}
        
        for reranker in self.rerankers:
            try:
                ranked = reranker.rerank(query, results, top_k=len(results))
                
                # Normalize scores to 0-1 range
                max_score = max(r.final_score for r in ranked) if ranked else 1
                min_score = min(r.final_score for r in ranked) if ranked else 0
                range_score = max_score - min_score if max_score != min_score else 1
                
                for r in ranked:
                    normalized = (r.final_score - min_score) / range_score
                    all_scores[r.id].append(normalized)
            except Exception:
                continue
        
        # Combine scores
        final_scores: Dict[str, float] = {}
        for doc_id, scores in all_scores.items():
            if scores:
                final_scores[doc_id] = sum(
                    s * w for s, w in zip(scores, self.weights)
                )
            else:
                final_scores[doc_id] = 0.0
        
        # Create ranked results
        ranked = []
        for doc_id, content, orig_score in results:
            score = final_scores.get(doc_id, 0.0)
            ranked.append(RankedResult(
                id=doc_id,
                content=content,
                original_rank=0,
                original_score=orig_score,
                rerank_score=score,
                final_score=score
            ))
        
        ranked.sort(key=lambda x: x.final_score, reverse=True)
        
        for i, r in enumerate(ranked):
            r.final_score = len(ranked) - i
        
        return ranked[:top_k]


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "RankedResult",
    "BaseReranker",
    "CrossEncoderReranker",
    "LLMReranker",
    "LearningToRankReranker",
    "DiversityReranker",
    "EnsembleReranker",
]

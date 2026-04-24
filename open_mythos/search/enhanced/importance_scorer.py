"""
Importance Scoring Module

Implements importance scoring for memory items:
- Weighted Memory Retrieval (WMR)
- Access frequency decay
- Recency scoring
- Relevance scoring
- Combined importance scoring
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time
import math


@dataclass
class ImportanceScore:
    """Complete importance score breakdown."""
    item_id: str
    final_score: float
    
    # Component scores
    base_score: float = 0.0
    access_score: float = 0.0
    recency_score: float = 0.0
    relevance_score: float = 0.0
    decay_score: float = 0.0
    
    # Metadata
    access_count: int = 0
    last_accessed: float = 0.0
    created_at: float = 0.0
    importance: float = 1.0
    
    @property
    def components(self) -> Dict[str, float]:
        return {
            "base": self.base_score,
            "access": self.access_score,
            "recency": self.recency_score,
            "relevance": self.relevance_score,
            "decay": self.decay_score,
            "final": self.final_score,
        }


@dataclass
class ScorerConfig:
    """Configuration for importance scoring."""
    # Access scoring
    access_weight: float = 0.25
    access_decay: float = 0.9  # How much each access matters less
    
    # Recency scoring
    recency_weight: float = 0.20
    recency_half_life: float = 3600 * 24 * 7  # 7 days in seconds
    
    # Relevance scoring
    relevance_weight: float = 0.30
    
    # Base importance
    base_weight: float = 0.15
    
    # Decay
    decay_weight: float = 0.10
    decay_rate: float = 0.01  # Per day
    
    # Time
    current_time_func: Callable[[], float] = field(default_factory=lambda: datetime.now().timestamp())


class ImportanceScorer:
    """
    Calculates importance scores for memory items.
    
    Combines:
    - Base importance (user-defined or content-based)
    - Access frequency and recency
    - Time-based decay
    - Relevance to current query
    """
    
    def __init__(self, config: Optional[ScorerConfig] = None):
        self.config = config or ScorerConfig()
    
    def score(
        self,
        item_id: str,
        base_importance: float = 1.0,
        access_count: int = 0,
        last_accessed: float = 0.0,
        created_at: float = 0.0,
        relevance_score: float = 0.0,
        current_time: Optional[float] = None
    ) -> ImportanceScore:
        """
        Calculate complete importance score.
        
        Returns ImportanceScore with all components.
        """
        now = current_time or self.config.current_time_func()
        
        # 1. Base score (normalized importance)
        base_score = self._calc_base_score(base_importance)
        
        # 2. Access score (frequency + recency)
        access_score = self._calc_access_score(
            access_count, last_accessed, now
        )
        
        # 3. Recency score
        recency_score = self._calc_recency_score(
            last_accessed or created_at, now
        )
        
        # 4. Relevance score
        relevance = self._calc_relevance_score(relevance_score)
        
        # 5. Time decay
        decay = self._calc_decay(created_at, now)
        
        # Combine scores
        final = (
            base_score * self.config.base_weight +
            access_score * self.config.access_weight +
            recency_score * self.config.recency_weight +
            relevance * self.config.relevance_weight +
            decay * self.config.decay_weight
        )
        
        return ImportanceScore(
            item_id=item_id,
            final_score=final,
            base_score=base_score,
            access_score=access_score,
            recency_score=recency_score,
            relevance_score=relevance,
            decay_score=decay,
            access_count=access_count,
            last_accessed=last_accessed,
            created_at=created_at,
            importance=base_importance,
        )
    
    def _calc_base_score(self, importance: float) -> float:
        """Normalize base importance to 0-1."""
        return min(1.0, max(0.0, importance / 10.0))
    
    def _calc_access_score(
        self,
        access_count: int,
        last_accessed: float,
        current_time: float
    ) -> float:
        """
        Calculate access-based score.
        
        Considers both frequency and recency of access.
        """
        if access_count == 0:
            return 0.0
        
        # Frequency component (logarithmic to reduce extreme values)
        freq_score = math.log(1 + access_count) / math.log(11)  # Max ~1.0
        
        # Recency component
        if last_accessed > 0:
            time_diff = current_time - last_accessed
            recency = math.exp(-time_diff / (self.config.recency_half_life * 2))
        else:
            recency = 0.0
        
        # Combine (weighted toward frequency)
        return 0.6 * freq_score + 0.4 * recency
    
    def _calc_recency_score(
        self,
        timestamp: float,
        current_time: float
    ) -> float:
        """
        Calculate recency score using exponential decay.
        
        Score = e^(-lambda * t)
        where lambda = ln(2) / half_life
        """
        if timestamp <= 0:
            return 0.0
        
        time_diff = current_time - timestamp
        half_life = self.config.recency_half_life
        
        # Exponential decay with half-life
        decay_constant = math.log(2) / half_life
        score = math.exp(-decay_constant * time_diff)
        
        return min(1.0, score)
    
    def _calc_relevance_score(self, relevance: float) -> float:
        """Normalize relevance score."""
        return min(1.0, max(0.0, relevance))
    
    def _calc_decay(
        self,
        created_at: float,
        current_time: float
    ) -> float:
        """
        Calculate time-based decay score.
        
        Older items decay, but never go below a floor.
        """
        if created_at <= 0:
            return 1.0
        
        days_elapsed = (current_time - created_at) / (3600 * 24)
        
        # Exponential decay
        decay = math.exp(-self.config.decay_rate * days_elapsed)
        
        # Floor at 0.1
        return max(0.1, decay)


class WMRScorer(ImportanceScorer):
    """
    Weighted Memory Retrieval (WMR) Scorer.
    
    Extends ImportanceScorer with:
    - Query-dependent scoring
    - Tag-based boosting
    - Layer-aware weighting
    """
    
    def __init__(self, config: Optional[ScorerConfig] = None):
        super().__init__(config)
        
        # Layer weights (working > short_term > long_term)
        self.layer_weights = {
            "working": 1.0,
            "short_term": 0.7,
            "long_term": 0.4,
        }
        
        # Tag weights
        self.tag_weights: Dict[str, float] = {}
    
    def score_with_query(
        self,
        item_id: str,
        query: str,
        item_content: str,
        layer: str = "short_term",
        **kwargs
    ) -> ImportanceScore:
        """Score with query-dependent relevance."""
        # Base score
        base_score = super().score(item_id, **kwargs)
        
        # Query relevance
        query_terms = set(query.lower().split())
        content_terms = set(item_content.lower().split())
        
        if query_terms:
            overlap = len(query_terms & content_terms)
            query_relevance = overlap / len(query_terms)
        else:
            query_relevance = 0.0
        
        # Boost base relevance with query
        base_score.relevance_score = (
            0.5 * base_score.relevance_score +
            0.5 * query_relevance
        )
        
        # Apply layer weight
        layer_weight = self.layer_weights.get(layer, 0.5)
        base_score.final_score *= layer_weight
        
        return base_score
    
    def set_tag_weight(self, tag: str, weight: float) -> None:
        """Set custom weight for a tag."""
        self.tag_weights[tag.lower()] = weight
    
    def get_tag_weight(self, tag: str) -> float:
        """Get weight for a tag."""
        return self.tag_weights.get(tag.lower(), 1.0)


class AdaptiveScorer:
    """
    Adaptive scorer that adjusts weights based on context.
    
    Learns from user behavior to optimize scoring.
    """
    
    def __init__(self, base_config: Optional[ScorerConfig] = None):
        self.base_config = base_config or ScorerConfig()
        self.scorer = ImportanceScorer(self.base_config)
        
        # Adaptive weights (learned from feedback)
        self.adaptive_weights = {
            "access": 1.0,
            "recency": 1.0,
            "relevance": 1.0,
        }
        
        # Feedback history
        self.feedback_history: List[Tuple[str, float]] = []  # (item_id, rating)
    
    def score(
        self,
        item_id: str,
        feedback_boost: float = 0.0,
        **kwargs
    ) -> ImportanceScore:
        """Score with adaptive weighting."""
        score = self.scorer.score(item_id, **kwargs)
        
        # Apply adaptive weights
        score.final_score = (
            score.final_score *
            self.adaptive_weights.get("access", 1.0) *
            self.adaptive_weights.get("recency", 1.0) *
            self.adaptive_weights.get("relevance", 1.0)
        )
        
        # Apply feedback boost
        if feedback_boost != 0:
            score.final_score *= (1 + feedback_boost)
        
        return score
    
    def record_feedback(self, item_id: str, rating: float) -> None:
        """
        Record user feedback to adjust weights.
        
        rating: -1 (negative) to 1 (positive)
        """
        self.feedback_history.append((item_id, rating))
        
        # Adjust weights based on feedback
        if len(self.feedback_history) > 10:
            # Analyze recent feedback
            recent = self.feedback_history[-10:]
            avg_rating = sum(r for _, r in recent) / len(recent)
            
            # If positive feedback, boost relevance
            # If negative, reduce
            adjustment = avg_rating * 0.1
            
            if avg_rating > 0.2:
                self.adaptive_weights["relevance"] += adjustment
            elif avg_rating < -0.2:
                self.adaptive_weights["access"] -= adjustment
    
    def decay_weights(self) -> None:
        """Gradually return adaptive weights to baseline."""
        for key in self.adaptive_weights:
            self.adaptive_weights[key] += (1.0 - self.adaptive_weights[key]) * 0.01


class MultiFactorScorer:
    """
    Multi-factor importance scoring.
    
    Supports custom factors and their weights.
    """
    
    def __init__(self):
        self.factors: Dict[str, Callable] = {}
        self.weights: Dict[str, float] = {}
    
    def register_factor(
        self,
        name: str,
        func: Callable[[Any], float],
        weight: float = 1.0
    ) -> None:
        """Register a custom scoring factor."""
        self.factors[name] = func
        self.weights[name] = weight
    
    def score(self, item: Any, **context) -> Tuple[float, Dict[str, float]]:
        """
        Score an item using all registered factors.
        
        Returns (final_score, factor_scores)
        """
        factor_scores = {}
        
        for name, func in self.factors.items():
            try:
                factor_scores[name] = func(item, **context)
            except Exception:
                factor_scores[name] = 0.0
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            normalized_weights = {
                k: v / total_weight for k, v in self.weights.items()
            }
        else:
            normalized_weights = {k: 1.0 / len(self.factors) for k in self.factors}
        
        # Calculate weighted score
        final_score = sum(
            factor_scores.get(name, 0.0) * normalized_weights.get(name, 0.0)
            for name in self.factors
        )
        
        return final_score, factor_scores


# Common factor functions

def access_count_factor(item: Any, **context) -> float:
    """Factor based on access count."""
    count = getattr(item, "access_count", 0) or context.get("access_count", 0)
    return min(1.0, math.log(1 + count) / math.log(11))


def recency_factor(item: Any, **context) -> float:
    """Factor based on recency."""
    last_accessed = getattr(item, "last_accessed", 0) or context.get("last_accessed", 0)
    created_at = getattr(item, "created_at", 0) or context.get("created_at", 0)
    timestamp = last_accessed or created_at
    
    if timestamp <= 0:
        return 0.0
    
    now = context.get("current_time", datetime.now().timestamp())
    half_life = 3600 * 24 * 7  # 7 days
    
    return math.exp(-math.log(2) * (now - timestamp) / half_life)


def importance_factor(item: Any, **context) -> float:
    """Factor based on base importance."""
    importance = getattr(item, "importance", 1.0) or context.get("importance", 1.0)
    return min(1.0, importance / 10.0)


def tag_match_factor(item: Any, **context) -> float:
    """Factor based on tag match with query."""
    query_tags = context.get("query_tags", set())
    if not query_tags:
        return 0.5
    
    item_tags = getattr(item, "tags", set()) or set()
    
    if not item_tags:
        return 0.5
    
    overlap = len(query_tags & item_tags)
    return overlap / max(len(query_tags), len(item_tags))


__all__ = [
    "ImportanceScore",
    "ScorerConfig",
    "ImportanceScorer",
    "WMRScorer",
    "AdaptiveScorer",
    "MultiFactorScorer",
    "access_count_factor",
    "recency_factor",
    "importance_factor",
    "tag_match_factor",
]

"""
BM25+ Retriever Implementation

BM25 (Best Matching 25) is a classic probabilistic retrieval algorithm.
BM25+ is an improvement that handles document length normalization better.
"""

from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import math
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


@dataclass
class BM25Document:
    """Represents a document for BM25 indexing."""
    id: str
    content: str
    tokens: List[str]
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BM25Result:
    """Represents a BM25 retrieval result."""
    doc_id: str
    score: float
    content: str
    metadata: Optional[Dict[str, Any]] = None


class BM25plus:
    """
    BM25+ implementation for keyword-based retrieval.
    
    BM25+ improves upon standard BM25 by adding a small constant to the
    term frequency component, which prevents documents with very long
    fields from receiving artificially low scores.
    
    Formula:
        BM25+ = sum over terms of:
            (tf + delta) / (tf + k1 * (1 - b + b * |d| / avgdl)) * idf
    
    where:
        - tf: term frequency in document
        - k1: term frequency saturation parameter (typical: 1.2-2.0)
        - b: length normalization parameter (typical: 0.75)
        - |d|: document length
        - avgdl: average document length
        - delta: the "+1" constant (typical: 1.0)
        - idf: inverse document frequency
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        min_doc_freq: int = 1,
        max_doc_freq: Optional[int] = None,
        stopwords: Optional[set] = None,
    ):
        """
        Initialize BM25+ retriever.
        
        Args:
            k1: Term frequency saturation parameter. Higher values give more
                importance to term frequency. Typical range: 1.2-2.0
            b: Length normalization parameter. Controls how much document
                length affects scoring. Typical value: 0.75
            delta: The constant added to term frequency. This prevents
                very long documents from being penalized too heavily.
                Typical value: 1.0
            min_doc_freq: Minimum document frequency for a term to be indexed
            max_doc_freq: Maximum document frequency (as ratio or count)
            stopwords: Optional set of stopwords to filter out
        """
        self.k1 = k1
        self.b = b
        self.delta = delta
        self.min_doc_freq = min_doc_freq
        self.max_doc_freq = max_doc_freq
        self.stopwords = stopwords or self._default_stopwords()
        
        self.doc_count = 0
        self.avg_doc_length = 0.0
        self.doc_lengths: Dict[str, int] = {}
        self.doc_term_freqs: Dict[str, Counter] = {}
        self.term_doc_freqs: Counter = Counter()
        self.term_idf: Dict[str, float] = {}
        self.documents: Dict[str, BM25Document] = {}
        
    def _default_stopwords(self) -> set:
        """Return a basic set of English stopwords."""
        return {
            "a", "an", "and", "are", "as", "at", "be", "by", "for",
            "from", "has", "he", "in", "is", "it", "its", "of", "on",
            "that", "the", "to", "was", "were", "will", "with"
        }
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into individual terms.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of lowercase tokens
        """
        # Simple tokenization: lowercase, split on non-alphanumeric
        tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        # Filter stopwords and short tokens
        return [t for t in tokens if t not in self.stopwords and len(t) > 1]
    
    def _calculate_idf(self, term: str, doc_freq: int) -> float:
        """
        Calculate IDF for a term.
        
        Args:
            term: The term
            doc_freq: Number of documents containing the term
            
        Returns:
            IDF value
        """
        # Smoothed IDF formula to avoid zero IDF for terms not in collection
        # IDF = log((N - df + 0.5) / (df + 0.5)) + 1
        n = self.doc_count
        df = doc_freq
        return math.log((n - df + 0.5) / (df + 0.5) + 1)
    
    def index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index a collection of documents.
        
        Args:
            documents: List of dicts with 'id' and 'content' keys
        """
        self.doc_count = len(documents)
        self.documents = {}
        self.doc_lengths = {}
        self.doc_term_freqs = {}
        self.term_doc_freqs = Counter()
        
        total_length = 0
        
        for doc in documents:
            doc_id = doc.get("id", str(hash(doc["content"])))
            content = doc["content"]
            metadata = doc.get("metadata")
            
            tokens = self._tokenize(content)
            doc_length = len(tokens)
            total_length += doc_length
            
            # Count term frequencies
            term_freqs = Counter(tokens)
            
            # Store document data
            self.documents[doc_id] = BM25Document(
                id=doc_id,
                content=content,
                tokens=tokens,
                metadata=metadata
            )
            self.doc_lengths[doc_id] = doc_length
            self.doc_term_freqs[doc_id] = term_freqs
            
            # Update term document frequencies
            for term in set(tokens):
                self.term_doc_freqs[term] += 1
        
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 1
        
        # Calculate IDF for all terms
        self.term_idf = {}
        for term, doc_freq in self.term_doc_freqs.items():
            if doc_freq >= self.min_doc_freq:
                if self.max_doc_freq is None or doc_freq <= self.max_doc_freq:
                    self.term_idf[term] = self._calculate_idf(term, doc_freq)
    
    def add_document(self, doc: Dict[str, Any]) -> None:
        """
        Add a single document to the index.
        
        Args:
            doc: Dict with 'id' and 'content' keys
        """
        doc_id = doc.get("id", str(hash(doc["content"])))
        content = doc["content"]
        metadata = doc.get("metadata")
        
        tokens = self._tokenize(content)
        doc_length = len(tokens)
        
        # Handle existing document replacement
        if doc_id in self.documents:
            old_tokens = self.documents[doc_id].tokens
            for term in set(old_tokens):
                self.term_doc_freqs[term] -= 1
        
        # Update doc count and average
        self.doc_count += 1
        old_avg = self.avg_doc_length
        self.avg_doc_length = ((old_avg * (self.doc_count - 1)) + doc_length) / self.doc_count
        
        # Store document
        term_freqs = Counter(tokens)
        self.documents[doc_id] = BM25Document(
            id=doc_id,
            content=content,
            tokens=tokens,
            metadata=metadata
        )
        self.doc_lengths[doc_id] = doc_length
        self.doc_term_freqs[doc_id] = term_freqs
        
        # Update term frequencies
        for term in set(tokens):
            self.term_doc_freqs[term] += 1
            if term not in self.term_idf:
                self.term_idf[term] = self._calculate_idf(term, self.term_doc_freqs[term])
    
    def _score_document(
        self,
        query_tokens: List[str],
        doc_id: str
    ) -> float:
        """
        Calculate BM25+ score for a single document.
        
        Args:
            query_tokens: Tokenized query
            doc_id: Document ID
            
        Returns:
            BM25+ score
        """
        if doc_id not in self.documents:
            return 0.0
        
        doc_length = self.doc_lengths[doc_id]
        term_freqs = self.doc_term_freqs[doc_id]
        
        score = 0.0
        doc_unique_terms = set(term_freqs.keys())
        
        for term in query_tokens:
            if term not in self.term_idf:
                continue
            
            tf = term_freqs.get(term, 0)
            idf = self.term_idf[term]
            
            # BM25+ formula
            numerator = (tf + self.delta)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
            
            term_score = idf * (numerator / denominator)
            score += term_score
        
        return score
    
    def get_scores(self, query: str) -> Dict[str, float]:
        """
        Get BM25+ scores for all documents for a query.
        
        Args:
            query: Query string
            
        Returns:
            Dict mapping doc_id to score
        """
        query_tokens = self._tokenize(query)
        
        scores = {}
        for doc_id in self.documents:
            scores[doc_id] = self._score_document(query_tokens, doc_id)
        
        return scores
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[BM25Result]:
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            score_threshold: Minimum score threshold
            
        Returns:
            List of BM25Result objects sorted by score descending
        """
        query_tokens = self._tokenize(query)
        
        # Score all documents
        scores = []
        for doc_id in self.documents:
            score = self._score_document(query_tokens, doc_id)
            if score > 0:
                doc = self.documents[doc_id]
                scores.append(BM25Result(
                    doc_id=doc_id,
                    score=score,
                    content=doc.content,
                    metadata=doc.metadata
                ))
        
        # Sort by score descending
        scores.sort(key=lambda x: x.score, reverse=True)
        
        # Apply thresholds
        if score_threshold is not None:
            scores = [s for s in scores if s.score >= score_threshold]
        
        return scores[:top_k]
    
    def get_bm25_scores_for_results(
        self,
        query: str,
        doc_ids: List[str]
    ) -> List[float]:
        """
        Get BM25 scores for specific documents.
        
        Useful when combining BM25 with vector retrieval.
        
        Args:
            query: Query string
            doc_ids: List of document IDs to score
            
        Returns:
            List of scores in same order as doc_ids
        """
        query_tokens = self._tokenize(query)
        return [self._score_document(query_tokens, doc_id) for doc_id in doc_ids]


class BM25Retriever:
    """
    High-level BM25 retriever interface.
    
    This class provides a simpler interface for using BM25 retrieval
    and can be easily integrated with other retrieval methods.
    """
    
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        delta: float = 1.0,
        min_doc_freq: int = 1,
        **kwargs
    ):
        """
        Initialize BM25 retriever.
        
        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
            delta: BM25+ constant
            min_doc_freq: Minimum document frequency
            **kwargs: Additional arguments (ignored)
        """
        self.bm25 = BM25plus(k1=k1, b=b, delta=delta, min_doc_freq=min_doc_freq)
        self._is_indexed = False
    
    def index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents for retrieval.
        
        Args:
            documents: List of dicts with 'id' and 'content' keys
        """
        self.bm25.index(documents)
        self._is_indexed = True
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the index.
        
        Args:
            documents: List of dicts with 'id' and 'content' keys
        """
        for doc in documents:
            self.bm25.add_document(doc)
        self._is_indexed = True
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            score_threshold: Minimum score threshold
            
        Returns:
            List of dicts with 'doc_id', 'score', 'content', 'metadata'
        """
        if not self._is_indexed:
            logger.warning("No documents indexed yet")
            return []
        
        results = self.bm25.retrieve(query, top_k, score_threshold)
        
        return [
            {
                "doc_id": r.doc_id,
                "score": r.score,
                "content": r.content,
                "metadata": r.metadata,
            }
            for r in results
        ]
    
    def get_scores(self, query: str) -> Dict[str, float]:
        """
        Get BM25 scores for all documents.
        
        Args:
            query: Query string
            
        Returns:
            Dict mapping doc_id to score
        """
        if not self._is_indexed:
            return {}
        return self.bm25.get_scores(query)

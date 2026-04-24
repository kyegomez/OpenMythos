"""
Hybrid Search Module

Implements hybrid search combining:
- BM25 (sparse keyword search)
- Dense Vector (semantic search)  
- Reciprocal Rank Fusion (RRF) for result merging

Query Enhancement:
- Query Expansion (generate multiple query variants)
- HyDE (Hypothetical Document Embedding)

Reranking:
- Cross-encoder reranking for precision
"""

import re
import math
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class SearchResult:
    """A single search result."""
    id: str
    content: str
    score: float
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"  # "bm25", "vector", "hybrid"
    
    def __lt__(self, other: "SearchResult") -> bool:
        return self.score > other.score


@dataclass
class SearchQuery:
    """A search query with metadata."""
    text: str
    original: str = ""
    expanded: List[str] = field(default_factory=list)
   hyde_doc: str = ""
    tags: List[str] = field(default_factory=list)
    layer: Optional[str] = None
    limit: int = 10
    offset: int = 0


@dataclass
class SearchConfig:
    """Configuration for hybrid search."""
    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_enabled: bool = True
    
    # Vector search parameters
    vector_enabled: bool = True
    vector_weight: float = 0.5
    
    # RRF parameters
    rrf_k: int = 60  # RRF constant
    rrf_enabled: bool = True
    
    # Query expansion
    expansion_enabled: bool = True
    expansion_count: int = 3
    expansion_weight: float = 0.3
    
    # HyDE
    hyde_enabled: bool = False
    hyde_weight: float = 0.2
    
    # Reranking
    rerank_enabled: bool = True
    rerank_top_k: int = 20
    rerank_final: int = 5
    
    # Weights for fusion
    bm25_weight: float = 0.4
    vector_weight_fusion: float = 0.6


# ============================================================================
# BM25 Implementation
# ============================================================================

class BM25:
    """
    BM25 (Best Matching 25) - Okapi BM25 implementation.
    
    A probabilistic relevance model for sparse keyword search.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []
        self.num_docs: int = 0
        self.corpus: List[str] = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words."""
        text = text.lower()
        words = re.findall(r'\w+', text)
        return words
    
    def _calculate_idf(self) -> None:
        """Calculate IDF for all terms."""
        for term, df in self.doc_freqs.items():
            self.idf[term] = math.log(
                (self.num_docs - df + 0.5) / (df + 0.5) + 1
            )
    
    def index(self, corpus: List[str]) -> None:
        """Index a corpus of documents."""
        self.corpus = corpus
        self.num_docs = len(corpus)
        self.doc_lengths = []
        self.doc_freqs = defaultdict(int)
        self.doc_term_freqs = []
        
        for doc in corpus:
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))
            
            term_freqs = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1
                self.doc_freqs[token] += 1
            
            self.doc_term_freqs.append(dict(term_freqs))
        
        self.avgdl = sum(self.doc_lengths) / self.num_docs if self.num_docs > 0 else 0
        self._calculate_idf()
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search the indexed corpus.
        
        Returns list of (doc_index, score) tuples.
        """
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        scores = []
        for doc_idx in range(self.num_docs):
            score = self._calculate_score(query_tokens, doc_idx)
            if score > 0:
                scores.append((doc_idx, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
    
    def _calculate_score(self, query_tokens: List[str], doc_idx: int) -> float:
        """Calculate BM25 score for a document."""
        doc_len = self.doc_lengths[doc_idx]
        term_freqs = self.doc_term_freqs[doc_idx]
        
        score = 0.0
        for token in query_tokens:
            if token not in term_freqs:
                continue
            
            tf = term_freqs[token]
            idf = self.idf.get(token, 0)
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            
            score += idf * (numerator / denominator)
        
        return score


# ============================================================================
# Vector Search (Simplified Embedding-based)
# ============================================================================

class SimpleVectorSearch:
    """
    Simple vector-based semantic search using TF-IDF embeddings.
    
    For production, replace with actual embedding models (OpenAI, SentenceTransformers, etc.)
    """
    
    def __init__(self):
        self.doc_vectors: List[Dict[str, float]] = []
        self.corpus: List[str] = []
        self.doc_freqs: Dict[str, int] = {}
        self.num_docs: int = 0
        self.avgdl: float = 0.0
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text."""
        text = text.lower()
        return re.findall(r'\w+', text)
    
    def _tfidf(self, tokens: List[str], doc_len: int) -> Dict[str, float]:
        """Calculate TF-IDF weights."""
        tf = defaultdict(int)
        for token in tokens:
            tf[token] += 1
        
        tfidf = {}
        for token, count in tf.items():
            # TF: 1 + log(count)
            tf_val = 1 + math.log(count) if count > 0 else 0
            # IDF from doc frequency
            df = self.doc_freqs.get(token, 0)
            idf_val = math.log(self.num_docs / (df + 1)) + 1
            tfidf[token] = tf_val * idf_val
        
        return tfidf
    
    def _cosine_sim(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors."""
        common_keys = set(vec1.keys()) & set(vec2.keys())
        if not common_keys:
            return 0.0
        
        dot_product = sum(vec1[k] * vec2[k] for k in common_keys)
        norm1 = math.sqrt(sum(v * v for v in vec1.values()))
        norm2 = math.sqrt(sum(v * v for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def index(self, corpus: List[str]) -> None:
        """Index a corpus."""
        self.corpus = corpus
        self.num_docs = len(corpus)
        self.doc_vectors = []
        self.doc_freqs = defaultdict(int)
        
        doc_lengths = []
        for doc in corpus:
            tokens = self._tokenize(doc)
            doc_lengths.append(len(tokens))
            
            for token in set(tokens):
                self.doc_freqs[token] += 1
        
        self.avgdl = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0
        
        for doc in corpus:
            tokens = self._tokenize(doc)
            vec = self._tfidf(tokens, len(tokens))
            # L2 normalize
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                vec = {k: v / norm for k, v in vec.items()}
            self.doc_vectors.append(vec)
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search using semantic similarity."""
        query_tokens = self._tokenize(query)
        query_len = len(query_tokens)
        
        if not query_tokens:
            return []
        
        query_vec = self._tfidf(query_tokens, query_len)
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in query_vec.values()))
        if norm > 0:
            query_vec = {k: v / norm for k, v in query_vec.items()}
        
        scores = []
        for doc_idx in range(self.num_docs):
            sim = self._cosine_sim(query_vec, self.doc_vectors[doc_idx])
            if sim > 0:
                scores.append((doc_idx, sim))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================================================
# Reciprocal Rank Fusion (RRF)
# ============================================================================

class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion for combining multiple ranked lists.
    
    RRF score = Σ 1/(k + rank_i)
    
    where k is a constant (typically 60) and rank_i is the rank in list i.
    """
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse(
        self,
        rankings: List[List[Tuple[str, float]]],
        weights: Optional[List[float]] = None
    ) -> List[Tuple[str, float]]:
        """
        Fuse multiple rankings into a single ranked list.
        
        Args:
            rankings: List of ranked lists, each containing (doc_id, score) tuples
            weights: Optional weights for each ranking (default: equal weights)
        
        Returns:
            Fused ranked list of (doc_id, rrf_score) tuples
        """
        if not rankings:
            return []
        
        if weights is None:
            weights = [1.0] * len(rankings)
        
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_sources: Dict[str, int] = {}  # Track which ranking each doc came from
        
        for rank_idx, ranking in enumerate(rankings):
            weight = weights[rank_idx]
            
            for rank, (doc_id, _score) in enumerate(ranking, 1):
                # RRF formula with weight
                rrf_score = weight * (1 / (self.k + rank))
                rrf_scores[doc_id] += rrf_score
                doc_sources[doc_id] = rank_idx
        
        # Sort by RRF score descending
        sorted_results = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results


# ============================================================================
# Query Expander
# ============================================================================

class QueryExpander:
    """
    Expand queries with multiple strategies:
    - Synonym expansion
    - Phrasing variations
    - Generalization (step-back)
    - Specialization (drill-down)
    """
    
    def __init__(self):
        # Common synonyms for expansion
        self.synonyms: Dict[str, List[str]] = {
            "find": ["search", "locate", "get", "retrieve"],
            "show": ["display", "list", "view", "get"],
            "create": ["make", "add", "new", "insert"],
            "update": ["edit", "modify", "change", "alter"],
            "delete": ["remove", "drop", "erase", "clear"],
            "error": ["bug", "issue", "problem", "failure"],
            "fix": ["repair", "resolve", "debug", "correct"],
            "test": ["check", "verify", "validate", "spec"],
            "config": ["settings", "options", "preferences"],
            "run": ["execute", "start", "launch", "begin"],
        }
    
    def expand(self, query: str, strategies: List[str] = None) -> List[str]:
        """
        Generate expanded query variations.
        
        Args:
            query: Original query
            strategies: List of strategies to use ["synonym", "generalize", "specialize", "rewrite"]
        
        Returns:
            List of expanded queries
        """
        if strategies is None:
            strategies = ["synonym", "rewrite"]
        
        expanded = [query]  # Always include original
        
        for strategy in strategies:
            if strategy == "synonym":
                expanded.extend(self._expand_synonyms(query))
            elif strategy == "generalize":
                expanded.append(self._generalize(query))
            elif strategy == "specialize":
                expanded.extend(self._specialize(query))
            elif strategy == "rewrite":
                expanded.append(self._rewrite(query))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_expanded = []
        for q in expanded:
            if q not in seen:
                seen.add(q)
                unique_expanded.append(q)
        
        return unique_expanded
    
    def _expand_synonyms(self, query: str) -> List[str]:
        """Replace words with synonyms."""
        words = query.lower().split()
        expanded = []
        
        for i, word in enumerate(words):
            if word in self.synonyms:
                for synonym in self.synonyms[word][:2]:  # Limit to 2 synonyms
                    new_words = words[:i] + [synonym] + words[i+1:]
                    expanded.append(" ".join(new_words))
        
        return expanded
    
    def _generalize(self, query: str) -> str:
        """Create a more general version of the query."""
        # Remove specific details to generalize
        words = query.split()
        
        # Remove numbers, dates, specific identifiers
        generalized = [w for w in words if not re.match(r'^\d+$', w)]
        
        # Remove adjectives that are too specific
        specific_adj = ["specific", "particular", "exact", "precise"]
        generalized = [w for w in generalized if w.lower() not in specific_adj]
        
        return " ".join(generalized) if generalized else query
    
    def _specialize(self, query: str) -> List[str]:
        """Create more specific versions."""
        specialized = []
        
        # Add context keywords
        if "file" in query.lower():
            specialized.append(query + " file path")
        if "error" in query.lower():
            specialized.append(query + " error message")
        if "config" in query.lower():
            specialized.append(query + " configuration settings")
        
        return specialized[:2]  # Limit to 2
    
    def _rewrite(self, query: str) -> str:
        """Rewrite query in different phrasing."""
        # Simple rewrites - in production use LLM
        rewrites = [
            ("how to", "ways to"),
            ("how do i", "how can I"),
            ("can't", "cannot"),
            ("don't", "do not"),
        ]
        
        result = query
        for old, new in rewrites:
            if old.lower() in query.lower():
                result = re.sub(old, new, result, flags=re.IGNORECASE)
                break
        
        return result


# ============================================================================
# HyDE (Hypothetical Document Embedding)
# ============================================================================

class HyDEGenerator:
    """
    HyDE - Hypothetical Document Embedding.
    
    Generates a hypothetical document that would answer the query,
    then uses that document's embedding for retrieval.
    
    Note: In production, use an LLM to generate the hypothetical document.
    """
    
    def __init__(self, llm_provider: Callable[[str], str] = None):
        """
        Initialize HyDE generator.
        
        Args:
            llm_provider: Optional LLM function that takes prompt and returns text
        """
        self.llm_provider = llm_provider
    
    def generate(self, query: str) -> str:
        """
        Generate a hypothetical document for a query.
        
        In production, this would use an LLM. For now, we generate
        a structured approximation.
        """
        if self.llm_provider:
            prompt = (
                f"Generate a hypothetical document that would answer this query: {query}\n\n"
                "The document should contain factual information that might answer the query. "
                "Format it as if it were a real document."
            )
            return self.llm_provider(prompt)
        
        # Fallback: generate structured approximation
        # This creates a pseudo-document based on query structure
        hyde_doc = f"""
Query: {query}

This document addresses the query about {query}.

Key Points:
- Regarding {query}, the main aspects are:
- The solution involves understanding the context of {query}
- Related considerations include implementation details for {query}

Summary: The answer to {query} involves careful consideration of requirements
and proper implementation approach.
"""
        return hyde_doc.strip()


# ============================================================================
# Cross-Encoder Reranker
# ============================================================================

class CrossEncoderReranker:
    """
    Cross-encoder based reranking.
    
    Cross-encoders evaluate query-document pairs together,
    providing more accurate relevance scores than bi-encoders.
    
    Note: In production, use a proper cross-encoder model
    (e.g., cross-encoder/ms-marco, SentenceTransformers).
    """
    
    def __init__(self, model: Callable[[str, str], float] = None):
        """
        Initialize reranker.
        
        Args:
            model: Function that takes (query, document) and returns relevance score
        """
        self.model = model or self._simple_relevance
    
    def rerank(
        self,
        query: str,
        documents: List[Tuple[str, Any]],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Rerank documents based on relevance to query.
        
        Args:
            query: Search query
            documents: List of (doc_id, doc_content_or_metadata) tuples
            top_k: Number of top results to return
        
        Returns:
            List of SearchResult objects, reranked
        """
        scored_docs = []
        
        for doc_id, doc_content in documents:
            # Get document content
            if isinstance(doc_content, dict):
                content = doc_content.get("content", str(doc_content))
            else:
                content = str(doc_content)
            
            # Calculate relevance score
            score = self.model(query, content)
            
            result = SearchResult(
                id=doc_id,
                content=content,
                score=score,
                source="reranked"
            )
            scored_docs.append(result)
        
        # Sort by score descending
        scored_docs.sort()
        
        # Assign ranks
        for i, doc in enumerate(scored_docs):
            doc.rank = i + 1
        
        return scored_docs[:top_k]
    
    def _simple_relevance(self, query: str, document: str) -> float:
        """
        Simple relevance scoring as fallback.
        
        Scores based on:
        - Term overlap
        - Query term density
        - Position
        """
        query_terms = set(query.lower().split())
        doc_lower = document.lower()
        doc_terms = set(re.findall(r'\w+', doc_lower))
        
        if not query_terms:
            return 0.0
        
        # Term overlap
        overlap = query_terms & doc_terms
        overlap_score = len(overlap) / len(query_terms)
        
        # Query density in document
        density = sum(1 for term in query_terms if term in doc_lower)
        density_score = density / len(query_terms)
        
        # Boost if query terms appear early
        position_boost = 0.0
        first_pos = float('inf')
        for term in query_terms:
            pos = doc_lower.find(term)
            if pos != -1 and pos < first_pos:
                first_pos = pos
                position_boost = max(0, 1.0 - (first_pos / 1000))
        
        return (overlap_score * 0.4 + density_score * 0.4 + position_boost * 0.2)


# ============================================================================
# Hybrid Search Engine
# ============================================================================

class HybridSearchEngine:
    """
    Full hybrid search engine combining:
    - BM25 (keyword/sparse search)
    - Vector search (semantic/dense search)
    - RRF fusion
    - Query expansion
    - HyDE (optional)
    - Cross-encoder reranking
    """
    
    def __init__(
        self,
        config: Optional[SearchConfig] = None,
        llm_provider: Callable[[str], str] = None
    ):
        self.config = config or SearchConfig()
        self.bm25 = BM25(
            k1=self.config.bm25_k1,
            b=self.config.bm25_b
        )
        self.vector = SimpleVectorSearch()
        self.rrf = ReciprocalRankFusion(k=self.config.rrf_k)
        self.expander = QueryExpander()
        self.hyde = HyDEGenerator(llm_provider=llm_provider)
        self.reranker = CrossEncoderReranker()
        
        self._indexed: bool = False
        self._doc_store: Dict[str, str] = {}
        self._id_to_idx: Dict[str, int] = {}
    
    def index(self, documents: List[Tuple[str, str]]) -> None:
        """
        Index documents for searching.
        
        Args:
            documents: List of (doc_id, content) tuples
        """
        self._doc_store = {doc_id: content for doc_id, content in documents}
        
        # Build index mapping
        doc_ids = list(self._doc_store.keys())
        self._id_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        
        # Get content in original order
        corpus = [self._doc_store[doc_id] for doc_id in doc_ids]
        
        # Index with BM25
        self.bm25.index(corpus)
        
        # Index with vector search
        self.vector.index(corpus)
        
        self._indexed = True
    
    def add_document(self, doc_id: str, content: str) -> None:
        """Add a single document to the index."""
        self._doc_store[doc_id] = content
        
        # Rebuild index (inefficient but works for small corpora)
        if self._indexed:
            self.index(list(self._doc_store.items()))
        else:
            self._id_to_idx = {doc_id: 0}
            self.bm25.index([content])
            self.vector.index([content])
            self._indexed = True
    
    def search(self, query: str, config: SearchConfig = None) -> List[SearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Search query
            config: Optional per-query config override
        
        cfg = config or self.config
        
        # Expand query if enabled
        expanded_queries = [query]
        if cfg.expansion_enabled:
            expanded_queries = self.expander.expand(query)
        
        # Generate HyDE document if enabled
        hyde_doc = ""
        if cfg.hyde_enabled:
            hyde_doc = self.hyde.generate(query)
            expanded_queries.append(hyde_doc)
        
        # Collect rankings from each query variant
        all_rankings: List[List[Tuple[str, float]]] = []
        
        for q in expanded_queries:
            bm25_ranking: List[Tuple[str, float]] = []
            vector_ranking: List[Tuple[str, float]] = []
            
            # BM25 search
            if cfg.bm25_enabled:
                bm25_results = self.bm25.search(q, top_k=cfg.rerank_top_k)
                bm25_ranking = [
                    (list(self._doc_store.keys())[idx], score)
                    for idx, score in bm25_results
                ]
                all_rankings.append(bm25_ranking)
            
            # Vector search
            if cfg.vector_enabled:
                vector_results = self.vector.search(q, top_k=cfg.rerank_top_k)
                vector_ranking = [
                    (list(self._doc_store.keys())[idx], score)
                    for idx, score in vector_results
                ]
                all_rankings.append(vector_ranking)
        
        # Fuse rankings using RRF
        if cfg.rrf_enabled and len(all_rankings) > 1:
            # Weight BM25 and vector differently
            weights = []
            for i, ranking in enumerate(all_rankings):
                if i == 0:  # First BM25 result
                    weights.append(cfg.bm25_weight)
                elif i == 1:  # First vector result
                    weights.append(cfg.vector_weight_fusion)
                else:  # Expansion queries
                    weights.append(cfg.expansion_weight)
            
            fused = self.rrf.fuse(all_rankings, weights=weights)
        elif all_rankings:
            fused = all_rankings[0]
        else:
            fused = []
        
        # Build SearchResult list
        results = []
        for doc_id, rrf_score in fused[:cfg.limit]:
            content = self._doc_store.get(doc_id, "")
            result = SearchResult(
                id=doc_id,
                content=content,
                score=rrf_score,
                source="hybrid"
            )
            results.append(result)
        
        # Rerank if enabled
        if cfg.rerank_enabled and results:
            doc_tuples = [(r.id, r.content) for r in results]
            reranked = self.reranker.rerank(
                query,
                doc_tuples,
                top_k=cfg.rerank_final
            )
            return reranked
        
        # Assign final ranks
        for i, result in enumerate(results):
            result.rank = i + 1
        
        return results[:cfg.limit]
    
    def search_bm25_only(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search using only BM25."""
        bm25_results = self.bm25.search(query, top_k=limit)
        return [
            SearchResult(
                id=list(self._doc_store.keys())[idx],
                content=self._doc_store[list(self._doc_store.keys())[idx]],
                score=score,
                source="bm25"
            )
            for idx, score in bm25_results
        ]
    
    def search_vector_only(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search using only vector similarity."""
        vector_results = self.vector.search(query, top_k=limit)
        return [
            SearchResult(
                id=list(self._doc_store.keys())[idx],
                content=self._doc_store[list(self._doc_store.keys())[idx]],
                score=score,
                source="vector"
            )
            for idx, score in vector_results
        ]


# ============================================================================
# Memory-Enhanced Search
# ============================================================================

class MemorySearch:
    """
    Search optimized for agent memory systems.
    
    Features:
    - Layer-aware search (working, short_term, long_term)
    - Tag-based filtering
    - Time-based filtering
    - Importance-weighted results
    """
    
    def __init__(self, config: Optional[SearchConfig] = None):
        self.hybrid = HybridSearchEngine(config=config)
        self.config = config or SearchConfig()
        
        # Layer-specific indices
        self.layer_indices: Dict[str, HybridSearchEngine] = {
            "working": HybridSearchEngine(config=config),
            "short_term": HybridSearchEngine(config=config),
            "long_term": HybridSearchEngine(config=config),
        }
    
    def index_memory(
        self,
        memories: List[Dict[str, Any]],
        layer: str
    ) -> None:
        """
        Index memories for a specific layer.
        
        Args:
            memories: List of memory entries with 'id' and 'content'
            layer: One of "working", "short_term", "long_term"
        """
        engine = self.layer_indices.get(layer)
        if not engine:
            raise ValueError(f"Invalid layer: {layer}")
        
        documents = [
            (mem["id"], mem["content"])
            for mem in memories
            if "id" in mem and "content" in mem
        ]
        
        engine.index(documents)
    
    def search(
        self,
        query: str,
        layers: List[str] = None,
        tags: List[str] = None,
        time_range: Tuple[Optional[float], Optional[float]] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search across memory layers.
        
        Args:
            query: Search query
            layers: Layers to search (default: all)
            tags: Filter by tags
            time_range: (start_time, end_time) tuple
            limit: Max results
        
        Returns:
            List of SearchResult objects
        """
        if layers is None:
            layers = ["working", "short_term", "long_term"]
        
        all_results: List[SearchResult] = []
        
        for layer in layers:
            engine = self.layer_indices.get(layer)
            if engine and engine._indexed:
                results = engine.search(query)
                for r in results:
                    r.metadata["layer"] = layer
                all_results.extend(results)
        
        # Filter by tags (if implemented in metadata)
        if tags:
            all_results = [
                r for r in all_results
                if any(tag in r.metadata.get("tags", []) for tag in tags)
            ]
        
        # Sort by score and limit
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:limit]


# ============================================================================
# Exports
# ============================================================================

__all__ = [
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
]

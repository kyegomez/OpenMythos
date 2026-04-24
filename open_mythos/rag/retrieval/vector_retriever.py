"""
Vector Retriever Implementation

Provides vector-based retrieval using embeddings with support
for various embedding backends.
"""

from typing import List, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass
import logging
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VectorResult:
    """Represents a vector retrieval result."""
    doc_id: str
    score: float
    content: str
    metadata: Optional[Dict[str, Any]] = None
    vector: Optional[List[float]] = None


class VectorRetriever:
    """
    Vector-based retriever using embeddings.
    
    This class provides semantic search capabilities using
    dense vector embeddings. It supports various backends
    for efficient similarity search.
    """
    
    def __init__(
        self,
        embedder: Optional[Any] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize vector retriever.
        
        Args:
            embedder: Embedder instance for creating/querying embeddings
            normalize_embeddings: Whether to normalize embeddings to unit length
            batch_size: Batch size for encoding multiple texts
        """
        self.embedder = embedder
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._doc_ids: List[str] = []
        self._embeddings: Optional[np.ndarray] = None
        self._is_indexed = False
    
    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding to unit length."""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            return embedding / norm
        return embedding
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        if self.normalize_embeddings:
            a = self._normalize(a)
            b = self._normalize(b)
        return float(np.dot(a, b))
    
    def _euclidean_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate similarity based on Euclidean distance."""
        dist = np.linalg.norm(a - b)
        # Convert distance to similarity (0 to 1)
        return 1.0 / (1.0 + dist)
    
    def index(self, documents: List[Dict[str, Any]]) -> None:
        """
        Index documents for vector retrieval.
        
        Args:
            documents: List of dicts with 'id', 'content', and optionally 'metadata'
        """
        if self.embedder is None:
            raise ValueError("Embedder is required for indexing documents")
        
        self._documents = {}
        self._doc_ids = []
        contents = []
        
        for doc in documents:
            doc_id = doc.get("id", str(hash(doc["content"])))
            self._documents[doc_id] = {
                "content": doc["content"],
                "metadata": doc.get("metadata"),
            }
            self._doc_ids.append(doc_id)
            contents.append(doc["content"])
        
        # Encode all documents
        self._embeddings = self.embedder.encode(contents, batch_size=self.batch_size)
        
        if self.normalize_embeddings:
            self._embeddings = np.array([
                self._normalize(e) for e in self._embeddings
            ])
        
        self._is_indexed = True
        logger.info(f"Indexed {len(documents)} documents")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the existing index.
        
        Args:
            documents: List of dicts with 'id', 'content', and optionally 'metadata'
        """
        if self.embedder is None:
            raise ValueError("Embedder is required for adding documents")
        
        if not self._is_indexed:
            # No existing index, just index all documents
            self.index(documents)
            return
        
        for doc in documents:
            doc_id = doc.get("id", str(hash(doc["content"])))
            self._documents[doc_id] = {
                "content": doc["content"],
                "metadata": doc.get("metadata"),
            }
            self._doc_ids.append(doc_id)
            
            # Encode and normalize the new document
            embedding = self.embedder.encode([doc["content"]])
            if self.normalize_embeddings:
                embedding = self._normalize(embedding)
            
            # Append to embeddings matrix
            self._embeddings = np.vstack([self._embeddings, embedding])
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string.
        
        Args:
            query: Query string to embed
            
        Returns:
            Query embedding as numpy array
        """
        if self.embedder is None:
            raise ValueError("Embedder is required for embedding queries")
        
        embedding = self.embedder.encode([query])
        if self.normalize_embeddings:
            embedding = self._normalize(embedding)
        return embedding[0]
    
    def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        similarity_metric: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents for a query.
        
        Args:
            query: Query string
            query_embedding: Pre-computed query embedding (optional)
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score threshold
            similarity_metric: 'cosine' or 'euclidean'
            
        Returns:
            List of dicts with 'doc_id', 'score', 'content', 'metadata'
        """
        if not self._is_indexed:
            logger.warning("No documents indexed yet")
            return []
        
        # Get query embedding
        if query_embedding is not None:
            q_emb = np.array(query_embedding)
            if self.normalize_embeddings:
                q_emb = self._normalize(q_emb)
        else:
            q_emb = self.embed_query(query)
        
        # Calculate similarities
        if similarity_metric == "cosine":
            similarities = np.dot(self._embeddings, q_emb)
        elif similarity_metric == "euclidean":
            similarities = np.array([
                self._euclidean_similarity(doc_emb, q_emb)
                for doc_emb in self._embeddings
            ])
        else:
            raise ValueError(f"Unknown similarity metric: {similarity_metric}")
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score < 0 and similarity_metric == "cosine":
                # Clip negative cosine similarities
                score = 0.0
            
            if score_threshold is not None and score < score_threshold:
                continue
            
            doc_id = self._doc_ids[idx]
            doc_data = self._documents[doc_id]
            
            results.append({
                "doc_id": doc_id,
                "score": score,
                "content": doc_data["content"],
                "metadata": doc_data.get("metadata"),
            })
        
        return results
    
    def get_vector_scores(
        self,
        query_embedding: np.ndarray,
        doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Get vector similarity scores for specific documents or all documents.
        
        Args:
            query_embedding: Pre-computed query embedding
            doc_ids: Optional list of specific document IDs to score
            
        Returns:
            Dict mapping doc_id to similarity score
        """
        if not self._is_indexed:
            return {}
        
        q_emb = np.array(query_embedding)
        if self.normalize_embeddings:
            q_emb = self._normalize(q_emb)
        
        scores = {}
        doc_indices = range(len(self._doc_ids))
        
        if doc_ids is not None:
            doc_indices = [self._doc_ids.index(did) for did in doc_ids if did in self._doc_ids]
        
        for idx in doc_indices:
            doc_id = self._doc_ids[idx]
            doc_emb = self._embeddings[idx]
            scores[doc_id] = float(np.dot(doc_emb, q_emb))
        
        return scores


class EmbedderWrapper:
    """
    Wrapper to provide a consistent interface for different embedding backends.
    
    Supports:
    - OpenAI embeddings
    - HuggingFace transformers
    - Sentence transformers
    - Custom embedders
    """
    
    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        embedder: Optional[Any] = None,
        dimension: int = 1536,
    ):
        """
        Initialize embedder wrapper.
        
        Args:
            model: Model name for OpenAI or HuggingFace
            api_key: API key for OpenAI
            embedder: Pre-configured embedder instance
            dimension: Embedding dimension
        """
        self.model = model
        self.api_key = api_key
        self.embedder = embedder
        self.dimension = dimension
        self._use_openai = embedder is None and api_key is not None
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            **kwargs: Additional arguments
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if self.embedder is not None:
            embeddings = self.embedder.encode(texts, **kwargs)
            if isinstance(embeddings, list):
                return np.array(embeddings)
            return embeddings
        
        if self._use_openai:
            return self._openai_encode(texts)
        
        raise ValueError("No embedder configured")
    
    def _openai_encode(self, texts: List[str]) -> np.ndarray:
        """Encode using OpenAI API."""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            embeddings = []
            for text in texts:
                response = client.embeddings.create(
                    model=self.model,
                    input=text,
                )
                embeddings.append(response.data[0].embedding)
            
            return np.array(embeddings)
        except ImportError:
            raise ImportError("openai package not installed")
    
    def encode_query(self, text: str, **kwargs) -> np.ndarray:
        """
        Encode a single query.
        
        Args:
            text: Query text
            **kwargs: Additional arguments
            
        Returns:
            Query embedding
        """
        return self.encode([text], **kwargs)[0]

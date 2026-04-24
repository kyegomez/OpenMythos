"""
HyDE (Hypothetical Document Embeddings) Query Expansion

HyDE generates a hypothetical document from the query using an LLM,
embeds that document, and uses the embedding for retrieval.
This helps capture the intent behind the query more effectively.
"""

from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HypotheticalDocument:
    """Represents a hypothetical document generated from a query."""
    content: str
    query: str
    metadata: Optional[Dict[str, Any]] = None


class HyDEQueryExpander:
    """
    HyDE (Hypothetical Document Embeddings) Query Expander.
    
    This class generates hypothetical documents from user queries
    and uses them for improved retrieval. The hypothetical document
    captures the likely structure and content of an ideal answer.
    
    Example:
        >>> expander = HyDEQueryExpander(llm_client=openai_client, embedder=embedder)
        >>> hypothetical_doc = await expander.generate("What is Python?")
        >>> embedding = await expander.embed(hypothetical_doc)
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        embedder: Optional[Any] = None,
        document_template: str = None,
        embedding_model: str = "text-embedding-ada-002",
        max_tokens: int = 256,
        temperature: float = 0.7,
    ):
        """
        Initialize the HyDE query expander.
        
        Args:
            llm_client: LLM client for generating hypothetical documents
            embedder: Embedder for creating document embeddings
            document_template: Optional template for generating documents
            embedding_model: Name of the embedding model to use
            max_tokens: Maximum tokens for generated document
            temperature: Temperature for LLM generation
        """
        self.llm_client = llm_client
        self.embedder = embedder
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self._default_template = (
            "Write a passage that would answer the following question. "
            "The passage should be informative, detailed, and written in a "
            "factual style typical of an encyclopedia article or textbook.\n\n"
            "Question: {query}\n\nPassage:"
        )
        self._document_template = document_template or self._default_template
        
    async def generate_async(
        self, 
        query: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> HypotheticalDocument:
        """
        Generate a hypothetical document from the query asynchronously.
        
        Args:
            query: The user's query
            system_prompt: Optional system prompt for the LLM
            **kwargs: Additional arguments for the LLM
            
        Returns:
            HypotheticalDocument with the generated content
        """
        if self.llm_client is None:
            raise ValueError("LLM client is required for generating hypothetical documents")
        
        prompt = self._document_template.format(query=query)
        
        if system_prompt is None:
            system_prompt = "You are a knowledgeable assistant that writes informative passages."
        
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **kwargs
            )
            content = response if isinstance(response, str) else response.get("content", "")
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {e}")
            # Fallback: use the original query as hypothetical content
            content = query
        
        return HypotheticalDocument(
            content=content,
            query=query,
            metadata={"model": self.embedding_model}
        )
    
    def generate(
        self, 
        query: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> HypotheticalDocument:
        """
        Generate a hypothetical document from the query synchronously.
        
        Args:
            query: The user's query
            system_prompt: Optional system prompt for the LLM
            **kwargs: Additional arguments for the LLM
            
        Returns:
            HypotheticalDocument with the generated content
        """
        if self.llm_client is None:
            raise ValueError("LLM client is required for generating hypothetical documents")
        
        prompt = self._document_template.format(query=query)
        
        if system_prompt is None:
            system_prompt = "You are a knowledgeable assistant that writes informative passages."
        
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **kwargs
            )
            content = response if isinstance(response, str) else response.get("content", "")
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {e}")
            content = query
        
        return HypotheticalDocument(
            content=content,
            query=query,
            metadata={"model": self.embedding_model}
        )
    
    async def embed_async(
        self, 
        document: HypotheticalDocument
    ) -> List[float]:
        """
        Embed the hypothetical document asynchronously.
        
        Args:
            document: The hypothetical document to embed
            
        Returns:
            List of embedding values
        """
        if self.embedder is None:
            raise ValueError("Embedder is required for creating embeddings")
        
        try:
            embedding = await self.embedder.embed(document.content)
            return embedding if isinstance(embedding, list) else embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding document: {e}")
            # Fallback: embed the original query
            embedding = await self.embedder.embed(document.query)
            return embedding if isinstance(embedding, list) else embedding.tolist()
    
    def embed(self, document: HypotheticalDocument) -> List[float]:
        """
        Embed the hypothetical document synchronously.
        
        Args:
            document: The hypothetical document to embed
            
        Returns:
            List of embedding values
        """
        if self.embedder is None:
            raise ValueError("Embedder is required for creating embeddings")
        
        try:
            embedding = self.embedder.embed(document.content)
            return embedding if isinstance(embedding, list) else embedding.tolist()
        except Exception as e:
            logger.error(f"Error embedding document: {e}")
            embedding = self.embedder.embed(document.query)
            return embedding if isinstance(embedding, list) else embedding.tolist()
    
    async def expand_query_async(
        self, 
        query: str, 
        return_embedding: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Expand a query using HyDE asynchronously.
        
        Args:
            query: The original user query
            return_embedding: Whether to return the embedding
            **kwargs: Additional arguments for generation
            
        Returns:
            Dictionary containing query, hypothetical document, and optionally embedding
        """
        doc = await self.generate_async(query, **kwargs)
        
        result = {
            "original_query": query,
            "hypothetical_document": doc.content,
            "document": doc,
        }
        
        if return_embedding:
            embedding = await self.embed_async(doc)
            result["embedding"] = embedding
        
        return result
    
    def expand_query(
        self, 
        query: str, 
        return_embedding: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Expand a query using HyDE synchronously.
        
        Args:
            query: The original user query
            return_embedding: Whether to return the embedding
            **kwargs: Additional arguments for generation
            
        Returns:
            Dictionary containing query, hypothetical document, and optionally embedding
        """
        doc = self.generate(query, **kwargs)
        
        result = {
            "original_query": query,
            "hypothetical_document": doc.content,
            "document": doc,
        }
        
        if return_embedding:
            embedding = self.embed(doc)
            result["embedding"] = embedding
        
        return result


class HyDEPipeline:
    """
    Complete HyDE pipeline combining generation and retrieval.
    
    This class provides a unified interface for using HyDE to
    improve retrieval performance.
    """
    
    def __init__(
        self,
        hyde_expander: HyDEQueryExpander,
        retriever: Any,
        use_rerank: bool = False,
        reranker: Optional[Any] = None,
    ):
        """
        Initialize the HyDE pipeline.
        
        Args:
            hyde_expander: HyDEQueryExpander instance
            retriever: Retriever to use for document retrieval
            use_rerank: Whether to use reranking after retrieval
            reranker: Optional reranker for improved results
        """
        self.hyde_expander = hyde_expander
        self.retriever = retriever
        self.use_rerank = use_rerank
        self.reranker = reranker
    
    async def retrieve_async(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using HyDE-enhanced query asynchronously.
        
        Args:
            query: The user query
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of retrieved documents with scores
        """
        # Expand query using HyDE
        expanded = await self.hyde_expander.expand_query_async(query)
        
        # Retrieve using the hypothetical document's embedding
        results = await self.retriever.retrieve_async(
            query=expanded["hypothetical_document"],
            query_embedding=expanded.get("embedding"),
            top_k=top_k,
            **kwargs
        )
        
        # Add HyDE metadata to results
        for result in results:
            result["hyde"] = {
                "original_query": query,
                "hypothetical_document": expanded["hypothetical_document"],
            }
        
        # Optional reranking
        if self.use_rerank and self.reranker:
            results = await self.reranker.rerank_async(
                query=query,
                documents=results,
                top_k=top_k
            )
        
        return results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using HyDE-enhanced query synchronously.
        
        Args:
            query: The user query
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            List of retrieved documents with scores
        """
        # Expand query using HyDE
        expanded = self.hyde_expander.expand_query(query)
        
        # Retrieve using the hypothetical document's embedding
        results = self.retriever.retrieve(
            query=expanded["hypothetical_document"],
            query_embedding=expanded.get("embedding"),
            top_k=top_k,
            **kwargs
        )
        
        # Add HyDE metadata to results
        for result in results:
            result["hyde"] = {
                "original_query": query,
                "hypothetical_document": expanded["hypothetical_document"],
            }
        
        # Optional reranking
        if self.use_rerank and self.reranker:
            results = self.reranker.rerank(
                query=query,
                documents=results,
                top_k=top_k
            )
        
        return results

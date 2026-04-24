"""
Example usage of HyDE Query Expansion and BM25+Vector Hybrid Retrieval

This example demonstrates how to use the retrieval module for
combined BM25 and vector search with Reciprocal Rank Fusion.
"""

from typing import List, Dict, Any
import numpy as np

# Example documents for demonstration
SAMPLE_DOCUMENTS = [
    {
        "id": "doc1",
        "content": "Python is a high-level programming language known for its simplicity and readability. "
                  "It supports multiple programming paradigms including procedural, object-oriented, and "
                  "functional programming. Python is widely used in web development, data science, "
                  "artificial intelligence, and scientific computing.",
        "metadata": {"category": "programming", "language": "Python"}
    },
    {
        "id": "doc2",
        "content": "JavaScript is a scripting language that enables interactive web pages. "
                  "It is an essential part of web applications alongside HTML and CSS. "
                  "Modern JavaScript supports features like async/await, classes, and modules. "
                  "Node.js allows JavaScript to run on the server side.",
        "metadata": {"category": "programming", "language": "JavaScript"}
    },
    {
        "id": "doc3",
        "content": "Machine learning is a subset of artificial intelligence that enables systems "
                  "to learn and improve from experience. It uses algorithms to identify patterns "
                  "in data and make decisions with minimal human intervention. "
                  "Common applications include image recognition, natural language processing, "
                  "and recommendation systems.",
        "metadata": {"category": "AI/ML", "field": "machine learning"}
    },
    {
        "id": "doc4",
        "content": "Deep learning is a branch of machine learning based on artificial neural networks "
                  "with multiple layers. These networks can learn hierarchical representations of data. "
                  "Convolutional neural networks are used for image tasks while recurrent neural networks "
                  "handle sequential data like text and time series.",
        "metadata": {"category": "AI/ML", "field": "deep learning"}
    },
    {
        "id": "doc5",
        "content": "Retrieval augmented generation (RAG) combines retrieval systems with language models. "
                  "It retrieves relevant documents from a knowledge base and uses them as context "
                  "for generating responses. This approach helps reduce hallucinations and provides "
                  "up-to-date information from external sources.",
        "metadata": {"category": "NLP", "field": "RAG"}
    },
    {
        "id": "doc6",
        "content": "BM25 is a classic probabilistic retrieval algorithm used for text search. "
                  "It improves upon simple TF-IDF by considering document length normalization "
                  "and term frequency saturation. BM25+ adds a constant to prevent very long "
                  "documents from being penalized too heavily.",
        "metadata": {"category": "IR", "field": "BM25"}
    },
    {
        "id": "doc7",
        "content": "Vector databases store information as embeddings in high-dimensional space. "
                  "They enable semantic search by finding documents similar to a query vector. "
                  "Popular vector databases include Pinecone, Weaviate, and Milvus. "
                  "Approximate nearest neighbor algorithms make search efficient at scale.",
        "metadata": {"category": "databases", "field": "vector search"}
    },
    {
        "id": "doc8",
        "content": "HyDE (Hypothetical Document Embeddings) is a query expansion technique. "
                  "It generates a hypothetical document that would answer the query, "
                  "then embeds this document for retrieval. This captures query intent better "
                  "than direct query embedding, especially for complex or ambiguous queries.",
        "metadata": {"category": "IR", "field": "HyDE"}
    },
]


class MockEmbedder:
    """Mock embedder for demonstration purposes."""
    
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        # Simple random embeddings for demonstration
        np.random.seed(42)
    
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts to embeddings."""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding for consistency
            hash_val = hash(text.lower())
            np.random.seed(hash_val % (2**31))
            embedding = np.random.randn(self.dimension)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings)


class MockLLM:
    """Mock LLM for HyDE demonstration."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a hypothetical document from prompt."""
        # Extract the question from the prompt
        if "Question:" in prompt:
            question = prompt.split("Question:")[1].split("Passage:")[0].strip()
        else:
            question = prompt
        
        # Generate a simple hypothetical document
        return (
            f"This passage discusses {question}. "
            f"It provides detailed information about the topic including "
            f"key concepts, applications, and practical examples. "
            f"The content is educational and informative in nature."
        )


def example_basic_bm25():
    """Example of basic BM25 retrieval."""
    print("=" * 60)
    print("Example 1: Basic BM25 Retrieval")
    print("=" * 60)
    
    from open_mythos.rag.retrieval.bm25 import BM25Retriever
    
    # Create and index
    bm25 = BM25Retriever()
    bm25.index(SAMPLE_DOCUMENTS)
    
    # Query
    results = bm25.retrieve("Python programming language", top_k=3)
    
    print(f"\nQuery: 'Python programming language'")
    print(f"Top 3 results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [BM25 Score: {r['score']:.4f}] {r['doc_id']}")
        print(f"     {r['content'][:100]}...")
    
    return results


def example_basic_vector():
    """Example of basic vector retrieval."""
    print("\n" + "=" * 60)
    print("Example 2: Basic Vector Retrieval")
    print("=" * 60)
    
    from open_mythos.rag.retrieval.vector_retriever import VectorRetriever
    
    # Create with mock embedder
    embedder = MockEmbedder()
    vector = VectorRetriever(embedder=embedder)
    vector.index(SAMPLE_DOCUMENTS)
    
    # Query
    results = vector.retrieve("machine learning and AI", top_k=3)
    
    print(f"\nQuery: 'machine learning and AI'")
    print(f"Top 3 results:")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [Vector Score: {r['score']:.4f}] {r['doc_id']}")
        print(f"     {r['content'][:100]}...")
    
    return results


def example_hybrid_retrieval():
    """Example of hybrid BM25+vector retrieval with RRF."""
    print("\n" + "=" * 60)
    print("Example 3: Hybrid BM25+Vector Retrieval with RRF")
    print("=" * 60)
    
    from open_mythos.rag.retrieval.bm25 import BM25Retriever
    from open_mythos.rag.retrieval.vector_retriever import VectorRetriever
    from open_mythos.rag.retrieval.hybrid_retriever import HybridRetriever, RRFConfig
    
    # Create retrievers
    bm25 = BM25Retriever()
    bm25.index(SAMPLE_DOCUMENTS)
    
    embedder = MockEmbedder()
    vector = VectorRetriever(embedder=embedder)
    vector.index(SAMPLE_DOCUMENTS)
    
    # Create hybrid retriever
    hybrid = HybridRetriever(
        bm25=bm25,
        vector=vector,
        rrf_config=RRFConfig(k=60, weight_bm25=1.0, weight_vector=1.0)
    )
    
    # Query
    results = hybrid.retrieve("Python machine learning", top_k=5)
    
    print(f"\nQuery: 'Python machine learning'")
    print(f"Top 5 results (BM25 + Vector with RRF):")
    for r in results:
        print(f"  Rank {r['rank']}: [Score: {r['score']:.4f}] {r['doc_id']}")
        print(f"           Type: {r['retriever_type']}")
        print(f"           {r['content'][:80]}...")
    
    return results


def example_hyde_expansion():
    """Example of HyDE query expansion."""
    print("\n" + "=" * 60)
    print("Example 4: HyDE Query Expansion")
    print("=" * 60)
    
    from open_mythos.rag.retrieval.hyde import HyDEQueryExpander, HyDEPipeline
    from open_mythos.rag.retrieval.bm25 import BM25Retriever
    from open_mythos.rag.retrieval.vector_retriever import VectorRetriever
    
    # Create components
    llm = MockLLM()
    embedder = MockEmbedder()
    
    # Create HyDE expander
    hyde = HyDEQueryExpander(
        llm_client=llm,
        embedder=embedder,
    )
    
    # Generate hypothetical document
    query = "What is deep learning?"
    expanded = hyde.expand_query(query)
    
    print(f"\nOriginal Query: '{query}'")
    print(f"\nHypothetical Document:")
    print(f"  {expanded['hypothetical_document']}")
    
    # Use in pipeline
    bm25 = BM25Retriever()
    bm25.index(SAMPLE_DOCUMENTS)
    
    vector = VectorRetriever(embedder=embedder)
    vector.index(SAMPLE_DOCUMENTS)
    
    pipeline = HyDEPipeline(
        hyde_expander=hyde,
        retriever=vector,  # Use vector retriever with HyDE
    )
    
    results = pipeline.retrieve(query, top_k=3)
    
    print(f"\nTop 3 results using HyDE-enhanced query:")
    for r in results:
        print(f"  [Score: {r['score']:.4f}] {r['doc_id']}")
        print(f"  {r['content'][:80]}...")
    
    return results


def example_custom_weights():
    """Example of adjusting retrieval weights."""
    print("\n" + "=" * 60)
    print("Example 5: Custom RRF Weights")
    print("=" * 60)
    
    from open_mythos.rag.retrieval.bm25 import BM25Retriever
    from open_mythos.rag.retrieval.vector_retriever import VectorRetriever
    from open_mythos.rag.retrieval.hybrid_retriever import HybridRetriever
    
    # Create retrievers
    bm25 = BM25Retriever()
    bm25.index(SAMPLE_DOCUMENTS)
    
    embedder = MockEmbedder()
    vector = VectorRetriever(embedder=embedder)
    vector.index(SAMPLE_DOCUMENTS)
    
    # Keyword-heavy search
    hybrid_keyword = HybridRetriever(
        bm25=bm25,
        vector=vector,
    )
    hybrid_keyword.set_weights(bm25_weight=2.0, vector_weight=0.5)
    
    # Semantic-heavy search
    hybrid_semantic = HybridRetriever(
        bm25=bm25,
        vector=vector,
    )
    hybrid_semantic.set_weights(bm25_weight=0.5, vector_weight=2.0)
    
    query = "neural networks deep learning"
    
    print(f"\nQuery: '{query}'")
    print(f"\nKeyword-heavy (BM25: 2.0, Vector: 0.5):")
    results = hybrid_keyword.retrieve(query, top_k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['doc_id']} ({r['retriever_type']})")
    
    print(f"\nSemantic-heavy (BM25: 0.5, Vector: 2.0):")
    results = hybrid_semantic.retrieve(query, top_k=3)
    for r in results:
        print(f"  [{r['score']:.4f}] {r['doc_id']} ({r['retriever_type']})")


def example_config_presets():
    """Example of using configuration presets."""
    print("\n" + "=" * 60)
    print("Example 6: Configuration Presets")
    print("=" * 60)
    
    from open_mythos.rag.retrieval.config import get_config, create_from_config
    from open_mythos.rag.retrieval.bm25 import BM25Retriever
    from open_mythos.rag.retrieval.vector_retriever import VectorRetriever
    from open_mythos.rag.retrieval.hybrid_retriever import HybridRetriever
    
    # Get a preset configuration
    config = get_config("hyde_enhanced")
    
    print(f"\nPreset 'hyde_enhanced' configuration:")
    print(f"  BM25 k1={config.bm25.k1}, b={config.bm25.b}")
    print(f"  Vector model={config.vector.model}")
    print(f"  HyDE enabled={config.hyde.use_hyde}")
    print(f"  RRF k={config.rrf.k}")
    print(f"  Weights: BM25={config.rrf.weight_bm25}, Vector={config.rrf.weight_vector}")
    
    # Create retrievers from config
    retrievers = create_from_config(config)
    
    print(f"\nCreated retrievers from config:")
    for name, retriever in retrievers.items():
        print(f"  {name}: {type(retriever).__name__}")


if __name__ == "__main__":
    # Run all examples
    example_basic_bm25()
    example_basic_vector()
    example_hybrid_retrieval()
    example_hyde_expansion()
    example_custom_weights()
    example_config_presets()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)

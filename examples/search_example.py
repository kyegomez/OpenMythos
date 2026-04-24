"""
Search Capabilities Example

Demonstrates the enhanced search features:
- Hybrid Search (BM25 + Vector + RRF)
- Query Expansion
- Reranking

Run: python examples/search_example.py
"""

from open_mythos.search import (
    HybridSearchEngine,
    SearchConfig,
    MultiStrategyQueryExpander,
    AdaptiveQueryExpander,
    CrossEncoderReranker,
    DiversityReranker,
    EnsembleReranker,
)


def main():
    print("=" * 60)
    print("OpenMythos Enhanced Search Demo")
    print("=" * 60)
    
    # Sample document corpus
    documents = [
        ("doc1", "Python is a high-level programming language. It is great for data science and machine learning."),
        ("doc2", "JavaScript is a scripting language for web development. It runs in browsers and on servers with Node.js."),
        ("doc3", "Machine learning is a subset of artificial intelligence. Deep learning uses neural networks."),
        ("doc4", "Web development involves HTML, CSS, and JavaScript. Frontend and backend are key parts."),
        ("doc5", "Data structures like arrays, trees, and graphs are fundamental to computer science."),
        ("doc6", "Natural language processing NLP is a field of AI. It deals with text and speech."),
        ("doc7", "Python has become popular for AI and ML. Libraries like TensorFlow and PyTorch are widely used."),
        ("doc8", "Programming languages each have strengths. Python excels in productivity, Java in enterprise."),
    ]
    
    # Initialize search engine
    config = SearchConfig(
        bm25_enabled=True,
        vector_enabled=True,
        rrf_enabled=True,
        expansion_enabled=True,
        rerank_enabled=True,
        bm25_weight=0.4,
        vector_weight_fusion=0.6,
        rerank_top_k=10,
        rerank_final=5,
    )
    
    engine = HybridSearchEngine(config=config)
    engine.index(documents)
    
    print("\n1. BASIC HYBRID SEARCH")
    print("-" * 40)
    
    # Basic search
    query = "Python machine learning"
    results = engine.search(query)
    
    print(f"Query: '{query}'")
    print(f"Results: {len(results)} found")
    for i, r in enumerate(results[:3], 1):
        print(f"  {i}. [{r.source}] {r.content[:50]}... (score: {r.score:.3f})")
    
    print("\n2. BM25 ONLY SEARCH")
    print("-" * 40)
    
    results_bm25 = engine.search_bm25_only("programming language")
    print(f"Query: 'programming language'")
    for i, r in enumerate(results_bm25[:3], 1):
        print(f"  {i}. {r.content[:50]}... (score: {r.score:.3f})")
    
    print("\n3. VECTOR ONLY SEARCH")
    print("-" * 40)
    
    results_vector = engine.search_vector_only("AI and data science")
    print(f"Query: 'AI and data science'")
    for i, r in enumerate(results_vector[:3], 1):
        print(f"  {i}. {r.content[:50]}... (score: {r.score:.3f})")
    
    print("\n4. QUERY EXPANSION")
    print("-" * 40)
    
    expander = MultiStrategyQueryExpander()
    original, expansions = expander.expand("fix error in code")
    
    print(f"Original: '{original}'")
    print(f"Expanded queries:")
    for i, exp in enumerate(expansions, 1):
        print(f"  {i}. '{exp}'")
    
    print("\n5. ADAPTIVE QUERY EXPANSION")
    print("-" * 40)
    
    adaptive = AdaptiveQueryExpander()
    
    test_queries = [
        "how to fix error in Python",
        "what is machine learning",
        "debug JavaScript issue",
    ]
    
    for q in test_queries:
        original, expansions = adaptive.expand(q)
        query_type = adaptive._classify_query(q)
        print(f"Query: '{q}'")
        print(f"  Type: {query_type}")
        print(f"  Expansions: {expansions[:3]}")
    
    print("\n6. CROSS-ENCODER RERANKING")
    print("-" * 40)
    
    reranker = CrossEncoderReranker()
    reranked = reranker.rerank(
        "Python programming",
        [(d[0], d[1], 0.5) for d in documents],
        top_k=3
    )
    
    print(f"Query: 'Python programming'")
    for r in reranked:
        print(f"  [{r.final_score}] {r.content[:50]}...")
    
    print("\n7. DIVERSITY RERANKING (MMR)")
    print("-" * 40)
    
    diverse_reranker = DiversityReranker(lambda_param=0.5)
    diverse_results = diverse_reranker.rerank(
        "Python JavaScript programming",
        [(d[0], d[1], 0.8) for d in documents],
        top_k=5
    )
    
    print(f"Query: 'Python JavaScript programming'")
    for r in diverse_results:
        print(f"  [{r.final_score}] diversity={r.diversity_score:.2f} {r.content[:40]}...")
    
    print("\n8. CUSTOM SEARCH CONFIG")
    print("-" * 40)
    
    # Custom config for precision
    precise_config = SearchConfig(
        bm25_weight=0.7,
        vector_weight_fusion=0.3,
        expansion_enabled=True,
        expansion_count=5,
    )
    
    precise_engine = HybridSearchEngine(config=precise_config)
    precise_engine.index(documents)
    
    results_precise = precise_engine.search("deep learning neural networks")
    print(f"Precision-focused search:")
    for r in results_precise[:3]:
        print(f"  [{r.source}] {r.content[:50]}... (score: {r.score:.3f})")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

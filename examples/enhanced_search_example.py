"""
Enhanced Search Example

Demonstrates:
- Tag System
- Semantic Chunking
- Importance Scoring (WMR)

Run: python examples/enhanced_search_example.py
"""

from open_mythos.search.enhanced import (
    TagExtractor, TagRepository, TagFilter,
    SemanticChunker, ChunkingConfig,
    ImportanceScorer, WMRScorer, ImportanceScore,
)


def main():
    print("=" * 60)
    print("Enhanced Search Features Demo")
    print("=" * 60)
    
    # ========================================================================
    # 1. TAG SYSTEM
    # ========================================================================
    print("\n1. TAG SYSTEM")
    print("-" * 40)
    
    # Extract tags from text
    extractor = TagExtractor()
    
    texts = [
        "Python is great for machine learning and data science. Python has TensorFlow and PyTorch.",
        "JavaScript is used for web development with Node.js and React.",
        "Deep learning neural networks are part of AI research.",
        "Web development uses HTML, CSS, and JavaScript for frontend.",
    ]
    
    for text in texts:
        tags = extractor.extract(text, max_tags=5)
        print(f"Text: {text[:50]}...")
        print(f"  Tags: {tags}")
    
    # Tag repository
    repo = TagRepository()
    repo.add_tag("python", category="language", weight=1.5)
    repo.add_tag("javascript", category="language", weight=1.5)
    repo.add_tag("ml", category="domain", aliases=["machine learning"])
    
    repo.add_item_to_tag("doc1", "python")
    repo.add_item_to_tag("doc1", "ml")
    repo.add_item_to_tag("doc2", "javascript")
    
    print("\nTag Repository:")
    print(f"  Tags: {[t.name for t in repo.tags.values()]}")
    print(f"  Items for 'python': {repo.items_by_tag.get('python', set())}")
    
    # Tag filtering
    print("\nTag Filtering:")
    from open_mythos.search.enhanced import TaggedItem
    items = [
        TaggedItem(id="1", content="Python tutorial", tags={"python", "tutorial"}),
        TaggedItem(id="2", content="JavaScript guide", tags={"javascript", "guide"}),
        TaggedItem(id="3", content="Python ML guide", tags={"python", "ml", "guide"}),
    ]
    
    tag_filter = TagFilter(repo)
    filtered = tag_filter.filter_items(items, include_tags=["python"], match_all=False)
    print(f"  Filtered by 'python': {[i.id for i in filtered]}")
    
    # ========================================================================
    # 2. SEMANTIC CHUNKING
    # ========================================================================
    print("\n2. SEMANTIC CHUNKING")
    print("-" * 40)
    
    sample_doc = """
    Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that enables systems to learn from data.
    There are three main types of machine learning: supervised, unsupervised, and reinforcement learning.
    
    Supervised Learning
    
    In supervised learning, the algorithm learns from labeled training data.
    Examples include classification and regression problems.
    Common algorithms include decision trees, random forests, and neural networks.
    
    Unsupervised Learning
    
    Unsupervised learning works with unlabeled data to find patterns.
    Clustering and dimensionality reduction are common unsupervised techniques.
    
    Deep Learning
    
    Deep learning uses neural networks with many layers.
    It has revolutionized computer vision and natural language processing.
    """
    
    # Different chunking strategies
    strategies = ["fixed", "sentence", "paragraph", "recursive"]
    
    for strategy in strategies:
        chunker = SemanticChunker(strategy=strategy)
        config = ChunkingConfig(chunk_size=150, chunk_overlap=20)
        chunks = chunker.chunk(sample_doc, chunk_size=150, chunk_overlap=20)
        
        print(f"\n  Strategy: {strategy}")
        print(f"    Chunks: {len(chunks)}")
        for i, chunk in enumerate(chunks[:2], 1):
            preview = chunk.content[:60].replace("\n", " ")
            print(f"    Chunk {i}: {preview}...")
    
    # ========================================================================
    # 3. IMPORTANCE SCORING (WMR)
    # ========================================================================
    print("\n3. IMPORTANCE SCORING (WMR)")
    print("-" * 40)
    
    # Basic importance scoring
    scorer = ImportanceScorer()
    
    items = [
        ("mem1", 1.0, 10, 1000000, 900000),   # High access, recent
        ("mem2", 0.5, 5, 500000, 800000),    # Medium
        ("mem3", 0.8, 2, 100000, 700000),    # High importance, low access
        ("mem4", 0.3, 20, 200000, 600000),   # Very high access
    ]
    
    print("  Item Scores:")
    for item_id, importance, access_count, last_access, created in items:
        score = scorer.score(
            item_id=item_id,
            base_importance=importance,
            access_count=access_count,
            last_accessed=last_access,
            created_at=created,
            relevance_score=0.5,
        )
        print(f"    {item_id}: final={score.final_score:.3f} "
              f"(base={score.base_score:.2f}, "
              f"access={score.access_score:.2f}, "
              f"recency={score.recency_score:.2f})")
    
    # WMR with query
    print("\n  WMR with Query:")
    wmr_scorer = WMRScorer()
    
    query = "machine learning python"
    content = "Python is great for machine learning and data science."
    layer = "working"
    
    wmr_score = wmr_scorer.score_with_query(
        item_id="mem1",
        query=query,
        item_content=content,
        layer=layer,
        base_importance=1.0,
        access_count=5,
        last_accessed=1000000,
        created_at=900000,
    )
    
    print(f"    Query: '{query}'")
    print(f"    Content: '{content}'")
    print(f"    Layer: {layer}")
    print(f"    WMR Score: {wmr_score.final_score:.3f}")
    
    # Component breakdown
    print("    Components:")
    for name, value in wmr_score.components.items():
        print(f"      {name}: {value:.3f}")
    
    # ========================================================================
    # 4. INTEGRATED EXAMPLE
    # ========================================================================
    print("\n4. INTEGRATED SEARCH EXAMPLE")
    print("-" * 40)
    
    # Simulated memory items
    memory_items = [
        {"id": "m1", "content": "Python async/await syntax for asynchronous programming", "layer": "working"},
        {"id": "m2", "content": "JavaScript Promise.all for parallel execution", "layer": "short_term"},
        {"id": "m3", "content": "Machine learning model training best practices", "layer": "long_term"},
        {"id": "m4", "content": "Deep learning transformer architecture", "layer": "long_term"},
        {"id": "m5", "content": "React useEffect hook usage", "layer": "short_term"},
    ]
    
    query = "python programming async"
    
    print(f"  Query: '{query}'")
    print(f"  Searching {len(memory_items)} memory items\n")
    
    # Score each item
    scored = []
    for item in memory_items:
        score = wmr_scorer.score_with_query(
            item_id=item["id"],
            query=query,
            item_content=item["content"],
            layer=item["layer"],
            base_importance=0.8,
            access_count=3,
            last_accessed=1000000,
            created_at=800000,
        )
        scored.append((item, score.final_score))
    
    # Sort by score
    scored.sort(key=lambda x: x[1], reverse=True)
    
    print("  Results (ranked by WMR):")
    for item, score in scored:
        print(f"    [{score:.3f}] {item['id']} ({item['layer']})")
        print(f"         {item['content']}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

"""
Example: Using the Three-Layer Memory System

This example demonstrates:
- Writing to different memory layers
- Searching across layers
- Managing importance decay
"""

from open_mythos.memory import ThreeLayerMemorySystem, MemoryLayer


def main():
    print("=== Three-Layer Memory System Example ===\n")
    
    # Create memory system
    memory = ThreeLayerMemorySystem()
    
    # Write to different layers
    print("1. Writing to different layers...")
    
    memory.write_to_layer(
        MemoryLayer.WORKING,
        "User is working on a Python project",
        importance=0.8,
        tags=["project", "python"],
        source="session"
    )
    
    memory.write_to_layer(
        MemoryLayer.SHORT_TERM,
        "Last session: debugging auth module",
        importance=0.7,
        tags=["auth", "debugging"],
        source="session"
    )
    
    memory.write_to_layer(
        MemoryLayer.LONG_TERM,
        "Python best practices: Use type hints, virtual environments",
        importance=0.9,
        tags=["python", "best-practices"],
        source="learned"
    )
    
    print("   - Working memory: current session info")
    print("   - Short-term memory: recent session info")
    print("   - Long-term memory: persistent knowledge")
    
    # Search across layers
    print("\n2. Searching for 'Python' across all layers...")
    
    from open_mythos.memory import MemoryQuery
    query = MemoryQuery(
        text="Python",
        layers=[MemoryLayer.WORKING, MemoryLayer.SHORT_TERM, MemoryLayer.LONG_TERM],
        limit=10
    )
    
    results = memory.search_unified(query)
    print(f"   Found {len(results)} matching entries")
    
    # Get context for prompt
    print("\n3. Getting context for prompt...")
    
    context = memory.get_context_for_prompt(max_tokens=500)
    print(f"   Context length: {len(context)} chars")
    print(f"   Preview: {context[:100]}...")
    
    # Get statistics
    print("\n4. Memory statistics:")
    
    stats = memory.get_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")
    
    # Skills (stored in long-term memory)
    print("\n5. Working with skills...")
    
    skill_content = """# TDD Skill

## Description
Test-Driven Development approach.

## When to Use
- Writing new functions
- Bug fixing
- Refactoring

## Approach
1. Write a failing test
2. Write minimal code to pass
3. Refactor
"""
    
    memory.long_term.add_skill("tdd", skill_content, {"author": "system"})
    
    skill = memory.long_term.get_skill("tdd")
    print(f"   Skill 'tdd' exists: {skill is not None}")
    
    skills = memory.long_term.list_skills()
    print(f"   All skills: {skills}")


if __name__ == "__main__":
    main()

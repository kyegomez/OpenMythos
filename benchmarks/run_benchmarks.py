"""
Performance Benchmarks for OpenMythos

Run with: python benchmarks/run_benchmarks.py
"""

import time
import statistics


def benchmark_memory_write(memory, iterations=1000):
    """Benchmark memory write operations."""
    times = []
    
    for i in range(iterations):
        start = time.perf_counter()
        memory.write_to_layer(
            layer="working",
            content=f"Benchmark test entry {i}",
            importance=0.5
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    return {
        "iterations": iterations,
        "total_ms": sum(times),
        "avg_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p95_ms": sorted(times)[int(len(times) * 0.95)],
    }


def benchmark_memory_search(memory, entries=1000, searches=100):
    """Benchmark memory search operations."""
    # First add entries
    for i in range(entries):
        memory.write_to_layer(
            layer="working",
            content=f"Searchable content {i} with keyword {i % 10}",
            importance=0.5
        )
    
    times = []
    for i in range(searches):
        start = time.perf_counter()
        memory.search_unified(f"keyword {i % 10}", limit=10)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    
    return {
        "entries_indexed": entries,
        "searches": searches,
        "avg_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_context_compression(engine, messages=100, iterations=50):
    """Benchmark context compression."""
    # Add messages
    for i in range(messages):
        engine.add_message("user", f"Message {i} with content", importance=0.5)
        engine.add_message("assistant", f"Response {i}", importance=0.5)
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        engine.compress(strategy="summarize", max_tokens=500)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    
    return {
        "messages": messages,
        "compressions": iterations,
        "avg_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_tool_execution(executor, tool_name="calculator", iterations=1000):
    """Benchmark tool execution."""
    import random
    
    times = []
    for i in range(iterations):
        args = {
            "operation": random.choice(["add", "subtract", "multiply", "divide"]),
            "a": random.randint(1, 100),
            "b": random.randint(1, 100)
        }
        
        start = time.perf_counter()
        executor.execute(tool_name, args)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)
    
    return {
        "iterations": iterations,
        "avg_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "p95_ms": sorted(times)[int(len(times) * 0.95)],
        "p99_ms": sorted(times)[int(len(times) * 0.99)],
    }


def run_all_benchmarks():
    """Run all benchmarks."""
    print("=" * 60)
    print("OpenMythos Performance Benchmarks")
    print("=" * 60)
    print()
    
    results = {}
    
    # Memory benchmarks (skip if memory module not available)
    try:
        from open_mythos.memory import ThreeLayerMemorySystem
        
        print("1. Memory Write Benchmark")
        print("-" * 40)
        
        memory = ThreeLayerMemorySystem()
        result = benchmark_memory_write(memory, iterations=500)
        results["memory_write"] = result
        
        print(f"   Iterations: {result['iterations']}")
        print(f"   Average:    {result['avg_ms']:.4f} ms")
        print(f"   Median:     {result['median_ms']:.4f} ms")
        print(f"   P95:        {result['p95_ms']:.4f} ms")
        print()
        
        print("2. Memory Search Benchmark")
        print("-" * 40)
        
        memory2 = ThreeLayerMemorySystem()
        result = benchmark_memory_search(memory2, entries=500, searches=100)
        results["memory_search"] = result
        
        print(f"   Entries:    {result['entries_indexed']}")
        print(f"   Searches:   {result['searches']}")
        print(f"   Average:    {result['avg_ms']:.4f} ms")
        print(f"   Median:     {result['median_ms']:.4f} ms")
        print()
        
    except ImportError as e:
        print(f"Skipping memory benchmarks: {e}")
        print()
    
    # Context benchmarks
    try:
        from open_mythos.context import ContextEngine
        
        print("3. Context Compression Benchmark")
        print("-" * 40)
        
        engine = ContextEngine()
        result = benchmark_context_compression(engine, messages=50, iterations=20)
        results["context_compression"] = result
        
        print(f"   Messages:   {result['messages']}")
        print(f"   Compressions: {result['compressions']}")
        print(f"   Average:    {result['avg_ms']:.4f} ms")
        print(f"   Median:     {result['median_ms']:.4f} ms")
        print()
        
    except ImportError as e:
        print(f"Skipping context benchmarks: {e}")
        print()
    
    # Tool execution benchmarks
    try:
        from open_mythos.tools import ToolExecutor, ExecutionPolicy, register_tool
        from open_mythos.tools.builtins.math_tools import add, subtract, multiply, divide
        
        print("4. Tool Execution Benchmark")
        print("-" * 40)
        
        # Register tools if not already
        register_tool(
            name="calc_add",
            toolset="math",
            schema={
                "name": "calc_add",
                "description": "Add two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number"},
                        "b": {"type": "number"}
                    },
                    "required": ["a", "b"]
                }
            },
            handler=add,
            description="Addition",
            emoji="[+]"
        )
        
        executor = ToolExecutor(ExecutionPolicy())
        result = benchmark_tool_execution(executor, tool_name="calc_add", iterations=500)
        results["tool_execution"] = result
        
        print(f"   Iterations: {result['iterations']}")
        print(f"   Average:    {result['avg_ms']:.4f} ms")
        print(f"   Median:     {result['median_ms']:.4f} ms")
        print(f"   P95:        {result['p95_ms']:.4f} ms")
        print()
        
    except ImportError as e:
        print(f"Skipping tool benchmarks: {e}")
        print()
    
    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, result in results.items():
        print(f"\n{name.upper().replace('_', ' ')}:")
        print(f"   Throughput: {1000 / result['avg_ms']:.2f} ops/sec")
    
    print("\n" + "=" * 60)
    
    return results


if __name__ == "__main__":
    run_all_benchmarks()

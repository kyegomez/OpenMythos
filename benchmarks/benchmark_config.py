"""
Benchmark Configuration

Defines benchmark parameters and targets.
"""

# Benchmark targets (minimum acceptable performance)
BENCHMARK_TARGETS = {
    "memory_write": {
        "min_throughput": 1000,  # ops/sec
        "max_latency_p95_ms": 5.0,
        "description": "Memory write operations"
    },
    "memory_search": {
        "min_throughput": 500,  # ops/sec
        "max_latency_p95_ms": 10.0,
        "description": "Memory search operations"
    },
    "context_compression": {
        "min_throughput": 10,  # ops/sec
        "max_latency_p95_ms": 100.0,
        "description": "Context compression operations"
    },
    "tool_execution": {
        "min_throughput": 5000,  # ops/sec
        "max_latency_p95_ms": 1.0,
        "description": "Tool execution operations"
    },
}

# Memory benchmarks
MEMORY_BENCHMARK = {
    "write_iterations": 1000,
    "search_entries": 1000,
    "search_queries": 100,
    "warm_up_iterations": 100,
}

# Context benchmarks
CONTEXT_BENCHMARK = {
    "initial_messages": 100,
    "compression_iterations": 50,
    "test_strategies": ["none", "reference", "priority", "summarize", "hybrid"],
    "max_tokens_options": [500, 1000, 2000, 4000],
}

# Tool benchmarks
TOOL_BENCHMARK = {
    "execution_iterations": 1000,
    "concurrent_executions": 10,
    "timeout_ms": 5000,
}

# Regression thresholds
REGRESSION_THRESHOLDS = {
    "latency_increase_percent": 20,  # Allow 20% latency increase before warning
    "throughput_decrease_percent": 20,  # Allow 20% throughput decrease before warning
}

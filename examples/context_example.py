"""
Example: Using Context Compression

This example demonstrates:
- Adding messages to context
- Checking compression pressure
- Using different compression strategies
"""

from open_mythos.context import ContextEngine, CompressionStrategy


def main():
    print("=== Context Compression Example ===\n")
    
    # Create context engine
    engine = ContextEngine()
    
    # Add messages
    print("1. Adding messages to context...")
    
    engine.add_system_message("You are a helpful AI assistant.")
    
    for i in range(10):
        engine.add_message("user", f"Can you help with task {i}?", importance=0.6)
        engine.add_message("assistant", f"I can help with task {i}. Here's how...", importance=0.6)
    
    print(f"   Total messages: {len(engine.messages)}")
    
    # Check pressure
    print("\n2. Checking context pressure...")
    
    pressure = engine.check_pressure(max_tokens=2000)
    print(f"   Current tokens: {pressure.current_tokens}")
    print(f"   Max tokens: {pressure.max_tokens}")
    print(f"   Pressure ratio: {pressure.pressure_ratio:.2%}")
    print(f"   Should compress: {pressure.should_compress}")
    
    # Get context before compression
    print("\n3. Context before compression:")
    
    context_before = engine.get_context(max_tokens=5000)
    print(f"   Messages: {len(context_before)}")
    
    # Compress
    print("\n4. Compressing with SUMMARIZE strategy...")
    
    result = engine.compress(
        strategy=CompressionStrategy.SUMMARIZE,
        max_tokens=500
    )
    
    print(f"   Original count: {result.original_count}")
    print(f"   Compressed count: {result.compressed_count}")
    print(f"   Compression ratio: {result.compression_ratio:.2%}")
    print(f"   Strategy used: {result.strategy.value}")
    
    # Check pressure after compression
    print("\n5. Context after compression:")
    
    pressure_after = engine.check_pressure(max_tokens=2000)
    print(f"   Pressure ratio: {pressure_after.pressure_ratio:.2%}")
    print(f"   Should compress: {pressure_after.should_compress}")
    
    # Different strategies
    print("\n6. Testing different strategies...")
    
    strategies = [
        CompressionStrategy.NONE,
        CompressionStrategy.REFERENCE,
        CompressionStrategy.PRIORITY,
        CompressionStrategy.HYBRID,
    ]
    
    for strategy in strategies:
        # Reset engine
        engine = ContextEngine()
        engine.add_system_message("System prompt")
        for i in range(20):
            engine.add_message("user", f"Message {i}", importance=0.5)
            engine.add_message("assistant", f"Response {i}", importance=0.5)
        
        result = engine.compress(strategy=strategy, max_tokens=300)
        
        print(f"   {strategy.value:12} -> {result.compressed_count:3} messages ({result.compression_ratio:.1%})")
    
    # Get statistics
    print("\n7. Context engine statistics:")
    
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"   - {key}: {value}")


if __name__ == "__main__":
    main()

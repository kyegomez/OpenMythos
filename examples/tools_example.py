"""
Example: Using the Tool System

This example demonstrates:
- Registering custom tools
- Executing tools
- Using MCP clients
"""

from open_mythos.tools import (
    registry, register_tool, tool_result,
    ToolExecutor, ExecutionPolicy
)


def calculator_impl(operation: str, a: float, b: float) -> str:
    """Calculator tool implementation."""
    if operation == "add":
        return tool_result({"result": a + b, "operation": "add"})
    elif operation == "subtract":
        return tool_result({"result": a - b, "operation": "subtract"})
    elif operation == "multiply":
        return tool_result({"result": a * b, "operation": "multiply"})
    elif operation == "divide":
        if b == 0:
            return tool_error("Cannot divide by zero")
        return tool_result({"result": a / b, "operation": "divide"})
    else:
        return tool_error(f"Unknown operation: {operation}")


def main():
    print("=== Tool System Example ===\n")
    
    # Register a custom tool
    print("1. Registering custom calculator tool...")
    
    register_tool(
        name="calculator",
        toolset="math",
        schema={
            "name": "calculator",
            "description": "Perform basic math operations",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["add", "subtract", "multiply", "divide"],
                        "description": "Math operation"
                    },
                    "a": {"type": "number", "description": "First operand"},
                    "b": {"type": "number", "description": "Second operand"}
                },
                "required": ["operation", "a", "b"]
            }
        },
        handler=calculator_impl,
        description="Basic calculator",
        emoji="[CALC]"
    )
    
    print("   Registered: calculator (add, subtract, multiply, divide)")
    
    # List available tools
    print("\n2. Available tools:")
    
    all_tools = registry.get_all_tool_names()
    print(f"   Total tools: {len(all_tools)}")
    for tool in all_tools[:10]:
        print(f"   - {tool}")
    
    # Toolsets
    print("\n3. Toolsets:")
    
    toolsets = registry.get_available_toolsets()
    for ts_name, ts_info in toolsets.items():
        print(f"   - {ts_name}: {ts_info['tool_count']} tools")
    
    # Execute a tool
    print("\n4. Executing calculator tool...")
    
    executor = ToolExecutor(ExecutionPolicy(max_result_size=1000))
    
    result = executor.execute("calculator", {
        "operation": "add",
        "a": 10,
        "b": 5
    })
    
    print(f"   Status: {result.status.value}")
    print(f"   Result: {result.result}")
    print(f"   Time: {result.execution_time_ms:.2f}ms")
    
    # Execute another operation
    print("\n5. Executing divide operation...")
    
    result = executor.execute("calculator", {
        "operation": "divide",
        "a": 100,
        "b": 3
    })
    
    print(f"   Status: {result.status.value}")
    print(f"   Result: {result.result}")
    
    # Error handling
    print("\n6. Testing error handling (divide by zero)...")
    
    result = executor.execute("calculator", {
        "operation": "divide",
        "a": 1,
        "b": 0
    })
    
    print(f"   Status: {result.status.value}")
    print(f"   Error: {result.error}")
    
    # Get tool schema
    print("\n7. Tool schema:")
    
    schema = registry.get_schema("calculator")
    print(f"   Name: {schema['name']}")
    print(f"   Description: {schema['description']}")


if __name__ == "__main__":
    main()

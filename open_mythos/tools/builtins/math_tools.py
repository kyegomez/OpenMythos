"""
Math Built-in Tools

Provides basic math operations.
"""

from typing import Any, Dict

from open_mythos.tools.registry import tool_result, tool_error, register_tool


def _register_math_tools():
    """Register all math tools."""
    
    def add(a: float, b: float) -> str:
        """Add two numbers."""
        return tool_result({"result": a + b, "operation": "add", "a": a, "b": b})
    
    def subtract(a: float, b: float) -> str:
        """Subtract b from a."""
        return tool_result({"result": a - b, "operation": "subtract", "a": a, "b": b})
    
    def multiply(a: float, b: float) -> str:
        """Multiply two numbers."""
        return tool_result({"result": a * b, "operation": "multiply", "a": a, "b": b})
    
    def divide(a: float, b: float) -> str:
        """Divide a by b."""
        if b == 0:
            return tool_error("Cannot divide by zero")
        return tool_result({"result": a / b, "operation": "divide", "a": a, "b": b})
    
    def modulo(a: float, b: float) -> str:
        """Return remainder of a divided by b."""
        if b == 0:
            return tool_error("Cannot perform modulo with zero")
        return tool_result({"result": a % b, "operation": "modulo", "a": a, "b": b})
    
    def power(base: float, exponent: float) -> str:
        """Raise base to exponent power."""
        return tool_result({"result": pow(base, exponent), "operation": "power", "base": base, "exponent": exponent})
    
    def sqrt(number: float) -> str:
        """Calculate square root."""
        if number < 0:
            return tool_error("Cannot take square root of negative number")
        return tool_result({"result": number ** 0.5, "operation": "sqrt", "input": number})
    
    def abs_value(number: float) -> str:
        """Get absolute value."""
        return tool_result({"result": abs(number), "operation": "abs", "input": number})
    
    def round_number(number: float, decimals: int = 0) -> str:
        """Round a number to specified decimal places."""
        return tool_result({"result": round(number, decimals), "operation": "round", "input": number, "decimals": decimals})
    
    def min_value(a: float, b: float) -> str:
        """Return minimum of two numbers."""
        return tool_result({"result": min(a, b), "operation": "min", "a": a, "b": b})
    
    def max_value(a: float, b: float) -> str:
        """Return maximum of two numbers."""
        return tool_result({"result": max(a, b), "operation": "max", "a": a, "b": b})
    
    def clamp(value: float, min_val: float, max_val: float) -> str:
        """Clamp value between min and max."""
        result = max(min_val, min(max_val, value))
        return tool_result({"result": result, "operation": "clamp", "value": value, "min": min_val, "max": max_val})
    
    # Register tools
    register_tool(
        name="math_add",
        toolset="math",
        schema={
            "name": "math_add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        },
        handler=add,
        description="Add two numbers",
        emoji="[+]"
    )
    
    register_tool(
        name="math_subtract",
        toolset="math",
        schema={
            "name": "math_subtract",
            "description": "Subtract second number from first",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number (minuend)"},
                    "b": {"type": "number", "description": "Second number (subtrahend)"}
                },
                "required": ["a", "b"]
            }
        },
        handler=subtract,
        description="Subtract b from a",
        emoji="[-]"
    )
    
    register_tool(
        name="math_multiply",
        toolset="math",
        schema={
            "name": "math_multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        },
        handler=multiply,
        description="Multiply two numbers",
        emoji="[*]"
    )
    
    register_tool(
        name="math_divide",
        toolset="math",
        schema={
            "name": "math_divide",
            "description": "Divide first number by second",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Dividend"},
                    "b": {"type": "number", "description": "Divisor"}
                },
                "required": ["a", "b"]
            }
        },
        handler=divide,
        description="Divide a by b",
        emoji="[/]"
    )
    
    register_tool(
        name="math_modulo",
        toolset="math",
        schema={
            "name": "math_modulo",
            "description": "Get remainder of division",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "Dividend"},
                    "b": {"type": "number", "description": "Divisor"}
                },
                "required": ["a", "b"]
            }
        },
        handler=modulo,
        description="Return remainder of a / b",
        emoji="[%]"
    )
    
    register_tool(
        name="math_power",
        toolset="math",
        schema={
            "name": "math_power",
            "description": "Raise to power",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"type": "number", "description": "Base number"},
                    "exponent": {"type": "number", "description": "Exponent"}
                },
                "required": ["base", "exponent"]
            }
        },
        handler=power,
        description="base raised to exponent power",
        emoji="[^]"
    )
    
    register_tool(
        name="math_sqrt",
        toolset="math",
        schema={
            "name": "math_sqrt",
            "description": "Calculate square root",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "number", "description": "Number to square root"}
                },
                "required": ["number"]
            }
        },
        handler=sqrt,
        description="Square root of number",
        emoji="[√]"
    )
    
    register_tool(
        name="math_abs",
        toolset="math",
        schema={
            "name": "math_abs",
            "description": "Get absolute value",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "number", "description": "Number"}
                },
                "required": ["number"]
            }
        },
        handler=abs_value,
        description="Absolute value",
        emoji="[|]"
    )
    
    register_tool(
        name="math_round",
        toolset="math",
        schema={
            "name": "math_round",
            "description": "Round a number",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "number", "description": "Number to round"},
                    "decimals": {"type": "integer", "description": "Decimal places", "default": 0}
                },
                "required": ["number"]
            }
        },
        handler=round_number,
        description="Round to decimal places",
        emoji="[~]"
    )
    
    register_tool(
        name="math_min",
        toolset="math",
        schema={
            "name": "math_min",
            "description": "Get minimum of two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        },
        handler=min_value,
        description="Minimum of a and b",
        emoji="[↓]"
    )
    
    register_tool(
        name="math_max",
        toolset="math",
        schema={
            "name": "math_max",
            "description": "Get maximum of two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"}
                },
                "required": ["a", "b"]
            }
        },
        handler=max_value,
        description="Maximum of a and b",
        emoji="[↑]"
    )
    
    register_tool(
        name="math_clamp",
        toolset="math",
        schema={
            "name": "math_clamp",
            "description": "Clamp value between min and max",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Value to clamp"},
                    "min_val": {"type": "number", "description": "Minimum value"},
                    "max_val": {"type": "number", "description": "Maximum value"}
                },
                "required": ["value", "min_val", "max_val"]
            }
        },
        handler=clamp,
        description="Clamp value between min and max",
        emoji="[⇄]"
    )


# Auto-register when imported
_register_math_tools()


__all__ = ["_register_math_tools"]

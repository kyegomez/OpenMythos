"""
JSON Built-in Tools

Provides JSON manipulation operations.
"""

import json
from typing import Any, Dict, List, Optional, Union

from open_mythos.tools.registry import tool_result, tool_error, register_tool


def _register_json_tools():
    """Register all JSON tools."""
    
    def parse_json(json_string: str) -> str:
        """Parse a JSON string."""
        try:
            data = json.loads(json_string)
            return tool_result({"success": True, "data": data})
        except json.JSONDecodeError as e:
            return tool_error(f"JSON parse error: {e}")
    
    def stringify_json(data: Any, indent: Optional[int] = None) -> str:
        """Convert data to JSON string."""
        try:
            if indent:
                result = json.dumps(data, indent=indent, ensure_ascii=False)
            else:
                result = json.dumps(data, ensure_ascii=False)
            return tool_result({"success": True, "json": result})
        except Exception as e:
            return tool_error(f"JSON stringify error: {e}")
    
    def get_value(data: Any, path: str) -> str:
        """Get value from nested data using dot notation path."""
        try:
            keys = path.split(".")
            current = data
            
            for key in keys:
                if isinstance(current, dict):
                    current = current[key]
                elif isinstance(current, list):
                    current = current[int(key)]
                else:
                    return tool_error(f"Cannot navigate through {type(current)}")
            
            return tool_result({"success": True, "value": current, "path": path})
        except (KeyError, IndexError, ValueError) as e:
            return tool_error(f"Path not found: {path} - {e}")
    
    def set_value(data: Any, path: str, value: Any) -> str:
        """Set value in nested data using dot notation path."""
        try:
            keys = path.split(".")
            current = data
            
            # Navigate to parent
            for key in keys[:-1]:
                if isinstance(current, dict):
                    current = current[key]
                elif isinstance(current, list):
                    current = current[int(key)]
                else:
                    return tool_error(f"Cannot navigate through {type(current)}")
            
            # Set the value
            final_key = keys[-1]
            if isinstance(current, dict):
                current[final_key] = value
            elif isinstance(current, list):
                current[int(final_key)] = value
            else:
                return tool_error(f"Cannot set value on {type(current)}")
            
            return tool_result({"success": True, "data": data, "path": path})
        except Exception as e:
            return tool_error(f"Error setting value: {e}")
    
    def merge_json(base: Any, updates: Any) -> str:
        """Merge two JSON objects."""
        try:
            if not isinstance(base, dict) or not isinstance(updates, dict):
                return tool_error("Both arguments must be objects")
            
            result = base.copy()
            result.update(updates)
            
            return tool_result({"success": True, "data": result})
        except Exception as e:
            return tool_error(f"Merge error: {e}")
    
    def validate_json(json_string: str) -> str:
        """Validate a JSON string."""
        try:
            json.loads(json_string)
            return tool_result({"valid": True})
        except json.JSONDecodeError as e:
            return tool_result({"valid": False, "error": str(e)})
    
    def pretty_print(json_string: str, indent: int = 2) -> str:
        """Pretty print JSON with indentation."""
        try:
            data = json.loads(json_string)
            result = json.dumps(data, indent=indent, ensure_ascii=False)
            return tool_result({"success": True, "pretty": result})
        except json.JSONDecodeError as e:
            return tool_error(f"JSON parse error: {e}")
    
    def minify_json(json_string: str) -> str:
        """Remove whitespace from JSON."""
        try:
            data = json.loads(json_string)
            result = json.dumps(data, separators=(',', ':'))
            return tool_result({"success": True, "minified": result})
        except json.JSONDecodeError as e:
            return tool_error(f"JSON parse error: {e}")
    
    # Register tools
    register_tool(
        name="json_parse",
        toolset="json",
        schema={
            "name": "json_parse",
            "description": "Parse a JSON string",
            "parameters": {
                "type": "object",
                "properties": {
                    "json_string": {"type": "string", "description": "JSON string to parse"}
                },
                "required": ["json_string"]
            }
        },
        handler=parse_json,
        description="Parse JSON string",
        emoji="[📥]"
    )
    
    register_tool(
        name="json_stringify",
        toolset="json",
        schema={
            "name": "json_stringify",
            "description": "Convert data to JSON string",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"description": "Data to convert"},
                    "indent": {"type": "integer", "description": "Indent spaces", "default": None}
                },
                "required": ["data"]
            }
        },
        handler=stringify_json,
        description="Convert to JSON",
        emoji="[📤]"
    )
    
    register_tool(
        name="json_get",
        toolset="json",
        schema={
            "name": "json_get",
            "description": "Get value from nested JSON using dot notation path",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"description": "JSON data"},
                    "path": {"type": "string", "description": "Dot notation path (e.g., 'a.b.c')"}
                },
                "required": ["data", "path"]
            }
        },
        handler=get_value,
        description="Get nested value",
        emoji="[🔍]"
    )
    
    register_tool(
        name="json_set",
        toolset="json",
        schema={
            "name": "json_set",
            "description": "Set value in nested JSON using dot notation path",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"description": "JSON data"},
                    "path": {"type": "string", "description": "Dot notation path"},
                    "value": {"description": "Value to set"}
                },
                "required": ["data", "path", "value"]
            }
        },
        handler=set_value,
        description="Set nested value",
        emoji="[📝]"
    )
    
    register_tool(
        name="json_merge",
        toolset="json",
        schema={
            "name": "json_merge",
            "description": "Merge two JSON objects",
            "parameters": {
                "type": "object",
                "properties": {
                    "base": {"description": "Base object"},
                    "updates": {"description": "Object with updates"}
                },
                "required": ["base", "updates"]
            }
        },
        handler=merge_json,
        description="Merge objects",
        emoji="[🔀]"
    )
    
    register_tool(
        name="json_validate",
        toolset="json",
        schema={
            "name": "json_validate",
            "description": "Validate JSON string",
            "parameters": {
                "type": "object",
                "properties": {
                    "json_string": {"type": "string", "description": "JSON string to validate"}
                },
                "required": ["json_string"]
            }
        },
        handler=validate_json,
        description="Validate JSON",
        emoji="[✓]"
    )
    
    register_tool(
        name="json_pretty",
        toolset="json",
        schema={
            "name": "json_pretty",
            "description": "Pretty print JSON",
            "parameters": {
                "type": "object",
                "properties": {
                    "json_string": {"type": "string", "description": "JSON string"},
                    "indent": {"type": "integer", "description": "Indent spaces", "default": 2}
                },
                "required": ["json_string"]
            }
        },
        handler=pretty_print,
        description="Pretty print",
        emoji="[✨]"
    )
    
    register_tool(
        name="json_minify",
        toolset="json",
        schema={
            "name": "json_minify",
            "description": "Minify JSON by removing whitespace",
            "parameters": {
                "type": "object",
                "properties": {
                    "json_string": {"type": "string", "description": "JSON string to minify"}
                },
                "required": ["json_string"]
            }
        },
        handler=minify_json,
        description="Minify JSON",
        emoji="[➜]"
    )


# Auto-register when imported
_register_json_tools()


__all__ = ["_register_json_tools"]

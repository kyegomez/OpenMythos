"""
Built-in Tools - File, Web, Math, DateTime operations

Auto-registers all built-in tools on import.
"""

from .file_tools import register_file_tools
from .web_tools import register_web_tools
from .math_tools import _register_math_tools
from .datetime_tools import _register_datetime_tools


def register_all_builtin_tools():
    """Register all built-in tools."""
    register_file_tools()
    register_web_tools()
    _register_math_tools()
    _register_datetime_tools()


# Auto-register all built-in tools when this module is imported
register_all_builtin_tools()


__all__ = [
    "register_all_builtin_tools",
    "register_file_tools",
    "register_web_tools",
]

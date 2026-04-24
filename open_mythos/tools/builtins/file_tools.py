"""
File Tools - Read, write, search files

Hermes-style file tools with safe defaults.
"""

import os
import glob as glob_module
import hashlib
from typing import Any, Dict, List, Optional

from ..registry import register_tool, tool_result, tool_error


def _check_path(path: str, base_dir: Optional[str] = None) -> bool:
    """Check if path is safe (no directory traversal)."""
    if base_dir:
        resolved = os.path.realpath(path)
        base = os.path.realpath(base_dir)
        return resolved.startswith(base)
    return ".." not in path and not os.path.isabs(path)


def read_file_impl(path: str, max_lines: int = 1000, offset: int = 0) -> str:
    """Read file contents."""
    if not _check_path(path):
        return tool_error("Path traversal detected")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        lines = lines[offset:offset+max_lines]
        
        content = "".join(lines)
        
        return tool_result({
            "path": path,
            "content": content,
            "total_lines": total_lines,
            "read_lines": len(lines),
            "offset": offset
        })
    except FileNotFoundError:
        return tool_error(f"File not found: {path}")
    except Exception as e:
        return tool_error(f"Error reading file: {str(e)}")


def write_file_impl(path: str, content: str, append: bool = False) -> str:
    """Write content to file."""
    if not _check_path(path):
        return tool_error("Path traversal detected")
    
    try:
        mode = "a" if append else "w"
        with open(path, mode, encoding="utf-8") as f:
            f.write(content)
        
        return tool_result({
            "path": path,
            "bytes_written": len(content.encode("utf-8")),
            "mode": mode
        })
    except Exception as e:
        return tool_error(f"Error writing file: {str(e)}")


def list_directory_impl(path: str, pattern: str = "*", recursive: bool = False) -> str:
    """List directory contents."""
    if not _check_path(path):
        return tool_error("Path traversal detected")
    
    try:
        if recursive:
            files = glob_module.glob(os.path.join(path, "**", pattern), recursive=True)
        else:
            files = glob_module.glob(os.path.join(path, pattern))
        
        entries = []
        for f in files:
            try:
                stat = os.stat(f)
                entries.append({
                    "name": os.path.basename(f),
                    "path": f,
                    "is_dir": os.path.isdir(f),
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
            except Exception:
                pass
        
        return tool_result({
            "path": path,
            "pattern": pattern,
            "count": len(entries),
            "entries": entries[:100]  # Limit to 100
        })
    except Exception as e:
        return tool_error(f"Error listing directory: {str(e)}")


def search_files_impl(directory: str, pattern: str, file_pattern: str = "*") -> str:
    """Search for pattern in files."""
    if not _check_path(directory):
        return tool_error("Path traversal detected")
    
    try:
        matches = []
        glob_pattern = os.path.join(directory, "**", file_pattern)
        
        for filepath in glob_module.glob(glob_pattern, recursive=True):
            if not os.path.isfile(filepath):
                continue
            
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if pattern.lower() in line.lower():
                            matches.append({
                                "file": filepath,
                                "line": i,
                                "content": line.strip()[:200]
                            })
                            
                            if len(matches) >= 100:  # Limit matches
                                break
            except Exception:
                continue
            
            if len(matches) >= 100:
                break
        
        return tool_result({
            "directory": directory,
            "pattern": pattern,
            "matches": len(matches),
            "results": matches
        })
    except Exception as e:
        return tool_error(f"Error searching files: {str(e)}")


def get_file_info_impl(path: str) -> str:
    """Get file information."""
    if not _check_path(path):
        return tool_error("Path traversal detected")
    
    try:
        stat = os.stat(path)
        
        return tool_result({
            "path": path,
            "exists": True,
            "is_file": os.path.isfile(path),
            "is_dir": os.path.isdir(path),
            "size": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "md5": hashlib.md5(open(path, "rb").read()).hexdigest() if os.path.isfile(path) else None
        })
    except FileNotFoundError:
        return tool_error(f"File not found: {path}")
    except Exception as e:
        return tool_error(f"Error getting file info: {str(e)}")


def register_file_tools() -> None:
    """Register all file tools."""
    
    register_tool(
        name="read_file",
        toolset="file",
        schema={
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "max_lines": {"type": "integer", "default": 1000, "description": "Maximum lines to read"},
                    "offset": {"type": "integer", "default": 0, "description": "Line offset to start reading"}
                },
                "required": ["path"]
            }
        },
        handler=read_file_impl,
        description="Read contents of a file with optional line limits",
        emoji="📄"
    )
    
    register_tool(
        name="write_file",
        toolset="file",
        schema={
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"},
                    "append": {"type": "boolean", "default": False, "description": "Append instead of overwrite"}
                },
                "required": ["path", "content"]
            }
        },
        handler=write_file_impl,
        description="Write content to a file",
        emoji="✏️",
        danger_level=1  # Elevated - can modify files
    )
    
    register_tool(
        name="list_directory",
        toolset="file",
        schema={
            "name": "list_directory",
            "description": "List directory contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path"},
                    "pattern": {"type": "string", "default": "*", "description": "File pattern"},
                    "recursive": {"type": "boolean", "default": False, "description": "Search recursively"}
                },
                "required": ["path"]
            }
        },
        handler=list_directory_impl,
        description="List files in a directory"
    )
    
    register_tool(
        name="search_files",
        toolset="file",
        schema={
            "name": "search_files",
            "description": "Search for pattern in files",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory to search"},
                    "pattern": {"type": "string", "description": "Text pattern to search"},
                    "file_pattern": {"type": "string", "default": "*", "description": "File pattern to match"}
                },
                "required": ["directory", "pattern"]
            }
        },
        handler=search_files_impl,
        description="Search for text pattern in files"
    )
    
    register_tool(
        name="get_file_info",
        toolset="file",
        schema={
            "name": "get_file_info",
            "description": "Get file information",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"}
                },
                "required": ["path"]
            }
        },
        handler=get_file_info_impl,
        description="Get file metadata and hash"
    )

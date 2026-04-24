"""
Web Tools - Search and extract web content

Hermes-style web tools.
"""

import urllib.request
import urllib.parse
import json
from typing import Any, Dict

from ..registry import register_tool, tool_result, tool_error


def web_search_impl(query: str, num_results: int = 5) -> str:
    """
    Search the web.
    
    Note: This is a placeholder that uses DuckDuckGo HTML.
    In production, use a proper search API.
    """
    try:
        # Encode query
        encoded_query = urllib.parse.quote(query)
        url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
        
        # Make request
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
        with urllib.request.urlopen(req, timeout=10) as response:
            html = response.read().decode("utf-8")
        
        # Simple extraction (in production, use proper parsing)
        results = []
        lines = html.split("\n")
        for i, line in enumerate(lines):
            if 'class="result__snippet"' in line:
                # Extract snippet
                start = line.find('">') + 2
                end = line.find("</a>", start)
                if start > 1 and end > start:
                    snippet = line[start:end]
                    # Clean HTML tags
                    import re
                    snippet = re.sub(r'<[^>]+>', '', snippet)
                    results.append({"snippet": snippet[:200]})
                    if len(results) >= num_results:
                        break
        
        return tool_result({
            "query": query,
            "num_results": len(results),
            "results": results
        })
        
    except Exception as e:
        return tool_error(f"Search failed: {str(e)}")


def web_extract_impl(url: str, selector: str = None) -> str:
    """
    Extract content from a URL.
    
    Note: This is a simplified implementation.
    For production, use a proper HTML parser like BeautifulSoup.
    """
    try:
        # Make request
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        
        with urllib.request.urlopen(req, timeout=15) as response:
            content_type = response.headers.get("Content-Type", "")
            
            if "text/html" not in content_type and "text/plain" not in content_type:
                return tool_error(f"URL does not return HTML: {content_type}")
            
            html = response.read().decode("utf-8", errors="ignore")
        
        # Simple text extraction (remove scripts, styles)
        import re
        
        # Remove script tags
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        # Remove style tags
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL)
        # Remove HTML tags but keep text
        text = re.sub(r'<[^>]+>', ' ', html)
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Limit content
        if len(text) > 10000:
            text = text[:10000] + "... [truncated]"
        
        return tool_result({
            "url": url,
            "content": text,
            "content_length": len(text)
        })
        
    except Exception as e:
        return tool_error(f"Extraction failed: {str(e)}")


def register_web_tools() -> None:
    """Register all web tools."""
    
    register_tool(
        name="web_search",
        toolset="web",
        schema={
            "name": "web_search",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "default": 5, "description": "Number of results"}
                },
                "required": ["query"]
            }
        },
        handler=web_search_impl,
        description="Search the web for information",
        emoji="🔍"
    )
    
    register_tool(
        name="web_extract",
        toolset="web",
        schema={
            "name": "web_extract",
            "description": "Extract content from a URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to extract from"},
                    "selector": {"type": "string", "description": "CSS selector (optional)"}
                },
                "required": ["url"]
            }
        },
        handler=web_extract_impl,
        description="Extract text content from a web page"
    )

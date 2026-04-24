"""
Provider - LLM Provider Abstraction

Supports multiple LLM backends:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- OpenRouter
- Local (Ollama, llama.cpp)
- Custom endpoints
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, AsyncIterator
import json


@dataclass
class ProviderConfig:
    """Base configuration for a provider."""
    model: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: float = 120.0


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    Usage:
        class MyProvider(LLMProvider):
            async def chat(self, messages):
                # Implementation
                pass
        
        provider = MyProvider(config)
        response = await provider.chat([{"role": "user", "content": "Hello"}])
    """
    
    def __init__(self, config: ProviderConfig):
        self.config = config
    
    @abstractmethod
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """
        Send a chat request.
        
        Args:
            messages: List of message dicts with role/content
            
        Returns:
            Response dict with content, tool_calls, usage, etc.
        """
        pass
    
    @abstractmethod
    async def stream(self, messages: List[Dict[str, Any]], **kwargs) -> AsyncIterator[str]:
        """
        Stream chat response.
        
        Args:
            messages: List of message dicts
            
        Yields:
            Text chunks as they arrive
        """
        pass
    
    def supports_tools(self) -> bool:
        """Check if provider supports tool calling."""
        return False
    
    def supports_vision(self) -> bool:
        """Check if provider supports vision."""
        return False


class OpenAIProvider(LLMProvider):
    """OpenAI provider (GPT-4, GPT-3.5)."""
    
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        import aiohttp
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        
        # Tools
        if "tools" in kwargs:
            body["tools"] = kwargs["tools"]
        
        url = f"{self.config.base_url}/chat/completions"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, timeout=self.config.timeout) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"OpenAI API error: {resp.status} - {error}")
                
                data = await resp.json()
                
                # Parse response
                choice = data["choices"][0]
                message = choice["message"]
                
                response = {
                    "content": message.get("content", ""),
                    "role": message.get("role"),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": data.get("usage", {})
                }
                
                # Tool calls
                if "tool_calls" in message:
                    response["tool_calls"] = message["tool_calls"]
                
                return response
    
    async def stream(self, messages: List[Dict[str, Any]], **kwargs) -> AsyncIterator[str]:
        import aiohttp
        import json
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True
        }
        
        url = f"{self.config.base_url}/chat/completions"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, timeout=self.config.timeout) as resp:
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line or line == "data: [DONE]":
                        continue
                    
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
    
    def supports_tools(self) -> bool:
        return True


class AnthropicProvider(LLMProvider):
    """Anthropic provider (Claude)."""
    
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        import aiohttp
        
        # Convert messages format
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue  # Handle separately
            anthropic_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        
        url = f"{self.config.base_url}/messages"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, timeout=self.config.timeout) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"Anthropic API error: {resp.status} - {error}")
                
                data = await resp.json()
                
                return {
                    "content": data.get("content", [{"text": ""}])[0].get("text", ""),
                    "role": "assistant",
                    "usage": data.get("usage", {})
                }
    
    async def stream(self, messages: List[Dict[str, Any]], **kwargs) -> AsyncIterator[str]:
        # Claude streaming implementation
        import aiohttp
        
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            anthropic_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        headers = {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True
        }
        
        url = f"{self.config.base_url}/messages"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, timeout=self.config.timeout) as resp:
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("type") == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                yield delta.get("text", "")
    
    def supports_tools(self) -> bool:
        return True


class OpenRouterProvider(LLMProvider):
    """OpenRouter provider."""
    
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        import aiohttp
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        
        url = f"{self.config.base_url}/chat/completions"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, timeout=self.config.timeout) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"OpenRouter API error: {resp.status} - {error}")
                
                data = await resp.json()
                choice = data["choices"][0]
                message = choice["message"]
                
                response = {
                    "content": message.get("content", ""),
                    "role": message.get("role"),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": data.get("usage", {})
                }
                
                if "tool_calls" in message:
                    response["tool_calls"] = message["tool_calls"]
                
                return response
    
    async def stream(self, messages: List[Dict[str, Any]], **kwargs) -> AsyncIterator[str]:
        import aiohttp
        import json
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": True
        }
        
        url = f"{self.config.base_url}/chat/completions"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, headers=headers, timeout=self.config.timeout) as resp:
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line or line == "data: [DONE]":
                        continue
                    
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
    
    def supports_tools(self) -> bool:
        return True


class OllamaProvider(LLMProvider):
    """Ollama local provider."""
    
    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        self.config.base_url = self.config.base_url or "http://localhost:11434"
    
    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        import aiohttp
        
        # Convert to Ollama format
        ollama_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        body = {
            "model": self.config.model,
            "messages": ollama_messages,
            "stream": False
        }
        
        url = f"{self.config.base_url}/api/chat"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, timeout=self.config.timeout) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    raise Exception(f"Ollama API error: {resp.status} - {error}")
                
                data = await resp.json()
                
                return {
                    "content": data.get("message", {}).get("content", ""),
                    "role": "assistant",
                    "usage": {}
                }
    
    async def stream(self, messages: List[Dict[str, Any]], **kwargs) -> AsyncIterator[str]:
        import aiohttp
        
        ollama_messages = []
        for msg in messages:
            if msg["role"] == "system":
                continue
            ollama_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        body = {
            "model": self.config.model,
            "messages": ollama_messages,
            "stream": True
        }
        
        url = f"{self.config.base_url}/api/chat"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=body, timeout=self.config.timeout) as resp:
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        if "message" in data:
                            content = data["message"].get("content", "")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue


def create_provider(provider_type: str, config: ProviderConfig) -> LLMProvider:
    """
    Factory function to create a provider.
    
    Usage:
        config = ProviderConfig(model="gpt-4", api_key="...")
        provider = create_provider("openai", config)
    """
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "openrouter": OpenRouterProvider,
        "ollama": OllamaProvider,
        "local": OllamaProvider,  # Alias for Ollama
    }
    
    provider_class = providers.get(provider_type.lower())
    if not provider_class:
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    return provider_class(config)

"""
Agent Loop - Core Hermes-Style Agent Orchestration

Provides the main agent loop that:
- Assembles prompts from memory, skills, context
- Dispatches to LLM provider
- Handles tool calls
- Manages retries and budgets
- Routes between providers on failure
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, AsyncIterator
import logging

logger = logging.getLogger(__name__)


class LoopState(Enum):
    """Agent loop states."""
    IDLE = "idle"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    WAITING_INPUT = "waiting_input"
    DONE = "done"
    ERROR = "error"


class TransitionReason(Enum):
    """Reasons for loop state transitions."""
    USER_INPUT = "user_input"
    API_RESPONSE = "api_response"
    TOOL_RESULT = "tool_result"
    COMPRESSION = "compression"
    BUDGET_EXCEEDED = "budget_exceeded"
    MAX_ITERATIONS = "max_iterations"
    ERROR = "error"
    YIELD = "yield"


@dataclass
class TokenBudget:
    """Token budget for a conversation turn."""
    max_tokens: int = 120000
    remaining: int = 120000
    prompt_tokens: int = 0
    completion_tokens: int = 0
    
    def consume(self, prompt: int, completion: int) -> None:
        """Consume tokens from budget."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.remaining = self.max_tokens - self.prompt_tokens - self.completion_tokens
    
    def reset(self) -> None:
        """Reset budget for new turn."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.remaining = self.max_tokens


@dataclass
class LoopEvent:
    """Event emitted by the agent loop."""
    type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentConfig:
    """Configuration for the agent loop."""
    max_iterations: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    compress_on_pressure: float = 0.8
    provider_timeout: float = 120.0
    stream: bool = True
    auto_compress: bool = True


class ProviderCallback:
    """Callbacks for provider events."""
    
    def __init__(self):
        self._callbacks: Dict[str, List[Callable]] = {
            "tool_progress": [],
            "tool_result": [],
            "step": [],
            "clarify": [],
            "compression": [],
        }
    
    def register(self, event: str, callback: Callable) -> None:
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def emit(self, event: str, data: Dict[str, Any]) -> None:
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")


class AIAgentLoop:
    """
    Core agent loop implementing Hermes-style orchestration.
    
    Usage:
        config = AgentConfig(max_iterations=50)
        agent = AIAgentLoop(provider=my_provider, config=config)
        
        # Async generator usage
        async for event in agent.run("Fix the bug in main.py"):
            if event.type == "text":
                print(event.data["content"], end="")
            elif event.type == "tool_call":
                result = await execute_tool(event.data["name"], event.data["args"])
                agent.submit_tool_result(result)
        
        # Get final response
        response = agent.get_response()
    """
    
    def __init__(
        self,
        provider: Any,
        config: Optional[AgentConfig] = None,
        callbacks: Optional[ProviderCallback] = None,
        enhanced_hermes: Optional[Any] = None
    ):
        self.provider = provider
        self.config = config or AgentConfig()
        self.callbacks = callbacks or ProviderCallback()
        self.hermes = enhanced_hermes
        
        # State
        self._state = LoopState.IDLE
        self._transition_reason: Optional[TransitionReason] = None
        self._iteration = 0
        self._retry_count = 0
        self._budget = TokenBudget()
        
        # Messages
        self._messages: List[Dict[str, Any]] = []
        self._pending_tool_calls: Dict[str, Dict[str, Any]] = {}
        
        # Response
        self._final_response: Optional[str] = None
        self._error: Optional[str] = None
    
    @property
    def state(self) -> LoopState:
        return self._state
    
    @property
    def iteration(self) -> int:
        return self._iteration
    
    def add_message(self, role: str, content: str, **kwargs) -> None:
        """Add a message to conversation history."""
        msg = {"role": role, "content": content, **kwargs}
        self._messages.append(msg)
    
    def add_system_message(self, content: str) -> None:
        """Add a system message."""
        self.add_message("system", content)
    
    async def run(self, prompt: str) -> AsyncIterator[LoopEvent]:
        """
        Run the agent loop.
        
        This is an async generator that yields events as they happen.
        """
        # Initialize
        self._state = LoopState.THINKING
        self._transition_reason = TransitionReason.USER_INPUT
        self._iteration = 0
        self._budget.reset()
        
        # Add user message
        self.add_message("user", prompt)
        
        # Emit start event
        yield LoopEvent(type="start", data={"prompt": prompt})
        
        try:
            # Main loop
            while self._iteration < self.config.max_iterations and self._budget.remaining > 0:
                self._iteration += 1
                self._state = LoopState.THINKING
                
                # Emit step event
                self.callbacks.emit("step", {
                    "iteration": self._iteration,
                    "state": self._state.value,
                    "messages": len(self._messages)
                })
                
                # Check for compression
                if self.config.auto_compress:
                    pressure = self._calculate_pressure()
                    if pressure > self.config.compress_on_pressure:
                        yield LoopEvent(type="compression", data={"pressure": pressure})
                        await self._compress_context()
                
                # Build API message
                api_messages = self._build_api_messages()
                
                # Make API call with retry
                response = await self._call_with_retry(api_messages)
                
                if response is None:
                    continue
                
                # Process response
                async for event in self._process_response(response):
                    yield event
                    
                    if self._state == LoopState.WAITING_INPUT:
                        return
                
                # Check if done
                if self._state == LoopState.DONE:
                    break
            
            # Budget exceeded
            if self._budget.remaining <= 0:
                yield LoopEvent(type="budget_exceeded", data={
                    "remaining": self._budget.remaining,
                    "iteration": self._iteration
                })
            
            # Max iterations
            if self._iteration >= self.config.max_iterations:
                yield LoopEvent(type="max_iterations", data={
                    "iterations": self._iteration
                })
                
        except Exception as e:
            self._state = LoopState.ERROR
            self._error = str(e)
            yield LoopEvent(type="error", data={"error": str(e)})
        
        # Emit final state
        yield LoopEvent(type="end", data={
            "iterations": self._iteration,
            "final_state": self._state.value,
            "response": self._final_response
        })
    
    async def _call_with_retry(self, messages: List[Dict]) -> Optional[Dict]:
        """Make API call with retry logic."""
        self._retry_count = 0
        
        while self._retry_count < self.config.max_retries:
            try:
                response = await asyncio.wait_for(
                    self.provider.chat(messages),
                    timeout=self.config.provider_timeout
                )
                
                # Track tokens
                if hasattr(response, "usage"):
                    self._budget.consume(
                        response.usage.get("prompt_tokens", 0),
                        response.usage.get("completion_tokens", 0)
                    )
                
                return response
                
            except asyncio.TimeoutError:
                self._retry_count += 1
                self._budget.consume(0, 0)  # Don't count failed attempts
                
                if self._retry_count < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * self._retry_count)
                    continue
                
                yield LoopEvent(type="error", data={
                    "error": "Provider timeout",
                    "retries": self._retry_count
                })
                return None
                
            except Exception as e:
                self._retry_count += 1
                
                if self._retry_count < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * self._retry_count)
                    continue
                
                yield LoopEvent(type="error", data={
                    "error": str(e),
                    "retries": self._retry_count
                })
                return None
        
        return None
    
    async def _process_response(self, response: Dict) -> AsyncIterator[LoopEvent]:
        """Process API response."""
        content = response.get("content", "")
        
        # Text content
        if content:
            self._final_response = content
            yield LoopEvent(type="text", data={"content": content})
        
        # Tool calls
        tool_calls = response.get("tool_calls", [])
        if tool_calls:
            self._state = LoopState.TOOL_CALL
            
            for tool_call in tool_calls:
                call_id = tool_call.get("id")
                func_name = tool_call.get("name")
                args = tool_call.get("arguments", {})
                
                # Store pending call
                self._pending_tool_calls[call_id] = {
                    "name": func_name,
                    "arguments": args
                }
                
                # Emit tool call event
                self.callbacks.emit("tool_progress", {
                    "id": call_id,
                    "name": func_name,
                    "args": args
                })
                
                yield LoopEvent(type="tool_call", data={
                    "id": call_id,
                    "name": func_name,
                    "arguments": args
                })
    
    def submit_tool_result(self, tool_call_id: str, result: Any, is_error: bool = False) -> None:
        """Submit a tool result back to the agent."""
        result_content = json.dumps({"result": result, "error": is_error})
        
        # Add tool result message
        self._messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result_content
        })
        
        # Remove from pending
        self._pending_tool_calls.pop(tool_call_id, None)
        
        # Update state
        self._state = LoopState.THINKING
        self._transition_reason = TransitionReason.TOOL_RESULT
    
    async def _compress_context(self) -> None:
        """Compress conversation context."""
        if not self.hermes:
            return
        
        original_count = len(self._messages)
        self.hermes.compress_context()
        
        # Rebuild messages from compressed context
        compressed = self.hermes.get_context(max_tokens=int(self._budget.remaining * 0.9))
        self._messages = self._messages[:2] + compressed  # Keep system + first user
        
        self.callbacks.emit("compression", {
            "original": original_count,
            "compressed": len(self._messages)
        })
    
    def _build_api_messages(self) -> List[Dict]:
        """Build messages for API call."""
        # Get memory context
        memory_context = ""
        if self.hermes:
            memory_context = self.hermes.get_memory_context(max_tokens=2000)
        
        # Build messages
        messages = []
        
        # System with memory
        if memory_context:
            system_content = f"{self._get_system_prompt()}\n\n# Memory Context\n{memory_context}"
        else:
            system_content = self._get_system_prompt()
        
        messages.append({"role": "system", "content": system_content})
        
        # Conversation history (respecting budget)
        for msg in self._messages:
            if msg["role"] == "system":
                continue
            
            msg_tokens = self._estimate_tokens(msg)
            if msg_tokens < self._budget.remaining * 0.1:
                messages.append(msg)
            else:
                break
        
        return messages
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt."""
        return """You are a helpful AI assistant. You have access to various tools to help the user.

Available tools:
- read_file(path): Read file contents
- write_file(path, content): Write content to file
- search_files(directory, pattern): Search for pattern in files
- web_search(query): Search the web
- web_extract(url): Extract content from URL

When you need to use a tool, respond with a tool call in this format:
{
  "tool_calls": [{
    "id": "call_123",
    "name": "tool_name",
    "arguments": {"arg1": "value1"}
  }]
}

After receiving tool results, continue your response.
"""
    
    def _estimate_tokens(self, message: Dict) -> int:
        """Estimate token count for a message."""
        content = message.get("content", "")
        return len(content) // 4  # Rough estimate
    
    def _calculate_pressure(self) -> float:
        """Calculate context pressure ratio."""
        if self._budget.max_tokens == 0:
            return 0
        used = self._budget.prompt_tokens + self._budget.completion_tokens
        return used / self._budget.max_tokens
    
    def get_response(self) -> Optional[str]:
        """Get the final response."""
        return self._final_response
    
    def get_messages(self) -> List[Dict]:
        """Get conversation messages."""
        return self._messages.copy()
    
    def reset(self) -> None:
        """Reset the agent state."""
        self._state = LoopState.IDLE
        self._messages.clear()
        self._pending_tool_calls.clear()
        self._final_response = None
        self._error = None
        self._iteration = 0
        self._retry_count = 0
        self._budget.reset()

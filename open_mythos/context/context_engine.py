"""
Context Engine - Hermes-Style Context Compression

Provides pluggable context compression strategies:
1. Summarization-based compression
2. Reference-based compression (keep pointers to original)
3. Priority-based compression (preserve important messages)
4. Hybrid compression (combine multiple strategies)
"""

import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import defaultdict


class CompressionStrategy(Enum):
    """Context compression strategies."""
    NONE = "none"
    SUMMARIZE = "summarize"
    REFERENCE = "reference"
    PRIORITY = "priority"
    HYBRID = "hybrid"
    SELECTIVE = "selective"


@dataclass
class Message:
    """A message in the conversation context."""
    role: str  # "system", "user", "assistant", "tool"
    content: str
    timestamp: float = field(default_factory=time.time)
    token_count: int = 0
    importance: float = 0.5  # [0, 1]
    is_compressed: bool = False
    compression_ref: Optional[str] = None  # Reference ID if compressed
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
            "importance": self.importance,
            "is_compressed": self.is_compressed,
            "compression_ref": self.compression_ref,
            "metadata": self.metadata,
        }


@dataclass
class CompressionResult:
    """Result of context compression."""
    original_count: int
    compressed_count: int
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    strategy: CompressionStrategy
    preserved_messages: List[Message]
    compressed_messages: List[Message]
    summary: str = ""


@dataclass
class ContextPressure:
    """Context pressure metrics."""
    current_tokens: int
    max_tokens: int
    pressure_ratio: float  # current / max
    warnings: List[str] = field(default_factory=list)
    should_compress: bool = False
    recommended_strategy: CompressionStrategy = CompressionStrategy.NONE


class BaseContextCompressor(ABC):
    """
    Abstract base class for context compressors.
    
    Implementations:
    - SummarizingCompressor: Summarizes old messages
    - ReferenceCompressor: Replaces old messages with references
    - PriorityCompressor: Keeps important messages, compresses rest
    - HybridCompressor: Combines multiple strategies
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def compress(
        self,
        messages: List[Message],
        max_tokens: int,
        **kwargs
    ) -> CompressionResult:
        """
        Compress messages to fit within max_tokens.
        
        Args:
            messages: List of messages to compress
            max_tokens: Maximum tokens allowed
            
        Returns:
            CompressionResult with compressed messages
        """
        pass
    
    @abstractmethod
    def get_importance(self, message: Message) -> float:
        """
        Calculate importance score for a message.
        
        Args:
            message: Message to score
            
        Returns:
            Importance score [0, 1]
        """
        pass


class SummarizingCompressor(BaseContextCompressor):
    """
    Summarizes old messages to preserve context in fewer tokens.
    
    Strategy:
    1. Separate recent messages (keep intact)
    2. Summarize older messages into a compact summary
    3. Preserve system messages and tool definitions
    """
    
    def __init__(
        self,
        recent_messages: int = 10,
        summary_tokens: int = 500,
        summarize_fn: Optional[Callable[[List[Message]], str]] = None
    ):
        super().__init__("summarizing")
        self.recent_messages = recent_messages
        self.summary_tokens = summary_tokens
        self.summarize_fn = summarize_fn or self._default_summarize
    
    def compress(
        self,
        messages: List[Message],
        max_tokens: int,
        **kwargs
    ) -> CompressionResult:
        """Compress by summarizing older messages."""
        if len(messages) <= self.recent_messages:
            return CompressionResult(
                original_count=len(messages),
                compressed_count=len(messages),
                original_tokens=sum(m.token_count for m in messages),
                compressed_tokens=sum(m.token_count for m in messages),
                compression_ratio=1.0,
                strategy=CompressionStrategy.SUMMARIZE,
                preserved_messages=messages,
                compressed_messages=[]
            )
        
        # Separate recent and older messages
        recent = messages[-self.recent_messages:]
        older = messages[:-self.recent_messages]
        
        # Calculate tokens
        recent_tokens = sum(m.token_count for m in recent)
        available_tokens = max_tokens - recent_tokens
        
        # Summarize older messages if needed
        if available_tokens < sum(m.token_count for m in older):
            summary = self.summarize_fn(older)
            summary_message = Message(
                role="system",
                content=f"[Previous conversation summary]\n{summary}",
                token_count=len(summary.split()),
                is_compressed=True,
                compression_ref="summary"
            )
            compressed = [summary_message] + recent
        else:
            compressed = older + recent
        
        # Calculate result
        original_tokens = sum(m.token_count for m in messages)
        compressed_tokens = sum(m.token_count for m in compressed)
        
        return CompressionResult(
            original_count=len(messages),
            compressed_count=len(compressed),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            strategy=CompressionStrategy.SUMMARIZE,
            preserved_messages=recent,
            compressed_messages=[m for m in compressed if m.is_compressed],
            summary=summary if available_tokens < sum(m.token_count for m in older) else ""
        )
    
    def get_importance(self, message: Message) -> float:
        """Calculate message importance."""
        # System messages are most important
        if message.role == "system":
            return 1.0
        
        # Recent messages are more important
        recency = min(1.0, (time.time() - message.timestamp) / 3600)  # Decay over 1 hour
        return 0.3 + (1 - recency) * 0.4 + message.importance * 0.3
    
    def _default_summarize(self, messages: List[Message]) -> str:
        """Default summarization using concatenation of key content."""
        # Extract key information from messages
        contents = []
        for msg in messages[-20:]:  # Last 20 messages
            if msg.role == "user":
                contents.append(f"User: {msg.content[:200]}")
            elif msg.role == "assistant":
                contents.append(f"Assistant: {msg.content[:200]}")
        
        return "\n".join(contents[-10:])  # Last 10 exchanges


class ReferenceCompressor(BaseContextCompressor):
    """
    Replaces old messages with lightweight references.
    
    Strategy:
    1. Keep recent messages intact
    2. Replace older messages with reference pointers
    3. Store compressed content in external storage
    """
    
    def __init__(
        self,
        recent_messages: int = 15,
        storage: Optional[Dict[str, Any]] = None
    ):
        super().__init__("reference")
        self.recent_messages = recent_messages
        self._storage = storage or {}  # External storage for compressed content
        self._ref_counter = 0
    
    def compress(
        self,
        messages: List[Message],
        max_tokens: int,
        **kwargs
    ) -> CompressionResult:
        """Compress by replacing older messages with references."""
        if len(messages) <= self.recent_messages:
            return CompressionResult(
                original_count=len(messages),
                compressed_count=len(messages),
                original_tokens=sum(m.token_count for m in messages),
                compressed_tokens=sum(m.token_count for m in messages),
                compression_ratio=1.0,
                strategy=CompressionStrategy.REFERENCE,
                preserved_messages=messages,
                compressed_messages=[]
            )
        
        # Keep recent messages
        recent = messages[-self.recent_messages:]
        older = messages[:-self.recent_messages]
        
        # Create references for older messages
        compressed = []
        for msg in older:
            ref_id = self._create_reference(msg)
            ref_message = Message(
                role=msg.role,
                content=f"[Reference: {ref_id}]",
                token_count=5,  # Reference is just a few tokens
                is_compressed=True,
                compression_ref=ref_id,
                metadata={"original_timestamp": msg.timestamp}
            )
            compressed.append(ref_message)
        
        compressed.extend(recent)
        
        # Calculate result
        original_tokens = sum(m.token_count for m in messages)
        compressed_tokens = sum(m.token_count for m in compressed)
        
        return CompressionResult(
            original_count=len(messages),
            compressed_count=len(compressed),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            strategy=CompressionStrategy.REFERENCE,
            preserved_messages=recent,
            compressed_messages=[m for m in compressed if m.is_compressed]
        )
    
    def get_importance(self, message: Message) -> float:
        """Calculate message importance."""
        if message.role == "system":
            return 1.0
        
        # Tool messages are important
        if message.role == "tool":
            return 0.9
        
        return 0.5
    
    def _create_reference(self, message: Message) -> str:
        """Create a reference ID and store compressed content."""
        self._ref_counter += 1
        ref_id = f"ref_{self._ref_counter}"
        
        self._storage[ref_id] = {
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp,
            "metadata": message.metadata
        }
        
        return ref_id
    
    def get_reference(self, ref_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve compressed content by reference."""
        return self._storage.get(ref_id)


class PriorityCompressor(BaseContextCompressor):
    """
    Preserves high-importance messages, compresses lower importance ones.
    
    Strategy:
    1. Score all messages by importance
    2. Preserve high-importance messages
    3. Compress/summarize lower-importance messages
    """
    
    def __init__(
        self,
        importance_threshold: float = 0.6,
        preserve_system: bool = True,
        preserve_tools: bool = True
    ):
        super().__init__("priority")
        self.importance_threshold = importance_threshold
        self.preserve_system = preserve_system
        self.preserve_tools = preserve_tools
    
    def compress(
        self,
        messages: List[Message],
        max_tokens: int,
        **kwargs
    ) -> CompressionResult:
        """Compress by preserving important messages."""
        # Score and categorize messages
        high_importance = []
        low_importance = []
        
        for msg in messages:
            importance = self.get_importance(msg)
            if importance >= self.importance_threshold:
                high_importance.append((msg, importance))
            else:
                low_importance.append((msg, importance))
        
        # Sort high importance by score
        high_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Calculate available tokens for low importance
        high_tokens = sum(m.token_count for m, _ in high_importance)
        available_tokens = max_tokens - high_tokens
        
        # Keep all high importance
        preserved = [m for m, _ in high_importance]
        compressed = []
        compressed_tokens_total = 0
        
        # Add low importance if tokens allow
        for msg, _ in low_importance:
            if compressed_tokens_total + msg.token_count <= available_tokens:
                preserved.append(msg)
                compressed_tokens_total += msg.token_count
            else:
                # Create summary reference
                summary_msg = Message(
                    role="system",
                    content=f"[Compressed {len(low_importance)} lower-importance messages]",
                    token_count=10,
                    is_compressed=True,
                    compression_ref="batch_summary"
                )
                compressed.append(summary_msg)
                break
        
        # Calculate result
        original_tokens = sum(m.token_count for m in messages)
        compressed_total_tokens = sum(m.token_count for m in preserved) + sum(m.token_count for m in compressed)
        
        return CompressionResult(
            original_count=len(messages),
            compressed_count=len(preserved) + len(compressed),
            original_tokens=original_tokens,
            compressed_tokens=compressed_total_tokens,
            compression_ratio=compressed_total_tokens / original_tokens if original_tokens > 0 else 1.0,
            strategy=CompressionStrategy.PRIORITY,
            preserved_messages=preserved,
            compressed_messages=compressed
        )
    
    def get_importance(self, message: Message) -> float:
        """Calculate message importance."""
        base = message.importance
        
        # Boost for role
        if message.role == "system":
            return 1.0
        elif message.role == "tool":
            return 0.95
        elif message.role == "user":
            base += 0.1
        
        # Recency boost
        age_hours = (time.time() - message.timestamp) / 3600
        recency_boost = max(0, 0.2 - age_hours * 0.02)  # Decay over 10 hours
        base += recency_boost
        
        return min(1.0, base)


class HybridCompressor(BaseContextCompressor):
    """
    Combines multiple compression strategies.
    
    Uses different strategies for different parts of context:
    - Recent messages: Keep intact
    - Middle messages: Summarize
    - Old messages: Reference
    """
    
    def __init__(
        self,
        recent_count: int = 10,
        middle_count: int = 20,
        summarize_fn: Optional[Callable[[List[Message]], str]] = None,
        storage: Optional[Dict[str, Any]] = None
    ):
        super().__init__("hybrid")
        self.recent_count = recent_count
        self.middle_count = middle_count
        self._summarizer = SummarizingCompressor(
            recent_messages=0,  # We handle recent separately
            summarize_fn=summarize_fn
        )
        self._reference_compressor = ReferenceCompressor(
            recent_messages=0,
            storage=storage
        )
    
    def compress(
        self,
        messages: List[Message],
        max_tokens: int,
        **kwargs
    ) -> CompressionResult:
        """Compress using hybrid strategy."""
        if len(messages) <= self.recent_count:
            return CompressionResult(
                original_count=len(messages),
                compressed_count=len(messages),
                original_tokens=sum(m.token_count for m in messages),
                compressed_tokens=sum(m.token_count for m in messages),
                compression_ratio=1.0,
                strategy=CompressionStrategy.HYBRID,
                preserved_messages=messages,
                compressed_messages=[]
            )
        
        # Divide messages
        recent = messages[-self.recent_count:]
        middle_start = max(0, len(messages) - self.recent_count - self.middle_count)
        middle = messages[middle_start:-self.recent_count]
        old = messages[:middle_start]
        
        # Recent: Keep intact
        preserved = list(recent)
        preserved_tokens = sum(m.token_count for m in recent)
        
        # Middle: Summarize if needed
        middle_tokens = sum(m.token_count for m in middle)
        available = max_tokens - preserved_tokens
        
        if middle_tokens > available:
            # Summarize middle
            summary_content = "\n".join(m.content[:100] for m in middle[-10:])
            summary = Message(
                role="system",
                content=f"[Earlier conversation summary]\n{summary_content}",
                token_count=len(summary_content.split()),
                is_compressed=True,
                compression_ref="middle_summary"
            )
            preserved.append(summary)
            preserved_tokens += summary.token_count
        else:
            preserved.extend(middle)
            preserved_tokens += middle_tokens
        
        # Old: Reference if exists and still over budget
        if preserved_tokens > max_tokens and old:
            ref_msg = Message(
                role="system",
                content=f"[Reference to {len(old)} earlier messages]",
                token_count=8,
                is_compressed=True,
                compression_ref="old_reference"
            )
            preserved.append(ref_msg)
        
        # Calculate result
        original_tokens = sum(m.token_count for m in messages)
        compressed_tokens = sum(m.token_count for m in preserved)
        
        return CompressionResult(
            original_count=len(messages),
            compressed_count=len(preserved),
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / original_tokens if original_tokens > 0 else 1.0,
            strategy=CompressionStrategy.HYBRID,
            preserved_messages=preserved,
            compressed_messages=[]
        )
    
    def get_importance(self, message: Message) -> float:
        """Calculate message importance."""
        if message.role == "system":
            return 1.0
        elif message.role == "tool":
            return 0.9
        elif message.role == "user":
            return 0.7
        return 0.5


class ContextEngine:
    """
    Main context management engine.
    
    Orchestrates compression strategies based on context pressure.
    
    Usage:
        engine = ContextEngine()
        
        # Add messages
        engine.add_message(role="user", content="Hello")
        engine.add_message(role="assistant", content="Hi!")
        
        # Get context for prompt
        context = engine.get_context(max_tokens=4000)
        
        # Check pressure
        pressure = engine.check_pressure(max_tokens=4000)
        if pressure.should_compress:
            result = engine.compress()
    """
    
    def __init__(
        self,
        default_strategy: CompressionStrategy = CompressionStrategy.HYBRID,
        warning_threshold: float = 0.7,
        compression_threshold: float = 0.9
    ):
        self.default_strategy = default_strategy
        self.warning_threshold = warning_threshold
        self.compression_threshold = compression_threshold
        
        self._messages: List[Message] = []
        self._compressor: BaseContextCompressor = self._create_compressor(default_strategy)
        
        # Statistics
        self._stats = {
            "total_compressions": 0,
            "total_tokens_saved": 0,
            "last_compression_ratio": 1.0
        }
    
    def _create_compressor(self, strategy: CompressionStrategy) -> BaseContextCompressor:
        """Create compressor for strategy."""
        if strategy == CompressionStrategy.SUMMARIZE:
            return SummarizingCompressor()
        elif strategy == CompressionStrategy.REFERENCE:
            return ReferenceCompressor()
        elif strategy == CompressionStrategy.PRIORITY:
            return PriorityCompressor()
        elif strategy == CompressionStrategy.HYBRID:
            return HybridCompressor()
        else:
            return SummarizingCompressor()
    
    def set_strategy(self, strategy: CompressionStrategy) -> None:
        """Change compression strategy."""
        self.default_strategy = strategy
        self._compressor = self._create_compressor(strategy)
    
    def add_message(
        self,
        role: str,
        content: str,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: Optional[int] = None
    ) -> Message:
        """Add a message to context."""
        if token_count is None:
            token_count = len(content.split())
        
        message = Message(
            role=role,
            content=content,
            token_count=token_count,
            importance=importance,
            metadata=metadata or {}
        )
        
        self._messages.append(message)
        return message
    
    def get_messages(self) -> List[Message]:
        """Get all messages."""
        return self._messages
    
    def get_context(self, max_tokens: int) -> List[Dict[str, Any]]:
        """Get context formatted for API."""
        total_tokens = sum(m.token_count for m in self._messages)
        
        if total_tokens <= max_tokens:
            return [m.to_dict() for m in self._messages]
        
        # Need to compress
        result = self._compressor.compress(self._messages, max_tokens)
        return [m.to_dict() for m in result.preserved_messages]
    
    def check_pressure(self, max_tokens: int) -> ContextPressure:
        """Check context pressure."""
        current_tokens = sum(m.token_count for m in self._messages)
        pressure_ratio = current_tokens / max_tokens if max_tokens > 0 else 0
        
        warnings = []
        should_compress = False
        recommended_strategy = self.default_strategy
        
        if pressure_ratio >= self.compression_threshold:
            should_compress = True
            warnings.append("Context pressure critical - compression required")
        elif pressure_ratio >= self.warning_threshold:
            warnings.append("Context pressure high - compression recommended")
        
        # Recommend strategy based on pressure
        if pressure_ratio > 0.95:
            recommended_strategy = CompressionStrategy.HYBRID
        elif pressure_ratio > 0.85:
            recommended_strategy = CompressionStrategy.PRIORITY
        
        return ContextPressure(
            current_tokens=current_tokens,
            max_tokens=max_tokens,
            pressure_ratio=pressure_ratio,
            warnings=warnings,
            should_compress=should_compress,
            recommended_strategy=recommended_strategy
        )
    
    def compress(
        self,
        strategy: Optional[CompressionStrategy] = None,
        max_tokens: Optional[int] = None
    ) -> CompressionResult:
        """
        Compress context.
        
        Args:
            strategy: Override compression strategy
            max_tokens: Override max tokens (uses current pressure if not provided)
            
        Returns:
            CompressionResult
        """
        if strategy and strategy != self.default_strategy:
            compressor = self._create_compressor(strategy)
        else:
            compressor = self._compressor
        
        if max_tokens is None:
            # Estimate from current pressure
            total = sum(m.token_count for m in self._messages)
            max_tokens = int(total / self.check_pressure(total).pressure_ratio * 0.9)
        
        result = compressor.compress(self._messages, max_tokens)
        
        # Update messages
        self._messages = result.preserved_messages
        
        # Update stats
        self._stats["total_compressions"] += 1
        self._stats["total_tokens_saved"] += result.original_tokens - result.compressed_tokens
        self._stats["last_compression_ratio"] = result.compression_ratio
        
        return result
    
    def clear(self) -> None:
        """Clear all messages."""
        self._messages.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return {
            **self._stats,
            "current_messages": len(self._messages),
            "current_tokens": sum(m.token_count for m in self._messages),
            "default_strategy": self.default_strategy.value
        }

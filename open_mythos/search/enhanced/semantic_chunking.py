"""
Semantic Chunking Module

Advanced text chunking strategies:
- Fixed-size chunking
- Sentence-based chunking
- Paragraph-based chunking
- Semantic similarity chunking
- Recursive chunking
- Document structure-aware chunking
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import re


@dataclass
class Chunk:
    """A text chunk with metadata."""
    id: str
    content: str
    start_char: int
    end_char: int
    chunk_index: int
    total_chunks: int
    similarity_to_prev: float = 0.0
    similarity_to_next: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def length(self) -> int:
        return len(self.content)
    
    @property
    def char_range(self) -> Tuple[int, int]:
        return (self.start_char, self.end_char)


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: List[str] = field(default_factory=lambda: ["\n\n", "\n", ". ", " "])
    embedding_func: Optional[Callable[[str], List[float]]] = None
    similarity_threshold: float = 0.5


class FixedSizeChunking:
    """Fixed-size character-based chunking."""
    
    def chunk(self, text: str, config: Optional[ChunkingConfig] = None) -> List[Chunk]:
        cfg = config or ChunkingConfig()
        chunks = []
        start = 0
        index = 0
        
        while start < len(text):
            end = start + cfg.chunk_size
            
            if end < len(text):
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space
            
            chunk_text = text[start:end]
            
            chunks.append(Chunk(
                id=f"chunk_{index}",
                content=chunk_text,
                start_char=start,
                end_char=end,
                chunk_index=index,
                total_chunks=0,
                metadata={"strategy": "fixed_size"}
            ))
            
            start = end - cfg.chunk_overlap
            index += 1
        
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks


class SentenceChunking:
    """Chunk by sentences."""
    
    def chunk(self, text: str, config: Optional[ChunkingConfig] = None) -> List[Chunk]:
        cfg = config or ChunkingConfig()
        chunks = []
        
        sentences = self._split_into_sentences(text)
        current_chunk = []
        current_length = 0
        index = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            if current_length + sentence_len > cfg.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                start_char = text.find(chunk_text)
                
                chunks.append(Chunk(
                    id=f"chunk_{index}",
                    content=chunk_text,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    chunk_index=index,
                    total_chunks=0,
                    metadata={"strategy": "sentence", "count": len(current_chunk)}
                ))
                
                if cfg.chunk_overlap > 0 and len(current_chunk) > 1:
                    current_chunk = current_chunk[-1:] + [sentence]
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = [sentence]
                    current_length = sentence_len
                
                index += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_len
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            start_char = text.find(chunk_text)
            if start_char == -1:
                start_char = max(0, len(text) - len(chunk_text))
            
            chunks.append(Chunk(
                id=f"chunk_{index}",
                content=chunk_text,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                chunk_index=index,
                total_chunks=0,
                metadata={"strategy": "sentence", "count": len(current_chunk)}
            ))
        
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        pattern = re.compile(r'[.!?]+\s+')
        parts = pattern.split(text)
        sentences = []
        
        for i, part in enumerate(parts):
            if part.strip():
                if i < len(parts) - 1:
                    sentences.append(part.strip() + ".")
                else:
                    sentences.append(part.strip())
        
        return sentences


class ParagraphChunking:
    """Chunk by paragraphs."""
    
    def chunk(self, text: str, config: Optional[ChunkingConfig] = None) -> List[Chunk]:
        cfg = config or ChunkingConfig()
        chunks = []
        
        paragraphs = re.split(r'\n\s*\n', text)
        current_chunk = []
        current_length = 0
        index = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_len = len(para)
            
            if para_len > cfg.chunk_size:
                if current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    start_char = text.find(chunk_text)
                    
                    chunks.append(Chunk(
                        id=f"chunk_{index}",
                        content=chunk_text,
                        start_char=start_char,
                        end_char=start_char + len(chunk_text),
                        chunk_index=index,
                        total_chunks=0,
                        metadata={"strategy": "paragraph"}
                    ))
                    index += 1
                    current_chunk = []
                    current_length = 0
                
                sub_chunker = SentenceChunking()
                sub_chunks = sub_chunker.chunk(para, cfg)
                for sc in sub_chunks:
                    sc.chunk_index = index
                    chunks.append(sc)
                    index += 1
            
            elif current_length + para_len > cfg.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                start_char = text.find(chunk_text)
                
                chunks.append(Chunk(
                    id=f"chunk_{index}",
                    content=chunk_text,
                    start_char=start_char,
                    end_char=start_char + len(chunk_text),
                    chunk_index=index,
                    total_chunks=0,
                    metadata={"strategy": "paragraph"}
                ))
                
                current_chunk = [para]
                current_length = para_len
                index += 1
            else:
                current_chunk.append(para)
                current_length += para_len
        
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            start_char = text.find(chunk_text)
            if start_char == -1:
                start_char = max(0, len(text) - len(chunk_text))
            
            chunks.append(Chunk(
                id=f"chunk_{index}",
                content=chunk_text,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                chunk_index=index,
                total_chunks=0,
                metadata={"strategy": "paragraph"}
            ))
        
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks


class RecursiveChunking:
    """Recursive chunking with multiple separator levels."""
    
    def chunk(self, text: str, config: Optional[ChunkingConfig] = None) -> List[Chunk]:
        cfg = config or ChunkingConfig()
        
        def split_text(text: str, separators: List[str], size: int) -> List[str]:
            if not separators:
                return [text]
            
            sep = separators[0]
            remaining = separators[1:]
            
            splits = text.split(sep)
            result = []
            current = []
            current_len = 0
            
            for split in splits:
                split_len = len(split)
                
                if current_len + split_len <= size:
                    current.append(split)
                    current_len += split_len
                else:
                    if current:
                        result.append(sep.join(current))
                    
                    if split_len > size:
                        sub = split_text(split, remaining, size)
                        result.extend(sub[:-1])
                        current = [sub[-1]]
                        current_len = len(sub[-1])
                    else:
                        current = [split]
                        current_len = split_len
            
            if current:
                result.append(sep.join(current))
            
            return result
        
        raw_chunks = split_text(text, cfg.separators, cfg.chunk_size)
        
        chunks = []
        start = 0
        for i, chunk_text in enumerate(raw_chunks):
            start_char = text.find(chunk_text, start)
            if start_char == -1:
                start_char = start
            
            chunks.append(Chunk(
                id=f"chunk_{i}",
                content=chunk_text,
                start_char=start_char,
                end_char=start_char + len(chunk_text),
                chunk_index=i,
                total_chunks=len(raw_chunks),
                metadata={"strategy": "recursive"}
            ))
            
            start = start_char + len(chunk_text)
        
        return chunks


class SemanticChunker:
    """Unified interface for chunking strategies."""
    
    STRATEGIES = {
        "fixed": FixedSizeChunking,
        "sentence": SentenceChunking,
        "paragraph": ParagraphChunking,
        "recursive": RecursiveChunking,
    }
    
    def __init__(self, strategy: str = "recursive"):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}")
        self.strategy_name = strategy
        self.strategy = self.STRATEGIES[strategy]()
    
    def chunk(self, text: str, chunk_size: int = 500, **kwargs) -> List[Chunk]:
        cfg = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=kwargs.get("chunk_overlap", 50),
            separators=kwargs.get("separators", ["\n\n", "\n", ". ", " "]),
        )
        return self.strategy.chunk(text, cfg)


__all__ = [
    "Chunk",
    "ChunkingConfig",
    "FixedSizeChunking",
    "SentenceChunking",
    "ParagraphChunking",
    "RecursiveChunking",
    "SemanticChunker",
]

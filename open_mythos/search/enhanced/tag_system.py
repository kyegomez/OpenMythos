"""
Tag System for Memory Management

Provides:
- Tag extraction (automatic and manual)
- Tag-based filtering
- Tag hierarchy and aliases
- Tag suggestions
- Weighted tag scoring
"""

from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import re


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Tag:
    """A tag with metadata."""
    name: str
    category: Optional[str] = None  # e.g., "tech", "domain", "type"
    aliases: Set[str] = field(default_factory=set)
    weight: float = 1.0  # Importance weight
    count: int = 0  # Usage count
    related_tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.name)
    
    def __eq__(self, other: object) -> bool:
        if isinstance(other, Tag):
            return self.name == other.name
        return False


@dataclass
class TaggedItem:
    """An item with associated tags."""
    id: str
    content: str
    tags: Set[str] = field(default_factory=set)
    auto_tags: Set[str] = field(default_factory=set)  # Extracted automatically
    manual_tags: Set[str] = field(default_factory=set)  # User-assigned
    tag_weights: Dict[str, float] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Tag Extractor
# ============================================================================

class TagExtractor:
    """
    Extract tags from text content.
    
    Supports:
    - Keyword extraction
    - Named entity recognition (simple)
    - Pattern-based extraction
    - LLM-based extraction (optional)
    """
    
    def __init__(self):
        # Common stop words to exclude
        self.stop_words: Set[str] = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "must", "shall", "can", "need",
            "this", "that", "these", "those", "i", "you", "he", "she", "it",
            "we", "they", "what", "which", "who", "when", "where", "why", "how",
            "all", "each", "every", "both", "few", "more", "most", "other",
            "some", "such", "only", "own", "same", "so", "than", "too", "very",
        }
        
        # Patterns for tag extraction
        self.patterns = [
            # CamelCase
            r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b',
            # snake_case
            r'\b[a-z]+_[a-z]+(?:_[a-z]+)+\b',
            # Hashtags
            r'#(\w+)',
            # @ mentions (convert to tag)
            r'@(\w+)',
            # Technical terms
            r'\b(?:Python|JavaScript|TypeScript|Java|C\+\+|Go|Rust|Ruby|PHP|Swift|Kotlin)\b',
            # Version numbers
            r'\bv?\d+(?:\.\d+)*\b',
            # File paths
            r'(?:[\w\-\.]+/)+[\w\-\.]+',
            # URLs
            r'https?://\S+',
            # Email
            r'\S+@\S+\.\S+',
        ]
        
        # Keyword indicators
        self.keyword_indicators = [
            "called", "named", "known as", "referred to as",
            "type of", "kind of", "specifically",
        ]
        
        # Domain-specific extractors
        self.domain_extractors: Dict[str, Callable[[str], List[str]]] = {}
    
    def extract(
        self,
        text: str,
        max_tags: int = 10,
        include_patterns: bool = True,
        include_keywords: bool = True,
        include_entities: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Extract tags from text.
        
        Args:
            text: Text to extract tags from
            max_tags: Maximum number of tags to return
            include_patterns: Extract pattern-based tags
            include_keywords: Extract keyword-based tags
            include_entities: Extract named entities
        
        Returns:
            List of (tag, confidence) tuples
        """
        tags: Dict[str, float] = {}
        text_lower = text.lower()
        
        # Extract patterns
        if include_patterns:
            for pattern in self.patterns:
                for match in re.findall(pattern, text):
                    tag = match.lower() if isinstance(match, str) else str(match)
                    if len(tag) > 2:
                        tags[tag] = tags.get(tag, 0.0) + 0.8
        
        # Extract keywords (non-stop words that appear multiple times)
        if include_keywords:
            words = re.findall(r'\b[a-z]{3,}\b', text_lower)
            word_freq = defaultdict(int)
            for word in words:
                if word not in self.stop_words:
                    word_freq[word] += 1
            
            # Score by frequency (with diminishing returns)
            for word, freq in word_freq.items():
                if freq >= 2:  # At least 2 occurrences
                    score = min(1.0, freq / 5)  # Cap at 5 occurrences
                    if word not in tags:
                        tags[word] = score * 0.7
        
        # Extract named entities (simple pattern-based)
        if include_entities:
            entities = self._extract_simple_entities(text)
            for entity in entities:
                tags[entity.lower()] = tags.get(entity.lower(), 0.0) + 0.9
        
        # Apply domain-specific extractors
        for domain, extractor in self.domain_extractors.items():
            try:
                domain_tags = extractor(text)
                for tag in domain_tags:
                    tags[tag.lower()] = tags.get(tag.lower(), 0.0) + 0.85
            except Exception:
                continue
        
        # Sort by score and return top tags
        sorted_tags = sorted(tags.items(), key=lambda x: x[1], reverse=True)
        return sorted_tags[:max_tags]
    
    def _extract_simple_entities(self, text: str) -> List[str]:
        """Extract simple named entities using patterns."""
        entities = []
        
        # Capitalized proper nouns (simple heuristic)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for entity in capitalized:
            # Filter out sentence starts and common words
            if entity.lower() not in self.stop_words and len(entity) > 2:
                entities.append(entity)
        
        # Known entity patterns
        known_patterns = [
            r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.\s+[A-Z][a-z]+',  # People
            r'\b[A-Z][a-z]+,?\s+Inc\.?',  # Companies
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:University|College|School)\b',  # Institutions
        ]
        
        for pattern in known_patterns:
            entities.extend(re.findall(pattern, text))
        
        return list(set(entities))
    
    def register_domain_extractor(
        self,
        domain: str,
        extractor: Callable[[str], List[str]]
    ) -> None:
        """Register a domain-specific tag extractor."""
        self.domain_extractors[domain] = extractor


# ============================================================================
# Tag Repository
# ============================================================================

class TagRepository:
    """
    Repository for managing tags and their relationships.
    
    Features:
    - Tag CRUD operations
    - Tag hierarchy (parent/child)
    - Tag aliases
    - Tag co-occurrence tracking
    - Tag suggestions
    """
    
    def __init__(self):
        self.tags: Dict[str, Tag] = {}
        self.items_by_tag: Dict[str, Set[str]] = defaultdict(set)
        self.tag_relationships: Dict[str, Set[str]] = defaultdict(set)
        self.tag_hierarchy: Dict[str, str] = {}  # child -> parent
    
    def add_tag(
        self,
        name: str,
        category: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        weight: float = 1.0
    ) -> Tag:
        """Add a new tag."""
        tag = Tag(
            name=name.lower(),
            category=category,
            aliases=set(aliases or []),
            weight=weight
        )
        self.tags[name.lower()] = tag
        return tag
    
    def get_tag(self, name: str) -> Optional[Tag]:
        """Get a tag by name (checks aliases)."""
        name_lower = name.lower()
        
        if name_lower in self.tags:
            return self.tags[name_lower]
        
        # Check aliases
        for tag in self.tags.values():
            if name_lower in tag.aliases:
                return tag
        
        return None
    
    def add_item_to_tag(self, item_id: str, tag_name: str) -> None:
        """Associate an item with a tag."""
        tag_name_lower = tag_name.lower()
        
        if tag_name_lower not in self.tags:
            self.add_tag(tag_name_lower)
        
        self.items_by_tag[tag_name_lower].add(item_id)
        self.tags[tag_name_lower].count = len(self.items_by_tag[tag_name_lower])
    
    def remove_item_from_tag(self, item_id: str, tag_name: str) -> None:
        """Remove an item from a tag."""
        tag_name_lower = tag_name.lower()
        
        if tag_name_lower in self.items_by_tag:
            self.items_by_tag[tag_name_lower].discard(item_id)
            if tag_name_lower in self.tags:
                self.tags[tag_name_lower].count = len(self.items_by_tag[tag_name_lower])
    
    def set_tag_hierarchy(self, child_tag: str, parent_tag: str) -> None:
        """Set parent-child relationship between tags."""
        self.tag_hierarchy[child_tag.lower()] = parent_tag.lower()
        self.tag_relationships[parent_tag.lower()].add(child_tag.lower())
    
    def get_child_tags(self, tag_name: str) -> Set[str]:
        """Get all child tags of a tag."""
        return self.tag_relationships.get(tag_name.lower(), set())
    
    def get_parent_tags(self, tag_name: str) -> List[str]:
        """Get all parent tags (ancestors) of a tag."""
        parents = []
        current = tag_name.lower()
        
        while current in self.tag_hierarchy:
            parent = self.tag_hierarchy[current]
            parents.append(parent)
            current = parent
        
        return parents
    
    def get_items_by_tags(
        self,
        tags: List[str],
        match_all: bool = False
    ) -> Set[str]:
        """
        Get items that match the given tags.
        
        Args:
            tags: List of tag names
            match_all: If True, item must have all tags. If False, any tag matches.
        
        Returns:
            Set of item IDs
        """
        if not tags:
            return set()
        
        result: Optional[Set[str]] = None
        
        for tag in tags:
            tag_lower = tag.lower()
            tag_items = self.items_by_tag.get(tag_lower, set())
            
            if result is None:
                result = set(tag_items)
            elif match_all:
                result &= tag_items
            else:
                result |= tag_items
        
        return result or set()
    
    def suggest_tags(
        self,
        text: str,
        existing_tags: Set[str],
        max_suggestions: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Suggest tags for text, avoiding already existing tags.
        
        Returns:
            List of (tag, score) tuples
        """
        extractor = TagExtractor()
        extracted = extractor.extract(text, max_tags=max_suggestions * 2)
        
        suggestions = []
        for tag, score in extracted:
            if tag.lower() not in existing_tags and tag.lower() not in self.stop_words:
                suggestions.append((tag, score))
        
        # Sort by score
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:max_suggestions]
    
    def track_cooccurrence(self, tag1: str, tag2: str) -> None:
        """Track co-occurrence of two tags."""
        t1, t2 = tag1.lower(), tag2.lower()
        self.tags[t1].related_tags.add(t2)
        self.tags[t2].related_tags.add(t1)
    
    def get_related_tags(self, tag_name: str, max_related: int = 5) -> List[Tuple[str, float]]:
        """Get tags that frequently co-occur with this tag."""
        tag = self.get_tag(tag_name)
        if not tag:
            return []
        
        related = []
        for related_tag_name in tag.related_tags:
            related_tag = self.get_tag(related_tag_name)
            if related_tag:
                # Score based on co-occurrence count and weight
                score = related_tag.count * related_tag.weight
                related.append((related_tag_name, score))
        
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:max_related]
    
    def search_tags(self, query: str, limit: int = 10) -> List[Tag]:
        """Search tags by name (includes aliases)."""
        query_lower = query.lower()
        results = []
        
        for tag in self.tags.values():
            if (query_lower in tag.name or
                query_lower in tag.aliases or
                (tag.category and query_lower in tag.category)):
                results.append(tag)
        
        # Sort by weight and count
        results.sort(key=lambda t: (t.weight * t.count), reverse=True)
        return results[:limit]
    
    @property
    def stop_words(self) -> Set[str]:
        """Get stop words set."""
        return {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"}


# ============================================================================
# Tag-based Filter
# ============================================================================

class TagFilter:
    """
    Filter items based on tags.
    
    Supports:
    - Include tags (OR/AND)
    - Exclude tags
    - Tag weight thresholds
    - Hierarchical tag matching
    """
    
    def __init__(self, repository: Optional[TagRepository] = None):
        self.repository = repository or TagRepository()
    
    def filter_items(
        self,
        items: List[TaggedItem],
        include_tags: Optional[List[str]] = None,
        exclude_tags: Optional[List[str]] = None,
        match_all: bool = False,
        min_weight: float = 0.0
    ) -> List[TaggedItem]:
        """
        Filter items by tags.
        
        Args:
            items: List of TaggedItem objects
            include_tags: Tags to include (None = all)
            exclude_tags: Tags to exclude
            match_all: If True, item must have ALL include_tags
            min_weight: Minimum tag weight threshold
        
        Returns:
            Filtered list of items
        """
        if include_tags is None and exclude_tags is None:
            return items
        
        filtered = []
        
        for item in items:
            # Check exclude tags
            if exclude_tags:
                if any(t.lower() in [et.lower() for et in exclude_tags] for t in item.tags):
                    continue
            
            # Check include tags
            if include_tags:
                item_tag_set = {t.lower() for t in item.tags}
                include_lower = {t.lower() for t in include_tags}
                
                if match_all:
                    # Must have all tags
                    if not include_lower.issubset(item_tag_set):
                        continue
                else:
                    # Must have at least one tag
                    if not include_lower & item_tag_set:
                        continue
            
            # Check weight threshold
            if min_weight > 0:
                max_item_weight = max(
                    item.tag_weights.get(t.lower(), 1.0)
                    for t in item.tags
                )
                if max_item_weight < min_weight:
                    continue
            
            filtered.append(item)
        
        return filtered
    
    def rank_by_tags(
        self,
        items: List[TaggedItem],
        query_tags: List[str],
        boost_hierarchy: bool = True
    ) -> List[Tuple[TaggedItem, float]]:
        """
        Rank items by tag relevance to query.
        
        Returns:
            List of (item, score) tuples sorted by score descending
        """
        query_lower = {t.lower() for t in query_tags}
        scored_items = []
        
        for item in items:
            item_tags_lower = {t.lower() for t in item.tags}
            
            # Direct match score
            direct_matches = query_lower & item_tags_lower
            match_score = len(direct_matches) / len(query_lower) if query_lower else 0
            
            # Hierarchy boost
            hierarchy_boost = 0.0
            if boost_hierarchy:
                for qt in query_lower:
                    for item_tag in item_tags_lower:
                        # Check if query tag is parent of item tag
                        parents = self.repository.get_parent_tags(item_tag)
                        if qt in parents:
                            hierarchy_boost += 0.1
            
            # Weight boost
            weight_boost = 0.0
            for tag in direct_matches:
                weight_boost += item.tag_weights.get(tag, 1.0)
            
            total_score = match_score + hierarchy_boost + (weight_boost * 0.1)
            scored_items.append((item, total_score))
        
        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "Tag",
    "TaggedItem",
    "TagExtractor",
    "TagRepository",
    "TagFilter",
]

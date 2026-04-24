"""
Query Expansion Module

Advanced query expansion techniques:
- Multi-query expansion
- Step-back prompting (generalization)
- Drill-down (specialization)
- HyDE (Hypothetical Document Embedding)
- Query decomposition
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import re


# ============================================================================
# Query Expansion Strategies
# ============================================================================

class QueryExpansionStrategy:
    """Base class for query expansion strategies."""
    
    def expand(self, query: str) -> List[str]:
        """Expand a query into multiple variants."""
        raise NotImplementedError


class SynonymExpansion(QueryExpansionStrategy):
    """
    Expand queries using synonyms.
    """
    
    def __init__(self):
        # Common programming/technical synonyms
        self.synonym_map: Dict[str, List[str]] = {
            # Actions
            "find": ["search", "locate", "get", "retrieve", "fetch"],
            "show": ["display", "view", "list", "present", "exhibit"],
            "create": ["make", "add", "new", "insert", "generate", "build"],
            "update": ["edit", "modify", "change", "alter", "revise", "patch"],
            "delete": ["remove", "drop", "erase", "eliminate", "clear"],
            "fix": ["repair", "resolve", "debug", "correct", "address"],
            "check": ["verify", "validate", "test", "examine", "inspect"],
            "get": ["obtain", "retrieve", "fetch", "acquire", "receive"],
            "set": ["configure", "establish", "define", "assign", "specify"],
            "run": ["execute", "start", "launch", "trigger", "invoke"],
            "stop": ["halt", "terminate", "end", "kill", "cancel"],
            "list": ["enumerate", "show", "display", "catalog"],
            
            # Concepts
            "error": ["bug", "issue", "problem", "failure", "exception"],
            "config": ["configuration", "settings", "options", "preferences"],
            "memory": ["storage", "state", "context", "cache"],
            "file": ["document", "data", "artifact", "resource"],
            "function": ["method", "procedure", "routine", "callable"],
            "class": ["type", "object", "entity", "structure"],
            "test": ["spec", "verification", "validation", "check"],
            "deploy": ["release", "publish", "push", "ship"],
            "build": ["compile", "construct", "assemble", "make"],
            "install": ["setup", "configure", "deploy", "initialize"],
            
            # Modifiers
            "fast": ["quick", "rapid", "efficient", "speedy"],
            "slow": ["laggy", "inefficient", "degraded", "poor"],
            "big": ["large", "huge", "major", "significant"],
            "small": ["tiny", "minor", "minimal", "compact"],
            "new": ["recent", "latest", "fresh", "updated"],
            "old": ["legacy", "deprecated", "outdated", "historic"],
        }
    
    def expand(self, query: str) -> List[str]:
        """Generate synonym-based expansions."""
        words = query.lower().split()
        expansions = []
        
        for i, word in enumerate(words):
            if word in self.synonym_map:
                for synonym in self.synonym_map[word][:2]:  # Limit to 2 per word
                    new_words = words[:i] + [synonym] + words[i+1:]
                    expansions.append(" ".join(new_words))
        
        return expansions


class GeneralizationExpansion(QueryExpansionStrategy):
    """
    Expand queries by generalizing (step-back prompting).
    
    Creates broader versions of the query.
    """
    
    def expand(self, query: str) -> List[str]:
        """Generate generalized versions."""
        expansions = []
        
        # Remove specific details
        # Remove numbers
        generalized = re.sub(r'\d+', '', query)
        
        # Remove specific terms
        specific_terms = [
            "specific", "particular", "exact", "precise",
            "certain", "individual", "single", "one"
        ]
        for term in specific_terms:
            generalized = generalized.replace(term, "")
        
        # Clean up extra spaces
        generalized = " ".join(generalized.split())
        
        if generalized and generalized != query:
            expansions.append(generalized)
        
        # Create broader category versions
        broadeners = {
            "error": "issue",
            "bug": "problem",
            "fail": "error",
            "slow": "performance",
            "crash": "failure",
        }
        
        for specific, broad in broadeners.items():
            if specific in query.lower():
                broad_query = query.lower().replace(specific, broad)
                if broad_query != query:
                    expansions.append(broad_query)
        
        return expansions[:2]  # Limit to 2


class SpecializationExpansion(QueryExpansionStrategy):
    """
    Expand queries by specializing (drill-down).
    
    Creates more specific versions of the query.
    """
    
    def expand(self, query: str) -> List[str]:
        """Generate specialized versions."""
        expansions = []
        query_lower = query.lower()
        
        # Add context keywords
        if "file" in query_lower:
            expansions.append(query + " file path location")
        if "error" in query_lower:
            expansions.append(query + " error message details")
        if "config" in query_lower:
            expansions.append(query + " configuration settings options")
        if "test" in query_lower:
            expansions.append(query + " test case specification")
        if "memory" in query_lower:
            expansions.append(query + " memory allocation storage")
        if "performance" in query_lower or "slow" in query_lower:
            expansions.append(query + " optimization efficiency")
        if "api" in query_lower:
            expansions.append(query + " endpoint request response")
        
        return expansions[:2]  # Limit to 2


class PhrasingExpansion(QueryExpansionStrategy):
    """
    Expand queries using different phrasings.
    """
    
    def expand(self, query: str) -> List[str]:
        """Generate different phrasings."""
        expansions = []
        
        # Pattern-based rewrites
        rewrites = [
            # Question patterns
            (r'^how\s+to\s+', 'ways to '),
            (r'^how\s+do\s+i\s+', 'how can I '),
            (r'^what\s+is\s+', 'explain '),
            (r'^why\s+does\s+', 'reason for '),
            (r'^when\s+to\s+', 'appropriate time for '),
            
            # Negation patterns
            (r"can't", 'cannot'),
            (r"don't", 'do not'),
            (r"won't", 'will not'),
            (r"isn't", 'is not'),
            (r"doesn't", 'does not'),
            
            # Technical patterns
            (r'memory', 'RAM'),
            (r'file', 'document'),
        ]
        
        for pattern, replacement in rewrites:
            new_query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
            if new_query != query:
                expansions.append(new_query)
        
        return expansions[:2]


class DecompositionExpansion(QueryExpansionStrategy):
    """
    Decompose complex queries into sub-queries.
    """
    
    def expand(self, query: str) -> List[str]:
        """Decompose into sub-queries."""
        expansions = []
        
        # Split on "and", "or", ","
        if " and " in query:
            parts = query.split(" and ")
            for part in parts:
                cleaned = part.strip()
                if cleaned:
                    expansions.append(cleaned)
        
        if " or " in query:
            parts = query.split(" or ")
            for part in parts:
                cleaned = part.strip()
                if cleaned:
                    expansions.append(cleaned)
        
        # Handle compound queries
        compound_patterns = [
            r'(\w+)\s+and\s+(\w+)',
            r'both\s+(\w+)\s+and\s+(\w+)',
        ]
        
        return expansions[:3]  # Limit to 3 sub-queries


# ============================================================================
# Query Expander Orchestrator
# ============================================================================

class MultiStrategyQueryExpander:
    """
    Orchestrates multiple query expansion strategies.
    
    Combines:
    - Synonym expansion
    - Generalization (step-back)
    - Specialization (drill-down)
    - Phrasing variations
    - Decomposition
    """
    
    def __init__(
        self,
        strategies: Optional[List[QueryExpansionStrategy]] = None,
        max_expansions: int = 5
    ):
        if strategies is None:
            strategies = [
                SynonymExpansion(),
                GeneralizationExpansion(),
                SpecializationExpansion(),
                PhrasingExpansion(),
            ]
        
        self.strategies = strategies
        self.max_expansions = max_expansions
    
    def expand(self, query: str) -> Tuple[str, List[str]]:
        """
        Expand a query using multiple strategies.
        
        Args:
            query: Original query
        
        Returns:
            Tuple of (primary_expansion, all_expansions)
        """
        all_expansions = [query]
        
        for strategy in self.strategies:
            try:
                expansions = strategy.expand(query)
                all_expansions.extend(expansions)
            except Exception:
                continue
        
        # Deduplicate while preserving order
        seen = set()
        unique_expansions = []
        for exp in all_expansions:
            if exp not in seen and exp.strip():
                seen.add(exp)
                unique_expansions.append(exp)
        
        # Limit total expansions
        if len(unique_expansions) > self.max_expansions:
            # Always keep original, add diverse others
            result = [unique_expansions[0]]
            remaining = unique_expansions[1:]
            
            # Pick diverse strategies
            step = max(1, len(remaining) // (self.max_expansions - 1))
            for i in range(0, len(remaining), step):
                result.append(remaining[i])
                if len(result) >= self.max_expansions:
                    break
            
            unique_expansions = result
        
        return unique_expansions[0] if unique_expansions else query, unique_expansions


# ============================================================================
# HyDE Integration
# ============================================================================

class HyDEQueryExpander:
    """
    HyDE-based query expansion.
    
    Uses hypothetical document generation to improve retrieval.
    """
    
    def __init__(
        self,
        llm_provider: Optional[Callable[[str], str]] = None,
        embed_provider: Optional[Callable[[str], List[float]]] = None
    ):
        """
        Initialize HyDE expander.
        
        Args:
            llm_provider: Function to generate hypothetical documents
            embed_provider: Function to embed text (for actual implementation)
        """
        self.llm_provider = llm_provider
        self.embed_provider = embed_provider
    
    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query.
        
        In production, this uses an LLM. For testing, we use a template.
        """
        if self.llm_provider:
            prompt = self._build_hyde_prompt(query)
            return self.llm_provider(prompt)
        
        # Fallback template-based generation
        return self._template_hyde(query)
    
    def _build_hyde_prompt(self, query: str) -> str:
        """Build prompt for HyDE document generation."""
        return f"""Generate a hypothetical document that directly answers the following query.

Query: {query}

The document should:
1. Contain specific, factual information that answers the query
2. Use appropriate technical terminology
3. Be structured clearly with key points
4. Include relevant examples where applicable

Write the hypothetical document:"""

    def _template_hyde(self, query: str) -> str:
        """Template-based HyDE (fallback when no LLM available)."""
        return f"""Document: {query}

This document addresses the query "{query}".

Key Information:
- The main topic relates to {query}
- Important aspects include implementation details
- Best practices suggest considering performance and reliability
- Common approaches involve proper error handling and validation

Related Topics:
- Configuration and setup for {query}
- Troubleshooting common issues with {query}
- Optimization techniques for {query}

Summary:
To properly handle {query}, one should understand the underlying principles,
follow established patterns, and consider the specific requirements of the use case.
""".strip()


# ============================================================================
# Adaptive Query Expansion
# ============================================================================

class AdaptiveQueryExpander:
    """
    Adaptive query expansion that selects strategies based on query type.
    """
    
    def __init__(self):
        self.base_expander = MultiStrategyQueryExpander()
        self.hyde = HyDEQueryExpander()
    
    def expand(self, query: str) -> Tuple[str, List[str]]:
        """
        Adaptively expand query based on its characteristics.
        """
        query_type = self._classify_query(query)
        
        strategies = []
        
        # Select strategies based on query type
        if query_type == "factual":
            strategies = ["synonym", "phrasing"]
        elif query_type == "how-to":
            strategies = ["generalization", "specialization"]
        elif query_type == "error":
            strategies = ["specialization", "decomposition"]
        elif query_type == "complex":
            strategies = ["decomposition", "generalization"]
        else:
            strategies = ["synonym", "phrasing"]
        
        # Build expander with selected strategies
        expander = self._build_expander(strategies)
        return expander.expand(query)
    
    def _classify_query(self, query: str) -> str:
        """Classify query type."""
        query_lower = query.lower()
        
        if any(kw in query_lower for kw in ["how to", "how do", "ways to"]):
            return "how-to"
        elif any(kw in query_lower for kw in ["error", "bug", "fail", "issue"]):
            return "error"
        elif "?" in query or query_lower.startswith(("what", "who", "when", "where", "why")):
            return "factual"
        elif len(query.split()) > 10:
            return "complex"
        
        return "general"
    
    def _build_expander(self, strategies: List[str]) -> MultiStrategyQueryExpander:
        """Build expander with specified strategies."""
        strategy_objects = []
        
        for s in strategies:
            if s == "synonym":
                strategy_objects.append(SynonymExpansion())
            elif s == "generalization":
                strategy_objects.append(GeneralizationExpansion())
            elif s == "specialization":
                strategy_objects.append(SpecializationExpansion())
            elif s == "phrasing":
                strategy_objects.append(PhrasingExpansion())
            elif s == "decomposition":
                strategy_objects.append(DecompositionExpansion())
        
        return MultiStrategyQueryExpander(strategy_objects)


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "QueryExpansionStrategy",
    "SynonymExpansion",
    "GeneralizationExpansion",
    "SpecializationExpansion",
    "PhrasingExpansion",
    "DecompositionExpansion",
    "MultiStrategyQueryExpander",
    "HyDEQueryExpander",
    "AdaptiveQueryExpander",
]

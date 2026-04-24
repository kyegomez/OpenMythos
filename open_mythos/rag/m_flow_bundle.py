"""
P4-1: M-Flow Style Bundle Search for OpenMythos
================================================

Graph-led retrieval replacing flat vector similarity.

Core insight from M-flow:
  "Relevance is not a score. It's a path."

This module implements:
  1. SemanticEdge — edges carrying natural language descriptions (M-flow Break #1)
  2. EpisodeBundle — retrieval result unit scored by min-cost path (not average)
  3. RelationshipIndex — graph relationship structure for fast traversal
  4. episodic_bundle_search() — the full 11-step retrieval pipeline
  5. GraphBundleMemory — integration with OpenMythos ThreeLayerMemorySystem

Design breaks from traditional RAG:
  - Break #1: Edges have semantics (edge_text is vectorized and scored)
  - Break #2: Minimum cost path wins, not average (one strong chain is enough)
  - Break #3: Direct hits are penalized (Episode summaries are too generic)
"""

from __future__ import annotations

import heapq
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Protocol

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    _HAS_TORCH = False


# ============================================================================
# Data Structures
# ============================================================================


class NodeType(Enum):
    """M-flow's four-layer cone graph node types."""

    EPISODE = auto()      # Base: bounded semantic focus (incident/decision/workflow)
    FACET = auto()         # Middle: one dimension of an Episode
    FACET_POINT = auto()   # Tip: atomic assertion or fact
    ENTITY = auto()        # Tip: named thing (person/tool/metric) linked across Episodes

    def priority(self) -> int:
        """Lower = expanded first in graph projection."""
        return {NodeType.EPISODE: 0, NodeType.FACET: 1, NodeType.FACET_POINT: 2, NodeType.ENTITY: 3}[self]

    def layer_name(self) -> str:
        return self.name.lower()


@dataclass
class SemanticEdge:
    """
    M-flow Break #1: Edges are first-class semantic carriers.

    Every edge carries a natural language description that is vectorized
    and searchable. During cost propagation, the system knows not just
    "a connection exists" but "how relevant this connection is to the query."

    Attributes:
        edge_id       -- unique edge ID
        from_node     -- source node ID
        to_node       -- target node ID
        edge_text     -- natural language description of the relationship
        embedding     -- vectorized edge_text for similarity scoring
        relationship  -- typed relationship name (e.g. "has_facet", "involves_entity")
        attributes    -- arbitrary metadata (weights, timestamps, etc.)
    """
    edge_id: str
    from_node: str
    to_node: str
    edge_text: str
    embedding: Optional[np.ndarray] = None
    relationship: str = ""
    attributes: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.edge_id is None:
            self.edge_id = str(uuid.uuid4())

    def get_edge_key(self) -> str:
        """Key used for embedding lookup in edge_hit_map."""
        return (
            self.attributes.get("edge_text")
            or self.edge_text
            or self.relationship
            or self.edge_id
        )


@dataclass
class GraphNode:
    """
    A node in the M-flow Cone Graph.

    Corresponds to one of four levels: Episode, Facet, FacetPoint, Entity.
    Each node has a text representation and an optional vector embedding.
    """
    node_id: str
    node_type: NodeType
    name: str
    description: str = ""
    content: str = ""          # Full content (for Episode summary, Facet detail, etc.)
    embedding: Optional[np.ndarray] = None
    display_only: str = ""     # Text preferred for display over description
    search_text: str = ""     # Text used for lexical search
    attributes: dict = field(default_factory=dict)

    def get_text_for_embedding(self) -> str:
        """Text used for vector embedding lookup."""
        return self.search_text or self.description or self.name


@dataclass
class EpisodeBundle:
    """
    Episode retrieval result unit — scored by minimum evidence path cost.

    M-flow Break #2: One strong chain of evidence is sufficient.
    We only look at the BEST path to each Episode, not the average of all paths.

    Attributes:
        episode_id     -- The Episode node ID
        score          -- Minimum cost across all paths (lower = better)
        best_path      -- Path type that achieved minimum: "direct"|"facet"|"point"|"entity"|"facet_entity"
        best_support_id-- ID of the supporting node (Facet/FacetPoint/Entity) on the best path
    """
    episode_id: str
    score: float
    best_path: str = "direct"
    best_support_id: Optional[str] = None
    best_facet_id: Optional[str] = None
    best_point_id: Optional[str] = None
    best_entity_id: Optional[str] = None

    def __lt__(self, other: EpisodeBundle) -> bool:
        return self.score < other.score


@dataclass
class RelationshipIndex:
    """
    Fast in-memory graph index for O(1) edge lookups.

    Built once from the MemoryGraph, then reused for all bundle computations.

    Structure:
        episode_ids / facet_ids / point_ids / entity_ids -- all node ID sets
        ep_facet_edge[(ep_id, facet_id)] -- Episode→Facet edge
        facet_point_edge[(facet_id, point_id)] -- Facet→FacetPoint edge
        ep_entity_edge[(ep_id, entity_id)] -- Episode→Entity edge
        facet_entity_edge[(facet_id, entity_id)] -- Facet→Entity edge

        facets_by_episode[ep_id] → Set[facet_ids]
        points_by_facet[facet_id] → Set[point_ids]
        entities_by_episode[ep_id] → Set[entity_ids]
        entities_by_facet[facet_id] → Set[entity_ids]
    """
    episode_ids: set = field(default_factory=set)
    facet_ids: set = field(default_factory=set)
    point_ids: set = field(default_factory=set)
    entity_ids: set = field(default_factory=set)

    ep_facet_edge: dict = field(default_factory=dict)
    facet_point_edge: dict = field(default_factory=dict)
    ep_entity_edge: dict = field(default_factory=dict)
    facet_entity_edge: dict = field(default_factory=dict)

    facets_by_episode: dict = field(default_factory=dict)
    points_by_facet: dict = field(default_factory=dict)
    entities_by_episode: dict = field(default_factory=dict)
    entities_by_facet: dict = field(default_factory=dict)

    # Raw edges list
    all_edges: list = field(default_factory=list)


# ============================================================================
# Relationship Index Builder
# ============================================================================


class RelationshipBuilder:
    """
    Builds a RelationshipIndex from an in-memory graph.

    Converts the graph structure into fast O(1) lookup tables for
    bundle scoring.
    """

    @staticmethod
    def build(nodes: list[GraphNode], edges: list[SemanticEdge]) -> RelationshipIndex:
        """
        Build a RelationshipIndex from nodes and edges.

        Args:
            nodes -- all GraphNodes (Episodes, Facets, FacetPoints, Entities)
            edges -- all SemanticEdges connecting the nodes

        Returns:
            RelationshipIndex with all lookup tables populated
        """
        index = RelationshipIndex()
        node_map: dict[str, GraphNode] = {}

        # Classify and register nodes
        for node in nodes:
            node_map[node.node_id] = node
            if node.node_type == NodeType.EPISODE:
                index.episode_ids.add(node.node_id)
            elif node.node_type == NodeType.FACET:
                index.facet_ids.add(node.node_id)
            elif node.node_type == NodeType.FACET_POINT:
                index.point_ids.add(node.node_id)
            elif node.node_type == NodeType.ENTITY:
                index.entity_ids.add(node.node_id)

        # Index edges by relationship type
        for edge in edges:
            rel = edge.relationship.lower()
            if rel in ("has_facet", "contains_facet"):
                index.ep_facet_edge[(edge.from_node, edge.to_node)] = edge
                index.facets_by_episode.setdefault(edge.from_node, set()).add(edge.to_node)
            elif rel in ("has_point", "contains_point"):
                index.facet_point_edge[(edge.from_node, edge.to_node)] = edge
                index.points_by_facet.setdefault(edge.from_node, set()).add(edge.to_node)
            elif rel in ("involves_entity", "has_entity"):
                # Could be Episode→Entity or Facet→Entity
                if edge.from_node in index.episode_ids:
                    index.ep_entity_edge[(edge.from_node, edge.to_node)] = edge
                    index.entities_by_episode.setdefault(edge.from_node, set()).add(edge.to_node)
                elif edge.from_node in index.facet_ids:
                    index.facet_entity_edge[(edge.from_node, edge.to_node)] = edge
                    index.entities_by_facet.setdefault(edge.from_node, set()).add(edge.to_node)
            elif rel in ("has_facet_entity",):
                index.facet_entity_edge[(edge.from_node, edge.to_node)] = edge
                index.entities_by_facet.setdefault(edge.from_node, set()).add(edge.to_node)

        index.all_edges = edges
        return index


# ============================================================================
# Episode Bundle Scorer — Core of M-flow Break #2
# ============================================================================


@dataclass
class BundleScorerConfig:
    """Configuration for bundle scoring."""

    # Edge cost when a required edge has no embedding match
    edge_miss_cost: float = 0.5

    # Cost added per graph hop during propagation
    hop_cost: float = 0.1

    # Penalty for direct Episode hit (Break #3: prevent generic summaries winning)
    direct_episode_penalty: float = 0.15

    # Facet direct-match discount thresholds (Break #2: low facet score → reduced edge costs)
    facet_discount_thresh_1: float = 0.1
    facet_discount_thresh_2: float = 0.2
    facet_discount_factor_1: float = 0.1   # for thresh_1
    facet_discount_factor_2: float = 0.3   # for thresh_2

    # Maximum facets per episode to consider
    max_facets_per_episode: int = 50
    max_points_per_facet: int = 20


class BundleScorer:
    """
    Computes EpisodeBundle scores using minimum-cost path ranking.

    M-flow Break #2: "Minimum, not Average"
    One strong chain of evidence is sufficient — we only look at the best path.
    """

    def __init__(self, config: Optional[BundleScorerConfig] = None):
        self.cfg = config or BundleScorerConfig()

    def compute_bundles(
        self,
        index: RelationshipIndex,
        node_distances: dict[str, float],
        edge_hit_map: dict[str, float],
    ) -> list[EpisodeBundle]:
        """
        Compute EpisodeBundle for all episodes.

        Args:
            index         -- RelationshipIndex from build_relationship_index()
            node_distances -- {node_id: vector_distance} for all node types
            edge_hit_map  -- {edge_text: vector_distance} for edge embeddings

        Returns:
            List[EpisodeBundle] — unsorted, scored by minimum path cost
        """
        bundles: list[EpisodeBundle] = []

        for ep_id in index.episode_ids:
            bundle = self._compute_episode_bundle(index, ep_id, node_distances, edge_hit_map)
            bundles.append(bundle)

        return bundles

    def _compute_episode_bundle(
        self,
        index: RelationshipIndex,
        ep_id: str,
        node_distances: dict[str, float],
        edge_hit_map: dict[str, float],
    ) -> EpisodeBundle:
        """
        Compute the minimum cost path to an Episode.

        Path options considered:
          1. Direct:      episode_direct + direct_episode_penalty
          2. Via Facet:   min over all facets: facet_cost + edge_cost + hop_cost
          3. Via Entity:  min over all entities: entity_direct + edge_cost + hop_cost
          4. Via Point:   min over all points: point_cost + 2*hop_cost + edge_cost

        Returns EpisodeBundle with the minimum cost across all paths.
        """
        INF = float("inf")

        def node_cost(nid: str) -> float:
            return float(node_distances.get(nid, INF))

        def edge_cost_fn(edge: Optional[SemanticEdge]) -> float:
            if edge is None:
                return self.cfg.edge_miss_cost
            key = edge.get_edge_key()
            return float(edge_hit_map.get(key, self.cfg.edge_miss_cost))

        direct = node_cost(ep_id)

        # ── Option 1: Direct hit (with penalty per Break #3) ──
        best_cost = direct + self.cfg.direct_episode_penalty
        best_path = "direct"
        best_support = ep_id

        # ── Option 2: Via Facet ──
        facet_ids = list(index.facets_by_episode.get(ep_id, []))[: self.cfg.max_facets_per_episode]
        for facet_id in facet_ids:
            facet_direct = node_cost(facet_id)
            # Find the edge: ep → facet
            ep_facet_edge = index.ep_facet_edge.get((ep_id, facet_id))

            # Effective edge/hop cost: discounted if facet direct score is low
            ec = edge_cost_fn(ep_facet_edge)
            hc = self.cfg.hop_cost
            if facet_direct < self.cfg.facet_discount_thresh_1:
                ec = ec * self.cfg.facet_discount_factor_1
                hc = hc * self.cfg.facet_discount_factor_1
            elif facet_direct < self.cfg.facet_discount_thresh_2:
                ec = ec * self.cfg.facet_discount_factor_2
                hc = hc * self.cfg.facet_discount_factor_2

            via_facet_cost = facet_direct + ec + hc
            if via_facet_cost < best_cost:
                best_cost = via_facet_cost
                best_path = "facet"
                best_support = facet_id

        # ── Option 3: Via Point (Episode → Facet → Point) ──
        for facet_id in facet_ids:
            point_ids = list(index.points_by_facet.get(facet_id, []))[: self.cfg.max_points_per_facet]
            for point_id in point_ids:
                point_direct = node_cost(point_id)
                facet_point_edge = index.facet_point_edge.get((facet_id, point_id))
                ep_facet_edge = index.ep_facet_edge.get((ep_id, facet_id))

                ec_fp = edge_cost_fn(facet_point_edge)
                ec_ep = edge_cost_fn(ep_facet_edge)
                # Two hops: ep→facet, facet→point
                via_point_cost = point_direct + ec_ep + ec_fp + 2 * self.cfg.hop_cost

                if via_point_cost < best_cost:
                    best_cost = via_point_cost
                    best_path = "point"
                    best_support = point_id

        # ── Option 4: Via Entity direct (Episode → Entity) ──
        entity_ids = list(index.entities_by_episode.get(ep_id, []))
        for entity_id in entity_ids:
            entity_direct = node_cost(entity_id)
            ep_entity_edge = index.ep_entity_edge.get((ep_id, entity_id))

            ec = edge_cost_fn(ep_entity_edge)
            via_entity_cost = entity_direct + ec + self.cfg.hop_cost
            if via_entity_cost < best_cost:
                best_cost = via_entity_cost
                best_path = "entity"
                best_support = entity_id

        # ── Option 5: Via Facet→Entity (Episode → Facet → Entity) ──
        for facet_id in facet_ids:
            facet_entity_ids = list(index.entities_by_facet.get(facet_id, []))
            for entity_id in facet_entity_ids:
                entity_direct = node_cost(entity_id)
                ep_facet_edge = index.ep_facet_edge.get((ep_id, facet_id))
                facet_entity_edge = index.facet_entity_edge.get((facet_id, entity_id))

                ec_ep = edge_cost_fn(ep_facet_edge)
                ec_fe = edge_cost_fn(facet_entity_edge)
                via_facet_entity_cost = entity_direct + ec_ep + ec_fe + 2 * self.cfg.hop_cost

                if via_facet_entity_cost < best_cost:
                    best_cost = via_facet_entity_cost
                    best_path = "facet_entity"
                    best_support = entity_id

        return EpisodeBundle(
            episode_id=ep_id,
            score=best_cost,
            best_path=best_path,
            best_support_id=best_support,
        )


# ============================================================================
# Vector Search — Phase 1 of Bundle Search
# ============================================================================


class VectorSearcher:
    """
    Phase 1 of Bundle Search: multi-collection vector search.

    Queries all 4 node-type collections in parallel and returns
    (node_distances, edge_distances) tuples.

    Embedding function is injected — can be OpenAI, local model, etc.
    """

    def __init__(
        self,
        embed_func: Callable[[list[str]], np.ndarray],
        collections: Optional[dict[NodeType, Any]] = None,
        top_k_per_collection: int = 100,
    ):
        """
        Args:
            embed_func   -- function that takes list[str] and returns (N, dim) embeddings
            collections  -- dict mapping NodeType → vector store (LanceDB, Chroma, etc.)
            top_k_per_collection -- candidates per collection
        """
        self.embed_func = embed_func
        self.collections = collections or {}
        self.top_k = top_k_per_collection

    async def search(
        self,
        query: str,
        node_map: dict[str, GraphNode],
        enable_adaptive: bool = False,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Search all collections and return node + edge distances.

        Args:
            query    -- user query string
            node_map -- {node_id: GraphNode} for text lookup
            enable_adaptive -- compute collection statistics before scoring

        Returns:
            (node_distances, edge_distances)
            node_distances: {node_id: cosine_distance}
            edge_distances: {edge_text: cosine_distance}
        """
        # Embed query
        query_emb = self.embed_func([query])[0]  # (dim,)

        node_distances: dict[str, float] = {}
        edge_distances: dict[str, float] = {}

        # Search each node-type collection in parallel
        import asyncio

        async def search_collection(ntype: NodeType, store: Any) -> dict[str, float]:
            if store is None:
                return {}
            try:
                results = await store.search(query_emb, top_k=self.top_k)
                return {r["id"]: float(r["distance"]) for r in results}
            except Exception:
                return {}

        # For now, synchronous fallback if collections are simple in-memory
        for ntype, store in self.collections.items():
            if store is None:
                continue
            try:
                results = store.search(query_emb, top_k=self.top_k)
                for r in results:
                    node_distances[r["id"]] = float(r["distance"])
            except (TypeError, AttributeError):
                # In-memory fallback: compute cosine manually
                for node_id, node in node_map.items():
                    if not node.embedding is not None:
                        continue
                    if ntype == NodeType.EPISODE and node.node_type != NodeType.EPISODE:
                        continue
                    dist = float(1 - np.dot(query_emb, node.embedding) / (np.linalg.norm(query_emb) * np.linalg.norm(node.embedding) + 1e-8))
                    node_distances[node_id] = min(node_distances.get(node_id, float("inf")), dist)

        # Search edge text collection if available
        edge_store = self.collections.get("edge")
        if edge_store is not None:
            try:
                results = edge_store.search(query_emb, top_k=self.top_k)
                for r in results:
                    edge_distances[r["id"]] = float(r["distance"])
            except (TypeError, AttributeError):
                pass

        return node_distances, edge_distances


# ============================================================================
# Query Preprocessor — extracts time, language, hybrid flags
# ============================================================================


@dataclass
class QueryAnalysis:
    """Structured analysis of a user query."""

    original: str
    vector_query: str
    has_time: bool
    time_text: Optional[str] = None
    is_hybrid: bool = False
    language: str = "en"


_EXPLICIT_DATE_PATTERN = re.compile(
    r"""
    \b\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}\b          # ISO: 2023-05-07
    |\b\d{4}年\d{1,2}月\d{1,2}[日号]?\b           # Chinese: 2023年5月7日
    |\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{4}\b  # English MDY
    |\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s*\d{4}\b   # English DMY
    """,
    re.VERBOSE | re.IGNORECASE,
)


class QueryPreprocessor:
    """
    Phase 1.5 of Bundle Search: query analysis.

    Extracts:
    - Vector query text (stripped of time expressions)
    - Whether query has explicit time
    - Time text for temporal filtering
    - Language detection
    """

    def analyze(self, query: str) -> QueryAnalysis:
        time_match = _EXPLICIT_DATE_PATTERN.search(query)
        has_time = time_match is not None
        time_text = time_match.group() if has_time else None

        # Strip time expressions for cleaner vector query
        vector_query = _EXPLICIT_DATE_PATTERN.sub("", query).strip()

        # Simple language detection
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", query))
        language = "zh" if has_chinese else "en"

        return QueryAnalysis(
            original=query,
            vector_query=vector_query,
            has_time=has_time,
            time_text=time_text,
            is_hybrid=has_time,  # time query = hybrid search
            language=language,
        )


# ============================================================================
# Graph Projection — Phase 2 of Bundle Search
# ============================================================================


class GraphProjector:
    """
    Phase 2 of Bundle Search: two-phase graph projection.

    Phase 1: Start from relevant_ids (anchor nodes from vector search)
    Phase 2: Expand one hop by node-type priority: Episode → Facet → FacetPoint → Entity

    This transforms isolated vector hits into a connected topological structure.
    """

    def __init__(self, max_relevant_ids: int = 100):
        self.max_relevant_ids = max_relevant_ids

    def project(
        self,
        relevant_ids: list[str],
        index: RelationshipIndex,
        node_map: dict[str, GraphNode],
    ) -> set[str]:
        """
        Expand anchor nodes into a subgraph.

        Args:
            relevant_ids -- node IDs from vector search to use as anchors
            index        -- RelationshipIndex
            node_map     -- {node_id: GraphNode}

        Returns:
            Set of all node IDs in the projected subgraph
        """
        # Phase 1: direct relevant nodes
        projected: set[str] = set(relevant_ids[: self.max_relevant_ids])

        # Phase 2: expand one hop by type priority (lowest number first)
        # Priority: Episode(0) → Facet(1) → FacetPoint(2) → Entity(3)
        def expand(ids: set) -> set:
            result: set = set()
            for nid in ids:
                ntype = node_map.get(nid)
                if ntype is None:
                    continue
                # Add neighbors based on edge types
                if ntype.node_type == NodeType.EPISODE:
                    result.update(index.facets_by_episode.get(nid, []))
                    result.update(index.entities_by_episode.get(nid, []))
                elif ntype.node_type == NodeType.FACET:
                    result.update(index.points_by_facet.get(nid, []))
                    result.update(index.entities_by_facet.get(nid, []))
            return result

        # One additional hop of expansion
        expanded = expand(projected)
        projected.update(expanded)

        # Limit total
        return set(list(projected)[: self.max_relevant_ids * 2])


# ============================================================================
# Exact Match Bonuses — Phase 3 of Bundle Search
# ============================================================================


@dataclass
class MatchBonusConfig:
    """Bonus multipliers for exact/keyword/number/English matches."""

    exact_bonus: float = 0.0     # exact string match (large reduction in cost)
    keyword_bonus: float = 0.0  # keyword overlap bonus
    number_bonus: float = 0.0    # numeric match bonus
    english_bonus: float = 0.0   # English text match bonus

    # Bonus application mode: "additive" (score += bonus) or "multiplicative" (score *= multiplier)
    mode: str = "multiplicative"

    # Minimum score to apply bonuses (prevents low-quality nodes getting bonuses)
    min_score_thresh: float = 0.5


class ExactMatchBonus:
    """
    Phase 3 of Bundle Search: apply match bonuses before scoring.

    After vector search, apply bonuses for:
    - Exact string match (query token == node/edge text)
    - Keyword overlap (shared tokens)
    - Number match (query has "500ms", node has "500")
    - English text match (non-Chinese queries matching English content)
    """

    def __init__(self, config: Optional[MatchBonusConfig] = None):
        self.cfg = config or MatchBonusConfig()

    def apply(
        self,
        node_distances: dict[str, float],
        node_map: dict[str, GraphNode],
        query: str,
        analysis: QueryAnalysis,
    ) -> dict[str, float]:
        """
        Apply bonuses to node distances.

        Returns new distances dict (original is not mutated).
        """
        import re

        query_tokens = set(re.findall(r"\w+", query.lower()))
        query_numbers = set(re.findall(r"\d+", query))
        is_chinese_query = analysis.language == "zh"

        bonuses: dict[str, float] = {}

        for node_id, dist in node_distances.items():
            if dist > self.cfg.min_score_thresh:
                continue

            node = node_map.get(node_id)
            if node is None:
                continue

            bonus = 0.0
            node_text = node.get_text_for_embedding().lower()
            node_tokens = set(re.findall(r"\w+", node_text))
            node_numbers = set(re.findall(r"\d+", node_text))

            # Exact match
            if query.lower() in node_text:
                bonus += self.cfg.exact_bonus

            # Keyword overlap
            overlap = len(query_tokens & node_tokens)
            if overlap > 0:
                bonus += self.cfg.keyword_bonus * overlap

            # Number match
            if query_numbers & node_numbers:
                bonus += self.cfg.number_bonus

            # English match: non-Chinese query matching English content
            if not is_chinese_query and not re.search(r"[\u4e00-\u9fff]", node_text):
                if node_tokens:
                    bonus += self.cfg.english_bonus

            if bonus != 0.0:
                if self.cfg.mode == "additive":
                    bonuses[node_id] = bonus
                else:
                    bonuses[node_id] = 1.0 + bonus

        # Apply bonuses
        result = dict(node_distances)
        for node_id, bonus in bonuses.items():
            if self.cfg.mode == "additive":
                result[node_id] = max(0.0, result.get(node_id, 1.0) - bonus)
            else:
                result[node_id] = result.get(node_id, 1.0) * (2.0 - min(bonus, 1.0))

        return result


# ============================================================================
# Time Bonus — Phase 9 of Bundle Search
# ============================================================================


@dataclass
class TimeBonusConfig:
    """Time-based score adjustments."""

    mentioned_time_weight: float = 0.2   # weight for event time matching
    created_at_weight: float = 0.1       # weight for ingestion time matching
    mismatch_penalty_max: float = 0.3     # max penalty when time definitely mismatches
    enabled: bool = True


class TimeBonus:
    """
    Phase 9 of Bundle Search: temporal scoring.

    If query has explicit time, apply time-based bonus/penalty:
    - Time match → score reduction (lower is better)
    - Time mismatch → score penalty
    """

    def apply(
        self,
        bundles: list[EpisodeBundle],
        node_map: dict[str, GraphNode],
        analysis: QueryAnalysis,
        config: Optional[TimeBonusConfig] = None,
    ) -> list[EpisodeBundle]:
        if not (config and config.enabled) or not analysis.has_time:
            return bundles

        cfg = config or TimeBonusConfig()
        query_time_lower = analysis.time_text.lower() if analysis.time_text else ""

        result: list[EpisodeBundle] = []
        for bundle in bundles:
            node = node_map.get(bundle.episode_id)
            if node is None:
                result.append(bundle)
                continue

            penalty = 0.0

            # Check mentioned_time attribute
            mentioned = node.attributes.get("mentioned_time_text", "")
            if mentioned and query_time_lower in mentioned.lower():
                penalty -= cfg.mentioned_time_weight
            elif mentioned:
                penalty += cfg.mismatch_penalty_max

            # Check created_at (ingestion time)
            created = node.attributes.get("created_at", "")
            if created and query_time_lower in str(created).lower():
                penalty -= cfg.created_at_weight

            new_bundle = EpisodeBundle(
                episode_id=bundle.episode_id,
                score=max(0.0, bundle.score + penalty),
                best_path=bundle.best_path,
                best_support_id=bundle.best_support_id,
            )
            result.append(new_bundle)

        return result


# ============================================================================
# Direct Hit Penalty — Break #3
# ============================================================================


class DirectHitPenalty:
    """
    Break #3: Penalize direct Episode summary hits.

    Episode summaries are high-level generalizations — they "look relevant"
    to many queries. We apply an extra penalty when the query directly hits
    an Episode summary to prevent generic summaries from winning.
    """

    def __init__(self, penalty: float = 0.15):
        self.penalty = penalty

    def apply(
        self,
        bundles: list[EpisodeBundle],
        node_distances: dict[str, float],
        node_map: dict[str, GraphNode],
    ) -> list[EpisodeBundle]:
        """
        Apply direct hit penalty to bundles whose Episode node was directly hit.

        Args:
            bundles        -- EpisodeBundle list from BundleScorer
            node_distances -- vector distances per node
            node_map       -- {node_id: GraphNode}

        Returns:
            Bundles with updated scores
        """
        result: list[EpisodeBundle] = []
        for bundle in bundles:
            ep_dist = node_distances.get(bundle.episode_id, 1.0)
            # If Episode had a direct hit (low distance), apply penalty
            if ep_dist < 0.2:
                new_bundle = EpisodeBundle(
                    episode_id=bundle.episode_id,
                    score=bundle.score + self.penalty,
                    best_path=bundle.best_path,
                    best_support_id=bundle.best_support_id,
                )
                result.append(new_bundle)
            else:
                result.append(bundle)
        return result


# ============================================================================
# Episodic Bundle Search — Main 11-Step Pipeline
# ============================================================================


@dataclass
class EpisodicBundleSearchConfig:
    """Top-level configuration for episodic bundle search."""

    # Vector search
    top_k: int = 5
    wide_search_top_k: int = 100
    collections: dict = field(default_factory=dict)  # NodeType → vector store

    # Graph projection
    max_relevant_ids: int = 100

    # Scoring
    edge_miss_cost: float = 0.5
    hop_cost: float = 0.1
    direct_episode_penalty: float = 0.15

    # Time
    enable_time_bonus: bool = True
    time_conf_min: float = 0.7

    # Adaptive weights
    enable_adaptive_weights: bool = False

    # Display mode
    display_mode: str = "summary"  # "summary" | "detail" | "highly_related_summary"


class EpisodicBundleSearch:
    """
    Main entry point for episodic bundle search.

    Orchestrates the 11-step M-flow retrieval pipeline:

    1.  Query Preprocessing — time parsing, language detection
    2.  Vector Search — multi-collection search with time enhancement
    2.5 Adaptive Scoring Context — compute collection stats
    3.  Apply Match Bonuses — exact/keyword/number/English matches
    4.  Two-Phase Projection — graph projection with neighbor expansion
    5.  Write Back Node Distances — store computed distances
    6.  Edge Distance Mapping — map vector distances to graph edges
    7.  Build Relationship Index — index episodes, facets, points, entities
    8.  Build Edge Hit Map — map edge distances
    9.  Bundle Scoring — compute episode bundles with optional time bonus
    10. Sort & Take Top-K — heapq.nsmallest
    11. Assemble Output — final edge assembly with time sorting
    """

    def __init__(
        self,
        config: Optional[EpisodicBundleSearchConfig] = None,
        embed_func: Optional[Callable[[list[str]], np.ndarray]] = None,
    ):
        self.cfg = config or EpisodicBundleSearchConfig()
        self.embed_func = embed_func or (lambda x: np.random.randn(len(x), 128).astype(np.float32))

        self.preprocessor = QueryPreprocessor()
        self.projector = GraphProjector(max_relevant_ids=self.cfg.max_relevant_ids)
        self.bonus_adder = ExactMatchBonus()
        self.scorer = BundleScorer(BundleScorerConfig(
            edge_miss_cost=self.cfg.edge_miss_cost,
            hop_cost=self.cfg.hop_cost,
            direct_episode_penalty=self.cfg.direct_episode_penalty,
        ))
        self.time_bonus = TimeBonus()
        self.direct_penalty = DirectHitPenalty(penalty=self.cfg.direct_episode_penalty)

    async def search(
        self,
        query: str,
        nodes: list[GraphNode],
        edges: list[SemanticEdge],
        index: Optional[RelationshipIndex] = None,
    ) -> list[EpisodeBundle]:
        """
        Main bundle search pipeline.

        Args:
            query  -- user query string
            nodes  -- all GraphNodes in the memory graph
            edges  -- all SemanticEdges
            index  -- pre-built RelationshipIndex (built if not provided)

        Returns:
            List[EpisodeBundle] sorted by score (ascending = lower cost = better)
        """
        # Step 1: Query preprocessing
        analysis = self.preprocessor.analyze(query)

        # Build node map for fast lookup
        node_map: dict[str, GraphNode] = {n.node_id: n for n in nodes}

        # Step 2: Vector search across all collections
        vector_searcher = VectorSearcher(
            embed_func=self.embed_func,
            collections=self.cfg.collections,
            top_k_per_collection=self.cfg.wide_search_top_k,
        )
        node_distances, edge_distances = await vector_searcher.search(
            query, node_map, enable_adaptive=self.cfg.enable_adaptive_weights
        )

        # Step 3: Apply exact match bonuses
        node_distances = self.bonus_adder.apply(node_distances, node_map, query, analysis)

        # Step 4: Two-phase graph projection
        relevant_ids = sorted(node_distances, key=node_distances.get)  # type: ignore
        projected = self.projector.project(relevant_ids, index, node_map)

        # Filter distances to only projected nodes
        node_distances = {k: v for k, v in node_distances.items() if k in projected}

        # Step 5: Write back node distances (already done in step 4)
        # Step 6: Edge distance mapping (done in scorer)
        # Step 7: Build relationship index (use provided or build new)
        if index is None:
            index = RelationshipBuilder.build(nodes, edges)

        # Step 8: Build edge hit map from edge_distances
        edge_hit_map = dict(edge_distances)

        # Step 9: Bundle scoring
        bundles = self.scorer.compute_bundles(index, node_distances, edge_hit_map)

        # Apply time bonus if applicable
        if self.cfg.enable_time_bonus:
            bundles = self.time_bonus.apply(bundles, node_map, analysis)

        # Apply direct hit penalty (Break #3)
        bundles = self.direct_penalty.apply(bundles, node_distances, node_map)

        # Step 10: Sort and take top-K
        bundles = heapq.nsmallest(self.cfg.top_k, bundles)

        # Step 11: Sort by score (ascending) and return
        bundles.sort(key=lambda b: b.score)
        return bundles


# ============================================================================
# Memory Graph Elements — GraphNode + SemanticEdge factory
# ============================================================================


class MemoryGraphElements:
    """
    Factory for creating GraphNodes and SemanticEdges from raw data.

    This is the ingestion-side counterpart to EpisodicBundleSearch.
    Call this to build the graph from documents/conversations before searching.
    """

    @staticmethod
    def create_episode(
        name: str,
        summary: str,
        content: str = "",
        embedding: Optional[np.ndarray] = None,
        **attrs,
    ) -> GraphNode:
        return GraphNode(
            node_id=str(uuid.uuid4()),
            node_type=NodeType.EPISODE,
            name=name,
            description=summary,
            content=content or summary,
            embedding=embedding,
            search_text=summary,
            attributes=attrs,
        )

    @staticmethod
    def create_facet(
        name: str,
        description: str,
        embedding: Optional[np.ndarray] = None,
        **attrs,
    ) -> GraphNode:
        return GraphNode(
            node_id=str(uuid.uuid4()),
            node_type=NodeType.FACET,
            name=name,
            description=description,
            embedding=embedding,
            search_text=description,
            attributes=attrs,
        )

    @staticmethod
    def create_facet_point(
        name: str,
        description: str,
        embedding: Optional[np.ndarray] = None,
        **attrs,
    ) -> GraphNode:
        return GraphNode(
            node_id=str(uuid.uuid4()),
            node_type=NodeType.FACET_POINT,
            name=name,
            description=description,
            embedding=embedding,
            search_text=description,
            attributes=attrs,
        )

    @staticmethod
    def create_entity(
        name: str,
        entity_type: str = "",
        embedding: Optional[np.ndarray] = None,
        **attrs,
    ) -> GraphNode:
        return GraphNode(
            node_id=str(uuid.uuid4()),
            node_type=NodeType.ENTITY,
            name=name,
            description=name,
            embedding=embedding,
            search_text=name,
            attributes={"entity_type": entity_type, **attrs},
        )

    @staticmethod
    def create_edge(
        from_node: str,
        to_node: str,
        edge_text: str,
        relationship: str,
        embedding: Optional[np.ndarray] = None,
        **attrs,
    ) -> SemanticEdge:
        return SemanticEdge(
            edge_id=str(uuid.uuid4()),
            from_node=from_node,
            to_node=to_node,
            edge_text=edge_text,
            relationship=relationship,
            embedding=embedding,
            attributes=attrs,
        )


# ============================================================================
# Graph Bundle Memory — Integration with OpenMythos ThreeLayerMemorySystem
# ============================================================================


class GraphBundleMemory:
    """
    P4-1: M-flow Bundle Search integrated into OpenMythos memory.

    This replaces the flat vector retrieval in ThreeLayerMemorySystem
    with graph-led retrieval. The core change:

    OLD: query → vector search → cosine similarity → top-k chunks
    NEW: query → vector search (anchors) → graph projection
              → path-cost propagation → EpisodeBundle scores → top-k Episodes

    Usage:
        gbm = GraphBundleMemory(embed_func=my_embed)
        gbm.add_episode(name="bug report", summary="Maria reported a deadline issue", ...)
        gbm.add_facet(episode_id, name="communication", description="...", ...)
        results = await gbm.search("What happened with Maria's deadline?")
    """

    def __init__(
        self,
        embed_func: Optional[Callable[[list[str]], np.ndarray]] = None,
        config: Optional[EpisodicBundleSearchConfig] = None,
    ):
        self.embed_func = embed_func or (lambda x: np.random.randn(len(x), 128).astype(np.float32))
        self.cfg = config or EpisodicBundleSearchConfig()
        self.searcher = EpisodicBundleSearch(self.cfg, self.embed_func)

        # In-memory graph storage
        self.nodes: list[GraphNode] = []
        self.edges: list[SemanticEdge] = []
        self.node_map: dict[str, GraphNode] = {}
        self.index: Optional[RelationshipIndex] = None
        self._index_dirty = True

    def _rebuild_index(self):
        """Rebuild RelationshipIndex when graph changes."""
        self.index = RelationshipBuilder.build(self.nodes, self.edges)
        self._index_dirty = False

    def add_episode(
        self,
        name: str,
        summary: str,
        content: str = "",
        **attrs,
    ) -> GraphNode:
        """Add an Episode node."""
        emb = self.embed_func([summary])[0] if self.embed_func else None
        node = MemoryGraphElements.create_episode(name, summary, content, emb, **attrs)
        self.nodes.append(node)
        self.node_map[node.node_id] = node
        self._index_dirty = True
        return node

    def add_facet(
        self,
        episode_id: str,
        name: str,
        description: str,
        edge_text: Optional[str] = None,
        **attrs,
    ) -> GraphNode:
        """Add a Facet node and connect it to an Episode."""
        emb = self.embed_func([description])[0] if self.embed_func else None
        facet = MemoryGraphElements.create_facet(name, description, emb, **attrs)
        self.nodes.append(facet)
        self.node_map[facet.node_id] = facet

        edge = MemoryGraphElements.create_edge(
            from_node=episode_id,
            to_node=facet.node_id,
            edge_text=edge_text or description,
            relationship="has_facet",
        )
        self.edges.append(edge)
        self._index_dirty = True
        return facet

    def add_facet_point(
        self,
        facet_id: str,
        name: str,
        description: str,
        edge_text: Optional[str] = None,
        **attrs,
    ) -> GraphNode:
        """Add a FacetPoint node and connect it to a Facet."""
        emb = self.embed_func([description])[0] if self.embed_func else None
        point = MemoryGraphElements.create_facet_point(name, description, emb, **attrs)
        self.nodes.append(point)
        self.node_map[point.node_id] = point

        edge = MemoryGraphElements.create_edge(
            from_node=facet_id,
            to_node=point.node_id,
            edge_text=edge_text or description,
            relationship="has_point",
        )
        self.edges.append(edge)
        self._index_dirty = True
        return point

    def add_entity(
        self,
        name: str,
        entity_type: str = "",
        connected_to: Optional[str] = None,  # Episode or Facet ID
        relationship: str = "involves_entity",
        **attrs,
    ) -> GraphNode:
        """Add an Entity node and optionally connect it."""
        emb = self.embed_func([name])[0] if self.embed_func else None
        entity = MemoryGraphElements.create_entity(name, entity_type, emb, **attrs)
        self.nodes.append(entity)
        self.node_map[entity.node_id] = entity

        if connected_to:
            connected_node = self.node_map.get(connected_to)
            edge_text = f"{name} involves {connected_node.name if connected_node else connected_to}"
            edge = MemoryGraphElements.create_edge(
                from_node=connected_to,
                to_node=entity.node_id,
                edge_text=edge_text,
                relationship=relationship,
            )
            self.edges.append(edge)

        self._index_dirty = True
        return entity

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> list[tuple[GraphNode, EpisodeBundle]]:
        """
        Search the graph using M-flow Bundle Search.

        Args:
            query  -- user query string
            top_k  -- number of results (overrides config.top_k)

        Returns:
            List of (Episode node, EpisodeBundle) tuples sorted by score
        """
        if self._index_dirty:
            self._rebuild_index()

        cfg = EpisodicBundleSearchConfig(**vars(self.cfg))
        if top_k is not None:
            cfg.top_k = top_k

        searcher = EpisodicBundleSearch(cfg, self.embed_func)
        bundles = await searcher.search(query, self.nodes, self.edges, self.index)

        results = []
        for bundle in bundles:
            node = self.node_map.get(bundle.episode_id)
            if node:
                results.append((node, bundle))

        return results

    def get_context(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "summary",
    ) -> str:
        """
        Synchronous context retrieval for OpenMythos integration.

        Args:
            query  -- user query
            top_k  -- number of episodes
            mode   -- "summary" | "detail" | "highly_related_summary"

        Returns:
            Formatted context string for LLM prompt injection
        """
        import asyncio
        results = asyncio.run(self.search(query, top_k))

        if mode == "summary":
            lines = []
            for node, bundle in results:
                time_prefix = node.attributes.get("mentioned_time_text", "")
                if time_prefix:
                    lines.append(f"[{time_prefix}] {node.description}")
                else:
                    lines.append(node.description)
            return "\n".join(lines)

        elif mode == "detail":
            lines = []
            for node, bundle in results:
                lines.append(f"## {node.name}")
                lines.append(node.content or node.description)
                lines.append("")
            return "\n".join(lines)

        elif mode == "highly_related_summary":
            # Only include summary paragraphs related to matched Facet
            lines = []
            for node, bundle in results:
                related = node.attributes.get("edge_text", "")
                if related:
                    lines.append(f"[via {bundle.best_path}] {related}")
                else:
                    lines.append(node.description)
            return "\n".join(lines)

        return ""

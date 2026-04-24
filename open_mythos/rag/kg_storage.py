"""
Phase 1: 知识图谱存储接口
==========================

定义 KG 存储的抽象接口，支持多种后端:
- Neo4j (原生图查询)
- PostgreSQL + pgvector (统一 SQL)
- MongoDB (灵活文档)
- OpenSearch (大规模 + LightRAG 原生)

接口设计:
- VectorStoreInterface: 向量存储
- GraphStoreInterface: 图结构存储
- StorageBackend: 统一存储后端
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import numpy as np


# ============================================================================
# Entity & Edge Data Structures
# ============================================================================


class EntityType(str, Enum):
    """实体类型"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    DOCUMENT = "document"
    PAGE = "page"
    CUSTOM = "custom"


class EdgeType(str, Enum):
    """边类型"""
    SEMANTIC_SIM = "semantic_sim"      # 语义相似
    CROSS_MODAL = "cross_modal"        # 跨模态关联
    BELONGS_TO = "belongs_to"           # 属于 (元素 → 页 → 文档)
    COREFERENCE = "coreference"        # 指代关联
    CAUSAL = "causal"                  # 因果关系
    PARALLEL = "parallel"              # 并列关系
    DERIVES_FROM = "derives_from"      # 派生关系


@dataclass
class Entity:
    """
    实体节点。

    Attributes:
        id: 唯一标识符
        type: 实体类型
        content: 内容 (文本或描述)
        embedding: 向量表示
        metadata: 额外元数据
    """
    id: str
    type: EntityType
    content: str = ""
    embedding: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "embedding": self.embedding.tolist() if self.embedding is not None else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        emb = data.get("embedding")
        if emb is not None:
            emb = np.array(emb)
        return cls(
            id=data["id"],
            type=EntityType(data["type"]),
            content=data.get("content", ""),
            embedding=emb,
            metadata=data.get("metadata", {}),
        )


@dataclass
class Edge:
    """
    图边（关系）。

    Attributes:
        source_id: 源实体 ID
        target_id: 目标实体 ID
        type: 边类型
        weight: 关系权重 (0-1)
        metadata: 额外元数据
    """
    source_id: str
    target_id: str
    type: EdgeType
    weight: float = 1.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
            "weight": self.weight,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Edge":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            type=EdgeType(data["type"]),
            weight=data.get("weight", 1.0),
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# Vector Store Interface
# ============================================================================


class VectorStoreInterface(ABC):
    """向量存储抽象接口"""

    @abstractmethod
    def upsert(self, entities: dict[str, np.ndarray]) -> None:
        """
        批量插入或更新向量。

        Args:
            entities: {entity_id: embedding}
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """
        向量相似度搜索。

        Args:
            query_embedding: 查询向量
            top_k: 返回数量
            filters: 过滤条件 (如 {"type": "text"})

        Returns:
            [{"id": str, "score": float, "embedding": np.ndarray}, ...]
        """
        pass

    @abstractmethod
    def delete(self, entity_ids: list[str]) -> None:
        """删除实体"""
        pass

    @abstractmethod
    def get(self, entity_ids: list[str]) -> dict[str, np.ndarray]:
        """批量获取向量"""
        pass


# ============================================================================
# Graph Store Interface
# ============================================================================


class GraphStoreInterface(ABC):
    """图结构存储抽象接口"""

    @abstractmethod
    def upsert_node(self, node_id: str, entity: Entity) -> None:
        """插入或更新节点"""
        pass

    @abstractmethod
    def upsert_edge(self, edge: Edge) -> None:
        """插入或更新边"""
        pass

    @abstractmethod
    def get_node(self, node_id: str) -> Optional[Entity]:
        """获取节点"""
        pass

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[list[EdgeType]] = None,
        direction: str = "both",  # out | in | both
    ) -> list[tuple[str, Edge]]:
        """
        获取邻居节点。

        Returns:
            [(neighbor_id, edge), ...]
        """
        pass

    @abstractmethod
    def delete_node(self, node_id: str) -> None:
        """删除节点及其所有边"""
        pass

    @abstractmethod
    def query(
        self,
        cypher: str,
        params: Optional[dict] = None,
    ) -> list[dict]:
        """
        原生图查询 (如 Cypher for Neo4j)。

        Returns:
            查询结果列表
        """
        pass


# ============================================================================
# Concrete Implementations
# ============================================================================


class Neo4jVectorStore(VectorStoreInterface):
    """
    Neo4j 向量存储实现。

    使用 Neo4j 的向量索引特性存储和检索实体向量。
    需要 neo4j >= 5.18 with vector index support.

    安装: pip install neo4j
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        index_name: str = "entity_vectors",
        dimensions: int = 1536,
    ):
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("neo4j package required. Install with: pip install neo4j")

        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self.index_name = index_name
        self.dimensions = dimensions

    def upsert(self, entities: dict[str, np.ndarray]) -> None:
        import json

        with self._driver.session() as session:
            for entity_id, embedding in entities.items():
                session.run(
                    f"""
                    MERGE (e:Entity {{id: $id}})
                    SET e.embedding = $embedding,
                        e.updated_at = timestamp()
                    """,
                    id=entity_id,
                    embedding=embedding.tolist(),
                )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        with self._driver.session() as session:
            # Neo4j vector search using db.index.vector.query
            result = session.run(
                f"""
                MATCH (e:Entity)
                WHERE e.embedding IS NOT NULL
                {"AND e.type = $type" if filters and filters.get("type") else ""}
                WITH e, vector.similarity.cosine(e.embedding, $query) AS score
                ORDER BY score DESC
                LIMIT $top_k
                RETURN e.id AS id, score, e.embedding AS embedding
                """,
                query=query_embedding.tolist(),
                top_k=top_k,
                type=filters.get("type") if filters else None,
            )

            return [
                {
                    "id": record["id"],
                    "score": record["score"],
                    "embedding": np.array(record["embedding"]),
                }
                for record in result
            ]

    def delete(self, entity_ids: list[str]) -> None:
        with self._driver.session() as session:
            for eid in entity_ids:
                session.run("MATCH (e:Entity {id: $id}) DETACH DELETE e", id=eid)

    def get(self, entity_ids: list[str]) -> dict[str, np.ndarray]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity) WHERE e.id IN $ids RETURN e.id AS id, e.embedding AS embedding",
                ids=entity_ids,
            )
            return {
                record["id"]: np.array(record["embedding"])
                for record in result
                if record["embedding"]
            }


class Neo4jGraphStore(GraphStoreInterface):
    """
    Neo4j 图存储实现。

    使用 Neo4j 存储图结构，支持 Cypher 查询。
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
    ):
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("neo4j package required. Install with: pip install neo4j")

        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    def upsert_node(self, node_id: str, entity: Entity) -> None:
        import json

        with self._driver.session() as session:
            session.run(
                """
                MERGE (e:Entity {id: $id})
                SET e.type = $type,
                    e.content = $content,
                    e.metadata = $metadata
                """,
                id=node_id,
                type=entity.type.value,
                content=entity.content,
                metadata=json.dumps(entity.metadata),
            )

    def upsert_edge(self, edge: Edge) -> None:
        with self._driver.session() as session:
            session.run(
                """
                MATCH (s:Entity {id: $source_id})
                MATCH (t:Entity {id: $target_id})
                MERGE (s)-[r:RELATES {type: $type}]->(t)
                SET r.weight = $weight,
                    r.metadata = $metadata
                """,
                source_id=edge.source_id,
                target_id=edge.target_id,
                type=edge.type.value,
                weight=edge.weight,
                metadata=json.dumps(edge.metadata),
            )

    def get_node(self, node_id: str) -> Optional[Entity]:
        with self._driver.session() as session:
            result = session.run(
                "MATCH (e:Entity {id: $id}) RETURN e",
                id=node_id,
            )
            for record in result:
                e = record["e"]
                return Entity(
                    id=e["id"],
                    type=EntityType(e["type"]),
                    content=e.get("content", ""),
                    metadata=json.loads(e.get("metadata", "{}")),
                )
        return None

    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[list[EdgeType]] = None,
        direction: str = "both",
    ) -> list[tuple[str, Edge]]:
        with self._driver.session() as session:
            if direction == "out":
                query = "MATCH (s:Entity {id: $id})-[r]->(t:Entity)"
                types_cond = "AND r.type IN $types" if edge_types else ""
            elif direction == "in":
                query = "MATCH (s:Entity)-[r]->(t:Entity {id: $id})"
                types_cond = "AND r.type IN $types" if edge_types else ""
            else:
                query = "MATCH (s:Entity {id: $id})-[r]-(t:Entity)"
                types_cond = "AND r.type IN $types" if edge_types else ""

            result = session.run(
                f"""
                {query}
                {types_cond}
                RETURN t.id AS target_id, r.type AS rel_type, r.weight AS weight, r.metadata AS metadata
                """,
                id=node_id,
                types=[et.value for et in edge_types] if edge_types else None,
            )

            neighbors = []
            for record in result:
                edge = Edge(
                    source_id=node_id,
                    target_id=record["target_id"],
                    type=EdgeType(record["rel_type"]),
                    weight=record.get("weight", 1.0),
                    metadata=json.loads(record.get("metadata", "{}")),
                )
                neighbors.append((record["target_id"], edge))

            return neighbors

    def delete_node(self, node_id: str) -> None:
        with self._driver.session() as session:
            session.run(
                "MATCH (e:Entity {id: $id}) DETACH DELETE e",
                id=node_id,
            )

    def query(self, cypher: str, params: Optional[dict] = None) -> list[dict]:
        with self._driver.session() as session:
            result = session.run(cypher, **(params or {}))
            return [dict(record) for record in result]


class PostgreSQLVectorStore(VectorStoreInterface):
    """
    PostgreSQL + pgvector 向量存储。

    使用 pgvector 扩展存储和检索向量。
    需要: PostgreSQL with pgvector extension enabled.

    安装: pip install psycopg2-binary pgvector
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "vectors",
        user: str = "postgres",
        password: str = "password",
        table_name: str = "entities",
        dimensions: int = 1536,
    ):
        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise ImportError("psycopg2 required. Install with: pip install psycopg2-binary")

        self.conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password
        )
        self.table_name = table_name
        self.dimensions = dimensions

        # 创建表
        with self.conn.cursor() as cur:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id TEXT PRIMARY KEY,
                    embedding vector({dimensions}),
                    content TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW()
                )
            """)
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {table_name}_embedding_idx
                ON {table_name} USING ivfflat (embedding vector_cosine_ops)
            """)
            self.conn.commit()

    def upsert(self, entities: dict[str, np.ndarray]) -> None:
        import json

        with self.conn.cursor() as cur:
            for entity_id, embedding in entities.items():
                cur.execute(f"""
                    INSERT INTO {self.table_name} (id, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (id) DO UPDATE SET embedding = EXCLUDED.embedding
                """, (entity_id, embedding.tolist()))

            self.conn.commit()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        with self.conn.cursor() as cur:
            type_cond = ""
            if filters and filters.get("type"):
                type_cond = f"AND metadata->>'type' = '{filters['type']}'"

            cur.execute(f"""
                SELECT id, 1 - (embedding <=> %s) AS score
                FROM {self.table_name}
                WHERE embedding IS NOT NULL
                {type_cond}
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_embedding.tolist(), query_embedding.tolist(), top_k))

            return [
                {"id": row[0], "score": float(row[1])}
                for row in cur.fetchall()
            ]

    def delete(self, entity_ids: list[str]) -> None:
        with self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table_name} WHERE id = ANY(%s)", (entity_ids,))
            self.conn.commit()

    def get(self, entity_ids: list[str]) -> dict[str, np.ndarray]:
        import numpy as np

        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT id, embedding FROM {self.table_name} WHERE id = ANY(%s)",
                (entity_ids,)
            )
            return {
                row[0]: np.array(row[1])
                for row in cur.fetchall()
                if row[1]
            }


class PostgreSQLGraphStore(GraphStoreInterface):
    """
    PostgreSQL 图存储实现 (使用邻接表)。

    使用关系表存储图结构，适合中小规模图。
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        dbname: str = "vectors",
        user: str = "postgres",
        password: str = "password",
    ):
        try:
            import psycopg2
        except ImportError:
            raise ImportError("psycopg2 required. Install with: pip install psycopg2-binary")

        self.conn = psycopg2.connect(
            host=host, port=port, dbname=dbname, user=user, password=password
        )

        # 创建表
        with self.conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kg_nodes (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT,
                    metadata JSONB DEFAULT '{}'
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS kg_edges (
                    id SERIAL PRIMARY KEY,
                    source_id TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
                    target_id TEXT NOT NULL REFERENCES kg_nodes(id) ON DELETE CASCADE,
                    type TEXT NOT NULL,
                    weight FLOAT DEFAULT 1.0,
                    metadata JSONB DEFAULT '{}',
                    UNIQUE(source_id, target_id, type)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS kg_edges_source_idx ON kg_edges(source_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS kg_edges_target_idx ON kg_edges(target_id)
            """)
            self.conn.commit()

    def upsert_node(self, node_id: str, entity: Entity) -> None:
        import json

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO kg_nodes (id, type, content, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    type = EXCLUDED.type,
                    content = EXCLUDED.content,
                    metadata = EXCLUDED.metadata
            """, (node_id, entity.type.value, entity.content, json.dumps(entity.metadata)))
            self.conn.commit()

    def upsert_edge(self, edge: Edge) -> None:
        import json

        with self.conn.cursor() as cur:
            cur.execute("""
                INSERT INTO kg_edges (source_id, target_id, type, weight, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source_id, target_id, type) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    metadata = EXCLUDED.metadata
            """, (
                edge.source_id, edge.target_id,
                edge.type.value, edge.weight,
                json.dumps(edge.metadata)
            ))
            self.conn.commit()

    def get_node(self, node_id: str) -> Optional[Entity]:
        import json

        with self.conn.cursor() as cur:
            cur.execute("SELECT id, type, content, metadata FROM kg_nodes WHERE id = %s", (node_id,))
            row = cur.fetchone()
            if row:
                return Entity(
                    id=row[0],
                    type=EntityType(row[1]),
                    content=row[2] or "",
                    metadata=json.loads(row[3]) if row[3] else {},
                )
        return None

    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[list[EdgeType]] = None,
        direction: str = "both",
    ) -> list[tuple[str, Edge]]:
        import json

        with self.conn.cursor() as cur:
            if direction == "out":
                cur.execute("""
                    SELECT t.id, e.type, e.weight, e.metadata
                    FROM kg_edges e
                    JOIN kg_nodes t ON t.id = e.target_id
                    WHERE e.source_id = %s
                    %s
                """, (node_id, "AND e.type = ANY(%s)" if edge_types else "",
                      [et.value for et in edge_types] if edge_types else None))
            elif direction == "in":
                cur.execute("""
                    SELECT s.id, e.type, e.weight, e.metadata
                    FROM kg_edges e
                    JOIN kg_nodes s ON s.id = e.source_id
                    WHERE e.target_id = %s
                    %s
                """, (node_id, "AND e.type = ANY(%s)" if edge_types else "",
                      [et.value for et in edge_types] if edge_types else None))
            else:
                cur.execute("""
                    SELECT neighbor_id, e.type, e.weight, e.metadata, direction
                    FROM (
                        SELECT target_id AS neighbor_id, type, weight, metadata, 'out' AS direction
                        FROM kg_edges WHERE source_id = %s
                        UNION ALL
                        SELECT source_id AS neighbor_id, type, weight, metadata, 'in' AS direction
                        FROM kg_edges WHERE target_id = %s
                    ) sub
                    WHERE 1=1
                    %s
                """, (node_id, node_id,
                      "AND type = ANY(%s)" if edge_types else "",
                      [et.value for et in edge_types] if edge_types else ()))

            return [
                (
                    row[0],
                    Edge(
                        source_id=node_id,
                        target_id=row[0],
                        type=EdgeType(row[1]),
                        weight=row[2],
                        metadata=json.loads(row[3]) if row[3] else {},
                    )
                )
                for row in cur.fetchall()
            ]

    def delete_node(self, node_id: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute("DELETE FROM kg_nodes WHERE id = %s", (node_id,))
            self.conn.commit()

    def query(self, sql: str, params: Optional[dict] = None) -> list[dict]:
        with self.conn.cursor() as cur:
            cur.execute(sql, params or {})
            return [dict(row) for row in cur.fetchall()]


class InMemoryVectorStore(VectorStoreInterface):
    """
    内存向量存储 (用于测试或小规模场景)。

    使用简单的余弦相似度搜索。
    """

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions
        self._vectors: dict[str, np.ndarray] = {}
        self._metadata: dict[str, dict] = {}

    def upsert(self, entities: dict[str, np.ndarray]) -> None:
        for entity_id, embedding in entities.items():
            self._vectors[entity_id] = embedding

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        if not self._vectors:
            return []

        scores = []
        for entity_id, embedding in self._vectors.items():
            if filters:
                meta = self._metadata.get(entity_id, {})
                if not all(meta.get(k) == v for k, v in filters.items()):
                    continue

            # Cosine similarity
            norm_q = np.linalg.norm(query_embedding)
            norm_e = np.linalg.norm(embedding)
            if norm_q > 0 and norm_e > 0:
                score = np.dot(query_embedding, embedding) / (norm_q * norm_e)
            else:
                score = 0.0

            scores.append((entity_id, float(score)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"id": eid, "score": score} for eid, score in scores[:top_k]]

    def delete(self, entity_ids: list[str]) -> None:
        for eid in entity_ids:
            self._vectors.pop(eid, None)
            self._metadata.pop(eid, None)

    def get(self, entity_ids: list[str]) -> dict[str, np.ndarray]:
        return {eid: self._vectors[eid] for eid in entity_ids if eid in self._vectors}

    def set_metadata(self, entity_id: str, metadata: dict) -> None:
        self._metadata[entity_id] = metadata


class InMemoryGraphStore(GraphStoreInterface):
    """
    内存图存储 (用于测试或小规模场景)。
    """

    def __init__(self):
        self._nodes: dict[str, Entity] = {}
        self._edges: list[Edge] = []
        self._neighbors_out: dict[str, list[tuple[str, Edge]]] = {}
        self._neighbors_in: dict[str, list[tuple[str, Edge]]] = {}

    def upsert_node(self, node_id: str, entity: Entity) -> None:
        self._nodes[node_id] = entity

    def upsert_edge(self, edge: Edge) -> None:
        # 移除已存在的相同边
        self._edges = [
            e for e in self._edges
            if not (e.source_id == edge.source_id and e.target_id == edge.target_id and e.type == edge.type)
        ]
        self._edges.append(edge)

        # 更新邻居索引
        if edge.source_id not in self._neighbors_out:
            self._neighbors_out[edge.source_id] = []
        if edge.target_id not in self._neighbors_in:
            self._neighbors_in[edge.target_id] = []

        self._neighbors_out[edge.source_id].append((edge.target_id, edge))
        self._neighbors_in[edge.target_id].append((edge.source_id, edge))

    def get_node(self, node_id: str) -> Optional[Entity]:
        return self._nodes.get(node_id)

    def get_neighbors(
        self,
        node_id: str,
        edge_types: Optional[list[EdgeType]] = None,
        direction: str = "both",
    ) -> list[tuple[str, Edge]]:
        neighbors = []

        if direction in ("out", "both"):
            neighbors.extend(self._neighbors_out.get(node_id, []))

        if direction in ("in", "both"):
            neighbors.extend(self._neighbors_in.get(node_id, []))

        if edge_types:
            neighbors = [(nid, e) for nid, e in neighbors if e.type in edge_types]

        return neighbors

    def delete_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)
        self._edges = [
            e for e in self._edges
            if e.source_id != node_id and e.target_id != node_id
        ]
        self._neighbors_out.pop(node_id, None)
        self._neighbors_in.pop(node_id, None)

    def query(self, cypher: str, params: Optional[dict] = None) -> list[dict]:
        # 简单的 Cypher 解析 (仅支持基本模式)
        # 格式: MATCH (a)-[r:type]->(b) WHERE ...
        if "MATCH" in cypher.upper():
            results = []
            for edge in self._edges:
                results.append({
                    "source": self._nodes.get(edge.source_id),
                    "edge": edge,
                    "target": self._nodes.get(edge.target_id),
                })
            return results
        return []


# ============================================================================
# Unified Storage Backend
# ============================================================================


class StorageBackend:
    """
    统一存储后端。

    组合向量存储和图存储，提供统一的接口。
    """

    def __init__(
        self,
        vector_store: VectorStoreInterface,
        graph_store: GraphStoreInterface,
    ):
        self.vector = vector_store
        self.graph = graph_store

    @classmethod
    def create(
        cls,
        backend: str = "memory",
        **kwargs,
    ) -> "StorageBackend":
        """
        创建存储后端。

        Args:
            backend: 后端类型
                - "memory": InMemory (测试用)
                - "neo4j": Neo4j (需要连接信息)
                - "postgresql": PostgreSQL + pgvector
            **kwargs: 后端特定参数
        """
        if backend == "memory":
            return cls(
                vector_store=InMemoryVectorStore(dimensions=kwargs.get("dimensions", 1536)),
                graph_store=InMemoryGraphStore(),
            )
        elif backend == "neo4j":
            return cls(
                vector_store=Neo4jVectorStore(**kwargs),
                graph_store=Neo4jGraphStore(**kwargs),
            )
        elif backend == "postgresql":
            return cls(
                vector_store=PostgreSQLVectorStore(**kwargs),
                graph_store=PostgreSQLGraphStore(**kwargs),
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

"""
MemPalace — Structured episodic memory for RAG sessions.

The "Method of Loci" memory technique mapped to RAG:

  PALACE  = one user session
  ROOMS   = topic clusters (e.g. "Chemistry", "Activity 3.13", "Ionic bonds")
  DRAWERS = entities within a room (e.g. "magnesium", "NaCl", "oxidation")
  ITEMS   = chunk IDs stored inside drawers (what was retrieved for this entity)

How it works across turns:
  Turn 1: "What is magnesium oxide?"
    → spaCy extracts: ["magnesium oxide", "magnesium"]
    → Room "chemistry" created, drawer "magnesium oxide" stores chunk IDs
    → Graph: magnesium_oxide → chunk_abc, chunk_def

  Turn 2: "What happens when you heat it?"
    → enrich_query() sees recent entity "magnesium oxide"
    → Expands to: "What happens when you heat it? [context: magnesium oxide]"
    → Retriever now finds the right chunks
    → Boosted chunk IDs from drawer are surfaced first

  Turn 3: "Show me Activity 3.13"
    → Graph finds "Activity 3.13" connected to prior chunks
    → Those chunks get a memory boost in RRF merge
"""

import json
import re
import redis
import networkx as nx
from loguru import logger
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from app.config import settings


# ── spaCy lazy loader ─────────────────────────────────────────────────────────
_nlp = None

def _get_nlp():
    """Load spaCy model once and reuse."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy en_core_web_sm loaded")
        except OSError:
            logger.error(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
            raise
    return _nlp


# ── Entity extraction ─────────────────────────────────────────────────────────

# Patterns for domain-specific identifiers (textbook references)
# These are not caught by spaCy NER but are critical for RAG over textbooks
_DOMAIN_PATTERNS = [
    r'\bactivity\s+\d+[\.\-]\d+\b',       # Activity 3.13
    r'\bproblem\s+\d+[\.\-]\d+\b',         # Problem 6.10
    r'\bexample\s+\d+[\.\-]\d+\b',         # Example 2.5
    r'\bfigure\s+\d+[\.\-]\d+\b',          # Figure 4.2
    r'\bchapter\s+\d+\b',                  # Chapter 3
    r'\bsection\s+\d+[\.\-]\d+\b',         # Section 2.1
    r'\beq(?:uation)?\s*[\.\-]?\s*\d+\b',  # Equation 5, Eq. 3
    r'\btable\s+\d+[\.\-]?\d*\b',          # Table 2.1
]

_STOP_ENTITIES = {
    # Generic words spaCy tags as entities but are useless for memory
    "one", "two", "three", "first", "second", "third",
    "today", "yesterday", "monday", "january",
    "pdf", "document", "page", "answer", "question",
    "topic", "context", "example", "note",
}


def extract_entities(text: str) -> List[str]:
    """
    Extract meaningful entities from text using spaCy + domain patterns.

    Returns deduplicated list of lowercase entity strings, e.g.:
    ["magnesium oxide", "activity 3.13", "ionic bond", "nacl"]
    """
    entities = []

    # 1 — Domain patterns (textbook identifiers) — highest priority
    text_lower = text.lower()
    for pattern in _DOMAIN_PATTERNS:
        matches = re.findall(pattern, text_lower)
        entities.extend(matches)

    # 2 — spaCy NER for scientific/general entities
    try:
        nlp = _get_nlp()
        doc = nlp(text[:1000])  # cap at 1000 chars for speed

        for ent in doc.ents:
            # Keep: chemicals, compounds, scientific terms, proper nouns
            if ent.label_ in {
                "CHEM", "ORG", "PRODUCT", "GPE", "LOC",
                "PERSON", "WORK_OF_ART", "LAW", "LANGUAGE",
                "NORP",  # nationalities, groups
            }:
                val = ent.text.strip().lower()
                if len(val) > 2 and val not in _STOP_ENTITIES:
                    entities.append(val)

        # 3 — Noun chunks (catches "magnesium ribbon", "ionic compound")
        for chunk in doc.noun_chunks:
            val = chunk.text.strip().lower()
            # Only multi-word noun phrases or long single words
            if (
                len(val) > 4
                and " " in val          # multi-word preferred
                and val not in _STOP_ENTITIES
                and not val.startswith("the ")
                and not val.startswith("a ")
                and not val.startswith("an ")
            ):
                entities.append(val)

    except Exception as e:
        logger.warning(f"spaCy extraction failed: {e}")

    # Deduplicate preserving order
    seen = set()
    result = []
    for e in entities:
        if e not in seen:
            seen.add(e)
            result.append(e)

    return result[:15]  # cap at 15 per turn


def _topic_from_entities(entities: List[str]) -> str:
    """
    Infer a room/topic label from entity list.
    This becomes the 'room' in the memory palace.
    """
    if not entities:
        return "general"

    # Check for activity/problem/chapter references first
    for e in entities:
        if re.match(r'activity|problem|example|chapter|section', e):
            return e  # e.g. "activity 3.13" becomes the room name

    # Otherwise use the first meaningful entity as the room
    return entities[0]


# ── MemPalace ─────────────────────────────────────────────────────────────────

class MemPalace:
    """
    Structured episodic memory with graph-based entity tracking.

    Storage layout in Redis:
      mempalace:session:{id}:episodes  → list of JSON episode objects
      mempalace:session:{id}:graph     → JSON serialized entity graph

    Graph structure (NetworkX DiGraph):
      Nodes:
        - entity nodes  (type="entity", label="magnesium oxide")
        - chunk nodes   (type="chunk",  chunk_id="abc123")
        - topic nodes   (type="topic",  label="activity 3.13")  ← ROOMS
      Edges:
        - entity → chunk  (entity was found in this chunk)
        - topic  → entity (this entity belongs to this room/topic)
        - entity → entity (co-occurred in same turn)
    """

    MEMORY_TURNS = 5  # how many past turns to use for context

    def __init__(self):
        self.redis = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
        )

    # ── Redis key helpers ─────────────────────────────────────────────────────

    def _episodes_key(self, session_id: str) -> str:
        return f"mempalace:session:{session_id}:episodes"

    def _graph_key(self, session_id: str) -> str:
        return f"mempalace:session:{session_id}:graph"

    # ── Graph persistence ─────────────────────────────────────────────────────

    def _load_graph(self, session_id: str) -> nx.DiGraph:
        """Load graph from Redis or create empty one."""
        raw = self.redis.get(self._graph_key(session_id))
        if raw:
            try:
                data = json.loads(raw)
                return nx.node_link_graph(data)
            except Exception as e:
                logger.warning(f"Graph load failed: {e} — starting fresh")
        return nx.DiGraph()

    def _save_graph(self, session_id: str, graph: nx.DiGraph):
        """Persist graph to Redis."""
        data = nx.node_link_data(graph)
        self.redis.set(
            self._graph_key(session_id),
            json.dumps(data),
            ex=86400,  # 24h TTL
        )

    # ── Core: save episode ────────────────────────────────────────────────────

    def save_episode(
        self,
        session_id: str,
        question: str,
        answer: str,
        chunk_ids: List[str],
        entities: Optional[List[str]] = None,
    ):
        """
        Store one Q&A turn and update the entity graph.

        If entities are not passed in, extract them automatically
        from both the question and answer using spaCy.
        """
        # Auto-extract entities if not provided
        if not entities:
            combined_text = f"{question} {answer[:500]}"
            entities = extract_entities(combined_text)

        logger.debug(f"MemPalace entities extracted: {entities}")

        # Store episode in Redis list
        episode = {
            "question": question,
            "answer": answer[:500],  # cap answer length in memory
            "chunk_ids": chunk_ids,
            "entities": entities,
            "timestamp": datetime.utcnow().isoformat(),
        }
        key = self._episodes_key(session_id)
        self.redis.rpush(key, json.dumps(episode))
        self.redis.expire(key, 86400)

        # Update graph with new entities and chunk relationships
        self._update_graph(session_id, entities, chunk_ids)

        logger.debug(
            f"Saved episode for session {session_id} | "
            f"entities={entities} | chunks={len(chunk_ids)}"
        )

    # ── Graph update ──────────────────────────────────────────────────────────

    def _update_graph(
        self,
        session_id: str,
        entities: List[str],
        chunk_ids: List[str],
    ):
        """
        Update the entity-chunk graph with new turn data.

        Rooms  = topic nodes (inferred from entities)
        Drawers = entity nodes within a room
        Items   = chunk_id nodes inside drawers
        """
        graph = self._load_graph(session_id)

        # Infer room (topic) for this turn
        topic = _topic_from_entities(entities)

        # Add topic node (the ROOM)
        if not graph.has_node(topic):
            graph.add_node(topic, type="topic", label=topic, visits=0)
        graph.nodes[topic]["visits"] = graph.nodes[topic].get("visits", 0) + 1

        # Add entity nodes (DRAWERS) and connect to room
        for entity in entities:
            if not graph.has_node(entity):
                graph.add_node(entity, type="entity", label=entity, frequency=0)
            graph.nodes[entity]["frequency"] = (
                graph.nodes[entity].get("frequency", 0) + 1
            )
            # Room → Drawer edge
            if not graph.has_edge(topic, entity):
                graph.add_edge(topic, entity, weight=1)
            else:
                graph[topic][entity]["weight"] += 1

        # Add chunk nodes (ITEMS in drawers) and connect to entities
        for chunk_id in chunk_ids:
            if not graph.has_node(chunk_id):
                graph.add_node(chunk_id, type="chunk", chunk_id=chunk_id, hits=0)
            graph.nodes[chunk_id]["hits"] = (
                graph.nodes[chunk_id].get("hits", 0) + 1
            )
            # Connect each entity to the chunks it retrieved
            for entity in entities:
                if not graph.has_edge(entity, chunk_id):
                    graph.add_edge(entity, chunk_id, weight=1)
                else:
                    graph[entity][chunk_id]["weight"] += 1

        # Add co-occurrence edges between entities in same turn
        # This links related concepts (e.g. "magnesium" ↔ "oxidation")
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                if not graph.has_edge(e1, e2):
                    graph.add_edge(e1, e2, weight=1, type="co-occurrence")
                else:
                    graph[e1][e2]["weight"] += 1

        self._save_graph(session_id, graph)

    # ── Query enrichment (called before retrieval) ────────────────────────────

    def enrich_query(self, session_id: str, query: str) -> str:
        """
        Expand a follow-up query using graph context.

        Examples:
          "What happens when you heat it?"
            → "What happens when you heat it? [context: magnesium oxide, heating]"

          "Explain Q2"
            → "Explain Q2 [context: activity 3.13, ionic compound]"

        Strategy:
          1. Extract entities from the new query
          2. Walk the graph to find related entities from prior turns
          3. Append them as context hints
        """
        history = self.get_session_history(session_id)
        if not history:
            return query

        # Extract entities from the current query
        current_entities = extract_entities(query)

        # Load the graph
        graph = self._load_graph(session_id)

        related_entities = []

        if current_entities and graph.number_of_nodes() > 0:
            # Walk graph: find neighbors of current query entities
            for entity in current_entities:
                if graph.has_node(entity):
                    neighbors = list(graph.neighbors(entity))
                    # Only return entity-type neighbors, not chunk nodes
                    entity_neighbors = [
                        n for n in neighbors
                        if graph.nodes[n].get("type") in ("entity", "topic")
                        and n != entity
                    ]
                    # Sort by edge weight (most co-occurring first)
                    entity_neighbors.sort(
                        key=lambda n: graph[entity][n].get("weight", 1),
                        reverse=True,
                    )
                    related_entities.extend(entity_neighbors[:3])

        # If no graph matches, fall back to recent entities from history
        if not related_entities:
            for ep in history[-2:]:
                related_entities.extend(ep.get("entities", []))

        # Deduplicate and remove entities already in query
        current_set = set(current_entities)
        unique_related = []
        seen = set()
        for e in related_entities:
            if e not in seen and e not in current_set:
                seen.add(e)
                unique_related.append(e)

        if unique_related:
            context_str = ", ".join(unique_related[:5])
            enriched = f"{query} [context: {context_str}]"
            logger.info(f"Query enriched: '{query}' → '{enriched}'")
            return enriched

        return query

    # ── Boosted chunk IDs (called during RRF merge) ───────────────────────────

    def get_boosted_chunk_ids(self, session_id: str) -> List[str]:
        """
        Return chunk IDs to boost during retrieval, ranked by graph hits.

        These are chunks that were useful in prior turns — surface them first.
        Uses graph hit count (how many turns retrieved this chunk) for ranking.
        """
        graph = self._load_graph(session_id)

        chunk_scores: Dict[str, float] = {}

        # Get chunks from graph, scored by hits
        for node, data in graph.nodes(data=True):
            if data.get("type") == "chunk":
                chunk_scores[node] = data.get("hits", 1)

        # Also add recency boost from episode history
        history = self.get_session_history(session_id)
        recent = history[-self.MEMORY_TURNS:]
        for i, ep in enumerate(reversed(recent)):
            recency_weight = 1.0 / (i + 1)  # most recent = highest weight
            for cid in ep.get("chunk_ids", []):
                chunk_scores[cid] = chunk_scores.get(cid, 0) + recency_weight

        # Sort by combined score
        sorted_chunks = sorted(
            chunk_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [cid for cid, _ in sorted_chunks[:10]]

    # ── Context summary for LLM prompt ───────────────────────────────────────

    def build_context_summary(self, session_id: str) -> str:
        """
        Build conversation history string for LLM prompt.
        Uses last MEMORY_TURNS turns. Also adds graph-derived topic context.
        """
        history = self.get_session_history(session_id)
        if not history:
            return ""

        recent = history[-self.MEMORY_TURNS:]

        lines = ["Prior conversation:"]
        for ep in recent:
            lines.append(f"Q: {ep['question']}")
            lines.append(f"A: {ep['answer'][:200]}...")

        # Add active topics from graph (what rooms are "open")
        graph = self._load_graph(session_id)
        active_topics = [
            data["label"]
            for _, data in graph.nodes(data=True)
            if data.get("type") == "topic"
            and data.get("visits", 0) > 0
        ]
        if active_topics:
            lines.append(f"\nActive topics in this session: {', '.join(active_topics[:5])}")

        return "\n".join(lines)

    # ── Session inspection (for debugging) ───────────────────────────────────

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve all episodes for a session."""
        key = self._episodes_key(session_id)
        raw = self.redis.lrange(key, 0, -1)
        return [json.loads(r) for r in raw]

    def get_graph_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Returns a summary of the memory graph for this session.
        Useful for debugging — shows what rooms/drawers exist.
        """
        graph = self._load_graph(session_id)

        rooms = [
            {"label": d["label"], "visits": d.get("visits", 0)}
            for _, d in graph.nodes(data=True)
            if d.get("type") == "topic"
        ]
        drawers = [
            {"label": d["label"], "frequency": d.get("frequency", 0)}
            for _, d in graph.nodes(data=True)
            if d.get("type") == "entity"
        ]
        items = [
            {"chunk_id": n, "hits": d.get("hits", 0)}
            for n, d in graph.nodes(data=True)
            if d.get("type") == "chunk"
        ]

        return {
            "session_id": session_id,
            "rooms": sorted(rooms, key=lambda x: x["visits"], reverse=True),
            "drawers": sorted(drawers, key=lambda x: x["frequency"], reverse=True),
            "items": sorted(items, key=lambda x: x["hits"], reverse=True),
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
        }

    def clear_session(self, session_id: str):
        self.redis.delete(self._episodes_key(session_id))
        self.redis.delete(self._graph_key(session_id))
        logger.info(f"Cleared session {session_id}")
"""
Graph Analytics — Memgraph-compatible queries .
"""

import logging
from typing import List, Dict, Optional
from collections import defaultdict
from neo4j import GraphDatabase
from config import MEMGRAPH_URI, MEMGRAPH_USER, MEMGRAPH_PASSWORD

logger = logging.getLogger(__name__)


def get_driver():
    return GraphDatabase.driver(
        MEMGRAPH_URI, auth=(MEMGRAPH_USER, MEMGRAPH_PASSWORD)
    )


# ── Entity neighbourhood ──────────────────────────────────────
def get_entity_neighborhood(
    entity_name: str,
    depth: int = 2,
    lens_filter: Optional[str] = None,
) -> Dict:
    """
    Return all direct neighbours using TWO separate queries.
    Memgraph UNION requires identical column names — avoided entirely.
    """
    if not entity_name or not entity_name.strip():
        return {"entity": entity_name, "connections": [], "depth": depth}

    driver = get_driver()
    result = {"entity": entity_name, "connections": [], "depth": depth}
    lf     = (lens_filter or "").lower().strip()

    try:
        with driver.session() as session:

            # ── outgoing: entity → neighbour ──────────────────
            out = session.run("""
                MATCH (start:__Entity__ {name: $name})-[r]->(nb:__Entity__)
                RETURN
                    type(r)      AS relation,
                    nb.name      AS neighbour,
                    r.confidence AS confidence,
                    r.lens       AS lens,
                    r.evidence   AS evidence
                ORDER BY r.confidence DESC
                LIMIT 50
            """, {"name": entity_name})

            for rec in out:
                stored_lens = (rec["lens"] or "").lower()
                if lf and lf not in stored_lens:
                    continue
                result["connections"].append({
                    "direction":  "outgoing",
                    "relation":   rec["relation"],
                    "target":     rec["neighbour"],
                    "source":     None,
                    "confidence": rec["confidence"],
                    "lens":       rec["lens"],
                    "evidence":   rec["evidence"],
                })

            # ── incoming: neighbour → entity ──────────────────
            inc = session.run("""
                MATCH (nb:__Entity__)-[r]->(start:__Entity__ {name: $name})
                RETURN
                    type(r)      AS relation,
                    nb.name      AS neighbour,
                    r.confidence AS confidence,
                    r.lens       AS lens,
                    r.evidence   AS evidence
                ORDER BY r.confidence DESC
                LIMIT 50
            """, {"name": entity_name})

            for rec in inc:
                stored_lens = (rec["lens"] or "").lower()
                if lf and lf not in stored_lens:
                    continue
                result["connections"].append({
                    "direction":  "incoming",
                    "relation":   rec["relation"],
                    "target":     None,
                    "source":     rec["neighbour"],
                    "confidence": rec["confidence"],
                    "lens":       rec["lens"],
                    "evidence":   rec["evidence"],
                })

    except Exception as e:
        logger.error(f"get_entity_neighborhood('{entity_name}'): {e}")
        result["error"] = str(e)
    finally:
        driver.close()

    return result


# ── Path between entities ─────────────────────────────────────
def get_path_between_entities(
    entity_a: str,
    entity_b: str,
    max_hops: int = 4,
) -> List[Dict]:
    """
    Find paths using variable-length matching (Memgraph-compatible).
    shortestPath() is NOT supported in Memgraph — use ORDER BY length(path).
    """
    if not entity_a or not entity_b:
        return []
    if entity_a.strip() == entity_b.strip():
        return []

    driver = get_driver()
    paths  = []
    try:
        with driver.session() as session:
            cypher = f"""
            MATCH path = (a:__Entity__ {{name: $entity_a}})
                         -[*1..{max_hops}]-
                         (b:__Entity__ {{name: $entity_b}})
            RETURN path
            ORDER BY length(path)
            LIMIT 5
            """
            records = session.run(
                cypher,
                {"entity_a": entity_a.strip(), "entity_b": entity_b.strip()},
            )
            for rec in records:
                path  = rec["path"]
                nodes = list(path.nodes)
                rels  = list(path.relationships)
                parts = []
                for i, node in enumerate(nodes):
                    parts.append(node.get("name", "?"))
                    if i < len(rels):
                        rel_dir = "→" if rels[i].start_node.id == nodes[i].id else "←"
                        parts.append(f"--[{rels[i].type}]--{rel_dir}")
                paths.append({
                    "path":      " ".join(parts),
                    "hop_count": len(rels),
                })
    except Exception as e:
        logger.error(
            f"get_path_between_entities('{entity_a}', '{entity_b}'): {e}"
        )
    finally:
        driver.close()
    return paths


# ── Lens subgraph ─────────────────────────────────────────────
def get_lens_subgraph(
    lens_name: str,
    limit: int = 100,
) -> Dict:
    """
    Get all triples for a lens (lowercase key).
    Matches exact lens OR multi-lens containing this lens.
    """
    lf     = lens_name.lower().strip()
    driver = get_driver()
    result = {"lens": lens_name, "triples": [], "entities": set()}

    try:
        with driver.session() as session:
            cypher = """
            MATCH (s:__Entity__)-[r]->(o:__Entity__)
            WHERE r.lens = $lf
               OR (r.lens STARTS WITH 'multi:' AND r.lens CONTAINS $lf)
            RETURN
                s.name        AS subject,
                s.entity_type AS subject_type,
                type(r)       AS relation,
                r.confidence  AS confidence,
                r.evidence    AS evidence,
                r.lens        AS lens,
                o.name        AS object,
                o.entity_type AS object_type
            ORDER BY r.confidence DESC
            LIMIT $limit
            """
            for rec in session.run(cypher, {"lf": lf, "limit": limit}):
                row = dict(rec)
                result["triples"].append(row)
                result["entities"].add(row["subject"])
                result["entities"].add(row["object"])

        result["entities"]     = list(result["entities"])
        result["triple_count"] = len(result["triples"])
        result["entity_count"] = len(result["entities"])

    except Exception as e:
        logger.error(f"get_lens_subgraph('{lens_name}'): {e}")
        result["error"] = str(e)
    finally:
        driver.close()

    return result


# ── High-confidence triples ───────────────────────────────────
def get_high_confidence_triples(
    min_confidence: float = 0.8,
    limit: int = 50,
) -> List[Dict]:
    driver  = get_driver()
    triples = []
    try:
        with driver.session() as session:
            for rec in session.run("""
                MATCH (s:__Entity__)-[r]->(o:__Entity__)
                WHERE r.confidence >= $min_confidence
                RETURN
                    s.name       AS subject,
                    type(r)      AS relation,
                    o.name       AS object,
                    r.confidence AS confidence,
                    r.lens       AS lens,
                    r.evidence   AS evidence
                ORDER BY r.confidence DESC
                LIMIT $limit
            """, {"min_confidence": min_confidence, "limit": limit}):
                triples.append(dict(rec))
    except Exception as e:
        logger.error(f"get_high_confidence_triples: {e}")
    finally:
        driver.close()
    return triples


# ── Context builder ───────────────────────────────────────────
def build_context_from_triples(triples: List[Dict]) -> str:
    """Build structured LLM context grouped by subject."""
    if not triples:
        return "No relevant knowledge graph context found."

    by_subject: Dict[str, List[Dict]] = defaultdict(list)
    for t in triples:
        by_subject[t.get("subject", "?")].append(t)

    lines = ["## Knowledge Graph Context\n"]
    for subject, rows in by_subject.items():
        lines.append(f"### {subject}")
        for t in rows:
            conf     = t.get("confidence", 0)
            lens     = t.get("lens", "?")
            evidence = t.get("evidence", "")
            conf_str = f"[conf:{conf:.2f}]" if isinstance(conf, float) else ""
            lines.append(
                f"  - **{subject}** —[{t.get('relation','?')}]→ "
                f"**{t.get('object','?')}** {conf_str} (lens: {lens})"
            )
            if evidence:
                lines.append(f"    *Evidence: {evidence[:120]}*")
        lines.append("")

    return "\n".join(lines)
"""
Database and index management utilities — fixed lens case + Memgraph Cypher.
"""

import logging
from typing import List, Optional, Dict
from neo4j import GraphDatabase
from config import (
    MEMGRAPH_URI, MEMGRAPH_USER, MEMGRAPH_PASSWORD, EMBED_DIMENSION,
)

logger = logging.getLogger(__name__)


def get_driver():
    return GraphDatabase.driver(
        MEMGRAPH_URI, auth=(MEMGRAPH_USER, MEMGRAPH_PASSWORD)
    )


def wipe_database():
    driver = get_driver()
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Database wiped.")
    finally:
        driver.close()


def drop_vector_index():
    driver = get_driver()
    try:
        with driver.session() as session:
            try:
                session.run("DROP VECTOR INDEX entity")
                logger.info("Old entity vector index dropped.")
            except Exception:
                logger.info("No entity vector index to drop.")
    finally:
        driver.close()


def drop_chunk_index():
    driver = get_driver()
    try:
        with driver.session() as session:
            try:
                session.run("DROP VECTOR INDEX chunk_text")
                logger.info("Old chunk vector index dropped.")
            except Exception:
                logger.info("No chunk vector index to drop.")
    finally:
        driver.close()


def create_vector_index(dimension: int = EMBED_DIMENSION):
    driver = get_driver()
    try:
        with driver.session() as session:
            session.run(f"""
                CREATE VECTOR INDEX entity ON :__Entity__(embedding)
                WITH CONFIG {{
                    "dimension": {dimension},
                    "capacity": 10000,
                    "metric": "cos"
                }}
            """)
            logger.info(f"Vector index 'entity' created (dim={dimension}).")
    except Exception as e:
        logger.warning(f"Entity index note: {e}")
    finally:
        driver.close()


def create_chunk_vector_index(dimension: int = EMBED_DIMENSION):
    driver = get_driver()
    try:
        with driver.session() as session:
            session.run(f"""
                CREATE VECTOR INDEX chunk_text ON :__Chunk__(embedding)
                WITH CONFIG {{
                    "dimension": {dimension},
                    "capacity": 10000,
                    "metric": "cos"
                }}
            """)
            logger.info(f"Vector index 'chunk_text' created (dim={dimension}).")
    except Exception as e:
        logger.warning(f"Chunk index note: {e}")
    finally:
        driver.close()


def reset_database_and_index(dimension: int = EMBED_DIMENSION):
    wipe_database()
    drop_vector_index()
    drop_chunk_index()
    create_vector_index(dimension)
    create_chunk_vector_index(dimension)


def fix_index_after_ingestion(dimension: int = EMBED_DIMENSION):
    drop_vector_index()
    drop_chunk_index()
    create_vector_index(dimension)
    create_chunk_vector_index(dimension)


# ── Chunk node ingestion ──────────────────────────────────────
def ingest_chunk_nodes(
    nodes: list,
    embed_model,
    progress_callback=None,
) -> Dict:
    driver = get_driver()
    stats  = {"chunks_created": 0, "errors": 0}
    try:
        total = len(nodes)
        logger.info(f"Ingesting {total} chunk nodes...")
        for i, node in enumerate(nodes):
            if progress_callback and i % 3 == 0:
                progress_callback(i, total, f"Storing chunk {i+1}/{total}...")
            try:
                embedding = embed_model.get_text_embedding(node.text)
                with driver.session() as session:
                    session.run("""
                        MERGE (c:__Chunk__ {chunk_id: $chunk_id})
                        SET c.text       = $text,
                            c.doc_id     = $doc_id,
                            c.page_label = $page,
                            c.char_count = $char_count,
                            c.embedding  = $embedding
                    """, {
                        "chunk_id":  node.node_id,
                        "text":      node.text,
                        "doc_id":    node.metadata.get("file_name", "unknown"),
                        "page":      node.metadata.get("page_label", ""),
                        "char_count":len(node.text),
                        "embedding": embedding,
                    })
                stats["chunks_created"] += 1
            except Exception as e:
                logger.error(f"Chunk ingest error {node.node_id[:8]}: {e}")
                stats["errors"] += 1
        logger.info(f"Chunk ingestion complete: {stats}")
    finally:
        driver.close()
    return stats


def link_entities_to_chunks(triples: list) -> Dict:
    """
    Link __Chunk__ nodes to __Entity__ nodes.
    Uses triple.lens (which is the lens KEY e.g. 'technical')
    so filtering later is case-consistent.
    """
    driver = get_driver()
    stats  = {"links_created": 0, "errors": 0}

    from collections import defaultdict
    chunk_entity_map: Dict[str, set] = defaultdict(set)
    for triple in triples:
        if triple.chunk_id == "cross_chunk":
            continue
        # Store lens as lowercase key for consistency
        lens_key = triple.lens.lower().split(":")[0] if "multi:" in triple.lens \
                   else triple.lens.lower()
        chunk_entity_map[triple.chunk_id].add((triple.subject, lens_key))
        chunk_entity_map[triple.chunk_id].add((triple.object,  lens_key))

    try:
        with driver.session() as session:
            for chunk_id, pairs in chunk_entity_map.items():
                for entity_name, lens_key in pairs:
                    try:
                        session.run("""
                            MATCH (c:__Chunk__  {chunk_id:   $chunk_id})
                            MATCH (e:__Entity__ {name: $entity_name})
                            MERGE (c)-[r:HAS_ENTITY]->(e)
                            SET r.lens = $lens
                        """, {
                            "chunk_id":    chunk_id,
                            "entity_name": entity_name,
                            "lens":        lens_key,
                        })
                        stats["links_created"] += 1
                    except Exception as e:
                        logger.debug(f"Link error: {e}")
                        stats["errors"] += 1
        logger.info(f"Entity-Chunk linking: {stats}")
    finally:
        driver.close()
    return stats


# ── Triple ingestion ──────────────────────────────────────────
def ingest_triples_to_memgraph(
    triples: list,
    embed_model,
    batch_size: int = 50,
    progress_callback=None,
) -> Dict:
    """
    Ingest triples. Stores r.lens as LOWERCASE KEY (e.g. 'technical')
    so lens filtering is case-consistent everywhere.
    """
    driver = get_driver()
    stats  = {
        "nodes_created":         0,
        "relationships_created": 0,
        "embeddings_added":      0,
        "errors":                0,
        "total_triples":         len(triples),
    }
    try:
        logger.info(f"Ingesting {len(triples)} triples...")

        for batch_start in range(0, len(triples), batch_size):
            batch = triples[batch_start: batch_start + batch_size]
            if progress_callback:
                progress_callback(
                    batch_start, len(triples),
                    f"Ingesting triples {batch_start}/{len(triples)}...",
                )
            with driver.session() as session:
                for triple in batch:
                    try:
                        rel_type = triple.relation.value

                        # ── Normalise lens to lowercase key ───
                        # triple.lens comes as e.g. "Technical", "Client_Market",
                        # "multi:HR_People,Technical", "organic_hub"
                        # We want to store it as lowercase for easy filtering:
                        # "technical", "client_market", "multi:hr_people,technical"
                        raw_lens = triple.lens
                        if raw_lens.startswith("multi:"):
                            parts     = raw_lens[6:].split(",")
                            norm_lens = "multi:" + ",".join(
                                p.lower() for p in parts
                            )
                        else:
                            norm_lens = raw_lens.lower()

                        cypher = f"""
                        MERGE (s:__Entity__ {{name: $subject}})
                        ON CREATE SET
                            s.entity_type     = $subject_type,
                            s.created_by_lens = $lens,
                            s.chunk_id        = $chunk_id,
                            s.doc_id          = $doc_id

                        MERGE (o:__Entity__ {{name: $object}})
                        ON CREATE SET
                            o.entity_type     = $object_type,
                            o.created_by_lens = $lens,
                            o.chunk_id        = $chunk_id,
                            o.doc_id          = $doc_id

                        MERGE (s)-[r:`{rel_type}`]->(o)
                        ON CREATE SET
                            r.confidence = $confidence,
                            r.evidence   = $evidence,
                            r.lens       = $lens,
                            r.chunk_id   = $chunk_id
                        ON MATCH SET
                            r.confidence = CASE
                                WHEN $confidence > r.confidence
                                THEN $confidence ELSE r.confidence
                            END
                        """
                        session.run(cypher, {
                            "subject":      triple.subject,
                            "subject_type": triple.subject_type.value,
                            "object":       triple.object,
                            "object_type":  triple.object_type.value,
                            "confidence":   triple.confidence,
                            "evidence":     triple.evidence[:500],
                            "lens":         norm_lens,   # ← lowercase key
                            "chunk_id":     triple.chunk_id,
                            "doc_id":       triple.doc_id,
                        })
                        stats["relationships_created"] += 1
                    except Exception as e:
                        logger.error(f"Triple error: {e}")
                        stats["errors"] += 1

            logger.info(
                f"Batch {batch_start // batch_size + 1} done | "
                f"total: {stats['relationships_created']}"
            )

        # Node count
        with driver.session() as session:
            r = session.run("MATCH (n:__Entity__) RETURN count(n) AS c")
            stats["nodes_created"] = r.single()["c"]

        # Embed entities
        with driver.session() as session:
            r = session.run(
                "MATCH (n:__Entity__) WHERE n.embedding IS NULL "
                "RETURN n.name AS name"
            )
            needs_embed = [rec["name"] for rec in r]

        logger.info(f"Embedding {len(needs_embed)} entity nodes...")
        for i, name in enumerate(needs_embed):
            if progress_callback and i % 10 == 0:
                progress_callback(
                    i, len(needs_embed),
                    f"Embedding entity {i}/{len(needs_embed)}: {name[:30]}...",
                )
            try:
                emb = embed_model.get_text_embedding(name)
                with driver.session() as session:
                    session.run(
                        "MATCH (n:__Entity__ {name: $name}) "
                        "SET n.embedding = $emb",
                        {"name": name, "emb": emb},
                    )
                stats["embeddings_added"] += 1
            except Exception as e:
                logger.error(f"Embed error '{name}': {e}")
                stats["errors"] += 1

        logger.info(f"Ingestion complete: {stats}")
    finally:
        driver.close()
    return stats


# ── Vector search ─────────────────────────────────────────────
def vector_search_entities(
    query_embedding: list,
    top_k: int = 5,
) -> List[Dict]:
    driver  = get_driver()
    results = []
    try:
        with driver.session() as session:
            cypher = """
            CALL vector_search.search("entity", $top_k, $embedding)
            YIELD node, distance
            RETURN
                node.name        AS name,
                node.entity_type AS entity_type,
                distance
            ORDER BY distance
            LIMIT $top_k
            """
            records = session.run(
                cypher, {"top_k": top_k, "embedding": query_embedding}
            )
            results = [dict(r) for r in records]
            logger.info(f"Vector search: {len(results)} entities found")
    except Exception as e:
        logger.error(f"Vector search error: {e}")
        try:
            with driver.session() as session:
                recs = session.run(
                    "MATCH (n:__Entity__) "
                    "RETURN n.name AS name, "
                    "       n.entity_type AS entity_type, "
                    "       0.5 AS distance "
                    "LIMIT $top_k",
                    {"top_k": top_k},
                )
                results = [dict(r) for r in recs]
                logger.warning(f"Fallback: {len(results)} entities")
        except Exception as e2:
            logger.error(f"Fallback failed: {e2}")
    finally:
        driver.close()
    return results


def vector_search_chunks(
    query_embedding: list,
    top_k: int = 3,
) -> List[Dict]:
    driver  = get_driver()
    results = []
    try:
        with driver.session() as session:
            cypher = """
            CALL vector_search.search("chunk_text", $top_k, $embedding)
            YIELD node, distance
            RETURN
                node.chunk_id   AS chunk_id,
                node.text       AS text,
                node.doc_id     AS doc_id,
                node.page_label AS page,
                distance
            ORDER BY distance
            LIMIT $top_k
            """
            records = session.run(
                cypher, {"top_k": top_k, "embedding": query_embedding}
            )
            results = [dict(r) for r in records]
            logger.info(f"Chunk search: {len(results)} chunks found")
    except Exception as e:
        logger.error(f"Chunk search error: {e}")
    finally:
        driver.close()
    return results


def get_triples_for_entities(
    entity_names: List[str],
    limit: int = 50,
    lens_filter: Optional[str] = None,
) -> List[Dict]:
    """
    Fetch triples for entities. lens_filter is a lowercase key
    e.g. 'technical', 'client_market', 'hr_people'.

    Matches:
      - exact:  r.lens = 'technical'
      - multi:  r.lens = 'multi:technical,client_market'
      - organic: always included (no lens filter on organic)
    """
    if not entity_names:
        return []

    driver  = get_driver()
    triples = []
    try:
        with driver.session() as session:
            if lens_filter:
                # Normalise filter to lowercase
                lf = lens_filter.lower()
                cypher = """
                MATCH (s:__Entity__)-[r]->(o:__Entity__)
                WHERE (s.name IN $names OR o.name IN $names)
                  AND (
                      r.lens = $lf
                      OR r.lens STARTS WITH ('multi:') AND r.lens CONTAINS $lf
                      OR r.lens STARTS WITH 'organic'
                  )
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
                records = session.run(
                    cypher,
                    {"names": entity_names, "lf": lf, "limit": limit},
                )
            else:
                cypher = """
                MATCH (s:__Entity__)-[r]->(o:__Entity__)
                WHERE s.name IN $names OR o.name IN $names
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
                records = session.run(
                    cypher, {"names": entity_names, "limit": limit}
                )
            triples = [dict(r) for r in records]
            logger.info(
                f"Fetched {len(triples)} triples "
                f"(lens_filter={lens_filter!r})"
            )
    except Exception as e:
        logger.error(f"get_triples_for_entities error: {e}")
    finally:
        driver.close()
    return triples


def get_chunk_text_for_entities(
    entity_names: List[str],
    limit: int = 5,
) -> List[Dict]:
    if not entity_names:
        return []
    driver = get_driver()
    chunks = []
    try:
        with driver.session() as session:
            cypher = """
            MATCH (c:__Chunk__)-[:HAS_ENTITY]->(e:__Entity__)
            WHERE e.name IN $names
            RETURN DISTINCT
                c.chunk_id   AS chunk_id,
                c.text       AS text,
                c.doc_id     AS doc_id,
                c.page_label AS page
            LIMIT $limit
            """
            records = session.run(
                cypher, {"names": entity_names, "limit": limit}
            )
            chunks = [dict(r) for r in records]
    except Exception as e:
        logger.error(f"get_chunk_text_for_entities error: {e}")
    finally:
        driver.close()
    return chunks


def get_all_triples_for_lens(
    lens_name: str,
    limit: int = 200,
) -> List[Dict]:
    """
    Get all triples for a lens. lens_name is the lowercase key.
    """
    driver  = get_driver()
    triples = []
    lf      = lens_name.lower()
    try:
        with driver.session() as session:
            cypher = """
            MATCH (s:__Entity__)-[r]->(o:__Entity__)
            WHERE r.lens = $lf
               OR (r.lens STARTS WITH 'multi:' AND r.lens CONTAINS $lf)
            OPTIONAL MATCH (c:__Chunk__ {chunk_id: r.chunk_id})
            RETURN
                s.name        AS subject,
                s.entity_type AS subject_type,
                type(r)       AS relation,
                r.confidence  AS confidence,
                r.evidence    AS evidence,
                r.lens        AS lens,
                r.chunk_id    AS chunk_id,
                o.name        AS object,
                o.entity_type AS object_type,
                c.text        AS source_text,
                c.doc_id      AS doc_id,
                c.page_label  AS page
            ORDER BY r.confidence DESC
            LIMIT $limit
            """
            records = session.run(cypher, {"lf": lf, "limit": limit})
            triples = [dict(r) for r in records]
            logger.info(
                f"get_all_triples_for_lens({lens_name!r}): "
                f"{len(triples)} triples"
            )
    except Exception as e:
        logger.error(f"get_all_triples_for_lens error: {e}")
    finally:
        driver.close()
    return triples


# ── Graph stats ───────────────────────────────────────────────
def get_graph_stats() -> dict:
    driver = get_driver()
    stats  = {}
    try:
        with driver.session() as session:
            stats["total_nodes"] = session.run(
                "MATCH (n) RETURN count(n) AS c"
            ).single()["c"]

            stats["total_relationships"] = session.run(
                "MATCH ()-[r]->() RETURN count(r) AS c"
            ).single()["c"]

            stats["entities"] = session.run(
                "MATCH (n:__Entity__) RETURN count(n) AS c"
            ).single()["c"]

            stats["chunks"] = session.run(
                "MATCH (n:__Chunk__) RETURN count(n) AS c"
            ).single()["c"]

            stats["entities_with_embeddings"] = session.run(
                "MATCH (n:__Entity__) WHERE n.embedding IS NOT NULL "
                "RETURN count(n) AS c"
            ).single()["c"]

            r = session.run("""
                MATCH (n:__Entity__)
                WHERE n.entity_type IS NOT NULL
                RETURN n.entity_type AS entity_type, count(*) AS count
                ORDER BY count DESC LIMIT 20
            """)
            stats["entity_type_counts"] = {
                rec["entity_type"]: rec["count"] for rec in r
            }

            r = session.run("""
                MATCH ()-[rel]->()
                WHERE rel.lens IS NOT NULL
                RETURN rel.lens AS lens, count(*) AS count
                ORDER BY count DESC
            """)
            stats["by_lens"] = {rec["lens"]: rec["count"] for rec in r}

            r = session.run("""
                MATCH ()-[rel]->()
                RETURN type(rel) AS rel_type, count(*) AS count
                ORDER BY count DESC LIMIT 15
            """)
            stats["relationship_types"] = {
                rec["rel_type"]: rec["count"] for rec in r
            }

            r = session.run("""
                MATCH (n)
                WITH labels(n) AS lbls
                UNWIND lbls AS label
                RETURN label, count(*) AS count
                ORDER BY count DESC LIMIT 10
            """)
            stats["label_counts"] = {
                rec["label"]: rec["count"] for rec in r
            }

    except Exception as e:
        logger.error(f"Stats error: {e}")
        stats["error"] = str(e)
    finally:
        driver.close()
    return stats
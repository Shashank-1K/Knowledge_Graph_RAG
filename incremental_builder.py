"""
Incremental Knowledge Graph Builder.

Adds new documents to an EXISTING knowledge graph without
rebuilding from scratch. Key operations:

1. Load + chunk only the NEW documents
2. Extract triples only from new chunks
3. Run organic discovery on ALL triples
   (old + new) to find cross-document relations
4. Merge new triples into existing graph
   (MERGE prevents duplicates)
5. Add embeddings for new entities only
6. Register new documents in the registry
"""

import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from config import (
    CHUNK_SIZE, CHUNK_OVERLAP, EMBED_DIMENSION,
    MIN_TRIPLE_CONFIDENCE, ACTIVE_LENSES,
    ORGANIC_MIN_COOCCURRENCE, ORGANIC_ENABLE_HUB,
    ORGANIC_ENABLE_TEMPORAL, ORGANIC_MAX_RELATIONS,
    INGESTION_BATCH_SIZE,
)
from directed_extractor import DirectedGraphExtractor
from organic_relations import OrganicRelationPipeline, EntityGraph
from graph_setup import (
    ingest_triples_to_memgraph,
    ingest_chunk_nodes,
    link_entities_to_chunks,
    fix_index_after_ingestion,
    get_driver,
)
from doc_registry import register_document, get_all_registered_docs
from schema import ExtractedTriple

logger = logging.getLogger(__name__)


# ── Fetch existing triples from Memgraph ──────────────────────
def fetch_existing_triples_from_graph() -> List[Dict]:
    """
    Load all currently stored triples from Memgraph.
    Used to run organic discovery across old + new triples.
    """
    driver = get_driver()
    triples = []
    try:
        with driver.session() as session:
            records = session.run("""
                MATCH (s:__Entity__)-[r]->(o:__Entity__)
                WHERE r.lens IS NOT NULL
                RETURN s.name        AS subject,
                       s.entity_type AS subject_type,
                       type(r)       AS relation,
                       r.confidence  AS confidence,
                       r.evidence    AS evidence,
                       r.lens        AS lens,
                       r.chunk_id    AS chunk_id,
                       o.name        AS object,
                       o.entity_type AS object_type
                LIMIT 5000
            """)
            triples = [dict(r) for r in records]
        logger.info(f"Fetched {len(triples)} existing triples from graph")
    except Exception as e:
        logger.error(f"fetch_existing_triples_from_graph: {e}")
    finally:
        driver.close()
    return triples


def dict_to_extracted_triple(row: Dict) -> Optional[ExtractedTriple]:
    """Convert a Memgraph record dict back to an ExtractedTriple."""
    from schema import get_entity_type_safe, get_relation_type_safe
    try:
        st = get_entity_type_safe(row.get("subject_type", ""))
        ot = get_entity_type_safe(row.get("object_type", ""))
        rt = get_relation_type_safe(row.get("relation", ""))
        if not (st and ot and rt):
            return None
        return ExtractedTriple(
            subject      = row["subject"],
            subject_type = st,
            relation     = rt,
            object       = row["object"],
            object_type  = ot,
            confidence   = float(row.get("confidence") or 0.7),
            evidence     = row.get("evidence") or "",
            lens         = row.get("lens") or "unknown",
            chunk_id     = row.get("chunk_id") or "unknown",
            doc_id       = row.get("doc_id") or "unknown",
        )
    except Exception:
        return None


# ── Remove document triples from graph ────────────────────────
def remove_document_from_graph(doc_id: str) -> Dict:
    """
    Remove all triples and chunks belonging to a specific document.
    Used when re-ingesting a changed document.
    """
    driver = get_driver()
    stats  = {"chunks_removed": 0, "entities_affected": 0}
    try:
        with driver.session() as session:
            # Remove chunk nodes for this doc
            result = session.run("""
                MATCH (c:__Chunk__ {doc_id: $doc_id})
                DETACH DELETE c
                RETURN count(c) AS removed
            """, {"doc_id": doc_id})
            rec = result.single()
            if rec:
                stats["chunks_removed"] = rec["removed"]

            # Remove relations that ONLY came from this doc
            # (keep entities that are shared with other docs)
            session.run("""
                MATCH ()-[r]->()
                WHERE r.chunk_id CONTAINS $doc_id
                DELETE r
            """, {"doc_id": doc_id})

        logger.info(f"Removed graph data for doc: {doc_id}")
    except Exception as e:
        logger.error(f"remove_document_from_graph({doc_id}): {e}")
        stats["error"] = str(e)
    finally:
        driver.close()
    return stats


# ── INCREMENTAL BUILD ─────────────────────────────────────────
def run_incremental_build(
    new_file_paths: List[str],
    llm,
    embed_model,
    active_lenses: List[str],
    status_container,
    force_reprocess: bool = False,
) -> Dict:
    """
    Add new documents to the existing knowledge graph.

    Steps:
    1. Load + chunk only the new files
    2. Store new chunk nodes (text + embeddings)
    3. Extract triples from new chunks using lenses
    4. Fetch all existing triples from graph
    5. Run organic discovery on old + new combined
    6. Ingest only the NEW triples (MERGE handles dedup)
    7. Link new entities → their source chunks
    8. Register new documents in registry
    9. Fix vector indexes

    Returns: build stats dict
    """
    result     = {}
    step_times = {}
    all_new_triples: List[ExtractedTriple] = []

    status_container.write(
        f"📂 Processing {len(new_file_paths)} new document(s)..."
    )

    # ── Step 1: Load new documents ────────────────────────────
    status_container.write("📄 Loading new documents...")
    t0 = time.time()

    all_nodes: List[TextNode] = []
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    per_doc_nodes: Dict[str, List[TextNode]] = {}
    for file_path in new_file_paths:
        path = Path(file_path)
        status_container.write(f"  → Loading: {path.name}")
        try:
            docs  = SimpleDirectoryReader(input_files=[file_path]).load_data()
            nodes = splitter.get_nodes_from_documents(docs)
            per_doc_nodes[file_path] = nodes
            all_nodes.extend(nodes)
            status_container.write(
                f"  ✅ {path.name}: {len(nodes)} chunk(s)"
            )
        except Exception as e:
            status_container.write(f"  ❌ Failed to load {path.name}: {e}")
            logger.error(f"Failed to load {file_path}: {e}")

    step_times["load"] = time.time() - t0
    result["new_chunks"] = len(all_nodes)
    status_container.write(
        f"✅ Loaded {len(all_nodes)} chunks from "
        f"{len(new_file_paths)} document(s) ({step_times['load']:.1f}s)"
    )

    if not all_nodes:
        return {"error": "No chunks could be loaded from new documents"}

    # ── Step 2: Store chunk nodes ─────────────────────────────
    status_container.write(
        f"💾 Storing {len(all_nodes)} chunk node(s) in Memgraph..."
    )
    t0 = time.time()

    def chunk_progress(cur, tot, msg):
        if cur % 3 == 0:
            status_container.write(f"  → {msg}")

    chunk_stats = ingest_chunk_nodes(all_nodes, embed_model, chunk_progress)
    step_times["chunk_storage"] = time.time() - t0
    status_container.write(
        f"✅ {chunk_stats['chunks_created']} chunk node(s) stored "
        f"({step_times['chunk_storage']:.1f}s)"
    )

    # ── Step 3: Extract triples from new chunks ───────────────
    status_container.write(
        f"🔬 Extracting with {len(active_lenses)} lens(es)..."
    )
    t0 = time.time()

    extractor = DirectedGraphExtractor(
        llm=llm,
        active_lenses=active_lenses,
        min_confidence=MIN_TRIPLE_CONFIDENCE,
    )

    def extract_progress(cur, tot, msg):
        status_container.write(f"  → {msg}")

    new_triples = extractor.extract_from_nodes(
        all_nodes, progress_callback=extract_progress
    )
    extraction_summary = extractor.get_extraction_summary(new_triples)
    step_times["extraction"] = time.time() - t0

    status_container.write(
        f"✅ {len(new_triples)} new triples from "
        f"{extraction_summary['unique_entities']} entities "
        f"({step_times['extraction']:.1f}s)"
    )
    result["extraction_summary"] = extraction_summary
    all_new_triples.extend(new_triples)

    # ── Step 4: Fetch existing triples from graph ─────────────
    status_container.write(
        "🔍 Fetching existing graph triples for organic discovery..."
    )
    t0 = time.time()
    existing_raw   = fetch_existing_triples_from_graph()
    existing_triples = [
        t for t in (dict_to_extracted_triple(r) for r in existing_raw)
        if t is not None
    ]
    step_times["fetch_existing"] = time.time() - t0
    status_container.write(
        f"✅ Fetched {len(existing_triples)} existing triples "
        f"({step_times['fetch_existing']:.1f}s)"
    )

    # ── Step 5: Organic discovery on combined corpus ──────────
    status_container.write(
        "🌱 Running organic discovery on combined graph "
        "(old + new)..."
    )
    t0 = time.time()

    combined_for_organic = existing_triples + new_triples
    organic_pipeline = OrganicRelationPipeline(
        min_cooccurrence=ORGANIC_MIN_COOCCURRENCE,
        enable_hub_discovery=ORGANIC_ENABLE_HUB,
        enable_temporal_discovery=ORGANIC_ENABLE_TEMPORAL,
        max_organic_relations=ORGANIC_MAX_RELATIONS,
    )
    _, organic_relations = organic_pipeline.run(
        combined_for_organic, source_doc="incremental"
    )
    organic_summary = organic_pipeline.get_organic_summary(organic_relations)
    step_times["organic"] = time.time() - t0

    # Only keep NEW organic relations (not already in graph)
    existing_keys = {
        (r["subject"].lower(), r["relation"], r["object"].lower())
        for r in existing_raw
    }
    new_organic = [
        r for r in organic_relations
        if (r.subject.lower(), r.relation.value, r.object.lower())
        not in existing_keys
    ]

    status_container.write(
        f"✅ {len(new_organic)} new organic relation(s) "
        f"({step_times['organic']:.1f}s)"
    )
    result["organic_summary"] = organic_summary
    result["new_organic_count"] = len(new_organic)

    # Combine new extracted + new organic
    triples_to_ingest = new_triples + new_organic
    result["total_new_triples"] = len(triples_to_ingest)

    # ── Step 6: Ingest new triples ────────────────────────────
    status_container.write(
        f"💾 Ingesting {len(triples_to_ingest)} new triple(s)..."
    )
    t0 = time.time()

    def ingest_progress(cur, tot, msg):
        if cur % 50 == 0:
            status_container.write(f"  → {msg}")

    ingestion_stats = ingest_triples_to_memgraph(
        triples_to_ingest,
        embed_model,
        batch_size=INGESTION_BATCH_SIZE,
        progress_callback=ingest_progress,
    )
    step_times["ingestion"] = time.time() - t0
    status_container.write(
        f"✅ {ingestion_stats['relationships_created']} relationship(s), "
        f"{ingestion_stats['embeddings_added']} new entity embedding(s) "
        f"({step_times['ingestion']:.1f}s)"
    )
    result["ingestion_stats"] = ingestion_stats

    # ── Step 7: Link new entities → their chunks ──────────────
    status_container.write("🔗 Linking new entities → source chunks...")
    t0 = time.time()
    link_stats = link_entities_to_chunks(triples_to_ingest)
    step_times["linking"] = time.time() - t0
    status_container.write(
        f"✅ {link_stats['links_created']} entity-chunk link(s) "
        f"({step_times['linking']:.1f}s)"
    )

    # ── Step 8: Register each new document ───────────────────
    status_container.write("📋 Registering documents in registry...")
    for file_path in new_file_paths:
        path       = Path(file_path)
        doc_nodes  = per_doc_nodes.get(file_path, [])
        doc_triples = [
            t for t in new_triples
            if t.doc_id == path.name
        ]
        reg = register_document(
            file_path    = file_path,
            chunk_count  = len(doc_nodes),
            triple_count = len(doc_triples),
            lenses_used  = active_lenses,
        )
        status_container.write(
            f"  ✅ Registered: {path.name} "
            f"({len(doc_nodes)} chunks, {len(doc_triples)} triples)"
        )

    # ── Step 9: Fix vector indexes ────────────────────────────
    status_container.write("🔧 Confirming vector indexes (768d)...")
    fix_index_after_ingestion(EMBED_DIMENSION)
    status_container.write("✅ Vector indexes confirmed")

    total_time = sum(step_times.values())
    status_container.write(f"🎉 Incremental update complete! Total: {total_time:.1f}s")

    result["step_times"]  = step_times
    result["all_new_triples"] = triples_to_ingest
    return result


# ── SAVE UPLOADED FILE ────────────────────────────────────────
def save_uploaded_file(uploaded_file, target_dir: str) -> Optional[str]:
    """
    Save a Streamlit UploadedFile to disk.
    Returns the saved file path, or None on failure.
    """
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    file_path = target_path / uploaded_file.name
    try:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"Saved uploaded file: {file_path}")
        return str(file_path)
    except Exception as e:
        logger.error(f"Failed to save {uploaded_file.name}: {e}")
        return None
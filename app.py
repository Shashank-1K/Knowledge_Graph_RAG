"""
Streamlit Knowledge Graph RAG
Directed Graph Generation · Stakeholder Lenses · Organic Relations
chunk text nodes, lens Q&A, graph visualisation by lens.
"""

import streamlit as st
import nest_asyncio
import logging
import sys
import os
import time
import traceback
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List 

nest_asyncio.apply()

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("APP")
_fh = logging.FileHandler("app_debug.log", mode="w")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(
    logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
)
logger.addHandler(_fh)
logger.info("APP STARTING")

# ── Imports ───────────────────────────────────────────────────
from config import (
    OLLAMA_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT,
    EMBED_MODEL, EMBED_DIMENSION, CHUNK_SIZE, CHUNK_OVERLAP,
    SIMILARITY_TOP_K, MEMGRAPH_USER, MEMGRAPH_PASSWORD, MEMGRAPH_URI,
    ACTIVE_LENSES, MIN_TRIPLE_CONFIDENCE,
    ORGANIC_MIN_COOCCURRENCE, ORGANIC_ENABLE_HUB, ORGANIC_ENABLE_TEMPORAL,
    ORGANIC_MAX_RELATIONS, INGESTION_BATCH_SIZE,
    GRAPH_QUERY_DEPTH, MAX_CONTEXT_TRIPLES,
)
from graph_setup import (
    reset_database_and_index,
    fix_index_after_ingestion,
    get_graph_stats,
    ingest_triples_to_memgraph,
    ingest_chunk_nodes,
    link_entities_to_chunks,
    vector_search_entities,
    vector_search_chunks,
    get_triples_for_entities,
    get_chunk_text_for_entities,
    get_all_triples_for_lens,
)
from directed_extractor import DirectedGraphExtractor
from organic_relations import OrganicRelationPipeline
from graph_analytics import (
    get_entity_neighborhood,
    get_path_between_entities,
    get_lens_subgraph,
    get_high_confidence_triples,
    build_context_from_triples,
)

# ── imports  ──────────────────────────
from doc_registry import (
    get_new_documents,
    get_all_registered_docs,
    get_registry_summary,
    ensure_registry_exists,
    remove_document_from_registry,
)
from incremental_builder import (
    run_incremental_build,
    save_uploaded_file,
    remove_document_from_graph,
)

from schema import STAKEHOLDER_LENSES

from llama_index.core import Settings, SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

logger.info("All imports successful")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Knowledge Graph RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header {
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(90deg, #4CAF50, #2196F3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.chunk-box {
    background: #0d1117;
    border-left: 3px solid #4CAF50;
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-family: monospace;
    white-space: pre-wrap;
    margin: 6px 0;
}
.triple-row {
    padding: 4px 8px;
    border-radius: 4px;
    margin: 2px 0;
    font-size: 0.87rem;
}
.lens-header {
    font-size: 1.1rem;
    font-weight: 600;
    padding: 6px 0;
    border-bottom: 2px solid #333;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
LENS_ICONS = {
    "executive":          "🔵",
    "technical":          "🟢",
    "hr_people":          "🟣",
    "operations":         "🟠",
    "client_market":      "🟡",
    "organic_hub":        "🌐",
    "organic_cooccurrence":"🔗",
}
LENS_COLORS = {
    "executive":     "#1565C0",
    "technical":     "#2E7D32",
    "hr_people":     "#6A1B9A",
    "operations":    "#E65100",
    "client_market": "#F57F17",
}
CONF_ICONS = {True: "🟢", False: "🟡"}  # True = high confidence



def get_lens_icon(lens_str: str) -> str:
    """
    Safely resolve an icon for any lens string, including:
      - plain keys:   'technical'
      - multi keys:   'multi:technical,client_market'
      - organic:      'organic_hub'
    """
    if not lens_str:
        return "⚪"
    ls = lens_str.lower()
    # Direct match
    if ls in LENS_ICONS:
        return LENS_ICONS[ls]
    # Multi-lens: pick icon of first lens in the list
    if ls.startswith("multi:"):
        first = ls[6:].split(",")[0].strip()
        return LENS_ICONS.get(first, "⚪")
    # Organic fallback
    if "organic" in ls:
        return "🌐"
    return "⚪"



def _register_all_existing(
    data_dir: str,
    build_result: dict,
    active_lenses: list,
):
    """
    After a full rebuild, register all documents from data_dir
    in the doc registry so incremental adds work correctly.
    """
    from doc_registry import register_document
    data_path = Path(data_dir)
    if not data_path.exists():
        return
    for f in data_path.glob("*"):
        if f.is_file() and not f.name.startswith("."):
            try:
                register_document(
                    file_path    = str(f),
                    chunk_count  = build_result.get("chunk_count", 0),
                    triple_count = build_result.get("total_triples", 0),
                    lenses_used  = active_lenses,
                )
            except Exception as e:
                logger.debug(f"Could not register {f.name}: {e}")



# ── Model init ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def init_models():
    llm = Ollama(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        request_timeout=LLM_TIMEOUT,
    )
    embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    Settings.llm        = llm
    Settings.embed_model = embed_model
    Settings.chunk_size  = CHUNK_SIZE
    logger.info(f"Models: LLM={LLM_MODEL}, Embed={EMBED_MODEL}")
    return llm, embed_model


# ── RAG query pipeline ────────────────────────────────────────
def run_graph_rag_query(
    question: str,
    embed_model,
    llm,
    top_k: int = SIMILARITY_TOP_K,
    max_triples: int = MAX_CONTEXT_TRIPLES,
    lens_filter: Optional[str] = None,   
) -> dict:
    """
    Full RAG pipeline using direct Memgraph calls.

    Steps:
    1. Embed question
    2. Vector-search __Entity__ nodes
    3. Also vector-search __Chunk__ nodes for raw text context
    4. Fetch graph triples for matched entities (optionally lens-filtered)
    5. Fetch source chunk text for context
    6. Build combined context (graph triples + raw text)
    7. LLM synthesis
    """

    t_start = time.time()

    # ── 1. Embed ──────────────────────────────────────────────
    t0 = time.time()
    q_emb = embed_model.get_text_embedding(question)
    embed_time = time.time() - t0
    logger.info(f"Embedded in {embed_time:.2f}s")

    # ── 2. Entity vector search ───────────────────────────────
    t0 = time.time()
    similar_entities = vector_search_entities(q_emb, top_k=top_k)
    entity_search_time = time.time() - t0
    entity_names = [e["name"] for e in similar_entities]
    logger.info(f"Entity search: {entity_names} ({entity_search_time:.2f}s)")

    # ── 3. Chunk vector search ────────────────────────────────
    t0 = time.time()
    similar_chunks = vector_search_chunks(q_emb, top_k=3)
    chunk_search_time = time.time() - t0
    logger.info(
        f"Chunk search: {len(similar_chunks)} chunks ({chunk_search_time:.2f}s)"
    )

    # ── 4. Fetch triples ──────────────────────────────────────
    t0 = time.time()
    triples = get_triples_for_entities(
        entity_names,
        limit=max_triples,
        lens_filter=lens_filter,
    )
    fetch_time = time.time() - t0
    logger.info(f"Fetched {len(triples)} triples ({fetch_time:.2f}s)")

    # ── 5. Fetch source chunk text ────────────────────────────
    source_chunks = get_chunk_text_for_entities(entity_names, limit=3)
    seen_ids = {c["chunk_id"] for c in source_chunks}
    for sc in similar_chunks:
        if sc.get("chunk_id") not in seen_ids:
            source_chunks.append(sc)

    # ── 6. Build context ──────────────────────────────────────
    graph_context = build_context_from_triples(triples)

    text_context_parts = []
    for i, chunk in enumerate(source_chunks[:3]):
        text = chunk.get("text", "")
        doc  = chunk.get("doc_id", "?")
        page = chunk.get("page", "")
        page_str = f" (page {page})" if page else ""
        text_context_parts.append(
            f"### Source Text {i+1} — {doc}{page_str}\n{text}"
        )
    text_context = "\n\n".join(text_context_parts) if text_context_parts else ""

    lens_note = (
        f"\n(Context filtered to '{lens_filter}' lens perspective)\n"
        if lens_filter else ""
    )

    # ── 7. LLM synthesis ─────────────────────────────────────
    prompt = f"""You are a knowledgeable assistant answering questions \
based on structured knowledge graph data and source document text.
{lens_note}
{graph_context}

## Source Document Text
{text_context if text_context else "No source text available."}

---

Question: {question}

Instructions:
- Answer based ONLY on the context provided above.
- Prefer specific facts from the knowledge graph (entities, relations, evidence).
- Use the source text to add detail and nuance.
- If context is insufficient, say so clearly.
- Be concise and factual.

Answer:"""

    t0 = time.time()
    response = llm.complete(prompt)
    llm_time = time.time() - t0
    answer   = str(response).strip()
    logger.info(f"LLM answered in {llm_time:.2f}s | len={len(answer)}")

    total_time = time.time() - t_start
    return {
        "answer":          answer,
        "entities_found":  similar_entities,
        "triples":         triples,
        "source_chunks":   source_chunks,
        "lens_filter":     lens_filter,
        "timing": {
            "embed_s":         round(embed_time,         2),
            "entity_search_s": round(entity_search_time, 2),
            "chunk_search_s":  round(chunk_search_time,  2),
            "fetch_s":         round(fetch_time,         2),
            "llm_s":           round(llm_time,           2),
            "total_s":         round(total_time,         2),
        },
        "question": question,
    }


# ── Build pipeline ────────────────────────────────────────────
def run_directed_build_pipeline(
    data_dir: str,
    llm,
    embed_model,
    active_lenses: list,
    reset_db: bool,
    status_container,
) -> dict:
    """Full directed KG build pipeline with chunk node storage."""
    result      = {}
    step_times  = {}

    # 1 – Reset
    if reset_db:
        status_container.write("🗑️ Resetting database...")
        t0 = time.time()
        reset_database_and_index(EMBED_DIMENSION)
        step_times["reset"] = time.time() - t0
        status_container.write(
            f"✅ Database reset ({step_times['reset']:.1f}s)"
        )

    # 2 – Load
    status_container.write("📄 Loading documents...")
    t0 = time.time()
    documents = SimpleDirectoryReader(data_dir).load_data()
    step_times["load"] = time.time() - t0
    status_container.write(
        f"✅ Loaded {len(documents)} document(s) ({step_times['load']:.1f}s)"
    )
    result["doc_count"] = len(documents)

    # 3 – Chunk
    status_container.write("✂️ Chunking documents...")
    t0 = time.time()
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    nodes = splitter.get_nodes_from_documents(documents)
    step_times["chunking"] = time.time() - t0
    status_container.write(
        f"✅ Created {len(nodes)} chunks ({step_times['chunking']:.1f}s)"
    )
    result["chunk_count"] = len(nodes)

    # 4 – Store chunk nodes in Memgraph (with embeddings + text)
    status_container.write(
        f"💾 Storing {len(nodes)} chunk nodes (text + embeddings)..."
    )
    t0 = time.time()

    def chunk_progress(cur, tot, msg):
        if cur % 3 == 0:
            status_container.write(f"  → {msg}")

    chunk_stats = ingest_chunk_nodes(nodes, embed_model, chunk_progress)
    step_times["chunk_storage"] = time.time() - t0
    status_container.write(
        f"✅ {chunk_stats['chunks_created']} chunk nodes stored "
        f"({step_times['chunk_storage']:.1f}s)"
    )

    # 5 – Directed extraction
    status_container.write(
        f"🔬 Extracting with {len(active_lenses)} stakeholder lenses..."
    )
    t0 = time.time()
    extractor = DirectedGraphExtractor(
        llm=llm,
        active_lenses=active_lenses,
        min_confidence=MIN_TRIPLE_CONFIDENCE,
    )

    def extract_progress(cur, tot, msg):
        status_container.write(f"  → {msg}")

    triples = extractor.extract_from_nodes(nodes, progress_callback=extract_progress)
    extraction_summary = extractor.get_extraction_summary(triples)
    step_times["extraction"] = time.time() - t0
    status_container.write(
        f"✅ {len(triples)} triples from "
        f"{extraction_summary['unique_entities']} entities "
        f"({step_times['extraction']:.1f}s)"
    )
    result["extraction_summary"] = extraction_summary

    # 6 – Organic relations
    status_container.write("🌱 Discovering organic cross-chunk relations...")
    t0 = time.time()
    organic_pipeline = OrganicRelationPipeline(
        min_cooccurrence=ORGANIC_MIN_COOCCURRENCE,
        enable_hub_discovery=ORGANIC_ENABLE_HUB,
        enable_temporal_discovery=ORGANIC_ENABLE_TEMPORAL,
        max_organic_relations=ORGANIC_MAX_RELATIONS,
    )
    all_triples, organic_relations = organic_pipeline.run(
        triples, source_doc=data_dir
    )
    organic_summary = organic_pipeline.get_organic_summary(organic_relations)
    step_times["organic"] = time.time() - t0
    status_container.write(
        f"✅ {len(organic_relations)} organic relations "
        f"({step_times['organic']:.1f}s)"
    )
    result["organic_summary"] = organic_summary
    result["total_triples"]   = len(all_triples)

    # 7 – Ingest triples
    status_container.write(
        f"💾 Ingesting {len(all_triples)} triples..."
    )
    t0 = time.time()

    def ingest_progress(cur, tot, msg):
        if cur % 50 == 0:
            status_container.write(f"  → {msg}")

    ingestion_stats = ingest_triples_to_memgraph(
        all_triples, embed_model,
        batch_size=INGESTION_BATCH_SIZE,
        progress_callback=ingest_progress,
    )
    step_times["ingestion"] = time.time() - t0
    status_container.write(
        f"✅ {ingestion_stats['relationships_created']} relationships, "
        f"{ingestion_stats['embeddings_added']} entity embeddings "
        f"({step_times['ingestion']:.1f}s)"
    )
    result["ingestion_stats"] = ingestion_stats

    # 8 – Link entities to chunks
    status_container.write("🔗 Linking entities → source chunks...")
    t0 = time.time()
    link_stats = link_entities_to_chunks(all_triples)
    step_times["linking"] = time.time() - t0
    status_container.write(
        f"✅ {link_stats['links_created']} entity-chunk links "
        f"({step_times['linking']:.1f}s)"
    )

    # 9 – Fix vector indexes
    status_container.write("🔧 Confirming vector indexes (768d)...")
    fix_index_after_ingestion(EMBED_DIMENSION)
    status_container.write("✅ Vector indexes confirmed")

    total_time = sum(step_times.values())
    status_container.write(f"🎉 Build complete! Total: {total_time:.1f}s")

    result["step_times"]  = step_times
    result["all_triples"] = all_triples
    return result


# ── Lens graph data builder ───────────────────────────────────
def build_lens_graph_data(lens_name: str) -> dict:
    """
    Build node/edge data for a lens-filtered graph visualisation.
    Returns data ready for rendering as an adjacency list + triples.
    """
    triples = get_all_triples_for_lens(lens_name, limit=200)

    nodes   = {}   # name -> {type, connections, evidence}
    edges   = []

    for t in triples:
        subj = t["subject"]
        obj  = t["object"]
        rel  = t["relation"]
        conf = t.get("confidence", 0)

        # Register nodes
        if subj not in nodes:
            nodes[subj] = {
                "type":        t.get("subject_type", "?"),
                "outgoing":    [],
                "incoming":    [],
                "source_text": t.get("source_text", ""),
            }
        if obj not in nodes:
            nodes[obj] = {
                "type":        t.get("object_type", "?"),
                "outgoing":    [],
                "incoming":    [],
                "source_text": t.get("source_text", ""),
            }

        nodes[subj]["outgoing"].append((rel, obj, conf))
        nodes[obj]["incoming"].append((subj, rel, conf))

        edges.append({
            "from":       subj,
            "to":         obj,
            "relation":   rel,
            "confidence": conf,
            "evidence":   t.get("evidence", ""),
            "lens":       t.get("lens", "?"),
            "source_text":t.get("source_text", ""),
        })

    return {"nodes": nodes, "edges": edges, "raw_triples": triples}


# ── Session state ─────────────────────────────────────────────
DEFAULTS = {
    "messages":         [],
    "graph_built":      False,
    "doc_count":        0,
    "chunk_count":      0,
    "build_stats":      {},
    "active_lenses":    ACTIVE_LENSES,
    "pending_question": None,
    "top_k":            SIMILARITY_TOP_K,

    "uploaded_file_paths": [],   # paths of files uploaded this session
    "doc_scan_result":     None, # result of last directory scan

}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

ensure_registry_exists()


llm, embed_model = init_models()



# --- AUTO-LOAD EXISTING GRAPH ON STARTUP ---
if not st.session_state.graph_built:
    # Check if Memgraph already has registered documents
    existing_docs = get_all_registered_docs()
    if existing_docs:
        st.session_state.graph_built = True
        st.session_state.doc_count = len(existing_docs)
        
        # Pre-fetch graph stats so the dashboard populates instantly
        if "graph_stats" not in st.session_state:
            st.session_state.graph_stats = get_graph_stats()
# -------------------------------------------



# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("## 🧠 Knowledge Graph RAG")
    st.markdown("---")

    # ── Config info ───────────────────────────────────────────
    st.markdown("### ⚙️ Configuration")
    st.info(
        f"**LLM:** {LLM_MODEL}\n\n"
        f"**Embeddings:** {EMBED_MODEL} ({EMBED_DIMENSION}d)"
    )
    st.markdown("---")

    # ── Data source ───────────────────────────────────────────
    st.markdown("### 📁 Data Source")
    # Hardcode the directory to remove UI redundancy
    data_dir = "./data"

    st.markdown("### 📁 Manage Documents")
    st.caption("Upload files to automatically add them to the graph.")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "docx", "md", "csv"],
        accept_multiple_files=True,
        key="file_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        # Prevent infinite loop by checking if we've already processed these exact files
        current_upload_names = [f.name for f in uploaded_files]
        previous_upload_paths = st.session_state.get("uploaded_file_paths", [])
        previous_upload_names = [Path(p).name for p in previous_upload_paths]

        if current_upload_names != previous_upload_names:
            saved_paths = []

            for uf in uploaded_files:
                saved = save_uploaded_file(uf, data_dir)
                if saved:
                    saved_paths.append(saved)

            if saved_paths:
                st.session_state.uploaded_file_paths = saved_paths
                st.success(f"**{len(saved_paths)}** file(s) ready to add")

                # Auto-scan immediately after a NEW upload
                st.session_state.doc_scan_result = get_new_documents(
                    data_dir,
                    uploaded_files=saved_paths
                )

                # Safe rerun (won't loop unless new files are uploaded)
                st.rerun()

    st.markdown("---")

    # ── Stakeholder lenses ────────────────────────────────────
    st.markdown("### 🔭 Stakeholder Lenses")
    selected_lenses = []
    for lname, lens in STAKEHOLDER_LENSES.items():
        icon = LENS_ICONS.get(lname, "⚪")
        if st.checkbox(
            f"{icon} {lens.name}",
            value=lname in (st.session_state.active_lenses or ACTIVE_LENSES),
            help=lens.description,
            key=f"lens_{lname}",
        ):
            selected_lenses.append(lname)
    if not selected_lenses:
        st.warning("Select at least one lens!")
    st.session_state.active_lenses = selected_lenses

    st.markdown("---")

    # ── Build settings ────────────────────────────────────────
    st.markdown("### 🏗️ Build Settings")
    c1, c2 = st.columns(2)
    reset_db = c1.checkbox("Reset DB", value=True,
                           help="Wipe graph and rebuild from scratch")
    top_k    = c2.slider("Top K", 1, 20, SIMILARITY_TOP_K)
    st.session_state.top_k = top_k

    # ── BUILD BUTTONS ─────────────────────────────────────────
    # Full rebuild button
    if st.button(
        "🚀 Build Full Graph",
        type="primary",
        use_container_width=True,
        help="Reset database and build from all documents",
    ):
        if not selected_lenses:
            st.error("Select at least one lens!")
        else:
            with st.status(
                "Building Knowledge Graph...", expanded=True
            ) as status:
                try:
                    build_result = run_directed_build_pipeline(
                        data_dir=data_dir,
                        llm=llm,
                        embed_model=embed_model,
                        active_lenses=selected_lenses,
                        reset_db=reset_db,
                        status_container=status,
                    )
                    st.session_state.doc_count   = build_result["doc_count"]
                    st.session_state.chunk_count = build_result["chunk_count"]
                    st.session_state.build_stats = build_result
                    st.session_state.graph_built = True
                    st.session_state.pop("graph_stats", None)
                    st.session_state.doc_scan_result = None  # Reset scan

                    # Register all docs from data_dir
                    _register_all_existing(data_dir, build_result, selected_lenses)

                    status.update(
                        label="✅ Knowledge Graph Ready!", state="complete"
                    )
                except Exception as e:
                    status.update(label="❌ Build Failed", state="error")
                    st.error(f"Error: {e}")
                    st.code(traceback.format_exc())

    # ── Incremental update button ─────────────────────────────
    # Only show if graph is built and there are new files
    scan        = st.session_state.get("doc_scan_result")
    new_files   = (scan or {}).get("new_files", [])
    changed_files = (scan or {}).get("changed_files", [])
    files_to_add  = new_files + changed_files

    if st.session_state.graph_built and files_to_add:
        st.markdown("---")
        st.markdown(
            f"### ➕ Add to Existing Graph\n"
            f"**{len(files_to_add)}** file(s) ready to add:"
        )
        for f in files_to_add:
            badge = "🆕" if f in new_files else "♻️"
            st.caption(f"  {badge} {f['name']} ({f['size_kb']} KB)")

        if st.button(
            f"➕ Update Graph ({len(files_to_add)} file(s))",
            type="secondary",
            use_container_width=True,
            help="Add new documents to existing graph — does NOT reset database",
        ):
            if not selected_lenses:
                st.error("Select at least one lens!")
            else:
                file_paths = [f["path"] for f in files_to_add]
                with st.status(
                    "Updating Knowledge Graph...", expanded=True
                ) as status:
                    try:
                        inc_result = run_incremental_build(
                            new_file_paths=file_paths,
                            llm=llm,
                            embed_model=embed_model,
                            active_lenses=selected_lenses,
                            status_container=status,
                        )

                        if "error" in inc_result:
                            status.update(
                                label=f"❌ {inc_result['error']}",
                                state="error",
                            )
                        else:
                            # Update session stats
                            bs = st.session_state.build_stats
                            bs["total_triples"] = (
                                bs.get("total_triples", 0)
                                + inc_result.get("total_new_triples", 0)
                            )
                            bs["chunk_count"] = (
                                bs.get("chunk_count", 0)
                                + inc_result.get("new_chunks", 0)
                            )
                            st.session_state.build_stats = bs
                            st.session_state.pop("graph_stats", None)
                            st.session_state.doc_scan_result = None

                            status.update(
                                label=(
                                    f"✅ Graph Updated! "
                                    f"+{inc_result.get('total_new_triples',0)} "
                                    f"triples, "
                                    f"+{inc_result.get('new_chunks',0)} chunks"
                                ),
                                state="complete",
                            )
                            st.rerun()
                    except Exception as e:
                        status.update(label="❌ Update Failed", state="error")
                        st.error(f"Error: {e}")
                        st.code(traceback.format_exc())

    st.markdown("---")

    # ── Graph statistics ──────────────────────────────────────
    if st.session_state.graph_built:
        st.markdown("### 📊 Graph Statistics")
        if st.button("🔄 Refresh Stats", use_container_width=True):
            st.session_state.pop("graph_stats", None)
        if "graph_stats" not in st.session_state:
            st.session_state.graph_stats = get_graph_stats()

        gs = st.session_state.graph_stats
        if "error" not in gs:
            c1, c2, c3 = st.columns(3)
            c1.metric("Nodes",     gs.get("total_nodes", 0))
            c2.metric("Relations", gs.get("total_relationships", 0))
            c3.metric("Chunks",    gs.get("chunks", 0))
            st.metric("🏷️ Entities", gs.get("entities", 0))

            with st.expander("By Lens"):
                for lens, count in gs.get("by_lens", {}).items():
                    icon = get_lens_icon(lens)
                    label = (
                        "multi: " + " + ".join(
                            p.title() for p in lens[6:].split(",")
                        ) if lens.startswith("multi:")
                        else lens.replace("_", " ").title()
                    )
                    st.text(f"  {icon} {label}: {count}")

            with st.expander("Entity Types"):
                for et, cnt in gs.get("entity_type_counts", {}).items():
                    st.text(f"  {et}: {cnt}")

            with st.expander("Relation Types"):
                for rt, cnt in gs.get("relationship_types", {}).items():
                    st.text(f"  {rt}: {cnt}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages         = []
        st.session_state.pending_question = None
        st.rerun()


# ================================================================
# MAIN AREA
# ================================================================
st.markdown(
    '<p class="main-header">🧠 Knowledge Graph RAG</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#888">Directed Graph · Stakeholder Lenses · '
    'Organic Relations · Source Text</p>',
    unsafe_allow_html=True,
)

if st.session_state.graph_built:
    s   = st.session_state.build_stats
    ext = s.get("extraction_summary", {})
    org = s.get("organic_summary",   {})
    st.success(
        f"✅ Graph Active | {s.get('total_triples', 0)} triples | "
        f"{ext.get('unique_entities', 0)} entities | "
        f"{s.get('chunk_count', 0)} chunks | "
        f"{org.get('total_organic', 0)} organic | "
        f"{LLM_MODEL}"
    )
    if ext.get("by_lens"):
        cols = st.columns(len(ext["by_lens"]))
        for i, (lns, cnt) in enumerate(ext["by_lens"].items()):
            icon = get_lens_icon(lns)
            cols[i].metric(f"{icon} {lns}", cnt)
else:
    st.warning(
        "⚠️ No knowledge graph built yet. "
        "Use the sidebar to load documents and build the graph."
    )

st.markdown("---")

# ================================================================
# TABS
# ================================================================
tab_chat, tab_explorer, tab_lenses, tab_compare, tab_chunks, tab_docs, tab_debug = st.tabs([
    "💬 Chat",
    "🔍 Graph Explorer",
    "🔭 Lens View & Q&A",
    "🔬 Lens Comparison",
    "📄 Source Chunks",
    "📂 Manage Documents",
    "🐛 Debug",
])


# ================================================================
# TAB 1: CHAT
# ================================================================
with tab_chat:
    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "metadata" in msg:
                meta = msg["metadata"]
                timing = meta.get("timing", {})
                with st.expander(
                    f"📊 Details — {timing.get('total_s','?')}s total | "
                    f"{len(meta.get('triples',[]))} triples | "
                    f"{len(meta.get('source_chunks',[]))} source chunks"
                ):
                    # Timing row
                    t_cols = st.columns(5)
                    for col, (k, label) in zip(
                        t_cols,
                        [
                            ("embed_s",        "Embed"),
                            ("entity_search_s","Entity Search"),
                            ("chunk_search_s", "Chunk Search"),
                            ("fetch_s",        "Fetch"),
                            ("llm_s",          "LLM"),
                        ],
                    ):
                        col.metric(label, f"{timing.get(k,'?')}s")

                    # Entities found
                    if meta.get("entities_found"):
                        st.markdown("**🔵 Entities matched by vector search:**")
                        ent_cols = st.columns(3)
                        for i, ent in enumerate(meta["entities_found"]):
                            dist = ent.get("distance", "?")
                            dist_str = f"{dist:.4f}" if isinstance(dist, float) else str(dist)
                            ent_cols[i % 3].caption(
                                f"• **{ent['name']}** "
                                f"({ent.get('entity_type','?')}) "
                                f"dist={dist_str}"
                            )

                    # Knowledge graph triples
                    if meta.get("triples"):
                        st.markdown("**🔗 Retrieved knowledge graph triples:**")
                        for t in meta["triples"][:15]:
                            conf = t.get("confidence", 0)
                            icon = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.6 else "🔴"
                            lens = t.get("lens", "?")
                            l_icon = get_lens_icon(lens)
                            st.markdown(
                                f"{icon} **{t['subject']}** "
                                f"—[{t['relation']}]→ "
                                f"**{t['object']}** "
                                f"| {conf:.2f} | {l_icon} {lens}"
                            )
                            if t.get("evidence"):
                                st.caption(f"  *{t['evidence'][:100]}*")

                    # Source chunk text
                    if meta.get("source_chunks"):
                        st.markdown("**📄 Source chunk text used:**")
                        for chunk in meta["source_chunks"]:
                            page = chunk.get("page", "")
                            doc  = chunk.get("doc_id", "?")
                            label = f"{doc} (page {page})" if page else doc
                            with st.expander(f"📄 {label}"):
                                st.markdown(
                                    f'<div class="chunk-box">'
                                    f'{chunk.get("text","")[:800]}'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

    # Quick questions
    if st.session_state.graph_built and not st.session_state.messages:
        st.markdown("### 💡 Try these questions:")
        quick_qs = [
            "What is the document about?",
            "What technologies are used?",
            "Who are the key people?",
            "What are the main services offered?",
            "What locations are mentioned?",
            "What are the strategic objectives?",
        ]
        cols = st.columns(3)
        for i, q in enumerate(quick_qs):
            if cols[i % 3].button(
                f"💬 {q}", key=f"quick_{i}", use_container_width=True
            ):
                st.session_state.pending_question = q
                st.rerun()

    # Resolve pending question (from quick buttons)
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            ph = st.empty()
            ph.info("🔍 Searching knowledge graph + source text...")
            try:
                result = run_graph_rag_query(
                    question=question,
                    embed_model=embed_model,
                    llm=llm,
                    top_k=st.session_state.top_k,
                )
                ph.empty()
                st.markdown(result["answer"])
                st.session_state.messages.append({
                    "role":     "assistant",
                    "content":  result["answer"],
                    "metadata": {
                        "timing":         result["timing"],
                        "entities_found": result["entities_found"],
                        "triples":        result["triples"],
                        "source_chunks":  result["source_chunks"],
                    },
                })
            except Exception as e:
                ph.empty()
                st.error(f"❌ {e}")
                st.code(traceback.format_exc())
                logger.exception("Query failed (pending)")
        st.rerun()

    # Chat input
    if prompt := st.chat_input(
        "Ask a question about your documents...",
        disabled=not st.session_state.graph_built,
    ):
        logger.info(f"NEW QUERY: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            ph = st.empty()
            ph.info("🔍 Searching knowledge graph + source text...")
            try:
                result = run_graph_rag_query(
                    question=prompt,
                    embed_model=embed_model,
                    llm=llm,
                    top_k=st.session_state.top_k,
                )
                ph.empty()
                st.markdown(result["answer"])
                st.session_state.messages.append({
                    "role":     "assistant",
                    "content":  result["answer"],
                    "metadata": {
                        "timing":         result["timing"],
                        "entities_found": result["entities_found"],
                        "triples":        result["triples"],
                        "source_chunks":  result["source_chunks"],
                    },
                })
            except Exception as e:
                ph.empty()
                st.error(f"❌ {e}")
                st.code(traceback.format_exc())
                logger.exception("Query failed (chat_input)")


# ================================================================
# TAB 2: GRAPH EXPLORER
# ================================================================
# ── GRAPH EXPLORER TAB: full replacement ─────────────────────
with tab_explorer:
    st.markdown("### 🔍 Explore the Knowledge Graph")

    if not st.session_state.graph_built:
        st.info("Build the knowledge graph first.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Entity Neighbourhood")
            ent_q = st.text_input(
                "Entity name (exact, case-sensitive):",
                key="ent_q",
                placeholder="e.g. HomeSoulAI",
            )
            depth = st.slider("Depth", 1, 3, GRAPH_QUERY_DEPTH, key="nbhd_depth")

            if st.button("🔍 Explore", use_container_width=True, key="btn_explore"):
                if not ent_q.strip():
                    st.warning("Enter an entity name first.")
                else:
                    with st.spinner(f"Querying neighbourhood of '{ent_q}'..."):
                        nbhd  = get_entity_neighborhood(ent_q.strip(), depth=depth)
                        conns = nbhd.get("connections", [])

                        if "error" in nbhd:
                            st.error(f"Graph error: {nbhd['error']}")
                        elif not conns:
                            st.warning(
                                f"No connections found for **{ent_q}**. "
                                "Check the exact entity name in the "
                                "'Source Chunks' or 'Graph Statistics' tabs."
                            )
                        else:
                            st.success(f"Found {len(conns)} connection(s)")
                            for conn in conns[:MAX_CONTEXT_TRIPLES]:
                                direction = conn.get("direction", "outgoing")
                                target    = conn.get("target") or conn.get("source", "?")
                                arrow     = "→" if direction == "outgoing" else "←"
                                conf      = conn.get("confidence")
                                cstr      = f" ({conf:.2f})" if isinstance(conf, float) else ""
                                lens      = conn.get("lens", "?")
                                l_icon    = get_lens_icon(lens)
                                st.markdown(
                                    f"**{ent_q}** {arrow}[{conn.get('relation','?')}]"
                                    f" **{target}**{cstr} {l_icon}"
                                )
                                if conn.get("evidence"):
                                    st.caption(f"  {conn['evidence'][:120]}")

            # ── Entity name helper ────────────────────────────
            with st.expander("💡 Known entity names (click to see)"):
                from graph_setup import get_driver as _get_driver
                try:
                    _drv = _get_driver()
                    with _drv.session() as _sess:
                        _recs = _sess.run(
                            "MATCH (n:__Entity__) "
                            "RETURN n.name AS name, n.entity_type AS type "
                            "ORDER BY n.name LIMIT 50"
                        )
                        _entities = [dict(r) for r in _recs]
                    _drv.close()
                    if _entities:
                        for _e in _entities:
                            st.caption(
                                f"• **{_e['name']}** ({_e.get('type','?')})"
                            )
                    else:
                        st.caption("No entities found yet.")
                except Exception as _ex:
                    st.caption(f"Could not load entity list: {_ex}")

        with c2:
            st.markdown("#### Path Between Entities")
            ea = st.text_input(
                "Entity A:",
                key="path_a",
                placeholder="e.g. HomeSoulAI",
            )
            eb = st.text_input(
                "Entity B:",
                key="path_b",
                placeholder="e.g. Residents",
            )
            max_hops = st.slider("Max hops", 2, 6, 4, key="max_hops")

            if st.button(
                "🔗 Find Path", use_container_width=True, key="btn_path"
            ):
                if not ea.strip() or not eb.strip():
                    st.warning("Enter both Entity A and Entity B.")
                elif ea.strip() == eb.strip():
                    st.warning("Entity A and B must be different.")
                else:
                    with st.spinner(
                        f"Finding path: {ea} → {eb} (max {max_hops} hops)..."
                    ):
                        paths = get_path_between_entities(
                            ea.strip(), eb.strip(), max_hops=max_hops
                        )
                        if paths:
                            st.success(f"Found {len(paths)} path(s)")
                            for p in paths:
                                st.code(p["path"], language=None)
                                st.caption(f"{p['hop_count']} hop(s)")
                        else:
                            st.warning(
                                f"No path found between **{ea}** and **{eb}** "
                                f"within {max_hops} hops. "
                                "Try increasing max hops or check entity names."
                            )

        st.markdown("---")
        st.markdown("#### 🏆 High-Confidence Triples")
        min_conf = st.slider(
            "Min Confidence", 0.5, 1.0, 0.8, 0.05, key="conf_slider"
        )
        if st.button(
            "📋 Show High-Confidence Triples",
            use_container_width=True,
            key="btn_hc",
        ):
            with st.spinner("Querying..."):
                hc = get_high_confidence_triples(
                    min_confidence=min_conf, limit=30
                )
                if hc:
                    st.success(f"Found {len(hc)} triple(s) with conf ≥ {min_conf}")
                    for t in hc:
                        conf   = t.get("confidence", 0)
                        c_icon = "🟢" if conf >= 0.8 else "🟡"
                        lens   = t.get("lens", "?")
                        l_icon = get_lens_icon(lens)
                        st.markdown(
                            f"{c_icon} **{t['subject']}** "
                            f"—[{t['relation']}]→ "
                            f"**{t['object']}** | {conf:.2f} "
                            f"| {l_icon} {lens}"
                        )
                        if t.get("evidence"):
                            st.caption(f"  *{t['evidence'][:100]}*")
                else:
                    st.info(
                        f"No triples found with confidence ≥ {min_conf}. "
                        "Try lowering the threshold."
                    )


# ================================================================
# TAB 3: LENS VIEW & Q&A
# ================================================================
with tab_lenses:
    st.markdown("### 🔭 Lens View & Q&A")
    st.markdown(
        "Explore the knowledge graph from a specific stakeholder's perspective "
        "and ask questions filtered to that lens."
    )

    if not st.session_state.graph_built:
        st.info("Build the knowledge graph first.")
    else:
        # Lens selector
        sel_lens = st.selectbox(
            "Select Stakeholder Lens",
            options=list(STAKEHOLDER_LENSES.keys()),
            format_func=lambda x: (
                f"{get_lens_icon(x)} {STAKEHOLDER_LENSES[x].name}"
            ),
            key="lens_selector",
        )
        lens_info = STAKEHOLDER_LENSES[sel_lens]

        # Lens metadata
        col_desc, col_types = st.columns([2, 1])
        with col_desc:
            color = LENS_COLORS.get(sel_lens, "#333")
            st.markdown(
                f'<div style="border-left:4px solid {color}; '
                f'padding:8px 12px; border-radius:4px; '
                f'background:#0d1117; margin-bottom:8px;">'
                f'<b>{lens_info.name}</b>: {lens_info.description}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with col_types:
            with st.expander("Entity & Relation Types"):
                st.markdown("**Entity Types:**")
                for et in lens_info.entity_types:
                    st.caption(f"  • {et.value}")
                st.markdown("**Relation Types:**")
                for rt in lens_info.relationship_types:
                    st.caption(f"  • {rt.value}")

        st.markdown("---")

        # ── Sub-tabs: Graph | Q&A ─────────────────────────────
        lens_graph_tab, lens_qa_tab = st.tabs([
            "📊 Lens Graph", "💬 Lens Q&A"
        ])

        # ── Lens Graph ────────────────────────────────────────
        # ── LENS VIEW TAB: replace the lens_graph_tab section ────────
        with lens_graph_tab:
            st.markdown(
                f"#### {get_lens_icon(sel_lens)} "
                f"{lens_info.name} Knowledge Graph"
            )

            # ── Check if this lens was used in the build ──────────────
            built_lenses = st.session_state.active_lenses or []
            lens_was_built = sel_lens in built_lenses

            if not lens_was_built:
                st.warning(
                    f"⚠️ **{lens_info.name}** lens was **not selected** during "
                    f"the last build. Rebuild with this lens enabled to see "
                    f"its graph. Current build lenses: "
                    f"{', '.join(built_lenses) or 'none'}"
                )
            else:
                if st.button(
                    f"🔭 Load {lens_info.name} Graph",
                    key="load_lens_graph",
                    use_container_width=True,
                ):
                    with st.spinner(f"Loading {lens_info.name} subgraph..."):
                        graph_data = build_lens_graph_data(sel_lens)

                    nodes   = graph_data["nodes"]
                    edges   = graph_data["edges"]
                    triples = graph_data["raw_triples"]

                    if not nodes:
                        st.warning(
                            f"No triples found for **{lens_info.name}** lens. "
                            f"The lens was selected during build but extracted 0 "
                            f"valid triples from the document. "
                            f"Check the Debug tab for extraction details."
                        )
                    else:
                        # ── Metrics ───────────────────────────────────
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Entities",  len(nodes))
                        m2.metric("Relations", len(edges))
                        avg_conf = (
                            sum(e["confidence"] for e in edges) / len(edges)
                            if edges else 0
                        )
                        m3.metric("Avg Confidence", f"{avg_conf:.2f}")

                        # ── Graph structure ───────────────────────────
                        st.markdown("#### 🗺️ Graph Structure")
                        by_subject: Dict = defaultdict(list)
                        for edge in edges:
                            by_subject[edge["from"]].append(edge)

                        color = LENS_COLORS.get(sel_lens, "#4CAF50")
                        for subj, subj_edges in by_subject.items():
                            node_info = nodes.get(subj, {})
                            st.markdown(
                                f'<div style="border-left:4px solid {color};'
                                f'padding:6px 12px;margin:4px 0;'
                                f'background:#0d1117;border-radius:4px;">'
                                f'<b>🔵 {subj}</b> '
                                f'<span style="color:#888;font-size:0.8rem;">'
                                f'({node_info.get("type","?")})</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            for edge in subj_edges:
                                conf   = edge["confidence"]
                                c_icon = (
                                    "🟢" if conf >= 0.8
                                    else "🟡" if conf >= 0.6
                                    else "🔴"
                                )
                                evid     = edge.get("evidence", "")
                                evid_str = f" *— {evid[:80]}*" if evid else ""
                                obj_type = nodes.get(edge["to"], {}).get("type", "?")
                                st.markdown(
                                    f"&nbsp;&nbsp;&nbsp;&nbsp;"
                                    f"{c_icon} ─[**{edge['relation']}**]─▶ "
                                    f"**{edge['to']}** ({obj_type}) "
                                    f"conf:{conf:.2f}{evid_str}"
                                )

                        # ── Memgraph Lab Cypher ───────────────────────
                        st.markdown("---")
                        st.markdown("#### 📋 Copy to Memgraph Lab")
                        st.code(f"""// {lens_info.name} — entity subgraph
        MATCH (s:__Entity__)-[r]->(o:__Entity__)
        WHERE r.lens = '{sel_lens}'
        OR (r.lens STARTS WITH 'multi:' AND r.lens CONTAINS '{sel_lens}')
        RETURN s, r, o LIMIT 100;""", language="cypher")

                        st.code(f"""// {lens_info.name} — with source chunks
        MATCH (c:__Chunk__)-[:HAS_ENTITY]->(e:__Entity__)
        MATCH (e)-[r]->(:__Entity__)
        WHERE r.lens = '{sel_lens}'
        OR (r.lens STARTS WITH 'multi:' AND r.lens CONTAINS '{sel_lens}')
        RETURN c, e, r LIMIT 100;""", language="cypher")

                        # ── Triples table ─────────────────────────────
                        st.markdown("---")
                        st.markdown("#### 📋 All Triples in This Lens")
                        for t in triples:
                            conf   = t.get("confidence", 0)
                            c_icon = (
                                "🟢" if conf >= 0.8
                                else "🟡" if conf >= 0.6
                                else "🔴"
                            )
                            st.markdown(
                                f"{c_icon} **{t['subject']}** "
                                f"—[{t['relation']}]→ "
                                f"**{t['object']}** | {conf:.2f}"
                            )
                            if t.get("evidence"):
                                st.caption(f"  *Evidence: {t['evidence'][:100]}*")
                            if t.get("source_text"):
                                with st.expander("📄 Source text"):
                                    st.markdown(
                                        f'<div class="chunk-box">'
                                        f'{t["source_text"][:600]}'
                                        f'</div>',
                                        unsafe_allow_html=True,
                                    )


        # ── Lens Q&A ──────────────────────────────────────────
        with lens_qa_tab:
            st.markdown(
                f"#### {get_lens_icon(sel_lens)} "
                f"Ask questions from the {lens_info.name} perspective"
            )
            st.info(
                f"Answers will be filtered to only use triples extracted "
                f"by the **{lens_info.name}** lens, giving you a "
                f"perspective-specific view of the data."
            )

            # Suggested questions per lens
            lens_questions = {
                "executive": [
                    "What are the strategic objectives?",
                    "What markets does the organisation operate in?",
                    "What are the key business outcomes?",
                ],
                "technical": [
                    "What technologies are used?",
                    "What are the key platform integrations?",
                    "What capabilities does the system provide?",
                ],
                "hr_people": [
                    "Who are the key people?",
                    "What roles exist in the organisation?",
                    "What skills are required?",
                ],
                "operations": [
                    "What processes are described?",
                    "What locations are involved?",
                    "What are the key operational metrics?",
                ],
                "client_market": [
                    "Who are the main clients?",
                    "What services are offered to clients?",
                    "What industries are targeted?",
                ],
            }

            suggested = lens_questions.get(sel_lens, [])
            if suggested:
                st.markdown("**💡 Suggested questions for this lens:**")
                sq_cols = st.columns(len(suggested))
                for i, sq in enumerate(suggested):
                    if sq_cols[i].button(
                        sq, key=f"sq_{sel_lens}_{i}", use_container_width=True
                    ):
                        st.session_state[f"lens_qa_{sel_lens}"] = sq
                        st.rerun()

            # Lens Q&A input
            lens_qa_key = f"lens_qa_{sel_lens}"
            prefill = st.session_state.get(lens_qa_key, "")

            lens_question = st.text_input(
                f"Ask a question (filtered to {lens_info.name} lens):",
                value=prefill,
                key=f"lens_q_input_{sel_lens}",
            )

            if st.button(
                f"🔍 Ask {lens_info.name}",
                key=f"ask_lens_{sel_lens}",
                use_container_width=True,
                type="primary",
            ):
                # Clear prefill
                if lens_qa_key in st.session_state:
                    del st.session_state[lens_qa_key]

                if not lens_question.strip():
                    st.warning("Enter a question first.")
                else:
                    with st.spinner(
                        f"Searching {lens_info.name} perspective..."
                    ):
                        try:
                            result = run_graph_rag_query(
                                question=lens_question,
                                embed_model=embed_model,
                                llm=llm,
                                top_k=st.session_state.top_k,
                                lens_filter=sel_lens,
                            )

                            # ── Answer ────────────────────────
                            color = LENS_COLORS.get(sel_lens, "#4CAF50")
                            st.markdown(
                                f'<div style="border-left:4px solid {color};'
                                f'padding:10px 14px;border-radius:6px;'
                                f'background:#0d1117;margin:8px 0;">'
                                f'<b>{get_lens_icon(sel_lens)} '
                                f'{lens_info.name} Answer:</b><br><br>'
                                f'{result["answer"]}'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            # ── Details ───────────────────────
                            timing = result["timing"]
                            with st.expander(
                                f"📊 Details — "
                                f"{timing['total_s']}s | "
                                f"{len(result['triples'])} triples | "
                                f"{len(result['source_chunks'])} chunks"
                            ):
                                t_cols = st.columns(5)
                                for col, (k, lbl) in zip(
                                    t_cols,
                                    [
                                        ("embed_s",        "Embed"),
                                        ("entity_search_s","Entity Search"),
                                        ("chunk_search_s", "Chunk Search"),
                                        ("fetch_s",        "Fetch"),
                                        ("llm_s",          "LLM"),
                                    ],
                                ):
                                    col.metric(lbl, f"{timing.get(k,'?')}s")

                                # ── LENS Q&A TAB ─

                                if result["triples"]:
                                    st.markdown(f"**Triples used ({lens_info.name} lens):**")
                                    for t in result["triples"][:10]:
                                        conf   = t.get("confidence", 0)
                                        c_icon = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.6 else "🔴"
                                        st.markdown(
                                            f"{c_icon} **{t['subject']}** "
                                            f"—[{t['relation']}]→ "
                                            f"**{t['object']}** | {conf:.2f}"
                                        )
                                        if t.get("evidence"):
                                            st.caption(f"  *{t['evidence'][:100]}*")
                                else:
                                    # Distinguish: lens not built vs lens built but no matches
                                    if sel_lens not in (st.session_state.active_lenses or []):
                                        st.warning(
                                            f"⚠️ **{lens_info.name}** was not used during the build. "
                                            f"The answer above is based on source text only (no graph "
                                            f"triples). Rebuild with this lens to get graph-grounded answers."
                                        )
                                    else:
                                        st.info(
                                            f"ℹ️ No **{lens_info.name}** triples matched these entities. "
                                            f"The answer uses organic/hub relations and source text. "
                                            f"Try a more specific question."
                                        )

                                if result["source_chunks"]:
                                    st.markdown("**📄 Source text:**")
                                    for chunk in result["source_chunks"]:
                                        doc  = chunk.get("doc_id", "?")
                                        page = chunk.get("page", "")
                                        lbl  = f"{doc} p.{page}" if page else doc
                                        with st.expander(f"📄 {lbl}"):
                                            st.markdown(
                                                f'<div class="chunk-box">'
                                                f'{chunk.get("text","")[:600]}'
                                                f'</div>',
                                                unsafe_allow_html=True,
                                            )

                        except Exception as e:
                            st.error(f"❌ Lens Q&A error: {e}")
                            st.code(traceback.format_exc())
                            logger.exception("Lens Q&A failed")



# ================================================================
# TAB: LENS COMPARISON
# ================================================================
with tab_compare:
    from lens_comparator import (
        run_lens_query,
        run_unfiltered_query,
        build_comparison_report,
        triple_key as _tkey,
    )

    st.markdown("### 🔬 Lens Comparison Report")
    st.markdown(
        "Ask **one question** and see how each stakeholder lens answers it "
        "differently — what each lens uniquely finds, what they agree on, "
        "and what the unfiltered baseline misses or adds."
    )

    if not st.session_state.graph_built:
        st.info("Build the knowledge graph first.")
    else:
        # ── Question presets ──────────────────────────────────
        st.markdown("#### 1️⃣ Enter Your Question")

        COMPARE_PRESETS = [
            "What does this document describe?",
            "What are the key capabilities?",
            "Who are the main stakeholders?",
            "What are the strategic outcomes?",
            "What technologies or processes are involved?",
        ]

        preset_cols = st.columns(len(COMPARE_PRESETS))
        for i, pq in enumerate(COMPARE_PRESETS):
            if preset_cols[i].button(
                pq, key=f"cmp_preset_{i}", use_container_width=True
            ):
                st.session_state["compare_question"] = pq
                st.rerun()

        compare_q = st.text_input(
            "Question to compare across all lenses:",
            value=st.session_state.get("compare_question", ""),
            placeholder="e.g. What are the main capabilities?",
            key="compare_q_input",
        )

        # ── Lens selector ─────────────────────────────────────
        st.markdown("#### 2️⃣ Select Lenses to Compare")
        built_lenses = st.session_state.active_lenses or []

        cmp_lenses: List[str] = []
        if not built_lenses:
            st.warning("No lenses were used in the build.")
        else:
            lc = st.columns(len(built_lenses))
            for i, lname in enumerate(built_lenses):
                lens_info = STAKEHOLDER_LENSES.get(lname)
                icon      = get_lens_icon(lname)
                label     = lens_info.name if lens_info else lname
                if lc[i].checkbox(
                    f"{icon} {label}",
                    value=True,
                    key=f"cmp_lens_{lname}",
                ):
                    cmp_lenses.append(lname)

            include_baseline = st.checkbox(
                "📊 Include unfiltered baseline (no lens)",
                value=True,
                key="cmp_include_baseline",
                help=(
                    "Run the same question with NO lens filter. "
                    "This is the traditional RAG result for comparison."
                ),
            )

        # ── Run button ────────────────────────────────────────
        st.markdown("#### 3️⃣ Run Comparison")
        run_disabled = (
            not compare_q.strip()
            or not built_lenses
            or not cmp_lenses
        )

        if st.button(
            "🔬 Run Lens Comparison",
            type="primary",
            use_container_width=True,
            disabled=run_disabled,
            key="btn_run_compare",
        ):
            st.session_state["compare_question"] = compare_q
            progress_area  = st.empty()
            results_store: Dict = {}

            # Baseline
            if include_baseline:
                progress_area.info("📊 Running baseline (no lens filter)…")
                baseline_result = run_unfiltered_query(
                    question    = compare_q,
                    embed_model = embed_model,
                    llm         = llm,
                    top_k       = st.session_state.top_k,
                    query_fn    = run_graph_rag_query,
                )
                results_store["no_lens"] = baseline_result
            else:
                baseline_result = {
                    "lens_name": "no_lens", "answer": "",
                    "triples": [], "entities_found": [],
                    "source_chunks": [], "timing": {},
                    "query_time_s": 0, "error": None,
                }

            # Each lens
            lens_results: List[dict] = []
            for li, lname in enumerate(cmp_lenses):
                lens_info = STAKEHOLDER_LENSES.get(lname)
                icon      = get_lens_icon(lname)
                label     = lens_info.name if lens_info else lname
                progress_area.info(
                    f"{icon} Running {label} lens "
                    f"({li+1}/{len(cmp_lenses)})…"
                )
                lr = run_lens_query(
                    question    = compare_q,
                    lens_name   = lname,
                    embed_model = embed_model,
                    llm         = llm,
                    top_k       = st.session_state.top_k,
                    query_fn    = run_graph_rag_query,
                )
                lens_results.append(lr)
                results_store[lname] = lr

            progress_area.empty()

            report = build_comparison_report(
                baseline      = baseline_result,
                lens_results  = lens_results,
                active_lenses = cmp_lenses,
            )

            st.session_state["compare_report"]            = report
            st.session_state["compare_results"]           = results_store
            st.session_state["compare_lenses"]            = cmp_lenses
            st.session_state["compare_baseline_included"] = include_baseline
            st.rerun()

        # ================================================================
        # RENDER REPORT
        # ================================================================
        if "compare_report" not in st.session_state:
            st.stop()

        report        = st.session_state["compare_report"]
        results       = st.session_state["compare_results"]
        cmp_lenses    = st.session_state.get("compare_lenses", [])
        show_baseline = st.session_state.get("compare_baseline_included", True)
        question      = st.session_state.get("compare_question", "")

        st.markdown("---")
        st.markdown("## 📊 Comparison Report")
        st.markdown(f"**Question:** *{question}*")
        st.markdown("---")

        import pandas as pd

        to_  = report.get("triple_overlap",      {})
        cpr  = report.get("confidence_profile",  {})
        uniq = to_.get("unique_per_lens",         {})

        # ============================================================
        # SECTION 1: EXECUTIVE SUMMARY
        # ============================================================
        st.markdown("### 📋 Executive Summary")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Baseline Triples",  to_.get("baseline_count",    0))
        s2.metric(
            "Lens Union Triples",
            to_.get("lens_union_count", 0),
            delta=f"+{to_.get('lens_gained', 0)} vs baseline",
        )
        s3.metric("Consensus Triples", to_.get("consensus_count",   0))
        s4.metric(
            "Lenses Run",
            len(cmp_lenses) + (1 if show_baseline else 0),
        )

        # ── Summary table — all types str to avoid Arrow errors ───
        st.markdown("#### At-a-Glance: What Each Lens Found")

        summary_rows: List[dict] = []

        if show_baseline:
            bl = results.get("no_lens", {})
            bc = cpr.get("no_lens", {})
            summary_rows.append({
                "Lens":           "📊 No Lens (Baseline)",
                "Triples":        str(len(bl.get("triples",       []))),
                "Unique Triples": "N/A",
                "Entities":       str(len(bl.get("entities_found",[]))),
                "Avg Conf":       f"{bc.get('avg', 0):.2f}",
                "Answer Words":   str(
                    report["answer_stats"]
                    .get("no_lens", {})
                    .get("words", 0)
                ),
                "Time (s)":       str(bl.get("query_time_s", 0)),
            })

        for lname in cmp_lenses:
            lr        = results.get(lname, {})
            lc_       = cpr.get(lname, {})
            lens_info = STAKEHOLDER_LENSES.get(lname)
            icon      = get_lens_icon(lname)
            label     = lens_info.name if lens_info else lname
            summary_rows.append({
                "Lens":           f"{icon} {label}",
                "Triples":        str(len(lr.get("triples",       []))),
                "Unique Triples": str(uniq.get(lname,              0)),
                "Entities":       str(len(lr.get("entities_found",[]))),
                "Avg Conf":       f"{lc_.get('avg', 0):.2f}",
                "Answer Words":   str(
                    report["answer_stats"]
                    .get(lname, {})
                    .get("words", 0)
                ),
                "Time (s)":       str(lr.get("query_time_s", 0)),
            })

        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        st.markdown("---")

        # ============================================================
        # SECTION 2: ANSWER COMPARISON
        # ============================================================
        st.markdown("### 💬 Answer Comparison")
        st.markdown(
            "The **same question**, answered from each perspective. "
            "Notice how the focus, vocabulary, and facts differ."
        )

        if show_baseline:
            bl_answer = results.get("no_lens", {}).get("answer", "")
            st.markdown("#### 📊 Baseline Answer (No Lens Filter)")
            st.markdown(
                f'<div style="border-left:4px solid #555;'
                f'padding:10px 14px;border-radius:6px;'
                f'background:#0d1117;margin:8px 0;">'
                f'{bl_answer or "<em>No answer generated.</em>"}'
                f'</div>',
                unsafe_allow_html=True,
            )
            bl_stats = report["answer_stats"].get("no_lens", {})
            st.caption(
                f"Words: {bl_stats.get('words', 0)} | "
                f"Triples used: "
                f"{len(results.get('no_lens', {}).get('triples', []))}"
            )

        if cmp_lenses:
            st.markdown("#### 🔭 Lens Answers")

        for i in range(0, len(cmp_lenses), 2):
            row_lenses = cmp_lenses[i : i + 2]
            cols       = st.columns(len(row_lenses))

            for col, lname in zip(cols, row_lenses):
                lr        = results.get(lname, {})
                lens_info = STAKEHOLDER_LENSES.get(lname)
                icon      = get_lens_icon(lname)
                label     = lens_info.name if lens_info else lname
                color     = LENS_COLORS.get(lname, "#4CAF50")
                answer    = lr.get("answer", "")
                err       = lr.get("error")

                with col:
                    st.markdown(
                        f'<div style="border-left:4px solid {color};'
                        f'padding:10px 14px;border-radius:6px;'
                        f'background:#0d1117;margin:8px 0;min-height:180px;">'
                        f'<b>{icon} {label}</b><br><br>'
                        f'{"❌ " + err if err else (answer or "<em>No answer.</em>")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    t_list = lr.get("triples", [])
                    st.caption(
                        f"Words: "
                        f"{report['answer_stats'].get(lname, {}).get('words', 0)} | "
                        f"Triples: {len(t_list)} | "
                        f"Time: {lr.get('query_time_s', 0)}s"
                    )

        st.markdown("---")

        # ============================================================
        # SECTION 3: UNIQUE INSIGHTS PER LENS
        # ============================================================
        st.markdown("### 🎯 Unique Insights Per Lens")
        st.markdown(
            "Facts that **only this lens found** — invisible to other "
            "lenses and to the unfiltered baseline."
        )

        unique_triples_map = report.get("_unique_triples", {})

        for lname in cmp_lenses:
            lr        = results.get(lname, {})
            lens_info = STAKEHOLDER_LENSES.get(lname)
            icon      = get_lens_icon(lname)
            label     = lens_info.name if lens_info else lname
            color     = LENS_COLORS.get(lname, "#4CAF50")

            unique_keys         = set(unique_triples_map.get(lname, []))
            unique_triple_dicts = [
                t for t in lr.get("triples", [])
                if _tkey(t) in unique_keys
            ]

            with st.expander(
                f"{icon} {label} — "
                f"{len(unique_triple_dicts)} unique fact(s) "
                f"not found by other lenses",
                expanded=True,
            ):
                if unique_triple_dicts:
                    for t in unique_triple_dicts:
                        conf = t.get("confidence", 0)
                        ci   = (
                            "🟢" if conf >= 0.8
                            else "🟡" if conf >= 0.6
                            else "🔴"
                        )
                        st.markdown(
                            f"{ci} **{t['subject']}** "
                            f"—[{t['relation']}]→ "
                            f"**{t['object']}** | {conf:.2f}"
                        )
                        if t.get("evidence"):
                            st.caption(f'  *"{t["evidence"][:120]}"*')
                else:
                    st.info(
                        "No exclusively unique triples for this lens — "
                        "it overlaps with other lenses on this question."
                    )

                # Topic keywords
                tf = report.get("topic_focus", {}).get(lname, [])
                if tf:
                    st.markdown("**Top keywords in this lens's answer:**")
                    st.markdown(
                        " · ".join(f"`{w}` ×{n}" for w, n in tf[:6])
                    )

        st.markdown("---")

        # ============================================================
        # SECTION 4: WHAT BASELINE MISSES
        # ============================================================
        if show_baseline:
            st.markdown("### 🕳️ What the Baseline (No Lens) Misses")
            st.markdown(
                "Facts found by **at least one lens** but "
                "**absent from the unfiltered baseline** — "
                "what you lose without lens filtering."
            )

            baseline_keys_set: set = set(
                _tkey(t)
                for t in results.get("no_lens", {}).get("triples", [])
            )
            all_lens_triples_flat: List[dict] = []
            for lname in cmp_lenses:
                all_lens_triples_flat.extend(
                    results.get(lname, {}).get("triples", [])
                )

            seen_gained:    set         = set()
            unique_gained:  List[dict]  = []
            for t in all_lens_triples_flat:
                k = _tkey(t)
                if k not in baseline_keys_set and k not in seen_gained:
                    seen_gained.add(k)
                    unique_gained.append(t)

            if unique_gained:
                st.warning(
                    f"⚠️ The baseline missed **{len(unique_gained)}** "
                    f"fact(s) that lens filtering reveals:"
                )
                for t in unique_gained[:20]:
                    conf  = t.get("confidence", 0)
                    ci    = (
                        "🟢" if conf >= 0.8
                        else "🟡" if conf >= 0.6
                        else "🔴"
                    )
                    lens  = t.get("lens", "?")
                    st.markdown(
                        f"{ci} **{t['subject']}** "
                        f"—[{t['relation']}]→ "
                        f"**{t['object']}** "
                        f"| {conf:.2f} "
                        f"| found by: {get_lens_icon(lens)} {lens}"
                    )
                    if t.get("evidence"):
                        st.caption(f'  *"{t["evidence"][:100]}"*')
                if len(unique_gained) > 20:
                    st.caption(
                        f"… and {len(unique_gained) - 20} more."
                    )
            else:
                st.success(
                    "✅ Baseline captured all lens-specific facts for "
                    "this question. Lenses provide perspective differences "
                    "in the answer text but not additional triples."
                )

            # What baseline has that lenses don't
            lenses_all_keys: set = set()
            for lname in cmp_lenses:
                for t in results.get(lname, {}).get("triples", []):
                    lenses_all_keys.add(_tkey(t))

            baseline_only_triples = [
                t for t in results.get("no_lens", {}).get("triples", [])
                if _tkey(t) not in lenses_all_keys
            ]

            if baseline_only_triples:
                with st.expander(
                    f"📊 {len(baseline_only_triples)} triple(s) only in "
                    f"baseline (lens filtering excluded these)"
                ):
                    st.caption(
                        "These may be organic/hub relations that span "
                        "multiple lens schemas."
                    )
                    for t in baseline_only_triples[:15]:
                        conf = t.get("confidence", 0)
                        ci   = (
                            "🟢" if conf >= 0.8
                            else "🟡" if conf >= 0.6
                            else "🔴"
                        )
                        st.markdown(
                            f"{ci} **{t['subject']}** "
                            f"—[{t['relation']}]→ "
                            f"**{t['object']}** | {conf:.2f}"
                        )

        st.markdown("---")

        # ============================================================
        # SECTION 5: RELATION TYPE PROFILE  (no matplotlib needed)
        # ============================================================
        st.markdown("### 🔥 Relation Type Profile")
        st.markdown(
            "Which **relation types** does each lens emphasise? "
            "Reveals each lens's structural bias."
        )

        rel_profile = report.get("relation_profile", {})

        all_relations: set = set()
        for rp in rel_profile.values():
            all_relations.update(rp.keys())

        if all_relations:
            lens_order_rel = (
                (["no_lens"] if show_baseline else []) + cmp_lenses
            )

            # Build plain int DataFrame — no styling, no matplotlib
            matrix_rows: List[dict] = []
            for rel in sorted(all_relations):
                row: dict = {"Relation": rel}
                for lname in lens_order_rel:
                    col_label = (
                        "Baseline"
                        if lname == "no_lens"
                        else (
                            STAKEHOLDER_LENSES[lname].name
                            if lname in STAKEHOLDER_LENSES
                            else lname
                        )
                    )
                    row[col_label] = int(
                        rel_profile.get(lname, {}).get(rel, 0)
                    )
                matrix_rows.append(row)

            df_rel = pd.DataFrame(matrix_rows)
            # Keep only rows where at least one count > 0
            numeric_cols = [c for c in df_rel.columns if c != "Relation"]
            df_rel = df_rel[df_rel[numeric_cols].sum(axis=1) > 0]

            st.dataframe(df_rel, use_container_width=True, hide_index=True)

            # Manual "heatmap" using emoji bars
            st.markdown("**Visual breakdown (count per lens):**")
            for _, row in df_rel.iterrows():
                rel_name = row["Relation"]
                parts    = []
                for col in numeric_cols:
                    val = int(row[col])
                    if val > 0:
                        bar   = "█" * min(val, 8)
                        parts.append(f"**{col}**: {bar} {val}")
                if parts:
                    st.markdown(f"`{rel_name}` — " + " | ".join(parts))

            st.caption(
                "Numbers = count of triples with that relation. "
                "Higher = more focus on that relation type."
            )

        st.markdown("---")

        # ============================================================
        # SECTION 6: CONFIDENCE COMPARISON
        # ============================================================
        st.markdown("### 📈 Confidence Profile")
        st.markdown(
            "Lens filtering often **raises** average confidence by "
            "excluding off-schema noise triples."
        )

        conf_rows: List[dict] = []
        lens_order_conf = (
            (["no_lens"] if show_baseline else []) + cmp_lenses
        )
        for lname in lens_order_conf:
            cp    = cpr.get(lname, {})
            label = (
                "📊 Baseline (No Lens)"
                if lname == "no_lens"
                else (
                    f"{get_lens_icon(lname)} "
                    f"{STAKEHOLDER_LENSES[lname].name}"
                    if lname in STAKEHOLDER_LENSES
                    else lname
                )
            )
            conf_rows.append({
                "Lens":      label,
                "Avg Conf":  f"{cp.get('avg',   0):.3f}",
                "Max Conf":  f"{cp.get('max',   0):.3f}",
                "Min Conf":  f"{cp.get('min',   0):.3f}",
                "# Triples": str(cp.get("count", 0)),
            })

        df_conf = pd.DataFrame(conf_rows)
        st.dataframe(df_conf, use_container_width=True, hide_index=True)

        # Visual confidence bars
        st.markdown("**Average confidence bars:**")
        for lname in lens_order_conf:
            cp    = cpr.get(lname, {})
            avg   = cp.get("avg", 0)
            label = (
                "Baseline"
                if lname == "no_lens"
                else (
                    STAKEHOLDER_LENSES[lname].name
                    if lname in STAKEHOLDER_LENSES
                    else lname
                )
            )
            icon  = "📊" if lname == "no_lens" else get_lens_icon(lname)
            bar   = "█" * int(avg * 20)   # scale 0-1 → 0-20 chars
            color = (
                "🟢" if avg >= 0.8
                else "🟡" if avg >= 0.6
                else "🔴"
            )
            st.markdown(f"{icon} **{label}**: {color} {bar} `{avg:.3f}`")

        st.markdown("---")

        # ============================================================
        # SECTION 7: TOPIC FOCUS
        # ============================================================
        st.markdown("### 🗂️ Topic Focus by Lens")
        st.markdown(
            "Top keywords from each lens's answer — "
            "shows what each lens **talks about** vs what others ignore."
        )

        tf_data      = report.get("topic_focus", {})
        lens_order_tf = (
            (["no_lens"] if show_baseline else []) + cmp_lenses
        )
        tf_cols = st.columns(min(len(lens_order_tf), 3))

        for idx, lname in enumerate(lens_order_tf):
            lens_info = STAKEHOLDER_LENSES.get(lname)
            icon      = "📊" if lname == "no_lens" else get_lens_icon(lname)
            label     = (
                "Baseline"
                if lname == "no_lens"
                else (lens_info.name if lens_info else lname)
            )
            color = (
                "#888"
                if lname == "no_lens"
                else LENS_COLORS.get(lname, "#4CAF50")
            )

            with tf_cols[idx % len(tf_cols)]:
                st.markdown(
                    f'<span style="color:{color};font-weight:bold;">'
                    f'{icon} {label}</span>',
                    unsafe_allow_html=True,
                )
                kws = tf_data.get(lname, [])
                if kws:
                    for word, count in kws[:6]:
                        bar = "█" * min(count, 12)
                        st.markdown(f"`{word}` {bar} ×{count}")
                else:
                    st.caption("No answer text available.")

        st.markdown("---")

        # ============================================================
        # SECTION 8: KEY INSIGHTS
        # ============================================================
        st.markdown("### 💡 Key Insights")

        # Most unique lens
        if uniq:
            most_unique_lens  = max(uniq, key=lambda x: uniq[x])
            most_unique_count = uniq[most_unique_lens]
            lmi               = STAKEHOLDER_LENSES.get(most_unique_lens)
            lml               = lmi.name if lmi else most_unique_lens
            if most_unique_count > 0:
                st.success(
                    f"🏆 **Most unique insights:** "
                    f"{get_lens_icon(most_unique_lens)} **{lml}** "
                    f"found **{most_unique_count}** fact(s) that "
                    f"no other lens found."
                )

        # Consensus
        cons = to_.get("consensus_count", 0)
        if cons > 0:
            st.info(
                f"🤝 **{cons}** triple(s) agreed on by ALL lenses — "
                f"these are the most reliable facts in the document."
            )
        elif len(cmp_lenses) > 1:
            st.warning(
                "⚠️ No consensus triples — each lens sees completely "
                "different aspects of the document for this question."
            )

        # Baseline gain/loss
        if show_baseline:
            gained = to_.get("lens_gained",    0)
            lost   = to_.get("baseline_only",  0)
            if gained > 0:
                st.warning(
                    f"🔍 Lens filtering **reveals {gained} additional "
                    f"fact(s)** not surfaced by the unfiltered baseline."
                )
            if lost > 0:
                st.info(
                    f"📊 The baseline contains **{lost}** triple(s) that "
                    f"lens filtering excludes (likely organic/hub relations)."
                )
            if gained == 0 and lost == 0:
                st.info(
                    "📊 Baseline and lenses use the same triple set. "
                    "The difference is in **how the LLM phrases the answer** "
                    "based on which lens perspective it adopts."
                )

        # Confidence improvement
        conf_avgs = {
            lname: cpr.get(lname, {}).get("avg", 0)
            for lname in cmp_lenses
        }
        if conf_avgs:
            best_lens  = max(conf_avgs, key=lambda x: conf_avgs[x])
            best_val   = conf_avgs[best_lens]
            bli        = STAKEHOLDER_LENSES.get(best_lens)
            bll        = bli.name if bli else best_lens
            bl_avg     = cpr.get("no_lens", {}).get("avg", 0)
            if show_baseline and best_val > bl_avg and bl_avg > 0:
                st.success(
                    f"📈 **{get_lens_icon(best_lens)} {bll}** has the "
                    f"highest average confidence ({best_val:.3f}) vs "
                    f"baseline ({bl_avg:.3f}) — lens focus improves "
                    f"extraction quality."
                )

        st.markdown("---")

        # ============================================================
        # SECTION 9: EXPORT
        # ============================================================
        with st.expander("📥 Export Raw Comparison Data"):
            import json as _json

            export_obj = {
                "question":           question,
                "lenses_compared":    cmp_lenses,
                "baseline_included":  show_baseline,
                "summary": {
                    "triple_overlap":     to_,
                    "confidence_profile": {
                        k: v for k, v in cpr.items()
                    },
                    "answer_stats":       report.get("answer_stats", {}),
                    "unique_per_lens":    uniq,
                },
                "answers": {
                    lname: results.get(lname, {}).get("answer", "")
                    for lname in (
                        (["no_lens"] if show_baseline else []) + cmp_lenses
                    )
                },
            }

            st.download_button(
                label     = "⬇️ Download JSON Report",
                data      = _json.dumps(export_obj, indent=2),
                file_name = f"lens_comparison_{int(time.time())}.json",
                mime      = "application/json",
                key       = "btn_download_report",
            )
            st.json(export_obj)



# ================================================================
# TAB 4: SOURCE CHUNKS
# ================================================================
with tab_chunks:
    st.markdown("### 📄 Source Document Chunks")
    st.markdown(
        "Browse the original document chunks stored in Memgraph "
        "with their embeddings and entity links."
    )

    if not st.session_state.graph_built:
        st.info("Build the knowledge graph first.")
    else:
        from graph_setup import get_driver

        if st.button("📥 Load All Chunks", use_container_width=True):
            driver = get_driver()
            try:
                with driver.session() as session:
                    records = session.run("""
                        MATCH (c:__Chunk__)
                        OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
                        WITH c,
                             collect(e.name + ' (' + e.entity_type + ')') AS entities
                        RETURN
                            c.chunk_id   AS chunk_id,
                            c.text       AS text,
                            c.doc_id     AS doc_id,
                            c.page_label AS page,
                            c.char_count AS chars,
                            entities
                        ORDER BY c.doc_id, c.page_label
                    """)
                    chunk_rows = [dict(r) for r in records]
            except Exception as e:
                st.error(f"Error loading chunks: {e}")
                chunk_rows = []
            finally:
                driver.close()

            if chunk_rows:
                st.success(f"Found {len(chunk_rows)} chunk(s) in Memgraph")
                for i, chunk in enumerate(chunk_rows):
                    doc  = chunk.get("doc_id", "?")
                    page = chunk.get("page", "")
                    chars = chunk.get("chars", 0)
                    ents  = chunk.get("entities", [])
                    cid   = (chunk.get("chunk_id") or "")[:8]

                    header = (
                        f"📄 Chunk {i+1} | {doc}"
                        f"{' p.'+str(page) if page else ''}"
                        f" | {chars} chars | {len(ents)} entities"
                        f" | id: {cid}"
                    )
                    with st.expander(header, expanded=(i == 0)):
                        # Chunk text
                        st.markdown("**Full Text:**")
                        st.markdown(
                            f'<div class="chunk-box">'
                            f'{chunk.get("text","")}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        # Linked entities
                        if ents:
                            st.markdown("**Linked Entities:**")
                            ent_cols = st.columns(3)
                            for j, ent in enumerate(ents):
                                ent_cols[j % 3].caption(f"• {ent}")
                        else:
                            st.caption("No entities linked to this chunk")
            else:
                st.warning("No chunks found in Memgraph")

        # Memgraph Lab Cypher for chunks
        st.markdown("---")
        st.markdown("#### 📋 View Chunks in Memgraph Lab")
        st.code("""// All chunks with their linked entities
MATCH (c:__Chunk__)
OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:__Entity__)
RETURN c, e
LIMIT 100;""", language="cypher")

        st.code("""// Full graph: chunks → entities → relations
MATCH (c:__Chunk__)-[:HAS_ENTITY]->(e:__Entity__)-[r]->(o:__Entity__)
RETURN c, e, r, o
LIMIT 100;""", language="cypher")




# ================================================================
# TAB: MANAGE DOCUMENTS
# ================================================================
with tab_docs:
    st.markdown("### 📂 Document Registry & Incremental Updates")
    st.markdown(
        "View all documents in the knowledge graph. "
        "Add new documents without rebuilding from scratch."
    )

    # ── Registry summary ──────────────────────────────────────
    reg_summary = get_registry_summary()
    total_docs    = reg_summary.get("total_docs", 0)
    total_chunks  = reg_summary.get("total_chunks", 0)
    total_triples = reg_summary.get("total_triples", 0)

    m1, m2, m3 = st.columns(3)
    m1.metric("📄 Documents in Graph", total_docs)
    m2.metric("🧩 Total Chunks",       total_chunks)
    m3.metric("🔗 Total Triples",      total_triples)

    st.markdown("---")

    # ── Upload new documents ──────────────────────────────────
    st.markdown("#### 📤 Add New Documents")

    col_upload, col_info = st.columns([2, 1])
    with col_upload:
        new_uploads = st.file_uploader(
            "Upload documents to add to the graph:",
            type=["pdf", "txt", "docx", "md", "csv"],
            accept_multiple_files=True,
            key="tab_uploader",
        )

    with col_info:
        st.info(
            "**Supported formats:**\n"
            "PDF, TXT, DOCX, MD, CSV\n\n"
            "**What happens:**\n"
            "Only new files are processed.\n"
            "Existing graph is preserved."
        )

    if new_uploads:
        st.markdown("**Files ready to process:**")
        upload_paths = []
        for uf in new_uploads:
            # Check if already ingested
            temp_path = Path(data_dir) / uf.name
            existing  = None
            if temp_path.exists():
                from doc_registry import is_document_ingested
                existing = is_document_ingested(str(temp_path))

            status_icon = "✅ Already in graph" if existing else "🆕 New"
            st.markdown(
                f"{'✅' if existing else '🆕'} **{uf.name}** "
                f"({uf.size / 1024:.1f} KB) — {status_icon}"
            )
            if existing:
                st.caption(
                    f"  Last ingested: {existing.get('ingested_at','?')[:10]} "
                    f"| {existing.get('chunk_count',0)} chunks "
                    f"| lenses: {existing.get('lenses_used','?')}"
                )
            upload_paths.append(uf)

        # Filter to only new files
        truly_new = []
        for uf in new_uploads:
            temp_path = Path(data_dir) / uf.name
            # Save to disk first to check hash
            saved = save_uploaded_file(uf, data_dir)
            if saved:
                from doc_registry import is_document_ingested
                existing = is_document_ingested(saved)
                if not existing:
                    truly_new.append(saved)

        if truly_new:
            st.success(f"**{len(truly_new)}** new document(s) ready to add")

            # Lens selection for incremental
            st.markdown("**Select lenses for extraction:**")
            inc_lenses = []
            lens_cols  = st.columns(len(STAKEHOLDER_LENSES))
            for i, (lname, lens) in enumerate(STAKEHOLDER_LENSES.items()):
                icon = LENS_ICONS.get(lname, "⚪")
                if lens_cols[i].checkbox(
                    f"{icon}",
                    value=lname in (st.session_state.active_lenses or ACTIVE_LENSES),
                    help=f"{lens.name}: {lens.description}",
                    key=f"inc_lens_{lname}",
                ):
                    inc_lenses.append(lname)

            if st.button(
                f"➕ Add {len(truly_new)} Document(s) to Graph",
                type="primary",
                use_container_width=True,
                disabled=not st.session_state.graph_built,
            ):
                if not st.session_state.graph_built:
                    st.error("Build the initial graph first!")
                elif not inc_lenses:
                    st.error("Select at least one lens!")
                else:
                    with st.status(
                        "Adding documents to graph...", expanded=True
                    ) as status:
                        try:
                            inc_result = run_incremental_build(
                                new_file_paths=truly_new,
                                llm=llm,
                                embed_model=embed_model,
                                active_lenses=inc_lenses,
                                status_container=status,
                            )
                            if "error" in inc_result:
                                status.update(
                                    label=f"❌ {inc_result['error']}",
                                    state="error",
                                )
                            else:
                                # Update session state
                                bs = st.session_state.build_stats
                                bs["total_triples"] = (
                                    bs.get("total_triples", 0)
                                    + inc_result.get("total_new_triples", 0)
                                )
                                st.session_state.build_stats = bs
                                st.session_state.pop("graph_stats", None)

                                status.update(
                                    label=(
                                        f"✅ Added! "
                                        f"+{inc_result['total_new_triples']} "
                                        f"triples, "
                                        f"+{inc_result['new_chunks']} chunks"
                                    ),
                                    state="complete",
                                )
                                st.rerun()
                        except Exception as e:
                            status.update(label="❌ Failed", state="error")
                            st.error(f"Error: {e}")
                            st.code(traceback.format_exc())
        elif new_uploads:
            st.info(
                "All uploaded files are already in the graph. "
                "No new processing needed."
            )

    st.markdown("---")

    # ── Registered documents list ─────────────────────────────
    st.markdown("#### 📋 Documents Currently in Graph")

    if st.button("🔄 Refresh Registry", use_container_width=True):
        st.rerun()

    reg_docs = get_all_registered_docs()

    if not reg_docs:
        st.info(
            "No documents registered yet. "
            "Build the graph first or check that the registry is initialised."
        )
    else:
        for i, doc in enumerate(reg_docs):
            ingested_date = (doc.get("ingested_at") or "")[:10]
            lenses_str    = ", ".join(doc.get("lenses_used") or [])
            col_info, col_action = st.columns([4, 1])

            with col_info:
                with st.expander(
                    f"📄 {doc.get('file_name','?')} "
                    f"({doc.get('file_size_kb','?')} KB) "
                    f"— {ingested_date}",
                    expanded=False,
                ):
                    detail_cols = st.columns(3)
                    detail_cols[0].metric(
                        "Chunks",  doc.get("chunk_count", 0)
                    )
                    detail_cols[1].metric(
                        "Triples", doc.get("triple_count", 0)
                    )
                    detail_cols[2].metric(
                        "Status",
                        doc.get("status", "?").title()
                    )
                    st.caption(f"**Lenses used:** {lenses_str or 'N/A'}")
                    st.caption(f"**Doc ID:** {doc.get('doc_id','?')}")
                    st.caption(f"**File path:** {doc.get('file_path','?')}")

    st.markdown("---")

    # ── Cross-document relationships ──────────────────────────
    st.markdown("#### 🔗 Cross-Document Relationships")
    st.caption(
        "Relationships discovered between entities from different documents"
    )

    if st.button("🔍 Find Cross-Doc Relations", use_container_width=True):
        driver = get_driver()
        try:
            with driver.session() as session:
                records = session.run("""
                    MATCH (s:__Entity__)-[r]->(o:__Entity__)
                    WHERE s.doc_id <> o.doc_id
                      AND s.doc_id IS NOT NULL
                      AND o.doc_id IS NOT NULL
                    RETURN s.name    AS subject,
                           s.doc_id  AS subject_doc,
                           type(r)   AS relation,
                           o.name    AS object,
                           o.doc_id  AS object_doc,
                           r.confidence AS confidence
                    ORDER BY r.confidence DESC
                    LIMIT 30
                """)
                cross_doc = [dict(r) for r in records]
            driver.close()

            if cross_doc:
                st.success(
                    f"Found {len(cross_doc)} cross-document relationship(s)"
                )
                for rel in cross_doc:
                    conf = rel.get("confidence", 0)
                    icon = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.6 else "🔴"
                    st.markdown(
                        f"{icon} **{rel['subject']}** "
                        f"*(from {rel['subject_doc']})*"
                        f" —[{rel['relation']}]→ "
                        f"**{rel['object']}** "
                        f"*(from {rel['object_doc']})*"
                        f" | {conf:.2f}"
                    )
            else:
                st.info(
                    "No cross-document relationships found yet. "
                    "Add more documents to discover connections between them!"
                )
        except Exception as e:
            driver.close()
            st.error(f"Query error: {e}")

    # ── Memgraph Lab queries ──────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Memgraph Lab — Multi-Document Queries")

    st.code("""// All documents in registry
MATCH (d:__DocRegistry__)
RETURN d.file_name, d.chunk_count, d.triple_count,
       d.lenses_used, d.ingested_at
ORDER BY d.ingested_at DESC;""", language="cypher")

    st.code("""// Cross-document entity relationships
MATCH (s:__Entity__)-[r]->(o:__Entity__)
WHERE s.doc_id <> o.doc_id
RETURN s, r, o LIMIT 50;""", language="cypher")

    st.code("""// Entities shared across documents
MATCH (e:__Entity__)
WHERE size(e.doc_id) > 0
WITH e.name AS entity, count(DISTINCT e.doc_id) AS doc_count
WHERE doc_count > 1
RETURN entity, doc_count
ORDER BY doc_count DESC;""", language="cypher")

    st.code("""// Full multi-doc graph
MATCH (c:__Chunk__)-[:HAS_ENTITY]->(e:__Entity__)-[r]->(o:__Entity__)
RETURN c, e, r, o LIMIT 100;""", language="cypher")



# ================================================================
# TAB 5: DEBUG
# ================================================================
with tab_debug:
    st.markdown("### 🐛 Debug Panel")

    c1, c2 = st.columns(2)
    with c1:
        st.json({
            "graph_built":   st.session_state.graph_built,
            "doc_count":     st.session_state.doc_count,
            "chunk_count":   st.session_state.chunk_count,
            "active_lenses": st.session_state.active_lenses,
            "messages":      len(st.session_state.messages),
            "top_k":         st.session_state.top_k,
        })
    with c2:
        if st.session_state.build_stats:
            st.json({
                "extraction": st.session_state.build_stats.get(
                    "extraction_summary", {}
                ),
                "organic":    st.session_state.build_stats.get(
                    "organic_summary", {}
                ),
                "ingestion":  st.session_state.build_stats.get(
                    "ingestion_stats", {}
                ),
                "step_times": st.session_state.build_stats.get(
                    "step_times", {}
                ),
            })

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🧪 Test LLM"):
            with st.spinner():
                try:
                    t0   = time.time()
                    resp = llm.complete("Say the word hello only.")
                    st.success(f"✅ '{resp}' ({time.time()-t0:.1f}s)")
                except Exception as e:
                    st.error(f"LLM Error: {e}")
    with c2:
        if st.button("🧪 Test Embeddings"):
            with st.spinner():
                try:
                    t0  = time.time()
                    emb = embed_model.get_text_embedding("hello world")
                    st.success(
                        f"✅ dim={len(emb)} ({time.time()-t0:.1f}s)"
                    )
                except Exception as e:
                    st.error(f"Embed Error: {e}")
    with c3:
        if st.session_state.graph_built:
            if st.button("🧪 Test Vector Search"):
                with st.spinner():
                    try:
                        t0   = time.time()
                        q    = embed_model.get_text_embedding("services")
                        ents = vector_search_entities(q, top_k=5)
                        chks = vector_search_chunks(q, top_k=3)
                        st.success(
                            f"✅ {len(ents)} entities, "
                            f"{len(chks)} chunks ({time.time()-t0:.1f}s)"
                        )
                        for e in ents:
                            st.write(
                                f"  Entity: **{e['name']}** "
                                f"({e.get('entity_type','?')}) "
                                f"dist={e.get('distance','?')}"
                            )
                        for c in chks:
                            st.write(
                                f"  Chunk: {c.get('doc_id','?')} "
                                f"| {len(c.get('text',''))} chars"
                                f" dist={c.get('distance','?')}"
                            )
                    except Exception as e:
                        st.error(f"Search error: {e}")
                        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align:center;color:#666;font-size:0.85rem;">'
    "🧠 Directed Graph RAG · LlamaIndex · Ollama · Memgraph · "
    "Stakeholder Lenses · Source Text"
    "</div>",
    unsafe_allow_html=True,
)
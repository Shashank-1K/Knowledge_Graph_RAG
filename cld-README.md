# 🧠 Knowledge Graph RAG System

A **Retrieval-Augmented Generation (RAG)** system powered by a **Directed Knowledge Graph**, built with **LlamaIndex**, **Ollama**, **Memgraph**, and **Streamlit**. Instead of letting the LLM freely decide what to extract, this system uses **structured stakeholder lenses** and **organic relation discovery** to build a rich, queryable knowledge graph from your documents — all running **100% locally** with no API keys required.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?logo=streamlit)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-green)
![Memgraph](https://img.shields.io/badge/Memgraph-Graph_DB-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Table of Contents

- [What Makes This Different](#-what-makes-this-different)
- [System Architecture](#-system-architecture)
- [Complete Data Flow](#-complete-data-flow)
- [Stakeholder Lenses](#-stakeholder-lenses)
- [Organic Relation Discovery](#-organic-relation-discovery)
- [Incremental Graph Updates](#-incremental-graph-updates)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [How It Works — Detailed](#-how-it-works--detailed)
- [Configuration](#-configuration)
- [Memgraph Lab Queries](#-memgraph-lab-queries)
- [Troubleshooting](#-troubleshooting)
- [Performance Notes](#-performance-notes)
- [Example Queries](#-example-queries)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 What Makes This Different

### Traditional RAG vs Graph RAG

| Aspect | Traditional RAG | This System |
|--------|----------------|-------------|
| **Storage** | Text chunks only | Text chunks + explicit entity relationships |
| **Retrieval** | Similar text passages | Similar entities + all their graph relationships |
| **Context** | Random text paragraphs | Structured facts with confidence scores + evidence |
| **Relations** | Implicit in text | Explicit typed edges (PROVIDES, USES, SERVES…) |
| **Perspectives** | Single view | 5 stakeholder lenses on the same data |
| **Cross-chunk** | No linking | Organic relations connect entities across chunks |
| **Provenance** | Which chunk | Which chunk + which lens + confidence + evidence quote |
| **Lens Q&A** | Not possible | Ask from Executive / Technical / HR perspective |
| **Doc updates** | Full rebuild required | Incremental add — existing graph preserved |

### The Core Innovation: Directed Extraction

Instead of:
```
LLM: "Extract anything you think is relevant"
→ Noisy, inconsistent, hard to query
```

This system uses:
```
LLM (5 times): "Extract ONLY these entity types and relations
                from THIS stakeholder's perspective"
→ Clean, schema-validated, queryable by lens
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         STREAMLIT UI                            │
│  💬 Chat  │ 🔍 Explorer │ 🔭 Lens View │ 📄 Chunks │ 📂 Docs  │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │  LLM Engine  │ │  Embedding   │ │   Memgraph   │
      │  Gemma3 via  │ │  nomic-embed │ │  Graph DB    │
      │  Ollama      │ │  -text 768d  │ │  bolt://7687 │
      └──────────────┘ └──────────────┘ └──────────────┘
              │               │               │
              ▼               ▼               ▼
      ┌─────────────────────────────────────────────────┐
      │              CORE MODULES                       │
      │                                                 │
      │  schema.py            → Entity/Relation types   │
      │  directed_extractor   → Multi-lens extraction   │
      │  organic_relations    → Cross-chunk discovery   │
      │  graph_setup.py       → Memgraph operations     │
      │  graph_analytics.py   → Query functions         │
      │  incremental_builder  → Add docs without reset  │
      │  doc_registry.py      → Track ingested docs     │
      └─────────────────────────────────────────────────┘
```

---

## 🔄 Complete Data Flow

### Build Phase (Full Rebuild)

```
PDF / TXT / DOCX
      │
      ▼ SimpleDirectoryReader
┌─────────────┐
│  Documents  │
└──────┬──────┘
       │ SentenceSplitter (chunk_size=512, overlap=64)
       ▼
┌─────────────────┐
│  Text Chunks    │  Each chunk: ~512 chars, unique UUID
│  (TextNodes)    │
└────┬────────────┘
     │                         │
     │ embed(text)             │ N × LLM calls per chunk
     ▼                         ▼
┌──────────────┐   ┌─────────────────────────────────┐
│ __Chunk__    │   │     STAKEHOLDER LENS EXTRACTION  │
│ nodes stored │   │                                  │
│ in Memgraph  │   │  🔵 Executive  → triples         │
│ with:        │   │  🟢 Technical  → triples         │
│ - full text  │   │  🟣 HR_People  → triples         │
│ - embedding  │   │  🟠 Operations → triples         │
│ - metadata   │   │  🟡 Client     → triples         │
└──────────────┘   └──────────────┬──────────────────┘
                                  │ validate + deduplicate
                                  ▼
                   ┌─────────────────────────────────┐
                   │    ORGANIC DISCOVERY             │
                   │  🔗 Cooccurrence relations       │
                   │  🌐 Hub-entity relations         │
                   │  🕐 Temporal relations           │
                   └──────────────┬──────────────────┘
                                  │
                                  ▼
                   ┌─────────────────────────────────┐
                   │         MEMGRAPH                 │
                   │                                  │
                   │  (:__Chunk__)                    │
                   │       │ HAS_ENTITY               │
                   │       ▼                          │
                   │  (:__Entity__)─[RELATION]→       │
                   │  (:__Entity__)                   │
                   │                                  │
                   │  Vector Index: entity   (768d)   │
                   │  Vector Index: chunk_text (768d) │
                   │                                  │
                   │  (:__DocRegistry__)              │
                   │  tracks every ingested file      │
                   └─────────────────────────────────┘
```

### Incremental Update Phase (Add New Docs)

```
New PDF / TXT / DOCX
      │
      ▼ doc_registry checks file hash
      │
      ├─ Already ingested? → Skip (no reprocessing)
      │
      └─ New file? → Process only this file
            │
            ▼ Load + Chunk (new file only)
            │
            ├─▶ Store new __Chunk__ nodes
            │
            ├─▶ Extract triples (new chunks only)
            │
            ├─▶ Fetch ALL existing triples from graph
            │
            ├─▶ Run organic discovery on OLD + NEW combined
            │         → finds cross-document relationships!
            │
            ├─▶ Filter: keep only NEW organic relations
            │         (MERGE prevents duplicates)
            │
            ├─▶ Ingest new triples (MERGE — no duplicates)
            │
            ├─▶ Link new entities → new chunks
            │
            └─▶ Register new doc in __DocRegistry__
                  (hash, chunk count, triple count, lenses used)
```

### Query Phase

```
User Question
      │
      ▼ embed(question) → 768-dim vector
      │
      ├──▶ vector_search("entity")     → Top-K similar entities
      │
      ├──▶ vector_search("chunk_text") → Top-3 similar chunks
      │
      ├──▶ get_triples_for_entities()  → Graph relationships
      │     (optionally filtered by lens)
      │
      └──▶ get_chunk_text_for_entities() → Source text
            │
            ▼
  Build context:
  ┌─────────────────────────────────────────────┐
  │ ## Knowledge Graph Context                  │
  │ ### HomeSoulAI                              │
  │   - PROVIDES → Maintenance Reporting [0.95] │
  │   - PROVIDES → Guidance [0.95]              │
  │                                             │
  │ ## Source Document Text                     │
  │ "Treppan Community is a comprehensive..."   │
  └─────────────────────────────────────────────┘
            │
            ▼ LLM synthesis (Gemma3)
            │
            ▼
  Answer + Details
  (entities matched, triples used, source text, timing)
```

---

## 🔭 Stakeholder Lenses

Each lens applies a different prompt to the same document chunk, extracting different entities and relationships:

### 🔵 Executive Lens
**Perspective:** C-suite, strategic view

| Entity Types | Relation Types |
|-------------|----------------|
| Organization, Market, Industry | OPERATES_IN, SERVES |
| Objective, Outcome, Client | TARGETS, ACHIEVES |
| Challenge, Metric, Timeframe | GENERATES, PARTNERS_WITH |
| | RESULTS_IN, INFLUENCES |

**Extracts:** Strategic goals, market presence, client relationships, business outcomes

---

### 🟢 Technical Lens
**Perspective:** Engineering and architecture view

| Entity Types | Relation Types |
|-------------|----------------|
| Technology, Tool, Platform | USES, INTEGRATES_WITH |
| Infrastructure, Process | DEPENDS_ON, SUPPORTS |
| Product, Service, Capability | ENABLES, PROVIDES, REQUIRES |

**Extracts:** Tech stack, integrations, platform capabilities, technical dependencies

---

### 🟣 HR_People Lens
**Perspective:** Human resources and organisational view

| Entity Types | Relation Types |
|-------------|----------------|
| Person, Role, Department | WORKS_AT, LEADS, MANAGES |
| Team, Organization | REPORTS_TO, BELONGS_TO |
| Capability, Location | RESPONSIBLE_FOR, HAS_SKILL |
| | COLLABORATES_WITH, LOCATED_IN |

**Extracts:** People, roles, org structure, reporting lines, skills

---

### 🟠 Operations Lens
**Perspective:** Operational and process view

| Entity Types | Relation Types |
|-------------|----------------|
| Process, Project, Location | LOCATED_IN, DURING |
| Metric, Timeframe, Policy | STARTED_AT, PRECEDED_BY |
| Event, Outcome | FOLLOWED_BY, RESULTS_IN |
| | REQUIRES, SUPPORTS, PART_OF |

**Extracts:** Processes, locations, timelines, metrics, policies

---

### 🟡 Client_Market Lens
**Perspective:** Sales and business development view

| Entity Types | Relation Types |
|-------------|----------------|
| Client, Market, Industry | SERVES, OPERATES_IN |
| Service, Product | PROVIDES, TARGETS |
| Capability, Outcome | PARTNERS_WITH, ACHIEVES, ENABLES |

**Extracts:** Clients, target markets, services offered, value propositions

---

### How Lenses Work Together

```
Same text: "HomeSoulAI helps residents report maintenance issues
            through the Treppan Community App"

🟢 Technical extracts:
   HomeSoulAI (Capability) -[PROVIDES]→ Maintenance Issue Reporting (Capability)
   Treppan App (Platform)  -[ENABLES]→  HomeSoulAI (Capability)

🟡 Client_Market extracts:
   Residents (Client)  -[USES]→    HomeSoulAI (Capability)
   HomeSoulAI          -[SERVES]→  Residents (Client)

🟣 HR_People extracts:
   (nothing relevant for this segment)

Result: 4 triples from one sentence, each revealing
        a different perspective on the same relationship
```

---

## 🌱 Organic Relation Discovery

After extraction, three discoverers find relationships that **no single chunk reveals**:

### 1. Cooccurrence Discovery
```
Logic: If entity A and entity B both appear in 2+ chunks together,
       they likely have an implicit relationship.

Example:
  chunk_001: {HomeSoulAI, Profile, Residents}
  chunk_002: {HomeSoulAI, Carbon Footprint, Profile}

  (HomeSoulAI, Profile) appear together in 2 chunks
  → Both are Capability type
  → Infer: HomeSoulAI -[COLLABORATES_WITH]→ Profile
  → Confidence: 0.6 + (0.1 × 2) = 0.8
```

### 2. Hub Entity Discovery
```
Logic: If a hub entity (many connections) connects to both A and B,
       then A and B are likely related too.

Example:
  Hub: "Treppan Community App" (8 connections)
  Neighbours: Profile, Carbon Footprint, Maintenance,
              Notification Prefs, Residents, HomeSoulAI

  Profile ↔ Carbon Footprint (not directly connected)
  → Create: Profile -[COLLABORATES_WITH]→ Carbon Footprint
  → Confidence: 0.50 (low — inferred, not explicit)
  → Evidence: "Both connected to hub: Treppan Community App"
```

### 3. Temporal Discovery
```
Logic: Timeframe entities with year numbers get PRECEDED_BY relations.

Example:
  "2020 revenue: $5M" and "2024 revenue: $25M"
  → 2020 -[PRECEDED_BY]→ 2024
  → Confidence: 0.75
```

### Cross-Document Organic Discovery

When you **add a second document**, organic discovery runs on the **combined corpus** (old + new triples). This means:

```
Document 1 entities: {HomeSoulAI, Residents, Treppan App}
Document 2 entities: {Property Manager, Residents, Maintenance Team}

Cross-doc organic relation discovered:
  "Residents" appears in both documents
  → HomeSoulAI -[COLLABORATES_WITH]→ Maintenance Team
     (both connected to "Residents" hub, from different docs)
  → Evidence: "Both connected to hub: Residents"
  → Lens: organic_hub

These cross-document relations are visible in:
  📂 Documents tab → "🔍 Find Cross-Doc Relations"
```

---

## ➕ Incremental Graph Updates

One of the key features of this system is the ability to **add new documents to an existing graph** without rebuilding from scratch.

### How It Works

```
Existing Graph                New Document
(Doc 1 fully ingested)        (Doc 2 uploaded)
        │                           │
        │                           ▼
        │                    1. Hash check → not ingested
        │                    2. Load + chunk Doc 2 only
        │                    3. Store new __Chunk__ nodes
        │                    4. Extract triples from Doc 2
        │                           │
        └───────────────────────────┤
                                    ▼
                        5. Fetch ALL existing triples
                           (Doc 1 already in graph)
                                    │
                                    ▼
                        6. Run organic discovery on
                           Doc1 triples + Doc2 triples
                           → finds cross-document links!
                                    │
                                    ▼
                        7. MERGE new triples
                           (existing nodes/edges untouched)
                                    │
                                    ▼
                        8. Register Doc 2 in __DocRegistry__
```

### Document Registry

Every ingested document is tracked in Memgraph as a `__DocRegistry__` node:

```cypher
(:__DocRegistry__ {
  doc_id:       "cfab85cf8bd82ebe",   // SHA-256 hash prefix
  file_name:    "company_report.pdf",
  file_path:    "./data/company_report.pdf",
  file_hash:    "cfab85cf8bd82ebe...", // full hash for change detection
  file_size_kb: 245.3,
  chunk_count:  12,
  triple_count: 87,
  lenses_used:  ["technical", "client_market"],
  ingested_at:  "2024-02-19T11:08:47",
  status:       "ingested"
})
```

### Deduplication Guarantee

The system uses Cypher `MERGE` statements throughout, so:

- ✅ Uploading the same file twice → detected by hash, skipped
- ✅ Same entity mentioned in two docs → single `__Entity__` node, multiple relations
- ✅ Same relation extracted by two lenses → highest confidence kept
- ✅ Re-running organic discovery → only NEW organic relations ingested

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🔭 **Multi-Lens Extraction** | 5 stakeholder perspectives on every document chunk |
| 🌱 **Organic Relations** | Cross-chunk relationship discovery (cooccurrence, hub, temporal) |
| ➕ **Incremental Updates** | Add new documents without rebuilding the entire graph |
| 📋 **Document Registry** | Tracks every ingested file with hash, chunk count, triple count |
| 🔗 **Cross-Doc Relations** | Organic discovery runs across all documents — finds hidden links |
| 📄 **Source Text Storage** | Full chunk text stored in Memgraph as `__Chunk__` nodes |
| 🔗 **Entity-Chunk Links** | Every entity traces back to its source paragraph |
| 💬 **Chat Interface** | Conversational Q&A with graph + text context |
| 🔭 **Lens Q&A** | Ask questions filtered to specific stakeholder perspective |
| 🔍 **Graph Explorer** | Entity neighbourhood, path finding, high-confidence triples |
| 📄 **Source Chunks Tab** | Browse all document chunks with linked entities |
| 📊 **Graph Statistics** | Live node, relation, chunk, and entity counts by lens |
| 📋 **Memgraph Lab Queries** | Ready-to-paste Cypher for graph visualisation |
| 🌐 **100% Local** | No cloud APIs — Ollama + Memgraph run on your machine |
| 🔄 **Schema Validation** | All triples validated against defined entity/relation types |
| 🔑 **Provenance Tracking** | Every triple records: lens, chunk_id, confidence, evidence |
| 🧹 **Deduplication** | Cross-lens duplicates merged, highest confidence kept |

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Ollama + Gemma3 | Text generation + KG extraction |
| **Embeddings** | nomic-embed-text (768d) | Vector representations |
| **Graph Database** | Memgraph | Store knowledge graph + vector search |
| **Framework** | LlamaIndex | Document loading + node parsing |
| **Frontend** | Streamlit | Web interface |
| **Driver** | Neo4j Python Driver | Memgraph connectivity |
| **Vector Search** | Memgraph built-in | Cosine similarity on embeddings |

---

## 📋 Prerequisites

- **Python 3.10+**
- **Docker** (for Memgraph)
- **Ollama** (for local LLM + embeddings)
- **8GB+ RAM** (16GB recommended for larger documents)
- **GPU recommended** (CPU works but extraction is slow: ~2 min/lens/chunk)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/GYTWorkz-Private-Limited/GraphRAG.git
cd GraphRAG
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Memgraph

```bash
docker run -d \
  --name memgraph \
  -p 7687:7687 \
  -p 7444:7444 \
  -p 3000:3000 \
  memgraph/memgraph-platform
```

Memgraph Lab (visual graph browser) available at:
```
http://localhost:3000
```

### 4. Pull Ollama Models

```bash
# LLM for extraction and Q&A
ollama pull gemma3

# Embedding model (768 dimensions)
ollama pull nomic-embed-text
```

### 5. Verify Ollama

```bash
ollama list
# Should show: gemma3, nomic-embed-text
```

### 6. Add Your Documents

```bash
mkdir data
# Place your PDF, TXT, or DOCX files in ./data/
```

### 7. Run the Application

```bash
python -m streamlit run app.py
```

App available at: `http://localhost:8501`

---

## 📁 Project Structure

```
knowledge-graph-rag/
│
├── app.py                  # Streamlit UI + RAG query pipeline
├── config.py               # All configuration constants
├── schema.py               # Entity types, relation types, lens definitions
├── directed_extractor.py   # Multi-lens LLM extraction pipeline
├── organic_relations.py    # Cross-chunk relation discovery
├── graph_setup.py          # Memgraph operations + triple ingestion
├── graph_analytics.py      # Query functions (neighbourhood, paths, etc.)
├── incremental_builder.py  # Add new docs without full rebuild
├── doc_registry.py         # Track ingested documents in Memgraph
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/                   # Put your documents here
│   ├── document_1.pdf      # Initially ingested via full build
│   └── document_2.pdf      # Added later via incremental update
│
├── .streamlit/
│   └── config.toml         # Streamlit theme settings
│
└── app_debug.log           # Detailed debug log (auto-generated)
```

---

## 📖 Usage

### Step 1: Add Documents

Place your documents in the `./data/` directory:

```bash
cp your_document.pdf ./data/
```

Supported formats: **PDF, TXT, DOCX, MD, CSV**

### Step 2: Configure Lenses

In the sidebar, select which stakeholder lenses to use:

| Selection | Lenses | Build Time (per chunk) |
|-----------|--------|----------------------|
| All 5 | Full coverage | ~10 min |
| 2–3 | Faster, focused | ~4 min |
| 1 | Quickest | ~2 min |

### Step 3: Build the Knowledge Graph

1. Set **Top K** (how many entities to match per query, default 5)
2. Check **Reset DB** (recommended for fresh builds)
3. Click **🚀 Build Full Graph**

Build steps:
```
1. Reset database          (~0.1s)
2. Load documents          (~0.3s)
3. Chunk documents         (~0.1s)
4. Store chunk nodes       (~2–10s per chunk)
5. Extract with lenses     (~2 min × lenses × chunks)
6. Discover organic links  (~0.1s)
7. Ingest triples          (~5–10s)
8. Link entities → chunks  (~0.1s)
9. Fix vector indexes      (~0.1s)
```

After the build, every document in `./data/` is registered in the **Document Registry** (stored in Memgraph as `__DocRegistry__` nodes).

### Step 4: Add More Documents (Incremental)

To add new documents **without rebuilding from scratch**:

1. Go to the **📂 Documents** tab
2. Use the file uploader to browse and upload new files
3. The system shows each file's status:
   - 🆕 **New** — not yet in graph, will be processed
   - ✅ **Already in graph** — detected by file hash, skipped
4. Select lenses for the new documents
5. Click **➕ Add N Document(s)**

What happens during incremental update:
```
✅ Existing graph is fully preserved
✅ Only new file(s) are chunked and extracted
✅ Organic discovery runs across ALL docs (old + new)
✅ New cross-document relationships are discovered
✅ MERGE prevents any duplicate nodes or edges
✅ New document registered in __DocRegistry__
```

### Step 5: Chat

Ask questions in the **💬 Chat** tab:

```
What is the document about?
What technologies are used?
Who are the key people?
What services does the company offer?
What are the strategic objectives?
```

Each answer shows:
- The answer text
- Which entities were matched (with similarity distances)
- Which graph triples were used (with confidence + lens)
- The source chunk text from the document
- Timing breakdown

### Step 6: Lens-Filtered Q&A

Go to **🔭 Lens View & Q&A**:

1. Select a stakeholder lens (e.g., 🟢 Technical)
2. Use suggested questions or type your own
3. Get answers filtered to that lens's perspective only

```
Technical lens:  "What platforms does the app use?"
Client lens:     "Who are the target customers?"
Executive lens:  "What are the strategic goals?"
HR lens:         "What roles exist in the org?"
```

### Step 7: Explore the Graph

Go to **🔍 Graph Explorer**:

**Entity Neighbourhood:**
```
Type: "HomeSoulAI"
→ See all direct connections with confidence + evidence
```

**Path Between Entities:**
```
Entity A: "Profile"    Entity B: "Residents"
→ Profile -[ENABLES]→ Residents (1 hop)
```

**High-Confidence Triples:**
```
Set threshold to 0.9
→ See only strongly evidenced triples
```

### Step 8: Manage Documents

Go to **📂 Documents** tab:

| Action | How |
|--------|-----|
| **View all ingested docs** | See list with chunk/triple counts and lenses |
| **Add new documents** | Upload via file browser |
| **Remove a document** | Click 🗑️ → confirm deletion |
| **Find cross-doc relations** | Click "🔍 Find Cross-Doc Relations" |
| **View registry in Memgraph** | Use the provided Cypher queries |

### Step 9: Memgraph Lab

Copy Cypher queries from the **🔭 Lens View** or **📂 Documents** tab into Memgraph Lab (`http://localhost:3000`) for interactive visual graph exploration.

---

## 🔧 How It Works — Detailed

### Extraction Pipeline (per chunk)

```python
# For each document chunk:
for lens in [executive, technical, hr_people, operations, client_market]:

    # 1. Build lens-specific prompt
    prompt = f"""
    ALLOWED ENTITY TYPES: {lens.entity_types}
    ALLOWED RELATION TYPES: {lens.relation_types}
    TEXT: {chunk.text}
    Output ONLY valid JSON array of triples.
    """

    # 2. Call Gemma3
    response = llm.complete(prompt)

    # 3. Parse + repair JSON
    triples = parse_json(response)

    # 4. Validate each triple
    for triple in triples:
        if entity_type_valid and relation_type_valid and confidence >= 0.6:
            keep(triple)

# 5. Deduplicate across lenses
# 6. Canonicalize entity names
```

### Incremental Build Pipeline

```python
def run_incremental_build(new_file_paths, llm, embed_model, active_lenses):

    # 1. Load + chunk ONLY the new files
    for file_path in new_file_paths:
        docs  = SimpleDirectoryReader([file_path]).load_data()
        nodes = splitter.get_nodes_from_documents(docs)

    # 2. Store new __Chunk__ nodes (with embeddings)
    ingest_chunk_nodes(new_nodes, embed_model)

    # 3. Extract triples from new chunks only
    new_triples = extractor.extract_from_nodes(new_nodes)

    # 4. Fetch ALL existing triples from graph
    existing_triples = fetch_existing_triples_from_graph()

    # 5. Organic discovery on COMBINED corpus
    combined = existing_triples + new_triples
    _, organic_relations = organic_pipeline.run(combined)

    # 6. Filter to only NEW organic relations
    existing_keys = {(s, r, o) for s, r, o in existing_triples}
    new_organic   = [r for r in organic_relations
                     if (r.subject, r.relation, r.object) not in existing_keys]

    # 7. Ingest new triples (MERGE — no duplicates)
    ingest_triples_to_memgraph(new_triples + new_organic)

    # 8. Link new entities → new chunks
    link_entities_to_chunks(new_triples)

    # 9. Register in __DocRegistry__
    register_document(file_path, chunk_count, triple_count, lenses_used)
```

### Validation (Fuzzy Matching)

```python
# Handles Gemma3's inconsistent casing
"Capability"  → EntityType.CAPABILITY  ✓  (exact)
"capability"  → EntityType.CAPABILITY  ✓  (lowercase)
"feature"     → EntityType.CAPABILITY  ✓  (substring)
"xyz"         → None                   ✗  (rejected)

"PROVIDES"    → RelationType.PROVIDES  ✓  (exact)
"provides"    → RelationType.PROVIDES  ✓  (normalised)
"gives"       → None                   ✗  (rejected, logged)
```

### Memgraph Schema

```cypher
// Chunk node — stores full text
(:__Chunk__ {
  chunk_id:   "uuid",
  text:       "full paragraph text...",
  doc_id:     "file.pdf",
  page_label: "1",
  char_count: 512,
  embedding:  [768 floats]
})

// Entity node — stores named entity
(:__Entity__ {
  name:            "HomeSoulAI",
  entity_type:     "Capability",
  created_by_lens: "client_market",
  chunk_id:        "uuid",
  embedding:       [768 floats]
})

// Typed relationship with full provenance
(:__Entity__)-[:PROVIDES {
  confidence: 0.95,
  evidence:   "exact quote from document",
  lens:       "client_market",
  chunk_id:   "uuid"
}]->(:__Entity__)

// Source link
(:__Chunk__)-[:HAS_ENTITY {lens: "technical"}]->(:__Entity__)

// Document registry
(:__DocRegistry__ {
  doc_id:       "cfab85cf8bd82ebe",
  file_name:    "report.pdf",
  file_hash:    "cfab85cf8bd82ebe...",
  chunk_count:  12,
  triple_count: 87,
  lenses_used:  ["technical", "client_market"],
  ingested_at:  "2024-02-19T11:08:47",
  status:       "ingested"
})
```

---

## ⚙️ Configuration

Edit `config.py`:

```python
# ── Ollama ────────────────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
LLM_MODEL         = "gemma3"          # or llama3, mistral
LLM_TEMPERATURE   = 0                 # 0 = deterministic
LLM_TIMEOUT       = 600.0             # seconds per LLM call
EMBED_MODEL       = "nomic-embed-text"
EMBED_DIMENSION   = 768

# ── Memgraph ──────────────────────────────────────────────
MEMGRAPH_URI      = "bolt://localhost:7687"
MEMGRAPH_USER     = ""
MEMGRAPH_PASSWORD = ""

# ── Chunking ──────────────────────────────────────────────
CHUNK_SIZE        = 512   # chars per chunk
CHUNK_OVERLAP     = 64    # overlap prevents context loss at boundaries

# ── Extraction ────────────────────────────────────────────
ACTIVE_LENSES = [
    "executive",
    "technical",
    "hr_people",
    "operations",
    "client_market",
]
MIN_TRIPLE_CONFIDENCE = 0.6   # discard triples below this

# ── Organic Discovery ─────────────────────────────────────
ORGANIC_MIN_COOCCURRENCE = 2    # min shared chunks for cooccurrence
ORGANIC_ENABLE_HUB       = True
ORGANIC_ENABLE_TEMPORAL  = True
ORGANIC_MAX_RELATIONS    = 200

# ── Query ─────────────────────────────────────────────────
SIMILARITY_TOP_K     = 5    # entities matched per query
MAX_CONTEXT_TRIPLES  = 30   # max triples sent to LLM
GRAPH_QUERY_DEPTH    = 2    # neighbourhood exploration depth
INGESTION_BATCH_SIZE = 50   # triples per Memgraph batch
```

---

## 📊 Memgraph Lab Queries

Open `http://localhost:3000` and use these Cypher queries:

```cypher
-- Full graph: chunks + entities + relations
MATCH (c:__Chunk__)-[:HAS_ENTITY]->(e:__Entity__)-[r]->(o:__Entity__)
RETURN c, e, r, o LIMIT 100;

-- All entity relations only
MATCH (s:__Entity__)-[r]->(o:__Entity__)
RETURN s, r, o LIMIT 100;

-- High-confidence relations (≥0.8)
MATCH (s:__Entity__)-[r]->(o:__Entity__)
WHERE r.confidence >= 0.8
RETURN s, r, o ORDER BY r.confidence DESC;

-- Technical lens subgraph
MATCH (s:__Entity__)-[r]->(o:__Entity__)
WHERE r.lens = 'technical'
   OR (r.lens STARTS WITH 'multi:' AND r.lens CONTAINS 'technical')
RETURN s, r, o LIMIT 100;

-- Client_Market lens subgraph
MATCH (s:__Entity__)-[r]->(o:__Entity__)
WHERE r.lens = 'client_market'
   OR (r.lens STARTS WITH 'multi:' AND r.lens CONTAINS 'client_market')
RETURN s, r, o LIMIT 100;

-- Entity neighbourhood (replace 'HomeSoulAI')
MATCH (e:__Entity__ {name: 'HomeSoulAI'})-[r]-(nb:__Entity__)
RETURN e, r, nb;

-- Path between two entities (replace names)
MATCH path = (a:__Entity__ {name: 'Profile'})
             -[*1..4]-
             (b:__Entity__ {name: 'Residents'})
WHERE ALL(n IN nodes(path) WHERE n:__Entity__)
RETURN path ORDER BY length(path) LIMIT 5;

-- Source text for an entity
MATCH (c:__Chunk__)-[:HAS_ENTITY]->(e:__Entity__ {name: 'HomeSoulAI'})
RETURN c.text, c.doc_id, c.page_label;

-- All entities with their types
MATCH (n:__Entity__)
RETURN n.name, n.entity_type, n.created_by_lens
ORDER BY n.entity_type, n.name;

-- Relation type distribution
MATCH ()-[r]->()
RETURN type(r) AS relation, count(*) AS count
ORDER BY count DESC;

-- Document registry: all ingested documents
MATCH (d:__DocRegistry__)
RETURN d.file_name, d.chunk_count, d.triple_count,
       d.lenses_used, d.ingested_at
ORDER BY d.ingested_at DESC;

-- Cross-document entity relationships
MATCH (s:__Entity__)-[r]->(o:__Entity__)
WHERE s.doc_id <> o.doc_id
  AND s.doc_id IS NOT NULL
  AND o.doc_id IS NOT NULL
RETURN s.name AS subject,  s.doc_id AS from_doc,
       type(r) AS relation,
       o.name AS object,   o.doc_id AS to_doc,
       r.confidence AS confidence
ORDER BY r.confidence DESC
LIMIT 50;

-- Entities shared across multiple documents
MATCH (e:__Entity__)
WHERE e.doc_id IS NOT NULL
WITH e.name AS entity, collect(DISTINCT e.doc_id) AS docs
WHERE size(docs) > 1
RETURN entity, docs, size(docs) AS doc_count
ORDER BY doc_count DESC;
```

---

## 🔍 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `streamlit: access denied` | Wrong command | Use `python -m streamlit run app.py` |
| `Memgraph connection refused` | Docker not running | `docker start memgraph` or re-run docker command |
| `Ollama connection refused` | Ollama not running | `ollama serve` in a separate terminal |
| `Vector search returns 0` | Wrong index dimension | Reset DB and rebuild — index recreated at 768d |
| `0 triples validated` | Gemma3 ignoring schema | Check `app_debug.log` for `REJECTED` lines |
| `JSON parse error` | Truncated LLM response | Increase `LLM_TIMEOUT` in config.py |
| `Lens Graph shows 0 triples` | Lens not in build | Rebuild with that lens enabled |
| `Path not found` | Entities disconnected | Increase max hops (up to 6) |
| `File shows as already ingested` | Same file hash detected | Expected — system skips re-processing correctly |
| `No cross-doc relations` | Only one document | Add a second document via incremental update |
| Build too slow | All 5 lenses, CPU only | Use 2–3 lenses; GPU is ~5× faster |

### Debug Workflow

```bash
# 1. Check for REJECTED warnings
grep "REJECTED" app_debug.log

# 2. Check what Gemma3 actually returned
grep "REJECTED subject_type" app_debug.log

# 3. Verify Memgraph has data
# Go to http://localhost:3000
# Run: MATCH (n) RETURN count(n);

# 4. Check document registry
# Run: MATCH (d:__DocRegistry__) RETURN d;

# 5. Test vector search
# In Debug tab → click "🧪 Test Vector Search"

# 6. Verify incremental build didn't skip files
grep "Saved uploaded file" app_debug.log
grep "Registered document" app_debug.log
```

---

## ⚡ Performance Notes

| Metric | Value | Notes |
|--------|-------|-------|
| Build time per lens per chunk | ~1–3 min | GPU vs CPU dependent |
| 1-page doc, 2 lenses | ~5 min | Typical demo build |
| 5-page doc, 5 lenses | ~60 min | Full extraction |
| Incremental add (1 new page) | ~5 min | Only new chunks processed |
| Query: embed | ~2s | nomic-embed-text |
| Query: vector search | ~0.05s | Memgraph in-memory |
| Query: LLM synthesis | ~30–60s | Gemma3 local |
| Total query time | ~35–65s | End-to-end |

### Speed Tips

```python
# config.py — faster builds
ACTIVE_LENSES = ["technical", "client_market"]  # 2 instead of 5

# Larger chunks = fewer LLM calls
CHUNK_SIZE = 1024

# Disable organic discovery if not needed
ORGANIC_ENABLE_HUB      = False
ORGANIC_ENABLE_TEMPORAL = False
```

---

## 💬 Example Queries

### General (Chat Tab)
```
What is the document about?
What are the main features described?
Who are the intended users?
What problem does this solve?
Summarize the key points.
```

### Technical Lens
```
What technologies are used?
What are the system capabilities?
How does the platform work?
What does the app provide technically?
```

### Client_Market Lens
```
Who are the target customers?
What services are offered to clients?
What value does this deliver to users?
Which industries or markets are targeted?
```

### Executive Lens
```
What are the strategic objectives?
What business outcomes are described?
What markets does the organisation operate in?
What are the key performance indicators?
```

### Graph Explorer
```
# Entity Neighbourhood
Type: "HomeSoulAI"
→ see all connected entities with confidence + evidence

# Path Finding
Entity A: "Profile"    Entity B: "Residents"
→ Profile -[ENABLES]→ Residents (1 hop)

# High-Confidence Facts
Threshold: 0.9
→ see only strongly evidenced triples

# Cross-Document (Documents tab)
→ find relationships between entities from different files
```

---

## Acknowledgments

- **[LlamaIndex](https://www.llamaindex.ai/)** — Document loading, node parsing, RAG orchestration
- **[Ollama](https://ollama.com/)** — Local LLM serving (Gemma3, nomic-embed-text)
- **[Memgraph](https://memgraph.com/)** — In-memory graph database with vector search
- **[Streamlit](https://streamlit.io/)** — Web application framework
- **[Google Gemma3](https://ai.google.dev/gemma)** — Open-source LLM for extraction and synthesis
- **[Nomic AI](https://www.nomic.ai/)** — nomic-embed-text embedding model
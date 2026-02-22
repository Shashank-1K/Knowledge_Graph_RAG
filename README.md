

# 🧠 Knowledge Graph RAG System


A **Retrieval-Augmented Generation (RAG)** system powered by a **Directed Knowledge Graph**, built with **LlamaIndex**, **Ollama**, **Memgraph**, and **Streamlit**. Instead of letting the LLM freely decide what to extract, this system uses **structured stakeholder lenses**, **organic relation discovery**, **smart incremental building**, and **cross-lens comparison** to build a rich, queryable knowledge graph from your documents — all running **100% locally** with no API keys required.

---

## 📋 Table of Contents

* [What Makes This Different](#-what-makes-this-different)
* [System Architecture](#-system-architecture)
* [Complete Data Flow](#-complete-data-flow)
* [Stakeholder Lenses](#-stakeholder-lenses)
* [Organic Relation Discovery](#-organic-relation-discovery)
* [Features](#-features)
* [Tech Stack](#-tech-stack)
* [Prerequisites](#-prerequisites)
* [Installation](#-installation)
* [Project Structure](#-project-structure)
* [Usage](#-usage)
* [Configuration](#configuration)
* [Memgraph Lab Queries](#-memgraph-lab-queries)
* [Troubleshooting](#-troubleshooting)
* [Performance Notes](#-performance-notes)
* [Acknowledgments](#-acknowledgments)


---

## 🎯 What Makes This Different

### Traditional RAG vs Graph RAG

| Aspect | Traditional RAG | This System |
| --- | --- | --- |
| **Storage** | Text chunks only | Text chunks + explicit entity relationships |
| **Retrieval** | Similar text passages | Similar entities + all their graph relationships |
| **Context** | Random text paragraphs | Structured facts with confidence scores + evidence |
| **Perspectives** | Single view | 5 stakeholder lenses on the same data |
| **Cross-chunk** | No linking | Organic relations connect entities across chunks & documents |
| **Updates** | Re-embed everything | **Incremental:** Only processes brand new documents |
| **Lens Q&A** | Not possible | Ask from Executive / Technical / HR perspective |

### The Core Innovation: Directed Extraction & Incremental Builds

Instead of naive LLM extraction:

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

Furthermore, as your knowledge base grows, **you never rebuild from scratch**. Drop in a new PDF, and the system uses cryptographic hashing to skip existing files, processing *only* the new content and organically merging it into the existing graph.

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
      │              CORE MODULES                       │
      │                                                 │
      │  schema.py            → Entity/Relation types   │
      │  directed_extractor   → Multi-lens extraction   │
      │  organic_relations    → Cross-chunk discovery   │
      │  graph_setup.py       → Memgraph operations     │
      │  graph_analytics.py   → Query functions         │
      │  incremental_builder  → Add docs without reset  │
      │  doc_registry.py      → Track ingested docs     │
      │  lens_comparator.py   → Cross-lens diff reports │
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

**Extracts:** Strategic goals, market presence, client relationships, business outcomes
**Entities:** Organization, Market, Industry, Objective, Outcome, Client, Metric

### 🟢 Technical Lens

**Extracts:** Tech stack, integrations, platform capabilities, technical dependencies
**Entities:** Technology, Tool, Platform, Infrastructure, Process, Product, Capability

### 🟣 HR_People Lens

**Extracts:** People, roles, org structure, reporting lines, skills
**Entities:** Person, Role, Department, Team, Organization, Capability, Location

### 🟠 Operations Lens

**Extracts:** Processes, locations, timelines, metrics, policies
**Entities:** Process, Project, Location, Metric, Timeframe, Policy, Event, Outcome

### 🟡 Client_Market Lens

**Extracts:** Clients, target markets, services offered, value propositions
**Entities:** Client, Market, Industry, Service, Product, Capability, Outcome

---

## 🌱 Organic Relation Discovery

After extraction, three discoverers find relationships that **no single chunk reveals**, dynamically bridging the gap between previously ingested documents and new ones:

### 1. Cooccurrence Discovery

```
Logic: If entity A and entity B both appear in 2+ chunks together, they likely have a relationship.
Example: (HomeSoulAI, Profile) co-occur in 3 chunks → Infer: HomeSoulAI -[COLLABORATES_WITH]→ Profile

```

### 2. Hub Entity Discovery

```
Logic: If a hub entity connects to both A and B, then A and B are likely indirectly related.
Example: Both "Profile" and "Carbon Footprint" connect to the Hub "Treppan App".
→ Create: Profile -[COLLABORATES_WITH]→ Carbon Footprint (confidence: 0.50)

```

### 3. Temporal Discovery

```
Logic: Timeframe entities with year numbers get PRECEDED_BY relations.
Example: "2020" and "2024" → 2020 -[PRECEDED_BY]→ 2024

```

---

## ✨ Features

| Feature | Description |
| --- | --- |
| 🚀 **Incremental Updates** | Drop in new files and *only* process the new content. Cryptographic hashing prevents duplicate work. |
| 🔭 **Multi-Lens Extraction** | 5 stakeholder perspectives on every document chunk. |
| 🌱 **Organic Relations** | Cross-document relationship discovery (cooccurrence, hub, temporal). |
| 📄 **Source Text Storage** | Full chunk text stored in Memgraph as `__Chunk__` nodes. |
| 🔗 **Entity-Chunk Links** | Every entity traces back to its exact source paragraph. |
| 💬 **Chat Interface** | Conversational Q&A grounded in graph + text context. |
| 🔭 **Lens Q&A** | Ask questions filtered exclusively to a specific stakeholder perspective. |
| ⚖️ **Lens Comparator** | Ask a single question across all active lenses simultaneously to generate a diff report highlighting blind spots, unique perspectives, and consensus. |
| 🔍 **Graph Explorer** | View entity neighbourhoods, path finding, and high-confidence triples. |
| 📁 **Document Registry** | Track exactly which files are in the graph, their sizes, and ingestion dates. |
| 📊 **Live Statistics** | Real-time node, relation, chunk, and entity counts. |
| 🌐 **100% Local** | No cloud APIs — Ollama + Memgraph run completely on your machine. |

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
| --- | --- | --- |
| **LLM** | Ollama + Gemma3 | Text generation + KG extraction |
| **Embeddings** | nomic-embed-text (768d) | Vector representations |
| **Graph Database** | Memgraph | Store knowledge graph + vector search |
| **Framework** | LlamaIndex | Document loading + node parsing |
| **Frontend** | Streamlit | Web interface |
| **Driver** | Neo4j Python Driver | Memgraph connectivity |

---

## 📋 Prerequisites

* **Python 3.10+**
* **Docker** (for Memgraph)
* **Ollama** (for local LLM + embeddings)
* **8GB+ RAM** (16GB recommended for larger documents)
* **GPU recommended** (CPU works but extraction is much slower)

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

Memgraph Lab (visual graph browser) is available at: `http://localhost:3000`

### 4. Pull Ollama Models

```bash
# LLM for extraction and Q&A
ollama pull gemma3

# Embedding model (768 dimensions)
ollama pull nomic-embed-text

```

### 5. Run the Application

```bash
python -m streamlit run app.py

```

App is available at: `http://localhost:8501`

---


### 📁 Project Structure

```text
GraphRAG/
│
├── app.py                  # Streamlit UI + RAG query pipeline
├── config.py               # Configuration constants (LLM, Graph, UI)
├── schema.py               # Entity types, relation types, lens definitions
├── directed_extractor.py   # Multi-lens LLM extraction logic
├── organic_relations.py    # Cross-chunk relation discovery (Hubs, Temporal)
├── graph_setup.py          # Memgraph connection & schema setup
├── graph_analytics.py      # Cypher queries for graph exploration
├── incremental_builder.py  # Orchestrates smart document updates
├── doc_registry.py         # SHA-256 state tracking for ingested files
├── lens_comparator.py      # Cross-lens Q&A analysis and diff reporting
│
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
│
├── data/                   # Place your PDF/TXT/DOCX files here
│   └── your_document.pdf
│
├── .streamlit/
│   └── config.toml         # Streamlit theme settings
│
└── app_debug.log           # detailed debug log (auto-generated)

```

---

## 📖 Usage

### Step 1: Initial Graph Build

1. Upload your first document(s) via the **📁 Manage Documents** tab.
2. Select your desired lenses in the sidebar.
3. Click **🚀 Build Full Graph** (this resets the DB and builds the baseline).

### Step 2: Incremental Updates (The Magic)

As you get new documents over time, you do *not* need to rebuild:

1. Go to the **📁 Manage Documents** tab.
2. Drag and drop your new PDF(s).
3. The system will automatically hash the files, recognize existing ones (marking them with a ♻️), and flag new ones (marking them with a 🆕).
4. Click **Update Graph**. The system will only extract triples from the new file and organically stitch them into your existing network!

### Step 3: Ask Questions

Use the **💬 Chat** tab to ask general questions, or use the **🔭 Lens View & Q&A** tab to force the LLM to only answer from a specific perspective (e.g., "From a technical perspective, what databases are used?").

### Step 4: Compare Perspectives (Lens Comparator)
Instead of asking one stakeholder, ask them all at once! 
1. Navigate to the **Lens Comparator** feature.
2. Ask a strategic question (e.g., *"What are the main risks of this project?"*).
3. The system queries all active lenses simultaneously and generates a **diff report** showing:
   - What facts all lenses agree on.
   - Unique topic focus per lens.
   - **Blind spots:** What the Technical lens missed that the Executive lens caught (and vice versa).

### Step 5: Explore the Graph
Use the **🔍 Graph Explorer** to find shortest paths between two disconnected entities, or view the immediate neighborhood of a hub entity.

---

## Configuration

Edit `config.py` to tune the system:

```python
# ── Ollama ────────────────────────────────────────────────
OLLAMA_BASE_URL   = "http://localhost:11434"
LLM_MODEL         = "gemma3"          # or llama3, mistral
EMBED_MODEL       = "nomic-embed-text"
EMBED_DIMENSION   = 768

# ── Chunking ──────────────────────────────────────────────
CHUNK_SIZE        = 512   # larger = more context per LLM call
CHUNK_OVERLAP     = 64    # overlap prevents losing context at boundaries

# ── Extraction ────────────────────────────────────────────
ACTIVE_LENSES = ["executive", "technical", "hr_people", "operations", "client_market"]
MIN_TRIPLE_CONFIDENCE = 0.6   # discard triples below this

# ── Organic Discovery ─────────────────────────────────────
ORGANIC_MIN_COOCCURRENCE = 2    # min shared chunks for cooccurrence

```

---

## 📊 Memgraph Lab Queries

Open `http://localhost:3000` and paste these to explore your data visually:

```cypher
-- 1. View all Document Registry entries
MATCH (d:DocRegistry)
RETURN d.file_name, d.chunk_count, d.triple_count, d.ingested_at
ORDER BY d.ingested_at DESC;

-- 2. View cross-document relationships (The "Aha!" moment of Graph RAG)
MATCH (s:__Entity__)-[r]->(o:__Entity__)
WHERE s.doc_id <> o.doc_id AND s.doc_id IS NOT NULL AND o.doc_id IS NOT NULL
RETURN s, r, o LIMIT 50;

-- 3. View entities shared across multiple documents
MATCH (e:__Entity__)
WHERE size(e.doc_id) > 0
WITH e.name AS entity, count(DISTINCT e.doc_id) AS doc_count
WHERE doc_count > 1
RETURN entity, doc_count ORDER BY doc_count DESC;

-- 4. View a specific lens subgraph
MATCH (s:__Entity__)-[r]->(o:__Entity__)
WHERE r.lens = 'technical' OR r.lens CONTAINS 'technical'
RETURN s, r, o LIMIT 100;

```

---

## ⚡ Performance Notes

| Metric | Value | Notes |
| --- | --- | --- |
| Build time per lens per chunk | ~1-3 min | Depends heavily on GPU/CPU |
| 1 page doc, 2 lenses | ~5 min | Typical demo build |
| Incremental Add (1 page) | ~1-2 min | Only processes the diff! |
| Query: vector search | ~0.05s | Memgraph in-memory speed |
| Query: LLM synthesis | ~15-30s | Gemma3 local |

**Speed Tips:** To drastically speed up builds, use only 2 lenses (e.g., `["technical", "client_market"]`) instead of all 5 in `config.py`.

---

## Acknowledgments

* **[LlamaIndex](https://www.llamaindex.ai/)** — Document loading, node parsing, RAG orchestration
* **[Ollama](https://ollama.com/)** — Local LLM serving (Gemma3, nomic-embed-text)
* **[Memgraph](https://memgraph.com/)** — In-memory graph database with vector search
* **[Streamlit](https://streamlit.io/)** — Web application framework
* **[Google Gemma3](https://ai.google.dev/gemma)** — Open-source LLM for extraction and synthesis

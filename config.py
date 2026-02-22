"""Configuration constants for the application."""

# Ollama Settings
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "gemma3"
LLM_TEMPERATURE = 0
LLM_TIMEOUT = 600.0
EMBED_MODEL = "nomic-embed-text"
EMBED_DIMENSION = 768

# Memgraph Settings
MEMGRAPH_URI = "bolt://localhost:7687"
MEMGRAPH_USER = ""
MEMGRAPH_PASSWORD = ""

# LlamaIndex Settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
MAX_PATHS_PER_CHUNK = 10
NUM_WORKERS = 1
SIMILARITY_TOP_K = 5

# ============================================================
# Directed Graph Settings
# ============================================================

# Which stakeholder lenses to use during extraction
# Options: "executive", "technical", "hr_people", "operations", "client_market"
# Set to None to use all lenses
ACTIVE_LENSES = [
    "executive",
    "technical",
    "hr_people",
    "operations",
    "client_market",
]

# Minimum confidence for a triple to be kept
MIN_TRIPLE_CONFIDENCE = 0.6

# Organic relation discovery settings
ORGANIC_MIN_COOCCURRENCE = 2       # Min chunks sharing entities for cooccurrence
ORGANIC_ENABLE_HUB = True          # Enable hub entity relation discovery
ORGANIC_ENABLE_TEMPORAL = True     # Enable temporal relation discovery
ORGANIC_MAX_RELATIONS = 200        # Cap organic relations

# Ingestion settings
INGESTION_BATCH_SIZE = 50          # Triples per Memgraph batch

# Query-time settings
GRAPH_QUERY_DEPTH = 2              # Neighborhood expansion depth
MAX_CONTEXT_TRIPLES = 30           # Max triples to include in LLM context
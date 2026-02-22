# Multi-Domain RAG Platform — Complete Project Guide

**Version:** Phase 2.2
**Last Updated:** February 2026
**Stack:** Python · ChromaDB · SentenceTransformers / Gemini / OpenAI · Gradio · Pydantic · BM25
**Branch:** phase2_testing

---

## Table of Contents

1. [What This Project Is](#1-what-this-project-is)
2. [Why This Architecture](#2-why-this-architecture)
3. [High-Level Architecture Diagram](#3-high-level-architecture-diagram)
4. [Layer-by-Layer Deep Dive](#4-layer-by-layer-deep-dive)
   - 4.1 UI Layer
   - 4.2 Service Layer
   - 4.3 Pipeline Layer
   - 4.4 Factory Layer
   - 4.5 Storage & Config Layer
5. [Every Component Explained](#5-every-component-explained)
   - 5.1 Config Manager
   - 5.2 Interfaces (Contracts)
   - 5.3 Chunking Strategies
   - 5.4 Embedding Providers
   - 5.5 Vector Stores
   - 5.6 Retrieval Strategies
   - 5.7 File Parsers
   - 5.8 Utilities
   - 5.9 Metadata Model
6. [Complete Data Flow Walkthroughs](#6-complete-data-flow-walkthroughs)
   - 6.1 Document Upload Flow
   - 6.2 Query Flow
   - 6.3 Deprecation Flow
7. [Configuration System In Depth](#7-configuration-system-in-depth)
8. [Design Patterns Used & Why](#8-design-patterns-used--why)
9. [Directory Structure (Annotated)](#9-directory-structure-annotated)
10. [CLI Tools Reference](#10-cli-tools-reference)
11. [Test Suite Structure](#11-test-suite-structure)
12. [Component Status Matrix](#12-component-status-matrix)
13. [Key Coding Decisions Explained](#13-key-coding-decisions-explained)
14. [Extending the Platform](#14-extending-the-platform)
15. [Implemented in Phase 2.2](#15-implemented-in-phase-22)
16. [Pending / Future Work](#16-pending--future-work)

---

## 1. What This Project Is

This is a **production-grade, multi-domain Retrieval-Augmented Generation (RAG) platform**. RAG is the dominant pattern for building AI assistants that answer questions grounded in private organizational documents — rather than hallucinating from the model's training data.

### The Problem It Solves

A company has hundreds of PDFs, DOCX files, and text documents spread across HR, Finance, Legal, and Engineering departments. Employees ask questions daily:
- "How many vacation days do I get?"
- "What is the capital expenditure limit for Q3?"
- "Does Article 5.3 of our vendor contract allow for SLA penalties?"

Sending every document to a general-purpose LLM is impossible (token limits, cost, privacy). **RAG solves this by:**

1. Pre-processing documents into a searchable vector database
2. At query time, fetching only the most relevant text passages
3. Passing those passages (not the whole document) to the LLM for an answer

### What This Platform Adds on Top of Basic RAG

| Basic RAG | This Platform |
|---|---|
| One domain | Multi-domain isolation (hr, finance, legal…) |
| Single retrieval method | Hybrid retrieval (semantic + keyword) |
| No metadata | 30+ metadata fields per chunk |
| No versioning | Document versioning and deprecation lifecycle |
| Hardcoded settings | Everything controlled by YAML config |
| Single embedding provider | Pluggable (SentenceTransformers or Gemini) |
| Single vector store | Pluggable (ChromaDB or Pinecone) |
| No audit trail | Full provenance: who uploaded, when, hash |
| No access control | Security settings per domain |

---

## 2. Why This Architecture

The project is built around **four strict separation-of-concerns layers**. Every design decision flows from one rule:

> **Zero business logic in the UI. Zero direct component instantiation outside factories.**

This matters because:
- **Testability:** The service layer can be fully tested without any UI or browser
- **Swappability:** Change the vector database from ChromaDB to Pinecone by editing one YAML line
- **Reusability:** The same `DocumentService` is called by the Gradio UI, the CLI tools, and the REST API
- **Maintainability:** Business rules (validation, metadata, deduplication) live in exactly one place

---

## 3. High-Level Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════╗
║                         EXTERNAL INTERFACES                              ║
║                                                                          ║
║   ┌──────────────────┐   ┌───────────────┐   ┌─────────────────────┐   ║
║   │  Gradio Web UI   │   │  CLI Tools    │   │  Future: FastAPI     │   ║
║   │  (app.py /       │   │  (cli/*.py)   │   │  (REST endpoint)     │   ║
║   │   app_admin.py)  │   │               │   │                      │   ║
║   └────────┬─────────┘   └──────┬────────┘   └──────────┬──────────┘   ║
║            │   ONLY service calls │                       │              ║
╚════════════╪═════════════════════╪═══════════════════════╪══════════════╝
             │                     │                       │
             ▼                     ▼                       ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                          SERVICE LAYER                                   ║
║                                                                          ║
║   ┌──────────────────────────┐   ┌──────────────────────────────────┐  ║
║   │     DocumentService      │   │         DomainService            │  ║
║   │                          │   │                                   │  ║
║   │  upload_document()        │   │  create_domain()                 │  ║
║   │  query()                  │   │    saves YAML + inits ChromaDB   │  ║
║   │  deprecate_document()     │   │  list_domains()                  │  ║
║   │  delete_document()        │   │  get_domain_vector_count()       │  ║
║   │  list_documents()         │   │  list_templates()                │  ║
║   │  list_chunks()            │   │  get_template_raw()              │  ║
║   │  get_document_info()      │   └──────────────────────────────────┘  ║
║   │  query_with_answer()      │ ← retrieval + LLM (config-driven)   ║
║   │  dry_run_retrieval()      │ ← debug mode                         ║
║   └────────────┬─────────────┘                                        ║
║                │                                                         ║
╚════════════════╪═════════════════════════════════════════════════════════╝
                                  │  delegates to
                                  ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                          PIPELINE LAYER                                  ║
║                                                                          ║
║                    ┌─────────────────────────┐                          ║
║                    │    DocumentPipeline      │                          ║
║                    │                          │                          ║
║                    │  process_document()       │ ← chunk→embed→store     ║
║                    │  query()                  │ ← multi-strategy search ║
║                    │  deprecate_document()     │ ← metadata update       ║
║                    │  delete_document()        │                          ║
║                    │  list_documents()         │ ← aggregate from chunks ║
║                    │  list_chunks()            │                          ║
║                    │  get_document_info()      │ ← aggregate stats       ║
║                    └────────────┬────────────┘                          ║
║                                 │                                        ║
╚═════════════════════════════════╪════════════════════════════════════════╝
                                  │  creates components via
                                  ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                          FACTORY LAYER                                   ║
║                                                                          ║
║  ┌─────────────────┐  ┌──────────────────┐  ┌─────────────────────┐   ║
║  │ ChunkingFactory  │  │ EmbeddingFactory  │  │ VectorStoreFactory  │   ║
║  │                  │  │                   │  │                     │   ║
║  │ recursive        │  │ sentence_trans..  │  │ chromadb            │   ║
║  │ semantic         │  │ gemini            │  │ pinecone            │   ║
║  └─────────────────┘  └──────────────────┘  └─────────────────────┘   ║
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐   ║
║  │                    RetrievalFactory                              │   ║
║  │  vector_similarity  │  bm25  │  hybrid                          │   ║
║  └─────────────────────────────────────────────────────────────────┘   ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
                                  │
                                  ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                     STORAGE & CONFIGURATION                              ║
║                                                                          ║
║   ┌─────────────────────┐      ┌──────────────────────────────────┐    ║
║   │  ChromaDB            │      │  YAML Config Files               │    ║
║   │  (Vector Store)      │      │                                   │    ║
║   │  · HNSW index        │      │  global_config.yaml  (defaults)  │    ║
║   │  · SQLite metadata   │      │  configs/domains/hr.yaml         │    ║
║   │  · File persistence  │      │  configs/domains/finance.yaml    │    ║
║   └─────────────────────┘      │  configs/templates/*.yaml        │    ║
║                                 └──────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 4. Layer-by-Layer Deep Dive

### 4.1 UI Layer (`app.py`, `app_admin.py`)

**Rule: ZERO business logic.**

The UI layer's only job is:
- Accept user input (file uploads, query text, dropdowns, buttons)
- Call one `DocumentService` method
- Display the result

```python
# CORRECT — UI calls service, displays result
service = DocumentService("hr")
result = service.upload_document(file_obj, metadata)
gr.Info(f"Uploaded: {result['chunks_ingested']} chunks")

# WRONG — UI directly touches pipeline / factories
pipeline = DocumentPipeline(config)   # ← architecture violation
pipeline.process_document(...)        # ← architecture violation
```

**Why this rule matters:** If you swap Gradio for a FastAPI endpoint tomorrow, you touch zero business logic. The service layer is the only thing the UI ever imports.

**Current UI apps:**

| File | Port | Status | Description |
|---|---|---|---|
| `app.py` | 7860 | ✅ Active | **User Chat Screen** — domain selector, conversation chatbot, LLM-grounded answers |
| `app_admin.py` | 7861 | ✅ Active | **Admin Console** — 4-tab interface for full platform management |

**Admin Console tabs (`app_admin.py`):**

| Tab | Purpose |
|---|---|
| **Templates** | View available domain templates and their raw YAML |
| **Domains** | Create new domains from templates, select LLM provider/model, list domains with vector counts |
| **Documents** | Upload files with metadata, browse/delete documents, drill-down into chunk details |
| **Playground** | Tune all RAG parameters (chunking, embeddings, retrieval, LLM) live and save as templates |

**Key workflow:** Admin creates a domain in `app_admin.py` → domain YAML saved + ChromaDB collection created immediately → uploads documents → users query via `app.py` with LLM answers grounded in those documents.

---

### 4.2 Service Layer (`core/services/`)

**The mandatory gateway between UI and all core logic.**

This layer contains two services. All validation, enrichment, error handling, and orchestration live here. It is the only layer the UI ever imports.

#### DocumentService (`core/services/document_service.py`)

```
DocumentService("hr")
    │
    ├── Loads HR domain config via ConfigManager
    ├── Creates DocumentPipeline(hr_config)
    └── Exposes 9 public methods to callers
```

**What the service does on upload:**
1. `_validate_file_type()` — checks extension against allowed list from config
2. `_validate_file_size()` — rejects files > max_file_size_mb
3. `_validate_metadata()` — ensures doc_id, title, doc_type, uploader_id are present
4. Saves to temp file (parsers need a file path, not an in-memory buffer)
5. Extracts text via `extract_text_from_file()`
6. Computes SHA-256 hash via `compute_file_hash()` for provenance
7. Enriches metadata: adds `upload_timestamp`, `source_file_hash`, `domain`
8. Delegates to `pipeline.process_document()`
9. Cleans up the temp file in a `finally` block

**What the service does on `query_with_answer()`:**
1. Applies `deprecated=False` filter by default
2. Calls `pipeline.query()` → retrieves relevant chunks (hybrid by default)
3. Resolves LLM provider/model using **priority chain**:
   - Explicit call-time params → config dict arg → **domain config `llm` section** → hardcoded fallback
4. Builds a grounded prompt from retrieved chunks
5. Calls the configured LLM (Gemini or OpenAI) and returns answer + sources + trace

**Custom exception hierarchy:**
```python
ValidationError         # bad input (file type, missing fields, etc.)
ProcessingError         # processing failed (chunking, embedding, storage)
DocumentNotFoundError   # no document with that doc_id exists
```

---

#### DomainService (`core/services/domain_service.py`)

**Manages the full domain lifecycle — creation, listing, and inspection.**

```
DomainService.create_domain(domain_id, domain_name, template_name, description)
    │
    ├── 1. Validate domain_id format (lowercase alphanumeric, 3-63 chars)
    ├── 2. Load template YAML from configs/templates/
    ├── 3. Override: collection_name = domain_id, persist_directory = ./data/chromadb/<id>
    ├── 4. Save configs/domains/<domain_id>.yaml
    ├── 5. Call DocumentService(domain_id)  ← triggers ChromaDB collection creation
    │      → DocumentPipeline.__init__()
    │        → VectorStoreFactory.create_vectorstore()
    │          → ChromaDBStore.__init__()
    │            → client.get_or_create_collection()  ← IMMEDIATE, no upload needed
    └── 6. Rollback: if ChromaDB init fails, delete the saved YAML
```

**Key design:** The ChromaDB collection is **created immediately** when a domain is created — not lazily on first upload. This gives instant feedback if the vector store is misconfigured.

**Other methods:**
```python
list_domains()              → list all YAML configs in configs/domains/
get_domain_vector_count()   → count of chunks currently in that domain's collection
list_templates()            → list all YAMLs in configs/templates/
get_template_raw()          → return raw YAML text of a named template
```

---

### 4.3 Pipeline Layer (`core/pipeline/document_pipeline.py`)

**Orchestrates the core RAG workflows. Has no knowledge of the UI.**

The pipeline is created once per `DocumentService` initialization and reused for all calls.

**Initialization sequence (in `__init__`):**
```python
1. EmbeddingFactory.create_embedder(config.embeddings)
   → stores as self.embedding_model

2. ChunkingFactory.create_chunker(config.chunking, embedding_model_name)
   → stores as self.chunker

3. VectorStoreFactory.create_vectorstore(config.vectorstore)
   → stores as self.vectorstore

4. self._init_retrieval_strategies()
   → builds BM25 index if corpus exists
   → creates retrievers for each configured strategy
   → stores as self.retrieval_strategies (dict)
```

**`_init_retrieval_strategies()` — the hybrid retrieval wiring:**
```python
def _init_retrieval_strategies(self):
    # Get strategy list from config (e.g. ["hybrid"])
    strategies = config.retrieval.strategies

    # If corpus is empty, skip BM25/hybrid (they need existing text)
    if corpus_is_empty:
        strategies = ["vector_similarity"]  # fallback

    for strategy_name in strategies:
        bm25_index = None

        # BM25 and hybrid need a pre-built keyword index
        if strategy_name in ["bm25", "hybrid"]:
            corpus, doc_ids = self.vectorstore.get_all_documents()
            bm25_index = BM25Retrieval(corpus=corpus, doc_ids=doc_ids)

        # Factory creates the right retriever
        retriever = RetrievalFactory.create_retriever(
            strategy_name=strategy_name,
            config=self.config,
            vectorstore=self.vectorstore,
            embedding_model=self.embedding_model,
            bm25_index=bm25_index,  # None for vector_similarity
        )
        retrievers[strategy_name] = retriever
```

**`process_document()` — the ingestion pipeline:**
```
text input
   │
   ├─ 1. Delete existing chunks (if replace_existing=True)
   │
   ├─ 2. self.chunker.chunk_text(text, doc_id, domain, ...)
   │      → List[ChunkMetadata]
   │
   ├─ 3. chunk_texts = [c.chunk_text for c in chunks]
   │
   ├─ 4. self.embedding_model.embed_texts(chunk_texts)
   │      → np.ndarray of shape (N, embedding_dim)
   │
   ├─ 5. self.vectorstore.upsert(chunks, embeddings)
   │
   └─ 6. return {doc_id, chunks_ingested, status, embedding_model, ...}
```

**`query()` — multi-strategy retrieval:**
```python
if strategy_name:
    # Single named strategy
    retriever = self.retrieval_strategies[strategy_name]
    return retriever.retrieve(query_text, metadata_filters, top_k)
else:
    # All configured strategies — aggregate and deduplicate
    for name, retriever in self.retrieval_strategies.items():
        results = retriever.retrieve(...)
        aggregated_results.extend(results)

    return self._deduplicate_results(aggregated_results)
    # Dedup keeps highest score per chunk_id, sorts descending
```

**`get_document_info()` — aggregate from chunks:**
```python
# ChromaDB collection.get(where={"doc_id": doc_id})
# → all chunk metadatas for this document
# Aggregate: count chunks, extract first-seen timestamp, check deprecated flag
return {
    "doc_id": ..., "title": ..., "chunk_count": len(metadatas),
    "upload_timestamp": ..., "version": ..., "deprecated": any(...),
    "embedding_model": ..., "chunking_strategy": ...
}
```

**`deprecate_document()` — metadata update:**
```python
# 1. Fetch all chunk IDs for doc_id
# 2. Merge {deprecated: True, deprecated_date, deprecation_reason} into each chunk's metadata
# 3. ChromaDB collection.update(ids=ids, metadatas=merged_metadatas)
# After this, all queries with deprecated=False will exclude this doc
```

---

### 4.4 Factory Layer

**No concrete class is ever instantiated outside a factory.**

Each factory:
- Reads the relevant config section
- Maps the strategy/provider name to the implementation class
- Returns an instance that implements the correct interface

#### ChunkingFactory (`core/factories/chunking_factory.py`)

```python
class ChunkingFactory:
    _registry = {
        "recursive": RecursiveChunker,
        "semantic":  SemanticChunker,
    }

    @staticmethod
    def create_chunker(config, embedding_model_name) -> ChunkerInterface:
        strategy = config.strategy          # "recursive" or "semantic"
        cls = ChunkingFactory._registry[strategy]
        return cls(config=config, embedding_model_name=embedding_model_name)
```

The registry pattern means adding a new chunker (e.g. `"sliding_window"`) requires only:
1. Writing a class that implements `ChunkerInterface`
2. Adding one line to the registry: `"sliding_window": SlidingWindowChunker`

#### EmbeddingFactory (`core/factories/embedding_factory.py`)

```python
class EmbeddingFactory:
    @staticmethod
    def create_embedder(config) -> EmbeddingInterface:
        provider = config.provider   # "sentence_transformers" or "gemini"

        if provider == "sentence_transformers":
            return SentenceTransformerEmbeddings(
                model_name=config.model_name,
                device=config.device,
                batch_size=config.batch_size
            )
        elif provider == "gemini":
            api_key = os.environ.get(config.api_key.replace("${","").replace("}",""))
            return GeminiEmbeddings(api_key=api_key, model_name=config.model_name)
```

#### VectorStoreFactory (`core/factories/vectorstore_factory.py`)

```python
class VectorStoreFactory:
    @staticmethod
    def create_vectorstore(config) -> VectorStoreInterface:
        provider = config.provider   # "chromadb" or "pinecone"

        if provider == "chromadb":
            return ChromaDBStore(
                persist_directory=config.persist_directory,
                collection_name=config.collection_name
            )
        elif provider == "pinecone":
            return PineconeStore(api_key=..., index_name=...)
```

#### RetrievalFactory (`core/factories/retrieval_factory.py`)

```python
class RetrievalFactory:
    @staticmethod
    def create_retriever(strategy_name, config, vectorstore,
                         embedding_model, bm25_index) -> RetrievalInterface:
        if strategy_name == "vector_similarity":
            return VectorSimilarityRetrieval(vectorstore, embedding_model, config)
        elif strategy_name == "bm25":
            return bm25_index  # BM25Retrieval is itself the retriever
        elif strategy_name == "hybrid":
            return HybridRetrieval(vectorstore, embedding_model, bm25_index, config)
```

---

### 4.5 Storage & Configuration Layer

**ChromaDB** stores:
- Chunk embeddings (float32 vectors, 384 or 768 dimensions)
- Chunk text (the raw string)
- All metadata fields (stored as ChromaDB metadata dict)

It uses **HNSW** (Hierarchical Navigable Small World) — a graph-based approximate nearest neighbor index that gives O(log N) query time.

**YAML configs** control all behavior. The `ConfigManager` merges them at startup:

```
global_config.yaml       ← base defaults for all domains
     +
configs/domains/hr.yaml  ← HR-specific overrides
     =
effective HR config      ← what DocumentPipeline receives
```

---

## 5. Every Component Explained

### 5.1 Config Manager (`core/config_manager.py`)

**What it does:** Loads YAML files, deep-merges domain config over global defaults, injects environment variables, and validates the result using Pydantic models.

**Key method:**
```python
def load_domain_config(domain_name: str) -> DomainConfig:
    global_raw   = self._load_yaml(self.global_config_path)
    domain_raw   = self._load_yaml(self.domain_dir / f"{domain_name}.yaml")
    merged       = self._merge_dicts(global_raw, domain_raw)
    injected     = self._inject_env_vars(merged)
    return DomainConfig(**injected)    # Pydantic validates here
```

**Environment variable injection:**
```yaml
# In YAML:
embeddings:
  api_key: "${GEMINI_API_KEY}"

# ConfigManager finds ${VAR_NAME} patterns and replaces with os.environ["VAR_NAME"]
```

**Pydantic validation catches:**
- Wrong types (e.g. chunk_size as a string instead of int)
- Out-of-range values (e.g. overlap >= chunk_size)
- Missing required fields

**Pydantic config models hierarchy:**
```
DomainConfig
├── ChunkingConfig
│   ├── RecursiveChunkingConfig (chunk_size, overlap)
│   └── SemanticChunkingConfig (similarity_threshold, max_chunk_size)
├── EmbeddingConfig (provider, model_name, api_key, device, batch_size)
├── VectorStoreConfig (provider, collection_name, persist_directory)
├── RetrievalConfig
│   ├── strategies: List[str]
│   └── HybridRetrievalConfig (alpha: float)
├── LLMConfig (provider, model_name, temperature, max_tokens, api_key)  ← NEW
├── SecurityConfig (allowed_file_types, max_file_size_mb)
└── MetadataConfig (track_versions, enable_deprecation, compute_file_hash)
```

**LLMConfig — new in Phase 2.2:**
```python
class LLMConfig(BaseModel):
    provider: str = "gemini"           # validated: "gemini" | "openai"
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.2           # 0.0 – 1.0
    max_tokens: int = 512              # 64 – 8192
    api_key: Optional[str] = None      # injected from environment if None
```

Each domain's YAML can specify its own LLM. At query time, `DocumentService.query_with_answer()` reads this config and calls the correct provider automatically — no manual parameter passing needed from the UI.

**Single source of truth:** All Pydantic models (`LLMConfig`, `DomainConfig`, etc.) are defined in `core/config_manager.py`. The `models/domain_config.py` file is a **thin re-export wrapper only** — it imports and re-exports from `core/config_manager.py` for backwards compatibility. Never duplicate model definitions.

---

### 5.2 Interfaces (Contracts)

All four interfaces are Python Abstract Base Classes (ABCs) that define the contract every implementation must fulfill.

#### `ChunkerInterface` (`core/interfaces/chunking_interface.py`)
```python
class ChunkerInterface(ABC):
    @abstractmethod
    def chunk_text(
        self, text, doc_id, domain, source_file_path,
        file_hash, uploader_id=None, page_num=None
    ) -> List[ChunkMetadata]:
        ...

    @abstractmethod
    def get_strategy_name(self) -> str:
        ...
```

#### `EmbeddingInterface` (`core/interfaces/embedding_interface.py`)
```python
class EmbeddingInterface(ABC):
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        ...  # Returns shape (N, embedding_dim), float32, L2-normalized

    @abstractmethod
    def get_model_name(self) -> str: ...

    @abstractmethod
    def get_embedding_dimension(self) -> int: ...
```

#### `VectorStoreInterface` (`core/interfaces/vectorstore_interface.py`)
```python
class VectorStoreInterface(ABC):
    @abstractmethod
    def upsert(self, chunks: List[ChunkMetadata], embeddings: np.ndarray) -> None: ...

    @abstractmethod
    def search(
        self, query_embedding: np.ndarray,
        top_k: int, filters: Optional[Dict] = None
    ) -> List[Dict]: ...

    @abstractmethod
    def delete_by_doc_id(self, doc_id: str) -> None: ...

    @abstractmethod
    def get_all_documents(self) -> Tuple[List[str], List[str]]:
        ...  # Returns (corpus_texts, chunk_ids) for BM25 indexing
```

#### `RetrievalInterface` (`core/interfaces/retrieval_interface.py`)
```python
class RetrievalInterface(ABC):
    @abstractmethod
    def retrieve(
        self, query_text: str,
        metadata_filters: Optional[Dict] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        ...  # Each dict: {id, score, document, metadata, strategy}
```

---

### 5.3 Chunking Strategies

#### RecursiveChunker (`core/chunking/recursive_chunker.py`)

**Algorithm — sliding window with overlap:**

```
Document: "AAAAABBBBBCCCCCDDDDDEEEEE" (chunk_size=10, overlap=2)

Chunk 1: AAAAABBBBB  (pos 0-10)
Chunk 2: BBBBBCCCCC  (pos 8-18)  ← starts 2 back (overlap=2)
Chunk 3: CCCCCDDDDDE (pos 16-26)
Chunk 4: DDDDDEEEEE  (pos 24-34)
```

Why overlap? Consider this split without overlap:
```
❌ Chunk 1: "Employee benefits include 15 vacation d"
❌ Chunk 2: "ays per year and unlimited sick leave."
```

With overlap both chunks contain the full context:
```
✅ Chunk 1: "Employee benefits include 15 vacation days"
✅ Chunk 2: "vacation days per year and unlimited sick leave."
```

**Key implementation detail:** Chunks are word-boundary aligned where possible to avoid splitting in the middle of a word.

**Output per chunk:**
```python
ChunkMetadata(
    chunk_id="uuid-auto-generated",
    doc_id="handbook_2025",
    domain="hr",
    chunk_text="Employee benefits include 15 vacation days...",
    char_range=(0, 500),     # position in original document
    uploader_id="admin",
    upload_timestamp="2026-02-22T10:00:00",
    source_file_hash="sha256hex...",
    chunking_strategy="recursive",
    chunking_params={"chunk_size": 500, "overlap": 50},
    ...
)
```

**Best for:** Structured documents, policies, reports. Fast, no dependencies.

**Config:**
```yaml
chunking:
  strategy: recursive
  chunk_size: 500      # characters per chunk
  overlap: 50          # characters repeated between adjacent chunks
```

---

#### SemanticChunker (`core/chunking/semantic_chunker.py`)

**Algorithm — similarity-based grouping:**

```
1. Split text into sentences using regex: [.!?]\s+

2. Embed all sentences in one batch (all-MiniLM-L6-v2)

3. For each sentence, compare cosine similarity with previous sentence:
   - similarity >= threshold (0.7) AND chunk not too large → same chunk
   - similarity < threshold OR chunk full → NEW chunk starts here

4. Result: topically coherent chunks, variable size
```

**Cosine similarity formula:**
```
cos(A, B) = (A · B) / (||A|| * ||B||)
Range: 0.0 (completely different) → 1.0 (identical)
```

**Example:**
```
Sentence 1: "Employees get 15 vacation days."              │
Sentence 2: "Leave must be requested 2 weeks in advance."  │ → Chunk 1
Sentence 3: "Unused leave can be carried forward."         │ (sim ≈ 0.82)
Sentence 4: "The CEO announced record profits for Q3."     ← new chunk starts
(sim ≈ 0.15 — completely different topic)
```

**Best for:** Narrative text, meeting notes, articles where topics shift gradually.

**Config:**
```yaml
chunking:
  strategy: semantic
  similarity_threshold: 0.7  # sentences below this → new chunk
  max_chunk_size: 1000        # hard ceiling in characters
```

---

### 5.4 Embedding Providers

#### SentenceTransformerEmbeddings (`core/embeddings/sentence_transformer_embeddings.py`)

**What it does:** Converts text into dense numerical vectors using a pre-trained transformer model that runs locally.

```python
class SentenceTransformerEmbeddings(EmbeddingInterface):
    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu", batch_size=32):
        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,  # L2 norm = 1.0 → enables cosine via dot product
            show_progress_bar=len(texts) > 50
        )
        return embeddings.astype(np.float32)
```

**Why normalize?** When all vectors have L2 norm = 1.0, cosine similarity equals the dot product. ChromaDB's HNSW index is optimized for cosine distance, so normalized embeddings give the best results.

**Popular models and their tradeoffs:**

| Model | Dimensions | Speed | Quality |
|---|---|---|---|
| `all-MiniLM-L6-v2` | 384 | ⚡ Very fast | Good (default) |
| `all-mpnet-base-v2` | 768 | Moderate | Best |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Fast | Good, multilingual |

**Config:**
```yaml
embeddings:
  provider: sentence_transformers
  model_name: all-MiniLM-L6-v2
  device: cpu         # or "cuda" for GPU
  batch_size: 32
```

---

#### GeminiEmbeddings (`core/embeddings/gemini_embeddings.py`)

**What it does:** Calls Google's Gemini embedding API to generate 768-dimensional embeddings.

```python
class GeminiEmbeddings(EmbeddingInterface):
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        embeddings = []
        for batch in self._batch(texts):
            result = genai.embed_content(
                model=self.model_name,
                content=batch,
                task_type=self.task_type  # "RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY"
            )
            embeddings.extend(result['embedding'])
        return np.array(embeddings, dtype=np.float32)
```

**Task types matter:** Gemini uses different internal representations depending on the task:
- `RETRIEVAL_DOCUMENT` — used when indexing chunks at upload time
- `RETRIEVAL_QUERY` — used when embedding a query at search time

Using mismatched task types degrades retrieval quality significantly.

**Config:**
```yaml
embeddings:
  provider: gemini
  model_name: models/embedding-001
  api_key: "${GEMINI_API_KEY}"    # injected from environment
  batch_size: 32
```

---

### 5.5 Vector Stores

#### ChromaDBStore (`core/vectorstores/chromadb_store.py`)

**What it does:** Wraps the ChromaDB library to provide persistent vector storage with metadata filtering.

```python
class ChromaDBStore(VectorStoreInterface):
    def __init__(self, persist_directory: str, collection_name: str):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # cosine distance metric
        )
```

**Upsert operation:**
```python
def upsert(self, chunks, embeddings):
    self.collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c.chunk_text for c in chunks],
        metadatas=[c.model_dump() for c in chunks]  # all 30+ fields
    )
    # "upsert" = update if id exists, insert if not → idempotent
```

**Search with filters:**
```python
def search(self, query_embedding, top_k, filters=None):
    results = self.collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
        where=filters,   # ChromaDB metadata filter syntax: {"doc_type": "policy"}
        include=["documents", "metadatas", "distances"]
    )
    # Convert ChromaDB's distance to similarity: score = 1 - distance
    return [{"id": id, "score": 1-dist, "document": doc, "metadata": meta}
            for id, doc, meta, dist in zip(...)]
```

**`get_all_documents()` — required for BM25:**
```python
def get_all_documents(self):
    raw = self.collection.get(include=["documents", "ids"])
    return raw["documents"], raw["ids"]  # (corpus_texts, chunk_ids)
```

**Storage layout on disk:**
```
data/chroma_db/
├── hr_docs/         ← each collection is a separate directory
│   ├── *.bin        ← HNSW index (the vectors)
│   └── chroma.sqlite3  ← metadata and document text
└── finance_docs/
    └── ...
```

---

### 5.6 Retrieval Strategies

#### VectorSimilarityRetrieval (`core/retrievals/vector_similarity_retrieval.py`)

**Pure semantic / dense search.**

```python
def retrieve(self, query_text, metadata_filters=None, top_k=10):
    # 1. Embed the query
    query_vector = self.embedding_model.embed_texts([query_text])[0]

    # 2. Search in vector store (HNSW approximate nearest neighbor)
    results = self.vectorstore.search(
        query_embedding=query_vector,
        top_k=top_k,
        filters=metadata_filters
    )
    return results
```

**Best for:** "What is the vacation policy?" (semantic / conceptual)
**Bad for:** "Form HR-22" (specific term — BM25 finds it better)

---

#### BM25Retrieval (`core/retrievals/bm25_retrieval.py`)

**Sparse / keyword-based search. No embeddings needed.**

BM25 is the algorithm behind classic search engines. It ranks documents by:
- **Term frequency (TF):** How often the query word appears in the chunk, with diminishing returns
- **Inverse document frequency (IDF):** Rare words are weighted more than common ones
- **Length normalization:** Shorter documents aren't penalized

```python
class BM25Retrieval(RetrievalInterface):
    def __init__(self, corpus, doc_ids, k1=1.5, b=0.75):
        tokenized = [text.lower().split() for text in corpus]
        self.bm25 = BM25Okapi(tokenized, k1=k1, b=b)
        self.corpus = corpus
        self.doc_ids = doc_ids

    def retrieve(self, query_text, metadata_filters=None, top_k=10):
        query_tokens = query_text.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        # Get indices sorted by score descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id": self.doc_ids[i],
                "score": float(scores[i]),
                "document": self.corpus[i],
                "metadata": self.metadata[i] if self.metadata else {},
            }
            for i in top_indices if scores[i] > 0
        ]
```

**BM25 parameters:**
- `k1=1.5` — controls term frequency saturation. Lower → keyword appears once is nearly as good as many times
- `b=0.75` — length normalization. 1.0 = full normalization, 0 = no normalization

**Best for:** "Article 5.3", "Form HR-22", exact company-specific terminology
**Bad for:** Paraphrases and synonyms (not in the keyword index)

---

#### HybridRetrieval (`core/retrievals/hybrid_retrieval.py`)

**Combines dense (semantic) + sparse (keyword) search with alpha weighting.**

```python
class HybridRetrieval(RetrievalInterface):
    def retrieve(self, query_text, metadata_filters=None, top_k=10):
        # 1. Dense search (fetch 2x top_k as candidate pool)
        query_vector = self.embedding_model.embed_texts([query_text])[0]
        dense_results = self.vectorstore.search(query_vector, top_k*2, metadata_filters)

        # 2. Sparse search (BM25, also 2x candidate pool)
        sparse_results = self.bm25_index.retrieve(query_text, top_k=top_k*2)

        # 3. Collect all unique chunk IDs
        all_ids = {r["id"] for r in dense_results + sparse_results}

        # 4. Build score lookup for each list
        dense_scores = {r["id"]: r["score"] for r in dense_results}
        sparse_scores = {r["id"]: r["score"] for r in sparse_results}

        # 5. Min-max normalize each score set to [0, 1]
        dense_scores  = self._normalize(dense_scores)
        sparse_scores = self._normalize(sparse_scores)

        # 6. Combine with alpha weighting
        final_scores = {}
        for chunk_id in all_ids:
            d = dense_scores.get(chunk_id, 0.0)
            s = sparse_scores.get(chunk_id, 0.0)
            final_scores[chunk_id] = self.alpha * d + (1 - self.alpha) * s

        # 7. Sort and return top_k
        sorted_ids = sorted(final_scores, key=lambda k: final_scores[k], reverse=True)
        return [self._build_result(id, final_scores[id]) for id in sorted_ids[:top_k]]
```

**Why normalize first?** Dense scores (cosine similarity, range ≈ 0-1) and BM25 scores (arbitrary positive numbers) cannot be directly combined. Min-max normalization puts both on [0, 1] so alpha weighting is meaningful.

**Alpha guide:**

| alpha | Effect |
|---|---|
| 1.0 | Pure semantic — best for conceptual queries |
| 0.7 | Mostly semantic, some keyword (production default) |
| 0.5 | Equal weight — balanced |
| 0.3 | Mostly keyword — for technical documents with specific terms |
| 0.0 | Pure BM25 — like a keyword search engine |

---

### 5.7 File Parsers (`core/utils/file_parsers/`)

```
parser_factory.py → decides which parser to use based on file extension
pdf_processor.py  → uses PyMuPDF or pdfplumber to extract text + page numbers
docx_processor.py → uses python-docx to iterate paragraphs
txt_processor.py  → reads UTF-8 text, strips control characters
```

**Parser factory function:**
```python
def extract_text_from_file(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    parser = FileParserFactory.create_parser(ext)
    return parser.extract_text(file_path)
```

**PDF special handling:** PDFs may have page numbers embedded. The PDF processor tracks which page each text block comes from and can return per-page chunks if needed.

---

### 5.8 Utilities

#### `core/utils/hashing.py`

```python
def compute_file_hash(file_obj: BinaryIO) -> str:
    sha256 = hashlib.sha256()
    file_obj.seek(0)
    for chunk in iter(lambda: file_obj.read(4096), b""):
        sha256.update(chunk)   # reads in 4KB blocks → handles large files efficiently
    file_obj.seek(0)           # reset for downstream use
    return sha256.hexdigest()  # 64-character hex string
```

**Uses:**
- **Deduplication:** If `source_file_hash` already exists in the collection, skip re-indexing
- **Integrity:** Verify a file hasn't been tampered with
- **Provenance:** Every chunk stores the hash of its source file

#### `core/utils/validation.py`

```python
def validate_file_type(filename: str, allowed_types: set) -> None:
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed_types:
        raise ValidationError(f"File type '.{ext}' not allowed. Allowed: {allowed_types}")

def validate_doc_id(doc_id: str) -> None:
    # Must be 3-100 chars, only alphanumeric/underscore/hyphen
    if not re.match(r'^[a-zA-Z0-9_-]{3,100}$', doc_id):
        raise ValidationError(f"Invalid doc_id format: {doc_id}")
```

---

### 5.9 Metadata Model (`models/metadata_models.py`)

Every chunk stored in ChromaDB carries this Pydantic model serialized as a flat dict.

```python
class ChunkMetadata(BaseModel):

    # ── IDENTITY ──────────────────────────────
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str                    # required — links all chunks of a document
    domain: str                    # required — domain isolation

    # ── CONTENT ───────────────────────────────
    chunk_text: str                # required — the actual text
    title: Optional[str] = None
    doc_type: Optional[str] = None # "policy", "faq", "manual", "guideline"
    tags: List[str] = []
    page_num: Optional[int] = None
    char_range: Optional[Tuple[int, int]] = None  # (start, end) in source

    # ── PROVENANCE ────────────────────────────
    uploader_id: Optional[str] = None
    upload_timestamp: Optional[str] = None        # UTC ISO format
    source_file_path: Optional[str] = None
    source_file_hash: Optional[str] = None        # SHA-256

    # ── VERSIONING ────────────────────────────
    version: str = "1.0"
    document_version: Optional[str] = None
    last_updated_timestamp: Optional[str] = None

    # ── PROCESSING ────────────────────────────
    embedding_model_name: Optional[str] = None
    embedding_dimension: Optional[int] = None
    embedding_version: Optional[str] = None
    chunking_strategy: Optional[str] = None
    chunking_params: Optional[Dict] = None
    chunk_type: str = "text"           # "text", "code", "list"
    processing_timestamp: Optional[str] = None

    # ── LIFECYCLE (Phase 2) ───────────────────
    deprecated: bool = False
    deprecated_date: Optional[str] = None
    deprecation_reason: Optional[str] = None
    superseded_by_chunk_id: Optional[str] = None  # points to replacement

    # ── QUALITY & AUTHORITY ───────────────────
    is_authoritative: bool = True
    confidence_score: Optional[float] = None      # 0.0 – 1.0
    authority_level: AuthorityLevel = AuthorityLevel.OFFICIAL
    review_status: ReviewStatus = ReviewStatus.APPROVED
    reviewed_by: Optional[str] = None
    reviewed_date: Optional[str] = None
    custom_metadata: Dict[str, Any] = {}          # domain-specific extensions
```

**Enums used:**
```python
class AuthorityLevel(str, Enum):
    OFFICIAL   = "official"    # Company policy — highest trust
    APPROVED   = "approved"    # Formally reviewed
    DRAFT      = "draft"       # Work in progress
    ARCHIVED   = "archived"    # Historical, not current
    DEPRECATED = "deprecated"  # Superseded

class ReviewStatus(str, Enum):
    APPROVED   = "approved"
    PENDING    = "pending"
    REJECTED   = "rejected"
    IN_REVIEW  = "in_review"
```

---

## 6. Complete Data Flow Walkthroughs

### 6.1 Document Upload Flow

```
User uploads "HR_Handbook_2025.pdf" via Gradio
    │
    ▼
app_phase2.py: service.upload_document(file_obj, metadata)
    │
    │  ── SERVICE LAYER (document_service.py) ──────────────────────────
    ▼
1. validate_file_type("HR_Handbook_2025.pdf", {"pdf","docx","txt"})
   → ext = "pdf" ✅

2. save file to temp: /tmp/tmpXYZ.pdf
   (parsers need file path, not in-memory buffer)

3. extract_text_from_file("/tmp/tmpXYZ.pdf")
   → PDFProcessor.extract_text() → 45,000 characters

4. compute_file_hash(file_obj)
   → SHA-256 = "a3f2c7d9..."

5. Enrich metadata:
   enriched = {
     doc_id: "handbook_2025",
     title: "HR Handbook 2025",
     doc_type: "policy",
     uploader_id: "alice@corp.com",
     source_file: "HR_Handbook_2025.pdf",
     source_file_hash: "a3f2c7d9...",
     upload_timestamp: "2026-02-22T10:00:00",
     domain: "hr"
   }

6. pipeline.process_document(text, doc_id, domain, ...)
    │
    │  ── PIPELINE LAYER (document_pipeline.py) ─────────────────────────
    ▼
7. Delete existing chunks if replace_existing=True
   vectorstore.delete_by_doc_id("handbook_2025")

8. chunker.chunk_text(text="45,000 chars...", doc_id, domain, ...)
   RecursiveChunker:
   → 500 chars at a time, 50-char overlap
   → 93 chunks created
   → Each chunk is a ChunkMetadata object

9. chunk_texts = [c.chunk_text for c in chunks]

10. embedding_model.embed_texts(chunk_texts)
    SentenceTransformerEmbeddings:
    → all-MiniLM-L6-v2 processes 93 texts in batches of 32
    → returns np.ndarray of shape (93, 384), dtype=float32

11. vectorstore.upsert(chunks=chunks, embeddings=embeddings)
    ChromaDBStore:
    → collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=embeddings.tolist(),
        documents=[chunk.chunk_text for chunk in chunks],
        metadatas=[chunk.model_dump() for chunk in chunks]
      )
    │
    ▼
12. Return {"doc_id": "handbook_2025", "chunks_ingested": 93, "status": "success",
            "embedding_model": "all-MiniLM-L6-v2", "chunking_strategy": "recursive"}
    │
    ▼
13. finally: Path(tmp_path).unlink()   # cleanup temp file

Return to UI → "✅ Uploaded 93 chunks"
```

---

### 6.2 Query Flow (with LLM Answer Generation)

```
User types: "How many vacation days?"  in app.py (domain: "hr")
    │
    ▼
app.py: svc.query_with_answer(query_text="How many vacation days?")
    │
    │  ── SERVICE LAYER (document_service.py) ────────────────────────────
    ▼
1. Add deprecated filter: metadata_filters = {"deprecated": False}

2. pipeline.query("How many vacation days?", strategy_name="hybrid",
                  metadata_filters={"deprecated": False}, top_k=10)
    │
    │  ── PIPELINE LAYER ─────────────────────────────────────────────────
    ▼
3. retriever = self.retrieval_strategies["hybrid"]
   → HybridRetrieval instance

4. retriever.retrieve("How many vacation days?", {"deprecated":False}, top_k=10)
    │
    │  ── HYBRID RETRIEVAL ────────────────────────────────────────────────
    ▼
5. DENSE PATH:
   a. query_vector = embedding_model.embed_texts(["How many vacation days?"])[0]
      → shape (384,), float32, normalized

   b. dense_results = vectorstore.search(query_vector, top_k=20, {"deprecated":False})
      ChromaDB HNSW search → 20 semantically similar chunks
      Each result: {id, score (0-1), document, metadata}

6. SPARSE PATH:
   a. bm25_index.retrieve("How many vacation days?", top_k=20)
      Tokenize: ["how", "many", "vacation", "days"]
      BM25 scores all corpus chunks
      → 20 highest-scoring chunks by keyword match

7. SCORE NORMALIZATION + COMBINE (alpha=0.7 for HR domain):
   final_score(id) = 0.7 * dense_norm(id) + 0.3 * sparse_norm(id)
   → Sort descending → return top 10

    │  ── BACK IN SERVICE LAYER (query_with_answer) ───────────────────────
    ▼
8. LLM PROVIDER RESOLUTION (priority chain):
   domain_llm = hr_domain_config.llm   ← LLMConfig from hr.yaml
   provider = domain_llm.provider      → "gemini"
   model    = domain_llm.model_name    → "gemini-1.5-flash"
   temp     = domain_llm.temperature   → 0.2
   tokens   = domain_llm.max_tokens    → 512

9. BUILD GROUNDED PROMPT:
   prompt = f"""
   You are an HR assistant. Answer only using the provided context.

   Context:
   [1] "Employees are entitled to 15 vacation days per year..."
   [2] "Unused leave must be carried forward within 12 months..."
   [3] ...

   Question: How many vacation days?
   """

10. CALL LLM (Gemini):
    import google.generativeai as genai
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(
        prompt,
        generation_config={"temperature": 0.2, "max_output_tokens": 512}
    )
    answer = response.text
    → "Employees are entitled to 15 vacation days per year based on the HR Handbook."

11. RETURN to app.py:
    {
      "answer": "Employees are entitled to 15 vacation days per year...",
      "sources": [
        {"title": "HR Handbook 2025", "page": 12, "score": 0.958},
        {"title": "Leave Policy", "page": 3, "score": 0.876},
        ...
      ],
      "trace": {
        "strategy": "hybrid",
        "raw_results_count": 10,
        "llm_provider": "gemini",
        "llm_model": "gemini-1.5-flash"
      }
    }

12. app.py appends source citations to the answer and shows in chatbot
```

**LLM Provider Switching:** To switch the HR domain to OpenAI GPT-4, change only `configs/domains/hr.yaml`:
```yaml
llm:
  provider: openai
  model_name: gpt-4o
  temperature: 0.2
```
Zero code changes required.

---

### 6.3 Deprecation Flow

```
Admin runs: python -m cli.manage deprecate --domain hr --doc-id handbook_2023
            --reason "Superseded by handbook_2025"
    │
    ▼
manage.py: service.deprecate_document("handbook_2023", reason="...", superseded_by=None)
    │
    │  ── SERVICE LAYER ──────────────────────────────────────────────────
    ▼
1. Validate: doc_id not empty, reason not empty

2. pipeline.deprecate_document("handbook_2023", reason="...", superseded_by=None)
    │
    │  ── PIPELINE LAYER ─────────────────────────────────────────────────
    ▼
3. collection.get(where={"doc_id": "handbook_2023"}, include=["metadatas"])
   → ids = ["chunk_001", "chunk_002", ..., "chunk_045"]  (45 chunks found)

4. For each chunk, merge update into existing metadata:
   updated_fields = {
     "deprecated": True,
     "deprecated_date": "2026-02-22T11:00:00",
     "deprecation_reason": "Superseded by handbook_2025"
   }
   merged_metadatas = [{**existing, **updated_fields} for existing in all_chunk_metadatas]

5. collection.update(ids=ids, metadatas=merged_metadatas)
   → 45 chunks updated in ChromaDB

Return: {"doc_id": "handbook_2023", "chunks_deprecated": 45,
         "deprecated_date": "2026-02-22T11:00:00", ...}

Effect: Any future query with metadata_filters={"deprecated": False}
        will NOT return any chunk from handbook_2023
```

---

## 7. Configuration System In Depth

### Config Hierarchy and Merging

```
global_config.yaml                 configs/domains/hr.yaml
─────────────────                  ──────────────────────
chunking:                          chunking:
  strategy: recursive     ┐          chunk_size: 300      ← overrides global 500
  chunk_size: 500         │  deep
  overlap: 50             │  merge
                          ┘
embeddings:                        embeddings:
  provider: gemini                   model_name: all-MiniLM-L6-v2 ← override
  model_name: gemini-1.0-pro

                    ↓  result  ↓
             effective_hr_config:
               chunking:
                 strategy: recursive    ← from global
                 chunk_size: 300        ← overridden by hr.yaml
                 overlap: 50            ← from global
               embeddings:
                 provider: gemini       ← from global
                 model_name: all-MiniLM-L6-v2 ← overridden by hr.yaml
```

### Environment Variable Injection

```yaml
# In config file:
embeddings:
  api_key: "${GEMINI_API_KEY}"

# ConfigManager._inject_env_vars() replaces this with:
embeddings:
  api_key: "AIzaSy..."    ← actual value from os.environ
```

This means API keys are **never** in version-controlled files. Only `${VAR_NAME}` placeholders are committed.

### Global Config (`configs/global_config.yaml`)

The global config sets defaults for all domains. Any key can be overridden in a domain-specific YAML.

```yaml
chunking:
  strategy: recursive
  recursive:
    chunk_size: 500
    overlap: 50

embeddings:
  provider: sentence_transformers   # free local model, no API key required
  model_name: all-MiniLM-L6-v2
  device: cpu
  batch_size: 32

vectorstore:
  provider: chromadb
  persist_directory: ./data/chromadb
  collection_name: default

retrieval:
  strategies:
    - hybrid                        # only valid values: hybrid, bm25, vector_similarity
  top_k: 10
  similarity: cosine
  hybrid:
    alpha: 0.7
    normalize_scores: true

llm:                                # ← NEW: default LLM for all domains
  provider: gemini
  model_name: gemini-1.5-flash
  temperature: 0.2
  max_tokens: 512

security:
  allowed_file_types: [pdf, docx, txt]
  max_file_size_mb: 20

metadata:
  track_versions: true
  enable_deprecation: true
  compute_file_hash: true
  extract_page_numbers: true
```

### Domain Config Examples

**HR domain (`configs/domains/hr.yaml`)** — balanced policy Q&A:
```yaml
domain_id: hr
name: Human Resources
description: "HR policies, employee handbook, leave rules, benefits"

vectorstore:
  provider: chromadb
  collection_name: hr           # matches domain_id
  persist_directory: ./data/chromadb/hr

chunking:
  strategy: recursive
  recursive:
    chunk_size: 600
    overlap: 80

embeddings:
  provider: sentence_transformers
  model_name: all-MiniLM-L6-v2
  device: cpu
  batch_size: 32

retrieval:
  strategies:
    - hybrid
  top_k: 10
  similarity: cosine
  hybrid:
    alpha: 0.7                    # 70% semantic, 30% keyword
    normalize_scores: true

llm:                              # ← domain-specific LLM
  provider: gemini
  model_name: gemini-1.5-flash
  temperature: 0.2
  max_tokens: 512

security:
  allowed_file_types: [pdf, docx, txt]
  max_file_size_mb: 20
```

**Finance domain (`configs/domains/finance.yaml`)** — precise financial answers:
```yaml
domain_id: finance
# ... (smaller chunks 400, overlap 60, alpha 0.6 — more keyword weight)
llm:
  provider: gemini
  model_name: gemini-1.5-flash
  temperature: 0.1              # very low — financial answers must be precise
  max_tokens: 512
```

**Legal domain (`configs/domains/legal.yaml`)** — high-quality legal precision:
```yaml
domain_id: legal
# ... (larger chunks 800, all-mpnet-base-v2 768-dim embeddings)
llm:
  provider: gemini
  model_name: gemini-1.5-pro    # higher quality model for legal precision
  temperature: 0.1
  max_tokens: 1024              # legal answers may be longer
```

### Active Domains

| domain_id | Embedding Model | Chunk Size | LLM Model | Alpha |
|---|---|---|---|---|
| `hr` | all-MiniLM-L6-v2 | 600 | gemini-1.5-flash | 0.70 |
| `finance` | all-MiniLM-L6-v2 | 400 | gemini-1.5-flash | 0.60 |
| `legal` | all-mpnet-base-v2 | 800 | gemini-1.5-pro | 0.65 |

---

## 8. Design Patterns Used & Why

### Factory Pattern

**Problem:** The pipeline needs to create embedders, chunkers, vector stores. If it instantiated them directly (`model = SentenceTransformerEmbeddings(...)`), swapping to Gemini would require changing the pipeline code.

**Solution:** All creation goes through factories. The pipeline only ever calls `EmbeddingFactory.create_embedder(config)`. The factory reads `config.provider` and returns the right object.

```python
# WITHOUT factory (bad — tightly coupled)
if config.provider == "sentence_transformers":
    model = SentenceTransformerEmbeddings(...)
elif config.provider == "gemini":
    model = GeminiEmbeddings(...)
# This code exists in the pipeline — business logic mixed with creation

# WITH factory (good — decoupled)
model = EmbeddingFactory.create_embedder(config)
# Zero if/elif in the pipeline — adding OpenAI embeddings only touches EmbeddingFactory
```

### Strategy Pattern

**Problem:** Different chunking algorithms, different retrieval methods. Code would be a mess of if/elif chains.

**Solution:** Each strategy implements a common interface. The caller (pipeline) holds a reference to the interface and calls `chunk_text()` or `retrieve()` — it never knows which concrete class it has.

```python
# Polymorphism: same call, different behavior
chunker = ChunkingFactory.create_chunker(config)   # RecursiveChunker or SemanticChunker
chunks  = chunker.chunk_text(text, ...)            # same call either way
```

### Repository Pattern

**Problem:** The pipeline would need to know ChromaDB's API (`collection.query(...)`) directly, making it impossible to switch to Pinecone.

**Solution:** `VectorStoreInterface` is a stable abstraction. The pipeline calls `vectorstore.search(query_vector, top_k)` — the same call works for ChromaDB or Pinecone.

### Template Method Pattern (in Pipeline)

`DocumentPipeline.__init__()` defines the initialization algorithm:
1. Create embedder
2. Create chunker
3. Create vector store
4. Create retrieval strategies

Each step calls a factory. The algorithm is fixed (template), but the concrete classes (method implementations) vary based on config.

### Registry Pattern (in Factories)

```python
class ChunkingFactory:
    _registry: Dict[str, Type[ChunkerInterface]] = {
        "recursive": RecursiveChunker,
        "semantic":  SemanticChunker,
    }

    @classmethod
    def register_strategy(cls, name: str, chunker_class: type):
        cls._registry[name] = chunker_class
```

Adding a new strategy at runtime: `ChunkingFactory.register_strategy("sliding_window", SlidingWindowChunker)`. Zero modification to existing code.

---

## 9. Directory Structure (Annotated)

```
genai_multi_domain_platform/
│
├── app.py                          ← User Chat Screen ✅ — domain selector + chatbot (port 7860)
├── app_admin.py                    ← Admin Console ✅ — 4-tab management UI (port 7861)
├── ARCHITECTURE.md                 ← Architecture reference (Phase 2.2)
├── PROJECT_GUIDE.md                ← This file — full project guide
├── requirements.txt                ← Python dependencies
├── .env                            ← Environment variables (git-ignored)
│
├── cli/                            ← Command-line interface tools ✅
│   ├── __init__.py
│   ├── ingest.py                   ← Batch-ingest files from directory
│   ├── query.py                    ← Query from terminal
│   ├── manage.py                   ← list-docs / deprecate / delete / info
│   └── evaluate.py                 ← Golden QA evaluation (Recall@K)
│
├── core/                           ← All business logic (no UI here)
│   ├── config_manager.py           ← YAML loader + Pydantic models (SINGLE SOURCE OF TRUTH) ✅
│   │                                  Defines: DomainConfig, LLMConfig, EmbeddingConfig, etc.
│   ├── playground_config_manager.py← Playground-specific config CRUD ✅
│   │
│   ├── interfaces/                 ← Abstract contracts for every component
│   │   ├── chunking_interface.py   ✅
│   │   ├── embedding_interface.py  ✅
│   │   ├── vectorstore_interface.py✅
│   │   └── retrieval_interface.py  ✅
│   │
│   ├── factories/                  ← Component creation (Factory Pattern)
│   │   ├── chunking_factory.py     ✅
│   │   ├── embedding_factory.py    ✅
│   │   ├── vectorstore_factory.py  ✅
│   │   └── retrieval_factory.py    ✅ (no sys.path manipulation)
│   │
│   ├── chunking/                   ← Chunking strategy implementations
│   │   ├── recursive_chunker.py    ✅ Fixed-size sliding window with overlap
│   │   └── semantic_chunker.py     ✅ Similarity-based topical grouping
│   │
│   ├── embeddings/                 ← Embedding provider implementations
│   │   ├── sentence_transformer_embeddings.py  ✅ Local (free, 384-dim)
│   │   └── gemini_embeddings.py    ✅ Google API (768-dim)
│   │
│   ├── vectorstores/               ← Vector database adapters
│   │   ├── chromadb_store.py       ✅ Local persistent, HNSW index
│   │   └── pinecone_store.py       ✅ Cloud production
│   │
│   ├── retrievals/                 ← Retrieval strategy implementations
│   │   ├── vector_similarity_retrieval.py  ✅ Dense only
│   │   ├── bm25_retrieval.py               ✅ Sparse keyword
│   │   └── hybrid_retrieval.py             ✅ Alpha-weighted fusion
│   │
│   ├── pipeline/
│   │   └── document_pipeline.py    ✅ Orchestration (chunk→embed→store→query)
│   │
│   ├── services/
│   │   ├── document_service.py     ✅ 9-method API: upload, query, query_with_answer, ...
│   │   └── domain_service.py       ✅ NEW: domain lifecycle — create, list, inspect
│   │
│   ├── registry/
│   │   └── component_registry.py   ✅ Global component registry
│   │
│   └── utils/
│       ├── hashing.py              ✅ SHA-256 file/string hashing
│       ├── validation.py           ✅ File type, size, metadata validation
│       └── file_parsers/
│           ├── parser_factory.py   ✅ Routes to correct parser by extension
│           ├── pdf_processor.py    ✅ PyMuPDF / pdfplumber
│           ├── docx_processor.py   ✅ python-docx
│           └── txt_processor.py    ✅ UTF-8 text
│
├── models/
│   ├── metadata_models.py          ✅ ChunkMetadata (30+ fields), Pydantic enums
│   └── domain_config.py            ✅ Thin re-export wrapper only — imports from core/config_manager.py
│                                      (NOT a duplicate — single source of truth enforced)
│
├── tests/
│   ├── unit/
│   │   ├── test_chunking.py        ✅ RecursiveChunker, ChunkingFactory tests
│   │   ├── test_document_service.py✅ Full service layer (all deps mocked)
│   │   └── test_retrieval.py       ✅ VectorSimilarity, BM25, Hybrid, Factory
│   ├── integration/
│   │   └── test_end_to_end.py      ✅ Full ingest→query→deprecate flow
│   ├── golden_qa/
│   │   └── hr_qa.json              ✅ 5 sample golden QA questions
│   └── test_chroma_db.py           ✅ Existing ChromaDB tests
│
├── configs/
│   ├── global_config.yaml          ✅ Base defaults (corrected keys — embeddings, llm section)
│   ├── domains/
│   │   ├── hr.yaml                 ✅ HR domain — 600 chunks, gemini-1.5-flash, alpha=0.7
│   │   ├── finance.yaml            ✅ Finance domain — 400 chunks, alpha=0.6 (keyword-heavy)
│   │   └── legal.yaml              ✅ Legal domain — 800 chunks, gemini-1.5-pro, all-mpnet
│   ├── templates/
│   │   └── test_template_hr_v1.yaml✅ Starting template for new domains
│   └── playground/                 ✅ Playground-specific saved configs
│
├── data/
│   ├── chromadb/                   ← ChromaDB persistence (git-ignored)
│   │   ├── hr/                     ← hr domain collection
│   │   ├── finance/                ← finance domain collection
│   │   └── legal/                  ← legal domain collection
│   └── samples/                    ← Sample documents for testing
│
├── notebooks/                      ← Jupyter exploration notebooks
│   ├── documents_testing.ipynb
│   └── playground_config_test.ipynb
│
├── docs/                           ← Reference documents
│
└── archive/                        ← Legacy / superseded files (not active)
    ├── legacy_code/                ← Old UI files before rename/refactor
    └── configs/                    ← Temp / experimental config drafts
```

---

## 10. CLI Tools Reference

### `cli/ingest.py` — Batch Ingestion

Processes every PDF/DOCX/TXT in a directory and ingests them into a domain.

```bash
# Basic: ingest all files from a directory
python -m cli.ingest --domain hr --dir ./docs/hr_policies/ --uploader admin

# Replace existing documents (re-index if file changed)
python -m cli.ingest --domain hr --dir ./docs/ --uploader admin --replace

# Only ingest PDFs
python -m cli.ingest --domain finance --dir ./docs/ --extensions pdf

# With verbose logging
python -m cli.ingest --domain hr --dir ./docs/ --uploader admin -v
```

**What it does internally:**
1. Discovers all matching files in directory
2. Opens each as binary (`"rb"`) — the `f.name` attribute is set so the service can detect the type
3. Calls `service.upload_document()` for each
4. Prints success/failure per file, exits 1 if any failures

---

### `cli/query.py` — Terminal Querying

```bash
# Basic query
python -m cli.query --domain hr --query "What is the vacation policy?"

# Specific strategy
python -m cli.query --domain hr --query "Form HR-22" --strategy bm25

# Filter by doc type
python -m cli.query --domain hr --query "leave policy" --filter doc_type=policy

# Show full metadata per result
python -m cli.query --domain hr --query "benefits" --show-metadata

# JSON output (for piping to other tools)
python -m cli.query --domain hr --query "vacation" --json | jq '.[0].metadata'

# Include deprecated docs (excluded by default)
python -m cli.query --domain hr --query "old policy" --include-deprecated
```

---

### `cli/manage.py` — Document Lifecycle Management

```bash
# List all documents in a domain
python -m cli.manage list-docs --domain hr

# List with deprecated docs included
python -m cli.manage list-docs --domain hr --include-deprecated

# Get stats for a specific document
python -m cli.manage info --domain hr --doc-id handbook_2025

# List chunks for a document (paginated)
python -m cli.manage list-chunks --domain hr --doc-id handbook_2025 --limit 10

# Deprecate a document (chunks remain but are excluded from queries)
python -m cli.manage deprecate --domain hr --doc-id handbook_2023 \
  --reason "Superseded by handbook_2025"

# Deprecate and point to replacement
python -m cli.manage deprecate --domain hr --doc-id handbook_2023 \
  --reason "Superseded" --superseded-by handbook_2025

# Permanently delete (requires --confirm flag for safety)
python -m cli.manage delete --domain hr --doc-id old_doc --confirm
```

---

### `cli/evaluate.py` — Quality Evaluation

Measures retrieval quality against a golden QA set.

```bash
# Basic evaluation
python -m cli.evaluate --domain hr --golden-file tests/golden_qa/hr_qa.json

# With specific strategy
python -m cli.evaluate --domain hr --golden-file tests/golden_qa/hr_qa.json \
  --strategy hybrid --top-k 5

# Save detailed results to file
python -m cli.evaluate --domain hr --golden-file tests/golden_qa/hr_qa.json \
  --output results.json

# Per-question breakdown
python -m cli.evaluate --domain hr --golden-file tests/golden_qa/hr_qa.json -v
```

**Output:**
```
Evaluating 5 questions against domain 'hr' (strategy=hybrid, top_k=5)

✅ [hr_q1]  recall=Y  keyword=Y  Q: How many vacation days do employees get...
✅ [hr_q2]  recall=Y  keyword=Y  Q: What is the process for requesting leave...
❌ [hr_q3]  recall=N  keyword=Y  Q: How does sick leave work?
✅ [hr_q4]  recall=Y  keyword=Y  Q: What is the performance review cycle?
✅ [hr_q5]  recall=Y  keyword=Y  Q: What are the working hours?

══════════════════════════════════════════════════
  Evaluation Summary — domain=hr
══════════════════════════════════════════════════
  Questions evaluated : 5
  Strategy            : hybrid
  Top-K               : 5
  Recall@5            : 80.0%  (4/5)
  Keyword Hit Rate    : 100.0% (5/5)
══════════════════════════════════════════════════
```

**Golden QA file format (`tests/golden_qa/hr_qa.json`):**
```json
[
  {
    "id": "hr_q1",
    "question": "How many vacation days do employees get per year?",
    "expected_doc_ids": ["employee_handbook", "leave_policy"],
    "expected_keywords": ["vacation", "annual leave", "15"]
  }
]
```

**Metrics:**
- **Recall@K:** The fraction of questions where at least one `expected_doc_id` appears in the top-K results
- **Keyword Hit Rate:** The fraction of questions where at least one `expected_keyword` appears in any retrieved chunk text

---

## 11. Test Suite Structure

### Unit Tests (`tests/unit/`) — Run without any real services

All unit tests mock ChromaDB, embedding models, and file I/O. They test logic, not infrastructure.

#### `test_chunking.py`

```
TestRecursiveChunker:
  test_produces_chunks                 — non-empty text → at least 1 chunk
  test_chunk_text_not_empty            — no blank chunks
  test_chunk_has_required_metadata     — doc_id, domain, chunk_id present
  test_empty_text_returns_no_chunks    — empty string → []
  test_short_text_produces_one_chunk   — very short text → single chunk
  test_chunk_size_respected            — chunk length ≤ chunk_size + 50 slack

TestChunkingFactory:
  test_creates_recursive_chunker       — factory returns chunker with chunk_text()
  test_unknown_strategy_raises         — invalid strategy → exception
```

#### `test_document_service.py`

```
TestValidation:
  test_file_type_validation_rejects_exe       — .exe blocked
  test_metadata_missing_doc_id_raises         — missing field → ValidationError
  test_metadata_all_required_passes           — all fields → no exception
  test_file_type_allowed_extensions           — pdf/docx/txt pass
  test_file_type_disallowed_raises            — .sh → ValidationError

TestQuery:
  test_query_returns_results                  — returns list
  test_query_adds_deprecated_filter           — deprecated=False auto-added
  test_query_pipeline_exception_raises        — pipeline error → ProcessingError

TestDeprecateDocument:
  test_deprecate_calls_pipeline               — correct args forwarded
  test_deprecate_missing_doc_id_raises        — ValidationError
  test_deprecate_missing_reason_raises        — ValidationError
  test_deprecate_not_found_raises             — ValueError → DocumentNotFoundError

TestListMethods:
  test_list_documents_returns_list
  test_list_documents_injects_domain_filter   — domain always set
  test_list_chunks_returns_list
  test_list_chunks_empty_doc_id_raises        — ValidationError
  test_list_chunks_not_found_raises           — DocumentNotFoundError

TestDeleteDocument:
  test_delete_calls_pipeline
  test_delete_pipeline_failure_raises         — ProcessingError
```

#### `test_retrieval.py`

```
TestVectorSimilarityRetrieval:
  test_retrieve_returns_list
  test_retrieve_calls_vectorstore_search
  test_retrieve_calls_embed
  test_empty_results_returns_empty_list

TestBM25Retrieval:
  test_retrieve_returns_list
  test_retrieve_top_k_respected
  test_relevant_chunk_ranked_high
  test_empty_query_does_not_crash

TestHybridRetrieval:
  test_retrieve_returns_list
  test_scores_are_normalized          — all scores in [0, 1]
  test_results_sorted_descending      — highest score first

TestRetrievalFactory:
  test_creates_vector_similarity
  test_unknown_strategy_raises
```

### Integration Tests (`tests/integration/test_end_to_end.py`)

These require real ChromaDB + embedding model. Marked with `@pytest.mark.integration`.

```
TestIngestionAndQuery:
  test_ingest_text_document           — upload returns success + chunks_ingested >= 1
  test_list_documents_contains_ingested
  test_query_returns_results
  test_query_result_contains_ingested_doc
  test_get_document_info              — chunk_count, deprecated=False
  test_list_chunks
  test_deprecate_document             — chunks_deprecated >= 1
  test_deprecated_doc_excluded_from_query   — not in results when deprecated=False
  test_deprecated_doc_included_when_requested — appears when include_deprecated=True
```

### Running Tests

```bash
# Unit tests only (no external dependencies needed)
pytest tests/unit/ -v

# Integration tests (requires running ChromaDB, embedding model)
pytest tests/integration/ -v

# Skip integration tests
pytest tests/ -v -m "not integration"

# With coverage report
pytest tests/unit/ --cov=core --cov-report=term-missing

# Evaluate retrieval quality (requires indexed documents)
python -m cli.evaluate --domain hr --golden-file tests/golden_qa/hr_qa.json
```

---

## 12. Component Status Matrix

### Core Infrastructure

| Component | File | Status | Notes |
|---|---|---|---|
| Config Manager | `core/config_manager.py` | ✅ Complete | Single source of truth for all Pydantic models |
| LLMConfig | `core/config_manager.py` | ✅ Complete — NEW | Per-domain LLM: provider, model, temperature, max_tokens |
| Playground Config | `core/playground_config_manager.py` | ✅ Complete | |
| Component Registry | `core/registry/component_registry.py` | ✅ Complete | |
| Metadata Model | `models/metadata_models.py` | ✅ Complete | 30+ fields |
| Domain Config Re-exports | `models/domain_config.py` | ✅ Refactored | Thin re-export wrapper, no duplicated models |

### Interfaces

| Interface | File | Status |
|---|---|---|
| ChunkerInterface | `core/interfaces/chunking_interface.py` | ✅ |
| EmbeddingInterface | `core/interfaces/embedding_interface.py` | ✅ |
| VectorStoreInterface | `core/interfaces/vectorstore_interface.py` | ✅ |
| RetrievalInterface | `core/interfaces/retrieval_interface.py` | ✅ |

### Factories

| Factory | File | Status |
|---|---|---|
| ChunkingFactory | `core/factories/chunking_factory.py` | ✅ |
| EmbeddingFactory | `core/factories/embedding_factory.py` | ✅ |
| VectorStoreFactory | `core/factories/vectorstore_factory.py` | ✅ |
| RetrievalFactory | `core/factories/retrieval_factory.py` | ✅ |

### Implementations

| Component | File | Status | Notes |
|---|---|---|---|
| RecursiveChunker | `core/chunking/recursive_chunker.py` | ✅ | Fixed-size, configurable overlap |
| SemanticChunker | `core/chunking/semantic_chunker.py` | ✅ | Embedding-based grouping |
| SentenceTransformerEmbeddings | `core/embeddings/sentence_transformer_embeddings.py` | ✅ | Local, 384-dim |
| GeminiEmbeddings | `core/embeddings/gemini_embeddings.py` | ✅ | Cloud API, 768-dim |
| ChromaDBStore | `core/vectorstores/chromadb_store.py` | ✅ | Local, HNSW |
| PineconeStore | `core/vectorstores/pinecone_store.py` | ✅ | Cloud |
| VectorSimilarityRetrieval | `core/retrievals/vector_similarity_retrieval.py` | ✅ | Dense-only |
| BM25Retrieval | `core/retrievals/bm25_retrieval.py` | ✅ | Sparse keyword |
| HybridRetrieval | `core/retrievals/hybrid_retrieval.py` | ✅ | Alpha-weighted fusion |

### Pipeline & Service

| Component | File | Status | Notes |
|---|---|---|---|
| DocumentPipeline | `core/pipeline/document_pipeline.py` | ✅ | All methods implemented |
| DocumentService | `core/services/document_service.py` | ✅ | 9-method API + LLM answer generation |
| DomainService | `core/services/domain_service.py` | ✅ NEW | Domain lifecycle: create, list, inspect |

### LLM Providers (answer generation in `query_with_answer()`)

| Provider | Status | Notes |
|---|---|---|
| Gemini (`gemini-1.5-flash`, `gemini-1.5-pro`) | ✅ Implemented | Default provider, uses `GEMINI_API_KEY` |
| OpenAI (`gpt-4o`, `gpt-3.5-turbo`, etc.) | ✅ Implemented | Uses `OPENAI_API_KEY` |
| Provider selection | ✅ Config-driven | Set `llm.provider` in domain YAML — no code changes |

### CLI Tools

| Tool | File | Status |
|---|---|---|
| Batch Ingestion | `cli/ingest.py` | ✅ |
| Query CLI | `cli/query.py` | ✅ |
| Management CLI | `cli/manage.py` | ✅ |
| Evaluation CLI | `cli/evaluate.py` | ✅ |

### Tests

| Test Category | File | Status |
|---|---|---|
| Unit — chunking | `tests/unit/test_chunking.py` | ✅ |
| Unit — service | `tests/unit/test_document_service.py` | ✅ |
| Unit — retrieval | `tests/unit/test_retrieval.py` | ✅ |
| Integration | `tests/integration/test_end_to_end.py` | ✅ |
| Golden QA | `tests/golden_qa/hr_qa.json` | ✅ |
| ChromaDB test | `tests/test_chroma_db.py` | ✅ existing |

---

## 13. Key Coding Decisions Explained

### Why Pydantic for Configs?

YAML is just text. Without Pydantic, a typo like `chunk_size: "five hundred"` would silently pass — the crash would only happen deep inside `RecursiveChunker` with an unhelpful `TypeError`. Pydantic catches it at load time with a clear message: `chunk_size must be an integer`.

It also gives IDE autocomplete: typing `config.chunking.` shows all available fields.

### Why Upsert Instead of Insert?

ChromaDB's `upsert` updates an existing record if the ID exists, or inserts it if not. Using `upsert` makes re-ingestion idempotent: if you ingest a document twice with `replace_existing=False`, the second run safely updates the same chunks rather than creating duplicates.

### Why SHA-256 File Hashing?

Two documents with different names may be identical. A 256-bit hash is a compact fingerprint. Comparing hashes before indexing avoids re-embedding identical content (expensive), and provides a provenance trail: you can always verify the stored chunks came from the exact file version that produced a given hash.

### Why Normalize Embeddings to L2 norm = 1?

When vectors are L2-normalized, cosine similarity = dot product:
```
cos(A, B) = A·B / (||A|| * ||B||)
If ||A|| = ||B|| = 1 → cos(A, B) = A·B
```

Dot product is much faster to compute, and ChromaDB's HNSW index is optimized for it. Normalization happens once at embedding time; all subsequent searches are faster.

### Why Min-Max Normalization in Hybrid Retrieval?

BM25 scores can be anything from 0 to 30+ depending on document length and term frequency. Cosine similarity is already in [0, 1]. Combining them directly with alpha=0.7 would give BM25 nearly zero influence. Min-max normalization puts both in [0, 1] so the alpha weight is actually meaningful.

### Why Deprecated Chunks Are Updated, Not Deleted?

Deleting deprecated chunks would lose the audit trail. A compliance team might need to know what was in the old handbook on a specific date. ChromaDB's metadata update (`collection.update()`) marks chunks as `deprecated=True` while preserving all original content and provenance data. The `deprecated=False` filter simply excludes them from normal queries.

### Why Temp File in Upload?

PDF/DOCX parsers (PyMuPDF, python-docx) require a file path string, not an in-memory file object. Gradio passes a file object. The service saves to a temp file, runs the parser, then deletes it in a `finally` block — ensuring cleanup even if parsing fails.

### Why `collection.update()` for Deprecation (Not Delete + Re-Insert)?

Re-inserting with updated metadata would:
1. Require re-embedding (expensive)
2. Risk creating new chunk IDs, losing continuity

ChromaDB's `update()` call mutates only the metadata dict, leaving vectors unchanged. It takes milliseconds and preserves all existing data.

---

## 14. Extending the Platform

### Adding a New Embedding Provider (e.g. OpenAI)

1. Create `core/embeddings/openai_embeddings.py`:
```python
class OpenAIEmbeddings(EmbeddingInterface):
    def embed_texts(self, texts): ...
    def get_model_name(self): return "text-embedding-3-small"
    def get_embedding_dimension(self): return 1536
```

2. Register in `EmbeddingFactory`:
```python
elif provider == "openai":
    return OpenAIEmbeddings(api_key=config.api_key)
```

3. Add to domain config:
```yaml
embeddings:
  provider: openai
  api_key: "${OPENAI_API_KEY}"
  model_name: text-embedding-3-small
```

That's it. Zero changes to pipeline, service, or UI.

---

### Adding a New Domain (e.g. Legal)

1. Create `configs/domains/legal.yaml`:
```yaml
domain_id: legal
description: Legal Document Knowledge Base

chunking:
  strategy: recursive
  chunk_size: 800     # Legal docs have longer relevant passages
  overlap: 100

vectorstore:
  collection_name: legal_docs
  persist_directory: ./data/chroma_db/legal
```

2. Use it:
```python
service = DocumentService("legal")
```

ChromaDB will automatically create a new `legal_docs` collection, isolated from `hr_docs`.

---

### Adding a New Retrieval Strategy (e.g. MMR — Maximal Marginal Relevance)

MMR re-ranks results to reduce redundancy while maintaining relevance.

1. Create `core/retrievals/mmr_retrieval.py` implementing `RetrievalInterface`
2. Add to `RetrievalFactory`: `"mmr": MMRRetrieval`
3. In domain config: `strategies: [hybrid, mmr]`

---

## 15. Implemented in Phase 2.2

The following were previously planned and are now complete:

| Feature | Status | Notes |
|---|---|---|
| `LLMConfig` in domain schema | ✅ Done | Each domain has its own LLM provider/model/temperature |
| `DomainService` | ✅ Done | Full domain lifecycle: create, list, vector count |
| Immediate ChromaDB init on domain creation | ✅ Done | `create_domain()` creates the collection instantly with rollback on failure |
| OpenAI LLM provider | ✅ Done | `provider: openai` in domain YAML selects OpenAI for answer generation |
| Config-driven LLM selection | ✅ Done | Priority chain: explicit params → config dict → domain YAML → hardcoded fallback |
| User Chat Screen (`app.py`) | ✅ Done | Clean single-screen chatbot with domain selector and source citations |
| Admin Console (`app_admin.py`) | ✅ Done | 4-tab interface: Templates, Domains, Documents, Playground |
| Finance and Legal domain configs | ✅ Done | `configs/domains/finance.yaml`, `configs/domains/legal.yaml` |
| `models/domain_config.py` single source of truth | ✅ Done | Now a re-export wrapper — no duplicated Pydantic models |
| Fix duplicate `register_strategy()` in ChunkingFactory | ✅ Done | Removed duplicate method definition |
| Remove `sys.path.insert()` from RetrievalFactory | ✅ Done | Proper package imports used instead |
| Fix `global_config.yaml` schema mismatches | ✅ Done | `embeddings:` key (not `embedding:`), correct nested chunking structure |

---

## 16. Pending / Future Work

### P2 — Nice-to-Have Enhancements

| Feature | Description | Complexity |
|---|---|---|
| `update_document_metadata()` | Update metadata on existing chunks without re-ingestion | Medium |
| `rollback_document()` | Restore a deprecated version | Medium |
| LLM Reranker | Post-retrieval LLM-based result reranking | High |
| Async ingestion | Background task queue for large batches | High |
| FastAPI server | REST API wrapper around DocumentService | Medium |
| Auth/RBAC | Role-based access per domain | High |
| Rate limiting | Request throttling per user/domain | Low |
| Prometheus metrics | Latency, throughput, error rate per operation | Medium |
| Hot config reload | Reload YAML configs without restart | Medium |
| Multi-tenancy | Tenant-level storage isolation | High |
| OpenAI Embeddings | `provider: openai` for embedding generation (currently: Gemini or SentenceTransformers) | Low |
| `CHANGELOG.md` | Versioned release notes | Low |

---

*Document updated for Phase 2.2 codebase — February 2026*
*All components listed as ✅ have been implemented and verified against the architecture spec.*

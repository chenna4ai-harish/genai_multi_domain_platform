# Architecture — GenAI Multi-Domain RAG Platform

**Last Updated:** February 2026
**Stack:** Python · ChromaDB · SentenceTransformers / Gemini · Gradio · Pydantic · BM25

---

## Overview

A **config-driven, multi-domain RAG (Retrieval-Augmented Generation) platform**. Each domain (HR, Finance, Legal, etc.) has its own isolated vector store, chunking strategy, embedding model, and retrieval configuration — all controlled entirely by YAML files with **no code changes required**.

---

## Design Goals

| Goal | How it is achieved |
|---|---|
| Config-driven | All behaviour controlled by `configs/domains/<name>.yaml` |
| Multi-domain isolation | Each domain gets its own ChromaDB collection |
| Swappable components | Factory pattern — change embedding/retrieval via config only |
| Zero business logic in UI | Service layer enforces all rules; UI only calls service methods |
| Testable without UI | Service and pipeline layers are UI-framework independent |

---

## Layer Diagram

```
┌─────────────────────────────────────────────────────┐
│  UI Layer                                           │
│  app.py          — User chat (domain select + Q&A)  │
│  app_admin.py    — Admin console (4 tabs)           │
│  cli/            — Command-line tools               │
└──────────────────┬──────────────────────────────────┘
                   │ calls only service methods
┌──────────────────▼──────────────────────────────────┐
│  Service Layer                                      │
│  DocumentService  — upload / query / list / delete  │
│  DomainService    — create domain / templates       │
└──────────────────┬──────────────────────────────────┘
                   │ delegates to
┌──────────────────▼──────────────────────────────────┐
│  Pipeline Layer                                     │
│  DocumentPipeline — chunk → embed → store           │
│                     query → retrieve → rank         │
└──────┬───────────┬───────────────┬──────────────────┘
       │           │               │
  ChunkingFactory  EmbeddingFactory  RetrievalFactory
       │           │               │
  Recursive    SentenceTransformer  Hybrid
  Semantic     Gemini               VectorSimilarity
                                    BM25
                    VectorStoreFactory
                    ChromaDBStore / PineconeStore

All components driven by:
┌─────────────────────────────────────────────────────┐
│  Config Layer                                       │
│  ConfigManager  — loads + validates YAML configs    │
│  DomainConfig   — Pydantic schema (single source)   │
│  global_config.yaml  +  domain.yaml  =  merged cfg  │
└─────────────────────────────────────────────────────┘
```

---

## File Structure

```
genai_multi_domain_platform/
│
├── app.py                        # User chat interface (domain select + Q&A)
├── app_admin.py                  # Admin console (4-tab: templates/domains/docs/playground)
│
├── configs/
│   ├── global_config.yaml        # Global defaults (all domains inherit)
│   ├── domains/                  # One YAML per domain
│   │   ├── hr.yaml
│   │   ├── finance.yaml
│   │   └── legal.yaml
│   ├── templates/                # Reusable config templates for domain creation
│   │   └── test_template_hr_v1.yaml
│   └── playground/               # Temporary experiment configs (auto-cleaned)
│
├── core/
│   ├── config_manager.py         # ★ Pydantic schemas + ConfigManager (SINGLE SOURCE OF TRUTH)
│   ├── playground_config_manager.py
│   │
│   ├── services/
│   │   ├── document_service.py   # upload / query / list / delete / deprecate
│   │   └── domain_service.py     # create domain + init vector store immediately
│   │
│   ├── pipeline/
│   │   └── document_pipeline.py  # orchestrates chunk→embed→store and retrieval
│   │
│   ├── factories/
│   │   ├── chunking_factory.py
│   │   ├── embedding_factory.py
│   │   ├── retrieval_factory.py
│   │   └── vectorstore_factory.py
│   │
│   ├── interfaces/               # Abstract base classes
│   │   ├── chunking_interface.py
│   │   ├── embedding_interface.py
│   │   ├── retrieval_interface.py
│   │   └── vectorstore_interface.py
│   │
│   ├── chunking/
│   │   ├── recursive_chunker.py  # Fixed-size with overlap
│   │   └── semantic_chunker.py   # Sentence-similarity grouping
│   │
│   ├── embeddings/
│   │   ├── sentence_transformer_embeddings.py  # Local, free
│   │   └── gemini_embeddings.py                # Google Cloud API
│   │
│   ├── retrievals/
│   │   ├── vector_similarity_retrieval.py
│   │   ├── bm25_retrieval.py
│   │   └── hybrid_retrieval.py   # Recommended default
│   │
│   ├── vectorstores/
│   │   ├── chromadb_store.py     # Local (MVP / dev)
│   │   └── pinecone_store.py     # Cloud (production)
│   │
│   ├── registry/
│   │   └── component_registry.py # Available components for UI dropdowns
│   │
│   └── utils/
│       ├── hashing.py
│       ├── validation.py
│       └── file_parsers/         # PDF / DOCX / TXT extraction
│
├── models/
│   ├── domain_config.py          # Re-exports from core/config_manager.py only
│   └── metadata_models.py        # ChunkMetadata Pydantic model
│
├── cli/
│   ├── query.py     # python -m cli.query --domain hr --query "..."
│   ├── ingest.py    # python -m cli.ingest --domain hr --dir ./docs/
│   ├── evaluate.py
│   └── manage.py
│
├── data/
│   └── chromadb/    # One subdir per domain (created automatically)
│
└── tests/
    ├── unit/
    └── integration/
```

---

## Configuration-Driven Design

### Merge order

```
configs/global_config.yaml   (baseline defaults)
        +
configs/domains/hr.yaml      (domain overrides)
        =
DomainConfig (Pydantic)      (validated, type-safe)
        ↓
DocumentService("hr")        (initialized with this config)
```

### YAML structure (must match `DomainConfig` schema exactly)

```yaml
domain_id: hr
name: Human Resources
description: "HR policies, leave rules, benefits"

chunking:
  strategy: recursive        # recursive | semantic
  recursive:
    chunk_size: 600
    overlap: 80

embeddings:
  provider: sentence_transformers   # sentence_transformers | gemini | openai
  model_name: all-MiniLM-L6-v2
  device: cpu

retrieval:
  strategies:
    - hybrid                 # hybrid | vector_similarity | bm25
  top_k: 10
  hybrid:
    alpha: 0.7              # 1.0=pure semantic, 0.0=pure keyword

vectorstore:
  provider: chromadb
  collection_name: hr       # convention: always = domain_id
  persist_directory: ./data/chromadb/hr

security:
  allowed_file_types: [pdf, docx, txt]
  max_file_size_mb: 20
```

### Adding a new domain — zero code changes

```bash
# Option A: use the Admin UI (Tab 2 — Domain Management)

# Option B: copy a template and edit
cp configs/templates/test_template_hr_v1.yaml configs/domains/engineering.yaml
# edit domain_id, name, collection_name, persist_directory
python -c "from core.services.domain_service import DomainService; DomainService().create_domain('engineering','Engineering','test_template_hr_v1')"
```

---

## Domain Lifecycle

```
1. Admin creates domain (DomainService.create_domain)
   ├── Validates domain_id format
   ├── Loads template YAML
   ├── Sets collection_name = domain_id
   ├── Saves configs/domains/<domain_id>.yaml
   └── Calls DocumentService(domain_id)
         └── DocumentPipeline → ChromaDBStore.get_or_create_collection()
               ← COLLECTION CREATED IMMEDIATELY

2. Admin uploads documents (DocumentService.upload_document)
   ├── Validates file type / size
   ├── Extracts text (PDF/DOCX/TXT)
   ├── Computes SHA-256 hash
   └── DocumentPipeline.process_document()
         ├── Chunk
         ├── Embed
         └── Store with metadata in ChromaDB

3. User asks question (DocumentService.query_with_answer)
   ├── DocumentPipeline.query()
   │     ├── Embed query
   │     ├── HybridRetrieval → top-K chunks
   │     └── Return ranked results
   └── LLM generates grounded answer from context
```

---

## Retrieval Strategies

| Strategy | Description | When to use |
|---|---|---|
| `hybrid` | Dense + Sparse (BM25), weighted by alpha | **Default — most use cases** |
| `vector_similarity` | Embedding cosine similarity only | Conceptual / semantic queries |
| `bm25` | Keyword frequency (Okapi BM25) | Exact terms, IDs, codes |

**Hybrid alpha guide:**

| Alpha | Meaning |
|---|---|
| `1.0` | Pure semantic (ignore keywords) |
| `0.7` | 70% semantic + 30% keyword (recommended) |
| `0.5` | Equal weight |
| `0.0` | Pure keyword |

---

## Service Layer Contract

The UI must only call service layer methods. It must never import from `core.pipeline`, `core.factories`, or `core.vectorstores`.

```python
# Correct — UI calls service only
from core.services.document_service import DocumentService
from core.services.domain_service import DomainService

svc = DocumentService("hr")
svc.upload_document(file_obj, metadata)
svc.query_with_answer("What is the leave policy?")

domain_svc = DomainService()
domain_svc.create_domain("hr", "Human Resources", "test_template_hr_v1")

# Wrong — UI must never do this
from core.pipeline.document_pipeline import DocumentPipeline
from core.factories.vectorstore_factory import VectorStoreFactory
```

---

## Pydantic Schema — Single Source of Truth

All configuration models are defined in `core/config_manager.py`.

`models/domain_config.py` is a thin re-export module only. It does not define any models. This ensures zero duplication.

```python
# Both resolve to the same class
from core.config_manager import DomainConfig
from models.domain_config import DomainConfig   # re-exports core/config_manager.py
```

---

## Supported Components

### Embedding Providers

| Provider | Model | Dimensions | Notes |
|---|---|---|---|
| `sentence_transformers` | `all-MiniLM-L6-v2` | 384 | Free, local, recommended default |
| `sentence_transformers` | `all-mpnet-base-v2` | 768 | Free, local, higher quality |
| `gemini` | `models/embedding-001` | 768 | Requires `GEMINI_API_KEY` |

### Vector Stores

| Provider | Type | Best for |
|---|---|---|
| `chromadb` | Local persistent | Development, MVP, < 1M vectors |
| `pinecone` | Cloud managed | Production, > 1M vectors |

### Chunking Strategies

| Strategy | Description | Key config |
|---|---|---|
| `recursive` | Fixed-size with overlap | `chunk_size`, `overlap` |
| `semantic` | Sentence similarity grouping | `similarity_threshold`, `max_chunk_size` |

---

## Environment Variables

```bash
GEMINI_API_KEY=...       # Required only if using gemini embedding/LLM provider
OPENAI_API_KEY=...       # Required only if using openai provider
```

Never commit API keys. Use a `.env` file (already in `.gitignore`).

---

## Known Limitations (Current State)

| Item | Status |
|---|---|
| OpenAI embedding provider | Config supported, implementation pending |
| Pinecone store | Interface defined, not fully tested |
| Qdrant / FAISS | Config allows, factory not implemented |
| Admin UI access control | Password protection not yet implemented |
| LLM reranking | Config present but not wired into pipeline |

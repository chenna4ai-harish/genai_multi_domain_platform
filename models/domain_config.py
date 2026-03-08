"""
models/domain_config.py

Single source of truth for all configuration Pydantic models lives in
core/config_manager.py.  This module re-exports every public symbol from
there so that any existing import of the form

    from models.domain_config import DomainConfig

continues to work without changes.

Do NOT define models here — edit core/config_manager.py instead.
"""

from core.config_manager import (  # noqa: F401  (re-exports)
    RecursiveChunkingConfig,
    SemanticChunkingConfig,
    ChunkingConfig,
    EmbeddingConfig,
    HybridRetrievalConfig,
    RetrievalConfig,
    VectorStoreConfig,
    SecurityConfig,
    MetadataConfig,
    LLMConfig,
    DomainConfig,
    ConfigManager,
)

__all__ = [
    "RecursiveChunkingConfig",
    "SemanticChunkingConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "HybridRetrievalConfig",
    "RetrievalConfig",
    "VectorStoreConfig",
    "SecurityConfig",
    "MetadataConfig",
    "LLMConfig",
    "DomainConfig",
    "ConfigManager",
]

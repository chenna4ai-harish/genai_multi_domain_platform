"""

core/config_manager.py

This module implements the Configuration Manager for Phase 2 architecture.

What is the Configuration Manager?
-----------------------------------
The ConfigManager is the central configuration loader and validator for the
Multi-Domain RAG System. It handles:
- Loading domain-specific configurations from YAML files
- Merging global defaults with domain overrides
- Validating configurations against Pydantic schemas
- Injecting environment variables (e.g., API keys)
- Supporting configuration templates

Phase 2 Configuration Architecture:
------------------------------------
configs/
  ├── global_config.yaml         # Global defaults
  ├── domains/                   # Domain-specific configs
  │   ├── hr.yaml
  │   ├── finance.yaml
  │   └── engineering.yaml
  └── templates/                 # Reusable templates
      ├── dev_template.yaml
      └── prod_template.yaml

Why Configuration-Driven Design?
---------------------------------
1. **No Hardcoded Business Rules**: All behavior controlled by YAML config
2. **Domain Customization**: Each domain (HR, Finance, etc.) has unique settings
3. **Easy A/B Testing**: Change strategies via config, no code deployment
4. **Environment Management**: Dev/staging/prod configurations separate
5. **Hot Reloading**: Change config without restarting (future enhancement)

Key Features:
-------------
- **Pydantic Validation**: Type-safe configuration with automatic validation
- **Environment Variables**: API keys from ${GEMINI_API_KEY} syntax
- **Deep Merging**: Domain configs override global defaults recursively
- **Template Support**: Reusable configuration templates

Example Usage:
--------------
# Initialize manager
config_mgr = ConfigManager()

# Load domain configuration
hr_config = config_mgr.load_domain_config("hr_domain")

# Access configuration
print(hr_config.chunking.strategy)  # "recursive"
print(hr_config.embeddings.provider)  # "sentence_transformers"

# Use in service/pipeline
service = DocumentService(hr_config)

References:
-----------
- Phase 2 Configuration: Section 12 (Configuration Management)
- Pydantic Docs: https://docs.pydantic.dev/

"""

import os
import glob
import copy
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ValidationError, field_validator

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# PHASE 2 PYDANTIC CONFIGURATION SCHEMAS
# =============================================================================

class RecursiveChunkingConfig(BaseModel):
    """Configuration for recursive (fixed-size) chunking strategy."""
    strategy: str = Field(default="recursive", description="Chunking strategy name")
    chunk_size: int = Field(default=500, ge=100, le=5000, description="Characters per chunk")
    overlap: int = Field(default=50, ge=0, le=500, description="Overlap between chunks")


class SemanticChunkingConfig(BaseModel):
    """Configuration for semantic (similarity-based) chunking strategy."""
    strategy: str = Field(default="semantic", description="Chunking strategy name")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")
    max_chunk_size: int = Field(default=1000, ge=100, le=5000, description="Maximum chunk size")


class ChunkingConfig(BaseModel):
    """Phase 2 chunking configuration."""
    strategy: str = Field(..., description="Chunking strategy: recursive or semantic")
    recursive: Optional[RecursiveChunkingConfig] = Field(default_factory=RecursiveChunkingConfig)
    semantic: Optional[SemanticChunkingConfig] = Field(default_factory=SemanticChunkingConfig)

    @field_validator('strategy')
    @classmethod
    def validate_strategy(cls, v):
        allowed = ["recursive", "semantic"]
        if v not in allowed:
            raise ValueError(f"Invalid chunking strategy: {v}. Allowed: {allowed}")
        return v


class EmbeddingConfig(BaseModel):
    """Phase 2 embedding configuration."""
    provider: str = Field(..., description="Embedding provider: sentence_transformers, gemini, openai")
    model_name: str = Field(..., description="Model name/identifier")
    device: Optional[str] = Field(default="cpu", description="Device for local models: cpu, cuda, mps")
    batch_size: Optional[int] = Field(default=32, ge=1, le=1000, description="Batch size for embedding")
    normalize: Optional[bool] = Field(default=True, description="Normalize embeddings (for sentence_transformers)")
    api_key: Optional[str] = Field(default=None, description="API key for cloud providers (from env)")

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        allowed = ["sentence_transformers", "gemini", "openai"]
        if v not in allowed:
            raise ValueError(f"Invalid embedding provider: {v}. Allowed: {allowed}")
        return v


class HybridRetrievalConfig(BaseModel):
    """Configuration for hybrid (dense + sparse) retrieval."""
    alpha: float = Field(default=0.7, ge=0.0, le=1.0, description="Dense weight (1-alpha = sparse weight)")
    normalize_scores: bool = Field(default=True, description="Normalize scores before combining")


class RetrievalConfig(BaseModel):
    """Phase 2 retrieval configuration."""
    strategies: List[str] = Field(default=["hybrid"], description="Enabled retrieval strategies")
    top_k: int = Field(default=10, ge=1, le=100, description="Default number of results")
    similarity: str = Field(default="cosine", description="Similarity metric")
    hybrid: Optional[HybridRetrievalConfig] = Field(default_factory=HybridRetrievalConfig)

    @field_validator('strategies')
    @classmethod
    def validate_strategies(cls, v):
        allowed = ["vector_similarity", "hybrid", "bm25"]
        for strategy in v:
            if strategy not in allowed:
                raise ValueError(f"Invalid retrieval strategy: {strategy}. Allowed: {allowed}")
        return v


class VectorStoreConfig(BaseModel):
    """Phase 2 vector store configuration."""
    provider: str = Field(..., description="Vector store provider: chromadb, pinecone, qdrant, faiss")
    collection_name: str = Field(..., description="Collection/index name")
    index_type: Optional[str] = Field(default="hnsw", description="Index algorithm")
    persist_directory: Optional[str] = Field(default="./data/chroma_db", description="ChromaDB persistence directory")
    cloud: Optional[str] = Field(default="aws", description="Pinecone cloud provider")
    region: Optional[str] = Field(default="us-east-1", description="Pinecone region")
    api_key: Optional[str] = Field(default=None, description="API key for cloud providers")
    dimension: Optional[int] = Field(default=None, description="Embedding dimension (auto-detected)")

    @field_validator('provider')
    @classmethod
    def validate_provider(cls, v):
        allowed = ["chromadb", "pinecone", "qdrant", "faiss"]
        if v not in allowed:
            raise ValueError(f"Invalid vector store provider: {v}. Allowed: {allowed}")
        return v


class SecurityConfig(BaseModel):
    """Phase 2 security configuration."""
    allowed_file_types: List[str] = Field(default=["pdf", "docx", "txt"], description="Allowed file extensions")
    max_file_size_mb: int = Field(default=20, ge=1, le=100, description="Maximum file size in MB")
    require_authentication: bool = Field(default=False, description="Require user authentication")


class MetadataConfig(BaseModel):
    """Phase 2 metadata configuration."""
    track_versions: bool = Field(default=True, description="Track document versions")
    enable_deprecation: bool = Field(default=True, description="Enable deprecation workflow")
    compute_file_hash: bool = Field(default=True, description="Compute file hashes for integrity")
    extract_page_numbers: bool = Field(default=True, description="Extract page numbers from PDFs")
    required_fields: List[str] = Field(
        default=["doc_id", "title", "domain", "doc_type", "uploader_id"],
        description="Required metadata fields for upload"
    )


class DomainConfig(BaseModel):
    """
    Complete Phase 2 domain configuration schema.

    This is the root configuration model that all domain configs must conform to.
    Validates all required fields and sub-configurations.
    """
    # Identity
    domain_id: str = Field(..., description="Unique domain identifier")
    name: str = Field(..., description="Human-readable domain name")
    description: Optional[str] = Field(default=None, description="Domain description")

    # Component configurations (CRITICAL: Match field names used in code)
    chunking: ChunkingConfig = Field(..., description="Chunking configuration")
    embeddings: EmbeddingConfig = Field(..., description="Embedding configuration")
    retrieval: RetrievalConfig = Field(..., description="Retrieval configuration")
    vectorstore: VectorStoreConfig = Field(..., description="Vector store configuration")

    # Optional configurations
    security: Optional[SecurityConfig] = Field(default_factory=SecurityConfig)
    metadata: Optional[MetadataConfig] = Field(default_factory=MetadataConfig)

    class Config:
        # Allow extra fields for future extensibility
        extra = "allow"


# =============================================================================
# CONFIGURATION MANAGER IMPLEMENTATION
# =============================================================================

class ConfigManager:
    """
    Central configuration manager for Phase 2 architecture.

    Responsibilities:
    - Load and validate domain configurations
    - Merge global defaults with domain overrides
    - Inject environment variables
    - Support configuration templates
    - Provide configuration discovery

    Directory Structure:
    --------------------
    configs/
      ├── global_config.yaml       # Global defaults (all domains inherit)
      ├── domains/                 # Domain-specific configs
      │   ├── hr.yaml              # HR domain configuration
      │   ├── finance.yaml         # Finance domain configuration
      │   └── engineering.yaml     # Engineering domain configuration
      └── templates/               # Reusable templates
          ├── dev_template.yaml    # Development environment template
          └── prod_template.yaml   # Production environment template

    Example:
    --------
    # Initialize manager
    config_mgr = ConfigManager()

    # List available domains
    domains = config_mgr.get_all_domain_names()
    print(f"Available domains: {domains}")

    # Load domain configuration
    hr_config = config_mgr.load_domain_config("hr_domain")

    # Access configuration values
    print(f"Chunking strategy: {hr_config.chunking.strategy}")
    print(f"Embedding provider: {hr_config.embeddings.provider}")
    print(f"Vector store: {hr_config.vectorstore.provider}")
    """

    def __init__(
            self,
            config_dir: str = "configs",
            domain_dir: str = "domains",
            template_dir: str = "templates",
            global_config_file: str = "global_config.yaml"
    ):
        """
        Initialize ConfigManager.

        Parameters:
        -----------
        config_dir : str
            Base configuration directory
        domain_dir : str
            Subdirectory for domain configs
        template_dir : str
            Subdirectory for template configs
        global_config_file : str
            Global configuration filename
        """
        self.config_dir = Path(config_dir)
        self.domain_dir = self.config_dir / domain_dir
        self.template_dir = self.config_dir / template_dir
        self.global_config_file = self.config_dir / global_config_file

        # Create directories if they don't exist
        self.domain_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Load global config once (merged into all domain configs)
        self.global_config = self._load_yaml(self.global_config_file) or {}
        self._inject_env_vars(self.global_config)

        logger.info(
            f"ConfigManager initialized:\n"
            f"  Config dir: {self.config_dir.absolute()}\n"
            f"  Global config: {self.global_config_file.name}\n"
            f"  Domains dir: {self.domain_dir}\n"
            f"  Templates dir: {self.template_dir}"
        )

    @staticmethod
    def _load_yaml(path: Path) -> Optional[Dict]:
        """Load YAML file and return dict."""
        if not path.exists():
            logger.warning(f"Config file not found: {path}")
            return None

        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
                logger.debug(f"Loaded YAML from {path}")
                return data
        except Exception as e:
            logger.error(f"Failed to load YAML from {path}: {e}")
            raise

    def _inject_env_vars(self, config: Dict) -> None:
        """
        Recursively inject environment variables.

        Replaces ${ENV_VAR} syntax with actual environment variable values.
        Example: api_key: ${GEMINI_API_KEY} → api_key: "actual-key-value"
        """
        if isinstance(config, dict):
            for k, v in config.items():
                if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                    env_var = v[2:-1]  # Extract ENV_VAR from ${ENV_VAR}
                    env_value = os.getenv(env_var)
                    if env_value:
                        config[k] = env_value
                        logger.debug(f"Injected env var: {env_var}")
                    else:
                        logger.warning(f"Environment variable not set: {env_var}")
                else:
                    self._inject_env_vars(v)
        elif isinstance(config, list):
            for entry in config:
                self._inject_env_vars(entry)

    @staticmethod
    def _merge_dicts(base: Dict, override: Dict) -> None:
        """
        Deep merge override into base dictionary.

        Recursively merges nested dictionaries. Non-dict values in override
        completely replace values in base.
        """
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                ConfigManager._merge_dicts(base[k], v)
            else:
                base[k] = v

    def get_all_domain_names(self) -> List[str]:
        """
        Get list of all available domain names.

        Returns:
        --------
        List[str]:
            List of domain names (without .yaml extension)
            Example: ["hr_domain", "finance", "engineering"]
        """
        yaml_files = glob.glob(str(self.domain_dir / "*.yaml"))
        names = [Path(f).stem for f in yaml_files]
        logger.debug(f"Found {len(names)} domain configs: {names}")
        return names

    def get_all_template_names(self) -> List[str]:
        """
        Get list of all available template names.

        Returns:
        --------
        List[str]:
            List of template names (without .yaml extension)
            Example: ["dev_template", "prod_template"]
        """
        yaml_files = glob.glob(str(self.template_dir / "*.yaml"))
        names = [Path(f).stem for f in yaml_files]
        logger.debug(f"Found {len(names)} template configs: {names}")
        return names

    def load_domain_config(self, domain_name: str) -> DomainConfig:
        """
        Load and validate domain configuration.

        Workflow:
        ---------
        1. Load global config (defaults)
        2. Load domain-specific config
        3. Deep merge: global <- domain (domain overrides global)
        4. Inject environment variables (API keys, etc.)
        5. Validate against Pydantic schema
        6. Return validated DomainConfig object

        Parameters:
        -----------
        domain_name : str
            Domain name (without .yaml extension)
            Example: "hr_domain", "finance", "engineering"

        Returns:
        --------
        DomainConfig:
            Validated domain configuration object

        Raises:
        -------
        FileNotFoundError:
            If domain config file doesn't exist
        ValidationError:
            If configuration is invalid

        Example:
        --------
        config_mgr = ConfigManager()
        hr_config = config_mgr.load_domain_config("hr_domain")

        # Access configuration
        print(hr_config.chunking.strategy)  # "recursive"
        print(hr_config.embeddings.provider)  # "sentence_transformers"
        """
        logger.info(f"Loading domain config: {domain_name}")

        # Load domain config
        domain_file = self.domain_dir / f"{domain_name}.yaml"
        if not domain_file.exists():
            raise FileNotFoundError(
                f"Domain config not found: {domain_file}\n"
                f"Available domains: {self.get_all_domain_names()}"
            )

        domain_conf = self._load_yaml(domain_file) or {}

        # Merge: global <- domain (domain overrides global)
        merged = copy.deepcopy(self.global_config)
        self._merge_dicts(merged, domain_conf)

        # Inject environment variables
        self._inject_env_vars(merged)

        # Add domain_id if not present
        if 'domain_id' not in merged:
            merged['domain_id'] = domain_name

        # Validate and convert to Pydantic model
        try:
            config = DomainConfig(**merged)
            logger.info(
                f"✅ Domain config loaded and validated: {domain_name}\n"
                f"   Chunking: {config.chunking.strategy}\n"
                f"   Embeddings: {config.embeddings.provider}\n"
                f"   Vector Store: {config.vectorstore.provider}\n"
                f"   Retrieval: {', '.join(config.retrieval.strategies)}"
            )
            return config

        except ValidationError as e:
            logger.error(f"Config validation failed for domain '{domain_name}':\n{e}")
            raise

    def load_template_config(self, template_name: str) -> DomainConfig:
        """
        Load and validate template configuration.

        Templates are reusable configurations (e.g., dev, prod environments).

        Parameters:
        -----------
        template_name : str
            Template name (without .yaml extension)

        Returns:
        --------
        DomainConfig:
            Validated template configuration
        """
        logger.info(f"Loading template config: {template_name}")

        # Load template config
        template_file = self.template_dir / f"{template_name}.yaml"
        if not template_file.exists():
            raise FileNotFoundError(
                f"Template config not found: {template_file}\n"
                f"Available templates: {self.get_all_template_names()}"
            )

        template_conf = self._load_yaml(template_file) or {}

        # Merge: global <- template
        merged = copy.deepcopy(self.global_config)
        self._merge_dicts(merged, template_conf)

        # Inject environment variables
        self._inject_env_vars(merged)

        # Add domain_id if not present
        if 'domain_id' not in merged:
            merged['domain_id'] = template_name

        # Validate
        try:
            return DomainConfig(**merged)
        except ValidationError as e:
            logger.error(f"Template validation failed for '{template_name}':\n{e}")
            raise

    def save_domain_config(self, domain_name: str, config: Dict) -> None:
        """
        Save domain configuration to YAML file.

        Parameters:
        -----------
        domain_name : str
            Domain name
        config : Dict
            Configuration dictionary to save
        """
        domain_file = self.domain_dir / f"{domain_name}.yaml"

        with open(domain_file, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

        logger.info(f"Saved domain config: {domain_file}")

    def save_template_config(self, template_name: str, config: Dict) -> None:
        """
        Save template configuration to YAML file.

        Parameters:
        -----------
        template_name : str
            Template name
        config : Dict
            Configuration dictionary to save
        """
        template_file = self.template_dir / f"{template_name}.yaml"

        with open(template_file, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)

        logger.info(f"Saved template config: {template_file}")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of ConfigManager usage.
    Run: python core/config_manager.py
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("=" * 70)
    print("ConfigManager Usage Examples")
    print("=" * 70)

    # Example 1: Initialize ConfigManager
    print("\n1. Initialize ConfigManager")
    print("-" * 70)

    config_mgr = ConfigManager()

    # Example 2: List available configurations
    print("\n2. Discover Available Configurations")
    print("-" * 70)

    domains = config_mgr.get_all_domain_names()
    templates = config_mgr.get_all_template_names()

    print(f"Available domains: {domains}")
    print(f"Available templates: {templates}")

    # Example 3: Load domain configuration
    print("\n3. Load Domain Configuration")
    print("-" * 70)

    if domains:
        domain_name = domains[0]
        try:
            config = config_mgr.load_domain_config(domain_name)

            print(f"\nDomain: {config.domain_id}")
            print(f"Name: {config.name}")
            print(f"Chunking strategy: {config.chunking.strategy}")
            print(f"Embedding provider: {config.embeddings.provider}")
            print(f"Vector store: {config.vectorstore.provider}")
            print(f"Retrieval strategies: {config.retrieval.strategies}")

        except FileNotFoundError as e:
            print(f"⚠️  No domain configs found. Create one in {config_mgr.domain_dir}/")
        except ValidationError as e:
            print(f"❌ Validation error:\n{e}")
    else:
        print("⚠️  No domain configurations found")
        print(f"Create configs in: {config_mgr.domain_dir}/")

    # Example 4: Configuration access patterns
    print("\n4. Configuration Access Patterns")
    print("-" * 70)

    print("""
# Access nested configuration
hr_config = config_mgr.load_domain_config("hr_domain")

# Chunking configuration
if hr_config.chunking.strategy == "recursive":
    chunk_size = hr_config.chunking.recursive.chunk_size
    overlap = hr_config.chunking.recursive.overlap

# Embedding configuration
provider = hr_config.embeddings.provider
model_name = hr_config.embeddings.model_name

# Vector store configuration
vectorstore_provider = hr_config.vectorstore.provider
collection_name = hr_config.vectorstore.collection_name

# Security configuration
allowed_types = hr_config.security.allowed_file_types
max_size = hr_config.security.max_file_size_mb
    """)

    print("\n" + "=" * 70)
    print("ConfigManager examples completed!")
    print("=" * 70)

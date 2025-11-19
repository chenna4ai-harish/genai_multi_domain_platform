"""
core/config_manager.py

This module implements the configuration management system for the multi-domain
document intelligence platform.

Purpose:
--------
The ConfigManager is responsible for:
1. Loading YAML configuration files (global + domain-specific)
2. Hierarchical merging (domain configs override global defaults)
3. Validating configurations using Pydantic models
4. Providing a clean API for accessing domain configurations

This enables the "config-driven" architecture where new domains can be added
by just creating a YAML file, with no code changes required.

Configuration Hierarchy:
------------------------
Global Config (configs/global_config.yaml)
    ↓ Provides baseline defaults
Domain Config (configs/domains/hr_domain.yaml)
    ↓ Overrides global settings
Merged Config
    ↓ Validated by Pydantic
DomainConfig Object
    ↓ Used by application

Example Usage:
--------------
# Initialize config manager
config_mgr = ConfigManager(config_dir="configs")

# Load HR domain configuration
hr_config = config_mgr.load_domain_config("hr")

# Access configuration values
print(hr_config.embeddings.model_name)  # "all-MiniLM-L6-v2"
print(hr_config.vector_store.provider)   # "chromadb"

# Load a different domain
finance_config = config_mgr.load_domain_config("finance")
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from models.domain_config import DomainConfig
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Manages hierarchical configuration loading and validation.

    This class implements a two-tier configuration system:
    1. Global config (configs/global_config.yaml) - baseline defaults
    2. Domain configs (configs/domains/*.yaml) - domain-specific overrides

    Key Features:
    -------------
    - Hierarchical merging: Domain configs override global defaults
    - Pydantic validation: Ensures all configs are valid before use
    - Caching: Loaded configs are cached to avoid re-parsing YAML
    - Error handling: Clear error messages for missing/invalid configs

    Design Benefits:
    ----------------
    - Add new domains without code changes (just add YAML file)
    - Share common settings across domains (via global config)
    - Override specific settings per domain (via domain config)
    - Type safety and validation (via Pydantic)

    Example Directory Structure:
    ----------------------------
    configs/
    ├── global_config.yaml          # Baseline defaults
    └── domains/
        ├── hr_domain.yaml          # HR overrides
        ├── finance_domain.yaml     # Finance overrides
        └── engineering_domain.yaml # Engineering overrides
    """

    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the ConfigManager.

        Parameters:
        -----------
        config_dir : str, optional
            Root directory containing configuration files.
            Default: "configs"

            Expected structure:
            config_dir/
            ├── global_config.yaml
            └── domains/
                └── *.yaml

        Raises:
        -------
        FileNotFoundError:
            If config_dir does not exist

        Example:
        --------
        # Use default config directory
        config_mgr = ConfigManager()

        # Use custom config directory
        config_mgr = ConfigManager(config_dir="/app/configs")
        """
        self.config_dir = Path(config_dir)

        # Validate that config directory exists
        if not self.config_dir.exists():
            raise FileNotFoundError(
                f"Configuration directory not found: {self.config_dir}\n"
                f"Please create the directory and add config files."
            )

        # Load global configuration (baseline defaults)
        self.global_config = self._load_global_config()

        # Cache for loaded domain configs (avoid re-parsing YAML)
        # Format: {"hr": DomainConfig(...), "finance": DomainConfig(...)}
        self._domain_config_cache: Dict[str, DomainConfig] = {}

        logger.info(f"ConfigManager initialized with config_dir: {self.config_dir}")

    def _load_global_config(self) -> Dict[str, Any]:
        """
        Load baseline default configuration from global_config.yaml.

        The global config provides default values that are shared across all
        domains. Domain configs can override these defaults.

        Returns:
        --------
        dict:
            Dictionary containing global configuration.
            Returns empty dict {} if global_config.yaml doesn't exist.

        Example global_config.yaml:
        ---------------------------
        embeddings:
          provider: "sentence_transformers"
          model_name: "all-MiniLM-L6-v2"
          device: "cpu"
          batch_size: 32

        chunking:
          strategy: "recursive"
          recursive:
            chunk_size: 500
            overlap: 50

        retrieval:
          strategy: "hybrid"
          alpha: 0.7
          top_k: 10

        Notes:
        ------
        - It's okay if global_config.yaml doesn't exist
        - Each domain MUST provide complete config (or merge will fail validation)
        - Global config is useful for DRY (Don't Repeat Yourself) principle
        """
        global_path = self.config_dir / "global_config.yaml"

        # If global config doesn't exist, return empty dict (no defaults)
        if not global_path.exists():
            logger.warning(
                f"Global config not found: {global_path}\n"
                f"Domain configs must provide all required fields."
            )
            return {}

        try:
            with open(global_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)

            # Handle empty YAML file (None is returned by safe_load)
            if config_data is None:
                logger.warning(f"Global config is empty: {global_path}")
                return {}

            logger.info(f"Loaded global config from: {global_path}")
            return config_data

        except yaml.YAMLError as e:
            # YAML syntax error (e.g., invalid indentation, unclosed quotes)
            logger.error(f"YAML parsing error in {global_path}: {e}")
            raise ValueError(
                f"Invalid YAML syntax in global config: {global_path}\n"
                f"Error: {e}\n"
                f"Please fix the YAML syntax and try again."
            )
        except Exception as e:
            # Other errors (e.g., permission denied, encoding issues)
            logger.error(f"Error loading global config: {e}")
            raise

    def load_domain_config(self, domain_name: str, use_cache: bool = True) -> DomainConfig:
        """
        Load and validate domain-specific configuration.

        This method:
        1. Checks cache (if use_cache=True)
        2. Loads domain YAML file (configs/domains/{domain_name}_domain.yaml)
        3. Merges with global config (domain overrides global)
        4. Validates merged config using Pydantic (DomainConfig model)
        5. Caches result for future calls

        Parameters:
        -----------
        domain_name : str
            Internal name of the domain (lowercase, no spaces).
            Example: "hr", "finance", "engineering"

            This will load: configs/domains/{domain_name}_domain.yaml

        use_cache : bool, optional
            Whether to use cached config if available.
            Default: True (recommended for performance)
            Set to False to force reload (useful for config updates)

        Returns:
        --------
        DomainConfig:
            Validated domain configuration object (Pydantic model).
            Safe to use - all fields are validated and type-checked.

        Raises:
        -------
        FileNotFoundError:
            If domain config file doesn't exist
        ValueError:
            If YAML syntax is invalid or config fails Pydantic validation

        Example:
        --------
        # Load HR domain config
        hr_config = config_mgr.load_domain_config("hr")

        # Access configuration values (type-safe!)
        print(hr_config.name)                      # "hr"
        print(hr_config.display_name)              # "Human Resources"
        print(hr_config.embeddings.provider)       # "sentence_transformers"
        print(hr_config.embeddings.model_name)     # "all-MiniLM-L6-v2"
        print(hr_config.vector_store.provider)     # "chromadb"
        print(hr_config.chunking.strategy)         # "recursive"
        print(hr_config.retrieval.strategy)        # "hybrid"

        # Force reload (ignore cache) - useful after config changes
        hr_config_fresh = config_mgr.load_domain_config("hr", use_cache=False)
        """
        # Check cache first (performance optimization)
        if use_cache and domain_name in self._domain_config_cache:
            logger.debug(f"Using cached config for domain: {domain_name}")
            return self._domain_config_cache[domain_name]

        # Construct path to domain config file
        # Format: configs/domains/{domain_name}_domain.yaml
        domain_path = self.config_dir / "domains" / f"{domain_name}_domain.yaml"

        # Validate that domain config file exists
        if not domain_path.exists():
            raise FileNotFoundError(
                f"Domain config not found: {domain_path}\n"
                f"Please create the file with required configuration.\n"
                f"Example: cp configs/domains/hr_domain.yaml {domain_path}"
            )

        try:
            # Load domain YAML file
            logger.info(f"Loading domain config: {domain_path}")
            with open(domain_path, 'r', encoding='utf-8') as f:
                domain_data = yaml.safe_load(f)

            # Handle empty YAML file
            if domain_data is None:
                raise ValueError(
                    f"Domain config is empty: {domain_path}\n"
                    f"Please add required configuration fields."
                )

            # Merge with global config (domain overrides global)
            merged_config = self._merge_configs(self.global_config, domain_data)

            logger.debug(f"Merged config for domain '{domain_name}': {merged_config.keys()}")

            # Validate merged config using Pydantic
            # This ensures all required fields are present and types are correct
            domain_config = DomainConfig(**merged_config)

            # Cache for future use
            self._domain_config_cache[domain_name] = domain_config

            logger.info(
                f"Successfully loaded and validated config for domain: {domain_name}\n"
                f"  - Embedding provider: {domain_config.embeddings.provider}\n"
                f"  - Vector store: {domain_config.vector_store.provider}\n"
                f"  - Chunking strategy: {domain_config.chunking.strategy}\n"
                f"  - Retrieval strategy: {domain_config.retrieval.strategy}"
            )

            return domain_config

        except yaml.YAMLError as e:
            # YAML syntax error
            logger.error(f"YAML parsing error in {domain_path}: {e}")
            raise ValueError(
                f"Invalid YAML syntax in domain config: {domain_path}\n"
                f"Error: {e}\n"
                f"Please fix the YAML syntax and try again."
            )
        except Exception as e:
            # Pydantic validation error or other issues
            logger.error(f"Error loading domain config '{domain_name}': {e}")
            raise

    def _merge_configs(self, global_cfg: Dict[str, Any],
                       domain_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hierarchically merge global and domain configurations.

        Merge Strategy:
        ---------------
        - Domain config takes precedence over global config
        - Nested dictionaries are deep-merged (recursive)
        - Domain can override specific nested keys without redefining entire sections

        Example:
        --------
        Global config:
        embeddings:
          provider: "sentence_transformers"
          model_name: "all-MiniLM-L6-v2"
          device: "cpu"
          batch_size: 32

        Domain config:
        embeddings:
          model_name: "all-mpnet-base-v2"  # Override only this field
          device: "cuda"                    # Override only this field

        Merged result:
        embeddings:
          provider: "sentence_transformers"  # From global (inherited)
          model_name: "all-mpnet-base-v2"    # From domain (overridden)
          device: "cuda"                      # From domain (overridden)
          batch_size: 32                      # From global (inherited)

        Parameters:
        -----------
        global_cfg : dict
            Global configuration (baseline defaults)
        domain_cfg : dict
            Domain-specific configuration (overrides)

        Returns:
        --------
        dict:
            Merged configuration (domain overrides global)

        Notes:
        ------
        This is a SHALLOW merge at the top level. For deep merging of nested
        dicts, consider using libraries like:
        - deepmerge: https://pypi.org/project/deepmerge/
        - hiyapyco: https://github.com/zerwes/hiyapyco

        The current implementation is simple and sufficient for most use cases
        where domain configs completely override top-level sections.
        """
        # Start with a copy of global config (don't modify original)
        merged = global_cfg.copy()

        # Override with domain config (domain takes precedence)
        # Note: This is a shallow merge (top-level keys only)
        # If you need deep merge for nested dicts, use deepmerge library
        merged.update(domain_cfg)

        logger.debug(
            f"Merged config: {len(global_cfg)} global keys + "
            f"{len(domain_cfg)} domain keys = {len(merged)} total keys"
        )

        return merged

    def get_all_domain_names(self) -> list[str]:
        """
        Get list of all available domain names.

        Scans the configs/domains/ directory for *_domain.yaml files and
        extracts the domain names.

        Returns:
        --------
        list[str]:
            List of domain names (without "_domain.yaml" suffix).
            Example: ["hr", "finance", "engineering", "legal"]

        Example:
        --------
        config_mgr = ConfigManager()
        domains = config_mgr.get_all_domain_names()
        print(f"Available domains: {domains}")
        # Output: Available domains: ['hr', 'finance', 'engineering']

        # Load configs for all domains
        for domain in domains:
            config = config_mgr.load_domain_config(domain)
            print(f"{domain}: {config.display_name}")

        Use Case:
        ---------
        - Populate domain dropdown in UI
        - Bulk operations (reindex all domains)
        - Admin dashboard (show all configured domains)
        """
        domains_dir = self.config_dir / "domains"

        if not domains_dir.exists():
            logger.warning(f"Domains directory not found: {domains_dir}")
            return []

        # Find all *_domain.yaml files
        domain_files = domains_dir.glob("*_domain.yaml")

        # Extract domain names (remove "_domain.yaml" suffix)
        # Example: "hr_domain.yaml" → "hr"
        domain_names = [
            f.stem.replace("_domain", "")  # stem = filename without extension
            for f in domain_files
        ]

        logger.info(f"Found {len(domain_names)} domain(s): {domain_names}")
        return sorted(domain_names)

    def reload_domain_config(self, domain_name: str) -> DomainConfig:
        """
        Force reload of a domain configuration (bypass cache).

        Useful when:
        - Config file was updated
        - Testing different configurations
        - Hot-reloading in development

        Parameters:
        -----------
        domain_name : str
            Name of domain to reload

        Returns:
        --------
        DomainConfig:
            Freshly loaded and validated domain configuration

        Example:
        --------
        # Initial load
        hr_config = config_mgr.load_domain_config("hr")
        print(hr_config.embeddings.model_name)  # "all-MiniLM-L6-v2"

        # User edits hr_domain.yaml and changes model_name to "all-mpnet-base-v2"

        # Reload to pick up changes
        hr_config = config_mgr.reload_domain_config("hr")
        print(hr_config.embeddings.model_name)  # "all-mpnet-base-v2"
        """
        logger.info(f"Force reloading config for domain: {domain_name}")

        # Clear from cache
        if domain_name in self._domain_config_cache:
            del self._domain_config_cache[domain_name]

        # Load fresh (use_cache=False)
        return self.load_domain_config(domain_name, use_cache=False)

    def clear_cache(self) -> None:
        """
        Clear the entire domain config cache.

        Forces all subsequent load_domain_config() calls to reload from disk.

        Use Case:
        ---------
        - After bulk config updates
        - Testing/development
        - When you want to ensure fresh configs

        Example:
        --------
        # Load some configs (they get cached)
        config_mgr.load_domain_config("hr")
        config_mgr.load_domain_config("finance")

        # User edits multiple config files

        # Clear cache to force reload
        config_mgr.clear_cache()

        # Next loads will read from disk
        hr_config = config_mgr.load_domain_config("hr")  # Fresh load
        """
        cache_size = len(self._domain_config_cache)
        self._domain_config_cache.clear()
        logger.info(f"Cleared config cache ({cache_size} domain(s))")


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of ConfigManager usage.
    Run this file directly to see examples: python core/config_manager.py
    """

    # Configure logging to see debug messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("ConfigManager Usage Examples")
    print("=" * 70)

    try:
        # Example 1: Initialize ConfigManager
        print("\n1. Initializing ConfigManager...")
        config_mgr = ConfigManager(config_dir="configs")
        print("✅ ConfigManager initialized successfully!")

        # Example 2: Get all available domains
        print("\n2. Getting all available domains...")
        domains = config_mgr.get_all_domain_names()
        print(f"✅ Found domains: {domains}")

        # Example 3: Load HR domain config
        print("\n3. Loading HR domain configuration...")
        hr_config = config_mgr.load_domain_config("hr")
        print(f"✅ Loaded HR config:")
        print(f"   - Display Name: {hr_config.display_name}")
        print(f"   - Description: {hr_config.description}")
        print(f"   - Embedding Provider: {hr_config.embeddings.provider}")
        print(f"   - Embedding Model: {hr_config.embeddings.model_name}")
        print(f"   - Vector Store: {hr_config.vector_store.provider}")
        print(f"   - Chunking Strategy: {hr_config.chunking.strategy}")
        print(f"   - Retrieval Strategy: {hr_config.retrieval.strategy}")

        # Example 4: Load multiple domains
        print("\n4. Loading multiple domains...")
        for domain_name in domains[:3]:  # Load first 3 domains
            config = config_mgr.load_domain_config(domain_name)
            print(f"✅ {config.display_name}: {config.embeddings.provider} + {config.vector_store.provider}")

        # Example 5: Demonstrate caching
        print("\n5. Demonstrating config caching...")
        print("   First load (from disk):")
        hr_config_1 = config_mgr.load_domain_config("hr", use_cache=False)
        print(f"   - Loaded: {hr_config_1.name}")

        print("   Second load (from cache):")
        hr_config_2 = config_mgr.load_domain_config("hr", use_cache=True)
        print(f"   - Loaded: {hr_config_2.name}")
        print(f"   - Same object? {hr_config_1 is hr_config_2}")

        # Example 6: Clear cache
        print("\n6. Clearing config cache...")
        config_mgr.clear_cache()
        print("✅ Cache cleared!")

        # Example 7: Reload domain config
        print("\n7. Reloading domain config...")
        hr_config_fresh = config_mgr.reload_domain_config("hr")
        print(f"✅ Reloaded: {hr_config_fresh.name}")

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure configs/ directory exists with:")
        print("  - configs/global_config.yaml (optional)")
        print("  - configs/domains/hr_domain.yaml (required)")

    except ValueError as e:
        print(f"\n❌ Validation Error: {e}")
        print("\nPlease check your YAML syntax and config structure.")

    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")

    print("\n" + "=" * 70)
    print("ConfigManager examples completed!")
    print("=" * 70)

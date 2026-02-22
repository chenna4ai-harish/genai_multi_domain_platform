"""
core/services/domain_service.py

Service layer for domain lifecycle management.

Responsibilities:
- Create a new domain from a template
- Immediately initialize the vector store collection on creation
- List, load, and delete domains
- Load and list templates

Design decision:
  When a domain is created, the ChromaDB collection is initialized immediately
  (not lazily on first upload). This ensures the domain is fully ready and
  the collection name is deterministic (always equals the domain_id).

Collection naming rule:
  collection_name = domain_id
  persist_directory = ./data/chromadb/<domain_id>
"""

import copy
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class DomainService:
    """
    Service layer for domain creation and management.

    All domain management operations go through this class.
    The UI must never call ConfigManager or VectorStoreFactory directly.
    """

    def __init__(self):
        self.config_manager = ConfigManager()

    # =========================================================================
    # DOMAIN CREATION
    # =========================================================================

    def create_domain(
        self,
        domain_id: str,
        domain_name: str,
        template_name: str,
        description: str = "",
    ) -> Dict[str, Any]:
        """
        Create a new domain from a template and immediately initialize
        the ChromaDB vector store collection.

        Steps:
        1. Validate inputs
        2. Check domain does not already exist
        3. Load template YAML
        4. Override domain-specific fields
        5. Save config to configs/domains/<domain_id>.yaml
        6. Initialize DocumentService → pipeline → vectorstore (collection created)
        7. Return status dict

        Parameters
        ----------
        domain_id : str
            Unique identifier for the domain (used as collection name).
            Must be lowercase alphanumeric + underscores only.
        domain_name : str
            Human-readable display name.
        template_name : str
            Name of the template to base this domain on (without .yaml).
        description : str
            Optional description.

        Returns
        -------
        Dict[str, Any]
            {
              "domain_id": str,
              "collection_name": str,
              "persist_directory": str,
              "vectors_in_collection": int,
              "status": "created"
            }

        Raises
        ------
        ValueError
            If domain_id is invalid or already exists.
        FileNotFoundError
            If template not found.
        RuntimeError
            If vector store initialization fails.
        """
        # --- 1. Validate domain_id ---
        if not domain_id or not domain_id.strip():
            raise ValueError("domain_id cannot be empty")

        domain_id = domain_id.strip().lower().replace(" ", "_")

        import re
        if not re.match(r'^[a-z0-9][a-z0-9_-]{1,61}[a-z0-9]$', domain_id):
            raise ValueError(
                f"Invalid domain_id '{domain_id}'. "
                "Must be 3-63 chars, lowercase alphanumeric, underscores or hyphens, "
                "start and end with alphanumeric."
            )

        # --- 2. Check domain does not already exist ---
        domain_file = self.config_manager.domain_dir / f"{domain_id}.yaml"
        if domain_file.exists():
            raise ValueError(
                f"Domain '{domain_id}' already exists. "
                "Choose a different domain_id or delete the existing one first."
            )

        # --- 3. Load template YAML ---
        template_file = self.config_manager.template_dir / f"{template_name}.yaml"
        if not template_file.exists():
            raise FileNotFoundError(
                f"Template '{template_name}' not found. "
                f"Available: {self.config_manager.get_all_template_names()}"
            )

        with open(template_file, "r") as f:
            template_dict = yaml.safe_load(f) or {}

        # --- 4. Build domain config from template + overrides ---
        domain_dict = copy.deepcopy(template_dict)

        # Remove playground-specific keys that don't belong in a domain config
        for key in ("playground_name", "session_id", "created_at", "last_modified",
                    "llm_rerank", "distance_metric"):
            domain_dict.pop(key, None)

        # Set domain identity fields
        domain_dict["domain_id"] = domain_id
        domain_dict["name"] = domain_name or domain_id
        domain_dict["description"] = description or f"Domain created from template '{template_name}'"

        # Force collection_name = domain_id (deterministic naming)
        persist_dir = f"./data/chromadb/{domain_id}"
        domain_dict.setdefault("vectorstore", {})
        domain_dict["vectorstore"]["collection_name"] = domain_id
        domain_dict["vectorstore"]["persist_directory"] = persist_dir

        # Timestamp
        domain_dict["created_at"] = datetime.utcnow().isoformat()

        # --- 5. Save YAML ---
        with open(domain_file, "w") as f:
            yaml.safe_dump(domain_dict, f, sort_keys=False, default_flow_style=False)

        logger.info(f"Saved domain config: {domain_file}")

        # --- 6. Initialize vector store collection immediately ---
        try:
            from core.services.document_service import DocumentService
            svc = DocumentService(domain_id)
            # collection.count() confirms the collection was created
            vector_count = svc.pipeline.vectorstore.collection.count()
            logger.info(
                f"✅ Domain '{domain_id}' created — "
                f"ChromaDB collection '{domain_id}' initialized "
                f"({vector_count} vectors, dir: {persist_dir})"
            )
        except Exception as e:
            # Config was saved — roll back the YAML so state stays consistent
            domain_file.unlink(missing_ok=True)
            logger.error(f"Vector store init failed for domain '{domain_id}': {e}")
            raise RuntimeError(
                f"Domain config saved but vector store initialization failed: {e}\n"
                f"Config file has been removed. Please fix the issue and retry."
            )

        return {
            "domain_id": domain_id,
            "collection_name": domain_id,
            "persist_directory": persist_dir,
            "vectors_in_collection": vector_count,
            "status": "created",
        }

    # =========================================================================
    # DOMAIN LISTING & LOADING
    # =========================================================================

    def list_domains(self) -> List[Dict[str, Any]]:
        """
        List all available domains with summary info.

        Returns
        -------
        List[Dict[str, Any]]
            Each entry: { domain_id, name, description, collection_name, created_at }
        """
        domains = []
        for name in self.config_manager.get_all_domain_names():
            domain_file = self.config_manager.domain_dir / f"{name}.yaml"
            try:
                with open(domain_file, "r") as f:
                    d = yaml.safe_load(f) or {}
                domains.append({
                    "domain_id": d.get("domain_id", name),
                    "name": d.get("name", name),
                    "description": d.get("description", ""),
                    "collection_name": (d.get("vectorstore") or {}).get("collection_name", name),
                    "created_at": d.get("created_at", ""),
                })
            except Exception as e:
                logger.warning(f"Could not read domain config '{name}': {e}")
        return domains

    def get_domain_vector_count(self, domain_id: str) -> Optional[int]:
        """
        Return the number of vectors currently in the domain's collection.

        Returns None if the collection cannot be accessed.
        """
        try:
            from core.services.document_service import DocumentService
            svc = DocumentService(domain_id)
            return svc.pipeline.vectorstore.collection.count()
        except Exception as e:
            logger.warning(f"Could not get vector count for '{domain_id}': {e}")
            return None

    # =========================================================================
    # TEMPLATE LISTING
    # =========================================================================

    def list_templates(self) -> List[Dict[str, Any]]:
        """
        List all available templates with summary info.

        Returns
        -------
        List[Dict[str, Any]]
            Each entry: { template_name, name, description, vectorstore_provider }
        """
        templates = []
        for name in self.config_manager.get_all_template_names():
            tpl_file = self.config_manager.template_dir / f"{name}.yaml"
            try:
                with open(tpl_file, "r") as f:
                    d = yaml.safe_load(f) or {}
                templates.append({
                    "template_name": name,
                    "name": d.get("name", name),
                    "description": d.get("description", ""),
                    "vectorstore_provider": (d.get("vectorstore") or {}).get("provider", ""),
                    "chunking_strategy": (d.get("chunking") or {}).get("strategy", ""),
                    "embedding_provider": (d.get("embeddings") or {}).get("provider", ""),
                })
            except Exception as e:
                logger.warning(f"Could not read template '{name}': {e}")
        return templates

    def get_template_raw(self, template_name: str) -> Dict[str, Any]:
        """
        Load raw template YAML as a dict (for pre-filling domain creation form).
        """
        tpl_file = self.config_manager.template_dir / f"{template_name}.yaml"
        if not tpl_file.exists():
            raise FileNotFoundError(f"Template '{template_name}' not found.")
        with open(tpl_file, "r") as f:
            return yaml.safe_load(f) or {}

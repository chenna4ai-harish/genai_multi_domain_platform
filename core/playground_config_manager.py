import yaml
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class PlaygroundConfigManager:
    """
    Configuration Manager for Playground experiments.
    Works exclusively with configs/playground/ directory.
    Separate from production ConfigManager (configs/domains/).
    """

    # Class-level attributes
    playground_dir = Path("configs/playground")
    template_dir = Path("configs/templates")

    def __init__(self):
        """Initialize PlaygroundConfigManager with directories and global config."""
        # Ensure directories exist
        self.playground_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Load global config
        self.global_config = self._load_global_config()

    def _load_global_config(self) -> Dict[str, Any]:
        """Load global default config (shared with production)."""
        global_path = Path("configs/global_config.yaml")
        if global_path.exists():
            try:
                with open(global_path) as f:
                    config = yaml.safe_load(f)
                    logger.info("Global config loaded successfully")
                    return config or {}
            except Exception as e:
                logger.warning(f"Failed to load global config: {e}")
                return {}
        else:
            logger.warning("global_config.yaml not found, using empty defaults")
            return {}

    @staticmethod
    def save_config(name: str, session_id: str, config: Dict[str, Any]) -> str:
        """
        Save playground config as {name}_{session_id}.yaml

        Args:
            name: Config name
            session_id: Unique session identifier
            config: Configuration dictionary

        Returns:
            Path to saved file as string
        """
        safe_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', name)
        filename = f"{safe_name}_{session_id}.yaml"

        # Add metadata
        config["playground_name"] = safe_name
        config["session_id"] = session_id
        config["created_at"] = datetime.now().isoformat()
        config["last_modified"] = datetime.now().isoformat()

        path = PlaygroundConfigManager.playground_dir / filename

        try:
            with open(path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            logger.info(f"Saved playground config: {filename}")
            return str(path)
        except Exception as e:
            logger.error(f"Failed to save config {filename}: {e}")
            raise

    @staticmethod
    def list_configs() -> List[Dict[str, Any]]:
        """
        Return list of all playground configs with metadata.

        Returns:
            List of dicts containing config metadata
        """
        configs = []
        for file in PlaygroundConfigManager.playground_dir.glob("*.yaml"):
            try:
                with open(file) as f:
                    cfg = yaml.safe_load(f)
                configs.append({
                    "filename": file.name,
                    "name": cfg.get("playground_name", cfg.get("name", "unknown")),
                    "session_id": cfg.get("session_id", "unknown"),
                    "created_at": cfg.get("created_at", "unknown"),
                    "description": cfg.get("description", "")
                })
            except Exception as e:
                logger.warning(f"Failed to load playground config {file}: {e}")
                continue

        # Sort by created_at (most recent first)
        configs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return configs

    @staticmethod
    def load_config(filename: str) -> Dict[str, Any]:
        """
        Load playground config given a filename.

        Args:
            filename: Name of the config file (e.g., 'test_config_abc123.yaml')

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        path = PlaygroundConfigManager.playground_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Playground config '{filename}' not found in {PlaygroundConfigManager.playground_dir}")

        try:
            with open(path) as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded playground config: {filename}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config {filename}: {e}")
            raise

    @staticmethod
    def find_config_by_name(name: str) -> Optional[str]:
        """
        Find latest playground config by name (returns filename).

        Args:
            name: Config name to search for

        Returns:
            Filename if found, None otherwise
        """
        safe_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', name)
        matches = sorted(
            PlaygroundConfigManager.playground_dir.glob(f"{safe_name}_*.yaml"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if matches:
            logger.info(f"Found config for '{name}': {matches[0].name}")
            return matches[0].name
        else:
            logger.warning(f"No config found for name '{name}'")
            return None

    @staticmethod
    def delete_config(filename: str) -> tuple:
        """
        Delete playground config file and optionally cleanup vector DB.

        Args:
            filename: Name of config file to delete

        Returns:
            Tuple of (success: bool, message: str)
        """
        path = PlaygroundConfigManager.playground_dir / filename
        if not path.exists():
            return False, f"Config '{filename}' does not exist."

        try:
            # Load config to get collection info before deleting
            with open(path) as f:
                cfg = yaml.safe_load(f)
            collection = cfg.get("vectorstore", {}).get("collection_name", "unknown")

            # TODO: Add vector store cleanup here if needed
            # Example: vectorstoreFactory.delete_collection(collection)

            # Delete the file
            path.unlink()
            logger.info(f"Deleted playground config: {filename}")
            return True, f"✅ Deleted config '{filename}' (collection: {collection})"
        except Exception as e:
            logger.error(f"Error deleting config {filename}: {e}")
            return False, f"❌ Error deleting config: {e}"

    @staticmethod
    def cleanup_expired_configs(expiry_hours: int = 48) -> int:
        """
        Remove playground configs older than expiry_hours.

        Args:
            expiry_hours: Age threshold in hours

        Returns:
            Number of configs deleted
        """
        now = datetime.now()
        deleted = 0

        for file in PlaygroundConfigManager.playground_dir.glob("*.yaml"):
            try:
                with open(file) as f:
                    cfg = yaml.safe_load(f)
                created = cfg.get("created_at")

                if created:
                    dt = datetime.fromisoformat(created)
                    age_hours = (now - dt).total_seconds() / 3600

                    if age_hours > expiry_hours:
                        file.unlink()
                        deleted += 1
                        logger.info(f"Cleaned up expired config: {file.name} (age: {age_hours:.1f}h)")
            except Exception as e:
                logger.warning(f"Error checking/deleting {file.name}: {e}")
                continue

        logger.info(f"Cleanup completed: {deleted} configs deleted")
        return deleted

    def merge_with_global(self, playground_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge playground config with global defaults.

        Args:
            playground_config: Playground-specific configuration

        Returns:
            Merged configuration dictionary
        """
        import copy
        merged = copy.deepcopy(self.global_config)
        self._deep_merge(merged, playground_config)
        return merged

    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> None:
        """
        Deep merge override dict into base dict (in-place).

        Args:
            base: Base dictionary to merge into
            override: Dictionary with values to override
        """
        for key, value in override.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                PlaygroundConfigManager._deep_merge(base[key], value)
            else:
                base[key] = value

    @staticmethod
    def get_config_metadata(filename: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific config file.

        Args:
            filename: Config filename

        Returns:
            Metadata dictionary or None if not found
        """
        path = PlaygroundConfigManager.playground_dir / filename
        if not path.exists():
            return None

        try:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            return {
                "name": cfg.get("playground_name", cfg.get("name")),
                "session_id": cfg.get("session_id"),
                "created_at": cfg.get("created_at"),
                "last_modified": cfg.get("last_modified"),
                "description": cfg.get("description"),
                "vectorstore": cfg.get("vectorstore", {})
            }
        except Exception as e:
            logger.error(f"Failed to get metadata for {filename}: {e}")
            return None

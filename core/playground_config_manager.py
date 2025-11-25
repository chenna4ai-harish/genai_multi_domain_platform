import yaml
from pathlib import Path
from datetime import datetime
import re

class PlaygroundConfigManager:
    playground_dir = Path("configs/playground")
    playground_dir.mkdir(parents=True, exist_ok=True)

    ###############################################
    # Core Save/Load/Delete Methods
    ###############################################

    @staticmethod
    def save_config(name, session_id, config):
        """Save config as {name}_{session_id}.yaml with creation metadata."""
        safe_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', name)
        filename = f"{safe_name}_{session_id}.yaml"
        config["playground_name"] = safe_name
        config["session_id"] = session_id
        config["created_at"] = datetime.now().isoformat()
        path = PlaygroundConfigManager.playground_dir / filename
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        return str(path)

    @staticmethod
    def list_configs():
        """Return list of all playground configs with metadata."""
        configs = []
        for file in PlaygroundConfigManager.playground_dir.glob("*.yaml"):
            try:
                with open(file) as f:
                    cfg = yaml.safe_load(f)
                configs.append({
                    "filename": file.name,
                    "name": cfg.get("playground_name", "unknown"),
                    "session_id": cfg.get("session_id", "unknown"),
                    "created_at": cfg.get("created_at", "-")
                })
            except Exception:
                continue
        return configs

    @staticmethod
    def load_config(filename):
        """Load config given a filename."""
        path = PlaygroundConfigManager.playground_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"No config with filename {filename}")
        with open(path) as f:
            return yaml.safe_load(f)

    @staticmethod
    def find_config_by_name(name):
        """Find latest config by name (returns filename)."""
        safe_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', name)
        matches = sorted([f for f in PlaygroundConfigManager.playground_dir.glob(f"{safe_name}_*.yaml")], key=lambda x: x.stat().st_mtime, reverse=True)
        return matches[0].name if matches else None

    @staticmethod
    def delete_config(filename):
        """Delete config file and (optionally) associated vector db collection."""
        path = PlaygroundConfigManager.playground_dir / filename
        if not path.exists():
            return False, "Config does not exist."
        # Optionally read config and call vector db cleanup here,
        # e.g., VectorStoreFactory.delete_collection(cfg['vectorstore']['collection_name'])
        with open(path) as f:
            cfg = yaml.safe_load(f)
            collection = cfg.get("vectorstore", {}).get("collection_name")
            # Call your backend vectorstore delete here
            # e.g., VectorStoreFactory.delete_collection(collection)
        path.unlink()
        return True, f"Deleted config and (if implemented) vector DB collection: {collection}"

    def save_as_template(
            template_name: str,
            config_name: str,
            session_id: str,  # you can ignore this inside if you don't need it yet
    ):
        from core.playground_config_manager import PlaygroundConfigManager

        if not template_name:
            return "⚠️ Please enter a template name."

        # Find the config by its name
        all_configs = PlaygroundConfigManager.list_configs()
        match = next((c for c in all_configs if c["name"] == config_name), None)
        if not match:
            return f"⚠️ No config named **{config_name}** found to save as template."

        # Load that config using the stored filename
        cfg = PlaygroundConfigManager.load_config(match["filename"])

        # Write it as a YAML template
        path = Path("configs/templates") / f"{template_name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(cfg, f)

        return f"⭐ Template **{template_name}** created from config **{config_name}** at `{path}`."


    ###############################################
    # Housekeeping/Utilities
    ###############################################

    @staticmethod
    def cleanup_expired_configs(expiry_hours=48):
        """Remove any config older than expiry_hours."""
        now = datetime.now()
        deleted = 0
        for file in PlaygroundConfigManager.playground_dir.glob("*.yaml"):
            try:
                with open(file) as f:
                    cfg = yaml.safe_load(f)
                created = cfg.get("created_at")
                if created:
                    dt = datetime.fromisoformat(created)
                    if (now - dt).total_seconds() > expiry_hours * 3600:
                        file.unlink()
                        deleted += 1
            except Exception:
                continue
        return deleted

    @staticmethod
    def config_metadata(filename):
        """Get metadata for a config file."""
        path = PlaygroundConfigManager.playground_dir / filename
        if not path.exists():
            return None
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return {
            "name": cfg.get("playground_name"),
            "session_id": cfg.get("session_id"),
            "created_at": cfg.get("created_at"),
            "vectorstore": cfg.get("vectorstore", {})
        }

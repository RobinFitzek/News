"""
Plugin Manager — Local file-based plugin system for Stockholm.

Plugins are .py files uploaded by the user via the Settings UI.
No external plugin store — everything runs locally.

Plugin contract:
  PLUGIN_NAME     = "My Plugin"        (required)
  PLUGIN_VERSION  = "1.0.0"           (optional, default "1.0.0")
  PLUGIN_TYPE     = "notifier"         (required: notifier | analyzer | screener | exporter)
  PLUGIN_DESCRIPTION = "What it does" (optional)
  PLUGIN_AUTHOR   = "Author"          (optional)
  PLUGIN_SETTINGS = {                  (optional)
      "key": {"type": "string|password|number|boolean",
               "label": "Human label",
               "required": True|False,
               "default": ...}
  }

  def run(context: dict, settings: dict) -> dict:
      # context keys by type:
      #   notifier:  ticker, signal, recommendation, confidence, risk_score
      #   analyzer:  ticker, analysis_result (dict)
      #   screener:  ticker, quant_data (dict)
      #   exporter:  watchlist (list), portfolio (list)
      return {"ok": True, "message": "..."}
"""
import ast
import importlib.util
import json
import logging
import re
import sys
import threading
from pathlib import Path
from typing import Optional

from core.config import BASE_DIR
from core.database import db

logger = logging.getLogger(__name__)

PLUGINS_DIR = BASE_DIR / "plugins"
PLUGINS_DIR.mkdir(exist_ok=True)

VALID_TYPES = {"notifier", "analyzer", "screener", "exporter"}
MAX_PLUGIN_SIZE = 100 * 1024  # 100 KB
PLUGIN_TIMEOUT = 30  # seconds


class PluginManager:
    """Manages lifecycle of local .py plugin files."""

    # ------------------------------------------------------------------ #
    # Metadata parsing (no code execution — uses AST)                     #
    # ------------------------------------------------------------------ #

    def get_metadata(self, filepath: Path) -> Optional[dict]:
        """
        Parse plugin metadata from a .py file using the AST module.
        Never executes the file. Returns None if required fields are missing.
        """
        try:
            source = filepath.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(filepath))
        except (SyntaxError, OSError) as e:
            logger.warning(f"Plugin {filepath.name}: parse error — {e}")
            return None

        constants = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith("PLUGIN_"):
                        try:
                            constants[target.id] = ast.literal_eval(node.value)
                        except (ValueError, TypeError):
                            pass

        if "PLUGIN_NAME" not in constants or "PLUGIN_TYPE" not in constants:
            return None

        plugin_type = constants.get("PLUGIN_TYPE", "notifier")
        if plugin_type not in VALID_TYPES:
            logger.warning(f"Plugin {filepath.name}: unknown type '{plugin_type}'")
            return None

        # Check run() function exists
        has_run = any(
            isinstance(n, ast.FunctionDef) and n.name == "run"
            for n in ast.walk(tree)
        )
        if not has_run:
            logger.warning(f"Plugin {filepath.name}: missing run() function")
            return None

        # Parse PLUGIN_SETTINGS if present
        settings_schema = constants.get("PLUGIN_SETTINGS", {})
        if not isinstance(settings_schema, dict):
            settings_schema = {}

        return {
            "name": str(constants.get("PLUGIN_NAME", filepath.stem)),
            "version": str(constants.get("PLUGIN_VERSION", "1.0.0")),
            "plugin_type": plugin_type,
            "description": str(constants.get("PLUGIN_DESCRIPTION", "")),
            "author": str(constants.get("PLUGIN_AUTHOR", "")),
            "settings_schema": settings_schema,
        }

    # ------------------------------------------------------------------ #
    # Install / Uninstall                                                  #
    # ------------------------------------------------------------------ #

    def install(self, filename: str, content: bytes) -> dict:
        """
        Save a .py file to the plugins directory, validate it, and register
        it in the database. Returns the metadata dict with 'id' added.
        Raises ValueError on validation errors.
        """
        # Sanitize filename
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", Path(filename).stem) + ".py"
        filepath = PLUGINS_DIR / safe_name

        if len(content) > MAX_PLUGIN_SIZE:
            raise ValueError(f"Plugin file exceeds maximum size of {MAX_PLUGIN_SIZE // 1024} KB")

        # Write temporarily to parse
        filepath.write_bytes(content)
        try:
            meta = self.get_metadata(filepath)
        except Exception as e:
            filepath.unlink(missing_ok=True)
            raise ValueError(f"Could not parse plugin: {e}")

        if meta is None:
            filepath.unlink(missing_ok=True)
            raise ValueError(
                "Invalid plugin file. Must define PLUGIN_NAME, PLUGIN_TYPE, and a run() function."
            )

        plugin_id = db.upsert_plugin(
            filename=safe_name,
            name=meta["name"],
            plugin_type=meta["plugin_type"],
            version=meta["version"],
            description=meta["description"],
            author=meta["author"],
        )
        meta["id"] = plugin_id
        meta["filename"] = safe_name
        logger.info(f"Plugin installed: {meta['name']} ({safe_name}) id={plugin_id}")
        return meta

    def uninstall(self, plugin_id: int):
        """Delete the plugin file and its DB record."""
        plugin = db.get_plugin(plugin_id)
        if not plugin:
            raise ValueError(f"Plugin id={plugin_id} not found")
        filepath = PLUGINS_DIR / plugin["filename"]
        filepath.unlink(missing_ok=True)
        # Remove from sys.modules to allow clean re-install
        mod_key = f"_stockholm_plugin_{plugin['filename']}"
        sys.modules.pop(mod_key, None)
        db.delete_plugin(plugin_id)
        logger.info(f"Plugin uninstalled: {plugin['name']} (id={plugin_id})")

    def reload_from_disk(self):
        """
        Sync DB with the plugins/ directory on startup.
        Adds missing DB records for .py files on disk; removes DB records
        for files no longer on disk.
        """
        on_disk = {p.name for p in PLUGINS_DIR.glob("*.py") if p.name != "example_notifier.py" or True}
        in_db = {p["filename"] for p in db.list_plugins()}

        # Register new files not yet in DB
        for fname in on_disk - in_db:
            filepath = PLUGINS_DIR / fname
            meta = self.get_metadata(filepath)
            if meta:
                db.upsert_plugin(
                    filename=fname,
                    name=meta["name"],
                    plugin_type=meta["plugin_type"],
                    version=meta["version"],
                    description=meta["description"],
                    author=meta["author"],
                )
                logger.info(f"Plugin discovered on disk: {fname}")

        # Remove stale DB records
        for fname in in_db - on_disk:
            plugin = db.get_plugin_by_filename(fname)
            if plugin:
                db.delete_plugin(plugin["id"])
                logger.info(f"Removed stale plugin record: {fname}")

    # ------------------------------------------------------------------ #
    # Module loading & execution                                           #
    # ------------------------------------------------------------------ #

    def _load_module(self, filename: str):
        """Load a plugin .py file as a Python module. Caches in sys.modules."""
        mod_key = f"_stockholm_plugin_{filename}"
        filepath = PLUGINS_DIR / filename
        spec = importlib.util.spec_from_file_location(mod_key, filepath)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_key] = module
        spec.loader.exec_module(module)
        return module

    def _safe_run(self, plugin_id: int, filename: str, context: dict, settings: dict) -> dict:
        """
        Execute plugin.run(context, settings) in a thread with a timeout.
        Catches all exceptions and logs them to the DB.
        Returns {"ok": bool, "message": str}.
        """
        result = {"ok": False, "message": "Plugin did not complete"}
        error_msg = None

        def _exec():
            nonlocal result, error_msg
            try:
                module = self._load_module(filename)
                r = module.run(context, settings)
                if isinstance(r, dict):
                    result = r
                else:
                    result = {"ok": True, "message": str(r)}
            except Exception as e:
                error_msg = str(e)
                result = {"ok": False, "message": f"Error: {e}"}
                logger.warning(f"Plugin {filename} error: {e}")

        t = threading.Thread(target=_exec, daemon=True)
        t.start()
        t.join(timeout=PLUGIN_TIMEOUT)

        if t.is_alive():
            error_msg = f"Plugin timed out after {PLUGIN_TIMEOUT}s"
            result = {"ok": False, "message": error_msg}
            logger.warning(f"Plugin {filename} timed out")

        db.set_plugin_last_run(plugin_id, error=error_msg)
        return result

    # ------------------------------------------------------------------ #
    # Public runner methods                                                #
    # ------------------------------------------------------------------ #

    def run_notifiers(self, context: dict):
        """
        Call all enabled notifier plugins.
        context: {ticker, signal, recommendation, confidence, risk_score}
        """
        plugins = [p for p in db.list_plugins()
                   if p["plugin_type"] == "notifier" and p["enabled"]]
        for plugin in plugins:
            try:
                settings = json.loads(plugin.get("settings") or "{}")
                self._safe_run(plugin["id"], plugin["filename"], context, settings)
            except Exception as e:
                logger.error(f"run_notifiers failed for {plugin['filename']}: {e}")

    def run_analyzers(self, ticker: str, analysis_result: dict) -> list:
        """
        Call all enabled analyzer plugins.
        Returns list of result dicts from each plugin.
        """
        plugins = [p for p in db.list_plugins()
                   if p["plugin_type"] == "analyzer" and p["enabled"]]
        results = []
        for plugin in plugins:
            try:
                settings = json.loads(plugin.get("settings") or "{}")
                ctx = {"ticker": ticker, "analysis_result": analysis_result}
                r = self._safe_run(plugin["id"], plugin["filename"], ctx, settings)
                results.append({"plugin": plugin["name"], **r})
            except Exception as e:
                logger.error(f"run_analyzers failed for {plugin['filename']}: {e}")
        return results

    def run_plugin(self, plugin_id: int, context: Optional[dict] = None) -> dict:
        """
        Execute a single plugin by id (used for manual test runs from the UI).
        If no context is provided, a dummy context matching the plugin type is used.
        """
        plugin = db.get_plugin(plugin_id)
        if not plugin:
            return {"ok": False, "message": f"Plugin id={plugin_id} not found"}

        dummy_contexts = {
            "notifier": {
                "ticker": "AAPL", "signal": "STRONG_BUY",
                "recommendation": "Test run from Settings UI",
                "confidence": 80, "risk_score": 4,
            },
            "analyzer": {
                "ticker": "AAPL",
                "analysis_result": {"signal": "BUY", "confidence": 75},
            },
            "screener": {
                "ticker": "AAPL",
                "quant_data": {"rsi": 45, "momentum_3m": 0.08},
            },
            "exporter": {
                "watchlist": [{"ticker": "AAPL", "name": "Apple Inc."}],
                "portfolio": [],
            },
        }
        ctx = context or dummy_contexts.get(plugin["plugin_type"], {})
        settings = json.loads(plugin.get("settings") or "{}")
        return self._safe_run(plugin["id"], plugin["filename"], ctx, settings)

    def get_plugin_settings_schema(self, plugin_id: int) -> dict:
        """
        Return the PLUGIN_SETTINGS schema dict from the plugin file (AST-parsed).
        Used by the UI to build the settings form dynamically.
        """
        plugin = db.get_plugin(plugin_id)
        if not plugin:
            return {}
        filepath = PLUGINS_DIR / plugin["filename"]
        meta = self.get_metadata(filepath)
        return meta.get("settings_schema", {}) if meta else {}


plugin_manager = PluginManager()

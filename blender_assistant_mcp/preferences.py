"""Extension preferences and settings."""

import bpy

LLAMA_CPP_AVAILABLE = True

from . import ollama_adapter as llama_manager

# Global cache for Ollama models
_ollama_models_cache = []
_ollama_models_enum_items = []


# Online Ollama Library search state
_library_search_state = {
    "active": False,
    "thread": None,
    "results": [],  # list[str]
    "status": "",
    "error": None,
}


def get_preferences():
    """Get the extension preferences (robust across extension IDs) and
    ensure Stock Photo tools are registered regardless of keys.

    Returns:
        AssistantPreferences or a shim with expected attributes.
    """
    # 1) Find the correct addon preferences entry across possible IDs
    try:
        addons = bpy.context.preferences.addons
    except Exception:
        addons = {}

    prefs_obj = None

    # Preferred: exact package key
    try:
        if __package__ in addons:
            prefs_obj = addons[__package__].preferences
    except Exception:
        prefs_obj = None

    # Common variants in extension builds (e.g., bl_ext.user_default.blender_assistant_mcp)
    if prefs_obj is None:
        try:
            for key in addons.keys():
                kl = key.lower()
                if (
                    "blender_assistant_mcp" in kl
                    or kl.endswith(".blender_assistant_mcp")
                    or kl == "bl_ext.user_default.blender_assistant_mcp"
                ):
                    prefs_obj = addons[key].preferences
                    break
        except Exception:
            prefs_obj = None

    # Fallback: scan for any addon preferences exposing expected fields
    if prefs_obj is None:
        try:
            for key in addons.keys():
                p = addons[key].preferences
                if hasattr(p, "pexels_api_key") or hasattr(p, "unsplash_api_key"):
                    prefs_obj = p
                    break
        except Exception:
            prefs_obj = None

    # Final fallback: return a shim with expected attributes to avoid crashes
    if prefs_obj is None:

        class _PrefsShim:
            pexels_api_key = ""
            unsplash_api_key = ""
            models_folder = ""
            use_external_ollama = False
            external_ollama_url = ""

        prefs_obj = _PrefsShim()

    # 2) Ensure Stock Photo tools are registered even if keys are missing.
    # Runtime checks inside the tools will prevent usage without valid keys.
    try:
        from .tools import tool_registry, stock_photo_tools

        registered = {t.get("name") for t in (tool_registry.get_tools_list() or [])}
        need_search = "search_stock_photos" not in registered
        need_download = "download_stock_photo" not in registered
        need_status = "check_stock_photo_download" not in registered

        if need_search or need_download:
            # Minimal schemas; runtime functions still validate API keys.
            search_schema = {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["unsplash", "pexels"],
                        "description": "Photo source to search",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'wood texture', 'sunset')",
                    },
                    "per_page": {
                        "type": "number",
                        "description": "Number of results (1-80, default: 10)",
                        "default": 10,
                    },
                    "orientation": {
                        "type": "string",
                        "description": "Orientation filter: landscape, portrait, square/squarish",
                    },
                },
                "required": ["source", "query"],
            }

            download_schema = {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["unsplash", "pexels"],
                        "description": "Photo source",
                    },
                    "photo_id": {
                        "type": "string",
                        "description": "Photo ID from search results",
                    },
                    "apply_as_texture": {
                        "type": "boolean",
                        "description": "Apply to active object as texture",
                        "default": True,
                    },
                    "use_background": {
                        "type": "boolean",
                        "description": "Run download in background thread",
                        "default": True,
                    },
                },
                "required": ["source", "photo_id"],
            }

            if need_search:
                tool_registry.register_tool(
                    "search_stock_photos",
                    stock_photo_tools.search_stock_photos,
                    "Search for stock photos (keys required at runtime).",
                    search_schema,
                    category="Stock Photos",
                )

            if need_download:
                tool_registry.register_tool(
                    "download_stock_photo",
                    stock_photo_tools.download_stock_photo,
                    "Download a stock photo by ID (keys required at runtime).",
                    download_schema,
                    category="Stock Photos",
                )

        if need_status and hasattr(stock_photo_tools, "check_download_status"):
            status_schema = {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Job ID returned from download_stock_photo",
                    }
                },
                "required": ["job_id"],
            }
            tool_registry.register_tool(
                "check_stock_photo_download",
                stock_photo_tools.check_download_status,
                "Check the status of a background stock photo download job",
                status_schema,
                category="Stock Photos",
            )
    except Exception as e:
        print(f"[Preferences] Stock tools self-registration failed: {e}")

    return prefs_obj


def get_llm_settings(context) -> dict:
    """Get common LLM settings (context, gpu, thinking) from preferences."""
    prefs = get_preferences() # Use robust getter
    
    # 1. Basic Settings
    settings = {
        "n_ctx": getattr(prefs, "num_ctx", 8192),
        "n_gpu_layers": getattr(prefs, "gpu_layers", -1),
        "n_batch": getattr(prefs, "batch_size", 256),
        "temperature": getattr(prefs, "temperature", 0.7),
        "keep_alive": getattr(prefs, "keep_alive", "5m"),
        "debug_mode": getattr(prefs, "debug_mode", False),
    }

    # 2. Thinking Level Mapping (Enum -> API Value)
    # Maps internal enum to Ollama 'think' parameter (or similar)
    thinking_level = getattr(prefs, "thinking_level", "LOW")
    
    if thinking_level == "OFF":
        # Pass nothing or explicit False depending on adapter
        # For now, we omit 'think' to rely on default or disable it?
        # Ollama without 'think' param simply doesn't enforce it.
        pass
    else:
        # Pass the lower-case value (low, medium, high) as expected by some models/adapters
        settings["think"] = thinking_level.lower()

    return settings


def infer_capabilities_from_name(model_name: str) -> dict:
    """Quickly infer model capabilities from name only (no HTTP request).

    This is much faster than get_model_capabilities() but less accurate.
    Use this for initial model list population.
    """
    name_lower = model_name.lower()

    # Check for vision capability (from name)
    has_vision = (
        "vision" in name_lower
        or "llava" in name_lower
        or "minicpm" in name_lower
        or "moondream" in name_lower
        or "bakllava" in name_lower
    )

    # Check for tool calling capability (from name/family)
    # Most modern models support tools
    tool_capable_indicators = [
        "llama",
        "mistral",
        "qwen",
        "gemma",
        "phi",
        "command",
        "deepseek",
        "mixtral",
        "yi",
        "solar",
        "gpt",
        "claude",
    ]
    has_tools = any(indicator in name_lower for indicator in tool_capable_indicators)

    # Extract size hint from name (e.g., "7b", "13b", "70b")
    import re

    size_match = re.search(r"(\d+\.?\d*[bm])", name_lower)
    size = size_match.group(1).upper() if size_match else ""

    return {
        "tools": has_tools,
        "vision": has_vision,
        "family": "",  # Not available without HTTP request
        "size": size,
    }


# Removed get_model_capabilities() - no longer needed
# We use Ollama's API to get model information

# Global download state (shared across all download operators)
_download_state = {
    "active": False,
    "thread": None,
    "progress": 0.0,
    "status": "",
    "error": None,
    "display_name": "",
}

def get_ollama_models():
    """Get list of installed Ollama models.

    Returns:
        List of model dicts with name and capabilities
    """
    global _ollama_models_cache
    from . import ollama_subprocess

    try:
        ollama = ollama_subprocess.get_ollama()

        if not ollama.is_running():
            print("[Info] Ollama server not running")
            return []

        # Get models from Ollama API
        models = ollama.list_models()

        if not models:
            print("[Info] No Ollama models found")
            return []

        # Format model list
        model_list = []
        for model in models:
            model_name = model.get("name", "")

            # Infer capabilities from model name
            caps = infer_capabilities_from_name(model_name)

            model_list.append(
                {
                    "name": model_name,
                    "capabilities": caps,
                    "size": model.get("size", 0),
                    "modified": model.get("modified_at", ""),
                }
            )

        _ollama_models_cache = model_list
        return model_list

    except Exception as e:
        print(f"Failed to get Ollama models: {e}")
        return []


def get_ollama_models_enum(self, context):
    """Dynamic enum callback for Ollama model selection.

    Note: This function must NOT modify any properties or it will crash Blender.
    It should only return the list of available enum items.
    """
    global _ollama_models_enum_items

    # Return cached items if available
    if _ollama_models_enum_items:
        return _ollama_models_enum_items

    # Default fallback
    return [("NONE", "Click Refresh Models", "Refresh to see installed Ollama models")]


def format_model_description(model_info: dict) -> str:
    """Format model description with capabilities."""
    name = model_info.get("name", "")
    caps = model_info.get("capabilities", {})

    # Build capability badges
    badges = []
    if caps.get("tools"):
        badges.append("🔧 Tools")
    if caps.get("vision"):
        badges.append("👁 Vision")

    # Add size if available
    size = caps.get("size", "")
    if size:
        badges.append(f"📦 {size}")

    # Combine
    if badges:
        return f"{name} - {' | '.join(badges)}"
    else:
        return name


def _sync_tools_to_json(prefs):
    """Sync tool_config_items checkboxes to schema_tools JSON."""
    import json

    enabled = [t.name for t in prefs.tool_config_items if t.expose_mcp]
    # Always ensure execute_code is included
    if "execute_code" not in enabled:
        enabled.append("execute_code")
    prefs.schema_tools = json.dumps(enabled)


def _on_models_folder_update(self, context):
    import os

    import bpy

    from . import ollama_subprocess

    def _norm_external_url(u: str) -> str:
        try:
            s = (u or "").strip()
            if not s:
                return ""
            # Strip trailing /api or /api/ if present
            if s.endswith("/api") or s.endswith("/api/"):
                s = s[: s.rfind("/api")]
            # Remove trailing slashes
            while s.endswith("/"):
                s = s[:-1]
            # Ensure scheme
            if not (
                s.lower().startswith("http://") or s.lower().startswith("https://")
            ):
                s = "http://" + s
            return s
        except Exception:
            return u

    try:
        # Update environment for the subprocess (models folder override)

        if getattr(self, "models_folder", ""):
            os.environ["OLLAMA_MODELS"] = self.models_folder

        else:
            # Clear override to fall back to default

            os.environ.pop("OLLAMA_MODELS", None)

        # Normalize and persist external Ollama URL (idempotent)
        try:
            norm = _norm_external_url(getattr(self, "external_ollama_url", ""))
            if norm and norm != self.external_ollama_url:
                self.external_ollama_url = norm
        except Exception:
            pass

        # Apply server mode based on external toggle
        if getattr(self, "use_external_ollama", False):
            # Ensure embedded server is stopped; do not start bundled server
            ollama_subprocess.stop_ollama()

            # Touch/get instance so it picks up latest prefs; external base_url is applied in get_ollama()
            _ = ollama_subprocess.get_ollama()
            started = False
        else:
            # Restart embedded Ollama so it picks up the current settings
            ollama_subprocess.stop_ollama()
            started = ollama_subprocess.start_ollama()

        # Refresh the models list (enum) without requiring a Blender restart

        # Always try to refresh; external mode will query the external server.
        try:
            bpy.ops.assistant.refresh_models()

        except Exception:
            pass

        # Redraw Preferences UI so changes are visible immediately

        try:
            for area in context.screen.areas:
                if area.type == "PREFERENCES":
                    area.tag_redraw()

        except Exception:
            pass

    except Exception as e:
        print(f"[Assistant] models_folder update failed: {e}")


class ASSISTANT_OT_refresh_models(bpy.types.Operator):
    """Refresh the list of installed Ollama models"""

    bl_idname = "assistant.refresh_models"
    bl_label = "Refresh Models"
    bl_options = {"REGISTER"}

    def execute(self, context):
        global _ollama_models_enum_items, _ollama_models_cache
        prefs = context.preferences.addons[__package__].preferences

        try:
            # Check if Ollama is running first
            from . import ollama_subprocess

            ollama = ollama_subprocess.get_ollama()

            if not ollama.is_running():
                self.report(
                    {"WARNING"},
                    "Ollama server is not running. Click 'Start Ollama' first.",
                )
                return {"CANCELLED"}

            self.report({"INFO"}, "Scanning for models...")

            # Get Ollama models
            models = get_ollama_models()
            if models:
                # Store ALL models in cache (including embedding models for detection)
                _ollama_models_cache = models

                # Format each model with capabilities (exclude embedding models from dropdown)
                _ollama_models_enum_items = [
                    (
                        m["name"],  # identifier (model name like "qwen2.5-coder:7b")
                        m["name"],  # display name
                        format_model_description(m),  # description with capabilities
                    )
                    for m in models
                    if "embed"
                    not in m["name"].lower()  # Skip embedding models in dropdown
                ]
                self.report(
                    {"INFO"}, f"Found {len(_ollama_models_enum_items)} chat models"
                )
                
                # Sanitize active model selection to prevent RNA warnings
                valid_ids = {item[0] for item in _ollama_models_enum_items}
                if prefs.model_file not in valid_ids and _ollama_models_enum_items:
                    print(f"[Assistant] 'model_file' had invalid value '{prefs.model_file}', resetting to '{_ollama_models_enum_items[0][0]}'")
                    prefs.model_file = _ollama_models_enum_items[0][0]

            else:
                _ollama_models_cache = []
                _ollama_models_enum_items = [
                    ("NONE", "No models found", "Use 'ollama pull' to download models")
                ]
                prefs.model_file = "NONE"
                self.report(
                    {"WARNING"},
                    "No Ollama models found. Use 'ollama pull <model>' to download.",
                )

            # Force UI redraw to update download buttons
            for area in context.screen.areas:
                if area.type == "PREFERENCES":
                    area.tag_redraw()

            return {"FINISHED"}

        except Exception as e:
            self.report({"ERROR"}, f"Failed to refresh models: {str(e)}")
            print(f"[Assistant] Error refreshing models: {e}")
            import traceback

            traceback.print_exc()
            return {"CANCELLED"}


class ASSISTANT_OT_search_ollama_library(bpy.types.Operator):
    """Search the Ollama online library (non-blocking)"""

    bl_idname = "assistant.search_ollama_library"
    bl_label = "Search Ollama Library"
    bl_options = {"REGISTER"}

    query: bpy.props.StringProperty(name="Query", default="")

    _timer = None

    def modal(self, context, event):
        global _library_search_state
        if event.type == "TIMER":
            # Update UI
            for area in context.screen.areas:
                if area.type == "PREFERENCES":
                    area.tag_redraw()

            if (
                _library_search_state["thread"]
                and not _library_search_state["thread"].is_alive()
            ):
                self.cancel(context)
                if _library_search_state["error"]:
                    self.report(
                        {"ERROR"}, f"Search failed: {_library_search_state['error']}"
                    )
                    _library_search_state["active"] = False
                    return {"CANCELLED"}
                else:
                    self.report(
                        {"INFO"},
                        f"Found {len(_library_search_state['results'])} models",
                    )
                    _library_search_state["active"] = False
                    return {"FINISHED"}
        return {"PASS_THROUGH"}

    def execute(self, context):
        global _library_search_state
        import re
        import threading
        import urllib.parse
        import urllib.request

        if _library_search_state["active"]:
            self.report({"WARNING"}, "Search already in progress")
            return {"CANCELLED"}

        def fetch():
            try:
                q = (self.query or "").strip()
                _library_search_state["status"] = f"Searching '{q or 'popular'}'..."
                # Choose URL: try search endpoint, fallback to library
                urls = []
                if q:
                    urls.append(f"https://ollama.com/search?q={urllib.parse.quote(q)}")
                urls.append("https://ollama.com/library")

                html = ""
                for url in urls:
                    try:
                        req = urllib.request.Request(
                            url, headers={"User-Agent": "BlenderAssistant/1.0"}
                        )
                        with urllib.request.urlopen(req, timeout=15) as resp:
                            html = resp.read().decode("utf-8", errors="ignore")
                            if html:
                                break
                    except Exception:
                        continue

                if not html:
                    _library_search_state["error"] = "No response from ollama.com"
                    return

                # Extract model slugs from links like /library/<name>
                names = re.findall(r'href="/library/([a-zA-Z0-9_.:-]+)"', html)
                # Deduplicate and filter obvious non-models
                unique = []
                seen = set()
                for n in names:
                    if n and n not in seen:
                        seen.add(n)
                        unique.append(n)
                # Limit list length
                _library_search_state["results"] = unique[:50]
                _library_search_state["status"] = (
                    f"Found {len(_library_search_state['results'])} models"
                )
            except Exception as e:
                _library_search_state["error"] = str(e)

        _library_search_state["active"] = True
        _library_search_state["error"] = None
        _library_search_state["results"] = []
        _library_search_state["status"] = "Starting search..."
        _library_search_state["thread"] = threading.Thread(target=fetch)
        _library_search_state["thread"].start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)


class ASSISTANT_OT_pull_model(bpy.types.Operator):
    """Download a model using Ollama"""

    bl_idname = "assistant.pull_model"
    bl_label = "Pull Model"
    bl_options = {"REGISTER"}

    model_name: bpy.props.StringProperty(name="Model Name")
    display_name: bpy.props.StringProperty(name="Display Name")

    _timer = None

    def modal(self, context, event):
        global _download_state

        if event.type == "TIMER":
            # Update UI
            for area in context.screen.areas:
                if area.type == "PREFERENCES":
                    area.tag_redraw()

            # Check if download complete
            if _download_state["thread"] and not _download_state["thread"].is_alive():
                self.cancel(context)

                if _download_state["error"]:
                    self.report({"ERROR"}, f"Pull failed: {_download_state['error']}")
                    _download_state["active"] = False
                    _download_state["status"] = ""
                    return {"CANCELLED"}
                else:
                    self.report({"INFO"}, f"Pulled {_download_state['display_name']}")
                    _download_state["active"] = False
                    _download_state["status"] = ""
                    # Refresh model list
                    bpy.ops.assistant.refresh_models()
                    return {"FINISHED"}

        return {"PASS_THROUGH"}

    def execute(self, context):
        global _download_state
        import threading

        from . import ollama_subprocess

        # Check if download already in progress
        if _download_state["active"]:
            self.report({"WARNING"}, "Download already in progress. Please wait.")
            return {"CANCELLED"}

        # Pull function
        def pull():
            try:
                ollama = ollama_subprocess.get_ollama()

                if not ollama.is_running():
                    _download_state["error"] = "Ollama server is not running"
                    return

                # Call Ollama pull API
                import json
                import urllib.request

                url = f"{ollama.base_url}/api/pull"
                data = json.dumps({"name": self.model_name}).encode("utf-8")

                req = urllib.request.Request(
                    url, data=data, headers={"Content-Type": "application/json"}
                )

                with urllib.request.urlopen(req, timeout=300) as response:
                    # Stream the response to get progress updates
                    for line in response:
                        if line:
                            try:
                                status = json.loads(line.decode("utf-8"))

                                # Update progress
                                if "completed" in status and "total" in status:
                                    completed = status["completed"]
                                    total = status["total"]
                                    if total > 0:
                                        _download_state["progress"] = completed / total
                                        completed_mb = completed / (1024 * 1024)
                                        total_mb = total / (1024 * 1024)
                                        _download_state["status"] = (
                                            f"Pulling {self.display_name}: "
                                            f"{completed_mb:.1f} / {total_mb:.1f} MB "
                                            f"({_download_state['progress'] * 100:.0f}%)"
                                        )

                                # Check for completion
                                if status.get("status") == "success":
                                    _download_state["status"] = (
                                        f"Completed: {self.display_name}"
                                    )
                                    break

                            except json.JSONDecodeError:
                                pass

            except Exception as e:
                _download_state["error"] = str(e)

        # Mark download as active
        _download_state["active"] = True
        _download_state["error"] = None
        _download_state["display_name"] = self.display_name
        _download_state["status"] = f"Starting pull: {self.display_name}..."

        # Start download in background thread
        _download_state["thread"] = threading.Thread(target=pull)
        _download_state["thread"].start()

        # Setup modal timer
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        wm.modal_handler_add(self)

        self.report({"INFO"}, f"Starting pull: {self.display_name}")
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)


class ASSISTANT_OT_start_ollama(bpy.types.Operator):
    """Start the Ollama server"""

    bl_idname = "assistant.start_ollama"
    bl_label = "Start Ollama Server"
    bl_options = {"REGISTER"}

    def execute(self, context):
        from . import ollama_subprocess

        ollama = ollama_subprocess.get_ollama()

        if ollama.is_running():
            self.report({"INFO"}, "Ollama server is already running")
            return {"CANCELLED"}

        success = ollama.start()

        if success:
            self.report({"INFO"}, "Ollama server started successfully")
            return {"FINISHED"}
        else:
            self.report({"ERROR"}, "Failed to start Ollama server")
            return {"CANCELLED"}


class ASSISTANT_OT_stop_ollama(bpy.types.Operator):
    """Stop the Ollama server"""

    bl_idname = "assistant.stop_ollama"
    bl_label = "Stop Ollama Server"
    bl_options = {"REGISTER"}

    def execute(self, context):
        from . import ollama_subprocess

        ollama = ollama_subprocess.get_ollama()

        if not ollama.is_running():
            self.report({"INFO"}, "Ollama server is not running")
            return {"CANCELLED"}

        success = ollama.stop()

        if success:
            self.report({"INFO"}, "Ollama server stopped")
            return {"FINISHED"}
        else:
            self.report({"ERROR"}, "Ollama server hopefully stopped")
            return {"CANCELLED"}


class ASSISTANT_OT_open_ollama_folder(bpy.types.Operator):
    # Online browse/search query

    """Open the Ollama models folder in file explorer"""

    bl_idname = "assistant.open_ollama_folder"
    bl_label = "Open Ollama Models Folder"
    bl_options = {"REGISTER"}

    def execute(self, context):
        import platform
        import subprocess
        from pathlib import Path

        # Get models directory from preferences
        prefs = context.preferences.addons[__package__].preferences

        if not prefs.models_folder:
            self.report({"ERROR"}, "Models folder not set in preferences")
            return {"CANCELLED"}

        models_dir = Path(prefs.models_folder)

        # Create directory if it doesn't exist
        models_dir.mkdir(parents=True, exist_ok=True)

        # Open in file explorer
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["explorer", str(models_dir)])
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", str(models_dir)])
            else:  # Linux
                subprocess.Popen(["xdg-open", str(models_dir)])

            self.report({"INFO"}, f"Opened: {models_dir}")
            return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"Failed to open folder: {str(e)}")
            return {"CANCELLED"}


class ASSISTANT_OT_delete_model(bpy.types.Operator):
    """Delete a model using Ollama"""

    bl_idname = "assistant.delete_model"
    bl_label = "Delete Model"
    bl_options = {"REGISTER"}

    model_name: bpy.props.StringProperty(name="Model Name")

    def invoke(self, context, event):
        return context.window_manager.invoke_confirm(self, event)

    def draw(self, context):
        layout = self.layout
        layout.label(text=f"Delete '{self.model_name}'?")
        layout.label(text="This will remove the model from Ollama.")

    def execute(self, context):
        import json
        import urllib.request

        from . import ollama_subprocess

        try:
            ollama = ollama_subprocess.get_ollama()

            if not ollama.is_running():
                self.report({"ERROR"}, "Ollama server is not running")
                return {"CANCELLED"}

            # Call Ollama delete API
            url = f"{ollama.base_url}/api/delete"
            data = json.dumps({"name": self.model_name}).encode("utf-8")

            req = urllib.request.Request(
                url, data=data, headers={"Content-Type": "application/json"}
            )
            req.get_method = lambda: "DELETE"

            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    self.report({"INFO"}, f"Deleted model: {self.model_name}")
                    # Refresh model list
                    bpy.ops.assistant.refresh_models()
                    return {"FINISHED"}
                else:
                    self.report({"ERROR"}, f"Failed to delete: HTTP {response.status}")
                    return {"CANCELLED"}

        except Exception as e:
            self.report({"ERROR"}, f"Failed to delete: {str(e)}")
            return {"CANCELLED"}



def _update_tool_enabled(self, context):
    """Callback when a tool is toggled."""
    # Find prefs
    try:
        prefs = context.preferences.addons[__package__].preferences
    except Exception:
        return
    _sync_tools_to_json(prefs)


class ToolConfigItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Tool Name")
    expose_mcp: bpy.props.BoolProperty(
        name="Expose MCP",
        default=True,
        description="Expose this tool to the LLM as a native MCP tool (JSON schema). If disabled, the agent must use the SDK (Python).",
        update=_update_tool_enabled,
    )
    category: bpy.props.StringProperty(name="Category")
    description: bpy.props.StringProperty(name="Description")



class ASSISTANT_OT_toggle_model_capability(bpy.types.Operator):
    bl_idname = "assistant.toggle_model_capability"
    bl_label = "Toggle Model Capability"

    model_name: bpy.props.StringProperty(name="Model Name")
    flag: bpy.props.EnumProperty(
        name="Capability",
        items=[
            ("tools", "Tools", ""),
            ("thinking", "Thinking", ""),
            ("vision", "Vision", ""),
        ],
    )
    value: bpy.props.BoolProperty(name="Value", default=True)

    def execute(self, context):
        import json as _json

        prefs = context.preferences.addons[__package__].preferences
        # Load current overrides
        try:
            data = (prefs.model_overrides or "").strip()
            overrides = _json.loads(data) if data else {}
        except Exception:
            overrides = {}

        # Apply toggle
        entry = dict(overrides.get(self.model_name) or {})
        entry[self.flag] = bool(self.value)
        overrides[self.model_name] = entry

        # Save back
        prefs.model_overrides = _json.dumps(overrides, separators=(",", ":"))

        # Refresh UI if possible
        try:
            if getattr(context, "area", None):
                context.area.tag_redraw()
        except Exception:
            pass

        return {"FINISHED"}


class AssistantPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    models_folder: bpy.props.StringProperty(
        name="Models Folder",
        description="Custom folder to store GGUF models (leave empty for default)",
        default="",
        subtype="DIR_PATH",
        update=_on_models_folder_update,
    )
    # External Ollama settings
    use_external_ollama: bpy.props.BoolProperty(
        name="Use External Ollama",
        description="If enabled, do not start the bundled Ollama server; use the URL below",
        default=False,
        update=_on_models_folder_update,
    )
    external_ollama_url: bpy.props.StringProperty(
        name="External Ollama URL",
        description="Base URL of an existing Ollama server (e.g., http://127.0.0.1:11434)",
        default="http://127.0.0.1:11434",
        update=_on_models_folder_update,
    )

    model_file: bpy.props.EnumProperty(
        name="Model",
        description="Select an Ollama model",
        items=get_ollama_models_enum,
    )
    vision_model: bpy.props.EnumProperty(
        name="Vision Model",
        description="Vision-capable model to use for viewport analysis (e.g., minicpm-v:8b)",
        items=lambda self, context: [
            e
            for e in get_ollama_models_enum(self, context)
            if self._is_vision_enabled(e[0])
        ],
    )

    custom_model_name: bpy.props.StringProperty(
        name="Custom",
        description="Enter any Ollama model name to pull (e.g., gemma2:27b, mistral:7b, gpt-oss:20b)",
        default="",
    )
    # Per-model capability overrides; JSON mapping:
    # { "model:name:tag": {"tools": bool, "thinking": bool, "vision": bool}, ... }
    model_overrides: bpy.props.StringProperty(
        name="Model Capabilities (JSON)",
        description="Per-model overrides for Tools, Thinking, and Vision (advanced)",
        default="{}",
    )

    use_rag: bpy.props.BoolProperty(
        name="Enable RAG (Documentation Search)",
        description="Use Retrieval-Augmented Generation to search Blender documentation and provide more accurate answers",
        default=True,
    )

    # Online browse/search query
    library_query: bpy.props.StringProperty(
        name="Search", description="Search the Ollama online library", default=""
    )

    rag_num_results: bpy.props.IntProperty(
        name="RAG Context Documents",
        description="Number of documentation chunks to retrieve for context (small sections of pages, not full pages)",
        default=5,
        min=1,
        max=20,
    )

    # RAG context selection mode
    rag_context_mode: bpy.props.EnumProperty(
        name="RAG Context Mode",
        description="How many context chunks to include with each augmented message",
        items=[
            ("FIXED", "Fixed", "Use exact number of chunks specified below"),
            (
                "AUTO",
                "Auto",
                "Scale number of chunks based on query complexity and token budget",
            ),
        ],
        default="AUTO",
    )

    rag_auto_min: bpy.props.IntProperty(
        name="Auto Min Chunks",
        description="Minimum chunks when in Auto mode",
        default=6,
        min=1,
        max=50,
    )

    rag_auto_max: bpy.props.IntProperty(
        name="Auto Max Chunks",
        description="Maximum chunks when in Auto mode",
        default=12,
        min=1,
        max=100,
    )

    rag_follow_up_augmentation: bpy.props.BoolProperty(
        name="Augment Follow-ups",
        description="Also auto-augment some follow-up user messages (e.g., when executing code or referencing bpy.*)",
        default=True,
    )

    rag_source_bias: bpy.props.EnumProperty(
        name="Source Bias",
        description="Prefer API vs Manual sources during retrieval",
        items=[
            (
                "AUTO",
                "Auto",
                "Detect from message keywords (e.g., bpy., script, code → API)",
            ),
            ("API_ONLY", "API Only", "Only retrieve from Blender Python API docs"),
            ("MANUAL_ONLY", "Manual Only", "Only retrieve from Blender Manual"),
            ("BOTH", "Both", "Consider both sources equally"),
        ],
        default="AUTO",
    )

    temperature: bpy.props.FloatProperty(
        name="Temperature",
        description="LLM temperature (higher = more creative, lower = more focused)",
        default=0.2,
        min=0.0,
        max=1.5,
    )

    thinking_level: bpy.props.EnumProperty(
        name="Thinking Level",
        description=(
            "Control Ollama 'think' reasoning. For GPT-OSS, choose Low/Medium/High. "
            "'Off' disables thinking for most models; for GPT-OSS it maps to Low."
        ),
        items=[
            ("OFF", "Off", "Disable thinking (GPT-OSS maps to Low)"),
            ("LOW", "Low", "Short reasoning"),
            ("MEDIUM", "Medium", "More reasoning"),
            ("HIGH", "High", "Detailed reasoning"),
        ],
        default="LOW",
    )

    enforce_json: bpy.props.BoolProperty(
        name="Enforce JSON Output",
        description="Force the Agent to output strictly JSON. Uncheck to allow Thinking/Text (experimental)",
        default=False
    )

    keep_alive: bpy.props.EnumProperty(
        name="Keep Model Loaded",
        description="How long to keep the model loaded in memory after a request",
        items=[
            ("5m", "5 Minutes", "Keep loaded for 5 minutes (default)"),
            ("15m", "15 Minutes", "Keep loaded for 15 minutes"),
            ("30m", "30 Minutes", "Keep loaded for 30 minutes"),
            ("60m", "1 Hour", "Keep loaded for 1 hour"),
            ("-1", "Forever", "Keep loaded until Blender closes or model is changed"),
        ],
        default="5m",
    )

    # Removed: lean_respect_tool_selector (Lean/API‑Lean now always use curated tool lists)
    # Removed: auto_scene_snapshot, snapshot_max_objects, planning_exploration, planning_temperature, max_iterations (Unused)

    # GPU Settings
    gpu_layers: bpy.props.IntProperty(
        name="GPU Layers",
        description="Number of model layers to load to GPU (999 = all layers, 0 = CPU only)",
        default=999,
        min=0,
        max=999,
    )

    num_ctx: bpy.props.IntProperty(
        name="Context Length",
        description="Maximum context window size in tokens",
        default=131072,
        min=2048,
        max=131072,
    )

    batch_size: bpy.props.IntProperty(
        name="Batch Size",
        description="Batch size for prompt processing (higher = faster, more VRAM)",
        default=512,
        min=128,
        max=2048,
    )

    # Advanced Sampling Options
    show_advanced_sampling: bpy.props.BoolProperty(
        name="Show Advanced Sampling",
        description="Show/hide advanced sampling options",
        default=False,
    )

    show_section_ui: bpy.props.BoolProperty(
        name="Show UI Section",
        default=True,
    )

    # UI / Visibility Options
    show_thinking: bpy.props.BoolProperty(
        name="Show Thinking",
        description="Show/hide the internal thought process of the agent",
        default=True,
    )

    show_system_updates: bpy.props.BoolProperty(
        name="Show System Updates",
        description="Show/hide system-level notifications (e.g. scene updates, hidden tool outputs) in the chat",
        default=False,
    )

    show_tool_outputs: bpy.props.BoolProperty(
        name="Show Tool/Agent Outputs",
        description="Show/hide the raw output from tools and sub-agents",
        default=True,
    )

    top_p: bpy.props.FloatProperty(
        name="Top P",
        description="Nucleus sampling - consider tokens with cumulative probability up to this value",
        default=0.9,
        min=0.0,
        max=1.0,
    )

    top_k: bpy.props.IntProperty(
        name="Top K",
        description="Sample from top K most likely tokens (0 = disabled)",
        default=40,
        min=0,
        max=100,
    )

    repeat_penalty: bpy.props.FloatProperty(
        name="Repeat Penalty",
        description="Penalize repetition in generated text (1.0 = no penalty)",
        default=1.1,
        min=0.0,
        max=2.0,
    )

    use_seed: bpy.props.BoolProperty(
        name="Use Fixed Seed",
        description="Use a fixed seed for reproducible outputs",
        default=False,
    )

    seed_value: bpy.props.IntProperty(
        name="Seed",
        description="Random seed for reproducible generation",
        default=42,
        min=0,
        max=2147483647,
    )

    # Performance Options
    low_vram_mode: bpy.props.BoolProperty(
        name="Low VRAM Mode",
        description="Reduce VRAM usage (slower but uses less memory)",
        default=False,
    )

    num_threads: bpy.props.IntProperty(
        name="CPU Threads",
        description="Number of CPU threads to use",
        default=8,
        min=1,
        max=64,
    )

    # Stock photo API keys
    unsplash_api_key: bpy.props.StringProperty(
        name="Unsplash API Key",
        description="API key for Unsplash (get free at https://unsplash.com/developers)",
        default="",
        subtype="PASSWORD",
    )

    pexels_api_key: bpy.props.StringProperty(
        name="Pexels API Key",
        description="API key for Pexels (get free at https://www.pexels.com/api/)",
        default="",
        subtype="PASSWORD",
    )

    show_ollama_config: bpy.props.BoolProperty(
        name="Show Ollama Configuration",
        description="Show/hide Ollama configuration section",
        default=False,  # Collapsed by default to save space
    )

    download_progress: bpy.props.FloatProperty(
        name="Download Progress",
        description="Current download progress",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype="FACTOR",
        get=lambda self: _download_state["progress"],
    )

    @property
    def ollama_model(self):
        """Return the active model filename (for backward compatibility)."""
        if self.model_file != "NONE":
            return self.model_file
        return ""

    @property
    def ollama_model_enum(self):
        """Alias for backward compatibility."""
        return self.model_file

    def _get_installed_models(self):
        """Get set of installed Ollama model names.
        Includes both full tag and base name (without tag) so UI buttons reflect installs correctly.
        """
        global _ollama_models_cache
        names = set()
        for m in _ollama_models_cache:
            name = m.get("name", "")
            if not name:
                continue
            names.add(name)
            names.add(name.split(":")[0])
        return names

    def _get_model_capabilities(self, model_name):
        """Get capabilities for a specific model."""
        global _ollama_models_cache
        for m in _ollama_models_cache:
            if m["name"] == model_name:
                return m.get("capabilities", {})
        return {}

    # ---- Per-model capability overrides helpers ----
    def _get_model_overrides_dict(self) -> dict:
        """Parse stored JSON overrides into a dict safely."""
        try:
            import json as _json

            data = getattr(self, "model_overrides", "") or ""
            return _json.loads(data) if isinstance(data, str) and data.strip() else {}
        except Exception:
            return {}

    def _get_default_flags_for_model(self, model_name: str) -> dict:
        """Heuristic defaults when no override exists."""
        name = (model_name or "").lower()
        defaults = {"tools": False, "thinking": False, "vision": False}
        # Chat default: GPT-OSS -> tools+thinking
        if name.startswith("gpt-oss"):
            defaults["tools"] = True
            defaults["thinking"] = True
        # Vision small models (commonly used)
        if any(
            v in name for v in ("minicpm-v", "llava", "moondream", "vision", "bakllava")
        ):
            defaults["vision"] = True
        # RAG embedding model: no checkboxes, defaults remain False
        if name == "nomic-embed-text":
            defaults["tools"] = False
            defaults["thinking"] = False
            defaults["vision"] = False
        return defaults

    def _is_flag_enabled(self, model_name: str, key: str) -> bool:
        """Return True if a capability flag is enabled for a model (override or default)."""
        overrides = self._get_model_overrides_dict()
        if isinstance(overrides.get(model_name), dict) and key in overrides[model_name]:
            return bool(overrides[model_name][key])
        return bool(self._get_default_flags_for_model(model_name).get(key, False))

    def _is_vision_enabled(self, model_name: str) -> bool:
        """True if model should appear in vision enum."""
        # Hide RAG/embedding model from vision list entirely
        if (model_name or "").split(":")[0] == "nomic-embed-text":
            return False
        return self._is_flag_enabled(model_name, "vision")

    def _is_tools_enabled(self, model_name: str) -> bool:
        """True if MCP tools should be enabled for the model."""
        return self._is_flag_enabled(model_name, "tools")

    def _is_thinking_enabled(self, model_name: str) -> bool:
        """True if 'think' should be enabled for the model."""
        return self._is_flag_enabled(model_name, "thinking")

    # ---- Per-model capability overrides helpers ----
    def _get_model_overrides_dict(self) -> dict:
        """Parse stored JSON overrides into a dict safely."""
        try:
            import json as _json

            data = getattr(self, "model_overrides", "") or ""
            return _json.loads(data) if isinstance(data, str) and data.strip() else {}
        except Exception:
            return {}

    def _get_default_flags_for_model(self, model_name: str) -> dict:
        """Heuristic defaults when no override exists."""
        name = (model_name or "").lower()
        defaults = {"tools": False, "thinking": False, "vision": False}
        # Defaults: GPT-OSS -> tools+thinking
        if name.startswith("gpt-oss"):
            defaults["tools"] = True
            defaults["thinking"] = True
        # Small/common vision models -> vision
        if any(
            v in name for v in ("minicpm-v", "llava", "moondream", "vision", "bakllava")
        ):
            defaults["vision"] = True
        # RAG/embedding model: no capabilities
        if name.split(":")[0] == "nomic-embed-text":
            defaults["tools"] = False
            defaults["thinking"] = False
            defaults["vision"] = False
        return defaults

    def _is_flag_enabled(self, model_name: str, key: str) -> bool:
        """Return True if a capability flag is enabled for a model (override or default)."""
        overrides = self._get_model_overrides_dict()
        if isinstance(overrides.get(model_name), dict) and key in overrides[model_name]:
            return bool(overrides[model_name][key])
        return bool(self._get_default_flags_for_model(model_name).get(key, False))

    def _is_vision_enabled(self, model_name: str) -> bool:
        """True if model should appear in the Vision model enum."""
        # Hide RAG/embedding model from vision list entirely
        if (model_name or "").split(":")[0] == "nomic-embed-text":
            return False
        return self._is_flag_enabled(model_name, "vision")

    def _is_tools_enabled(self, model_name: str) -> bool:
        """True if MCP tools should be enabled for the model."""
        return self._is_flag_enabled(model_name, "tools")

    def _is_thinking_enabled(self, model_name: str) -> bool:
        """True if 'think' should be enabled for the model."""
        return self._is_flag_enabled(model_name, "thinking")

    def _draw_model_management(self, layout):
        """Draw model management section."""

        box = layout.box()
        col = box.column(align=True)
        # External Ollama controls
        col.prop(self, "use_external_ollama", text="Use External Ollama")
        col.prop(self, "external_ollama_url", text="External Ollama URL")
        col.separator()
        box.label(text="Model Management", icon="PREFERENCES")

        # Custom models folder path

        col.prop(self, "models_folder", text="Models Folder")

        col.label(text="Leave empty to use default extension folder", icon="INFO")
        col.separator()

        # Model selection dropdown

        row = col.row(align=True)

        row.prop(self, "model_file", text="Active Model")

        row.operator("assistant.refresh_models", text="", icon="FILE_REFRESH")

        # Vision model selector

        vrow = col.row(align=True)

        vrow.prop(self, "vision_model", text="Vision Model")

        # Model actions

        action_row = col.row(align=True)

        action_row.operator(
            "assistant.open_ollama_folder", text="Open Folder", icon="FILE_FOLDER"
        )

        # Per-model capabilities (render-only UI below the selection controls)

        col.separator()

        cap_box = col.box()

        cap_box.label(text="Per-model Capabilities", icon="CHECKMARK")

        caps_col = cap_box.column(align=True)

        installed = self._get_installed_models()

        if installed:
            # Group models by family (prefix before ':') and render headers without delete buttons
            grouped = {}
            for m in sorted(installed):
                family = m.split(":")[0]
                if family == "nomic-embed-text":
                    continue
                grouped.setdefault(family, []).append(m)

            for family in sorted(grouped.keys()):
                # Family header (render as category-like row)
                hdr = caps_col.row()
                hdr.label(text=family, icon="PRESET")

                # Render each model in this family with capability toggles and delete
                for name in sorted(grouped[family]):
                    tools_on = self._is_tools_enabled(name)

                    think_on = self._is_thinking_enabled(name)

                    vision_on = self._is_vision_enabled(name)

                    r = caps_col.row(align=True)

                    r.label(text=name)

                    btn_tools = r.operator(
                        "assistant.toggle_model_capability",
                        text="",
                        icon="CHECKBOX_HLT" if tools_on else "CHECKBOX_DEHLT",
                    )

                    if btn_tools:
                        btn_tools.model_name = name

                        btn_tools.flag = "tools"

                        btn_tools.value = not tools_on

                    # Thinking toggle

                    btn_think = r.operator(
                        "assistant.toggle_model_capability",
                        text="",
                        icon="CHECKBOX_HLT" if think_on else "CHECKBOX_DEHLT",
                    )

                    if btn_think:
                        btn_think.model_name = name

                        btn_think.flag = "thinking"

                        btn_think.value = not think_on

                    # Vision toggle

                    btn_vis = r.operator(
                        "assistant.toggle_model_capability",
                        text="",
                        icon="CHECKBOX_HLT" if vision_on else "CHECKBOX_DEHLT",
                    )

                    if btn_vis:
                        btn_vis.model_name = name

                        btn_vis.flag = "vision"

                        btn_vis.value = not vision_on

                    # Per-model delete button

                    del_op = r.operator(
                        "assistant.delete_model",
                        text="",
                        icon="TRASH",
                    )

                    if del_op:
                        del_op.model_name = name

        else:
            caps_col.label(text="No installed models found", icon="INFO")

        # Delete button (only if model selected)

        if self.model_file and self.model_file != "NONE":
            delete_op = action_row.operator(
                "assistant.delete_model", text="Delete", icon="TRASH"
            )

            delete_op.model_name = self.model_file

        # Show capability icons for selected model

        if self.model_file and self.model_file != "NONE":
            caps = self._get_model_capabilities(self.model_file)

            if caps:
                caps_row = col.row(align=True)

                caps_row.label(text="Capabilities:")

                if caps.get("tools"):
                    caps_row.label(text="Tools", icon="TOOL_SETTINGS")

                if caps.get("vision"):
                    caps_row.label(text="Vision", icon="CAMERA_DATA")

                if caps.get("size"):
                    caps_row.label(text=caps["size"])

        col.separator()

    def _draw_model_downloads(self, layout):
        """Draw model download section."""

        global _download_state

        # Browse Ollama Library (Online)
        layout.separator()
        browse_box = layout.box()
        browse_box.label(text="Browse Ollama Library (Online)", icon="URL")
        browse_col = browse_box.column(align=True)
        row = browse_col.row(align=True)
        row.prop(self, "library_query", text="Search")
        if _library_search_state["active"]:
            row.enabled = False
            row.label(
                text=_library_search_state.get("status") or "Searching...",
                icon="SORTTIME",
            )
        else:
            op = row.operator(
                "assistant.search_ollama_library", text="Search", icon="VIEWZOOM"
            )
            op.query = self.library_query

        # Results list
        results = _library_search_state.get("results") or []
        if results:
            browse_col.label(text=f"Results ({len(results)}):", icon="PRESET")
            installed_models = self._get_installed_models()
            max_show = min(len(results), 25)
            for i in range(max_show):
                name = results[i]
                r = browse_col.row(align=True)
                is_installed = (
                    name in installed_models or name.split(":")[0] in installed_models
                )
                if is_installed:
                    r.label(text=f"\u2713 {name}", icon="CHECKMARK")
                    del_op = r.operator("assistant.delete_model", text="", icon="TRASH")
                    del_op.model_name = name
                else:
                    r.label(text=name)
                    if _download_state["active"]:
                        r.label(text="Wait...", icon="TIME")
                    else:
                        pull_op = r.operator(
                            "assistant.pull_model", text="Download", icon="IMPORT"
                        )
                        pull_op.model_name = name
                        pull_op.display_name = name

        box = layout.box()
        box.label(text="Download Models", icon="IMPORT")
        col = box.column(align=True)

        # Show download progress if active
        if _download_state["active"]:
            progress_box = col.box()
            progress_col = progress_box.column(align=True)
            progress_col.label(text=_download_state["status"], icon="SORTTIME")

            # Progress bar
            progress_row = progress_col.row()
            progress_row.scale_y = 1.5
            progress_row.prop(
                self, "download_progress", text="", slider=True, emboss=True
            )
            col.separator()

        installed_models = self._get_installed_models()

        # Chat and Vision core models
        col.label(text="Core Chat/Vision Models:", icon="PRESET")
        core_chat = [
            ("gpt-oss:20b", "gpt-oss 20B", "10GB"),
            ("minicpm-v:8b", "MiniCPM-V 8B", "~6–8GB"),
        ]
        for model_name, display_name, size in core_chat:
            model_row = col.row(align=True)

            is_installed = model_name in installed_models

            if is_installed:
                model_row.label(text=f"✓ {display_name}", icon="CHECKMARK")
                model_row.label(text=size)

                delete_op = model_row.operator(
                    "assistant.delete_model", text="", icon="TRASH"
                )
                delete_op.model_name = model_name

            else:
                model_row.label(text=display_name, icon="NONE")
                model_row.label(text=size)

                if _download_state["active"]:
                    model_row.label(text="Wait...", icon="TIME")
                else:
                    pull_op = model_row.operator(
                        "assistant.pull_model", text="Download", icon="IMPORT"
                    )
                    pull_op.model_name = model_name

                    pull_op.display_name = display_name

        col.separator()

        # Embedding model (for RAG)
        col.label(text="Embedding Model (for RAG):", icon="PRESET")
        embed_models = [
            ("nomic-embed-text", "Nomic Embed Text", "~130MB"),
        ]
        for model_name, display_name, size in embed_models:
            model_row = col.row(align=True)

            is_installed = model_name in installed_models

            if is_installed:
                model_row.label(text=f"✓ {display_name}", icon="CHECKMARK")
                model_row.label(text=size)

                delete_op = model_row.operator(
                    "assistant.delete_model", text="", icon="TRASH"
                )
                delete_op.model_name = model_name

            else:
                model_row.label(text=display_name, icon="NONE")
                model_row.label(text=size)

                if _download_state["active"]:
                    model_row.label(text="Wait...", icon="TIME")
                else:
                    pull_op = model_row.operator(
                        "assistant.pull_model", text="Download", icon="IMPORT"
                    )
                    pull_op.model_name = model_name

                    pull_op.display_name = display_name

        col.separator()

        # Custom model pull/delete
        col.label(text="Custom Model:", icon="PRESET")
        custom_row = col.row(align=True)
        custom_row.prop(self, "custom_model_name", text="")

        if self.custom_model_name.strip():
            custom_name = self.custom_model_name.strip()
            is_custom_installed = (
                custom_name in installed_models
                or custom_name.split(":")[0] in installed_models
            )
            if is_custom_installed:
                del_op = custom_row.operator(
                    "assistant.delete_model", text="Delete", icon="TRASH"
                )
                del_op.model_name = custom_name
            else:
                if _download_state["active"]:
                    custom_row.label(text="Wait...", icon="TIME")
                else:
                    pull_op = custom_row.operator(
                        "assistant.pull_model", text="Download", icon="IMPORT"
                    )
                    pull_op.model_name = custom_name
                    pull_op.display_name = custom_name
        else:
            custom_row.enabled = True

    def _draw_gpu_settings(self, layout):
        """Draw GPU optimization settings."""
        gpu_box = layout.box()
        gpu_box.label(text="GPU Optimization", icon="SHADING_RENDERED")
        gpu_col = gpu_box.column(align=True)

        # GPU settings
        gpu_col.prop(self, "gpu_layers")

        gpu_col.prop(self, "num_ctx")
        gpu_col.prop(self, "batch_size")

        # Performance options
        gpu_col.separator()
        gpu_col.prop(self, "low_vram_mode")
        gpu_col.prop(self, "num_threads")

    def _draw_generation_settings(self, layout):
        """Draw generation settings."""
        settings_box = layout.box()
        settings_box.label(text="Generation Settings", icon="SETTINGS")
        settings_col = settings_box.column(align=True)

        # Basic settings
        settings_col.prop(self, "temperature")

        # Thinking / execution controls
        settings_col.prop(self, "thinking_level")
        settings_col.prop(self, "enforce_json")
        settings_col.prop(self, "keep_alive")

        # Advanced sampling (collapsible)
        settings_col.separator()
        settings_col.prop(
            self,
            "show_advanced_sampling",
            icon="TRIA_DOWN" if self.show_advanced_sampling else "TRIA_RIGHT",
            emboss=False,
        )

        if self.show_advanced_sampling:
            adv_box = settings_col.box()
            adv_col = adv_box.column(align=True)
            adv_col.label(text="Advanced Sampling:", icon="MODIFIER")
            adv_col.prop(self, "top_p")
            adv_col.prop(self, "top_k")
            adv_col.prop(self, "repeat_penalty")

            adv_col.separator()
            adv_col.prop(self, "use_seed")
            if self.use_seed:
                adv_col.prop(self, "seed_value")

    def _draw_rag_settings(self, layout):
        """Draw RAG configuration."""
        rag_box = layout.box()
        rag_box.label(text="RAG (Documentation Search)", icon="DOCUMENTS")
        rag_col = rag_box.column(align=True)
        rag_col.prop(self, "use_rag")

        if self.use_rag:
            rag_col.prop(self, "rag_context_mode")
            if self.rag_context_mode == "FIXED":
                rag_col.prop(self, "rag_num_results")
            else:
                row = rag_col.row(align=True)
                row.prop(self, "rag_auto_min")
                row.prop(self, "rag_auto_max")
            rag_col.prop(self, "rag_follow_up_augmentation")
            rag_col.prop(self, "rag_source_bias")

            # Show RAG status (non-blocking). Kick off background load on first open.
            try:
                from . import rag_system

                # Start async load if needed; returns immediately
                rag_system.ensure_rag_loaded_async()
                rag = rag_system.get_rag_instance()
                # If still loading, show spinner; else show status
                if getattr(rag, "loading", False):
                    status_row = rag_col.row()
                    status_row.label(text="Loading RAG database...", icon="SORTTIME")
                else:
                    stats = rag.get_stats()
                    if stats.get("enabled"):
                        status_row = rag_col.row()
                        status_row.label(
                            text=f"✓ Loaded {stats.get('document_count', 0)} documents",
                            icon="CHECKMARK",
                        )
                    else:
                        status_row = rag_col.row()
                        status_row.label(text="⚠ RAG database not found", icon="ERROR")

            except Exception as e:
                rag_col.label(text=f"RAG status: {str(e)[:50]}", icon="ERROR")

    def _draw_api_keys(self, layout):
        """Draw stock photo API keys section."""
        api_box = layout.box()
        api_box.label(text="Stock Photo APIs (Optional)", icon="IMAGE_DATA")
        api_col = api_box.column(align=True)
        api_col.prop(self, "unsplash_api_key")
        api_col.prop(self, "pexels_api_key")
        api_col.separator()
        api_col.label(text="Get free API keys:", icon="URL")
        api_col.label(text="  • Unsplash: unsplash.com/developers")
        api_col.label(text="  • Pexels: pexels.com/api")

    # def _draw_tools_settings(self, layout):
    #     """Draw tools configuration section with checkboxes."""
    #     from .tools import tool_registry

    #     tools_box = layout.box()
    #     tools_box.label(text="Schema-Based Tools Configuration", icon="TOOL_SETTINGS")

    #     tools_col = tools_box.column(align=True)
    #     tools_col.label(
    #         text="Select which tools to expose to the LLM via OpenAI schema:",
    #         icon="INFO",
    #     )

    #     # Quick preset buttons
    #     tools_col.separator()
    #     tools_col.label(text="Quick Presets:", icon="PRESET")
    #     preset_row = tools_col.row(align=True)

    #     op = preset_row.operator("assistant.set_tool_preset", text="Lean (Default)")
    #     op.preset = "lean"

    #     op = preset_row.operator("assistant.set_tool_preset", text="Core Only")
    #     op.preset = "core"

    #     op = preset_row.operator("assistant.set_tool_preset", text="All Tools")
    #     op.preset = "all"

    #     # Refresh button
    #     tools_col.separator()
    #     tools_col.operator(
    #         "assistant.refresh_tool_config",
    #         text="Refresh Tool List",
    #         icon="FILE_REFRESH",
    #     )

    #     # Show tools grouped by category with checkboxes
    #     tools_col.separator()

    #     if not hasattr(self, "tool_config_items") or len(self.tool_config_items) == 0:
    #         tools_col.label(
    #             text="No tools found. Click 'Refresh Tool List'.", icon="INFO"
    #         )
    #         return

    #     # Count enabled tools
    #     enabled_count = sum(1 for t in self.tool_config_items if t.enabled)
    #     tools_col.label(
    #         text=f"Enabled: {enabled_count} / {len(self.tool_config_items)} tools",
    #         icon="CHECKBOX_HLT",
    #     )

    #     tools_col.separator()

    #     # Get tools grouped by category
    #     tools_by_category = {}
    #     for tool in self.tool_config_items:
    #         cat = tool.category or "Other"
    #         if cat not in tools_by_category:
    #             tools_by_category[cat] = []
    #         tools_by_category[cat].append(tool)

    #     # Draw category sections
    #     for category in sorted(tools_by_category.keys()):
    #         tools = tools_by_category[category]

    #         # Category header
    #         box = tools_col.box()
    #         row = box.row()

    #         enabled_in_cat = sum(1 for t in tools if t.enabled)
    #         row.label(
    #             text=f"{category} ({enabled_in_cat}/{len(tools)})", icon="TOOL_SETTINGS"
    #         )

    #         # Category enable/disable buttons
    #         op = row.operator(
    #             "assistant.toggle_category_tools_prefs",
    #             text="",
    #             icon="CHECKBOX_HLT",
    #             emboss=False,
    #         )
    #         op.category = category
    #         op.enable = True

    #         op = row.operator(
    #             "assistant.toggle_category_tools_prefs",
    #             text="",
    #             icon="CHECKBOX_DEHLT",
    #             emboss=False,
    #         )
    #         op.category = category
    #         op.enable = False

    #         # Tool list with checkboxes
    #         for tool in sorted(tools, key=lambda t: t.name):
    #             row = box.row()
    #             row.prop(tool, "enabled", text=tool.name)
    #             # Show execute_code as always enabled
    #             if tool.name == "execute_code":
    #                 row.enabled = False
    #                 row.label(text="(always enabled)", icon="LOCKED")

    #     tools_col.separator()
    #     tools_col.label(
    #         text="Note: execute_code is always enabled",
    #         icon="KEYFRAME_HLT",
    #     )

    # Schema-based tools configuration (checkbox-based UI)
    tool_config_items: bpy.props.CollectionProperty(type=ToolConfigItem)

    # Collapsible section toggles
    show_section_models: bpy.props.BoolProperty(
        name="Show Model Management", default=True
    )
    show_section_downloads: bpy.props.BoolProperty(
        name="Show Download Models", default=True
    )
    show_section_gpu: bpy.props.BoolProperty(name="Show GPU Optimization", default=True)
    show_section_generation: bpy.props.BoolProperty(
        name="Show Generation Settings", default=True
    )
    show_section_rag: bpy.props.BoolProperty(name="Show RAG Settings", default=True)
    show_section_api: bpy.props.BoolProperty(name="Show Stock Photo APIs", default=True)
    show_section_tools: bpy.props.BoolProperty(
        name="Show MCP enabled Tools", default=False
    )
    show_section_debug: bpy.props.BoolProperty(
        name="Show Debug Settings", default=False
    )

    debug_mode: bpy.props.BoolProperty(
        name="Enable Debug Logging",
        description="Print full LLM payloads and responses to the system console",
        default=False,
    )

    def draw(self, context):
        """Draw preferences UI"""
        layout = self.layout

        # Model Management (collapsible)
        row = layout.row(align=True)
        row.prop(
            self,
            "show_section_models",
            text="",
            icon="TRIA_DOWN" if self.show_section_models else "TRIA_RIGHT",
            emboss=False,
        )
        row.label(text="Model Management", icon="PREFERENCES")
        if self.show_section_models:
            self._draw_model_management(layout)

        # Download Models (collapsible)
        row = layout.row(align=True)
        row.prop(
            self,
            "show_section_downloads",
            text="",
            icon="TRIA_DOWN" if self.show_section_downloads else "TRIA_RIGHT",
            emboss=False,
        )
        row.label(text="Download Models", icon="IMPORT")
        if self.show_section_downloads:
            self._draw_model_downloads(layout)

        # UI / Visibility Options (collapsible)
        row = layout.row(align=True)
        row.prop(
            self,
            "show_section_ui",
            text="",
            icon="TRIA_DOWN" if self.show_section_ui else "TRIA_RIGHT",
            emboss=False,
        )
        row.label(text="UI & Visibility Options", icon="RESTRICT_VIEW_OFF")
        if self.show_section_ui:
            self._draw_ui_settings(layout)

        # GPU Optimization (collapsible)
        row = layout.row(align=True)
        row.prop(
            self,
            "show_section_gpu",
            text="",
            icon="TRIA_DOWN" if self.show_section_gpu else "TRIA_RIGHT",
            emboss=False,
        )
        row.label(text="GPU Optimization", icon="SHADING_RENDERED")
        if self.show_section_gpu:
            self._draw_gpu_settings(layout)

        # Generation Settings (collapsible)
        row = layout.row(align=True)
        row.prop(
            self,
            "show_section_generation",
            text="",
            icon="TRIA_DOWN" if self.show_section_generation else "TRIA_RIGHT",
            emboss=False,
        )
        row.label(text="Generation Settings", icon="SETTINGS")
        if self.show_section_generation:
            self._draw_generation_settings(layout)

        # RAG (collapsible)
        row = layout.row(align=True)
        row.prop(
            self,
            "show_section_rag",
            text="",
            icon="TRIA_DOWN" if self.show_section_rag else "TRIA_RIGHT",
            emboss=False,
        )
        row.label(text="RAG & Documentation Search", icon="DOCUMENTS")
        if self.show_section_rag:
            self._draw_rag_settings(layout)

        # Stock Photo APIs (collapsible)
        row = layout.row(align=True)
        row.prop(
            self,
            "show_section_api",
            text="",
            icon="TRIA_DOWN" if self.show_section_api else "TRIA_RIGHT",
            emboss=False,
        )
        row.label(text="Stock Photo APIs (Optional)", icon="IMAGE_DATA")
        if self.show_section_api:
            self._draw_api_keys(layout)

        # Tools Configuration (collapsible)
        row = layout.row(align=True)
        row.prop(
            self,
            "show_section_tools",
            text="",
            icon="TRIA_DOWN" if self.show_section_tools else "TRIA_RIGHT",
            emboss=False,
        )
        row.label(text="MCP Tools - Note: More Tools means more context bloat, which can reduce performance", icon="TOOL_SETTINGS")
        if self.show_section_tools:
            self._draw_tools_config(layout)

        # Debug Settings (collapsible)
        row = layout.row(align=True)
        row.prop(
            self,
            "show_section_debug",
            text="",
            icon="TRIA_DOWN" if self.show_section_debug else "TRIA_RIGHT",
            emboss=False,
        )
        row.label(text="Debug Settings", icon="CONSOLE")
        if self.show_section_debug:
            self._draw_debug_settings(layout)

    def _draw_debug_settings(self, layout):
        """Draw debug settings."""
        box = layout.box()
        box.label(text="Debug Configuration", icon="CONSOLE")
        col = box.column(align=True)
        col.prop(self, "debug_mode")
        col.operator("assistant.view_request_payload", text="Dump Prompt & Tools JSON", icon="TEXT")
        col.label(text="Check system console for output", icon="INFO")



    def _draw_ui_settings(self, layout):
        """Draw UI visibility settings."""
        box = layout.box()
        box.label(text="Message Visibility Control", icon="RESTRICT_VIEW_OFF")
        
        col = box.column(align=True)
        col.prop(self, "show_thinking", text="Show Thinking Process")
        col.prop(self, "show_system_updates", text="Show System Updates")
        col.prop(self, "show_tool_outputs", text="Show Tool Details")
        
        col.separator()
        col.label(text="Uncheck these to reduce clutter in the chat.", icon="INFO")

    def _draw_api_settings(self, layout):
        """Draw API key settings."""
        box = layout.box()
        box.label(text="Stock Photo API Keys", icon="WORLD")
        
        col = box.column(align=True)
        col.prop(self, "unsplash_api_key")
        col.prop(self, "pexels_api_key")
        
        col.separator()
        col.label(text="Keys are stored securely in Blender preferences.", icon="LOCKED")
        col.label(text="Required for searching and downloading stock photos.", icon="INFO")

    def _draw_tools_config(self, layout):
        """Draw tools configuration section"""
        box = layout.box()
        box.label(text="Schema-Based Tools Configuration", icon="TOOL_SETTINGS")

        tools_col = box.column(align=True)
        tools_col.label(
            text="Unchecked tools are hidden from context but available via SDK (cheaper).",
            icon="INFO",
        )
        
        # Presets
        row = box.row()
        row.label(text="Presets:")
        row.operator("assistant.set_tool_preset", text="Lean (Default)").preset = "lean"
        row.operator("assistant.set_tool_preset", text="Core Only").preset = "core"
        row.operator("assistant.set_tool_preset", text="All Tools").preset = "all"
        
        box.separator()
        
        # Refresh button
        row = box.row()
        row.operator("assistant.refresh_tool_config", icon="FILE_REFRESH")
        
        box.separator()
        
        # Tools list
        if len(self.tool_config_items) > 0:
            # Group by category
            by_category = {}
            for item in self.tool_config_items:
                cat = item.category or "Other"
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(item)
            
            # Draw categories
            for cat in sorted(by_category.keys()):
                # Create a visually distinct box for each category
                cat_box = box.box()
                
                # Category header row
                row = cat_box.row()
                row.label(text=cat, icon="PREFERENCES")  # Using a generic icon instead of misleading arrow
                
                # Toggle All Button
                op = row.operator("assistant.toggle_category_tools_prefs", text="Toggle All", icon="CHECKBOX_HLT")
                op.category = cat
                op.enable = True # Logic handled in operator
                
                col = cat_box.column(align=True)
                for item in sorted(by_category[cat], key=lambda x: x.name):
                    r = col.row(align=True)
                    if item.name == "execute_code" or item.name == 'sdk_help':
                        r.enabled = False # Always enabled
                    
                    # Left side: Tool Name
                    r.label(text=item.name)
                    
                    # Right side: Toggle
                    if item.expose_mcp:
                        r.prop(item, "expose_mcp", text="MCP Tool", icon="CHECKBOX_HLT", toggle=True)
                    else:
                        r.prop(item, "expose_mcp", text="SDK Only", icon="CHECKBOX_DEHLT", toggle=True)
        else:
            box.label(text="No tools found. Click Refresh.", icon="ERROR")


class ASSISTANT_OT_refresh_tool_config(bpy.types.Operator):
    bl_idname = "assistant.refresh_tool_config"
    bl_label = "Refresh Tool Configuration"
    bl_description = "Refresh the tool list from MCP registry"

    def execute(self, context):
        import json

        from .tools import tool_registry

        prefs = context.preferences.addons[__package__].preferences

        # Get current enabled tools to preserve state
        try:
            current_enabled = json.loads(prefs.schema_tools)
        except Exception:
            current_enabled = []

        # Clear and rebuild
        prefs.tool_config_items.clear()

        # Add all registered tools
        # Add all registered tools
        for name, tool_data in tool_registry._TOOLS.items():
            # Hide CORE TOOLS from UI (they are auto-injected for Agents)
            if name in ["execute_code", "sdk_help"]:
                continue
                
            item = prefs.tool_config_items.add()
            item.name = name
            item.category = tool_data.get("category", "Other")
            item.description = tool_data.get("description", "")
            # Default to FALSE for new tools (per user preference)
            item.expose_mcp = False
            
            # Logic:
            # 1. If currently enabled in JSON, keep enabled.
            # 2. If JSON is empty (first run), enable ALL defaults? Or stick to False?
            #    Let's stick to strict logic: Enabled only if explicitly in list, OR if list is empty (fresh install).
            
            if len(current_enabled) == 0:
                # Fresh install/reset: Enable all (User can disable later)
                # To be safer/cleaner, let's enable all initially so they see them.
                item.expose_mcp = True
            elif name in current_enabled:
                item.expose_mcp = True
            
            # exception: always enable critical schema tools? No, execute_code is handled separately.


        # Sync to schema_tools
        _sync_tools_to_json(prefs)

        self.report({"INFO"}, f"Refreshed {len(prefs.tool_config_items)} tools")
        return {"FINISHED"}


class ASSISTANT_OT_toggle_category_tools_prefs(bpy.types.Operator):
    bl_idname = "assistant.toggle_category_tools_prefs"
    bl_label = "Toggle Category Tools"
    bl_description = "Enable or disable all tools in this category"

    category: bpy.props.StringProperty()
    enable: bpy.props.BoolProperty()

    def execute(self, context):
        prefs = context.preferences.addons[__package__].preferences

        for tool in prefs.tool_config_items:
            if tool.category == self.category:
                # Always keep execute_code enabled
                if tool.name == "execute_code":
                    tool.expose_mcp = True
                else:
                    tool.expose_mcp = self.enable

        # Sync to schema_tools
        _sync_tools_to_json(prefs)

        return {"FINISHED"}


class ASSISTANT_OT_set_tool_preset(bpy.types.Operator):
    bl_idname = "assistant.set_tool_preset"
    bl_label = "Set Tool Preset"
    bl_description = "Set a predefined tool configuration"

    preset: bpy.props.StringProperty()

    def execute(self, context):
        from .tools import tool_registry

        prefs = context.preferences.addons[__package__].preferences

        if self.preset == "lean":
            # Default lean tool set - core Blender operations
            tools = [
                "execute_code",
                "get_scene_info",
                "get_object_info",
                "list_collections",
                "get_collection_info",
                "create_collection",
                "move_to_collection",
                "set_collection_color",
                "delete_collection",
                "get_selection",
                "get_active",
                "set_selection",
                "set_active",
                "select_by_type",
                "sdk_help",
                "capture_viewport_for_vision",
            ]
        elif self.preset == "core":
            # Minimal core set - just code execution and scene info
            tools = [
                "execute_code",
                "get_scene_info",
                "get_object_info",
                "sdk_help",
            ]
        elif self.preset == "all":
            # All registered tools
            all_tools = tool_registry.get_tools_list()
            tools = [t.get("name") for t in all_tools if t.get("name")]
        else:
            self.report({"ERROR"}, f"Unknown preset: {self.preset}")
            return {"CANCELLED"}

        # Update checkboxes
        tools_set = set(tools)
        for tool in prefs.tool_config_items:
            tool.expose_mcp = tool.name in tools_set or tool.name == "execute_code"

        # Sync to schema_tools
        _sync_tools_to_json(prefs)

        self.report({"INFO"}, f"Set {len(tools)} tools from preset '{self.preset}'")
        return {"FINISHED"}


def register():
    """Register preferences and operators, auto-scan for Ollama models."""
    bpy.utils.register_class(ToolConfigItem)

    bpy.utils.register_class(ASSISTANT_OT_refresh_models)
    bpy.utils.register_class(ASSISTANT_OT_pull_model)
    bpy.utils.register_class(ASSISTANT_OT_delete_model)
    bpy.utils.register_class(ASSISTANT_OT_start_ollama)
    bpy.utils.register_class(ASSISTANT_OT_stop_ollama)
    bpy.utils.register_class(ASSISTANT_OT_open_ollama_folder)

    bpy.utils.register_class(ASSISTANT_OT_search_ollama_library)

    bpy.utils.register_class(ASSISTANT_OT_toggle_model_capability)
    bpy.utils.register_class(ASSISTANT_OT_refresh_tool_config)
    bpy.utils.register_class(ASSISTANT_OT_toggle_category_tools_prefs)
    bpy.utils.register_class(ASSISTANT_OT_set_tool_preset)

    bpy.utils.register_class(AssistantPreferences)

    # Initialize tool config after a delay (tools need to be registered first)
    def delayed_tool_init():
        try:
            if bpy.context:
                import json

                from .tools import tool_registry

                prefs = bpy.context.preferences.addons[__package__].preferences

                # Sync with registry (add missing, update existing)
                existing_names = {t.name for t in prefs.tool_config_items}
                
                # Get default enabled tools (fallback)
                try:
                    default_enabled = json.loads(prefs.schema_tools)
                except Exception:
                    default_enabled = []

                for name, tool_data in tool_registry._TOOLS.items():
                    if name in ["execute_code", "sdk_help"]:
                        continue

                    if name not in existing_names:
                        # Add new tool
                        item = prefs.tool_config_items.add()
                        item.name = name
                        cat = tool_data.get("category", "Other")
                        item.category = cat
                        item.description = tool_data.get("description", "")
                        
                        # Enable by default if in schema_tools OR if it's a Web tool (User Request)
                        # Fix: Property is 'expose_mcp', not 'enabled'
                        is_web = (cat == "Web")
                        item.expose_mcp = (name in default_enabled) or is_web
                        
                        print(f"[Tool Config] Added new tool: {name} (Enabled: {item.expose_mcp})")
                    else:
                        # Update description/category of existing tool
                        for item in prefs.tool_config_items:
                            if item.name == name:
                                item.category = tool_data.get("category", "Other")
                                item.description = tool_data.get("description", "")
                                break
                
                print(f"[Tool Config] Synced {len(prefs.tool_config_items)} tools")
        except Exception as e:
            print(f"[Tool Config] Delayed init failed: {e}")
        return None

    bpy.app.timers.register(delayed_tool_init, first_interval=1.0)

    # Auto-scan for Ollama models on startup (non-blocking)
    global _ollama_models_enum_items, _ollama_models_cache
    try:
        # Check if Ollama is running first (with very short timeout)
        from . import ollama_subprocess

        ollama = ollama_subprocess.get_ollama()

        if ollama.is_running():
            models = get_ollama_models()
            if models:
                # Store ALL models in cache (including embedding models)
                _ollama_models_cache = models

                # But only show chat models in the dropdown (exclude embedding models)
                _ollama_models_enum_items = [
                    (m["name"], m["name"], format_model_description(m))
                    for m in models
                    if "embed"
                    not in m["name"].lower()  # Exclude embedding models from dropdown
                ]
                print(f"[Assistant] Auto-detected {len(models)} Ollama models")
        else:
            print(
                "[Assistant] Ollama not running, skipping auto-detect. Click 'Refresh Models' after starting Ollama."
            )
    except Exception as e:
        print(f"[Assistant] Could not auto-scan models on startup: {e}")
        # Don't crash, just use default empty list


def unregister():
    """Unregister preferences and operators."""
    bpy.utils.unregister_class(AssistantPreferences)
    bpy.utils.unregister_class(ASSISTANT_OT_set_tool_preset)
    bpy.utils.unregister_class(ASSISTANT_OT_toggle_category_tools_prefs)
    bpy.utils.unregister_class(ASSISTANT_OT_refresh_tool_config)
    bpy.utils.unregister_class(ASSISTANT_OT_toggle_model_capability)
    bpy.utils.unregister_class(ASSISTANT_OT_search_ollama_library)
    bpy.utils.unregister_class(ASSISTANT_OT_open_ollama_folder)
    bpy.utils.unregister_class(ASSISTANT_OT_stop_ollama)
    bpy.utils.unregister_class(ASSISTANT_OT_start_ollama)
    bpy.utils.unregister_class(ASSISTANT_OT_delete_model)
    bpy.utils.unregister_class(ASSISTANT_OT_pull_model)
    bpy.utils.unregister_class(ASSISTANT_OT_refresh_models)

    bpy.utils.unregister_class(ToolConfigItem)

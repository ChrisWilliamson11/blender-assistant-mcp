"""
Ollama subprocess manager for Blender extension.

This module manages an Ollama server process that runs alongside Blender,
providing GPU-accelerated LLM inference without DLL conflicts.

This approach uses Ollama's binary directly, which internally uses llama.cpp
with full CUDA support. By running it as a separate process, we avoid all
DLL loading conflicts that occur when trying to use llama-cpp-python inside Blender.
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Generator, List, Optional


class OllamaSubprocess:
    """Manages an Ollama server subprocess."""

    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize Ollama subprocess manager.

        Args:
            models_dir: Directory for Ollama model storage (sets OLLAMA_MODELS env var)
        """
        self.process: Optional[subprocess.Popen] = None
        self.models_dir = models_dir
        self.host = "127.0.0.1"
        self.port = 11435  # Use different port than system Ollama (11434)
        self.base_url = f"http://{self.host}:{self.port}"

        # Get extension directory
        extension_dir = Path(__file__).parent
        self.ollama_exe = extension_dir / "bin" / "ollama.exe"

        print(f"[Ollama] Initialized with models_dir: {models_dir}")
        print(f"[Ollama] Binary path: {self.ollama_exe}")

    def is_running(self) -> bool:
        """Check if Ollama server is running and responding."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=1) as response:
                return response.status == 200
        except:
            return False

    def start(self) -> bool:
        """Start the Ollama server subprocess.



        Returns:

            True if started successfully, False otherwise

        """

        # External mode: do not start a bundled server; just verify availability
        if getattr(self, "use_external", False):
            if self.is_running():
                print(f"[Ollama] Using external Ollama at {self.base_url}")
                return True
            print(
                f"[Ollama] External Ollama not reachable at {self.base_url}. Please start it manually."
            )
            return False

        if self.is_running():
            print(f"[Ollama] Server already running at {self.base_url}")
            return True

        if not self.ollama_exe.exists():
            print(f"[Ollama] Note: Bundled ollama.exe not found (Lite mode?). Assuming external server.")
            return False

        try:
            # Set environment variables
            env = os.environ.copy()
            env["OLLAMA_HOST"] = f"{self.host}:{self.port}"

            if self.models_dir:
                env["OLLAMA_MODELS"] = str(self.models_dir)

            # Start Ollama server
            print(f"[Ollama] Starting server at {self.base_url}")
            self.process = subprocess.Popen(
                [str(self.ollama_exe), "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )

            # Wait for server to start (max 10 seconds)
            for i in range(20):
                if self.is_running():
                    print(f"[Ollama] Server started successfully")
                    return True
                time.sleep(0.5)

            print(f"[Ollama] ERROR: Server failed to start within 10 seconds")
            self.stop()
            return False

        except Exception as e:
            print(f"[Ollama] ERROR starting server: {e}")
            return False

    def stop(self):
        """Stop the Ollama server subprocess and all child processes."""

        # External mode: never attempt to stop a user-managed server
        if getattr(self, "use_external", False):
            print("[Ollama] External mode: not stopping external server")
            return
        if self.process:
            print(f"[Ollama] Stopping server")

            try:
                # On Windows, use taskkill to kill process tree
                if os.name == "nt":
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                            capture_output=True,
                            timeout=10,
                        )
                        print(
                            f"[Ollama] Killed process tree for PID {self.process.pid}"
                        )
                    except Exception as e:
                        print(f"[Ollama] taskkill failed: {e}, trying terminate()")
                        self.process.terminate()
                        self.process.wait(timeout=5)
                else:
                    # On Unix, terminate should handle process group
                    self.process.terminate()
                    self.process.wait(timeout=5)

            except subprocess.TimeoutExpired:
                print(f"[Ollama] Force killing server")
                self.process.kill()
            except Exception as e:
                print(f"[Ollama] Error stopping server: {e}")
            finally:
                self.process = None

        # Also kill any orphaned ollama.exe processes
        if os.name == "nt":
            try:
                # Kill any ollama.exe processes that might be orphaned
                subprocess.run(
                    ["taskkill", "/F", "/IM", "ollama.exe"],
                    capture_output=True,
                    timeout=5,
                )
                subprocess.run(
                    ["taskkill", "/F", "/IM", "ollama_llama_server.exe"],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass  # Ignore errors if no processes found

        def chat(self, model: str, messages: list, stream: bool = False, **kwargs):
            """Send a chat request to Ollama.



            Args:

                model: Model name (e.g., "qwen2.5-coder:7b")

                messages: List of message dicts with 'role' and 'content'

                stream: Whether to stream the response

                **kwargs: Additional parameters (temperature, num_ctx, etc.)



            Returns:

                Response dict or generator if streaming

            """

            if not self.is_running():
                raise RuntimeError("Ollama server is not running")

            url = f"{self.base_url}/api/chat"

            payload = {"model": model, "messages": messages, "stream": stream, **kwargs}

            data = json.dumps(payload).encode("utf-8")

            req = urllib.request.Request(url, data=data)

            req.add_header("Content-Type", "application/json")

            with urllib.request.urlopen(req, timeout=300) as response:
                if stream:
                    # Stream line-delimited JSON; accumulate content and tool_calls
                    content_parts = []
                    final_tool_calls = None
                    top_level_tool_calls = None
                    last_message = None

                    for raw_line in response:
                        try:
                            line = raw_line.decode("utf-8").strip()
                        except Exception:
                            continue
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            # Ignore non-JSON keep-alives
                            continue

                        # Accumulate assistant message content and tool calls
                        msg = obj.get("message")
                        if isinstance(msg, dict):
                            last_message = msg
                            delta = msg.get("content")
                            if isinstance(delta, str) and delta:
                                content_parts.append(delta)
                            if isinstance(msg.get("tool_calls"), list) and msg.get(
                                "tool_calls"
                            ):
                                final_tool_calls = msg.get("tool_calls")

                        # Some servers may emit tool_calls at top-level
                        if isinstance(obj.get("tool_calls"), list) and obj.get(
                            "tool_calls"
                        ):
                            top_level_tool_calls = obj.get("tool_calls")

                        if obj.get("done") is True:
                            break

                    content = "".join(content_parts) if content_parts else ""
                    tool_calls = final_tool_calls or top_level_tool_calls or None

                    final_message = {
                        "role": "assistant",
                        "content": content,
                    }
                    if tool_calls:
                        final_message["tool_calls"] = tool_calls

                    return {"message": final_message}
                else:
                    return json.loads(response.read().decode("utf-8"))

    def pull_model(self, model: str):
        """Pull a model from the Ollama library."""
        if not self.is_running():
            return False
            
        print(f"[Ollama] Pulling model {model}...")
        try:
            # Use subprocess to pull so we can see progress or at least block until done
            import subprocess
            
            # Use startupinfo to hide console window on Windows
            startupinfo = None
            if os.name == 'nt':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
            subprocess.run(
                [self.binary_path, "pull", model], 
                check=True,
                startupinfo=startupinfo,
                capture_output=True # Capture output to avoid spamming console
            )
            print(f"[Ollama] Successfully pulled {model}")
            return True
        except Exception as e:
            print(f"[Ollama] Failed to pull {model}: {e}")
            return False

    def generate_embedding(self, model: str, text: str, keep_alive: str = "5m"):
        """Generate embeddings using Ollama.

        Args:
            model: Embedding model name (e.g., "nomic-embed-text")
            text: Text to embed
            keep_alive: How long to keep model loaded (default "5m")

        Returns:
            List of floats (embedding vector)
        """
        if not self.is_running():
            raise RuntimeError("Ollama server is not running")

        # Check if model exists, if not try to pull
        if not self.is_model_loaded(model):
            # is_model_loaded checks running models. We need to check available models.
            available = self.list_models()
            model_exists = False
            for m in available:
                if m.get("name") == model or m.get("name", "").startswith(model + ":"):
                    model_exists = True
                    break
            
            if not model_exists:
                print(f"[Ollama] Model {model} not found. Attempting to pull...")
                if not self.pull_model(model):
                    print(f"[Ollama] Failed to pull embedding model {model}")
                    return []

        url = f"{self.base_url}/api/embed"
        payload = {
            "model": model,
            "input": text,
            "keep_alive": keep_alive,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        req.add_header("Content-Type", "application/json")

        # Increase timeout to 120s - first load can be slow if chat model is loaded
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            print(f"[Ollama] Embedding request failed: {e}")
            return []

        # Return first embedding (Ollama returns list of embeddings)
        embeddings = result.get("embeddings")
        if not embeddings:
            print(f"[Ollama] ERROR: No embeddings in response: {result}")
            return []
        return embeddings[0]

    def list_models(self):
        """List available models.

        Returns:
            List of model dicts
        """
        if not self.is_running():
            print("[Ollama] Server not running, cannot list models")
            return []

        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("models", [])
        except Exception as e:
            print(f"[Ollama] Failed to list models: {e}")
            return []

    def list_running_models(self) -> Dict:
        """Get list of currently loaded models.

        Returns:
            Dict with 'models' array containing loaded model info
        """
        if not self.is_running():
            return {"models": []}

        url = f"{self.base_url}/api/ps"
        req = urllib.request.Request(url)

        try:
            # Use very short timeout to avoid blocking UI
            with urllib.request.urlopen(req, timeout=0.5) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            # Don't print timeout errors - they're expected during startup
            if "timed out" not in str(e).lower():
                print(f"[Ollama] Failed to get running models: {e}")
            return {"models": []}

    def is_model_loaded(self, model_name: str) -> Optional[Dict]:
        """Check if a specific model is loaded.

        Args:
            model_name: Name of model to check (can be filename or model name)

        Returns:
            Model info dict if loaded, None if not loaded
        """
        running = self.list_running_models()
        for model in running.get("models", []):
            # Match against both 'name' and 'model' fields
            if model.get("name") == model_name or model.get("model") == model_name:
                return model
            # Also try matching without extension
            if model.get("name", "").split(":")[0] == model_name.split(":")[0]:
                return model
        return None

    def preload_model(self, model_name: str, keep_alive: str = "30m") -> bool:
        """Preload a model into memory without generating.

        Args:
            model_name: Name of model to load
            keep_alive: How long to keep loaded ("-1" = forever, "5m" = 5 minutes, etc.)

        Returns:
            True if successful
        """
        if not self.is_running():
            print("[Ollama] Cannot preload model: server not running")
            return False

        # Use /api/generate with empty prompt to just load the model
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model_name,
            "prompt": "",  # Empty prompt just loads the model
            "stream": False,
            "keep_alive": keep_alive,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        req.add_header("Content-Type", "application/json")

        try:
            print(f"[Ollama] Preloading model: {model_name}")
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                if result.get("done", False):
                    print(f"[Ollama] Model {model_name} loaded successfully")
                    # Print ps for debugging GPU/CPU
                    try:
                        ps = self.list_running_models()
                        print(f"[Ollama] Running models: {ps}")
                    except Exception:
                        pass
                    return True
                return False
        except urllib.error.HTTPError as e:
            # Read error body for diagnostics and retry without keep_alive
            body = (
                e.read().decode("utf-8", errors="ignore") if hasattr(e, "read") else ""
            )
            print(f"[Ollama] Preload HTTP {e.code}: {body}")
            try:
                # Retry without keep_alive in case it's unsupported
                alt_payload = {"model": model_name, "prompt": "", "stream": False}
                alt_req = urllib.request.Request(
                    url, data=json.dumps(alt_payload).encode("utf-8")
                )
                alt_req.add_header("Content-Type", "application/json")
                with urllib.request.urlopen(alt_req, timeout=120) as response:
                    result = json.loads(response.read().decode("utf-8"))
                    if result.get("done", False):
                        print(
                            f"[Ollama] Model {model_name} loaded successfully (retry)"
                        )
                        try:
                            ps = self.list_running_models()
                            print(f"[Ollama] Running models: {ps}")
                        except Exception:
                            pass
                        return True
            except Exception as e2:
                print(f"[Ollama] Preload retry failed: {e2}")
            return False
        except Exception as e:
            print(f"[Ollama] Failed to preload model: {e}")
            return False

    def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory.

        Args:
            model_name: Name of model to unload

        Returns:
            True if successful
        """
        if not self.is_running():
            print("[Ollama] Cannot unload model: server not running")
            return False

        url = f"{self.base_url}/api/chat"
        payload = {"model": model_name, "messages": [], "keep_alive": 0}

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data)
        req.add_header("Content-Type", "application/json")

        try:
            print(f"[Ollama] Unloading model: {model_name}")
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read().decode("utf-8"))
                done_reason = result.get("done_reason", "")
                success = done_reason == "unload" or result.get("done", False)
                if success:
                    print(f"[Ollama] Model {model_name} unloaded successfully")
                return success
        except Exception as e:
            print(f"[Ollama] Failed to unload model: {e}")
            return False


# Global instance
_ollama_instance: Optional[OllamaSubprocess] = None


def get_ollama() -> OllamaSubprocess:
    """Get the global Ollama subprocess instance."""
    global _ollama_instance
    if _ollama_instance is None:
        import bpy

        # Try to get preferences - handle both addon name formats
        models_dir = None
        try:
            # Try extension format first (Blender 4.2+)
            if __package__ in bpy.context.preferences.addons:
                prefs = bpy.context.preferences.addons[__package__].preferences
                models_dir = Path(prefs.models_folder) if prefs.models_folder else None
            # Fallback to old format
            elif "blender_assistant_mcp" in bpy.context.preferences.addons:
                prefs = bpy.context.preferences.addons[
                    "blender_assistant_mcp"
                ].preferences
                models_dir = Path(prefs.models_folder) if prefs.models_folder else None
        except Exception as e:
            print(f"[Ollama] Warning: Could not get preferences: {e}")
            print(f"[Ollama] Using default models directory")

        _ollama_instance = OllamaSubprocess(models_dir=models_dir)
        # Apply external Ollama preferences if available
        try:
            use_ext = getattr(prefs, "use_external_ollama", False)
            ext_url = getattr(prefs, "external_ollama_url", "")
            if use_ext and ext_url:
                _ollama_instance.use_external = True
                _ollama_instance.base_url = ext_url
        except Exception:
            pass

    # Always apply external Ollama preferences on each call and normalize URL
    try:
        import bpy as _bpy

        prefs2 = None
        if __package__ in _bpy.context.preferences.addons:
            prefs2 = _bpy.context.preferences.addons[__package__].preferences
        elif "blender_assistant_mcp" in _bpy.context.preferences.addons:
            prefs2 = _bpy.context.preferences.addons[
                "blender_assistant_mcp"
            ].preferences
        use_ext = getattr(prefs2, "use_external_ollama", False) if prefs2 else False
        ext_url = getattr(prefs2, "external_ollama_url", "") if prefs2 else ""
        if use_ext and ext_url:
            u = str(ext_url).strip()
            # Strip trailing /api and slashes
            if u.endswith("/api") or u.endswith("/api/"):
                u = u[: u.rfind("/api")]
            while u.endswith("/"):
                u = u[:-1]
            # Add scheme if missing
            if not (
                u.lower().startswith("http://") or u.lower().startswith("https://")
            ):
                u = "http://" + u
            _ollama_instance.use_external = True
            _ollama_instance.base_url = u
        else:
            _ollama_instance.use_external = False
            _ollama_instance.base_url = (
                f"http://{_ollama_instance.host}:{_ollama_instance.port}"
            )
    except Exception:
        # Do not fail if preferences unavailable
        pass

    return _ollama_instance


def start_ollama() -> bool:
    """Start the Ollama subprocess."""
    return get_ollama().start()


def stop_ollama():
    """Stop the Ollama subprocess."""
    global _ollama_instance
    if _ollama_instance:
        _ollama_instance.stop()
        _ollama_instance = None


def list_running_models() -> Dict:
    """Get list of currently loaded models.

    Returns:
        Dict with 'models' array containing loaded model info
    """
    return get_ollama().list_running_models()


def is_model_loaded(model_name: str) -> Optional[Dict]:
    """Check if a specific model is loaded.

    Args:
        model_name: Name of model to check

    Returns:
        Model info dict if loaded, None if not loaded
    """
    return get_ollama().is_model_loaded(model_name)


def preload_model(model_name: str, keep_alive: str = "-1") -> bool:
    """Preload a model into memory without generating.

    Args:
        model_name: Name of model to load
        keep_alive: How long to keep loaded ("-1" = forever, "5m" = 5 minutes, etc.)

    Returns:
        True if successful
    """
    return get_ollama().preload_model(model_name, keep_alive)


def unload_model(model_name: str) -> bool:
    """Unload a model from memory.

    Args:
        model_name: Name of model to unload

    Returns:
        True if successful
    """
    return get_ollama().unload_model(model_name)

"""Blender Automation Assistant with MCP - Extension for Blender 4.2+

A self-contained Automation assistant for Blender with built-in MCP tool support.
Uses Ollama subprocess for GPU-accelerated LLM inference without DLL conflicts.

This approach runs Ollama's binary (which uses llama.cpp internally) as a separate
process, avoiding all DLL loading issues that occur when using llama-cpp-python
directly inside Blender's Python environment.
"""

import os
import sys

__version__ = "2.1.0"
_extension_dir = os.path.dirname(os.path.abspath(__file__))

    "version": (2, 1, 0),
bl_info = {
    "blender": (4, 2, 0),
    "category": "3D View",
}

# Import modules
from . import (
    assistant,
    mcp_tools,
    polyhaven_tools,
    preferences,
    selection_tools,
    stock_photo_tools,
    tool_selector,
    ui,
)

# Module list for registration
_modules = [
    mcp_tools,
    selection_tools,
    polyhaven_tools,
    stock_photo_tools,
    tool_selector,  # Register tool selector UI
    assistant,
    ui,
    preferences,
]


def register():
    """Register all extension modules."""
    for mod in _modules:
        if hasattr(mod, "register"):
            mod.register()

    print(f"[{__package__}] Blender Automation Assistant registered successfully!")

    # Start Ollama subprocess

    import bpy

    from .ollama_subprocess import start_ollama

    try:
        prefs = bpy.context.preferences.addons[__package__].preferences

        models_dir = prefs.models_folder if hasattr(prefs, "models_folder") else None

        if start_ollama():
            print(f"[{__package__}] Ollama server started successfully")

            # Schedule a non-blocking auto-refresh of models after Ollama starts
            def _auto_refresh_models_timer():
                try:
                    from .ollama_subprocess import get_ollama

                    if not get_ollama().is_running():
                        return 1.0  # retry in 1s until server is up
                    # Try to fire the refresh operator; if not yet registered, retry shortly
                    try:
                        bpy.ops.assistant.refresh_models()
                    except Exception:
                        return 1.0
                    return None  # stop timer
                except Exception:
                    # Stop retries on unexpected errors to avoid spamming
                    return None

            try:
                bpy.app.timers.register(_auto_refresh_models_timer, first_interval=1.0)
            except Exception as _te:
                print(
                    f"[{__package__}] Warning: Could not schedule auto-refresh: {_te}"
                )
        else:
            print(f"[{__package__}] Warning: Failed to start Ollama server")
    except Exception as e:
        print(f"[{__package__}] Warning: Could not start Ollama: {e}")


def unregister():
    """Unregister all extension modules."""
    # Stop Ollama subprocess first
    from .ollama_subprocess import stop_ollama

    try:
        stop_ollama()
        print(f"[{__package__}] Ollama server stopped")
    except Exception as e:
        print(f"[{__package__}] Warning: Error stopping Ollama: {e}")

    for mod in reversed(_modules):
        if hasattr(mod, "unregister"):
            mod.unregister()

    print(f"[{__package__}] Blender Automation Assistant with MCP unregistered")


if __name__ == "__main__":
    register()

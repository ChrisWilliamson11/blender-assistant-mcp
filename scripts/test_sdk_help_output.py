
import sys
import unittest.mock
from unittest.mock import MagicMock
import types

# 1. SETUP PACKAGE STRUCTURE
# usage: from .tool_registry import ... requires the parent package to exist in sys.modules

# Define the full package path
FULL_PKG = "bl_ext.user_default.blender_assistant_mcp"
TOOLS_PKG = f"{FULL_PKG}.tools"

# Create package modules
sys.modules["bl_ext"] = types.ModuleType("bl_ext")
sys.modules["bl_ext.user_default"] = types.ModuleType("bl_ext.user_default")
sys.modules[FULL_PKG] = types.ModuleType(FULL_PKG)
sys.modules[FULL_PKG].__path__ = ["."]  # Mark as package
sys.modules[TOOLS_PKG] = types.ModuleType(TOOLS_PKG)
sys.modules[TOOLS_PKG].__path__ = ["."] # Mark as package

# Mock bpy
mock_bpy = MagicMock()
sys.modules["bpy"] = mock_bpy

# 2. LOAD MODULES MANUALLY into the fake package structure
import importlib.util

def load_module_into_sys(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # CRITICAL: set __item__ and __package__ correctly
    mod.__package__ = name.rsplit(".", 1)[0]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load Registry (fake dependencies)
registry_name = f"{TOOLS_PKG}.tool_registry"
registry = load_module_into_sys(registry_name, "h:/blender-assistant-mcp/blender_assistant_mcp/tools/tool_registry.py")
sys.modules[registry_name] = registry # Explicitly ensure it's there

# Mock the registry tools list with a fake Polyhaven tool
registry._TOOLS = {
    "download_polyhaven": {
        "name": "download_polyhaven",
        "category": "PolyHaven",
        "description": "Download stuff.",
        "inputSchema": {
            "type": "object", 
            "properties": {
                "asset_id": {"type": "string"},
                "asset_type": {"type": "string", "enum": ["hdri", "texture", "model"]}
            },
            "required": ["asset_id"]
        }
    }
}
registry.get_tool_schema = lambda x: registry._TOOLS.get(x)
registry._TOOLS_LIST = list(registry._TOOLS.values())
registry.get_tools_list = lambda: registry._TOOLS_LIST
registry.get_tool_schema = lambda x: registry._TOOLS.get(x)

# Load assistant_sdk stub
sdk_mod = types.ModuleType(f"{FULL_PKG}.assistant_sdk")
sdk_obj = MagicMock()
sdk_mod.get_assistant_sdk = lambda: sdk_obj
sys.modules[f"{FULL_PKG}.assistant_sdk"] = sdk_mod

# Load core_tools (TEST TARGET)
core_tools_name = f"{TOOLS_PKG}.core_tools"
core_tools = load_module_into_sys(core_tools_name, "h:/blender-assistant-mcp/blender_assistant_mcp/tools/core_tools.py")

def run_test(tool_name, exposed=True):
    # Mock preferences
    mock_prefs = MagicMock()
    mock_item = MagicMock()
    mock_item.name = tool_name
    mock_item.expose_mcp = exposed
    
    # Setup prefs list
    # If exposed=True, include it in list with expose_mcp=True
    # If exposed=False, include it with expose_mcp=False
    mock_prefs.tool_config_items = [mock_item]
    
    # Inject into context
    mock_bpy.context.preferences.addons[FULL_PKG].preferences = mock_prefs
    
    # Trick __package__ lookup in core_tools (uses it to find prefs)
    # It accesses __package__ global inside the function, which in our loaded module is incorrect.
    # We might need to patch the function or relying on the try/catch blocks.
    # Let's patch the 'bpy' inside core_tools directly
    core_tools.bpy = mock_bpy
    core_tools.__package__ = FULL_PKG

    print(f"\n--- TESTING MODE: {'MCP (Native)' if exposed else 'SDK (Hidden)'} ---")
    result = core_tools.sdk_help(tool_name=tool_name)
    import json
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    print("Simulating SDK Help for 'download_polyhaven'...")
    run_test("download_polyhaven", exposed=True)
    run_test("download_polyhaven", exposed=False)

"""
Tool Manager for Blender Assistant.

This module centralizes tool definitions, schema generation, and context management.
It helps manage the trade-off between exposing tools natively (MCP) vs via SDK hints
to control context window usage.
"""

import json
from typing import Dict, List, Any, Set, Optional
from . import mcp_tools

class ToolManager:
    """Manages tool definitions and schema generation."""

    def __init__(self):
        # Core tools that should always be available natively
        self.core_tools: Set[str] = {
            "execute_code",
            "get_scene_info",
            "get_object_info",
            "list_collections",
            "get_collection_info",
            "get_selection",
            "get_active",
            "assistant_help",
        }
        
        # Tools that are enabled by default in the "lean" profile
        self.default_tools: Set[str] = self.core_tools | {
            "create_collection",
            "move_to_collection",
            "set_collection_color",
            "delete_collection",
            "set_selection",
            "set_active",
            "select_by_type",
            "capture_viewport_for_vision",
        }

    def get_enabled_tools(self, preferences=None) -> Set[str]:
        """Get the set of enabled tools based on preferences."""
        enabled = set()
        
        # Always include core tools
        enabled.update(self.core_tools)
        
        if preferences:
            # If preferences has a tool config, use it
            tool_config_items = getattr(preferences, "tool_config_items", None)
            if tool_config_items and len(tool_config_items) > 0:
                for t in tool_config_items:
                    if t.enabled:
                        enabled.add(t.name)
                return enabled

        # Fallback to default set
        return self.default_tools

    def get_openai_tools(self, enabled_tools: Set[str]) -> List[Dict[str, Any]]:
        """Generate OpenAI-style tool definitions for enabled tools."""
        tools = []
        all_tools = mcp_tools.get_tools_list()
        
        for t in all_tools:
            name = t.get("name")
            if name in enabled_tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": t.get("description", ""),
                        "parameters": t.get("inputSchema", {"type": "object", "properties": {}})
                    }
                })
        return tools

    def get_system_prompt_hints(self, enabled_tools: Set[str]) -> str:
        """Generate SDK hints for tools that are NOT enabled natively."""
        all_tool_names = {t.get("name") for t in mcp_tools.get_tools_list()}
        
        # Identify which domains are missing from native tools
        hints = []
        
        def _is_missing(names: List[str]) -> bool:
            return not any(n in enabled_tools for n in names)

        # Check specific domains
        if _is_missing(["create_object", "modify_object", "delete_object", "set_material"]):
            hints.append("- assistant_sdk.blender.* — scene/objects/collections/selection")
            
        if _is_missing(["search_polyhaven_assets", "download_polyhaven"]):
            hints.append("- assistant_sdk.polyhaven.search/download — PolyHaven assets")
            
        if _is_missing(["search_stock_photos", "download_stock_photo"]):
            hints.append("- assistant_sdk.stock_photos.search/download — Pexels/Unsplash (API keys)")
            
        if _is_missing(["sketchfab_login", "sketchfab_search", "sketchfab_download_model"]):
            hints.append("- assistant_sdk.sketchfab.login/search/download — Sketchfab")
            
        if _is_missing(["web_search"]):
            hints.append("- assistant_sdk.web.search — web results")
            
        if _is_missing(["rag_query", "rag_get_stats"]):
            hints.append("- assistant_sdk.rag.query/get_stats — Blender docs RAG")

        if not hints:
            return ""

        return "\nSDK Hints (use assistant_help for exact usage):\n" + "\n".join(hints) + "\n"

    def get_compact_tool_list(self, enabled_tools: Set[str]) -> str:
        """Get a compact string representation of enabled tools for the system prompt."""
        return mcp_tools.get_tools_schema(enabled_tools=sorted(list(enabled_tools)))

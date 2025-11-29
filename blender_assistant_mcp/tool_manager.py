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
            "assistant_help",
            "search_memory",
            "capture_viewport_for_vision",
            "rag_query",
        }
        
        # Default tools are now just the core tools (strict SDK-first approach)
        self.default_tools: Set[str] = self.core_tools.copy()

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
        # Map categories to SDK namespaces
        namespace_map = {
            "Blender": "assistant_sdk.blender",
            "PolyHaven": "assistant_sdk.polyhaven",
            "Stock Photos": "assistant_sdk.stock_photos",
            "Sketchfab": "assistant_sdk.sketchfab",
            "Web": "assistant_sdk.web",
            "RAG": "assistant_sdk.rag",
            "Memory": "assistant_sdk.memory",
            "Vision": "assistant_sdk.vision",
        }
        
        return mcp_tools.get_sdk_tools_schema(enabled_tools, namespace_map)

    def get_compact_tool_list(self, enabled_tools: Set[str]) -> str:
        """Get a compact string representation of enabled tools for the system prompt."""
        return mcp_tools.get_tools_schema(enabled_tools=sorted(list(enabled_tools)))

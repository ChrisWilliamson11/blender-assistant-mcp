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
        all_tools = mcp_tools.get_tools_list()
        
        # Group hidden tools by category
        hidden_by_category: Dict[str, List[str]] = {}
        
        for t in all_tools:
            name = t.get("name")
            if name not in enabled_tools:
                cat = t.get("category", "Other")
                if cat not in hidden_by_category:
                    hidden_by_category[cat] = []
                hidden_by_category[cat].append(name)
                
        if not hidden_by_category:
            return ""

        hints = []
        
        # Map categories to SDK namespaces and descriptions
        cat_map = {
            "Blender": ("assistant_sdk.blender", "scene/objects/collections/selection"),
            "PolyHaven": ("assistant_sdk.polyhaven", "PolyHaven assets"),
            "Stock Photos": ("assistant_sdk.stock_photos", "Pexels/Unsplash"),
            "Sketchfab": ("assistant_sdk.sketchfab", "Sketchfab models"),
            "Web": ("assistant_sdk.web", "Web search/download"),
            "RAG": ("assistant_sdk.rag", "Blender docs RAG"),
            "Memory": ("assistant_sdk.memory", "Long-term memory"),
            "Vision": ("assistant_sdk.vision", "Vision analysis"),
        }
        
        for cat, tools in hidden_by_category.items():
            if cat in cat_map:
                namespace, desc = cat_map[cat]
                # Simplify tool names for the hint (remove namespace prefix if present, though names are usually flat)
                # We just list the namespace and a generic description, or maybe a few key verbs?
                # The original hints were like "assistant_sdk.polyhaven.search/download".
                # Let's try to infer verbs from tool names (e.g. "search_polyhaven" -> "search")
                
                verbs = set()
                for t in tools:
                    # heuristic: extract verb from tool name (e.g. "create_object" -> "create")
                    parts = t.split("_")
                    if parts:
                        # Handle special cases or just take the first/last part?
                        # "search_polyhaven_assets" -> "search"
                        # "download_stock_photo" -> "download"
                        # "web_search" -> "search"
                        # "rag_query" -> "query"
                        
                        # Common verbs
                        if "search" in t: verbs.add("search")
                        elif "download" in t: verbs.add("download")
                        elif "create" in t: verbs.add("create")
                        elif "get" in t: verbs.add("get")
                        elif "set" in t: verbs.add("set")
                        elif "delete" in t: verbs.add("delete")
                        elif "query" in t: verbs.add("query")
                        elif "fetch" in t: verbs.add("fetch")
                        elif "login" in t: verbs.add("login")
                        else: verbs.add(parts[0]) # Fallback
                
                verb_str = "/".join(sorted(list(verbs)))
                hints.append(f"- {namespace}.{verb_str} — {desc}")
            else:
                # Fallback for unknown categories
                hints.append(f"- assistant_sdk.{cat.lower()}.* — {', '.join(tools[:3])}...")

        return "\nSDK Hints (use assistant_help for exact usage):\n" + "\n".join(hints) + "\n"

    def get_compact_tool_list(self, enabled_tools: Set[str]) -> str:
        """Get a compact string representation of enabled tools for the system prompt."""
        return mcp_tools.get_tools_schema(enabled_tools=sorted(list(enabled_tools)))

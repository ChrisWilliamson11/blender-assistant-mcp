"""
Tool Manager for Blender Assistant.

This module centralizes tool definitions, schema generation, and context management.
It helps manage the trade-off between exposing tools natively (MCP) vs via SDK hints
to control context window usage.
"""

import json
from typing import Dict, List, Any, Set, Optional
from .tools import tool_registry

class ToolManager:
    """Manages tool definitions and schema generation."""

    def __init__(self):
        # Core tools - empty to allow full user control via preferences
        self.core_tools: Set[str] = set()
        
        # Default tools enabled if no preferences are found (first run)
        self.default_tools: Set[str] = {
            "execute_code",
            "assistant_help",
            "get_scene_info",
            "get_object_info",
            "search_memory",
            "capture_viewport_for_vision",
            "rag_query",
            "web_search",
            "fetch_webpage",
        }

    def get_enabled_tools(self, preferences=None) -> Set[str]:
        """Get the set of enabled tools based on preferences."""
        enabled = set()
        
        if preferences:
            # Check global feature flags
            rag_enabled = getattr(preferences, "use_rag", True)
            memory_enabled = getattr(preferences, "use_memory", True)
            vision_enabled = getattr(preferences, "use_vision", True)

            # If preferences has a tool config, use it
            tool_config_items = getattr(preferences, "tool_config_items", None)
            if tool_config_items and len(tool_config_items) > 0:
                for t in tool_config_items:
                    if t.enabled:
                        # Respect feature flags for specific tools
                        if t.name == "rag_query" and not rag_enabled:
                            continue
                        if t.name == "search_memory" and not memory_enabled:
                            continue
                        if t.name == "capture_viewport_for_vision" and not vision_enabled:
                            continue
                        enabled.add(t.name)
                return enabled
            
            # If no tool config (first run?), use defaults but respect feature flags
            for tool_name in self.default_tools:
                if tool_name == "rag_query" and not rag_enabled:
                    continue
                if tool_name == "search_memory" and not memory_enabled:
                    continue
                if tool_name == "capture_viewport_for_vision" and not vision_enabled:
                    continue
                enabled.add(tool_name)
            return enabled

        # Fallback to default set
        return self.default_tools

    def get_openai_tools(self, enabled_tools: Set[str]) ->List[Dict[str, Any]]:
        """Generate OpenAI-style tool definitions for enabled tools."""
        tools = []
        all_tools = tool_registry.get_tools_list()
        
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

    def get_system_prompt_hints(self, enabled_tools: Set[str], allowed_tools: Set[str] = None) -> str:
        """Generate SDK hints for tools that are NOT enabled natively.
        
        Args:
            enabled_tools: Tools that are enabled as MCP (schemas injected).
            allowed_tools: Validation set interaction. Only show hints for tools in this set (Universe).
                           If None, shows all known tools (legacy behavior).
        """
        # Group tools by namespace
        tools_by_namespace = {}
        all_tools = tool_registry.get_tools_list()
        
        for tool in all_tools:
            name = tool.get("name")
            
            # Skip if enabled as MCP (native)
            if name in enabled_tools:
                continue
                
            # Skip if not allowed for this role (Universe restriction)
            if allowed_tools is not None and name not in allowed_tools:
                continue
                
            category = tool.get("category", "Other")
            namespace = category.lower().replace(" ", "_").replace("-", "_")
            
            if namespace not in tools_by_namespace:
                tools_by_namespace[namespace] = []
            tools_by_namespace[namespace].append(name)
            
        if not tools_by_namespace:
            return ""

        lines = ["SDK CAPABILITIES (Use `assistant_help` to find tools):"]
        for namespace, tools in sorted(tools_by_namespace.items()):
            # Format: - namespace: tool1, tool2, ...
            tool_list = ", ".join(sorted(tools))
            lines.append(f"- **{namespace}**: {tool_list}")
            
        return "\n".join(lines)

    def get_enabled_tools_for_role(self, role: str) -> Set[str]:
        """Get the specific set of tools allowed for a given agent role."""
        if role == "MANAGER":
            return {
                "get_object_info", "get_scene_info", "inspect_data", "search_data",
                "remember_fact", "remember_learning", "remember_preference",
                "task_add", "task_clear", "task_list", "task_update", "task_complete",
                "consult_specialist",
                "get_active", "get_selection", "capture_viewport_for_vision"
            }
        elif role == "TASK_AGENT":
            return {
                # Core
                "execute_code", "assistant_help",
                # Blender
                "create_collection", "delete_collection", "get_collection_info", 
                "list_collections", "move_to_collection", "set_collection_color",
                "get_object_info", "get_scene_info", "inspect_data", "search_data",
                "get_active", "get_selection", "select_by_type", "set_active", "set_selection",
                # Polyhaven
                "download_polyhaven", "search_polyhaven_assets",
                # Web/RAG
                "web_search", "fetch_webpage", "search_image_url", "search_wikimedia_image", 
                "extract_image_urls_from_webpage", "download_image_as_texture", 
                "rag_get_stats", "rag_query",
                # Vision
                "capture_viewport_for_vision",
                # Sketchfab/Stock (if available)
                "sketchfab_download", "sketchfab_login", "sketchfab_search",
                "check_stock_photo_download", "download_stock_photo", "search_stock_photos"
            }
        elif role == "COMPLETION":
            return {
                "get_scene_info", "get_object_info", "inspect_data", "search_data", 
                "list_collections", "task_list",
                "get_active", "get_selection"
            }
        else:
            return set()

    def get_compact_tool_list(self, enabled_tools: Set[str]) -> str:
        """Get a compact string representation of enabled tools for the system prompt."""
        return tool_registry.get_tools_schema(enabled_tools=sorted(list(enabled_tools)))

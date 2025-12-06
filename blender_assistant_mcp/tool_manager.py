"""
Tool Manager for Blender Assistant.

This module centralizes tool definitions, schema generation, and context management.
It helps manage the trade-off between exposing tools MCPly (MCP) vs via SDK hints
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
        
        # Default tools - Removed per user request to enforce explicit configuration.
        self.default_tools: Set[str] = set()

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
            if tool_config_items:
                for t in tool_config_items:
                    if t.enabled:
                        # Respect global feature flags by Category
                        # This avoids hardcoding tool names
                        if t.category == "RAG" and not rag_enabled:
                            continue
                        if t.category == "Memory" and not memory_enabled:
                            continue
                        if t.category == "Vision" and not vision_enabled:
                            continue
                        
                        enabled.add(t.name)
                return enabled
            
        # If no tool config or preferences could not be read, return empty set (Pure SDK)
        # We do NOT fallback to defaults anymore.
        return set()

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
        """Generate SDK hints for tools that are NOT enabled MCPly.
        
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
            
            # Skip if enabled as MCP (MCP)
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

    def get_allowed_tools_for_role(self, role: str) -> Set[str]:
        """Get the 'Universe' of tools allowed for a role (ignoring preferences)."""
        base_set = {"execute_code", "assistant_help"} # Always allowed

        if role == "MANAGER":
            return base_set.union({
                # System / Memory / Task
                "remember_fact", "remember_learning", "remember_preference",
                "task_add", "task_clear", "task_list", "task_update", "task_complete",
                "consult_specialist",
                # Inspection / Vision
                "get_object_info", "get_scene_info", "inspect_data", "search_data",
                "capture_viewport_for_vision"
            })
        
        elif role == "TASK_AGENT":
            return base_set.union({
                 # Blender
                "create_collection", "delete_collection", "get_collection_info", 
                "list_collections", "move_to_collection", "set_collection_color",
                "get_object_info", "get_scene_info", "inspect_data", "search_data",
                "get_active", "get_selection", "select_by_type", "set_active", "set_selection",
                # Polyhaven
                "download_polyhaven", "search_polyhaven_assets", "get_polyhaven_asset_info",
                # Web/RAG
                "web_search", "fetch_webpage", "search_image_url", "search_wikimedia_image", 
                "extract_image_urls_from_webpage", "download_image_as_texture", 
                "rag_get_stats", "rag_query",
                # Vision
                "capture_viewport_for_vision",
                # Sketchfab/Stock (if available)
                "sketchfab_download", "sketchfab_login", "sketchfab_search",
                "check_stock_photo_download", "download_stock_photo", "search_stock_photos"
            })

        elif role == "COMPLETION_AGENT":
            return base_set.union({
                 "get_scene_info", "get_object_info", "inspect_data", "search_data", 
                 "list_collections", "task_list",
                 "get_active", "get_selection"
            })
            
        return set()

    def get_enabled_tools_for_role(self, role: str, preferences=None) -> Set[str]:
        """Get the specific set of tools enabled for a role (intersected with preferences)."""
        allowed = self.get_allowed_tools_for_role(role)
        
        # Start with core tools (ALWAYS ENABLED)
        enabled = {"execute_code", "assistant_help"}
        
        # If we don't have prefs, usually implies "Show Everything" (or default behavior)
        # But per user request: "Enabled = MCP, Disabled = SDK". 
        # If prefs is none, we default to ALL Allowed tools being MCP enabled.
        if not getattr(preferences, "tool_config_items", None):
            return allowed
            
        # Filter by user preferences
        pref_enabled = {t.name for t in preferences.tool_config_items if t.enabled}
        
        for tool in allowed:
            if tool in pref_enabled:
                enabled.add(tool)
        
        return enabled

    def get_compact_tool_list(self, enabled_tools: Set[str]) -> str:
        """Get a compact string representation of enabled tools for the system prompt."""
        return tool_registry.get_tools_schema(enabled_tools=sorted(list(enabled_tools)))

    def get_common_behavior_prompt(self) -> str:
        """Get the standardized BEHAVIOR section for all agents."""
        return """BEHAVIOR
- **PLAN FIRST**: If a request is complex, briefly plan before executing.
- **ACCESS METHODS**: You have two ways to act:
  1. **MCP Tools**: Call these using your built-in tool calling mechanism, using the format specified (e.g., `get_scene_info`).
  2. **Python Code**: Use `execute_code` to run scripts. Use this for `assistant_sdk.*` methods and raw `bpy` commands.
- **FINDING TOOLS**: Do not guess tool names. Use `assistant_help` to find SDK methods, `rag_query` for docs, or `search_memory` for past solutions.
- **SCENE AWARENESS**: 'SCENE UPDATES' provide a SUMMARY of changes (added/modified objects). Use `inspect_data` or `get_scene_info(detailed=True)` to fetch detailed properties (like vertices, modifiers, or custom props) if needed.
- **CLEANUP**: Keep the scene organized. Use collections to group new objects.
- **VERIFY**: Always verify your actions.
- **TEST OVER GUESS**: If unsure about API behavior, write a small test script using `execute_code` instead of speculating.
- **LEARN**: Use `remember_learning` to record pitfalls or version quirks."""

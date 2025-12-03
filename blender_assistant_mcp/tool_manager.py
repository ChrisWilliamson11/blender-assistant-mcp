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
            "research_topic",
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

    def get_system_prompt_hints(self, enabled_tools: Set[str]) -> str:
        """Generate SDK hints for tools that are NOT enabled natively."""
        lines = []
        all_tools = tool_registry.get_tools_list()
        
        # Sort by category then name for better organization
        sorted_tools = sorted(all_tools, key=lambda x: (x.get("category", "Other"), x.get("name")))
        
        for tool in sorted_tools:
            name = tool.get("name")
            
            # Skip if natively enabled
            if name in enabled_tools:
                continue
                
            category = tool.get("category", "Other")
            namespace = category.lower().replace(" ", "_").replace("-", "_")
            
            # Construct SDK usage string
            # e.g. assistant_sdk.blender.create_object(type, name)
            schema = tool.get("inputSchema", {})
            props = schema.get("properties", {})
            arg_names = list(props.keys())
            args_str = ", ".join(arg_names)
            
            usage = f"assistant_sdk.{namespace}.{name}({args_str})"
            desc = (tool.get("description", "") or "").strip()
            
            # Truncate description if too long
            if len(desc) > 150:
                desc = desc[:147] + "..."
                
            lines.append(f"- {usage}: {desc}")
                
        header = "SDK TOOLS (Call via `execute_code`):\n"
        return header + "\n".join(lines) if lines else ""

    def get_compact_tool_list(self, enabled_tools: Set[str]) -> str:
        """Get a compact string representation of enabled tools for the system prompt."""
        return tool_registry.get_tools_schema(enabled_tools=sorted(list(enabled_tools)))

"""Tools for interacting with the memory system."""

from typing import Dict, Any
from ..memory import MemoryManager

# Global instance (lazy loaded or injected)
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager

def remember_preference(key: str, value: str) -> Dict[str, Any]:
    """Remember a user preference for future sessions.
    
    Args:
        key: The preference category/name (e.g., "color_scheme", "unit_system")
        value: The preference value (e.g., "dark mode", "metric")
    """
    mem = get_memory_manager()
    mem.remember_preference(key, value)
    return {"success": True, "message": f"Remembered preference: {key}={value}"}

def remember_fact(fact: str, category: str = "general") -> Dict[str, Any]:
    """Remember a fact or instruction for the future.
    
    Args:
        fact: The content to remember (e.g., "The user likes low-poly style")
        category: Optional category tag
    """
    mem = get_memory_manager()
    mem.remember_fact(fact, category)
    return {"success": True, "message": f"Remembered fact: {fact}"}

def register():
    """Register memory tools."""
    from . import tool_registry
    
    tool_registry.register_tool(
        "remember_preference",
        remember_preference,
        "Store a user preference persistently.",
        {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Preference key"},
                "value": {"type": "string", "description": "Preference value"}
            },
            "required": ["key", "value"]
        },
        category="Memory"
    )
    
    tool_registry.register_tool(
        "remember_fact",
        remember_fact,
        "Store a general fact or instruction persistently.",
        {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "Fact to remember"},
                "category": {"type": "string", "description": "Category (optional)"}
            },
            "required": ["fact"]
        },
        category="Memory"
    )

    tool_registry.register_tool(
        "remember_learning",
        remember_learning,
        "Record a technical learning, pitfall, or version quirk.",
        {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic (e.g., 'Blender 4.0', 'BSDF')"},
                "insight": {"type": "string", "description": "The technical insight or pitfall to avoid"}
            },
            "required": ["topic", "insight"]
        },
        category="Memory"
    )

    tool_registry.register_tool(
        "search_memory",
        search_memory,
        "Semantically search memory for relevant facts/learnings.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        },
        category="Memory"
    )

def remember_learning(topic: str, insight: str) -> Dict[str, Any]:
    """Record a technical learning."""
    mem = get_memory_manager()
    mem.remember_learning(topic, insight)
    return {"success": True, "message": f"Recorded learning on [{topic}]"}

def search_memory(query: str) -> Dict[str, Any]:
    """Search memory."""
    mem = get_memory_manager()
    results = mem.search_memory(query)
    return {"results": results}

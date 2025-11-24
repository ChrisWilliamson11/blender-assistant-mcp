"""Tools for interacting with the memory system."""

from typing import Dict, Any
from .memory import MemoryManager

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

def register_tools():
    """Register memory tools."""
    from . import mcp_tools
    
    mcp_tools.register_tool(
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
    
    mcp_tools.register_tool(
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

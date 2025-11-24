"""Memory manager for Blender Assistant.

Handles loading, saving, and retrieving persistent memory (preferences, facts, etc.)
from a JSON file in the extension directory.
"""

import json
import os
from typing import Dict, List, Any, Optional

class MemoryManager:
    def __init__(self, memory_file: str = "memory.json"):
        # Use extension directory by default if just a filename is given
        if not os.path.isabs(memory_file):
            ext_dir = os.path.dirname(os.path.abspath(__file__))
            self.file_path = os.path.join(ext_dir, memory_file)
        else:
            self.file_path = memory_file
            
        self.data: Dict[str, Any] = {
            "user_preferences": {},
            "facts": [],
            "project_context": {}
        }
        self.load()

    def load(self):
        """Load memory from disk."""
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # Merge with default structure to ensure keys exist
                    self.data.update(loaded)
            except Exception as e:
                print(f"[Memory] Failed to load memory: {e}")

    def save(self):
        """Save memory to disk."""
        try:
            with open(self.file_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            print(f"[Memory] Failed to save memory: {e}")

    def remember_preference(self, key: str, value: Any):
        """Store a user preference."""
        self.data["user_preferences"][key] = value
        self.save()

    def remember_fact(self, fact: str, category: str = "general"):
        """Store a general fact."""
        self.data["facts"].append({
            "content": fact,
            "category": category,
            "timestamp": 0 # TODO: Add real timestamp if needed
        })
        self.save()

    def get_context(self) -> str:
        """Get a formatted string of relevant memory for the system prompt."""
        lines = []
        
        prefs = self.data.get("user_preferences", {})
        if prefs:
            lines.append("User Preferences:")
            for k, v in prefs.items():
                lines.append(f"- {k}: {v}")
        
        facts = self.data.get("facts", [])
        if facts:
            lines.append("Learned Facts:")
            # Show last 5 facts to avoid context bloat
            # In a real system, we'd use semantic search here
            for f in facts[-5:]:
                lines.append(f"- {f['content']}")
                
        return "\n".join(lines)

    def clear(self):
        """Clear all memory."""
        self.data = {
            "user_preferences": {},
            "facts": [],
            "project_context": {}
        }
        self.save()

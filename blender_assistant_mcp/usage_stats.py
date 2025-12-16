import bpy
import json
import os
from pathlib import Path
from typing import Dict, Any

class ToolUsageTracker:
    """Tracks tool usage stats persistently across sessions."""
    
    def __init__(self):
        # Store in Blender's user config directory
        config_dir = Path(bpy.utils.user_resource('CONFIG'))
        self.stats_file = config_dir / "tool_usage_stats.json"
        
        # Structure: { tool_name: { "MCP": int, "SDK": int, "total": int } }
        self.stats: Dict[str, Dict[str, int]] = {}
        self._load()
        
    def _load(self):
        """Load stats from file."""
        if self.stats_file.exists():
            try:
                with open(self.stats_file, 'r') as f:
                    self.stats = json.load(f)
            except Exception as e:
                print(f"[UsageTracker] Error loading stats: {e}")
                self.stats = {}
                
    def save(self):
        """Save stats to file."""
        try:
            with open(self.stats_file, 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception as e:
            print(f"[UsageTracker] Error saving stats: {e}")

    def track_usage(self, tool_name: str, source: str = "MCP"):
        """Increment usage count for a tool."""
        if tool_name not in self.stats:
            self.stats[tool_name] = {"MCP": 0, "SDK": 0, "total": 0}
            
        entry = self.stats[tool_name]
        
        if source not in entry:
            entry[source] = 0
            
        entry[source] += 1
        entry["total"] += 1
        
        # Auto-save on every update (low frequency enough)
        self.save()

# Global instance
tracker = ToolUsageTracker()

def track_usage(tool_name: str, source: str = "MCP"):
    tracker.track_usage(tool_name, source)

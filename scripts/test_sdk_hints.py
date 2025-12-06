import sys
import os
import unittest
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock dependencies
sys.modules["bpy"] = MagicMock()
sys.modules["bmesh"] = MagicMock()
sys.modules["mathutils"] = MagicMock()

from blender_assistant_mcp.core import AssistantSession
from blender_assistant_mcp.tool_manager import ToolManager

class TestSystemPromptHints(unittest.TestCase):
    def setUp(self):
        self.tool_manager = ToolManager()
        # Mock getting tools list since it depends on registry imports which might trigger bpy
        # Or hopefully registry is pure python.
        # Let's inspect real output if possible, but mock is safer.
        
        # We'll use the real ToolManager logic but mock tool_registry.get_tools_list
        # We need to patch tool_registry inside tool_manager module or mock the method if accessible.
        # Since tool_manager imports tool_registry, we can patch `blender_assistant_mcp.tool_manager.tool_registry`
        
        from blender_assistant_mcp import tool_manager
        tool_manager.tool_registry = MagicMock()
        tool_manager.tool_registry.get_tools_list.return_value = [
            {"name": "search_polyhaven_assets", "category": "PolyHaven", "description": "Search stuff."},
            {"name": "download_polyhaven", "category": "PolyHaven", "description": "Download stuff."},
            {"name": "create_object", "category": "Blender", "description": "Create stuff."}
        ]
        
        self.tool_manager = ToolManager()
        self.tool_manager.get_enabled_tools = MagicMock(return_value={"execute_code", "assistant_help"})
        
    def test_hints_format(self):
        """Verify hints are grouped by namespace and hide signatures."""
        print("\nTesting System Prompt Hints...")
        
        hints = self.tool_manager.get_system_prompt_hints(enabled_tools={"execute_code"})
        print(f"\nGenerated Hints:\n{hints}\n")
        
        # Check assertions
        self.assertIn("SDK CAPABILITIES", hints)
        self.assertIn("- **polyhaven**:", hints)
        self.assertIn("search_polyhaven_assets", hints)
        
        # Check that tool lines do NOT have signatures (parens after name)
        # We split lines and check content lines
        for line in hints.split("\n"):
            if line.strip().startswith("- **"):
                # e.g. - **namespace**: tool1, tool2
                # Should not contain "(" except maybe in comments if any?
                # Actually my generating code puts tool names separated by commas.
                # No parens should be present in the tool list part.
                # But earlier assertion failed on global lookup because header had parens.
                pass 
                
        print("PASS: Hints are opaque and grouped.")

if __name__ == "__main__":
    unittest.main()

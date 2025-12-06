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

from blender_assistant_mcp.core import AssistantSession, ToolCall
from blender_assistant_mcp.tool_manager import ToolManager

class TestAutoTranslation(unittest.TestCase):
    def setUp(self):
        self.tool_manager = MagicMock(spec=ToolManager)
        self.tool_manager.get_enabled_tools.return_value = ["execute_code"]
        self.session = AssistantSession("test-model", self.tool_manager)
        
    def test_auto_translation(self):
        """Verify that SDK calls are translated to execute_code."""
        print("\nTesting Auto-Translation...")
        
        # Mock ResponseParser to return the SDK call
        # We can't easily mock the static method if it's imported in core.py
        # But we can mock the response dict that ResponseParser parses.
        # Assuming ResponseParser handles standard tool_calls format.
        
        response = {
            "message": {
                "content": "I will download the lamp.",
                "tool_calls": [
                    {
                        "function": {
                            "name": "assistant_sdk.polyhaven.search_polyhaven_assets",
                            "arguments": {"query": "lamp", "limit": 5}
                        }
                    }
                ]
            }
        }
        
        # We need to ensure ResponseParser works or mock it.
        # Let's try to run it with the real ResponseParser (imported in core).
        # If ResponseParser fails, we might need to adjust the input.
        
        # Run process_response
        # Note: process_response calls ResponseParser.parse(response)
        
        valid_calls, thinking = self.session.process_response(response)
        
        # Verify results
        self.assertEqual(len(valid_calls), 1)
        call = valid_calls[0]
        
        self.assertEqual(call.tool, "execute_code")
        self.assertIn("assistant_sdk.polyhaven.search_polyhaven_assets", call.args["code"])
        self.assertIn("query='lamp'", call.args["code"])
        
        print(f"Translated Code: {call.args['code']}")
        print("PASS: Auto-Translation verified.")

if __name__ == "__main__":
    unittest.main()

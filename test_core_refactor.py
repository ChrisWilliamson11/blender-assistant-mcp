import json
import unittest
from dataclasses import dataclass
from typing import Dict, Any, List

# Mock ToolCall for testing if not importing from core
@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any]

# Import the refactored ResponseParser from core.py
# Since we can't easily import from the extension structure in a standalone script without setup,
# I will copy the class definition here for testing the LOGIC, or I can try to import if path allows.
# Given the environment, I'll try to import by adding the path.

import sys
import os
from unittest.mock import MagicMock

# Mock bpy
sys.modules["bpy"] = MagicMock()

sys.path.append("h:/blender-assistant-mcp")
# We need to be careful about imports. core imports mcp_tools, tool_manager, memory, scene_watcher.
# scene_watcher imports bpy.
# memory imports bpy.
# So mocking bpy should work.

from blender_assistant_mcp.core import ResponseParser, ToolCall

class TestResponseParser(unittest.TestCase):
    def test_native_calls(self):
        response = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "test_tool",
                            "arguments": {"arg": 1}
                        }
                    }
                ]
            }
        }
        calls = ResponseParser.parse(response)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tool, "test_tool")
        self.assertEqual(calls[0].args, {"arg": 1})

    def test_json_blocks(self):
        content = """
        Thinking...
        ```json
        {"name": "json_tool", "arguments": {"x": 2}}
        ```
        """
        response = {"message": {"content": content}}
        calls = ResponseParser.parse(response)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tool, "json_tool")
        self.assertEqual(calls[0].args, {"x": 2})

    def test_code_blocks(self):
        content = """
        I will run this code:
        ```python
        import bpy
        bpy.ops.mesh.primitive_cube_add()
        ```
        """
        response = {"message": {"content": content}}
        calls = ResponseParser.parse(response)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tool, "execute_code")
        self.assertIn("import bpy", calls[0].args["code"])

if __name__ == "__main__":
    unittest.main()

import sys
from unittest.mock import MagicMock

# Mock bpy
mock_bpy = MagicMock()
sys.modules["bpy"] = mock_bpy

import unittest
from blender_assistant_mcp.core import AssistantSession
from blender_assistant_mcp.tool_manager import ToolManager

class TestSystemPrompt(unittest.TestCase):
    def test_prompt_contains_new_instructions(self):
        tm = ToolManager()
        session = AssistantSession("test-model", tm)
        prompt = session.get_system_prompt()
        
        self.assertIn("PLAN FIRST", prompt)
        self.assertIn("IDEMPOTENCY", prompt)
        self.assertIn("SCENE AWARENESS", prompt)
        self.assertIn("DATA ANALYSIS", prompt)
        self.assertIn("SDK FIRST", prompt)

if __name__ == '__main__':
    unittest.main()

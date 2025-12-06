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

from blender_assistant_mcp.agent_manager import AgentTools
from blender_assistant_mcp.core import AssistantSession
from blender_assistant_mcp.tool_manager import ToolManager

class TestAgentTools(unittest.TestCase):
    def setUp(self):
        # Mock LLM client
        self.mock_llm = MagicMock()
        self.mock_llm.chat.return_value = [{"content": "I am the specialist."}]
        
        self.agent_tools = AgentTools(llm_client=self.mock_llm)
        
    def test_consult_specialist_prompt(self):
        """Verify that consult_specialist generates the correct prompt."""
        print("\nTesting consult_specialist prompt generation...")
        
        # Consult Modeler
        response = self.agent_tools.consult_specialist("MODELER", "Bevel the cube", focus_object="Cube")
        
        # Check that LLM was called
        self.mock_llm.chat.assert_called_once()
        
        # Inspect the messages passed to LLM
        call_args = self.mock_llm.chat.call_args
        messages = call_args[0][0] # First arg is messages list
        
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]
        
        print(f"System Prompt:\n{system_prompt[:200]}...")
        
        self.assertIn("You are the Modeler Specialist", system_prompt)
        self.assertIn("FOCUS OBJECT: Cube", system_prompt)
        self.assertIn("Mesh data unfolded", system_prompt)
        self.assertEqual(user_prompt, "Bevel the cube")
        
        print("PASS: Prompt contains correct role and context.")

if __name__ == "__main__":
    unittest.main()

import sys
import os
import unittest
import json
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock dependencies
sys.modules["bpy"] = MagicMock()
sys.modules["bmesh"] = MagicMock()
sys.modules["mathutils"] = MagicMock()

from blender_assistant_mcp.agent_manager import AgentTools

class TestAgenticTools(unittest.TestCase):
    def setUp(self):
        # Mock LLM client
        self.mock_llm = MagicMock()
        self.agent_tools = AgentTools(llm_client=self.mock_llm)
        
        # Mock ResearchAgent
        self.agent_tools.research_agent = MagicMock()
        self.agent_tools.research_agent.research.return_value = "Research Summary"
        
    def test_consult_specialist(self):
        """Verify consult_specialist prompt and flow."""
        print("\nTesting consult_specialist...")
        
        # Mock LLM response
        expected_response = {
            "thought": "I will bevel the cube.",
            "code": "bpy.ops.mesh.bevel()",
            "expected_changes": {
                "modified": [{"name": "Cube", "changes": "Added Bevel Modifier"}]
            }
        }
        self.mock_llm.chat_completion.return_value = {"content": json.dumps(expected_response)}
        
        response = self.agent_tools.consult_specialist("MODELER", "Bevel the cube", focus_object="Cube")
        
        # Verify prompt
        call_args = self.mock_llm.chat_completion.call_args
        messages = call_args[1]["messages"] # kwargs['messages']
        system_prompt = messages[0]["content"]
        
        self.assertIn("You are the MODELER Specialist", system_prompt)
        self.assertIn("expected_changes", system_prompt)
        self.assertIn("FOCUS OBJECT: Cube", system_prompt)
        
        # Verify response
        self.assertIn("expected_changes", response)
        print("PASS: consult_specialist flow verified.")
        
    def test_research_topic(self):
        """Verify research_topic integration."""
        print("\nTesting research_topic...")
        
        result = self.agent_tools.research_topic("Blender Geometry Nodes")
        
        self.agent_tools.research_agent.research.assert_called_with("Blender Geometry Nodes")
        self.assertEqual(result, "Research Summary")
        print("PASS: research_topic integration verified.")

if __name__ == "__main__":
    unittest.main()

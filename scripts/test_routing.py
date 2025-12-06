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
from blender_assistant_mcp.agent_manager import AgentManager

class TestRouting(unittest.TestCase):
    def setUp(self):
        self.tool_manager = MagicMock(spec=ToolManager)
        self.tool_manager.get_enabled_tools.return_value = ["switch_agent", "get_scene_info"]
        self.tool_manager.get_compact_tool_list.return_value = "Tools: ..."
        self.tool_manager.get_system_prompt_hints.return_value = "Hints: ..."
        
        self.session = AssistantSession("test-model", self.tool_manager)
        
    def test_initial_agent(self):
        """Verify initial agent is Generalist."""
        print("\nTesting Initial Agent...")
        agent = self.session.agent_manager.get_active_agent()
        self.assertEqual(agent.name, "GENERALIST")
        print("PASS: Initial agent is GENERALIST")
        
    def test_switch_agent(self):
        """Verify switching agent updates state and context."""
        print("\nTesting Switch Agent...")
        
        # Switch to Modeler
        result = self.session.agent_manager.switch_agent("MODELER", focus_object="Cube")
        print(f"Switch Result: {result}")
        
        agent = self.session.agent_manager.get_active_agent()
        self.assertEqual(agent.name, "MODELER")
        self.assertEqual(self.session.agent_manager.context_manager.focus_object_name, "Cube")
        print("PASS: Switched to MODELER with focus object")
        
        # Check System Prompt
        prompt = self.session.get_system_prompt()
        self.assertIn("CURRENT AGENT: MODELER", prompt)
        self.assertIn("Mesh data unfolded", prompt)
        print("PASS: System prompt reflects MODELER context")
        
    def test_switch_animator(self):
        """Verify switching to Animator unfolds animation data."""
        print("\nTesting Switch Animator...")
        
        self.session.agent_manager.switch_agent("ANIMATOR", focus_object="Armature")
        
        prompt = self.session.get_system_prompt()
        self.assertIn("CURRENT AGENT: ANIMATOR", prompt)
        self.assertIn("Animation data unfolded", prompt)
        print("PASS: System prompt reflects ANIMATOR context")

if __name__ == "__main__":
    unittest.main()

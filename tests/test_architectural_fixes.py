import unittest
import bpy
import json
from blender_assistant_mcp.blender_tools import assistant_help, execute_code
from blender_assistant_mcp.core import AssistantSession
from blender_assistant_mcp.tool_manager import ToolManager

class TestArchitecturalFixes(unittest.TestCase):
    def setUp(self):
        # Clear scene
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
    def test_assistant_help_flexibility(self):
        """Verify assistant_help handles various input formats."""
        
        # 1. Test combined path with prefix
        res = assistant_help(tool="assistant_sdk.polyhaven.search/download")
        self.assertNotIn("error", res)
        self.assertIn("results", res)
        aliases = [r["alias"] for r in res["results"]]
        self.assertIn("polyhaven.search", aliases)
        self.assertIn("polyhaven.download", aliases)
        
        # 2. Test fuzzy match
        res = assistant_help(tool="polyhaven")
        self.assertNotIn("error", res)
        aliases = [r["alias"] for r in res["results"]]
        self.assertIn("polyhaven.search", aliases)
        
        # 3. Test exact match
        res = assistant_help(tool="blender.create_object")
        self.assertNotIn("error", res)
        self.assertEqual(res["results"][0]["alias"], "blender.create_object")

    def test_scene_awareness_injection(self):
        """Verify AssistantSession injects scene changes into history."""
        
        # Setup session
        tm = ToolManager()
        session = AssistantSession("test-model", tm)
        
        # Mock a tool call that creates an object via execute_code
        # We simulate the process_response -> execute_next_tool flow manually
        
        # 1. Queue a tool call
        from blender_assistant_mcp.core import ToolCall
        code = "import bpy; bpy.ops.mesh.primitive_cube_add(location=(0,0,0))"
        session.tool_queue.append(ToolCall(tool="execute_code", args={"code": code}))
        
        # 2. Execute
        result = session.execute_next_tool()
        self.assertTrue(result["success"])
        
        # 3. Check history for SCENE UPDATES
        # History should have:
        # - tool result
        # - system message with scene updates
        
        self.assertTrue(len(session.history) >= 2)
        last_msg = session.history[-1]
        
        # The last message might be the system update OR the tool result depending on order
        # My implementation adds tool result THEN checks changes.
        # So the LAST message should be the system update if changes were detected.
        
        print("History:", session.history)
        
        has_update = False
        for msg in session.history:
            if msg["role"] == "system" and "SCENE UPDATES" in msg["content"]:
                has_update = True
                self.assertIn("Cube", msg["content"])
                
        self.assertTrue(has_update, "Scene updates were not injected into history")

if __name__ == '__main__':
    unittest.main()

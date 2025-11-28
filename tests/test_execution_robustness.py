import unittest
import bpy
from blender_assistant_mcp.blender_tools import execute_code

class TestExecutionRobustness(unittest.TestCase):
    def setUp(self):
        # Clear scene
        bpy.ops.wm.read_factory_settings(use_empty=True)
        
    def test_sdk_move_to_collection(self):
        # Create an object
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        cube.name = "TestCube"
        
        # Verify it's in the master collection
        self.assertTrue(cube.name in bpy.context.collection.objects)
        
        # Use execute_code with the SDK (the new "Right Way")
        code = """
import bpy
# We don't need to import assistant_sdk, it's injected
result = assistant_sdk.blender.move_to_collection(
    object_names=["TestCube"], 
    collection_name="SDKCollection"
)
"""
        result = execute_code(code)
        
        # Check result
        self.assertTrue(result.get("success"), f"Execution failed: {result.get('error')}")
        
        # Verify collection exists
        self.assertIn("SDKCollection", bpy.data.collections)
        new_col = bpy.data.collections["SDKCollection"]
        
        # Verify object is in new collection
        self.assertIn("TestCube", new_col.objects)
        
        # Verify object is NOT in old collection
        self.assertEqual(len(cube.users_collection), 1)
        self.assertEqual(cube.users_collection[0], new_col)

if __name__ == '__main__':
    unittest.main()

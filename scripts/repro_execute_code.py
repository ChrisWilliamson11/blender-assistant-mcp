import unittest
from unittest.mock import MagicMock
import sys
import os

# Mock bpy and other dependencies
class MockProperty:
    def __init__(self, **kwargs): pass

mock_bpy = MagicMock()
mock_bpy.props.StringProperty = MockProperty
mock_bpy.props.EnumProperty = MockProperty
mock_bpy.props.CollectionProperty = MockProperty
mock_bpy.types.PropertyGroup = object
mock_bpy.context = MagicMock()
mock_bpy.utils.register_class = MagicMock()
mock_bpy.utils.unregister_class = MagicMock()

sys.modules["bpy"] = mock_bpy
sys.modules["mathutils"] = MagicMock()

# Add current directory to path
sys.path.append(os.getcwd())

# Import module under test
from blender_assistant_mcp.tools import blender_tools

class TestExecuteCode(unittest.TestCase):
    def test_assistant_sdk_access(self):
        # Code that uses assistant_sdk
        code = """
import bpy
# Try to access assistant_sdk
print(f"SDK Type: {type(assistant_sdk)}")
result = "Success"
"""
        # Execute
        res = blender_tools.execute_code(code)
        print(f"DEBUG: Result: {res}")
        
        # Verify
        self.assertIn("Success", res.get("result", ""))
        self.assertNotIn("NameError", res.get("output", ""))
        self.assertNotIn("traceback", res)

if __name__ == "__main__":
    unittest.main()

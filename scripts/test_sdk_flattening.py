
import sys
import unittest
from unittest.mock import MagicMock
import os

# 1. Setup Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 2. Mock 'bpy' BEFORE import
if "bpy" not in sys.modules:
    sys.modules["bpy"] = MagicMock()

# 3. Import the module under test
# We intercept the 'tools' package to avoid loading real tools
if "blender_assistant_mcp.tools" not in sys.modules:
    sys.modules["blender_assistant_mcp.tools"] = MagicMock()
    sys.modules["blender_assistant_mcp.tools.tool_registry"] = MagicMock()

from blender_assistant_mcp import assistant_sdk

class TestFlattenedSDK(unittest.TestCase):
    def setUp(self):
        # Overwrite the tool_registry imported by assistant_sdk
        self.mock_registry = MagicMock()
        assistant_sdk.tool_registry = self.mock_registry
        
        # Reset Singleton if exists
        assistant_sdk._assistant_sdk = None
        
        self.tools_list = [
            {"name": "unique_tool", "category": "CategoryA", "inputSchema": {}},
            {"name": "ambiguous_tool", "category": "CategoryA", "inputSchema": {}},
            {"name": "ambiguous_tool", "category": "CategoryB", "inputSchema": {}},
        ]
        self.mock_registry.get_tools_list.return_value = self.tools_list
        self.mock_registry.execute_tool = MagicMock(return_value={"status": "success"})

    def test_flattened_access(self):
        sdk = assistant_sdk.get_assistant_sdk()
        
        # 1. Test Namespace Access
        self.assertTrue(hasattr(sdk, "category_a"))
        self.assertTrue(hasattr(sdk.category_a, "unique_tool"))
        
        # 2. Test Unique Flattened Access
        self.assertTrue(hasattr(sdk, "unique_tool"))
        sdk.unique_tool(arg=1)
        # Check if execute_tool was called on the registry
        self.mock_registry.execute_tool.assert_called_with("unique_tool", {"arg": 1})
        
    def test_ambiguous_access(self):
        sdk = assistant_sdk.get_assistant_sdk()
        
        # 2. Ambiguous tool should raise helpful AttributeError
        with self.assertRaises(AttributeError) as cm:
            _ = sdk.ambiguous_tool
        
        err_msg = str(cm.exception)
        print(f"Ambiguous Error: {err_msg}")
        self.assertIn("ambiguous", err_msg)
        self.assertIn("multiple namespaces", err_msg)
        
    def test_missing_tool(self):
        sdk = assistant_sdk.get_assistant_sdk()
        with self.assertRaises(AttributeError) as cm:
            _ = sdk.non_existent_tool
            
        err_msg = str(cm.exception)
        print(f"Missing Error: {err_msg}")
        self.assertIn("Available namespaces", err_msg)
        self.assertIn("category_a", err_msg)

if __name__ == "__main__":
    unittest.main()

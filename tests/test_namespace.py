import sys
from unittest.mock import MagicMock

# Mock modules
mock_bpy = MagicMock()
mock_mathutils = MagicMock()
mock_bmesh = MagicMock()
mock_numpy = MagicMock()

sys.modules["bpy"] = mock_bpy
sys.modules["mathutils"] = mock_mathutils
sys.modules["bmesh"] = mock_bmesh
sys.modules["numpy"] = mock_numpy

# Setup mathutils classes
mock_mathutils.Vector = MagicMock
mock_mathutils.Matrix = MagicMock
mock_mathutils.Euler = MagicMock
mock_mathutils.Color = MagicMock

import unittest
from blender_assistant_mcp.blender_tools import execute_code, _get_code_namespace

class TestNamespaceInjection(unittest.TestCase):
    def test_namespace_has_batteries(self):
        # Force namespace initialization
        ns = _get_code_namespace()
        
        # Verify keys exist in namespace
        self.assertIn("Vector", ns)
        self.assertIn("Matrix", ns)
        self.assertIn("Euler", ns)
        self.assertIn("Color", ns)
        self.assertIn("bmesh", ns)
        self.assertIn("numpy", ns)
        self.assertIn("np", ns)

if __name__ == '__main__':
    unittest.main()

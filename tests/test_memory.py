import unittest
import os
import json
import tempfile
import shutil
import sys
from unittest.mock import MagicMock

# Mock bpy before importing anything from the package
sys.modules["bpy"] = MagicMock()

from blender_assistant_mcp.memory import MemoryManager

class TestMemoryManager(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()
        self.memory_file = os.path.join(self.test_dir, "test_memory.json")
        self.memory_manager = MemoryManager(self.memory_file)

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_initialization(self):
        """Test that memory manager initializes with empty data."""
        self.assertEqual(self.memory_manager.data["user_preferences"], {})
        self.assertEqual(self.memory_manager.data["facts"], [])

    def test_remember_preference(self):
        """Test storing and retrieving a preference."""
        self.memory_manager.remember_preference("theme", "dark")
        self.assertEqual(self.memory_manager.data["user_preferences"]["theme"], "dark")
        
        # Verify persistence
        new_manager = MemoryManager(self.memory_file)
        self.assertEqual(new_manager.data["user_preferences"]["theme"], "dark")

    def test_remember_fact(self):
        """Test storing and retrieving a fact."""
        self.memory_manager.remember_fact("The sky is blue", "science")
        self.assertEqual(len(self.memory_manager.data["facts"]), 1)
        self.assertEqual(self.memory_manager.data["facts"][0]["content"], "The sky is blue")
        self.assertEqual(self.memory_manager.data["facts"][0]["category"], "science")

    def test_get_context(self):
        """Test context string generation."""
        self.memory_manager.remember_preference("unit", "metric")
        self.memory_manager.remember_fact("User likes cubes")
        
        context = self.memory_manager.get_context()
        self.assertIn("User Preferences:", context)
        self.assertIn("unit: metric", context)
        self.assertIn("Learned Facts:", context)
        self.assertIn("User likes cubes", context)

    def test_clear(self):
        """Test clearing memory."""
        self.memory_manager.remember_preference("foo", "bar")
        self.memory_manager.clear()
        self.assertEqual(self.memory_manager.data["user_preferences"], {})

if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import MagicMock
import sys

# Mock bpy
class MockProperty:
    def __init__(self, **kwargs):
        pass

class MockCollectionProperty(list):
    def __init__(self, type=None):
        self.type = type
        
    def add(self):
        item = self.type()
        self.append(item)
        return item
        
    def clear(self):
        super().clear()

class MockScene:
    def __init__(self):
        self.assistant_tasks = MockCollectionProperty(type=MockTaskItem)

class MockTaskItem:
    def __init__(self):
        self.name = ""
        self.status = "TODO"
        self.notes = ""

class MockContext:
    def __init__(self):
        self.scene = MockScene()

mock_bpy = MagicMock()
mock_bpy.props.StringProperty = MockProperty
mock_bpy.props.EnumProperty = MockProperty
mock_bpy.props.CollectionProperty = MockProperty
mock_bpy.types.PropertyGroup = object
mock_bpy.context = MockContext()
mock_bpy.utils.register_class = MagicMock()
mock_bpy.utils.unregister_class = MagicMock()

sys.modules["bpy"] = mock_bpy

# Add current directory to path to find the package
import os
sys.path.append(os.getcwd())

# Now import the module under test
from blender_assistant_mcp.tools import task_tools

class TestTaskTools(unittest.TestCase):
    def setUp(self):
        # Reset scene state
        mock_bpy.context.scene = MockScene()

    def test_add_task(self):
        res = task_tools.add_task("Test Task 1")
        self.assertTrue(res["success"])
        self.assertEqual(res["task_index"], 0)
        
        tasks = task_tools.list_tasks()
        self.assertEqual(len(tasks["tasks"]), 1)
        self.assertEqual(tasks["tasks"][0]["name"], "Test Task 1")
        self.assertEqual(tasks["tasks"][0]["status"], "TODO")

    def test_update_task(self):
        task_tools.add_task("Test Task 2")
        
        res = task_tools.update_task(0, "IN_PROGRESS", "Started working")
        self.assertTrue(res["success"])
        
        tasks = task_tools.list_tasks()
        self.assertEqual(tasks["tasks"][0]["status"], "IN_PROGRESS")
        self.assertEqual(tasks["tasks"][0]["notes"], "Started working")
        
        # Test invalid status
        res = task_tools.update_task(0, "INVALID_STATUS")
        self.assertFalse(res["success"])

    def test_clear_tasks(self):
        task_tools.add_task("Task A")
        task_tools.add_task("Task B")
        
        res = task_tools.clear_tasks()
        self.assertTrue(res["success"])
        
        res = task_tools.list_tasks()
        self.assertIn("message", res) # "No tasks in plan."

if __name__ == "__main__":
    unittest.main()

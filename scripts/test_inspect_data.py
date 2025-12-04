import sys
import os
import json
from unittest.mock import MagicMock

# Add package to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock bpy
mock_bpy = MagicMock()
sys.modules["bpy"] = mock_bpy
mock_bpy.utils.user_resource.return_value = "."

# Mock bmesh
mock_bmesh = MagicMock()
sys.modules["bmesh"] = mock_bmesh

# Mock mathutils
mock_mathutils = MagicMock()
sys.modules["mathutils"] = mock_mathutils

# Mock bpy data structures for testing
class MockProperty:
    def __init__(self, identifier):
        self.identifier = identifier

class MockType:
    def __init__(self, name, props):
        self.name = name
        self.properties = [MockProperty(p) for p in props]

class MockObject:
    def __init__(self, name, rna_type_name, props):
        self.name = name
        self.rna_type = MockType(rna_type_name, props.keys())
        for k, v in props.items():
            setattr(self, k, v)
    
    def __repr__(self):
        return f"<MockObject {self.name}>"

# Setup mock data
cube = MockObject("Cube", "Object", {
    "location": [1.0, 2.0, 3.0],
    "type": "MESH",
    "hide_viewport": False
})
camera = MockObject("Camera", "Object", {
    "location": [0.0, -10.0, 0.0],
    "type": "CAMERA",
    "hide_viewport": False
})

# Mock bpy.data.objects
mock_objects = MagicMock()
mock_objects.__iter__.return_value = [cube, camera]
mock_objects.__getitem__.side_effect = lambda key: next((o for o in [cube, camera] if o.name == key), None)
mock_objects.get.side_effect = lambda key, default=None: next((o for o in [cube, camera] if o.name == key), default)
mock_bpy.data.objects = mock_objects

# Import tools after mocking
from blender_assistant_mcp.tools.blender_tools import inspect_data, search_data, _get_code_namespace

# Inject mock bpy into the tool's namespace
_get_code_namespace()["bpy"] = mock_bpy

def test_inspect_data():
    print("\n=== Testing inspect_data ===")
    # Test inspecting Cube
    result = inspect_data("bpy.data.objects['Cube']", depth=1)
    print(json.dumps(result, indent=2))
    
    if "error" in result:
        print("FAIL: Error in inspect_data")
        return

    data = result.get("data", {})
    if data.get("name") == "Cube" and data.get("location") == [1.0, 2.0, 3.0]:
        print("PASS: Cube inspected correctly")
    else:
        print("FAIL: Cube data mismatch")

def test_search_data():
    print("\n=== Testing search_data ===")
    # Test searching for MESH objects
    result = search_data("bpy.data.objects", filter_props={"type": "MESH"})
    print(json.dumps(result, indent=2))

    if result.get("count") == 1 and result["matches"][0]["name"] == "Cube":
        print("PASS: Found Cube by type=MESH")
    else:
        print("FAIL: Search failed")

    # Test searching for CAMERA objects
    result = search_data("bpy.data.objects", filter_props={"type": "CAMERA"})
    if result.get("count") == 1 and result["matches"][0]["name"] == "Camera":
        print("PASS: Found Camera by type=CAMERA")
    else:
        print("FAIL: Search failed")

def test_get_scene_info_detailed():
    print("\n=== Testing get_scene_info(detailed=True) ===")
    from blender_assistant_mcp.tools.blender_tools import get_scene_info
    
    # Mock context for get_scene_info
    mock_bpy.context.scene.collection.children = []
    mock_bpy.context.scene.collection.objects = [cube, camera]
    mock_bpy.context.view_layer.active_layer_collection.collection = mock_bpy.context.scene.collection
    
    # We need to mock collection objects properly for the tool to work
    # The tool iterates root collections. Let's mock a root collection.
    root_col = MagicMock()
    root_col.name = "Collection"
    root_col.objects = [cube]
    root_col.children = []
    mock_bpy.context.scene.collection.children = [root_col]
    
    result = get_scene_info(detailed=True)
    
    # Check if nodes have 'data' field
    nodes = result.get("outliner", {}).get("nodes", [])
    found_data = False
    for node in nodes:
        if node.get("kind") == "object" and node.get("name") == "Cube":
            if "data" in node and node["data"].get("location") == [1.0, 2.0, 3.0]:
                found_data = True
                break
    
    if found_data:
        print("PASS: Detailed info found in get_scene_info")
    else:
        print("FAIL: Detailed info missing")
        print(json.dumps(nodes, indent=2))

if __name__ == "__main__":
    test_inspect_data()
    test_search_data()
    test_get_scene_info_detailed()

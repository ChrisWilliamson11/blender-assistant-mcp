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

# Mock bmesh and mathutils
sys.modules["bmesh"] = MagicMock()
sys.modules["mathutils"] = MagicMock()

# Mock bpy data structures
class MockObject:
    def __init__(self, name):
        self.name = name
        self.type = "MESH"
        self.hide_viewport = False
        self.hide_render = False
        self.location = [0.0, 0.0, 0.0]
        self.rotation_euler = [0.0, 0.0, 0.0]
        self.scale = [1.0, 1.0, 1.0]
        self.dimensions = [2.0, 2.0, 2.0]
        self.parent = None
        self.users_collection = []
        self.data = MagicMock()
        self.data.name = "CubeData"
        self.material_slots = []
        self.modifiers = []
        self.constraints = []
        self.vertex_groups = []
        
        # RNA mocking
        self.rna_type = MagicMock()
        self.rna_type.name = "Object"
        self.rna_type.properties = []

cube = MockObject("Cube")
# Mock bpy.data.objects as a MagicMock that behaves like a list/dict
mock_objects = MagicMock()
mock_objects.__iter__.return_value = [cube]
mock_objects.__getitem__.side_effect = lambda key: cube if key == "Cube" else None
mock_objects.get.side_effect = lambda key, default=None: cube if key == "Cube" else default
mock_bpy.data.objects = mock_objects

# Import tools
from blender_assistant_mcp.tools.blender_tools import get_object_info, _serialize_rna
from blender_assistant_mcp.scene_watcher import SceneWatcher

def test_get_object_info_consistency():
    print("\n=== Testing get_object_info consistency ===")
    info = get_object_info("Cube")
    
    if "data" in info:
        print("PASS: 'data' field present in get_object_info")
        # Check if it looks like AST
        if info["data"].get("_type") == "Object":
            print("PASS: 'data' field has correct AST structure")
        else:
            print(f"FAIL: Unexpected data structure: {info['data']}")
    else:
        print("FAIL: 'data' field missing")

def test_scene_watcher_structured():
    print("\n=== Testing SceneWatcher structured output ===")
    watcher = SceneWatcher()
    
    # Initial capture
    mock_bpy.context.selected_objects = []
    mock_bpy.context.active_object = None
    mock_bpy.context.mode = "OBJECT"
    watcher.capture_state()
    
    # Simulate adding a new object
    new_obj = MockObject("Sphere")
    mock_bpy.data.objects = [cube, new_obj]
    
    # Check changes
    changes = watcher.get_changes()
    print(f"Changes: {json.dumps(changes, indent=2)}")
    
    if "added" in changes and len(changes["added"]) == 1:
        print("PASS: Detected added object")
        added_data = changes["added"][0]
        # Check for summary fields
        if added_data.get("name") == "Sphere" and "location" in added_data and "type" in added_data:
            print("PASS: Added object has summary structure")
        else:
            print(f"FAIL: Added object missing summary structure: {added_data}")
            
        # Ensure no heavy fields
        if "data" in added_data or "rna_type" in added_data:
             print(f"FAIL: Summary contains heavy fields: {added_data.keys()}")
        else:
             print("PASS: Summary is concise")
    else:
        print("FAIL: Did not detect added object")

    # Test consume_changes JSON format
    json_output = watcher.consume_changes()
    try:
        parsed = json.loads(json_output)
        if "added" in parsed:
            print("PASS: consume_changes returns valid JSON")
        else:
            print("FAIL: consume_changes JSON missing expected keys")
    except json.JSONDecodeError:
        print(f"FAIL: consume_changes returned invalid JSON: {json_output}")

if __name__ == "__main__":
    test_get_object_info_consistency()
    test_scene_watcher_structured()

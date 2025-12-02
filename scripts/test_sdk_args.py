import sys
import os

# Mock tool registry
class MockToolRegistry:
    def __init__(self):
        self.tools = [
            {
                "name": "web_search",
                "category": "web",
                "description": "Search the web",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {"type": "integer"}
                    }
                }
            },
            {
                "name": "create_cube",
                "category": "blender",
                "description": "Create a cube",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "size": {"type": "number"},
                        "location": {"type": "array"}
                    }
                }
            }
        ]
        self.last_call = None

    def get_tools_list(self):
        return self.tools

    def execute_tool(self, name, kwargs):
        self.last_call = (name, kwargs)
        return {"status": "success", "kwargs": kwargs}

# Mock the package structure
import sys
from types import ModuleType

# Create mock registry module
registry_mod = ModuleType('blender_assistant_mcp.tools.tool_registry')
mock_registry = MockToolRegistry()
registry_mod.get_tools_list = mock_registry.get_tools_list
registry_mod.execute_tool = mock_registry.execute_tool
# Also attach the mock object for inspection in the test
registry_mod.mock_instance = mock_registry

# Create mock tools package
tools_pkg = ModuleType('blender_assistant_mcp.tools')
tools_pkg.tool_registry = registry_mod

# Register in sys.modules
sys.modules['blender_assistant_mcp'] = ModuleType('blender_assistant_mcp')
sys.modules['blender_assistant_mcp.tools'] = tools_pkg
sys.modules['blender_assistant_mcp.tools.tool_registry'] = registry_mod

# Now we can import the SDK module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "blender_assistant_mcp.assistant_sdk", 
    "h:/blender-assistant-mcp/blender_assistant_mcp/assistant_sdk.py"
)
assistant_sdk = importlib.util.module_from_spec(spec)
sys.modules['blender_assistant_mcp.assistant_sdk'] = assistant_sdk
spec.loader.exec_module(assistant_sdk)

# Helper to access the mock for assertions
assistant_sdk.tool_registry = registry_mod

def test_sdk_args():
    sdk = assistant_sdk.get_assistant_sdk()
    mock_inst = registry_mod.mock_instance
    
    print("Testing positional args...")
    # Test 1: Positional args
    sdk.web.web_search("python news", 5)
    last_call = mock_inst.last_call
    assert last_call[0] == "web_search"
    assert last_call[1] == {"query": "python news", "num_results": 5}
    print("PASS: Positional args mapped correctly")

    # Test 2: Keyword args
    sdk.web.web_search(query="blender", num_results=3)
    last_call = mock_inst.last_call
    assert last_call[1] == {"query": "blender", "num_results": 3}
    print("PASS: Keyword args preserved")

    # Test 3: Mixed args
    sdk.web.web_search("mixed", num_results=10)
    last_call = mock_inst.last_call
    assert last_call[1] == {"query": "mixed", "num_results": 10}
    print("PASS: Mixed args handled correctly")
    
    # Test 4: Too many args
    try:
        sdk.web.web_search("a", 1, "extra")
        print("FAIL: Should have raised TypeError for too many args")
    except TypeError as e:
        print(f"PASS: Caught expected TypeError: {e}")

    # Test 5: Duplicate args
    try:
        sdk.web.web_search("query", query="duplicate")
        print("FAIL: Should have raised TypeError for duplicate args")
    except TypeError as e:
        print(f"PASS: Caught expected TypeError: {e}")

if __name__ == "__main__":
    test_sdk_args()

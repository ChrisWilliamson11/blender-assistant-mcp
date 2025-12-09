
import sys
import os

# Mock tool registry
_TOOLS = {
    "task_add": {
        "description": "Add a task",
        "inputSchema": {},
        "category": "Planning"
    }
}

def get_tool_schema(name):
    if name not in _TOOLS: return None
    tool = _TOOLS[name]
    return {
        "name": name,
        "description": tool["description"],
        "inputSchema": tool["inputSchema"],
        "category": tool.get("category", "Other"),
    }

def test_sdk_help():
    target_tool = "task_add"
    found_tool = _TOOLS[target_tool]
    
    print(f"Testing sdk_help logic for {target_tool}...")
    try:
        # BUG: Accessing "name" on the tool dict, which doesn't have it
        tool_name_key = found_tool["name"]
        print(f"Success: {tool_name_key}")
    except KeyError as e:
        print(f"Caught expected KeyError: {e}")
    except Exception as e:
        print(f"Caught unexpected exception: {e}")

if __name__ == "__main__":
    test_sdk_help()

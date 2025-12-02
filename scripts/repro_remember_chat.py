import sys
import os
from typing import Dict, Any

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock bpy
from unittest.mock import MagicMock
sys.modules["bpy"] = MagicMock()

from blender_assistant_mcp.tools import tool_registry

# Mock remember_learning function
def mock_remember_learning(topic: str, insight: str) -> Dict[str, Any]:
    print(f"CALLED remember_learning with topic='{topic}', insight='{insight}'")
    return {"success": True}

# Register the tool as it is in memory_tools.py
tool_registry.register_tool(
    "remember_learning",
    mock_remember_learning,
    "Record a technical learning.",
    {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "insight": {"type": "string"}
        },
        "required": ["topic", "insight"]
    },
    category="Memory"
)

def test_mismatch():
    print("\n--- Testing Argument Mismatch ---")
    # Simulate what LLM sends based on logs
    args = {"summary": "User requested to stack 5 boxes..."}
    
    print(f"Calling execute_tool with: {args}")
    result = tool_registry.execute_tool("remember_learning", args)
    
    print(f"Result: {result}")
    
    if "error" in result:
        print("PASS: Error returned as expected")
    else:
        print("FAIL: No error returned")

    print("\n--- Testing Correct Arguments ---")
    args_correct = {"topic": "Chat Summary", "insight": "User requested to stack 5 boxes..."}
    print(f"Calling execute_tool with: {args_correct}")
    result = tool_registry.execute_tool("remember_learning", args_correct)
    print(f"Result: {result}")
    
    if result.get("success"):
        print("PASS: Tool executed successfully")
    else:
        print("FAIL: Tool execution failed")

if __name__ == "__main__":
    test_mismatch()

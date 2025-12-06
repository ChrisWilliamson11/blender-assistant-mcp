import sys
import os
from unittest.mock import MagicMock

# Mock dependencies before importing core
sys.modules['bpy'] = MagicMock()
sys.modules['blender_assistant_mcp.tools'] = MagicMock()
sys.modules['blender_assistant_mcp.tool_manager'] = MagicMock()
sys.modules['blender_assistant_mcp.memory'] = MagicMock()
sys.modules['blender_assistant_mcp.scene_watcher'] = MagicMock()
sys.modules['blender_assistant_mcp.agent_manager'] = MagicMock()

# Import AssistantSession
# We need to add the parent directory to path so we can import from blender_assistant_mcp
sys.path.append('h:\\blender-assistant-mcp')
from blender_assistant_mcp.core import AssistantSession

# Mock the components
mock_tool_manager = MagicMock()
mock_tool_manager.get_enabled_tools.return_value = ["some_tool"]
mock_tool_manager.get_compact_tool_list.return_value = "- mock_tool(arg): description"
mock_tool_manager.get_system_prompt_hints.return_value = "- sdk.mock_method(): description"

session = AssistantSession("model-name", mock_tool_manager)
session.memory_manager.get_context.return_value = "Mock Memory Context"
session.scene_watcher.consume_changes.return_value = "Mock Scene Changes"
# Mock _load_protocol to avoid file reading issues
session._load_protocol = MagicMock(return_value="# Mock Protocol")

prompt = session.get_system_prompt()

print("--- SYSTEM PROMPT START ---")
print(prompt)
print("--- SYSTEM PROMPT END ---")

# Verification checks
required_strings = [
    "Mock Memory Context",
    "# Mock Protocol",
    "MCP TOOLS",
    "- mock_tool(arg): description",
    "SDK TOOLS",
    "- sdk.mock_method(): description",
    "Mock Scene Changes"
]

missing = [s for s in required_strings if s not in prompt]
if missing:
    print(f"\nFAILED: Missing sections: {missing}")
    sys.exit(1)
else:
    print("\nSUCCESS: All sections present.")

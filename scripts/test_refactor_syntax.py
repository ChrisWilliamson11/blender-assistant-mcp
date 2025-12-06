
import sys
import os

# Add repo root to path
sys.path.append(os.getcwd())

# Mock bpy
from unittest.mock import MagicMock
import sys
sys.modules["bpy"] = MagicMock()
import bpy
bpy.types.PropertyGroup = object # Needs to be a class for inheritance
bpy.props.StringProperty = MagicMock()
bpy.props.EnumProperty = MagicMock()
bpy.props.CollectionProperty = MagicMock()
bpy.utils = MagicMock()
bpy.context = MagicMock()

try:
    print("Importing tool_manager...")
    from blender_assistant_mcp.tool_manager import ToolManager
    tm = ToolManager()
    print("ToolManager instantiated.")
    
    print("Checking tool roles...")
    manager_tools = tm.get_enabled_tools_for_role("MANAGER")
    worker_tools = tm.get_enabled_tools_for_role("TASK_AGENT")
    print(f"Manager Tools: {len(manager_tools)}")
    print(f"Worker Tools: {len(worker_tools)}")
    
    print("Importing agent_manager...")
    from blender_assistant_mcp.agent_manager import AgentTools
    print("AgentTools imported.")
    
    print("Importing core...")
    from blender_assistant_mcp.core import AssistantSession
    print("AssistantSession imported.")
    
    # Mocking Blender context for session init if needed?
    # AssistantSession init tries to read prefs. We might fail there if not inside Blender.
    # But let's see if we get that far.
    
    print("SUCCESS: Syntax checks passed.")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()

import sys
import types
from unittest.mock import MagicMock

# Mock bpy
mock_bpy = MagicMock()
sys.modules["bpy"] = mock_bpy
sys.modules["bpy.types"] = MagicMock()
sys.modules["bpy.props"] = MagicMock()
sys.modules["bpy.utils"] = MagicMock()

# Mock other dependencies
sys.modules["blender_assistant_mcp.tools.tool_registry"] = MagicMock()
sys.modules["blender_assistant_mcp.memory"] = MagicMock()

# Mock AssistantSDK
class MockAssistantSDK:
    def __init__(self):
        self.system = MagicMock()
        self.blender = MagicMock()
        self.polyhaven = MagicMock()
        # Add a dummy method to system to verify it's wrapped
        self.system.consult_specialist = MagicMock()

sys.modules["blender_assistant_mcp.assistant_sdk"] = MagicMock()
sys.modules["blender_assistant_mcp.assistant_sdk"].AssistantSDK = MockAssistantSDK

# Now import blender_tools
# We need to make sure we are importing from the right path
# Assuming the script is run from the root of the repo (h:\blender-assistant-mcp)
try:
    from blender_assistant_mcp.tools import blender_tools
except ImportError:
    # If running from brain dir, we might need to adjust path
    sys.path.append("h:/blender-assistant-mcp")
    from blender_assistant_mcp.tools import blender_tools

def verify():
    # Test SDK Shim
    print("Testing SDK Shim...")
    # Force reset of namespace to trigger rebuild
    blender_tools._CODE_NAMESPACE = None
    
    shim = blender_tools._get_code_namespace()
    sdk = shim["assistant_sdk"]
    
    if hasattr(sdk, "system"):
        print("SUCCESS: assistant_sdk.system is present in shim.")
        # Verify we can access attributes
        if hasattr(sdk.system, "consult_specialist"):
             print("SUCCESS: assistant_sdk.system.consult_specialist is accessible.")
        else:
             print("FAILURE: assistant_sdk.system.consult_specialist is NOT accessible.")
    else:
        print("FAILURE: assistant_sdk.system is MISSING from shim.")
        print(f"Available attributes: {dir(sdk)}")

    # Test assistant_help
    print("\nTesting assistant_help...")
    help_result = blender_tools.assistant_help(tool="system.consult_specialist")
    if "results" in help_result and len(help_result["results"]) > 0:
        print("SUCCESS: assistant_help returned docs for system.consult_specialist.")
        print(f"Usage: {help_result['results'][0]['sdkUsage']}")
    else:
        print("FAILURE: assistant_help returned no results.")
        print(help_result)

if __name__ == "__main__":
    verify()

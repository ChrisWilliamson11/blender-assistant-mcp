import sys
import os
import unittest
from unittest.mock import MagicMock

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock bpy and dependencies
sys.modules["bpy"] = MagicMock()
sys.modules["httpx"] = MagicMock()

# Import SDK and registry
from blender_assistant_mcp import assistant_sdk
from blender_assistant_mcp.tools import tool_registry, web_tools

class TestSDKIntegration(unittest.TestCase):
    def test_search_image_url_in_sdk(self):
        # Register web tools
        web_tools.register()
        
        # Rebuild SDK (it might happen on init, but let's be sure)
        sdk = assistant_sdk.AssistantSDK()
        
        # Check if web namespace exists
        self.assertTrue(hasattr(sdk, "web"))
        
        # Check if search_image_url exists in web namespace
        self.assertTrue(hasattr(sdk.web, "search_image_url"))
        
        # Check docstring
        method = sdk.web.search_image_url
        self.assertIn("Search for direct image URLs", method.__doc__)
        
        print("\nSDK Integration Verified: assistant_sdk.web.search_image_url exists.")

if __name__ == "__main__":
    unittest.main()

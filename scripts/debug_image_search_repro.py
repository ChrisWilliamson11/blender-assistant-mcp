import sys
import os
import json

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock bpy
from unittest.mock import MagicMock
sys.modules["bpy"] = MagicMock()

from blender_assistant_mcp.tools import web_tools

def test_search():
    print("Testing search_image_url with 'random photo'...")
    try:
        result = web_tools.search_image_url("random photo", num_results=5)
        print(json.dumps(result, indent=2))
        
        if result.get("success"):
            urls = result.get("image_urls", [])
            print(f"Found {len(urls)} URLs")
            for url in urls:
                print(f"- {url}")
        else:
            print("Search failed")
            
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    test_search()

import sys
import os
import json
from unittest.mock import MagicMock

# Add parent dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock bpy and preferences
sys.modules["bpy"] = MagicMock()
mock_prefs = MagicMock()
mock_prefs.unsplash_api_key = "mock_unsplash_key"
mock_prefs.pexels_api_key = "mock_pexels_key"

# Mock preferences module
mock_preferences_module = MagicMock()
mock_preferences_module.get_preferences.return_value = mock_prefs
sys.modules["blender_assistant_mcp.preferences"] = mock_preferences_module

# Mock httpx to avoid real API calls and return dummy data
import httpx
original_client = httpx.Client

class MockResponse:
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data
        self.text = json.dumps(json_data)

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("Error", request=None, response=self)

class MockClient:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def get(self, url, params=None, headers=None, **kwargs):
        print(f"MockClient GET: {url} params={params} headers={headers}")
        
        if "unsplash.com" in url:
            if headers.get("Authorization") == "Client-ID mock_unsplash_key":
                return MockResponse(200, {
                    "results": [
                        {"urls": {"regular": "https://images.unsplash.com/photo-1"}},
                        {"urls": {"regular": "https://images.unsplash.com/photo-2"}}
                    ]
                })
            else:
                return MockResponse(401, {})
                
        if "pexels.com" in url:
            if headers.get("Authorization") == "mock_pexels_key":
                return MockResponse(200, {
                    "photos": [
                        {"src": {"large2x": "https://images.pexels.com/photo-1"}},
                        {"src": {"large2x": "https://images.pexels.com/photo-2"}}
                    ]
                })
            else:
                return MockResponse(401, {})
                
        return MockResponse(404, {})

# Patch httpx.Client
httpx.Client = MockClient

from blender_assistant_mcp.tools import web_tools

def test_api_search():
    print("\n--- Testing Unsplash Search ---")
    # Should use Unsplash because key is present (mocked)
    result = web_tools.search_image_url("test query", num_results=2)
    print(json.dumps(result, indent=2))
    
    if result.get("source") == "Unsplash" and len(result.get("image_urls")) == 2:
        print("PASS: Unsplash search used")
    else:
        print("FAIL: Unsplash search not used or failed")

    print("\n--- Testing Pexels Search (Unsplash disabled) ---")
    # Disable Unsplash key to test Pexels fallback
    mock_prefs.unsplash_api_key = ""
    result = web_tools.search_image_url("test query", num_results=2)
    print(json.dumps(result, indent=2))
    
    if result.get("source") == "Pexels" and len(result.get("image_urls")) == 2:
        print("PASS: Pexels search used")
    else:
        print("FAIL: Pexels search not used or failed")

    print("\n--- Testing Scraping Fallback (No keys) ---")
    # Disable both keys
    mock_prefs.pexels_api_key = ""
    # We expect it to try scraping (which will fail in this mock setup or return empty/mocked web search)
    # Since we mocked httpx.Client, web_search will also use MockClient which returns 404 by default
    # So we expect an error or empty result from scraping path
    result = web_tools.search_image_url("test query", num_results=2)
    print(json.dumps(result, indent=2))
    
    if "source" not in result: # Scraping doesn't set source="Unsplash/Pexels"
        print("PASS: Fallback to scraping triggered")
    else:
        print("FAIL: Unexpected source")

if __name__ == "__main__":
    test_api_search()

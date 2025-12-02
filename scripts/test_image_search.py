import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add parent dir to path to import blender_assistant_mcp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock bpy
sys.modules["bpy"] = MagicMock()

# Mock tool_registry
sys.modules["blender_assistant_mcp.tools.tool_registry"] = MagicMock()

from blender_assistant_mcp.tools import web_tools

class TestImageSearch(unittest.TestCase):
    @patch("blender_assistant_mcp.tools.web_tools.web_search")
    @patch("blender_assistant_mcp.tools.web_tools.extract_image_urls")
    def test_search_image_url_direct(self, mock_extract, mock_search):
        # Setup mock search results with direct image links
        mock_search.return_value = {
            "results": [
                {"url": "https://example.com/image1.jpg", "title": "Image 1"},
                {"url": "https://example.com/page1", "title": "Page 1"},
                {"url": "https://example.com/image2.png", "title": "Image 2"},
            ]
        }
        
        # Test
        result = web_tools.search_image_url("test query", num_results=5)
        
        # Verify
        self.assertTrue(result["success"])
        self.assertEqual(len(result["image_urls"]), 2)
        self.assertIn("https://example.com/image1.jpg", result["image_urls"])
        self.assertIn("https://example.com/image2.png", result["image_urls"])
        
        # Verify extract was NOT called (since we found enough direct links? No, we iterate all results)
        # Actually logic: it iterates results. If direct link, adds it. If not, extracts.
        # In this case, we have 2 direct links. 
        # "Page 1" is not direct link. It should trigger extract.
        mock_extract.assert_called_once()

    @patch("blender_assistant_mcp.tools.web_tools.web_search")
    @patch("blender_assistant_mcp.tools.web_tools.extract_image_urls")
    def test_search_image_url_extraction(self, mock_extract, mock_search):
        # Setup mock search results with NO direct image links
        mock_search.return_value = {
            "results": [
                {"url": "https://example.com/page1", "title": "Page 1"},
            ]
        }
        
        # Setup mock extraction
        mock_extract.return_value = {
            "success": True,
            "images": ["https://example.com/extracted1.jpg", "https://example.com/extracted2.jpg"]
        }
        
        # Test
        result = web_tools.search_image_url("test query", num_results=5)
        
        # Verify
        self.assertTrue(result["success"])
        self.assertEqual(len(result["image_urls"]), 2)
        self.assertIn("https://example.com/extracted1.jpg", result["image_urls"])
        
        mock_extract.assert_called_with("https://example.com/page1", max_images=3)

if __name__ == "__main__":
    unittest.main()

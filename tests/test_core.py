import unittest
import json
import sys
from unittest.mock import MagicMock

# Mock bpy before importing core
sys.modules["bpy"] = MagicMock()
from blender_assistant_mcp.core import ResponseParser, ToolCall

class TestResponseParser(unittest.TestCase):
    def test_native_tool_call(self):
        response = {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "create_object",
                            "arguments": {"type": "CUBE"}
                        }
                    }
                ]
            }
        }
        calls = ResponseParser.parse(response)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tool, "create_object")
        self.assertEqual(calls[0].args, {"type": "CUBE"})

    def test_json_markdown_block(self):
        response = {
            "message": {
                "role": "assistant",
                "content": "I will create a cube.\n```json\n{\"name\": \"create_object\", \"arguments\": {\"type\": \"CUBE\"}}\n```"
            }
        }
        calls = ResponseParser.parse(response)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tool, "create_object")
        self.assertEqual(calls[0].args, {"type": "CUBE"})

    def test_python_code_block(self):
        response = {
            "message": {
                "role": "assistant",
                "content": "Here is the code:\n```python\nimport bpy\nbpy.ops.mesh.primitive_cube_add()\n```"
            }
        }
        calls = ResponseParser.parse(response)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tool, "execute_code")
        self.assertIn("import bpy", calls[0].args["code"])

    def test_thinking_model_format(self):
        # Some models output thinking then a JSON block
        response = {
            "message": {
                "role": "assistant",
                "content": "<think>I need to make a sphere</think>\n```json\n{\"tool\": \"create_object\", \"args\": {\"type\": \"SPHERE\"}}\n```"
            }
        }
        calls = ResponseParser.parse(response)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tool, "create_object")
        self.assertEqual(calls[0].args, {"type": "SPHERE"})

if __name__ == '__main__':
    unittest.main()

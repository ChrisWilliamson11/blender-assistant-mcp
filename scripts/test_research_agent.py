import sys
import os
from unittest.mock import MagicMock

# Mock bpy
sys.modules["bpy"] = MagicMock()
sys.modules["bpy"].utils.user_resource.return_value = "."

# Mock ollama_adapter
mock_ollama = MagicMock()
sys.modules["blender_assistant_mcp.ollama_adapter"] = mock_ollama

# Mock tool_registry
mock_registry = MagicMock()
mock_registry._TOOLS = {
    "rag": ["rag_search"],
    "web": ["web_search"],
    "memory": ["search_memory"]
}
mock_registry.execute_tool.return_value = "Mock Tool Output"
sys.modules["blender_assistant_mcp.tools.tool_registry"] = mock_registry

# Add package to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from blender_assistant_mcp.agents.research_agent import ResearchAgent

def test_research_agent():
    print("Testing ResearchAgent...")
    
    # Setup mocks
    mock_ollama.chat_completion.side_effect = [
        # 1. Plan
        {"content": '{"tools": [{"name": "rag_search", "args": {"query": "test"}}]}'},
        # 2. Reflect
        {"content": '{"enough": true}'},
        # 3. Synthesize
        {"content": "Final Answer"}
    ]
    
    agent = ResearchAgent()
    # Mock memory manager
    agent.memory_manager = MagicMock()
    agent.memory_manager.get_abstracts.return_value = ["Abstract 1"]
    
    result = agent.research("Test Topic")
    
    print(f"Result: {result}")
    assert result == "Final Answer"
    print("Test Passed!")

if __name__ == "__main__":
    test_research_agent()

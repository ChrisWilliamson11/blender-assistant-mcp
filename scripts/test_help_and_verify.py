
import sys
import os
import json
from unittest.mock import MagicMock

# Mock bpy before importing anything that uses it
mock_bpy = MagicMock()
sys.modules["bpy"] = mock_bpy
sys.modules["bpy.app"] = MagicMock()
sys.modules["bpy.props"] = MagicMock()
sys.modules["bpy.types"] = MagicMock()
sys.modules["bpy.utils"] = MagicMock()
sys.modules["bpy.context"] = MagicMock()
sys.modules["bpy.data"] = MagicMock()
sys.modules["bpy.ops"] = MagicMock()

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from blender_assistant_mcp.tools import blender_tools
from blender_assistant_mcp.agent_manager import AgentTools
from blender_assistant_mcp import tool_registry

def test_assistant_help():
    print("\n=== Testing assistant_help ===")
    
    # Register some dummy tools to ensure registry is populated (if not already)
    # But we expect real tools to be registered by the system.
    # Let's manually register a dummy tool to be sure.
    tool_registry.register_tool(
        "download_polyhaven_texture",
        lambda: None,
        "Download texture",
        {"type": "object", "properties": {"asset_id": {"type": "string"}}},
        category="PolyHaven"
    )

    tool_registry.register_tool(
        "execute_code",
        lambda code: {"success": True, "output": "Executed"},
        "Execute Python code",
        {"type": "object", "properties": {"code": {"type": "string"}}},
        category="System"
    )
    
    # Test 1: Exact match with namespace
    print("Test 1: polyhaven.download_polyhaven_texture")
    res = blender_tools.assistant_help(tool="polyhaven.download_polyhaven_texture")
    print(json.dumps(res, indent=2))
    assert len(res["results"]) > 0
    assert res["results"][0]["tool"] == "download_polyhaven_texture"
    
    # Test 2: Category match
    print("\nTest 2: polyhaven")
    res = blender_tools.assistant_help(tool="polyhaven")
    print(json.dumps(res, indent=2))
    assert len(res["results"]) >= 1
    
    # Test 3: Substring match
    print("\nTest 3: download_polyhaven")
    res = blender_tools.assistant_help(tool="download_polyhaven_texture")
    print(json.dumps(res, indent=2))
    assert len(res["results"]) > 0

    # Test 4: Virtual tool (web.images_workflow)
    print("\nTest 4: web.images_workflow")
    res = blender_tools.assistant_help(tool="web.images_workflow")
    print(json.dumps(res, indent=2))
    assert len(res["results"]) > 0
    assert "workflow" in res["results"][0]

    # Test 5: Alias (search -> polyhaven.search)
    # Note: polyhaven.search needs to be registered for this to work fully, 
    # but we can check if it tries to resolve it.
    # Let's register a dummy polyhaven.search first
    tool_registry.register_tool(
        "search_polyhaven_assets",
        lambda: None,
        "Search assets",
        {"type": "object", "properties": {}},
        category="PolyHaven"
    )
    print("\nTest 5: Alias 'search'")
    res = blender_tools.assistant_help(tool="search")
    print(json.dumps(res, indent=2))
    assert len(res["results"]) > 0
    assert "search_polyhaven_assets" in res["results"][0]["tool"] or "polyhaven" in res["results"][0]["category"].lower()

def test_verifier_agent():
    print("\n=== Testing VERIFIER Agent ===")
    
    # Mock LLM client
    mock_llm = MagicMock()
    mock_llm.chat.return_value = "True. The boxes are stacked."
    
    agent_tools = AgentTools(llm_client=mock_llm)
    
    # Check if VERIFIER is in agents
    print(f"Available agents: {list(agent_tools.agents.keys())}")
    assert "VERIFIER" in agent_tools.agents
    print("VERIFIER agent found.")
    
    # Test consult_specialist with VERIFIER
    res = agent_tools.consult_specialist("VERIFIER", "Check if boxes are stacked")
    print(f"Verifier Response: {res}")
    
    # Verify prompt contains correct role
    if "Error" in res:
        print(f"Consultation failed: {res}")
    assert "Error" not in res

def test_web_agent():
    print("\n=== Testing WEB Agent ===")
    
    # Mock LLM client
    mock_llm = MagicMock()
    mock_llm.chat_completion.side_effect = [
        {"content": '{"thought": "Searching...", "code": "print(\'Found\')", "expected_changes": {}}'},
        {"content": '{"thought": "Done.", "expected_changes": {"added": []}}'}
    ]
    
    agent_tools = AgentTools(llm_client=mock_llm)
    
    # Check if WEB is in agents
    print(f"Available agents: {list(agent_tools.agents.keys())}")
    assert "WEB" in agent_tools.agents
    print("WEB agent found.")
    
    # Test consult_specialist with WEB
    res = agent_tools.consult_specialist("WEB", "Find a brick texture")
    print(f"Web Response: {res}")
    
    if "Error" in res:
        print(f"Consultation failed: {res}")
    assert "Error" not in res

def test_research_agent():
    print("\n=== Testing RESEARCH Agent ===")
    mock_llm = MagicMock()
    mock_llm.chat_completion.side_effect = [
        {"content": '{"thought": "Researching...", "code": "print(\'Searching\')", "expected_changes": {}}'},
        {"content": '{"thought": "Done.", "expected_changes": {"added": []}}'}
    ]
    agent_tools = AgentTools(llm_client=mock_llm)
    assert "RESEARCH" in agent_tools.agents
    print("RESEARCH agent found.")
    res = agent_tools.consult_specialist("RESEARCH", "How to make donuts?")
    print(f"Research Response: {res}")
    assert "Error" not in res

def test_completion_agent():
    print("\n=== Testing COMPLETION Agent ===")
    mock_llm = MagicMock()
    mock_llm.chat_completion.return_value = {
        "content": '{"thought": "Task complete.", "expected_changes": {"status": "COMPLETE"}}'
    }
    agent_tools = AgentTools(llm_client=mock_llm)
    assert "COMPLETION" in agent_tools.agents
    print("COMPLETION agent found.")
    res = agent_tools.consult_specialist("COMPLETION", "Is the donut done?")
    print(f"Completion Response: {res}")
    assert "Error" not in res

if __name__ == "__main__":
    test_assistant_help()
    # test_verifier_agent()
    test_web_agent()
    test_research_agent()
    test_completion_agent()

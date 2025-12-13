"""Scene Update Agent - The "Researcher" for maintaining context."""

import bpy
from typing import Dict, Any, List, Optional
from .tools import tool_registry

SCENE_AGENT_PROMPT = """You are the Scene Agent (The Researcher).
Your job is to INVESTIGATE the Blender scene to answer a specific query from the Manager or Task Agent.

CRITICAL: The "Scene Changes" summary is a historical delta.
DATA ACCURACY PROTOCOL: To ensure global consistency, you MUST query the live state using tools.
VERIFICATION: Use `inspect_data` or `get_scene_info` to fetch the current ground truth.

GOAL: Provide a concise, factual summary of the scene state relevant to the query.

PROCESS:
1. ANALYZE: What information is needed?
2. SEARCH: Use introspection tools to find the ABSOLUTE TRUTH.
   - Use `inspect_data(path='bpy.data.objects["Name"]')` to verify existence and properties.
   - Use `search_data(query="...")` to find objects/blocks.
   - Use `get_scene_info(detailed=True)` for hierarchy.
3. SYNTHESIZE: Summarize your findings into a clear text report.

OUTPUT FORMAT:
Return a JSON object with your findings in the `thought` field, and mark status as DONE.
{
    "thought": "The Cube has 2 material slots: 'MatA' (Red) and 'MatB' (Blue). It is currently located at (0,0,0).",
    "expected_changes": {"status": "DONE"}
}
"""

def consult_scene_agent(query: str, focus_object: str = None) -> Dict[str, Any]:
    """
    Spawns the Scene Agent to investigate the scene and return a textual summary.
    
    Args:
        query: The question or investigation goal (e.g. "What materials are on the selected objects?").
        focus_object: Optional object context.
        
    Returns:
        Dictionary containing the agent's findings.
    """
    # Break circular import by runtime lookup
    from .session_manager import get_session
    
    session = get_session(bpy.context)
    if not session:
        return {"error": "No active assistant session found."}
    
    if not hasattr(session, "agent_tools"):
        return {"error": "Session agent tools not initialized."}
        
    # Spawn the agent
    # We expect the agent to return a result dict
    if hasattr(session.agent_tools, "spawn_agent"):
        return session.agent_tools.spawn_agent("SCENE_AGENT", query, focus_object)
    
    return {"error": "AgentTools.spawn_agent not found."}


def register():
    """Register the Scene Agent tool."""
    tool_registry.register_tool(
        name="consult_scene_agent",
        func=consult_scene_agent,
        description=(
            "Ask the specialized Scene Agent to investigate the scene.\n"
            "USE THIS WHEN: You need to verify complex state, find deep properties, or check results before proceeding.\n"
            "RETURNS: A text summary of the findings."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What do you want to know about the scene?",
                },
                "focus_object": {
                    "type": "string",
                    "description": "Optional object to focus on",
                }
            },
            "required": ["query"]
        },
        category="System",
        requires_main_thread=False
    )

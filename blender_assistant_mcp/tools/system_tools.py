"""System tools for agent communication and task completion."""

import bpy
from typing import Dict, Any, List, Optional
from . import tool_registry

def spawn_agent(role: str, query: str, focus_object: str = None) -> Dict[str, Any]:
    """
    Delegate a task to a Specialist Agent (e.g., TASK_AGENT) or Verify completion (COMPLETION_AGENT).
    
    Args:
        role: The role to spawn (TASK_AGENT or COMPLETION_AGENT).
        query: The specific task or question for the specialist.
        focus_object: Optional object name to focus the specialist's attention on.
        
    Returns:
        JSON with 'thought', 'code', 'expected_changes', or 'error'.
    """
    # Break circular import by runtime lookup
    from ..session_manager import get_session
    
    session = get_session(bpy.context)
    if not session:
        return {"error": "No active assistant session found."}
    
    if not hasattr(session, "agent_tools"):
        return {"error": "Session agent tools not initialized."}
        
    # Delegate to the AgentTools instance
    # NOTE: The instance method starts a background thread and returns metadata
    if hasattr(session.agent_tools, "spawn_agent"):
        return session.agent_tools.spawn_agent(role, query, focus_object)
    
    return {"error": "AgentTools.spawn_agent not found."}

def finish_task(expected_changes: List[str], summary: str = "") -> Dict[str, Any]:
    """
    Call this tool when you have completed your objective.
    
    Args:
        expected_changes: List of strings describing what changed (e.g. "Added Cube").
        summary: Brief summary of what was done.
    """
    return {
        "status": "DONE",
        "summary": summary,
        "expected_changes": expected_changes
    }

def register():
    """Register system tools."""
    tool_registry.register_tool(
        name="spawn_agent",
        func=spawn_agent,
        description=(
            "Delegate a task to a Worker Agent (TASK_AGENT) or Verify completion (COMPLETION_AGENT).\n"
            "RETURNS: A JSON Report containing 'status' (DONE/INCOMPLETE/ERROR), 'summary', and 'expected_changes'.\n"
            "BEHAVIOR:\n"
            "- BLOCKING: This tool waits for the agent to finish and returns the Final Report immediately.\n"
            "- ERRORS: If status is 'ERROR' or 'INCOMPLETE', read the summary and plan your next step to fix it.\n"
            "- SYNC: Use the returned 'expected_changes' to update your internal Task List tracking."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "role": {
                    "type": "string", 
                    "enum": ["TASK_AGENT", "COMPLETION_AGENT"],
                    "description": "Agent to spawn."
                },
                "query": {
                    "type": "string",
                    "description": "Task description."
                },
                "focus_object": {
                    "type": "string",
                    "description": "Optional name of object to focus on."
                }
            },
            "required": ["role", "query"]
        },
        category="System",
        requires_main_thread=False
    )

    tool_registry.register_tool(
        name="finish_task",
        func=finish_task,
        description=(
            "Signal completion of the assigned task.\n"
            "USAGE: Call this when your objective is fully complete.\n"
            "RETURNS: {'status': 'DONE', 'summary': '...', 'expected_changes': [...]}"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "Summary of work done."},
                "expected_changes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of changes made to the scene."
                }
            },
            "required": ["expected_changes"]
        },
        category="System",
        requires_main_thread=False
    )

"""Tools for managing assistant tasks and checkpoints."""

import bpy
from typing import Dict, Any, List
from . import tool_registry

# -----------------------------------------------------------------------------
# Data Structures (Blender Properties)
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------
# AssistantTaskItem is defined in ui.py to be part of AssistantChatSession
# We access it via wm.assistant_chat_sessions[wm.assistant_active_chat_index].tasks

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------

def _get_active_task_list():
    """Helper to get the task list for the active chat session."""
    wm = bpy.context.window_manager
    if not wm.assistant_chat_sessions:
        return None
    idx = wm.assistant_active_chat_index
    if idx < 0 or idx >= len(wm.assistant_chat_sessions):
        return None
    return wm.assistant_chat_sessions[idx].tasks

def task_add(description: str) -> Dict[str, Any]:
    """Add a new task to the plan.
    
    Args:
        description: The task description (e.g., "Create base mesh")
    """
    tasks = _get_active_task_list()
    if tasks is None:
         return {"success": False, "error": "No active chat session found."}

    item = tasks.add()
    item.name = description
    item.status = "TODO"
    
    return {
        "success": True, 
        "message": f"Added task: {description}", 
        "task_index": len(tasks) - 1
    }

def task_update(index: int, status: str, notes: str = "") -> Dict[str, Any]:
    """Update a task's status and notes (Checkpoint).
    
    Args:
        index: The index of the task to update (0-based)
        status: New status (TODO, IN_PROGRESS, DONE, FAILED, SKIPPED)
        notes: Optional notes or verification results
    """
    tasks = _get_active_task_list()
    if tasks is None:
         return {"success": False, "error": "No active chat session found."}

    if index < 0 or index >= len(tasks):
        return {"success": False, "error": f"Invalid task index: {index}"}
    
    item = tasks[index]
    
    # Validate status
    valid_statuses = {"TODO", "IN_PROGRESS", "DONE", "FAILED", "SKIPPED"}
    if status not in valid_statuses:
         return {"success": False, "error": f"Invalid status: {status}. Must be one of {valid_statuses}"}

    item.status = status
    if notes:
        item.notes = notes
        
    return {
        "success": True, 
        "message": f"Updated task {index} ('{item.name}') to {status}",
        "current_state": {
            "name": item.name,
            "status": item.status,
            "notes": item.notes
        }
    }

def task_complete(index: int) -> Dict[str, Any]:
    """Mark a task as DONE and provide a summary log."""
    return task_update(index, "DONE")

def task_list() -> Dict[str, Any]:
    """List all tasks and their current status."""
    tasks = _get_active_task_list()
    if tasks is None:
         return {"message": "No active chat session found."}

    output_tasks = []
    for i, item in enumerate(tasks):
        output_tasks.append({
            "index": i,
            "name": item.name,
            "status": item.status,
            "notes": item.notes
        })
        
    if not output_tasks:
        return {"message": "No tasks in plan."}
        
    return {"tasks": output_tasks}

def task_clear(reason: str = "") -> Dict[str, Any]:
    """Clear all tasks from the plan."""
    tasks = _get_active_task_list()
    if tasks is None:
         return {"success": False, "error": "No active chat session found."}
    
    tasks.clear()
    msg = f"Task list cleared. Reason: {reason}" if reason else "Task list cleared."
    return {"success": True, "message": msg}

# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

def register():
    """Register tools and properties."""
    # AssistantTaskItem is registered in ui.py now, and attached to AssistantChatSession
    # We used to register it here but moved it for scope control.
    # bpy.utils.register_class(AssistantTaskItem)
    # bpy.types.Scene.assistant_tasks = bpy.props.CollectionProperty(type=AssistantTaskItem)
    
    # Register Tools
    tool_registry.register_tool(
        "task_add",
        task_add,
        "Add a new task to the plan.",
        {
            "type": "object",
            "properties": {
                "description": {"type": "string", "description": "Task description"}
            },
            "required": ["description"]
        },
        category="Planning"
    )
    
    tool_registry.register_tool(
        "task_update",
        task_update,
        "Update a task's status and add notes (Checkpoint).",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Task index (0-based)"},
                "status": {"type": "string", "enum": ["TODO", "IN_PROGRESS", "DONE", "FAILED", "SKIPPED"], "description": "New status"},
                "notes": {"type": "string", "description": "Verification notes or details"}
            },
            "required": ["index", "status"]
        },
        category="Planning"
    )

    tool_registry.register_tool(
        "task_complete",
        task_complete,
        "Mark a task as DONE.",
        {
            "type": "object",
            "properties": {
                "index": {"type": "integer", "description": "Task index (0-based)"}
            },
            "required": ["index"]
        },
        category="Planning"
    )
    
    tool_registry.register_tool(
        "task_list",
        task_list,
        "List all tasks and their status.",
        {
            "type": "object",
            "properties": {},
            "required": []
        },
        category="Planning"
    )
    
    tool_registry.register_tool(
        "task_clear",
        task_clear,
        "Clear the task list.",
        {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Reason for clearing tasks"}
            },
            "required": []
        },
        category="Planning"
    )

def unregister():
    """Unregister tools and properties."""
    if hasattr(bpy.types.Scene, "assistant_tasks"):
        del bpy.types.Scene.assistant_tasks
    # bpy.utils.unregister_class(AssistantTaskItem) # Registered in ui.py

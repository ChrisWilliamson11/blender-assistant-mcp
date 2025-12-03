"""Tools for managing assistant tasks and checkpoints."""

import bpy
from typing import Dict, Any, List
from . import tool_registry

# -----------------------------------------------------------------------------
# Data Structures (Blender Properties)
# -----------------------------------------------------------------------------

class AssistantTaskItem(bpy.types.PropertyGroup):
    """A single task item in the assistant's todo list."""
    name: bpy.props.StringProperty(name="Task Name")
    status: bpy.props.EnumProperty(
        name="Status",
        items=[
            ("TODO", "To Do", "Task is pending"),
            ("IN_PROGRESS", "In Progress", "Task is currently being worked on"),
            ("DONE", "Done", "Task is completed"),
            ("FAILED", "Failed", "Task failed to complete"),
            ("SKIPPED", "Skipped", "Task was skipped"),
        ],
        default="TODO"
    )
    notes: bpy.props.StringProperty(name="Notes", description="Additional notes or checkpoint details")

# -----------------------------------------------------------------------------
# Tools
# -----------------------------------------------------------------------------

def add_task(description: str) -> Dict[str, Any]:
    """Add a new task to the plan.
    
    Args:
        description: The task description (e.g., "Create base mesh")
    """
    scene = bpy.context.scene
    item = scene.assistant_tasks.add()
    item.name = description
    item.status = "TODO"
    
    return {
        "success": True, 
        "message": f"Added task: {description}", 
        "task_index": len(scene.assistant_tasks) - 1
    }

def update_task(index: int, status: str, notes: str = "") -> Dict[str, Any]:
    """Update a task's status and notes (Checkpoint).
    
    Args:
        index: The index of the task to update (0-based)
        status: New status (TODO, IN_PROGRESS, DONE, FAILED, SKIPPED)
        notes: Optional notes or verification results
    """
    scene = bpy.context.scene
    if index < 0 or index >= len(scene.assistant_tasks):
        return {"success": False, "error": f"Invalid task index: {index}"}
    
    item = scene.assistant_tasks[index]
    
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

def list_tasks() -> Dict[str, Any]:
    """List all tasks and their current status."""
    scene = bpy.context.scene
    tasks = []
    for i, item in enumerate(scene.assistant_tasks):
        tasks.append({
            "index": i,
            "name": item.name,
            "status": item.status,
            "notes": item.notes
        })
        
    if not tasks:
        return {"message": "No tasks in plan."}
        
    return {"tasks": tasks}

def clear_tasks() -> Dict[str, Any]:
    """Clear all tasks from the plan."""
    scene = bpy.context.scene
    scene.assistant_tasks.clear()
    return {"success": True, "message": "Task list cleared."}

# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------

def register():
    """Register tools and properties."""
    # Register PropertyGroup
    bpy.utils.register_class(AssistantTaskItem)
    bpy.types.Scene.assistant_tasks = bpy.props.CollectionProperty(type=AssistantTaskItem)
    
    # Register Tools
    tool_registry.register_tool(
        "add_task",
        add_task,
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
        "update_task",
        update_task,
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
        "list_tasks",
        list_tasks,
        "List all tasks and their status.",
        {
            "type": "object",
            "properties": {},
            "required": []
        },
        category="Planning"
    )
    
    tool_registry.register_tool(
        "clear_tasks",
        clear_tasks,
        "Clear the task list.",
        {
            "type": "object",
            "properties": {},
            "required": []
        },
        category="Planning"
    )

def unregister():
    """Unregister tools and properties."""
    if hasattr(bpy.types.Scene, "assistant_tasks"):
        del bpy.types.Scene.assistant_tasks
    bpy.utils.unregister_class(AssistantTaskItem)

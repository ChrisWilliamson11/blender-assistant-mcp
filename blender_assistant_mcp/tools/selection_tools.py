"""Selection and active object management tools.

This module contains tools for getting and setting object selection and active objects.
"""

import bpy
from . import tool_registry


def get_selection() -> dict:
    """Get the currently selected objects.
    
    Returns:
        Dictionary with list of selected object names and count
    """
    try:
        selected = [obj.name for obj in bpy.context.selected_objects]
        return {
            "selected_objects": selected,
            "count": len(selected),
            "message": f"Found {len(selected)} selected object(s): {', '.join(selected) if selected else 'none'}"
        }
    except Exception as e:
        return {"error": f"Failed to get selection: {str(e)}"}


def get_active() -> dict:
    """Get the currently active object.
    
    Returns:
        Dictionary with active object name and type
    """
    try:
        active = bpy.context.active_object
        if active:
            return {
                "active_object": active.name,
                "type": active.type,
                "message": f"Active object: {active.name} (type: {active.type})"
            }
        else:
            return {
                "active_object": None,
                "message": "No active object"
            }
    except Exception as e:
        return {"error": f"Failed to get active object: {str(e)}"}


def set_selection(object_names: list) -> dict:
    """Set the selection to specific objects.

    Args:
        object_names: List of object names to select (also accepts a single string)

    Returns:
        Dictionary with success status and selected objects
    """
    try:
        # Normalize input
        if isinstance(object_names, str):
            object_names = [object_names]
        if not isinstance(object_names, (list, tuple)):
            return {"error": "object_names must be a list of strings or a single string"}
        names = [str(n) for n in object_names if str(n).strip()]
        if not names:
            return {"success": False, "selected": [], "not_found": [], "message": "No object names provided"}

        # Deselect all first
        bpy.ops.object.select_all(action='DESELECT')

        selected = []
        not_found = []
        seen = set()

        for name in names:
            if name in seen:
                continue
            seen.add(name)
            obj = bpy.data.objects.get(name)
            if obj:
                obj.select_set(True)
                selected.append(name)
            else:
                not_found.append(name)

        message = f"Selected {len(selected)} object(s)"
        if not_found:
            message += f". Not found: {', '.join(not_found)}"
            if len(not_found) == len(names):
                message += ". Tip: use select_by_type(object_type='MESH') to select meshes, or check name spelling."

        return {
            "success": True,
            "selected": selected,
            "not_found": not_found,
            "message": message
        }
    except Exception as e:
        return {"error": f"Failed to set selection: {str(e)}"}


def set_active(object_name: str) -> dict:
    """Set the active object.
    
    Args:
        object_name: Name of the object to make active
        
    Returns:
        Dictionary with success status
    """
    try:
        obj = bpy.data.objects.get(object_name)
        if not obj:
            return {"error": f"Object '{object_name}' not found"}
        
        # Set as active
        bpy.context.view_layer.objects.active = obj
        
        # Also select it if not already selected
        if not obj.select_get():
            obj.select_set(True)
        
        return {
            "success": True,
            "active_object": object_name,
            "message": f"Set active object to: {object_name}"
        }
    except Exception as e:
        return {"error": f"Failed to set active object: {str(e)}"}


def select_by_type(object_type: str) -> dict:
    """Select all objects of a specific type.

    Args:
        object_type: Type of objects to select (MESH, CURVE, LIGHT, CAMERA, etc.)

    Returns:
        Dictionary with selected objects
    """
    try:
        allowed = {"MESH", "CURVE", "SURFACE", "META", "FONT", "ARMATURE", "LATTICE", "EMPTY", "GPENCIL", "CAMERA", "LIGHT", "SPEAKER", "LIGHT_PROBE"}
        ot = (object_type or "").upper()
        if ot not in allowed:
            return {"error": f"Invalid object_type '{object_type}'.", "allowed": sorted(list(allowed))}

        # Deselect all first
        bpy.ops.object.select_all(action='DESELECT')

        selected = []
        for obj in bpy.data.objects:
            if obj.type == ot:
                obj.select_set(True)
                selected.append(obj.name)

        return {
            "success": True,
            "selected": selected,
            "count": len(selected),
            "message": f"Selected {len(selected)} {ot} object(s)"
        }
    except Exception as e:
        return {"error": f"Failed to select by type: {str(e)}"}


def register():
    """Register all selection tools with the MCP registry."""
    
    tool_registry.register_tool(
        "get_selection",
        get_selection,
        (
            "Get the list of currently selected objects.\n"
            "RETURNS: {'selected_objects': ['Cube', ...], 'count': 1}\n"
            "USAGE: Use to verify what acts as input for subsequent operations."
        ),
        {
            "type": "object",
            "properties": {},
            "required": []
        },
        category="Selection"
    )

    tool_registry.register_tool(
        "get_active",
        get_active,
        (
            "Get the currently active object (highlighted in yellow).\n"
            "RETURNS: {'active_object': 'Cube', 'type': 'MESH'}\n"
            "USAGE: Many operators apply *only* to the active object."
        ),
        {
            "type": "object",
            "properties": {},
            "required": []
        },
        category="Selection"
    )

    tool_registry.register_tool(
        "set_selection",
        set_selection,
        (
            "Set the selection to specific objects by name.\n"
            "USAGE: Pass `object_names=['Cube', 'Sphere']`. Clears previous selection."
        ),
        {
            "type": "object",
            "properties": {
                "object_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of object names to select (e.g., ['Cube', 'Sphere'])"
                }
            },
            "required": ["object_names"]
        },
        category="Selection"
    )

    tool_registry.register_tool(
        "set_active",
        set_active,
        (
            "Set the active object by name (must be visible).\n"
            "USAGE: Pass `object_name='Cube'`. Ensures object is also selected."
        ),
        {
            "type": "object",
            "properties": {
                "object_name": {
                    "type": "string",
                    "description": "Name of the object to make active (e.g., 'Cube')"
                }
            },
            "required": ["object_name"]
        },
        category="Selection"
    )

    tool_registry.register_tool(
        "select_by_type",
        select_by_type,
        (
            "Select all objects of a specific type.\n"
            "USAGE: Pass `object_type='MESH'` or 'LIGHT', 'CAMERA'. Useful for batch operations.\n"
            "RETURNS: List of selected names."
        ),
        {
            "type": "object",
            "properties": {
                "object_type": {
                    "type": "string",
                    "description": "Type of objects to select (MESH, CURVE, LIGHT, CAMERA, EMPTY, etc.)",
                    "enum": ["MESH", "CURVE", "SURFACE", "META", "FONT", "ARMATURE",
                            "LATTICE", "EMPTY", "GPENCIL", "CAMERA", "LIGHT", "SPEAKER",
                            "LIGHT_PROBE"]
                }
            },
            "required": ["object_type"]
        },
        category="Selection"
    )


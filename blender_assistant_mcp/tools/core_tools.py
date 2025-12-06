"""Core tools for the assistant (execute_code, assistant_help)."""

import bpy
import json
from .tool_registry import register_tool

def execute_code(code: str):
    """
    Execute Python code in Blender.
    
    Args:
        code: Python code to execute.
    """
    # Real execution implementation
    import sys
    from io import StringIO
    
    # Capture stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = StringIO()
    
    try:
        sys.stdout = redirected_output
        sys.stderr = redirected_output
        
        # Execute the code
        exec(code, {"bpy": bpy, "print": print})
        
        result_output = redirected_output.getvalue()
        return {"status": "success", "output": result_output}
        
    except Exception as e:
        import traceback
        return {"status": "error", "error": str(e), "traceback": traceback.format_exc(), "output": redirected_output.getvalue()}
        
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def assistant_help(tool_name: str = None):
    """
    Get help for a specific tool or SDK method.
    
    Args:
        tool_name: The name of the tool (e.g., 'assistant_sdk.blender.get_scene_info')
    """
    from .tool_registry import get_tool_schema, _TOOLS, get_tools_list
    
    
    # 1. List Categories Mode
    if not tool_name or tool_name.lower() in ("list", "help", "categories"):
        categories = sorted(list({t.get("category", "Other") for t in _TOOLS.values()}))
        return {
            "available_categories": categories,
            "usage": "assistant_help(tool_name='<category>') to list tools in a category, or assistant_help(tool_name='<tool_name>') for schema."
        }
        
    # 2. Normalize Input
    # Remove 'assistant_sdk.' prefix if present
    clean_name = tool_name
    if clean_name.startswith("assistant_sdk."):
        clean_name = clean_name.replace("assistant_sdk.", "", 1)
        
    # Check for dot notation (category.tool OR category.submodule.tool)
    parts = clean_name.split(".")
    
    target_tool = clean_name
    target_category = None
    
    if len(parts) > 1:
        # Assumption: Last part is tool name, first part is category
        target_tool = parts[-1]
        target_category = parts[0] # Hint
    else:
        # No dot, could be category or tool
        # Check if it matches a known category exactly
        all_cats = {t.get("category", "Other").lower() for t in _TOOLS.values()}
        if clean_name.lower() in all_cats:
             target_category = clean_name
             target_tool = None
            
    # 3. Validation
    
    # Try Direct Tool Lookup First
    if target_tool:
        schema = get_tool_schema(target_tool)
        if schema:
            return schema
            
    # Try Category Lookup
    if target_category:
        tools_in_cat = []
        for name, data in _TOOLS.items():
            cat = data.get("category", "Other")
            if cat.lower() == target_category.lower():
                tools_in_cat.append(name)
        
        if tools_in_cat:
            return {
                "category": target_category,
                "tools": sorted(tools_in_cat),
                "hint": f"Use assistant_help(tool_name='<tool_name>') for details."
            }

    # 4. Failure
    categories = sorted(list({t.get("category", "Other") for t in _TOOLS.values()}))
    return {
        "error": f"Tool or Category '{tool_name}' not found.",
        "available_categories": categories,
        "tip": "Try assistant_help('list') to see all categories."
    }


def register():
    """Register core tools."""
    register_tool(
        name="execute_code",
        func=execute_code,
        description="Run Python code in Blender. Use this to call `assistant_sdk` methods or raw `bpy`.",
        input_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute (multi-line supported)."
                }
            },
            "required": ["code"]
        },
        category="Core"
    )

    register_tool(
        name="assistant_help",
        func=assistant_help,
        description="Get help/schema for tools. Call without arguments to discover available Categories. Call with 'CategoryName' to list tools in a category. Call with 'tool_name' for detailed schema.",
        input_schema={
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Tool name, Category name, or empty to list all categories."
                }
            },
            "required": []
        },
        category="Core"
    )

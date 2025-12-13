"""Core tools for the assistant (execute_code, sdk_help)."""

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
    import ast
    import traceback
    from io import StringIO
    from .. import assistant_sdk

    # Initialize namespace (Do not cache to avoid stale references on reload)
    
    # MAGIC FIX: Inject assistant_sdk into sys.modules so 'import assistant_sdk' works
    # We always overwrite to ensure we have the fresh module instance, not a stale one from before reload
    import sys
    sys.modules["assistant_sdk"] = assistant_sdk

    namespace = {
        "bpy": bpy,
        "print": print,
        "assistant_sdk": assistant_sdk.get_assistant_sdk()
    }

    # Capture stdout/stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    redirected_output = StringIO()
    
    result_data = {"status": "success"}

    try:
        sys.stdout = redirected_output
        sys.stderr = redirected_output
        
        # Parse code to check if last statement is an expression (REPL behavior)
        try:
            tree = ast.parse(code)
            last_node = tree.body[-1] if tree.body else None
            
            if isinstance(last_node, ast.Expr):
                # Split into body (exec) and tail (eval)
                body_nodes = tree.body[:-1]
                last_expr = last_node.value
                
                # Execute body
                if body_nodes:
                    exec(compile(ast.Module(body_nodes, type_ignores=[]), filename="<string>", mode="exec"), namespace)
                    
                # Eval tail
                # Note: eval() needs the same namespace
                eval_ret = eval(compile(ast.Expression(last_expr), filename="<string>", mode="eval"), namespace)
                if eval_ret is not None:
                     print(eval_ret) # Capture in output
                     result_data["return_value"] = str(eval_ret) # Also store explicitly
            else:
                # Normal execution
                exec(code, namespace)
                
        except Exception:
             # Fallback to standard exec if AST fails
             exec(code, namespace)
        
        # Collect output
        output_str = redirected_output.getvalue()
        if output_str:
            result_data["output"] = output_str

        # Check for explicit 'result' variable (User fallback)
        if "result" in namespace:
            try:
                result_data["result_var"] = namespace["result"]
            except:
                result_data["result_var"] = str(namespace["result"])
        elif "__result__" in namespace:
             try:
                result_data["result_var"] = namespace["__result__"]
             except:
                result_data["result_var"] = str(namespace["__result__"])

        return result_data
        
    except Exception as e:
        tb = traceback.format_exc()
        result_data["status"] = "error"
        result_data["error"] = str(e)
        result_data["traceback"] = tb
        result_data["output"] = redirected_output.getvalue()
        
        # Helper: Smart Error Hints
        err_low = (str(e) or "").lower()
        code_low = (code or "").lower()
        
        if (
            "no module named 'assistant_sdk'" in err_low
            or "name 'assistant_sdk' is not defined" in err_low
            or "import assistant_sdk" in code_low
            or "from assistant_sdk" in code_low
        ):
            result_data["hint"] = (
                "assistant_sdk is ALREADY available in the execute_code namespace. "
                "Use it directly (e.g. `assistant_sdk.polyhaven.search(...)`). "
                "Do NOT import it."
            )
            
        if (
            "string indices must be integers" in err_low
            and "polyhaven.search" in code_low
            and " in results" in code_low
        ):
            result_data["hint"] = (
                "PolyHaven search returns a dict {'assets': [...]}. "
                "Iterate over `results.get('assets', [])`."
            )

        return result_data
        
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def sdk_help(tool_name: str = None, tool_names: list[str] = None):
    """
    Get help for specific tool(s) or SDK methods.
    
    Args:
        tool_name: (DEPRECATED) Single tool name.
        tool_names: List of tool names to retrieve schemas for. ALWAYS USE THIS FOR BATCHING.
    """
    from .tool_registry import get_tool_schema, _TOOLS, get_tools_list
    
    # helper for single tool processing
    def _process_single_tool(clean_name):
        # Normalize
        if clean_name.startswith("assistant_sdk."):
            clean_name = clean_name.replace("assistant_sdk.", "", 1)
        
        # SPECIAL CASE: Help for 'assistant_sdk' itself
        if clean_name == "assistant_sdk":
            return {
                "name": "assistant_sdk",
                "description": "Python SDK for programmatic access to all tools + Skills.",
                "usage": "assistant_sdk.<tool_name>(...) OR assistant_sdk.<namespace>.<tool_name>(...)",
                "namespaces": sorted(list({t.get("category", "Other").lower().replace(" ", "_") for t in _TOOLS.values()})),
                "mapping_rule": "Unified Access: use `assistant_sdk.tool_name` directly if unique. Else use namespace."
            }

        parts = clean_name.split(".")
        target_tool = clean_name
        target_category = None
        
        if len(parts) > 1:
            target_tool = parts[-1]
            target_category = parts[0]
        elif clean_name.lower() in {t.get("category", "Other").lower() for t in _TOOLS.values()}:
             target_category = clean_name
             target_tool = None
             
        # Lookup
        found_tool = None
        if target_tool and target_tool in _TOOLS:
             found_tool = _TOOLS[target_tool]
             
        # Category Mode
        if not found_tool and target_category:
            import bpy
            # Robust package name resolution
            package_name = __package__
            if ".tools" in package_name:
                package_name = package_name.partition(".tools")[0]
            
            try:
                prefs = bpy.context.preferences.addons[package_name].preferences
            except:
                # Fallback for legacy parsing if needed, but the above should cover Extensions & Zip
                prefs = bpy.context.preferences.addons[__package__.split(".")[0]].preferences

            exposed_tools = {t.name for t in prefs.tool_config_items if t.expose_mcp}
            # Always expose core tools
            exposed_tools.add("execute_code")
            exposed_tools.add("sdk_help")
            
            tools_in_cat = []
            
            # Helper to check exposure
            def is_exposed(t_name):
                return t_name in exposed_tools or t_name in ["execute_code", "sdk_help", "spawn_agent", "finish_task"]

            for name, data in _TOOLS.items():
                cat = data.get("category", "Other")
                if cat.lower() == target_category.lower():
                    # SHOW ALL TOOLS (Filtered or not)
                    # But annotate if they are SDK-only
                    if is_exposed(name):
                        tools_in_cat.append(name)
                    else:
                        tools_in_cat.append(f"{name} (SDK Only)")
                        
            if tools_in_cat:
                return {
                    "category": target_category,
                    "tools": sorted(tools_in_cat),
                    "hint": "Tools marked '(SDK Only)' available via `assistant_sdk` but not as native prompt tools. Use `sdk_help(tool_name='...')` for usage."
                }
            else:
                 return {
                    "category": target_category,
                    "tools": [],
                    "hint": "No tools found in this category. Check spelling or inspect `assistant_sdk` directly."
                }

        # Schema Mode
        if found_tool:
            try:
                # Fix: found_tools values don't have "name" key inside, use the lookup key
                tool_name_key = target_tool
                tool_def = get_tool_schema(tool_name_key)
            except Exception:
                # Robustness fallback
                return {
                    "name": target_tool or "unknown",
                    "description": "Error retrieving schema.",
                    "error": "Tool definition invalid or missing name."
                }
            
            # SMART SDK HINT: Check if flattened access is valid
            from .. import assistant_sdk
            sdk = assistant_sdk.get_assistant_sdk()
            
            # Default to Namespaced
            category = found_tool.get("category", "Other")
            namespace = category.lower().replace(" ", "_").replace("-", "_")
            sdk_path = f"assistant_sdk.{namespace}.{tool_name_key}"
            
            # Check for shorter path
            if hasattr(sdk, "_flat_map"):
                flat_lookup = sdk._flat_map.get(tool_name_key)
                if flat_lookup and flat_lookup != "AMBIGUOUS":
                    sdk_path = f"assistant_sdk.{tool_name_key}"
            
            # Check exposure status for clearer hint
            import bpy
            package_name = __package__
            if ".tools" in package_name:
                package_name = package_name.partition(".tools")[0]
            try:
                prefs = bpy.context.preferences.addons[package_name].preferences
            except:
                prefs = bpy.context.preferences.addons[__package__.split(".")[0]].preferences

            is_exposed = False
            for t in prefs.tool_config_items:
                if t.name == tool_name_key:
                    is_exposed = t.expose_mcp
                    break
            
            # Use exposure status to refine the hint
            if not is_exposed and tool_name_key not in ["execute_code", "sdk_help", "spawn_agent", "finish_task"]:
                 tool_def["hint"] = (
                    f"⚠️ SDK ONLY: This tool is hidden from the prompt. "
                    f"Use `{sdk_path}(...)` via `execute_code`."
                )
                 tool_def["usage_method"] = "SDK (Python)"
                 tool_def["python_sdk_usage"] = f"{sdk_path}(...)"
            else:
                 tool_def["hint"] = (
                    f"✅ MCP TOOL: This tool is exposed. Call `{tool_name_key}(...)` directly."
                )
                 tool_def["usage_method"] = "MCP"
                 # No python_sdk_usage for Native tools to reduce clutter
            
            return tool_def

        return None # Not found

    # BATCH MODE
    if tool_names:
        results = {}
        missing = []
        for name in tool_names:
            res = _process_single_tool(name)
            if res:
                # If it's a category listing, just key it by category
                key = res.get("name", name) 
                results[key] = res
            else:
                missing.append(name)
        
        return {
            "schemas": results,
            "missing": missing,
            "count": len(results)
        }

    # SINGLE MODE
    if tool_name and tool_name.lower() not in ("list", "help", "categories"):
        res = _process_single_tool(tool_name)
        if res:
            return res
        
        # Fuzzy match fallback
        matches = []
        for t in _TOOLS.values():
             try:
                 if t.get("name") and tool_name.lower() in t["name"].lower():
                     matches.append(t["name"])
             except: pass
        
        return {
            "error": f"Tool '{tool_name}' not found.",
            "candidates": matches[:5]
        }

    # LIST CATEGORIES (Default)
    # LIST CATEGORIES (Default)
    import bpy
    package_name = __package__
    if ".tools" in package_name:
        package_name = package_name.partition(".tools")[0]
    try:
        prefs = bpy.context.preferences.addons[package_name].preferences
    except:
        prefs = bpy.context.preferences.addons[__package__.split(".")[0]].preferences
    exposed_tools = {t.name for t in prefs.tool_config_items if t.expose_mcp}
    exposed_tools.add("execute_code")
    exposed_tools.add("sdk_help")

    categories = set()
    for name, tool_data in _TOOLS.items():
        if name in exposed_tools:
            categories.add(tool_data.get("category", "Other"))
            
    return {
        "available_categories": sorted(list(categories)),
        "usage": "sdk_help(tool_names=['tool1', 'tool2']) for batch schemas."
    }


def register():
    """Register core tools."""
    register_tool(
        name="execute_code",
        func=execute_code,
        description=(
            "Execute Python code in Blender. This is your PRIMARY way to interact with Blender and the SDK.\n"
            "Capabilities:\n"
            "1. Run raw `bpy` commands (e.g. `bpy.ops.mesh.primitive_cube_add()`).\n"
            "2. Access SDK Tools via `assistant_sdk` (e.g. `assistant_sdk.polyhaven.search(...)`).\n"
            "3. Use `print()` to output information for reasoning (stdout is captured).\n"
            "4. Define functions and classes for reuse in subsequent calls.\n\n"
            "BEST PRACTICE: Filter large data in Python before printing! "
            "Don't dump 10,000 items; use `print(items[:5])` or `[i.name for i in items if ...]`.\n\n"
            "RETURNS: A dictionary containing 'status', 'output' (stdout), and 'scene_changes' (a summary of objects added/removed/modified). "
            "USE 'scene_changes' FOR VERIFICATION."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute. Maintains state across calls."
                }
            },
            "required": ["code"]
        },
        category="Core"
    )

    register_tool(
        name="sdk_help",
        func=sdk_help,
        description=(
            "The capabilities discovery tool. Use this to find tools and learn how to use them.\n"
            "MODES:\n"
            "1. DISCOVER: Call without arguments to list all available categories/namespaces.\n"
            "2. SCHEMA: Pass `tool_name` (e.g. 'polyhaven') to get tools in that category, or a specific tool name (e.g. 'download_polyhaven') to get its full signature.\n"
            "3. BATCH: Pass `tool_names=['tool_a', 'tool_b']` to get multiple schemas at once. ALWAYS use this when you need multiple tools to save steps.\n"
            "4. SDK: Pass `tool_name='assistant_sdk'` to learn about the Python SDK structure."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "description": "Single tool name or Category (e.g. 'PolyHaven')."
                },
                "tool_names": {
                    "type": "array",
                    "items": { "type": "string" },
                    "description": "List of tool names to retrieve schemas for at once. Use this to discover multiple tools in one turn."
                }
            },
            "required": []
        },
        category="Core"
    )

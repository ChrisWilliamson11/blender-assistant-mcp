"""Internal MCP tool registry - no external server needed!

This module provides an internal MCP-compatible tool registry that replaces
the need for an external MCP server. Tools are registered here and can be
called directly from the assistant.
"""

import ast

# Global tool registry
import json
from typing import Any, Callable, Dict, List


def _parse_str_to_obj(s):
    """Best-effort parse of a JSON/Python-literal string into a Python object.
    Returns original input on failure.
    """
    try:
        return json.loads(s)
    except Exception:
        pass
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def _coerce_args_to_schema(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce stringly-typed arguments into the shapes expected by the tool schema.

    - If schema expects array/object but value is a string, try to parse it.
    - If schema expects boolean and value is a string, coerce common forms.
    - If schema expects integer/number and value is a numeric-ish string, convert.
    """
    tool = _TOOLS.get(tool_name) or {}
    schema = tool.get("inputSchema", {}) or {}
    props = schema.get("properties") or {}

    if not props or not isinstance(args, dict):
        return args

    coerced = dict(args)

    for key, prop_schema in props.items():
        if key not in coerced:
            continue
        val = coerced[key]
        if val is None:
            continue

        # Determine expected types
        expected = prop_schema.get("type")
        if isinstance(expected, list):
            expected_types = set(expected)
        elif isinstance(expected, str):
            expected_types = {expected}
        else:
            expected_types = set()

        # Coerce array/object from string
        if isinstance(val, str) and (
            "array" in expected_types or "object" in expected_types
        ):
            parsed = _parse_str_to_obj(val)
            if "array" in expected_types and isinstance(parsed, list):
                coerced[key] = parsed
                continue
            if "object" in expected_types and isinstance(parsed, dict):
                coerced[key] = parsed
                continue
            # Fallback: if array expected and parsing failed, wrap single string into a list
            if "array" in expected_types:
                coerced[key] = [val]
                continue

        # Coerce boolean from string
        if "boolean" in expected_types and isinstance(val, str):
            lv = val.strip().lower()
            if lv in ("true", "1", "yes", "y", "on"):
                coerced[key] = True
                continue
            if lv in ("false", "0", "no", "n", "off"):
                coerced[key] = False
                continue

        # Coerce integer/number from string
        if isinstance(val, str) and (
            "integer" in expected_types or "number" in expected_types
        ):
            try:
                if "integer" in expected_types:
                    coerced[key] = int(float(val))
                elif "number" in expected_types:
                    coerced[key] = float(val)
                continue
            except Exception:
                pass

    return coerced


def _to_number_list(seq):
    """Convert a sequence to a list of floats/ints where possible; leave others as-is."""
    out = []
    for v in list(seq):
        if isinstance(v, (int, float)):
            out.append(v)
        elif isinstance(v, str):
            try:
                if "." in v or "e" in v.lower():
                    out.append(float(v))
                else:
                    out.append(int(float(v)))
            except Exception:
                # keep as-is
                try:
                    out.append(float(v))
                except Exception:
                    out.append(v)
        else:
            out.append(v)
    return out


def _dict_to_vec3(d):
    """Normalize dicts like {x,y,z} or {0,1,2} to [x,y,z]. Returns None if not applicable."""
    if not isinstance(d, dict):
        return None
    # Key variants
    keys = {str(k).lower(): k for k in d.keys()}
    if all(k in keys for k in ("x", "y", "z")):
        return [_to_number_list([d[keys["x"]], d[keys["y"]], d[keys["z"]]])][0]
    # Numeric index keys
    if all(k in keys for k in ("0", "1", "2")):
        return _to_number_list([d[keys["0"]], d[keys["1"]], d[keys["2"]]])
    # Partial XY; default z=0
    if all(k in keys for k in ("x", "y")) and "z" not in keys:
        return _to_number_list([d[keys["x"]], d[keys["y"]], 0.0])
    return None


def _dict_to_color(d):
    """Normalize dicts like {r,g,b,a?} or {0,1,2,3?} to [r,g,b,a]. Returns None if not applicable."""
    if not isinstance(d, dict):
        return None
    keys = {str(k).lower(): k for k in d.keys()}
    if all(k in keys for k in ("r", "g", "b")):
        r = d[keys["r"]]
        g = d[keys["g"]]
        b = d[keys["b"]]
        a = d[keys["a"]] if "a" in keys else 1.0
        return _to_number_list([r, g, b, a])
    if all(k in keys for k in ("0", "1", "2")):
        r = d[keys["0"]]
        g = d[keys["1"]]
        b = d[keys["2"]]
        a = d.get(keys.get("3"), 1.0)
        return _to_number_list([r, g, b, a])
    return None


def _normalize_vectorish_args(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common vector-like arguments across tools.

    - location/rotation/scale: accept dicts {x,y,z} or {0,1,2}
    - color: accept dicts {r,g,b[,a]} or {0,1,2[,3]} and add alpha if missing
    - Also coerce arrays of strings -> numbers where appropriate
    - For batch specs (objects=[{...}]), apply recursively
    """
    if not isinstance(args, dict):
        return args

    def norm_vec3(v):
        if isinstance(v, dict):
            vec = _dict_to_vec3(v)
            if vec is not None:
                return vec
        if isinstance(v, (list, tuple)):
            lst = _to_number_list(v)
            # Pad/trim to 3
            if len(lst) == 2:
                lst = [lst[0], lst[1], 0.0]
            return lst[:3]
        return v

    def norm_color(v):
        if isinstance(v, dict):
            c = _dict_to_color(v)
            if c is not None:
                return c
        if isinstance(v, (list, tuple)):
            lst = _to_number_list(v)
            if len(lst) == 3:
                lst = lst + [1.0]
            if len(lst) >= 4:
                lst = lst[:4]
            return lst
        return v

    for k in list(args.keys()):
        kl = k.lower()
        if kl in ("location", "rotation", "scale"):
            args[k] = norm_vec3(args[k])
        elif kl == "color":
            args[k] = norm_color(args[k])
        elif kl == "objects" and isinstance(args[k], list):
            normalized = []
            for item in args[k]:
                if isinstance(item, dict):
                    item = dict(item)
                    if "location" in item:
                        item["location"] = norm_vec3(item["location"])
                    if "rotation" in item:
                        item["rotation"] = norm_vec3(item["rotation"])
                    if "scale" in item:
                        item["scale"] = norm_vec3(item["scale"])
                    if "color" in item:
                        item["color"] = norm_color(item["color"])
                normalized.append(item)
            args[k] = normalized

    return args


def _apply_arg_aliases(tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Apply common argument aliases based on the tool's schema.

    Only remap when:
    - The canonical key exists in the tool's properties
    - The canonical key is missing from args
    - The alias key exists in args AND is NOT itself a declared property for this tool
    This avoids clobbering legitimate fields like 'name' used by many tools.
    """
    tool = _TOOLS.get(tool_name) or {}
    schema = tool.get("inputSchema", {}) or {}
    props = schema.get("properties") or {}
    prop_keys = set(props.keys())

    if not isinstance(args, dict) or not props:
        return args

    aliases = {
        # Selection / object tools
        "object_names": ["objects", "names"],
        "object_name": ["obj", "object", "name"],
        # Collection tools
        "collection_name": ["collection", "target_collection", "to_collection"],
        "collection_names": ["collections", "names"],
        "parent": ["parent_collection", "parent_name"],
        # Web
        "image_url": ["url", "image", "link"],
        # Meta-tools
        "items": ["list", "values"],
        "operation": ["tool", "function"],
        "operation_args": ["args", "arguments", "parameters"],
        # Common options
        "limit": ["max_results", "n", "count"],
        "num_results": ["n", "limit"],
        "color_tag": ["color", "tag"],
    }

    remapped = dict(args)

    for canonical, syns in aliases.items():
        if canonical in prop_keys and canonical not in remapped:
            for alias in syns:
                # Only map from alias if alias is present and NOT a defined property name in this tool
                if alias in remapped and alias not in prop_keys:
                    remapped[canonical] = remapped.pop(alias)
                    break
    return remapped


_TOOLS: Dict[str, Dict[str, Any]] = {}


def register_tool(
    name: str,
    func: Callable,
    description: str,
    input_schema: Dict[str, Any],
    category: str = "Other",
):
    """Register a tool in the MCP registry.

    Args:
        name: Tool name (e.g., "create_object")
        func: Function to call when tool is invoked
        description: Human-readable description
        input_schema: JSON Schema for tool parameters
        category: Tool category for UI organization (e.g., "Blender", "Web", "PolyHaven")
    """
    _TOOLS[name] = {
        "function": func,
        "description": description,
        "inputSchema": input_schema,
        "category": category,
    }
    print(f"[MCP] Registered tool: {name} (category: {category})")


def unregister_tool(name: str):
    """Unregister a tool from the registry."""
    if name in _TOOLS:
        del _TOOLS[name]
        print(f"[MCP] Unregistered tool: {name}")


def execute_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a registered tool.

    Args:
        name: Tool name
        args: Tool arguments

    Returns:
        Tool result as dict
    """
    if name not in _TOOLS:
        available = ", ".join(_TOOLS.keys())
        return {"error": f"Unknown tool: {name}. Available: {available}"}

    # Normalize args
    if args is None:
        args = {}

    # If args came in as a JSON/Python string, try to parse into a dict
    if isinstance(args, str):
        parsed = _parse_str_to_obj(args)
        if isinstance(parsed, dict):
            args = parsed
        else:
            schema = _TOOLS[name].get("inputSchema", {})
            props = list((schema.get("properties") or {}).keys())
            return {
                "error": f"Invalid argument type for {name}: expected object/dict",
                "received_type": type(args).__name__,
                "expected_params": props,
            }

    if not isinstance(args, dict):
        # Provide a helpful error with expected parameters
        schema = _TOOLS[name].get("inputSchema", {})
        props = list((schema.get("properties") or {}).keys())
        return {
            "error": f"Invalid argument type for {name}: expected object/dict",
            "received_type": type(args).__name__,
            "expected_params": props,
        }

    # Unwrap common envelopes some models produce: {"name": ..., "arguments": {...}} or {"name":..., "args": {...}}
    try:
        if isinstance(args, dict):
            if "arguments" in args and isinstance(args.get("arguments"), (dict, str)):
                inner = args.get("arguments")
                if isinstance(inner, str):
                    parsed = _parse_str_to_obj(inner)
                    if isinstance(parsed, dict):
                        inner = parsed
                if isinstance(inner, dict):
                    args = inner
            elif (
                "args" in args
                and isinstance(args.get("args"), (dict, str))
                and (len(args) == 1 or "name" in args)
            ):
                inner = args.get("args")
                if isinstance(inner, str):
                    parsed = _parse_str_to_obj(inner)
                    if isinstance(parsed, dict):
                        inner = parsed
                if isinstance(inner, dict):
                    args = inner
    except Exception:
        # Best effort; leave args as-is on error
        pass

    # Apply common alias mapping to smooth over LLM variations
    args = _apply_arg_aliases(name, args)
    # Coerce stringly-typed fields to expected schema shapes
    args = _coerce_args_to_schema(name, args)
    # Normalize common vector-like arguments (location/rotation/scale/color, and batch objects)
    args = _normalize_vectorish_args(name, args)

    try:
        result = _TOOLS[name]["function"](**args)
        return result if isinstance(result, dict) else {"result": result}
    except TypeError as e:
        # Argument mismatch
        schema = _TOOLS[name].get("inputSchema", {})
        props = list((schema.get("properties") or {}).keys())
        return {
            "error": f"Invalid arguments for {name}: {str(e)}",
            "expected_params": props,
        }
    except Exception as e:
        # Other errors - include traceback for debugging
        import traceback

        tb = traceback.format_exc()
        print(f"[ERROR] Tool {name} failed:\n{tb}")
        return {"error": f"Tool {name} failed: {str(e)}"}


def get_tools_list() -> List[Dict[str, Any]]:
    """Get list of all registered tools in MCP format.

    Returns:
        List of tool definitions compatible with MCP protocol
    """
    return [
        {
            "name": name,
            "description": tool["description"],
            "inputSchema": tool["inputSchema"],
            "category": tool.get("category", "Other"),
        }
        for name, tool in _TOOLS.items()
    ]


def get_tools_schema(enabled_tools: List[str] = None) -> str:
    """Return a compact tools cheat-sheet for the system prompt.

    Format per tool: "- name(arg1, arg2, ...): short description"
    Only parameter names are listed to keep context small. Full JSON schemas are
    provided via the OpenAI-style "tools" field in the API payload.
    """
    lines: List[str] = []
    for name, tool in _TOOLS.items():
        if enabled_tools is not None and name not in enabled_tools:
            continue
        schema = tool.get("inputSchema", {})
        props = schema.get("properties", {})
        # Parameter names only, preserve a stable order
        arg_names = list(props.keys())
        args_str = ", ".join(arg_names) if arg_names else ""
        # Shorten description
        desc = (tool.get("description", "") or "").strip()
        # Increase limit to avoid truncation of important details
        if len(desc) > 120:
            desc = desc[:117] + "..."
        if args_str:
            lines.append(f"- {name}({args_str}): {desc}")
        else:
            lines.append(f"- {name}(): {desc}")
    header = "NATIVE TOOLS (Call directly):\n"
    return header + "\n".join(lines) if lines else "NATIVE TOOLS (Call directly):\n(none)"


def get_sdk_tools_schema(enabled_tools: List[str], namespace_map: Dict[str, str]) -> str:
    """Return a compact tools cheat-sheet for SDK tools (not natively enabled).

    Format per tool: "- namespace.name(arg1, arg2, ...): short description"
    """
    lines: List[str] = []
    # Sort by category then name for grouping
    sorted_items = sorted(_TOOLS.items(), key=lambda x: (x[1].get("category", "Other"), x[0]))
    
    for name, tool in sorted_items:
        if name in enabled_tools:
            continue
            
        category = tool.get("category", "Other")
        namespace = namespace_map.get(category, "assistant_sdk")
        
        # Construct full name
        full_name = f"{namespace}.{name}"
        
        schema = tool.get("inputSchema", {})
        props = schema.get("properties", {})
        arg_names = list(props.keys())
        args_str = ", ".join(arg_names) if arg_names else ""
        
        desc = (tool.get("description", "") or "").strip()
        if len(desc) > 120:
            desc = desc[:117] + "..."
            
        if args_str:
            lines.append(f"- {full_name}({args_str}): {desc}")
        else:
            lines.append(f"- {full_name}(): {desc}")
            
    header = "SDK TOOLS (Call via `execute_code`):\n"
    return header + "\n".join(lines) if lines else ""


def clear_tools():
    """Clear all registered tools."""
    _TOOLS.clear()
    print("[MCP] Cleared all tools")


# Module-level functions for compatibility
def register():
    """Register all MCP tools (called from __init__.py)."""
    # Note: Other modules register their own tools via their own register() functions
    # which are called by __init__.py. We don't need to do it here.
    pass


def unregister():
    """Unregister all MCP tools."""
    clear_tools()

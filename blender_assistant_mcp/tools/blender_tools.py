"""Blender-specific MCP tools.

This module contains all the Blender manipulation tools that can be called
by the Automation assistant.
"""

import os
import tempfile
import typing

import bpy

from . import tool_registry
from ..memory import MemoryManager
from ..assistant_sdk import AssistantSDK

_memory_manager = None


def get_memory_manager():
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager


def get_scene_info(
    expand_depth: int = 1,
    expand: list | None = None,
    focus: list | None = None,
    fold_state: dict | None = None,
    max_children: int = 50,
    include_icons: bool = True,
    include_counts: bool = True,
    root_filter: str | None = None,
    detailed: bool = False,
) -> dict:
    """Return an Outliner-style, persistent view of the scene (hierarchical, compact, and stateful).

    Args:
        expand_depth: How many levels to expand by default from the root (default: 1)
        expand: Additional node paths to expand beyond expand_depth (e.g., ["SolarSystem/Planets"])
        focus: Node names or paths to auto-expand and include ancestors
        fold_state: Opaque expansion state from prior calls; this will be updated and returned
        max_children: Maximum number of children to list per node (truncates with has_more flag)
        include_icons: Include presence icons (types, modifiers, materials, children) on nodes

        include_counts: Include type counts on collections (e.g., {"MESH": 3, "LIGHT": 1})
        root_filter: If provided, render the outliner starting from a specific collection name

    Returns:
        Dict with keys:
          - requested: normalized parameters used for this call
          - outliner: { "lines": [...], "nodes": [ {path, kind, name, ...} ] }
          - changed, unchanged, failed: empty maps for consistency with other tools
          - fold_state: updated expansion state (pass back to persist expansion/collapse)
          - summary: concise one-liner about what is shown
    """
    import bpy

    # Normalize inputs
    expand = list(expand or [])
    focus = list(focus or [])
    fold_state = dict(fold_state or {})
    expanded = set(fold_state.get("expanded", []))

    # Helper: build node path strings (Collection/ChildCollection/Object)
    def path_join(*parts) -> str:
        return "/".join([p for p in parts if p])

    def object_icons(obj) -> list[str]:
        icons = []

        # Type icon (basic set)

        t = (getattr(obj, "type", "") or "").upper()

        type_map = {
            "MESH": "M",
            "ARMATURE": "AR",
            "LIGHT": "Lt",
            "CAMERA": "Cam",
            "CURVE": "Cur",
            "EMPTY": "E",
        }

        if t in type_map:
            icons.append(type_map[t])

        # Modifiers, Materials, Children presence

        try:
            if getattr(obj, "modifiers", None) and len(obj.modifiers) > 0:
                icons.append("Mo")

        except Exception:
            pass

        try:
            if getattr(obj.data, "materials", None) and len(obj.data.materials) > 0:
                icons.append("Mat")

        except Exception:
            pass

        try:
            if len(getattr(obj, "children", []) or []) > 0:
                icons.append("Ch")

        except Exception as e:
            icons.append(f"Err:{str(e)}")

        # Selection and Active indicators (avoid [A] conflict with ARMATURE type icon)
        try:
            if hasattr(obj, "select_get") and obj.select_get():
                if "S" not in icons:
                    icons.append("S")
        except Exception:
            pass
        try:
            # Only add [A] if 'A' is not already reserved in type_map (ARMATURE)
            import bpy

            if bpy.context and bpy.context.active_object is obj:
                if "A" not in type_map.values() and "A" not in icons:
                    icons.append("A")
        except Exception:
            pass
        return icons

    # Helper: summarize a collection’s contained types
    def collection_type_counts(col) -> dict:
        counts: dict[str, int] = {}
        try:
            for o in col.objects:
                counts[o.type] = counts.get(o.type, 0) + 1
        except Exception:
            pass
        return counts

    # Determine root collection(s)
    scene_root = (
        bpy.context.scene.collection if bpy.context and bpy.context.scene else None
    )
    root_collections = []
    if root_filter:
        c = bpy.data.collections.get(root_filter)
        if c:
            root_collections.append(c)
    if not root_collections:
        # Default to top-level children of the scene root
        try:
            root_collections = list(scene_root.children) if scene_root else []
        except Exception:
            root_collections = []

    # Compute effective expanded set: expand_depth + explicit expand + focus
    # We record expanded collection paths; objects expand only when their parent is expanded.
    to_expand = set(expanded)
    to_expand.update(expand)
    to_expand.update(focus)

    # Utility to check whether a node (path) should be expanded
    def is_expanded(path: str, depth: int) -> bool:
        if path in to_expand:
            return True
        return depth < expand_depth

    # Build tree (nodes and lines). We track nodes by path for machine use.
    nodes = []
    lines = []

    def add_collection(col, path: str, depth: int):
        # Build summary line for collection
        col_name = col.name
        col_tag = getattr(col, "color_tag", "NONE")
        cpath = path_join(path, col_name)
        # Presence icons (types)
        icons = []
        counts = collection_type_counts(col) if include_counts else {}
        if include_icons:
            type_map = {
                "MESH": "M",
                "ARMATURE": "AR",
                "LIGHT": "Lt",
                "CAMERA": "Cam",
                "CURVE": "Cur",
                "EMPTY": "E",
            }

            for k, v in counts.items():
                if k in type_map and v > 0 and type_map[k] not in icons:
                    icons.append(type_map[k])

            # Active collection indicator

            try:
                vl = bpy.context.view_layer if bpy.context else None

                active_col = (
                    vl.active_layer_collection.collection
                    if vl and getattr(vl, "active_layer_collection", None)
                    else (bpy.context.collection if bpy.context else None)
                )

                if active_col is col and "Ac" not in icons:
                    icons.append("Ac")

            except Exception:
                pass

            # If collection is collapsed, surface selection/active object presence on the collection itself

            collapsed = not is_expanded(cpath, depth)

            if include_icons and collapsed:
                try:
                    # Prefer recursive all_objects if available to mirror Outliner collapsed behavior

                    objs = list(getattr(col, "all_objects", col.objects))

                except Exception:
                    try:
                        objs = list(col.objects)

                    except Exception:
                        objs = []

                has_sel = False

                has_act = False

                try:
                    active_obj = bpy.context.active_object if bpy.context else None

                except Exception:
                    active_obj = None

                for o in objs:
                    try:
                        if not has_sel and hasattr(o, "select_get") and o.select_get():
                            has_sel = True

                        if not has_act and active_obj is o:
                            has_act = True

                        if has_sel and has_act:
                            break

                    except Exception:
                        pass

                if has_sel and "S" not in icons:
                    icons.append("S")

                if has_act and "A" not in icons:
                    icons.append("A")

                    # Node record

        node = {
            "path": cpath,
            "kind": "collection",
            "name": col_name,
            "color_tag": col_tag,
            "icons": icons if include_icons else [],
            "counts": counts if include_counts else {},
            "children": [],
        }
        nodes.append(node)
        # Line summary
        ico_txt = "".join(f"[{i}]" for i in (icons if include_icons else []))
        count_txt = ""
        if include_counts:
            total_objs = sum(counts.values())
            count_txt = f" ({total_objs} obj)" if total_objs > 0 else " (0 obj)"
        lines.append(
            ("  " * depth)
            + f"[C] {col_name} [{col_tag}]{(' ' + ico_txt) if ico_txt else ''}{count_txt}"
        )

        # Decide expansion
        if not is_expanded(cpath, depth):
            fold_state.setdefault("expanded", [])
            # Keep current state unchanged if not previously expanded
            return

        # Children: sub-collections then objects
        # Sub-collections
        try:
            subcols = list(col.children)

        except Exception:
            subcols = []
        # Enforce max_children per node
        has_more_subcols = len(subcols) > max_children
        subcols = subcols[:max_children]
        for sc in subcols:
            node["children"].append(path_join(cpath, sc.name))
            add_collection(sc, cpath, depth + 1)
        if has_more_subcols:
            lines.append(("  " * (depth + 1)) + "... more collections")

        # Objects
        try:
            objs = list(col.objects)
        except Exception:
            objs = []
        has_more_objs = len(objs) > max_children
        objs = objs[:max_children]
        for o in objs:
            opath = path_join(cpath, o.name)
            oicons = object_icons(o) if include_icons else []
            nodes.append(
                {
                    "path": opath,
                    "kind": "object",
                    "name": o.name,
                    "type": getattr(o, "type", ""),
                    "icons": oicons,
                    "children": [],
                }
            )
            node["children"].append(opath)
            oico_txt = "".join(f"[{i}]" for i in (oicons if include_icons else []))
            lines.append(
                ("  " * (depth + 1))
                + f"{o.name} [{getattr(o, 'type', '')}]{(' ' + oico_txt) if oico_txt else ''}"
            )
            
            # Detailed mode: populate data
            if detailed:
                # Use _serialize_rna but limit depth to avoid massive dumps
                # We reuse the inspect_data logic
                nodes[-1]["data"] = _serialize_rna(o, depth=0, max_depth=1)

        if has_more_objs:
            lines.append(("  " * (depth + 1)) + "... more objects")

        # Record expansion
        exp_list = set(fold_state.get("expanded", []))
        exp_list.add(cpath)
        fold_state["expanded"] = sorted(exp_list)

    # Build from root collections
    shown_collections = 0
    shown_objects = 0
    for rc in root_collections:
        add_collection(rc, "", 0)
    # Count objects/collections shown from nodes
    for n in nodes:
        if n.get("kind") == "collection":
            shown_collections += 1
        else:
            shown_objects += 1

    requested = {
        "expand_depth": int(expand_depth),
        "expand": expand,
        "focus": focus,
        "max_children": int(max_children),
        "include_icons": bool(include_icons),
        "include_counts": bool(include_counts),
        "root_filter": root_filter or "",
    }

    legend = (
        {
            "M": "Mesh",
            "AR": "Armature",
            "Lt": "Light",
            "Cam": "Camera",
            "Cur": "Curve",
            "E": "Empty",
            "Mo": "Has modifiers",
            "Mat": "Has material(s)",
            "Ch": "Has children",
            "S": "Selected",
            "A": "Active object",
            "Ac": "Active collection",
        }
        if include_icons
        else {}
    )
    result = {
        "requested": requested,
        "outliner": {
            "lines": lines,
            "nodes": nodes,
            "legend": legend,
        },
        "changed": {},
        "unchanged": {},
        "failed": {},
        "fold_state": fold_state,
        "summary": f"{shown_collections} collections, {shown_objects} objects shown",
    }
    return result


def get_object_info(name: str) -> dict:
    """Get detailed information about a specific object."""
    obj = bpy.data.objects.get(name)
    if not obj:
        return {"error": f"Object not found: {name}"}

    info = {
        "name": obj.name,
        "type": obj.type,
        "visible": not obj.hide_viewport,
        "render": not obj.hide_render,
        "location": [round(v, 4) for v in obj.location],
        "rotation_euler": [round(v, 4) for v in obj.rotation_euler],
        "scale": [round(v, 4) for v in obj.scale],
        "dimensions": [round(v, 4) for v in obj.dimensions],
        "parent": obj.parent.name if obj.parent else None,
        "collections": [c.name for c in obj.users_collection],
        # Unified AST-like representation
        "data": _serialize_rna(obj, depth=0, max_depth=1)
    }

    # Data block info
    # Data block info
    if obj.data:
        info["mesh_stats"] = {"name": obj.data.name}

        # Mesh specific
        if obj.type == "MESH":
            mesh = obj.data
            info["mesh_stats"]["vertices"] = len(mesh.vertices)
            info["mesh_stats"]["polygons"] = len(mesh.polygons)
            info["mesh_stats"]["shape_keys"] = (
                [k.name for k in mesh.shape_keys.key_blocks] if mesh.shape_keys else []
            )

            # Vertex Groups
            info["vertex_groups"] = [vg.name for vg in obj.vertex_groups]

    # Materials
    info["materials"] = []
    for slot in obj.material_slots:
        mat_info = {
            "slot": slot.name,
            "material": slot.material.name if slot.material else None,
        }
        info["materials"].append(mat_info)

    # Modifiers
    info["modifiers"] = []
    for mod in obj.modifiers:
        info["modifiers"].append(
            {"name": mod.name, "type": mod.type, "enabled": mod.show_viewport}
        )

    # Constraints
    info["constraints"] = []
    for const in obj.constraints:
        info["constraints"].append(
            {"name": const.name, "type": const.type, "enabled": const.enabled}
        )

        # Animation
        info["action"] = obj.animation_data.action.name

    return info



def _serialize_rna(data, depth: int = 1, max_depth: int = 1) -> typing.Any:
    """Recursively serialize a Blender RNA object to a JSON-compatible dict."""
    if depth > max_depth:
        return str(data)

    # Handle basic types
    if data is None:
        return None
    if isinstance(data, (str, int, float, bool)):
        return data
    
    # Handle sequences (bpy_prop_array, lists, tuples)
    if hasattr(data, "__iter__") and not isinstance(data, (str, dict)):
        # Limit sequence length to avoid massive output (e.g. vertices)
        try:
            l = len(data)
            if l > 100:
                return f"<Sequence length={l}>"
            return [_serialize_rna(item, depth, max_depth) for item in data]
        except:
            pass

    # Handle Blender Objects (ID types)
    if hasattr(data, "rna_type"):
        # It's a Blender struct
        result = {"_type": data.rna_type.name}
        
        # If it's an ID (has a name), include it
        if hasattr(data, "name"):
            result["name"] = data.name
            
        # If we are at max depth, just return the name/type reference
        if depth == max_depth:
            return result

        # Introspect properties
        try:
            for prop in data.rna_type.properties:
                if prop.identifier in {"rna_type"}:
                    continue
                
                # Skip large collections/arrays at this level unless explicitly requested?
                # For now, we rely on the sequence length check above.
                
                try:
                    val = getattr(data, prop.identifier)
                    result[prop.identifier] = _serialize_rna(val, depth + 1, max_depth)
                except Exception:
                    result[prop.identifier] = "<Error>"
        except Exception as e:
            result["_error"] = str(e)
            
        return result

    # Fallback
    return str(data)


def _get_object_summary(obj) -> dict:
    """Return a concise summary of an object for scene updates."""
    if not obj:
        return None
        
    summary = {
        "name": obj.name,
        "type": getattr(obj, "type", "UNKNOWN"),
    }
    
    # Add transform info if available (rounded for brevity)
    if hasattr(obj, "location"):
        summary["location"] = [round(v, 2) for v in obj.location]
    if hasattr(obj, "dimensions"):
        summary["dimensions"] = [round(v, 2) for v in obj.dimensions]
        
    # Hierarchy info
    if getattr(obj, "parent", None):
        summary["parent"] = obj.parent.name
        
    # Collection info
    if hasattr(obj, "users_collection"):
        summary["collections"] = [c.name for c in obj.users_collection]
        
    return summary


def inspect_data(path: str, depth: int = 1) -> dict:
    """Introspect a Blender data block and return its structure (AST-like).
    
    Args:
        path: Python path to the data (e.g. "bpy.data.objects['Cube']", "bpy.context.scene")
        depth: How deep to traverse relationships (default: 1)
        
    Returns:
        Dict containing the serialized data structure.
    """
    try:
        # Resolve path
        # We use the persistent namespace to resolve 'bpy'
        namespace = _get_code_namespace()
        
        # Security/Safety check: only allow access to bpy
        if not path.startswith("bpy."):
             return {"error": "Path must start with 'bpy.'"}

        try:
            # Eval the path to get the object
            obj = eval(path, namespace, namespace)
        except Exception as e:
            return {"error": f"Failed to resolve path '{path}': {str(e)}"}
            
        # Serialize
        return {
            "path": path,
            "data": _serialize_rna(obj, 0, depth)
        }
        
    except Exception as e:
        return {"error": f"Inspection failed: {str(e)}"}


def search_data(root_path: str, filter_props: dict = None, max_results: int = 10) -> dict:
    """Search for data blocks matching specific criteria (AST-grep like).
    
    Args:
        root_path: Path to a collection (e.g. "bpy.data.objects")
        filter_props: Dict of property=value to match (e.g. {"type": "MESH", "hide_viewport": False})
        max_results: Max number of matches to return
        
    Returns:
        List of matches with summaries.
    """
    try:
        namespace = _get_code_namespace()
        
        if not root_path.startswith("bpy."):
             return {"error": "Root path must start with 'bpy.'"}
             
        try:
            root = eval(root_path, namespace, namespace)
        except Exception as e:
            return {"error": f"Failed to resolve root '{root_path}': {str(e)}"}
            
        # Ensure root is iterable
        if not hasattr(root, "__iter__"):
            return {"error": f"Root '{root_path}' is not iterable"}
            
        matches = []
        count = 0
        
        for item in root:
            match = True
            if filter_props:
                for key, target_val in filter_props.items():
                    # Support nested keys? e.g. "data.materials"
                    # For now, simple attributes
                    try:
                        val = getattr(item, key, None)
                        if val != target_val:
                            match = False
                            break
                    except:
                        match = False
                        break
            
            if match:
                # Create a summary
                summary = {
                    "name": getattr(item, "name", str(item)),
                    "type": getattr(item, "type", "Unknown") if hasattr(item, "type") else type(item).__name__
                }
                # Add the filtered props to summary for confirmation
                if filter_props:
                    for k in filter_props:
                        summary[k] = getattr(item, k, None)
                        
                matches.append(summary)
                count += 1
                if count >= max_results:
                    break
                    
        return {
            "root": root_path,
            "filter": filter_props,
            "count": len(matches),
            "matches": matches
        }

    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


# Persistent namespace for execute_code (maintains state between calls)
_CODE_NAMESPACE = None


def _get_code_namespace():
    """Get or create the persistent code execution namespace."""

    global _CODE_NAMESPACE

    if _CODE_NAMESPACE is None:
        import bmesh
        import mathutils
        import numpy as np

        try:
            from . import context_utils as _ctx_utils

        except Exception:
            _ctx_utils = None

        _CODE_NAMESPACE = {
            "bpy": bpy,
            "mathutils": mathutils,
            "bmesh": bmesh,
            "numpy": np,
            "np": np,
            "Vector": mathutils.Vector,
            "Matrix": mathutils.Matrix,
            "Euler": mathutils.Euler,
            "Color": mathutils.Color,
            "context_utils": _ctx_utils,
            "__builtins__": __builtins__,
            # Helper to explore available functions
            "help": help,
            "dir": dir,
            "globals": lambda: _CODE_NAMESPACE,
            "globals": lambda: _CODE_NAMESPACE,
            "assistant_sdk": AssistantSDK(),
        }

        # Install a robust importable shim module
        try:
            import sys
            import types

            sdk_obj = _CODE_NAMESPACE["assistant_sdk"]

            def _create_shim_module(name, obj):
                """Recursively create a module shim for an object."""
                mod = types.ModuleType(name)

                # Copy attributes
                for attr in dir(obj):
                    if not attr.startswith("_"):
                        val = getattr(obj, attr)
                        setattr(mod, attr, val)

                        # If attribute is a class/object with its own attributes,
                        # create a submodule for it too (for 'from X.Y import Z')
                        if hasattr(val, "__dict__") and not isinstance(
                            val, (int, float, str, bool, list, dict, tuple)
                        ):
                            # Check if it looks like a namespace/class we want to expose as a module
                            pass

                return mod

            # 1. Create top-level assistant_sdk module
            _as_mod = _create_shim_module("assistant_sdk", sdk_obj)
            sys.modules["assistant_sdk"] = _as_mod

            # 2. Create submodules for each tool category dynamically
            for category in dir(sdk_obj):
                if category.startswith("_") or category == "help":
                    continue
                    
                cat_obj = getattr(sdk_obj, category)
                
                # Only process if it looks like a namespace object (has _tools or similar)
                # or just try to wrap it if it's not a basic type
                if isinstance(cat_obj, (int, float, str, bool, list, dict, tuple)):
                    continue

                cat_mod_name = f"assistant_sdk.{category}"

                # Create the submodule
                cat_mod = _create_shim_module(cat_mod_name, cat_obj)
                sys.modules[cat_mod_name] = cat_mod

                # Link it to the parent
                setattr(_as_mod, category, cat_mod)

            # Bind module into namespace so 'import assistant_sdk' works inside exec()
            _CODE_NAMESPACE["assistant_sdk"] = _as_mod

        except Exception as e:
            print(f"[Assistant] Failed to create SDK shim: {e}")
            # Best-effort; if this fails, direct namespace access still works
            pass
    return _CODE_NAMESPACE


def execute_code(code: str) -> dict:
    """Execute Python code in Blender's context with persistent state.

    This is the most powerful tool - it gives you direct access to Blender's Python API (bpy).
    State persists between calls, so you can define variables/functions and reuse them.

    Available in namespace:
    - bpy: Blender Python API
    - mathutils: Vector, Matrix, Euler, etc.
    - assistant_sdk: Pre-initialized SDK for tools (blender, polyhaven, sketchfab, stock_photos, web, rag)
    - Any variables/functions you define persist between calls


    **BEST PRACTICE**: Use `assistant_sdk.<namespace>.*` methods (e.g., `assistant_sdk.blender.move_to_collection`)

    instead of writing raw `bpy` logic for common tasks. Call methods directly via `assistant_sdk` and avoid importing submodules (e.g., do NOT use `from assistant_sdk.web import search`). They are safer and handle edge cases.



    IMPORTANT: assistant_sdk is already available in the namespace. Do NOT import it, and do NOT import its submodules.
    Always call via `assistant_sdk.web.search(...)`, `assistant_sdk.web.download_image(...)`, `assistant_sdk.blender.create_object(...)`, etc. Importing like `from assistant_sdk.web import search` is not supported with the runtime shim and can fail or produce stale references.


    Examples:
        # Use assistant_sdk (already available, no import needed!)
        assets = assistant_sdk.polyhaven.search(asset_type='model', query='tree', limit=5)
        for asset in assets.get('assets', []):
            assistant_sdk.polyhaven.download(asset=asset, asset_type='model')

        # Create and modify objects with bpy
        bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
        cube = bpy.context.active_object
        cube.scale = (2, 2, 2)

        # Or use assistant_sdk.blender
        assistant_sdk.blender.create_object('CUBE', location=[0, 0, 0])
        assistant_sdk.blender.set_material(object_name='Cube', color=[1, 0, 0, 1])

        # Define reusable functions
        def create_sphere(name, location):
            bpy.ops.mesh.primitive_uv_sphere_add(location=location)
            sphere = bpy.context.active_object
            sphere.name = name
            return sphere

        # Use state from previous calls
        my_sphere = create_sphere("MySphere", (5, 0, 0))


    Returns:

        Dict with success status and any print() output.

        To return a structured value from your code, assign it to a variable named
        'result' or '__result__' (e.g., result = assistant_sdk.polyhaven.download(...)).
        The value will be included in the tool response under the 'result' key.
    """

    try:
        import io
        import sys
        import traceback
        
        # Sanitize code input: Convert literal \n to actual newlines if detected
        # LLMs often send "line1\nline2" as a single line string with literal backslashes
        if "\\n" in code and "\n" not in code:
             code = code.replace("\\n", "\n")
             
        import io
        import sys
        import traceback

        # Get persistent namespace
        namespace = _get_code_namespace()

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            exec(code, namespace, namespace)
            output = captured_output.getvalue()
        finally:
            sys.stdout = old_stdout

        # Return format compatible with original, but with output

        result = {"executed": True, "success": True}

        if output:
            result["output"] = output.strip()

        # If the executed code assigned a value to 'result' or '__result__',
        # include it in the tool response for better feedback.
        if "result" in namespace:
            try:
                result["result"] = namespace["result"]
            except Exception:
                result["result"] = str(namespace["result"])
        elif "__result__" in namespace:
            try:
                result["result"] = namespace["__result__"]
            except Exception:
                result["result"] = str(namespace["__result__"])

        return result

    except Exception as e:
        import traceback

        tb = traceback.format_exc()

        result = {"error": str(e), "traceback": tb}

        # Recovery hint: assistant_sdk is injected into the execute_code namespace already.

        # If the user tries to import it, guide them to use it directly.

        err_low = (str(e) or "").lower()

        code_low = (code or "").lower()

        if (
            "no module named 'assistant_sdk'" in err_low
            or "no module named 'assistant_sdk'" in tb.lower()
            or "name 'assistant_sdk' is not defined" in err_low
            or "import assistant_sdk" in code_low
            or "from assistant_sdk" in code_low
        ):
            result["hint"] = (
                "assistant_sdk is already available in the execute_code namespace. "
                "Use it directly, e.g., assistant_sdk.polyhaven.search(...); do not import it."
            )

        # Recovery hint: PolyHaven search returns a dict with an 'assets' list.
        # Iterating the whole 'results' dict and indexing like asset['id'] will fail.
        if (
            "string indices must be integers" in err_low
            and "polyhaven.search" in code_low
            and " in results" in code_low
        ):
            extra = (
                "PolyHaven search returns a dict; iterate results['assets'] and pass asset_type to download. Example:\n"
                "assets = assistant_sdk.polyhaven.search(asset_type='model', query='tree', limit=5)\n"
                "for a in assets.get('assets', []):\n"
                "    assistant_sdk.polyhaven.download(asset=a, asset_type='model')"
            )
            if "hint" in result:
                result["hint"] += " | " + extra
            else:
                result["hint"] = extra

        return result


def capture_viewport_for_vision(
    question: str,
    max_size: int = 1024,
    vision_model: str = "",
    timeout_s: int = 15,
) -> dict:
    """Synchronously capture the viewport and run a dedicated vision model.

    Behavior:
    - Captures the current viewport and scales it to max_size if needed.
    - Calls the configured vision model with the image and the provided question.
    - Returns a single result with textual description and metadata.
    - Never returns the image and never sends images to the chat model.

    Args:
        question: Instruction for the vision model (what to analyze).
        max_size: Max width/height for the screenshot.
        vision_model: Optional override model name (uses preferences if empty).
        timeout_s: Max seconds to wait for the vision analysis.

    Returns:
        dict with keys:
            success: bool
            description: str
            original_resolution: [w,h]
            scaled_resolution: [w,h]
            viewport_info: { ... }
            vision_model: str
    """
    import base64
    import threading

    try:
        import tempfile

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "blender_viewport_vision.png")

        # Capture viewport (UI thread operation)
        scene = bpy.context.scene
        prev_fp = scene.render.filepath
        scene.render.filepath = temp_path
        try:
            bpy.ops.render.opengl(write_still=True)
        finally:
            scene.render.filepath = prev_fp

        # Load and scale image
        if not os.path.exists(temp_path):
            return {"error": "Failed to capture viewport"}

        img = bpy.data.images.load(temp_path)

        try:
            original_width = img.size[0]
            original_height = img.size[1]

            if original_width > max_size or original_height > max_size:
                if original_width >= original_height:
                    new_width = max_size
                    new_height = int(original_height * (max_size / original_width))
                else:
                    new_height = max_size
                    new_width = int(original_width * (max_size / original_height))
                img.scale(new_width, new_height)
                img.save_render(temp_path)
            else:
                new_width = original_width
                new_height = original_height

            # Read and encode as base64 for the VLM call only (not returned)
            with open(temp_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode("utf-8")

            # Clean up Blender image datablock
            bpy.data.images.remove(img)

            # Viewport info
            area = None
            for a in bpy.context.screen.areas:
                if a.type == "VIEW_3D":
                    area = a
                    break

            viewport_info = {}
            if area:
                space = area.spaces.active
                viewport_info = {
                    "shading": space.shading.type,
                    "overlay": space.overlay.show_overlays,
                    "camera_view": space.region_3d.view_perspective == "CAMERA",
                }

            # Resolve vision model from preferences or override
            model_name = vision_model or ""
            
            # Handle "default" alias from LLM or preferences
            if model_name.lower() == "default":
                model_name = ""

            try:
                if not model_name:
                    from ..preferences import get_preferences
                    prefs = get_preferences()
                    if prefs:
                        # Prefer dedicated vision model, fallback to active chat model
                        model_name = getattr(prefs, "vision_model", "") or getattr(
                            prefs, "model_file", ""
                        )
                        
                        # If preference is also "default" (unlikely but possible), clear it
                        if model_name and model_name.lower() == "default":
                            model_name = ""
                            
                        # Hard fallback to minicpm-v:8b if nothing else is set
                        if not model_name:
                            model_name = "minicpm-v:8b"
            except Exception as e:
                print(f"[Vision] Error reading preferences: {e}")
                pass
            
            print(f"[Vision] Using model: {model_name}")

            if not model_name or model_name == "NONE":
                return {
                    "error": "Vision model not set. Configure a vision-capable model in preferences or pass vision_model.",
                    "original_resolution": [original_width, original_height],
                    "scaled_resolution": [new_width, new_height],
                    "viewport_info": viewport_info,
                }

            prompt = (
                (question or "").strip()
                or "Provide a concise description of the viewport contents (objects, layout, materials, lighting)."
            )

            # Call the VLM with a bounded wait using a worker thread
            result_box = {"resp": None, "err": None}

            def _call_vlm():
                try:
                    from .. import ollama_adapter as llama_manager

                    messages = [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_b64],
                        }
                    ]
                    print(f"[Vision] Sending request to {model_name} with prompt: {prompt[:50]}...")
                    result_box["resp"] = llama_manager.chat_completion(
                        model_path=model_name,
                        messages=messages,
                        temperature=0.1,
                        max_tokens=1024,
                    )
                except Exception as e:
                    result_box["err"] = str(e)

            th = threading.Thread(target=_call_vlm, daemon=True)
            th.start()
            th.join(max(1, int(timeout_s)))

            if th.is_alive():
                # Keep thread daemon; return timeout error without image
                return {
                    "error": f"Vision analysis timed out after {timeout_s}s",
                    "original_resolution": [original_width, original_height],
                    "scaled_resolution": [new_width, new_height],
                    "viewport_info": viewport_info,
                    "vision_model": model_name,
                }

            if result_box["err"]:
                return {"error": f"Vision analysis failed: {result_box['err']}"}

            resp = result_box["resp"] or {}
            desc = ""
            if isinstance(resp, dict):
                if "message" in resp and isinstance(resp["message"], dict):
                    desc = resp["message"].get("content", "") or ""
                elif "content" in resp:
                    desc = resp.get("content", "") or ""
            
            if not desc:
                return {
                    "error": f"Vision model '{model_name}' returned empty response. Ensure it is a vision-capable model (e.g. llava).",
                    "vision_model": model_name,
                    "original_resolution": [original_width, original_height],
                    "scaled_resolution": [new_width, new_height],
                    "viewport_info": viewport_info,
                }

            return {
                "success": True,
                "description": (desc or "").strip(),
                "original_resolution": [original_width, original_height],
                "scaled_resolution": [new_width, new_height],
                "viewport_info": viewport_info,
                "vision_model": model_name,
            }

        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

    except Exception as e:
        import traceback

        return {
            "error": f"Failed to capture/analyze vision: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def list_collections() -> dict:
    """List all collections in the scene.

    Returns:
        Dictionary with list of collections and their hierarchy
    """
    try:
        collections_info = []

        def get_collection_info_recursive(col, depth=0):
            """Recursively get collection info."""
            objects = [obj.name for obj in col.objects]
            children = [child.name for child in col.children]

            return {
                "name": col.name,
                "depth": depth,
                "object_count": len(objects),
                "objects": objects[:10],  # Limit to first 10
                "has_more_objects": len(objects) > 10,
                "children": children,
            }

        # Get master collection
        master = bpy.context.scene.collection
        collections_info.append(get_collection_info_recursive(master, 0))

        # Get all other collections recursively
        def add_children(col, depth):
            for child in col.children:
                collections_info.append(get_collection_info_recursive(child, depth))
                add_children(child, depth + 1)

        add_children(master, 1)

        return {
            "success": True,
            "total_collections": len(collections_info),
            "collections": collections_info,
        }
    except Exception as e:
        return {"error": f"Failed to list collections: {str(e)}"}


def get_collection_info(collection_name: str) -> dict:
    """Get detailed information about a specific collection.

    Args:
        collection_name: Name of the collection

    Returns:
        Dictionary with collection details
    """
    try:
        col = bpy.data.collections.get(collection_name)

        if not col:
            # Check if it's the master collection
            if collection_name == bpy.context.scene.collection.name:
                col = bpy.context.scene.collection
            else:
                return {"error": f"Collection '{collection_name}' not found"}

        objects = [obj.name for obj in col.objects]
        children = [child.name for child in col.children]

        # Find parent
        parent_name = None
        for potential_parent in bpy.data.collections:
            if col in potential_parent.children.values():
                parent_name = potential_parent.name
                break

        if not parent_name and col != bpy.context.scene.collection:
            # Check if parent is master collection
            if col in bpy.context.scene.collection.children.values():
                parent_name = bpy.context.scene.collection.name

        return {
            "success": True,
            "name": col.name,
            "parent": parent_name,
            "object_count": len(objects),
            "objects": objects,
            "child_count": len(children),
            "children": children,
        }
    except Exception as e:
        return {"error": f"Failed to get collection info: {str(e)}"}


# Helpers for robust collection resolution


def _get_scene_root():
    try:
        return bpy.context.scene.collection
    except Exception:
        return None


def _normalize_name(name: str) -> str:
    return (name or "").strip().lower()


def resolve_collection_by_name(name: str) -> typing.Optional[bpy.types.Collection]:
    """Resolve a collection by name with special handling for scene root and case-insensitive matches.
    - Treats "Scene Collection"/"Master Collection" (case-insensitive) as the scene root
    - Falls back to case-insensitive match across bpy.data.collections
    """
    if not name:
        return None
    scene_root = _get_scene_root()
    norm = _normalize_name(name)
    # Scene root synonyms
    if scene_root is not None and (
        norm == _normalize_name(scene_root.name)
        or norm in {"scene collection", "master collection"}
    ):
        return scene_root
    # Exact match first
    col = bpy.data.collections.get(name)
    if col:
        return col
    # Case-insensitive search
    try:
        for c in bpy.data.collections:
            if _normalize_name(c.name) == norm:
                return c
    except Exception:
        pass
    return None


def create_collection(
    name: str = None, parent: str = None, collections: list = None
) -> dict:
    """Create one or more collections.

    Args:
        name: Name for a single collection (use this OR collections, not both)
        parent: Parent collection name for single collection
        collections: List of dicts with 'name' and optional 'parent' keys for batch creation
                    Example: [{"name": "Buildings", "parent": "City"}, {"name": "Streets"}]

    Returns:
        Dictionary with created collection states (not generic success message)
    """
    try:
        # Batch mode: create multiple collections
        if collections:
            created = {}
            failed = {}

            # If a top-level name is provided alongside batch, ensure it exists first.
            # This supports calls like: name="SpaceScene", collections=[{"name":"Earth","parent":"SpaceScene"}, ...]
            if name:
                if name in bpy.data.collections:
                    # Already exists; ok
                    pass
                else:
                    # Create the parent collection and link it to the scene or given parent
                    top_col = bpy.data.collections.new(name)
                    if parent:
                        parent_col = bpy.data.collections.get(parent)
                        if not parent_col:
                            if parent == bpy.context.scene.collection.name:
                                parent_col = bpy.context.scene.collection
                            else:
                                # Could not find designated parent; fall back to scene root
                                parent_col = bpy.context.scene.collection
                        parent_col.children.link(top_col)
                    else:
                        bpy.context.scene.collection.children.link(top_col)

            for col_spec in collections:
                col_name = col_spec.get("name")
                col_parent = col_spec.get("parent")

                if not col_name:
                    failed[f"spec_{len(failed)}"] = {
                        "error": "Collection spec missing 'name'",
                        "spec": col_spec,
                    }
                    continue

                # Check if already exists
                if col_name in bpy.data.collections:
                    failed[col_name] = {"error": "Already exists"}
                    continue

                # Create collection
                new_col = bpy.data.collections.new(col_name)

                # Link to parent or scene
                if col_parent:
                    parent_col = bpy.data.collections.get(col_parent)
                    if not parent_col:
                        if col_parent == bpy.context.scene.collection.name:
                            parent_col = bpy.context.scene.collection
                        else:
                            # If parent matches the provided top-level name, we just ensured it exists
                            if (
                                name
                                and col_parent == name
                                and col_parent in bpy.data.collections
                            ):
                                parent_col = bpy.data.collections[col_parent]
                            else:
                                failed[col_name] = {
                                    "error": f"Parent '{col_parent}' not found"
                                }
                                bpy.data.collections.remove(new_col)
                                continue
                    parent_col.children.link(new_col)
                else:
                    bpy.context.scene.collection.children.link(new_col)

                # Record created collection state
                created[col_name] = {
                    "parent": col_parent or "Scene",
                    "object_count": 0,
                    "color_tag": getattr(new_col, "color_tag", None),
                }

            # Build result
            result = {
                "action": "create_collection",
                "requested": {
                    "collections": [c.get("name") for c in collections if c.get("name")]
                },
                "created": created,
                "failed": failed,
                "created_count": len(created),
                "failed_count": len(failed),
            }

            # Add summary
            summary_parts = []
            if created:
                summary_parts.append(f"Created {len(created)} collection(s)")
            if failed:
                summary_parts.append(f"{len(failed)} failed")

            result["summary"] = (
                ". ".join(summary_parts) if summary_parts else "No collections created"
            )

            return result

        # Single mode: create one collection
        if not name:
            return {"error": "Must provide 'name' or 'collections' parameter"}

        # Check if collection already exists
        if name in bpy.data.collections:
            return {"error": f"Collection '{name}' already exists"}

        # Create new collection
        new_col = bpy.data.collections.new(name)

        # Link to parent or scene
        parent_name = None
        if parent:
            parent_col = resolve_collection_by_name(parent)
            if not parent_col:
                return {"error": f"Parent collection '{parent}' not found"}
            parent_col.children.link(new_col)
            parent_name = parent_col.name
        else:
            # Link to scene master collection
            scene_root = _get_scene_root()
            (scene_root or bpy.context.scene.collection).children.link(new_col)
            parent_name = "Scene"

        # Return state of created collection
        return {
            "action": "create_collection",
            "created": {
                name: {
                    "parent": parent_name,
                    "object_count": 0,
                    "color_tag": getattr(new_col, "color_tag", None),
                }
            },
            "summary": f"Created collection '{name}' in {parent_name}",
        }
    except Exception as e:
        return {"error": f"Failed to create collection: {str(e)}"}


def move_to_collection(
    object_names: list,
    collection_name: str,
    unlink_from_others: bool = True,
    create_if_missing: bool = True,
) -> dict:
    """Move one or more objects into a target collection (preferred over manual link/unlink).

    This tool:
      - Unlinks from all other collections by default (unlink_from_others=True) — safe and idempotent
      - Creates the target collection when missing (create_if_missing=True)
      - Returns before/after collection membership per object

    Args:
        object_names: List of object names to move

        collection_name: Target collection name

        unlink_from_others: If True, unlink from all other collections first (default True)

        create_if_missing: If True, create the target collection if missing (default True)

    Returns:
        Dict with requested parameters, per-object before/after membership, and final collection state
    """

    try:
        # Resolve or create target collection
        target_col = resolve_collection_by_name(collection_name)

        if not target_col:
            if create_if_missing:
                create_result = create_collection(name=collection_name)
                if isinstance(create_result, dict) and create_result.get("error"):
                    return create_result
                target_col = resolve_collection_by_name(collection_name)
            else:
                return {"error": f"Collection '{collection_name}' not found"}

        # Track state changes
        changed = {}
        failed = {}

        for obj_name in object_names:
            obj = bpy.data.objects.get(obj_name)

            if not obj:
                failed[obj_name] = {"error": "Object not found"}
                continue

            # Capture before state (which collections this object is in)
            before_collections = [col.name for col in obj.users_collection]

            # Unlink from other collections if requested
            if unlink_from_others:
                for col in list(obj.users_collection):
                    try:
                        col.objects.unlink(obj)
                    except Exception:
                        pass

            # Link to target collection
            if obj.name not in target_col.objects:
                target_col.objects.link(obj)

            # Capture after state
            after_collections = [col.name for col in obj.users_collection]

            # Record change
            changed[obj_name] = {
                "before": {"collections": before_collections},
                "after": {"collections": after_collections},
            }

        # Get final collection state
        collection_state = {
            collection_name: {
                "object_count": len(target_col.objects),
                "objects": [obj.name for obj in target_col.objects],
            }
        }

        # Build result
        result = {
            "action": "move_to_collection",
            "requested": {
                "objects": object_names,
                "collection": collection_name,
                "unlink_from_others": unlink_from_others,
            },
            "changed": changed,
            "failed": failed,
            "collection_state": collection_state,
            "completed": len(changed),
            "failed_count": len(failed),
        }

        # Add natural language summary
        summary_parts = []
        if changed:
            summary_parts.append(
                f"Moved {len(changed)}/{len(object_names)} objects to {collection_name}"
            )
        if failed:
            failed_names = ", ".join(failed.keys())
            summary_parts.append(f"Failed: {failed_names}")

        result["summary"] = (
            ". ".join(summary_parts) if summary_parts else "No objects moved"
        )

        return result

    except Exception as e:
        return {"error": f"Failed to move objects: {str(e)}"}


def set_collection_color(
    collection_name: str = None,
    color_tag: str = "COLOR_01",
    collection_names: list = None,
) -> dict:
    """Set the Outliner color tag on one or more collections (Blender 4.2+).



    Prefer this for scene organization over coloring object materials. Supports single
    collection_name or batch collection_names; reports changed/unchanged/failed.

    Args:
        collection_name: Single collection name
        collection_names: List of collections

        color_tag: COLOR_01..COLOR_08 or NONE



    Returns:

        Dict with changed/unchanged/failed and a summary
    """

    try:
        names = []
        if collection_names:
            names.extend(collection_names or [])
        if collection_name:
            names.append(collection_name)

        if not names:
            return {"error": "Provide collection_name or collection_names"}

        # Track before/after states
        changed = {}
        unchanged = {}
        failed = {}

        for name in names:
            col = resolve_collection_by_name(name)
            if not col:
                failed[name] = {"error": "Collection not found"}
                continue
            if not hasattr(col, "color_tag"):
                failed[name] = {
                    "error": "No color_tag property (Blender 4.2+ required)"
                }
                continue

            # Get object count for context
            obj_count = len(col.objects)

            # Capture before state
            before_color = col.color_tag

            # Apply change
            col.color_tag = color_tag

            # Record state change
            if before_color == color_tag:
                # No actual change
                unchanged[name] = {"color_tag": color_tag, "object_count": obj_count}
            else:
                changed[name] = {
                    "before": {"color_tag": before_color, "object_count": obj_count},
                    "after": {"color_tag": color_tag, "object_count": obj_count},
                }

        # Build result
        result = {
            "action": "set_collection_color",
            "requested": {"collections": names, "color_tag": color_tag},
            "changed": changed,
            "unchanged": unchanged,
            "failed": failed,
            "completed": len(changed),
            "unchanged_count": len(unchanged),
            "failed_count": len(failed),
        }

        # Add natural language summary
        summary_parts = []
        if changed:
            summary_parts.append(f"Colored {len(changed)}/{len(names)} collections")
        if unchanged:
            summary_parts.append(f"{len(unchanged)} already had {color_tag}")
        if failed:
            failed_names = ", ".join(failed.keys())
            summary_parts.append(f"Failed: {failed_names}")

        result["summary"] = (
            ". ".join(summary_parts) if summary_parts else "No changes made"
        )

        return result
    except Exception as e:
        return {"error": f"Failed to set collection color: {str(e)}"}


def delete_collection(collection_name: str, delete_objects: bool = False) -> dict:
    """Delete a collection.

    Args:
        collection_name: Name of collection to delete
        delete_objects: If True, also delete objects in the collection

    Returns:
        Dictionary with deletion result
    """
    try:
        col = resolve_collection_by_name(collection_name)
        if not col:
            return {"error": f"Collection '{collection_name}' not found"}

        # Cannot delete master collection
        scene_root = _get_scene_root()
        if scene_root and col == scene_root:
            return {"error": "Cannot delete the master scene collection"}

        object_count = len(col.objects)

        if delete_objects:
            # Delete all objects in collection
            for obj in list(col.objects):
                bpy.data.objects.remove(obj, do_unlink=True)

        # Remove collection
        bpy.data.collections.remove(col)

        return {
            "success": True,
            "deleted": collection_name,
            "objects_deleted": object_count if delete_objects else 0,
            "objects_preserved": 0 if delete_objects else object_count,
        }
    except Exception as e:
        return {"error": f"Failed to delete collection: {str(e)}"}


def assistant_help(tool: str = "", tools: list | None = None, **kwargs) -> dict:
    """Return usage information for assistant_sdk tools.

    Can look up tools by alias (e.g. "polyhaven.search") or namespace (e.g. "assistant_sdk.polyhaven").
    Supports multiple lookups and fuzzy matching.
    """
    try:
        # Build list of requested queries
        queries: list[str] = []
        if tools and isinstance(tools, (list, tuple)):
            queries.extend([str(t).strip() for t in tools if str(t).strip()])
        if tool and str(tool).strip():
            queries.append(str(tool).strip())

        if not queries:
            return {"error": "Missing 'tool' or 'tools' parameter"}

        # Define the knowledge base of SDK tools (Manual Overrides & Virtual Tools)
        # Format: alias -> {sdkUsage, notes, returns?}
        sdk_docs = {
            # Virtual / Workflow items (Not real tools)
            "web.images_workflow": {
                "workflow": [
                    "# 3-STEP WORKFLOW for downloading web images:",
                    "1. results = assistant_sdk.web.search(query='kittens')  # Get pages about kittens",
                    "2. images = assistant_sdk.web.extract_images(results['results'][0]['url'])  # Extract image URLs",
                    "3. assistant_sdk.web.download_image(images['images'][0])  # Download first image",
                ],
                "notes": "Multi-step process: SEARCH for pages → EXTRACT image URLs → DOWNLOAD the image.",
            },
            # Common aliases
            "search": "polyhaven.search",
            "download": "polyhaven.download",
            "scene": "blender.get_scene_info",
            "info": "blender.get_object_info",
        }

        # Get all registered tools dynamically
        all_tools = tool_registry.get_tools_list()
        results = []

        for q in queries:
            # Normalize query: remove 'assistant_sdk.' prefix
            clean_q = q.replace("assistant_sdk.", "").lower()
            
            # Check manual docs first (for workflows/aliases)
            if clean_q in sdk_docs:
                entry = sdk_docs[clean_q]
                # If it's an alias string, resolve it
                if isinstance(entry, str):
                    clean_q = entry # Resolve alias and continue to dynamic lookup
                else:
                    # It's a virtual tool/workflow
                    results.append({
                        "tool": clean_q,
                        "sdkUsage": entry.get("sdkUsage", "N/A"),
                        "notes": entry.get("notes", ""),
                        "returns": entry.get("returns", "dict"),
                        "workflow": entry.get("workflow", None)
                    })
                    continue

            # Split by slash if present (e.g. "polyhaven.search/download")
            sub_queries = clean_q.split("/")
            
            for sub_q in sub_queries:
                sub_q = sub_q.strip()
                if not sub_q:
                    continue

                # Handle namespace.tool_name (e.g., polyhaven.download_polyhaven -> download_polyhaven)
                query_parts = sub_q.split(".")
                query_tool_name = query_parts[-1]
                
                found_for_query = False
                
                for t in all_tools:
                    t_name = t["name"]
                    t_cat = t.get("category", "Other").lower()
                    
                    # Check for match
                    is_match = False
                    
                    # 1. Exact match on tool name
                    if query_tool_name == t_name.lower():
                        is_match = True
                    
                    # 2. Namespace match (e.g. "polyhaven" matches all tools in PolyHaven category)
                    elif query_tool_name == t_cat:
                        is_match = True
                        
                    # 3. Substring match (if query is specific enough)
                    elif len(query_tool_name) > 3 and query_tool_name in t_name.lower():
                        is_match = True
                        
                    if is_match:
                        # Format usage string
                        schema = t["inputSchema"]
                        props = schema.get("properties", {})
                        args_list = []
                        for prop_name, prop_info in props.items():
                            arg_str = f"{prop_name}"
                            if "default" in prop_info:
                                arg_str += f"={repr(prop_info['default'])}"
                            args_list.append(arg_str)
                        
                        # Construct namespace for usage (e.g. assistant_sdk.polyhaven.download_polyhaven)
                        # We use the category as the namespace
                        usage = f"assistant_sdk.{t_cat}.{t_name}({', '.join(args_list)})"
                        
                        results.append({
                            "tool": t_name,
                            "sdkUsage": usage,
                            "notes": t["description"],
                            "returns": "Dict (see description)",
                            "category": t_cat
                        })
                        found_for_query = True
                
                if not found_for_query:
                    # No match found
                    pass

        # Remove duplicates
        unique_results = []
        seen = set()
        for r in results:
            key = r["sdkUsage"]
            if key not in seen:
                seen.add(key)
                unique_results.append(r)

        return {"results": unique_results}

    except Exception as e:
        import traceback
        return {"error": f"Help failed: {str(e)}", "traceback": traceback.format_exc()}


def register():
    """Register all Blender tools with the MCP registry."""

    # get_scene_info

    tool_registry.register_tool(
        "get_scene_info",
        get_scene_info,
        "Get the scene state in an AST-like format. Use this to verify changes.",
        {
            "type": "object",
            "properties": {
                "expand_depth": {
                    "type": "integer",
                    "description": "Depth of hierarchy to expand (default: 2)",
                    "default": 2,
                    "minimum": 0,
                    "maximum": 6,
                }
            },
            "required": [],
        },
        category="Blender",
    )

    # get_object_info

    tool_registry.register_tool(
        "get_object_info",
        get_object_info,
        "Get detailed information about a specific object",
        {
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Object name"}},
            "required": ["name"],
        },
        category="Blender",
    )

    # create_object

    # modify_object

    # delete_object (supports batch via 'names')

    # set_material

    # execute_code

    tool_registry.register_tool(
        "execute_code",
        execute_code,
        "Execute Python code in Blender's context (use with caution)",
        {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"],
        },
        category="Code",
    )

    # capture_viewport_for_vision (vision models only)

    tool_registry.register_tool(
        "capture_viewport_for_vision",
        capture_viewport_for_vision,
        "Synchronously capture the current viewport and run a vision model to answer a question about the scene. Returns only the textual description and metadata (no image).",
        {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What to analyze in the captured viewport image (e.g., 'Are there 9 cubes? Label them.').",
                },
                "max_size": {
                    "type": "integer",
                    "description": "Max width/height in pixels for the captured image",
                    "default": 1024,
                },
                "vision_model": {
                    "type": "string",
                    "description": "Optional override for the vision model name (uses default from preferences if omitted)",
                },
                "timeout_s": {
                    "type": "integer",
                    "description": "Max seconds to wait for vision analysis before timing out",
                    "default": 15,
                    "minimum": 1,
                    "maximum": 120,
                },
            },
            "required": ["question"],
        },
        category="Vision",
    )

    # Collection tools

    # list_collections

    tool_registry.register_tool(
        "list_collections",
        list_collections,
        "List all collections with hierarchy and object counts",
        {"type": "object", "properties": {}, "required": []},
        category="Blender",
    )

    # get_collection_info

    tool_registry.register_tool(
        "get_collection_info",
        get_collection_info,
        "Get info about a specific collection (objects and children)",
        {
            "type": "object",
            "properties": {
                "collection_name": {
                    "type": "string",
                    "description": "Collection name",
                }
            },
            "required": ["collection_name"],
        },
        category="Blender",
    )

    # create_collection (supports batch via 'collections')

    tool_registry.register_tool(
        "create_collection",
        create_collection,
        "Create one or more collections. Batch via 'collections'.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Single collection name"},
                "parent": {
                    "type": "string",
                    "description": "Parent for single collection (omit for scene root)",
                },
                "collections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "parent": {"type": "string"},
                        },
                        "required": ["name"],
                    },
                    "description": "List of {name, parent?}",
                },
            },
            "required": [],
        },
        category="Blender",
    )

    # move_to_collection

    tool_registry.register_tool(
        "move_to_collection",
        move_to_collection,
        "Move objects to a collection; can create the target when missing.",
        {
            "type": "object",
            "properties": {
                "object_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Object names to move",
                },
                "collection_name": {
                    "type": "string",
                    "description": "Target collection name",
                },
                "unlink_from_others": {
                    "type": "boolean",
                    "description": "Unlink from other collections",
                    "default": True,
                },
                "create_if_missing": {
                    "type": "boolean",
                    "description": "Create target if missing",
                    "default": True,
                },
            },
            "required": ["object_names", "collection_name"],
        },
        category="Blender",
    )

    # set_collection_color (now supports batch via 'collection_names')
    tool_registry.register_tool(
        "set_collection_color",
        set_collection_color,
        "Set collection color tag (Blender 4.2+). Accepts single name or list.",
        {
            "type": "object",
            "properties": {
                "collection_name": {
                    "type": "string",
                    "description": "Single collection name",
                },
                "collection_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of collections",
                },
                "color_tag": {
                    "type": "string",
                    "description": "COLOR_01..COLOR_08 or NONE",
                    "enum": [
                        "COLOR_01",
                        "COLOR_02",
                        "COLOR_03",
                        "COLOR_04",
                        "COLOR_05",
                        "COLOR_06",
                        "COLOR_07",
                        "COLOR_08",
                        "NONE",
                    ],
                    "default": "COLOR_01",
                },
            },
            "required": [],
        },
        category="Blender",
    )

    tool_registry.register_tool(
        "delete_collection",
        delete_collection,
        "Delete a collection. Optionally delete all objects within it.",
        {
            "type": "object",
            "properties": {
                "collection_name": {
                    "type": "string",
                    "description": "Collection to delete",
                },
                "delete_objects": {
                    "type": "boolean",
                    "description": "Also delete objects",
                    "default": False,
                },
            },
            "required": ["collection_name"],
        },
        category="Blender",
    )

    # capture_viewport_for_vision (vision models only)

    tool_registry.register_tool(
        "capture_viewport_for_vision",
        capture_viewport_for_vision,
        "Synchronously capture the current viewport and run a vision model to answer a question about the scene. Returns only the textual description and metadata (no image).",
        {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "What to analyze in the captured viewport image (e.g., 'Are there 9 cubes? Label them.').",
                },
                "max_size": {
                    "type": "integer",
                    "description": "Max width/height in pixels for the captured image",
                    "default": 1024,
                },
                "vision_model": {
                    "type": "string",
                    "description": "Optional override for the vision model name (uses default from preferences if omitted)",
                },
                "timeout_s": {
                    "type": "integer",
                    "description": "Max seconds to wait for vision analysis before timing out",
                    "default": 15,
                    "minimum": 1,
                    "maximum": 120,
                },
            },
            "required": ["question"],
        },
        category="Vision",
    )

    # Collection tools

    # list_collections

    tool_registry.register_tool(
        "list_collections",
        list_collections,
        "List all collections with hierarchy and object counts",
        {"type": "object", "properties": {}, "required": []},
        category="Blender",
    )

    # get_collection_info

    tool_registry.register_tool(
        "get_collection_info",
        get_collection_info,
        "Get info about a specific collection (objects and children)",
        {
            "type": "object",
            "properties": {
                "collection_name": {
                    "type": "string",
                    "description": "Collection name",
                }
            },
            "required": ["collection_name"],
        },
        category="Blender",
    )

    # create_collection (supports batch via 'collections')

    tool_registry.register_tool(
        "create_collection",
        create_collection,
        "Create one or more collections. Batch via 'collections'.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Single collection name"},
                "parent": {
                    "type": "string",
                    "description": "Parent for single collection (omit for scene root)",
                },
                "collections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "parent": {"type": "string"},
                        },
                        "required": ["name"],
                    },
                    "description": "List of {name, parent?}",
                },
            },
            "required": [],
        },
        category="Blender",
    )

    # move_to_collection

    tool_registry.register_tool(
        "move_to_collection",
        move_to_collection,
        "Move objects to a collection; can create the target when missing.",
        {
            "type": "object",
            "properties": {
                "object_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Object names to move",
                },
                "collection_name": {
                    "type": "string",
                    "description": "Target collection name",
                },
                "unlink_from_others": {
                    "type": "boolean",
                    "description": "Unlink from other collections",
                    "default": True,
                },
                "create_if_missing": {
                    "type": "boolean",
                    "description": "Create target if missing",
                    "default": True,
                },
            },
            "required": ["object_names", "collection_name"],
        },
        category="Blender",
    )

    # set_collection_color (now supports batch via 'collection_names')
    tool_registry.register_tool(
        "set_collection_color",
        set_collection_color,
        "Set collection color tag (Blender 4.2+). Accepts single name or list.",
        {
            "type": "object",
            "properties": {
                "collection_name": {
                    "type": "string",
                    "description": "Single collection name",
                },
                "collection_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of collections",
                },
                "color_tag": {
                    "type": "string",
                    "description": "COLOR_01..COLOR_08 or NONE",
                    "enum": [
                        "COLOR_01",
                        "COLOR_02",
                        "COLOR_03",
                        "COLOR_04",
                        "COLOR_05",
                        "COLOR_06",
                        "COLOR_07",
                        "COLOR_08",
                        "NONE",
                    ],
                    "default": "COLOR_01",
                },
            },
            "required": [],
        },
        category="Blender",
    )

    tool_registry.register_tool(
        "delete_collection",
        delete_collection,
        "Delete a collection. Optionally delete all objects within it.",
        {
            "type": "object",
            "properties": {
                "collection_name": {
                    "type": "string",
                    "description": "Collection to delete",
                },
                "delete_objects": {
                    "type": "boolean",
                    "description": "Also delete objects",
                    "default": False,
                },
            },
            "required": ["collection_name"],
        },
        category="Blender",
    )

    tool_registry.register_tool(
        "assistant_help",
        assistant_help,
        "Return JSON Schemas for one or more assistant_sdk tool methods (e.g., 'polyhaven.search').",
        {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "Single SDK method name or MCP tool name (e.g., 'polyhaven.search', 'get_scene_info')",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of SDK method names or MCP tool names",
                },
            },
            "required": [],
        },
        category="Info",
    )

    tool_registry.register_tool(
        "inspect_data",
        inspect_data,
        "Introspect a Blender data block and return its structure (AST-like). Use this to see properties of an object.",
        {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Python path to the data (e.g. \"bpy.data.objects['Cube']\")",
                },
                "depth": {
                    "type": "integer",
                    "description": "Traversal depth (default: 1)",
                    "default": 1,
                },
            },
            "required": ["path"],
        },
        category="Blender",
    )

    tool_registry.register_tool(
        "search_data",
        search_data,
        "Search for data blocks matching specific criteria (AST-grep like).",
        {
            "type": "object",
            "properties": {
                "root_path": {
                    "type": "string",
                    "description": "Path to a collection (e.g. \"bpy.data.objects\")",
                },
                "filter_props": {
                    "type": "object",
                    "description": "Key-value pairs to match (e.g. {\"type\": \"MESH\"})",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return",
                    "default": 10,
                },
            },
            "required": ["root_path"],
        },
        category="Blender",
    )

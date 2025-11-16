"""Blender-specific MCP tools.

This module contains all the Blender manipulation tools that can be called
by the Automation assistant.
"""

import os
import tempfile
import typing

import bpy

from . import mcp_tools


def get_scene_info(
    expand_depth: int = 1,
    expand: list | None = None,
    focus: list | None = None,
    fold_state: dict | None = None,
    max_children: int = 50,
    include_icons: bool = True,
    include_counts: bool = True,
    root_filter: str | None = None,
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

    # Helper: presence icons for an object
    def object_icons(obj) -> list[str]:
        icons = []
        # Type icon (basic set)
        t = (getattr(obj, "type", "") or "").upper()
        type_map = {
            "MESH": "M",
            "ARMATURE": "A",
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
                "ARMATURE": "A",
                "LIGHT": "Lt",
                "CAMERA": "Cam",
                "CURVE": "Cur",
                "EMPTY": "E",
            }
            for k, v in counts.items():
                if k in type_map and v > 0 and type_map[k] not in icons:
                    icons.append(type_map[k])

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

    result = {
        "requested": requested,
        "outliner": {
            "lines": lines,
            "nodes": nodes,
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

    mats = (
        [m.name for m in obj.data.materials]
        if getattr(obj.data, "materials", None)
        else []
    )
    return {
        "name": obj.name,
        "type": obj.type,
        "materials": mats,
        "location": list(obj.location),
        "rotation_euler": list(obj.rotation_euler),
        "scale": list(obj.scale),
    }


def create_object(
    type: str = "CUBE",
    name: str = None,
    location: list = None,
    rotation: list = None,
    scale: list = None,
    text: str = None,
) -> dict:
    """Create an object (mesh primitive, camera, light, text, curve, etc.) with optional parameters."""
    obj_type = type.upper()

    # Create the appropriate object type
    # MESH PRIMITIVES
    if obj_type == "CUBE":
        bpy.ops.mesh.primitive_cube_add()
    elif obj_type == "SPHERE":
        bpy.ops.mesh.primitive_uv_sphere_add()
    elif obj_type == "CYLINDER":
        bpy.ops.mesh.primitive_cylinder_add()
    elif obj_type == "CONE":
        bpy.ops.mesh.primitive_cone_add()
    elif obj_type == "TORUS":
        bpy.ops.mesh.primitive_torus_add()
    elif obj_type == "PLANE":
        bpy.ops.mesh.primitive_plane_add()
    elif obj_type == "MONKEY":
        bpy.ops.mesh.primitive_monkey_add()
    elif obj_type == "ICOSPHERE":
        bpy.ops.mesh.primitive_ico_sphere_add()

    # CAMERAS
    elif obj_type == "CAMERA":
        bpy.ops.object.camera_add()

    # LIGHTS
    elif obj_type == "LIGHT" or obj_type == "POINT_LIGHT":
        bpy.ops.object.light_add(type="POINT")
    elif obj_type == "SUN" or obj_type == "SUN_LIGHT":
        bpy.ops.object.light_add(type="SUN")
    elif obj_type == "SPOT" or obj_type == "SPOT_LIGHT":
        bpy.ops.object.light_add(type="SPOT")
    elif obj_type == "AREA" or obj_type == "AREA_LIGHT":
        bpy.ops.object.light_add(type="AREA")

    # TEXT
    elif obj_type == "TEXT":
        bpy.ops.object.text_add()
        obj = bpy.context.active_object
        if text:
            obj.data.body = text

    # CURVES
    elif obj_type == "BEZIER_CURVE" or obj_type == "CURVE":
        bpy.ops.curve.primitive_bezier_curve_add()
    elif obj_type == "BEZIER_CIRCLE":
        bpy.ops.curve.primitive_bezier_circle_add()
    elif obj_type == "NURBS_CURVE":
        bpy.ops.curve.primitive_nurbs_curve_add()
    elif obj_type == "NURBS_CIRCLE":
        bpy.ops.curve.primitive_nurbs_circle_add()
    elif obj_type == "PATH":
        bpy.ops.curve.primitive_nurbs_path_add()

    # EMPTIES
    elif obj_type == "EMPTY":
        bpy.ops.object.empty_add(type="PLAIN_AXES")
    elif obj_type == "EMPTY_ARROWS":
        bpy.ops.object.empty_add(type="ARROWS")
    elif obj_type == "EMPTY_CUBE":
        bpy.ops.object.empty_add(type="CUBE")
    elif obj_type == "EMPTY_SPHERE":
        bpy.ops.object.empty_add(type="SPHERE")

    else:
        supported = [
            "CUBE",
            "SPHERE",
            "CYLINDER",
            "CONE",
            "TORUS",
            "PLANE",
            "MONKEY",
            "ICOSPHERE",
            "CAMERA",
            "LIGHT",
            "SUN",
            "SPOT",
            "AREA",
            "TEXT",
            "CURVE",
            "BEZIER_CIRCLE",
            "NURBS_CURVE",
            "NURBS_CIRCLE",
            "PATH",
            "EMPTY",
            "EMPTY_ARROWS",
            "EMPTY_CUBE",
            "EMPTY_SPHERE",
        ]
        return {
            "error": f"Unsupported type: {obj_type}. Supported: {', '.join(supported)}"
        }

    obj = bpy.context.active_object

    # Set name if provided
    if name:
        obj.name = name

    # Apply transforms if provided
    if location:
        obj.location = location
    if rotation:
        obj.rotation_euler = rotation
    if scale:
        obj.scale = scale

    return {"created": obj.name, "type": obj.type, "name": obj.name}


def modify_object(
    name: str = None,
    location: list = None,
    rotation: list = None,
    scale: list = None,
    visible: bool = None,
    objects: list = None,
) -> dict:
    """Modify one or more object properties.

    Args:
        name: Single object name (use this OR objects, not both)
        location: Location [x, y, z] for single object
        rotation: Rotation [x, y, z] for single object
        scale: Scale [x, y, z] for single object
        visible: Visibility for single object
        objects: List of dicts with 'name' and optional transform properties for batch modification
                Example: [{"name": "Cube", "location": [0,0,1]}, {"name": "Sphere", "scale": [2,2,2]}]

    Returns:
        Dictionary with modification result(s)
    """
    # Batch mode: modify multiple objects
    if objects:
        results = []
        modified = []
        errors = []

        for obj_spec in objects:
            obj_name = obj_spec.get("name")
            if not obj_name:
                errors.append({"error": "Object spec missing 'name'", "spec": obj_spec})
                continue

            obj = bpy.data.objects.get(obj_name)
            if not obj:
                errors.append({"name": obj_name, "error": "Object not found"})
                continue

            # Apply transforms
            if "location" in obj_spec and obj_spec["location"] is not None:
                obj.location = obj_spec["location"]
            if "rotation" in obj_spec and obj_spec["rotation"] is not None:
                obj.rotation_euler = obj_spec["rotation"]
            if "scale" in obj_spec and obj_spec["scale"] is not None:
                obj.scale = obj_spec["scale"]
            if "visible" in obj_spec and obj_spec["visible"] is not None:
                obj.hide_viewport = not obj_spec["visible"]
                obj.hide_render = not obj_spec["visible"]

            modified.append(obj_name)
            results.append({"name": obj_name, "modified": True})

        return {
            "modified": modified,
            "count": len(modified),
            "results": results,
            "errors": errors if errors else None,
            "message": f"Modified {len(modified)} object(s)"
            + (f", {len(errors)} failed" if errors else ""),
        }

    # Single mode: modify one object
    if not name:
        return {"error": "Must provide 'name' or 'objects' parameter"}

    obj = bpy.data.objects.get(name)
    if not obj:
        return {"error": f"Object not found: {name}"}

    if location is not None:
        obj.location = location
    if rotation is not None:
        obj.rotation_euler = rotation
    if scale is not None:
        obj.scale = scale
    if visible is not None:
        obj.hide_viewport = not visible
        obj.hide_render = not visible

    return {"modified": obj.name, "name": obj.name}


def delete_object(name: str = None, names: list = None) -> dict:
    """Delete one or more objects from the scene.

    Args:
        name: Single object name to delete
        names: List of object names to delete (batch mode)

    Returns:
        Dictionary with deletion result(s)
    """
    # Batch mode
    if names:
        deleted = []
        not_found = []

        for obj_name in names:
            obj = bpy.data.objects.get(obj_name)
            if obj:
                bpy.data.objects.remove(obj, do_unlink=True)
                deleted.append(obj_name)
            else:
                not_found.append(obj_name)

        return {
            "deleted": deleted,
            "count": len(deleted),
            "not_found": not_found if not_found else None,
            "message": f"Deleted {len(deleted)} object(s)"
            + (f", {len(not_found)} not found" if not_found else ""),
        }

    # Single mode
    if not name:
        return {"error": "Must provide 'name' or 'names' parameter"}

    obj = bpy.data.objects.get(name)
    if not obj:
        return {"error": f"Object not found: {name}"}

    bpy.data.objects.remove(obj, do_unlink=True)
    return {"deleted": name}


def set_material(
    object_name: str = None,
    object_names: list = None,
    material_name: str = None,
    color: list = None,
) -> dict:
    """Set or create a material on one or more objects with optional color.

    Args:
        object_name: Single object name (use this OR object_names)
        object_names: List of object names for batch mode
        material_name: Material name (if None, generates per-object names in single mode)
        color: RGBA color [r, g, b, a] or RGB [r, g, b]

    Returns:
        Dictionary with before/after material states showing what changed
    """
    # Batch mode
    if object_names:
        # If no material name provided, generate a shared one for batch
        if not material_name:
            material_name = "BatchMaterial"

        # Get or create material once for batch
        mat = bpy.data.materials.get(material_name)
        mat_existed = mat is not None
        if not mat:
            mat = bpy.data.materials.new(name=material_name)
            mat.use_nodes = True

        # Set color if provided
        if color:
            if len(color) == 3:
                color = list(color) + [1.0]
            if mat.use_nodes:
                nodes = mat.node_tree.nodes
                principled = nodes.get("Principled BSDF")
                if principled:
                    principled.inputs[0].default_value = color

        # Track state changes
        changed = {}
        failed = {}

        # Apply to all objects
        for obj_name in object_names:
            obj = bpy.data.objects.get(obj_name)
            if not obj:
                failed[obj_name] = {"error": "Object not found"}
                continue

            # Capture before state
            before_material = None
            if (
                hasattr(obj, "data")
                and hasattr(obj.data, "materials")
                and obj.data.materials
            ):
                before_material = (
                    obj.data.materials[0].name if obj.data.materials[0] else None
                )

            # Apply material
            if obj.data.materials:
                obj.data.materials[0] = mat
            else:
                obj.data.materials.append(mat)

            # Record state change
            changed[obj_name] = {
                "before": {"material": before_material},
                "after": {"material": material_name, "color": color if color else None},
            }

        # Get final material state
        material_state = {
            material_name: {
                "color": color if color else None,
                "assigned_to": list(changed.keys()),
                "existed_before": mat_existed,
            }
        }

        # Build result
        result = {
            "action": "set_material",
            "requested": {
                "objects": object_names,
                "material": material_name,
                "color": color,
            },
            "changed": changed,
            "failed": failed,
            "material_state": material_state,
            "completed": len(changed),
            "failed_count": len(failed),
        }

        # Add summary
        summary_parts = []
        if changed:
            summary_parts.append(
                f"Applied {material_name} to {len(changed)}/{len(object_names)} object(s)"
            )
        if failed:
            failed_names = ", ".join(failed.keys())
            summary_parts.append(f"Failed: {failed_names}")

        result["summary"] = (
            ". ".join(summary_parts) if summary_parts else "No materials applied"
        )

        return result

    # Single mode
    if not object_name:
        return {"error": "Must provide 'object_name' or 'object_names' parameter"}

    obj = bpy.data.objects.get(object_name)
    if not obj:
        return {"error": f"Object not found: {object_name}"}

    # Capture before state
    before_material = None
    if hasattr(obj, "data") and hasattr(obj.data, "materials") and obj.data.materials:
        before_material = obj.data.materials[0].name if obj.data.materials[0] else None

    # If no material name provided, generate one
    if not material_name:
        material_name = f"Material_{object_name}"

    # Get or create material
    mat = bpy.data.materials.get(material_name)
    mat_existed = mat is not None
    if not mat:
        mat = bpy.data.materials.new(name=material_name)
        mat.use_nodes = True

    # Set color if provided
    if color:
        if len(color) == 3:
            color = list(color) + [1.0]
        if mat.use_nodes:
            nodes = mat.node_tree.nodes
            principled = nodes.get("Principled BSDF")
            if principled:
                principled.inputs[0].default_value = color

    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    # Return state information
    return {
        "action": "set_material",
        "changed": {
            object_name: {
                "before": {"material": before_material},
                "after": {"material": material_name, "color": color if color else None},
            }
        },
        "material_state": {
            material_name: {
                "color": color if color else None,
                "assigned_to": [object_name],
                "existed_before": mat_existed,
            }
        },
        "summary": f"Applied {material_name} to {object_name}",
    }


# Persistent namespace for execute_code (maintains state between calls)
_CODE_NAMESPACE = None


def _get_code_namespace():
    """Get or create the persistent code execution namespace."""
    global _CODE_NAMESPACE
    if _CODE_NAMESPACE is None:
        import mathutils

        try:
            from . import context_utils as _ctx_utils
        except Exception:
            _ctx_utils = None
        _CODE_NAMESPACE = {
            "bpy": bpy,
            "mathutils": mathutils,
            "context_utils": _ctx_utils,
            "__builtins__": __builtins__,
            # Helper to explore available functions
            "help": help,
            "dir": dir,
            "globals": lambda: _CODE_NAMESPACE,
            "assistant_sdk": _get_assistant_sdk(),
        }
    return _CODE_NAMESPACE


class _AssistantSDK:
    def __init__(self, mcp):
        self._mcp = mcp
        self.polyhaven = self._Polyhaven(mcp)
        self.blender = self._Blender(mcp)
        self.sketchfab = self._Sketchfab(mcp)
        self.stock_photos = self._StockPhotos(mcp)
        self.web = self._Web(mcp)
        self.rag = self._RAG(mcp)

    def call(self, name: str, **kwargs):
        return mcp_tools.execute_tool(name, kwargs)

    def help(self):
        return (
            "assistant_sdk quick reference:\n"
            "- blender.get_scene_info(expand_depth=1, expand=[], focus=[], fold_state=None, max_children=50, include_icons=True, include_counts=True)\n"
            "- blender.create_object(type, name=None, location=None, rotation=None, scale=None, text=None)\n"
            "- blender.modify_object(name, location=None, rotation=None, scale=None, visible=None)\n"
            "- blender.delete_object(name=None, names=None)\n"
            "- blender.set_material(object_name=None, object_names=None, material_name=None, color=None)\n"
            "- Scene organization (preferred):\n"
            "  blender.list_collections(); blender.get_collection_info(collection_name); blender.create_collection(name, parent=None);\n"
            "  blender.move_to_collection(object_names, collection_name, unlink_from_others=True, create_if_missing=True)  # unlinks from other collections by default; creates target if missing\n"
            "  blender.set_collection_color(collection_name=None, collection_names=None, color_tag='COLOR_01')  # Blender 4.2+ Outliner color tag (not object materials)\n"
            "  blender.delete_collection(collection_name, delete_objects=False)\n"
            "- blender.get_selection(); blender.get_active(); blender.set_selection(object_names); blender.set_active(object_name); blender.select_by_type(object_type)\n"
            "- polyhaven.search(asset_type='hdri'|'texture'|'model', query='', limit=10);\n"
            "- polyhaven.download(asset=a or asset_id='id', asset_type='hdri'|'texture'|'model', resolution='2k')\n"
            "- sketchfab.login(email, password, save_token=False); sketchfab.search(query, page=1, per_page=24, downloadable_only=True, sort_by='relevance'); sketchfab.download(uid, import_into_scene=True, name_hint='')\n"
            "- stock_photos.search(source, query, per_page=10, orientation=''); stock_photos.download(source, photo_id, apply_as_texture=True)\n"
            "- web.search(query, num_results=5)\n"
            "- rag.query(query, num_results=5, prefer_source=None, page_types=None, excerpt_chars=600); rag.get_stats()\n"
        )

    class _Polyhaven:
        def __init__(self, mcp):
            self._mcp = mcp

        def search(
            self, asset_type: str | None = None, query: str = "", limit: int = 10
        ):
            """Search PolyHaven assets.



            Ergonomics:
                - You can call search('lamp', limit=5) and it will assume asset_type='model'.
                - Or use explicit asset_type='hdri'|'texture'|'model' with query='...'.

            Args:
                asset_type: One of "hdri", "texture", "model". If omitted/None, defaults to "model".
                query: Free text query (e.g., "wood floor"). If the first positional argument

                       is not a recognized asset_type, it is treated as the query and asset_type
                       defaults to "model".
                limit: Max assets to return.



            Returns:

                Dict with keys:

                  - success: bool

                  - assets: list of {id, name, ...}

                  - count: int

                  - formatted: str (human-friendly summary)



            Usage:

                # Shorthand: positional query defaults to models
                assets = assistant_sdk.polyhaven.search('lamp', limit=5)

                # Explicit
                assets = assistant_sdk.polyhaven.search(asset_type="texture", query="wood floor", limit=5)

                for a in assets.get("assets", []):

                    # Use download() as shown below

                    pass

            """

            # Accept positional query as first arg; default asset_type to 'model' when ambiguous
            valid_types = {"hdri", "hdris", "texture", "textures", "model", "models"}
            if isinstance(asset_type, str) and asset_type.lower() not in valid_types:
                # Treat provided first arg as query
                query = str(asset_type)
                asset_type = "model"
            if not asset_type:
                asset_type = "model"

            return mcp_tools.execute_tool(
                "search_polyhaven_assets",
                {"asset_type": asset_type, "query": query, "limit": limit},
            )

        def download(
            self,
            asset: dict | str | None = None,
            asset_type: str | None = None,
            asset_id: str | None = None,
            resolution: str = "2k",
            file_format: str | None = None,
        ):
            """Download a PolyHaven asset ergonomically.

            Accepts either:
              - asset (dict) from assistant_sdk.polyhaven.search(...), and optionally asset_type
              - asset_id (str) and asset_type (str)

            Args:
                asset: Asset dict (ideally from search) or an asset ID string.
                asset_type: "hdri" | "texture" | "model". Required if not inferable from asset.
                asset_id: Asset ID string. Required if not provided via asset.
                resolution: e.g., "2k", "4k" (depends on asset/type).
                file_format: Optional format override (varies by type).

            Returns:
                Dict from the underlying download tool, or a helpful error if parameters are incomplete.

            Examples:
                # Using results from a known search
                res = assistant_sdk.polyhaven.search(asset_type="texture", query="wood floor", limit=5)
                for a in res.get("assets", []):
                    assistant_sdk.polyhaven.download(asset=a, asset_type="texture")

                # Explicit ID + type
                assistant_sdk.polyhaven.download(asset_id="lythwood_room", asset_type="hdri")

                # Common mistake to avoid:
                # assistant_sdk.polyhaven.download(a["id"])  # Missing asset_type -> add asset_type
            """
            # Normalize inputs
            if asset is not None:
                # If a dict was provided, try extracting fields
                if isinstance(asset, dict):
                    if asset_id is None:
                        asset_id = asset.get("id") or asset.get("asset_id")
                    # Some sources may include a "type" or "asset_type" field; prefer explicit arg if given
                    if asset_type is None:
                        asset_type = asset.get("type") or asset.get("asset_type")
                elif isinstance(asset, str):
                    # Treat the string 'asset' as an asset_id if asset_id wasn't provided
                    if asset_id is None:
                        asset_id = asset

            # Validate required params
            if not asset_id or not asset_type:
                return {
                    "error": "polyhaven.download requires both asset_type ('hdri' | 'texture' | 'model') and asset_id",
                    "hint": (
                        "If you called search(asset_type='texture', ...), pass the same asset_type to download. "
                        "Examples:\n"
                        "  assistant_sdk.polyhaven.download(asset=a, asset_type='texture')\n"
                        "  assistant_sdk.polyhaven.download(asset_id='lythwood_room', asset_type='hdri')"
                    ),
                }

            return mcp_tools.execute_tool(
                "download_polyhaven",
                {
                    "asset_type": asset_type,
                    "asset_id": asset_id,
                    "resolution": resolution,
                    "file_format": file_format,
                },
            )

    class _Blender:
        def __init__(self, mcp):
            self._mcp = mcp

        # Collections

        def list_collections(self):
            """List all collections in the current scene.

            Returns:
                Dict containing collections with hierarchy and basic info.
            """
            return mcp_tools.execute_tool("list_collections", {})

        def get_collection_info(self, collection_name: str):
            """Get detailed information about a specific collection.

            Args:
                collection_name: The collection name to inspect.

            Returns:
                Dict with color tag, object list, and child collections.
            """
            return mcp_tools.execute_tool(
                "get_collection_info", {"collection_name": collection_name}
            )

        def create_collection(
            self,
            name: str | None = None,
            parent: str | None = None,
            collections: list | None = None,
        ):
            """Create collections. Single or batch.

            Args:
                name: New collection name (single mode).

                parent: Optional parent collection name for single mode; links to scene root if None.

                collections: Optional batch list of {name, parent?} dicts.

            Returns:

                Dict describing created/failed collections.
            """

            payload = {}
            if name is not None:
                payload["name"] = name
            if parent is not None:
                payload["parent"] = parent
            if collections is not None:
                payload["collections"] = collections
            return mcp_tools.execute_tool("create_collection", payload)

        def ensure_collection(self, name: str, parent: str | None = None):
            """Ensure a collection exists (idempotent). Creates if missing and links to parent or scene root.

            Args:
                name: Collection name to ensure.
                parent: Optional parent collection name.

            Returns:
                Dict describing the resulting collection state.
            """
            return mcp_tools.execute_tool(
                "create_collection", {"name": name, "parent": parent}
            )

        def delete_collection(self, collection_name: str, delete_objects: bool = False):
            """Delete a collection, optionally deleting its objects.

            Args:
                collection_name: Name of the collection to delete.
                delete_objects: If True, also delete objects contained inside.

            Returns:
                Dict describing the deletion result.
            """
            return mcp_tools.execute_tool(
                "delete_collection",
                {"collection_name": collection_name, "delete_objects": delete_objects},
            )

        def move_to_collection(
            self,
            object_names: list,
            collection_name: str,
            unlink_from_others: bool = True,
            create_if_missing: bool = True,
        ):
            """Move one or more objects into a target collection.

            Args:
                object_names: List of object names to move.
                collection_name: Target collection name.
                unlink_from_others: If True, unlink objects from other collections first.
                create_if_missing: If True, create the target collection if it doesn't exist.

            Returns:
                Dict with changed/failed and final collection state.
            """
            return mcp_tools.execute_tool(
                "move_to_collection",
                {
                    "object_names": object_names,
                    "collection_name": collection_name,
                    "unlink_from_others": unlink_from_others,
                    "create_if_missing": create_if_missing,
                },
            )

        def set_collection_color(
            self,
            collection_name: str | None = None,
            collection_names: list | None = None,
            color_tag: str = "COLOR_01",
        ):
            """Set the color tag on one or more collections (Blender 4.2+).

            Args:
                collection_name: Single collection name (optional).
                collection_names: Multiple collection names (optional; takes precedence if provided).
                color_tag: COLOR_01..COLOR_08 or NONE.

            Returns:
                Dict with changed/unchanged/failed and a summary.
            """
            return mcp_tools.execute_tool(
                "set_collection_color",
                {
                    "collection_name": collection_name,
                    "collection_names": collection_names,
                    "color_tag": color_tag,
                },
            )

        def get_scene_info(
            self,
            expand_depth: int = 1,
            expand: list | None = None,
            focus: list | None = None,
            fold_state: dict | None = None,
            max_children: int = 50,
            include_icons: bool = True,
            include_counts: bool = True,
            root_filter: str | None = None,
        ):
            """Outliner-style, persistent scene view (hierarchical, compact, and stateful)."""
            return mcp_tools.execute_tool(
                "get_scene_info",
                {
                    "expand_depth": expand_depth,
                    "expand": expand,
                    "focus": focus,
                    "fold_state": fold_state,
                    "max_children": max_children,
                    "include_icons": include_icons,
                    "include_counts": include_counts,
                    "root_filter": root_filter,
                },
            )

        # Objects
        def get_object_info(self, name: str):
            return mcp_tools.execute_tool("get_object_info", {"name": name})

        def create_object(
            self,
            type: str,
            name: str | None = None,
            location: list | None = None,
            rotation: list | None = None,
            scale: list | None = None,
            text: str | None = None,
        ):
            """Create an object (mesh primitive, text, camera/light, etc.).

            Args:
                type: Object type (e.g., 'CUBE', 'SPHERE', 'TEXT', 'CAMERA', 'LIGHT').
                name: Optional object name.
                location: Optional [x, y, z].
                rotation: Optional [x, y, z] Euler (radians).
                scale: Optional [x, y, z].
                text: TEXT object content (when type='TEXT').

            Returns:
                Dict describing the created object (via underlying tool).
            """
            return mcp_tools.execute_tool(
                "create_object",
                {
                    "type": type,
                    "name": name,
                    "location": location,
                    "rotation": rotation,
                    "scale": scale,
                    "text": text,
                },
            )

        def modify_object(
            self,
            name: str | None = None,
            location: list | None = None,
            rotation: list | None = None,
            scale: list | None = None,
            visible: bool | None = None,
            objects: list | None = None,
        ):
            """Modify properties of objects.

            Args:
                name: Object name to modify (single mode).

                location: Optional [x, y, z] (single mode).

                rotation: Optional [x, y, z] Euler (radians) (single mode).

                scale: Optional [x, y, z] (single mode).

                visible: Optional visibility toggle (True/False) (single mode).

                objects: Optional batch list of {name, location?, rotation?, scale?, visible?}.

            Returns:

                Dict describing the modification result.

            """

            payload = {}
            if name is not None:
                payload["name"] = name
            if location is not None:
                payload["location"] = location
            if rotation is not None:
                payload["rotation"] = rotation
            if scale is not None:
                payload["scale"] = scale
            if visible is not None:
                payload["visible"] = visible
            if objects is not None:
                payload["objects"] = objects
            return mcp_tools.execute_tool("modify_object", payload)

        def delete_object(self, name: str | None = None, names: list | None = None):
            """Delete object(s) by name.

            Args:
                name: Single object name.
                names: List of object names (batch delete).

            Returns:
                Dict describing deleted items and any failures.
            """
            payload = {}

            if name is not None:
                payload["name"] = name

            if names is not None:
                payload["names"] = names

            return mcp_tools.execute_tool("delete_object", payload)

        def set_material(
            self,
            object_name: str | None = None,
            object_names: list | None = None,
            material_name: str | None = None,
            color: list | None = None,
        ):
            """Assign or create a material on one or more objects.

            Args:
                object_name: Single object name (use this OR object_names).
                object_names: List of object names (batch).
                material_name: Target material name (created if missing).
                color: Optional [r, g, b, a] or [r, g, b] base color for Principled BSDF.

            Returns:
                Dict with changed/failed and material state.
            """
            payload = {}
            if object_name is not None:
                payload["object_name"] = object_name

            if object_names is not None:
                payload["object_names"] = object_names

            if material_name is not None:
                payload["material_name"] = material_name

            if color is not None:
                payload["color"] = color

            return mcp_tools.execute_tool("set_material", payload)

        # Selection helpers
        def get_selection(self):
            return mcp_tools.execute_tool("get_selection", {})

        def get_active(self):
            """Get the active object name (if any)."""
            return mcp_tools.execute_tool("get_active", {})

        def set_selection(self, object_names: list | str):
            """Set the current selection.

            Args:
                object_names: A string (single object) or a list of object names.

            Returns:
                Dict describing selection state.
            """
            if isinstance(object_names, str):
                object_names = [object_names]
            return mcp_tools.execute_tool(
                "set_selection", {"object_names": object_names}
            )

        def set_active(self, object_name: str):
            """Set the active object by name."""
            return mcp_tools.execute_tool("set_active", {"object_name": object_name})

        def select_by_type(self, object_type: str):
            """Select all objects of a given Blender type (e.g., 'MESH')."""
            return mcp_tools.execute_tool(
                "select_by_type", {"object_type": object_type}
            )

    class _Sketchfab:
        def __init__(self, mcp):
            self._mcp = mcp

        def login(self, email: str, password: str, save_token: bool = False):
            """Login to Sketchfab (stores access token in memory).

            Args:
                email: Account email.
                password: Account password.
                save_token: Persist token (if supported); otherwise kept in memory only.

            Returns:
                Dict indicating login success or error.
            """
            return mcp_tools.execute_tool(
                "sketchfab_login",
                {"email": email, "password": password, "save_token": save_token},
            )

        def search(
            self,
            query: str,
            page: int = 1,
            per_page: int = 24,
            downloadable_only: bool = True,
            sort_by: str = "relevance",
        ):
            """Search downloadable Sketchfab models.

            Args:
                query: Free text search query.
                page: Page number (1-based).
                per_page: Results per page.
                downloadable_only: Restrict to downloadable assets (recommended True).
                sort_by: One of 'relevance' | 'likes' | 'views' | 'recent'.

            Returns:
                Dict with results and pagination info.
            """
            return mcp_tools.execute_tool(
                "sketchfab_search",
                {
                    "query": query,
                    "page": page,
                    "per_page": per_page,
                    "downloadable_only": downloadable_only,
                    "sort_by": sort_by,
                },
            )

        def download(
            self, uid: str, import_into_scene: bool = True, name_hint: str = ""
        ):
            """Download a Sketchfab model by UID and optionally import it.

            Args:
                uid: Sketchfab model UID.
                import_into_scene: If True, import the model after download.
                name_hint: Optional name hint for imported content.

            Returns:
                Dict describing download/import result or an error.
            """
            return mcp_tools.execute_tool(
                "sketchfab_download_model",
                {
                    "uid": uid,
                    "import_into_scene": import_into_scene,
                    "name_hint": name_hint,
                },
            )

    class _StockPhotos:
        def __init__(self, mcp):
            self._mcp = mcp

        def search(
            self, source: str, query: str, per_page: int = 10, orientation: str = ""
        ):
            """Search stock photos by source.

            Args:
                source: 'unsplash' | 'pexels'.
                query: Free text query.
                per_page: Max results to return.
                orientation: Optional orientation filter ('landscape' | 'portrait' | 'squarish').

            Returns:
                Dict with results list and metadata.
            """
            return mcp_tools.execute_tool(
                "search_stock_photos",
                {
                    "source": source,
                    "query": query,
                    "per_page": per_page,
                    "orientation": orientation,
                },
            )

        def download(self, source: str, photo_id: str, apply_as_texture: bool = True):
            """Download a stock photo and optionally apply as a texture to the active object.

            Args:
                source: 'unsplash' | 'pexels'.
                photo_id: Identifier returned by search().
                apply_as_texture: If True, applies as an image texture to the active object (if possible).

            Returns:
                Dict describing the download (and application) result.
            """
            return mcp_tools.execute_tool(
                "download_stock_photo",
                {
                    "source": source,
                    "photo_id": photo_id,
                    "apply_as_texture": apply_as_texture,
                },
            )

    class _Web:
        def __init__(self, mcp):
            self._mcp = mcp

        def search(self, query: str, num_results: int = 5):
            """Search the web (DuckDuckGo HTML) for top results.

            Args:
                query: Free text query.
                num_results: Number of results to return (default 5, max 10).

            Returns:
                Dict with titles, URLs, and descriptions.
            """
            return mcp_tools.execute_tool(
                "web_search", {"query": query, "num_results": num_results}
            )

    class _RAG:
        def __init__(self, mcp):
            self._mcp = mcp

        def query(
            self,
            query: str,
            num_results: int = 5,
            prefer_source: str | None = None,
            page_types: list | None = None,
            excerpt_chars: int = 600,
        ):
            """Query the local Blender docs RAG store.

            Args:
                query: Search query or question.
                num_results: Number of results to return.
                prefer_source: Optional source bias (e.g., 'python_api').
                page_types: Optional list of page types to filter.
                excerpt_chars: Excerpt length per result.

            Returns:
                Dict with top results (title, url, excerpt, source).
            """
            payload = {
                "query": query,
                "num_results": num_results,
                "excerpt_chars": excerpt_chars,
            }
            if prefer_source is not None:
                payload["prefer_source"] = prefer_source

            if page_types is not None:
                payload["page_types"] = page_types

            return mcp_tools.execute_tool("rag_query", payload)

        def get_stats(self):
            """Get RAG subsystem statistics (enabled, document count, db path)."""
            return mcp_tools.execute_tool("rag_get_stats", {})


def _get_assistant_sdk():
    try:
        return _AssistantSDK(mcp_tools)
    except Exception:
        return None


def execute_code(code: str) -> dict:
    """Execute Python code in Blender's context with persistent state.

    This is the most powerful tool - it gives you direct access to Blender's Python API (bpy).
    State persists between calls, so you can define variables/functions and reuse them.

    Available in namespace:
    - bpy: Blender Python API
    - mathutils: Vector, Matrix, Euler, etc.
    - assistant_sdk: Pre-initialized SDK for tools (blender, polyhaven, sketchfab, stock_photos, web, rag)
    - Any variables/functions you define persist between calls

    IMPORTANT: assistant_sdk is already available in the namespace. Do NOT import it!

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
            try:
                if not model_name:
                    addon = bpy.context.preferences.addons.get(__package__)
                    prefs = addon.preferences if addon else None
                    # Prefer dedicated vision model, fallback to active chat model
                    model_name = getattr(prefs, "vision_model", "") or getattr(
                        prefs, "model_file", ""
                    )
            except Exception:
                pass

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
                    from . import ollama_adapter as llama_manager

                    messages = [
                        {
                            "role": "user",
                            "content": prompt,
                            "images": [image_b64],
                        }
                    ]
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


def ensure_collections(names: list, parent: str = None) -> dict:
    """Ensure a list of collections exist. Create missing ones.

    Args:
        names: List of collection names to ensure exist
        parent: Optional parent collection for newly created ones (omit for scene root)

    Returns:
        Dict with created, existing, and errors
    """
    try:
        created = []
        existing = []
        errors = []

        parent_col = None
        if parent:
            parent_col = resolve_collection_by_name(parent)
            if not parent_col:
                return {"error": f"Parent collection '{parent}' not found"}

        for name in names or []:
            col = resolve_collection_by_name(name)
            if col:
                existing.append(name)
                continue
            # Create and link
            new_col = bpy.data.collections.new(name)
            if parent_col:
                parent_col.children.link(new_col)
            else:
                scene_root = _get_scene_root()
                (scene_root or bpy.context.scene.collection).children.link(new_col)
            created.append(name)

        return {
            "success": True,
            "created": created,
            "existing": existing,
            "message": f"Ensured {len(names or [])} collection(s): {len(created)} created, {len(existing)} existing",
        }
    except Exception as e:
        return {"error": f"Failed to ensure collections: {str(e)}"}


def safe_move_objects(
    object_names: list,
    collection_name: str,
    create_if_missing: bool = True,
    unlink_from_others: bool = True,
) -> dict:
    """Move objects to a target collection; create the collection if needed.

    Args:
        object_names: Objects to move
        collection_name: Target collection
        create_if_missing: Create target if it's missing (default True)
        unlink_from_others: Unlink from other collections
    """
    try:
        target_col = resolve_collection_by_name(collection_name)
        if not target_col:
            if create_if_missing:
                # Create at scene root
                create_result = create_collection(name=collection_name)
                if isinstance(create_result, dict) and create_result.get("error"):
                    return create_result
                target_col = resolve_collection_by_name(collection_name)
            else:
                return {"error": f"Collection '{collection_name}' not found"}

        return move_to_collection(
            object_names=object_names,
            collection_name=collection_name,
            unlink_from_others=unlink_from_others,
        )
    except Exception as e:
        return {"error": f"Failed to move objects: {str(e)}"}


def set_collections_color_batch(
    collection_names: list, color_tag: str = "COLOR_01"
) -> dict:
    """Set a color tag on multiple collections.

    Args:
        collection_names: List of collections to color-tag
        color_tag: COLOR_01..COLOR_08 or NONE
    """
    try:
        updated = []
        not_found = []
        skipped = []
        for name in collection_names or []:
            col = resolve_collection_by_name(name)
            if not col:
                not_found.append(name)
                continue
            if hasattr(col, "color_tag"):
                col.color_tag = color_tag
                updated.append(name)
            else:
                skipped.append(
                    {"name": name, "reason": "Collection has no color_tag property"}
                )
        result = {"success": True, "updated": updated, "color_tag": color_tag}
        if not_found:
            result["not_found"] = not_found
            result["suggestion"] = (
                "Use ensure_collections(names=[...]) to create missing collections"
            )
        if skipped:
            result["skipped"] = skipped
        return result
    except Exception as e:
        return {"error": f"Failed to set collection colors: {str(e)}"}


def assistant_help(tool: str = "", tools: list | None = None) -> dict:
    """Return JSON Schemas for one or more assistant_sdk tool aliases.

    Usage examples (tool aliases):

      - "polyhaven.search"            -> search_polyhaven_assets
      - "polyhaven.download"          -> download_polyhaven

      - "blender.get_scene_info"      -> get_scene_info
      - "blender.get_object_info"     -> get_object_info
      - "blender.list_collections"    -> list_collections
      - "blender.get_collection_info" -> get_collection_info
      - "blender.create_collection"   -> create_collection
      - "blender.move_to_collection"  -> move_to_collection
      - "blender.set_collection_color"-> set_collection_color
      - "blender.delete_collection"   -> delete_collection
      - "blender.get_selection"       -> get_selection
      - "blender.get_active"          -> get_active
      - "blender.set_selection"       -> set_selection
      - "blender.set_active"          -> set_active
      - "blender.select_by_type"      -> select_by_type
      - "blender.create_object"       -> create_object
      - "blender.modify_object"       -> modify_object
      - "blender.delete_object"       -> delete_object
      - "blender.set_material"        -> set_material

      - "vision.capture"              -> capture_viewport_for_vision
      - "vision.capture_viewport"     -> capture_viewport_for_vision

      - "web.search"                  -> web_search
      - "web.fetch"                   -> fetch_webpage
      - "web.extract_image_urls"      -> extract_image_urls
      - "web.wikimedia_image"         -> search_wikimedia_image
      - "web.download_image"          -> download_image_as_texture

      - "sketchfab.login"             -> sketchfab_login
      - "sketchfab.search"            -> sketchfab_search
      - "sketchfab.download"          -> sketchfab_download_model

      - "stock_photos.search"         -> search_stock_photos    (requires configured API keys)
      - "stock_photos.download"       -> download_stock_photo   (requires configured API keys)

      - "rag.query"                   -> rag_query
      - "rag.get_stats"               -> rag_get_stats

    Namespace expansion (pass the namespace to list its common tools):

      - "assistant_sdk.web"           -> expands to web.search/fetch/extract_image_urls/wikimedia_image/download_image
      - "assistant_sdk.sketchfab"     -> expands to sketchfab.login/search/download
      - "assistant_sdk.stock_photos"  -> expands to stock_photos.search/download
      - "assistant_sdk.rag"           -> expands to rag.query/get_stats
      - "assistant_sdk.blender"       -> expands to common blender.* tools
      - "assistant_sdk"               -> expands to all supported namespaces above

    Also accepts underlying tool names directly (e.g., "get_scene_info").
    """
    try:
        # Build list of requested aliases
        aliases: list[str] = []
        if tools and isinstance(tools, (list, tuple)):
            aliases.extend([str(t).strip() for t in tools if str(t).strip()])
        if tool and str(tool).strip():
            aliases.append(str(tool).strip())
        if not aliases:
            return {"error": "Missing 'tool' or 'tools' parameter"}

        # Map SDK-like aliases to MCP tool names
        alias_map = {
            # PolyHaven
            "polyhaven.search": "search_polyhaven_assets",
            "polyhaven.download": "download_polyhaven",
            # Blender
            "blender.get_scene_info": "get_scene_info",
            "blender.get_object_info": "get_object_info",
            "blender.list_collections": "list_collections",
            "blender.get_collection_info": "get_collection_info",
            "blender.create_collection": "create_collection",
            "blender.move_to_collection": "move_to_collection",
            "blender.set_collection_color": "set_collection_color",
            "blender.delete_collection": "delete_collection",
            "blender.get_selection": "get_selection",
            "blender.get_active": "get_active",
            "blender.set_selection": "set_selection",
            "blender.set_active": "set_active",
            "blender.select_by_type": "select_by_type",
            "blender.create_object": "create_object",
            "blender.modify_object": "modify_object",
            "blender.delete_object": "delete_object",
            "blender.set_material": "set_material",
            # Vision
            "vision.capture": "capture_viewport_for_vision",
            "vision.capture_viewport": "capture_viewport_for_vision",
            # Web
            "web.search": "web_search",
            "web.fetch": "fetch_webpage",
            "web.extract_image_urls": "extract_image_urls",
            "web.wikimedia_image": "search_wikimedia_image",
            "web.download_image": "download_image_as_texture",
            # Sketchfab
            "sketchfab.login": "sketchfab_login",
            "sketchfab.search": "sketchfab_search",
            "sketchfab.download": "sketchfab_download_model",
            # Stock Photos (conditional registration based on API keys)
            "stock_photos.search": "search_stock_photos",
            "stock_photos.download": "download_stock_photo",
            # RAG
            "rag.query": "rag_query",
            "rag.get_stats": "rag_get_stats",
            "rag.stats": "rag_get_stats",
        }

        # Define namespace expansions (each expands to a list of alias_map keys)
        namespace_map = {
            "web": [
                "web.search",
                "web.fetch",
                "web.extract_image_urls",
                "web.wikimedia_image",
                "web.download_image",
            ],
            "sketchfab": [
                "sketchfab.login",
                "sketchfab.search",
                "sketchfab.download",
            ],
            "stock_photos": [
                "stock_photos.search",
                "stock_photos.download",
            ],
            "vision": [
                "vision.capture",
                "vision.capture_viewport",
            ],
            "rag": [
                "rag.query",
                "rag.get_stats",
            ],
            "polyhaven": [
                "polyhaven.search",
                "polyhaven.download",
            ],
            "blender": [
                "blender.get_scene_info",
                "blender.get_object_info",
                "blender.list_collections",
                "blender.get_collection_info",
                "blender.create_collection",
                "blender.move_to_collection",
                "blender.set_collection_color",
                "blender.delete_collection",
                "blender.get_selection",
                "blender.get_active",
                "blender.set_selection",
                "blender.set_active",
                "blender.create_object",
                "blender.modify_object",
                "blender.delete_object",
                "blender.set_material",
                "blender.select_by_type",
            ],
        }

        # Pull current MCP tool registry
        tool_defs = mcp_tools.get_tools_list() or []
        names = {
            t.get("name"): t for t in tool_defs if isinstance(t, dict) and t.get("name")
        }
        lowered = {k.lower(): k for k in names.keys()}

        results = []
        for alias in aliases:
            norm = alias.lower().strip()

            # Strip assistant_sdk. prefix when present
            if norm.startswith("assistant_sdk."):
                norm = norm[len("assistant_sdk.") :]

            # Namespace expansions (e.g., "web", "sketchfab", "stock_photos", "rag", "blender")
            expanded_aliases: list[str] = []
            if norm in ("assistant_sdk",):
                for ns_aliases in namespace_map.values():
                    expanded_aliases.extend(ns_aliases)
            elif norm in namespace_map:
                expanded_aliases.extend(namespace_map[norm])

            if expanded_aliases:
                for ns_alias in expanded_aliases:
                    resolved_name = alias_map.get(ns_alias)
                    if not resolved_name or resolved_name not in names:
                        continue
                    t = names[resolved_name]
                    results.append(
                        {
                            "tool": resolved_name,
                            "description": t.get("description", ""),
                            "inputSchema": t.get("inputSchema", {}),
                        }
                    )
                continue

            # Single alias resolution path
            resolved = alias_map.get(norm)
            if not resolved:
                # Try exact or case-insensitive MCP name
                if alias in names:
                    resolved = alias
                elif norm in lowered:
                    resolved = lowered[norm]
            if not resolved or resolved not in names:
                continue

            t = names[resolved]
            results.append(
                {
                    "tool": resolved,
                    "description": t.get("description", ""),
                    "inputSchema": t.get("inputSchema", {}),
                }
            )

        return {"results": results}

    except Exception as e:
        return {"error": f"assistant_help failed: {str(e)}"}


def register_tools():
    """Register all Blender tools with the MCP registry."""

    # get_scene_info

    mcp_tools.register_tool(
        "get_scene_info",
        get_scene_info,
        "Outliner-style, persistent scene view (hierarchical, compact, stateful).",
        {
            "type": "object",
            "properties": {
                "expand_depth": {
                    "type": "integer",
                    "description": "Default expansion depth from root",
                    "default": 1,
                    "minimum": 0,
                    "maximum": 6,
                },
                "expand": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional node paths to expand (e.g., 'SolarSystem/Planets')",
                },
                "focus": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Nodes to auto-expand and include ancestors",
                },
                "fold_state": {
                    "type": "object",
                    "description": "Opaque expansion state (pass from prior call)",
                },
                "max_children": {
                    "type": "integer",
                    "description": "Max children per node",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 500,
                },
                "include_icons": {
                    "type": "boolean",
                    "description": "Include presence icons on nodes",
                    "default": True,
                },
                "include_counts": {
                    "type": "boolean",
                    "description": "Include type counts on collections",
                    "default": True,
                },
                "root_filter": {
                    "type": "string",
                    "description": "Start rendering from a specific collection name",
                },
            },
            "required": [],
        },
        category="Blender",
    )

    # get_object_info

    mcp_tools.register_tool(
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

    mcp_tools.register_tool(
        "create_object",
        create_object,
        "Create objects: meshes, cameras, lights, text, curves, empties",
        {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Object type",
                    "enum": [
                        # Mesh primitives
                        "CUBE",
                        "SPHERE",
                        "CYLINDER",
                        "CONE",
                        "TORUS",
                        "PLANE",
                        "MONKEY",
                        "ICOSPHERE",
                        # Cameras
                        "CAMERA",
                        # Lights
                        "LIGHT",
                        "POINT_LIGHT",
                        "SUN",
                        "SUN_LIGHT",
                        "SPOT",
                        "SPOT_LIGHT",
                        "AREA",
                        "AREA_LIGHT",
                        # Text
                        "TEXT",
                        # Curves
                        "CURVE",
                        "BEZIER_CURVE",
                        "BEZIER_CIRCLE",
                        "NURBS_CURVE",
                        "NURBS_CIRCLE",
                        "PATH",
                        # Empties
                        "EMPTY",
                        "EMPTY_ARROWS",
                        "EMPTY_CUBE",
                        "EMPTY_SPHERE",
                    ],
                },
                "name": {"type": "string", "description": "Optional name"},
                "location": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Location [x, y, z]",
                },
                "rotation": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Euler rotation [x, y, z]",
                },
                "scale": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Scale [x, y, z]",
                },
                "text": {"type": "string", "description": "Text content for TEXT"},
            },
            "required": [],
        },
        category="Blender",
    )

    # modify_object
    mcp_tools.register_tool(
        "modify_object",
        modify_object,
        "Modify an existing object's properties",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Object name"},
                "location": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "New location [x, y, z]",
                },
                "rotation": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "New Euler rotation [x, y, z]",
                },
                "scale": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "New scale [x, y, z]",
                },
                "visible": {"type": "boolean", "description": "Set visibility"},
                "objects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "location": {"type": "array", "items": {"type": "number"}},
                            "rotation": {"type": "array", "items": {"type": "number"}},
                            "scale": {"type": "array", "items": {"type": "number"}},
                            "visible": {"type": "boolean"},
                        },
                        "required": ["name"],
                    },
                    "description": "Batch list of objects to modify ({name, location?, rotation?, scale?, visible?})",
                },
            },
            "required": [],
        },
        category="Blender",
    )

    # delete_object (supports batch via 'names')

    mcp_tools.register_tool(
        "delete_object",
        delete_object,
        "Delete one or more objects. Prefer batch via 'names'.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Single object name"},
                "names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of object names",
                },
            },
            "required": [],
        },
        category="Blender",
    )

    # set_material

    mcp_tools.register_tool(
        "set_material",
        set_material,
        "Set or create a material on an object with optional color",
        {
            "type": "object",
            "properties": {
                "object_name": {
                    "type": "string",
                    "description": "Object name (single)",
                },
                "object_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of object names (batch)",
                },
                "material_name": {
                    "type": "string",
                    "description": "Material name (auto if omitted)",
                },
                "color": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "RGBA [r, g, b, a] 0.0-1.0",
                },
            },
            "required": [],
        },
        category="Blender",
    )

    # execute_code

    mcp_tools.register_tool(
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

    mcp_tools.register_tool(
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

    mcp_tools.register_tool(
        "list_collections",
        list_collections,
        "List all collections with hierarchy and object counts",
        {"type": "object", "properties": {}, "required": []},
        category="Blender",
    )

    # get_collection_info

    mcp_tools.register_tool(
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

    mcp_tools.register_tool(
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

    mcp_tools.register_tool(
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
    mcp_tools.register_tool(
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

    mcp_tools.register_tool(
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

    mcp_tools.register_tool(
        "assistant_help",
        assistant_help,
        "Return JSON Schemas for one or more assistant_sdk tool aliases (e.g., 'polyhaven.search').",
        {
            "type": "object",
            "properties": {
                "tool": {
                    "type": "string",
                    "description": "Single SDK alias or MCP tool name (e.g., 'polyhaven.search', 'get_scene_info')",
                },
                "tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of SDK aliases or MCP tool names",
                },
            },
            "required": [],
        },
        category="Info",
    )

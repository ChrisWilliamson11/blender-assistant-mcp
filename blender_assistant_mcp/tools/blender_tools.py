"""Blender-specific MCP tools.

This module contains all the Blender manipulation tools that can be called
by the Automation assistant.
"""

import os
import tempfile
import typing

import bpy

from . import tool_registry
from .memory import MemoryManager

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
    }

    # Data block info
    if obj.data:
        info["data"] = {"name": obj.data.name}

        # Mesh specific
        if obj.type == "MESH":
            mesh = obj.data
            info["data"]["vertices"] = len(mesh.vertices)
            info["data"]["polygons"] = len(mesh.polygons)
            info["data"]["shape_keys"] = (
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
            "assistant_sdk": _get_assistant_sdk(),
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

            # 2. Create submodules for each tool category (blender, polyhaven, etc.)
            for category in [
                "blender",
                "polyhaven",
                "sketchfab",
                "stock_photos",
                "web",
                "rag",
            ]:
                if hasattr(sdk_obj, category):
                    cat_obj = getattr(sdk_obj, category)
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


class _AssistantSDK:
    def __init__(self, mcp):
        self._mcp = mcp
        self.polyhaven = self._Polyhaven(mcp)
        self.blender = self._Blender(mcp)
        self.sketchfab = self._Sketchfab(mcp)
        self.web = self._Web(mcp)
        self.rag = self._RAG(mcp)
        self.memory = self._Memory(mcp)
        # Only expose stock_photos if API keys are configured
        try:
            from .preferences import get_preferences

            prefs = get_preferences()
            if prefs and (prefs.unsplash_api_key or prefs.pexels_api_key):
                self.stock_photos = self._StockPhotos(mcp)
        except Exception:
            pass  # No preferences or no API keys

    def call(self, name: str, **kwargs):
        return tool_registry.execute_tool(name, kwargs)

    def help(self) -> str:
        """Return a concise SDK quick reference for first-turn planning.

        This is purpose-only (no full signatures). The model should call
        assistant_help('<namespace>') to fetch exact usage before coding.
        """
        return (
            "assistant_sdk quick reference; use assistant_help('assistant_sdk.<namespace>') for signatures\n"
            "- blender.* — scene/objects/collections/selection\n"
            "- polyhaven.search/download — PolyHaven assets (HDRIs, textures, models)\n"
            "- stock_photos.search/download — Pexels/Unsplash images (requires API keys)\n"
            "- sketchfab.login/search/download — Sketchfab models\n"
            "- web.search/fetch_page/extract_images/download_image — web tools (for images: search→extract→download)\n"
            "- rag.query/get_stats — Blender docs RAG (API/Manual)\n"
            "- memory.remember_fact/remember_preference/remember_learning/search — long-term knowledge\n"
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

            return tool_registry.execute_tool(
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

            return tool_registry.execute_tool(
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
            return tool_registry.execute_tool("list_collections", {})

        def get_collection_info(self, collection_name: str):
            """Get detailed information about a specific collection.

            Args:
                collection_name: The collection name to inspect.

            Returns:
                Dict with color tag, object list, and child collections.
            """
            return tool_registry.execute_tool(
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
            return tool_registry.execute_tool("create_collection", payload)

        def ensure_collection(self, name: str, parent: str | None = None):
            """Ensure a collection exists (idempotent). Creates if missing and links to parent or scene root.

            Args:
                name: Collection name to ensure.
                parent: Optional parent collection name.

            Returns:
                Dict describing the resulting collection state.
            """
            return tool_registry.execute_tool(
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
            return tool_registry.execute_tool(
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
            return tool_registry.execute_tool(
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
            return tool_registry.execute_tool(
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
            return tool_registry.execute_tool(
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
            return tool_registry.execute_tool("get_object_info", {"name": name})

        def create_object(
            self,
            type: str,
            name: str | None = None,
            location: list | None = None,
            rotation: list | None = None,
            scale: list | None = None,
            text: str | None = None,
        ):
            """Create an object (mesh primitive, text, camera/light, etc.)."""
            try:
                # Convert type to uppercase
                obj_type = type.upper()

                # Handle mesh primitives
                if obj_type == "CUBE":
                    bpy.ops.mesh.primitive_cube_add(location=location or (0, 0, 0))
                elif obj_type == "SPHERE":
                    bpy.ops.mesh.primitive_uv_sphere_add(location=location or (0, 0, 0))
                elif obj_type == "PLANE":
                    bpy.ops.mesh.primitive_plane_add(location=location or (0, 0, 0))
                elif obj_type == "CYLINDER":
                    bpy.ops.mesh.primitive_cylinder_add(location=location or (0, 0, 0))
                elif obj_type == "CONE":
                    bpy.ops.mesh.primitive_cone_add(location=location or (0, 0, 0))
                elif obj_type == "TORUS":
                    bpy.ops.mesh.primitive_torus_add(location=location or (0, 0, 0))
                elif obj_type == "MONKEY":
                    bpy.ops.mesh.primitive_monkey_add(location=location or (0, 0, 0))

                # Handle other types
                elif obj_type == "TEXT":
                    bpy.ops.object.text_add(location=location or (0, 0, 0))
                    if text:
                        bpy.context.active_object.data.body = text
                elif obj_type == "CAMERA":
                    bpy.ops.object.camera_add(location=location or (0, 0, 0))
                elif obj_type == "LIGHT":
                    bpy.ops.object.light_add(
                        type="POINT", location=location or (0, 0, 0)
                    )
                else:
                    return {"error": f"Unknown object type: {type}"}

                obj = bpy.context.active_object
                if name:
                    obj.name = name

                if rotation:
                    obj.rotation_euler = rotation
                if scale:
                    obj.scale = scale

                return {"success": True, "name": obj.name, "type": obj.type}
            except Exception as e:
                return {"error": f"Failed to create object: {str(e)}"}

        def modify_object(
            self,
            name: str | None = None,
            location: list | None = None,
            rotation: list | None = None,
            scale: list | None = None,
            visible: bool | None = None,
            objects: list | None = None,
        ):
            """Modify properties of objects."""
            try:
                targets = []
                if objects:
                    targets.extend(objects)
                if name:
                    # Single object mode
                    item = {"name": name}
                    if location is not None:
                        item["location"] = location
                    if rotation is not None:
                        item["rotation"] = rotation
                    if scale is not None:
                        item["scale"] = scale
                    if visible is not None:
                        item["visible"] = visible
                    targets.append(item)

                modified = []
                for item in targets:
                    obj_name = item.get("name")
                    obj = bpy.data.objects.get(obj_name)
                    if not obj:
                        continue

                    if "location" in item:
                        obj.location = item["location"]
                    if "rotation" in item:
                        obj.rotation_euler = item["rotation"]
                    if "scale" in item:
                        obj.scale = item["scale"]
                    if "visible" in item:
                        obj.hide_viewport = not item["visible"]
                        obj.hide_render = not item["visible"]

                    modified.append(obj.name)

                return {"success": True, "modified": modified}
            except Exception as e:
                return {"error": f"Failed to modify objects: {str(e)}"}

        def delete_object(self, name: str | None = None, names: list | None = None):
            """Delete object(s) by name."""
            try:
                targets = []
                if names:
                    targets.extend(names)
                if name:
                    targets.append(name)

                deleted = []
                for obj_name in targets:
                    obj = bpy.data.objects.get(obj_name)
                    if obj:
                        bpy.data.objects.remove(obj, do_unlink=True)
                        deleted.append(obj_name)

                return {"success": True, "deleted": deleted}
            except Exception as e:
                return {"error": f"Failed to delete objects: {str(e)}"}

        def set_material(
            self,
            object_name: str | None = None,
            object_names: list | None = None,
            material_name: str | None = None,
            color: list | None = None,
        ):
            """Assign or create a material on one or more objects."""
            try:
                targets = []
                if object_names:
                    targets.extend(object_names)
                if object_name:
                    targets.append(object_name)

                if not targets:
                    return {"error": "No objects specified"}

                # Get or create material
                mat = None
                if material_name:
                    mat = bpy.data.materials.get(material_name)
                    if not mat:
                        mat = bpy.data.materials.new(name=material_name)
                        mat.use_nodes = True

                # If color provided, update material (or create temp one if no name)
                if color:
                    if not mat:
                        mat = bpy.data.materials.new(name="Material")
                        mat.use_nodes = True

                    # Set base color on Principled BSDF
                    if mat.node_tree:
                        bsdf = None
                        for n in mat.node_tree.nodes:
                            if n.type == "BSDF_PRINCIPLED":
                                bsdf = n
                                break
                        if not bsdf:
                            bsdf = mat.node_tree.nodes.new("ShaderNodeBsdfPrincipled")

                        # Handle 3-tuple or 4-tuple color
                        c = list(color)
                        if len(c) == 3:
                            c.append(1.0)
                        bsdf.inputs["Base Color"].default_value = c

                if not mat:
                    return {"error": "No material name or color specified"}

                # Assign to objects
                assigned = []
                for obj_name in targets:
                    obj = bpy.data.objects.get(obj_name)
                    if obj and obj.type == "MESH":
                        if not obj.data.materials:
                            obj.data.materials.append(mat)
                        else:
                            obj.data.materials[0] = mat
                        assigned.append(obj_name)

                return {"success": True, "material": mat.name, "assigned_to": assigned}
            except Exception as e:
                return {"error": f"Failed to set material: {str(e)}"}

        def get_active(self):
            """Get the active object name (if any)."""
            return tool_registry.execute_tool("get_active", {})

        def set_selection(self, object_names: list | str):
            """Set the current selection.

            Args:
                object_names: A string (single object) or a list of object names.

            Returns:
                Dict describing selection state.
            """
            if isinstance(object_names, str):
                object_names = [object_names]
            return tool_registry.execute_tool(
                "set_selection", {"object_names": object_names}
            )

        def set_active(self, object_name: str):
            """Set the active object by name."""
            return tool_registry.execute_tool("set_active", {"object_name": object_name})

        def select_by_type(self, object_type: str):
            """Select all objects of a given Blender type (e.g., 'MESH')."""
            return tool_registry.execute_tool(
                "select_by_type", {"object_type": object_type}
            )

        def capture_viewport(
            self,
            question: str,
            max_size: int = 1024,
            vision_model: str | None = None,
            timeout_s: int = 15,
        ):
            """Capture the viewport and ask a vision model a question."""
            return tool_registry.execute_tool(
                "capture_viewport_for_vision",
                {
                    "question": question,
                    "max_size": max_size,
                    "vision_model": vision_model,
                    "timeout_s": timeout_s,
                },
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
            return tool_registry.execute_tool(
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
            return tool_registry.execute_tool(
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
            return tool_registry.execute_tool(
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
            self,
            source: str = "pexels",
            query: str = "",
            per_page: int = 10,
            orientation: str = "",
            limit: int | None = None,
            **kwargs,
        ):
            """Search stock photos by source (resilient interface).



            Accepts extra/unknown kwargs (ignored) and supports 'limit' as an alias for 'per_page'.
            Returns a dict-like object that is also iterable/sliceable over the photo list,
            so code like `results[:5]` and `for r in results:` works.

            Args:
                source: 'unsplash' | 'pexels'
                query: Free text query
                per_page: Max results to return (default 10)
                orientation: Optional orientation filter ('landscape' | 'portrait' | 'squarish')
                limit: Alias for per_page (common LLM pattern)
                **kwargs: Ignored (for robustness)

            Returns:

                Dict-like with metadata, and iterable/sliceable over photos.
            """

            # Normalize alias
            if limit is not None and (per_page is None or per_page == 10):
                try:
                    per_page = int(limit)
                except Exception:
                    pass

            # Map 'squarish' (Unsplash) to 'square' (Pexels) transparently when possible
            if orientation and orientation.lower() == "squarish":
                orientation = "square"

            data = tool_registry.execute_tool(
                "search_stock_photos",
                {
                    "source": source,
                    "query": query,
                    "per_page": per_page,
                    "orientation": orientation,
                },
            )

            # Wrap results so they behave like a list when iterated/sliced
            class _SearchResult(dict):
                def __init__(self, payload: dict):
                    super().__init__(payload if isinstance(payload, dict) else {})
                    self._items = []
                    if isinstance(payload, dict):
                        # Prefer 'photos', fallback to 'results'
                        self._items = (
                            payload.get("photos") or payload.get("results") or []
                        )
                        # Ensure list type
                        if not isinstance(self._items, list):
                            self._items = []
                    else:
                        # If some provider returns a list directly, keep it
                        self._items = payload if isinstance(payload, list) else []

                def __iter__(self):
                    return iter(self._items)

                def __len__(self):
                    return len(self._items)

                def __getitem__(self, key):
                    # Allow list-style access/slicing: results[0], results[:5]
                    if isinstance(key, (int, slice)):
                        return self._items[key]
                    # Fallback to dict access for metadata keys
                    return super().__getitem__(key)

            return _SearchResult(data)

        def download(
            self,
            source: str | None = None,
            photo_id: str | int | None = None,
            apply_as_texture: bool = False,
            **kwargs,
        ):
            """Download a stock photo and optionally apply it as a texture (resilient interface).



            This method tolerates common LLM call patterns:

            - download(photo_id=...)                      # explicit

            - download(source='pexels', photo_id=...)     # explicit

            - download('123456')                          # positional photo_id only; source inferred

            - download('abc123')                          # positional unsplash-like ID; source inferred

            - download(photo_id, destination='...')       # ignores unknown kwargs like 'destination'



            Args:

                source: 'unsplash' | 'pexels' (inferred if omitted)

                photo_id: Identifier string/int from search results

                apply_as_texture: Apply to active object if possible

                **kwargs: Ignored (e.g., destination)



            Returns:

                Dict with download status.
            """
            # Support positional-only (download(photo_id)) calls
            if (
                source is not None
                and photo_id is None
                and isinstance(source, (str, int))
            ):
                photo_id, source = source, None

            # Infer source if missing based on photo_id shape

            if source is None:
                if isinstance(photo_id, int) or (
                    isinstance(photo_id, str) and photo_id.isdigit()
                ):
                    source = "pexels"

                else:
                    source = "unsplash"

            # Ignore unknown kwargs such as 'destination'
            return tool_registry.execute_tool(
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
            return tool_registry.execute_tool(
                "web_search", {"query": query, "num_results": num_results}
            )

        def fetch_page(self, url: str, max_length: int = 10000):
            """Fetch and extract text content from a webpage."""
            return tool_registry.execute_tool(
                "fetch_webpage", {"url": url, "max_length": max_length}
            )

        def extract_images(self, url: str, min_width: int = 400, max_images: int = 10):
            """Extract likely content image URLs from a webpage."""
            return tool_registry.execute_tool(
                "extract_image_urls",
                {"url": url, "min_width": min_width, "max_images": max_images},
            )

        def download_image(
            self, image_url: str, apply_to_active: bool = True, pack_image: bool = True
        ):
            """Download an image and optionally apply it as a texture."""
            return tool_registry.execute_tool(
                "download_image_as_texture",
                {
                    "image_url": image_url,
                    "apply_to_active": apply_to_active,
                    "pack_image": pack_image,
                },
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

            return tool_registry.execute_tool("rag_query", payload)

        def get_stats(self):
            """Get RAG subsystem statistics (enabled, document count, db path)."""
            return tool_registry.execute_tool("rag_get_stats", {})

    class _Memory:
        def __init__(self, mcp):
            self._mcp = mcp

        def remember_fact(self, key: str, value: str):
            """Store a factual memory (e.g., 'user likes blue')."""
            return get_memory_manager().remember_fact(key, value)

        def remember_preference(self, key: str, value: str):
            """Store a user preference (e.g., 'default_material: metal')."""
            return get_memory_manager().remember_preference(key, value)

        def remember_learning(self, key: str, value: str):
            """Store a learning/insight (e.g., 'UV unwrap before texturing')."""
            return get_memory_manager().remember_learning(key, value)

        def search(self, query: str, limit: int = 10):
            """Search memory for relevant entries."""
            return get_memory_manager().search_memory(query, limit)


def _get_assistant_sdk():
    """Get the assistant SDK instance for execute_code namespace."""
    try:
        from .. import assistant_sdk
        return assistant_sdk.get_assistant_sdk()
    except Exception as e:
        print(f"[Assistant] CRITICAL: Failed to initialize assistant_sdk: {e}")
        import traceback
        traceback.print_exc()

        # Return a minimal stub so code doesn't crash entirely
        class _StubSDK:
            pass

        return _StubSDK()


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

        # Define the knowledge base of SDK tools
        # Format: alias -> {sdkUsage, notes, returns?}
        sdk_docs = {
            # PolyHaven
            "polyhaven.search": {
                "sdkUsage": "assistant_sdk.polyhaven.search(asset_type='hdri'|'texture'|'model', query='', limit=10)",
                "notes": "Returns dict with 'assets' list. Iterate results['assets'] to find items.",
                "returns": "{'success': bool?, 'assets': [ { 'id': str, 'name': str, ... } ], 'count': int}",
            },
            "polyhaven.download": {
                "sdkUsage": "assistant_sdk.polyhaven.download(asset=asset_dict, asset_type='type', resolution='2k')",
                "notes": "Downloads and imports asset. Pass the full asset dict from search, or asset_id.",
                "returns": "{'success': bool, 'path': str?, 'message': str?}",
            },
            # Blender - Scene
            "blender.get_scene_info": {
                "sdkUsage": "assistant_sdk.blender.get_scene_info(expand_depth=2)",
                "notes": "Returns JSON tree of scene objects/collections.",
                "returns": "{'outliner': {'lines': [...], 'nodes': [...]}, 'fold_state': {...}, 'summary': str}",
            },
            "blender.get_object_info": {
                "sdkUsage": "assistant_sdk.blender.get_object_info(object_names=['Name'])",
                "notes": "Get detailed info for specific objects.",
            },
            "blender.list_collections": {
                "sdkUsage": "assistant_sdk.blender.list_collections()",
                "notes": "List all collection names.",
            },
            "blender.get_collection_info": {
                "sdkUsage": "assistant_sdk.blender.get_collection_info(collection_names=['Name'])",
                "notes": "Get info about specific collections.",
            },
            # Blender - Manipulation
            "blender.create_collection": {
                "sdkUsage": "assistant_sdk.blender.create_collection(name='Name', parent='Parent')",
                "notes": "Create a new collection.",
                "returns": "{'success': bool, 'name': str, 'parent': str|null}",
            },
            "blender.move_to_collection": {
                "sdkUsage": "assistant_sdk.blender.move_to_collection(object_names=['Obj'], collection_name='Col')",
                "notes": "Move objects to a collection.",
                "returns": "{'success': bool, 'moved': ['Obj', ...]?}",
            },
            "blender.set_collection_color": {
                "sdkUsage": "assistant_sdk.blender.set_collection_color(collection_name='Col', color_tag='COLOR_01')",
                "notes": "Set collection color tag.",
                "returns": "{'success': bool}",
            },
            "blender.delete_collection": {
                "sdkUsage": "assistant_sdk.blender.delete_collection(collection_names=['Col'], delete_objects=False)",
                "notes": "Delete collections.",
                "returns": "{'success': bool, 'deleted': ['Col', ...]}",
            },
            # Blender - Selection/Active
            "blender.get_selection": {
                "sdkUsage": "assistant_sdk.blender.get_selection()",
                "notes": "Get list of selected object names.",
                "returns": "{'selected_objects': ['Name', ...], 'count': int, 'message': str}",
            },
            "blender.set_selection": {
                "sdkUsage": "assistant_sdk.blender.set_selection(object_names=['Obj'], replace=True)",
                "notes": "Set selected objects.",
                "returns": "{'success': bool, 'selected': ['Obj', ...], 'not_found': ['Name', ...], 'message': str}",
            },
            "blender.get_active": {
                "sdkUsage": "assistant_sdk.blender.get_active()",
                "notes": "Get active object name.",
                "returns": "{'active_object': 'Name'|None, 'type': str|None, 'message': str}",
            },
            "blender.set_active": {
                "sdkUsage": "assistant_sdk.blender.set_active(object_name='Obj')",
                "notes": "Set active object.",
                "returns": "{'success': bool, 'active_object': 'Obj', 'message': str}",
            },
            "blender.select_by_type": {
                "sdkUsage": "assistant_sdk.blender.select_by_type(type='MESH')",
                "notes": "Select objects by type.",
                "returns": "{'success': bool, 'selected': ['Name', ...], 'count': int, 'message': str}",
            },
            # Vision
            "vision.capture": {
                "sdkUsage": "assistant_sdk.vision.capture_viewport_for_vision()",
                "notes": "SDK (execute_code): Capture 3D viewport as base64 image.",
                "returns": "{'description': str, 'model': str, 'width': int, 'height': int}",
            },
            # Blender - Object/Material manipulation
            "blender.create_object": {
                "sdkUsage": "assistant_sdk.blender.create_object(type='PLANE'|'CUBE'|'SPHERE'|..., name='Name', location=[x,y,z], rotation=[x,y,z], scale=[x,y,z], text='...')",
                "notes": "Create an object (mesh primitive, text, camera/light).",
                "returns": "{'success': bool, 'name': str, 'type': str}",
            },
            "blender.modify_object": {
                "sdkUsage": "assistant_sdk.blender.modify_object(name='Obj', location=[x,y,z], rotation=[x,y,z], scale=[x,y,z], visible=True|False, objects=[{...}])",
                "notes": "Modify properties of one or more objects (supports batch via 'objects').",
                "returns": "{'success': bool, 'modified': ['Name', ...]}",
            },
            "blender.delete_object": {
                "sdkUsage": "assistant_sdk.blender.delete_object(name='Obj'|None, names=['ObjA','ObjB'])",
                "notes": "Delete object(s) by name (supports batch via 'names').",
                "returns": "{'success': bool, 'deleted': ['Name', ...]}",
            },
            "blender.set_material": {
                "sdkUsage": "assistant_sdk.blender.set_material(object_name='Obj'|None, object_names=['ObjA','ObjB'], material_name='Mat'|None, color=[r,g,b(,a)])",
                "notes": "Assign or create a material; sets Principled Base Color if 'color' specified.",
            },
            "web.images_workflow": {
                "workflow": [
                    "# 3-STEP WORKFLOW for downloading web images:",
                    "1. results = assistant_sdk.web.search(query='kittens')  # Get pages about kittens",
                    "2. images = assistant_sdk.web.extract_images(results['results'][0]['url'])  # Extract image URLs",
                    "3. assistant_sdk.web.download_image(images['images'][0])  # Download first image",
                ],
                "notes": "Multi-step process: SEARCH for pages → EXTRACT image URLs → DOWNLOAD the image.",
            },
            "web.extract_image_urls": {
                "sdkUsage": "assistant_sdk.web.extract_image_urls(url='url')",
                "notes": "SDK (execute_code): Find image URLs on a page.",
                "returns": "{'success': bool, 'url': str, 'count': int, 'images': [str, ...]}",
            },
            "web.download_image": {
                "sdkUsage": "assistant_sdk.web.download_image(image_url='url')",
                "notes": "SDK (execute_code): Download image and load as Blender image/texture.",
                "returns": "{'success': bool, 'image_name': str, 'packed': bool, 'size': 'WxH', 'applied_to': str|None, 'material': str?, 'message': str}",
            },
            # Sketchfab
            "sketchfab.search": {
                "sdkUsage": "assistant_sdk.sketchfab.search(query='q', type='models')",
                "notes": "SDK (execute_code): Search Sketchfab.",
                "returns": "{'success': bool?, 'results': [...], 'count': int?}",
            },
            "sketchfab.download": {
                "sdkUsage": "assistant_sdk.sketchfab.download(uid='uid')",
                "notes": "SDK (execute_code): Download model from Sketchfab.",
                "returns": "{'success': bool, 'path': str?, 'message': str?}",
            },
            # RAG
            "rag.query": {
                "sdkUsage": "assistant_sdk.rag.query(text='question')",
                "notes": "SDK (execute_code): Query documentation.",
                "returns": "{'results': [{'title': str, 'url': str, 'excerpt': str, 'source': str}], 'count': int}",
            },
            # Memory
            "memory.remember_fact": {
                "sdkUsage": "assistant_sdk.memory.remember_fact(fact='fact', category='general')",
                "notes": "SDK (execute_code): Store a general fact.",
                "returns": "{'success': bool}",
            },
            "memory.remember_preference": {
                "sdkUsage": "assistant_sdk.memory.remember_preference(key='key', value='value')",
                "notes": "SDK (execute_code): Store a user preference.",
                "returns": "{'success': bool}",
            },
            "memory.remember_learning": {
                "sdkUsage": "assistant_sdk.memory.remember_learning(topic='topic', insight='insight')",
                "notes": "SDK (execute_code): Record a technical learning/pitfall.",
                "returns": "{'success': bool}",
            },
            "memory.search": {
                "sdkUsage": "assistant_sdk.memory.search(query='query')",
                "notes": "SDK (execute_code): Semantic search of memory.",
                "returns": "{'results': [{'type': str, 'key': str, 'value': str}], 'count': int}",
            },
        }

        results = []

        for q in queries:
            # Normalize query: remove 'assistant_sdk.' prefix, handle 'tool/subtool'
            clean_q = q.replace("assistant_sdk.", "").lower()

            # Split by slash if present (e.g. "polyhaven.search/download")
            sub_queries = clean_q.split("/")

            for sub_q in sub_queries:
                sub_q = sub_q.strip()
                if not sub_q:
                    continue

                # 1. Exact match
                if sub_q in sdk_docs:
                    res = sdk_docs[sub_q]
                    # res["alias"] = sub_q <-- REMOVED
                    results.append(res)
                    continue

                # 2. Namespace match (e.g., "polyhaven")
                namespace_matches = [
                    k for k in sdk_docs.keys() if k.startswith(sub_q + ".")
                ]
                if namespace_matches:
                    for k in namespace_matches:
                        res = sdk_docs[k].copy()
                        # res["alias"] = k  <-- REMOVED
                        results.append(res)
                    continue

                # 3. Fuzzy/Contains match
                fuzzy_matches = [k for k in sdk_docs.keys() if sub_q in k]
                if fuzzy_matches:
                    for k in fuzzy_matches:
                        res = sdk_docs[k].copy()
                        # res["alias"] = k <-- REMOVED
                        results.append(res)
                    continue

        # Remove duplicates
        unique_results = []
        seen = set()
        for r in results:
            # Use sdkUsage as unique key since alias is gone
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

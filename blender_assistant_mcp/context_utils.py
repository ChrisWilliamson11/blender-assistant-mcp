"""Utilities for robust Blender context and collection operations.

Designed to be imported into the execute_code namespace so LLMs can rely on
safer primitives that avoid common pitfalls (VIEW_3D context, idempotent links,
collection lookup, etc.).
"""
from __future__ import annotations

import bpy
from contextlib import contextmanager
from typing import Optional


@contextmanager
def ensure_view3d_context():
    """Yield a temp override that guarantees a VIEW_3D area when possible.

    Usage:
        with ensure_view3d_context() as override:
            if override is None:
                # Fallback: operator may fail without a VIEW_3D
                bpy.ops.mesh.primitive_cube_add()
            else:
                with bpy.context.temp_override(**override):
                    bpy.ops.mesh.primitive_cube_add()
    """
    try:
        screen = bpy.context.screen
        if not screen or not getattr(screen, "areas", None):
            yield None
            return
        view3d_area = None
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                view3d_area = area
                break
        if view3d_area is None:
            yield None
            return

        # Find a matching WINDOW region
        region = None
        for r in view3d_area.regions:
            if r.type == 'WINDOW':
                region = r
                break

        override = bpy.context.copy()
        override['area'] = view3d_area
        if region:
            override['region'] = region
        with bpy.context.temp_override(**override):
            yield override
    except Exception:
        # On any error, do not crash the caller
        yield None


def get_or_create_collection(name: str, link_to_scene: bool = True) -> bpy.types.Collection:
    """Return a collection by name, creating it if needed.

    - Case-sensitive exact match first; if not found, try case-insensitive
      match before creating a new collection.
    - Optionally link newly created collection to the scene root.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("Collection name must be a non-empty string")

    col = bpy.data.collections.get(name)
    if not col:
        # Case-insensitive fallback
        low = name.lower()
        for c in bpy.data.collections:
            if c.name.lower() == low:
                col = c
                break
    if not col:
        col = bpy.data.collections.new(name)
        if link_to_scene:
            try:
                scene_col = bpy.context.scene.collection
                if not any(child is col for child in scene_col.children):
                    scene_col.children.link(col)
            except Exception:
                pass
    return col


def link_object_safe(obj: bpy.types.Object, col: bpy.types.Collection) -> None:
    """Link an object to a collection; ignore if already linked."""
    try:
        if obj is None or col is None:
            return
        if col not in obj.users_collection:
            col.objects.link(obj)
    except RuntimeError:
        # Already linked
        pass
    except Exception:
        pass


def unlink_object_safe(obj: bpy.types.Object, col: bpy.types.Collection) -> None:
    """Unlink an object from a collection if present."""
    try:
        if obj is None or col is None:
            return
        if obj.name in col.objects:
            col.objects.unlink(obj)
    except Exception:
        pass


def link_child_collection_safe(parent: bpy.types.Collection, child: bpy.types.Collection) -> None:
    """Link a child collection under a parent; ignore if already linked."""
    try:
        if parent is None or child is None:
            return
        # Some Blender versions accept Collection instance; else use name
        try:
            parent.children.link(child)
        except RuntimeError:
            # Already linked
            pass
        except TypeError:
            # Fallback by name
            if child.name not in parent.children:
                parent.children.link(child)
    except Exception:
        pass


def move_object_to_collection(obj: bpy.types.Object, target: bpy.types.Collection) -> None:
    """Move object into target collection and out of scene root if redundant."""
    if obj is None or target is None:
        return
    link_object_safe(obj, target)
    # If object is in multiple collections including scene root, remove from scene root
    try:
        scene_col = bpy.context.scene.collection
        if scene_col in obj.users_collection and len(obj.users_collection) > 1:
            try:
                scene_col.objects.unlink(obj)
            except Exception:
                pass
    except Exception:
        pass


def ensure_unique_name(base: str, existing: Optional[set] = None) -> str:
    """Produce a unique name with numeric suffix if needed."""
    if not base:
        base = "Item"
    if existing is None:
        existing = {bpy.data.objects.get(n).name for n in bpy.data.objects.keys()} if bpy.data.objects else set()
    if base not in existing:
        return base
    i = 1
    while True:
        candidate = f"{base}.{i:03d}"
        if candidate not in existing:
            return candidate
        i += 1


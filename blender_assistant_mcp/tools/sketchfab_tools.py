"""Sketchfab MCP tools: login with email/password and download/import models.

Notes & safety:
- This uses Sketchfab's OAuth password grant (same endpoint their official plugin uses).
- We DO NOT persist passwords. By default we keep the access token only in-memory.
- Optional save_token lets you persist the access token in Blender's session via
  add-on preferences in future; currently kept in module-level memory only.
- Requires built-in glTF importer (bpy.ops.import_scene.gltf) for import.
"""
from __future__ import annotations

from typing import Dict, Any, List
import os
import io
import json
import zipfile
import tempfile

import bpy
import httpx

from . import tool_registry

SKETCHFAB_URL = "https://sketchfab.com"
SKETCHFAB_API = "https://api.sketchfab.com"
SKETCHFAB_SEARCH = f"{SKETCHFAB_API}/v3/search"
SKETCHFAB_MODEL = f"{SKETCHFAB_API}/v3/models"
SKETCHFAB_OAUTH = f"{SKETCHFAB_URL}/oauth2/token/"
# Public client id from the official addon (Apache 2.0):
SKETCHFAB_CLIENT_ID = "hGC7unF4BHyEB0s7Orz5E1mBd3LluEG0ILBiZvF9"

# Module-level session state
_ACCESS_TOKEN: str | None = None
_HEADERS: Dict[str, str] = {}


def _build_headers() -> Dict[str, str]:
    global _HEADERS
    if _ACCESS_TOKEN:
        _HEADERS = {"Authorization": f"Bearer {_ACCESS_TOKEN}"}
    else:
        _HEADERS = {}
    return _HEADERS


def sketchfab_login(email: str = "", password: str = "", save_token: bool = False) -> dict:
    """Login to Sketchfab using email/password and obtain an OAuth access token.

    Returns a limited success payload; token is stored in memory for subsequent calls.
    """
    try:
        if not email or not password:
            return {"error": "email and password are required"}
        data = {
            "grant_type": "password",
            "client_id": SKETCHFAB_CLIENT_ID,
            "username": email,
            "password": password,
        }
        with httpx.Client(timeout=20.0) as client:
            r = client.post(SKETCHFAB_OAUTH, data=data)
            if r.status_code != 200:
                # Attempt to surface server-provided error
                try:
                    info = r.json()
                except Exception:
                    info = {"status": r.status_code, "text": r.text[:200]}
                return {"error": "Failed to authenticate", "details": info}
            payload = r.json()
            token = payload.get("access_token")
            if not token:
                return {"error": "No access_token in response", "response": payload}

        # Store in-memory
        global _ACCESS_TOKEN
        _ACCESS_TOKEN = token
        _build_headers()

        # Optionally return a hint for persistence (not implemented here to avoid storing secrets)
        return {"success": True, "token_preview": token[:6] + "…", "saved": save_token}
    except Exception as e:
        return {"error": f"Login failed: {str(e)}"}


def sketchfab_search(query: str, page: int = 1, per_page: int = 24, downloadable_only: bool = True,
                     sort_by: str = "relevance") -> dict:
    """Search downloadable Sketchfab models via the v3 API.

    sort_by one of: relevance, likes, views, recent (best-effort mapping).
    """
    try:
        if not _ACCESS_TOKEN:
            return {"error": "Not logged in. Call sketchfab_login(email, password) first."}

        # Map sort_by to API params
        sort_map = {
            "relevance": None,
            "likes": "-likeCount",
            "views": "-viewCount",
            "recent": "-publishedAt",
        }
        sort_param = sort_map.get(sort_by.lower()) if isinstance(sort_by, str) else None

        params = {
            "type": "models",
            "q": query or "",
            "downloadable": "true" if downloadable_only else "false",
            "per_page": max(1, min(per_page, 48)),
            "page": max(1, page),
        }
        if sort_param:
            params["sort_by"] = sort_param

        with httpx.Client(timeout=20.0) as client:
            r = client.get(SKETCHFAB_SEARCH, params=params, headers=_build_headers())
            r.raise_for_status()
            data = r.json() or {}

        results: List[dict] = []
        for item in data.get("results", []) or []:
            # Only include minimal fields to keep context small
            results.append({
                "uid": item.get("uid"),
                "name": item.get("name"),
                "user": {
                    "username": item.get("user", {}).get("username"),
                    "displayName": item.get("user", {}).get("displayName"),
                },
                "thumbnails": item.get("thumbnails", {}).get("images", []),
                "archives": item.get("archives", {}),
                "license": (item.get("license", {}) or {}).get("label"),
            })

        return {
            "success": True,
            "query": query,
            "count": len(results),
            "page": page,
            "per_page": per_page,
            "next": data.get("next"),
            "previous": data.get("previous"),
            "results": results,
        }
    except httpx.HTTPStatusError as e:
        return {"error": f"Search HTTP {e.response.status_code}", "details": e.response.text[:200]}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


def _resolve_download_url(uid: str) -> dict:
    """Get the glTF archive URL for a given model UID."""
    try:
        url = f"{SKETCHFAB_MODEL}/{uid}/download"
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            r = client.get(url, headers=_build_headers())
            if r.status_code != 200:
                try:
                    info = r.json()
                except Exception:
                    info = {"status": r.status_code, "text": r.text[:200]}
                return {"error": "Model is not downloadable or not accessible", "details": info}
            data = r.json() or {}
        gltf = data.get("gltf") or {}
        if not gltf or "url" not in gltf:
            return {"error": "No glTF archive available for this model"}
        return {"success": True, "gltf": gltf, "expires": data.get("expires")}
    except Exception as e:
        return {"error": f"Failed to resolve download: {str(e)}"}


def _import_gltf_zip(zip_bytes: bytes, name_hint: str | None = None) -> dict:
    """Import a glTF/GLB from a Zip archive buffer into the current scene."""
    try:
        # Write to temp zip
        temp_dir = tempfile.mkdtemp(prefix="skfb_")
        zip_path = os.path.join(temp_dir, "model.zip")
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        # Extract
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(temp_dir)

        # Find a .glb or .gltf file
        gltf_path = None
        for root, _, files in os.walk(temp_dir):
            for fn in files:
                if fn.lower().endswith((".glb", ".gltf")):
                    gltf_path = os.path.join(root, fn)
                    break
            if gltf_path:
                break
        if not gltf_path:
            return {"error": "Archive does not contain a .glb/.gltf"}

        # Import
        before = set(o.name for o in bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=gltf_path)
        after = set(o.name for o in bpy.data.objects)
        created = sorted(list(after - before))

        # Optionally rename a single root object
        if name_hint and len(created) == 1:
            try:
                obj = bpy.data.objects.get(created[0])
                if obj:
                    obj.name = name_hint
                    created = [obj.name]
            except Exception:
                pass

        return {"success": True, "imported_objects": created, "file": os.path.basename(gltf_path)}
    except Exception as e:
        return {"error": f"Import failed: {str(e)}"}


def sketchfab_download_model(uid: str, import_into_scene: bool = True, name_hint: str = "") -> dict:
    """Download a downloadable Sketchfab model by UID and optionally import into scene."""
    try:
        if not _ACCESS_TOKEN:
            return {"error": "Not logged in. Call sketchfab_login first."}
        if not uid:
            return {"error": "uid is required"}

        info = _resolve_download_url(uid)
        if "error" in info:
            return info
        gltf = info.get("gltf", {})
        file_url = gltf.get("url")
        if not file_url:
            return {"error": "No glTF download URL available"}

        with httpx.Client(timeout=120.0, follow_redirects=True) as client:
            r = client.get(file_url)
            r.raise_for_status()
            zip_bytes = r.content

        if not import_into_scene:
            # Return metadata only
            return {
                "success": True,
                "uid": uid,
                "archive_size": len(zip_bytes),
                "message": "Downloaded archive (not imported)",
            }

        result = _import_gltf_zip(zip_bytes, name_hint=name_hint or None)
        if "error" in result:
            return result
        result.update({"uid": uid})
        return result
    except httpx.HTTPStatusError as e:
        return {"error": f"Download HTTP {e.response.status_code}", "details": e.response.text[:200]}
    except Exception as e:
        return {"error": f"Download failed: {str(e)}"}


def register():
    """Register Sketchfab tools in the MCP registry."""
    tool_registry.register_tool(
        "sketchfab_login",
        sketchfab_login,
        (
            "Login to Sketchfab using email/password.\n"
            "USAGE: Required before searching or downloading. Stores token in memory.\n"
            "RETURNS: {'success': True, ...} or error."
        ),
        {
            "type": "object",
            "properties": {
                "email": {"type": "string", "description": "Sketchfab account email"},
                "password": {"type": "string", "description": "Password", "format": "password"},
                "save_token": {"type": "boolean", "description": "Persist token (disabled; kept in memory only)", "default": False},
            },
            "required": ["email", "password"]
        },
        category="Sketchfab",
    )

    tool_registry.register_tool(
        "sketchfab_search",
        sketchfab_search,
        (
            "Search for free downloadable Sketchfab models.\n"
            "USAGE: Requires login. Query 'car', 'tree', etc.\n"
            "RETURNS: {'results': [{'uid': '...', 'name': '...', 'user': {...}}, ...]}"
        ),
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "page": {"type": "integer", "default": 1, "minimum": 1},
                "per_page": {"type": "integer", "default": 24, "minimum": 1, "maximum": 48},
                "downloadable_only": {"type": "boolean", "default": True},
                "sort_by": {"type": "string", "enum": ["relevance", "likes", "views", "recent"], "default": "relevance"},
            },
            "required": ["query"]
        },
        category="Sketchfab",
    )

    tool_registry.register_tool(
        "sketchfab_download_model",
        sketchfab_download_model,
        (
            "Download and Import a Sketchfab Model by UID.\n"
            "USAGE: Find UID via `sketchfab_search`. Imports glTF automatically.\n"
            "RETURNS: {'success': True, 'imported_objects': ['Object', ...]}"
        ),
        {
            "type": "object",
            "properties": {
                "uid": {"type": "string", "description": "Sketchfab model UID (32-char)"},
                "import_into_scene": {"type": "boolean", "default": True},
                "name_hint": {"type": "string", "description": "Rename root object if single root is imported", "default": ""}
            },
            "required": ["uid"]
        },
        category="Sketchfab",
    )



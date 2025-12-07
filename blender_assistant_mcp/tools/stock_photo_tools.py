"""Stock photo download tools for Unsplash and Pexels.

This module provides tools for searching and downloading stock photos
from Unsplash and Pexels to use as textures and references in Blender.
"""

import os
import threading
import time

import bpy
import httpx

from . import tool_registry

# Global state for tracking background downloads
_download_threads = []
_download_results = {}


def _cleanup_finished_threads():
    """Remove finished threads from the tracking list."""
    global _download_threads
    _download_threads = [t for t in _download_threads if t.is_alive()]


def _run_download_in_background(download_func, *args, **kwargs):
    """Run a download function in a background thread to avoid blocking the UI.

    Args:
        download_func: The download function to run
        *args, **kwargs: Arguments to pass to the download function

    Returns:
        Dictionary with status and job tracking info
    """
    global _download_threads, _download_results

    # Clean up old threads
    _cleanup_finished_threads()

    # Generate a unique job ID
    job_id = f"download_{int(time.time() * 1000)}_{len(_download_threads)}"

    # Initialize result
    _download_results[job_id] = {"status": "downloading", "progress": 0, "result": None}

    def thread_target():
        try:
            result = download_func(*args, **kwargs)
            _download_results[job_id]["status"] = "completed"
            _download_results[job_id]["result"] = result
            _download_results[job_id]["progress"] = 100
        except Exception as e:
            _download_results[job_id]["status"] = "failed"
            _download_results[job_id]["result"] = {"error": str(e)}
            _download_results[job_id]["progress"] = 0

    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    _download_threads.append(thread)

    return {
        "success": True,
        "job_id": job_id,
        "status": "downloading",
        "message": f"Download started in background. Use check_download_status('{job_id}') to check progress.",
        "note": "The download is running in the background and won't block the UI. The result will be available when complete.",
    }


def check_download_status(job_id: str) -> dict:
    """Check the status of a background download.

    Args:
        job_id: The job ID returned from a download operation

    Returns:
        Dictionary with current status and result if completed
    """
    global _download_results

    if job_id not in _download_results:
        return {
            "error": f"Unknown job ID: {job_id}",
            "hint": "The job may have expired or never existed.",
        }

    job_info = _download_results[job_id]

    response = {
        "job_id": job_id,
        "status": job_info["status"],
        "progress": job_info["progress"],
    }

    if job_info["status"] == "completed":
        response["result"] = job_info["result"]
        response["message"] = "Download completed successfully!"
    elif job_info["status"] == "failed":
        response["result"] = job_info["result"]
        response["message"] = "Download failed."
    elif job_info["status"] == "downloading":
        response["message"] = "Download in progress..."

    return response


def search_unsplash_photos(
    query: str, per_page: int = 10, orientation: str = ""
) -> dict:
    """Search for photos on Unsplash.

    Args:
        query: Search query (e.g., "wood texture", "brick wall")
        per_page: Number of results to return (1-30, default: 10)
        orientation: Filter by orientation (landscape, portrait, squarish, or empty for all)

    Returns:
        Dictionary with search results
    """
    try:
        from ..preferences import get_preferences

        prefs = get_preferences()

        if not prefs.unsplash_api_key:
            return {
                "error": "Unsplash API key not configured. "
                "Get a free key at https://unsplash.com/developers and add it in preferences."
            }

        # Unsplash API endpoint
        api_url = "https://api.unsplash.com/search/photos"

        params = {
            "query": query,
            "per_page": min(max(1, per_page), 30),  # Clamp between 1-30
        }

        if orientation and orientation.lower() in ["landscape", "portrait", "squarish"]:
            params["orientation"] = orientation.lower()

        headers = {"Authorization": f"Client-ID {prefs.unsplash_api_key}"}

        with httpx.Client() as client:
            response = client.get(api_url, params=params, headers=headers, timeout=10.0)
            response.raise_for_status()
            data = response.json()

        results = data.get("results", [])

        # Format results
        formatted = f"Found {data.get('total', 0)} photos for '{query}':\n\n"
        for photo in results:
            photographer = photo.get("user", {}).get("name", "Unknown")
            description = photo.get("description") or photo.get(
                "alt_description", "No description"
            )
            width = photo.get("width", 0)
            height = photo.get("height", 0)

            formatted += f"• {description}\n"
            formatted += f"  By: {photographer}\n"
            formatted += f"  Size: {width}x{height}\n"
            formatted += f"  ID: {photo.get('id')}\n\n"

        return {
            "success": True,
            "total": data.get("total", 0),
            "photos": [
                {
                    "id": p.get("id"),
                    "description": p.get("description") or p.get("alt_description", ""),
                    "photographer": p.get("user", {}).get("name", "Unknown"),
                    "photographer_url": p.get("user", {})
                    .get("links", {})
                    .get("html", ""),
                    "width": p.get("width", 0),
                    "height": p.get("height", 0),
                    "color": p.get("color", "#000000"),
                    "download_url": p.get("urls", {}).get("raw", ""),
                }
                for p in results
            ],
            "formatted": formatted,
        }

    except Exception as e:
        return {"error": f"Failed to search Unsplash: {str(e)}"}


def download_unsplash_photo(photo_id: str, apply_as_texture: bool = False) -> dict:
    """Download a photo from Unsplash and optionally apply it as a texture.

    Args:
        photo_id: The Unsplash photo ID
        apply_as_texture: If True, apply to active object as texture (default: True)
        use_background: If True, run download in background thread to avoid UI lockup (default: False for compatibility)

    Returns:
        Dictionary with download status or job tracking info if use_background=True
    """
    # If background download requested, delegate to background runner
    if use_background:
        return _run_download_in_background(
            _download_unsplash_photo_impl, photo_id, apply_as_texture
        )
    else:
        return _download_unsplash_photo_impl(photo_id, apply_as_texture)


def _download_unsplash_photo_impl(photo_id: str, apply_as_texture: bool = True) -> dict:
    """Internal implementation of Unsplash photo download.

    Args:
        photo_id: The Unsplash photo ID
        apply_as_texture: If True, apply to active object as texture

    Returns:
        Dictionary with download status
    """
    try:
        from ..preferences import get_preferences

        prefs = get_preferences()

        if not prefs.unsplash_api_key:
            return {"error": "Unsplash API key not configured"}

        headers = {"Authorization": f"Client-ID {prefs.unsplash_api_key}"}

        # Get photo details
        api_url = f"https://api.unsplash.com/photos/{photo_id}"

        with httpx.Client() as client:
            response = client.get(api_url, headers=headers, timeout=10.0)
            response.raise_for_status()
            photo = response.json()

        # Trigger download tracking (required by Unsplash API)
        download_location = photo.get("links", {}).get("download_location")
        if download_location:
            with httpx.Client() as client:
                client.get(download_location, headers=headers, timeout=10.0)

        # Download the image
        download_url = photo.get("urls", {}).get("raw", "")
        if not download_url:
            return {"error": "No download URL found"}

        # Add parameters for high quality
        download_url += "&q=85&fm=jpg"

        # Create download directory
        download_dir = os.path.join(bpy.app.tempdir, "unsplash_photos")
        os.makedirs(download_dir, exist_ok=True)

        # Download file
        file_path = os.path.join(download_dir, f"{photo_id}.jpg")

        print(f"[Unsplash] Downloading photo: {photo_id}...")

        # Download with progress bar (safe context handling)
        wm = bpy.context.window_manager
        has_progress = wm is not None

        if has_progress:
            try:
                wm.progress_begin(0, 100)
            except:
                has_progress = False

        try:
            with httpx.Client() as client:
                with client.stream("GET", download_url, timeout=60.0) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0

                    with open(file_path, "wb") as f:
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if has_progress and total_size > 0:
                                try:
                                    progress = int((downloaded / total_size) * 100)
                                    wm.progress_update(progress)
                                except:
                                    pass

        finally:
            if has_progress:
                try:
                    wm.progress_end()

                except:
                    pass

                # Ensure the image is loaded and standardized with both prefixed and plain ID names
                # Primary name: unsplash_<id>; Alias: <id>
                try:
                    img = bpy.data.images.load(file_path, check_existing=True)
                except Exception:
                    img = bpy.data.images.get(str(photo_id))
                if img:
                    # Set primary standardized name
                    try:
                        img.name = f"unsplash_{photo_id}"
                    except Exception:
                        pass
                    # Ensure a plain-ID alias exists as a separate datablock (same file path)
                    try:
                        if not bpy.data.images.get(str(photo_id)):
                            alias = img.copy()
                            alias.name = str(photo_id)
                    except Exception:
                        pass

                photographer = photo.get("user", {}).get("name", "Unknown")

        photographer_url = photo.get("user", {}).get("links", {}).get("html", "")
        description = photo.get("description") or photo.get(
            "alt_description", "Unsplash photo"
        )

        result = {
            "success": True,
            "file_path": file_path,
            "photo_id": photo_id,
            "photographer": photographer,
            "photographer_url": photographer_url,
            "description": description,
            "message": f"Downloaded: {description} by {photographer}",
            "image_name": f"unsplash_{photo_id}",
            "image_names": [f"unsplash_{photo_id}", str(photo_id)],
        }

        # Apply as texture if requested
        if apply_as_texture:
            obj = bpy.context.active_object
            if obj and obj.type == "MESH":
                # Create material
                mat_name = f"unsplash_{photo_id}"
                mat = bpy.data.materials.get(mat_name)
                if not mat:
                    mat = bpy.data.materials.new(name=mat_name)

                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links

                # Clear existing nodes
                nodes.clear()

                # Create nodes
                bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
                bsdf.location = (0, 0)

                output = nodes.new(type="ShaderNodeOutputMaterial")
                output.location = (300, 0)

                tex = nodes.new(type="ShaderNodeTexImage")

                # Prefer standardized names: unsplash_<id> first, then plain ID, then load from file
                img = (
                    bpy.data.images.get(f"unsplash_{photo_id}")
                    or bpy.data.images.get(str(photo_id))
                    or bpy.data.images.load(file_path, check_existing=True)
                )

                tex.image = img
                tex.location = (-300, 0)

                # Link nodes
                links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
                links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

                # Assign material
                if obj.data.materials:
                    obj.data.materials[0] = mat
                else:
                    obj.data.materials.append(mat)

                result["applied_to"] = obj.name
                result["message"] += f" and applied to {obj.name}"
            else:
                result["message"] += ". No active mesh object to apply to."

        # Add attribution info
        result["attribution"] = (
            f"Photo by {photographer} on Unsplash ({photographer_url})"
        )

        return result

    except Exception as e:
        return {"error": f"Failed to download Unsplash photo: {str(e)}"}


def search_stock_photos(
    source: str, query: str, per_page: int = 10, orientation: str = ""
) -> dict:
    """Search for stock photos on Unsplash or Pexels (consolidated).

    Args:
        source: Photo source - "unsplash" or "pexels"
        query: Search query
        per_page: Number of results
        orientation: Filter by orientation

    Returns:
        Dictionary with search results
    """
    if source == "unsplash":
        return search_unsplash_photos(query, per_page, orientation)
    elif source == "pexels":
        return search_pexels_photos(query, per_page, orientation)
    else:
        return {"error": f"Invalid source: {source}. Use 'unsplash' or 'pexels'"}


def download_stock_photo(
    source: str,
    photo_id: str,
    apply_as_texture: bool = False,
) -> dict:
    """Download a stock photo and optionally apply as texture (consolidated).



    Args:

        source: Photo source - "unsplash" or "pexels"

        photo_id: Photo ID from search results

        apply_as_texture: Apply to active object as texture


    Returns:
        Dictionary with download result
    """
    if source == "unsplash":
        return download_unsplash_photo(photo_id, apply_as_texture)
    elif source == "pexels":
        return download_pexels_photo(photo_id, apply_as_texture)
    else:
        return {"error": f"Invalid source: {source}. Use 'unsplash' or 'pexels'"}


# Registration function (called from __init__.py)
def register():
    """Register stock photo tools conditionally based on API keys."""
    try:
        from ..preferences import get_preferences

        prefs = get_preferences()
    except:
        prefs = None

    # Check which sources have API keys
    has_unsplash = prefs and prefs.unsplash_api_key
    has_pexels = prefs and prefs.pexels_api_key

    # Only register if at least one API key is configured
    if has_unsplash or has_pexels:
        # Build available sources list
        available_sources = []
        if has_unsplash:
            available_sources.append("unsplash")
        if has_pexels:
            available_sources.append("pexels")

        # Consolidated search tool
        tool_registry.register_tool(
            "search_stock_photos",
            search_stock_photos,
            f"Search for stock photos. Available sources: {', '.join(available_sources)}. Returns high-quality free photos for textures and references.",
            {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": f"Photo source to search",
                        "enum": available_sources,
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'wood texture', 'brick wall', 'concrete surface')",
                    },
                    "per_page": {
                        "type": "number",
                        "description": "Number of results (1-30, default: 10)",
                        "default": 10,
                    },
                    "orientation": {
                        "type": "string",
                        "description": "Filter by orientation: landscape, portrait, square/squarish, or empty for all",
                    },
                },
                "required": ["source", "query"],
            },
            category="Stock Photos",
            requires_main_thread=False
        )

        tool_registry.register_tool(
            "download_stock_photo",
            download_stock_photo,
            f"Download a stock photo by ID (optionally apply as texture). Available sources: {', '.join(available_sources)}.",
            {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": f"Photo source",
                        "enum": available_sources,
                    },
                    "photo_id": {
                        "type": "string",
                        "description": "Photo ID from search results",
                    },
                    "apply_as_texture": {
                        "type": "boolean",
                        "description": "Apply to active object as texture",
                        "default": False,
                    },
                },
                "required": ["source", "photo_id"],
            },
        )


def search_pexels_photos(query: str, per_page: int = 10, orientation: str = "") -> dict:
    """Search for photos on Pexels.

    Args:
        query: Search query (e.g., "concrete texture", "metal surface")
        per_page: Number of results to return (1-80, default: 10)
        orientation: Filter by orientation (landscape, portrait, square, or empty for all)

    Returns:
        Dictionary with search results
    """
    try:
        from ..preferences import get_preferences

        prefs = get_preferences()

        if not prefs.pexels_api_key:
            return {
                "error": "Pexels API key not configured. "
                "Get a free key at https://www.pexels.com/api/ and add it in preferences."
            }

        # Pexels API endpoint
        api_url = "https://api.pexels.com/v1/search"

        params = {
            "query": query,
            "per_page": min(max(1, per_page), 80),  # Clamp between 1-80
        }

        if orientation and orientation.lower() in ["landscape", "portrait", "square"]:
            params["orientation"] = orientation.lower()

        headers = {"Authorization": prefs.pexels_api_key}

        with httpx.Client() as client:
            response = client.get(api_url, params=params, headers=headers, timeout=10.0)
            response.raise_for_status()
            data = response.json()

        photos = data.get("photos", [])

        # Format results
        formatted = f"Found {data.get('total_results', 0)} photos for '{query}':\n\n"
        for photo in photos:
            photographer = photo.get("photographer", "Unknown")
            alt = photo.get("alt", "No description")
            width = photo.get("width", 0)
            height = photo.get("height", 0)

            formatted += f"• {alt}\n"
            formatted += f"  By: {photographer}\n"
            formatted += f"  Size: {width}x{height}\n"
            formatted += f"  ID: {photo.get('id')}\n\n"

        return {
            "success": True,
            "total": data.get("total_results", 0),
            "photos": [
                {
                    "id": p.get("id"),
                    "description": p.get("alt", ""),
                    "photographer": p.get("photographer", "Unknown"),
                    "photographer_url": p.get("photographer_url", ""),
                    "width": p.get("width", 0),
                    "height": p.get("height", 0),
                    "avg_color": p.get("avg_color", "#000000"),
                    "download_url": p.get("src", {}).get("original", ""),
                }
                for p in photos
            ],
            "formatted": formatted,
        }

    except Exception as e:
        return {"error": f"Failed to search Pexels: {str(e)}"}


def download_pexels_photo(photo_id: int, apply_as_texture: bool = False) -> dict:
    """Download a photo from Pexels and optionally apply it as a texture.

    Args:
        photo_id: The Pexels photo ID
        apply_as_texture: If True, apply to active object as texture (default: True)
        use_background: If True, run download in background thread to avoid UI lockup (default: False for compatibility)

    Returns:
        Dictionary with download status or job tracking info if use_background=True
    """
    # If background download requested, delegate to background runner
    if use_background:
        return _run_download_in_background(
            _download_pexels_photo_impl, photo_id, apply_as_texture
        )
    else:
        return _download_pexels_photo_impl(photo_id, apply_as_texture)


def _download_pexels_photo_impl(photo_id: int, apply_as_texture: bool = True) -> dict:
    """Internal implementation of Pexels photo download.

    Args:
        photo_id: The Pexels photo ID
        apply_as_texture: If True, apply to active object as texture

    Returns:
        Dictionary with download status
    """
    try:
        from ..preferences import get_preferences

        prefs = get_preferences()

        if not prefs.pexels_api_key:
            return {"error": "Pexels API key not configured"}

        headers = {"Authorization": prefs.pexels_api_key}

        # Get photo details
        api_url = f"https://api.pexels.com/v1/photos/{photo_id}"

        with httpx.Client() as client:
            response = client.get(api_url, headers=headers, timeout=10.0)
            response.raise_for_status()
            photo = response.json()

        # Get download URL (original quality)
        download_url = photo.get("src", {}).get("original", "")
        if not download_url:
            return {"error": "No download URL found"}

        # Create download directory
        download_dir = os.path.join(bpy.app.tempdir, "pexels_photos")
        os.makedirs(download_dir, exist_ok=True)

        # Determine file extension from URL
        ext = ".jpg"
        if ".png" in download_url.lower():
            ext = ".png"

        # Download file
        file_path = os.path.join(download_dir, f"{photo_id}{ext}")

        print(f"[Pexels] Downloading photo: {photo_id}...")

        # Download with progress bar (safe context handling)
        wm = bpy.context.window_manager
        has_progress = wm is not None

        if has_progress:
            try:
                wm.progress_begin(0, 100)
            except:
                has_progress = False

        try:
            with httpx.Client() as client:
                with client.stream("GET", download_url, timeout=60.0) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0

                    with open(file_path, "wb") as f:
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if has_progress and total_size > 0:
                                try:
                                    progress = int((downloaded / total_size) * 100)
                                    wm.progress_update(progress)
                                except:
                                    pass

        finally:
            if has_progress:
                try:
                    wm.progress_end()

                except:
                    pass

                # Ensure the image is loaded and standardized with both prefixed and plain ID names
                # Primary name: pexels_<id>; Alias: <id>
                try:
                    img = bpy.data.images.load(file_path, check_existing=True)
                except Exception:
                    img = bpy.data.images.get(str(photo_id))
                if img:
                    # Set primary standardized name
                    try:
                        img.name = f"pexels_{photo_id}"
                    except Exception:
                        pass
                    # Ensure a plain-ID alias exists as a separate datablock (same file path)
                    try:
                        if not bpy.data.images.get(str(photo_id)):
                            alias = img.copy()
                            alias.name = str(photo_id)
                    except Exception:
                        pass

                photographer = photo.get("photographer", "Unknown")

        photographer_url = photo.get("photographer_url", "")
        description = photo.get("alt", "Pexels photo")

        result = {
            "success": True,
            "file_path": file_path,
            "photo_id": photo_id,
            "photographer": photographer,
            "photographer_url": photographer_url,
            "description": description,
            "message": f"Downloaded: {description} by {photographer}",
            "image_name": f"pexels_{photo_id}",
            "image_names": [f"pexels_{photo_id}", str(photo_id)],
        }

        # Apply as texture if requested
        if apply_as_texture:
            obj = bpy.context.active_object
            if obj and obj.type == "MESH":
                # Create material
                mat_name = f"pexels_{photo_id}"
                mat = bpy.data.materials.get(mat_name)
                if not mat:
                    mat = bpy.data.materials.new(name=mat_name)

                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                links = mat.node_tree.links

                # Clear existing nodes
                nodes.clear()

                # Create nodes
                bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
                bsdf.location = (0, 0)

                output = nodes.new(type="ShaderNodeOutputMaterial")
                output.location = (300, 0)

                tex = nodes.new(type="ShaderNodeTexImage")

                # Prefer standardized names: pexels_<id> first, then plain ID, then load from file
                img = (
                    bpy.data.images.get(f"pexels_{photo_id}")
                    or bpy.data.images.get(str(photo_id))
                    or bpy.data.images.load(file_path, check_existing=True)
                )

                tex.image = img
                tex.location = (-300, 0)

                # Link nodes
                links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
                links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

                # Assign material
                if obj.data.materials:
                    obj.data.materials[0] = mat
                else:
                    obj.data.materials.append(mat)

                result["applied_to"] = obj.name
                result["message"] += f" and applied to {obj.name}"
            else:
                result["message"] += ". No active mesh object to apply to."

        # Add attribution info
        result["attribution"] = (
            f"Photo by {photographer} on Pexels ({photographer_url})"
        )

        return result

    except Exception as e:
        return {"error": f"Failed to download Pexels photo: {str(e)}"}

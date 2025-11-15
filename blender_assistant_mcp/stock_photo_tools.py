"""Stock photo download tools for Unsplash and Pexels.

This module provides tools for searching and downloading stock photos
from Unsplash and Pexels to use as textures and references in Blender.
"""

import httpx
import os
import bpy
from . import mcp_tools


def search_unsplash_photos(query: str, per_page: int = 10, orientation: str = "") -> dict:
    """Search for photos on Unsplash.
    
    Args:
        query: Search query (e.g., "wood texture", "brick wall")
        per_page: Number of results to return (1-30, default: 10)
        orientation: Filter by orientation (landscape, portrait, squarish, or empty for all)
        
    Returns:
        Dictionary with search results
    """
    try:
        from .preferences import get_preferences
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
            "per_page": min(max(1, per_page), 30)  # Clamp between 1-30
        }
        
        if orientation and orientation.lower() in ["landscape", "portrait", "squarish"]:
            params["orientation"] = orientation.lower()
        
        headers = {
            "Authorization": f"Client-ID {prefs.unsplash_api_key}"
        }
        
        with httpx.Client() as client:
            response = client.get(api_url, params=params, headers=headers, timeout=10.0)
            response.raise_for_status()
            data = response.json()
        
        results = data.get("results", [])
        
        # Format results
        formatted = f"Found {data.get('total', 0)} photos for '{query}':\n\n"
        for photo in results:
            photographer = photo.get("user", {}).get("name", "Unknown")
            description = photo.get("description") or photo.get("alt_description", "No description")
            width = photo.get("width", 0)
            height = photo.get("height", 0)
            
            formatted += f"• {description}\n"
            formatted += f"  By: {photographer}\n"
            formatted += f"  Size: {width}x{height}\n"
            formatted += f"  ID: {photo.get('id')}\n\n"
        
        return {
            "success": True,
            "total": data.get("total", 0),
            "photos": [{
                "id": p.get("id"),
                "description": p.get("description") or p.get("alt_description", ""),
                "photographer": p.get("user", {}).get("name", "Unknown"),
                "photographer_url": p.get("user", {}).get("links", {}).get("html", ""),
                "width": p.get("width", 0),
                "height": p.get("height", 0),
                "color": p.get("color", "#000000"),
                "download_url": p.get("urls", {}).get("raw", "")
            } for p in results],
            "formatted": formatted
        }
        
    except Exception as e:
        return {"error": f"Failed to search Unsplash: {str(e)}"}


def download_unsplash_photo(photo_id: str, apply_as_texture: bool = True) -> dict:
    """Download a photo from Unsplash and optionally apply it as a texture.
    
    Args:
        photo_id: The Unsplash photo ID
        apply_as_texture: If True, apply to active object as texture (default: True)
        
    Returns:
        Dictionary with download status
    """
    try:
        from .preferences import get_preferences
        prefs = get_preferences()
        
        if not prefs.unsplash_api_key:
            return {"error": "Unsplash API key not configured"}
        
        headers = {
            "Authorization": f"Client-ID {prefs.unsplash_api_key}"
        }
        
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
        
        # Download with progress bar
        bpy.context.window_manager.progress_begin(0, 100)
        try:
            with httpx.Client() as client:
                with client.stream("GET", download_url, timeout=60.0) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = int((downloaded / total_size) * 100)
                                bpy.context.window_manager.progress_update(progress)
        finally:
            bpy.context.window_manager.progress_end()
        
        photographer = photo.get("user", {}).get("name", "Unknown")
        photographer_url = photo.get("user", {}).get("links", {}).get("html", "")
        description = photo.get("description") or photo.get("alt_description", "Unsplash photo")
        
        result = {
            "success": True,
            "file_path": file_path,
            "photo_id": photo_id,
            "photographer": photographer,
            "photographer_url": photographer_url,
            "description": description,
            "message": f"Downloaded: {description} by {photographer}"
        }
        
        # Apply as texture if requested
        if apply_as_texture:
            obj = bpy.context.active_object
            if obj and obj.type == 'MESH':
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
                bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf.location = (0, 0)
                
                output = nodes.new(type='ShaderNodeOutputMaterial')
                output.location = (300, 0)
                
                tex = nodes.new(type='ShaderNodeTexImage')
                tex.image = bpy.data.images.load(file_path)
                tex.location = (-300, 0)
                
                # Link nodes
                links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
                links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
                
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
        result["attribution"] = f"Photo by {photographer} on Unsplash ({photographer_url})"
        
        return result
        
    except Exception as e:
        return {"error": f"Failed to download Unsplash photo: {str(e)}"}


def search_stock_photos(source: str, query: str, per_page: int = 10, orientation: str = "") -> dict:
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


def download_stock_photo(source: str, photo_id: str, apply_as_texture: bool = True) -> dict:
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
        from .preferences import get_preferences
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
        mcp_tools.register_tool(
            "search_stock_photos",
            search_stock_photos,
            f"Search for stock photos. Available sources: {', '.join(available_sources)}. Returns high-quality free photos for textures and references.",
            {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": f"Photo source to search",
                        "enum": available_sources
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'wood texture', 'brick wall', 'concrete surface')"
                    },
                    "per_page": {
                        "type": "number",
                        "description": "Number of results (1-30, default: 10)",
                        "default": 10
                    },
                    "orientation": {
                        "type": "string",
                        "description": "Filter by orientation: landscape, portrait, square/squarish, or empty for all"
                    }
                },
                "required": ["source", "query"]
            },
            category="Stock Photos"
        )

        # Consolidated download tool
        mcp_tools.register_tool(
            "download_stock_photo",
            download_stock_photo,
            f"Download a stock photo and optionally apply as texture. Available sources: {', '.join(available_sources)}.",
            {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Photo source (must match the source used in search)",
                        "enum": available_sources
                    },
                    "photo_id": {
                        "type": "string",
                        "description": "The photo ID from search results"
                    },
                    "apply_as_texture": {
                        "type": "boolean",
                        "description": "Apply to active object as texture (default: true)",
                        "default": True
                    }
                },
                "required": ["source", "photo_id"]
            },
            category="Stock Photos"
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
        from .preferences import get_preferences
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
            "per_page": min(max(1, per_page), 80)  # Clamp between 1-80
        }

        if orientation and orientation.lower() in ["landscape", "portrait", "square"]:
            params["orientation"] = orientation.lower()

        headers = {
            "Authorization": prefs.pexels_api_key
        }

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
            "photos": [{
                "id": p.get("id"),
                "description": p.get("alt", ""),
                "photographer": p.get("photographer", "Unknown"),
                "photographer_url": p.get("photographer_url", ""),
                "width": p.get("width", 0),
                "height": p.get("height", 0),
                "avg_color": p.get("avg_color", "#000000"),
                "download_url": p.get("src", {}).get("original", "")
            } for p in photos],
            "formatted": formatted
        }

    except Exception as e:
        return {"error": f"Failed to search Pexels: {str(e)}"}


def download_pexels_photo(photo_id: int, apply_as_texture: bool = True) -> dict:
    """Download a photo from Pexels and optionally apply it as a texture.

    Args:
        photo_id: The Pexels photo ID
        apply_as_texture: If True, apply to active object as texture (default: True)

    Returns:
        Dictionary with download status
    """
    try:
        from .preferences import get_preferences
        prefs = get_preferences()

        if not prefs.pexels_api_key:
            return {"error": "Pexels API key not configured"}

        headers = {
            "Authorization": prefs.pexels_api_key
        }

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

        # Download with progress bar
        bpy.context.window_manager.progress_begin(0, 100)
        try:
            with httpx.Client() as client:
                with client.stream("GET", download_url, timeout=60.0) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0

                    with open(file_path, 'wb') as f:
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = int((downloaded / total_size) * 100)
                                bpy.context.window_manager.progress_update(progress)
        finally:
            bpy.context.window_manager.progress_end()

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
            "message": f"Downloaded: {description} by {photographer}"
        }

        # Apply as texture if requested
        if apply_as_texture:
            obj = bpy.context.active_object
            if obj and obj.type == 'MESH':
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
                bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
                bsdf.location = (0, 0)

                output = nodes.new(type='ShaderNodeOutputMaterial')
                output.location = (300, 0)

                tex = nodes.new(type='ShaderNodeTexImage')
                tex.image = bpy.data.images.load(file_path)
                tex.location = (-300, 0)

                # Link nodes
                links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
                links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

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
        result["attribution"] = f"Photo by {photographer} on Pexels ({photographer_url})"

        return result

    except Exception as e:
        return {"error": f"Failed to download Pexels photo: {str(e)}"}

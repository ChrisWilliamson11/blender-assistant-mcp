"""PolyHaven asset download and management tools.

This module provides tools for searching and downloading assets from PolyHaven.org
(HDRIs, textures, and models).
"""

import os

import bpy
import httpx

from . import tool_registry


def search_polyhaven_assets(asset_type: str, query: str = "", limit: int = 10) -> dict:
    """Search for assets on PolyHaven.

    Args:
        asset_type: Type of asset (hdris, textures, models)
        query: Search query (optional)
        limit: Maximum number of results to return

    Returns:
        Dictionary with search results
    """
    # PolyHaven API endpoint - get ALL assets (no type filter in URL)
    api_url = "https://api.polyhaven.com/assets"

    with httpx.Client() as client:
        response = client.get(api_url, timeout=10.0)
        response.raise_for_status()
        all_assets = response.json()

    # Map asset_type string to type number
    # 0 = HDRI, 1 = Texture, 2 = Model
    type_map = {
        "hdri": 0,
        "hdris": 0,
        "texture": 1,
        "textures": 1,
        "model": 2,
        "models": 2,
        "": None,  # Empty string = all types
    }

    type_num = type_map.get(asset_type.lower())

    # Filter by type if specified
    if type_num is not None:
        assets = {k: v for k, v in all_assets.items() if v.get("type") == type_num}
    else:
        assets = all_assets

    print(
        f"[DEBUG] PolyHaven search: total assets={len(all_assets)}, after type filter={len(assets)}, type_num={type_num}"
    )

    # Filter by query if provided
    if query:
        query_lower = query.lower()
        # Split query into words for better matching (e.g., "starry sky" → ["starry", "sky"])
        query_words = query_lower.split()

        # Match if ANY word appears in ID, name, or categories
        filtered = {
            k: v
            for k, v in assets.items()
            if all(
                word in k.lower()
                or word in v.get("name", "").lower()
                or any(word in cat.lower() for cat in v.get("categories", []))
                for word in query_words
            )
        }
        print(
            f"[DEBUG] PolyHaven search: query='{query}' (words: {query_words}), results after filter={len(filtered)}"
        )
    else:
        filtered = assets

    # Limit results
    results = list(filtered.items())[:limit]

    # Format results
    type_name = asset_type if asset_type else "assets"
    formatted = f"Found {len(results)} {type_name}"

    if results:
        formatted += f". Use download_polyhaven() with one of these asset IDs:\n\n"
        for asset_id, info in results:
            name = info.get("name", asset_id)
            categories = ", ".join(info.get("categories", []))
            formatted += f"• {name} (ID: {asset_id})\n"
            if categories:
                formatted += f"  Categories: {categories}\n"
            formatted += "\n"
    else:
        formatted += f". Try a different search term or browse all by searching with empty query.\n"

    return {
        "success": True,
        "assets": [
            {
                "id": k,
                "name": v.get("name", k),
                "categories": v.get("categories", []),
            }
            for k, v in results
        ],
        "count": len(results),
        "formatted": formatted,
    }


def get_polyhaven_asset_info(asset_id: str) -> dict:
    """Get detailed information about a specific PolyHaven asset.

    Args:
        asset_id: The asset ID (e.g., 'abandoned_warehouse_04')

    Returns:
        Dictionary with asset information
    """
    api_url = f"https://api.polyhaven.com/info/{asset_id}"

    with httpx.Client() as client:
        response = client.get(api_url, timeout=10.0)
        # Check for 404 manually to give better error
        if response.status_code == 404:
            raise KeyError(f"Asset '{asset_id}' not found on PolyHaven.")
        response.raise_for_status()
        info = response.json()

    # Format information
    formatted = f"Asset: {info.get('name', asset_id)}\n"
    formatted += f"Type: {info.get('type', 'unknown')}\n"
    formatted += f"Categories: {', '.join(info.get('categories', []))}\n"
    formatted += f"Author: {info.get('author', 'unknown')}\n"

    # Available resolutions
    files = info.get("files", {})
    if files:
        formatted += "\nAvailable downloads:\n"
        for file_type, resolutions in files.items():
            formatted += f"  {file_type}: {', '.join(resolutions.keys())}\n"

    return {"success": True, "info": info, "formatted": formatted}


def download_polyhaven_hdri(
    asset_id: str, resolution: str = "2k", file_format: str = "exr"
) -> dict:
    """Download an HDRI from PolyHaven and set it as the world background.

    Args:
        asset_id: The HDRI ID (e.g., 'abandoned_warehouse_04')
        resolution: Resolution to download (1k, 2k, 4k, 8k, 16k)
        file_format: File format (exr, hdr)

    Returns:
        Dictionary with download status and file path
    """
    # Get asset info to find download URL
    api_url = f"https://api.polyhaven.com/files/{asset_id}"

    with httpx.Client() as client:
        response = client.get(api_url, timeout=10.0)
        # Check for 404 manually to give better error
        if response.status_code == 404:
             raise ValueError(f"Asset '{asset_id}' not found on PolyHaven.")
        response.raise_for_status()
        files = response.json()

    # Find the requested file
    if "hdri" not in files:
        raise ValueError(f"Asset '{asset_id}' is not an HDRI")

    hdri_files = files["hdri"]
    if resolution not in hdri_files:
        available = ", ".join(hdri_files.keys())
        raise ValueError(f"Resolution '{resolution}' not available. Available: {available}")

    if file_format not in hdri_files[resolution]:
        available = ", ".join(hdri_files[resolution].keys())
        raise ValueError(f"Format '{file_format}' not available. Available: {available}")

    download_url = hdri_files[resolution][file_format]["url"]

    # Download the file
    download_dir = os.path.join(bpy.app.tempdir, "polyhaven_hdris")
    os.makedirs(download_dir, exist_ok=True)

    file_path = os.path.join(download_dir, f"{asset_id}_{resolution}.{file_format}")

    print(f"[PolyHaven] Downloading {asset_id} ({resolution}, {file_format})...")

    # Download with progress bar (if context available)
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
                        if total_size > 0 and has_progress:
                            progress = int((downloaded / total_size) * 100)
                            try:
                                wm.progress_update(progress)
                            except:
                                has_progress = False
    finally:
        if has_progress:
            try:
                wm.progress_end()
            except:
                pass

    # Load the HDRI into Blender's world
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    # Clear existing nodes
    nodes.clear()

    # Create environment texture node
    env_tex = nodes.new(type="ShaderNodeTexEnvironment")
    env_tex.image = bpy.data.images.load(file_path)
    env_tex.location = (-300, 300)

    # Create background node
    background = nodes.new(type="ShaderNodeBackground")
    background.location = (0, 300)

    # Create output node
    output = nodes.new(type="ShaderNodeOutputWorld")
    output.location = (300, 300)

    # Link nodes
    links.new(env_tex.outputs["Color"], background.inputs["Color"])
    links.new(background.outputs["Background"], output.inputs["Surface"])

    return {
        "success": True,
        "file_path": file_path,
        "asset_id": asset_id,
        "resolution": resolution,
        "message": f"Downloaded and loaded HDRI: {asset_id} ({resolution}) from {file_path}",
    }


def download_polyhaven_texture(asset_id: str, resolution: str = "2k") -> dict:
    """Download a PBR texture set from PolyHaven and apply it to the active object.

    Args:
        asset_id: The texture ID (e.g., 'brick_wall_001')
        resolution: Resolution to download (1k, 2k, 4k, 8k)

    Returns:
        Dictionary with download status and file paths
    """
    # Get asset info to find download URLs
    api_url = f"https://api.polyhaven.com/files/{asset_id}"

    with httpx.Client() as client:
        response = client.get(api_url, timeout=10.0)
        if response.status_code == 404:
            raise ValueError(f"Asset '{asset_id}' not found on PolyHaven.")
        response.raise_for_status()
        files = response.json()

    # Check if this is a texture
    if "Diffuse" not in files and "blend" not in files:
        raise ValueError(f"Asset '{asset_id}' is not a texture")

    # Download directory
    download_dir = os.path.join(bpy.app.tempdir, "polyhaven_textures", asset_id)
    os.makedirs(download_dir, exist_ok=True)

    # Download all texture maps
    texture_maps = {}
    # Map types to look for (handle various naming conventions)
    # PolyHaven and other sources use different names for the same maps
    map_types_config = {
        "Diffuse": [
            "Diffuse",
            "diff",
            "col",
            "Color",
            "BaseColor",
            "Albedo",
            "albedo",
        ],
        "Displacement": ["Displacement", "disp", "Disp", "height", "Height"],
        "Normal": [
            "Normal",
            "nor_gl",
            "nor_dx",
            "norm",
            "Norm",
            "normal",
        ],  # nor_gl = OpenGL, nor_dx = DirectX
        "Roughness": ["Roughness", "Rough", "rough", "Gloss", "gloss"],
        "AO": ["AO", "ao", "AmbientOcclusion", "ambient_occlusion"],
        "Metalness": ["Metalness", "Metal", "metal", "Metallic", "metallic"],
        "Specular": ["Specular", "spec", "Spec", "specular"],
        "Bump": ["Bump", "bump"],
        "ARM": ["arm", "ARM"],  # Combined AO+Roughness+Metalness (packed texture)
    }

    print(f"[PolyHaven] Downloading texture set: {asset_id} ({resolution})...")
    print(f"[PolyHaven] Available map types in API response: {list(files.keys())}")

    # Find which maps are available (check all variants)
    available_maps = {}
    for map_name, variants in map_types_config.items():
        for variant in variants:
            if variant in files and resolution in files[variant]:
                available_maps[map_name] = variant
                break

    print(f"[PolyHaven] Maps found for {resolution}: {list(available_maps.keys())}")
    total_maps = len(available_maps)

    # Start progress bar (if context available)
    wm = bpy.context.window_manager
    has_progress = wm is not None

    if has_progress:
        try:
            wm.progress_begin(0, total_maps)
        except:
            has_progress = False

    current_map = 0

    try:
        for map_name, api_key in available_maps.items():
            # Get the first available format (usually jpg or png)
            formats = files[api_key][resolution]
            file_format = list(formats.keys())[0]
            download_url = formats[file_format]["url"]

            file_path = os.path.join(
                download_dir, f"{asset_id}_{map_name}_{resolution}.{file_format}"
            )

            if has_progress:
                try:
                    wm.progress_update(current_map)
                except:
                    has_progress = False

            with httpx.Client() as client:
                with client.stream("GET", download_url, timeout=60.0) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0

                    with open(file_path, "wb") as f:
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0 and has_progress:
                                file_progress = current_map + (
                                    downloaded / total_size
                                )
                                try:
                                    wm.progress_update(file_progress)
                                except:
                                    has_progress = False

            texture_maps[map_name] = file_path
            print(f"  Downloaded: {map_name} (from '{api_key}')")
            current_map += 1
    finally:
        if has_progress:
            try:
                wm.progress_end()
            except:
                pass

    if not texture_maps:
        raise ValueError(f"No texture maps found for resolution '{resolution}'")

    # Create material (ALWAYS)
    mat_name = f"{asset_id}_material"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create Principled BSDF
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    # Create output node
    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (300, 0)

    # Link BSDF to output
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    y_offset = 300

    # Helper to load image
    def load_image_node(path, label, x, y, is_data=True):
        try:
            img = bpy.data.images.load(path)
            node = nodes.new(type="ShaderNodeTexImage")
            node.image = img
            node.label = label
            node.location = (x, y)
            if is_data:
                node.image.colorspace_settings.name = "Non-Color"
            return node
        except Exception as e:
            print(f"Failed to load image {path}: {e}")
            return None

    # Add texture nodes
    if "Diffuse" in texture_maps:
        tex = load_image_node(texture_maps["Diffuse"], "Diffuse", -300, y_offset, is_data=False)
        if tex:
            links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        y_offset -= 300

    # Handle Roughness (can be standalone or in ARM texture)
    if "Roughness" in texture_maps:
        tex = load_image_node(texture_maps["Roughness"], "Roughness", -300, y_offset, is_data=True)
        if tex:
            links.new(tex.outputs["Color"], bsdf.inputs["Roughness"])
        y_offset -= 300
    elif "ARM" in texture_maps:
        # ARM texture: R=AO, G=Roughness, B=Metalness
        tex = load_image_node(texture_maps["ARM"], "ARM", -600, y_offset, is_data=True)
        if tex:
            # Separate RGB to extract channels
            separate = nodes.new(type="ShaderNodeSeparateColor") # Blender 3.3+ uses Separate Color
            # Fallback for older Blender versions if needed, but 4.0+ uses Separate Color
            if "Separate Color" not in [n.name for n in bpy.types.ShaderNode.bl_rna.properties['type'].enum_items]:
                    separate = nodes.new(type="ShaderNodeSeparateRGB")
            
            separate.location = (-300, y_offset)
            links.new(tex.outputs["Color"], separate.inputs[0]) # Input is usually index 0 'Image' or 'Color'

            # Green channel = Roughness
            links.new(separate.outputs["Green"], bsdf.inputs["Roughness"])

            # Blue channel = Metalness
            links.new(separate.outputs["Blue"], bsdf.inputs["Metallic"])

            # Red channel = AO (we'll handle this separately if no standalone AO)
            if "AO" not in texture_maps:
                # Mix AO (red channel) with base color if we have diffuse
                if "Diffuse" in texture_maps:
                    mix = nodes.new(type="ShaderNodeMix") # Blender 3.4+ uses Mix Node
                    mix.data_type = 'RGBA'
                    mix.blend_type = "MULTIPLY"
                    mix.inputs[0].default_value = 1.0 # Factor
                    mix.location = (-150, y_offset + 300)

                    # Find existing diffuse connection
                    diffuse_source = None
                    for link in list(links):
                        if link.to_socket == bsdf.inputs["Base Color"]:
                            diffuse_source = link.from_socket
                            links.remove(link)
                            break

                    if diffuse_source:
                        links.new(diffuse_source, mix.inputs[6]) # A
                        links.new(separate.outputs["Red"], mix.inputs[7]) # B
                        links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])

        y_offset -= 300

    # Normal map
    if "Normal" in texture_maps:
        tex = load_image_node(texture_maps["Normal"], "Normal", -600, y_offset, is_data=True)
        if tex:
            normal_map = nodes.new(type="ShaderNodeNormalMap")
            normal_map.location = (-300, y_offset)
            links.new(tex.outputs["Color"], normal_map.inputs["Color"])
            links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
        y_offset -= 300

    # Bump map (alternative to normal map)
    elif "Bump" in texture_maps:
        tex = load_image_node(texture_maps["Bump"], "Bump", -600, y_offset, is_data=True)
        if tex:
            bump_node = nodes.new(type="ShaderNodeBump")
            bump_node.location = (-300, y_offset)
            links.new(tex.outputs["Color"], bump_node.inputs["Height"])
            links.new(bump_node.outputs["Normal"], bsdf.inputs["Normal"])
        y_offset -= 300

    if "Displacement" in texture_maps:
        tex = load_image_node(texture_maps["Displacement"], "Displacement", -300, y_offset, is_data=True)
        if tex:
            disp = nodes.new(type="ShaderNodeDisplacement")
            disp.location = (0, -300)
            links.new(tex.outputs["Color"], disp.inputs["Height"])
            links.new(disp.outputs["Displacement"], output.inputs["Displacement"])
            mat.cycles.displacement_method = "DISPLACEMENT"
        y_offset -= 300

    # Handle Metalness (standalone only if not in ARM)
    if "Metalness" in texture_maps and "ARM" not in texture_maps:
        tex = load_image_node(texture_maps["Metalness"], "Metalness", -300, y_offset, is_data=True)
        if tex:
            links.new(tex.outputs["Color"], bsdf.inputs["Metallic"])
        y_offset -= 300

    # Specular map
    if "Specular" in texture_maps:
        tex = load_image_node(texture_maps["Specular"], "Specular", -300, y_offset, is_data=True)
        if tex:
            # Check for Specular input (Blender 4.0+ vs older)
            if "Specular IOR Level" in bsdf.inputs:
                    links.new(tex.outputs["Color"], bsdf.inputs["Specular IOR Level"])
            elif "Specular" in bsdf.inputs:
                    links.new(tex.outputs["Color"], bsdf.inputs["Specular"])
        y_offset -= 300

    if "AO" in texture_maps:
        # AO needs to be mixed with base color
        tex = load_image_node(texture_maps["AO"], "AO", -600, y_offset, is_data=True)
        if tex:
            # Create Mix Node
            mix = nodes.new(type="ShaderNodeMix")
            mix.data_type = 'RGBA'
            mix.blend_type = "MULTIPLY"
            mix.inputs[0].default_value = 1.0
            mix.location = (-150, y_offset)

            # Find existing diffuse connection
            diffuse_source = None
            for link in list(links):
                if link.to_socket == bsdf.inputs["Base Color"]:
                    diffuse_source = link.from_socket
                    links.remove(link)
                    break

            if diffuse_source:
                links.new(diffuse_source, mix.inputs[6]) # A
                links.new(tex.outputs["Color"], mix.inputs[7]) # B
                links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])

    # Apply to active object if one exists
    obj = bpy.context.active_object
    if obj and obj.type == "MESH":
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        return {
            "success": True,
            "texture_maps": texture_maps,
            "asset_id": asset_id,
            "material_name": mat_name,
            "file_path": texture_maps.get("Diffuse") or list(texture_maps.values())[0], # Legacy/Consistency helper
            "message": f"Downloaded and created material '{mat_name}'. Assigned to active object: {obj.name}"
        }

    return {
        "success": True,
        "texture_maps": texture_maps,
        "asset_id": asset_id,
        "material_name": mat_name,
        "resolution": resolution,
        "message": f"Downloaded and created material '{mat_name}'. No active mesh to apply to."
    }


def download_polyhaven_model(asset_id: str, file_format: str = "blend") -> dict:
    """Download a 3D model from PolyHaven and import it into the scene.

    Args:
        asset_id: The model ID (e.g., 'wooden_chair_01')
        file_format: File format (blend, fbx, gltf)

    Returns:
        Dictionary with download status and imported objects
    """
    # Get asset info to find download URL
    api_url = f"https://api.polyhaven.com/files/{asset_id}"

    with httpx.Client() as client:
        response = client.get(api_url, timeout=10.0)
        # Check for 404
        if response.status_code == 404:
            raise ValueError(f"Asset '{asset_id}' not found on PolyHaven.")
        response.raise_for_status()
        files = response.json()

    # Normalize format name
    format_lower = file_format.lower()

    # Find the requested format (case-insensitive)
    format_data = None
    actual_format_key = None

    for key in files.keys():
        if key.lower() == format_lower:
            format_data = files[key]
            actual_format_key = key
            break

    if format_data is None:
        # Filter out texture map keys (they contain underscores or are texture types)
        model_formats = [
            k
            for k in files.keys()
            if k.lower() in ["blend", "fbx", "gltf", "glb", "usd", "obj"]
        ]
        available = (
            ", ".join(model_formats) if model_formats else ", ".join(files.keys())
        )
        raise ValueError(f"Format '{file_format}' not available. Available model formats: {available}")

    # Get download URL
    download_url = None

    if isinstance(format_data, dict):
        # Look for resolution keys (1k, 2k, 4k, 8k)
        resolution_keys = ["2k", "4k", "1k", "8k"]  # Prefer 2k

        for res_key in resolution_keys:
            if res_key in format_data:
                res_data = format_data[res_key]

                # Now look for the format key again (nested structure)
                if isinstance(res_data, dict):
                    # Try the format key itself
                    if actual_format_key in res_data:
                        file_info = res_data[actual_format_key]
                        if isinstance(file_info, dict) and "url" in file_info:
                            download_url = file_info["url"]
                            print(f"[PolyHaven] Found URL at resolution {res_key}")
                            break

                    # Also try lowercase version
                    format_lower_key = actual_format_key.lower()
                    if format_lower_key in res_data:
                        file_info = res_data[format_lower_key]
                        if isinstance(file_info, dict) and "url" in file_info:
                            download_url = file_info["url"]
                            print(f"[PolyHaven] Found URL at resolution {res_key}")
                            break

    if not download_url:
        raise ValueError(f"Could not find download URL for format '{file_format}'. API structure may have changed.")

    # Determine actual file extension
    file_ext = actual_format_key.lower()
    if file_ext == "gltf":
        file_ext = "gltf"  # Keep as gltf

    # Download the file and any included textures
    download_dir = os.path.join(bpy.app.tempdir, "polyhaven_models", asset_id)
    os.makedirs(download_dir, exist_ok=True)

    file_path = os.path.join(download_dir, f"{asset_id}.{file_ext}")

    print(f"[PolyHaven] Downloading model: {asset_id} ({file_ext})...")
    print(f"[PolyHaven] Download URL: {download_url}")

    # Start progress bar
    wm = bpy.context.window_manager
    has_progress = wm is not None
    if has_progress:
        try:
            wm.progress_begin(0, 100)
        except:
             has_progress = False

    try:
        # Download main file
        with httpx.Client() as client:
            with client.stream("GET", download_url, timeout=120.0) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(file_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0 and has_progress:
                            progress = int((downloaded / total_size) * 100)
                            try:
                                wm.progress_update(progress)
                            except:
                                has_progress = False
    finally:
        if has_progress:
             try:
                wm.progress_end()
             except:
                pass

    # Import the model
    # Use appropriate importer based on extension
    imported_objects = []

    if file_ext == 'blend':
        # Link from .blend file
        with bpy.data.libraries.load(file_path, link=False) as (data_from, data_to):
            data_to.objects = data_from.objects

        scene = bpy.context.scene
        for obj in data_to.objects:
            if obj is not None:
                scene.collection.objects.link(obj)
                imported_objects.append(obj)
    elif file_ext in ['gltf', 'glb']:
        bpy.ops.import_scene.gltf(filepath=file_path)
        imported_objects = bpy.context.selected_objects
    elif file_ext == 'fbx':
        bpy.ops.import_scene.fbx(filepath=file_path)
        imported_objects = bpy.context.selected_objects
    elif file_ext == 'obj':
        bpy.ops.import_scene.obj(filepath=file_path)
        imported_objects = bpy.context.selected_objects

    return {
        "success": True,
        "file_path": file_path,
        "asset_id": asset_id,
        "imported_count": len(imported_objects),
        "imported_names": [o.name for o in imported_objects],
        "message": f"Downloaded and imported model '{asset_id}'."
    }


def download_polyhaven(
    asset_type: str, asset_id: str, resolution: str = "2k", file_format: str = None
) -> dict:
    """Download a PolyHaven asset (HDRI, texture, or model).



    Args:

        asset_type: 'hdri' | 'texture' | 'model' (plurals accepted)

        asset_id: Asset ID from search results

        resolution: For hdri/texture only ('1k'...'16k', default '2k')

        file_format: For hdri/model only (e.g., 'exr'/'hdr', or 'blend'/'fbx'/'gltf'); auto if omitted



    Returns:

        Dict describing the download/import result or an error.

    """

    # Normalize asset_type to singular lowercase (accept common plurals/synonyms)

    at = (asset_type or "").strip().lower()

    normalize = {
        "hdris": "hdri",
        "hdri": "hdri",
        "textures": "texture",
        "texture": "texture",
        "models": "model",
        "model": "model",
    }

    at = normalize.get(at, at)

    if at == "hdri":
        return download_polyhaven_hdri(asset_id, resolution, file_format or "exr")

    elif at == "texture":
        return download_polyhaven_texture(asset_id, resolution)

    elif at == "model":
        return download_polyhaven_model(asset_id, file_format or "blend")

    else:
        return {
            "error": f"Invalid asset_type: {asset_type}. Use 'hdri', 'texture', or 'model'"
        }


def register():
    """Register all PolyHaven tools with the MCP registry."""

    tool_registry.register_tool(
        "search_polyhaven_assets",
        search_polyhaven_assets,
        "Search PolyHaven (HDRIs, textures, models). Returns a dict with 'assets' (list of {id, name, categories}) and 'count'.",
        {
            "type": "object",
            "properties": {
                "asset_type": {
                    "type": "string",
                    "description": "Type of asset to search for (use singular: hdri, texture, model)",
                    "enum": ["hdri", "hdris", "texture", "textures", "model", "models"],
                },
                "query": {
                    "type": "string",
                    "description": "Search query (searches in asset name, ID, and categories). Examples: 'night sky', 'wood', 'brick'",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["asset_type"],
        },
        category="PolyHaven",
    )

    tool_registry.register_tool(
        "get_polyhaven_asset_info",
        get_polyhaven_asset_info,
        "Get detailed info about a PolyHaven asset (including available resolutions).",
        {
            "type": "object",
            "properties": {
                "asset_id": {
                    "type": "string",
                    "description": "The asset ID",
                },
            },
            "required": ["asset_id"],
        },
        category="PolyHaven",
    )

    # Consolidated download tool

    tool_registry.register_tool(
        "download_polyhaven",
        download_polyhaven,
        "Download from PolyHaven (hdri, texture, model). Requires 'asset_type' and 'asset_id'. 'resolution' applies to hdri/texture; 'file_format' applies to hdri/model (auto if omitted).",
        {
            "type": "object",
            "properties": {
                "asset_type": {
                    "type": "string",
                    "description": "Type of asset to download",
                    "enum": ["hdri", "hdris", "texture", "textures", "model", "models"],
                },
                "asset_id": {
                    "type": "string",
                    "description": "Asset ID (from search)",
                },
                "resolution": {
                    "type": "string",
                    "description": "Resolution for hdri/texture (1k–16k). Default: 2k",
                    "enum": ["1k", "2k", "4k", "8k", "16k"],
                    "default": "2k",
                },
                "file_format": {
                    "type": "string",
                    "description": "Format for hdri/model; auto if omitted.",
                },
            },
            "required": ["asset_type", "asset_id"],
        },
        category="PolyHaven",
    )

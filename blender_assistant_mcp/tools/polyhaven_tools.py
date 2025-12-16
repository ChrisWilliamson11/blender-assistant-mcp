"""PolyHaven asset download and management tools.

This module provides tools for searching and downloading assets from PolyHaven.org
(HDRIs, textures, and models).
"""

import os

import bpy
import httpx

from . import tool_registry



# Headers for API compliance
REQ_HEADERS = {
    "User-Agent": "BlenderAssistant/1.0 (Educational)"
}

def search_polyhaven_assets(asset_type: str = "", query: str = "", limit: int = 10, sort: str = "relevant") -> dict:
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

    with httpx.Client(headers=REQ_HEADERS) as client:
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

    # print(
    #     f"[DEBUG] PolyHaven search: total assets={len(all_assets)}, after type filter={len(assets)}, type_num={type_num}"
    # )

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
        # print(
        #     f"[DEBUG] PolyHaven search: query='{query}' (words: {query_words}), results after filter={len(filtered)}"
        # )
    else:
        filtered = assets

    # Sorting Logic
    # Convert to list of tuples for sorting
    results = list(filtered.items())

    if sort == "random":
        import random
        random.shuffle(results)
    elif sort == "newest":
        # Sort by date_published (descending)
        # date_published is unix timestamp
        results.sort(key=lambda x: x[1].get("date_published", 0), reverse=True)
    # else: "relevant" - implicit order from dict usually by ID or whatever python implementation does. 
    # If query was used, strict relevancy isn't implemented beyond filtering, 
    # but we could sort by string match quality if we wanted. For now, kept simple.

    # Limit results
    results = results[:limit]

    # Format results
    type_name = asset_type if asset_type else "assets"
    formatted = f"Found {len(results)} {type_name}"

    if results:
        formatted += f". Use download_polyhaven(asset_type='{type_name}', asset_id='<id>') with one of these:\n\n"
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

    with httpx.Client(headers=REQ_HEADERS) as client:
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


def _solve_resolution(available_res: list, target_res: str) -> str:
    """Find the best matching resolution from available options."""
    if not available_res:
        return target_res
        
    target = str(target_res).lower()
    if target in available_res:
        return target
        
    def parse_res(r):
        if not r: return 0
        r = r.lower()
        if 'k' in r:
            try: return int(r.replace('k', '')) * 1024
            except: return 0
        try: return int(r)
        except: return 0
            
    target_val = parse_res(target)
    valid = [(r, parse_res(r)) for r in available_res if parse_res(r) > 0]
    valid.sort(key=lambda x: x[1])
    
    if not valid: return available_res[0]

    # Max/Highest logic
    if target_val == 0 or target in ["max", "highest", "best"]:
        return valid[-1][0]
        
    best = valid[0]
    min_diff = abs(valid[0][1] - target_val)
    
    for r, val in valid:
        diff = abs(val - target_val)
        if diff < min_diff:
            min_diff = diff
            best = (r, val)
        elif diff == min_diff:
            if val > best[1]: best = (r, val)
            
    if best[0] != target_res:
        print(f"[PolyHaven] Resolution '{target_res}' not found. Using '{best[0]}'.")
        
    return best[0]


def _download_polyhaven_hdri_by_id(
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

    with httpx.Client(headers=REQ_HEADERS) as client:
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
    
    # Check format
    if file_format not in hdri_files:
        available_formats = list(hdri_files.keys())
        # Prefer EXR > HDR
        if "exr" in available_formats: file_format = "exr"
        elif "hdr" in available_formats: file_format = "hdr"
        else: file_format = available_formats[0]
        print(f"[PolyHaven] Requested format not found. Switching to {file_format}.")

    # Solve Resolution
    available_res = list(hdri_files[file_format].keys())
    resolution = _solve_resolution(available_res, resolution)

    if resolution not in hdri_files[file_format]:
        available = ", ".join(hdri_files[file_format].keys())
        raise ValueError(f"Resolution '{resolution}' not available for format '{file_format}'. Available: {available}")

    download_url = hdri_files[file_format][resolution]["url"]

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
        with httpx.Client(headers=REQ_HEADERS) as client:
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


def _download_polyhaven_texture_by_id(asset_id: str, resolution: str = "2k") -> dict:
    """Download a PBR texture set from PolyHaven and apply it to the active object.

    Args:
        asset_id: The texture ID (e.g., 'brick_wall_001')
        resolution: Resolution to download (1k, 2k, 4k, 8k)

    Returns:
        Dictionary with download status and file paths
    """
    # Get asset info to find download URLs
    api_url = f"https://api.polyhaven.com/files/{asset_id}"

    with httpx.Client(headers=REQ_HEADERS) as client:
        response = client.get(api_url, timeout=10.0)
        # Check for 404 manually
        if response.status_code == 404:
            raise ValueError(f"Asset '{asset_id}' not found on PolyHaven.")
        response.raise_for_status()
        files = response.json()

    # Check if this is a texture
    # Textures typically have keys like 'Diffuse', 'Roughness' OR 'blend' etc.
    # We should check if it's NOT a Model or HDRI predominantly.
    # But usually valid texture assets have specific map names OR 'blend' files.
    
    # Download directory
    download_dir = os.path.join(bpy.app.tempdir, "polyhaven_textures", asset_id)
    os.makedirs(download_dir, exist_ok=True)

    # Download all texture maps
    texture_maps = {}
    
    # Map types config
    map_types_config = {
        "Diffuse": ["Diffuse", "diff", "col", "Color", "BaseColor", "Albedo", "albedo"],
        "Displacement": ["Displacement", "disp", "Disp", "height", "Height"],
        "Normal": ["Normal", "nor_gl", "nor_dx", "norm", "Norm", "normal"], 
        "Roughness": ["Roughness", "Rough", "rough", "Gloss", "gloss"],
        "AO": ["AO", "ao", "AmbientOcclusion", "ambient_occlusion"],
        "Metalness": ["Metalness", "Metal", "metal", "Metallic", "metallic"],
        "Specular": ["Specular", "spec", "Spec", "specular"],
        "Bump": ["Bump", "bump"],
        "ARM": ["arm", "ARM"],
    }
    
    # Gather Available Resolutions
    # Check all map variants to find common resolution set
    candidates = set()
    for map_key, map_variants in map_types_config.items():
         for variant in map_variants:
              if variant in files:
                   # files[variant] is {res: {fmt: ...}}
                   candidates.update(files[variant].keys())
    
    # Also check blend if present (sometimes textures just have blend)
    if "blend" in files:
        candidates.update(files["blend"].keys())
        
    available_res = list(candidates)
    
    # Solve Resolution
    target_res = resolution
    resolution = _solve_resolution(available_res, resolution)
    
    if resolution != target_res:
         print(f"[PolyHaven] Switching texture download to {resolution} (Available: {available_res})")

    # Find which maps are available for this resolution
    available_maps = {}
    for map_name, variants in map_types_config.items():
        for variant in variants:
            if variant in files and resolution in files[variant]:
                available_maps[map_name] = variant
                break

    if not available_maps and "blend" in files and resolution in files["blend"]:
         # Fallback to downloading blend? 
         # The function promises to apply to active object.
         # For now, let's error if individual maps aren't found, OR warn.
         pass
         
    total_maps = len(available_maps)
    
    if total_maps == 0:
         hint = available_res if available_res else "None"
         raise ValueError(f"No compatible texture maps found for resolution '{resolution}'. Available resolutions: {hint}")

    print(f"[PolyHaven] Downloading texture set: {asset_id} ({resolution})...")
    print(f"[PolyHaven] Maps: {list(available_maps.keys())}")

    # Start progress bar
    wm = bpy.context.window_manager
    has_progress = wm is not None
    if has_progress:
        try: wm.progress_begin(0, total_maps)
        except: has_progress = False

    current_map = 0

    try:
        for map_name, api_key in available_maps.items():
            # Get the first available format
            formats = files[api_key][resolution]
            file_format = list(formats.keys())[0]
            download_url = formats[file_format]["url"]

            file_path = os.path.join(download_dir, f"{asset_id}_{map_name}_{resolution}.{file_format}")

            if has_progress:
                try: wm.progress_update(current_map)
                except: has_progress = False

            with httpx.Client(headers=REQ_HEADERS) as client:
                with client.stream("GET", download_url, timeout=60.0) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0
                    with open(file_path, "wb") as f:
                        for chunk in response.iter_bytes(chunk_size=8192):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0 and has_progress:
                                fp = current_map + (downloaded / total_size)
                                try: wm.progress_update(fp)
                                except: has_progress = False

            texture_maps[map_name] = file_path
            current_map += 1
    finally:
        if has_progress:
            try: wm.progress_end()
            except: pass

    # Create Material Logic (Existing)
    # ... (Rest of logic is cleaner to keep similar but condensed)
    
    mat_name = f"{asset_id}_material"
    mat = bpy.data.materials.get(mat_name)
    if not mat:
        mat = bpy.data.materials.new(name=mat_name)

    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)

    output = nodes.new(type="ShaderNodeOutputMaterial")
    output.location = (300, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    y_offset = 300

    def load_image_node(path, label, x, y, is_data=True):
        try:
            img = bpy.data.images.load(path)
            node = nodes.new(type="ShaderNodeTexImage")
            node.image = img
            node.label = label
            node.location = (x, y)
            if is_data: node.image.colorspace_settings.name = "Non-Color"
            return node
        except Exception as e:
            print(f"Failed to load image {path}: {e}")
            return None

    # Diffuse
    if "Diffuse" in texture_maps:
        tex = load_image_node(texture_maps["Diffuse"], "Diffuse", -300, y_offset, is_data=False)
        if tex: links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
        y_offset -= 300

    # Roughness / ARM
    if "Roughness" in texture_maps:
        tex = load_image_node(texture_maps["Roughness"], "Roughness", -300, y_offset, is_data=True)
        if tex: links.new(tex.outputs["Color"], bsdf.inputs["Roughness"])
        y_offset -= 300
    elif "ARM" in texture_maps:
        tex = load_image_node(texture_maps["ARM"], "ARM", -600, y_offset, is_data=True)
        if tex:
            sep = nodes.new(type="ShaderNodeSeparateColor")
            # Fallback check
            if "Separate Color" not in [n.name for n in bpy.types.ShaderNode.bl_rna.properties['type'].enum_items]:
                 sep = nodes.new(type="ShaderNodeSeparateRGB")
            
            sep.location = (-300, y_offset)
            links.new(tex.outputs["Color"], sep.inputs[0])
            links.new(sep.outputs["Green"], bsdf.inputs["Roughness"])
            links.new(sep.outputs["Blue"], bsdf.inputs["Metallic"])
            
            if "AO" not in texture_maps and "Diffuse" in texture_maps:
                 mix = nodes.new(type="ShaderNodeMix")
                 mix.data_type = 'RGBA'
                 mix.blend_type = "MULTIPLY"
                 mix.inputs[0].default_value = 1.0
                 mix.location = (-150, y_offset + 300)
                 
                 # Hook up
                 # Find existing diffuse link
                 for link in list(links):
                      if link.to_socket == bsdf.inputs["Base Color"]:
                           links.new(link.from_socket, mix.inputs[6])
                           links.remove(link)
                           break
                 links.new(sep.outputs["Red"], mix.inputs[7])
                 links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])
        y_offset -= 300

    # Normal / Bump
    if "Normal" in texture_maps:
         tex = load_image_node(texture_maps["Normal"], "Normal", -600, y_offset, is_data=True)
         if tex:
              nmap = nodes.new(type="ShaderNodeNormalMap")
              nmap.location = (-300, y_offset)
              links.new(tex.outputs["Color"], nmap.inputs["Color"])
              links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
         y_offset -= 300
    elif "Bump" in texture_maps:
         tex = load_image_node(texture_maps["Bump"], "Bump", -600, y_offset, is_data=True)
         if tex:
              bump = nodes.new(type="ShaderNodeBump")
              bump.location = (-300, y_offset)
              links.new(tex.outputs["Color"], bump.inputs["Height"])
              links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
         y_offset -= 300

    # Displacement
    if "Displacement" in texture_maps:
         tex = load_image_node(texture_maps["Displacement"], "Displacement", -300, y_offset, is_data=True)
         if tex:
              disp = nodes.new(type="ShaderNodeDisplacement")
              disp.location = (0, -300)
              links.new(tex.outputs["Color"], disp.inputs["Height"])
              links.new(disp.outputs["Displacement"], output.inputs["Displacement"])
              mat.cycles.displacement_method = "DISPLACEMENT"
         y_offset -= 300

    # Metalness
    if "Metalness" in texture_maps and "ARM" not in texture_maps:
         tex = load_image_node(texture_maps["Metalness"], "Metalness", -300, y_offset, is_data=True)
         if tex: links.new(tex.outputs["Color"], bsdf.inputs["Metallic"])
         y_offset -= 300
         
    # Specular
    if "Specular" in texture_maps:
         tex = load_image_node(texture_maps["Specular"], "Specular", -300, y_offset, is_data=True)
         if tex:
              if "Specular IOR Level" in bsdf.inputs:
                   links.new(tex.outputs["Color"], bsdf.inputs["Specular IOR Level"])
              elif "Specular" in bsdf.inputs:
                   links.new(tex.outputs["Color"], bsdf.inputs["Specular"])
         y_offset -= 300

    # AO (Standalone)
    if "AO" in texture_maps:
         tex = load_image_node(texture_maps["AO"], "AO", -600, y_offset, is_data=True)
         if tex:
              mix = nodes.new(type="ShaderNodeMix")
              mix.data_type = 'RGBA'
              mix.blend_type = "MULTIPLY"
              mix.inputs[0].default_value = 1.0
              mix.location = (-150, y_offset)
              
              for link in list(links):
                   if link.to_socket == bsdf.inputs["Base Color"]:
                        links.new(link.from_socket, mix.inputs[6])
                        links.remove(link)
                        break
              links.new(tex.outputs["Color"], mix.inputs[7])
              links.new(mix.outputs["Result"], bsdf.inputs["Base Color"])

    # Apply to object
    obj = bpy.context.active_object
    if obj and obj.type == "MESH":
         if obj.data.materials: obj.data.materials[0] = mat
         else: obj.data.materials.append(mat)
         
         return {
             "success": True, 
             "texture_maps": texture_maps, 
             "asset_id": asset_id,
             "material_name": mat_name,
             "file_path": texture_maps.get("Diffuse") or list(texture_maps.values())[0],
             "message": f"Downloaded texture set '{asset_id}' ({resolution}). Applied to {obj.name}."
         }
         
    return {
         "success": True,
         "texture_maps": texture_maps,
         "asset_id": asset_id,
         "material_name": mat_name,
         "resolution": resolution,
         "message": f"Downloaded texture set '{asset_id}' ({resolution}). Material created but no active mesh selected."
    }


def _download_polyhaven_model_by_id(asset_id: str, file_format: str = "blend", resolution: str = "2k") -> dict:
    """Download a 3D model from PolyHaven and import it into the scene.

    Args:
        asset_id: The model ID (e.g., 'wooden_chair_01')
        file_format: File format (blend, fbx, gltf)

    Returns:
        Dictionary with download status and imported objects
    """
    # Get asset info to find download URL
    api_url = f"https://api.polyhaven.com/files/{asset_id}"
    
    with httpx.Client(headers=REQ_HEADERS) as client:
        response = client.get(api_url, timeout=10.0)
        if response.status_code == 404:
            raise ValueError(f"Asset '{asset_id}' not found on PolyHaven.")
        response.raise_for_status()
        files = response.json()

    format_lower = file_format.lower() if file_format else "blend"
    
    # Aliases
    if format_lower == "glb": 
        format_lower = "gltf"

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
    files_to_download = {}  # { relative_path: url }

    if isinstance(format_data, dict):
        # Look for resolution keys (1k, 2k, 4k, 8k)
        # Prioritize requested resolution
        default_prio = ["2k", "4k", "1k", "8k"]
        resolution_keys = [resolution] + [r for r in default_prio if r != resolution]

        for res_key in resolution_keys:
            if res_key in format_data:
                res_data = format_data[res_key]

                # Now look for the format key again (nested structure)
                if isinstance(res_data, dict):
                    # Try the format key itself
                    main_key = None
                    if actual_format_key in res_data:
                        main_key = actual_format_key
                    elif actual_format_key.lower() in res_data:
                        main_key = actual_format_key.lower()

                    if main_key:
                        file_info = res_data[main_key]
                        if isinstance(file_info, dict) and "url" in file_info:
                            download_url = file_info["url"]
                            print(f"[PolyHaven] Found URL at resolution {res_key}")
                            
                            # Collect included files (textures, etc.)
                            if "include" in file_info and isinstance(file_info["include"], dict):
                                for rel_path, include_info in file_info["include"].items():
                                    if isinstance(include_info, dict) and "url" in include_info:
                                        files_to_download[rel_path] = include_info["url"]
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
    print(f"[PolyHaven] Included files: {len(files_to_download)}")

    # Start progress bar
    wm = bpy.context.window_manager
    has_progress = wm is not None
    if has_progress:
        try:
            wm.progress_begin(0, 100)
        except:
            has_progress = False

    try:
        # 1. Download Main File
        with httpx.Client(headers=REQ_HEADERS) as client:
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

        # 2. Download Included Files (Textures)
        for rel_path, url in files_to_download.items():
            # Handle paths like "textures/foo.jpg"
            target_path = os.path.join(download_dir, rel_path.replace("/", os.sep))
            target_dir = os.path.dirname(target_path)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            
            print(f"[PolyHaven] Downloading included: {rel_path}...")
            
            with httpx.Client(headers=REQ_HEADERS) as client:
                 # Simple download (no progress bar needed per file, or minimal)
                 r = client.get(url, timeout=60.0)
                 r.raise_for_status()
                 with open(target_path, "wb") as f:
                     f.write(r.content)

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
        try:
            with bpy.data.libraries.load(file_path, link=False) as (data_from, data_to):
                data_to.objects = data_from.objects

            scene = bpy.context.scene
            for obj in data_to.objects:
                if obj is not None:
                    scene.collection.objects.link(obj)
                    imported_objects.append(obj)
        except Exception as e:
            # If load fails, it might be due to missing libraries if not packed?
            # But we downloaded the textures now!
            print(f"Blend load error: {e}")
            raise

    elif file_ext in ['gltf', 'glb']:
        bpy.ops.import_scene.gltf(filepath=file_path)
        imported_objects = bpy.context.selected_objects
    elif file_ext == 'fbx':
        bpy.ops.import_scene.fbx(filepath=file_path)
        imported_objects = bpy.context.selected_objects
    elif file_ext == 'obj':
        bpy.ops.import_scene.obj(filepath=file_path)
        imported_objects = bpy.context.selected_objects
    
    # Ensure textures are packed (Critical for Blend files that reference external textures)
    # The 'include' files we downloaded are in relative paths. 
    # Blender 'Find Missing Files' might be needed if the paths aren't exactly what the .blend expects.
    # But usually Polyhaven .blends are relative.
    
    # Attempt to use 'find_missing_files' logic if textures are pink?
    # For now, just pack.
    try:
        bpy.ops.file.pack_all()
    except Exception as e:
        print(f"Warning: Failed to pack resources: {e}")
        
    if not imported_objects and file_ext != 'blend':
        # Fallback if importers don't select objects
        # This is tricky without tracking scene delta, but good enough for now
        pass

    return {
        "success": True,
        "file_path": file_path,
        "asset_id": asset_id,
        "imported_count": len(imported_objects),
        "imported_names": [o.name for o in imported_objects],
        "message": f"Downloaded and imported model '{asset_id}' with {len(files_to_download)} included files."
    }


# SMART WRAPPERS
def download_polyhaven_model(query_or_id: str, file_format: str = "blend", resolution: str = "2k", index: int = 0, sort: str = "random") -> dict:
    """Download a 3D model from PolyHaven (Auto-Search).

    Args:
        query_or_id: Asset ID (e.g. 'wooden_chair_01') OR search query (e.g. 'chair')
        file_format: blend (default), gltf, fbx
        resolution: Texture resolution (1k, 2k, 4k). Default 2k.
        index: Result index to pick (0 = 1st result, 1 = 2nd, etc.)
        sort: relevant, random, newest. Default "random".

    Returns:
        Dict having keys 'success', 'imported_objects', etc.
    """
    print(f"[PolyHaven] Smart Download Model: '{query_or_id}' (Sort: {sort}, Index: {index})")
    
    # Needs limit to reach the desired index
    limit_needed = index + 1
    
    search_res = search_polyhaven_assets(asset_type="model", query=query_or_id, limit=limit_needed, sort=sort)
    
    if not search_res.get("assets") or len(search_res["assets"]) <= index:
        return {"error": f"Model not found for query: '{query_or_id}' (Index {index} out of range)"}
        
    best_asset = search_res["assets"][index]
    asset_id = best_asset["id"]
    asset_name = best_asset["name"]
    
    print(f"[PolyHaven] Selected asset: {asset_name} (ID: {asset_id})")
    
    # Call internal download
    # Note: _download_model_by_id implementation doesn't support resolution arg yet in previous code?
    # Let's check. _download_polyhaven_model_by_id(asset_id, file_format)
    # It hardcodes resolution/format logic inside? 
    # Wait, the previous impl of download_polyhaven_model DID find resolution keys.
    # But strict signature was (asset_id, file_format). 
    # We should update internal one to respect resolution if possible, but for now strict proxy.
    
    try:
        return _download_polyhaven_model_by_id(asset_id, file_format=file_format, resolution=resolution)
    except Exception as e:
        return {"error": str(e)}


def download_polyhaven_texture(query_or_id: str, resolution: str = "2k", index: int = 0, sort: str = "random") -> dict:
    """Download a Texture from PolyHaven (Auto-Search) and apply to active object.

    Args:
        query_or_id: Asset ID or Search Query (e.g. "brick", "floor")
        resolution: 1k, 2k, 4k, 8k. Default 2k.
        index: Result index to pick.
        sort: relevant, random, newest. Default "random".
    """
    print(f"[PolyHaven] Smart Download Texture: '{query_or_id}' (Sort: {sort}, Index: {index})")
    
    limit_needed = index + 1
    search_res = search_polyhaven_assets(asset_type="texture", query=query_or_id, limit=limit_needed, sort=sort)
    
    if not search_res.get("assets") or len(search_res["assets"]) <= index:
        return {"error": f"Texture not found for query: '{query_or_id}' (Index {index} out of range)"}
        
    best_asset = search_res["assets"][index]
    asset_id = best_asset["id"]
    print(f"[PolyHaven] Selected texture: {best_asset['name']} (ID: {asset_id})")
    
    try:
        return _download_polyhaven_texture_by_id(asset_id, resolution=resolution)
    except Exception as e:
        return {"error": str(e)}


def download_polyhaven_hdri(query_or_id: str, resolution: str = "2k", index: int = 0, sort: str = "random") -> dict:
    """Download an HDRI from PolyHaven (Auto-Search) and set as World Background.

    Args:
        query_or_id: Asset ID or Search Query (e.g. "sunset", "studio")
        resolution: 1k, 2k, 4k, 8k. Default 2k.
        index: Result index to pick.
        sort: relevant, random, newest. Default "random".
    """
    print(f"[PolyHaven] Smart Download HDRI: '{query_or_id}' (Sort: {sort}, Index: {index})")
    
    limit_needed = index + 1
    search_res = search_polyhaven_assets(asset_type="hdri", query=query_or_id, limit=limit_needed, sort=sort)
    
    if not search_res.get("assets") or len(search_res["assets"]) <= index:
        return {"error": f"HDRI not found for query: '{query_or_id}' (Index {index} out of range)"}
        
    best_asset = search_res["assets"][index]
    asset_id = best_asset["id"]
    print(f"[PolyHaven] Selected HDRI: {best_asset['name']} (ID: {asset_id})")
    
    try:
        # Default format exr
        return _download_polyhaven_hdri_by_id(asset_id, resolution=resolution, file_format="exr")
    except Exception as e:
        return {"error": str(e)}


def register():
    """Register all PolyHaven tools with the MCP registry."""

    tool_registry.register_tool(
        "search_polyhaven_assets",
        search_polyhaven_assets,
        (
            "Search for HDRIs, textures, or models on PolyHaven.\n"
            "RETURNS: {'assets': [{'id': '...', 'name': '...', 'categories': [...]}, ...]}\n"
            "USAGE: Pass `asset_type` (hdri, texture, model) and `query` string."
        ),
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
                "sort": {
                    "type": "string",
                    "description": "Sort order: 'relevant', 'random', 'newest'",
                    "enum": ["relevant", "random", "newest"],
                    "default": "relevant"
                }
            },
            "required": ["asset_type"],
        },
        category="PolyHaven",
        requires_main_thread=False
    )

    tool_registry.register_tool(
        "get_polyhaven_asset_info",
        get_polyhaven_asset_info,
        (
            "Get detailed info including available resolutions for an asset.\n"
            "RETURNS: {'info': {'name': '...', 'files': {...}}}\n"
            "USAGE: Use this to check if '4k' or '8k' is available before downloading."
        ),
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
        requires_main_thread=False
    )

    # Consolidated download tool

    # Smart Tools Registration

    tool_registry.register_tool(
        "download_polyhaven_model",
        download_polyhaven_model,
        (
            "Find and download a 3D model from PolyHaven. Automatically searches for best match.\n"
            "USAGE: `download_polyhaven_model(query_or_id='red chair')`"
        ),
        {
            "type": "object",
            "properties": {
                "query_or_id": {"type": "string", "description": "Search query or precise Asset ID (e.g. 'wooden_table')"},
                "resolution": {"type": "string", "description": "Texture resolution (1k, 2k, 4k). Default '2k'", "default": "2k"},
                "file_format": {"type": "string", "description": "Format (blend, gltf, fbx). Default 'blend'", "default": "blend"},
                "index": {"type": "integer", "description": "Index of result to download (0=1st, 1=2nd). Default 0.", "default": 0},
                "sort": {"type": "string", "description": "Sort: 'relevant', 'random', 'newest'. Default 'random'", "default": "random", "enum": ["relevant", "random", "newest"]}
            },
            "required": ["query_or_id"]
        },
        category="PolyHaven"
    )

    tool_registry.register_tool(
        "download_polyhaven_texture",
        download_polyhaven_texture,
        (
            "Find and download a PBR Texture from PolyHaven and apply to Active Object.\n"
            "USAGE: `download_polyhaven_texture(query_or_id='brick wall')`"
        ),
        {
            "type": "object",
            "properties": {
                "query_or_id": {"type": "string", "description": "Search query or precise Asset ID"},
                "resolution": {"type": "string", "description": "Resolution (1k, 2k, 4k, 8k). Default '2k'", "default": "2k"},
                "index": {"type": "integer", "description": "Index of result to download (0=1st, 1=2nd). Default 0.", "default": 0},
                "sort": {"type": "string", "description": "Sort: 'relevant', 'random', 'newest'. Default 'random'", "default": "random", "enum": ["relevant", "random", "newest"]}
            },
            "required": ["query_or_id"]
        },
        category="PolyHaven"
    )

    tool_registry.register_tool(
        "download_polyhaven_hdri",
        download_polyhaven_hdri,
        (
            "Find and download an HDRI environment from PolyHaven and set as World Background.\n"
            "USAGE: `download_polyhaven_hdri(query_or_id='sunset')`"
        ),
        {
            "type": "object",
            "properties": {
                "query_or_id": {"type": "string", "description": "Search query or precise Asset ID"},
                "resolution": {"type": "string", "description": "Resolution (1k, 2k, 4k, 8k). Default '2k'", "default": "2k"},
                "index": {"type": "integer", "description": "Index of result to download (0=1st, 1=2nd). Default 0.", "default": 0},
                "sort": {"type": "string", "description": "Sort: 'relevant', 'random', 'newest'. Default 'random'", "default": "random", "enum": ["relevant", "random", "newest"]}
            },
            "required": ["query_or_id"]
        },
        category="PolyHaven"
    )

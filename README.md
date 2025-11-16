blender-assistant-mcp

`blender-assistant-mcp` is an open source project that integrates an LLM into Blender using the Model Context Protocol (MCP).

Features:

- **Control Blender with Natural Language:** Send prompts to an Ollama LLM model to perform actions in Blender.
- **MCP Integration:** Uses the Model Context Protocol for structured communication between the AI model and Blender.
- **Ollama Support:** Designed to work with Ollama for easy local model management.
- **Ollama bundled Add-on:** Includes the Ollama exe, and CUDA DLLS so is self contained

MCP Tools:

Below is a list of the tools exposed to the assistant. Each tool takes JSON-style arguments (shown by name only) but run through an SDK assistant_sdk to keep context light. It can use assistant_help to get information about tools, as well as get_scene_info to maintain a current image of the scene hierarchy, or other “info” tools such as viewport capture to understand the scene, then act with creation/modification tools.

- Blender
  - get_scene_info(an outliner style representation of the scene)
  - get_object_info(name) — Detailed info for a single object
  - create_object(type, name, location, rotation, scale, text) — Create meshes, cameras, lights, text, curves, empties
  - modify_object(name, location, rotation, scale, visible) — Edit object transforms/visibility
  - delete_object(name | names[]) — Delete one or more objects
  - set_material(object_name, material_name, color) — Set or create a material on an object
  - capture_viewport_for_vision(max_size, question) — Non-blocking viewport screenshot + VLM analysis
  - list_collections() — List scene collections and hierarchy
  - get_collection_info(collection_name) — Info for a specific collection
  - create_collection(name, parent, collections[]) — Create one or many collections
  - move_to_collection(object_names[], collection_name, unlink_from_others) — Move objects to a collection
  - set_collection_color(collection_name, color_tag) — Set collection color tag (Blender 4.2+)
  - delete_collection(collection_name, delete_objects) — Delete a collection (and optionally its objects)

- Selection
  - get_selection() — List selected objects
  - get_active() — Get the active object
  - set_selection(object_names[]) — Select one or more objects by name
  - set_active(object_name) — Make an object active
  - select_by_type(object_type) — Select all objects of a given type (MESH, LIGHT, etc.)

- PolyHaven
  - search_polyhaven_assets(asset_type, query, limit) — Find HDRIs, textures, or models
  - download_polyhaven(asset_id, file_format, resolution, apply_to_active?) — Download assets into Blender

- Web
  - web_search(query, num_results) — Simple web search (DuckDuckGo HTML)
  - search_wikimedia_image(query, apply_to_active) — Find and download free images from Wikimedia
  - fetch_webpage(url, max_length) — Fetch page and extract text content
  - download_image_as_texture(url, apply_to_active, pack_image) — Download an image and apply as texture

- Sketchfab
  - sketchfab_login(api_token) — Store token for authenticated requests
  - sketchfab_search(query, category, max_results) — Search Sketchfab
  - sketchfab_download_model(model_uid, apply_to_scene?) — Download and import a model

- RAG (Documentation)
  - rag_query(query, num_results, source_bias) — Retrieve doc snippets to augment prompts
  - rag_get_stats() — Database stats/status

- Code
  - execute_code(code) — Execute Python in Blender (persistent namespace between calls)

to build it yourself:

Clone the repository

build the rag database with 'build_rag_database.py' - this downloads the blender manual & API reference & builds a vector database from them to use as reference at inference time - you may need to update the download links if they change.

Install Ollama

Copy the exe & DLL's from your Ollama installation with 'update_ollama_bins.py

download the required python wheels with download_wheels.py


package the extension with scripts/build_extension.py

Usage:
- python scripts/build_extension.py
- python scripts/build_extension.py --dry-run
- python scripts/build_extension.py --output-dir dist


install in blender as per usual, when its installed you can download models in the preferences, or point it to your existing Ollama models folder.

A lot was learnt from Blender open MCP! (https://github.com/dhakalnirajan/blender-open-mcp)

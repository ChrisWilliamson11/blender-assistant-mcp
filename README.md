blender-assistant-mcp

`blender-assistant-mcp` is an open source project that integrates an LLM into Blender using the Model Context Protocol (MCP).

Its meant to be an interactive help / automation tool to allow you to free up some of the more mundane 3D tasks you dont enjoy, so you can spend more time on those you do. It excells at doing slightly 'fuzzy' but simple tasks than a built in operator doesnt do on its own.for example 'copy the bone constraints from every selected bone to its parent' or 'randomise the phase values of all the noise modifiers on every animation channel on my selected object' or 'why is my scene rendering black???' :D


Features:

- **Control Blender with Natural Language:** Send prompts to an Ollama LLM model to perform actions in Blender.
- **MCP Integration:** Uses the Model Context Protocol for structured communication between the AI model and Blender.
- **Code execution with MCP** Also has a code path to create an assistant_sdk the model can access to reduce context bloat.
- **RAG Database** includes a RAG database of both the blender manual, and API reference, which the assistant can access.
- **Vision Tool** it can capture the viewport and send the image to a dedicated vision LLM, to give it rudimentary 'vision'
- **Ollama Support:** Designed to work with Ollama for easy local model management.
- **Ollama bundled Add-on:** Includes the Ollama exe, and CUDA DLLS so is self contained install (the release)

MCP Tools:

Below is a list of the tools. Each tool takes JSON-style arguments & can be exposed as an MCP tool, or discoverable through an SDK assistant_sdk to keep context light. If running through the sdk, the LLM can use sdk_help to get information about tools. It maintains a 'state' of the scene through 'get_scene_info' to maintain a current image of the scene hierarchy, similar to using an outliner.

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
  - sdk_help(tool, tools[]) — Get JSON schemas and usage examples for assistant_sdk tool aliases

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
  - extract_image_urls(url, min_width, max_images) — Extract likely content image URLs from a webpage
  - download_image_as_texture(url, apply_to_active, pack_image) — Download an image and apply as texture

- Stock Photos
  - search_stock_photos(source, query, per_page, orientation) — Search Unsplash or Pexels for stock photos
  - download_stock_photo(source, photo_id, apply_as_texture) — Download and apply stock photos as textures

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

install in blender as per usual, when its installed you can download models in the preferences, or point it to your existing Ollama models folder.

A lot was learnt from Blender open MCP! (https://github.com/dhakalnirajan/blender-open-mcp)

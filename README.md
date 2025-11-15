blender-assistant-mcp

`blender-assistant-mcp` is an open source project that integrates an LLM into Blender using the Model Context Protocol (MCP). 

Features:

- **Control Blender with Natural Language:** Send prompts to an  Ollama LLM model to perform actions in Blender.
- **MCP Integration:** Uses the Model Context Protocol for structured communication between the AI model and Blender.
- **Ollama Support:** Designed to work with Ollama for easy local model management.
- **Ollama bundled Add-on:** Includes the Ollama exe, and CUDA DLLS so is self contained

MCP Tools:


## MCP Tools (a list of all included MCP tools):



Below is a concise, categorized list of the tools exposed to the assistant. Each tool takes JSON-style arguments (shown by name only). Use get_scene_info or other “info” tools first to understand the scene, then act with creation/modification tools.

- Blender
  - get_scene_info(info_level, object_type, object_names, limit) — Summarize or detail scene contents
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
  - ensure_collections(names[], parent) — Ensure collections exist (create missing ones)
  - safe_move_objects(object_names[], collection_name, create_if_missing, unlink_from_others) — Move objects (auto-create target if needed)
  - set_collections_color_batch(collection_names[], color_tag) — Color-tag multiple collections

- Selection
  - get_selection() — List selected objects
  - get_active() — Get the active object
  - set_selection(object_names[]) — Select one or more objects by name
  - set_active(object_name) — Make an object active
  - select_by_type(object_type) — Select all objects of a given type (MESH, LIGHT, etc.)

- PolyHaven
  - search_polyhaven_assets(asset_type, query, limit) — Find HDRIs, textures, or models
  - get_polyhaven_asset_info(asset_id) — Detailed info for a PolyHaven asset
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

- Meta (Task Planning / Helpers)
  - create_task_plan(task_description, subtasks[], expected_end_state) — Start a multi-step plan
  - get_current_subtask() — What’s next in the plan
  - complete_current_subtask() — Mark subtask done
  - get_task_progress() — Summary of plan status
  - cancel_task_plan() — Cancel current plan
  - set_task_context(key, value) / get_task_context(key) — Store/retrieve transient context
  - skip_current_subtask() — Skip ahead
  - check_task_completion() — Validate task completion
  - apply_to_each_selected(operation, operation_args) — Apply any tool to each selected item
  - for_each_in_list(items[], operation, operation_args) — Batch-apply a tool to a provided list

- Code
  - execute_code(code) — Execute Python in Blender (persistent namespace between calls)


A lot was learnt from Blender open MCP! (https://github.com/dhakalnirajan/blender-open-mcp)



# Tools Configuration System

## Overview

The Blender Assistant uses a **checkbox-based tools configuration system** in preferences to control which tools are exposed to the LLM via the OpenAI-style tools schema. This provides an intuitive way to customize the assistant's capabilities and optimize context usage.

## What Changed (Version 2.2.0+)

### Previous System
- Tools were hardcoded in `build_openai_tools()` function
- A separate UI-based tool selector panel existed in the 3D Viewport sidebar
- Tool selector panel wasn't integrated with schema-based tools
- No easy way to switch between tool sets

### New System (Current)
- **Checkbox-based UI** directly in preferences
- Tools grouped by category with visual enable/disable buttons
- Three quick presets available: Lean (Default), Core Only, All Tools
- `execute_code` is always enabled and locked (safety guarantee)
- Old 3D Viewport tool selector panel removed (deprecated)

## Location

**Edit → Preferences → Add-ons → Blender Assistant → Tools Configuration (Advanced)**

The section is collapsed by default. Click the triangle to expand it.

## How to Use

### Quick Presets

Three presets are available for common scenarios:

1. **Lean (Default)** - Recommended balanced set
   - Core Blender operations (scene info, object info, collections)
   - Selection management (get/set selection, select by type)
   - Code execution
   - Vision capture
   - Assistant help
   - ~16 tools total

2. **Core Only** - Minimal set for low-context scenarios
   - Just the essentials: `execute_code`, `get_scene_info`, `get_object_info`, `assistant_help`
   - Use when token budget is tight
   - ~4 tools total

3. **All Tools** - Maximum capabilities
   - Every registered tool in the MCP registry
   - Includes PolyHaven, stock photos, web search, Sketchfab, etc.
   - Higher context usage
   - ~30+ tools depending on what's registered

**Usage**: Simply click one of the three preset buttons at the top of the Tools Configuration section.

### Checkbox-Based UI

#### Category Organization

Tools are organized by category in collapsible boxes:

- **Blender** - Core Blender operations
- **Selection** - Object selection and activation
- **Collections** - Collection management
- **Vision** - Viewport capture
- **Web** - Web search and image download
- **PolyHaven** - Asset library integration
- **Stock Photos** - Unsplash/Pexels
- **Sketchfab** - 3D model downloads
- **Code Execution** - Python execution
- **RAG** - Documentation search

#### Using Categories

Each category box shows:
- Category name
- Enabled/total count: `Category Name (5/8)`
- **✓ button** - Enable all tools in this category
- **✗ button** - Disable all tools in this category

#### Individual Tool Checkboxes

- **Check/uncheck** any tool to enable/disable it
- **execute_code** is locked (always enabled) for safety
- Changes take effect immediately
- No need to save or apply

#### Refresh Tool List

If new tools are registered (e.g., after installing additional modules), click the **"Refresh Tool List"** button to update the UI with the latest registry.

### Status Display

At the top of the tools section, you'll see:
```
Enabled: 16 / 30 tools
```

This shows how many tools are currently enabled out of the total available.

## Tool Categories Explained

### Blender (Core Operations)
- `get_scene_info` - Outliner-style scene hierarchy
- `get_object_info` - Detailed object information
- `create_object` - Create primitives, cameras, lights, text
- `modify_object` - Edit transforms and visibility
- `delete_object` - Remove objects
- `set_material` - Apply/create materials
- `capture_viewport_for_vision` - Screenshot + VLM analysis
- `assistant_help` - SDK documentation access

### Collections
- `list_collections` - List all collections
- `get_collection_info` - Collection details
- `create_collection` - Create new collections
- `move_to_collection` - Move objects to collections
- `set_collection_color` - Set color tags
- `delete_collection` - Remove collections

### Selection
- `get_selection` - List selected objects
- `get_active` - Get active object
- `set_selection` - Select objects by name
- `set_active` - Make object active
- `select_by_type` - Select by type (MESH, LIGHT, etc.)

### Web
- `web_search` - DuckDuckGo web search
- `search_wikimedia_image` - Find free images
- `fetch_webpage` - Extract webpage content
- `extract_image_urls` - Find images on any webpage
- `download_image_as_texture` - Download and apply images

### Stock Photos
- `search_stock_photos` - Search Unsplash/Pexels
- `download_stock_photo` - Download and apply stock photos

### PolyHaven
- `search_polyhaven_assets` - Search HDRIs, textures, models
- `download_polyhaven` - Download and apply assets

### Sketchfab
- `sketchfab_login` - Authenticate with API token
- `sketchfab_search` - Search 3D models
- `sketchfab_download_model` - Download and import models

### RAG (Documentation)
- `rag_query` - Search Blender documentation
- `rag_get_stats` - Database statistics

### Code Execution
- `execute_code` - Execute Python code (always enabled)

## Benefits

1. **Reduced Context Usage** - Fewer tools = smaller OpenAI payload
2. **Visual Interface** - See all tools at a glance with checkboxes
3. **Category Management** - Enable/disable entire categories quickly
4. **Persistent Settings** - Saved with Blender preferences
5. **Flexible** - Easy presets or granular control
6. **Safe** - `execute_code` always included as safety net

## Technical Details

### For Developers

The tools configuration system consists of:

1. **Property Group** (`preferences.py`)
   ```python
   class ToolConfigItem(bpy.types.PropertyGroup):
       name: bpy.props.StringProperty(name="Tool Name")
       enabled: bpy.props.BoolProperty(name="Enabled", default=True)
       category: bpy.props.StringProperty(name="Category")
       description: bpy.props.StringProperty(name="Description")
   ```

2. **Collection Property** (on AssistantPreferences)
   ```python
   tool_config_items: bpy.props.CollectionProperty(type=ToolConfigItem)
   ```

3. **Sync Function** (`preferences.py`)
   ```python
   def _sync_tools_to_json(prefs):
       """Sync checkboxes to internal schema_tools JSON."""
       enabled = [t.name for t in prefs.tool_config_items if t.enabled]
       prefs.schema_tools = json.dumps(enabled)
   ```

4. **Build Function** (`assistant.py`)
   ```python
   def build_openai_tools() -> list:
       """Build OpenAI-style tools from checkbox configuration."""
       # Reads from tool_config_items checkboxes
   ```

5. **UI Drawing** (`preferences.py`)
   ```python
   def _draw_tools_settings(self, layout):
       """Draw checkbox-based tools configuration."""
   ```

6. **Operators**
   - `ASSISTANT_OT_refresh_tool_config` - Refresh from registry
   - `ASSISTANT_OT_toggle_category_tools_prefs` - Category enable/disable
   - `ASSISTANT_OT_set_tool_preset` - Apply presets

### Data Storage

- **Primary**: `tool_config_items` collection (checkboxes)
- **Secondary**: `schema_tools` string property (JSON, for backward compatibility)
- Changes to checkboxes are synced to JSON automatically
- `build_openai_tools()` reads from checkboxes, falls back to JSON

### Initialization

Tools are auto-populated on addon startup:
1. Timer waits 1 second for tool registration
2. Reads from MCP `_TOOLS` registry
3. Populates `tool_config_items` with all registered tools
4. Sets initial enabled state from default `schema_tools` JSON
5. Always ensures `execute_code` is enabled

### Integration Points

- **MCP Tools Registry** (`mcp_tools.py`) - All tools registered here
- **Tool Execution** - Only enabled tools exposed via OpenAI schema
- **Context Optimization** - Fewer tools = smaller payload = more conversation room

## Migration Notes

### From Old Tool Selector Panel

The old tool selector panel in the 3D Viewport sidebar has been **removed**. Its functionality has been fully integrated into the new preference-based system.

**What to do:**
1. Open Preferences → Add-ons → Blender Assistant → Tools Configuration
2. Review your enabled tools (defaults to Lean preset)
3. Adjust as needed using checkboxes or presets
4. Changes persist across sessions

### Backward Compatibility

- The internal `schema_tools` JSON property still exists for compatibility
- Old configurations will be imported on first launch
- Checkboxes become the primary interface

## Recommendations

### General Use
Start with **"Lean (Default)"** preset - provides good balance of capabilities and context efficiency.

### Simple Tasks
Use **"Core Only"** preset when:
- Working on simple modeling tasks
- Token budget is tight
- You want maximum conversation context

### Complex Workflows
Use **"All Tools"** preset when:
- Downloading assets from PolyHaven
- Searching for reference images
- Using web search for documentation
- Working with Sketchfab models

### Custom Scenarios
Manually configure tools when:
- You know exactly which operations you'll need
- You want to exclude specific categories
- You're testing tool functionality

## Best Practices

1. **Start Small** - Begin with Lean or Core presets, add more as needed
2. **Review Enabled Tools** - Periodically check what's enabled
3. **Category Toggles** - Use category enable/disable for quick adjustments
4. **Execute Code Always On** - Don't worry about it being locked - it's essential
5. **Refresh After Updates** - If you add new tool modules, click Refresh

## Troubleshooting

**Q: I don't see any tools in the preferences**
- Click "Refresh Tool List" button
- Wait a moment for tools to register on startup
- Check console for errors

**Q: Changes don't seem to take effect**
- Changes are immediate - restart not needed
- Check "Enabled: X / Y tools" counter updates
- Verify checkbox state matches expectation

**Q: execute_code checkbox is disabled/grayed out**
- This is intentional - it's always enabled for safety
- Cannot be disabled to ensure core functionality

**Q: I want to go back to defaults**
- Click the "Lean (Default)" preset button
- This restores the recommended tool set

**Q: Can I disable all tools?**
- No, `execute_code` is always enabled
- This ensures the assistant can always write and execute Python code

**Q: Where did the old Tool Selection panel go?**
- It was removed in version 2.2.0+
- All functionality is now in Preferences → Tools Configuration
- The new system is more powerful and better integrated

## Performance Notes

### Context Impact

Tool count directly affects the OpenAI tools payload size:

- **Core Only (4 tools)**: ~2-3 KB payload
- **Lean (16 tools)**: ~8-10 KB payload  
- **All Tools (30+ tools)**: ~15-20 KB payload

**Impact**: More tools = less room for conversation history in context window.

### Recommendations by Context Size

- **128K+ context**: Use All Tools without concern
- **32K-128K context**: Use Lean or custom selection
- **8K-32K context**: Consider Core Only for long conversations
- **<8K context**: Definitely use Core Only

## Future Enhancements

Potential improvements being considered:

- Per-session tool overrides (temporary enable/disable)
- Tool usage analytics (which tools are actually being called)
- Smart recommendations based on conversation context
- Tool dependency tracking (automatically enable required tools)
- Export/import tool configurations
- Named custom presets
- Search/filter tools by name

---

**Last Updated**: January 2025  
**Version**: 2.2.0+  
**Status**: Stable
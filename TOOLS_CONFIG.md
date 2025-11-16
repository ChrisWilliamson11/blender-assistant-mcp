# Tools Configuration System

## Overview

The Blender Assistant now uses a **preference-based tools configuration system** instead of hardcoded tool lists. This provides flexibility to customize which tools are exposed to the LLM via the OpenAI-style tools schema, enabling you to optimize context usage and control the assistant's capabilities.

## What Changed

### Previous System
- Tools were hardcoded in `build_openai_tools()` function
- A separate UI-based tool selector existed but wasn't integrated with schema-based tools
- No easy way to switch between tool sets

### New System
- Tools are configured via **Preferences → Tools Configuration (Advanced)**
- Tool list is stored as JSON in preferences
- Three quick presets available: Lean (Default), Core Only, All Tools
- `execute_code` is always included regardless of settings (safety guarantee)

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
   - Includes PolyHaven, stock photos, web search, etc.
   - Higher context usage
   - ~30+ tools depending on what's registered

### Manual Configuration

The `schema_tools` field accepts a JSON array of tool names:

```json
[
  "execute_code",
  "get_scene_info",
  "get_object_info",
  "list_collections",
  "assistant_help"
]
```

#### Adding Tools
1. Check the "Available Tools in Registry" section to see all registered tools
2. Copy the tool name exactly as shown
3. Add it to the JSON array in the `Tools List (JSON)` field
4. Tools are grouped by category (Blender, Selection, Collections, Web, PolyHaven, Stock Photos)

#### Removing Tools
Simply remove the tool name from the JSON array. Note that `execute_code` will always be re-added automatically.

## Tool Categories

Tools in the registry are organized by category:

- **Blender** - Core Blender operations (scene/object info, data access)
- **Selection** - Object selection and activation
- **Collections** - Collection management
- **Vision** - Viewport capture for vision models
- **Web** - Web search capabilities
- **PolyHaven** - Asset search and download from PolyHaven
- **Stock Photos** - Unsplash/Pexels integration
- **Code Execution** - Python code execution (always enabled)
- **Help** - Assistant SDK documentation access

## Technical Details

### For Developers

The tools configuration system consists of:

1. **Preference Property** (`preferences.py`)
   ```python
   schema_tools: bpy.props.StringProperty(
       name="Schema Tools",
       description="JSON list of tool names to expose via OpenAI-style tools schema",
       default='[...]',  # Default lean toolset
   )
   ```

2. **Helper Function** (`assistant.py`)
   ```python
   def get_schema_tools() -> list:
       """Get the list of schema-based tools from preferences."""
   ```

3. **Build Function** (`assistant.py`)
   ```python
   def build_openai_tools() -> list:
       """Build OpenAI-style tools list from MCP registry using schema-based tools."""
   ```

4. **UI Panel** (`preferences.py`)
   ```python
   def _draw_tools_settings(self, layout):
       """Draw tools configuration section."""
   ```

5. **Preset Operator** (`preferences.py`)
   ```python
   class ASSISTANT_OT_set_tool_preset(bpy.types.Operator):
       """Set a predefined tool configuration."""
   ```

### Integration Points

- **MCP Tools Registry** (`mcp_tools.py`) - All tools are registered here with categories
- **Tool Execution** - Only tools in the schema list are exposed to the LLM, but all can be called directly
- **Context Optimization** - Fewer tools = smaller OpenAI tools payload = more room for conversation

### Migration from Old Tool Selector

The old `tool_selector.py` UI panel in the 3D Viewport sidebar is still present but **no longer affects schema-based tools**. The new preference-based system takes precedence for tools exposed to the LLM via the OpenAI schema.

If you were using the old tool selector, your enabled tools may differ from the new default. Review the new preferences to ensure your desired tools are enabled.

## Benefits

1. **Reduced Context Usage** - Lean toolset reduces OpenAI tools payload size
2. **Centralized Configuration** - All settings in one place (preferences)
3. **Persistent** - Settings saved with Blender preferences
4. **Flexible** - Easy to switch between presets or customize
5. **Safe** - `execute_code` always included as safety net

## Recommendations

- **General Use**: Start with "Lean (Default)" preset
- **Simple Tasks**: Use "Core Only" to maximize conversation context
- **Complex Workflows**: Use "All Tools" when you need web search, assets, etc.
- **Custom Scenarios**: Manually configure based on your specific needs

## Future Enhancements

Potential future improvements:

- Per-session tool sets (override preferences temporarily)
- Tool usage statistics (which tools are actually being used)
- Smart tool recommendations based on conversation context
- Tool groups/profiles for different workflows
- UI for visual tool selection (drag-and-drop)

## Troubleshooting

**Q: My tools aren't showing up in the assistant**
- Check that the tool name is spelled exactly as shown in "Available Tools in Registry"
- Verify the JSON is valid (use a JSON validator if needed)
- `execute_code` is always available even if not listed

**Q: I want to go back to defaults**
- Click the "Lean (Default)" preset button

**Q: Can I disable all tools?**
- No, `execute_code` is always enabled for safety and core functionality

**Q: The old Tool Selector panel is still there**
- The old UI is deprecated but not yet removed
- It no longer affects schema-based tools
- Use the new preference-based system instead

---

**Last Updated**: 2025
**Version**: 2.2.0+
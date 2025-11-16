# Changelog

All notable changes to Blender Assistant MCP will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2025-01-XX

### Added
- **Checkbox-based Tools Configuration UI** in Preferences
  - Visual, intuitive interface for enabling/disabling tools
  - Tools organized by category (Blender, Selection, Collections, Web, etc.)
  - Category-level enable/disable buttons for quick bulk changes
  - Live status display showing enabled/total tool counts
- **Quick Tool Presets** for common scenarios
  - "Lean (Default)" - Balanced 16-tool set for general use
  - "Core Only" - Minimal 4-tool set for tight context scenarios
  - "All Tools" - Maximum capabilities with all registered tools
- **Refresh Tool List** operator to update UI from MCP registry
- **Execute Code Lock** - `execute_code` tool is always enabled and visually locked for safety
- Comprehensive `TOOLS_CONFIG.md` documentation with usage guide

### Changed
- **Tools configuration moved to Preferences** from hardcoded lists
  - Previously hardcoded in `build_openai_tools()` function
  - Now user-configurable via checkbox UI
  - Persists across Blender sessions
- **Tool selection is now preferences-based** instead of runtime property
  - Uses `tool_config_items` collection property
  - Automatic sync to internal JSON for compatibility
  - Initialization on addon startup with sensible defaults
- **Updated README.md** with complete tool documentation
  - Added missing `assistant_help` tool
  - Added missing `extract_image_urls` tool
  - Added entire Stock Photos category (search_stock_photos, download_stock_photo)
  - All 30+ tools now accurately documented

### Removed
- **Old Tool Selector Panel** from 3D Viewport sidebar
  - Functionality fully integrated into preferences
  - `tool_selector.py` module no longer registered
  - Simplified UI with single source of truth

### Fixed
- **Removed reference to non-existent `snapshot_info_level` property**
  - Fixed RNA property error in preferences UI
  - Cleaned up generation settings display

### Technical Details
- New `ToolConfigItem` PropertyGroup for checkbox state management
- `_sync_tools_to_json()` function keeps JSON in sync with checkboxes
- `build_openai_tools()` now reads from checkbox configuration
- `get_schema_tools()` helper function for accessing enabled tools
- Auto-initialization timer populates tools from MCP registry on startup
- Backward compatibility maintained via internal `schema_tools` JSON property

### Migration Notes
- Existing installations will auto-import default "Lean" toolset
- Old tool selector panel removed - functionality in Preferences → Tools Configuration
- No user action required - tools default to recommended configuration

---

## [2.1.3] - Previous Release

### Features
- MCP-based tool architecture
- Code execution with assistant_sdk
- RAG database (Blender Manual + API Reference)
- Vision tool for viewport analysis
- Ollama bundled with CUDA support
- PolyHaven integration
- Stock photo APIs (Unsplash, Pexels)
- Sketchfab integration
- Web search capabilities

---

## Future Roadmap

- Per-session tool overrides
- Tool usage analytics
- Smart tool recommendations based on context
- Tool dependency tracking
- Named custom presets
- Export/import tool configurations
- Search/filter tools by name

---

**Note**: For detailed tool configuration documentation, see [TOOLS_CONFIG.md](TOOLS_CONFIG.md)
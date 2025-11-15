"""Tool selector UI for enabling/disabling tools by category."""

import bpy
from . import mcp_tools


# Property group for individual tool
class ToolItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Tool Name")
    enabled: bpy.props.BoolProperty(name="Enabled", default=True)
    category: bpy.props.StringProperty(name="Category")
    description: bpy.props.StringProperty(name="Description")


# Panel for tool selection
class ASSISTANT_PT_tool_selector(bpy.types.Panel):
    bl_label = "Tool Selection"
    bl_idname = "ASSISTANT_PT_tool_selector"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Assistant'
    bl_options = {'DEFAULT_CLOSED'}
    
    def draw(self, context):
        layout = self.layout
        wm = context.window_manager

        if not hasattr(wm, 'assistant_tools'):
            layout.label(text="Tools not initialized", icon='ERROR')
            return

        if len(wm.assistant_tools) == 0:
            layout.label(text="No tools found", icon='INFO')
            return

        # Get tools grouped by category
        tools_by_category = {}
        for tool in wm.assistant_tools:
            cat = tool.category or "Other"
            if cat not in tools_by_category:
                tools_by_category[cat] = []
            tools_by_category[cat].append(tool)

        # Draw category sections (always expanded for simplicity)
        for category in sorted(tools_by_category.keys()):
            tools = tools_by_category[category]

            # Category header
            box = layout.box()
            row = box.row()
            row.label(text=f"{category} ({len(tools)} tools)", icon='TOOL_SETTINGS')

            # Category actions
            op = row.operator("assistant.toggle_category_tools", text="", icon='CHECKBOX_HLT')
            op.category = category
            op.enable = True

            op = row.operator("assistant.toggle_category_tools", text="", icon='CHECKBOX_DEHLT')
            op.category = category
            op.enable = False

            # Tool list
            for tool in sorted(tools, key=lambda t: t.name):
                row = box.row()
                row.prop(tool, "enabled", text=tool.name)


# Operator to toggle all tools in a category
class ASSISTANT_OT_toggle_category_tools(bpy.types.Operator):
    bl_idname = "assistant.toggle_category_tools"
    bl_label = "Toggle Category Tools"
    bl_description = "Enable or disable all tools in this category"
    
    category: bpy.props.StringProperty()
    enable: bpy.props.BoolProperty()
    
    def execute(self, context):
        wm = context.window_manager
        
        for tool in wm.assistant_tools:
            if tool.category == self.category:
                tool.enabled = self.enable
        
        return {'FINISHED'}


# Operator to refresh tool list
class ASSISTANT_OT_refresh_tool_list(bpy.types.Operator):
    bl_idname = "assistant.refresh_tool_list"
    bl_label = "Refresh Tool List"
    bl_description = "Refresh the tool list from registry"

    def execute(self, context):
        refresh_tool_list(context)
        self.report({'INFO'}, f"Refreshed {len(context.window_manager.assistant_tools)} tools")
        return {'FINISHED'}


def refresh_tool_list(context):
    """Refresh the tool list from the MCP registry."""
    wm = context.window_manager
    
    # Clear existing
    wm.assistant_tools.clear()
    
    # Add all registered tools
    for name, tool_data in mcp_tools._TOOLS.items():
        item = wm.assistant_tools.add()
        item.name = name
        item.category = tool_data.get("category", "Other")
        item.description = tool_data.get("description", "")
        item.enabled = True  # Default to enabled
    
    print(f"[Tool Selector] Refreshed {len(wm.assistant_tools)} tools")


def get_enabled_tools():
    """Get list of enabled tool names."""
    try:
        wm = bpy.context.window_manager
        if hasattr(wm, 'assistant_tools'):
            return [tool.name for tool in wm.assistant_tools if tool.enabled]
    except Exception:
        pass
    
    # Fallback: all tools
    return list(mcp_tools._TOOLS.keys())


# Registration
classes = (
    ToolItem,
    ASSISTANT_PT_tool_selector,
    ASSISTANT_OT_toggle_category_tools,
    ASSISTANT_OT_refresh_tool_list,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    # Add collection property to WindowManager
    bpy.types.WindowManager.assistant_tools = bpy.props.CollectionProperty(type=ToolItem)

    # Initialize tool list after a delay (tools need to be registered first)
    def delayed_init():
        try:
            if bpy.context:
                refresh_tool_list(bpy.context)
                print("[Tool Selector] Initialized tool list")
        except Exception as e:
            print(f"[Tool Selector] Delayed init failed: {e}")
        return None

    bpy.app.timers.register(delayed_init, first_interval=1.0)


def unregister():
    # Remove collection property
    if hasattr(bpy.types.WindowManager, 'assistant_tools'):
        del bpy.types.WindowManager.assistant_tools
    
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


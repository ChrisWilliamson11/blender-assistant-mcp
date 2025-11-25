"""UI panels and operators for the automation Assistant."""

import bpy

# Global set to track which code blocks are expanded (message_idx, block_idx)
_expanded_code_blocks = set()

# Global set to track which JSON hierarchy nodes are expanded (message_idx, path)
# Global set to track which JSON hierarchy nodes are expanded (message_idx, path)
_expanded_json_nodes = set()


class AssistantChatMessage(bpy.types.PropertyGroup):
    """A single message in the chat history"""
    role: bpy.props.StringProperty(name="Role", default="User")
    content: bpy.props.StringProperty(name="Content", default="")
    image_data: bpy.props.StringProperty(name="Image Data", default="")  # Base64 encoded
    tool_name: bpy.props.StringProperty(name="Tool Name", default="")


class AssistantChatSession(bpy.types.PropertyGroup):
    """A chat session containing multiple messages"""
    name: bpy.props.StringProperty(name="Name", default="Chat")
    messages: bpy.props.CollectionProperty(type=AssistantChatMessage)
    created_at: bpy.props.StringProperty(name="Created At", default="")


class ASSISTANT_PT_panel(bpy.types.Panel):
    """Main automation Assistant panel in the 3D View sidebar"""

    bl_idname = "ASSISTANT_PT_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Assistant"
    bl_label = "MCP Assistant"

    def draw(self, context):
        layout = self.layout
        prefs = context.preferences.addons[__package__].preferences
        wm = context.window_manager
        col = layout.column(align=True)

        # Model Configuration - collapsible box
        box = col.box()
        row = box.row()
        row.prop(
            prefs,
            "show_ollama_config",
            icon="TRIA_DOWN" if prefs.show_ollama_config else "TRIA_RIGHT",
            icon_only=True,
            emboss=False,
        )
        row.label(text="Model Configuration", icon="PREFERENCES")

        if prefs.show_ollama_config:
            # Model selection
            box.label(text="Active Model:")
            model_row = box.row(align=True)
            model_row.prop(prefs, "model_file", text="")
            model_row.operator("assistant.refresh_models", text="", icon="FILE_REFRESH")

            # Model status indicator
            from pathlib import Path

            from . import ollama_subprocess

            # Simple status - just show load/unload buttons without checking actual status
            # (checking status blocks UI thread)
            status_box = box.box()

            # Ollama server control with status indicator
            server_row = status_box.row(align=True)

            # Check if model is loaded (cached, non-blocking)
            try:
                from . import ollama_subprocess

                ollama = ollama_subprocess.get_ollama()
                # Use cached status if available, don't block UI
                # Store status for use in Chat label below
                is_loaded = getattr(ollama, "_cached_model_loaded", None)
            except Exception:
                is_loaded = None

            server_row.label(text="Ollama Server:")
            server_row.operator("assistant.start_ollama", text="Start", icon="PLAY")
            server_row.operator("assistant.stop_ollama", text="Stop", icon="CANCEL")

            # Model control
            status_row = status_box.row(align=True)
            status_row.label(text="Model Control:")
            status_row.operator("assistant.load_model", text="Load", icon="IMPORT")
            status_row.operator("assistant.unload_model", text="Unload", icon="EXPORT")
            status_row.operator(
                "assistant.refresh_model_status", text="", icon="FILE_REFRESH"
            )

            box.separator()

            # Generation settings
            box.label(text="Generation Settings:", icon="SETTINGS")
            box.prop(prefs, "max_iterations", text="Max Steps")
            box.prop(prefs, "use_rag", text="Enable Documentation")
            if prefs.use_rag:
                box.prop(prefs, "rag_num_results", text="Results")

        col.separator()

        # Chat session management
        box = col.box()
        row = box.row(align=True)

        # Model status indicator - red/green dot
        try:
            from . import ollama_subprocess

            ollama = ollama_subprocess.get_ollama()
            is_loaded = getattr(ollama, "_cached_model_loaded", None)
            if is_loaded is True:
                row.label(text="●", icon="NONE")  # Green dot (will be theme green)
            elif is_loaded is False:
                row.alert = True
                row.label(text="●", icon="NONE")  # Red dot (alert makes it red)
                row.alert = False  # Reset alert for rest of row
            else:
                row.label(text="○", icon="NONE")  # Empty circle (unknown)
        except Exception:
            pass

        # row.label(text="Chat:", icon="DOCUMENTS")

        # Session selector with +/- buttons
        if wm.assistant_chat_sessions:
            # Dropdown with +/- buttons inline
            # row = box.row(align=True)
            row.prop(wm, "assistant_active_chat_enum", text="")
            row.operator("assistant.new_chat", text="", icon="ADD")
            row.operator("assistant.delete_chat", text="", icon="REMOVE")
            row.operator("assistant.rename_chat", text="", icon="GREASEPENCIL")
        else:
            box.operator("assistant.new_chat", text="Create First Chat", icon="ADD")

        col.separator()

        # Input field with paste button on the left
        col.label(text="Message:")
        msg_row = col.row(align=True)
        msg_row.operator("assistant.paste_text", text="", icon="PASTEDOWN")
        msg_row.prop(context.window_manager, "assistant_message", text="")

        # Send and Stop buttons with status indicator
        # Check if assistant is running
        from .assistant import ASSISTANT_OT_send

        is_running = ASSISTANT_OT_send._is_running

        row = col.row(align=True)

        # Send button - disabled when running
        send_row = row.row(align=True)
        send_row.enabled = not is_running
        send_row.operator("assistant.submit_message", text="Send", icon="PLAY")

        # Continue button - quickly ask the model to continue
        cont_row = row.row(align=True)
        cont_row.enabled = not is_running
        cont_row.operator("assistant.continue_chat", text="Continue", icon="FORWARD")

        # Paste Image button - only show if vision model might be available
        # (We'll show it always for now, operator will handle if PIL is missing)
        paste_row = row.row(align=True)
        paste_row.enabled = not is_running
        paste_row.operator("assistant.paste_image", text="", icon="IMAGE_DATA")

        # Show indicator if image is pending
        if wm.assistant_pending_image:
            row.label(text="📎", icon="NONE")

        # Show pending image preview
        if wm.assistant_pending_image:
            pending_box = col.box()
            pending_box.label(
                text="Pending Image (will be sent with next message):",
                icon="IMAGE_DATA",
            )
            if "AssistantPendingImage" in bpy.data.images:
                pending_img = bpy.data.images["AssistantPendingImage"]
                try:
                    pending_img.preview_ensure()
                except Exception:
                    pass
                preview_col = pending_box.column()
                preview_col.scale_y = 6.0
                # template_preview doesn't accept Image IDs; wrap in a Texture datablock when available
                tex_name = "AssistantPendingImage_PreviewTex"
                if hasattr(bpy.data, "textures"):
                    tex_block = bpy.data.textures.get(tex_name)
                    if tex_block is None:
                        try:
                            tex_block = bpy.data.textures.new(tex_name, type="IMAGE")
                        except Exception:
                            tex_block = None
                    if tex_block:
                        tex_block.image = pending_img
                        preview_col.template_preview(tex_block, show_buttons=False)
                    else:
                        pending_box.label(
                            text="Preview unavailable (texture datablock not available)"
                        )
                else:
                    pending_box.label(text="Preview unavailable (no texture support)")

        # Status indicator and Stop button
        if is_running:
            # Show processing indicator
            row.label(text="Processing...", icon="TIME")
            # Stop button - red/prominent when running
            stop_row = row.row(align=True)
            stop_row.alert = True  # Makes it red
            stop_row.operator("assistant.stop", text="", icon="CANCEL")
        else:
            # Stop button - small, disabled when not running
            stop_row = row.row(align=True)
            stop_row.enabled = False
            stop_row.operator("assistant.stop", text="", icon="CANCEL")

        col.separator()

        # Chat interface - show messages from active session
        if wm.assistant_chat_sessions and 0 <= wm.assistant_active_chat_index < len(
            wm.assistant_chat_sessions
        ):
            active_session = wm.assistant_chat_sessions[wm.assistant_active_chat_index]
            col.template_list(
                "ASSISTANT_UL_chat",
                "",
                active_session,
                "messages",
                wm,
                "assistant_chat_message_index",
                rows=8,
            )
            # Copy debug conversation button
            col.operator(
                "assistant.copy_debug_conversation",
                text="Copy Debug Conversation",
                icon="TEXT",
            )

            # Multi-line display of selected message
            if (
                wm.assistant_chat_message_index >= 0
                and wm.assistant_chat_message_index < len(active_session.messages)
            ):
                selected_item = active_session.messages[wm.assistant_chat_message_index]
                box = col.box()

                # Header with role and copy button
                header_row = box.row()
                header_row.label(
                    text=f"{selected_item.role}:",
                    icon="USER" if selected_item.role == "You" else "CONSOLE",
                )
                if selected_item.image_data:
                    header_row.label(text="📎 Image", icon="IMAGE_DATA")
                header_row.operator("assistant.copy_message", text="", icon="COPYDOWN")

                # Show image preview if attached
                if selected_item.image_data:
                    img_box = box.box()
                    img_box.label(text="Attached Image:", icon="IMAGE_DATA")

                    # Display image preview (image created by update callback)
                    img_name = f"AssistantMsg_{wm.assistant_chat_message_index}"
                    if img_name in bpy.data.images:
                        blender_img = bpy.data.images[img_name]

                        # Show a large image preview filling the panel width
                        try:
                            blender_img.preview_ensure()
                        except Exception:
                            pass
                        preview_col = img_box.column()
                        preview_col.scale_y = 6.0
                        # template_preview doesn't accept Image IDs; wrap in a Texture datablock when available
                        tex_name = f"{img_name}_PreviewTex"
                        if hasattr(bpy.data, "textures"):
                            tex_block = bpy.data.textures.get(tex_name)
                            if tex_block is None:
                                try:
                                    tex_block = bpy.data.textures.new(
                                        tex_name, type="IMAGE"
                                    )
                                except Exception:
                                    tex_block = None
                            if tex_block:
                                tex_block.image = blender_img
                                preview_col.template_preview(
                                    tex_block, show_buttons=False
                                )
                            else:
                                img_box.label(
                                    text="Preview unavailable (texture datablock not available)"
                                )
                        else:
                            img_box.label(
                                text="Preview unavailable (no texture support)"
                            )

                    else:
                        # Fallback if image not created yet
                        img_box.label(
                            text=f"({len(selected_item.image_data)} bytes base64)"
                        )
                        img_box.label(text="Loading preview...")

                # Use a scrollable text area for multi-line display with improved formatting
                import json
                import textwrap

                content = selected_item.content

                # Special formatting for Tool messages with JSON
                if selected_item.role == "Tool":
                    try:
                        # Try to parse as JSON and display hierarchically
                        data = json.loads(content)
                        self._draw_json_hierarchy(
                            box, data, wm.assistant_chat_message_index, depth=0, path=""
                        )
                        # Skip normal text display for tool messages with valid JSON
                        lines = []
                    except (json.JSONDecodeError, Exception):
                        # Not JSON or parsing failed, display as normal text
                        lines = content.split("\n")
                else:
                    lines = content.split("\n")

                # Process content line by line, detecting code blocks
                in_code_block = False
                code_block_lines = []
                code_blocks = []  # Store all code blocks found
                line_count = 0
                max_lines = 30

                # First pass - extract all code blocks
                for line in lines:
                    if line.strip().startswith("```"):
                        if in_code_block:
                            # End of code block
                            if code_block_lines:
                                code_blocks.append(list(code_block_lines))
                                code_block_lines = []
                            in_code_block = False
                        else:
                            # Start of code block
                            in_code_block = True
                        continue

                    if in_code_block:
                        code_block_lines.append(line)
                    else:
                        # Regular text - wrap and display
                        if line_count >= max_lines:
                            box.label(
                                text="... (truncated, copy to see full text)",
                                icon="INFO",
                            )
                            break
                        if len(line) > 80:
                            wrapped = textwrap.wrap(line, width=80)
                            for wrapped_line in wrapped:
                                box.label(text=wrapped_line if wrapped_line else " ")
                                line_count += 1
                        else:
                            box.label(text=line if line else " ")
                            line_count += 1

                # Handle unclosed code block
                if in_code_block and code_block_lines:
                    code_blocks.append(list(code_block_lines))

                # Display all code blocks as collapsible (default collapsed)
                for idx, code_lines in enumerate(code_blocks):
                    code_box = box.box()
                    header_row = code_box.row(align=True)

                    # Check if this code block is expanded
                    block_key = (wm.assistant_chat_message_index, idx)
                    is_expanded = block_key in _expanded_code_blocks

                    # Toggle button
                    icon = "TRIA_DOWN" if is_expanded else "TRIA_RIGHT"
                    toggle_op = header_row.operator(
                        "assistant.toggle_code_block",
                        text="",
                        icon=icon,
                        emboss=False,
                    )
                    toggle_op.message_index = wm.assistant_chat_message_index
                    toggle_op.block_index = idx

                    header_row.label(
                        text=f"Code ({len(code_lines)} lines):", icon="CONSOLE"
                    )

                    # Copy button for this code block
                    copy_op = header_row.operator(
                        "assistant.copy_code_block",
                        text="",
                        icon="COPYDOWN",
                        emboss=False,
                    )
                    copy_op.message_index = wm.assistant_chat_message_index
                    copy_op.block_index = idx

                    # Show code content if expanded
                    if is_expanded:
                        code_col = code_box.column(align=True)
                        for code_line in code_lines[:30]:
                            code_col.label(text=code_line if code_line else " ")
                        if len(code_lines) > 30:
                            code_col.label(
                                text="... (code truncated, copy message for full code)"
                            )

                # Action buttons
                footer_row = box.row()
                footer_row.operator(
                    "assistant.copy_message", text="Copy to Clipboard", icon="COPYDOWN"
                )
                footer_row.operator(
                    "assistant.save_preset",
                    text="Save Preset",
                    icon="OUTLINER_DATA_LIGHT",
                )
        else:
            col.label(text="No active chat session")

    def _draw_json_hierarchy(
        self, parent_box, data, message_idx, depth=0, path="", max_depth=5
    ):
        """Draw JSON data as nested boxes with collapsible sections.

        Args:
            parent_box: Parent UI box to draw into
            data: JSON data (dict, list, or primitive)
            message_idx: Index of the message (for tracking expand state)
            depth: Current nesting depth
            path: Current path in the hierarchy (for tracking expand state)
            max_depth: Maximum depth to expand
        """
        if depth > max_depth:
            parent_box.label(text="... (too deeply nested)", icon="INFO")
            return

        wm = bpy.context.window_manager

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, (dict, list)) and value:
                    # Nested structure - collapsible
                    node_key = (message_idx, current_path)
                    is_expanded = node_key in _expanded_json_nodes or depth == 0

                    sub_box = parent_box.box()
                    row = sub_box.row(align=True)

                    # Toggle button
                    icon = "TRIA_DOWN" if is_expanded else "TRIA_RIGHT"
                    toggle_op = row.operator(
                        "assistant.toggle_json_node",
                        text="",
                        icon=icon,
                        emboss=False,
                    )
                    toggle_op.message_index = message_idx
                    toggle_op.node_path = current_path

                    row.label(text=f"{key}:", icon="DISCLOSURE_TRI_DOWN")

                    # Show content if expanded
                    if is_expanded:
                        self._draw_json_hierarchy(
                            sub_box,
                            value,
                            message_idx,
                            depth + 1,
                            current_path,
                            max_depth,
                        )
                else:
                    # Simple value - no box, just plain text
                    if isinstance(value, bool):
                        parent_box.label(
                            text=f"{key}: {str(value).lower()}",
                            icon="CHECKMARK" if value else "PANEL_CLOSE",
                        )
                    elif value is None:
                        parent_box.label(text=f"{key}: null", icon="X")
                    else:
                        parent_box.label(text=f"{key}: {value}")

        elif isinstance(data, list):
            if not data:
                parent_box.label(text="(empty list)")
            elif all(isinstance(item, (str, int, float, bool)) for item in data):
                # Simple list - display items as plain text
                for item in data[:10]:  # Limit to 10 items
                    parent_box.label(text=f"• {item}")
                if len(data) > 10:
                    parent_box.label(
                        text=f"... and {len(data) - 10} more items", icon="INFO"
                    )
            else:
                # Complex list - collapsible items
                for idx, item in enumerate(data[:10]):
                    current_path = f"{path}[{idx}]"
                    node_key = (message_idx, current_path)
                    is_expanded = node_key in _expanded_json_nodes or depth == 0

                    item_box = parent_box.box()
                    row = item_box.row(align=True)

                    # Toggle button
                    icon = "TRIA_DOWN" if is_expanded else "TRIA_RIGHT"
                    toggle_op = row.operator(
                        "assistant.toggle_json_node",
                        text="",
                        icon=icon,
                        emboss=False,
                    )
                    toggle_op.message_index = message_idx
                    toggle_op.node_path = current_path

                    row.label(text=f"[{idx}]", icon="DISCLOSURE_TRI_DOWN")

                    # Show content if expanded
                    if is_expanded:
                        self._draw_json_hierarchy(
                            item_box,
                            item,
                            message_idx,
                            depth + 1,
                            current_path,
                            max_depth,
                        )
                if len(data) > 10:
                    parent_box.label(
                        text=f"... and {len(data) - 10} more items", icon="INFO"
                    )

        else:
            # Primitive value
            parent_box.label(text=str(data))


class ASSISTANT_UL_chat(bpy.types.UIList):
    """UI list for displaying chat messages"""

    bl_idname = "ASSISTANT_UL_chat"

    def draw_item(
        self, context, layout, data, item, icon, active_data, active_propname, index
    ):
        if item is None:
            return

        # Role-based row colors for better readability
        # User (lightest), Assistant (medium), Tool (darkest)
        row = layout.row(align=True)
        if item.role == "You":
            # User messages - lightest (almost white)
            row.emboss = "NONE"
        elif item.role == "Assistant":
            # Assistant messages - medium gray
            row.emboss = "PULLDOWN_MENU"
        elif item.role == "Tool":
            # Tool messages - darkest
            row.emboss = "NORMAL"
        else:
            # System or other - default
            row.emboss = "NONE_OR_STATUS"

        split = row.split(factor=0.18)
        split.label(text=item.role)

        # Show content with image indicator if attached
        content_row = split.row(align=True)
        if item.image_data:
            content_row.label(text="", icon="IMAGE_DATA")
        content_row.label(text=item.content)


class ASSISTANT_OT_new_chat(bpy.types.Operator):
    """Create a new chat session"""
    bl_idname = "assistant.new_chat"
    bl_label = "New Chat"
    bl_options = {"REGISTER"}

    def execute(self, context):
        wm = context.window_manager
        new_session = wm.assistant_chat_sessions.add()
        new_session.name = f"Chat {len(wm.assistant_chat_sessions)}"
        import datetime
        new_session.created_at = datetime.datetime.now().isoformat()
        
        # Switch to new chat
        wm.assistant_active_chat_index = len(wm.assistant_chat_sessions) - 1
        return {"FINISHED"}


class ASSISTANT_OT_delete_chat(bpy.types.Operator):
    """Delete the active chat session"""
    bl_idname = "assistant.delete_chat"
    bl_label = "Delete Chat"
    bl_options = {"REGISTER"}

    def execute(self, context):
        wm = context.window_manager
        idx = wm.assistant_active_chat_index
        if 0 <= idx < len(wm.assistant_chat_sessions):
            wm.assistant_chat_sessions.remove(idx)
            # Adjust index
            if wm.assistant_active_chat_index >= len(wm.assistant_chat_sessions):
                wm.assistant_active_chat_index = len(wm.assistant_chat_sessions) - 1
        return {"FINISHED"}


class ASSISTANT_OT_rename_chat(bpy.types.Operator):
    """Rename the active chat session"""
    bl_idname = "assistant.rename_chat"
    bl_label = "Rename Chat"
    bl_options = {"REGISTER"}
    
    new_name: bpy.props.StringProperty(name="New Name")

    def invoke(self, context, event):
        wm = context.window_manager
        idx = wm.assistant_active_chat_index
        if 0 <= idx < len(wm.assistant_chat_sessions):
            self.new_name = wm.assistant_chat_sessions[idx].name
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        wm = context.window_manager
        idx = wm.assistant_active_chat_index
        if 0 <= idx < len(wm.assistant_chat_sessions):
            wm.assistant_chat_sessions[idx].name = self.new_name
        return {"FINISHED"}


class ASSISTANT_OT_toggle_code_block(bpy.types.Operator):
    """Toggle code block expansion"""

    bl_idname = "assistant.toggle_code_block"
    bl_label = "Toggle Code Block"
    bl_options = {"REGISTER"}

    message_index: bpy.props.IntProperty()
    block_index: bpy.props.IntProperty()

    def execute(self, context):
        global _expanded_code_blocks
        block_key = (self.message_index, self.block_index)
        if block_key in _expanded_code_blocks:
            _expanded_code_blocks.remove(block_key)
        else:
            _expanded_code_blocks.add(block_key)

        # Trigger UI redraw
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        return {"FINISHED"}


class ASSISTANT_OT_toggle_json_node(bpy.types.Operator):
    """Toggle JSON hierarchy node expansion"""

    bl_idname = "assistant.toggle_json_node"
    bl_label = "Toggle JSON Node"
    bl_options = {"REGISTER"}

    message_index: bpy.props.IntProperty()
    node_path: bpy.props.StringProperty()

    def execute(self, context):
        global _expanded_json_nodes
        node_key = (self.message_index, self.node_path)
        if node_key in _expanded_json_nodes:
            _expanded_json_nodes.remove(node_key)
        else:
            _expanded_json_nodes.add(node_key)

        # Trigger UI redraw
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        return {"FINISHED"}


class ASSISTANT_OT_copy_code_block(bpy.types.Operator):
    """Copy code block content to clipboard"""

    bl_idname = "assistant.copy_code_block"
    bl_label = "Copy Code Block"
    bl_options = {"REGISTER"}

    message_index: bpy.props.IntProperty()
    block_index: bpy.props.IntProperty()

    def execute(self, context):
        wm = context.window_manager

        # Get the message
        if not wm.assistant_chat_sessions:
            return {"CANCELLED"}

        active_idx = wm.assistant_active_chat_index
        if active_idx < 0 or active_idx >= len(wm.assistant_chat_sessions):
            return {"CANCELLED"}

        active_session = wm.assistant_chat_sessions[active_idx]
        if self.message_index < 0 or self.message_index >= len(active_session.messages):
            return {"CANCELLED"}

        selected_item = active_session.messages[self.message_index]
        content = selected_item.content
        lines = content.split("\n")

        # Extract the specific code block
        in_code_block = False
        code_blocks = []
        code_block_lines = []

        for line in lines:
            if line.strip().startswith("```"):
                if in_code_block:
                    if code_block_lines:
                        code_blocks.append(code_block_lines)
                        code_block_lines = []
                    in_code_block = False
                else:
                    in_code_block = True
                continue

            if in_code_block:
                code_block_lines.append(line)

        # Handle unclosed code block
        if in_code_block and code_block_lines:
            code_blocks.append(code_block_lines)

        # Get the requested block
        if self.block_index < len(code_blocks):
            code_text = "\n".join(code_blocks[self.block_index])
            context.window_manager.clipboard = code_text
            self.report(
                {"INFO"}, f"Copied {len(code_blocks[self.block_index])} lines of code"
            )
        else:
            self.report({"WARNING"}, "Code block not found")
            return {"CANCELLED"}

        return {"FINISHED"}


class ASSISTANT_OT_paste_text(bpy.types.Operator):
    """Paste text from clipboard into message box"""

    bl_idname = "assistant.paste_text"
    bl_label = "Paste Text"
    bl_options = {"REGISTER"}

    def execute(self, context):
        wm = context.window_manager

        # Get text from clipboard
        clipboard_text = context.window_manager.clipboard

        if not clipboard_text:
            self.report({"WARNING"}, "Clipboard is empty")
            return {"CANCELLED"}

        # Append to existing message or replace if empty
        if wm.assistant_message:
            wm.assistant_message += clipboard_text
        else:
            wm.assistant_message = clipboard_text

        self.report({"INFO"}, f"Pasted {len(clipboard_text)} characters")
        return {"FINISHED"}


class ASSISTANT_OT_paste_image(bpy.types.Operator):
    """Paste image from clipboard (requires PIL/Pillow)"""
    bl_idname = "assistant.paste_image"
    bl_label = "Paste Image"
    bl_options = {"REGISTER"}

    def execute(self, context):
        try:
            from PIL import ImageGrab
            import io
            import base64
            
            # Grab image from clipboard
            img = ImageGrab.grabclipboard()
            
            if img is None:
                self.report({"WARNING"}, "No image in clipboard")
                return {"CANCELLED"}
                
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Save to bytes
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Store in window manager
            context.window_manager.assistant_pending_image = img_str
            self.report({"INFO"}, "Image pasted successfully")
            
            # Trigger redraw
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
                    
            return {"FINISHED"}
            
        except ImportError:
            self.report({"ERROR"}, "PIL/Pillow not installed. Cannot paste images.")
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Failed to paste image: {str(e)}")
            return {"CANCELLED"}


class ASSISTANT_OT_copy_message(bpy.types.Operator):
    """Copy selected message to clipboard"""

    bl_idname = "assistant.copy_message"
    bl_label = "Copy Message"
    bl_options = {"REGISTER"}

    def execute(self, context):
        wm = context.window_manager

        # Get active chat session
        if not wm.assistant_chat_sessions:
            self.report({"WARNING"}, "No chat session")
            return {"CANCELLED"}

        active_idx = wm.assistant_active_chat_index
        if active_idx < 0 or active_idx >= len(wm.assistant_chat_sessions):
            self.report({"WARNING"}, "Invalid chat session")
            return {"CANCELLED"}

        active_session = wm.assistant_chat_sessions[active_idx]

        # Get selected message
        msg_idx = wm.assistant_chat_message_index
        if msg_idx < 0 or msg_idx >= len(active_session.messages):
            self.report({"WARNING"}, "No message selected")
            return {"CANCELLED"}

        selected_item = active_session.messages[msg_idx]

        # Copy to clipboard
        context.window_manager.clipboard = selected_item.content

        self.report(
            {"INFO"}, f"Copied {len(selected_item.content)} characters to clipboard"
        )
        return {"FINISHED"}


class ASSISTANT_OT_copy_debug_conversation(bpy.types.Operator):
    """Copy the entire active conversation plus useful debug context"""

    bl_idname = "assistant.copy_debug_conversation"
    bl_label = "Copy Debug Conversation"
    bl_options = {"REGISTER"}

    def execute(self, context):
        wm = context.window_manager
        if not wm.assistant_chat_sessions:
            self.report({"WARNING"}, "No chat session")
            return {"CANCELLED"}
        idx = wm.assistant_active_chat_index
        if idx < 0 or idx >= len(wm.assistant_chat_sessions):
            self.report({"WARNING"}, "Invalid chat session")
            return {"CANCELLED"}
        session = wm.assistant_chat_sessions[idx]

        # Gather prefs and runtime info
        try:
            prefs = context.preferences.addons[__package__].preferences
        except Exception:
            prefs = None
        try:
            from . import assistant, ollama_subprocess

            enabled_tools = assistant.get_schema_tools()
            tool_count = len(enabled_tools)
            ollama = ollama_subprocess.get_ollama()
            ollama_running = bool(ollama.is_running())
        except Exception:
            enabled_tools = []
            tool_count = 0
            ollama_running = False

        import datetime

        header = []
        header.append("Blender Assistant — Debug Transcript")
        header.append(
            f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        header.append(f"Blender: {getattr(bpy.app, 'version_string', bpy.app.version)}")
        header.append(f"Add-on: {__package__}")
        if prefs:
            header.append(f"Model: {getattr(prefs, 'model_file', '')}")
            header.append(f"Max Steps: {getattr(prefs, 'max_iterations', '')}")
            header.append(f"RAG Enabled: {getattr(prefs, 'use_rag', False)}")
            header.append(
                f"Planning Exploration: {getattr(prefs, 'planning_exploration', False)}"
            )
        header.append(
            f"Enabled Tools: {tool_count} -> {', '.join(enabled_tools)[:200]}"
        )
        header.append(f"Ollama running: {ollama_running}")
        header.append("")
        header.append(f"Chat: {session.name} ({len(session.messages)} messages)")
        header.append("-----")

        lines = []
        lines.extend(header)
        # Messages
        for i, msg in enumerate(session.messages):
            role = msg.role or ""
            name = getattr(msg, "tool_name", "") if role == "Tool" else ""
            prefix = f"[{role}{'/' + name if name else ''}]"
            content = msg.content or ""
            # Keep messages readable; we keep full content since the user asked for all messages
            lines.append(f"{i + 1:02d} {prefix} {content}")
        text = "\n".join(lines)

        context.window_manager.clipboard = text
        self.report(
            {"INFO"}, f"Copied debug transcript ({len(session.messages)} messages)"
        )
        return {"FINISHED"}


# Preset storage helpers
def _assistant_preset_file_path():
    import os

    import bpy

    cfg = bpy.utils.user_resource("CONFIG") or bpy.app.tempdir
    return os.path.join(cfg, "blender_assistant_presets.json")


def _assistant_load_presets():
    import json
    import os

    path = _assistant_preset_file_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                # normalize
                return [
                    p
                    for p in data
                    if isinstance(p, dict) and "name" in p and "text" in p
                ]
    except Exception:
        pass
    return []


def _assistant_save_presets(items):
    import json
    import os

    path = _assistant_preset_file_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


class ASSISTANT_OT_save_preset(bpy.types.Operator):
    """Save the selected message as a preset"""

    bl_idname = "assistant.save_preset"
    bl_label = "Save Preset"
    bl_options = {"REGISTER"}

    preset_name: bpy.props.StringProperty(name="Preset Name", default="")

    def invoke(self, context, event):
        # Suggest a default name from selected content
        wm = context.window_manager
        default = ""
        try:
            if wm.assistant_chat_sessions and 0 <= wm.assistant_active_chat_index < len(
                wm.assistant_chat_sessions
            ):
                sess = wm.assistant_chat_sessions[wm.assistant_active_chat_index]
                idx = wm.assistant_chat_message_index
                if 0 <= idx < len(sess.messages):
                    default = (sess.messages[idx].content or "").strip()
        except Exception:
            default = ""
        default = (default[:24] + "…") if len(default) > 24 else default
        if not default:
            default = "Preset"
        self.preset_name = default
        return context.window_manager.invoke_props_dialog(self)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "preset_name")

    def execute(self, context):
        wm = context.window_manager
        # Get selected message content
        if not wm.assistant_chat_sessions:
            self.report({"WARNING"}, "No chat session")
            return {"CANCELLED"}
        idx = wm.assistant_active_chat_index
        if idx < 0 or idx >= len(wm.assistant_chat_sessions):
            self.report({"WARNING"}, "Invalid chat session")
            return {"CANCELLED"}
        sess = wm.assistant_chat_sessions[idx]
        msg_idx = wm.assistant_chat_message_index
        content = ""
        if 0 <= msg_idx < len(sess.messages):
            content = sess.messages[msg_idx].content or ""
        # Fallback to current input if no selection
        if not content:
            content = wm.assistant_message or ""
        if not content.strip():
            self.report({"WARNING"}, "Nothing to save (empty content)")
            return {"CANCELLED"}
        name = (self.preset_name or "").strip()
        if not name:
            self.report({"WARNING"}, "Preset name is required")
            return {"CANCELLED"}

        items = _assistant_load_presets()
        # Overwrite if same name exists
        updated = False
        for it in items:
            if it.get("name") == name:
                it["text"] = content
                updated = True
                break
        if not updated:
            items.append({"name": name, "text": content})

        ok = _assistant_save_presets(items)
        if ok:
            # Force UI refresh
            for area in context.screen.areas:
                if area.type in {"VIEW_3D", "PREFERENCES"}:
                    area.tag_redraw()
            self.report({"INFO"}, f"Saved preset: {name}")
            return {"FINISHED"}
        else:
            self.report({"ERROR"}, "Failed to save preset file")
            return {"CANCELLED"}


class ASSISTANT_OT_run_preset(bpy.types.Operator):
    """Run a preset (Alt+Click to delete)"""

    bl_idname = "assistant.run_preset"
    bl_label = "Run Preset"
    bl_options = {"REGISTER"}

    preset_name: bpy.props.StringProperty(name="Preset")

    _delete = False

    def invoke(self, context, event):
        self._delete = bool(getattr(event, "alt", False))
        return self.execute(context)

    def execute(self, context):
        name = (self.preset_name or "").strip()
        if not name:
            self.report({"WARNING"}, "No preset name")
            return {"CANCELLED"}
        items = _assistant_load_presets()
        if self._delete:
            new_items = [it for it in items if it.get("name") != name]
            if len(new_items) == len(items):
                self.report({"WARNING"}, f"Preset not found: {name}")
                return {"CANCELLED"}
            if _assistant_save_presets(new_items):
                for area in context.screen.areas:
                    if area.type in {"VIEW_3D", "PREFERENCES"}:
                        area.tag_redraw()
                self.report({"INFO"}, f"Deleted preset: {name}")
                return {"FINISHED"}
            else:
                self.report({"ERROR"}, "Failed to update preset file")
                return {"CANCELLED"}
        # Run preset: set message and send
        text = ""
        for it in items:
            if it.get("name") == name:
                text = it.get("text") or ""
                break
        if not text:
            self.report({"WARNING"}, f"Preset '{name}' has no content")
            return {"CANCELLED"}
        try:
            context.window_manager.assistant_message = text
            bpy.ops.assistant.send("INVOKE_DEFAULT")
            return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"Failed to send preset: {e}")
            return {"CANCELLED"}


class ASSISTANT_PT_presets(bpy.types.Panel):
    bl_label = "Presets"
    bl_idname = "ASSISTANT_PT_presets"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Assistant"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        layout = self.layout
        items = _assistant_load_presets()
        if not items:
            layout.label(text="No presets saved", icon="INFO")
            layout.label(text="Save one from message view")
            return
        col = layout.column(align=True)
        for it in items:
            name = it.get("name") or "(unnamed)"
            row = col.row(align=True)
            op = row.operator("assistant.run_preset", text=name, icon="FORWARD")
            op.preset_name = name
            # Alt+click to delete is handled in the operator's invoke


def update_message_selection(self, context):
    """Called when a message is selected - create image preview if needed."""
    wm = context.window_manager

    # Get selected message
    if not wm.assistant_chat_sessions or wm.assistant_active_chat_index < 0:
        return

    if wm.assistant_active_chat_index >= len(wm.assistant_chat_sessions):
        return

    active_session = wm.assistant_chat_sessions[wm.assistant_active_chat_index]

    if wm.assistant_chat_message_index < 0 or wm.assistant_chat_message_index >= len(
        active_session.messages
    ):
        return

    selected_item = active_session.messages[wm.assistant_chat_message_index]

    # If message has image data, create Blender image for preview
    if selected_item.image_data:
        try:
            from .assistant import base64_to_blender_image

            img_name = f"AssistantMsg_{wm.assistant_chat_message_index}"
            base64_to_blender_image(selected_item.image_data, img_name)
        except Exception as e:
            print(f"[ERROR] Failed to create image preview: {e}")


class ASSISTANT_OT_load_model(bpy.types.Operator):
    """Load the selected model into GPU memory (non-blocking)"""

    bl_idname = "assistant.load_model"
    bl_label = "Load Model"
    bl_description = "Preload model into GPU memory for faster responses"

    _timer = None
    _thread = None
    _error = None
    _success = False
    _model_name = ""

    def modal(self, context, event):
        if event.type == "TIMER":
            # Keep UI responsive and show progress via status bar reports
            for area in context.screen.areas:
                if area.type in {"VIEW_3D", "PREFERENCES"}:
                    area.tag_redraw()
            if self._thread and not self._thread.is_alive():
                self.cancel(context)
                if self._success:
                    self.report(
                        {"INFO"}, f"Model {self._model_name} loaded successfully"
                    )
                    return {"FINISHED"}
                else:
                    self.report(
                        {"ERROR"}, self._error or f"Failed to load {self._model_name}"
                    )
                    return {"CANCELLED"}
        return {"PASS_THROUGH"}

    def execute(self, context):
        import threading

        from . import ollama_subprocess

        prefs = context.preferences.addons[__package__].preferences
        self._model_name = prefs.model_file  # e.g., "qwen2.5-coder:7b"

        ollama = ollama_subprocess.get_ollama()
        if not ollama.is_running():
            self.report({"ERROR"}, "Ollama server is not running")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Loading {self._model_name}...")

        def _bg_load():
            try:
                ok = ollama_subprocess.preload_model(self._model_name, keep_alive="30m")
                self._success = bool(ok)
                if not ok:
                    self._error = "Preload call returned failure"
            except Exception as e:
                self._success = False
                self._error = str(e)

        self._success = False
        self._error = None
        self._thread = threading.Thread(target=_bg_load, daemon=True)
        self._thread.start()

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.2, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        wm = context.window_manager
        if self._timer:
            wm.event_timer_remove(self._timer)
            self._timer = None


class ASSISTANT_OT_unload_model(bpy.types.Operator):
    """Unload the model from GPU memory"""

    bl_idname = "assistant.unload_model"
    bl_label = "Unload Model"
    bl_description = "Unload model from GPU memory to free VRAM"

    def execute(self, context):
        from . import ollama_subprocess

        prefs = context.preferences.addons[__package__].preferences
        model_name = prefs.model_file  # Ollama model name (e.g., "llama3.2:3b")

        ollama = ollama_subprocess.get_ollama()

        if not ollama.is_running():
            self.report({"ERROR"}, "Ollama server is not running")
            return {"CANCELLED"}

        self.report({"INFO"}, f"Unloading {model_name}...")

        success = ollama_subprocess.unload_model(model_name)

        if success:
            self.report({"INFO"}, f"Model {model_name} unloaded")
            # Redraw UI to update status
            for area in context.screen.areas:
                if area.type == "VIEW_3D":
                    area.tag_redraw()
            return {"FINISHED"}
        else:
            self.report({"ERROR"}, f"Failed to unload {model_name}")
            return {"CANCELLED"}


class ASSISTANT_OT_refresh_model_status(bpy.types.Operator):
    """Refresh model loading status"""

    bl_idname = "assistant.refresh_model_status"
    bl_label = "Refresh Status"
    bl_description = "Refresh model loading status (cached for UI performance)"

    def execute(self, context):
        # Check and cache model loaded status (non-blocking on subsequent UI draws)
        try:
            from . import ollama_subprocess

            prefs = context.preferences.addons[__package__].preferences
            model_name = prefs.model_file if hasattr(prefs, "model_file") else None

            if model_name and model_name != "NONE":
                ollama = ollama_subprocess.get_ollama()
                # Check if model is loaded
                is_loaded = ollama.is_model_loaded(model_name)
                # Cache the result for UI to read without blocking
                ollama._cached_model_loaded = is_loaded is not None
            else:
                # No model selected
                from . import ollama_subprocess

                ollama = ollama_subprocess.get_ollama()
                ollama._cached_model_loaded = False
        except Exception as e:
            print(f"[UI] Failed to check model status: {e}")
            # Cache unknown state
            try:
                from . import ollama_subprocess

                ollama = ollama_subprocess.get_ollama()
                ollama._cached_model_loaded = None
            except Exception:
                pass

        # Trigger UI redraw
        for area in context.screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        return {"FINISHED"}


def _update_model_status_cache():
    """Periodic timer to update model status cache (non-blocking)."""
    try:
        if not bpy.context:
            return 5.0  # Retry in 5 seconds

        from . import ollama_subprocess

        prefs = bpy.context.preferences.addons[__package__].preferences
        model_name = prefs.model_file if hasattr(prefs, "model_file") else None

        if model_name and model_name != "NONE":
            ollama = ollama_subprocess.get_ollama()
            is_loaded = ollama.is_model_loaded(model_name)
            ollama._cached_model_loaded = is_loaded is not None
        else:
            ollama = ollama_subprocess.get_ollama()
            ollama._cached_model_loaded = False
    except Exception:
        # Silently fail, cache remains at previous value
        pass

    return 5.0  # Check again in 5 seconds


def update_message_selection(self, context):
    """Callback when message selection changes"""
    # Trigger redraw of the panel to show details of selected message
    for area in context.screen.areas:
        if area.type == "VIEW_3D":
            area.tag_redraw()


def get_chat_sessions_enum(self, context):
    """Callback to populate chat session dropdown"""
    items = []
    wm = context.window_manager
    if not wm.assistant_chat_sessions:
        return []
    
    for i, session in enumerate(wm.assistant_chat_sessions):
        name = session.name if session.name else f"Chat {i+1}"
        items.append((str(i), name, f"Switch to {name}"))
    return items


def update_active_chat(self, context):
    """Callback when active chat changes"""
    wm = context.window_manager
    try:
        idx = int(wm.assistant_active_chat_enum)
        wm.assistant_active_chat_index = idx
        # Scroll to bottom of new chat
        if 0 <= idx < len(wm.assistant_chat_sessions):
            wm.assistant_chat_message_index = len(wm.assistant_chat_sessions[idx].messages) - 1
    except ValueError:
        pass


def register():
    # Register window manager properties
    bpy.utils.register_class(AssistantChatMessage)
    bpy.utils.register_class(AssistantChatSession)
    
    bpy.types.WindowManager.assistant_chat_sessions = bpy.props.CollectionProperty(
        type=AssistantChatSession
    )
    bpy.types.WindowManager.assistant_active_chat_index = bpy.props.IntProperty(
        default=-1, update=update_message_selection
    )
    bpy.types.WindowManager.assistant_active_chat_enum = bpy.props.EnumProperty(
        items=get_chat_sessions_enum,
        name="Chat Session",
        description="Select active chat session",
        update=update_active_chat
    )
    bpy.types.WindowManager.assistant_message = bpy.props.StringProperty(
        name="Message", default=""
    )
    bpy.types.WindowManager.assistant_pending_image = bpy.props.StringProperty(
        name="Pending Image", default=""
    )
    bpy.types.WindowManager.assistant_chat_message_index = bpy.props.IntProperty(
        default=-1, update=update_message_selection
    )

    # Register classes
    bpy.utils.register_class(ASSISTANT_UL_chat)
    bpy.utils.register_class(ASSISTANT_OT_new_chat)
    bpy.utils.register_class(ASSISTANT_OT_delete_chat)
    bpy.utils.register_class(ASSISTANT_OT_rename_chat)
    bpy.utils.register_class(ASSISTANT_OT_toggle_code_block)
    bpy.utils.register_class(ASSISTANT_OT_toggle_json_node)
    bpy.utils.register_class(ASSISTANT_OT_copy_code_block)
    bpy.utils.register_class(ASSISTANT_OT_paste_text)
    bpy.utils.register_class(ASSISTANT_OT_paste_image)
    bpy.utils.register_class(ASSISTANT_OT_copy_message)
    bpy.utils.register_class(ASSISTANT_OT_copy_debug_conversation)
    bpy.utils.register_class(ASSISTANT_OT_save_preset)
    bpy.utils.register_class(ASSISTANT_OT_run_preset)
    bpy.utils.register_class(ASSISTANT_OT_load_model)
    bpy.utils.register_class(ASSISTANT_OT_unload_model)
    bpy.utils.register_class(ASSISTANT_OT_refresh_model_status)
    bpy.utils.register_class(ASSISTANT_PT_panel)
    bpy.utils.register_class(ASSISTANT_PT_presets)

    # Start periodic model status cache update (every 5 seconds)
    # Initialize immediately on startup
    if not bpy.app.timers.is_registered(_update_model_status_cache):
        _update_model_status_cache()  # Call once immediately
        bpy.app.timers.register(_update_model_status_cache, first_interval=5.0)


def unregister():
    """Unregister UI panel and operators"""
    # Stop model status update timer
    if bpy.app.timers.is_registered(_update_model_status_cache):
        bpy.app.timers.unregister(_update_model_status_cache)

    # Unregister in reverse order
    bpy.utils.unregister_class(ASSISTANT_PT_presets)
    bpy.utils.unregister_class(ASSISTANT_PT_panel)
    bpy.utils.unregister_class(ASSISTANT_OT_refresh_model_status)
    bpy.utils.unregister_class(ASSISTANT_OT_unload_model)
    bpy.utils.unregister_class(ASSISTANT_OT_load_model)
    bpy.utils.unregister_class(ASSISTANT_OT_copy_debug_conversation)
    bpy.utils.unregister_class(ASSISTANT_OT_run_preset)
    bpy.utils.unregister_class(ASSISTANT_OT_save_preset)
    bpy.utils.unregister_class(ASSISTANT_OT_copy_message)
    bpy.utils.unregister_class(ASSISTANT_OT_paste_image)
    bpy.utils.unregister_class(ASSISTANT_OT_paste_text)
    bpy.utils.unregister_class(ASSISTANT_OT_copy_code_block)
    bpy.utils.unregister_class(ASSISTANT_OT_toggle_json_node)
    bpy.utils.unregister_class(ASSISTANT_OT_toggle_code_block)
    bpy.utils.unregister_class(ASSISTANT_UL_chat)

    # Delete window manager properties
    del bpy.types.WindowManager.assistant_chat_message_index

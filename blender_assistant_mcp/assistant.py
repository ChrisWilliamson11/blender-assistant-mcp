"""Automation Assistant operator with agentic loop.

This module provides the main automation assistant that uses Ollama
to run models locally with full GPU acceleration.
"""

import json
import threading
import bpy
from . import mcp_tools
from .core import AssistantSession
from .tool_manager import ToolManager
from . import ollama_adapter as llama_manager

# Global session state
_session: AssistantSession = None
_stop_requested = False

def get_session(context) -> AssistantSession:
    """Get or create the global assistant session."""
    global _session
    if _session is None:
        prefs = context.preferences.addons[__package__].preferences
        model_name = getattr(prefs, "model_file", "qwen2.5-coder:7b")
        _session = AssistantSession(model_name, ToolManager())
    return _session

class ASSISTANT_OT_stop(bpy.types.Operator):
    """Stop the current assistant operation"""
    bl_idname = "assistant.stop"
    bl_label = "Stop"
    bl_options = {"REGISTER"}

    def execute(self, context):
        global _stop_requested
        _stop_requested = True
        self.report({"INFO"}, "Stop requested - will cancel after current step")
        return {"FINISHED"}

class ASSISTANT_OT_continue_chat(bpy.types.Operator):
    """Ask the assistant to continue generating"""
    bl_idname = "assistant.continue_chat"
    bl_label = "Continue"
    bl_options = {"REGISTER"}

    def execute(self, context):
        # Set a standard continue message
        bpy.ops.assistant.send(message="Please continue.")
        return {"FINISHED"}

class ASSISTANT_OT_send(bpy.types.Operator):
    """Send message to assistant (runs in background, UI stays responsive)"""
    bl_idname = "assistant.send"
    bl_label = "Send"
    bl_options = {"REGISTER"}

    message: bpy.props.StringProperty(name="Message", default="")
    
    _timer = None
    _thread = None
    _response = None
    _error = None
    _is_running = False

    def _add_message(self, role, content, tool_name=None, images=None):
        """Add message to UI chat history."""
        wm = bpy.context.window_manager
        
        # Ensure we have an active session
        if not wm.assistant_chat_sessions:
            if not wm.assistant_chat_sessions:
                # Create default session if none exists
                bpy.ops.assistant.new_chat()
                
        # Get active session
        if wm.assistant_active_chat_index < 0 or wm.assistant_active_chat_index >= len(wm.assistant_chat_sessions):
             wm.assistant_active_chat_index = 0
             
        active_session = wm.assistant_chat_sessions[wm.assistant_active_chat_index]
        
        # Add message to UI collection
        item = active_session.messages.add()
        item.role = role
        item.content = content
        if images and len(images) > 0:
            item.image_data = images[0] # UI only supports showing one image for now
            
        if tool_name:
            # Store tool name in the message item if property exists, 
            # otherwise it's just part of the history.
            # For now, we rely on the role being "Tool" and content being the result.
            pass
            
        # Scroll to bottom
        wm.assistant_chat_message_index = len(active_session.messages) - 1
        
        # Force redraw
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

    def _http_worker(self, model_name, messages, system_prompt, tools):
        """Background thread for LLM request."""
        try:
            # Prepend system prompt
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            
            response = llama_manager.chat_completion(
                model_path=model_name,
                messages=full_messages,
                tools=tools,
                temperature=0.1 # Low temp for tool use
            )
            self._response = response
        except Exception as e:
            self._error = str(e)

    def execute(self, context):
        global _stop_requested
        _stop_requested = False
        
        # Set running flag
        ASSISTANT_OT_send._is_running = True
        
        session = get_session(context)
        
        # Ensure watcher is initialized
        if not session.scene_watcher._initialized:
            session.scene_watcher.capture_state()
            
        # Check for message in WindowManager if not provided in operator
        wm = context.window_manager
        if not self.message and wm.assistant_message:
            self.message = wm.assistant_message
            wm.assistant_message = "" # Clear input
            
        # Check for pending image
        images = []
        if wm.assistant_pending_image:
            images.append(wm.assistant_pending_image)
            wm.assistant_pending_image = "" # Clear pending image
            
        # Add user message
        if self.message:
            session.add_message("user", self.message, images=images)
            self._add_message("User", self.message, images=images)
        else:
            # No message to send
            self.report({"WARNING"}, "Please enter a message")
            ASSISTANT_OT_send._is_running = False
            return {"CANCELLED"}
            
        # Start the loop
        self._start_step(context, session)
        
        context.window_manager.modal_handler_add(self)
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        return {"RUNNING_MODAL"}

    def _start_step(self, context, session):
        """Start an LLM request."""
        prefs = context.preferences.addons[__package__].preferences
        model_name = getattr(prefs, "model_file", "qwen2.5-coder:7b")
        
        # Prepare tools for OpenAI format
        tools = session.tool_manager.get_openai_tools(session.enabled_tools)
        
        self._thread = threading.Thread(
            target=self._http_worker,
            args=(model_name, session.history, session.get_system_prompt(), tools),
            daemon=True
        )
        self._thread.start()
        session.state = "THINKING"

    def modal(self, context, event):
        global _stop_requested
        session = get_session(context)

        if _stop_requested:
            self._add_message("System", "Stopped by user.")
            return self._finish(context)

        if event.type == 'TIMER':
            if session.state == "THINKING":
                if not self._thread.is_alive():
                    # Request done
                    if self._error:
                        self._add_message("Error", self._error)
                        return self._finish(context)
                    
                    if self._response:
                        # Process response
                        tool_calls = session.process_response(self._response)
                        
                        # Add assistant message to UI
                        msg = self._response.get("message", {})
                        content = msg.get("content", "")
                        if content:
                            self._add_message("Assistant", content)
                            
                        if tool_calls:
                            session.tool_queue = tool_calls
                            session.state = "EXECUTING"
                            # Immediate execution of first tool?
                            # Let's wait for next timer tick to keep UI responsive
                        else:
                            # No tools, we are done
                            return self._finish(context)
                    else:
                        self._add_message("Error", "No response from model")
                        return self._finish(context)

            elif session.state == "EXECUTING":
                if session.tool_queue:
                    # Execute one tool
                    result = session.execute_next_tool()
                    
                    # Show result
                    last_msg = session.history[-1]
                    self._add_message("Tool", last_msg["content"], tool_name=last_msg.get("name"))
                    
                    # Continue executing?
                    if not session.tool_queue:
                        # All tools done, feed back to LLM?
                        # For now, let's auto-continue to let LLM see results
                        self._start_step(context, session)
                else:
                    # Queue empty (shouldn't happen in EXECUTING state unless transitioned)
                    session.state = "IDLE"
                    
                    # Update scene watcher baseline so we don't report our own changes as user changes next time
                    session.scene_watcher.capture_state()
                    
                    return self._finish(context)

        return {"PASS_THROUGH"}

    def _finish(self, context):
        ASSISTANT_OT_send._is_running = False
        if self._timer:
            context.window_manager.event_timer_remove(self._timer)
        return {"FINISHED"}

def get_schema_tools():
    """Get the list of enabled tools for debug/UI purposes."""
    # Create a temporary session or tool manager to get default tools
    # Ideally we should get this from the active session if it exists
    if _session:
        return list(_session.enabled_tools)
    else:
        # Fallback to defaults
        tm = ToolManager()
        return list(tm.default_tools)

def register():
    bpy.utils.register_class(ASSISTANT_OT_stop)
    bpy.utils.register_class(ASSISTANT_OT_continue_chat)
    bpy.utils.register_class(ASSISTANT_OT_send)

def unregister():
    bpy.utils.unregister_class(ASSISTANT_OT_stop)
    bpy.utils.unregister_class(ASSISTANT_OT_continue_chat)
    bpy.utils.unregister_class(ASSISTANT_OT_send)

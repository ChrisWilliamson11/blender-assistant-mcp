"""Automation Assistant operator with agentic loop.

This module provides the main automation assistant that uses Ollama
to run models locally with full GPU acceleration.
"""

import json
import threading
import bpy
from .tools import tool_registry
from .core import AssistantSession
from .tool_manager import ToolManager
from . import ollama_adapter as llama_manager
from . import session_manager
from .session_manager import get_session

# Global session state delegated to session_manager
_stop_requested = False

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

class ASSISTANT_OT_submit_message(bpy.types.Operator):
    """Submit message from UI to assistant"""
    bl_idname = "assistant.submit_message"
    bl_label = "Submit Message"
    bl_options = {"REGISTER"}

    def execute(self, context):
        wm = context.window_manager
        msg = wm.assistant_message
        img = wm.assistant_pending_image
        
        if not msg and not img:
            self.report({"WARNING"}, "Please enter a message")
            return {"CANCELLED"}
            
        # Clear UI
        wm.assistant_message = ""
        wm.assistant_pending_image = ""
        
        # Call worker
        bpy.ops.assistant.send(message=msg, image_data=img)
        return {"FINISHED"}

class ASSISTANT_OT_send(bpy.types.Operator):
    """Send message to assistant (runs in background, UI stays responsive)"""
    bl_idname = "assistant.send"
    bl_label = "Send"
    bl_options = {"REGISTER"}

    message: bpy.props.StringProperty(name="Message", default="")
    image_data: bpy.props.StringProperty(name="Image Data", default="")
    
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
            
        # Scroll to bottom only for primary conversation items
        if role in {"You", "System", "Assistant"}:
            wm.assistant_chat_message_index = len(active_session.messages) - 1
        
        # Force redraw
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()

    def _http_worker(self, model_name, messages, system_prompt, tools, debug_mode=False, keep_alive="5m"):
        """Background thread for LLM request."""
        try:
            # Prepend system prompt
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            
            response = llama_manager.chat_completion(
                model_path=model_name,
                messages=full_messages,
                tools=tools,
                temperature=0.1, # Low temp for tool use
                max_tokens=8192, # Allow space for long chain-of-thought
                debug_mode=debug_mode,
                keep_alive=keep_alive
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
        
        if session is None:
            # No session exists, try to create one
            bpy.ops.assistant.new_chat()
            session = get_session(context)
            
            if session is None:
                self.report({"ERROR"}, "Failed to create assistant session")
                ASSISTANT_OT_send._is_running = False
                return {"CANCELLED"}
        
        # Ensure watcher is initialized
        if not session.scene_watcher._initialized:
            session.scene_watcher.capture_state()
            
        # Prepare images list
        images = []
        if self.image_data:
            images.append(self.image_data)
            
        # Add user message
        if self.message or images:
            # Note: We pass message even if empty if there's an image
            session.add_message("user", self.message, images=images)
            self._add_message("User", self.message, images=images)
        else:
            # Should not happen if called via submit_message or continue_chat
            self.report({"WARNING"}, "No message or image to send")
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
        model_name = getattr(prefs, "model_file", "gpt-oss:20b")
        if model_name == "NONE":
            model_name = "gpt-oss:20b"
        debug_mode = getattr(prefs, "debug_mode", False)
        
        
        
        # Get fresh enabled tools from preferences
        enabled_tools = session.tool_manager.get_enabled_tools_for_role(
            "MANAGER", preferences=prefs
        )
        
        if debug_mode:
             print(f"[Debug] Session Enabled Tools: {enabled_tools}")
        
        # Build system prompt with current tools
        system_prompt = session.get_system_prompt(enabled_tools)
        
        # Prepare tools for OpenAI format
        tools = session.tool_manager.get_openai_tools(enabled_tools)
        
        keep_alive = getattr(prefs, "keep_alive", "5m")
        
        self._thread = threading.Thread(
            target=self._http_worker,
            args=(model_name, session.history, system_prompt, tools, debug_mode, keep_alive),
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
            # Service Execution Queue (Main Thread Tasks from Background Agents)
            # This allows background threads to execute Blender API calls (which must be on Main Thread)
            if hasattr(session, "execution_queue"):
                queued_processed = False
                while not session.execution_queue.empty():
                    try:
                        task = session.execution_queue.get_nowait()
                        func, kwargs, result_container, completion_event = task
                        try:
                            # Execute on Main Thread
                            result_container['result'] = func(**kwargs)
                        except Exception as e:
                           result_container['error'] = e
                        finally:
                            # Notify waiting thread
                            completion_event.set()
                            session.execution_queue.task_done()
                            queued_processed = True
                    except:
                        break
                
                if queued_processed:
                    # Sync Python session history to Blender UI Collection
                    # This ensures messages added by Agents (via session.add_message) appear in the UI
                    try:
                        wm = context.window_manager
                        if wm.assistant_chat_sessions and 0 <= wm.assistant_active_chat_index < len(wm.assistant_chat_sessions):
                            session_ui = wm.assistant_chat_sessions[wm.assistant_active_chat_index]
                            
                            # Sync Python session history to Blender UI Collection
                            hist_len = len(session.history)
                            ui_len = len(session_ui.messages)
                            
                            # Detect Desync (Compression or truncated history)
                            if hist_len < ui_len:
                                # History was compressed/truncated. Rebuild UI.
                                session_ui.messages.clear()
                                ui_len = 0
                                
                            # Append new messages
                            if hist_len > ui_len:
                                for i in range(ui_len, hist_len):
                                    msg_data = session.history[i]
                                    new_msg = session_ui.messages.add()
                                    raw_role = msg_data.get("role", "System")
                                    
                                    # Normalize for UI (User Request: "User")
                                    if raw_role == "user" or raw_role == "You":
                                        new_msg.role = "User"
                                    elif raw_role == "assistant":
                                        new_msg.role = "Assistant"
                                    else:
                                        new_msg.role = raw_role
                                        
                                    new_msg.content = msg_data.get("content", "")
                                    
                                # Scroll to bottom on new content
                                wm.assistant_chat_message_index = len(session_ui.messages) - 1
                                
                    except Exception as e:
                        print(f"Failed to sync history to UI: {e}")

                    # Force UI redraw so Agent messages appear immediately
                    for region in context.area.regions:
                        if region.type == 'WINDOW':
                            region.tag_redraw()

            if session.state == "THINKING":
                if not self._thread.is_alive():
                    # Request done
                    if self._error:
                        self._add_message("Error", self._error)
                        return self._finish(context)
                    
                    if self._response:
                        # CRITICAL FIX: Check for adapter errors first
                        if self._response.get("error"):
                             self._add_message("Error", f"LLM Error: {self._response['error']}")
                             return self._finish(context)

                        # Process response
                        try:
                            prefs = context.preferences.addons[__package__].preferences
                            enabled_tools = session.tool_manager.get_enabled_tools_for_role("MANAGER", preferences=prefs)
                        except:
                            enabled_tools = set()

                        tool_calls, thinking = session.process_response(self._response, enabled_tools)
                        
                        if thinking:
                            self._add_message("Thinking", thinking)
                        
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
                            # No tools, we are done?
                            if not content:
                                if thinking:
                                    # Model thought but didn't act or speak.
                                    # This is a "stalled" state.
                                    # We should consult the COMPLETION agent to see if we are actually done.
                                    self._add_message("System", "Verifying completion...")
                                    
                                    # Consult COMPLETION agent
                                    # We need to run this in a thread or just queue it?
                                    # Since we are in the timer, we can't block.
                                    # But spawn_agent is synchronous (calls LLM).
                                    # We should probably queue a "forced" tool call to spawn_agent.
                                    
                                    # Better approach: Inject a system message and restart step
                                    # asking the model to either act or confirm completion.
                                    # OR, we can just assume it's NOT done and ask it to continue.
                                    
                                    # Let's try to use the COMPLETION agent as intended.
                                    # We'll inject a tool call to `spawn_agent` into the queue manually.
                                    from .core import ToolCall
                                    
                                    # Get original user request
                                    user_request = "Unknown request"
                                    # Find the last message from the user
                                    for msg in reversed(session.history):
                                        if msg.get("role") == "user":
                                            user_request = msg.get("content", "Unknown request")
                                            break
                                    
                                    comp_call = ToolCall(
                                        tool="spawn_agent",
                                        args={
                                            "role": "COMPLETION_AGENT",
                                            "query": f"Verify if this request is satisfied: {user_request}"
                                        }
                                    )

                                    session.tool_queue = [comp_call]
                                    session.state = "EXECUTING"
                                    # Return to let timer pick up EXECUTING state
                                    return {"PASS_THROUGH"}
                                else:
                                    # Empty response (No tools, no content, no thinking).
                                    # This usually means the model stopped prematurely or messed up JSON.
                                    # We should KICK it to continue.
                                    self._add_message("System", "Empty response detected. Auto-continuing...")
                                    session.add_message("user", "System: You returned an empty response. Please continue your work or call `finish_task` if done.")
                                    self._start_step(context, session)
                                    return {"PASS_THROUGH"}
                            return self._finish(context)
                    else:
                        self._add_message("Error", "No response from model")
                        return self._finish(context)

            elif session.state == "EXECUTING":
                if session.tool_queue:
                    # Execute one tool
                    result = session.execute_next_tool()
                    
                    # If tool switched us to WAITING state (Async Agent), stop here.
                    if session.state == "WAITING_FOR_AGENT":
                        return {"PASS_THROUGH"}
                    
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
                    # Check if we woke up from an Async Agent with a result
                    last_msg_role = session.history[-1].get("role") if session.history else ""
                    if last_msg_role == "tool":
                        # We have new results from an agent. Continue thinking.
                        self._start_step(context, session)
                    else:
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



class ASSISTANT_OT_view_request_payload(bpy.types.Operator):
    """View the exact system prompt and tools sent to the LLM"""
    bl_idname = "assistant.view_request_payload"
    bl_label = "View Request Payload"
    
    def execute(self, context):
        session = get_session(context)
        if not session:
            self.report({"ERROR"}, "No active session")
            return {"CANCELLED"}
            
        import json
        
        # reconstruct payload - Use ToolManager with fresh prefs
        try:
             prefs = context.preferences.addons[__package__].preferences
        except:
             prefs = None

        enabled_tools = session.tool_manager.get_enabled_tools_for_role("MANAGER", preferences=prefs)
        system_prompt = session.get_system_prompt(enabled_tools)
        tools = session.tool_manager.get_openai_tools(enabled_tools)
        
        # Capture Sub-Agent Prompts (Dry Run)
        try:
             task_data = session.agent_tools.spawn_agent("TASK_AGENT", "DEBUG_QUERY", dry_run=True)
             completion_data = session.agent_tools.spawn_agent("COMPLETION_AGENT", "DEBUG_QUERY", dry_run=True)
        except Exception as e:
             task_data = f"Error: {e}"
             completion_data = f"Error: {e}"
        
        # Build Readable Report
        lines = []
        lines.append("="*60)
        lines.append(" ASSISTANT DEBUG REPORT")
        lines.append("="*60)
        lines.append("")
        
        # MANAGER
        lines.append(">>> MANAGER AGENT <<<")
        lines.append("-" * 30)
        lines.append("SYSTEM PROMPT:")
        lines.append(system_prompt)
        lines.append("")
        lines.append("TOOLS (MCP):")
        lines.append(json.dumps(tools, indent=2))
        lines.append("\n" + "="*60 + "\n")
        
        # SUB-AGENTS
        for role, data in [("TASK_AGENT", task_data), ("COMPLETION_AGENT", completion_data)]:
             lines.append(f">>> {role} <<<")
             lines.append("-" * 30)
             if isinstance(data, dict):
                 lines.append("SYSTEM PROMPT:")
                 lines.append(str(data.get("system_prompt", "")))
                 lines.append("")
                 lines.append("TOOLS (MCP):")
                 lines.append(json.dumps(data.get("tool_schemas", []), indent=2))
             else:
                 lines.append(f"ERROR/RAW: {data}")
             lines.append("\n" + "="*60 + "\n")

        report = "\n".join(lines)
        
        text_name = "Debug_Assistant_Payload.txt"
        text_block = bpy.data.texts.get(text_name)
        if not text_block:
            text_block = bpy.data.texts.new(text_name)
            
        text_block.clear()
        text_block.write(report)
        
        # Switch area to Text Editor if possible, or just report
        self.report({"INFO"}, f"Payload dumped to '{text_name}'")
        return {"FINISHED"}


def register():
    bpy.utils.register_class(ASSISTANT_OT_stop)
    bpy.utils.register_class(ASSISTANT_OT_continue_chat)
    bpy.utils.register_class(ASSISTANT_OT_submit_message)
    bpy.utils.register_class(ASSISTANT_OT_send)
    bpy.utils.register_class(ASSISTANT_OT_view_request_payload)

def unregister():
    bpy.utils.unregister_class(ASSISTANT_OT_stop)
    bpy.utils.unregister_class(ASSISTANT_OT_continue_chat)
    bpy.utils.unregister_class(ASSISTANT_OT_submit_message)
    bpy.utils.unregister_class(ASSISTANT_OT_send)
    bpy.utils.unregister_class(ASSISTANT_OT_view_request_payload)

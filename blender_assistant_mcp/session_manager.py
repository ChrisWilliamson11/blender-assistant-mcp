"""Session Management for Blender Assistant.
Handles global session state and depsgraph updates.
"""

import bpy
from .core import AssistantSession
from .tool_manager import ToolManager
from . import ollama_adapter as llama_manager

# Global session state
_sessions: dict[str, AssistantSession] = {}

def get_session(context) -> AssistantSession:
    """Get or create the assistant session for the active chat tab."""
    global _sessions
    
    if not hasattr(context, "window_manager"):
        return None
        
    wm = context.window_manager
    if not hasattr(wm, "assistant_chat_sessions"):
        return None
        
    # Ensure items exist
    if not wm.assistant_chat_sessions:
        return None

    idx = wm.assistant_active_chat_index
    if idx < 0 or idx >= len(wm.assistant_chat_sessions):
        idx = 0
        
    session_ui = wm.assistant_chat_sessions[idx]
    session_id = session_ui.session_id
    
    # If session_id is empty (legacy/bug), generate one
    if not session_id:
        import uuid
        session_id = str(uuid.uuid4())
        session_ui.session_id = session_id
        
    if session_id not in _sessions:
        # We need to access preferences. Package name resolution:
        try:
            pkg = __name__.split(".")[0]
            prefs = context.preferences.addons[pkg].preferences
            model_name = getattr(prefs, "model_file", "gpt-oss:20b")
            if model_name == "NONE":
                 model_name = "gpt-oss:20b"
        except:
            model_name = "gpt-oss:20b"
            
        _sessions[session_id] = AssistantSession(model_name, ToolManager(), llm_client=llama_manager)
        
    return _sessions[session_id]

def reset_session(session_id: str):
    """Reset a specific session."""
    global _sessions
    if session_id in _sessions:
        del _sessions[session_id]

# --- Depsgraph Handling ---

def on_depsgraph_update(scene, depsgraph):
    """Handler for depsgraph updates to track modified objects."""
    global _sessions
    if not _sessions:
        return
        
    # Collect dirty object names
    dirty_names = set()
    for update in depsgraph.updates:
        # We only care about Objects for deep diffing
        if isinstance(update.id, bpy.types.Object):
            dirty_names.add(update.id.name)
            
    if not dirty_names:
        return
        
    # Notify all active session watchers
    for session in _sessions.values():
        if hasattr(session, "scene_watcher"):
            for name in dirty_names:
                session.scene_watcher.mark_dirty(name)

def register():
    """Register depsgraph handlers."""
    if on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update)

def unregister():
    """Unregister depsgraph handlers."""
    if on_depsgraph_update in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update)

import bpy
from typing import Set, Dict, List, Optional, Tuple, Any

class SceneWatcher:
    """Tracks changes in the Blender scene to provide context to the assistant."""

    def __init__(self):
        self.last_objects: Set[str] = set()
        self.last_materials: Set[str] = set()
        self.last_selection: Set[str] = set()
        self.last_active: Optional[str] = None
        self.last_mode: str = "OBJECT"
        self._initialized = False

    def capture_state(self):
        """Capture the current state of the scene as the baseline."""
        if not bpy.context:
            return

        # Capture objects (names)
        # We use names as unique IDs for simplicity, though UUIDs would be better if Blender had them natively exposed easily
        self.last_objects = {obj.name for obj in bpy.data.objects}
        self.last_materials = {mat.name for mat in bpy.data.materials}
        
        # Capture selection
        self.last_selection = {obj.name for obj in bpy.context.selected_objects}
        
        # Capture active object
        active = bpy.context.active_object
        self.last_active = active.name if active else None
        
        # Capture mode
        self.last_mode = bpy.context.mode
        
        self._initialized = True

    def get_changes(self) -> Dict[str, Any]:
        """Check for changes since the last capture and return a structured dict of changes."""
        if not self._initialized:
            self.capture_state()
            return {}

        if not bpy.context:
            return {}

        changes = {}
        
        # Current state
        current_objects = {obj.name: obj for obj in bpy.data.objects}
        current_object_names = set(current_objects.keys())
        current_selection = {obj.name for obj in bpy.context.selected_objects}
        active = bpy.context.active_object
        current_active = active.name if active else None
        current_mode = bpy.context.mode

        # 1. Object Additions/Deletions
        added_names = current_object_names - self.last_objects
        removed_names = self.last_objects - current_object_names
        
        if added_names:
            # Serialize added objects so LLM knows what they are
            from .tools.blender_tools import _get_object_summary
            changes["added"] = []
            for name in added_names:
                obj = current_objects[name]
                # Use summary to save context
                changes["added"].append(_get_object_summary(obj))

        if removed_names:
            changes["removed"] = list(removed_names)

        # 1b. Material Additions
        current_materials = {mat.name for mat in bpy.data.materials}
        added_materials = current_materials - self.last_materials
        if added_materials:
            changes["materials_added"] = list(added_materials)

        # 2. Selection Changes
        newly_selected = current_selection - self.last_selection
        deselected = self.last_selection - current_selection
        
        # Filter out added objects from "selected" report
        newly_selected_existing = newly_selected - added_names
        
        if newly_selected_existing:
            changes["selected"] = list(newly_selected_existing)
        
        if deselected and not current_selection:
            changes["deselected_all"] = True

        # 3. Active Object Change
        if current_active != self.last_active and current_active not in added_names:
            changes["active"] = current_active

        # 4. Mode Change
        if current_mode != self.last_mode:
            changes["mode"] = current_mode

        return changes

    def consume_changes(self) -> str:
        """Get changes and update the baseline (consume the event)."""
        changes = self.get_changes()
        if changes:
            self.capture_state()
            import json
            return json.dumps(changes)
        return ""

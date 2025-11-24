import bpy
from typing import Set, Dict, List, Optional, Tuple

class SceneWatcher:
    """Tracks changes in the Blender scene to provide context to the assistant."""

    def __init__(self):
        self.last_objects: Set[str] = set()
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
        
        # Capture selection
        self.last_selection = {obj.name for obj in bpy.context.selected_objects}
        
        # Capture active object
        active = bpy.context.active_object
        self.last_active = active.name if active else None
        
        # Capture mode
        self.last_mode = bpy.context.mode
        
        self._initialized = True

    def get_changes(self) -> List[str]:
        """Check for changes since the last capture and return a list of natural language descriptions."""
        if not self._initialized:
            self.capture_state()
            return []

        if not bpy.context:
            return []

        changes = []
        
        # Current state
        current_objects = {obj.name for obj in bpy.data.objects}
        current_selection = {obj.name for obj in bpy.context.selected_objects}
        active = bpy.context.active_object
        current_active = active.name if active else None
        current_mode = bpy.context.mode

        # 1. Object Additions/Deletions
        added = current_objects - self.last_objects
        removed = self.last_objects - current_objects
        
        if added:
            changes.append(f"Objects added: {', '.join(added)}")
        if removed:
            changes.append(f"Objects deleted: {', '.join(removed)}")

        # 2. Selection Changes (only report if objects weren't just added/removed to avoid noise)
        # If I add a cube, it becomes selected. I don't need to say "Cube added" AND "Cube selected".
        # So we filter selection changes to exclude newly added objects.
        
        newly_selected = current_selection - self.last_selection
        deselected = self.last_selection - current_selection
        
        # Filter out added objects from "selected" report
        newly_selected_existing = newly_selected - added
        
        if newly_selected_existing:
            changes.append(f"User selected: {', '.join(newly_selected_existing)}")
        
        # We generally don't report deselection unless EVERYTHING was deselected
        if deselected and not current_selection:
            changes.append("User cleared selection")

        # 3. Active Object Change
        if current_active != self.last_active and current_active not in added:
            if current_active:
                changes.append(f"Active object changed to: {current_active}")
            else:
                changes.append("No active object")

        # 4. Mode Change
        if current_mode != self.last_mode:
            changes.append(f"Mode changed to: {current_mode}")

        return changes

    def consume_changes(self) -> str:
        """Get changes and update the baseline (consume the event)."""
        changes = self.get_changes()
        if changes:
            self.capture_state()
            return " ".join(changes)
        return ""

import bpy
from typing import Set, Dict, List, Optional, Any, Tuple

class SceneWatcher:
    """Tracks deeply the changes in the Blender scene to provide rich feedback."""

    def __init__(self):
        # Maps object_name -> ObjectSignature (dict)
        self.last_state: Dict[str, Dict] = {}
        
        self.last_materials: Set[str] = set()
        self.last_selection: Set[str] = set()
        self.last_active: Optional[str] = None
        self.last_mode: str = "OBJECT"
        self._initialized = False
        
        # Optimization: Only deeply check objects marked as dirty by depsgraph
        self.dirty_object_names: Set[str] = set()

    def mark_dirty(self, obj_name: str):
        """Mark an object as needing deep inspection."""
        self.dirty_object_names.add(obj_name)

    def _get_obj_signature(self, obj: bpy.types.Object) -> Dict[str, Any]:
        """Compute a hashable-like signature for an object to detect changes."""
        # Modifiers: list of (name, type, show_viewport)
        mods = []
        for m in obj.modifiers:
            mods.append((m.name, m.type, m.show_viewport))
            
        # Constraints: list of (name, type, target_name)
        constrs = []
        for c in obj.constraints:
            target = c.target.name if getattr(c, 'target', None) else None
            constrs.append((c.name, c.type, target))

        # Materials: list of material names (or None)
        mats = []
        for slot in obj.material_slots:
            if slot.material:
                mats.append(slot.material.name)
            else:
                mats.append(None)
                
        return {
            "name": obj.name,
            "pointer": obj.as_pointer(), # Track identity
            "type": obj.type,
            "location": tuple(round(x, 4) for x in obj.location),
            "dimensions": tuple(round(x, 4) for x in obj.dimensions),
            "modifiers": tuple(mods),
            "constraints": tuple(constrs),
            "materials": tuple(mats),
            "parent": obj.parent.name if obj.parent else None,
            "collection": obj.users_collection[0].name if obj.users_collection else None
        }

    def capture_state(self):
        """Capture the current state of the scene as the baseline."""
        if not bpy.context:
            return

        # 1. Capture Object Signatures
        self.last_state = {}
        for obj in bpy.data.objects:
            self.last_state[obj.name] = self._get_obj_signature(obj)

        # 2. Capture Global Materials
        self.last_materials = {mat.name for mat in bpy.data.materials}
        
        self.last_mode = bpy.context.mode
        
        # Clear dirty list as we have a fresh baseline
        self.dirty_object_names.clear()
        self._initialized = True

    def get_changes(self) -> Dict[str, Any]:
        """Check for changes since last capture."""
        if not self._initialized:
            self.capture_state()
            return {}

        changes = {}
        
        # Current State
        current_state = {}
        current_objects = {}
        counts = {} # Stats
        
        for obj in bpy.data.objects:
            sig = self._get_obj_signature(obj)
            current_state[obj.name] = sig
            current_objects[obj.name] = obj
            # Stats
            t = obj.type
            counts[t] = counts.get(t, 0) + 1
            
        current_materials = {mat.name for mat in bpy.data.materials}
        current_selection = [obj.name for obj in bpy.context.selected_objects] # List is better for JSON
        active = bpy.context.active_object
        current_active = active.name if active else None
        current_mode = bpy.context.mode

        # A. Object Adds/Removes
        last_names = set(self.last_state.keys())
        curr_names = set(current_state.keys())
        
        added = curr_names - last_names
        removed = last_names - curr_names
        
        if added:
            changes["added"] = []
            for name in added:
                # Provide brief summary
                sig = current_state[name]
                changes["added"].append({
                    "name": name, 
                    "type": sig["type"], 
                    "modifiers": [m[0] for m in sig["modifiers"]]
                })
                
        if removed:
            changes["removed"] = list(removed)

        # B. Object Modifications (Deep Diff)
        candidates = last_names & curr_names
        if self.dirty_object_names:
            candidates = candidates & self.dirty_object_names
            
        replaced = []
        modified = []
        
        for name in candidates:
            old_sig = self.last_state[name]
            new_sig = current_state[name]
            
            # 0. Identity Check (Replacement)
            if old_sig.get("pointer") != new_sig.get("pointer"):
                replaced.append({
                    "name": name,
                    "new_type": new_sig["type"]
                })
                continue 
            
            diff = []
            
            # 1. Modifiers
            if old_sig["modifiers"] != new_sig["modifiers"]:
                diff.append("modifiers_changed")
            
            # 1.5 Constraints
            if old_sig.get("constraints") != new_sig.get("constraints"):
                diff.append("constraints_changed")

            # 2. Materials
            if old_sig["materials"] != new_sig["materials"]:
                diff.append("materials_changed")
                
            # 3. Transform (approx)
            if old_sig["dimensions"] != new_sig["dimensions"]:
                 diff.append("dimensions_changed")
            elif old_sig["location"] != new_sig["location"]: 
                diff.append("moved")

            if diff:
                modified.append(f"{name} ({', '.join(diff)})")
                
        if replaced:
            changes["replaced"] = replaced
            
        if modified:
            changes["modified"] = modified

        # C. Materials
        added_mats = current_materials - self.last_materials
        if added_mats:
            changes["new_materials"] = list(added_mats)

        # D. Selection
        new_sel_set = set(current_selection)
        if new_sel_set != self.last_selection:
             # Only report if it's NOT just the added objects (which are auto-selected)
             real_sel = new_sel_set - added
             if real_sel: changes["selected"] = list(real_sel)

        # E. Active/Mode
        if current_active != self.last_active and current_active:
            changes["active_object"] = current_active
        if current_mode != self.last_mode:
            changes["mode"] = current_mode
            
        # F. CONTEXT REFRESH (Visual Grounding)
        # Always provide a lightweight summary so the agent implies existence
        changes["context"] = {
            "active": current_active,
            "selected": current_selection,
            "stats": counts
        }

        return changes

    def consume_changes(self) -> Dict[str, Any]:
        """Consume and return changes as a dictionary. Resets state."""
        changes = self.get_changes()
        if changes:
            self.capture_state()
            return changes
        return {}

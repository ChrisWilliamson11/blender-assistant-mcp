"""Skills Management System.

Allows agents to learn (persist) and forget (delete) Python functions as "skills".
Skills are saved to the `skills/` directory and dynamically registered as tools.
"""

import os
import sys
import json
import inspect
import importlib.util
from typing import Dict, Any, Optional

import bpy
from . import tool_registry

# Location for persisted skills
SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


class SkillManager:
    """Manages the lifecycle of agent skills."""
    
    def __init__(self):
        self._ensure_skills_dir()
        self.skills = {} # name -> metadata
    
    def _ensure_skills_dir(self):
        if not os.path.exists(SKILLS_DIR):
            os.makedirs(SKILLS_DIR)
            # Create __init__.py so it's a package
            with open(os.path.join(SKILLS_DIR, "__init__.py"), "w") as f:
                f.write("")

    def register_all_skills(self):
        """Scan skills directory and register all valid skills as tools."""
        self.skills = {}
        for filename in os.listdir(SKILLS_DIR):
            if filename.endswith(".json"):
                # Load metadata
                skill_name = filename[:-5]
                self._load_and_register_skill(skill_name)
    
    def _load_and_register_skill(self, skill_name: str):
        """Load skill metadata and code, then register with tool_registry."""
        meta_path = os.path.join(SKILLS_DIR, f"{skill_name}.json")
        py_path = os.path.join(SKILLS_DIR, f"{skill_name}.py")
        
        if not os.path.exists(meta_path) or not os.path.exists(py_path):
            return

        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            
            # Dynamic Import
            spec = importlib.util.spec_from_file_location(f"blender_assistant_mcp.skills.{skill_name}", py_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"blender_assistant_mcp.skills.{skill_name}"] = module
            spec.loader.exec_module(module)
            
            # Find the function
            if not hasattr(module, skill_name):
                print(f"[SkillManager] Error: Function '{skill_name}' not found in {py_path}")
                return

            func = getattr(module, skill_name)
            
            # Register Tool
            tool_registry.register_tool(
                name=skill_name,
                function=func,
                description=metadata.get("description", "Custom agent skill"),
                input_schema=metadata.get("input_schema", {}),
                category="Skills"
            )
            
            self.skills[skill_name] = metadata
            print(f"[SkillManager] Registered skill: {skill_name}")
            
        except Exception as e:
            print(f"[SkillManager] Failed to load skill '{skill_name}': {e}")

    def learn_skill(self, name: str, code: str, description: str, input_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Save a new skill."""
        clean_name = name.lower().replace(" ", "_").strip()
        
        if not clean_name.isidentifier():
             return {"success": False, "error": f"Invalid skill name '{clean_name}'. Must be a valid Python identifier."}

        # 1. Save Python Code
        py_path = os.path.join(SKILLS_DIR, f"{clean_name}.py")
        with open(py_path, "w") as f:
            f.write(code)
            
        # 2. Save Metadata
        meta_path = os.path.join(SKILLS_DIR, f"{clean_name}.json")
        metadata = {
            "name": clean_name,
            "description": description,
            "input_schema": input_schema
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        # 3. Register immediately
        self._load_and_register_skill(clean_name)
        
        # 4. Trigger SDK rebuild so it appears immediately
        import blender_assistant_mcp.assistant_sdk as sdk
        sdk.get_assistant_sdk()._rebuild()
        
        return {"success": True, "message": f"Learned skill '{clean_name}'. It is now available in 'Skills' category."}

    def forget_skill(self, name: str) -> Dict[str, Any]:
        """Delete a skill."""
        clean_name = name.lower().replace(" ", "_").strip()
        
        py_path = os.path.join(SKILLS_DIR, f"{clean_name}.py")
        meta_path = os.path.join(SKILLS_DIR, f"{clean_name}.json")
        
        deleted = False
        if os.path.exists(py_path):
            os.remove(py_path)
            deleted = True
        if os.path.exists(meta_path):
            os.remove(meta_path)
            deleted = True
            
        if deleted:
            # Unregister logic currently not supported by ToolRegistry (no unregister_tool).
            # But we can remove it from SDK namespace on next rebuild.
            import blender_assistant_mcp.assistant_sdk as sdk
            if clean_name in self.skills:
                del self.skills[clean_name]
            
            # Since we can't unregister from registry easily without restart, 
            # we just rebuild accessors. The tool will still be in registry but failing if called?
            # Ideally ToolRegistry needs unregister supported. 
            # For now, we accept it lingers in registry until reload.
            sdk.get_assistant_sdk()._rebuild()
            
            return {"success": True, "message": f"Forgot skill '{clean_name}'."}
        else:
            return {"success": False, "error": f"Skill '{clean_name}' not found."}


# Global Manager
_skill_manager = SkillManager()

# -----------------------------------------------------------------------------
# Tool Wrappers
# -----------------------------------------------------------------------------

def learn_skill(name: str, code: str, description: str, input_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Learn a new skill (Python function) and save it for future use.
    
    The code must define a function with the EXACT same name as the `name` argument.
    The function should be self-contained (imports inside).
    
    Args:
        name: Name of the skill (snake_case, valid python identifier)
        code: Full Python code string defining the function.
        description: Description of what the skill does.
        input_schema: JSON schema for the function arguments.
    """
    return _skill_manager.learn_skill(name, code, description, input_schema)

def forget_skill(name: str) -> Dict[str, Any]:
    """Delete a learned skill.
    
    Args:
        name: Name of the skill to delete.
    """
    return _skill_manager.forget_skill(name)


def register():
    """Register skill management tools and load existing skills."""
    
    # Register Management Tools
    tool_registry.register_tool(
        "learn_skill",
        learn_skill,
        "Learn a new reusable skill (Python function).",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name (snake_case)"},
                "code": {"type": "string", "description": "Python code defining the function"},
                "description": {"type": "string", "description": "What the skill does"},
                "input_schema": {"type": "object", "description": "JSON schema for arguments"}
            },
            "required": ["name", "code", "description", "input_schema"]
        },
        category="Skills"
    )
    
    tool_registry.register_tool(
        "forget_skill",
        forget_skill,
        "Delete a learned skill.",
        {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Skill name"}
            },
            "required": ["name"]
        },
        category="Skills"
    )
    
    # Load User Skills
    _skill_manager.register_all_skills()

def unregister():
    pass # No cleanup needed

#!/usr/bin/env python3
"""
Debug script to generate the system prompt and conversation context exactly as the LLM sees it.
Run this to inspect what context is being sent to the model.
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock

# -----------------------------------------------------------------------------
# MOCKING BLENDER API (bpy)
# -----------------------------------------------------------------------------
# We need to mock bpy before importing the package modules
mock_bpy = MagicMock()
mock_bpy.app.version = (4, 2, 0)
mock_bpy.context.preferences.addons = {}
mock_bpy.context.window_manager = MagicMock()
mock_bpy.data.objects = []
mock_bpy.context.selected_objects = []
mock_bpy.context.active_object = None
mock_bpy.context.mode = "OBJECT"

sys.modules["bpy"] = mock_bpy
sys.modules["bpy.types"] = MagicMock()
sys.modules["bpy.props"] = MagicMock()
sys.modules["bpy.utils"] = MagicMock()
sys.modules["mathutils"] = MagicMock()

# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from blender_assistant_mcp.core import AssistantSession
from blender_assistant_mcp.tool_manager import ToolManager
import blender_assistant_mcp

def main():
    print("="*80)
    print("BLENDER ASSISTANT PROMPT DEBUGGER")
    print("="*80)
    
    # Register tools using the EXACT same list as the extension
    print("\n[0] Registering Tools (using blender_assistant_mcp._modules)...")
    if hasattr(blender_assistant_mcp, "_modules"):
        for module in blender_assistant_mcp._modules:
            if hasattr(module, "register"):
                try:
                    module.register()
                    print(f"  - Registered: {module.__name__}")
                except Exception as e:
                    print(f"  - Failed to register {module.__name__}: {e}")
    else:
        print("ERROR: Could not find _modules in blender_assistant_mcp package!")
        return
    
    # 1. Initialize Session
    print("\n[1] Initializing Session...")
    tool_manager = ToolManager()
    # Mock preferences for tool manager if needed, or rely on defaults
    session = AssistantSession("debug-model", tool_manager)
    
    # 2. Generate System Prompt
    print("\n[2] Generating System Prompt...")
    system_prompt = session.get_system_prompt()
    
    print("-" * 40)
    print("SYSTEM PROMPT START")
    print("-" * 40)
    print(system_prompt)
    print("-" * 40)
    print("SYSTEM PROMPT END")
    print("-" * 40)
    
    # 3. Simulate Conversation Flow
    print("\n[3] Simulating Conversation...")
    
    # User Message
    user_msg = "Create a red cube."
    print(f"\n> User: {user_msg}")
    session.add_message("user", user_msg)
    
    # Assistant Tool Call (Thinking + Tool)
    thinking = "I need to create a cube and then set its color to red."
    tool_call = {"tool": "execute_code", "args": {"code": "import bpy\nbpy.ops.mesh.primitive_cube_add()"}}
    
    print(f"> Assistant (Thinking): {thinking}")
    print(f"> Assistant (Tool Call): {tool_call}")
    
    # Manually append to history as process_response would
    session.history.append({
        "role": "assistant", 
        "content": thinking,
        "tool_calls": [{
            "function": {
                "name": tool_call["tool"],
                "arguments": tool_call["args"]
            }
        }]
    })
    
    # Tool Result
    tool_result = {"success": True, "message": "Created Cube"}
    print(f"> Tool Result: {tool_result}")
    session.add_message("tool", str(tool_result), name=tool_call["tool"])
    
    # 4. Show Full Context for Next Turn
    print("\n[4] Full Context for Next Turn (Messages List):")
    print("-" * 40)
    
    full_messages = [{"role": "system", "content": system_prompt}] + session.history
    
    import json
    print(json.dumps(full_messages, indent=2))
    print("-" * 40)
    
    print("\nDone! This is exactly what the LLM receives.")
    
    # 5. Debug Worker Agent Prompts (Task & Completion)
    print("\n[5] Worker Agent Prompts (Simulated)")
    
    agent_tools = session.agent_tools
    
    for role in ["TASK_AGENT", "COMPLETION"]:
        print("\n" + "="*40)
        print(f"AGENT: {role}")
        print("="*40)
        
        agent = agent_tools.agents[role]
        
        # Simulate logic from consult_specialist
        universe = tool_manager.get_enabled_tools_for_role(role)
        # Mock active prefs (default tools)
        active_mcp_set = tool_manager.get_enabled_tools(None) 
        core_tools = {"execute_code", "assistant_help"}
        
        # Native Tools (Intersection)
        injected_tools = universe.intersection(active_mcp_set.union(core_tools))
        
        # SDK Hints (The rest)
        sdk_hints = tool_manager.get_system_prompt_hints(
            enabled_tools=injected_tools,
            allowed_tools=universe
        )
        
        # Schemas
        from blender_assistant_mcp.tools import tool_registry
        schemas = []
        for t in injected_tools:
            s = tool_registry.get_tool_schema(t)
            if s: schemas.append(s)
            
        tools_text = json.dumps(schemas, indent=2)
        
        simulated_prompt = (
            f"You are the {agent.name}.\n"
            f"{agent.system_prompt}\n\n"
            "(CONTEXT WOULD GO HERE)\n\n"
            f"AVAILABLE TOOLS (Native):\n{tools_text}\n\n"
            f"{sdk_hints}\n\n"
            "Your goal is to solve the user's query efficiently. "
            "Use your tools."
        )
        
        print(simulated_prompt)
        print("-" * 40)

if __name__ == "__main__":
    main()

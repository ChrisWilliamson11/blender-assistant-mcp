"""
Core logic for Blender Assistant.

This module contains the AssistantSession class which manages the conversation state,
and the ResponseParser class which handles LLM output parsing.
"""

import json
import re
import ast
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from .tools import tool_registry
from .tool_manager import ToolManager
from .memory import MemoryManager
from .scene_watcher import SceneWatcher

@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any]

class ResponseParser:
    """Parses LLM responses into structured tool calls."""

    @classmethod
    def parse(cls, response: Dict[str, Any]) -> List[ToolCall]:
        """Parse a raw LLM response dict into a list of ToolCall objects."""
        # 1. Native tool calls (OpenAI/Ollama format)
        tool_calls = cls._parse_native_calls(response)
        if tool_calls:
            return tool_calls

        # 2. Content-based tool calls (Thinking models / text fallback)
        message = response.get("message", {})
        content = message.get("content", "")
        if not content:
            return []

        # Try JSON blocks
        tool_calls = cls._parse_json_blocks(content)
        if tool_calls:
            return tool_calls

        # 3. Fallback: Auto-wrap Python code blocks
        return cls._parse_code_blocks(content)

    @staticmethod
    def _parse_native_calls(response: Dict[str, Any]) -> List[ToolCall]:
        """Extract native tool calls from the response."""
        tool_calls = []
        message = response.get("message", {})
        if not isinstance(message, dict):
            return []

        native_calls = message.get("tool_calls", [])
        for call in native_calls:
            func = call.get("function", {})
            name = func.get("name")
            args = func.get("arguments", {})
            
            # Handle stringified JSON args
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    # Try python literal eval as fallback
                    try:
                        args = ast.literal_eval(args)
                    except:
                        pass
                        
            if name:
                tool_calls.append(ToolCall(tool=name, args=args if isinstance(args, dict) else {}))
        
        return tool_calls

    @staticmethod
    def _parse_json_blocks(content: str) -> List[ToolCall]:
        """Extract tool calls from JSON blocks in markdown."""
        tool_calls = []
        
        # Try to extract JSON blocks from markdown
        json_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        
        # If no markdown blocks, look for raw JSON objects
        if not json_blocks:
            # Simple brace matching for standalone JSON
            matches = re.findall(r"(\{[\s\S]*?\})", content)
            for m in matches:
                if '"name"' in m and '"arguments"' in m:
                    json_blocks.append(m)

        for block in json_blocks:
            try:
                data = json.loads(block)
                # Check for standard tool call format: {"name": "...", "arguments": {...}}
                if "name" in data and "arguments" in data:
                    tool_calls.append(ToolCall(tool=data["name"], args=data["arguments"]))
                # Check for alternative format: {"tool": "...", "args": {...}}
                elif "tool" in data and "args" in data:
                    tool_calls.append(ToolCall(tool=data["tool"], args=data["args"]))
            except json.JSONDecodeError:
                pass
                
        return tool_calls

    @staticmethod
    def _parse_code_blocks(content: str) -> List[ToolCall]:
        """Extract Python code blocks and wrap them in execute_code."""
        tool_calls = []
        code_blocks = re.findall(r"```(?:python)?\s*(.+?)\s*```", content, re.DOTALL | re.IGNORECASE)
        
        for block in code_blocks:
            # Heuristic: Only treat as code execution if it looks like Blender API usage
            if "import bpy" in block or "bpy." in block:
                tool_calls.append(ToolCall(tool="execute_code", args={"code": block}))
                break # Only take the first code block to avoid confusion
                
        return tool_calls

class AssistantSession:
    """Manages the state of a single assistant session."""

    def __init__(self, model_name: str, tool_manager: ToolManager):
        self.model_name = model_name
        self.tool_manager = tool_manager
        self.memory_manager = MemoryManager()
        self.scene_watcher = SceneWatcher()
        self.history: List[Dict[str, str]] = []
        self.tool_queue: List[ToolCall] = []
        self.state = "IDLE" # IDLE, THINKING, EXECUTING, DONE
        self.last_error = None
        
        # Load enabled tools
        self.enabled_tools = self.tool_manager.get_enabled_tools()

    def add_message(self, role: str, content: str, name: str = None, images: List[str] = None):
        """Add a message to history."""
        msg = {"role": role, "content": content}
        if name:
            msg["name"] = name
        if images:
            msg["images"] = images
        self.history.append(msg)
        
        # Prune history if too long (simple sliding window)
        if len(self.history) > 50:
            self.history = self.history[-50:]

    def get_system_prompt(self) -> str:
        """Build the system prompt based on enabled tools."""
        compact_tools = self.tool_manager.get_compact_tool_list(self.enabled_tools)
        sdk_hints = self.tool_manager.get_system_prompt_hints(self.enabled_tools)
        memory_context = self.memory_manager.get_context()
        
        # Check for scene changes
        scene_changes = self.scene_watcher.consume_changes()
        scene_context = ""
        if scene_changes:
            scene_context = f"SCENE UPDATES (Since last turn)\n{scene_changes}\n\n"
        
        return (
            "You are Blender Assistant — control Blender by calling native tools or writing Python code using the Blender API & the Assistant_SDK.\n\n"
            f"{scene_context}"
            "MEMORY\n"
            f"{memory_context}\n\n"
            "BEHAVIOR\n"
            "- **PLAN FIRST**: If a request is complex, briefly plan before executing.\n"
            "- **ACCESS METHODS**: You have two ways to act:\n"
            "  1. **Native Tools**: Call these directly (e.g., `get_scene_info`).\n"
            "  2. **Python Code**: Use `execute_code` to run scripts. You MUST use this for `assistant_sdk.*` methods and raw `bpy` commands.\n"
            "- **FINDING TOOLS**: Do not guess tool names. Use `assistant_help` to find SDK methods, `rag_query` for docs, or `search_memory` for past solutions.\n"
            "- **SCENE AWARENESS**: 'SCENE UPDATES' show a diff of what changed since the last turn (added/deleted objects, selection changes). Use `get_scene_info` for a full snapshot.\n"
            "- **CLEANUP**: Keep the scene organized. Use collections to group new objects.\n"
            "- **VERIFY**: Always verify your actions.\n"
            "- **TEST OVER GUESS**: If unsure about API behavior, write a small test script using `execute_code` instead of speculating.\n"
            "- **LEARN**: Use `remember_learning` to record pitfalls or version quirks.\n\n"
            "TOOLS\n"
            f"{compact_tools}\n"
            f"{sdk_hints}"
        )

    def process_response(self, response: Dict[str, Any]) -> List[ToolCall]:
        """Process a raw response from the LLM."""
        # Extract thinking/content for history
        message = response.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking", "")
        tool_calls = message.get("tool_calls", [])
        
        if thinking:
            # We don't add thinking to history to save context,
            pass
            
        # Parse tools
        calls = ResponseParser.parse(response)
        
        # Add assistant message to history if there is content OR tool calls
        # Note: We store the raw tool_calls in the history for the API to see
        if content or tool_calls:
            # If we have native tool calls, we should include them in the message object
            msg = {"role": "assistant", "content": content}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            self.history.append(msg)
            # Prune history
            if len(self.history) > 50:
                self.history = self.history[-50:]
        
        # Queue valid calls
        valid_calls = []
        for call in calls:
            if call.tool in self.enabled_tools or call.tool == "execute_code":
                valid_calls.append(call)
            else:
                # Tool not enabled/found - Feedback to LLM is critical!
                print(f"[Session] Warning: Tool '{call.tool}' not enabled or unknown.")
                self.add_message(
                    "user", 
                    f"SYSTEM ERROR: Tool '{call.tool}' is not a native tool. You must use the Python SDK via `execute_code` instead. Check `assistant_help` for the correct SDK method signature."
                )
                
        return valid_calls, thinking

    def execute_next_tool(self) -> Dict[str, Any]:
        """Execute the next tool in the queue."""
        if not self.tool_queue:
            return None
            
        call = self.tool_queue.pop(0)
        
        try:
            print(f"[Session] Executing {call.tool} with {call.args}")
            
            # Execute tool
            result = tool_registry.execute_tool(call.tool, call.args)
            
            # Add result to history
            self.add_message("tool", json.dumps(result), name=call.tool)
            
            # Check for scene side-effects immediately
            # This ensures the LLM knows what changed *before* it generates the next step
            changes = self.scene_watcher.consume_changes()
            if changes:
                self.add_message("system", f"SCENE UPDATES: {changes}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing {call.tool}: {str(e)}"
            self.add_message("tool", json.dumps({"error": error_msg}), name=call.tool)
            return {"error": error_msg}

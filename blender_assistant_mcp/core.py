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
from . import mcp_tools
from .tool_manager import ToolManager

@dataclass
class ToolCall:
    tool: str
    args: Dict[str, Any]

class ResponseParser:
    """Parses LLM responses into structured tool calls."""

    @staticmethod
    def parse(response: Dict[str, Any]) -> List[ToolCall]:
        """Parse a raw LLM response dict into a list of ToolCall objects."""
        tool_calls = []
        
        # 1. Check for native tool calls (OpenAI/Ollama format)
        message = response.get("message", {})
        if isinstance(message, dict):
            native_calls = message.get("tool_calls", [])
            if native_calls:
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

        # 2. Check for content-based tool calls (Thinking models / text fallback)
        content = message.get("content", "")
        if not content:
            return []

        # Try to extract JSON blocks from markdown
        json_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        
        # If no markdown blocks, look for raw JSON objects
        if not json_blocks:
            # Simple brace matching for standalone JSON
            # This is a basic heuristic; for complex nesting it might fail but covers most LLM outputs
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

        # 3. Fallback: Auto-wrap Python code blocks into execute_code
        if not tool_calls:
            code_blocks = re.findall(r"```(?:python)?\s*(.+?)\s*```", content, re.DOTALL | re.IGNORECASE)
            for block in code_blocks:
                if "import bpy" in block or "bpy." in block:
                    tool_calls.append(ToolCall(tool="execute_code", args={"code": block}))
                    break # Only take the first code block to avoid confusion

        return tool_calls

class AssistantSession:
    """Manages the state of a single assistant session."""

    def __init__(self, model_name: str, tool_manager: ToolManager):
        self.model_name = model_name
        self.tool_manager = tool_manager
        self.history: List[Dict[str, str]] = []
        self.tool_queue: List[ToolCall] = []
        self.state = "IDLE" # IDLE, THINKING, EXECUTING, DONE
        self.max_iterations = 15
        self.current_iteration = 0
        self.last_error = None
        
        # Load enabled tools
        self.enabled_tools = self.tool_manager.get_enabled_tools()

    def add_message(self, role: str, content: str, name: str = None):
        """Add a message to history."""
        msg = {"role": role, "content": content}
        if name:
            msg["name"] = name
        self.history.append(msg)
        
        # Prune history if too long (simple sliding window)
        if len(self.history) > 50:
            self.history = self.history[-50:]

    def get_system_prompt(self) -> str:
        """Build the system prompt based on enabled tools."""
        compact_tools = self.tool_manager.get_compact_tool_list(self.enabled_tools)
        sdk_hints = self.tool_manager.get_system_prompt_hints(self.enabled_tools)
        
        return (
            "You are Blender Assistant — control Blender by writing Python code or calling native tools.\n\n"
            "BEHAVIOR\n"
            "- Prefer native tools when they map to your task.\n"
            "- Use 'execute_code' for custom logic or complex operations not covered by tools.\n"
            "- Always verify your actions (e.g., check if objects were created).\n"
            "- If a tool fails, try a different approach or use Python code.\n\n"
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
        
        if thinking:
            # We don't add thinking to history to save context, but we could log it
            pass
            
        if content:
            self.add_message("assistant", content)

        # Parse tools
        calls = ResponseParser.parse(response)
        
        # Queue valid calls
        valid_calls = []
        for call in calls:
            if call.tool in self.enabled_tools or call.tool == "execute_code":
                valid_calls.append(call)
            else:
                # Tool not enabled/found
                print(f"[Session] Warning: Tool '{call.tool}' not enabled or unknown.")
                
        return valid_calls

    def execute_next_tool(self) -> Dict[str, Any]:
        """Execute the next tool in the queue."""
        if not self.tool_queue:
            return None
            
        call = self.tool_queue.pop(0)
        
        try:
            print(f"[Session] Executing {call.tool} with {call.args}")
            result = mcp_tools.execute_tool(call.tool, call.args)
            
            # Add result to history
            self.add_message("tool", json.dumps(result), name=call.tool)
            return result
            
        except Exception as e:
            error_msg = f"Error executing {call.tool}: {str(e)}"
            self.add_message("tool", json.dumps({"error": error_msg}), name=call.tool)
            return {"error": error_msg}

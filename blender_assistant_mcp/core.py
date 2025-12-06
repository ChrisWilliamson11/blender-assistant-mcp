"""
Core logic for Blender Assistant.

This module contains the AssistantSession class which manages the conversation state,
and the ResponseParser class which handles LLM output parsing.
"""

import json
import re
import ast
import queue
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from .tools import tool_registry
from .tool_manager import ToolManager
from .memory import MemoryManager
from .scene_watcher import SceneWatcher
from .agent_manager import AgentTools

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
        """Extract tool calls from JSON blocks in markdown or raw text."""
        tool_calls = []
        
        # Helper to extract balanced JSON objects
        def extract_balanced_objects(text: str) -> List[str]:
            objects = []
            brace_count = 0
            start_index = -1
            i = 0
            length = len(text)
            
            while i < length:
                char = text[i]
                
                # Handle strings to ignore braces inside them
                if char == '"':
                    # Check for triple quote
                    if i + 2 < length and text[i:i+3] == '"""':
                        i += 3
                        # Find end of triple quote
                        while i + 2 < length:
                            if text[i:i+3] == '"""' and text[i-1] != '\\':
                                i += 3
                                break
                            i += 1
                        continue
                    else:
                        # Normal string
                        i += 1
                        while i < length:
                            if text[i] == '"' and text[i-1] != '\\':
                                i += 1
                                break
                            i += 1
                        continue
                
                # Count braces
                if char == '{':
                    if brace_count == 0:
                        start_index = i
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and start_index != -1:
                        objects.append(text[start_index:i+1])
                        start_index = -1
                
                i += 1
                
            return objects

        # 1. Try markdown blocks first (they are usually well-formed)
        json_blocks = re.findall(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        
        # 2. If no markdown blocks, use robust extraction
        if not json_blocks:
            json_blocks = extract_balanced_objects(content)

        for block in json_blocks:
            try:
                data = json.loads(block)
                if "name" in data and "arguments" in data:
                    tool_calls.append(ToolCall(tool=data["name"], args=data["arguments"]))
                elif "tool" in data and "args" in data:
                    tool_calls.append(ToolCall(tool=data["tool"], args=data["args"]))
            except json.JSONDecodeError:
                # Fallback: Try parsing as Python literal
                try:
                    data = ast.literal_eval(block)
                    if isinstance(data, dict):
                        if "name" in data and "arguments" in data:
                            tool_calls.append(ToolCall(tool=data["name"], args=data["arguments"]))
                        elif "tool" in data and "args" in data:
                            tool_calls.append(ToolCall(tool=data["tool"], args=data["args"]))
                except:
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

    def __init__(self, model_name: str, tool_manager: ToolManager, llm_client=None):
        self.model_name = model_name
        self.tool_manager = tool_manager
        self.memory_manager = MemoryManager()
        self.scene_watcher = SceneWatcher()
        self.execution_queue = queue.Queue()
        self.agent_tools = AgentTools(
            llm_client=llm_client, 
            tool_manager=tool_manager, 
            memory_manager=self.memory_manager,
            message_callback=self.add_message,
            get_model_name=self.get_model_name_from_prefs,
            execution_queue=self.execution_queue,
            on_agent_finish=self.on_agent_finish,
            scene_watcher=self.scene_watcher,
            session=self
        ) # Initialize AgentTools
        self.history: List[Dict[str, str]] = []
        self.tool_queue: List[ToolCall] = []
        self.state = "IDLE" # IDLE, THINKING, EXECUTING, DONE
        self.last_error = None
        
        # Load enabled tools (MANAGER ROLE)
        # The main assistant is now the MANAGER and cannot EXECUTE code directly.
        # It must delegate to TASK_AGENT.
        self.enabled_tools = self.tool_manager.get_enabled_tools_for_role("MANAGER")

        # Register consult_specialist tool
        tool_registry.register_tool(
            name="consult_specialist",
            func=self.agent_tools.consult_specialist,
            description="Delegate a task to a Worker Agent (TASK_AGENT) or Verify completion (COMPLETION). Returns a JSON with 'thought', 'code', and 'expected_changes'.",
            input_schema={
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "description": "Agent role (TASK_AGENT, COMPLETION)",
                        "enum": ["TASK_AGENT", "COMPLETION"]
                    },
                    "query": {
                        "type": "string",
                        "description": "Specific task description for the agent"
                    }
                },
                "required": ["role", "query"]
            },
            category="System"
        )



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
        mcp_tools = self.tool_manager.get_compact_tool_list(self.enabled_tools)
        sdk_tools = self.tool_manager.get_system_prompt_hints(self.enabled_tools)
        memory_context = self.memory_manager.get_context()
        if not memory_context:
            memory_context = "(No memories yet)"
        
        # Check for scene changes
        scene_changes = self.scene_watcher.consume_changes()
        scene_context = ""
        if scene_changes:
            scene_context = f"SCENE UPDATES (Since last turn)\n{scene_changes}\n\n"
            
        return (
            "You are the Blender Assistant MANAGER. You CANNOT execute code or modify the scene directly.\n"
            "**YOUR GOAL**: Plan and EXECUTE the task by delegating to the `TASK_AGENT`.\n"
            "**CRITICAL**: Do NOT write a text plan. You must ACT using tools immediately.\n"
            "**WORKFLOW**:\n"
            "1.  Call `task_add` to log the first step.\n"
            "2.  IMMEDIATELY call `consult_specialist(role='TASK_AGENT', query='...')` to execute it.\n"
            "3.  Wait for the result, then call `task_complete`.\n"
            "4.  Repeat for the next step.\n"
            "5.  Use `remember_*` only for critical facts.\n\n"
            "**CRITICAL RULES**:\n"
            "- **NO CODE EXECUTION**: You have NO execute_code tool. Delegate ALL coding/blender work to `TASK_AGENT`.\n"
            "- **NO PLAN LISTS**: Do not output markdown checklists. Call `task_add` instead.\n"
            "- **USE TOOLS**: If you need to check something, delegate to `TASK_AGENT` or `COMPLETION`.\n"
            "- **VERIFY**: Use `consult_specialist(role='COMPLETION', ...)` before finishing complex tasks.\n\n"
            "CHAT PARTICIPANTS\n"
            "- **User**: The human manager.\n"
            "- **Manager (You)**: The planner. NO CODE EXECUTION.\n"
            "- **Task Agent**: The worker who executes code/tools.\n"
            "- **Completion Agent**: Verifier.\n\n"
            f"{scene_context}"
            "MEMORY\n"
            f"{memory_context}\n\n"
            "TOOLS (Manager Only)\n"
            f"{mcp_tools}\n"
            "PROTOCOL & BEHAVIORAL RULES\n" 
            f"{self._load_protocol()}"
        )

    def _load_protocol(self) -> str:
        """Load the agentic protocol from protocol.md."""
        try:
            import os
            # Assume protocol.md is in the same directory as this file (core.py)
            protocol_path = os.path.join(os.path.dirname(__file__), "protocol.md")
            if os.path.exists(protocol_path):
                with open(protocol_path, "r", encoding="utf-8") as f:
                    return f.read()
        except Exception as e:
            print(f"[Assistant] Failed to load protocol.md: {e}")
        
        # Fallback if file missing
        return "Follow standard coding best practices."

    def get_model_name_from_prefs(self) -> Optional[str]:
        """Fetch the currently selected model from Blender preferences."""
        import bpy
        try:
            # We need to get the package name, but __package__ might not be available here directly.
            # We can try to guess or use a hardcoded name if imported relatively
            # Or assume standard name if we are consistent.
            # Re-using the logic from __init__ (simplified)
            pkg = __name__.split(".")[0]
            prefs = bpy.context.preferences.addons[pkg].preferences
            return prefs.model_file # Assuming 'model_file' is the property name
        except Exception:
             # Fallback to the one passed in Init if prefs fail, or None
             return self.model_name
             
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
            elif call.tool.startswith("assistant_sdk."):
                # Auto-Translate SDK calls to execute_code
                print(f"[Session] Auto-Translating SDK call '{call.tool}' to execute_code")
                
                # Construct Python code
                # Handle args: if they are simple, pass them. If complex, might be tricky.
                # We assume args are kwargs.
                args_str = ", ".join([f"{k}={repr(v)}" for k, v in call.args.items()])
                code = f"result = {call.tool}({args_str})"
                
                valid_calls.append(ToolCall(tool="execute_code", args={"code": code}))
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
            
            # Check for Async Agent Start (Direct JSON or Wrapped in dict)
            agent_signal = None
            if isinstance(result, str):
                try: 
                    agent_signal = json.loads(result)
                except: pass
            elif isinstance(result, dict) and "result" in result and isinstance(result["result"], str):
                try:
                    agent_signal = json.loads(result["result"])
                except: pass

            if isinstance(agent_signal, dict) and agent_signal.get("type") == "AGENT_STARTED":
                self.state = "WAITING_FOR_AGENT"
                self.add_message("system", f"Agent {agent_signal.get('role')} started in background...")
                return result

            
            # Add result to history (Truncate if too large to prevent context overflow)
            result_str = json.dumps(result)
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "... [Truncated]"
            self.add_message("tool", result_str, name=call.tool)
            
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

    def on_agent_finish(self, result):
        """Callback from background agent thread."""
        # Add result to history
        result_str = json.dumps(result)
        self.add_message("tool", result_str, name="consult_specialist")
        
        # Wake up assistant logic
        # We set state to EXECUTING so the main loop picks it up and triggers the next LLM step
        self.state = "EXECUTING"

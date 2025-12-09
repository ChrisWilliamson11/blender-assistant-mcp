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
        # 1. MCP tool calls (OpenAI/Ollama format)
        tool_calls = cls._parse_MCP_calls(response)
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
    def _parse_MCP_calls(response: Dict[str, Any]) -> List[ToolCall]:
        """Extract MCP tool calls from the response."""
        tool_calls = []
        message = response.get("message", {})
        if not isinstance(message, dict):
            return []

        MCP_calls = message.get("tool_calls", [])
        for call in MCP_calls:
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
        self.full_history: List[Dict[str, str]] = [] # Permanent record for UI
        self.tool_queue: List[ToolCall] = []
        self.state = "IDLE" # IDLE, THINKING, EXECUTING, DONE
        self.last_error = None
        self.current_task_state = ""

    def add_message(self, role: str, content: str, name: str = None, images: List[str] = None, tool_calls: List[Dict] = None, usage: Dict = None):
        """Add a message to history."""
        msg = {"role": role, "content": content}
        if name:
            msg["name"] = name
        if images:
            msg["images"] = images
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if usage:
            msg["usage"] = usage
            
        self.history.append(msg)
        self.full_history.append(msg)
        
        # Trigger Gradient Compression if history gets long
        if len(self.history) > 20:
            self.history = self.memory_manager.compact_history(
                self.history, 
                llm_client=self.agent_tools.llm_client
            )
        # Fallback pruning
        if len(self.history) > 50:
            self.history = self.history[-50:]

    def get_system_prompt(self, enabled_tools: set) -> str:
        """Build the system prompt based on enabled tools."""
        # Get Allowed tools (Universe) to filter hints correctly
        allowed_tools = self.tool_manager.get_allowed_tools_for_role("MANAGER")
        
        # Generate hints for disabled tools (SDK Mode support)
        # Sdk Hints = Allowed - Enabled
        sdk_hints = self.tool_manager.get_system_prompt_hints(enabled_tools, allowed_tools=allowed_tools)
        
        memory_context = self.memory_manager.get_context() or "(No memories yet)"
        
        # Check for scene changes
        raw_changes = self.scene_watcher.consume_changes()
        scene_context = f"Scene Changes (Since last turn):\n{raw_changes}" if raw_changes else "Scene Changes: (no scene changes since last turn)"

        # Lean Protocol Strategy
        include_protocol = len(self.history) <= 1
        protocol_section = self._load_protocol() if include_protocol else "Refer to protocol for behavioral rules."

        return f"""You are the Manager Agent (The "Brain").

        GOAL: Plan, delegate, and track complex Blender tasks. If the task is very simple (a 1-liner) complete it yourself with execute_code & bpy.
        When using python for simple tasks, "Inline Code Execution" is available for efficiency. Feel free to write Python blocks (` ```python ... ``` `) directly in your response to run quick checks or simple actions without formal tool calls.
        Multiple code blocks will be processed one after the other, and the execution environment maintains a persistant state, so you could define a function in one code block then run it later from another for a real free-form thinking/coding experience.
        
        CURRENT TASK STATUS:
        {self.current_task_state if self.current_task_state else "(No active task state)"}

        CONTEXT:
        - Memory: {memory_context}
        - {scene_context}

        - **User**: The human manager.
        - **Manager (You)**: The planner.
        - **Task Agent**: (Role: `TASK_AGENT`) The worker. Use `spawn_agent(role="TASK_AGENT", ...)` for: **Web Search**, **Downloading Images**, **PolyHaven/Sketchfab Assets**, and rigorous Blender operations.
        - **Completion Agent**: (Role: `COMPLETION_AGENT`) The verifier. Use `spawn_agent(role="COMPLETION_AGENT", ...)` to verify your work.

        MCP TOOLS:
        {json.dumps(list(enabled_tools), indent=2)}
        
        {sdk_hints}

        PROTOCOL & BEHAVIORAL RULES
        {protocol_section}"""

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
        """Get the model name from preferences."""
        try:
            import bpy
            prefs = bpy.context.preferences.addons[__package__].preferences
            return prefs.model_file
        except Exception:
            return None

    def process_response(self, response: Dict[str, Any], enabled_tools: set, usage: Dict = None) -> tuple[List[ToolCall], str]:
        """Process a raw response from the LLM."""
        # Extract thinking/content for history
        message = response.get("message", {})
        content = message.get("content", "")
        thinking = message.get("thinking", "")
        tool_calls = message.get("tool_calls", [])
        
        if thinking:
            # Add thinking to history with special role for UI visibility choice
            self.add_message("thinking", thinking, usage=usage)
            
        # Parse tools
        calls = ResponseParser.parse(response)
        
        # Add assistant message to history if there is content OR tool calls
        if content or calls:
            self.add_message(
                "assistant", 
                content, 
                tool_calls=message.get("tool_calls"), # Original MCP calls for history
                usage=usage
            )
            
        valid_calls = []
        for call in calls:
            if call.tool in enabled_tools or call.tool == "execute_code":
                valid_calls.append(call)
            elif call.tool == "python":
                # Handle generic python blocks as execute_code
                code = call.args.get("code", "") or call.args.get("source", "")
                if not code and isinstance(call.args, str):
                    code = call.args
                
                valid_calls.append(ToolCall(tool="execute_code", args={"code": code}))
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
                    f"SYSTEM ERROR: Tool '{call.tool}' is not a MCP tool. You must use the Python SDK via `execute_code` instead. Check `sdk_help` for the correct SDK method signature."
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
        self.add_message("tool", result_str, name="spawn_agent")
        
        # Wake up assistant logic
        # We set state to EXECUTING so the main loop picks it up and triggers the next LLM step
        self.state = "EXECUTING"

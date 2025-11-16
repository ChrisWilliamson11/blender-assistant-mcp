"""Automation Assistant operator with agentic loop.

This module provides the main automation assistant that uses Ollama
to run models locally with full GPU acceleration.
"""

import ast
import json
import os
import threading

import bpy

from . import mcp_tools
from . import (
    ollama_adapter as llama_manager,  # Use Ollama adapter instead of llama-cpp-python
)

# Ollama is always available (subprocess-based)
LLAMA_CPP_AVAILABLE = True


# Limit how many prior chat messages are sent to the LLM per request (context budget)

MAX_HISTORY_MESSAGES = 120


# Meta/planning tools deprecated in favor of verification system
META_TOOL_NAMES = set()


CODE_FIRST_CAP = {
    "execute_code",
    "get_scene_info",
    "get_object_info",
    "list_collections",
    "get_collection_info",
    "create_collection",
    "move_to_collection",
    "set_collection_color",
    "delete_collection",
    "get_selection",
    "get_active",
    "set_selection",
    "set_active",
    "select_by_type",
    "assistant_help",
    "capture_viewport_for_vision",
}


# Back-compat aliases (all modes use the same fixed set)
# API_LEAN_CAP removed (code-first always)


# LEAN_CAP removed (code-first always)


def _model_supports_thinking(model_path: str) -> bool:
    """Return True if the model is expected to support the Ollama 'think' param.
    Conservative: mistral/nemo families often do not support it, so return False.
    """
    name = (model_path or "").lower()
    if "mistral" in name or "nemo" in name:
        return False
    return True


def build_system_prompt() -> str:
    """Build a single, code-first system prompt (no lean modes; fixed minimal tool set) with verification workflow."""
    # Get only enabled tools (Tool Selector may disable tools; always include execute_code)
    from . import tool_selector

    enabled_tools = []
    try:
        enabled_tools = tool_selector.get_enabled_tools() or []

    except Exception:
        enabled_tools = []

    code_first_cap = {
        "execute_code",
        "get_scene_info",
        "get_object_info",
        "list_collections",
        "get_collection_info",
        "create_collection",
        "move_to_collection",
        "set_collection_color",
        "delete_collection",
        "get_selection",
        "get_active",
        "set_selection",
        "set_active",
        "select_by_type",
        "assistant_help",
        "capture_viewport_for_vision",
    }

    # Intersect with Tool Selector if present (always include execute_code)

    try:
        allowed_set = (
            set(code_first_cap)
            if not enabled_tools
            else (set(code_first_cap) & set(enabled_tools))
        )

    except Exception:
        allowed_set = set(code_first_cap)
    # Ensure execute_code is always available
    allowed_set.add("execute_code")

    tools_section = mcp_tools.get_tools_schema(enabled_tools=sorted(allowed_set))

    return (
        "You are Blender Assistant — control Blender by writing Python code or calling native tools.\n\n"
        "BEHAVIOR\n"
        "- Prefer native tools when they map to your task; use execute_code for custom logic\n"
        "- If you plan to use assistant_sdk in execute_code, first call assistant_help with the SDK alias(es) to fetch the exact schema(s), then write code accordingly\n"
        "- Assign a dict to 'result' or '__result__' in execute_code to return structured values (e.g., result = assistant_sdk.polyhaven.download(...))\n"
        "- Minimal explanations unless asked\n\n"
        "SCENE VIEW (Outliner)\n"
        "- Use get_scene_info (Outliner) to see the current scene structure. Control expansion with expand_depth/expand/focus and persist fold_state across turns\n\n"
        "ASSISTANT_SDK API (in execute_code)\n"
        "- assistant_sdk.blender.*  (scene & objects, collections, selection)\n"
        "  e.g. blender.create_object(...), blender.modify_object(...), blender.set_material(...)\n"
        "  e.g. blender.move_to_collection([...], 'Props')\n"
        "- assistant_sdk.polyhaven.search/download, assistant_sdk.sketchfab.login/search/download\n"
        "- assistant_sdk.stock_photos.search/download, assistant_sdk.web.search, assistant_sdk.rag.query/get_stats\n"
        "TOOLS\n" + tools_section + "\n"
    )


def build_openai_tools() -> list:
    """Build OpenAI-style tools list from MCP registry using a fixed code-first tool set."""
    from . import tool_selector

    code_first_cap = {
        "execute_code",
        "get_scene_info",
        "get_object_info",
        "list_collections",
        "get_collection_info",
        "create_collection",
        "move_to_collection",
        "set_collection_color",
        "delete_collection",
        "get_selection",
        "get_active",
        "set_selection",
        "set_active",
        "select_by_type",
        "assistant_help",
        "capture_viewport_for_vision",
    }

    # Intersect with Tool Selector if present
    try:
        enabled_tools = tool_selector.get_enabled_tools() or []

    except Exception:
        enabled_tools = []

    allowed_set = (
        set(code_first_cap)
        if not enabled_tools
        else (set(code_first_cap) & set(enabled_tools))
    )
    # Ensure execute_code is always included regardless of Tool Selector
    allowed_set.add("execute_code")

    tools = []
    try:
        for t in mcp_tools.get_tools_list():
            name = t.get("name")
            if name in allowed_set:
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": t.get("description", ""),
                            "parameters": t.get(
                                "inputSchema", {"type": "object", "properties": {}}
                            ),
                        },
                    }
                )
    except Exception as e:
        print(f"[DEBUG] Failed to build tools list: {e}")

    return tools


def llama_chat(
    model_path: str,
    messages: list,
    temperature: float = 0.7,
    num_ctx: int = 131072,
    gpu_layers: int = -1,
    batch_size: int = 512,
    tools: list | None = None,
    format: dict | str | None = None,
    think: str | bool | None = None,
) -> dict:
    """Call Ollama for chat completion with GPU optimization.

    Args:
        model_path: Ollama model name (e.g., 'qwen2.5-coder:7b')
        messages: List of message dicts with 'role' and 'content'
        temperature: Temperature for generation (default: 0.7)
        num_ctx: Context window size (default: 131072 = 128k)
        gpu_layers: Number of layers to load to GPU (default: -1 = all)
        batch_size: Batch size for prompt processing (default: 512)
        tools: Optional OpenAI-style tools list to enable native tool calling
        format: Optional structured output format ("json" or a JSON Schema dict)

    Returns:
        Dict with 'message' containing 'content', or 'error' on failure
    """
    if not LLAMA_CPP_AVAILABLE:
        return {"error": "llama-cpp-python not available"}

    try:
        # Use model manager for chat completion
        return llama_manager.chat_completion(
            model_path=model_path,
            messages=messages,
            temperature=temperature,
            n_ctx=num_ctx,
            n_gpu_layers=gpu_layers,
            n_batch=batch_size,
            tools=tools,
            format=format,
            think=think,
        )

    except Exception as e:
        return {"error": f"llama-cpp-python error: {str(e)}"}


# Global flag to stop execution
_stop_requested = False


class ASSISTANT_OT_stop(bpy.types.Operator):
    """Stop the current assistant operation"""

    bl_idname = "assistant.stop"
    bl_label = "Stop"
    bl_options = {"REGISTER"}

    def execute(self, context):
        global _stop_requested
        _stop_requested = True
        self.report({"INFO"}, "Stop requested - will cancel after current step")
        return {"FINISHED"}


class ASSISTANT_OT_send(bpy.types.Operator):
    """Send message to AI assistant (runs in background, UI stays responsive)"""

    bl_idname = "assistant.send"
    bl_label = "Send"
    bl_options = {"REGISTER"}

    message: bpy.props.StringProperty(name="Message", default="")

    # Modal state variables (using class-level dict to avoid RNA threading issues)
    # We can't set instance attributes from threads, so use a shared dict
    _modal_state = {}
    _is_running = False  # Flag to prevent concurrent operations

    def modal(self, context, event):
        global _stop_requested

        state = self._modal_state

        # Handle ESC or stop button
        if event.type == "ESC" or _stop_requested:
            self._add_message("System", "⚠️ Stopped by user")
            self.cancel(context)
            self.report({"WARNING"}, "Operation cancelled")
            return {"CANCELLED"}

        # Timer event - check our state
        if event.type == "TIMER":
            if state.get("state") == "EXECUTE_QUEUED":
                # Execute next queued tool call
                queue = state.get("tool_call_queue", [])
                if queue:
                    next_call = queue.pop(0)
                    tool_name = next_call.get("tool")
                    args = next_call.get("args", {})

                    # Dedupe within this turn for queued tool calls
                    try:
                        key = f"{tool_name}|{json.dumps(args, sort_keys=True)}"
                    except Exception:
                        key = f"{tool_name}|{str(args)}"
                    exec_map = state.get("executed_this_turn", {}) or {}
                    if exec_map.get(key):
                        self._add_message(
                            "Tool", "Skipping duplicate tool call (queued) in this turn"
                        )
                        # If more queued, remain in EXECUTE_QUEUED; else proceed to next HTTP
                        if state.get("tool_call_queue"):
                            state["state"] = "EXECUTE_QUEUED"
                            return {"PASS_THROUGH"}
                        else:
                            # Inject brief scene snapshot before continuing
                            self._maybe_append_scene_snapshot(context)
                            self._start_http_request(context)
                            return {"PASS_THROUGH"}
                    exec_map[key] = 1
                    state["executed_this_turn"] = exec_map

                    # Skip execution if blocked
                    blocked = set(state.get("blocked_tools") or [])
                    # Never block execute_code; allow the model to self-correct code attempts
                    if tool_name in blocked and tool_name != "execute_code":
                        self._add_message(
                            "System",
                            f"Tool '{tool_name}' is temporarily blocked due to repeated failures. Skipping queued call.",
                        )
                    else:
                        print(
                            f"[DEBUG] Executing queued tool: {tool_name} with args: {args}"
                        )
                        self._add_message("Assistant", f"→ Calling {tool_name}({args})")
                        # Track objects/collections touched this turn for feedback loop
                        self._update_working_focus(tool_name, args)
                        try:
                            result = mcp_tools.execute_tool(tool_name, args)
                            print(f"[DEBUG] Tool result: {result}")
                            self._add_message(
                                "Tool",
                                json.dumps(result, indent=2),
                                tool_name=tool_name,
                            )
                            # Process result for loop hardening
                            self._process_tool_result(tool_name, args, result)
                            # Reset per-turn think override after a successful tool action
                            try:
                                state["think_override"] = None
                            except Exception:
                                pass
                        except Exception as e:
                            print(f"[DEBUG] Tool execution error: {str(e)}")
                            self._add_message(
                                "Tool", f"❌ Error: {str(e)}", tool_name=tool_name
                            )

                    # Check iteration limit

                    exempt_info_tools = {
                        "get_scene_info",
                        "get_object_info",
                        "list_collections",
                        "get_collection_info",
                        "get_selection",
                        "get_active",
                    }
                    if tool_name not in exempt_info_tools:
                        state["iteration"] += 1
                    if state["iteration"] >= state["max_iterations"]:
                        self._add_message("System", "⚠️ Max iterations reached")
                        state["state"] = "DONE"
                        return {"PASS_THROUGH"}

                    # If more queued calls, stay in EXECUTE_QUEUED state
                    if state.get("tool_call_queue"):
                        state["state"] = "EXECUTE_QUEUED"
                    else:
                        # All queued calls done, make HTTP request for next step
                        state["executing_queue"] = False
                        # Inject brief scene snapshot before continuing
                        self._maybe_append_scene_snapshot(context)
                        self._start_http_request(context)
                else:
                    # No more queued calls
                    state["executing_queue"] = False
                    # Inject brief scene snapshot before continuing
                    self._maybe_append_scene_snapshot(context)
                    self._start_http_request(context)

            elif state.get("state") == "WAITING_HTTP":
                # Check if HTTP thread is done
                thread = state.get("http_thread")
                if thread and not thread.is_alive():
                    if state.get("http_error"):
                        self._add_message("System", f"❌ Error: {state['http_error']}")
                        self.cancel(context)
                        return {"CANCELLED"}

                    # Got response, process it immediately
                    self._process_response(context)

                    # Check if processing set state to DONE (important for immediate exit)
                    if state.get("state") == "DONE":
                        self.cancel(context)
                        return {"FINISHED"}

            elif state.get("state") == "DONE":
                self.cancel(context)
                return {"FINISHED"}

        return {"PASS_THROUGH"}

    def _process_response(self, context):
        """Process LLM response and execute tool if needed."""
        state = self._modal_state
        reply = state.get("http_response")
        raw_reply_dict = state.get(
            "http_raw_response"
        )  # Get the raw dict for native tool calls

        # Check for native tool calls first (gpt-oss, newer Ollama models)
        native_tool_call = None
        native_tool_calls = []  # Collect ALL tool calls

        if raw_reply_dict and isinstance(raw_reply_dict, dict):
            message = raw_reply_dict.get("message", {})
            if isinstance(message, dict):
                # Support both OpenAI 'tool_calls' (array) and legacy 'function_call' (single)
                tool_calls = message.get("tool_calls", [])
                if (not tool_calls) and isinstance(message.get("function"), dict):
                    # Some providers use 'function' instead of tool_calls
                    tool_calls = [{"function": message.get("function")}]
                if (not tool_calls) and isinstance(message.get("function_call"), dict):
                    # OpenAI 0613 style: single function_call
                    fc = message.get("function_call")
                    tool_calls = [
                        {
                            "function": {
                                "name": fc.get("name"),
                                "arguments": fc.get("arguments"),
                            }
                        }
                    ]
                if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                    # Persist raw assistant tool_calls so we can include them in the next request
                    try:
                        state["last_assistant_tool_calls"] = tool_calls
                    except Exception:
                        pass

                    # Detect multiple tool calls
                    if len(tool_calls) > 1:
                        print(
                            f"[INFO] ✅ Model output {len(tool_calls)} tool calls - will execute them sequentially!"
                        )

                    # Extract ALL tool calls
                    for call_idx, call in enumerate(tool_calls):
                        if "function" not in call:
                            continue

                        func = call["function"]

                        func_name = func.get("name", "")

                        arguments = func.get("arguments", {})

                        extracted_call = None

                        # Wrapper format used by some models (e.g., assistant<|channel|>commentary)
                        # arguments looks like: {'name': '<tool_name>', 'arguments': { ... }}
                        if (
                            isinstance(arguments, dict)
                            and "name" in arguments
                            and "arguments" in arguments
                        ):
                            inner_name = arguments.get("name")
                            inner_args = arguments.get("arguments", {})
                            # If inner_args is a JSON string, try to parse it
                            if isinstance(inner_args, str):
                                try:
                                    inner_args = json.loads(inner_args)
                                except Exception:
                                    try:
                                        inner_args = ast.literal_eval(inner_args)
                                    except Exception:
                                        inner_args = {}
                            extracted_call = {"tool": inner_name, "args": inner_args}
                            if call_idx == 0:
                                print(
                                    f"[DEBUG] Unwrapped wrapper tool call: {extracted_call}"
                                )

                        # Format 1: Nested format {'name': 'execute_code', 'arguments': {'tool': 'create_object', 'args': {...}}}
                        elif (
                            isinstance(arguments, dict)
                            and "tool" in arguments
                            and "args" in arguments
                        ):
                            extracted_call = {
                                "tool": arguments.get("tool"),
                                "args": arguments.get("args", {}),
                            }
                            if call_idx == 0:
                                print(
                                    f"[DEBUG] Extracted nested native tool call: {extracted_call}"
                                )

                        # Format 2: Direct format {'name': 'execute_code', 'arguments': {'code': '...'}}
                        elif isinstance(arguments, dict):
                            # Extract tool name from function name (remove 'tool.' prefix if present)
                            tool_name = func_name.replace("tool.", "")

                            # If tool name is just "tool" (generic wrapper), infer from arguments
                            if tool_name == "tool" or not tool_name:
                                inferred = self._infer_tool_from_args(arguments)
                                if inferred:
                                    tool_name = inferred
                                    if call_idx == 0:
                                        print(
                                            f"[DEBUG] Inferred tool '{tool_name}' from arguments"
                                        )
                                else:
                                    if call_idx == 0:
                                        print(
                                            f"[DEBUG] Could not infer tool from arguments: {arguments}"
                                        )

                            extracted_call = {"tool": tool_name, "args": arguments}
                            if call_idx == 0:
                                print(
                                    f"[DEBUG] Extracted direct native tool call: {extracted_call}"
                                )

                        # Format 3: String arguments (need to parse JSON)
                        elif isinstance(arguments, str):
                            parsed_args = None
                            # Try JSON first
                            try:
                                parsed_args = json.loads(arguments)
                            except Exception:
                                # Fallback: Python literal (handles single quotes)
                                try:
                                    parsed_args = ast.literal_eval(arguments)
                                except Exception as e2:
                                    if call_idx == 0:
                                        print(
                                            f"[DEBUG] Failed to parse string arguments via JSON and literal_eval: {e2}"
                                        )
                                    parsed_args = None
                            if parsed_args is not None:
                                tool_name = func_name.replace("tool.", "")
                                extracted_call = {
                                    "tool": tool_name,
                                    "args": parsed_args,
                                }
                                if call_idx == 0:
                                    print(
                                        f"[DEBUG] Extracted string-parsed native tool call: {extracted_call}"
                                    )

                        if extracted_call:
                            native_tool_calls.append(extracted_call)

                    # Set first call as the primary one
                    if native_tool_calls:
                        native_tool_call = native_tool_calls[0]

            # Fallback: some adapters return tool_calls at the top level (no 'message' wrapper)
            if not native_tool_calls:
                top_level_calls = raw_reply_dict.get("tool_calls", [])
                if isinstance(top_level_calls, list) and top_level_calls:
                    # Persist raw assistant tool_calls so we can include them in the next request
                    try:
                        state["last_assistant_tool_calls"] = top_level_calls
                    except Exception:
                        pass
                    if len(top_level_calls) > 1:
                        print(
                            f"[INFO] ✅ Model output {len(top_level_calls)} tool calls (top-level) - will execute them sequentially!"
                        )

                    for call_idx, call in enumerate(top_level_calls):
                        try:
                            func = (
                                call.get("function", {})
                                if isinstance(call, dict)
                                else {}
                            )
                            func_name = func.get("name", "")

                            arguments = func.get("arguments", {})

                            extracted_call = None

                            # Wrapper format used by some models (e.g., assistant<|channel|>commentary)

                            # arguments looks like: {'name': '<tool_name>', 'arguments': { ... }}

                            if (
                                isinstance(arguments, dict)
                                and "name" in arguments
                                and "arguments" in arguments
                            ):
                                inner_name = arguments.get("name")

                                inner_args = arguments.get("arguments", {})

                                if isinstance(inner_args, str):
                                    try:
                                        inner_args = json.loads(inner_args)

                                    except Exception:
                                        try:
                                            inner_args = ast.literal_eval(inner_args)

                                        except Exception:
                                            inner_args = {}

                                extracted_call = {
                                    "tool": inner_name,
                                    "args": inner_args,
                                }

                                if call_idx == 0:
                                    print(
                                        f"[DEBUG] Unwrapped wrapper tool call (top-level): {extracted_call}"
                                    )

                            # Nested format {'arguments': {'tool': 'create_object', 'args': {...}}}

                            elif (
                                isinstance(arguments, dict)
                                and "tool" in arguments
                                and "args" in arguments
                            ):
                                extracted_call = {
                                    "tool": arguments.get("tool"),
                                    "args": arguments.get("args", {}),
                                }

                                if call_idx == 0:
                                    print(
                                        f"[DEBUG] Extracted nested native tool call (top-level): {extracted_call}"
                                    )

                            # Direct format {'name': 'execute_code', 'arguments': {...}}

                            elif isinstance(arguments, dict):
                                tool_name = (func_name or "").replace("tool.", "")

                                if tool_name == "tool" or not tool_name:
                                    inferred = self._infer_tool_from_args(arguments)

                                    if inferred:
                                        tool_name = inferred

                                        if call_idx == 0:
                                            print(
                                                f"[DEBUG] Inferred tool '{tool_name}' from arguments (top-level)"
                                            )

                                    else:
                                        if call_idx == 0:
                                            print(
                                                f"[DEBUG] Could not infer tool from arguments (top-level): {arguments}"
                                            )

                                extracted_call = {"tool": tool_name, "args": arguments}

                                if call_idx == 0:
                                    print(
                                        f"[DEBUG] Extracted direct native tool call (top-level): {extracted_call}"
                                    )

                            # String arguments (JSON)

                            elif isinstance(arguments, str):
                                try:
                                    parsed_args = json.loads(arguments)

                                    tool_name = (func_name or "").replace("tool.", "")

                                    extracted_call = {
                                        "tool": tool_name,
                                        "args": parsed_args,
                                    }

                                    if call_idx == 0:
                                        print(
                                            f"[DEBUG] Extracted string-parsed native tool call (top-level): {extracted_call}"
                                        )

                                except Exception as e:
                                    if call_idx == 0:
                                        print(
                                            f"[DEBUG] Failed to parse string arguments (top-level): {e}"
                                        )

                            if extracted_call:
                                native_tool_calls.append(extracted_call)

                        except Exception as e:
                            if call_idx == 0:
                                print(
                                    f"[DEBUG] Failed to extract top-level tool call: {e}"
                                )

                    if native_tool_calls:
                        native_tool_call = native_tool_calls[0]

        # If we have a native tool call, use it
        if native_tool_call and native_tool_call.get("tool"):
            tool_name = native_tool_call.get("tool")
            args = native_tool_call.get("args", {})

            # Check for repeated tool calls (loop detection)
            last_call = state.get("last_tool_call")
            if (
                last_call
                and last_call["tool"] == tool_name
                and last_call["args"] == args
            ):
                state["repeat_count"] = state.get("repeat_count", 0) + 1
                if state["repeat_count"] >= 3:
                    self._add_message(
                        "System",
                        f"⚠️ Detected loop: same tool called 3 times in a row. Stopping.",
                    )
                    state["state"] = "DONE"
                    return
            else:
                state["repeat_count"] = 0

            state["last_tool_call"] = {"tool": tool_name, "args": args}

            # Queue remaining tool calls if we have multiple
            if len(native_tool_calls) > 1:
                state["tool_call_queue"] = native_tool_calls[
                    1:
                ]  # Queue all except first
                print(
                    f"[DEBUG] Queued {len(state['tool_call_queue'])} additional tool calls"
                )

            # Dedupe within this turn for native tool calls
            try:
                key = f"{tool_name}|{json.dumps(args, sort_keys=True)}"
            except Exception:
                key = f"{tool_name}|{str(args)}"
            exec_map = state.get("executed_this_turn", {}) or {}
            if exec_map.get(key):
                self._add_message(
                    "Tool", "Skipping duplicate tool call (native) in this turn"
                )
                # Continue to next iteration without executing duplicate
                if len(state.get("tool_call_queue", [])) > 0:
                    state["state"] = "EXECUTE_QUEUED"
                else:
                    self._start_http_request(context)
                return
            exec_map[key] = 1
            state["executed_this_turn"] = exec_map

            # If tool is blocked, skip execution and prompt replanning
            blocked = set(state.get("blocked_tools") or [])
            # Never block execute_code; allow the model to self-correct code attempts
            if tool_name in blocked and tool_name != "execute_code":
                self._add_message(
                    "System",
                    f"Tool '{tool_name}' is temporarily blocked due to repeated failures. Skipping.",
                )
                # Continue loop without executing
                state["iteration"] += 1
                if state["iteration"] >= state["max_iterations"]:
                    self._add_message("System", "⚠️ Max iterations reached")
                    state["state"] = "DONE"
                    return
                if state.get("tool_call_queue"):
                    state["state"] = "EXECUTE_QUEUED"
                    return
                self._start_http_request(context)
                return

            self._add_message("Assistant", f"→ Calling {tool_name}({args})")
            # Track objects/collections touched this turn for feedback loop
            self._update_working_focus(tool_name, args)

            print(f"[DEBUG] Executing native tool: {tool_name} with args: {args}")

            try:
                result = mcp_tools.execute_tool(tool_name, args)
                print(f"[DEBUG] Tool result: {result}")
                # Process result for loop hardening
                self._process_tool_result(tool_name, args, result)
                # Reset per-turn think override after a successful tool action
                try:
                    state["think_override"] = None
                except Exception:
                    pass

                self._add_message(
                    "Tool", json.dumps(result, indent=2), tool_name=tool_name
                )

                # Vision tool now returns synchronously with textual description; no auto-polling.

                # Detect empty web search results and add hint
                if tool_name == "web_search" and isinstance(result, dict):
                    results = result.get("results", [])
                    if results:
                        # Check if all snippets are empty
                        all_empty = all(
                            not r.get("snippet", "").strip() for r in results
                        )
                        if all_empty and results:
                            hint = (
                                "\n⚠️ HINT: All search snippets are empty. "
                                "Use fetch_webpage() on one of the URLs to read the actual content. "
                                f"Example: fetch_webpage('{results[0]['url']}')"
                            )
                            self._add_message("System", hint)
            except Exception as e:
                print(f"[DEBUG] Tool execution error: {str(e)}")
                self._add_message("Tool", f"❌ Error: {str(e)}", tool_name=tool_name)

            # Check iteration limit

            exempt_info_tools = {
                "get_scene_info",
                "get_object_info",
                "list_collections",
                "get_collection_info",
                "get_selection",
                "get_active",
                "capture_viewport_for_vision",
            }

            if tool_name not in exempt_info_tools:
                state["iteration"] += 1

            if state["iteration"] >= state["max_iterations"]:
                self._add_message("System", "⚠️ Max iterations reached")
                state["state"] = "DONE"
                return

            # Check if we have queued tool calls to execute
            if state.get("tool_call_queue"):
                # Execute next queued call instead of making HTTP request
                state["executing_queue"] = True
                state["state"] = "EXECUTE_QUEUED"
                return


            # Start next iteration (inject a brief scene snapshot to keep context fresh)

            self._maybe_append_scene_snapshot(context)
            self._start_http_request(context)

            return


        # Fall back to text-based processing
        if not reply:
            # Check if we have thinking content to show (Ollama 'thinking' field)
            raw_response = state.get("http_raw_response", {})
            message = raw_response.get("message", {})
            thinking = message.get("thinking", "") if isinstance(message, dict) else ""

            if thinking:
                # Show thinking in chat log for user visibility, but do not echo it back to the LLM
                self._add_message("Assistant", f"💭 Thinking: {thinking}")

                # Nudge at most once per session to act on prior reasoning
                if not state.get("thinking_nudge_done"):
                    nudge = (
                        "Based on your prior reasoning, call the appropriate tool(s) now. "
                        "Respond with ONLY the tool call payload(s) (no extra text). If done, reply TASK_COMPLETE."
                    )
                    state["active_session"].messages.add()
                    state["active_session"].messages[-1].role = "System"
                    state["active_session"].messages[-1].content = nudge
                    self._add_message(
                        "System", "⚠️ Thinking detected — prompting model to act..."
                    )
                    state["thinking_nudge_done"] = True
                else:
                    # Enforce: disable thinking for next request to break the loop
                    state["think_override"] = False
                    self._add_message(
                        "System",
                        "⚠️ Thinking persisted — disabling chain-of-thought and enforcing tool-call-only output. "
                        "Respond with ONLY the tool call payload(s); no extra text.",
                    )

                # Continue to next iteration
                state["iteration"] += 1
                if state["iteration"] >= state["max_iterations"]:
                    self._add_message("System", "⚠️ Max iterations reached")
                    state["state"] = "DONE"
                    return
                self._start_http_request(context)
                return
            else:
                # Truly empty response: nudge once, then give up
                if not state.get("empty_retry_done"):
                    nudge = (
                        "No output received. If you cannot call a tool, reply with a brief text answer "
                        "summarizing your reasoning or next steps."
                    )
                    # Add to chat history so LLM sees it on retry
                    state["active_session"].messages.add()
                    state["active_session"].messages[-1].role = "System"
                    state["active_session"].messages[-1].content = nudge

                    self._add_message(
                        "System",
                        "⚠️ Empty response — nudging model to reply or call a tool...",
                    )
                    state["empty_retry_done"] = True

                    # Continue to next iteration with the nudge
                    state["iteration"] += 1
                    if state["iteration"] >= state["max_iterations"]:
                        self._add_message("System", "⚠️ Max iterations reached")
                        state["state"] = "DONE"
                        return
                    self._start_http_request(context)
                else:
                    # Second time empty — stop
                    self._add_message("System", "❌ Empty response from LLM")
                    state["state"] = "DONE"
            return

        print(f"[DEBUG] LLM Response: {reply[:200]}...")  # Debug output

        # Check for completion (exact match or contained in response)
        if "TASK_COMPLETE" in reply.upper() or reply.strip().upper() == "TASK_COMPLETE":
            self._add_message("Assistant", "✅ Task complete!")
            self.report({"INFO"}, f"Completed in {state['iteration']} step(s)")
            state["state"] = "DONE"
            return

        # Try to extract tool call via structured outputs first (for non-native models)
        tool_call = None
        try:
            if self._should_use_structured_outputs():
                tool_call = self._structured_extract_tool_json()
                print(f"[DEBUG] Structured outputs tool extraction: {tool_call}")
                # If structured returned a single tool, also scan the reply for a JSON array and
                # queue any additional steps (without re-executing the first one).
                if tool_call and not getattr(self, "_pending_json_calls", []):
                    try:
                        aux = self._extract_tool(reply)
                        # If the extractor found an array, it will have populated _pending_json_calls with the rest
                        if getattr(self, "_pending_json_calls", []):
                            print(
                                f"[DEBUG] Also found additional tool calls in reply; queuing {len(self._pending_json_calls)}"
                            )
                    except Exception as _e2:
                        print(f"[DEBUG] Auxiliary reply scan failed: {_e2}")
        except Exception as _e:
            print(f"[DEBUG] Structured extraction failed: {_e}")

        if not tool_call:
            tool_call = self._extract_tool(reply)
            print(f"[DEBUG] Extracted tool from text: {tool_call}")  # Debug output

        if tool_call:
            # Check if we have pending JSON calls to queue
            if hasattr(self, "_pending_json_calls") and self._pending_json_calls:
                state["tool_call_queue"] = self._pending_json_calls
                print(
                    f"[DEBUG] Queued {len(state['tool_call_queue'])} additional JSON tool calls"
                )
                self._pending_json_calls = []  # Clear

            # Execute tool (synchronous - fast enough)
            tool_name = tool_call.get("tool")
            args = tool_call.get("args", {})

            # Check for repeated tool calls (loop detection)
            last_call = state.get("last_tool_call")
            if (
                last_call
                and last_call["tool"] == tool_name
                and last_call["args"] == args
            ):
                state["repeat_count"] = state.get("repeat_count", 0) + 1
                if state["repeat_count"] >= 3:
                    self._add_message(
                        "System",
                        f"⚠️ Detected loop: same tool called 3 times in a row. Stopping.",
                    )
                    state["state"] = "DONE"
                    return
            else:
                state["repeat_count"] = 0

            state["last_tool_call"] = {"tool": tool_name, "args": args}

            # Dedupe within this turn for text-inferred tool calls
            try:
                key = f"{tool_name}|{json.dumps(args, sort_keys=True)}"
            except Exception:
                key = f"{tool_name}|{str(args)}"
            exec_map = state.get("executed_this_turn", {}) or {}
            if exec_map.get(key):
                self._add_message(
                    "Tool", "Skipping duplicate tool call (text) in this turn"
                )
                self._start_http_request(context)
                return
            exec_map[key] = 1
            state["executed_this_turn"] = exec_map

            # If tool is blocked, skip execution and prompt replanning
            blocked = set(state.get("blocked_tools") or [])
            # Never block execute_code; allow the model to self-correct code attempts
            if tool_name in blocked and tool_name != "execute_code":
                self._add_message(
                    "System",
                    f"Tool '{tool_name}' is temporarily blocked due to repeated failures. Skipping.",
                )
                state["iteration"] += 1
                if state["iteration"] >= state["max_iterations"]:
                    self._add_message("System", "⚠️ Max iterations reached")
                    state["state"] = "DONE"
                    return
                if state.get("tool_call_queue"):
                    state["state"] = "EXECUTE_QUEUED"
                    return
                self._start_http_request(context)
                return

            self._add_message("Assistant", f"→ Calling {tool_name}({args})")
            # Track objects/collections touched this turn for feedback loop
            self._update_working_focus(tool_name, args)

            print(f"[DEBUG] Executing tool: {tool_name} with args: {args}")

            try:
                result = mcp_tools.execute_tool(tool_name, args)
                print(f"[DEBUG] Tool result: {result}")
                # Process result for loop hardening
                self._process_tool_result(tool_name, args, result)
                # Reset per-turn think override after a successful tool action
                try:
                    state["think_override"] = None
                except Exception:
                    pass

                self._add_message(
                    "Tool", json.dumps(result, indent=2), tool_name=tool_name
                )

                # Detect empty web search results and add hint
                if tool_name == "web_search" and isinstance(result, dict):
                    results = result.get("results", [])
                    if results:
                        # Check if all snippets are empty
                        all_empty = all(
                            not r.get("snippet", "").strip() for r in results
                        )
                        if all_empty and results:
                            hint = (
                                "\n⚠️ HINT: All search snippets are empty. "
                                "Use fetch_webpage() on one of the URLs to read the actual content. "
                                f"Example: fetch_webpage('{results[0]['url']}')"
                            )
                            self._add_message("System", hint)
            except Exception as e:
                print(f"[DEBUG] Tool execution error: {str(e)}")
                self._add_message("Tool", f"❌ Error: {str(e)}", tool_name=tool_name)

            # Check iteration limit

            exempt_info_tools = {
                "get_scene_info",
                "get_object_info",
                "list_collections",
                "get_collection_info",
                "get_selection",
                "get_active",
            }
            if tool_name not in exempt_info_tools:
                state["iteration"] += 1

            if state["iteration"] >= state["max_iterations"]:
                self._add_message("System", "⚠️ Max iterations reached")
                state["state"] = "DONE"
                return

            # Check if we have queued tool calls to execute
            if state.get("tool_call_queue"):
                # Execute next queued call instead of making HTTP request
                state["executing_queue"] = True
                state["state"] = "EXECUTE_QUEUED"
                return


            # Start next iteration (inject a brief scene snapshot to keep context fresh)

            self._maybe_append_scene_snapshot(context)
            self._start_http_request(context)

        else:

            # No tool call - conversational response
            self._add_message("Assistant", reply)
            state["state"] = "DONE"

    def _start_http_request(self, context=None):
        """Start HTTP request to Ollama in background thread."""
        state = self._modal_state
        state["state"] = "WAITING_HTTP"
        state["http_response"] = None
        state["http_raw_response"] = None
        state["http_error"] = None
        # Reset per-turn executed tool calls map to avoid duplicate executions in one turn
        state["executed_this_turn"] = {}
        # Reset working focus for the new turn
        state["working_focus"] = {"objects": [], "collections": []}

        # Build messages from chat history

        messages = [{"role": "system", "content": state["system_prompt"]}]
        # One-off SDK quick reference injection on first turn to aid planning
        try:
            if int(state.get("iteration", 0) or 0) == 0 and not state.get(
                "sdk_quickref_injected"
            ):
                from . import blender_tools as _bt

                _sdk = _bt._get_assistant_sdk()
                _help = _sdk.help() if _sdk else ""
                if _help:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "SDK QUICK REFERENCE\n" + _help,
                        }
                    )
                    state["sdk_quickref_injected"] = True
        except Exception:
            # Never fail if help cannot be produced
            pass

        # Track if we need to augment the first user message with RAG and optionally follow-ups
        first_user_message = True
        rag_enabled = (
            state["prefs"].use_rag if hasattr(state["prefs"], "use_rag") else False
        )
        rag_follow_ups = getattr(state["prefs"], "rag_follow_up_augmentation", True)

        # RAG follow-up augmentation is controlled only by user preference; lean/api_lean removed

        rag_sent_indices = state.get("rag_sent_indices") or set()
        state["rag_sent_indices"] = rag_sent_indices

        # If the last turn used native tools, inject an assistant tool_calls message before tool results
        pending_tool_calls = state.get("last_assistant_tool_calls")
        injected_tool_calls = False

        history = state["active_session"].messages
        start_idx = max(0, len(history) - MAX_HISTORY_MESSAGES)
        for it in history[start_idx:]:
            role = (it.role or "").lower()
            if role == "you":
                content = it.content

                # Augment with RAG (first message always if enabled; some follow-ups if configured)
                try:
                    from . import rag_system

                    # Start RAG load in background if needed (non-blocking)
                    rag_system.ensure_rag_loaded_async()
                    rag = rag_system.get_rag_instance()
                    if rag_enabled and rag.enabled:
                        # Prefer source selection
                        bias_pref = getattr(state["prefs"], "rag_source_bias", "AUTO")
                        # Force API docs in API-only Lean Mode
                        # lean/api_lean removed; no bias override
                        prefer_source = None
                        lc = (content or "").lower()
                        api_keywords = ["bpy.", "bpy.ops", "script", "code", "execute"]
                        if bias_pref == "API_ONLY":
                            prefer_source = "API"
                        elif bias_pref == "MANUAL_ONLY":
                            prefer_source = "Manual"
                        elif bias_pref == "BOTH":
                            prefer_source = None
                        else:  # AUTO
                            if any(k in lc for k in api_keywords):
                                prefer_source = "API"

                        # Determine chunk count
                        mode = getattr(state["prefs"], "rag_context_mode", "AUTO")
                        if mode == "AUTO":
                            mn = getattr(state["prefs"], "rag_auto_min", 6)
                            mx = getattr(state["prefs"], "rag_auto_max", 12)
                            n_results = mx if any(k in lc for k in api_keywords) else mn
                        else:
                            n_results = getattr(state["prefs"], "rag_num_results", 5)

                        triggers = any(k in lc for k in api_keywords)
                        if first_user_message:
                            content = rag.augment_prompt(
                                content,
                                n_results=n_results,
                                prefer_source=prefer_source,
                                exclude_indices=rag_sent_indices,
                            )
                            if getattr(rag, "last_indices", None):
                                rag_sent_indices.update(rag.last_indices)
                            print(
                                f"[RAG] Augmented first user message with {n_results} documentation chunks"
                            )
                            first_user_message = False
                        elif rag_follow_ups and triggers:
                            content = rag.augment_prompt(
                                content,
                                n_results=n_results,
                                prefer_source=prefer_source,
                                exclude_indices=rag_sent_indices,
                            )
                            if getattr(rag, "last_indices", None):
                                rag_sent_indices.update(rag.last_indices)
                            print(
                                f"[RAG] Augmented follow-up user message with {n_results} documentation chunks"
                            )
                except Exception as e:
                    print(f"[RAG] Failed to augment message: {e}")

                msg = {"role": "user", "content": content}
                # Add image if attached to this message
                if it.image_data:
                    msg["images"] = [it.image_data]
                messages.append(msg)
            elif role == "assistant":
                content = it.content
                # Do not echo chain-of-thought back to the model
                if content and content.strip().startswith("💭 Thinking"):
                    continue
                messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                # Only include real tool results (those with a stored tool_name)
                tname = getattr(it, "tool_name", "") if hasattr(it, "tool_name") else ""
                if tname:
                    # Inject the assistant tool_calls message once, right before the first tool result
                    if pending_tool_calls and not injected_tool_calls:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": pending_tool_calls,
                            }
                        )
                        injected_tool_calls = True
                        # Clear after injecting so we don't duplicate next turn
                        state["last_assistant_tool_calls"] = None

                    messages.append(
                        {"role": "tool", "name": tname, "content": it.content or ""}
                    )
                else:
                    # Skip non-result tool logs (e.g., dedupe notices) from API payload
                    pass





        print(f"[DEBUG] Starting HTTP request with {len(messages)} messages")

        # Keep a copy of the exact messages we sent (for structured extraction reuse)
        state["last_request_messages"] = list(messages)

        # Get model name from preferences (Ollama model name like "qwen2.5-coder:7b")
        model_name = state["prefs"].model_file
        if not model_name or model_name == "NONE":
            state["http_error"] = (
                "No model selected. Please select a model in preferences."
            )
            return

        # Build OpenAI-style tools list for native tool calling
        tools_all = build_openai_tools()
        # Filter out any tools blocked due to failures this session
        blocked = set(state.get("blocked_tools") or [])
        # Never filter out execute_code; allow self-correction attempts
        effective_blocked = {b for b in blocked if b != "execute_code"}
        if effective_blocked:
            before = len(tools_all)
            tools_all = [
                t
                for t in tools_all
                if t.get("function", {}).get("name") not in effective_blocked
            ]
            print(
                f"[DEBUG] Tools filtered by blocklist: {before} -> {len(tools_all)} (blocked: {', '.join(sorted(effective_blocked))})"
            )
        print(f"[DEBUG] OpenAI tools prepared: {len(tools_all)}")

        # Decide whether to enable native tools for this model (per-model flag + heuristic fallback)
        mn = (model_name or "").lower()

        disable_native_tools = mn.startswith("gemma") or mn.split(":")[0] in {
            "gemma",
            "gemma2",
            "gemma3",
        }

        use_tools = True
        try:
            # If preferences expose a per-model tools flag, respect it
            if hasattr(state["prefs"], "_is_tools_enabled"):
                use_tools = bool(state["prefs"]._is_tools_enabled(model_name))
        except Exception:
            pass
        if disable_native_tools:
            use_tools = False
        tools_to_send = tools_all if use_tools else None
        if not use_tools:
            print(
                "[DEBUG] Native tools disabled for this model; using text-based tool extraction."
            )

        # Determine effective temperature (boost on first turn if planning exploration is enabled)
        effective_temp = state.get("prefs").temperature if state.get("prefs") else 0.2
        try:
            if (
                getattr(state["prefs"], "planning_exploration", False)
                and state.get("iteration", 0) == 0
            ):
                effective_temp = getattr(
                    state["prefs"], "planning_temperature", effective_temp
                )
        except Exception:
            pass

        # Compute Ollama 'think' parameter from preferences and per-model capability flags
        think_level = getattr(state["prefs"], "thinking_level", "LOW")

        base_name = (model_name or "").split(":")[0].lower()

        # Per-model allow/deny for thinking (if provided by preferences UI)
        think_allowed = True
        try:
            if hasattr(state["prefs"], "_is_thinking_enabled"):
                think_allowed = bool(state["prefs"]._is_thinking_enabled(model_name))
        except Exception:
            think_allowed = True
        # Per-turn override (e.g., after detecting repeated thinking without action)

        if "think_override" in state and state["think_override"] is not None:
            think_param = state["think_override"]

        else:
            if base_name == "gpt-oss":
                # OFF must fully disable chain-of-thought; LOW/MEDIUM/HIGH map to strings

                if think_level == "OFF":
                    think_param = False

                elif think_level == "LOW":
                    think_param = "low"

                elif think_level == "MEDIUM":
                    think_param = "medium"

                else:
                    think_param = "high"

            else:
                # Other models: OFF -> False, otherwise enable generic thinking

                think_param = False if think_level == "OFF" else True

        # Enforce per-model flag: if disabled, do not send 'think'
        if not think_allowed:
            think_param = False
        print(f"[DEBUG] Think param: {think_param}")

        # Start thread with GPU settings
        thread = threading.Thread(
            target=self._llama_worker,
            args=(
                model_name,  # Pass Ollama model name directly
                messages,
                tools_to_send,
                effective_temp,
                state["prefs"].gpu_layers,
                state["prefs"].num_ctx,
                state["prefs"].batch_size,
                think_param,
                state,
            ),
        )
        thread.daemon = True
        thread.start()
        state["http_thread"] = thread

    def _llama_worker(
        self,
        model_path,
        messages,
        tools,
        temperature,
        gpu_layers,
        context_length,
        batch_size,
        think_param,
        state_dict,
    ):
        """Worker thread for Ollama request (runs in background)."""

        try:
            print(f"[DEBUG] Ollama worker starting request with model: {model_path}")

            reply_dict = llama_chat(
                model_path=model_path,
                messages=messages,
                temperature=temperature,
                num_ctx=context_length,
                gpu_layers=gpu_layers,
                batch_size=batch_size,
                tools=tools,
                think=think_param,
            )

            print(f"[DEBUG] llama.cpp worker got reply_dict: {reply_dict}")

            # Error handling with targeted retries
            if "error" in reply_dict:
                error_msg = str(reply_dict.get("error", ""))
                print(f"[DEBUG] HTTP worker got error: {error_msg}")

                # Retry once for HTTP 500 (transient server crash)
                if (
                    ("500" in error_msg)
                    or ("internal server error" in error_msg.lower())
                ) and not state_dict.get("server_retry_done"):
                    import time as _time

                    state_dict["server_retry_done"] = True

                    print("[DEBUG] Retrying once after HTTP 500...")

                    _time.sleep(0.75)

                    reply_dict = llama_chat(
                        model_path=model_path,
                        messages=messages,
                        temperature=temperature,
                        num_ctx=context_length,
                        gpu_layers=gpu_layers,
                        batch_size=batch_size,
                        tools=tools,
                        think=think_param,
                    )
                    print(f"[DEBUG] Retry got reply_dict: {reply_dict}")

                    if "error" in reply_dict:
                        state_dict["http_error"] = reply_dict["error"]

                        return

                # Retry once for HTTP 400 due to 'think' not supported
                elif (
                    (("400" in error_msg) or ("http error 400" in error_msg.lower()))
                    and (
                        "think" in error_msg.lower() or "thinking" in error_msg.lower()
                    )
                    and think_param
                    and not state_dict.get("think_retry_done")
                ):
                    state_dict["think_retry_done"] = True

                    print(
                        "[DEBUG] Retrying once without thinking due to model limitation (HTTP 400)"
                    )

                    reply_dict = llama_chat(
                        model_path=model_path,
                        messages=messages,
                        temperature=temperature,
                        num_ctx=context_length,
                        gpu_layers=gpu_layers,
                        batch_size=batch_size,
                        tools=tools,
                        think=False,
                    )
                    print(f"[DEBUG] Retry (no think) got reply_dict: {reply_dict}")

                    if "error" in reply_dict:
                        state_dict["http_error"] = reply_dict["error"]

                        return

                else:
                    state_dict["http_error"] = error_msg

                    return

            # Success path: store raw response and extract text
            state_dict["http_raw_response"] = reply_dict

            text = None
            # Format 1: Standard Ollama chat format {"message": {"content": "..."}}

            if "message" in reply_dict:
                message = reply_dict.get("message", {})

                if isinstance(message, dict):
                    text = message.get("content", "")

                elif isinstance(message, str):
                    text = message

            # Format 2: Direct response format {"response": "..."}

            elif "response" in reply_dict:
                text = reply_dict.get("response", "")

            # Format 3: Direct content format {"content": "..."}

            elif "content" in reply_dict:
                text = reply_dict.get("content", "")

            # Format 4: Choices format (OpenAI-style)
            elif (
                "choices" in reply_dict
                and isinstance(reply_dict["choices"], list)
                and len(reply_dict["choices"]) > 0
            ):
                choice = reply_dict["choices"][0]

                if "message" in choice:
                    text = choice["message"].get("content", "")

                elif "text" in choice:
                    text = choice.get("text", "")

            if text is None:
                # Check if we have native tool calls - if so, empty text is OK

                message = reply_dict.get("message", {})

                if isinstance(message, dict) and message.get("tool_calls"):
                    print(
                        "[DEBUG] HTTP worker: Empty content but has native tool_calls"
                    )

                    text = ""
                else:
                    print("[DEBUG] HTTP worker: Could not extract text from reply_dict")

                    print(f"[DEBUG] Reply dict keys: {list(reply_dict.keys())}")

                    print(f"[DEBUG] Full reply_dict: {reply_dict}")

                    state_dict["http_error"] = (
                        f"Invalid response format from Ollama. Keys: {list(reply_dict.keys())}"
                    )

                    return

            text = (text or "").strip()
            print(
                f"[DEBUG] HTTP worker extracted text: {text[:100] if text else '(empty)'}..."
            )

            state_dict["http_response"] = text

        except Exception as e:
            print(f"[DEBUG] HTTP worker exception: {e}")

            import traceback

            traceback.print_exc()

            state_dict["http_error"] = str(e)

    def _add_message(self, role, content, image_data=None, tool_name: str = ""):
        """Add message to chat and update UI.

        Args:
            role: Message role (You, Assistant, Tool, System)
            content: Message text content
            image_data: Optional base64 encoded image data
            tool_name: When role=="Tool", the originating tool's name for proper
                       Ollama tool result messages.
        """
        state = self._modal_state
        item = state["active_session"].messages.add()
        item.role = role
        item.content = str(content)
        if image_data:
            item.image_data = image_data
        # Persist tool name only for Tool messages
        try:
            if (role or "").lower() == "tool":
                item.tool_name = tool_name or item.tool_name
        except Exception:
            pass

        # Update UI
        state["wm"].assistant_chat_message_index = (
            len(state["active_session"].messages) - 1
        )

        # Force redraw
        for area in state["context_ref"].screen.areas:
            if area.type == "VIEW_3D":
                area.tag_redraw()
        bpy.ops.wm.redraw_timer(type="DRAW_WIN_SWAP", iterations=1)

    def _process_tool_result(self, tool_name: str, args, result: dict):
        """Analyze a tool result and harden the loop by blocking bad patterns.
        - Blocks unknown tools after first occurrence and informs the model.
        - Blocks execute_code after repeated operator/context errors in a turn.
        - Tracks failure signatures to avoid repeated attempts.
        """
        try:
            state = self._modal_state
            if not isinstance(result, dict):
                return
            err = str(result.get("error", ""))
            if not err:
                return

            err_low = err.lower()

            # Unknown tool: block and inform with available tool list
            if err_low.startswith("unknown tool:"):
                try:
                    unknown = err.split(":", 1)[1].strip()
                    if "." in unknown:
                        unknown = unknown.split(".")[0].strip()
                except Exception:
                    unknown = tool_name
                counts = dict(state.get("unknown_tool_counts") or {})
                counts[unknown] = counts.get(unknown, 0) + 1
                state["unknown_tool_counts"] = counts

                # Add to blocked list so we short-circuit next time
                blocked = set(state.get("blocked_tools") or [])
                blocked.add(unknown)
                state["blocked_tools"] = list(blocked)

                from . import tool_selector

                try:
                    enabled = tool_selector.get_enabled_tools() or []
                except Exception:
                    enabled = []
                try:
                    # Fixed minimal, code-first capability set
                    base = {
                        "execute_code",
                        "get_scene_info",
                        "get_object_info",
                        "list_collections",
                        "get_collection_info",
                        "create_collection",
                        "move_to_collection",
                        "set_collection_color",
                        "delete_collection",
                        "get_selection",
                        "get_active",
                        "set_selection",
                        "set_active",
                        "select_by_type",
                        "assistant_help",
                        "capture_viewport_for_vision",
                    }
                    allowed_set = base if not enabled else (base & set(enabled))
                    allowed_names = sorted(n for n in allowed_set if n)
                    available = ", ".join(allowed_names)
                except Exception:
                    available = "(tool list unavailable)"
                self._add_message(
                    "System",
                    f"Tool '{unknown}' does not exist. Use available tools: {available}. Do not call '{unknown}' again.",
                )
                return

            # Failure signature counting
            sig = err_low[:120]
            key = f"{tool_name}|{sig}"
            fc = dict(state.get("failure_counts") or {})
            fc[key] = fc.get(key, 0) + 1
            state["failure_counts"] = fc

            # Execute code specific hardening
            if tool_name == "execute_code":
                bad_patterns = (
                    "1-2 args execution context",
                    "unrecognized",
                    "syntaxerror",
                    "attributeerror",
                    "nameerror",
                )
                if any(p in err_low for p in bad_patterns) or fc[key] >= 2:
                    # Do not block execute_code; models often self-correct via code.
                    # Instead, show a one-time guidance hint to steer toward native tools when appropriate.
                    if not state.get("execute_code_warned"):
                        self._add_message(
                            "System",
                            "Hint: Repeated Blender operator/context errors detected in execute_code. "
                            "Consider using native tools (create_object, modify_object) or adjust the code context.",
                        )
                        state["execute_code_warned"] = True

        except Exception:
            # Never let debug helpers crash the loop
            pass

    def execute(self, context):
        global _stop_requested
        _stop_requested = False  # Reset stop flag at start

        # Check if already running - prevent concurrent operations
        if ASSISTANT_OT_send._is_running:
            self.report(
                {"WARNING"}, "Assistant is already processing a message. Please wait."
            )
            return {"CANCELLED"}

        prefs = context.preferences.addons[__package__].preferences
        wm = context.window_manager

        # Pull message from WindowManager
        user_text = getattr(wm, "assistant_message", "").strip()

        # Validate input
        if not user_text:
            self.report({"WARNING"}, "Please enter a message")
            return {"CANCELLED"}

        # Get active chat session
        if not wm.assistant_chat_sessions:
            self.report({"ERROR"}, "No chat session available")
            return {"CANCELLED"}

        active_idx = wm.assistant_active_chat_index
        if active_idx < 0 or active_idx >= len(wm.assistant_chat_sessions):
            self.report({"ERROR"}, "Invalid chat session")
            return {"CANCELLED"}

        active_session = wm.assistant_chat_sessions[active_idx]

        # Mark as running
        ASSISTANT_OT_send._is_running = True

        # Initialize modal state (thread-safe dict)
        self._modal_state = {
            "prefs": prefs,
            "wm": wm,
            "active_session": active_session,
            "context_ref": context,
            "max_iterations": prefs.max_iterations,
            "iteration": 0,
            "state": "IDLE",
            "http_thread": None,
            "http_response": None,
            "http_raw_response": None,  # Store raw dict for native tool calls
            "http_error": None,
            "system_prompt": build_system_prompt(),
            "timer": None,
            "last_tool_call": None,  # For loop detection
            "repeat_count": 0,  # For loop detection
            "tool_call_queue": [],  # Queue for multiple tool calls from one response
            "executing_queue": False,  # Flag to track if we're executing queued calls
            # Loop hardening state
            "blocked_tools": [],  # Names blocked this session due to failures
            "failure_counts": {},  # (tool|sig) -> count
            "unknown_tool_counts": {},  # unknown tool name -> count
            "execute_code_warned": False,  # one-time hint shown when execute_code repeats errors
            # Thinking control
            "thinking_nudge_done": False,  # only nudge once per session when thinking has no action
            "think_override": None,  # per-turn override for Ollama 'think' param (e.g., disable after loop)
            # Scene feedback loop tracking (objects/collections touched this turn)
            "working_focus": {"objects": [], "collections": []},
        }

        print(
            f"[DEBUG] System prompt ready; allowed tools: {len(build_openai_tools())}"
        )

        # Add user message to chat (with image if pending)
        pending_image = getattr(wm, "assistant_pending_image", "")
        self._add_message(
            "You", user_text, image_data=pending_image if pending_image else None
        )

        # Image (if any) is attached directly to the chat message; no separate pending image handoff.

        # Clear input box and pending image
        try:
            wm.assistant_message = ""
            wm.assistant_pending_image = ""
        except Exception:
            pass

        # Start modal timer
        timer = wm.event_timer_add(0.1, window=context.window)
        self._modal_state["timer"] = timer
        context.window_manager.modal_handler_add(self)

        # Start first HTTP request
        self._start_http_request(context)

        self.report({"INFO"}, "Working... (Press ESC to cancel)")
        return {"RUNNING_MODAL"}

    def cancel(self, context):
        """Cleanup when modal is cancelled."""
        state = self._modal_state

        timer = state.get("timer")
        if timer:
            context.window_manager.event_timer_remove(timer)
            state["timer"] = None

        # Wait for thread to finish (with timeout)
        thread = state.get("http_thread")
        if thread and thread.is_alive():
            thread.join(timeout=1.0)

        # Cap message history to prevent memory issues
        active_session = state.get("active_session")
        if active_session and len(active_session.messages) > 200:
            # Remove oldest messages
            while len(active_session.messages) > 200:
                active_session.messages.remove(0)

        # Clear running flag so UI Stop becomes disabled and Send enabled
        ASSISTANT_OT_send._is_running = False

    def _update_working_focus(self, tool_name: str, args: dict):
        """Track objects/collections referenced by the executed tool.
        Keeps a short, recency-ordered list in modal state for scene snapshots.

        Hardened to avoid treating wrapper fields or tool names as scene objects.
        """
        try:
            state = self._modal_state
            wf = state.get("working_focus") or {"objects": [], "collections": []}

            # Build a set of known tool names to filter out accidental additions
            try:
                from . import mcp_tools as _mtools

                _tool_defs = _mtools.get_tools_list()
                known_tools = {
                    t.get("name")
                    for t in _tool_defs
                    if isinstance(t, dict) and t.get("name")
                }
            except Exception:
                known_tools = set()

            def _should_keep_name(s: str) -> bool:
                # Filter out empty, the current tool name, and any known tool names
                if not s:
                    return False
                s2 = str(s).strip()
                if not s2:
                    return False
                if isinstance(tool_name, str) and s2 == tool_name:
                    return False
                if s2 in known_tools:
                    return False
                # Otherwise keep
                return True

            def add_unique(lst: list, val: str, max_len: int = 40):
                try:
                    if not _should_keep_name(val):
                        return
                    s = str(val).strip()
                    if s in lst:
                        lst.remove(s)
                    lst.append(s)
                    while len(lst) > max_len:
                        lst.pop(0)
                except Exception:
                    pass

            # Heuristic: treat tools with 'collection' in the name as collection tools
            is_collection_tool = isinstance(tool_name, str) and (
                "collection" in tool_name
            )

            # Known keys to inspect
            object_keys = [
                "name",
                "names",
                "object_name",
                "object_names",
                "source_object",
                "target_object",
                "new_name",
            ]
            collection_keys = [
                "collection",
                "collection_name",
                "collection_names",
                "target_collection",
                "parent",
                "parent_collection",
            ]

            # Unwrap common wrappers if a provider returned a nested envelope by mistake
            unwrapped = args
            if isinstance(unwrapped, dict):
                # Cases like {'name': 'create_collection', 'arguments': {...}}
                if (
                    "arguments" in unwrapped
                    and isinstance(unwrapped.get("arguments"), dict)
                    and ("name" in unwrapped or "function" in unwrapped)
                ):
                    unwrapped = unwrapped.get("arguments")
                # Cases like {'tool': 'create_object', 'args': {...}}
                elif (
                    "tool" in unwrapped
                    and "args" in unwrapped
                    and isinstance(unwrapped.get("args"), dict)
                ):
                    unwrapped = unwrapped.get("args")
                # Cases like {'parameters': {...}, 'name': '...'}
                elif (
                    "parameters" in unwrapped
                    and isinstance(unwrapped.get("parameters"), dict)
                    and "name" in unwrapped
                ):
                    unwrapped = unwrapped.get("parameters")

            if isinstance(unwrapped, dict):
                # Collections first if it's a collection-related tool
                if is_collection_tool:
                    for k in collection_keys:
                        v = unwrapped.get(k)
                        if isinstance(v, str):
                            add_unique(wf["collections"], v)
                        elif isinstance(v, (list, tuple)):
                            for it in v:
                                if isinstance(it, str):
                                    add_unique(wf["collections"], it)
                                elif isinstance(it, dict) and "name" in it:
                                    add_unique(wf["collections"], it.get("name"))
                        elif isinstance(v, dict) and "name" in v:
                            add_unique(wf["collections"], v.get("name"))
                # Always inspect for object names
                for k in object_keys:
                    v = unwrapped.get(k)
                    if isinstance(v, str):
                        add_unique(wf["objects"], v)
                    elif isinstance(v, (list, tuple)):
                        for it in v:
                            if isinstance(it, str):
                                add_unique(wf["objects"], it)
                            elif isinstance(it, dict) and "name" in it:
                                add_unique(wf["objects"], it.get("name"))
                    elif isinstance(v, dict) and "name" in v:
                        add_unique(wf["objects"], v.get("name"))

                # Special-case: moving objects to a collection often includes both
                if "target_collection" in unwrapped:
                    add_unique(wf["collections"], unwrapped.get("target_collection"))

            state["working_focus"] = wf
        except Exception:
            # Non-fatal
            pass

    def _maybe_append_scene_snapshot(self, context=None):
        """Append a compact outliner snapshot to keep model context fresh."""

        try:
            state = self._modal_state
            prefs = state.get("prefs")

            auto = getattr(prefs, "auto_scene_snapshot", True) if prefs else True



            if not auto:

                return



            # Persist and reuse fold_state across turns

            fold = state.get("snapshot_fold_state") or {}
            # Focus recently touched names

            wf = state.get("working_focus") or {"objects": [], "collections": []}



            focus = list((wf.get("collections") or []) + (wf.get("objects") or []))[-8:]

            args = {
                "expand_depth": 1,
                "expand": [],
                "focus": focus,
                "fold_state": fold,
                "max_children": int(getattr(prefs, "snapshot_max_objects", 8) or 8),
                "include_icons": True,
                "include_counts": True,
            }
            result = mcp_tools.execute_tool("get_scene_info", args)

            if isinstance(result, dict):
                # Save updated fold_state
                if "fold_state" in result:
                    state["snapshot_fold_state"] = result["fold_state"]
                # Emit compact Outliner lines
                lines = (result.get("outliner", {}) or {}).get("lines", [])
                if lines:
                    self._add_message(
                        "Tool", "\n".join(lines), tool_name="get_scene_info"
                    )
        except Exception:
            # Do not break the loop if snapshotting fails

            pass

    def _extract_tool(self, text_s: str):
        """Extract tool call JSON from LLM response.

        Handles multiple formats:
        1. Conversational: "→ Calling tool_name({...})"
        2. Multiple JSON objects on separate lines (Llama format)
        3. Single JSON object
        4. Fenced code blocks (json and python)
        """
        import ast
        import re

        # Clean up escaped characters that some models add
        text_s = text_s.replace("\\_", "_")  # Fix escaped underscores
        text_s = text_s.replace("\\{", "{").replace("\\}", "}")  # Fix escaped braces

        # 1) Conversational format: "→ Calling tool_name({...})"
        # Example: "→ Calling create_object({'type': 'SPHERE', 'name': 'Venus'})"
        # Also support empty args: "→ Calling get_current_subtask()"
        pattern = r"(?:→|->)\s*Calling\s+(\w+)\s*\((.*)\)\s*$"
        match = re.search(pattern, text_s, re.DOTALL | re.MULTILINE)
        if match:
            tool_name = match.group(1)
            args_str = match.group(2).strip()

            # If no args provided, treat as empty dict
            if args_str == "" or args_str == "{}":
                return {"tool": tool_name, "args": {}}

            # Try multiple parsing strategies
            try:
                # Strategy 1: Python literal eval (handles single quotes)
                args = ast.literal_eval(args_str)
                return {"tool": tool_name, "args": args}
            except Exception as e:
                print(
                    f"[DEBUG] Failed to parse conversational tool call with ast.literal_eval: {e}"
                )

            try:
                # Strategy 2: Simple quote replacement
                args_str_json = args_str.replace("'", '"')
                args = json.loads(args_str_json)
                return {"tool": tool_name, "args": args}
            except Exception as e2:
                print(f"[DEBUG] Fallback JSON parse also failed: {e2}")

            try:
                # Strategy 3: Fix mixed quotes (Gemma issue: {'code": "..."})
                # Replace mixed quote patterns
                args_str_fixed = re.sub(r"{\s*'(\w+)\":", r'{"\1":', args_str)
                args_str_fixed = args_str_fixed.replace("'", '"')
                args = json.loads(args_str_fixed)
                return {"tool": tool_name, "args": args}
            except Exception as e3:
                print(f"[DEBUG] Mixed quote fix also failed: {e3}")

            try:
                # Strategy 4: Extract just the inner content and rebuild
                # For execute_code, extract the code string more aggressively
                if tool_name == "execute_code":
                    # Try to find code between the first quote after "code" and the last quote before }
                    # Handle both escaped and unescaped strings
                    code_pattern = (
                        r'["\']?code["\']?\s*:\s*["\'](.+)["\']?\s*\}?\s*\)?\s*$'
                    )
                    code_match = re.search(code_pattern, args_str, re.DOTALL)
                    if code_match:
                        code_content = code_match.group(1)
                        # Remove trailing quotes and braces
                        code_content = code_content.rstrip("\"})' ")
                        # Unescape common escape sequences
                        code_content = (
                            code_content.replace("\\n", "\n")
                            .replace("\\t", "\t")
                            .replace("\\'", "'")
                            .replace('\\"', '"')
                        )
                        print(
                            f"[DEBUG] Extracted code via aggressive parsing: {len(code_content)} chars"
                        )
                        return {"tool": tool_name, "args": {"code": code_content}}
            except Exception as e4:
                print(f"[DEBUG] Code extraction also failed: {e4}")

            # If all parsing failed but we have a tool name, return with empty args
            print(
                f"[DEBUG] All parsing strategies failed, returning tool '{tool_name}' with empty args"
            )
            return {"tool": tool_name, "args": {}}

        # 1b) ReAct-style: "Action" / "Action Input" blocks (Qwen-style)
        react_calls = []
        try:
            # Capture multiple Action + Action Input pairs; stop args at blank line or end
            react_iter = re.finditer(
                r"(?mi)^\s*Action\s*:\s*([A-Za-z_]\w*)\s*$[\s\S]*?^\s*Action Input\s*:\s*(.+?)(?:\r?\n\s*\r?\n|$)",
                text_s,
            )
            for m in react_iter:
                tname = m.group(1).strip()
                ain = m.group(2).strip()
                body = ain
                # If fenced code follows, strip fence header and closing
                if body.startswith("```"):
                    parts3 = body.split("```", 2)
                    if len(parts3) >= 2:
                        body = parts3[1].strip()
                # Try parse body as dict
                obj = {}
                parsed = False
                if body.startswith("{"):
                    try:
                        obj = json.loads(body)
                        parsed = isinstance(obj, dict)
                    except Exception:
                        try:
                            obj = ast.literal_eval(body)
                            parsed = isinstance(obj, dict)
                        except Exception:
                            parsed = False
                if not parsed:
                    a = body.find("{")
                    b = body.rfind("}")
                    if a != -1 and b != -1 and b > a:
                        frag = body[a : b + 1]
                        try:
                            obj = json.loads(frag)
                            parsed = isinstance(obj, dict)
                        except Exception:
                            try:
                                obj = ast.literal_eval(frag)
                                parsed = isinstance(obj, dict)
                            except Exception:
                                parsed = False
                if not isinstance(obj, dict):
                    obj = {}
                react_calls.append({"tool": tname, "args": obj})
            if react_calls:
                if len(react_calls) > 1:
                    self._pending_json_calls = react_calls[1:]
                return react_calls[0]
        except Exception:
            pass

        # 1c) Function-like text patterns outside fences
        func_like_patterns = [
            r"(?mi)^\s*CALL\s+([A-Za-z_]\w*)\s*\((.*?)\)\s*$",
            r"(?mi)^\s*tool\.([A-Za-z_]\w*)\s*\((.*?)\)\s*$",
            r"(?mi)^\s*([A-Za-z_]\w*)\s*\((.*?)\)\s*$",
        ]
        for pat in func_like_patterns:
            m = re.search(pat, text_s, re.DOTALL)
            if m:
                tname = m.group(1)
                args_str = (m.group(2) or "").strip()
                if args_str == "" or args_str == "{}":
                    return {"tool": tname, "args": {}}
                try:
                    args_obj = json.loads(args_str)
                    if isinstance(args_obj, dict):
                        return {"tool": tname, "args": args_obj}
                except Exception:
                    try:
                        args_obj = ast.literal_eval(args_str)
                        if isinstance(args_obj, dict):
                            return {"tool": tname, "args": args_obj}
                    except Exception:
                        pass

        # 1d) YAML-ish pairs: tool: name / arguments: {...}
        yaml_match = re.search(
            r"(?mis)^\s*tool\s*:\s*([A-Za-z_]\w*)\s*$[\s\S]*?^\s*(parameters|arguments|args)\s*:\s*(\{[\s\S]*?\})",
            text_s,
        )
        if yaml_match:
            tname = yaml_match.group(1)
            arg_str = yaml_match.group(3)
            try:
                obj = json.loads(arg_str)
            except Exception:
                try:
                    obj = ast.literal_eval(arg_str)
                except Exception:
                    obj = {}
            if not isinstance(obj, dict):
                obj = {}
            return {"tool": tname, "args": obj}

        # 2) Multiple JSON objects on separate lines (Llama 3.1 format)
        # Example:
        # {"tool": "create_object", "args": {...}}
        # {"tool": "modify_object", "args": {...}}
        lines = text_s.strip().split("\n")

        # Extract ALL tool calls from JSON lines
        json_tool_calls = []
        for line in lines:
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        # Format 1: {"tool": "name", "args": {...}}
                        if obj.get("tool"):
                            json_tool_calls.append(obj)
                        # Format 2: {"name": "tool_name", "arguments"|"parameters"|"args": {...}}
                        elif obj.get("name"):
                            tool_name = obj.get("name")
                            args = obj.get(
                                "parameters", obj.get("arguments", obj.get("args", {}))
                            )
                            json_tool_calls.append({"tool": tool_name, "args": args})
                        # Format 3: {"function": {"name": "tool_name", "arguments"|"parameters"|"args": {...}}}
                        elif obj.get("function") and isinstance(obj["function"], dict):
                            func = obj["function"]
                            tool_name = func.get("name")
                            args = func.get(
                                "parameters",
                                func.get("arguments", func.get("args", {})),
                            )
                            if tool_name:
                                json_tool_calls.append(
                                    {"tool": tool_name, "args": args}
                                )
                        # Format 3b: {"function": "tool_name", "arguments"|"parameters"|"args": {...}}
                        elif isinstance(obj.get("function"), str):
                            tool_name = obj.get("function")
                            args = obj.get(
                                "parameters", obj.get("arguments", obj.get("args", {}))
                            )
                            json_tool_calls.append({"tool": tool_name, "args": args})
                        # Format 4: {"type": "function", "function": {...}}
                        elif obj.get("type") == "function" and obj.get("function"):
                            func = obj["function"]
                            tool_name = func.get("name")
                            args = func.get(
                                "parameters",
                                func.get("arguments", func.get("args", {})),
                            )
                            if tool_name:
                                json_tool_calls.append(
                                    {"tool": tool_name, "args": args}
                                )
                        else:
                            # Infer tool from known argument patterns
                            tool_name = self._infer_tool_from_args(obj)
                            if tool_name:
                                # Unwrap common envelopes if present
                                inferred_args = (
                                    obj.get(
                                        "parameters",
                                        obj.get("arguments", obj.get("args", obj)),
                                    )
                                    if isinstance(obj, dict)
                                    else obj
                                )
                                json_tool_calls.append(
                                    {"tool": tool_name, "args": inferred_args}
                                )
                except Exception:
                    continue

        if len(json_tool_calls) > 1:
            print(
                f"[INFO] ✅ Model output {len(json_tool_calls)} tool calls - will execute them sequentially!"
            )
            # Store additional calls in state for queuing
            # This is a bit hacky - we'll return the first and store the rest in a class variable
            self._pending_json_calls = json_tool_calls[1:]

        # Return the first tool call if we found any
        if json_tool_calls:
            print(f"[DEBUG] Found tool call in line: {str(json_tool_calls[0])[:100]}")
            return json_tool_calls[0]

        # 3) fenced code block ```json ... ``` or ```python ... ```
        if "```" in text_s:
            parts = text_s.split("```")
            for i in range(1, len(parts), 2):
                block = parts[i].strip()

                # Handle ```python function_name(args) ``` format
                if block.lower().startswith("python"):
                    block = block[6:].strip()
                    # Try to parse Python function call: tool_name(arg1=val1, arg2=val2)
                    func_pattern = r"^(\w+)\s*\((.*)\)\s*$"
                    func_match = re.match(func_pattern, block, re.DOTALL)
                    if func_match:
                        tool_name = func_match.group(1)
                        args_str = func_match.group(2).strip()

                        try:
                            # Parse Python kwargs into dict
                            import ast

                            # Wrap in dict() to make it parseable
                            dict_str = f"dict({args_str})"
                            args = ast.literal_eval(dict_str)
                            print(
                                f"[DEBUG] Parsed Python function call: {tool_name}({args})"
                            )
                            return {"tool": tool_name, "args": args}
                        except Exception as e:
                            print(f"[DEBUG] Failed to parse Python function call: {e}")
                            # Try simpler parsing for common patterns
                            try:
                                # Handle simple cases like: task="...", subtasks=[...]
                                args = {}
                                for param in args_str.split(","):
                                    if "=" in param:
                                        key, val = param.split("=", 1)
                                        key = key.strip()
                                        val = val.strip()
                                        args[key] = ast.literal_eval(val)
                                print(
                                    f"[DEBUG] Parsed Python function call (simple): {tool_name}({args})"
                                )
                                return {"tool": tool_name, "args": args}
                            except Exception as e2:
                                print(f"[DEBUG] Simple parsing also failed: {e2}")

                # Handle ```tool:<name>``` header with JSON body
                header_match = re.match(
                    r"^(tool|tool_call|call)\s*:\s*([A-Za-z_]\w*)\s*\n([\s\S]+)$",
                    block,
                    re.IGNORECASE,
                )
                if header_match:
                    tool_name = header_match.group(2)
                    body = header_match.group(3).strip()
                    if body.lower().startswith("json"):
                        body = body[4:].strip()
                    try:
                        obj = json.loads(body)
                    except Exception:
                        try:
                            obj = ast.literal_eval(body)
                        except Exception:
                            # Try to find a JSON object inside
                            a = body.find("{")
                            b = body.rfind("}")
                            obj = {}
                            if a != -1 and b != -1 and b > a:
                                frag = body[a : b + 1]
                                try:
                                    obj = json.loads(frag)
                                except Exception:
                                    try:
                                        obj = ast.literal_eval(frag)
                                    except Exception:
                                        obj = {}
                    if isinstance(obj, dict):
                        print(f"[DEBUG] Parsed tool header block: {tool_name}")
                        return {"tool": tool_name, "args": obj}

                # Handle ```tool_code function_name(args)``` or ```tool ...``` formats
                if (
                    block.lower().startswith("tool_code")
                    or block.lower().startswith("toolcall")
                    or block.lower().startswith("tool")
                ):
                    # Remove the leading label (e.g., 'tool_code') and parse the function call
                    parts2 = block.split(None, 1)
                    block_body = parts2[1].strip() if len(parts2) > 1 else ""
                    func_pattern = r"^(\w+)\s*\((.*)\)\s*$"
                    func_match = re.match(func_pattern, block_body, re.DOTALL)
                    if func_match:
                        tool_name = func_match.group(1)
                        args_str = func_match.group(2).strip()
                        try:
                            # Simple parse: split kwargs and ast-eval values
                            import ast

                            args = {}
                            if args_str:
                                for param in args_str.split(","):
                                    if "=" in param:
                                        key, val = param.split("=", 1)
                                        key = key.strip()
                                        val = val.strip()
                                        args[key] = ast.literal_eval(val)
                            print(
                                f"[DEBUG] Parsed tool_code function call: {tool_name}({args})"
                            )
                            return {"tool": tool_name, "args": args}
                        except Exception as e:
                            print(
                                f"[DEBUG] Failed to parse tool_code function call: {e}"
                            )

                # Generic function-call fallback inside fenced block (no language label)
                generic_match = re.match(r"^(\w+)\s*\((.*)\)\s*$", block, re.DOTALL)
                if generic_match:
                    tool_name = generic_match.group(1)
                    args_str = generic_match.group(2).strip()
                    try:
                        import ast

                        args = {}
                        if args_str:
                            for param in args_str.split(","):
                                if "=" in param:
                                    key, val = param.split("=", 1)
                                    key = key.strip()
                                    val = val.strip()
                                    args[key] = ast.literal_eval(val)
                        print(
                            f"[DEBUG] Parsed generic function call in fenced block: {tool_name}({args})"
                        )
                        return {"tool": tool_name, "args": args}
                    except Exception as e:
                        print(
                            f"[DEBUG] Failed to parse generic function call in fenced block: {e}"
                        )

                # Handle ```json {...} ``` format
                if block.lower().startswith("json"):
                    block = block[4:].strip()
                try:
                    obj = json.loads(block)
                except Exception:
                    try:
                        import ast

                        obj = ast.literal_eval(block)
                    except Exception:
                        obj = None
                if obj is not None:
                    # Handle list of tool calls (take first, queue the rest)
                    if isinstance(obj, list):
                        calls = [x for x in obj if isinstance(x, dict)]

                        def _norm(it):
                            if it.get("tool"):
                                return {
                                    "tool": it.get("tool"),
                                    "args": it.get("args", {}),
                                }
                            if it.get("name"):
                                return {
                                    "tool": it.get("name"),
                                    "args": it.get(
                                        "parameters",
                                        it.get("arguments", it.get("args", {})),
                                    ),
                                }
                            if it.get("function") and isinstance(it["function"], dict):
                                func = it["function"]
                                tn = func.get("name")
                                args = func.get(
                                    "parameters",
                                    func.get("arguments", func.get("args", {})),
                                )
                                return {"tool": tn, "args": args} if tn else None
                            if it.get("type") == "function" and it.get("function"):
                                func = it["function"]
                                tn = func.get("name")
                                args = func.get(
                                    "parameters",
                                    func.get("arguments", func.get("args", {})),
                                )
                                return {"tool": tn, "args": args} if tn else None
                            return None

                        norm_calls = [c for c in (_norm(item) for item in calls) if c]
                        if norm_calls:
                            first = norm_calls[0]
                            rest = norm_calls[1:]
                            if rest:
                                setattr(self, "_pending_json_calls", rest)
                            return first
                        # Try inference if we had raw calls but none normalized
                        if calls:
                            inferred = self._infer_tool_from_args(calls[0])
                            if inferred:
                                rest = calls[1:] if len(calls) > 1 else []
                                if rest:
                                    setattr(self, "_pending_json_calls", rest)
                                return {"tool": inferred, "args": calls[0]}
                    elif isinstance(obj, dict):
                        # Single object
                        if obj.get("tool"):
                            return obj
                        elif obj.get("name"):
                            return {
                                "tool": obj.get("name"),
                                "args": obj.get(
                                    "parameters",
                                    obj.get("arguments", obj.get("args", {})),
                                ),
                            }
                        elif obj.get("function") and isinstance(obj["function"], dict):
                            func = obj["function"]
                            tn = func.get("name")
                            args = func.get(
                                "parameters",
                                func.get("arguments", func.get("args", {})),
                            )
                            if tn:
                                return {"tool": tn, "args": args}
                        elif obj.get("type") == "function" and obj.get("function"):
                            func = obj["function"]
                            tn = func.get("name")
                            args = func.get(
                                "parameters",
                                func.get("arguments", func.get("args", {})),
                            )
                            if tn:
                                return {"tool": tn, "args": args}

        # 4a) fallback: first [...] span (list of tool calls)
        try:
            lstart = text_s.find("[")
            lend = text_s.rfind("]")
            if lstart != -1 and lend != -1 and lend > lstart:
                json_list_str = text_s[lstart : lend + 1]
                try:
                    arr = json.loads(json_list_str)
                except Exception:
                    try:
                        import ast

                        arr = ast.literal_eval(json_list_str)
                    except Exception as e2:
                        print(f"[DEBUG] Fallback list parse failed (json+ast): {e2}")
                        arr = None
                if isinstance(arr, list):
                    calls = [x for x in arr if isinstance(x, dict)]

                    def _norm(it):
                        if it.get("tool"):
                            return {"tool": it.get("tool"), "args": it.get("args", {})}
                        if it.get("name"):
                            return {
                                "tool": it.get("name"),
                                "args": it.get(
                                    "parameters",
                                    it.get("arguments", it.get("args", {})),
                                ),
                            }
                        if it.get("function") and isinstance(it["function"], dict):
                            func = it["function"]
                            tn = func.get("name")
                            args = func.get(
                                "parameters",
                                func.get("arguments", func.get("args", {})),
                            )
                            return {"tool": tn, "args": args} if tn else None
                        if it.get("type") == "function" and it.get("function"):
                            func = it["function"]
                            tn = func.get("name")
                            args = func.get(
                                "parameters",
                                func.get("arguments", func.get("args", {})),
                            )
                            return {"tool": tn, "args": args} if tn else None
                        return None

                    norm_calls = [c for c in (_norm(item) for item in calls) if c]
                    if norm_calls:
                        first = norm_calls[0]
                        rest = norm_calls[1:]
                        if rest:
                            setattr(self, "_pending_json_calls", rest)
                        print(
                            f"[DEBUG] Found tool call via fallback list parser: {first['tool']}"
                        )
                        return first
                    if calls:
                        inferred = self._infer_tool_from_args(calls[0])
                        if inferred:
                            rest = calls[1:] if len(calls) > 1 else []
                            if rest:
                                setattr(self, "_pending_json_calls", rest)
                            print(
                                f"[DEBUG] Inferred tool '{inferred}' from fallback list item: {calls[0]}"
                            )
                            obj0 = calls[0]
                            # Unwrap common envelopes if present
                            args0 = (
                                obj0.get(
                                    "parameters",
                                    obj0.get("arguments", obj0.get("args", obj0)),
                                )
                                if isinstance(obj0, dict)
                                else obj0
                            )
                            return {"tool": inferred, "args": args0}
        except Exception as e:
            print(f"[DEBUG] Fallback list JSON parse failed: {e}")

        # 4) fallback: first {...} span (handles multi-line JSON)
        try:
            start = text_s.find("{")
            end = text_s.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = text_s[start : end + 1]
                try:
                    obj = json.loads(json_str)
                except Exception:
                    try:
                        import ast

                        obj = ast.literal_eval(json_str)
                    except Exception as e2:
                        print(f"[DEBUG] Fallback object parse failed (json+ast): {e2}")
                        obj = None
                if isinstance(obj, dict):
                    # Format 1: {"tool": "name", "args": {...}}
                    if obj.get("tool"):
                        print(
                            f"[DEBUG] Found tool call via fallback parser: {obj.get('tool')}"
                        )
                        return obj
                    # Format 2: {"name": "tool_name", "arguments": {...}}
                    elif obj.get("name"):
                        tool_name = obj.get("name")
                        args = obj.get("arguments", obj.get("args", {}))
                        print(
                            f"[DEBUG] Found tool call via fallback parser (name format): {tool_name}"
                        )
                        return {"tool": tool_name, "args": args}
                    # Format 3: {"function": {"name": "...", "arguments": {...}}}
                    elif obj.get("function") and isinstance(obj["function"], dict):
                        func = obj["function"]
                        tool_name = func.get("name")
                        args = func.get("arguments", func.get("args", {}))
                        if tool_name:
                            print(
                                f"[DEBUG] Found tool call via fallback parser (function format): {tool_name}"
                            )
                            return {"tool": tool_name, "args": args}
                    # Format 4: {"type": "function", "function": {...}}
                    elif obj.get("type") == "function" and obj.get("function"):
                        func = obj["function"]
                        tool_name = func.get("name")
                        args = func.get("arguments", func.get("args", {}))
                        if tool_name:
                            print(
                                f"[DEBUG] Found tool call via fallback parser (type:function format): {tool_name}"
                            )
                            return {"tool": tool_name, "args": args}
                    # Infer tool from args if none of the formats matched
                    tool_name = self._infer_tool_from_args(obj)
                    if tool_name:
                        print(
                            f"[DEBUG] Inferred tool '{tool_name}' from fallback args: {obj}"
                        )
                        # Unwrap common envelopes if present
                        inferred_args = (
                            obj.get("arguments", obj.get("args", obj))
                            if isinstance(obj, dict)
                            else obj
                        )
                        return {"tool": tool_name, "args": inferred_args}
        except Exception as e:
            print(f"[DEBUG] Fallback JSON parse failed: {e}")
            pass

        print(
            f"[DEBUG] No tool call found in response. First 200 chars: {text_s[:200]}"
        )
        return None

    def _should_use_structured_outputs(self) -> bool:
        """Decide whether to use Ollama Structured Outputs for tool extraction.

        We keep native tool-calling as-is for models that support it (e.g., gpt-oss).
        For Gemma-family models and other non-native models, prefer structured outputs
        over brittle text extraction.
        """
        try:
            state = getattr(self, "_modal_state", {}) or {}
            prefs = state.get("prefs")
            mn = (getattr(prefs, "model_file", "") or "").lower()
            if not mn:
                return False
            base = mn.split(":")[0]
            # Never override native tools path for gpt-oss
            if base in {"gpt-oss"}:
                return False
            # Prefer structured outputs for Gemma family (no native tools)
            return base.startswith("gemma") or base in {"gemma", "gemma2", "gemma3"}
        except Exception:
            return False

    def _structured_extract_tool_json(self):
        """Call Ollama with a JSON Schema 'format' to extract a single tool and args.

        Returns {"tool": name, "args": {...}} or None.
        """
        try:
            from . import mcp_tools
        except Exception as e:
            print(f"[DEBUG] structured extract: cannot import mcp_tools: {e}")
            return None

        state = getattr(self, "_modal_state", {}) or {}
        prefs = state.get("prefs")
        model = getattr(prefs, "model_file", None) if prefs else None
        if not model:
            return None

        # Build allowed tool name list using a fixed code-first capability set + Tool Selector
        try:
            tool_defs = mcp_tools.get_tools_list()
            from . import tool_selector

            # Tool Selector can disable tools; default to no filter
            try:
                enabled = tool_selector.get_enabled_tools() or []
            except Exception:
                enabled = []

            base = {
                "execute_code",
                "get_scene_info",
                "get_object_info",
                "list_collections",
                "get_collection_info",
                "create_collection",
                "move_to_collection",
                "set_collection_color",
                "delete_collection",
                "get_selection",
                "get_active",
                "set_selection",
                "set_active",
                "select_by_type",
                "assistant_help",
                "capture_viewport_for_vision",
            }
            tool_set = base if not enabled else (base & set(enabled))
            # Always include execute_code regardless of Tool Selector
            tool_set.add("execute_code")

            tool_names = sorted(
                t.get("name")
                for t in tool_defs
                if isinstance(t, dict) and t.get("name") in tool_set
            )
        except Exception as e:
            print(f"[DEBUG] structured extract: failed to get tools list: {e}")
            return None

        if not tool_names:
            return None

        # Envelope schema (keep args flexible; per-tool validation happens downstream)
        item_schema = {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": sorted(tool_names + ["none"])},
                "args": {"type": "object"},
            },
            "required": ["tool"],
            "additionalProperties": False,
        }
        # Accept either a single tool call object or an array of them
        format_schema = {
            "anyOf": [
                item_schema,
                {"type": "array", "items": item_schema, "minItems": 1},
            ]
        }

        # Use the exact messages we sent for the turn, append a short system reminder
        messages = list(state.get("last_request_messages") or [])
        if not messages:
            return None
        allowed_list_txt = ", ".join(sorted(tool_names))
        messages = messages + [
            {
                "role": "system",
                "content": (
                    "Decide the best next tool(s) to call and return ONLY JSON matching the provided schema. "
                    f"Allowed tools: {allowed_list_txt}. "
                    "If multiple steps are clearly required, return an array of tool calls in execution order; "
                    "if only one is needed, return a single object. "
                    'If no tool is needed, set tool to "none".'
                ),
            }
        ]

        # Make a non-streaming call with 'format' set to the schema
        try:
            result = llama_chat(
                model_path=model,
                messages=messages,
                temperature=0.1,
                num_ctx=getattr(prefs, "ctx_size", 131072) if prefs else 131072,
                gpu_layers=getattr(prefs, "gpu_layers", -1) if prefs else -1,
                batch_size=getattr(prefs, "batch_size", 512) if prefs else 512,
                tools=None,
                format=format_schema,
            )
        except Exception as e:
            print(f"[DEBUG] structured extract: llama_chat exception: {e}")
            return None

        if not isinstance(result, dict) or result.get("error"):
            print(
                f"[DEBUG] structured extract call error: {result.get('error') if isinstance(result, dict) else result}"
            )
            return None

        content = ""
        if "message" in result and isinstance(result["message"], dict):
            content = (result["message"].get("content") or "").strip()
        elif "content" in result:
            content = str(result.get("content") or "").strip()
        if not content:
            return None

        try:
            obj = json.loads(content)
        except Exception as e:
            print(f"[DEBUG] structured extract: JSON parse failed: {e}")
            return None

        # Handle either a single object or an array of tool calls
        if isinstance(obj, list):
            calls = [x for x in obj if isinstance(x, dict)]

            def _norm(it):
                if it.get("tool"):
                    return {"tool": it.get("tool"), "args": it.get("args", {})}
                if it.get("name"):
                    return {
                        "tool": it.get("name"),
                        "args": it.get("arguments", it.get("args", {})),
                    }
                if it.get("function") and isinstance(it["function"], dict):
                    func = it["function"]
                    tn = func.get("name")
                    args = func.get("arguments", func.get("args", {}))
                    return {"tool": tn, "args": args} if tn else None
                if it.get("type") == "function" and it.get("function"):
                    func = it["function"]
                    tn = func.get("name")
                    args = func.get("arguments", func.get("args", {}))
                    return {"tool": tn, "args": args} if tn else None
                return None

            norm_calls = [c for c in (_norm(item) for item in calls) if c]
            if norm_calls:
                first = norm_calls[0]
                rest = norm_calls[1:]
                if rest:
                    setattr(self, "_pending_json_calls", rest)
                return first
            if calls:
                inferred = self._infer_tool_from_args(calls[0])
                if inferred:
                    rest = calls[1:] if len(calls) > 1 else []
                    if rest:
                        setattr(self, "_pending_json_calls", rest)
                    return {"tool": inferred, "args": calls[0]}
            return None

        if not isinstance(obj, dict):
            return None

        tool = obj.get("tool")
        args = obj.get("args", {})
        if tool in (None, "", "none"):
            return None
        if not isinstance(args, dict):
            args = {}
        return {"tool": tool, "args": args}

    def _infer_tool_from_args(self, args: dict) -> str:
        """Infer tool name from argument keys.

        Some models return only the arguments without the tool name.
        We can infer the tool from unique argument patterns.
        """
        if not isinstance(args, dict):
            return None

        # Map unique argument combinations to tool names
        arg_keys = set(args.keys())

        # get_scene_info - has "info_level" key
        if "info_level" in arg_keys:
            return "get_scene_info"

        # execute_code - has "code" key
        if "code" in arg_keys and len(arg_keys) == 1:
            return "execute_code"

        # create_object - has "type" and optionally "name", "location"
        if "type" in arg_keys and arg_keys <= {
            "type",
            "name",
            "location",
            "rotation",
            "scale",
        }:
            return "create_object"

        # PolyHaven downloads - has "asset_id"
        # Need to distinguish between texture, model, and HDRI
        if "asset_id" in arg_keys:
            # HDRI has resolution + file_format (exr/hdr)
            if "file_format" in arg_keys and args.get("file_format") in ["exr", "hdr"]:
                return "download_polyhaven_hdri"
            # Model has file_format (blend/fbx/gltf)
            if "file_format" in arg_keys and args.get("file_format") in [
                "blend",
                "fbx",
                "gltf",
            ]:
                return "download_polyhaven_model"
            # Texture has resolution
            if "resolution" in arg_keys:
                return "download_polyhaven_texture"
            # Just asset_id - ambiguous, but check context from thinking
            # Default to texture as it's most common
            print(
                f"[DEBUG] Ambiguous asset_id call - defaulting to download_polyhaven_texture"
            )
            return "download_polyhaven_texture"

        # search_polyhaven_assets - has "query" and "asset_type"
        if "query" in arg_keys and "asset_type" in arg_keys:
            return "search_polyhaven_assets"

        # set_active - has "object_name" only
        if arg_keys == {"object_name"}:
            return "set_active"

        # get_selection - no args
        if len(arg_keys) == 0:
            return "get_selection"

        # create_task_plan - has "task_description" and "subtasks"
        if "task_description" in arg_keys and "subtasks" in arg_keys:
            return "create_task_plan"

        # complete_current_subtask - has "result" or no args
        if arg_keys <= {"result"}:
            # Could be complete_current_subtask or get_current_subtask
            # Default to complete if has result
            if "result" in arg_keys:
                return "complete_current_subtask"

        print(f"[DEBUG] Could not infer tool from args: {arg_keys}")
        return None


class ASSISTANT_OT_clear(bpy.types.Operator):
    bl_idname = "assistant.clear"
    bl_label = "Clear Chat"
    bl_options = {"REGISTER"}

    def execute(self, context):
        wm = context.window_manager
        if wm.assistant_chat_sessions:
            active_idx = wm.assistant_active_chat_index
            if 0 <= active_idx < len(wm.assistant_chat_sessions):
                wm.assistant_chat_sessions[active_idx].messages.clear()
                self.report({"INFO"}, "Chat cleared")
            else:
                self.report({"WARNING"}, "No active chat")
        return {"FINISHED"}


class ASSISTANT_OT_new_chat(bpy.types.Operator):
    """Create a new chat session"""

    bl_idname = "assistant.new_chat"
    bl_label = "New Chat"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        import datetime

        wm = context.window_manager
        session = wm.assistant_chat_sessions.add()
        session.name = f"Chat {len(wm.assistant_chat_sessions)}"
        session.created = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        # Switch to the newly created chat and sync the enum so the dropdown shows the correct label
        new_idx = len(wm.assistant_chat_sessions) - 1
        wm.assistant_active_chat_index = new_idx
        try:
            wm.assistant_active_chat_enum = str(new_idx)
        except Exception:
            pass
        self.report({"INFO"}, f"Created new chat: {session.name}")
        return {"FINISHED"}


class ASSISTANT_OT_delete_chat(bpy.types.Operator):
    """Delete the current chat session"""

    bl_idname = "assistant.delete_chat"
    bl_label = "Delete Chat"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        wm = context.window_manager
        if not wm.assistant_chat_sessions:
            self.report({"WARNING"}, "No chats to delete")
            return {"CANCELLED"}

        active_idx = wm.assistant_active_chat_index
        if 0 <= active_idx < len(wm.assistant_chat_sessions):
            session_name = wm.assistant_chat_sessions[active_idx].name
            wm.assistant_chat_sessions.remove(active_idx)

            # Adjust active index and sync enum to keep dropdown label correct
            if wm.assistant_active_chat_index >= len(wm.assistant_chat_sessions):
                wm.assistant_active_chat_index = max(
                    0, len(wm.assistant_chat_sessions) - 1
                )
            try:
                wm.assistant_active_chat_enum = str(wm.assistant_active_chat_index)
            except Exception:
                pass

            self.report({"INFO"}, f"Deleted chat: {session_name}")
        return {"FINISHED"}


class ASSISTANT_OT_rename_chat(bpy.types.Operator):
    """Rename the current chat session"""

    bl_idname = "assistant.rename_chat"
    bl_label = "Rename Chat"
    bl_options = {"REGISTER", "UNDO"}

    new_name: bpy.props.StringProperty(name="New Name", default="")

    def invoke(self, context, event):
        wm = context.window_manager
        if wm.assistant_chat_sessions:
            active_idx = wm.assistant_active_chat_index
            if 0 <= active_idx < len(wm.assistant_chat_sessions):
                self.new_name = wm.assistant_chat_sessions[active_idx].name
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        wm = context.window_manager
        if wm.assistant_chat_sessions:
            active_idx = wm.assistant_active_chat_index
            if 0 <= active_idx < len(wm.assistant_chat_sessions):
                wm.assistant_chat_sessions[active_idx].name = self.new_name
                self.report({"INFO"}, f"Renamed to: {self.new_name}")
        return {"FINISHED"}


def base64_to_blender_image(base64_data: str, image_name: str) -> bpy.types.Image:
    """Convert base64 image data to Blender Image datablock.

    Args:
        base64_data: Base64 encoded image (PNG format)
        image_name: Name for the Blender image datablock

    Returns:
        Blender Image datablock
    """
    import base64
    import io

    import numpy as np
    from PIL import Image

    # Decode base64 to bytes
    img_bytes = base64.b64decode(base64_data)

    # Load with PIL
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    # Convert to numpy array
    img_array = np.array(pil_img)
    height, width = img_array.shape[:2]

    # Remove old image if exists
    if image_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[image_name])

    # Create Blender image
    blender_img = bpy.data.images.new(
        image_name, width=width, height=height, alpha=True
    )

    # Blender expects pixels in range [0, 1] and flipped vertically
    pixels = img_array.astype(np.float32) / 255.0
    pixels = np.flipud(pixels)  # Flip vertically
    blender_img.pixels = pixels.flatten()
    blender_img.update()

    # Pack the image so it's embedded and has a preview
    blender_img.pack()

    # Ensure preview is generated
    blender_img.preview_ensure()

    return blender_img


class ASSISTANT_OT_paste_image(bpy.types.Operator):
    """Paste image from clipboard to attach to next message"""

    bl_idname = "assistant.paste_image"
    bl_label = "Paste Image"
    bl_options = {"REGISTER"}

    def execute(self, context):
        import base64
        import io

        try:
            # Try to get image from clipboard using PIL
            from PIL import Image, ImageGrab

            # Get image from clipboard
            img = ImageGrab.grabclipboard()

            if img is None:
                self.report({"WARNING"}, "No image in clipboard")
                return {"CANCELLED"}

            if not isinstance(img, Image.Image):
                self.report({"WARNING"}, "Clipboard content is not an image")
                return {"CANCELLED"}

            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            img_bytes = buffer.getvalue()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")

            # Store in window manager for next message
            context.window_manager.assistant_pending_image = img_base64

            # Create Blender Image datablock for preview
            base64_to_blender_image(img_base64, "AssistantPendingImage")

            self.report(
                {"INFO"},
                f"📎 Image pasted ({img.width}x{img.height}px) - will be sent with next message",
            )
            return {"FINISHED"}

        except ImportError:
            self.report(
                {"ERROR"}, "PIL/Pillow not available. Install with: pip install Pillow"
            )
            return {"CANCELLED"}
        except Exception as e:
            self.report({"ERROR"}, f"Failed to paste image: {str(e)}")
            return {"CANCELLED"}


class ASSISTANT_OT_continue(bpy.types.Operator):
    """Insert 'please continue' and send it to the assistant"""

    bl_idname = "assistant.continue_chat"
    bl_label = "Continue"
    bl_options = {"REGISTER"}

    prompt: bpy.props.StringProperty(name="Prompt", default="please continue")

    def execute(self, context):
        # Prevent overlapping runs
        if ASSISTANT_OT_send._is_running:
            self.report({"WARNING"}, "Assistant is already processing. Please wait.")
            return {"CANCELLED"}
        wm = context.window_manager
        wm.assistant_message = self.prompt
        # Invoke the send operator
        try:
            bpy.ops.assistant.send("INVOKE_DEFAULT")
            return {"FINISHED"}
        except Exception as e:
            self.report({"ERROR"}, f"Failed to send: {e}")
            return {"CANCELLED"}


# Chat item property group
class AssistantChatItem(bpy.types.PropertyGroup):
    role: bpy.props.StringProperty(name="Role", default="")
    content: bpy.props.StringProperty(name="Content", default="")
    image_data: bpy.props.StringProperty(
        name="Image Data", default=""
    )  # Base64 encoded image
    # When role == "Tool", store the originating tool name so we can send proper
    # Ollama tool result messages in the next request.
    tool_name: bpy.props.StringProperty(name="Tool Name", default="")


# Chat session property group
class AssistantChatSession(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty(name="Session Name", default="New Chat")
    messages: bpy.props.CollectionProperty(type=AssistantChatItem)
    created: bpy.props.StringProperty(name="Created", default="")


def get_chat_sessions_enum(self, context):
    """Dynamic enum for chat session dropdown."""
    wm = context.window_manager
    items = []
    for i, session in enumerate(wm.assistant_chat_sessions):
        items.append((str(i), session.name, f"Switch to {session.name}"))
    return items if items else [("0", "No Chats", "Create a chat first")]


def register():
    import datetime

    bpy.utils.register_class(AssistantChatItem)
    bpy.utils.register_class(AssistantChatSession)
    bpy.utils.register_class(ASSISTANT_OT_send)
    bpy.utils.register_class(ASSISTANT_OT_stop)
    bpy.utils.register_class(ASSISTANT_OT_clear)
    bpy.utils.register_class(ASSISTANT_OT_new_chat)
    bpy.utils.register_class(ASSISTANT_OT_delete_chat)
    bpy.utils.register_class(ASSISTANT_OT_rename_chat)
    bpy.utils.register_class(ASSISTANT_OT_paste_image)
    bpy.utils.register_class(ASSISTANT_OT_continue)

    # Register chat sessions collection
    bpy.types.WindowManager.assistant_chat_sessions = bpy.props.CollectionProperty(
        type=AssistantChatSession
    )
    bpy.types.WindowManager.assistant_active_chat_index = bpy.props.IntProperty(
        name="Active Chat Index", default=0, min=0
    )
    bpy.types.WindowManager.assistant_active_chat_enum = bpy.props.EnumProperty(
        name="Active Chat",
        description="Select active chat session",
        items=get_chat_sessions_enum,
        update=lambda self, context: setattr(
            context.window_manager,
            "assistant_active_chat_index",
            int(context.window_manager.assistant_active_chat_enum),
        ),
    )
    bpy.types.WindowManager.assistant_message = bpy.props.StringProperty(
        name="Message", default=""
    )
    bpy.types.WindowManager.assistant_pending_image = bpy.props.StringProperty(
        name="Pending Image",
        description="Base64 encoded image to attach to next message",
        default="",
    )

    # Create default chat session if none exist
    try:
        wm = bpy.context.window_manager
        if not wm.assistant_chat_sessions:
            session = wm.assistant_chat_sessions.add()
            session.name = "Chat 1"
            session.created = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            wm.assistant_active_chat_index = 0
            try:
                wm.assistant_active_chat_enum = "0"
            except Exception:
                pass
    except Exception as e:
        print(f"[Assistant] Could not create default chat: {e}")


def unregister():
    del bpy.types.WindowManager.assistant_pending_image
    del bpy.types.WindowManager.assistant_message
    del bpy.types.WindowManager.assistant_active_chat_enum
    del bpy.types.WindowManager.assistant_active_chat_index
    del bpy.types.WindowManager.assistant_chat_sessions

    bpy.utils.unregister_class(ASSISTANT_OT_continue)
    bpy.utils.unregister_class(ASSISTANT_OT_paste_image)
    bpy.utils.unregister_class(ASSISTANT_OT_rename_chat)
    bpy.utils.unregister_class(ASSISTANT_OT_delete_chat)
    bpy.utils.unregister_class(ASSISTANT_OT_new_chat)
    bpy.utils.unregister_class(ASSISTANT_OT_clear)
    bpy.utils.unregister_class(ASSISTANT_OT_stop)
    bpy.utils.unregister_class(ASSISTANT_OT_send)
    bpy.utils.unregister_class(AssistantChatSession)
    bpy.utils.unregister_class(AssistantChatItem)

from dataclasses import dataclass
from typing import List, Dict, Optional, Any, Callable
import json
import threading

@dataclass
class Agent:
    name: str
    description: str
    system_prompt: str

class ContextManager:
    """Manages the 'Fold/Focus' state of the context."""
    
    def __init__(self, tool_manager=None, memory_manager=None):
        self.tool_manager = tool_manager
        self.memory_manager = memory_manager
    
    def get_context_prompt(self, agent_name: str, focus_object_name: Optional[str]) -> str:
        """Generate context string based on agent and focus object."""
        context = []
        
        if focus_object_name:
            context.append(f"FOCUS OBJECT: {focus_object_name}")
            
            # Agent-specific unfolding (Simple Logic)
            context.append(f"FOCUS OBJECT: {focus_object_name}")
            
        return "\n".join(context)

class AgentTools:
    """Provides agentic tools (fuzzy functions) for the assistant."""
    
    def __init__(self, llm_client=None, tool_manager=None, memory_manager=None, message_callback: Optional[Callable] = None, get_model_name: Optional[Callable] = None, execution_queue=None, on_agent_finish: Optional[Callable] = None, scene_watcher=None, session=None):
        self.llm_client = llm_client
        self.tool_manager = tool_manager
        self.memory_manager = memory_manager
        self.message_callback = message_callback
        self.get_model_name = get_model_name
        self.execution_queue = execution_queue
        self.on_agent_finish = on_agent_finish
        self.scene_watcher = scene_watcher
        self.session = session  # Context Sharing
        self.context_manager = ContextManager(tool_manager, memory_manager)
        self.agents: Dict[str, Agent] = self._setup_agents()
        
    def _setup_agents(self) -> Dict[str, Agent]:
        # Import prompts here to avoid top-level circular deps if any
        try:
            from .scene_agent import SCENE_AGENT_PROMPT
        except ImportError:
            SCENE_AGENT_PROMPT = "You are the Scene Agent. Analyze the scene and report findings."

        return {
            "TASK_AGENT": Agent(
                name="TASK_AGENT",
                description="General Purpose Worker Agent. Can execute code, use Blender, search Web, download assets, etc.",
                system_prompt="""You are the Task Agent. Your job is to EXECUTE the task given by the Manager.
                
                PROCESS:
                - PLAN FIRST: If the task is complex, use `task_plan` to break it down.
                - BE CONCISE: Keep your 'thought' field short. Long thoughts cause JSON errors.
                - NO MARKDOWN IN JSON: Do not put code blocks (```) inside the JSON string values.
                
                1. ANALYZE: Understand the specific task.
                2. PLAN: If complex, break down into steps.
                3. EXECUTE: Use tools to achieve the goal.
                4. VERIFY: Check if your actions worked (Read the 'scene_changes' in the Tool Result).
                5. REPORT: Return `expected_changes` JSON summary when done.
                
                RESPONSE FORMAT:
                You MUST return a JSON object with a 'thought' field explaining your reasoning, and either 'tool_call', 'code', or 'expected_changes'.
                Example:
                {
                    "thought": "I need to find the object. I will use get_scene_info...",
                    "tool_call": { ... }
                }

                RULES:
                - BACKGROUND PROCESS: You are running in a non-interactive loop. Use tools to interact with the system.
                - TASK LIST PROTOCOL: **CRITICAL**: Start by calling `task_list`. Do NOT rely solely on the query string. If tasks exist, work through them sequentially. Mark items as IN_PROGRESS or DONE using `task_update`.
                - BATCHING: Do NOT stop after one task. Execute ALL pending tasks in the list until blocked.
                - CODING GUIDELINES:
                    - DEFENSIVE CODING: Check `if obj.name in collection.objects:` before `unlink`. Avoid assumptions about object state.
                    - ERROR HANDLING: If a tool fails, report it in JSON. Do NOT panic and output raw text.
                - COMPLETION PROTOCOL: When satisfied (or all tasks DONE), you MUST call `finish_task` with a summary. This is the only way to report success.
                - VERIFICATION PROTOCOL: Inspect `scene_changes` in the tool result first. If it confirms the action (e.g. 'added': [...]), you are DONE. Do NOT call `get_scene_info` unless the result is ambiguous. 
                - MISSING INFO PROTOCOL: If critical information is missing, use `finish_task` to report the gap and request user input.
                - Use `sdk_help` to learn tool usage.
                - If a tool fails, try a different parameter."""
            ),
            "COMPLETION_AGENT": Agent(
                name="COMPLETION_AGENT",
                description="Specializes in verifying if the user's request is fully satisfied.",
                system_prompt="""You are the Completion Agent. 
                
                - Do NOT guess. If unsure, use `inspect_data` or `get_scene_info` to find the truth.
                - If a check fails, report the specific property that mismatch.
                - When checks pass, calling `finish_task` is MANDATORY.

                CONTEXT SAFETY:
                - Do NOT call `get_object_info(verbose=True)` for everything. Use `get_scene_info` for high-level checks.
                - For deep inspection, use `consult_scene_agent`.
                
                REPORTING FORMAT (When Step 3 is reached):
                1. IF COMPLETE:
                {
                    "expected_changes": {"status": "COMPLETE"}
                }
                
                2. IF INCOMPLETE:
                {
                    "expected_changes": {"status": "INCOMPLETE", "missing": ["X"]}
                }
                
                RULES:
                - CONTEXT SAFETY: **CRITICAL**: Do NOT use `get_object_info(verbose=True)` for multiple objects or validation. It dumps massive data and crashes the session. Use `inspect_data` for specific properties if needed.
                - EVIDENCE REQUIREMENT: Base all decision on concrete tool outputs (e.g. `get_object_info` default mode).
                - CHECK HISTORY: Before searching blindly, READ the 'scene_changes' or 'expected_changes' from the Task Agent's output in the chat history. If the object is listed there, verify it specifically.
                - TASK LIST PROTOCOL: Use `task_list` to see the "Plan of Record". Verify EACH item marked DONE.
                - INCOMPLETION PROTOCOL: If verification fails or data is missing, report as INCOMPLETE with details."""
            ),
            "SCENE_AGENT": Agent(
                name="SCENE_AGENT",
                description="Specializes in researching the scene state to answer queries.",
                system_prompt=SCENE_AGENT_PROMPT
            )
        }
        
    def execute_in_main_thread(self, func, *args, **kwargs):
        """Execute a function on the main thread via the session queue."""
        # Check if we are already on the main thread to avoid deadlock
        if threading.current_thread() is threading.main_thread():
             return func(*args, **kwargs)

        if not self.execution_queue:
            return func(*args, **kwargs)
            
        result_container = {}
        completion_event = threading.Event()
        
        # Support partial binding if queue requires (func, kwargs, result_container, completion_event)
        from functools import partial
        bound_func = partial(func, *args)
        
        self.execution_queue.put((bound_func, kwargs, result_container, completion_event))
        completion_event.wait()
        
        if 'error' in result_container:
            raise result_container['error']
        return result_container.get('result')

    def _run_autonomous_loop(self, role, messages, model_to_use, tools=None, llm_settings=None):
        """Background thread loop for the agent."""
        turn_count = 0
        empty_turns = 0 # Track consecutive empty turns
        from .tools import tool_registry
        
        # Default settings if none provided
        if not llm_settings:
            llm_settings = {
                "temperature": 0.2, # Override default 0.7 for agents
                "n_ctx": 8192
            }
        
        while True:
            turn_count += 1
            if turn_count > 50:
                print(f"[AgentTools] [AGENT: {role}] Turn limit exceeded ({turn_count}).")
                if self.message_callback:
                    self.execute_in_main_thread(self.message_callback, "system", f"Agent {role} turn limit exceeded.")
                
                # Signal Manager of timeout
                timeout_result = {
                    "status": "ERROR",
                    "agent": role,
                    "error": f"Agent {role} exceeded turn limit ({turn_count})."
                }

                def finish_callback():
                    if hasattr(self, "on_agent_finish") and self.on_agent_finish:
                         self.on_agent_finish(timeout_result)
                
                self.execute_in_main_thread(finish_callback)
                return

            print(f"[AgentTools] [AGENT: {role}] Turn {turn_count}...")
            
            try:
                # DYNAMIC PROMPT INJECTION
                # Recover tools + behavior + protocol
                base_prompt = self.agents[role].system_prompt
                
                # 1. SDK Support (What tools can I call via execute_code?)
                # Retrieve universe for this role, then filter out what is enabled as MCP
                allowed_tools = self.tool_manager.get_allowed_tools_for_role(role)
                # Default to all tools enabled in the request that are allowed for this role
                enabled_mcp = {t["function"]["name"] for t in tools if t["function"]["name"] in allowed_tools} if tools else set()
                
                # Filter out low-level Polyhaven tools for everyone to enforce smart wrappers
                polyhaven_hiding = {"search_polyhaven_assets", "get_polyhaven_asset_info"}
                enabled_mcp = enabled_mcp - polyhaven_hiding

                # For sub-agents (non-ASSISTANT), remove task management tools so they only use 'task_list' and 'finish_task'
                if role != "ASSISTANT":
                    task_tools_to_hide = {"task_add", "task_update", "task_complete", "task_clear", "task_plan"}
                    enabled_mcp = enabled_mcp - task_tools_to_hide
                
                # Add Task Completion Protocol for TASK_AGENT role
                if role == "TASK_AGENT":
                    task_completion_section = """
                    TASK COMPLETION PROTOCOL:
                    - After completing all required actions, CALL `finish_task` with a list of expected changes and a brief summary.
                    - This will signal the calling agent that the work is done and return the normal response structure.
                    - Do NOT manually mark individual tasks as DONE; rely on `finish_task`.
                    """
                else:
                    task_completion_section = ""
                # Generate the SDK list (Tools allowed but NOT enabled as MCP)
                sdk_section = self.tool_manager.get_system_prompt_hints(enabled_mcp, allowed_tools)
                
                # 2. Behavior (Common rules)
                behavior_section = self.tool_manager.get_common_behavior_prompt()
                
                # 3. Protocol (If applicable - usually via context or memory, but we'll stick to basics for now)
                protocol_section = ""
                if self.session: 
                     # Reuse the session's protocol loading logic or just use the text if available
                    protocol_section = self.session._load_protocol() if hasattr(self.session, '_load_protocol') else "Refer to protocol for behavioral rules."
                
                # 4. MCP Tools (Mini-List)
                # We list just the names to avoid token waste, as the adapter sends the full schema in the 'tools' param.
                mcp_list_str = json.dumps(sorted(list(enabled_mcp)), indent=2)
                mcp_section = f"MCP TOOLS (Primary Interactions):\n{mcp_list_str}"

                # 5. Concurrent Edit Protocol
                concurrent_section = """
NOTE ON SCENE SYNCHRONIZATION:
You are collaborating with a human USER who is also editing the scene live. 
'Scene Changes' logs may reflect USER actions (active deletions, moves, etc.).
**PROTOCOL**:
1. Treat the current scene state as the TRUTH.
2. Do NOT try to 'restore' objects that were deleted unless explicitly asked.
3. If a required object is missing, report it as a blocker rather than silently recreating it."""

                # Add Task Completion Protocol instruction
                
                # Build the full system prompt, including task_completion_section only if non-empty
                full_system_prompt = f"{base_prompt}\n\n{mcp_section}\n\n{sdk_section}\n\n{behavior_section}\n\n{concurrent_section}\n\n{protocol_section}"
                if task_completion_section:
                    full_system_prompt += f"\n\n{task_completion_section}"

                
                
                # Inject into messages if it's the first turn or if we want to reinforce it
                # Ideally, we replace the first 'system' message if it exists, or prepend it
                current_messages = list(messages)
                if current_messages and current_messages[0].get("role") == "system":
                     current_messages[0]["content"] = full_system_prompt
                else:
                     current_messages.insert(0, {"role": "system", "content": full_system_prompt})


                # LLM Request
                # Prepare kwargs from settings
                kwargs = {
                     "stream": True,
                     "tools": tools,
                     "model_path": model_to_use,
                     "messages": current_messages
                }
                
                # Check thinking preference
                prefs = self.preferences.get_preferences()
                if prefs.enforce_json:
                    kwargs["format"] = "json"
                
                # Merge unified settings
                if llm_settings:
                    # Map settings to chat_completion arguments
                    # standard: temperature, n_ctx
                    # adapter-specific (passed via **kwargs handling in adapter? no, adapter takes specific args)
                    # Checking chat_completion signature in Core or Adapter? 
                    # Adapter `chat_completion` accepts `**kwargs`.
                    # We pass the settings dict as kwargs.
                    kwargs.update(llm_settings)
                    
                    # Force agent-specific overrides if needed (e.g. lower temp for tasks)
                    # But user preference should arguably rule. 
                    # For now, let's respect global preference unless it's unset.
                    if "temperature" not in llm_settings:
                        kwargs["temperature"] = 0.2
                else:
                    kwargs["temperature"] = 0.2

                response = self.llm_client.chat_completion(**kwargs)
                
                # CRITICAL: Check for LLM errors (e.g. Ollama offline)
                if response.get("error"):
                    err_msg = response["error"]
                    print(f"[AgentTools] [AGENT: {role}] LLM Error: {err_msg}")
                    if self.message_callback:
                        self.execute_in_main_thread(
                            self.message_callback, 
                            "system", 
                            f"FATAL ERROR: LLM Request Failed: {err_msg}", 
                            name="System"
                        )
                    # Abort loop
                    return
                
                # Context Metrics (Post-Execution)
                
                # Usage extraction (Defer logging until after thinking display)
                usage = response.get("usage", {})

                
                
                # --- CONSOLIDATED PARSING LOGIC ---
                from .core import ResponseParser
                
                # Use core ResponseParser for robust handling
                parsed_tools = ResponseParser.parse(response)
                
                # Extract content/thinking
                raw_content = response.get("content", "") or ""
                native_thinking = ResponseParser.extract_thinking(response)
                
                # Check for legacy JSON in content (if no tool calls)
                legacy_data = {}
                if isinstance(raw_content, str) and (raw_content.strip().startswith("{") or raw_content.strip().startswith("```json")):
                     try:
                         # Use ResponseParser internal helper if we could, but simple load is fine for legacy
                         # Strip markdown if needed
                         json_str = raw_content.strip()
                         if json_str.startswith("```json"): 
                             json_str = json_str[7:].split("```")[0]
                         legacy_data = json.loads(json_str)
                     except: pass
                
                # Construct Normalized Data
                thought = native_thinking or legacy_data.get("thought", "")
                
                # Fallback: If no structured thought, but we have text content that ISN'T just a JSON block, treat as thought
                # This fixes visibility for models like GPT-OSS that output text before tool calls without tags
                if not thought and raw_content:
                    stripped = raw_content.strip()
                    # If it doesn't look like a pure JSON/Code block starter, treat as thought
                    if not (stripped.startswith("```json") or stripped.startswith("{") or stripped.startswith("```python")):
                         thought = raw_content

                data = {
                    "thought": thought,
                    "code": legacy_data.get("code", ""),
                    "tool_calls": [{"name": t.tool, "args": t.args} for t in parsed_tools],
                    "expected_changes": legacy_data.get("expected_changes", []),
                    "finish_summary": legacy_data.get("finish_summary", ""),
                    "reply": legacy_data.get("content", "") # Persist main content/reply
                }
                
                # Re-serialize for consistent downstream processing variables and history
                # This ensures 'content' used in history includes our extracted thought
                content = json.dumps(data)
                usage = response.get("usage")
                
                # --- (Loop logic falls through to check valid data) ---

                # Processing Content
                # Processing Content
                if not content or str(content).strip() == "":
                     # Handle completely empty response
                     print(f"[AgentTools] [AGENT: {role}] Empty Content Received.")
                     # Fall through to empty_turns logic (we populate empty data)
                     data = {}
                else:
                    try:
                        data = json.loads(content)
                    except json.JSONDecodeError:
                        # Fallback: Content is not JSON, but possibly a final Plain Text response.
                        if content and not content.strip().startswith("{"):
                             # Treat as valid text response (Reply)
                             # We use 'reply' field to distinguish from internal thought
                             data = {"reply": content} 
                        else:
                            # Genuine JSON error (e.g. broken JSON)
                            print(f"[AgentTools] [AGENT: {role}] JSON Parse Error. Content: {content[:100]}...")
                            empty_turns += 1
                            
                            if empty_turns >= 5:
                                 if self.message_callback:
                                     self.execute_in_main_thread(self.message_callback, role, "Agent stuck in invalid format loop. Aborting.", name=role)
                                 return
                            
                            if self.message_callback:
                                 # Show the RAW content so user can debug "Invalid Format"
                                 # Truncate for sanity
                                 raw_preview = content
                                 if len(raw_preview) > 500: raw_preview = raw_preview[:500] + "... [Truncated]"
                                 error_msg = f"Agent returned invalid format. Raw Output:\n```json\n{raw_preview}\n```"
                                 self.execute_in_main_thread(self.message_callback, "system", error_msg, name=role)
                                 
                                 # Retry logic remains
                                 self.execute_in_main_thread(self.message_callback, role, "Agent returned invalid format. Retrying...", name=role)
                            messages.append({"role": "user", "content": "Error: Invalid JSON response. Please return valid JSON."})
                            continue
                    
                thought = data.get("thought", "")
                # New: capture plain assistant message as reply if present
                reply = data.get("reply", "")
                code = data.get("code", "")
                tool_call = data.get("tool_call")
                tool_calls = data.get("tool_calls", [])
                expected_changes = data.get("expected_changes")
                finish_summary = data.get("finish_summary")

                # DISPLAY REPLY (If accepted as final text response)
                if reply and self.message_callback:
                     self.execute_in_main_thread(self.message_callback, role, reply, name=role)
                
                # SILENT LOOP PROTECTION
                if not thought and not reply and not code and not tool_call and not tool_calls and not expected_changes:
                     empty_turns += 1
                     print(f"[AgentTools] [AGENT: {role}] Empty turn detected ({empty_turns}/5). Retrying...")
                     
                     if empty_turns >= 5:
                         print(f"[AgentTools] [AGENT: {role}] ABORTING: Too many consecutive empty responses.")
                         if self.message_callback:
                             self.execute_in_main_thread(self.message_callback, role, "Agent stuck in silence. Aborting.", name=role)
                         return
 
                     if self.message_callback:
                         self.execute_in_main_thread(self.message_callback, role, f"Empty Response (Retrying {empty_turns}/5)...", name=role)
                     messages.append({"role": "user", "content": "Error: You returned empty JSON. Please return 'thought' AND ('code' OR 'tool_call' OR 'expected_changes')."})
                     continue
                
                # Reset empty turn counter on valid response
                empty_turns = 0

                # THINKING STALL PROTECTION
                # If agent returns ONLY thought (no code/tool/finish/reply) for multiple turns
                if thought and not reply and not code and not tool_call and not tool_calls and not expected_changes:
                     # Check if we are stalling
                     if not "thought_only_turns" in locals(): thought_only_turns = 0
                     thought_only_turns += 1
                     
                     if thought_only_turns > 8:
                         print(f"[AgentTools] [AGENT: {role}] ABORTING: Agent stuck in thinking loop.")
                         if self.message_callback:
                             self.execute_in_main_thread(self.message_callback, role, "Agent stuck in thinking loop. Aborting.", name=role)
                         return {"status": "ERROR", "error": "Agent stuck in thinking loop."}
                         
                     # Prompt for action
                     print(f"[AgentTools] [AGENT: {role}] Thought only turn ({thought_only_turns}/8). Prompting for action...")
                     messages.append({"role": "assistant", "content": content})
                     messages.append({"role": "user", "content": "System: You provided 'thought' but no action. Please Execute Code, Call a Tool, or Return expected_changes."})
                     continue
                else:
                    thought_only_turns = 0

                print(f"[AgentTools] [AGENT: {role}] Thought: {thought}")
                
                if self.message_callback and thought:
                    # Pass usage statistics if available (so UI can store it)
                    # We pass 'thinking' as role so UI renders it as thinking bubble
                    self.execute_in_main_thread(self.message_callback, "thinking", thought, name=role, usage=usage)
                


                # Execution Priority: Code > Tool Call > Expected Changes
                # Execution Priority: Tool Calls (List) > Legacy Code/ToolCall
                # Normalize everything into a list of executions
                pending_executions = []
                
                # 1. New List Format
                if data.get("tool_calls"):
                    pending_executions.extend(data["tool_calls"])
                
                # 2. Legacy Formats
                if code:
                    pending_executions.append({"name": "execute_code", "args": {"code": code}})
                if tool_call:
                     pending_executions.append(tool_call)
                     
                if pending_executions:
                    messages.append({"role": "assistant", "content": content})
                    
                    # Execute all planned tools
                    for i, t_call in enumerate(pending_executions):
                        t_name = t_call.get("name")
                        t_args = t_call.get("args", {})
                        
                        # Logging
                        print(f"[AgentTools] [AGENT: {role}] Executing '{t_name}' ({i+1}/{len(pending_executions)})...")
                        
                        if self.message_callback:
                             # Status update
                             status_msg = f"Executing {t_name}..."
                             if t_name == "execute_code":
                                 code_preview = t_args.get("code", "").strip()
                                 if code_preview: status_msg = f"Executing code:\n```python\n{code_preview}\n```"
                             elif t_name == "finish_task":
                                 status_msg = f"Finishing Task: {t_args.get('summary', '')}"
                             
                             # Send as Agent message so it is visible (not hidden by system filter)
                             self.execute_in_main_thread(self.message_callback, role, status_msg, name=role)

                        # Execution
                        t_info = tool_registry.get_tool_info(t_name)
                        requires_main = t_info.get("requires_main_thread", True) if t_info else True
                        
                        try:
                            if requires_main:
                                result = self.execute_in_main_thread(tool_registry.execute_tool, t_name, t_args)
                            else:
                                result = tool_registry.execute_tool(t_name, t_args)
                                
                            # Scene Changes
                            if self.scene_watcher:
                                changes = self.execute_in_main_thread(self.scene_watcher.consume_changes)
                                if changes:
                                     if isinstance(result, dict): result["scene_changes"] = changes
                                     else: result = {"output": result, "scene_changes": changes}
                        except Exception as e:
                            result = {"status": "ERROR", "error": str(e)}

                        # Result Reporting
                        result_str = json.dumps(result)
                        print(f"[AgentTools] [AGENT: {role}] Result: {result_str}")
                        
                        # Append Result to History
                        # We append distinct user messages for each result so the LLM sees the sequence
                        messages.append({"role": "user", "content": f"Result ({t_name}): {result_str}"})
                        
                        # UI Feedback
                        if self.message_callback:
                             self.execute_in_main_thread(self.message_callback, "system", f"Result: {result_str}", name=role)
                             
                        # Termination Check (finish_task)
                        if t_name == "finish_task" and isinstance(result, dict) and result.get("status") == "DONE":
                            print(f"[AgentTools] [AGENT: {role}] explicit finish_task called. Exiting loop.")
                            if not result.get("agent"): result["agent"] = role
                            return result
                            
                    # Continue Loop (LLM sees results and decides next step)
                    continue
                
                # FALLBACK (Legacy Block Placeholder)
                elif False and code:
                    pass



                elif expected_changes is not None:
                     print(f"[AgentTools] [AGENT: {role}] Task Finished. Changes: {expected_changes}")
                     
                     if self.message_callback:
                         if finish_summary:
                             self.execute_in_main_thread(self.message_callback, "system", f"Task Completed: {finish_summary}", name=role)
                         else:
                             self.execute_in_main_thread(self.message_callback, role, "Task Finished.", name=role)

                     # Final Result
                     return {
                         "status": "DONE",
                         "agent": role,
                         "expected_changes": expected_changes,
                         "summary": finish_summary or thought
                     }
                    
                else:
                    # Generic text response? Usually shouldn't happen in Agent Mode
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": "continue"})
                    
            except Exception as e:
                print(f"[AgentTools] [AGENT: {role}] Error: {e}")
                if self.message_callback:
                    self.execute_in_main_thread(self.message_callback, "system", f"Agent Error: {e}")
                return 

    def spawn_agent(self, role: str, query: str, focus_object: str = None, dry_run: bool = False):
        """delegate a task to a Worker Agent (Starts Background Thread)."""
        role = role.upper().replace(" ", "_")
        if role not in self.agents:
             valid_roles = list(self.agents.keys())
             return json.dumps({"error": f"Agent role '{role}' not found. Valid roles: {valid_roles}"})
            
        agent = self.agents[role]
        
        # Build Context (User Requirement)
        initial_context = ""
        if self.session and len(self.session.history) > 0:
            # Find the LATEST user message (Dynamic North Star)
            # Iterate backwards to find the most recent "user" or "User" message
            last_user_msg = ""
            for m in reversed(self.session.history):
                msg_role = m.get("role", "")
                if msg_role in ("user", "User", "You"):
                    last_user_msg = m["content"]
                    break
            
            if last_user_msg:
                initial_context = f"\n\n[OVERALL GOAL (NORTH STAR)]: {last_user_msg}"
        
        context_prompt = self.context_manager.get_context_prompt(role, focus_object)
        
        tool_schemas = []
        sdk_hints = ""

        if self.tool_manager:
            from .tools import tool_registry
            import bpy

            # 1. Determine Agent Universe & User Permissions
            universe = self.tool_manager.get_allowed_tools_for_role(role)
            prefs = None
            try:
                # Robustly find preferences (handles Zip vs Extension)
                prefs = bpy.context.preferences.addons[__package__].preferences
            except:
                # Fallback or dev mode
                pass
            
            injected_tools = self.tool_manager.get_enabled_tools_for_role(role, preferences=prefs)
            
            # 4. Generate Tool Schemas
            tool_schemas = self.tool_manager.get_openai_tools(injected_tools)

            # finish_task is now injected via ToolManager (system_tools)
            
            # 5. Generate SDK Hints (For Allowed but Disabled Tools)
            sdk_hints = self.tool_manager.get_system_prompt_hints(
                enabled_tools=injected_tools, 
                allowed_tools=universe
            )
            
            # 6. Get Common Behavior
            behavior_prompt = self.tool_manager.get_common_behavior_prompt()

        # Inject tool definitions into system prompt for robustness
        # We use NAMES only here to avoid duplication if the LLM also sees the 'tools' param.
        tool_names = sorted([t["function"]["name"] for t in tool_schemas])
        tools_text = json.dumps(tool_names, indent=2)
        
        # Define Goal
        if role == "COMPLETION_AGENT":
             goal_text = "Your goal is to verify the clients query is complete. Use your tools."
        else:
             goal_text = "Your goal is to solve the user's query efficiently. Use your tools."

        system_prompt = (
            f"You are the {agent.name}.\n"
            f"{agent.system_prompt}\n\n"
            f"{context_prompt}\n"
            f"{initial_context}\n\n"
            f"{behavior_prompt}\n\n"
            f"MCP TOOLS:\n{tools_text}\n\n"
            f"{sdk_hints}\n\n"
            f"{goal_text}"
        )

        if prefs and getattr(prefs, "debug_mode", False):
            print(f"\n[AgentTools] [AGENT: {agent.name}] System Prompt:\n{'-'*40}\n{system_prompt}\n{'-'*40}\n")
        
        if dry_run:
            return {
                "system_prompt": system_prompt,
                "tool_schemas": tool_schemas,
                "enabled_tools": list(injected_tools)
            }
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        print(f"[AgentTools] [AGENT: {role}] Starting Agent Loop. Query: {query}")
        
        # Determine model - Sync with Session
        model_to_use = "gpt-oss:20b" # Default fallback
        
        # Priority 1: Dynamic Callback (Fresh from Prefs)
        if self.get_model_name:
             try:
                 name = self.get_model_name()
                 if name: model_to_use = name
             except: pass
             
        # Priority 2: Session Static (Backup)
        elif self.session and hasattr(self.session, "model_name"):
            model_to_use = self.session.model_name

        # BLOCKING/SYNC MODE (For Sub-Agents called by Manager)
        # If we are already in a background thread, run synchronously to enable "Context Handoff"
        if threading.current_thread() is not threading.main_thread():
             print(f"[AgentTools] [AGENT: {role}] Running SYNCHRONOUSLY (Recursion).")
             result = self._run_autonomous_loop(role, messages, model_to_use, tool_schemas)
             return result or {"status": "ERROR", "error": "Agent loop returned None"}

        # ASYNC/THREAD MODE (For Top-Level Agents called by UI)
        # Fetch settings globally (Main Thread Safe)
        from .preferences import get_llm_settings
        import bpy
        try:
            llm_settings = get_llm_settings(bpy.context)
        except:
             llm_settings = {}

        def thread_target(r, m, mod, sch, settings):
            res = self._run_autonomous_loop(r, m, mod, sch, settings)
            if res and hasattr(self, "on_agent_finish") and self.on_agent_finish:
                self.execute_in_main_thread(self.on_agent_finish, res)

        t = threading.Thread(target=thread_target, args=(role, messages, model_to_use, tool_schemas, llm_settings), daemon=True)
        t.start()
        
        return json.dumps({"type": "AGENT_STARTED", "role": role})

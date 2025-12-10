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
        return {
            "TASK_AGENT": Agent(
                name="TASK_AGENT",
                description="General Purpose Worker Agent. Can execute code, use Blender, search Web, download assets, etc.",
                system_prompt="""You are the Task Agent. Your job is to EXECUTE the task given by the Manager.
                
                PROCESS:
                1. ANALYZE: Understand the specific task.
                2. PLAN: If complex, break down into steps.
                3. EXECUTE: Use tools to achieve the goal.
                4. VERIFY: Check if your actions worked.
                5. REPORT: Return `expected_changes` JSON summary when done.
                
                RULES:
                - CHECK BEFORE CREATING: Always check if an object or material exists (using `get_scene_info` or `inspect_data`) before trying to create it! Do not create duplicate objects (e.g. Floor, Floor.001) if the goal is to modify the existing one.
                - NO INTERACTIVE QUESTIONS: You are running in a background loop. You CANNOT ask the user questions directly. If you are missing critical information (e.g. file paths, dimensions) that prevents you from proceeding, you must ABORT by calling `finish_task` with a clear summary of what is missing.
                - Use `sdk_help(tool_names=['...'])` to get multiple schemas at once.
                - For PolyHaven, always `search_polyhaven_assets` first to get an ID before downloading.
                - If a tool fails with a specific error (e.g. Resolution not found), try a different parameter instead of giving up."""
            ),
            "COMPLETION_AGENT": Agent(
                name="COMPLETION_AGENT",
                description="Specializes in verifying if the user's request is fully satisfied.",
                system_prompt="""You are the Completion Agent. 
                
                YOUR JOB: Verify if the user's request is satisfied.
                
                PROCESS:
                1. INSPECT: Use tools like `get_scene_info` or `get_object_info` to check the scene.
                2. EVALUATE: Compare scene state against the User's goal.
                3. REPORT: Output the JSON decision.

                RESPONSE FORMAT (Choose One):
                1. IF COMPLETE:
                {
                    "thought": "I have verified X, Y, Z...",
                    "expected_changes": {"status": "COMPLETE"}
                }
                
                2. IF INCOMPLETE:
                {
                    "thought": "Missing X...",
                    "expected_changes": {"status": "INCOMPLETE", "missing": ["X"]}
                }
                
                RULES:
                - You MUST use tools to verify. Do not guess.
                - If you cannot verify, assume INCOMPLETE."""
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

    def _run_autonomous_loop(self, role, messages, model_to_use, tools=None):
        """Background thread loop for the agent."""
        turn_count = 0
        empty_turns = 0 # Track consecutive empty turns
        from .tools import tool_registry
        
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
                enabled_mcp = {t["function"]["name"] for t in tools} if tools else set()
                
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

                full_system_prompt = f"{base_prompt}\n\n{mcp_section}\n\n{sdk_section}\n\n{behavior_section}\n\n{protocol_section}"
                
                # Inject into messages if it's the first turn or if we want to reinforce it
                # Ideally, we replace the first 'system' message if it exists, or prepend it
                current_messages = list(messages)
                if current_messages and current_messages[0].get("role") == "system":
                     current_messages[0]["content"] = full_system_prompt
                else:
                     current_messages.insert(0, {"role": "system", "content": full_system_prompt})


                # LLM Request                # LLM Request
                response = self.llm_client.chat_completion(
                    model_path=model_to_use, 
                    messages=current_messages,
                    format="json",
                    temperature=0.2,
                    tools=tools
                )
                
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
                usage = response.get("usage", {})
                if usage:
                    prompt_tokens = usage.get("prompt_eval_count", 0)
                    completion_tokens = usage.get("eval_count", 0)
                    total_dur = usage.get("total_duration", 0) / 1e9 # ns to s
                    
                    print(f"[AgentTools] [AGENT: {role}] Usage: {prompt_tokens} input tokens, {completion_tokens} output tokens ({total_dur:.2f}s)")
                    
                    if self.message_callback:
                         self.execute_in_main_thread(
                            self.message_callback, 
                            "system", 
                            f"Context: {prompt_tokens} tokens | Output: {completion_tokens} tokens | Time: {total_dur:.2f}s", 
                            name=role
                        )
                
                content = response.get("content", "{}")
                tool_calls = response.get("tool_calls", []) or response.get("message", {}).get("tool_calls", [])

                # MCP Tool Logic
                if (not content or content == "{}") and tool_calls:
                     try:
                         call = tool_calls[0]
                         fn_name = call.get("function", {}).get("name")
                         fn_args = call.get("function", {}).get("arguments")
                         
                         if isinstance(fn_args, str):
                             fn_args = json.loads(fn_args)
                             
                         new_data = {"thought": f"Invoking MCP tool: {fn_name}"}
                         
                         if fn_name == "execute_code":
                             new_data["code"] = fn_args.get("code", "")
                         elif fn_name == "finish_task":
                             new_data["expected_changes"] = fn_args.get("expected_changes", [])
                             if "summary" in fn_args:
                                 new_data["thought"] += f" Summary: {fn_args['summary']}"
                         else:
                             new_data["tool_call"] = {"name": fn_name, "args": fn_args}
                             
                         content = json.dumps(new_data)
                     except Exception as e:
                         print(f"[AgentTools] [AGENT: {role}] Failed to parse MCP tool call: {e}")

                # Processing Content
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    print(f"[AgentTools] [AGENT: {role}] JSON Parse Error. Content: {content[:100]}...")
                    if self.message_callback:
                         self.execute_in_main_thread(self.message_callback, role, f"Raw Response (Invalid JSON): {content}", name=role)
                    messages.append({"role": "user", "content": "Error: Invalid JSON response. Please return valid JSON."})
                    continue
                    
                thought = data.get("thought", "")
                code = data.get("code", "")
                tool_call = data.get("tool_call")
                expected_changes = data.get("expected_changes")
                
                # SILENT LOOP PROTECTION
                if not thought and not code and not tool_call and not expected_changes:
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
                # If agent returns ONLY thought (no code/tool/finish) for multiple turns
                if thought and not code and not tool_call and not expected_changes:
                     # Check if we are stalling
                     if not "thought_only_turns" in locals(): thought_only_turns = 0
                     thought_only_turns += 1
                     
                     if thought_only_turns > 3:
                         print(f"[AgentTools] [AGENT: {role}] ABORTING: Agent stuck in thinking loop.")
                         if self.message_callback:
                             self.execute_in_main_thread(self.message_callback, role, "Agent stuck in thinking loop. Aborting.", name=role)
                         return {"status": "ERROR", "error": "Agent stuck in thinking loop."}
                         
                     # Prompt for action
                     print(f"[AgentTools] [AGENT: {role}] Thought only turn ({thought_only_turns}/3). Prompting for action...")
                     messages.append({"role": "assistant", "content": content})
                     messages.append({"role": "user", "content": "System: You provided 'thought' but no action. Please Execute Code, Call a Tool, or Return expected_changes."})
                     continue
                else:
                    thought_only_turns = 0

                print(f"[AgentTools] [AGENT: {role}] Thought: {thought}")
                
                if self.message_callback and thought:
                    # Pass usage statistics if available (so UI can store it)
                    # We need to access 'usage' which was captured earlier (Line 213)
                    self.execute_in_main_thread(self.message_callback, "thinking", thought, name=role, usage=usage if "usage" in locals() else None)
                
                # Execution Priority: Code > Tool Call > Expected Changes
                if code:
                    print(f"[AgentTools] [AGENT: {role}] Executing Code via Queue...")
                    if self.message_callback:
                        self.execute_in_main_thread(self.message_callback, role, "Executing code...", name=role)
                        
                    result = self.execute_in_main_thread(tool_registry.execute_tool, "execute_code", {"code": code})
                    
                    # Scene Watcher Update (Merged into Result)
                    if self.scene_watcher:
                        changes = self.execute_in_main_thread(self.scene_watcher.consume_changes)
                        if changes:
                             if isinstance(result, dict):
                                 result["scene_changes"] = changes
                             else:
                                 result = {"output": result, "scene_changes": changes}
                             print(f"[AgentTools] [AGENT: {role}] SCENE UPDATES merged: {changes}")

                    result_str = json.dumps(result)
                    print(f"[AgentTools] [AGENT: {role}] Execution Result: {result_str}")
                    
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"Result: {result_str}"})

                    # Show Result in UI
                    if self.message_callback:
                         self.execute_in_main_thread(self.message_callback, "system", f"Result: {result_str}", name=role)


                elif tool_call:
                    t_name = tool_call.get("name")
                    t_args = tool_call.get("args", {})
                    print(f"[AgentTools] [AGENT: {role}] Executing MCP Tool '{t_name}' via Queue...")
                    
                    if self.message_callback:
                        self.execute_in_main_thread(self.message_callback, role, f"Executing {t_name}...", name=role)

                    # SPECIAL CASE: spawn_agent must run in THIS thread to allow recursion/blocking
                    # All other tools run in Main Thread via Queue
                    # UPDATE: Now using metadata from registry
                    t_info = tool_registry.get_tool_info(t_name)
                    requires_main = t_info.get("requires_main_thread", True)
                    
                    if requires_main:
                         result = self.execute_in_main_thread(tool_registry.execute_tool, t_name, t_args)
                    else:
                         # Safe to run in background (e.g. spawn_agent, finish_task)
                         result = tool_registry.execute_tool(t_name, t_args)

                     
                    # Scene Watcher Update (Merged into Result)
                    if self.scene_watcher:
                        changes = self.execute_in_main_thread(self.scene_watcher.consume_changes)
                        if changes:
                             if isinstance(result, dict):
                                 result["scene_changes"] = changes
                             else:
                                 result = {"output": result, "scene_changes": changes}
                             print(f"[AgentTools] [AGENT: {role}] SCENE UPDATES merged: {changes}")

                    result_str = json.dumps(result)
                    
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"Result: {result_str}"})

                    if self.message_callback:
                         self.execute_in_main_thread(self.message_callback, "system", f"Result: {result_str}", name=role)


                elif expected_changes:
                    # Final Answer
                    print(f"[AgentTools] [AGENT: {role}] Finished.")
                    if self.message_callback:
                        self.execute_in_main_thread(self.message_callback, role, "Task finished. Reporting back.", name=role)
                    
                    final_result = {
                        "status": "DONE",
                        "agent": role,
                        "expected_changes": expected_changes,
                        "summary": thought
                    }
                    
                    def finish_callback():
                        if hasattr(self, "on_agent_finish"):
                             self.on_agent_finish(final_result)
                    
                    self.execute_in_main_thread(finish_callback)
                    # Helper return for synchronous calls
                    return final_result
                    
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
        tools_text = json.dumps(tool_schemas, indent=2)
        
        system_prompt = (
            f"You are the {agent.name}.\n"
            f"{agent.system_prompt}\n\n"
            f"{context_prompt}\n"
            f"{initial_context}\n\n"
            f"{behavior_prompt}\n\n"
            f"MCP TOOLS:\n{tools_text}\n\n"
            f"{sdk_hints}\n\n"
            "Your goal is to solve the user's query efficiently. "
            "Use your tools."
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
        t = threading.Thread(target=self._run_autonomous_loop, args=(role, messages, model_to_use, tool_schemas), daemon=True)
        t.start()
        
        return json.dumps({"type": "AGENT_STARTED", "role": role})

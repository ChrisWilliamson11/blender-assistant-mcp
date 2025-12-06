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
            if agent_name == "MODELER":
                context.append("(Mesh data unfolded: stats, topology, vertex groups, attributes - Use `inspect_data` for details)")
            elif agent_name == "ANIMATOR":
                context.append("(Animation data unfolded: f-curves, action, NLA tracks - Use `inspect_data` for details)")
            
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
                
                CAPABILITIES (You have full access):
                - Python Scripting (`execute_code`) & Blender API
                - Web Search & Scrape
                - PolyHaven & Sketchfab Asset Download
                - Vision (Viewport Capture)
                
                PROCESS:
                1. ANALYZE: Understand the specific task.
                2. PLAN: If complex, break down into steps.
                3. EXECUTE: Use tools to achieve the goal.
                4. VERIFY: Check if your actions worked.
                5. REPORT: Return `expected_changes` JSON summary when done.
                
                CRITICAL RULES:
                - If a tool fails (e.g., arguments mismatch), YOU MUST use `assistant_help` to check the tool signature immediately. DO NOT GUESS AGAIN.
                - For PolyHaven/Sketchfab: Use `get_polyhaven_asset_info` (or equivalent) to check availability (resolutions/formats) BEFORE downloading.
                - If stuck, think step-by-step why it failed."""
            ),
            "COMPLETION_AGENT": Agent(
                name="COMPLETION_AGENT",
                description="Specializes in verifying if the user's request is fully satisfied.",
                system_prompt="""You are the Completion Agent. Your ONLY job is to check if the user's original request has been fully completed.
                
                INPUT:
                - User Query
                - Current Scene State (via `get_scene_info`)
                - Execution History
                
                OUTPUT:
                - If complete: Return `{"thought": "Task is complete because...", "expected_changes": {"status": "COMPLETE"}}`
                - If incomplete: Return `{"thought": "Task is incomplete because...", "expected_changes": {"status": "INCOMPLETE", "missing": ["..."]}}`
                
                CRITICAL RULES:
                - Output VALID JSON only.
                - Verify efficiently using `get_scene_info` or `execute_code` (for complex checks).
                - If a tool fails, check usage with `assistant_help`."""
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
                # LLM Request
                response = self.llm_client.chat_completion(
                    model_path=model_to_use, 
                    messages=messages,
                    format="json",
                    temperature=0.2,
                    tools=tools
                )
                
                content = response.get("content", "{}")
                tool_calls = response.get("tool_calls", []) or response.get("message", {}).get("tool_calls", [])

                # Native Tool Logic
                if (not content or content == "{}") and tool_calls:
                     try:
                         call = tool_calls[0]
                         fn_name = call.get("function", {}).get("name")
                         fn_args = call.get("function", {}).get("arguments")
                         
                         if isinstance(fn_args, str):
                             fn_args = json.loads(fn_args)
                             
                         new_data = {"thought": f"Invoking native tool: {fn_name}"}
                         
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
                         print(f"[AgentTools] [AGENT: {role}] Failed to parse native tool call: {e}")

                # Processing Content
                try:
                    data = json.loads(content)
                except json.JSONDecodeError:
                    messages.append({"role": "user", "content": "Error: Invalid JSON response. Please return valid JSON."})
                    continue
                    
                thought = data.get("thought", "")
                code = data.get("code", "")
                tool_call = data.get("tool_call")
                expected_changes = data.get("expected_changes")
                
                print(f"[AgentTools] [AGENT: {role}] Thought: {thought}")
                
                if self.message_callback and thought:
                    self.execute_in_main_thread(self.message_callback, role, thought, name=role)
                
                # Execution Priority: Code > Tool Call > Expected Changes
                if code:
                    print(f"[AgentTools] [AGENT: {role}] Executing Code via Queue...")
                    if self.message_callback:
                        self.execute_in_main_thread(self.message_callback, role, "Executing code...", name=role)
                        
                    result = self.execute_in_main_thread(tool_registry.execute_tool, "execute_code", {"code": code})
                    result_str = json.dumps(result)
                    print(f"[AgentTools] [AGENT: {role}] Execution Result: {result_str}")
                    
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"Result: {result_str}"})

                    # Show Result in UI
                    if self.message_callback:
                         self.execute_in_main_thread(self.message_callback, "system", f"Result: {result_str}", name=role)

                    # Scene Watcher Update
                    if self.scene_watcher:
                        changes = self.execute_in_main_thread(self.scene_watcher.consume_changes)
                        if changes:
                             print(f"[AgentTools] [AGENT: {role}] SCENE UPDATES: {changes}")
                             messages.append({"role": "system", "content": f"SCENE UPDATES (Objects Modified/Created): {changes}"})

                elif tool_call:
                    t_name = tool_call.get("name")
                    t_args = tool_call.get("args", {})
                    print(f"[AgentTools] [AGENT: {role}] Executing Native Tool '{t_name}' via Queue...")
                    
                    if self.message_callback:
                        self.execute_in_main_thread(self.message_callback, role, f"Executing {t_name}...", name=role)

                    result = self.execute_in_main_thread(tool_registry.execute_tool, t_name, t_args)
                    result_str = json.dumps(result)
                    
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": f"Result: {result_str}"})

                    if self.message_callback:
                         self.execute_in_main_thread(self.message_callback, "system", f"Result: {result_str}", name=role)

                    if self.scene_watcher:
                        changes = self.execute_in_main_thread(self.scene_watcher.consume_changes)
                        if changes:
                             print(f"[AgentTools] [AGENT: {role}] SCENE UPDATES: {changes}")
                             messages.append({"role": "system", "content": f"SCENE UPDATES (Objects Modified/Created): {changes}"})

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
                    return
                    
                else:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": "continue"})
                    
            except Exception as e:
                print(f"[AgentTools] [AGENT: {role}] Error: {e}")
                if self.message_callback:
                    self.execute_in_main_thread(self.message_callback, "system", f"Agent Error: {e}")
                return 

    def consult_specialist(self, role: str, query: str, focus_object: str = None) -> str:
        """delegate a task to a Worker Agent (Starts Background Thread)."""
        role = role.upper()
        if role not in self.agents:
            if role in ["MODELER", "ANIMATOR", "RIGGER", "SHADERS", "CODER", "WEB", "RESEARCH"]:
                 role = "TASK_AGENT"
            else:
                 return json.dumps({"error": f"Specialist '{role}' not found."})
            
        agent = self.agents[role]
        
        # Build Context (User Requirement)
        initial_context = ""
        if self.session and len(self.session.history) > 0:
            # Find the first user message
            first_user_msg = next((m["content"] for m in self.session.history if m["role"] == "user"), "")
            initial_context = f"\n\n[OVERALL GOAL (NORTH STAR)]: {first_user_msg}"
        
        context_prompt = self.context_manager.get_context_prompt(role, focus_object)
        
        tool_schemas = []
        sdk_hints = ""

        if self.tool_manager:
            from .tools import tool_registry
            import bpy

            # 1. Determine Agent Universe
            universe = self.tool_manager.get_allowed_tools_for_role(role)
            
            # 2. Determine User Preferences (MCP Active Tools)
            prefs = None
            try:
                addon_name = "blender_assistant_mcp"
                if addon_name in bpy.context.preferences.addons:
                    prefs = bpy.context.preferences.addons[addon_name].preferences
            except: pass
            
            active_mcp_set = self.tool_manager.get_enabled_tools(prefs)
            
            # 3. Intersect + Force CORE TOOLS (execute_code, assistant_help)
            core_tools = {"execute_code", "assistant_help"}
            injected_tools = universe.intersection(active_mcp_set.union(core_tools))
            
            # 4. Generate Tool Schemas
            tool_schemas = self.tool_manager.get_openai_tools(injected_tools)

            # Inject 'finish_task'
            tool_schemas.append({
                "type": "function",
                "function": {
                    "name": "finish_task",
                    "description": "Call this tool when you have completed your objective.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "expected_changes": {
                                "type": "array", 
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["expected_changes"]
                    }
                }
            })
            
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
            f"AVAILABLE TOOLS (Native):\n{tools_text}\n\n"
            f"{sdk_hints}\n\n"
            "Your goal is to solve the user's query efficiently. "
            "Use your tools."
        )

        # Debug: Print system prompt if enabled
        if prefs and getattr(prefs, "debug_mode", False):
            print(f"\n[AgentTools] [AGENT: {agent.name}] System Prompt:\n{'-'*40}\n{system_prompt}\n{'-'*40}\n")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        print(f"[AgentTools] [AGENT: {role}] Starting Background Thread. Query: {query}")
        
        # Determine model
        model_to_use = "qwen2.5-coder:7b"
        if self.get_model_name:
            try:
                 name = self.get_model_name()
                 if name: model_to_use = name
            except: pass

        t = threading.Thread(target=self._run_autonomous_loop, args=(role, messages, model_to_use, tool_schemas), daemon=True)
        t.start()
        
        return json.dumps({"type": "AGENT_STARTED", "role": role})

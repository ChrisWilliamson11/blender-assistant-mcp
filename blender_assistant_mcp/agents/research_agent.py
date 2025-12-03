import json
import logging
from typing import List, Dict, Any, Optional
from .. import ollama_adapter

logger = logging.getLogger(__name__)

class ResearchAgent:
    """
    Agent that performs iterative research using available tools (RAG, Web, Memory).
    Follows a Plan -> Search -> Integrate -> Reflect loop.
    """

    def __init__(self, model_name: str = "qwen2.5-coder:7b"):
        self.model_name = model_name
        self.max_iters = 3
        
        # Initialize MemoryManager
        from ..memory import MemoryManager
        self.memory_manager = MemoryManager()

    def research(self, topic: str) -> str:
        """
        Perform deep research on a topic.
        """
        print(f"[ResearchAgent] Starting research on: {topic}")
        
        # 1. Gather initial context (Abstracts)
        abstracts = self.memory_manager.get_abstracts()
        if abstracts:
            memory_context = "Prior Context (Abstracts):\n" + "\n".join([f"- {a}" for a in abstracts])
        else:
            memory_context = "No prior memory context." 
        
        accumulated_info = []
        current_focus = topic
        
        for i in range(self.max_iters):
            print(f"[ResearchAgent] Iteration {i+1}/{self.max_iters}: {current_focus}")
            
            # --- PLANNING ---
            plan = self._plan(current_focus, memory_context, accumulated_info)
            if not plan.get("tools"):
                print("[ResearchAgent] No tools needed, skipping search.")
                break
                
            # --- EXECUTION ---
            results = self._execute_plan(plan)
            
            # --- INTEGRATION ---
            # We don't necessarily need a full LLM integration step here, 
            # we can just append the raw results to our accumulated info
            # and let the Reflection step decide if it's enough.
            accumulated_info.extend(results)
            
            # --- REFLECTION ---
            decision = self._reflect(topic, accumulated_info)
            
            if decision["enough"]:
                print("[ResearchAgent] Information sufficient.")
                break
            
            if not decision.get("new_focus"):
                print("[ResearchAgent] No new focus suggested, stopping.")
                break
                
            current_focus = decision["new_focus"]
            
        # --- FINAL SYNTHESIS ---
        final_answer = self._synthesize(topic, accumulated_info)
        return final_answer

    def _plan(self, topic: str, memory_context: str, accumulated_info: List[str]) -> Dict:
        """
        Decide which tools to use.
        """
        # Simple prompt to ask LLM for tool calls
        # We can list available tools: rag_search, web_search, search_memory
        
        prompt = f"""You are a Research Planner.
Goal: Find information about "{topic}".
Available Tools:
- rag_search(query): Search Blender documentation.
- web_search(query): Search the internet.
- fetch_webpage(url): Read content from a specific URL.
- search_memory(query): Search user's past conversations.

Current Knowledge:
{json.dumps(accumulated_info[:5], indent=2)}... (truncated)

Return a JSON object with a list of tool calls to execute.
Format:
{{
    "tools": [
        {{"name": "rag_search", "args": {{"query": "..."}}}},
        {{"name": "web_search", "args": {{"query": "..."}}}}
    ]
}}
If no more info is needed, return {{"tools": []}}.
"""
        response = ollama_adapter.chat_completion(
            model_path=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
            format="json" # Force JSON output
        )
        
        try:
            content = response.get("content", "{}")
            return json.loads(content)
        except Exception as e:
            print(f"[ResearchAgent] Planning failed: {e}")
            return {"tools": []}

    def _execute_plan(self, plan: Dict) -> List[str]:
        results = []
        for tool_call in plan.get("tools", []):
            name = tool_call.get("name")
            args = tool_call.get("args", {})
            
            print(f"[ResearchAgent] Executing {name}({args})")
            
            # Map tool names to SDK calls
            # We need to find the tool in the SDK. 
            # SDK tools are namespaced (e.g. blender.get_scene_info).
            # But rag_search and web_search might be in different namespaces.
            # For now, we'll hardcode the mapping or search the registry.
            
            try:
                # Direct mapping for known tools
                output = ""
                if name == "rag_search":
                    # Assuming rag_search is available in the 'rag' namespace or similar
                    # Actually, rag_search is in rag_tools.py, usually registered under 'rag' or 'search'?
                    # Let's assume 'rag.search' or similar. 
                    # Wait, the tool name in tool_registry is just the function name usually.
                    # Let's try to find it via SDK help or just assume 'rag_search' is the name.
                    # In tool_registry.py, tools are stored by name.
                    # SDK.execute_tool takes (category, tool_name, **kwargs).
                    # We don't know the category easily.
                    # But we can use tool_registry directly if we import it, OR use sdk to find it.
                    
                    # Better: The SDK exposes namespaces. 
                    # Let's try to find the tool in the registry.
                    from ..tools import tool_registry
                    
                    # Find category for tool
                    category = None
                    for cat, tools in tool_registry._TOOLS.items():
                        if name in tools:
                            category = cat
                            break
                    
                    if category:
                        output = tool_registry.execute_tool(category, name, **args)
                    else:
                        output = f"Tool {name} not found."
                        
                elif name == "web_search":
                     # Similar lookup
                    from ..tools import tool_registry
                    category = None
                    for cat, tools in tool_registry._TOOLS.items():
                        if name in tools:
                            category = cat
                            break
                    if category:
                        output = tool_registry.execute_tool(category, name, **args)
                    else:
                        output = f"Tool {name} not found."

                elif name == "fetch_webpage":
                     # Similar lookup
                    from ..tools import tool_registry
                    category = None
                    for cat, tools in tool_registry._TOOLS.items():
                        if name in tools:
                            category = cat
                            break
                    if category:
                        output = tool_registry.execute_tool(category, name, **args)
                    else:
                        output = f"Tool {name} not found."

                elif name == "search_memory":
                     # Similar lookup
                    from ..tools import tool_registry
                    category = None
                    for cat, tools in tool_registry._TOOLS.items():
                        if name in tools:
                            category = cat
                            break
                    if category:
                        output = tool_registry.execute_tool(category, name, **args)
                    else:
                        output = f"Tool {name} not found."
                else:
                    output = f"Unknown tool: {name}"

                results.append(f"Result from {name}({args}):\n{output}")
                
            except Exception as e:
                results.append(f"Error executing {name}: {e}")
                
        return results

    def _reflect(self, topic: str, accumulated_info: List[str]) -> Dict:
        """
        Check if we have enough info.
        """
        info_text = "\n\n".join(accumulated_info)
        prompt = f"""You are a Research Critic.
Topic: "{topic}"
Gathered Information:
{info_text}

Do we have enough information to comprehensively answer the topic?
If yes, return {{"enough": true}}.
If no, return {{"enough": false, "new_focus": "what specific sub-topic to research next"}}.
Return JSON.
"""
        response = ollama_adapter.chat_completion(
            model_path=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256,
            format="json"
        )
        try:
            return json.loads(response.get("content", "{}"))
        except:
            return {"enough": True}

    def _synthesize(self, topic: str, accumulated_info: List[str]) -> str:
        """
        Generate final answer.
        """
        info_text = "\n\n".join(accumulated_info)
        prompt = f"""You are a Research Assistant.
Topic: "{topic}"
Gathered Information:
{info_text}

Please write a comprehensive answer to the topic based on the gathered information.
Cite sources if available in the text.
"""
        response = ollama_adapter.chat_completion(
            model_path=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048
        )
        return response.get("content", "Failed to generate summary.")

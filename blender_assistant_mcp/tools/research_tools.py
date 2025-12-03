from ..agents.research_agent import ResearchAgent

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = ResearchAgent()
    return _agent

def research_topic(topic: str) -> str:
    """Perform deep, iterative research on a topic using RAG, Web, and Memory.
    
    Args:
        topic: The research topic or question (e.g. "How to create procedural buildings in Blender?")
        
    Returns:
        A comprehensive answer based on the research.
    """
    agent = get_agent()
    return agent.research(topic)

def register():
    from . import tool_registry
    
    tool_registry.register_tool(
        name="research_topic",
        func=research_topic,
        description="Perform deep, iterative research on a topic using RAG, Web, and Memory. Use this for complex questions that require multiple steps or external info.",
        input_schema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The research topic or question"
                }
            },
            "required": ["topic"]
        },
        category="Research"
    )

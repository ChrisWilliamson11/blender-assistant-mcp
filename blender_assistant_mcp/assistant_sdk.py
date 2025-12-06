"""Auto-generated SDK wrapper for all registered tools.

This module provides the `assistant_sdk` object which allows calling any
registered tool via namespaced methods (e.g., `assistant_sdk.blender.create_object(...)`).

The SDK is dynamically generated from the tool registry, so tools are automatically
available regardless of whether they're MCP-exposed or not.
"""

from typing import Any, Dict
from .tools import tool_registry


class _Namespace:
    """Dynamic namespace for a tool category."""
    
    def __init__(self, category: str, tools: list):
        self._category = category
        self._tools = {tool['name']: tool for tool in tools}
        
        # Dynamically create methods for each tool
        for tool_name in self._tools.keys():
            setattr(self, tool_name, self._make_tool_method(tool_name))
    
    def _make_tool_method(self, tool_name: str):
        """Create a method that calls tool_registry.execute_tool."""
        tool = self._tools[tool_name]
        
        def tool_method(*args, **kwargs) -> Dict[str, Any]:
            # Handle positional arguments by mapping them to schema properties
            if args:
                # Get parameter names from schema
                schema = tool.get('inputSchema', {})
                props = schema.get('properties', {})
                # Since we don't have explicit order in standard JSON schema, 
                # we'll use the order of keys in 'properties' as a best effort,
                # which usually matches definition order in Python dicts.
                param_names = list(props.keys())
                
                if len(args) > len(param_names):
                    raise TypeError(f"{tool_name}() takes {len(param_names)} positional arguments but {len(args)} were given")
                
                for i, arg in enumerate(args):
                    param_name = param_names[i]
                    if param_name in kwargs:
                        raise TypeError(f"{tool_name}() got multiple values for argument '{param_name}'")
                    kwargs[param_name] = arg
            
            result = tool_registry.execute_tool(tool_name, kwargs)
            
            # CRITICAL FIX: If tool returned an error, RAISE it so execute_code catches it.
            # Otherwise, the agent thinks it succeeded because it got a dict back.
            if isinstance(result, dict) and 'error' in result:
                raise RuntimeError(f"Tool {tool_name} failed: {result['error']}")
                
            return result
        
        # Set docstring from tool description
        tool_method.__doc__ = tool.get('description', f'{tool_name} tool')
        tool_method.__name__ = tool_name
        
        return tool_method


class AssistantSDK:
    """Auto-generated SDK providing access to all registered tools via namespaces.
    
    Usage:
        assistant_sdk.blender.create_object(type='CUBE', name='MyCube')
        assistant_sdk.web.search(query='cats')
        assistant_sdk.memory.remember_fact(fact='User likes blue')
    
    Each namespace corresponds to a tool category from the registry.
    """
    
    def __init__(self):
        self._namespaces = {}
        self._rebuild()
    
    def _rebuild(self):
        """Rebuild namespaces from current tool registry."""
        from collections import defaultdict
        
        # Group tools by category
        tools_by_category = defaultdict(list)
        for tool in tool_registry.get_tools_list():
            category = tool.get('category', 'other')
            tools_by_category[category].append(tool)
        
        # Create namespace for each category
        for category, tools in tools_by_category.items():
            namespace_name = category.lower().replace(' ', '_').replace('-', '_')
            namespace = _Namespace(category, tools)
            setattr(self, namespace_name, namespace)
            self._namespaces[namespace_name] = namespace
    
    def help(self, namespace: str = None) -> Dict[str, Any]:
        """Get help for namespaces or specific namespace.
        
        Args:
            namespace: Optional namespace name (e.g., 'blender', 'web')
            
        Returns:
            Dictionary of available namespaces/methods
        """
        if namespace:
            if namespace in self._namespaces:
                ns = self._namespaces[namespace]
                return {
                    'namespace': namespace,
                    'category': ns._category,
                    'tools': list(ns._tools.keys())
                }
            else:
                return {'error': f'Namespace "{namespace}" not found',
                       'available': list(self._namespaces.keys())}
        else:
            return {
                'namespaces': list(self._namespaces.keys()),
                'usage': 'assistant_sdk.<namespace>.<tool_name>(**kwargs)'
            }


# Global singleton instance
_assistant_sdk = None


def get_assistant_sdk() -> AssistantSDK:
    """Get the global AssistantSDK instance.
    
    Returns:
        AssistantSDK: The singleton SDK instance
    """
    global _assistant_sdk
    if _assistant_sdk is None:
        _assistant_sdk = AssistantSDK()
    return _assistant_sdk

"""MCP (Model Context Protocol) tool registration"""

from collections.abc import Callable


def register_tool(fn: Callable, name: str = '', description: str = '') -> None:
    """Register a Python function as an MCP tool"""

def unregister_tool(name: str) -> None:
    """Unregister an MCP tool"""

def list_tools() -> list[str]:
    """List all registered Python MCP tools"""

def tool(name: str = '', description: str = '') -> object:
    """Decorator to register a function as an MCP tool"""

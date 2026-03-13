"""MCP (Model Context Protocol) tool registration"""

from collections.abc import Callable


def register_tool(fn: Callable, name: str = '', description: str = '') -> None:
    """Register a Python function as an MCP tool"""

def unregister_tool(name: str) -> None:
    """Unregister an MCP tool"""

def list_tools() -> list[str]:
    """List all registered shared capabilities/tools"""

def list_python_tools() -> list[str]:
    """List Python-provided MCP tools registered through this module"""

def describe_tools() -> list:
    """Describe all registered shared capabilities/tools"""

def list_resources() -> list[str]:
    """List all registered MCP resources"""

def read_resource(uri: str) -> list:
    """Read one registered MCP resource"""

def call_tool(name: str, args: object | None = None) -> object:
    """Invoke a registered shared capability/tool"""

def tool(name: str = '', description: str = '') -> object:
    """Decorator to register a function as an MCP tool"""

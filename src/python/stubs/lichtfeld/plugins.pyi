"""Plugin system - management, capabilities, panels, and settings"""

from collections.abc import Callable


API_VERSION: str = '1.0'

FEATURES: list = ...

def discover() -> object:
    """Discover plugins in ~/.lichtfeld/plugins/"""

def load(name: str) -> bool:
    """Load plugin"""

def unload(name: str) -> bool:
    """Unload plugin"""

def reload(name: str) -> bool:
    """Reload plugin"""

def load_all() -> object:
    """Load all user-enabled plugins"""

def list_loaded() -> object:
    """List loaded plugins"""

def start_watcher() -> None:
    """Start file watcher"""

def stop_watcher() -> None:
    """Stop file watcher"""

def get_state(name: str) -> object:
    """Get plugin state"""

def get_error(name: str) -> object:
    """Get plugin error"""

def get_traceback(name: str) -> object:
    """Get plugin error traceback"""

def install(url: str, auto_load: bool = True, transport: str = 'archive') -> str:
    """Install from GitHub URL"""

def update(name: str) -> bool:
    """Update plugin"""

def uninstall(name: str) -> bool:
    """Uninstall plugin"""

def search(query: str) -> object:
    """Search plugin registry"""

def install_from_registry(plugin_id: str, version: str = '', auto_load: bool = True, transport: str = 'archive') -> str:
    """Install plugin from registry"""

def check_updates() -> object:
    """Check for plugin updates"""

def register_capability(name: str, handler: Callable, description: str = '', schema: dict | None = None, plugin_name: str | None = None, requires_gui: bool = True) -> None:
    """
    Register a capability (handler signature: def handler(args: dict, ctx: PluginContext) -> dict)
    """

def unregister_capability(name: str) -> bool:
    """Unregister a capability"""

def invoke(name: str, args: dict = {}) -> object:
    """Invoke a capability by name"""

def has_capability(name: str) -> bool:
    """Check if a capability is registered"""

def list_capabilities() -> list:
    """List all registered capabilities"""

def settings(plugin_name: str) -> object:
    """Get settings object for a plugin"""

def create(name: str) -> str:
    """Create a new plugin from template (returns path to created plugin)"""

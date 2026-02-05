# Plugin Developer Workflow

This guide covers the plugin development workflow using a portable LichtFeld Studio distribution.

## Distribution Contents

The portable build includes everything needed for plugin development:

```
dist/
├── bin/
│   ├── LichtFeld-Studio      # Main application
│   └── uv                    # Package manager for plugin dependencies
├── lib/python/
│   ├── lichtfeld.abi3.so     # Python bindings module
│   ├── lfs_plugins/          # Plugin infrastructure
│   └── lichtfeld/            # Type stubs for IDE support
└── share/doc/LichtFeld-Studio/
    ├── PYTHON_API.md         # Complete API reference
    ├── plugin-system.md      # Architecture documentation
    └── plugin-use-cases.md   # Example implementations
```

## Creating a Plugin

### 1. Scaffold a New Plugin

From the app's Python console or a startup script:

```python
import lichtfeld as lf

path = lf.plugins.create("my_plugin")
print(f"Plugin created at: {path}")
```

This creates the following structure:

```
~/.lichtfeld/plugins/my_plugin/
├── pyproject.toml           # Plugin manifest
├── __init__.py              # Entry point (on_load/on_unload hooks)
└── panels/
    ├── __init__.py
    └── main_panel.py        # Example UI panel
```

### 2. Plugin Manifest

Edit `pyproject.toml` to configure your plugin:

```toml
[project]
name = "my_plugin"
version = "1.0.0"
description = "My awesome plugin"
authors = [{name = "Your Name"}]
dependencies = [
    "numpy>=1.20.0",
    "pillow>=9.0.0",
]

[tool.lichtfeld]
auto_start = true
hot_reload = true
min_lichtfeld_version = "1.0.0"
```

### 3. Entry Point

Edit `__init__.py`:

```python
import lichtfeld as lf
from .panels.main_panel import MainPanel

_classes = [MainPanel]

def on_load():
    """Called when plugin is loaded."""
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("my_plugin loaded")

def on_unload():
    """Called when plugin is unloaded."""
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("my_plugin unloaded")
```

### 4. UI Panel

Edit `panels/main_panel.py`:

```python
from lfs_plugins.types import Panel

class MainPanel(Panel):
    label = "My Plugin"
    space = "SIDE_PANEL"
    order = 200

    def __init__(self):
        self.counter = 0

    def draw(self, layout):
        layout.label("Hello from my_plugin!")

        if layout.button("Click Me"):
            self.counter += 1

        layout.label(f"Count: {self.counter}")
```

## Development Workflow

### Loading and Testing

```python
import lichtfeld as lf

# Discover all plugins
lf.plugins.discover()

# Load your plugin
lf.plugins.load("my_plugin")

# Enable hot reload (watches for file changes)
lf.plugins.start_watcher()
```

With hot reload enabled:
1. Edit your plugin code
2. Save the file
3. Plugin automatically reloads

### Manual Reload

```python
lf.plugins.reload("my_plugin")
```

### Checking Plugin State

```python
state = lf.plugins.get_state("my_plugin")
print(state)  # PluginState.ACTIVE, PluginState.ERROR, etc.

# If there's an error
error = lf.plugins.get_error("my_plugin")
traceback = lf.plugins.get_traceback("my_plugin")
```

### Unloading

```python
lf.plugins.unload("my_plugin")
```

## Adding Dependencies

Plugins have isolated virtual environments. Dependencies are specified in `pyproject.toml` under `[project].dependencies` and installed via `uv sync`.

```toml
[project]
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "torch>=2.0.0",
]
```

On plugin load, dependencies are installed to `~/.lichtfeld/plugins/my_plugin/.venv/`.

## IDE Setup

### Type Stubs for Autocompletion

The distribution includes type stubs in `lib/python/lichtfeld/`. Configure your IDE to find them:

**VS Code** (`settings.json`):
```json
{
    "python.analysis.extraPaths": [
        "/path/to/dist/lib/python"
    ]
}
```

**PyCharm**:
1. Right-click `lib/python/` directory
2. Mark Directory as → Sources Root

### Copying Stubs to Plugin

Alternatively, copy stubs into your plugin for portability:

```bash
cp -r dist/lib/python/lichtfeld ~/.lichtfeld/plugins/my_plugin/typings/
```

Then in your IDE, add `typings/` to the Python path.

## Plugin APIs

### Core Management

```python
lf.plugins.discover()              # Find plugins in ~/.lichtfeld/plugins/
lf.plugins.load("name")            # Load a plugin
lf.plugins.unload("name")          # Unload a plugin
lf.plugins.reload("name")          # Reload a plugin
lf.plugins.load_all()              # Load all auto_start plugins
lf.plugins.list_loaded()           # List active plugins
```

### Hot Reload

```python
lf.plugins.start_watcher()         # Monitor plugin files for changes
lf.plugins.stop_watcher()          # Stop monitoring
```

### UI Panels

```python
lf.register_class(MyPanel)       # Register panel, operator, or menu
lf.unregister_class(MyPanel)     # Unregister
```

### Persistent Settings

```python
settings = lf.plugins.settings("my_plugin")
settings.set("key", value)
value = settings.get("key", default=None)
settings.update({"key1": val1, "key2": val2})
settings.delete("key")
```

Settings are stored in `~/.lichtfeld/plugins/my_plugin/settings.json`.

### Cross-Plugin Capabilities

Register a capability:
```python
lf.plugins.register_capability(
    name="my_feature",
    handler=my_handler_function,
    description="What this does",
    schema={"properties": {...}, "required": [...]},
    plugin_name="my_plugin",
)
```

Invoke another plugin's capability:
```python
result = lf.plugins.invoke("other_feature", {"arg1": value})
```

Query capabilities:
```python
lf.plugins.has_capability("name")
lf.plugins.list_capabilities()
```

## Installing from GitHub

```python
# Install from GitHub URL
lf.plugins.install("https://github.com/user/my-lichtfeld-plugin")

# Install from plugin registry
lf.plugins.install_from_registry("plugin_id", version="1.0.0")

# Check for updates
lf.plugins.check_updates()

# Update a plugin
lf.plugins.update("my_plugin")
```

## Distribution

### Sharing Your Plugin

Package your plugin directory:
```bash
cd ~/.lichtfeld/plugins
zip -r my_plugin.zip my_plugin/
```

Users install by extracting to their `~/.lichtfeld/plugins/` directory.

### Plugin Structure Requirements

Minimum required files:
```
my_plugin/
├── pyproject.toml   # Required: manifest
└── __init__.py      # Required: entry point with on_load/on_unload
```

## Debugging

### Logging

```python
lf.log.debug("Debug message")    # Only shown with --log-level debug
lf.log.info("Info message")      # Default level
lf.log.warn("Warning message")
lf.log.error("Error message")
```

### Error Handling

```python
state = lf.plugins.get_state("my_plugin")
if state == lf.plugins.PluginState.ERROR:
    print(lf.plugins.get_error("my_plugin"))
    print(lf.plugins.get_traceback("my_plugin"))
```

## Further Reading

- [PYTHON_API.md](PYTHON_API.md) - Complete API reference
- [plugin-system.md](plugin-system.md) - Architecture and internals
- [plugin-use-cases.md](plugin-use-cases.md) - Real-world examples

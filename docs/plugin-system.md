# LichtFeld Plugin System

A Python-based plugin system for LichtFeld Studio with per-plugin virtual environments, hot reload support, and dependency isolation.

## Architecture

```
~/.lichtfeld/
├── plugins/                          # Plugin directory
│   ├── simple_plugin/
│   │   ├── pyproject.toml            # Manifest (required — discovery trigger)
│   │   ├── .venv/                    # Isolated virtual environment
│   │   └── __init__.py               # Entry point (on_load, on_unload)
│   └── pytorch_plugin/
│       ├── pyproject.toml            # Manifest (required)
│       ├── .venv/                    # Isolated virtual environment
│       └── __init__.py
└── venv/                             # Global venv (existing)
```

## Plugin Manifest (pyproject.toml)

Every plugin **must** have a `pyproject.toml` with a `[tool.lichtfeld]` section — its existence is how LichtFeld discovers plugins.

```toml
[project]
name = "my_plugin"
version = "1.0.0"
description = "Plugin description"
authors = [{name = "Author Name"}]
dependencies = [
    "some-package>=1.0.0",
]

[tool.lichtfeld]
auto_start = true
hot_reload = true
min_lichtfeld_version = "1.0.0"
```

## Dependencies

All dependencies are declared in `[project].dependencies` and resolved via `uv sync`. The installer runs `uv sync --project <plugin_dir>` which creates the plugin's `.venv` and installs everything.

For plugins that need packages from custom indexes (e.g., PyTorch CUDA wheels), add `[tool.uv.index]` and `[tool.uv.sources]` sections to the same `pyproject.toml`:

```toml
[project]
name = "my-pytorch-plugin"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = ["torch>=2.6.0"]

[tool.lichtfeld]
auto_start = true

[[tool.uv.index]]
name = "pytorch-cu124"
url = "https://download.pytorch.org/whl/cu124"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu124" }]
```

## Core Components

### Phase 1 - Plugin System (`scripts/lfs_plugins/`)

| File | Purpose |
|------|---------|
| `plugin.py` | PluginInfo, PluginInstance, PluginState dataclasses |
| `errors.py` | PluginError, PluginLoadError, PluginDependencyError exceptions |
| `installer.py` | Per-plugin venv creation and dependency installation via uv |
| `manager.py` | PluginManager singleton with discover/load/unload/reload |
| `watcher.py` | File polling for hot reload support |
| `__init__.py` | Package exports |

### C++ Bindings (`src/python/lfs/`)

| File | Purpose |
|------|---------|
| `py_plugins.hpp` | Header for plugin bindings |
| `py_plugins.cpp` | nanobind bindings exposing `lichtfeld.plugins` submodule |

## Plugin States

```
UNLOADED → INSTALLING → LOADING → ACTIVE
                ↓           ↓
              ERROR       ERROR
```

- `UNLOADED` - Plugin discovered but not loaded
- `INSTALLING` - Installing dependencies into plugin venv
- `LOADING` - Importing plugin module
- `ACTIVE` - Plugin loaded and running
- `ERROR` - Error during install/load
- `DISABLED` - Manually disabled by user

## Usage

### Python API

```python
from lfs_plugins import PluginManager, PluginState

mgr = PluginManager.instance()

# Discover available plugins
plugins = mgr.discover()
for p in plugins:
    print(f"{p.name} v{p.version}")

# Load a plugin
mgr.load("colmap")

# Check state
state = mgr.get_state("colmap")
assert state == PluginState.ACTIVE

# Hot reload
mgr.reload("colmap")

# Unload
mgr.unload("colmap")

# Load all auto_start plugins
mgr.load_all()

# Hot reload file watcher
mgr.start_watcher()  # Start watching for file changes
mgr.stop_watcher()   # Stop watcher
```

### Via lichtfeld Module

```python
import lichtfeld as lf

plugins = lf.plugins.discover()
lf.plugins.load("colmap")
lf.plugins.reload("colmap")
lf.plugins.unload("colmap")
lf.plugins.load_all()
lf.plugins.start_watcher()
lf.plugins.stop_watcher()
```

## Writing a Plugin

### Minimal Plugin Structure

```
~/.lichtfeld/plugins/my_plugin/
├── pyproject.toml
└── __init__.py
```

### Entry Point (`__init__.py`)

```python
"""My Plugin for LichtFeld Studio."""

import lichtfeld as lf

def on_load():
    """Called when plugin loads."""
    # Register panels, callbacks, etc.
    pass

def on_unload():
    """Called when plugin unloads."""
    # Cleanup resources
    pass
```

### Registering a GUI Panel

```python
import lichtfeld as lf
from lfs_plugins.types import Panel

class MyPanel(Panel):
    label = "My Panel"
    space = "SIDE_PANEL"
    order = 10

    def __init__(self):
        self.value = 0

    def draw(self, layout):
        layout.label("Hello from plugin!")
        if layout.button("Click me"):
            self.value += 1
        layout.label(f"Clicked: {self.value}")

_classes = [MyPanel]

def on_load():
    for cls in _classes:
        lf.register_class(cls)

def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
```

## COLMAP Plugin

The COLMAP plugin (`~/.lichtfeld/plugins/colmap/`) provides Structure-from-Motion reconstruction:

### Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Manifest with pycolmap dependency |
| `utils.py` | ColmapConfig, ReconstructionResult dataclasses |
| `features.py` | SIFT feature extraction |
| `matching.py` | Feature matching (exhaustive/sequential/vocab_tree/spatial) |
| `reconstruction.py` | Incremental SfM |
| `runner.py` | ColmapJob background thread with progress tracking |
| `pipeline.py` | Synchronous and async pipeline entry points |
| `panels/reconstruction.py` | GUI panel for reconstruction workflow |

### Usage

```python
# After plugin is loaded
import colmap

# Synchronous
result = colmap.run_pipeline("path/to/images")
print(f"Reconstructed {result.num_images} images, {result.num_points} points")

# Asynchronous with callbacks
job = colmap.run_pipeline_async(
    "path/to/images",
    on_progress=lambda stage, pct, msg: print(f"{stage}: {pct}% - {msg}"),
    on_complete=lambda result: print(f"Done: {result.success}"),
)

# Cancel if needed
job.cancel()
```

## Testing

```bash
# Run plugin system tests
PYTHONPATH="scripts:build" ./build/vcpkg_installed/x64-linux/tools/python3/python3.12 \
    -m pytest tests/python/test_plugin_system.py -v
```

## Features

- **Plugin discovery** in `~/.lichtfeld/plugins/`
- **Per-plugin virtual environments** with isolated dependencies
- **Dependency installation** via uv package manager
- **Hot reload** via file watcher (polling)
- **Plugin lifecycle hooks** (`on_load`, `on_unload`)
- **State tracking** with error reporting
- **C++ bindings** via nanobind for use from lichtfeld module

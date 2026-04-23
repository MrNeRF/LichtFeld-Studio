# LichtFeld Plugin System

LichtFeld Studio plugins are Python packages discovered from `~/.lichtfeld/plugins/`. They can add panels, operators, hooks, tools, capabilities, and plugin-local dependencies.

## Filesystem layout

```text
~/.lichtfeld/
└── plugins/
    └── my_plugin/
        ├── pyproject.toml
        ├── __init__.py
        ├── .venv/                  # Created by the CLI scaffold or during dependency setup
        ├── pyrightconfig.json      # Created by the CLI scaffold
        ├── .vscode/                # Created by the CLI scaffold
        └── panels/
            ├── __init__.py
            ├── main_panel.py
            ├── main_panel.rml      # Scaffolded for v1; optional to customize
            └── main_panel.rcss     # Scaffolded sibling stylesheet
```

Discovery is manifest-driven: a plugin is discovered because it has a `pyproject.toml` with a `[tool.lichtfeld]` section.

## Core pieces

Current implementation lives in:

| Path | Purpose |
|---|---|
| `src/python/lfs_plugins/` | Python-side plugin manager, installer, watcher, registry, settings, and templates |
| `src/python/lfs/py_plugins.cpp` | `lichtfeld.plugins` bindings |
| `src/python/lfs/py_ui*.cpp` | Unified panel and UI bindings |
| `src/python/plugin_runner.cpp` | CLI `plugin` subcommand runner |
| `src/core/argument_parser.cpp` | CLI parsing for `LichtFeld-Studio plugin ...` |

## Lifecycle

Plugin states:

```text
UNLOADED -> INSTALLING -> LOADING -> ACTIVE
                \            \
                 \            -> ERROR
                  -> ERROR
```

Available states:

- `UNLOADED`
- `INSTALLING`
- `LOADING`
- `ACTIVE`
- `ERROR`
- `DISABLED`

Inspect them from Python:

```python
import lichtfeld as lf

state = lf.plugins.get_state("my_plugin")
error = lf.plugins.get_error("my_plugin")
traceback = lf.plugins.get_traceback("my_plugin")
```

## Scaffolding

Two creation paths exist and they are intentionally different:

### `lf.plugins.create(name)`

Creates the v1 source package scaffold:

```text
pyproject.toml
__init__.py
panels/__init__.py
panels/main_panel.py
panels/main_panel.rml
panels/main_panel.rcss
```

The generated panel still starts as immediate-mode `draw(ui)` content. The retained files are there so plugin authors can move into hybrid UI without reshaping the package later.

### `LichtFeld-Studio plugin create <name>`

Creates the same source files and then adds:

- `.venv/`
- `.vscode/settings.json`
- `.vscode/launch.json`
- `pyrightconfig.json`

This is the convenience path for local development.

### Why `.rml` and `.rcss` are generated in v1

v1 intentionally ships a hybrid-ready scaffold. `main_panel.py` is still the first file you edit, but the sibling `main_panel.rml` and `main_panel.rcss` are already wired up with an `#im-root` mount point for embedded immediate widgets.

## Compatibility contract

The v1 plugin system is strict and does not preserve the old compatibility fields. Every plugin manifest must declare:

```toml
[tool.lichtfeld]
hot_reload = true
plugin_api = ">=1,<2"
lichtfeld_version = ">=0.4.2"
required_features = []
```

Rules:

- `plugin_api` targets the public plugin API contract.
- `lichtfeld_version` targets the host application/runtime version.
- `required_features` is a list of optional host features your plugin depends on.
- Legacy `min_lichtfeld_version` / `max_lichtfeld_version` fields are removed in v1 and rejected.

The runtime exposes the current host contract through:

- `lf.PLUGIN_API_VERSION`
- `lf.plugins.API_VERSION`
- `lf.plugins.FEATURES`

## Unified panel model

`lf.ui.Panel` is the only public panel base class. It supports three common levels of complexity:

| Level | What you add | Typical files |
|---|---|---|
| Immediate | `draw(self, ui)` only | `main_panel.py` |
| Mixed | `style`, `height_mode`, `on_update()`, `on_mount()`, `on_bind_model()` | `main_panel.py` |
| Hybrid | `template` plus optional embedded `draw(ui)` | `main_panel.py`, `main_panel.rml`, `main_panel.rcss` |

### Default retained shells

If a panel uses retained features and leaves `template` empty, LichtFeld selects a shell automatically:

- `FLOATING` -> `rmlui/floating_window.rml`
- `STATUS_BAR` -> `rmlui/status_bar_panel.rml`
- other retained panel spaces -> `rmlui/docked_panel.rml`

Built-in template aliases:

- `builtin:docked-panel`
- `builtin:floating-window`
- `builtin:status-bar`

### Styling path

Use these layers in order:

1. Start with `draw(ui)` and the immediate widget API.
2. Add `style` when you want retained shell tweaks without introducing a custom template.
3. Add `template = str(Path(...))` and a sibling `.rcss` when you need custom DOM structure or a full stylesheet.

Retained template rules:

- plugin-local `template` values should be absolute paths
- `style` is RCSS text, not a filename
- a sibling `.rcss` file is loaded automatically for a plugin-local `.rml`
- a sibling `.theme.rcss` file is loaded automatically for palette-dependent overrides
- include `<div id="im-root"></div>` in the template when you want embedded `draw(ui)` content

## Dependency isolation

Plugin dependencies live in `[project].dependencies` inside the plugin's `pyproject.toml`.

LichtFeld creates per-plugin virtual environments with its bundled Python and `uv`, using:

1. `uv venv <plugin>/.venv --python <bundled_python> --no-managed-python --no-python-downloads`
2. `uv sync --project <plugin> --python <plugin>/.venv/.../python --no-managed-python --no-python-downloads`

This keeps plugin packages isolated while ensuring they run against the same Python runtime as the app.

## Management surfaces

### CLI

The `plugin` subcommand currently supports:

```text
LichtFeld-Studio plugin create <name>
LichtFeld-Studio plugin check <name>
LichtFeld-Studio plugin list
```

The CLI is intentionally narrow. It handles scaffolding, validation, and discovery listing.

### Python API

The richer management surface is exposed through `lichtfeld.plugins`:

```python
import lichtfeld as lf

lf.plugins.discover()
lf.plugins.load("my_plugin")
lf.plugins.reload("my_plugin")
lf.plugins.unload("my_plugin")
lf.plugins.load_all()
lf.plugins.start_watcher()
lf.plugins.stop_watcher()
lf.plugins.install("owner/repo")
lf.plugins.update("my_plugin")
lf.plugins.search("viewer")
lf.plugins.install_from_registry("plugin_id")
lf.plugins.check_updates()
```

## Validation and debugging

Quick checks:

```bash
LichtFeld-Studio plugin check my_plugin
LichtFeld-Studio plugin list
```

Runtime logging:

```python
import lichtfeld as lf

lf.log.debug("debug")
lf.log.info("info")
lf.log.warn("warn")
lf.log.error("error")
```

## Where to go next

- [docs/plugins/getting-started.md](plugins/getting-started.md) for the learning path
- [docs/plugins/examples/README.md](plugins/examples/README.md) for progressive examples
- [docs/plugins/api-reference.md](plugins/api-reference.md) for exact APIs

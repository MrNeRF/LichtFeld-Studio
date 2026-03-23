# Plugin Developer Workflow

This guide explains the current plugin scaffolding workflow and the intended UI progression from a single `draw(ui)` file to full retained RML/RCSS.

## Scaffold paths

There are two supported ways to create a plugin:

### CLI scaffold

```bash
LichtFeld-Studio plugin create my_plugin
```

This creates the minimal plugin package and also adds local development helpers:

```text
~/.lichtfeld/plugins/my_plugin/
├── .venv/
├── .vscode/
│   ├── launch.json
│   └── settings.json
├── pyproject.toml
├── pyrightconfig.json
├── __init__.py
└── panels/
    ├── __init__.py
    ├── main_panel.py
    ├── main_panel.rml
    └── main_panel.rcss
```

Useful companion commands:

```bash
LichtFeld-Studio plugin check my_plugin
LichtFeld-Studio plugin list
```

### Python scaffold

```python
import lichtfeld as lf

path = lf.plugins.create("my_plugin")
print(path)
```

This creates only the plugin package itself:

```text
~/.lichtfeld/plugins/my_plugin/
├── pyproject.toml
├── __init__.py
└── panels/
    ├── __init__.py
    ├── main_panel.py
    ├── main_panel.rml
    └── main_panel.rcss
```

## Why the scaffold now creates `.rml` and `.rcss`

The v1 scaffold is intentionally hybrid-ready. The generated panel still starts as a simple immediate-mode `draw(ui)` panel, but it also includes a sibling RML/RCSS shell so authors can move into retained layout without reshaping the package later:

```python
import lichtfeld as lf


class MainPanel(lf.ui.Panel):
    id = "my_plugin.main_panel"
    label = "My Plugin"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB

    def draw(self, ui):
        ui.heading("Hello")
        ui.text_disabled("Start simple and keep state on self.")
```

The default scaffold keeps `draw(ui)` as the content source and mounts it into `<div id="im-root"></div>` inside `main_panel.rml`. You can ignore the retained files until you need them, but they are already wired correctly for v1.

## Styling progression

### 1. Immediate-only panel

Stay in `main_panel.py` when you only need widgets, layout composition, and panel-local state.

Typical tools:

- `draw(self, ui)`
- `button_styled()`
- `row()`, `column()`, `box()`, `grid_flow()`
- `push_style_var()` and `push_style_color()` for local widget overrides

### 2. Mixed panel with inline RCSS

When you want retained shell styling but still want Python to generate the content, add `style`, `height_mode`, or retained hooks on the same panel class:

```python
class StatusPanel(lf.ui.Panel):
    id = "my_plugin.status"
    label = "Status"
    space = lf.ui.PanelSpace.STATUS_BAR
    height_mode = lf.ui.PanelHeightMode.CONTENT
    style = """
body.status-bar-panel { padding: 0 12dp; }
#im-root .im-label { color: #f3c96d; font-weight: bold; }
"""

    def draw(self, ui):
        ui.label("Ready")
```

Notes:

- `style` is inline RCSS text, not a file path.
- `STATUS_BAR` automatically gets the status-bar shell.
- `draw(ui)` still provides the actual panel content.

### 3. Hybrid panel with template and stylesheet

Add a template when you need custom DOM structure, model binding, or direct document event handling:

```python
from pathlib import Path


class HybridPanel(lf.ui.Panel):
    id = "my_plugin.hybrid"
    label = "Hybrid"
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT

    def draw(self, ui):
        ui.text_disabled("Rendered into #im-root")
```

Important details:

- Use an absolute path for plugin-local `template` values.
- A sibling `main_panel.rcss` is scaffolded and loaded automatically with `main_panel.rml`.
- Keep `<div id="im-root"></div>` in the template if you want embedded `draw(ui)` content.

## Development loop

Load and reload from the app:

```python
import lichtfeld as lf

lf.plugins.discover()
lf.plugins.load("my_plugin")
lf.plugins.reload("my_plugin")
lf.plugins.start_watcher()
```

Inspect failures:

```python
state = lf.plugins.get_state("my_plugin")
error = lf.plugins.get_error("my_plugin")
traceback = lf.plugins.get_traceback("my_plugin")
```

Unload when needed:

```python
lf.plugins.unload("my_plugin")
```

## Dependencies

Declare plugin dependencies in `pyproject.toml`:

```toml
[project]
dependencies = [
    "numpy>=1.26",
]
```

Plugin environments are isolated per plugin. The CLI scaffold creates `.venv` immediately. The Python scaffold creates only the source files; dependency installation still happens through the plugin system when the plugin is loaded.

## Compatibility manifest

The v1 release requires an explicit compatibility contract in `[tool.lichtfeld]`:

```toml
[tool.lichtfeld]
hot_reload = true
plugin_api = ">=1,<2"
lichtfeld_version = ">=0.4.2"
required_features = []
```

Rules:

- `plugin_api` targets the public plugin API surface, not the plugin's own package version.
- `lichtfeld_version` targets the host application/runtime version.
- `required_features` lists optional host features the plugin needs. Inspect the runtime surface with `lf.plugins.API_VERSION`, `lf.plugins.FEATURES`, or `lf.PLUGIN_API_VERSION`.
- `LichtFeld-Studio plugin check` validates these fields and rejects incompatible plugins before load time.
- Legacy `min_lichtfeld_version` / `max_lichtfeld_version` fields are removed in v1 and are not supported.

## IDE support

`LichtFeld-Studio plugin create` also writes:

- `.vscode/settings.json`
- `.vscode/launch.json`
- `pyrightconfig.json`

Those files point the editor at the plugin venv and the LichtFeld typings.

If you scaffold through `lf.plugins.create()`, configure your IDE manually or run the CLI scaffold path for new plugins when you want a ready-made editor setup.

## Recommended reading order

1. Read [docs/plugins/getting-started.md](plugins/getting-started.md).
2. Walk through [docs/plugins/examples/README.md](plugins/examples/README.md).
3. Use [docs/plugins/api-reference.md](plugins/api-reference.md) for exact panel, hook, and widget APIs.

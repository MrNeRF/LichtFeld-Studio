# Python Development Guide

[Python docs](README.md) · [Application API](api-reference.md) · [Plugin and UI API](plugin-reference.md)

The native `lichtfeld` module runs inside LichtFeld Studio's embedded Python
runtime: use the integrated Python console, an app-run script, or a plugin.
The `.pyi` SDK files provide editor completion; they are not an importable
implementation for a system Python interpreter. Use `lf.help()` and
`help(lf.Tensor.linspace)` in the app to inspect the running build.

LichtFeld Studio plugins extend the application with panels, operators, tools, signals, and capabilities. Plugins live in `~/.lichtfeld/plugins/` and are just Python packages with a small manifest and entrypoint.

## On this page

- [Learning path](#learning-path)
- [Quick start](#quick-start)
- [Panels](#panels)
- [UI Hooks and Menus](#ui-hooks-and-menus)
- [Operators](#operators)
- [Toolbar Tools](#toolbar-tools)
- [Properties](#properties)
- [Scene Access](#scene-access)
- [Signals](#signals)
- [Capabilities](#capabilities)
- [Training Hooks](#training-hooks)
- [Plugin Runtime and Dependencies](#plugin-runtime-and-dependencies)
- [Hot Reload & Debugging](#hot-reload--debugging)
- [IDE Setup](#ide-setup)
- [Installing & Publishing](#installing--publishing)

## Learning path

Read the examples in this order:

| Step | Goal | Example |
|---|---|---|
| 1 | Pure immediate-mode panel with `draw(ui)` only | [`examples/01_draw_only.py`](examples/01_draw_only.py) |
| 2 | Add shell and styling without rewriting `draw(ui)`; use periodic updates only for animation-like UI | [`examples/02_status_bar_mixed.py`](examples/02_status_bar_mixed.py) |
| 3 | Build a dirty-policy retained panel with an RML data model and `RuntimeState` subscriptions | [Reactive retained panels](#reactive-retained-panels) |
| 4 | Build a full hybrid panel with template, RCSS, data model, DOM hooks, and embedded `draw(ui)` | [`examples/03_hybrid_plugin/`](examples/03_hybrid_plugin/) |
| 5 | Explore focused feature demos | [`examples/README.md`](examples/README.md) |
| 6 | See an end-to-end multi-file plugin | [`examples/full_plugin/`](examples/full_plugin/) |

The key idea is that `lf.ui.Panel` is one public base class that scales from the smallest `draw(ui)` panel to full retained/hybrid UI. You do not need to switch APIs or rewrite the panel body when you add advanced features.

## Quick start

### Plugin directory structure

```text
~/.lichtfeld/plugins/my_plugin/
├── pyproject.toml       # Plugin manifest (required)
├── __init__.py          # Entry point with on_load/on_unload (required)
├── panels/
│   ├── __init__.py
│   ├── main_panel.py
│   ├── main_panel.rml   # Scaffolded for v1; optional to customize
│   └── main_panel.rcss  # Scaffolded sibling stylesheet
├── operators/           # Optional
│   └── my_operator.py
└── icons/               # Optional PNG icons for custom tools
    └── my_icon.png
```

### Scaffold with CLI or Python

Create a plugin from the command line when you also want a venv and editor config:

```bash
LichtFeld-Studio plugin create my_plugin
LichtFeld-Studio plugin check my_plugin
LichtFeld-Studio plugin list
```

Create a plugin from Python when you only want the source package:

```python
import lichtfeld as lf

path = lf.plugins.create("my_plugin")
print(path)
```

Important scaffold behavior:

- `lf.plugins.create()` writes `pyproject.toml`, `__init__.py`, `panels/__init__.py`, `panels/main_panel.py`, `panels/main_panel.rml`, and `panels/main_panel.rcss`.
- `LichtFeld-Studio plugin create` writes the same source files and also adds `.venv/`, `.vscode/`, and `pyrightconfig.json`.
- The scaffold is hybrid-ready, but you can ignore the retained files until you actually need custom DOM or standalone RCSS.

That is intentional. Most plugins should still start by editing `draw(ui)` in `main_panel.py`; v1 just removes the later migration work when you decide to add a custom template.

### `pyproject.toml`

Every plugin needs `[project]` metadata and a `[tool.lichtfeld]` section:

```toml
[project]
name = "my_plugin"
version = "0.1.0"
description = "What this plugin does"
authors = [{name = "Your Name"}]
dependencies = []

[tool.lichtfeld]
hot_reload = true
plugin_api = ">=1,<2"
lichtfeld_version = ">=0.4.2"
required_features = []
```

Notes:

- `name`, `version`, and `description` are required.
- `hot_reload` is required.
- `plugin_api`, `lichtfeld_version`, and `required_features` are required in v1.
- Legacy `min_lichtfeld_version` / `max_lichtfeld_version` fields are removed and rejected.
- Plugin-local Python dependencies go in `project.dependencies`.
- Inspect the current host contract from Python with `lf.PLUGIN_API_VERSION`, `lf.plugins.API_VERSION`, and `lf.plugins.FEATURES`.

### `__init__.py`

Your entrypoint must define `on_load()` and `on_unload()`:

```python
import lichtfeld as lf
from .panels.main_panel import MainPanel

_classes = [MainPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("my_plugin loaded")


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("my_plugin unloaded")
```

## Panels

Panels are the main UI surface for most plugins. The same `lf.ui.Panel` class supports both immediate-mode and retained/hybrid UI.

### Step 1: start with `draw(ui)`

This is the smallest useful panel:

```python
import lichtfeld as lf


class HelloPanel(lf.ui.Panel):
    id = "hello_world.main_panel"
    label = "Hello World"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 200

    def __init__(self):
        self._clicks = 0

    def draw(self, ui):
        ui.heading("Hello from my plugin")
        ui.text_disabled("This panel uses only draw(ui).")

        if ui.button_styled(f"Greet ({self._clicks})", "primary"):
            self._clicks += 1
            lf.log.info("Hello, LichtFeld!")
```

That alone is enough to ship a plugin panel. Keep state on `self`, render with `draw(ui)`, and register the class in `on_load()`.

See the full version in [`examples/01_draw_only.py`](examples/01_draw_only.py).

### Panel attributes

```python
import lichtfeld as lf


class MyPanel(lf.ui.Panel):
    id = "my_plugin.panel"
    label = "My Panel"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    parent = ""
    order = 100
    options = set()
    poll_dependencies = {
        lf.ui.PollDependency.SCENE,
        lf.ui.PollDependency.SELECTION,
        lf.ui.PollDependency.TRAINING,
    }
    size = None
    template = ""
    style = ""
    height_mode = lf.ui.PanelHeightMode.FILL
    update_policy = "dirty"
    update_interval_ms = 100

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def draw(self, ui):
        ui.label("Content here")
```

See [panel metadata and lifecycle](plugin-reference.md#panel) for defaults,
validation rules, retained hooks, and reactive updates. Use `id`,
`poll_dependencies`, and `lf.ui.PanelSpace`/`PanelOption`/`PanelHeightMode`
enums. Removed legacy fields and string enum values are rejected during
`lf.register_class()`.

### Step 2: add shell and retained behavior without rewriting `draw(ui)`

The unified API is designed for progressive disclosure. You can keep `draw(ui)` as your content source and opt into advanced features on the same class:

```python
import lichtfeld as lf


class StatusBarPanel(lf.ui.Panel):
    id = "my_plugin.status"
    label = "Build Up 2"
    space = lf.ui.PanelSpace.STATUS_BAR
    height_mode = lf.ui.PanelHeightMode.CONTENT
    update_policy = "interval"
    update_interval_ms = 120
    style = """
body.status-bar-panel { padding: 0 12dp; }
#im-root .im-label { color: #f3c96d; font-weight: bold; }
"""

    def __init__(self):
        self._progress = 0.2

    def draw(self, ui):
        ui.label("STATUS")
        ui.progress_bar(self._progress, f"{int(self._progress * 100)}%")

    def on_update(self, doc):
        del doc
        self._progress = (self._progress + 0.02) % 1.0
        return True
```

What changes here:

- `style` adds inline RCSS.
- `height_mode` controls how the retained shell sizes itself.
- `on_update()` adds periodic behavior for this status-bar animation example.
- `draw(ui)` still renders the actual content.

This is useful for UI that must advance on time, such as a progress animation. Most data-driven panels should not poll; use a dirty-policy retained panel instead.

See the full version in [`examples/02_status_bar_mixed.py`](examples/02_status_bar_mixed.py).

### Reactive retained panels

For normal retained panels, make updates explicit. Set `update_policy = "dirty"` and use `PanelStateBinding` to connect runtime-state changes to model invalidation. This keeps subscription lifetime, refresh work, and RML dirtying in one place.

```python
from pathlib import Path
import lichtfeld as lf
from lfs_plugins.ui import RuntimeState, PanelStateBinding

MODEL_NAME = "my_plugin_scene_summary"


class SceneSummaryPanel(lf.ui.Panel):
    id = "my_plugin.scene_summary"
    label = "Scene Summary"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    template = str(Path(__file__).resolve().with_name("scene_summary.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT
    update_policy = "dirty"

    def __init__(self):
        self._handle = None
        self._store_binding = PanelStateBinding()
        self._title = "No scene"

    def on_bind_model(self, ctx):
        model = ctx.create_data_model(MODEL_NAME)
        if model is None:
            return
        model.bind_func("title", lambda: self._title)
        self._handle = model.get_handle()

    def on_mount(self, doc):
        self._store_binding.set_handle(self._handle).watch(
            RuntimeState.scene_generation,
            RuntimeState.selection_generation,
            refresh=self._refresh_summary,
            dirty="title",
            immediate=True,
        )

    def on_unmount(self, doc):
        self._store_binding.close()
        doc.remove_data_model(MODEL_NAME)
        self._handle = None

    def _refresh_summary(self):
        scene = lf.get_scene()
        self._title = getattr(scene, "name", "Scene") if scene else "No scene"
```

Use this pattern when panel content depends on scene, selection, training, language, task progress, or active tool state. The low-level `RuntimeState.<field>.subscribe(...)` API exists for non-panel code, but retained panels should normally use `PanelStateBinding`.

### Retained shells and template resolution

When a panel uses retained features, LichtFeld chooses a shell automatically if `template` is empty:

| Space | Default retained shell |
|---|---|
| `FLOATING` | `rmlui/floating_window.rml` |
| `STATUS_BAR` | `rmlui/status_bar_panel.rml` |
| Other retained panel spaces | `rmlui/docked_panel.rml` |

Built-in aliases:

- `builtin:docked-panel`
- `builtin:floating-window`
- `builtin:status-bar`

For plugin-local templates, prefer absolute paths:

```python
from pathlib import Path

template = str(Path(__file__).resolve().with_name("main_panel.rml"))
```

When a template file exists at `main_panel.rml`, LichtFeld automatically looks for a sibling `main_panel.rcss` file and loads it as the base stylesheet for that document. A sibling `main_panel.theme.rcss` file is also loaded for palette-dependent overrides.

### Which styling path should you use?

| Goal | Best tool | Extra files |
|---|---|---|
| Start simple and ship quickly | `draw(ui)` plus built-in widgets and sub-layouts | None |
| Tweak spacing, colors, or typography on a retained shell | `style` with inline RCSS | None |
| Own the DOM structure and stylesheet | `template` plus sibling `.rml` and `.rcss` | `main_panel.rml`, `main_panel.rcss` |

Use that ladder in order. The scaffold still starts with immediate-mode content on the first row even though the retained shell files are already present.

### Step 3: go full hybrid

Use a custom template when you want retained DOM structure, data binding, or direct event listeners, but still keep an embedded immediate-mode area when that is convenient.

```python
from pathlib import Path
import lichtfeld as lf

MODEL_NAME = "my_plugin_hybrid"


class HybridPanel(lf.ui.Panel):
    id = "my_plugin.hybrid"
    label = "Hybrid"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT

    def draw(self, ui):
        ui.text_disabled("This block is rendered into #im-root.")

    def on_bind_model(self, ctx):
        model = ctx.create_data_model(MODEL_NAME)
        if model is None:
            return
        model.bind_func("title", lambda: "Hybrid Panel")
        self._handle = model.get_handle()

    def on_mount(self, doc):
        header = doc.get_element_by_id("header")
        if header:
            header.add_event_listener("click", lambda _ev: lf.log.info("Header clicked"))

    def on_update(self, doc):
        del doc
        if getattr(self, "_handle", None):
            self._handle.dirty_all()
```

Key retained hooks:

- `on_bind_model(ctx)`: create and bind a retained data model before the document loads.
- `on_mount(doc)`: wire DOM listeners or build dynamic DOM content after the document mounts.
- `on_unmount(doc)`: clean up document-local state.
- `on_update(doc)`: retained update hook. It is periodic for `update_policy = "interval"` and invalidation-driven for `update_policy = "dirty"`. Return `True` to mark content dirty.
- `on_scene_changed(doc)`: respond to active scene generation changes.

To mix retained and immediate content, include `<div id="im-root"></div>` somewhere in your template. `draw(ui)` will render into that node.

See the complete multi-file example in [`examples/03_hybrid_plugin/`](examples/03_hybrid_plugin/).

### Panel spaces

| Space | Description |
|---|---|
| `MAIN_PANEL_TAB` | Own tab in the right panel. Default for plugin panels. |
| `SIDE_PANEL` | Right sidebar panel. |
| `VIEWPORT_OVERLAY` | Drawn over the 3D viewport. |
| `SCENE_HEADER` | Header area above the scene tree. |
| `FLOATING` | Free-floating window. |
| `BOTTOM_DOCK` | Bottom dock area. |
| `LEFT_DOCK` | Left dock area. |
| `STATUS_BAR` | Bottom status bar. |

### Embedding in an existing tab

Use `parent` to place your panel inside a built-in tab as a collapsible section:

```python
class MyAnalysis(lf.ui.Panel):
    label = "My Analysis"
    parent = "lfs.rendering"
    order = 200

    def draw(self, ui):
        ui.label("Analysis results here")
```

Common parent ids:

| `parent` value | Effect |
|---|---|
| `"lfs.rendering"` | Collapsible section inside Rendering |
| `"lfs.training"` | Collapsible section inside Training |

### Register and unregister

```python
import lichtfeld as lf

lf.register_class(MyPanel)
lf.unregister_class(MyPanel)
```

### Panel replacement

Registering a panel with the same `id` as an existing panel replaces it. This is how plugins override built-in panels:

```python
import lichtfeld as lf


class MyTrainingPanel(lf.ui.Panel):
    id = "lfs.training"
    label = "Training"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 20

    def draw(self, ui):
        ui.label("Custom training controls")
```

Third-party plugins load after built-ins, so the replacement takes effect automatically while keeping the same slot in the UI.

### Panel management API

```python
import lichtfeld as lf

lf.ui.set_panel_enabled("my_plugin.panel", False)
lf.ui.is_panel_enabled("my_plugin.panel")

lf.ui.get_panel("my_plugin.panel")
lf.ui.set_panel_label("my_plugin.panel", "New Name")
lf.ui.set_panel_order("my_plugin.panel", 50)
lf.ui.set_panel_space("my_plugin.panel", lf.ui.PanelSpace.FLOATING)
lf.ui.set_panel_parent("my_plugin.panel", "lfs.rendering")
lf.ui.get_panel_names(lf.ui.PanelSpace.MAIN_PANEL_TAB)
```

`lf.ui.get_panel()` returns a typed `lf.ui.PanelInfo | None`, and `lf.ui.get_main_panel_tabs()` returns `list[lf.ui.PanelSummary]`.

### Layout composition

The `ui` object passed to `draw()` exposes a large widget/layout API. Start with direct calls, then use sub-layouts when structure matters:

```python
def draw(self, ui):
    with ui.row() as row:
        row.button("Action A")
        row.button("Action B")

    with ui.column() as col:
        col.label("Top")
        col.label("Bottom")

    with ui.box() as box:
        box.heading("Settings")
        box.prop(self, "opacity")

    with ui.split(0.3) as split:
        split.label("Name")
        split.prop(self, "name")

    with ui.grid_flow(columns=3) as grid:
        for item in items:
            grid.button(item.name)
```

See [examples/README.md](examples/README.md) for the recommended progression through the example files.

### Example: viewport overlay

```python
import lichtfeld as lf
from lfs_plugins.ui import RuntimeState


class StatsOverlay(lf.ui.Panel):
    label = "Stats"
    space = lf.ui.PanelSpace.VIEWPORT_OVERLAY
    order = 10

    @classmethod
    def poll(cls, context) -> bool:
        return context.has_scene

    def draw(self, ui):
        n = RuntimeState.num_gaussians.value
        x, y = ui.get_viewport_pos()
        ui.draw_text(x + 10, y + 10, f"Gaussians: {n:,}", (1.0, 1.0, 1.0, 0.8))
```

### Displaying GPU tensors

Panel `draw(ui)` receives `lf.ui.RmlUILayout`. Keep a `DynamicTexture` on
the panel and use its opaque Vulkan texture `id` with `ui.image()`:

```python
class PreviewPanel(lf.ui.Panel):
    id = "my_plugin.preview"
    label = "Preview"
    space = lf.ui.PanelSpace.FLOATING

    def __init__(self):
        self._texture = lf.ui.DynamicTexture()
        self._image = lf.Tensor.rand([256, 256, 3], device="cuda")

    def draw(self, ui):
        self._texture.update(self._image)
        ui.image(self._texture.id, (256, 256))

    def on_unmount(self, doc):
        self._texture.destroy()
```

The input must be `[H,W,3]` or `[H,W,4]`. CPU tensors are transferred to CUDA;
non-float32 values are converted, and uint8 pixels are normalized to `[0,1]`.
A new resolution recreates the texture. For changing image data, update the
texture after the image changes rather than doing expensive work in every draw.

`image_tensor()` and `image_texture()` are not methods of panel `RmlUILayout`.
They exist on compatibility `UILayout`, where images are currently inert.
See [images and their limitations](plugin-reference.md#images) and
[DynamicTexture](plugin-reference.md#dynamictexture).

---

## UI Hooks and Menus

New plugin UI should use panels, retained data models, and `RuntimeState`
subscriptions. Hook registration is deprecated for external plugins. Existing
hooks can be discovered using `lf.ui.get_hook_points()`; availability and the
layout supplied by a host determine whether interactive widgets work. Avoid
assuming a hook point always provides a live panel layout.

See [hook registration](plugin-reference.md#ui-hooks) for the compatibility
API and [Menu](plugin-reference.md#menu) for declarative menu entries.

---

## Operators

Operators are actions that can be invoked by buttons, menus, or keyboard shortcuts. They extend `PropertyGroup`, so they can have typed properties.

### Operator base class

```python
from lfs_plugins.types import Operator, Event

class MyOperator(Operator):
    label = "My Action"
    description = "What this operator does"
    options = set()          # e.g. {'UNDO', 'BLOCKING'}

    @classmethod
    def poll(cls, context) -> bool:
        """Return False to disable the operator."""
        return True

    def invoke(self, context, event: Event) -> set:
        """Called when operator is first triggered. Can start modal."""
        return self.execute(context)

    def execute(self, context) -> set:
        """Synchronous execution."""
        return {"FINISHED"}

    def modal(self, context, event: Event) -> set:
        """Handle events during modal execution."""
        return {"FINISHED"}

    def cancel(self, context):
        """Called when the operator is cancelled."""
        pass
```

### Return sets

| Value             | Meaning                              |
|-------------------|--------------------------------------|
| `{"FINISHED"}`    | Operator completed successfully      |
| `{"CANCELLED"}`   | Operator was cancelled               |
| `{"RUNNING_MODAL"}` | Operator is running in modal mode |
| `{"PASS_THROUGH"}`  | Pass event to other handlers       |

Operators can also return a dict: `{"status": "FINISHED", "result": data}`.

### Event object

`invoke()` and `modal()` receive an event with mouse, key, modifier, scroll,
and GUI-hit fields. See [Event](plugin-reference.md#event) for the field list.
Discover native operators with `lf.ops.get_all()` and inspect their metadata
with `lf.ops.get_descriptor(id)`. Invoke an operator with a positional ID,
`lf.ops.invoke(id, **properties)`, and check its `finished`/`cancelled` flags.

### Example: simple execute-only operator

```python
import lichtfeld as lf
from lfs_plugins.types import Operator

class PrintGaussianCount(Operator):
    label = "Print Gaussian Count"
    description = "Log the number of gaussians in the current scene"

    @classmethod
    def poll(cls, context) -> bool:
        return context.has_scene

    def execute(self, context) -> set:
        scene = lf.get_scene()
        if scene is None:
            return {"CANCELLED"}
        lf.log.info(f"Gaussians: {scene.total_gaussian_count:,}")
        return {"FINISHED"}
```

### Example: modal operator (interactive tool)

```python
import lichtfeld as lf
from lfs_plugins.types import Operator, Event

class MeasureTool(Operator):
    label = "Measure Distance"
    description = "Click two points to measure distance"
    # Only set UNDO when the operator actually implements undo()/redo()
    # or when all mutations go through history-aware scene APIs.
    options = set()

    def __init__(self):
        super().__init__()
        self.start_pos = None

    def invoke(self, context, event: Event) -> set:
        self.start_pos = None
        lf.log.info("Click first point...")
        return {"RUNNING_MODAL"}

    def modal(self, context, event: Event) -> set:
        if event.type == "ESC":
            lf.log.info("Measurement cancelled")
            return {"CANCELLED"}

        if event.type == "LEFTMOUSE" and event.value == "PRESS":
            pos = (event.mouse_x, event.mouse_y)
            if self.start_pos is None:
                self.start_pos = pos
                lf.log.info("Click second point...")
                return {"RUNNING_MODAL"}
            else:
                dx = pos[0] - self.start_pos[0]
                dy = pos[1] - self.start_pos[1]
                dist = (dx * dx + dy * dy) ** 0.5
                lf.log.info(f"Distance: {dist:.2f} pixels")
                return {"FINISHED"}

        return {"RUNNING_MODAL"}

    def cancel(self, context):
        self.start_pos = None
```

If an operator performs custom side effects that are not already covered by the shared scene history,
wrap them in `with lf.undo.transaction("My Change"):` or push a custom step with `lf.undo.push(...)`.

---

## Toolbar Tools

Tools appear in the viewport toolbar and can have submodes and pivot modes.

### ToolDef dataclass

```python
from lfs_plugins.tool_defs.definition import ToolDef, SubmodeDef, PivotModeDef

tool = ToolDef(
    id="my_plugin.my_tool",         # Unique identifier
    label="My Tool",                # Display label
    icon="star",                    # Icon name
    group="utility",                # "select", "transform", "utility"
    order=200,                      # Sort order within group
    description="Tool tooltip",     # Tooltip
    shortcut="",                    # Keyboard shortcut
    gizmo="",                       # "translate", "rotate", "scale", or ""
    operator="",                    # Operator to invoke on activation
    submodes=(),                    # Tuple of SubmodeDef
    pivot_modes=(),                 # Tuple of PivotModeDef
    poll=None,                      # Callable[[context], bool]
    plugin_name="my_plugin",        # For custom icon loading
    plugin_path="/path/to/plugin",  # For custom icon loading
)
```

### Register and unregister

```python
from lfs_plugins.tools import ToolRegistry

ToolRegistry.register_tool(tool)
ToolRegistry.unregister_tool("my_plugin.my_tool")
```

### Custom icons

Place PNG icons in your plugin's `icons/` folder. Reference them by name (without extension) and set `plugin_name` and `plugin_path` on the `ToolDef`.

### Example: custom tool with submodes

```python
from pathlib import Path
from lfs_plugins.tool_defs.definition import ToolDef, SubmodeDef, PivotModeDef
from lfs_plugins.tools import ToolRegistry

measure_tool = ToolDef(
    id="my_plugin.measure",
    label="Measure",
    icon="ruler",
    group="utility",
    order=100,
    description="Measure scene attributes",
    submodes=(
        SubmodeDef("distance", "Distance", "ruler"),
        SubmodeDef("angle", "Angle", "angle"),
        SubmodeDef("bounds", "Bounds", "box"),
    ),
    pivot_modes=(
        PivotModeDef("center", "Selection Center", "circle-dot"),
        PivotModeDef("cursor", "3D Cursor", "crosshair"),
    ),
    poll=lambda ctx: ctx.has_scene,
    plugin_name="my_plugin",
    plugin_path=str(Path(__file__).parent),
)

ToolRegistry.register_tool(measure_tool)
```

### Native TRS gizmos

Use the built-in viewport transform handles directly from Python when a tool needs a reusable translate,
rotate, or scale handle that is not tied to the active toolbar selection.

```python
import lichtfeld as lf

box_matrix = lf.compose_transform([0, 0, 0], [0, 0, 0], [1, 1, 1])

gizmo = lf.TransformGizmo("translate", box_matrix, id="my_plugin.box_move")
gizmo.space = "local"          # or "world"
gizmo.snap = True
gizmo.translate_snap = 0.1


def read_box_transform():
    return box_matrix


def write_box_transform(matrix):
    global box_matrix
    box_matrix = list(matrix)


gizmo.attach_to_callbacks(read_box_transform, write_box_transform)
```

Convenience constructors are available for each operation:

```python
move = lf.TranslationGizmo(id="move_anchor")
rotate = lf.RotationGizmo(id="rotate_anchor")
scale = lf.ScaleGizmo(id="scale_anchor")
```

For scene nodes, attach directly:

```python
gizmo = lf.TransformGizmo("rotate", id="my_plugin.node_rotate")
gizmo.attach_to_node("Model")  # visualizer-world transform by default
```

Keep the returned gizmo object if you want to change settings later. Attached gizmos stay registered
until `gizmo.detach()` or `lf.clear_transform_gizmos()` is called.

---

## Properties

Properties provide typed, validated attributes for operators and property groups.

### Property types

| Type                | Default     | Key Parameters                            |
|---------------------|-------------|-------------------------------------------|
| `FloatProperty`     | `0.0`       | `min`, `max`, `step`, `precision`, `subtype` |
| `IntProperty`       | `0`         | `min`, `max`, `step`                      |
| `BoolProperty`      | `False`     |                                           |
| `StringProperty`    | `""`        | `maxlen`, `subtype`                       |
| `EnumProperty`      | first item  | `items=[(id, label, desc), ...]`          |
| `FloatVectorProperty` | `(0,0,0)` | `size`, `min`, `max`, `subtype`           |
| `IntVectorProperty` | `(0,0,0)`  | `size`, `min`, `max`                      |
| `TensorProperty`    | `None`      | `shape`, `dtype`, `device`                |
| `CollectionProperty`| `None`        | `type=PropertyGroupSubclass`              |
| `PointerProperty`   | `None`      | `type=PropertyGroupSubclass`              |

All properties accept `name`, `description`, `subtype`, and `update`.
The update callback receives `(owner, None)` after assignment; its second
argument is a context placeholder, not the new value. Read `owner.<property>`
in the callback. See [property signatures](plugin-reference.md#properties).

### PropertyGroup base class

```python
from lfs_plugins.props import PropertyGroup, FloatProperty, FloatVectorProperty, StringProperty

class MaterialSettings(PropertyGroup):
    color = FloatVectorProperty(default=(1, 1, 1), size=3, subtype="COLOR")
    roughness = FloatProperty(default=0.5, min=0.0, max=1.0)
    name = StringProperty(default="Untitled", maxlen=64)

# Singleton access
settings = MaterialSettings.get_instance()
settings.roughness = 0.8
print(settings.roughness)  # 0.8 (validated and clamped)
```

### Subtypes

| Subtype        | Applies To       | Effect                           |
|----------------|------------------|----------------------------------|
| `COLOR`        | FloatVector      | Color picker widget              |
| `COLOR_GAMMA`  | FloatVector      | Color picker with gamma          |
| `FILE_PATH`    | String           | File picker widget               |
| `DIR_PATH`     | String           | Folder picker widget             |
| `FACTOR`       | Float            | 0-1 slider                       |
| `PERCENTAGE`   | Float            | 0-100 slider                     |
| `ANGLE`        | Float            | Radians, displayed as degrees    |
| `TRANSLATION`  | FloatVector      | 3D translation                   |
| `EULER`        | FloatVector      | Euler rotation angles            |
| `QUATERNION`   | FloatVector(4)   | Quaternion rotation              |
| `XYZ`          | FloatVector      | Generic XYZ values               |

### Example: settings group with typed properties

```python
from lfs_plugins.props import (
    PropertyGroup, FloatProperty, IntProperty, BoolProperty,
    StringProperty, EnumProperty, FloatVectorProperty, TensorProperty,
)

class TrainingSettings(PropertyGroup):
    learning_rate = FloatProperty(
        default=0.001, min=0.0001, max=0.1,
        name="Learning Rate",
        description="Base learning rate for optimization",
    )
    max_iterations = IntProperty(default=30000, min=1000, max=100000)
    use_ssim = BoolProperty(default=True, name="Use SSIM Loss")
    output_path = StringProperty(default="output", subtype="DIR_PATH")
    strategy = EnumProperty(items=[
        ("mcmc", "MCMC", "Markov Chain Monte Carlo strategy"),
        ("default", "Default", "Default densification strategy"),
    ])
    background_color = FloatVectorProperty(
        default=(0.0, 0.0, 0.0), size=3, subtype="COLOR"
    )
    custom_mask = TensorProperty(shape=(-1,), dtype="bool", device="cuda")
```

---

## Scene Access

The `lichtfeld` module (`lf`) provides access to the scene graph, node operations, selection, and transforms.

### Getting the scene

```python
import lichtfeld as lf

scene = lf.get_scene()          # Get scene object (None if no scene loaded)
if lf.has_scene():
    print(f"Total gaussians: {scene.total_gaussian_count}")
```

### Node operations

```python
scene = lf.get_scene()

# Add nodes
group_id = scene.add_group("My Group")
splat_id = scene.add_splat(
    "My Splat",
    means=lf.Tensor.zeros([100, 3], device="cuda"),
    sh0=lf.Tensor.zeros([100, 1, 3], device="cuda"),
    shN=lf.Tensor.zeros([100, 0, 3], device="cuda"),
    scaling=lf.Tensor.zeros([100, 3], device="cuda"),
    rotation=lf.Tensor.cat([
        lf.Tensor.ones([100, 1], device="cuda"),
        lf.Tensor.zeros([100, 3], device="cuda"),
    ], dim=1),  # Identity quaternions (w, x, y, z)
    opacity=lf.Tensor.zeros([100, 1], device="cuda"),
)

# Query nodes
nodes = scene.get_nodes()
node = scene.get_node("My Splat")
visible = scene.get_visible_nodes()

# Modify
scene.rename_node("My Splat", "Renamed Splat")
scene.reparent(splat_id, group_id)
scene.remove_node("Renamed Splat", keep_children=False)
new_name = scene.duplicate_node("My Group")
```

### Selection

```python
import lichtfeld as lf

lf.select_node("My Splat")
names = lf.get_selected_node_names()
lf.deselect_all()
has_sel = lf.has_selection()

# Gaussian-level selection (mask-based)
scene = lf.get_scene()
mask = lf.Tensor.zeros([scene.total_gaussian_count], dtype="bool", device="cuda")
mask[0:100] = True
scene.set_selection_mask(mask)
scene.clear_selection()
```

### Transforms

```python
import lichtfeld as lf

# Get/set a 16-float column-major transform matrix
matrix = lf.get_node_transform("My Splat")
lf.set_node_transform("My Splat", matrix)

# Decompose/compose
components = lf.decompose_transform(matrix)
# components includes translation, rotation_quat, rotation_euler,
# rotation_euler_deg, and scale

new_matrix = lf.compose_transform(
    translation=[1.0, 2.0, 3.0],
    euler_deg=[0.0, 45.0, 0.0],
    scale=[1.0, 1.0, 1.0],
)
```

### Splat data access

Splat data can be accessed from the combined model or from individual scene
nodes. `combined_model()` merges visible splats and can return `None`. Treat it
as render data; edit each node's `splat_data()` to update that node and call
`scene.notify_changed()` after changing raw tensors.

```python
scene = lf.get_scene()

# Combined model (visible splats merged; may be None)
model = scene.combined_model()

# Per-node access
for node in scene.get_nodes():
    sd = node.splat_data()       # None for non-splat nodes
    if sd is not None:
        print(f"{node.name}: {sd.num_points} gaussians")
```

```python
# Raw data views; use per-node data for edits
means = model.means_raw           # [N, 3] positions
sh0 = model.sh0_raw               # [N, 1, 3] base SH coefficients
shN = model.shN_raw               # [N, K, 3] higher-order SH
scaling = model.scaling_raw        # [N, 3] log-space scaling
rotation = model.rotation_raw      # [N, 4] quaternion rotation
opacity = model.opacity_raw        # [N, 1] logit-space opacity

# Activated data (transformed to usable form)
activated_opacity = model.get_opacity()     # sigmoid applied, [N]
activated_scaling = model.get_scaling()     # exp applied
activated_rotation = model.get_rotation()   # normalized quaternions

# Metadata
count = model.num_points
sh_deg = model.active_sh_degree
```

### Soft delete

Soft delete hides gaussians without removing them. After modifying the deletion mask, call `scene.notify_changed()` to update the viewport:

```python
scene = lf.get_scene()
for node in scene.get_nodes():
    sd = node.splat_data()
    if sd is None:
        continue

    # Hide gaussians with opacity below threshold
    opacity = sd.get_opacity()            # [N] in [0, 1]
    mask = opacity < 0.1
    sd.soft_delete(mask)

# Trigger viewport redraw — required after modifying scene data
scene.notify_changed()

# Restore all hidden gaussians
for node in scene.get_nodes():
    sd = node.splat_data()
    if sd is not None:
        sd.clear_deleted()
scene.notify_changed()
```

> **Note:** `scene.invalidate_cache()` only clears the internal cache. It does **not** trigger a viewport redraw. Use `scene.notify_changed()` instead — it invalidates the cache and signals the renderer.

### Example: scene manipulation plugin

```python
import lichtfeld as lf
from lfs_plugins.types import Operator

class SceneInfo(Operator):
    label = "Print Scene Info"

    def execute(self, context) -> set:
        scene = lf.get_scene()
        if scene is None:
            lf.log.warn("No scene loaded")
            return {"CANCELLED"}

        for node in scene.get_nodes():
            bounds = scene.get_node_bounds(node.id)
            lf.log.info(f"Node: {node.name}, bounds: {bounds}")

        return {"FINISHED"}

class CenterSelection(Operator):
    label = "Center Selection"

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_selection() and lf.can_transform_selection()

    def execute(self, context) -> set:
        center = lf.get_selection_visualizer_world_center()
        if center:
            lf.log.info(f"Selection center: {center}")
        return {"FINISHED"}
```

---

## Signals

Signals provide reactive state management. When a signal's value changes, all subscribers are notified.

### Signal types

```python
from lfs_plugins.ui.signals import Signal, ComputedSignal, ThrottledSignal, Batch

# Basic signal
count = Signal(0, name="count")
count.value = 5                          # Notifies subscribers
current = count.value                    # Read current value
current = count.peek()                   # Read without tracking

# Subscribe
unsub = count.subscribe(lambda v: print(f"Count: {v}"))
unsub()                                  # Stop receiving updates

# Owner-tracked subscription (auto-cleanup on plugin unload)
unsub = count.subscribe_as("my_plugin", lambda v: print(v))

# Computed signal (derived from others)
a = Signal(2)
b = Signal(3)
product = ComputedSignal(lambda: a.value * b.value, [a, b])
print(product.value)                     # 6

# Throttled signal (rate-limited notifications)
iteration = ThrottledSignal(0, max_rate_hz=30)
iteration.value = 1000                   # Only notifies ~30 times/sec
iteration.flush()                        # Force pending notification
```

### Batch context manager

Defer notifications until all updates are complete:

```python
from lfs_plugins.ui.signals import Batch

with Batch():
    state.x.value = 10
    state.y.value = 20
    state.z.value = 30
# Each changed signal notifies after all three assignments are complete
```

### RuntimeState

Use native state names for new plugins:

```python
from lfs_plugins.ui import RuntimeState

RuntimeState.training_running.value
RuntimeState.training_state.value
RuntimeState.trainer_loaded.value
RuntimeState.iteration.value
RuntimeState.total_iterations.value
RuntimeState.loss.value
RuntimeState.eval_psnr.value             # float or None
RuntimeState.scene_generation.value
RuntimeState.selection_generation.value
```

The [runtime-state reference](plugin-reference.md#reactive-retained-panels)
lists fields, compatibility aliases, and `PanelStateBinding`. For current
scene/selection availability, use `lf.ui.context()` or native queries.

### Example: reactive training monitor

```python
import lichtfeld as lf
from lfs_plugins.ui import RuntimeState
from lfs_plugins.ui.signals import Signal

class TrainingMonitor(lf.ui.Panel):
    label = "Training Monitor"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 50

    def __init__(self):
        self.best_loss = Signal(float("inf"), name="best_loss")
        self.loss_history = []

        RuntimeState.loss.subscribe_as("my_plugin", self._on_loss_change)

    def _on_loss_change(self, loss: float):
        if loss > 0:
            self.loss_history.append(loss)
            if loss < self.best_loss.value:
                self.best_loss.value = loss

    @classmethod
    def poll(cls, context) -> bool:
        return context.has_trainer

    def draw(self, ui):
        ui.heading("Training Monitor")

        state = RuntimeState.training_state.value
        ui.label(f"State: {state}")
        ui.label(f"Iteration: {RuntimeState.iteration.value}")
        ui.label(f"Loss: {RuntimeState.loss.value:.6f}")
        ui.label(f"Best Loss: {self.best_loss.value:.6f}")
        psnr = RuntimeState.eval_psnr.value
        ui.label("PSNR: unavailable" if psnr is None else f"PSNR: {psnr:.2f}")
        ui.label(f"Gaussians: {RuntimeState.num_gaussians.value:,}")

        progress = RuntimeState.training_progress.value
        ui.progress_bar(progress, f"{progress * 100:.1f}%")

        if self.loss_history:
            ui.plot_lines(
                "Loss##monitor",
                self.loss_history[-200:],
                0.0, max(self.loss_history[-200:]),
                (0, 80),
            )
```

---

## Capabilities

Capabilities allow plugins to expose features that other plugins (or the application) can invoke.

### CapabilityRegistry

```python
from lfs_plugins.capabilities import CapabilityRegistry, CapabilitySchema
from lfs_plugins.context import PluginContext

registry = CapabilityRegistry.instance()

# Register a capability
def my_handler(args: dict, ctx: PluginContext) -> dict:
    threshold = args.get("threshold", 0.5)
    if ctx.scene:
        # Do something with the scene
        pass
    return {"success": True, "count": 42}

registry.register(
    name="my_plugin.analyze",
    handler=my_handler,
    description="Analyze gaussians by threshold",
    schema=CapabilitySchema(
        properties={"threshold": {"type": "number", "default": 0.5}},
        required=[],
    ),
    plugin_name="my_plugin",
    requires_gui=True,
)

# Invoke a capability
result = registry.invoke("my_plugin.analyze", {"threshold": 0.3})
# result = {"success": True, "count": 42}

# Query
registry.has("my_plugin.analyze")    # True
caps = registry.list_all()           # List[Capability]

# Unregister
registry.unregister("my_plugin.analyze")
registry.unregister_all_for_plugin("my_plugin")
```

### PluginContext

Capability handlers receive a `PluginContext` with scene and view data:

```python
from lfs_plugins.context import PluginContext, SceneContext, ViewContext

def handler(args: dict, ctx: PluginContext) -> dict:
    # Scene access
    if ctx.scene:
        ctx.scene.scene               # PyScene object
        ctx.scene.set_selection_mask(mask)

    # Viewport access
    if ctx.view:
        ctx.view.image                 # [H, W, 3] tensor
        ctx.view.screen_positions      # [N, 2] tensor or None
        ctx.view.width, ctx.view.height
        ctx.view.fov
        ctx.view.rotation              # [3, 3] tensor
        ctx.view.translation           # [3] tensor

    # Invoke other capabilities
    if ctx.capabilities.has("other_plugin.feature"):
        result = ctx.capabilities.invoke("other_plugin.feature", {"key": "value"})

    return {"success": True}
```

---

## Training Hooks

Register callbacks for training lifecycle events.

### Decorators

```python
import lichtfeld as lf

@lf.on_training_start
def on_start(_hook):
    lf.log.info("Training started")

@lf.on_iteration_start
def on_iter(_hook):
    pass

@lf.on_pre_optimizer_step
def on_pre_opt(_hook):
    pass

@lf.on_post_step
def on_post(_hook):
    ctx = lf.context()
    if ctx.iteration % 1000 == 0:
        lf.log.info(f"Iteration {ctx.iteration}, loss: {ctx.loss:.6f}")

@lf.on_training_end
def on_end(_hook):
    lf.log.info(f"Training finished: {lf.finish_reason()}")
```

For unloadable plugins, keep registrations in a `lf.ControlSession()` and
call `clear()` in `on_unload()`. Module-level decorators register until app
shutdown; dropping a Python reference does not remove the callback.

### Training context

```python
ctx = lf.context()
ctx.iteration          # Current iteration (int)
ctx.max_iterations     # Target iterations (int)
ctx.loss               # Current loss (float)
ctx.num_gaussians      # Gaussian count (int)
ctx.is_refining        # Currently refining (bool)
ctx.is_training        # Training active (bool)
ctx.is_paused          # Training paused (bool)
ctx.phase              # Current phase (str)
ctx.strategy           # Training strategy (str)
ctx.refresh()          # Update snapshot
```

### Training control

```python
import lichtfeld as lf

lf.start_training()
lf.pause_training()
lf.resume_training()
lf.stop_training()
lf.reset_training()
lf.project_save()
```

### Example: custom training callback

```python
import lichtfeld as lf

class AutoSavePlugin:
    """Request a project save every N iterations."""

    def __init__(self, interval=5000):
        self.interval = interval
        self.last_save = 0
        self._hooks = lf.ControlSession()
        self._hooks.on_post_step(self.on_post_step)

    def close(self):
        self._hooks.clear()

    def on_post_step(self, _hook):
        ctx = lf.context()
        if ctx.iteration - self.last_save >= self.interval:
            lf.project_save()
            self.last_save = ctx.iteration
            lf.log.info(f"Auto-save requested at iteration {ctx.iteration}")

_auto_save = None

def on_load():
    global _auto_save
    _auto_save = AutoSavePlugin(interval=5000)
    lf.log.info("Auto-save plugin loaded")

def on_unload():
    global _auto_save
    if _auto_save is not None:
        _auto_save.close()
    _auto_save = None
```

`lf.log.info(...)` writes to the main LichtFeld application log. Use `print(...)` if you want temporary debug output in the integrated Python console output panel, or call both if you want messages in both places.

---

## Plugin Runtime and Dependencies

Discovery requires a `pyproject.toml` with `[tool.lichtfeld]`. The manager in
`src/python/lfs_plugins/` validates the manifest, installs dependencies, imports
the entrypoint, calls `on_load()`, and tracks lifecycle state. `lf.plugins`
provides the native convenience bindings. The CLI handles `create`, `check`,
and `list`; installs/updates are available through Python and the Marketplace.

States are `UNLOADED`, `INSTALLING`, `LOADING`, `ACTIVE`, `ERROR`, and
`DISABLED`. Inspect state and the captured traceback with `lf.plugins.get_state`,
`get_error`, and `get_traceback`; see [management APIs](plugin-reference.md#pluginmanager).

Declare dependencies in `[project].dependencies`. Dependencies are isolated
in each plugin's `.venv` using bundled Python and `uv`. The CLI scaffold creates
the environment immediately; the Python scaffold writes source files only and
loading the plugin performs dependency setup. The installer uses `uv venv` and
`uv sync` with `--no-managed-python --no-python-downloads` to use the app's Python.

Compatibility ranges target separate versions: `plugin_api` matches the public
plugin contract, `lichtfeld_version` matches the host application, and
`required_features` names host features. Read `lf.PLUGIN_API_VERSION`,
`lf.plugins.API_VERSION`, and `lf.plugins.FEATURES` for the current host.

## Hot Reload & Debugging

### File watcher

When `hot_reload = true` in `pyproject.toml`, LichtFeld watches your plugin directory for changes. Changes detected by the running watcher unload and reload the plugin. Use
`lf.plugins.start_watcher()`/`stop_watcher()` to control watching explicitly.

### Logging

```python
import lichtfeld as lf

lf.log.info("Informational message")
lf.log.warn("Warning message")
lf.log.error("Error message")
lf.log.debug("Debug message")    # Only visible with --log-level debug
```

`lf.log.*()` messages go to the main application log. `print(...)` and Python tracebacks go to the integrated Python console output.

### Plugin state inspection

```python
from lfs_plugins.manager import PluginManager

mgr = PluginManager.instance()
state = mgr.get_state("my_plugin")       # PluginState enum
error = mgr.get_error("my_plugin")       # Error message or None
tb = mgr.get_traceback("my_plugin")      # Traceback string or None
```

Or via the `lf` module:

```python
import lichtfeld as lf

lf.plugins.get_state("my_plugin")
lf.plugins.get_error("my_plugin")
lf.plugins.get_traceback("my_plugin")
```

---

## IDE Setup

### Completion and type checking

`LichtFeld-Studio plugin create` writes `pyrightconfig.json` and VS Code settings
inside the **plugin directory**, pointing at its venv and the host typings.
For a Python-created plugin, configure these paths yourself:

```python
import lichtfeld as lf
print(lf.packages.embedded_python_path())
print(lf.packages.typings_dir())
```

The committed [SDK stubs](../../src/python/stubs/lichtfeld/) cover the native
module and submodules. Generated stubs are under
`<build-dir>/src/python/typings/`; use your actual build directory rather than
a hard-coded `build/` path. Pure Python `lfs_plugins` helpers live in
[src/python/lfs_plugins](../../src/python/lfs_plugins/).

For binding maintainers, the full app build checks committed stubs against the
generated output. The `refresh_python_stubs` maintenance target updates the
committed SDK after intentional binding changes; it is not needed to author a
plugin or change these docs.

### debugpy attach

Add to your plugin's `on_load()` for VS Code debugging:

```python
def on_load():
    try:
        import debugpy
        debugpy.listen(5678)
        lf.log.info("debugpy listening on port 5678")
    except ImportError:
        pass
```

VS Code launch config:

```json
{
    "name": "Attach to LichtFeld Plugin",
    "type": "debugpy",
    "request": "attach",
    "connect": {"host": "localhost", "port": 5678}
}
```

---

## Installing & Publishing

### Create a new plugin

```python
import lichtfeld as lf

path = lf.plugins.create("my_new_plugin")
```

That Python API creates the minimal source package only. If you also want a plugin venv and editor config, use:

```bash
LichtFeld-Studio plugin create my_new_plugin
```

Both scaffold paths start with the same step-1 panel template and now include `main_panel.rml` and `main_panel.rcss` up front. You can ignore those files until you move into the custom-template styling path.

### Install from GitHub

```python
import lichtfeld as lf

lf.plugins.install("owner/repo")
lf.plugins.install("https://github.com/owner/repo")
```

### Plugin registry

```python
import lichtfeld as lf

results = lf.plugins.search("neural rendering")
lf.plugins.install_from_registry("plugin_id")
lf.plugins.check_updates()
lf.plugins.update("my_plugin")
```

Registry installs use the same v1 compatibility contract as local plugins. A version is eligible only if its `plugin_api`, `lichtfeld_version`, and `required_features` match the current host.

### Manage plugins

```python
import lichtfeld as lf

lf.plugins.discover()              # Scan for installed plugins
lf.plugins.load("my_plugin")       # Load a specific plugin
lf.plugins.unload("my_plugin")     # Unload
lf.plugins.reload("my_plugin")     # Reload (hot reload)
lf.plugins.uninstall("my_plugin")  # Remove
lf.plugins.list_loaded()           # Show loaded plugins
```

### pyproject.toml packaging requirements

For publishing, ensure your `pyproject.toml` includes:
- `name` - Unique plugin identifier
- `version` - Semantic version (e.g., `"1.0.0"`)
- `description` - Clear description of what the plugin does
- `authors` - Your name or organization
- `dependencies` - Any Python dependencies
- `plugin_api` - Supported plugin API range, such as `"~=1.0"` or `">=1,<2"`
- `lichtfeld_version` - Supported host/runtime range, such as `">=0.4.2"`
- `required_features` - Optional host features the plugin requires

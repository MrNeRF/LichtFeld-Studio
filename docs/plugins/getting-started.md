# Plugin Developer Guide

LichtFeld Studio plugins extend the application with panels, operators, tools, signals, and capabilities. Plugins live in `~/.lichtfeld/plugins/` and are just Python packages with a small manifest and entrypoint.

## Learning path

Read the examples in this order:

| Step | Goal | Example |
|---|---|---|
| 1 | Pure immediate-mode panel with `draw(ui)` only | [`examples/01_draw_only.py`](examples/01_draw_only.py) |
| 2 | Add shell, styling, and periodic updates without rewriting `draw(ui)` | [`examples/02_status_bar_mixed.py`](examples/02_status_bar_mixed.py) |
| 3 | Build a full hybrid panel with template, RCSS, data model, DOM hooks, and embedded `draw(ui)` | [`examples/03_hybrid_plugin/`](examples/03_hybrid_plugin/) |
| 4 | Explore focused feature demos | [`examples/README.md`](examples/README.md) |
| 5 | See an end-to-end multi-file plugin | [`examples/full_plugin/`](examples/full_plugin/) |

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
    update_interval_ms = 100

    @classmethod
    def poll(cls, context) -> bool:
        return True

    def draw(self, ui):
        ui.label("Content here")
```

| Attribute | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | `module.qualname` | Unique panel identifier. Used for replacement, visibility, and API lookups. |
| `label` | `str` | `""` | Display name in the UI. Falls back to `id` when empty. |
| `space` | `lf.ui.PanelSpace` | `lf.ui.PanelSpace.MAIN_PANEL_TAB` | Where the panel appears when `parent` is empty. |
| `parent` | `str` | `""` | Parent panel id. When set, the panel embeds as a collapsible section and must not also override `space`. |
| `order` | `int` | `100` | Sort order within its space. Lower values appear earlier. |
| `options` | `set[lf.ui.PanelOption]` | `set()` | Panel options such as `lf.ui.PanelOption.DEFAULT_CLOSED` and `lf.ui.PanelOption.HIDE_HEADER`. |
| `poll_dependencies` | `set[lf.ui.PollDependency]` | `{SCENE, SELECTION, TRAINING}` | Which app-state changes should re-run `poll()`. |
| `size` | `tuple[float, float] \| None` | `None` | Initial width/height hint, mainly useful for floating panels. |
| `template` | `str \| os.PathLike[str]` | `""` | Optional retained RML template. Use an absolute path for plugin-local files. |
| `style` | `str` | `""` | Optional inline RCSS appended to the retained document. This is RCSS text, not a file path. |
| `height_mode` | `lf.ui.PanelHeightMode` | `lf.ui.PanelHeightMode.FILL` | `FILL` or `CONTENT` for retained panels. |
| `update_interval_ms` | `int` | `100` | Update cadence for retained/hybrid `on_update()` work. |

The panel API is strict in v1: use the enum values above, not string literals.

Panel definitions are validated eagerly. Invalid enum values, removed legacy fields, retained-only settings in `VIEWPORT_OVERLAY`, and conflicting fields such as `parent` plus explicit `space` raise `ValueError`, `TypeError`, or `AttributeError` during `lf.register_class()`.

### Step 2: add shell and retained behavior without rewriting `draw(ui)`

The unified API is designed for progressive disclosure. You can keep `draw(ui)` as your content source and opt into advanced features on the same class:

```python
import lichtfeld as lf


class StatusBarPanel(lf.ui.Panel):
    id = "my_plugin.status"
    label = "Build Up 2"
    space = lf.ui.PanelSpace.STATUS_BAR
    height_mode = lf.ui.PanelHeightMode.CONTENT
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
- `on_update()` adds periodic behavior.
- `draw(ui)` still renders the actual content.

This is the normal upgrade path. You do not need to rewrite the panel as full DOM/RML just because you added styling or retained hooks.

See the full version in [`examples/02_status_bar_mixed.py`](examples/02_status_bar_mixed.py).

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
- `on_update(doc)`: periodic updates while the panel is visible. Return `True` to mark content dirty.
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
from lfs_plugins.ui.state import AppState


class StatsOverlay(lf.ui.Panel):
    label = "Stats"
    space = lf.ui.PanelSpace.VIEWPORT_OVERLAY
    order = 10

    @classmethod
    def poll(cls, context) -> bool:
        return AppState.has_scene.value

    def draw(self, ui):
        n = AppState.num_gaussians.value
        ui.draw_text(10, 10, f"Gaussians: {n:,}", (1.0, 1.0, 1.0, 0.8))
```

### Displaying GPU tensors

Use `image_tensor` to render a CUDA tensor directly in a panel with no manual texture management:

```python
class PreviewPanel(lf.ui.Panel):
    label = "Preview"
    space = lf.ui.PanelSpace.FLOATING

    def draw(self, ui):
        tensor = lf.Tensor.rand([256, 256, 3], device="cuda")
        ui.image_tensor("my_preview", tensor, (256, 256))
```

The `label` argument (`"my_preview"`) caches the underlying GL texture between frames. Passing a tensor with a different resolution automatically recreates the texture. The tensor must be `[H, W, 3]` (RGB) or `[H, W, 4]` (RGBA). CPU tensors and integer dtypes are converted automatically.

For advanced use cases, use `DynamicTexture`:

```python
class AdvancedPanel(lf.ui.Panel):
    label = "Advanced"
    space = lf.ui.PanelSpace.FLOATING

    def __init__(self):
        self.tex = lf.ui.DynamicTexture()

    def draw(self, ui):
        self.tex.update(my_tensor)
        ui.image_texture(self.tex, (256, 256))
```

See the [DynamicTexture API reference](api-reference.md#dynamictexture) for all properties and methods.

---

## UI Hooks

Hooks let you inject UI into existing panels without replacing them. A hook callback receives a `layout` object and draws into the host panel at a predefined hook point.

### Hook pattern

```python
import lichtfeld as lf


class MyHookPanel:
    def draw(self, layout):
        if not layout.collapsing_header("My Section", default_open=True):
            return
        layout.label("Injected into the rendering panel")


_instance = None


def _draw_hook(layout):
    global _instance
    if _instance is None:
        _instance = MyHookPanel()
    _instance.draw(layout)


def register():
    lf.ui.add_hook("rendering", "selection_groups", _draw_hook, "append")


def unregister():
    lf.ui.remove_hook("rendering", "selection_groups", _draw_hook)
```

The `position` argument controls whether the hook draws before (`"prepend"`) or after (`"append"`) the native content at that hook point.

### Available hook points

| Panel | Section | Description |
|---|---|---|
| `"rendering"` | `"selection_groups"` | Rendering panel, between settings and tools |

### Decorator form

```python
@lf.ui.hook("rendering", "selection_groups", "append")
def my_hook(layout):
    layout.label("Hello from hook")
```

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

The `Event` object is passed to `invoke()` and `modal()`:

| Attribute        | Type    | Description                              |
|------------------|---------|------------------------------------------|
| `type`           | `str`   | `'MOUSEMOVE'`, `'LEFTMOUSE'`, `'KEY_A'`-`'KEY_Z'`, `'ESC'`, `'RET'`, `'SPACE'`, `'WHEELUPMOUSE'`, `'WHEELDOWNMOUSE'`, etc. |
| `value`          | `str`   | `'PRESS'`, `'RELEASE'`, `'NOTHING'`      |
| `mouse_x`        | `float` | Mouse X (viewport coords)               |
| `mouse_y`        | `float` | Mouse Y (viewport coords)               |
| `mouse_region_x` | `float` | Mouse X relative to region               |
| `mouse_region_y` | `float` | Mouse Y relative to region               |
| `delta_x`        | `float` | Mouse delta X                            |
| `delta_y`        | `float` | Mouse delta Y                            |
| `scroll_x`       | `float` | Scroll X offset                          |
| `scroll_y`       | `float` | Scroll Y offset                          |
| `shift`          | `bool`  | Shift held                               |
| `ctrl`           | `bool`  | Ctrl held                                |
| `alt`            | `bool`  | Alt held                                 |
| `pressure`       | `float` | Tablet pressure (1.0 for mouse)          |
| `over_gui`       | `bool`  | True if mouse is over a GUI element      |
| `key_code`       | `int`   | Key code (see `key_codes.hpp`)           |

### Example: simple execute-only operator

```python
import lichtfeld as lf
from lfs_plugins.types import Operator
from lfs_plugins.props import FloatProperty

class ResetOpacity(Operator):
    label = "Reset Opacity"
    description = "Set opacity of all gaussians to a given value"

    target_opacity: float = FloatProperty(default=1.0, min=0.0, max=1.0)

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()

    def execute(self, context) -> set:
        scene = lf.get_scene()
        model = scene.combined_model()
        n = model.num_points
        mask = lf.Tensor.ones([n, 1], device="cuda")
        scaled = mask * self.target_opacity
        # Apply to opacity (working in logit space requires inverse sigmoid)
        lf.log.info(f"Reset {n} gaussians to opacity {self.target_opacity}")
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
    group="utility",                # "select", "transform", "paint", "utility"
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

paint_tool = ToolDef(
    id="my_plugin.paint",
    label="Paint",
    icon="paint",
    group="paint",
    order=100,
    description="Paint gaussian attributes",
    submodes=(
        SubmodeDef("opacity", "Opacity", "opacity"),
        SubmodeDef("color", "Color", "color"),
        SubmodeDef("scale", "Scale", "scale"),
    ),
    pivot_modes=(
        PivotModeDef("center", "Selection Center", "circle-dot"),
        PivotModeDef("cursor", "3D Cursor", "crosshair"),
    ),
    poll=lambda ctx: ctx.has_scene,
    plugin_name="my_plugin",
    plugin_path=str(Path(__file__).parent),
)

ToolRegistry.register_tool(paint_tool)
```

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
| `CollectionProperty`| `[]`        | `type=PropertyGroupSubclass`              |
| `PointerProperty`   | `None`      | `type=PropertyGroupSubclass`              |

All properties accept: `name`, `description`, `subtype`, `update` (callback).

### PropertyGroup base class

```python
from lfs_plugins.props import PropertyGroup, FloatProperty, StringProperty

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
    rotation=lf.Tensor.zeros([100, 4], device="cuda"),
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

# Get/set as 16-float column-major matrix
matrix = lf.get_node_transform("My Splat")
lf.set_node_transform("My Splat", matrix)

# Decompose/compose
components = lf.decompose_transform(matrix)
# components = {"translation": [x,y,z], "euler": [rx,ry,rz], "scale": [sx,sy,sz]}

new_matrix = lf.compose_transform(
    translation=[1.0, 2.0, 3.0],
    euler_deg=[0.0, 45.0, 0.0],
    scale=[1.0, 1.0, 1.0],
)
```

### Splat data access

Splat data can be accessed from the combined model or from individual scene nodes:

```python
scene = lf.get_scene()

# Combined model (all splat nodes merged)
model = scene.combined_model()

# Per-node access
for node in scene.get_nodes():
    sd = node.splat_data()       # None for non-splat nodes
    if sd is not None:
        print(f"{node.name}: {sd.num_points} gaussians")
```

```python
# Raw data (views into GPU memory — no copy)
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
        center = lf.get_selection_world_center()
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
# Subscribers notified once here, not three times
```

### AppState

Pre-defined signals for application state:

```python
from lfs_plugins.ui.state import AppState

# Training
AppState.is_training              # Signal[bool]
AppState.trainer_state            # Signal[str] - "idle", "ready", "running", "paused", "stopping"
AppState.has_trainer              # Signal[bool]
AppState.iteration                # Signal[int]
AppState.max_iterations           # Signal[int]
AppState.loss                     # Signal[float]
AppState.psnr                     # Signal[float]
AppState.num_gaussians            # Signal[int]

# Scene
AppState.has_scene                # Signal[bool]
AppState.scene_generation         # Signal[int] - increments on scene change
AppState.scene_path               # Signal[str]

# Selection
AppState.has_selection            # Signal[bool]
AppState.selection_count          # Signal[int]
AppState.selection_generation     # Signal[int]

# Viewport
AppState.viewport_width           # Signal[int]
AppState.viewport_height          # Signal[int]

# Computed
AppState.training_progress        # ComputedSignal[float] - 0.0 to 1.0
AppState.can_start_training       # ComputedSignal[bool]
```

### Example: reactive training monitor

```python
import lichtfeld as lf
from lfs_plugins.ui.state import AppState
from lfs_plugins.ui.signals import Signal

class TrainingMonitor(lf.ui.Panel):
    label = "Training Monitor"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 50

    def __init__(self):
        self.best_loss = Signal(float("inf"), name="best_loss")
        self.loss_history = []

        AppState.loss.subscribe_as("my_plugin", self._on_loss_change)

    def _on_loss_change(self, loss: float):
        if loss > 0:
            self.loss_history.append(loss)
            if loss < self.best_loss.value:
                self.best_loss.value = loss

    @classmethod
    def poll(cls, context) -> bool:
        return AppState.has_trainer.value

    def draw(self, ui):
        ui.heading("Training Monitor")

        state = AppState.trainer_state.value
        ui.label(f"State: {state}")
        ui.label(f"Iteration: {AppState.iteration.value}")
        ui.label(f"Loss: {AppState.loss.value:.6f}")
        ui.label(f"Best Loss: {self.best_loss.value:.6f}")
        ui.label(f"PSNR: {AppState.psnr.value:.2f}")
        ui.label(f"Gaussians: {AppState.num_gaussians.value:,}")

        progress = AppState.training_progress.value
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
lf.save_checkpoint()
```

### Example: custom training callback

```python
import lichtfeld as lf

class AutoSavePlugin:
    """Automatically save checkpoints every N iterations."""

    def __init__(self, interval=5000):
        self.interval = interval
        self.last_save = 0

    def on_post_step(self, _hook):
        ctx = lf.context()
        if ctx.iteration - self.last_save >= self.interval:
            lf.save_checkpoint()
            self.last_save = ctx.iteration
            lf.log.info(f"Auto-save requested at iteration {ctx.iteration}")

_auto_save = None

def on_load():
    global _auto_save
    _auto_save = AutoSavePlugin(interval=5000)
    lf.on_post_step(_auto_save.on_post_step)
    lf.log.info("Auto-save plugin loaded")

def on_unload():
    global _auto_save
    _auto_save = None
```

`lf.log.info(...)` writes to the main LichtFeld application log. Use `print(...)` if you want temporary debug output in the integrated Python console output panel, or call both if you want messages in both places.

---

## Hot Reload & Debugging

### File watcher

When `hot_reload = true` in `pyproject.toml`, LichtFeld watches your plugin directory for changes. On any `.py` file save, the plugin is automatically unloaded and reloaded.

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

### Auto-generated pyrightconfig.json

LichtFeld generates a `pyrightconfig.json` in the project root that includes the correct Python paths for type checking.

### VS Code

Add to `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": [
        "/path/to/gaussian-splatting-cuda/src/python",
        "/path/to/gaussian-splatting-cuda/build/src/python/typings"
    ]
}
```

### Type stubs

Type stubs are generated at `build/src/python/typings/` and provide autocomplete for:
- `lichtfeld` - Main API (scene, training, rendering, etc.)
- `lichtfeld.ui` - UI functions
- `lichtfeld.scene` - Scene types
- `lichtfeld.selection` - Selection types
- `lichtfeld.plugins` - Plugin management

The committed SDK stubs live in `src/python/stubs/` and are checked against the generated output during the build. If you intentionally change the Python API surface, refresh the committed stubs with:

```bash
cmake --build build --target refresh_python_stubs
```

You can also run the check explicitly with:

```bash
cmake --build build --target check_python_stubs
```

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

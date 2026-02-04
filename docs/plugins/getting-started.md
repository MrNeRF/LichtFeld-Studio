# Plugin Developer Guide

LichtFeld Studio plugins extend the application with custom panels, operators, toolbar tools, and more. Plugins are written in Python and live in `~/.lichtfeld/plugins/`.

## Quick Start

### Plugin directory structure

```
~/.lichtfeld/plugins/my_plugin/
├── plugin.toml          # Plugin manifest (required)
├── __init__.py          # Entry point with on_load/on_unload (required)
├── panels/
│   ├── __init__.py
│   └── main_panel.py
├── operators/
│   └── my_operator.py
└── icons/               # Custom icons (PNG)
    └── my_icon.png
```

### plugin.toml manifest

Every plugin requires a `plugin.toml` at its root:

```toml
[plugin]
name = "my_plugin"
version = "0.1.0"
description = "What this plugin does"
author = "Your Name"

[dependencies]
packages = []               # Python packages to install, e.g. ["numpy>=1.20"]

[lifecycle]
auto_start = true           # Load automatically on startup
hot_reload = true           # Reload on file changes during development
```

### Entry point

`__init__.py` must define `on_load()` and `on_unload()`:

```python
import lichtfeld as lf
from .panels.main_panel import HelloPanel

_classes = [HelloPanel]

def on_load():
    for cls in _classes:
        lf.register_class(cls)
    lf.log.info("my_plugin loaded")

def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
    lf.log.info("my_plugin unloaded")
```

### Minimal "Hello World" plugin

**plugin.toml**:
```toml
[plugin]
name = "hello_world"
version = "0.1.0"
description = "Hello World example"
author = ""

[dependencies]
packages = []

[lifecycle]
auto_start = true
hot_reload = true
```

**__init__.py**:
```python
import lichtfeld as lf
from lfs_plugins.types import Panel

class HelloPanel(Panel):
    label = "Hello World"
    space = "SIDE_PANEL"
    order = 200

    def draw(self, layout):
        layout.label("Hello from my plugin!")

_classes = [HelloPanel]

def on_load():
    for cls in _classes:
        lf.register_class(cls)

def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)
```

---

## Panels

Panels are UI elements that draw into designated spaces using the layout API.

### Panel base class

```python
from lfs_plugins.types import Panel

class MyPanel(Panel):
    label = "My Panel"       # Display name
    space = "SIDE_PANEL"     # Where it appears
    order = 100              # Sort order (lower = higher)

    @classmethod
    def poll(cls, context) -> bool:
        """Return False to hide the panel."""
        return True

    def draw(self, layout):
        """Draw panel contents using the layout API."""
        layout.label("Content here")
```

### Panel spaces

| Space             | Description                        |
|-------------------|------------------------------------|
| `SIDE_PANEL`      | Right sidebar panel                |
| `MAIN_PANEL_TAB`  | Tab in the main panel area         |
| `VIEWPORT_OVERLAY`| Drawn over the 3D viewport         |
| `SCENE_HEADER`    | Header area above the scene tree   |
| `FLOATING`        | Free-floating window               |
| `DOCKABLE`        | Dockable window                    |
| `STATUS_BAR`      | Bottom status bar                  |

### Register and unregister

```python
import lichtfeld as lf

lf.register_class(MyPanel)      # Makes the panel visible
lf.unregister_class(MyPanel)    # Removes the panel
```

### Example: side panel with interactive widgets

```python
import lichtfeld as lf
from lfs_plugins.types import Panel

class SettingsPanel(Panel):
    label = "Settings"
    space = "SIDE_PANEL"
    order = 50

    def __init__(self):
        self.opacity = 0.8
        self.name = "Default"
        self.enabled = True

    def draw(self, layout):
        layout.heading("Plugin Settings")

        changed, self.enabled = layout.checkbox("Enable##settings", self.enabled)

        if self.enabled:
            changed, self.opacity = layout.slider_float(
                "Opacity##settings", self.opacity, 0.0, 1.0
            )
            changed, self.name = layout.input_text("Name##settings", self.name)

            layout.separator()

            if layout.collapsing_header("Advanced", default_open=False):
                layout.text_wrapped("Advanced settings go here.")
                if layout.button("Reset All"):
                    self.opacity = 0.8
                    self.name = "Default"
                    self.enabled = True
```

### Layout composition

Sub-layouts let you compose UI structure declaratively. Each sub-layout is a context manager that positions its children automatically.

```python
def draw(self, layout):
    # Row: children placed side by side
    with layout.row() as row:
        row.button("A")
        row.button("B")

    # Box: bordered container
    with layout.box() as box:
        box.heading("Settings")
        box.prop(self, "opacity")

    # Split: two-column layout
    with layout.split(0.3) as split:
        split.label("Name")
        split.prop(self, "name")

    # State cascading: disable all children
    with layout.column() as col:
        col.enabled = self.is_active
        col.prop(self, "value")
        with col.row() as row:  # inherits disabled state
            row.button("Apply")
            row.button("Cancel")

    # Responsive grid
    with layout.grid_flow(columns=3) as grid:
        for item in items:
            grid.button(item.name)

    # Enum toggle buttons
    with layout.row() as row:
        row.prop_enum(self, "mode", "fast", "Fast")
        row.prop_enum(self, "mode", "quality", "Quality")
```

See [layout examples](examples/) for more patterns.

### Example: viewport overlay

```python
from lfs_plugins.types import Panel
from lfs_plugins.ui.state import AppState

class StatsOverlay(Panel):
    label = "Stats"
    space = "VIEWPORT_OVERLAY"
    order = 10

    @classmethod
    def poll(cls, context) -> bool:
        return AppState.has_scene.value

    def draw(self, layout):
        n = AppState.num_gaussians.value
        layout.draw_text(10, 10, f"Gaussians: {n:,}", (1.0, 1.0, 1.0, 0.8))
```

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
| `key_code`       | `int`   | Raw GLFW key code                        |

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
    options = {"UNDO"}

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
from lfs_plugins.types import Panel, Operator

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
from lfs_plugins.types import Panel
from lfs_plugins.ui.state import AppState
from lfs_plugins.ui.signals import Signal

class TrainingMonitor(Panel):
    label = "Training Monitor"
    space = "SIDE_PANEL"
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

    def draw(self, layout):
        layout.heading("Training Monitor")

        state = AppState.trainer_state.value
        layout.label(f"State: {state}")
        layout.label(f"Iteration: {AppState.iteration.value}")
        layout.label(f"Loss: {AppState.loss.value:.6f}")
        layout.label(f"Best Loss: {self.best_loss.value:.6f}")
        layout.label(f"PSNR: {AppState.psnr.value:.2f}")
        layout.label(f"Gaussians: {AppState.num_gaussians.value:,}")

        progress = AppState.training_progress.value
        layout.progress_bar(progress, f"{progress * 100:.1f}%")

        if self.loss_history:
            layout.plot_lines(
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
def on_start():
    lf.log.info("Training started")

@lf.on_iteration_start
def on_iter():
    pass

@lf.on_pre_optimizer_step
def on_pre_opt():
    pass

@lf.on_post_step
def on_post():
    ctx = lf.context()
    if ctx.iteration % 1000 == 0:
        lf.log.info(f"Iteration {ctx.iteration}, loss: {ctx.loss:.6f}")

@lf.on_training_end
def on_end():
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
from lfs_plugins.types import Panel
from lfs_plugins.ui.state import AppState

class AutoSavePlugin:
    """Automatically save checkpoints every N iterations."""

    def __init__(self, interval=5000):
        self.interval = interval
        self.last_save = 0

    def on_post_step(self):
        ctx = lf.context()
        if ctx.iteration - self.last_save >= self.interval:
            lf.save_checkpoint()
            self.last_save = ctx.iteration
            lf.log.info(f"Auto-saved at iteration {ctx.iteration}")

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

---

## Hot Reload & Debugging

### File watcher

When `hot_reload = true` in `plugin.toml`, LichtFeld watches your plugin directory for changes. On any `.py` file save, the plugin is automatically unloaded and reloaded.

### Logging

```python
import lichtfeld as lf

lf.log.info("Informational message")
lf.log.warn("Warning message")
lf.log.error("Error message")
lf.log.debug("Debug message")    # Only visible with --log-level debug
```

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
path = lf.plugins.create("my_new_plugin")  # Creates from template
```

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

### plugin.toml packaging requirements

For publishing, ensure your `plugin.toml` includes:
- `name` - Unique plugin identifier
- `version` - Semantic version (e.g., `"1.0.0"`)
- `description` - Clear description of what the plugin does
- `author` - Your name or organization
- `packages` - Any Python dependencies

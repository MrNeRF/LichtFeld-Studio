# Plugin API Reference

Complete API reference for LichtFeld Studio plugins.

This page is aligned against the committed Python stubs in
`src/python/stubs/lichtfeld/` and the plugin framework in
`src/python/lfs_plugins/`. When in doubt, the stubs are the quickest way to
verify exact signatures.

---

## Python API Surface Map

The public Python surface is split between the native `lichtfeld` module and
the pure-Python plugin helpers in `lfs_plugins`.

| Module | Main responsibility |
|---|---|
| `lichtfeld` | Training control, scene shortcuts, tensors, rendering, viewport, transform gizmos, registration, app helpers |
| `lichtfeld.app` | Application-level file open helper |
| `lichtfeld.animation` | Tracks, clips, and timeline evaluation |
| `lichtfeld.io` | Load/save splats, point clouds, datasets, images, and supported format queries |
| `lichtfeld.keymap` | Input binding profiles, action lookup, capture, import/export |
| `lichtfeld.log` | Native log sink |
| `lichtfeld.mcp` | Register Python MCP tools and call/read shared MCP capabilities/resources |
| `lichtfeld.mesh` | OpenMesh-style mesh data, handles, iterators, readers/writers, decimaters |
| `lichtfeld.ops` | Native operator invocation, descriptors, built-in operator/tool enums |
| `lichtfeld.packages` | `uv`-backed package install/list/uninstall and stub path helpers |
| `lichtfeld.pipeline` | Chainable selection/edit/transform operation stages |
| `lichtfeld.plugins` | Plugin discovery, lifecycle, registry install/update, capabilities, settings, scaffolding |
| `lichtfeld.scene` | Scene graph, nodes, splat data, point clouds, cameras, selection groups |
| `lichtfeld.scripts` | Script panel state and batch execution |
| `lichtfeld.selection` | Gaussian stroke/preview/brush/lasso/depth/crop selection primitives |
| `lichtfeld.ui` | Panels, menus, immediate UI, dialogs, theme, tools, app UI state, RML bridge |
| `lichtfeld.undo` | Undo/redo stack, transactions, memory accounting, subscriptions |
| `lfs_plugins.*` | Plugin base types, properties, runtime state bindings, tool definitions, capabilities, templates, managers |

The sections below focus on the APIs plugin authors most often call directly.
Large low-level surfaces such as `lichtfeld.mesh.TriMesh` and every tensor
operator are intentionally summarized; inspect the `.pyi` files for exhaustive
method lists.

## Documentation Map

Python and plugin API documentation currently lives in these places:

| Path | Purpose |
|---|---|
| `docs/plugins/getting-started.md` | Plugin authoring guide, common workflows, panel/operator examples, and runtime patterns |
| `docs/plugins/api-reference.md` | Practical Python/plugin API reference aligned with the committed stubs |
| `docs/plugins/examples/README.md` and `docs/plugins/examples/` | Runnable example plugins and focused API examples |
| `docs/plugin-system.md` | Plugin runtime architecture, manager responsibilities, scaffolding, and packaging overview |
| `docs/plugin-dev-workflow.md` | CLI/Python workflow for creating, validating, installing, and iterating on plugins |
| `docs/Python_UI.md` | Compatibility redirect for the old Python UI document |
| `docs/Python_API_issues.md` | Known Python API issues, binding gaps, stale-doc corrections, and follow-up recommendations |
| `docs/docs/development/mcp/` | MCP automation guide, resources, tools, and workflow recipes |
| `docs/docs/development/rmlui-styling.md` | RmlUI/RCSS styling rules used by retained Python panels |

## Registration

```python
import lichtfeld as lf

lf.register_class(cls)           # Register a Panel, Operator, or Menu class
lf.unregister_class(cls)         # Unregister a Panel, Operator, or Menu class
```

---

## Scrub Controls

```python
from lfs_plugins import ScrubFieldController, ScrubFieldSpec
```

Retained panels can use these helpers to turn a range slider row (`input.setting-slider`) into a scrub field:

- Drag horizontally on the scrub field to scrub values.
- Click the numeric text area to type a value directly.
- The controller keeps the displayed value in sync and applies clamping, snapping, and fill width updates.

```python
SCRUB_FIELD_SPECS = {
    "quality": ScrubFieldSpec(min_value=0.0, max_value=1.0, step=0.01, fmt="%.2f"),
}

class MyPanel(lf.ui.Panel):
    # ...
    def on_bind_model(self, ctx):
        model = ctx.create_data_model("my_panel")
        if model is None:
            return
        model.bind("quality", lambda: f"{self._quality:.2f}", self._set_quality)

    def __init__(self):
        self._scrub_fields = ScrubFieldController(
            SCRUB_FIELD_SPECS,
            self._get_scrub_value,
            self._set_scrub_value,
        )

    def on_mount(self, doc):
        self._scrub_fields.mount(doc)

    def on_unmount(self, doc):
        self._scrub_fields.unmount()

    def on_update(self, doc):
        return self._scrub_fields.sync_all()
```

Each scrubbed `data-value` still needs a normal `model.bind(...)` entry. The controller upgrades the range input UI, but it does not create data-model variables for you.

`ScrubFieldSpec` fields are `min_value`, `max_value`, `step`, `fmt`,
`data_type` (default `float`), and `pixels_per_step` (unused in the current controller implementation).

## Panel

```python
import lichtfeld as lf
# lf.ui.Panel is the base class for all panels
```

| Attribute | Type | Default | Description |
|---|---|---|---|
| `id` | `str` | `module.qualname` | Unique panel identifier |
| `label` | `str` | `""` | Display name (`id` fallback when empty) |
| `space` | `lf.ui.PanelSpace` | `lf.ui.PanelSpace.MAIN_PANEL_TAB` | Panel space (see below) |
| `parent` | `str` | `""` | Parent panel id. Embeds as a collapsible section; embedded panels must not override `space` |
| `order` | `int` | `100` | Sort order (lower = higher) |
| `options` | `set[lf.ui.PanelOption]` | `set()` | `DEFAULT_CLOSED`, `HIDE_HEADER` |
| `poll_dependencies` | `set[lf.ui.PollDependency]` | `{SCENE, SELECTION, TRAINING}` | Which state changes trigger `poll()` |
| `size` | `tuple[float, float] \| None` | `None` | Initial width/height hint, mainly for floating panels |
| `template` | `str \| os.PathLike[str]` | `""` | Retained RML template. Use an absolute path for plugin-local files |
| `style` | `str` | `""` | Inline RCSS appended to the retained document |
| `height_mode` | `lf.ui.PanelHeightMode` | `lf.ui.PanelHeightMode.FILL` | `FILL` or `CONTENT` for retained panels |
| `update_policy` | `str` | `"interval"` | Set to `"dirty"` or `"reactive"` for retained panels that update from explicit model/store invalidation |
| `update_interval_ms` | `int` | `100` | Fallback cadence for retained/hybrid `on_update()` work. Prefer `update_policy = "dirty"` for data-driven panels |

| Method | Returns | Description |
|---|---|---|
| `poll(cls, context)` | `bool` | Classmethod. Show/hide condition |
| `draw(self, ui)` | `None` | Immediate-mode content |
| `on_bind_model(self, ctx)` | `None` | Bind retained data models before document load |
| `on_mount(self, doc)` | `None` | Called once after the retained document mounts |
| `on_unmount(self, doc)` | `None` | Called before the retained document is destroyed |
| `on_update(self, doc)` | `None \| bool` | Retained update hook. With `update_policy = "interval"` it runs on the interval; with `"dirty"` it runs only after explicit invalidation, scene changes, or update requests. Return `True` to mark content dirty |
| `on_scene_changed(self, doc)` | `None` | Called when the active scene generation changes |

Registering a panel with the same `id` as an existing panel replaces it (see [Panel replacement](getting-started.md#panel-replacement)).

`lf.ui.Panel` is unified: a panel can start as `draw(ui)` only and later add `template`, `style`, `height_mode`, or retained hooks without switching base classes or rewriting the panel body.

Panel definitions are validated during `lf.register_class()`. Invalid enum values, removed legacy field names, unsupported retained features on `VIEWPORT_OVERLAY`, or conflicting embedded-panel fields raise `ValueError`, `TypeError`, or `AttributeError`.

The panel API is strict in v1: use the enum values above, not string literals.

### Reactive retained panels

For retained RML panels, prefer dirty-policy updates over timer polling. A dirty-policy panel runs `on_update()` only when scene state changes, document/model state is marked dirty, or an explicit update is requested.

```python
import lichtfeld as lf
from lfs_plugins.ui import RuntimeState, PanelStateBinding


class MyPanel(lf.ui.Panel):
    id = "my_plugin.panel"
    label = "My Panel"
    template = "/absolute/path/to/main_panel.rml"
    update_policy = "dirty"

    def __init__(self):
        self._handle = None
        self._store_binding = PanelStateBinding()
        self._title = "No scene"

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("my_plugin_panel")
        if model is None:
            return
        model.bind_func("title", lambda: self._title)
        self._handle = model.get_handle()

    def on_mount(self, doc):
        self._store_binding.set_handle(self._handle).watch(
            RuntimeState.scene_generation,
            RuntimeState.selection_generation,
            refresh=self._refresh_title,
            dirty="title",
            immediate=True,
        )

    def on_unmount(self, doc):
        self._store_binding.close()
        doc.remove_data_model("my_plugin_panel")
        self._handle = None

    def _refresh_title(self):
        scene = lf.get_scene()
        self._title = getattr(scene, "name", "Scene") if scene else "No scene"
```

Use `PanelStateBinding` for normal panel subscriptions. It keeps subscription lifetime and RML invalidation together:

| API | Purpose |
|---|---|
| `RuntimeState.<field>.value` | Read or publish a current app value |
| `RuntimeState.<field>.subscribe(callback)` | Low-level subscription, mostly for non-panel code |
| `PanelStateBinding(handle).watch(...)` | Preferred retained-panel subscription helper |
| `dirty=None` | Request `on_update()` without dirtying every bound variable |
| `dirty="field"` | Dirty one data-model variable |
| `dirty=("a", "b")` | Dirty several data-model variables |
| `dirty="*"` | Dirty the full data model |
| `batch_updates()` | Publish several store fields atomically |

Store fields currently exposed to plugins:

```python
RuntimeState.iteration
RuntimeState.total_iterations
RuntimeState.loss
RuntimeState.num_gaussians
RuntimeState.max_gaussians
RuntimeState.training_running
RuntimeState.training_state
RuntimeState.trainer_loaded
RuntimeState.eval_psnr
RuntimeState.eval_ssim
RuntimeState.scene_generation
RuntimeState.selection_generation
RuntimeState.fps
RuntimeState.mode_text
RuntimeState.active_tool
RuntimeState.active_submode
RuntimeState.transform_space
RuntimeState.pivot_mode
RuntimeState.import_overlay_state
RuntimeState.video_export_overlay_state
RuntimeState.export_progress_state
RuntimeState.mesh2splat_state
RuntimeState.splat_simplify_state
RuntimeState.scripts_generation
RuntimeState.language_generation
```

`AppState`, `AppStore`, and `NativeAppStore` remain as compatibility aliases for older plugins. New plugin code should import `RuntimeState` from `lfs_plugins.ui`.

Old Python UI hooks still compile, but hook registration is deprecated for external plugins. Use retained RML data models plus `RuntimeState` subscriptions for new UI.

### Panel spaces

`MAIN_PANEL_TAB`, `SIDE_PANEL`, `VIEWPORT_OVERLAY`, `SCENE_HEADER`, `FLOATING`, `STATUS_BAR`

### Retained shell behavior

If a panel uses retained features and `template` is empty, LichtFeld selects a shell automatically:

- `FLOATING` -> `rmlui/floating_window.rml`
- `STATUS_BAR` -> `rmlui/status_bar_panel.rml`
- Other retained panel spaces -> `rmlui/docked_panel.rml`

Built-in template aliases:

- `builtin:docked-panel`
- `builtin:floating-window`
- `builtin:status-bar`

### Panel styling guide

| Goal | Use | Notes |
|---|---|---|
| Minimal panel | `draw(self, ui)` | No extra files needed |
| Light retained styling | `style` | Inline RCSS text, not a path |
| Full custom retained UI | `template` | Use an absolute path for plugin-local `.rml` |
| Hybrid panel | `template` plus `draw(ui)` | Render immediate content into `<div id="im-root"></div>` |

When a plugin-local template file such as `main_panel.rml` is present, LichtFeld automatically loads a sibling `main_panel.rcss` stylesheet if it exists. A sibling `main_panel.theme.rcss` file is also loaded for palette-dependent overrides.

---

## Operator

```python
from lfs_plugins.types import Operator, Event
```

Operator extends `PropertyGroup`, so it supports typed properties as class attributes.

| Attribute     | Type       | Description                              |
|---------------|------------|------------------------------------------|
| `label`       | `str`      | Display name                             |
| `description` | `str`      | Tooltip text                             |
| `options`     | `Set[str]` | `{'UNDO', 'BLOCKING'}`                   |

| Method                          | Returns | Description                       |
|---------------------------------|---------|-----------------------------------|
| `poll(cls, context)`            | `bool`  | Classmethod. Can the op run?      |
| `invoke(self, context, event)`  | `set`   | Called on trigger, can start modal|
| `execute(self, context)`        | `set`   | Synchronous execution             |
| `modal(self, context, event)`   | `set`   | Handle events in modal mode       |
| `cancel(self, context)`         | `None`  | Called on cancellation             |

### Return sets

`{"FINISHED"}`, `{"CANCELLED"}`, `{"RUNNING_MODAL"}`, `{"PASS_THROUGH"}`

Or dict form: `{"status": "FINISHED", "key": value, ...}`

---

## Event

```python
from lfs_plugins.types import Event
```

| Attribute        | Type    | Description                                    |
|------------------|---------|------------------------------------------------|
| `type`           | `str`   | `'MOUSEMOVE'`, `'LEFTMOUSE'`, `'RIGHTMOUSE'`, `'MIDDLEMOUSE'`, `'KEY_A'`-`'KEY_Z'`, `'WHEELUPMOUSE'`, `'WHEELDOWNMOUSE'`, `'ESC'`, `'RET'`, `'SPACE'` |
| `value`          | `str`   | `'PRESS'`, `'RELEASE'`, `'NOTHING'`            |
| `mouse_x`        | `float` | Mouse X (viewport coords)                     |
| `mouse_y`        | `float` | Mouse Y (viewport coords)                     |
| `mouse_region_x` | `float` | Mouse X relative to region                     |
| `mouse_region_y` | `float` | Mouse Y relative to region                     |
| `delta_x`        | `float` | Mouse delta X                                  |
| `delta_y`        | `float` | Mouse delta Y                                  |
| `scroll_x`       | `float` | Scroll X offset                                |
| `scroll_y`       | `float` | Scroll Y offset                                |
| `shift`          | `bool`  | Shift modifier                                 |
| `ctrl`           | `bool`  | Ctrl modifier                                  |
| `alt`            | `bool`  | Alt modifier                                   |
| `pressure`       | `float` | Tablet pressure (1.0 for mouse)                |
| `over_gui`       | `bool`  | Mouse is over GUI element                      |
| `key_code`       | `int`   | Key code (see `key_codes.hpp`)                 |

---

## Properties

```python
from lfs_plugins.props import (
    Property, FloatProperty, IntProperty, BoolProperty,
    StringProperty, EnumProperty, FloatVectorProperty,
    IntVectorProperty, TensorProperty, CollectionProperty,
    PointerProperty, PropertyGroup, PropSubtype,
)
```

### FloatProperty

```python
FloatProperty(
    default: float = 0.0,
    min: float = -inf,
    max: float = inf,
    step: float = 0.1,
    precision: int = 3,
    subtype: str = "",       # FACTOR, PERCENTAGE, ANGLE, TIME, DISTANCE, POWER
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### IntProperty

```python
IntProperty(
    default: int = 0,
    min: int = -2**31,
    max: int = 2**31 - 1,
    step: int = 1,
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### BoolProperty

```python
BoolProperty(
    default: bool = False,
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### StringProperty

```python
StringProperty(
    default: str = "",
    maxlen: int = 0,         # 0 = unlimited
    subtype: str = "",       # FILE_PATH, DIR_PATH, FILE_NAME
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### EnumProperty

```python
EnumProperty(
    items: list[tuple[str, str, str]] = [],  # (identifier, label, description)
    default: str = None,     # First item if None
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### FloatVectorProperty

```python
FloatVectorProperty(
    default: tuple = (0.0, 0.0, 0.0),
    size: int = 3,
    min: float = -inf,
    max: float = inf,
    subtype: str = "",       # COLOR, COLOR_GAMMA, TRANSLATION, DIRECTION,
                             # VELOCITY, ACCELERATION, XYZ, EULER, QUATERNION
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### IntVectorProperty

```python
IntVectorProperty(
    default: tuple = (0, 0, 0),
    size: int = 3,
    min: int = -2**31,
    max: int = 2**31 - 1,
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### TensorProperty

```python
TensorProperty(
    shape: tuple = (),       # Use -1 for variable dims, e.g. (-1, 3)
    dtype: str = "float32",
    device: str = "cuda",
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### CollectionProperty

```python
CollectionProperty(
    type: Type[PropertyGroup],  # Item type
    name: str = "",
    description: str = "",
)
```

| Method               | Returns           | Description              |
|----------------------|-------------------|--------------------------|
| `add()`              | `PropertyGroup`   | Add new item             |
| `remove(index)`      | `None`            | Remove by index          |
| `clear()`            | `None`            | Remove all items         |
| `move(from, to)`     | `None`            | Reorder items            |
| `__len__()`          | `int`             | Item count               |
| `__getitem__(index)` | `PropertyGroup`   | Access by index          |
| `__iter__()`         | `Iterator`        | Iterate items            |

### PointerProperty

```python
PointerProperty(
    type: Type[PropertyGroup],  # Referenced type
    name: str = "",
    description: str = "",
)
```

| Method           | Returns         | Description                    |
|------------------|-----------------|--------------------------------|
| `get_instance()` | `PropertyGroup` | Get or create referenced object|

### PropertyGroup

```python
from lfs_plugins.props import PropertyGroup
```

| Method                    | Returns                | Description                         |
|---------------------------|------------------------|-------------------------------------|
| `get_instance()`          | `cls`                  | Classmethod. Singleton access       |
| `add_property(name, prop)`| `None`                 | Add property at runtime             |
| `remove_property(name)`   | `None`                 | Remove runtime property             |
| `get_all_properties()`    | `dict[str, Property]`  | All properties (class + runtime)    |
| `get(prop_id)`            | `Any`                  | Get property value by name          |
| `set(prop_id, value)`     | `None`                 | Set property value by name          |

### PropSubtype constants

```python
from lfs_plugins.props import PropSubtype

PropSubtype.NONE             # ""
PropSubtype.FILE_PATH        # "FILE_PATH"
PropSubtype.DIR_PATH         # "DIR_PATH"
PropSubtype.FILE_NAME        # "FILE_NAME"
PropSubtype.COLOR            # "COLOR"
PropSubtype.COLOR_GAMMA      # "COLOR_GAMMA"
PropSubtype.TRANSLATION      # "TRANSLATION"
PropSubtype.DIRECTION        # "DIRECTION"
PropSubtype.VELOCITY         # "VELOCITY"
PropSubtype.ACCELERATION     # "ACCELERATION"
PropSubtype.XYZ              # "XYZ"
PropSubtype.EULER            # "EULER"
PropSubtype.QUATERNION       # "QUATERNION"
PropSubtype.AXISANGLE        # "AXISANGLE"
PropSubtype.ANGLE            # "ANGLE"
PropSubtype.FACTOR           # "FACTOR"
PropSubtype.PERCENTAGE       # "PERCENTAGE"
PropSubtype.TIME             # "TIME"
PropSubtype.DISTANCE         # "DISTANCE"
PropSubtype.POWER            # "POWER"
PropSubtype.TEMPERATURE      # "TEMPERATURE"
PropSubtype.PIXEL            # "PIXEL"
PropSubtype.UNSIGNED         # "UNSIGNED"
PropSubtype.LAYER            # "LAYER"
PropSubtype.LAYER_MEMBER     # "LAYER_MEMBER"
```

---

## ToolDef / ToolRegistry

### ToolDef

```python
from lfs_plugins.tool_defs.definition import ToolDef, SubmodeDef, PivotModeDef
```

```python
@dataclass(frozen=True)
class ToolDef:
    id: str                                      # Unique tool ID
    label: str                                   # Display label
    icon: str                                    # Icon name
    group: str = "default"                       # "select", "transform", "utility"
    order: int = 100                             # Sort order within group
    description: str = ""                        # Tooltip
    shortcut: str = ""                           # Keyboard shortcut
    gizmo: str = ""                              # "translate", "rotate", "scale", ""
    operator: str = ""                           # Operator to invoke on activation
    submodes: tuple[SubmodeDef, ...] = ()
    pivot_modes: tuple[PivotModeDef, ...] = ()
    poll: Callable[[Any], bool] | None = None    # Availability check
    plugin_name: str = ""                        # For custom icon loading
    plugin_path: str = ""                        # For custom icon loading
```

| Method                  | Returns | Description                          |
|-------------------------|---------|--------------------------------------|
| `can_activate(context)` | `bool`  | Check if tool can be activated       |
| `to_dict()`             | `dict`  | Convert to dict for C++ interop      |

### SubmodeDef

```python
@dataclass(frozen=True)
class SubmodeDef:
    id: str           # Unique submode ID
    label: str        # Display label
    icon: str         # Icon name
    shortcut: str = ""
```

### PivotModeDef

```python
@dataclass(frozen=True)
class PivotModeDef:
    id: str           # Unique pivot mode ID
    label: str        # Display label
    icon: str         # Icon name
```

### ToolRegistry

```python
from lfs_plugins.tools import ToolRegistry
```

| Method                     | Returns              | Description                              |
|----------------------------|----------------------|------------------------------------------|
| `register_tool(tool)`      | `None`               | Register a custom tool                   |
| `unregister_tool(tool_id)` | `None`               | Unregister by ID                         |
| `get(tool_id)`             | `Optional[ToolDef]`  | Get tool by ID (builtins first)          |
| `get_all()`                | `list[ToolDef]`      | All tools (builtins + custom, sorted)    |
| `set_active(tool_id)`      | `bool`               | Activate a tool                          |
| `get_active()`             | `Optional[ToolDef]`  | Get active tool                          |
| `get_active_id()`          | `str`                | Get active tool ID                       |

---

## Native Transform Gizmos

| API | Returns | Description |
|-----|---------|-------------|
| `lf.TransformGizmo(operation="translate", matrix=[], id="")` | `TransformGizmo` | Reusable native TRS gizmo |
| `lf.TranslationGizmo(matrix=[], id="")` | `TransformGizmo` | Translate handle |
| `lf.RotationGizmo(matrix=[], id="")` | `TransformGizmo` | Rotate handle |
| `lf.ScaleGizmo(matrix=[], id="")` | `TransformGizmo` | Scale handle |
| `lf.get_transform_gizmo_ids()` | `list[str]` | Attached transform gizmo IDs |
| `lf.has_transform_gizmos()` | `bool` | Whether any native TRS gizmos are attached |
| `lf.clear_transform_gizmos()` | `None` | Detach all native TRS gizmos |

`TransformGizmo` properties:

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Stable ID |
| `operation` | `str` | `"translate"`, `"rotate"`, or `"scale"` |
| `space` | `str` | `"local"` or `"world"` |
| `matrix` | `list[float]` | 16 floats, column-major |
| `translation` | `list[float]` | Translation component |
| `visible`, `enabled`, `input_enabled` | `bool` | Runtime draw/input controls |
| `active`, `hovered`, `changed` | `bool` | Last-frame interaction state |
| `snap` | `bool` | Enable snapping |
| `translate_snap`, `rotate_snap_degrees`, `scale_snap_ratio` | `float` | Per-operation snap settings |

| Method | Description |
|--------|-------------|
| `attach()` | Draw without an automatic target |
| `attach_to_callbacks(getter, setter)` | Bind to arbitrary Python transform callbacks |
| `attach_to_node(node_name, visualizer_world=True)` | Bind to a scene node |
| `detach()` | Remove from viewport drawing |
| `set_on_begin(callback)`, `set_on_change(callback)`, `set_on_end(callback)` | Drag lifecycle callbacks |

---

## Signals

```python
from lfs_plugins.ui.signals import Signal, ComputedSignal, ThrottledSignal, Batch, batch
```

### Signal[T]

```python
Signal(initial_value: T, name: str = "")
```

| Property/Method                          | Returns           | Description                      |
|------------------------------------------|-------------------|----------------------------------|
| `.value`                                 | `T`               | Get/set current value            |
| `.peek()`                                | `T`               | Get without tracking             |
| `.subscribe(callback)`                   | `() -> None`      | Subscribe; returns unsubscribe fn|
| `.subscribe_as(owner, callback)`         | `() -> None`      | Owner-tracked subscription       |

### ComputedSignal[T]

```python
ComputedSignal(compute: Callable[[], T], dependencies: list[Signal])
```

| Property/Method                          | Returns           | Description                      |
|------------------------------------------|-------------------|----------------------------------|
| `.value`                                 | `T`               | Get computed value (lazy)        |
| `.subscribe(callback)`                   | `() -> None`      | Subscribe to changes             |
| `.subscribe_as(owner, callback)`         | `() -> None`      | Owner-tracked subscription       |

### ThrottledSignal[T]

```python
ThrottledSignal(initial_value: T, max_rate_hz: float = 60.0, name: str = "")
```

| Property/Method                          | Returns           | Description                      |
|------------------------------------------|-------------------|----------------------------------|
| `.value`                                 | `T`               | Get/set current value            |
| `.flush()`                               | `None`            | Force pending notification       |
| `.subscribe(callback)`                   | `() -> None`      | Subscribe to changes             |
| `.subscribe_as(owner, callback)`         | `() -> None`      | Owner-tracked subscription       |

### Batch / batch()

```python
with Batch():       # Class form
    ...

with batch():       # Function form
    ...
```

Defers all signal notifications until the block exits.

### SubscriptionRegistry

```python
from lfs_plugins.ui.subscription_registry import SubscriptionRegistry

registry = SubscriptionRegistry.instance()
unsub = registry.register(owner="my_plugin", unsubscribe_fn=fn)
registry.unregister_all("my_plugin")   # Cleanup on unload
```

---

## Capabilities

```python
from lfs_plugins.capabilities import CapabilityRegistry, CapabilitySchema, Capability
from lfs_plugins.context import PluginContext, SceneContext, ViewContext, CapabilityBroker
```

### CapabilityRegistry

```python
registry = CapabilityRegistry.instance()
```

| Method                                    | Returns              | Description                          |
|-------------------------------------------|----------------------|--------------------------------------|
| `register(name, handler, ...)`            | `None`               | Register a capability                |
| `unregister(name)`                        | `bool`               | Unregister by name                   |
| `unregister_all_for_plugin(plugin_name)`  | `int`                | Unregister all for plugin            |
| `invoke(name, args)`                      | `dict`               | Invoke capability                    |
| `get(name)`                               | `Optional[Capability]`| Get by name                         |
| `list_all()`                              | `list[Capability]`   | List all capabilities                |
| `has(name)`                               | `bool`               | Check existence                      |

#### register() parameters

```python
registry.register(
    name: str,                    # Unique name, e.g. "my_plugin.feature"
    handler: Callable,            # fn(args: dict, ctx: PluginContext) -> dict
    description: str = "",
    schema: CapabilitySchema = None,
    plugin_name: str = None,
    requires_gui: bool = True,
)
```

### CapabilitySchema

```python
@dataclass
class CapabilitySchema:
    properties: dict[str, dict[str, Any]]   # JSON Schema-like property defs
    required: list[str]                      # Required property names
```

### Capability

```python
@dataclass
class Capability:
    name: str
    description: str
    handler: Callable
    schema: CapabilitySchema
    plugin_name: Optional[str]
    requires_gui: bool
```

### PluginContext

```python
@dataclass
class PluginContext:
    scene: Optional[SceneContext]
    view: Optional[ViewContext]
    capabilities: CapabilityBroker
```

| Method                               | Returns         | Description                    |
|--------------------------------------|-----------------|--------------------------------|
| `build(registry, include_view=True)` | `PluginContext`  | Classmethod. Build from state  |

### SceneContext

```python
@dataclass
class SceneContext:
    scene: Any                          # PyScene object
```

| Method                       | Returns | Description                  |
|------------------------------|---------|------------------------------|
| `set_selection_mask(mask)`   | `None`  | Apply selection mask         |

### ViewContext

```python
@dataclass
class ViewContext:
    image: Any                          # [H, W, 3] tensor
    screen_positions: Optional[Any]     # [N, 2] tensor or None
    width: int
    height: int
    fov: float
    rotation: Any                       # [3, 3] tensor
    translation: Any                    # [3] tensor
```

### CapabilityBroker

```python
class CapabilityBroker:
    def invoke(self, name: str, args: dict = None) -> dict
    def has(self, name: str) -> bool
    def list_all(self) -> list[str]
```

---

## PluginManager

```python
from lfs_plugins.manager import PluginManager
```

```python
mgr = PluginManager.instance()
```

| Method                                | Returns                    | Description                       |
|---------------------------------------|----------------------------|-----------------------------------|
| `plugins_dir`                         | `Path`                     | Property. `~/.lichtfeld/plugins/` |
| `discover()`                          | `list[PluginInfo]`         | Scan for plugins                  |
| `load(name, on_progress=None)`        | `bool`                     | Load a plugin                     |
| `unload(name)`                        | `bool`                     | Unload a plugin                   |
| `reload(name)`                        | `bool`                     | Hot-reload a plugin               |
| `load_all()`                          | `dict[str, bool]`          | Load all user-enabled plugins     |
| `install(url, on_progress=None, auto_load=True)` | `str`          | Install from Git URL              |
| `uninstall(name)`                     | `bool`                     | Remove a plugin                   |
| `update(name, on_progress=None)`      | `bool`                     | Update a plugin                   |
| `search(query, compatible_only=True)` | `list[RegistryPluginInfo]` | Search registry                   |
| `check_updates()`                     | `dict[str, tuple]`         | Check installed plugin updates    |
| `get_state(name)`                     | `Optional[PluginState]`    | Get plugin state                  |
| `get_error(name)`                     | `Optional[str]`            | Get error message                 |
| `get_traceback(name)`                 | `Optional[str]`            | Get error traceback               |

### PluginInfo

```python
@dataclass
class PluginInfo:
    name: str
    version: str
    path: Path
    description: str = ""
    author: str = ""
    entry_point: str = "__init__"
    dependencies: list[str] = []
    auto_start: bool = False
    hot_reload: bool = True
    plugin_api: str = ""
    lichtfeld_version: str = ""
    required_features: list[str] = []
```

### PluginState

```python
class PluginState(Enum):
    UNLOADED = "unloaded"
    INSTALLING = "installing"
    LOADING = "loading"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"
```

### `lichtfeld.plugins` convenience API

```python
import lichtfeld as lf
```

| Function | Returns | Description |
|---|---|---|
| `lf.plugins.discover()` | `list[PluginInfo]` | Discover plugins in `~/.lichtfeld/plugins/` |
| `lf.plugins.load(name)` | `bool` | Load a plugin |
| `lf.plugins.unload(name)` | `bool` | Unload a plugin |
| `lf.plugins.reload(name)` | `bool` | Reload a plugin |
| `lf.plugins.load_all()` | `dict[str, bool]` | Load all user-enabled plugins |
| `lf.plugins.start_watcher()` | `None` | Start the hot-reload watcher |
| `lf.plugins.stop_watcher()` | `None` | Stop the hot-reload watcher |
| `lf.plugins.get_state(name)` | `PluginState \| None` | Read plugin state |
| `lf.plugins.get_error(name)` | `str \| None` | Read the last plugin error |
| `lf.plugins.get_traceback(name)` | `str \| None` | Read the full traceback |
| `lf.plugins.create(name)` | `str` | Create the v1 source scaffold in `~/.lichtfeld/plugins/<name>` |

`lf.plugins.create()` writes the source package, including `panels/main_panel.py`, `panels/main_panel.rml`, and `panels/main_panel.rcss`. If you want a scaffold that also adds `.venv`, `.vscode`, and `pyrightconfig.json`, use the CLI command `LichtFeld-Studio plugin create <name>`.

Runtime compatibility constants:

| Constant | Type | Description |
|---|---|---|
| `lf.PLUGIN_API_VERSION` | `str` | Host plugin API version |
| `lf.plugins.API_VERSION` | `str` | Same plugin API version through the plugin namespace |
| `lf.plugins.FEATURES` | `list[str]` | Supported optional plugin features on this host |

---

## Layout API

The `ui` object passed to a regular `Panel.draw()` is a live `RmlUILayout`,
an immediate widget API reconciled into retained RmlUi elements. Viewport
overlay panels and document-less draw hooks receive the compatibility
`UILayout`: its viewport drawing methods are live during the overlay frame,
while interactive controls warn once and return inert defaults.

### Text

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `label(text)`                                       | `None`  | Plain text               |
| `label_centered(text)`                              | `None`  | Centered text            |
| `heading(text)`                                     | `None`  | Large heading            |
| `text_colored(text, color)`                         | `None`  | Colored text (RGBA tuple)|
| `text_colored_centered(text, color)`                | `None`  | Centered colored text    |
| `text_selectable(text, height=0)`                   | `None`  | Selectable text          |
| `text_wrapped(text)`                                | `None`  | Word-wrapped text        |
| `text_disabled(text)`                               | `None`  | Grayed-out text          |
| `bullet_text(text)`                                 | `None`  | Bulleted text            |

### Buttons

| Method                                              | Returns | Description                    |
|-----------------------------------------------------|---------|--------------------------------|
| `button(label, size=(0,0))`                         | `bool`  | Standard button                |
| `button_styled(label, style, size=(0,0))`           | `bool`  | Styled: "success", "error", "warning", "primary", "secondary" |
| `button_callback(label, callback=None, size=(0,0))` | `bool`  | Button with callback           |
| `small_button(label)`                               | `bool`  | Compact button                 |
| `invisible_button(id, size)`                        | `bool`  | Invisible clickable area       |

### Input

| Method                                                              | Returns             | Description              |
|---------------------------------------------------------------------|---------------------|--------------------------|
| `checkbox(label, value)`                                            | `(bool, bool)`      | (changed, new_value)     |
| `radio_button(label, current, value)`                               | `(bool, int)`       | Radio button             |
| `input_text(label, value)`                                          | `(bool, str)`       | Text input               |
| `input_text_with_hint(label, hint, value)`                          | `(bool, str)`       | Text with placeholder    |
| `input_text_enter(label, value)`                                    | `(bool, str)`       | Confirm on Enter         |
| `input_float(label, value, step=0, step_fast=0, format='%.3f')`    | `(bool, float)`     | Float input              |
| `input_int(label, value, step=1, step_fast=100)`                    | `(bool, int)`       | Integer input            |
| `input_int_formatted(label, value, step=0, step_fast=0)`           | `(bool, int)`       | Formatted int input      |

### Sliders & Drags

| Method                                                              | Returns              | Description           |
|---------------------------------------------------------------------|----------------------|-----------------------|
| `slider_float(label, value, min, max)`                              | `(bool, float)`      | Float slider          |
| `slider_int(label, value, min, max)`                                | `(bool, int)`        | Integer slider        |
| `slider_float2(label, value, min, max)`                             | `(bool, tuple)`      | 2-component slider    |
| `slider_float3(label, value, min, max)`                             | `(bool, tuple)`      | 3-component slider    |
| `drag_float(label, value, speed=1, min=0, max=0)`                  | `(bool, float)`      | Float drag            |
| `drag_int(label, value, speed=1, min=0, max=0)`                    | `(bool, int)`        | Integer drag          |

### Selection

| Method                                                              | Returns              | Description              |
|---------------------------------------------------------------------|----------------------|--------------------------|
| `combo(label, current_idx, items)`                                  | `(bool, int)`        | Dropdown selector        |
| `listbox(label, current_idx, items, height_items=-1)`               | `(bool, int)`        | List selector            |
| `selectable(label, selected=False, height=0)`                       | `bool`               | Selectable item          |
| `prop_search(data, prop_id, search_data, search_prop, text='')`     | `(bool, int)`        | Searchable dropdown      |

### Color

| Method                                              | Returns              | Description              |
|-----------------------------------------------------|----------------------|--------------------------|
| `color_edit3(label, color)`                         | `(bool, tuple)`      | RGB color picker         |
| `color_edit4(label, color)`                         | `(bool, tuple)`      | RGBA color picker        |
| `color_button(label, color, size=(0,0))`            | `bool`               | Color swatch button      |

### File/Path

| Method                                              | Returns              | Description              |
|-----------------------------------------------------|----------------------|--------------------------|
| `path_input(label, value, folder_mode=True, dialog_title='')` | `(bool, str)` | Editable path plus native browse button on `RmlUILayout`; `folder_mode` selects folder vs file. A non-empty title is passed to the custom-title dialog path, while an empty title uses the native default. Unsupported on compatibility `UILayout`. |

### Property Binding

| Method                                              | Returns              | Description                              |
|-----------------------------------------------------|----------------------|------------------------------------------|
| `prop(data, prop_id, text=None)`                    | `(bool, Any)`        | Auto-widget based on property type       |

### Layout Structure

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `separator()`                                       | `None`  | Horizontal line          |
| `spacing()`                                         | `None`  | Vertical space           |
| `same_line(offset=0, spacing=-1)`                   | `None`  | Next widget on same line |
| `new_line()`                                        | `None`  | Force new line           |
| `indent(width=0)`                                   | `None`  | Increase indent          |
| `unindent(width=0)`                                 | `None`  | Decrease indent          |
| `begin_group()` / `end_group()`                     | `None`  | Logical widget group     |
| `set_next_item_width(width)`                        | `None`  | Width for next widget    |
| `dummy(size)`                                       | `None`  | Empty space placeholder  |

### Collapsible / Tree

| Method                                              | Returns | Description                    |
|-----------------------------------------------------|---------|--------------------------------|
| `collapsing_header(label, default_open=False)`      | `bool`  | Collapsible section            |
| `tree_node(label)`                                  | `bool`  | Tree node (call `tree_pop()`)  |
| `tree_node_ex(label, flags='')`                     | `bool`  | Extended tree node             |
| `tree_pop()`                                        | `None`  | Close tree node                |

### Tables

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `begin_table(id, columns)`                          | `bool`  | Start table              |
| `table_setup_column(label, width=0)`                | `None`  | Define column            |
| `table_headers_row()`                               | `None`  | Draw header row          |
| `table_next_row()`                                  | `None`  | Next row                 |
| `table_next_column()`                               | `None`  | Next column              |
| `table_set_column_index(column)`                    | `bool`  | Jump to column           |
| `table_set_bg_color(target, color)`                 | `None`  | Set row/cell background  |
| `end_table()`                                       | `None`  | End table                |

Rows are position-identified by default. If rows can be removed or reordered,
call `push_id()` with a stable value (a hidden `##key` is accepted) after
`begin_table()` and before `table_next_row()`, and keep that id active through
the row's cells. The Rml bridge then preserves the matching row, focus, caret,
listeners, and cell state across reconciliation.

### Images

| Method                                                       | Returns | Description            |
|--------------------------------------------------------------|---------|------------------------|
| `image(texture_id, size, tint=(1,1,1,1))`                   | `None`  | Display image          |
| `image_uv(texture_id, size, uv0, uv1, tint=(1,1,1,1))`     | `None`  | Image with UV coords   |
| `image_button(id, texture_id, size, tint=(1,1,1,1))`        | `bool`  | Clickable image        |
| `toolbar_button(id, tex, size, selected=F, disabled=F, tooltip='')` | `bool` | Toolbar icon button |
| `image_tensor(label, tensor, size, tint=None)`               | `None`  | Display a tensor as an image (cached by label) |
| `image_texture(texture, size, tint=None)`                    | `None`  | Display a `DynamicTexture` |

`image_tensor` is the simplest way to display a GPU tensor — it internally manages a `DynamicTexture` cached by `label`. The tensor must be `[H, W, 3]` or `[H, W, 4]` (RGB/RGBA). CPU tensors and non-float32 dtypes are converted automatically.

```python
ui.image_tensor("preview", my_tensor, (256, 256))
```

For full control (e.g. reusing one texture across multiple draw calls), use `DynamicTexture` directly:

```python
tex = lf.ui.DynamicTexture(tensor)   # or DynamicTexture() + tex.update(tensor)
ui.image_texture(tex, (256, 256))
```

---

### DynamicTexture

GPU tensor to UI texture bridge. In the Vulkan viewer this uses an opaque Vulkan UI texture id (`uint64`).

```python
tex = lf.ui.DynamicTexture()          # Empty
tex = lf.ui.DynamicTexture(tensor)    # From tensor
```

| Method / Property  | Returns              | Description                                    |
|--------------------|----------------------|------------------------------------------------|
| `update(tensor)`   | `None`               | Upload `[H, W, 3\|4]` tensor (auto-converts CPU→CUDA, uint8→float32) |
| `destroy()`        | `None`               | Release UI texture resources                   |
| `id`               | `int`                | Opaque Vulkan UI texture id (`uint64`)         |
| `width`            | `int`                | Current width in pixels                        |
| `height`           | `int`                | Current height in pixels                       |
| `valid`            | `bool`               | `True` if texture is initialized               |
| `uv1`             | `tuple[float, float]` | UV scale factors for power-of-2 padding        |

Calling `update()` with a different resolution automatically recreates the backend texture. Textures are freed on plugin unload via `lf.ui.free_plugin_textures(name)`.

### Drag & Drop

| Method                                              | Returns       | Description              |
|-----------------------------------------------------|---------------|--------------------------|
| `begin_drag_drop_source()`                          | `bool`        | Start drag source        |
| `set_drag_drop_payload(type, data)`                 | `None`        | Set drag payload         |
| `end_drag_drop_source()`                            | `None`        | End drag source          |
| `begin_drag_drop_target()`                          | `bool`        | Start drag target        |
| `accept_drag_drop_payload(type)`                    | `str or None` | Accept payload           |
| `end_drag_drop_target()`                            | `None`        | End drag target          |

### Popups & Menus

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `begin_popup(id)`                                   | `bool`  | Start popup              |
| `begin_context_menu(id='')`                         | `bool`  | Styled context menu      |
| `begin_popup_modal(title)`                          | `bool`  | Modal popup              |
| `open_popup(id)`                                    | `None`  | Trigger popup open       |
| `end_popup()` / `end_popup_modal()`                 | `None`  | End popup/modal          |
| `end_context_menu()`                                | `None`  | End context menu         |
| `close_current_popup()`                             | `None`  | Close current popup      |
| `begin_menu(label)`                                 | `bool`  | Start menu               |
| `end_menu()`                                        | `None`  | End menu                 |
| `begin_menu_bar()` / `end_menu_bar()`               | `bool`  | Menu bar                 |
| `menu_item(label, enabled=True)`                    | `bool`  | Menu item                |
| `menu_item_toggle(label, shortcut, selected)`       | `bool`  | Toggle menu item         |
| `menu_item_shortcut(label, shortcut, enabled=True)` | `bool`  | Menu item with shortcut  |
| `menu(menu_id, text='', icon='')`                   | `None`  | Inline menu reference    |
| `popover(panel_id, text='', icon='')`               | `None`  | Panel popover            |

### Windows & Children

| Method                                                        | Returns        | Description           |
|---------------------------------------------------------------|----------------|-----------------------|
| `begin_window(title, flags=0)`                                | `bool`         | Start window          |
| `begin_window_closable(title, flags=0)`                       | `(bool, bool)` | Closable window       |
| `end_window()`                                                | `None`         | End window            |
| `begin_child(id, size=(0,0), border=False)`                   | `bool`         | Start child region    |
| `end_child()`                                                 | `None`         | End child region      |
| `set_next_window_pos(pos, first_use=False)`                   | `None`         | Set window position   |
| `set_next_window_size(size, first_use=False)`                 | `None`         | Set window size       |
| `set_next_window_pos_center()`                                | `None`         | Center window         |
| `set_next_window_pos_centered(first_use=False)`               | `None`         | Center next window (main viewport) |
| `set_next_window_pos_viewport_center()`                       | `None`         | Viewport center       |
| `set_next_window_focus()`                                     | `None`         | Focus next window     |
| `set_next_window_bg_alpha(alpha)`                             | `None`         | Set next window BG alpha |
| `push_window_style()` / `pop_window_style()`                  | `None`         | Window style stack    |
| `push_modal_style()` / `pop_modal_style()`                    | `None`         | Modal style stack     |

### Drawing (Viewport)

On a viewport-overlay `UILayout`, these primitives enqueue into the active
viewport-scoped `ScreenOverlayRenderer`. Coordinates are absolute screen
coordinates. Outside an active overlay frame they emit no command.

| Method                                                         | Returns | Description            |
|----------------------------------------------------------------|---------|------------------------|
| `draw_line(x0, y0, x1, y1, color, thickness=1)`               | `None`  | Line                   |
| `draw_rect(x0, y0, x1, y1, color, thickness=1)`               | `None`  | Rectangle outline      |
| `draw_rect_filled(x0, y0, x1, y1, color, bg=False)`           | `None`  | Filled rectangle       |
| `draw_rect_rounded(x0, y0, x1, y1, color, r, thick=1, bg=F)`  | `None`  | Rounded rect outline   |
| `draw_rect_rounded_filled(x0, y0, x1, y1, color, r, bg=F)`    | `None`  | Filled rounded rect    |
| `draw_circle(x, y, radius, color, segments=32, thickness=1)`  | `None`  | Circle outline         |
| `draw_circle_filled(x, y, radius, color, segments=32)`        | `None`  | Filled circle          |
| `draw_triangle_filled(x0, y0, x1, y1, x2, y2, color, bg=F)`  | `None`  | Filled triangle        |
| `draw_text(x, y, text, color, bg=False)`                      | `None`  | Text at position       |
| `draw_polyline(points, color, closed=False, thickness=1)`      | `None`  | Polyline               |
| `draw_poly_filled(points, color)`                              | `None`  | Filled polygon         |
| `plot_lines(label, values, scale_min, scale_max, size)`        | `None`  | Line plot              |

### Drawing (Window)

The `draw_window_*` names are compatibility aliases and use the same absolute
screen-coordinate convention as `draw_*`; they do not add a window origin.

| Method                                                         | Returns | Description            |
|----------------------------------------------------------------|---------|------------------------|
| `draw_window_rect_filled(x0, y0, x1, y1, color)`              | `None`  | Filled rectangle       |
| `draw_window_rect(x0, y0, x1, y1, color, thickness=1)`        | `None`  | Rectangle outline      |
| `draw_window_rect_rounded(x0, y0, x1, y1, color, r, thickness=1)` | `None` | Rounded outline    |
| `draw_window_rect_rounded_filled(x0, y0, x1, y1, color, r)`   | `None`  | Filled rounded rect    |
| `draw_window_line(x0, y0, x1, y1, color, thickness=1)`        | `None`  | Line                   |
| `draw_window_text(x, y, text, color)`                          | `None`  | Text                   |
| `draw_window_triangle_filled(x0, y0, x1, y1, x2, y2, color)` | `None`  | Filled triangle        |

The legacy `background` arguments on applicable `draw_*` calls are accepted
but ignored: there is no separate background draw list. Overlay calls share
one command stream, packed as shapes followed by text/images. Enqueue order is
retained within each batch, but cross-batch call order is not a z-order
guarantee.

### Progress & Status

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `progress_bar(fraction, overlay='', width=0)`       | `None`  | Progress bar             |
| `set_tooltip(text)`                                 | `None`  | Tooltip for last item    |

### State Queries

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `is_item_hovered()`                                 | `bool`  | Last item hovered        |
| `is_item_clicked(button=0)`                         | `bool`  | Last item clicked        |
| `is_item_active()`                                  | `bool`  | Last item active         |
| `is_window_focused()`                               | `bool`  | Window has focus         |
| `is_window_hovered()`                               | `bool`  | Window is hovered        |
| `is_mouse_double_clicked(button=0)`                 | `bool`  | Double click detected    |
| `is_mouse_dragging(button=0)`                       | `bool`  | Mouse dragging           |
| `get_mouse_wheel()`                                 | `float` | Scroll wheel delta       |
| `get_mouse_delta()`                                 | `tuple` | Mouse delta (dx, dy)     |

### Position / Size

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `get_cursor_pos()`                                  | `tuple` | Cursor position          |
| `get_cursor_screen_pos()`                           | `tuple` | Cursor screen position   |
| `get_window_pos()`                                  | `tuple` | Window position          |
| `get_window_width()`                                | `float` | Window width             |
| `get_text_line_height()`                            | `float` | Text line height         |
| `get_content_region_avail()`                        | `tuple` | Available content area   |
| `get_viewport_pos()`                                | `tuple` | Viewport position        |
| `get_viewport_size()`                               | `tuple` | Viewport size            |
| `get_dpi_scale()`                                   | `float` | DPI scale factor         |
| `calc_text_size(text)`                              | `tuple` | Text dimensions          |

### Styling

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `push_style_var(var, value)`                        | `None`  | Push float style var     |
| `push_style_var_vec2(var, value)`                   | `None`  | Push vec2 style var      |
| `pop_style_var(count=1)`                            | `None`  | Pop style vars           |
| `push_style_color(col, color)`                      | `None`  | Push color override      |
| `pop_style_color(count=1)`                          | `None`  | Pop color overrides      |
| `push_item_width(width)` / `pop_item_width()`      | `None`  | Item width stack         |
| `begin_disabled(disabled=True)` / `end_disabled()`  | `None`  | Disable widget region. For composable disabled regions, prefer `SubLayout.enabled` (see Layout Composition below). |

### Keyboard / Mouse Capture

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `set_keyboard_focus_here()`                         | `None`  | Focus next widget        |
| `capture_keyboard_from_app(capture=True)`           | `None`  | Capture keyboard input   |
| `capture_mouse_from_app(capture=True)`              | `None`  | Capture mouse input      |
| `set_mouse_cursor_hand()`                           | `None`  | Set hand cursor          |

### Cursor Control

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `set_cursor_pos(pos)`                               | `None`  | Set cursor position      |
| `set_cursor_pos_x(x)`                               | `None`  | Set cursor X             |
| `set_scroll_here_y(ratio=0.5)`                      | `None`  | Scroll to current Y      |

### Specialized Widgets

| Method                                                                            | Returns        | Description           |
|-----------------------------------------------------------------------------------|----------------|-----------------------|
| `crf_curve_preview(label, gamma, toe, shoulder, gamma_r=0, gamma_g=0, gamma_b=0)`| `None`         | Unsupported layout method; warns once. Use retained `<crf-curve>`. |
| `chromaticity_diagram(label, rx, ry, gx, gy, bx, by, nx, ny, range=0.5)`         | `(bool, list)` | Unsupported layout method; warns once. Use retained `<chromaticity-diagram>`. |
| `template_list(list_type_id, list_id, data, prop_id, active_data, active_prop, rows=5)` | `(int, int)` | Live on `RmlUILayout`. Compatibility `UILayout` raises `TypeError` outside draw hooks and warns/returns inert values in draw hooks. |

### Layout Composition

Create composable sub-layouts with automatic widget positioning and state cascading.

| Method                                             | Returns     | Description                     |
|----------------------------------------------------|-------------|---------------------------------|
| `row()`                                            | `SubLayout` | Horizontal layout               |
| `column()`                                         | `SubLayout` | Vertical layout                 |
| `split(factor=0.5)`                                | `SubLayout` | Two-child split; factor controls first/second width ratio |
| `box()`                                            | `SubLayout` | Bordered container              |
| `grid_flow(columns=0, even_columns=True, even_rows=True)` | `SubLayout` | Responsive grid with explicit column/row sizing controls |
| `prop_enum(data, prop_id, value, text='')`          | `bool`      | Enum toggle button              |

`SubLayout` is a context manager. Use `with ui.row() as row:` to enter the layout, then call widget methods on `row` instead of `ui`. Sub-layouts nest arbitrarily.

For `split`, values are clamped to `[0, 1]`; a 4dp gap is accounted for while
preserving the requested ratio. Only two children are shown. For `grid_flow`,
positive `columns` with `even_columns=True` assigns equal percentage widths;
`columns=0` uses a wrapping 100dp basis. `even_columns=False` uses content
width. `even_rows=True` grows and stretches cells to the row height;
`even_rows=False` preserves natural height.

#### SubLayout state properties

| Property  | Type    | Description                                |
|-----------|---------|--------------------------------------------|
| `enabled` | `bool`  | Disabled state (cascades to children)      |
| `active`  | `bool`  | Active state (cascades to children)        |
| `alert`   | `bool`  | One-shot alert styling (red text/bg)       |

#### Example

```python
def draw(self, ui):
    with ui.row() as row:
        row.prop_enum(self, "mode", "fast", "Fast")
        row.prop_enum(self, "mode", "quality", "Quality")

    with ui.box() as box:
        box.heading("Settings")
        box.prop(self, "opacity")

    with ui.column() as col:
        col.enabled = self.is_active
        col.prop(self, "value")
        with col.row() as row:
            row.button("Apply")
            row.button("Cancel")

    with ui.grid_flow(columns=3) as grid:
        for item in items:
            with grid.box() as cell:
                cell.label(item.name)
                cell.button("Select")
```

---

## Scene API (lf module)

```python
import lichtfeld as lf
```

### Scene Management

| Function                  | Returns          | Description                       |
|---------------------------|------------------|-----------------------------------|
| `get_scene()`             | `Scene or None`  | Get scene object                  |
| `get_render_scene()`      | `Scene or None`  | Get render scene (PyScene)        |
| `has_scene()`             | `bool`           | Whether scene is loaded           |
| `clear_scene()`           | `None`           | Clear all scene content           |
| `load_file(path, is_dataset=False)` | `None` | Load PLY or dataset               |
| `load_config_file(path)`  | `None`           | Load JSON config                  |
| `get_scene_generation()`  | `int`            | Scene generation counter          |
| `list_scene()`            | `None`           | Print scene tree                  |

### Node Operations (on Scene object)

| Method                                            | Returns          | Description                       |
|---------------------------------------------------|------------------|-----------------------------------|
| `add_group(name, parent=-1)`                      | `int`            | Add group node                    |
| `add_splat(name, means, sh0, shN, scaling, rotation, opacity, ...)` | `int` | Add splat node       |
| `add_point_cloud(name, points, colors, parent=-1)`| `int`            | Add point cloud                   |
| `add_camera(name, parent, R, T, fx, fy, w, h, ...)` | `int`         | Add camera node                   |
| `remove_node(name, keep_children=False)`          | `None`           | Remove node                       |
| `rename_node(old, new)`                           | `bool`           | Rename node                       |
| `reparent(node_id, new_parent_id)`                | `None`           | Change parent                     |
| `duplicate_node(name)`                            | `str`            | Duplicate, returns new node name  |
| `merge_group(group_name)`                         | `str`            | Merge group children, returns merged node name |
| `get_node(name)`                                  | `SceneNode`      | Get node by name                  |
| `get_node_by_id(id)`                              | `SceneNode`      | Get node by ID                    |
| `get_nodes()`                                     | `list[SceneNode]`| All nodes                         |
| `get_visible_nodes()`                             | `list[SceneNode]`| Visible nodes only                |
| `root_nodes()`                                    | `list[int]`      | Root node IDs                     |
| `is_node_effectively_visible(id)`                 | `bool`           | Considers parent visibility       |
| `total_gaussian_count`                            | `int`            | Property. Total gaussians         |
| `invalidate_cache()`                              | `None`           | Clear internal cache (no redraw)  |
| `notify_changed()`                                | `None`           | Invalidate cache + trigger viewport redraw |

### SceneNode Properties

| Property/Method     | Returns              | Description                       |
|---------------------|----------------------|-----------------------------------|
| `id`                | `int`                | Node ID                           |
| `name`              | `str`                | Node name                         |
| `type`              | `NodeType`           | Node type enum (SPLAT, POINTCLOUD, GROUP, etc.) |
| `parent_id`         | `int`                | Parent node ID (-1 for root)      |
| `children`          | `list[int]`          | Child node IDs                    |
| `visible`           | `bool`               | Visibility flag                   |
| `locked`            | `bool`               | Lock flag                         |
| `gaussian_count`    | `int`                | Number of gaussians (splat nodes) |
| `centroid`          | `tuple[float, float, float]` | Node centroid             |
| `world_transform`   | `tuple`              | World-space 4x4 transform, row-major |
| `splat_data()`      | `SplatData or None`  | Splat data for this node (None if not a splat) |
| `point_cloud()`     | `PointCloud or None` | Point cloud data (None if not a point cloud) |
| `cropbox()`         | `CropBox or None`    | Crop box data                     |
| `ellipsoid()`       | `Ellipsoid or None`  | Ellipsoid data                    |

### Selection

| Function                              | Returns          | Description                       |
|---------------------------------------|------------------|-----------------------------------|
| `select_node(name)`                   | `None`           | Select node by name               |
| `deselect_all()`                      | `None`           | Clear selection                   |
| `has_selection()`                     | `bool`           | Any selection active              |
| `get_selected_node_name()`            | `str`            | First selected node name          |
| `get_selected_node_names()`           | `list[str]`      | All selected node names           |
| `can_transform_selection()`           | `bool`           | Selection is transformable        |
| `get_selected_node_transform()`       | `list[float]`    | 16 column-major floats            |
| `set_selected_node_transform(matrix)` | `None`           | Set transform from 16 column-major floats |
| `get_selection_center()`              | `list[float]`    | Local space center                |
| `get_selection_visualizer_world_center()` | `list[float]` | Visualizer world-space center     |
| `get_selection_world_center()`        | `list[float]`    | Deprecated legacy data-world center; use `get_selection_visualizer_world_center()` |
| `capture_selection_transforms()`      | `dict`           | Snapshot for undo                 |

### Scene Shortcuts

Module-level shortcuts for common scene operations (equivalent to `Scene` object methods):

| Function | Returns | Description |
|----------|---------|-------------|
| `set_node_visibility(name, visible)` | `None` | Toggle node visibility |
| `remove_node(name, keep_children=False)` | `None` | Remove node |
| `reparent_node(name, new_parent)` | `None` | Reparent node |
| `rename_node(old_name, new_name)` | `None` | Rename node |
| `add_group(name, parent="")` | `None` | Add group node |
| `get_num_gaussians()` | `int` | Total gaussian count |

### Gaussian-Level Selection (on Scene object)

| Method                          | Returns     | Description                  |
|---------------------------------|-------------|------------------------------|
| `set_selection_mask(mask)`      | `None`      | Apply bool tensor mask       |
| `preview_selection_mask(mask)`  | `None`      | Preview transient selection without an undo step |
| `commit_selection_preview()`    | `None`      | Commit the transient preview as one undo step |
| `cancel_selection_preview()`    | `None`      | Restore the selection before preview |
| `clear_selection()`             | `None`      | Clear gaussian selection     |
| `has_selection()`               | `bool`      | Any gaussians selected       |
| `selection_mask`                | `Tensor`    | Property. Current mask       |
| `set_selection(indices)`        | `None`      | Select by index list         |
| `add_selection_group(name, color)` | `int`    | Create a named group         |
| `selection_groups()`            | `list[SelectionGroup]` | Current groups      |
| `active_selection_group`        | `int`       | Property. Active group id    |
| `set_selection_group_locked(id, locked)` | `None` | Lock/unlock group edits |

### Selection Primitives (`lf.selection`)

`lichtfeld.selection` is the lower-level selection-tool API. It operates on
selection strokes, previews, and viewport-space hit data.

| Function | Returns | Description |
|---|---|---|
| `begin_stroke()` / `commit_stroke(mode)` / `cancel_stroke()` | `None` / `bool` / `None` | Manage one undoable selection stroke |
| `get_stroke_selection()` | `Tensor \| None` | Current stroke mask `[N]` uint8 |
| `set_preview(add_mode=True)` / `clear_preview()` | `None` | Show/clear add/remove preview overlay |
| `draw_brush_circle(x, y, radius, add_mode=True)` | `None` | Brush cursor overlay |
| `draw_rect_preview(x0, y0, x1, y1, add_mode=True)` | `None` | Rectangle preview overlay |
| `draw_polygon_preview(points, closed=False, add_mode=True)` | `None` | Polygon preview overlay |
| `draw_lasso_preview(points, add_mode=True)` | `None` | Lasso preview overlay |
| `has_screen_positions()` / `get_screen_positions()` | `bool` / `Tensor \| None` | Viewport-projected positions `[N, 2]` |
| `set_depth_filter_range(enabled, depth_near=0, depth_far=100, frustum_half_width=50)` | `None` | Camera-space selection depth filter |
| `get_depth_filter_range()` | `tuple[bool, float, float, float]` | `(enabled, near, far, half_width)` |
| `set_crop_filter(enabled)` / `apply_crop_filter()` | `None` | Crop-box constrained selection |
| `screen_to_render(x, y)` | `tuple[float, float]` | Convert screen to render coordinates |
| `pick_at_screen(x, y)` | `PickResult \| None` | Depth/world hit at screen point |
| `ring_select(index, add=True)` | `None` | Select/deselect one gaussian |
| `grow(radius, iterations=1)` / `shrink(radius, iterations=1)` | `None` | Spatial grow/shrink selection |
| `by_opacity(min_opacity=0, max_opacity=1)` | `None` | Select by activated opacity range |
| `by_scale(max_scale)` | `None` | Select by activated scale threshold |
| `by_color(gaussian_index, threshold=0.2)` | `None` | Select by SH DC color similarity |

`PickResult.index` is the gaussian under the current cursor, not necessarily
the queried screen coordinate. Use `depth` and `world_position` for the queried
coordinate data.

### Transforms

| Function                                    | Returns        | Description                      |
|---------------------------------------------|----------------|----------------------------------|
| `get_node_transform(name)`                  | `list[float]`  | 16 column-major floats           |
| `set_node_transform(name, matrix)`          | `None`         | Set 4x4 transform from 16 column-major floats |
| `decompose_transform(matrix)`               | `dict`         | Decompose 16 column-major floats; see keys below |
| `compose_transform(translation, euler_deg, scale)` | `list[float]` | Build 16 column-major floats from components (Euler in degrees) |

`decompose_transform` returns a dict with these keys:

| Key | Type | Description |
|-----|------|-------------|
| `translation` | `[x, y, z]` | Position |
| `rotation_quat` | `[x, y, z, w]` | Quaternion |
| `rotation_euler` | `[rx, ry, rz]` | Euler angles (radians) |
| `rotation_euler_deg` | `[rx, ry, rz]` | Euler angles (degrees) |
| `scale` | `[sx, sy, sz]` | Scale |

### Splat Data (combined_model() / node.splat_data())

Accessible via `scene.combined_model()` (all nodes merged) or `node.splat_data()` (per-node).

| Property/Method       | Returns      | Description                     |
|-----------------------|--------------|---------------------------------|
| `means_raw`           | `Tensor`     | [N, 3] positions (view)        |
| `sh0_raw`             | `Tensor`     | [N, 1, 3] base SH (view)       |
| `shN_raw`             | `Tensor`     | [N, K, 3] higher SH (view)     |
| `scaling_raw`         | `Tensor`     | [N, 3] log-space (view)        |
| `rotation_raw`        | `Tensor`     | [N, 4] quaternions (view)      |
| `opacity_raw`         | `Tensor`     | [N, 1] logit-space (view)      |
| `get_means()`         | `Tensor`     | Positions                       |
| `get_opacity()`       | `Tensor`     | [N] sigmoid applied             |
| `get_scaling()`       | `Tensor`     | Exp applied                     |
| `get_rotation()`      | `Tensor`     | Normalized quaternions          |
| `get_shs()`           | `Tensor`     | SH0 + SHN concatenated         |
| `num_points`          | `int`        | Gaussian count                  |
| `active_sh_degree`    | `int`        | Current SH degree               |
| `max_sh_degree`       | `int`        | Maximum SH degree               |
| `scene_scale`         | `float`      | Scene scale factor              |
| `soft_delete(mask)`   | `Tensor`     | Mark for deletion, returns newly deleted mask |
| `undelete(mask)`      | `None`       | Restore deleted gaussians       |
| `apply_deleted()`     | `int`        | Permanently remove, returns count|
| `clear_deleted()`     | `None`       | Clear deletion mask             |
| `deleted`             | `Tensor`     | Property. [N] bool deletion mask|
| `has_deleted_mask()`  | `bool`       | Whether deletion mask exists    |
| `visible_count()`     | `int`        | Number of non-deleted gaussians |

> After calling `soft_delete()`, `undelete()`, or `clear_deleted()`, call `scene.notify_changed()` to update the viewport.

### Point Clouds, Cameras, and Dataset Nodes

| API | Returns | Description |
|---|---|---|
| `Scene.add_point_cloud(name, points, colors, parent=-1)` | `int` | Add `[N,3]` position/color tensors |
| `SceneNode.point_cloud()` | `PointCloud \| None` | Point-cloud payload for point-cloud nodes |
| `PointCloud.means` / `PointCloud.colors` | `Tensor` | Position and color tensors |
| `PointCloud.normals`, `sh0`, `shN`, `opacity`, `scaling`, `rotation` | `Tensor \| None` | Optional gaussian-like attributes |
| `PointCloud.normalize_colors()` | `None` | Normalize colors to `[0,1]` |
| `PointCloud.filter(mask)` / `filter_indices(indices)` | `int` | Keep matching points, return removed count |
| `PointCloud.set_means(points)` / `set_colors(colors)` | `None` | Update positions/colors without replacing both |
| `Scene.add_camera_group(name, parent, camera_count)` | `int` | Add camera group |
| `Scene.add_camera(name, parent, R, T, focal_x, focal_y, width, height, image_path='', uid=-1, mask=None)` | `int` | Add camera node |
| `SceneNode.load_mask(...)` / `load_depth(...)` | `Tensor \| None` | Load camera mask/depth from node metadata |
| `Scene.get_active_cameras()` | `list[SceneNode]` | Cameras enabled for training |
| `CameraDataset.cameras()` | `list[Camera]` | Dataset cameras as Python objects |
| `Camera.load_image(resize_factor=1, max_width=0, output_uint8=False)` | `Tensor` | `[C,H,W]` CUDA image |
| `Camera.rotation` / `Camera.translation` | `Tensor` | Visualizer camera pose, directly usable with `render_view()` |

Deprecated raw dataset-camera properties are still available on `Camera`:
`R`, `T`, `world_view_transform`, and `cam_position`. Prefer
`rotation`, `translation`, `K`, and `view_matrix` for new code.

### Training Control

| Function                    | Returns          | Description                   |
|-----------------------------|------------------|-------------------------------|
| `start_training()`         | `None`           | Start training                |
| `pause_training()`         | `None`           | Pause                         |
| `resume_training()`        | `None`           | Resume                        |
| `stop_training()`          | `None`           | Stop                          |
| `reset_training()`         | `None`           | Reset to iteration 0          |
| `save_checkpoint()`        | `None`           | Save checkpoint               |
| `switch_to_edit_mode()`    | `None`           | Enter edit mode               |
| `has_trainer()`            | `bool`           | Trainer loaded                |
| `trainer_state()`          | `str`            | State string                  |
| `finish_reason()`          | `str or None`    | Why training ended            |
| `trainer_error()`          | `str or None`    | Error message                 |
| `context()`                | `Context`        | Training context snapshot     |
| `optimization_params()`    | `OptimizationParams` | Training parameters      |
| `dataset_params()`         | `DatasetParams`  | Dataset parameters            |
| `loss_buffer()`            | `list[float]`    | Loss history                  |
| `load_checkpoint_for_training(checkpoint_path, dataset_path, output_path)` | `None` | Load checkpoint for training |

### Training Status

| Function | Returns | Description |
|----------|---------|-------------|
| `trainer_elapsed_seconds()` | `float` | Elapsed training time |
| `trainer_eta_seconds()` | `float` | Estimated remaining time (-1 if unavailable) |
| `trainer_strategy_type()` | `str` | Strategy type (mcmc, default, etc.) |
| `trainer_is_gut_enabled()` | `bool` | GUT enabled |
| `trainer_max_gaussians()` | `int` | Max gaussians |
| `trainer_num_splats()` | `int` | Current splat count |
| `trainer_current_iteration()` | `int` | Current iteration |
| `trainer_total_iterations()` | `int` | Total iterations |
| `trainer_current_loss()` | `float` | Current loss |

### Training Hooks

| Decorator                    | Description                         |
|------------------------------|-------------------------------------|
| `@lf.on_training_start`     | Called when training starts          |
| `@lf.on_iteration_start`    | Called at start of each iteration    |
| `@lf.on_pre_optimizer_step` | Called before optimizer step         |
| `@lf.on_post_step`          | Called after each step               |
| `@lf.on_training_end`       | Called when training ends            |

Callbacks registered with `lf.on_*` receive one positional hook payload. If you do not need it, declare the parameter as `_hook`.

### Rendering

| Function                                              | Returns          | Description                |
|-------------------------------------------------------|------------------|----------------------------|
| `get_current_view()`                                  | `ViewInfo`       | Current camera view        |
| `get_viewport_render()`                               | `ViewportRender` | Current viewport image     |
| `capture_viewport()`                                  | `ViewportRender` | Capture for async use      |
| `export_viewport_image(path, format='', width=0, height=0, transparent=False, jpeg_quality=95)` | `dict` | Export active viewport to PNG/JPEG |
| `look_at(eye, target, up=(0,1,0))`                    | `(Tensor, Tensor)` | Compute `(rotation, translation)` for rendering |
| `render_view(rotation, translation, width, height, fov=60, bg_color=None, with_depth=False, depth_mode='median')` | `Tensor \| tuple \| None` | Render active scene from camera |
| `render_view_u8(rotation, translation, width, height, fov=60, bg_color=None, orthographic=None, ortho_scale=None)` | `Tensor \| None` | Render active scene as CPU uint8 RGB |
| `render_at(eye, target, width, height, fov=60, up=(0,1,0), bg_color=None)` | `Tensor \| None` | Convenience look-at render |
| `render_asset_preview(path, width=512, height=224, focal_length_mm=35)` | `Tensor \| None` | Offscreen asset thumbnail without mutating live scene |
| `render_asset_preview_from_camera(path, eye, target, ...)` | `Tensor \| None` | Offscreen thumbnail from custom camera |
| `compute_screen_positions(rotation, translation, width, height, fov=60)` | `Tensor` | [N, 2] screen positions |
| `get_render_settings()`                               | `RenderSettings` | Current render settings    |
| `get_render_mode()` / `set_render_mode(mode)`         | `RenderMode`     | Render mode                |

### Viewport Control

| Function                        | Returns | Description              |
|---------------------------------|---------|--------------------------|
| `reset_camera()`               | `None`  | Reset camera             |
| `toggle_fullscreen()`          | `None`  | Toggle fullscreen        |
| `is_fullscreen()`              | `bool`  | Fullscreen state         |
| `toggle_ui()`                  | `None`  | Toggle UI visibility     |
| `set_orthographic(ortho)`      | `None`  | Set projection mode      |
| `is_orthographic()`            | `bool`  | Orthographic state       |

### Export

```python
lf.export_scene(
    format: int,             # 0=PLY, 1=SOG, 2=SPZ, 3=HTML, 4=USD,
                             # 5=USDZ NuRec, 6=RAD, 7=COLMAP
    path: str,
    node_names: list[str],
    sh_degree: int,
    rad_flip_y: bool = False,
    rad_streamable: bool = True,
)
lf.save_config_file(path: str)
```

### File I/O (`lf.io`)

Use `lf.io` when you need data objects directly instead of loading into the
live application scene.

| Function | Returns | Description |
|---|---|---|
| `lf.io.load(path, format=None, resize_factor=None, max_width=None, images_folder=None, progress=None, min_track_length=None)` | `LoadResult` | Load splat file, point cloud, mesh-derived data, or dataset |
| `lf.io.load_point_cloud(path)` | `(Tensor, Tensor)` | Load PLY point cloud as positions/colors |
| `lf.io.save_ply(data, path, binary=True, progress=None, extra_attributes=None)` | `None` | Save `SplatData` as PLY |
| `lf.io.save_point_cloud_ply(point_cloud, path, extra_attributes=None)` | `None` | Save `PointCloud` as PLY |
| `lf.io.save_sog(data, path, kmeans_iterations=10, use_gpu=True, progress=None)` | `None` | Save SOG-compressed splats |
| `lf.io.save_spz(data, path)` | `None` | Save SPZ splats |
| `lf.io.save_usd(data, path)` | `None` | Save OpenUSD gaussian file |
| `lf.io.save_nurec_usdz(data, path)` | `None` | Save NuRec-compatible USDZ |
| `lf.io.export_html(data, path, kmeans_iterations=10, progress=None)` | `None` | Self-contained HTML viewer |
| `lf.io.save_image(path, image)` | `None` | Save PNG/JPG/TIFF/EXR from `[H,W,C]` or `[C,H,W]` tensor |
| `lf.io.is_dataset_path(path)` | `bool` | Dataset directory detection |
| `lf.io.is_gaussian_splat_ply(path)` | `bool` | PLY schema check for gaussian splat attributes |
| `lf.io.get_supported_formats()` / `get_supported_extensions()` | `list[str]` | Loader/exporter support |

`LoadResult` exposes `splat_data`, `point_cloud`, `cameras`,
`scene_center`, `loader_used`, `load_time_ms`, `warnings`, and `is_dataset`.
PLY extra attributes must avoid reserved gaussian names and must match the
visible/raw point count expected by the exporter.

### Pipeline Operations (`lf.pipeline`)

Pipelines compose built-in operations and execute them as one chain.

```python
pipe = (
    lf.pipeline.Pipeline("grow-and-move")
    | lf.pipeline.select.grow(radius=0.05)
    | lf.pipeline.transform.translate(offset=(0.0, 0.1, 0.0))
)
if pipe.poll():
    result = pipe.execute()
```

| API | Description |
|---|---|
| `lf.pipeline.Pipeline(name='')` | Create an empty chain |
| `Pipeline.add(stage)` / `Pipeline \| stage` | Append a stage |
| `Pipeline.poll()` / `execute()` | Check and execute all stages |
| `Stage.execute()` | Execute a single stage immediately |
| `lf.pipeline.select.all/none/invert/grow/shrink(**kwargs)` | Selection stages |
| `lf.pipeline.transform.translate/rotate/scale/set(**kwargs)` | Transform stages |
| `lf.pipeline.edit.duplicate(**kwargs)` | Duplicate stage |

### Logging

| Function           | Description       |
|--------------------|-------------------|
| `lf.log.info(msg)` | Info level        |
| `lf.log.warn(msg)` | Warning level     |
| `lf.log.error(msg)`| Error level       |
| `lf.log.debug(msg)`| Debug level       |

### Undo

```python
lf.undo.push(name: str, undo: Callable, redo: Callable, validate: Callable | None = None)
lf.undo.transaction(name: str = "Grouped Changes") -> Transaction
lf.undo.stack() -> dict
```

- `lf.undo.transaction(...)` groups multiple undoable mutations into one history step.
- `lf.undo.stack()` returns structured undo/redo items with `id`, `label`, `source`, `scope`, and `estimated_bytes`.

### UI Functions

| Function                                    | Returns          | Description                |
|---------------------------------------------|------------------|----------------------------|
| `lf.ui.tr(key)`                             | `str`            | Translate string           |
| `lf.ui.theme()`                             | `Theme`          | Current theme              |
| `lf.ui.context()`                           | `AppContext`     | App context                |
| `lf.ui.request_redraw()`                    | `None`           | Request UI redraw          |
| `lf.ui.set_language(lang_code)`             | `None`           | Set UI language            |
| `lf.ui.get_current_language()`              | `str`            | Active language code       |
| `lf.ui.get_languages()`                     | `list[tuple[str, str]]` | Available languages  |
| `lf.ui.set_theme(name)`                     | `None`           | Theme switch by stable theme id |
| `lf.ui.get_theme()`                         | `str`            | Active stable theme id     |
| `lf.ui.themes()`                            | `list[dict]`     | Available theme presets    |
| `lf.ui.set_panel_enabled(panel_id, enabled)`  | `None`           | Toggle panel by id         |
| `lf.ui.is_panel_enabled(panel_id)`            | `bool`           | Panel enabled state        |
| `lf.ui.get_panel_names(space=lf.ui.PanelSpace.FLOATING)` | `list[str]` | Panel ids for a space |
| `lf.ui.get_panel(panel_id)`                   | `lf.ui.PanelInfo \| None`   | Typed panel info |
| `lf.ui.get_main_panel_tabs()`                 | `list[lf.ui.PanelSummary]` | Typed summaries for main-panel tabs |
| `lf.ui.set_panel_label(panel_id, label)`      | `bool`           | Change panel display name  |
| `lf.ui.set_panel_order(panel_id, order)`      | `bool`           | Change panel sort order    |
| `lf.ui.set_panel_space(panel_id, space)`      | `bool`           | Move panel to a different space (`lf.ui.PanelSpace`) |
| `lf.ui.set_panel_parent(panel_id, parent)`    | `bool`           | Embed panel inside a tab as collapsible section |
| `lf.ui.ops.invoke(op_id, **kwargs)`         | `OperatorReturnValue` | Invoke operator       |
| `lf.ui.ops.poll(op_id)`                     | `bool`           | Operator poll              |
| `lf.ui.ops.cancel_modal()`                  | `None`           | Cancel modal operator      |
| `lf.ui.get_active_tool()`                   | `str`            | Active tool ID             |
| `lf.ui.get_active_submode()`                | `str`            | Active submode             |
| `lf.ui.set_selection_mode(mode)`            | `None`           | Set selection submode      |
| `lf.ui.get_transform_space()`               | `int`            | Transform space enum index |
| `lf.ui.set_transform_space(space)`          | `None`           | Set transform space index  |
| `lf.ui.get_pivot_mode()` / `set_pivot_mode(mode)` | `int`      | Pivot mode enum index      |
| `lf.ui.get_fps()`                           | `float`          | Current FPS                |
| `lf.ui.get_git_commit()`                    | `str`            | Git commit hash            |
| `lf.ui.is_key_pressed(key, repeat=False)`    | `bool`           | SDL-backed rising edge for the current UI frame; UI thread only, no repeat events |
| `lf.ui.is_key_down(key)`                     | `bool`           | Current SDL keyboard level |

For a coarse CUDA memory number in Python plugin code, use
`lfs_plugins.get_gpu_memory()` from the helper package. There is no
`lf.ui.get_gpu_memory()` binding in the current stubs.

### File Dialogs

| Function                                    | Returns          |
|---------------------------------------------|------------------|
| `lf.ui.open_image_dialog(start_dir='')`     | `str`            |
| `lf.ui.open_folder_dialog(title='Select Folder', start_dir='')` | `str` |
| `lf.ui.open_dataset_folder_dialog()`        | `str`            |
| `lf.ui.open_ply_file_dialog(start_dir='')`  | `str`            |
| `lf.ui.open_mesh_file_dialog(start_dir='')` | `str`            |
| `lf.ui.open_checkpoint_file_dialog()`       | `str`            |
| `lf.ui.open_ppisp_file_dialog(start_dir='')`| `str`            |
| `lf.ui.open_json_file_dialog()`             | `str`            |
| `lf.ui.open_csv_file_dialog()`              | `str`            |
| `lf.ui.open_xml_file_dialog()`              | `str`            |
| `lf.ui.open_las_file_dialog()`              | `str`            |
| `lf.ui.open_video_file_dialog()`            | `str`            |
| `lf.ui.select_colmap_sparse_folder_dialog(default_path='')` | `str` |
| `lf.ui.open_environment_map_dialog(start_dir='')` | `str`     |
| `lf.ui.save_las_file_dialog(default_name='export')` | `str` |
| `lf.ui.save_laz_file_dialog(default_name='export')` | `str` |
| `lf.ui.save_json_file_dialog(default_name='config.json')` | `str` |
| `lf.ui.save_png_file_dialog(default_name='export.png')`   | `str` |
| `lf.ui.save_jpg_file_dialog(default_name='export.jpg')`   | `str` |
| `lf.ui.save_ply_file_dialog(default_name='export')`       | `str` |
| `lf.ui.save_sog_file_dialog(default_name='export')`       | `str` |
| `lf.ui.save_spz_file_dialog(default_name='export')`       | `str` |
| `lf.ui.save_usd_file_dialog(default_name='export')`       | `str` |
| `lf.ui.save_usdz_file_dialog(default_name='export')`      | `str` |
| `lf.ui.save_html_file_dialog(default_name='viewer')`      | `str` |
| `lf.ui.save_rad_file_dialog(default_name='export')`       | `str` |

`lf.ui.open_folder_dialog()` accepts `title` for compatibility with older scripts. The current native dialog backend ignores it.

### UI Hooks

Inject UI into existing panels at predefined hook points. Callbacks receive a `layout` object.

| Function | Description |
|---|---|
| `lf.ui.add_hook(panel, section, callback, position="append")` | Register a hook. `position`: `"prepend"` or `"append"` |
| `lf.ui.remove_hook(panel, section, callback)` | Remove a specific hook callback |
| `lf.ui.clear_hooks(panel, section="")` | Clear hooks for panel/section (or all sections if empty) |
| `lf.ui.clear_all_hooks()` | Clear all registered hooks |
| `lf.ui.get_hook_points()` | List all registered hook point keys |
| `lf.ui.invoke_hooks(panel, section, prepend=False)` | Invoke hooks (`prepend=True` for prepend, `False` for append) |
| `@lf.ui.hook(panel, section, position="append")` | Decorator form of `add_hook` |

Hook points are runtime-defined. Query them with `lf.ui.get_hook_points()` instead of hard-coding.
Callbacks whose layout cannot host interactive widgets warn once per method
and return inert controls. Viewport drawing remains available during an active
overlay frame.

### Tensor API

```python
import lichtfeld as lf
t = lf.Tensor
```

The tables below list the most-used tensor APIs. For the full bound surface, see `src/python/stubs/lichtfeld/__init__.pyi`.

**Creation:**

| Function                                    | Returns  | Description              |
|---------------------------------------------|----------|--------------------------|
| `t.zeros(shape, device='cuda', dtype='float32')` | `Tensor` | Zero-filled tensor   |
| `t.ones(shape, device, dtype)`              | `Tensor` | Ones tensor              |
| `t.full(shape, value, device, dtype)`       | `Tensor` | Constant-filled tensor   |
| `t.eye(n, device, dtype)`                   | `Tensor` | Identity matrix          |
| `t.arange(start, end, step, device, dtype)` | `Tensor` | Range tensor             |
| `t.linspace(start, end, steps, device, dtype)` | `Tensor` | Linear space          |
| `t.rand(shape, device, dtype)`              | `Tensor` | Uniform random [0, 1)   |
| `t.randn(shape, device, dtype)`             | `Tensor` | Normal random            |
| `t.empty(shape, device, dtype)`             | `Tensor` | Uninitialized tensor     |
| `t.randint(low, high, shape, device)`       | `Tensor` | Random integers          |
| `t.from_numpy(arr, copy=True)`              | `Tensor` | From NumPy array         |
| `t.cat(tensors, dim=0)`                     | `Tensor` | Concatenate              |
| `t.stack(tensors, dim=0)`                   | `Tensor` | Stack                    |
| `t.where(condition, x, y)`                  | `Tensor` | Conditional select       |

**Properties:**

| Property         | Type    | Description              |
|------------------|---------|--------------------------|
| `.shape`         | `tuple` | Tensor dimensions        |
| `.ndim`          | `int`   | Number of dimensions     |
| `.numel`         | `int`   | Total elements           |
| `.device`        | `str`   | `'cpu'` or `'cuda'`     |
| `.dtype`         | `str`   | Data type string         |
| `.is_contiguous` | `bool`  | Memory contiguous        |
| `.is_cuda`       | `bool`  | On GPU                   |

**Methods:**

| Method                              | Returns  | Description              |
|-------------------------------------|----------|--------------------------|
| `.clone()`                          | `Tensor` | Deep copy                |
| `.cpu()` / `.cuda()`               | `Tensor` | Move device              |
| `.contiguous()`                     | `Tensor` | Make contiguous          |
| `.sync()`                           | `None`   | CUDA synchronize         |
| `.numpy(copy=True)`                 | `ndarray`| Convert to NumPy         |
| `.to(dtype)`                        | `Tensor` | Convert dtype            |
| `.size(dim)`                        | `int`    | Size at dimension        |
| `.item()`                           | `scalar` | Extract scalar           |
| `.sum(dim=None, keepdim=False)`     | `Tensor` | Reduce sum               |
| `.mean(dim=None, keepdim=False)`    | `Tensor` | Reduce mean              |
| `.max(dim=None, keepdim=False)`     | `Tensor` | Reduce max               |
| `.min(dim=None, keepdim=False)`     | `Tensor` | Reduce min               |
| `.reshape(shape)`                   | `Tensor` | Reshape                  |
| `.view(shape)`                      | `Tensor` | View reshape             |
| `.squeeze(dim=None)`                | `Tensor` | Remove size-1 dims       |
| `.unsqueeze(dim)`                   | `Tensor` | Add size-1 dim           |
| `.transpose(dim0, dim1)`            | `Tensor` | Swap dimensions          |
| `.permute(dims)`                    | `Tensor` | Reorder dimensions       |
| `.flatten(start=0, end=-1)`         | `Tensor` | Flatten range            |
| `.expand(sizes)`                    | `Tensor` | Broadcast view           |
| `.repeat(repeats)`                  | `Tensor` | Tile tensor              |
| `.prod()`, `.std()`, `.var()`      | `Tensor` | Additional reductions    |
| `.argmax()`, `.argmin()`            | `Tensor` | Index reductions         |
| `.all()`, `.any()`                  | `Tensor` | Logical reductions       |
| `.matmul()`, `.mm()`, `.bmm()`      | `Tensor` | Matrix products          |
| `.masked_select()`, `.masked_fill()`| `Tensor` | Masked operations        |
| `.zeros_like()`, `.ones_like()` etc.| `Tensor` | Like-constructors        |
| `.from_dlpack()` / `.__dlpack__()`  | `Tensor` | DLPack interop           |

**Operators:** `+`, `-`, `*`, `/`, `**`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `[]` (indexing/slicing)

### Application

| Function             | Description              |
|----------------------|--------------------------|
| `lf.request_exit()`  | Exit with confirmation   |
| `lf.force_exit()`    | Immediate exit           |
| `lf.run(path)`       | Execute Python script    |
| `lf.on_frame(cb)`    | Per-frame callback       |
| `lf.stop_animation()`| Clear frame callback     |
| `lf.mat4(rows)`      | Create 4x4 matrix        |
| `lf.help()`          | Show help                |

### Native Operators (`lf.ops` / `lf.ui.ops`)

Use `lf.ops` for the native operator registry and descriptor metadata. The
`lf.ui.ops` namespace exposes the same common invoke/poll/modal controls for UI
code.

| API | Returns | Description |
|---|---|---|
| `lf.ops.invoke(id, **kwargs)` | `OperatorReturnValue` | Invoke a native or Python operator |
| `lf.ops.poll(id)` | `bool` | Check whether the operator can run |
| `lf.ops.get_all()` | `list[str]` | Registered operator IDs |
| `lf.ops.get_descriptor(id)` | `OperatorDescriptor \| None` | Label, description, icon, shortcut, flags |
| `lf.ops.has_modal()` / `cancel_modal()` | `bool` / `None` | Modal operator state/control |

`OperatorReturnValue` has boolean helpers: `finished`, `cancelled`,
`running_modal`, `pass_through`, and `bool(result)` for successful completion.
Extra return data can be accessed by attribute.

### MCP Tools (`lf.mcp`)

Python plugins can register MCP tools into the same local MCP surface used by
automation clients.

```python
@lf.mcp.tool(name="my_plugin.echo", description="Echo a message")
def echo(args):
    return {"message": args.get("message", "")}
```

| Function | Returns | Description |
|---|---|---|
| `register_tool(fn, name='', description='')` | `None` | Register a Python function as an MCP tool |
| `@tool(name='', description='')` | decorator | Decorator form |
| `unregister_tool(name)` | `None` | Remove a Python MCP tool |
| `list_tools()` / `describe_tools()` | `list` | All shared MCP tools/capabilities |
| `list_python_tools()` | `list[str]` | Python tools registered through `lf.mcp` |
| `list_resources()` / `read_resource(uri)` | `list` | Shared MCP resource discovery/read |
| `call_tool(name, args=None)` | `object` | Invoke a registered tool/capability |

### Packages (`lf.packages`)

Package installation is backed by `uv` and the LichtFeld-managed Python
environment.

| Function | Returns | Description |
|---|---|---|
| `init()` | `str` | Initialize `~/.lichtfeld/venv` |
| `install(package)` / `uninstall(package)` | `str` | Synchronous package change |
| `list()` | `list[PackageInfo]` | Installed packages |
| `is_installed(package)` | `bool` | Package presence check |
| `install_async(package)` | `bool` | Start non-blocking install |
| `install_torch(cuda='auto', version='')` | `str` | Install PyTorch with CUDA detection |
| `install_torch_async(cuda='auto', version='')` | `bool` | Non-blocking PyTorch install |
| `is_busy()` | `bool` | Async operation running |
| `is_uv_available()` / `uv_path()` | `bool` / `str` | `uv` discovery |
| `embedded_python_path()` / `site_packages_dir()` / `typings_dir()` | `str` | Runtime paths |

There is no `uninstall_async()` API in the current stubs; use synchronous
`uninstall()`.

### Scripts (`lf.scripts`)

These functions back the Scripts panel and are useful for plugin-managed script
batches.

| Function | Returns | Description |
|---|---|---|
| `get_scripts()` | `list` | Loaded script records |
| `set_script_enabled(index, enabled)` | `None` | Toggle one script |
| `set_script_error(index, error)` | `None` | Set or clear one script error |
| `clear_errors()` / `clear()` | `None` | Clear errors or all scripts |
| `run(paths)` | `dict` | Run scripts, returns success/error data |
| `get_enabled_paths()` | `list[str]` | Enabled script paths |
| `count()` | `int` | Script count |

### Keymaps (`lf.keymap`)

`lf.keymap` exposes input bindings for tools and global actions.

| Function | Description |
|---|---|
| `get_action_for_key(mode, key, modifiers=0)` / `get_action_for_scroll(mode, modifiers=0, held_keys=[])` | Resolve input to action |
| `get_key_for_action(action, mode=GLOBAL)` / `get_trigger(action, mode=GLOBAL)` | Read current binding |
| `set_binding(mode, action, key, modifiers=0)` / `set_trigger_binding(mode, action, trigger)` | Change binding |
| `clear_binding(mode, action)` / `reset_to_default()` | Remove or reset bindings |
| `find_conflict_for_action(mode, action)` | Detect binding conflicts |
| `get_available_profiles()` / `get_current_profile()` | Profile discovery |
| `load_profile(name)` / `save_profile(name)` / `export_profile(path)` / `import_profile(path)` | Profile persistence |
| `start_capture(mode, action)` / `capture_scroll(...)` / `cancel_capture()` | Interactive capture |
| `is_capturing()` / `get_captured_trigger()` / `bindings_revision()` | Capture and change state |

The enum surfaces include `Action`, `ToolMode`, `Modifier`, `MouseButton`,
`KeyTrigger`, and `MouseButtonTrigger`.

### Animation (`lf.animation`)

| Class | Purpose |
|---|---|
| `AnimationTrack` | Keyframes for one target property path; supports `add_keyframe()`, `remove_keyframe()`, `evaluate()`, and `keyframes()` |
| `AnimationClip` | Multi-track clip with `add_track(value_type, target_path)`, `remove_track()`, `get_track()`, and `evaluate(time)` |
| `Timeline` | Camera keyframe timeline plus optional animation clip; exposes `animation_clip()` and `evaluate_clip(time)` |

Track value types are `"bool"`, `"int"`, `"float"`, `"vec2"`, `"vec3"`,
`"vec4"`, `"quat"`, and `"mat4"`.

### Mesh (`lf.mesh`)

`lf.mesh` is a broad OpenMesh binding surface. The most common plugin entry
points are:

| API | Returns | Description |
|---|---|---|
| `MeshData` | class | Tensor-backed mesh payload used by scene/rendering |
| `read_trimesh(filename, **options)` / `read_polymesh(filename, **options)` | `TriMesh` / `PolyMesh` | Load OpenMesh meshes |
| `write_mesh(filename, mesh, **options)` | `None` | Save `TriMesh` or `PolyMesh` |
| `write_mesh(mesh_data, path)` | `None` | Save tensor-backed mesh data |
| `TriMesh` / `PolyMesh` | classes | OpenMesh-style topology, geometry, and property APIs |
| `TriMeshDecimater` / `PolyMeshDecimater` | classes | Decimation module management |

For exact method coverage, use `src/python/stubs/lichtfeld/mesh.pyi`; that
file is intentionally the canonical exhaustive reference for the mesh binding.

---

## pyproject.toml Schema

```toml
[project]
name = ""                    # string, required - Unique plugin identifier
version = ""                 # string, required - Semantic version
description = ""             # string, required
authors = []                 # list[{name, email}], optional - PEP 621 authors
dependencies = []            # list[string], optional - Python packages (PEP 508)

[tool.lichtfeld]
hot_reload = true            # bool, required
entry_point = "__init__"     # string, optional - Module to load (default: __init__)
plugin_api = ">=1,<2"        # string, required - Supported plugin API range (PEP 440)
lichtfeld_version = ">=0.4.2"  # string, required - Supported host app/runtime range (PEP 440)
required_features = []       # list[string], required - Optional host features this plugin needs
author = ""                  # string, optional - Author fallback (if no [project].authors)
```

v1 is strict. Legacy `min_lichtfeld_version` / `max_lichtfeld_version` fields are removed and rejected.

---

## Icon System

```python
from lfs_plugins.icon_manager import get_icon, get_ui_icon, get_scene_icon, get_plugin_icon
```

| Function                                    | Returns | Description                              |
|---------------------------------------------|---------|------------------------------------------|
| `get_icon(name)`                            | `int`   | Load `assets/icon/{name}.png`            |
| `get_ui_icon(name)`                         | `int`   | Load `assets/icon/{name}` (include ext)  |
| `get_scene_icon(name)`                      | `int`   | Load `assets/icon/scene/{name}.png`      |
| `get_plugin_icon(name, plugin_path, plugin_name)` | `int` | Load `{plugin_path}/icons/{name}.png` with fallback |

All return opaque Vulkan UI texture ids (0 on failure). Icons are cached by C++ (`IconCache`).

Direct loading:
```python
import lichtfeld as lf
texture_id = lf.load_icon(name)
lf.free_icon(texture_id)
```

---

## Errors

Plugin errors are captured and accessible via the plugin manager:

```python
import lichtfeld as lf

state = lf.plugins.get_state("my_plugin")   # PluginState enum
error = lf.plugins.get_error("my_plugin")   # Error message string
tb = lf.plugins.get_traceback("my_plugin")  # Full traceback string
```

### PluginState values

| State        | Description                    |
|--------------|--------------------------------|
| `UNLOADED`   | Plugin is not loaded           |
| `INSTALLING` | Plugin is being installed      |
| `LOADING`    | Plugin is loading              |
| `ACTIVE`     | Plugin is running              |
| `ERROR`      | Plugin failed to load/run      |
| `DISABLED`   | Plugin is manually disabled    |

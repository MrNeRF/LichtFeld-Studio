# Python Plugin and UI API

[Python docs](README.md) · [Development guide](getting-started.md) · [Application API](api-reference.md) · [Plugin and UI API](plugin-reference.md)

This reference describes the bindings in this checkout. The committed
[SDK stubs](../../src/python/stubs/lichtfeld/) list overloads and less common
members; [native bindings](../../src/python/lfs/) and
[Python helpers](../../src/python/lfs_plugins/) define behavior. A name present
in a stub does not imply that it works on every layout or outside the app.

## On this page

- [Registration](#registration)
- [Menu](#menu)
- [Scrub Controls](#scrub-controls)
- [Panel](#panel)
- [Operator](#operator)
- [Event](#event)
- [Properties](#properties)
- [ToolDef / ToolRegistry](#tooldef--toolregistry)
- [Signals](#signals)
- [Capabilities](#capabilities)
- [PluginManager](#pluginmanager)
- [Layout API](#layout-api)
- [UI Functions](#ui-functions)
- [File Dialogs](#file-dialogs)
- [UI Hooks](#ui-hooks)
- [pyproject.toml Schema](#pyprojecttoml-schema)
- [Icon System](#icon-system)
- [Plugin Errors](#plugin-errors)

## Registration

```python
import lichtfeld as lf

lf.register_class(cls)           # Register a Panel, Operator, or Menu class
lf.unregister_class(cls)         # Unregister a Panel, Operator, or Menu class
```

---

## Menu

```python
from lfs_plugins.types import Menu
from lfs_plugins.layouts.menus import menu_action, menu_operator, menu_separator, menu_submenu, menu_toggle

class MyMenu(Menu):
    label = "My Plugin"
    location = "MENU_BAR"
    order = 200

    def menu_items(self):
        return [menu_action("Print status", lambda: lf.log.info("Ready"))]
```

Register/unregister the class with `lf.register_class()`/`lf.unregister_class()`.
Prefer a declarative `menu_items()` list. `draw(layout)` is a legacy fallback;
interactive widgets on a compatibility `UILayout` are inert. Operator IDs can
be discovered with `lf.ops.get_all()` and `get_descriptor()`.

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
| `update_policy` | `str` | `"dirty"` | Use `"interval"` only for panels that need periodic updates; normal data panels update from explicit invalidation |
| `update_interval_ms` | `int \| None` | `100` | Interval refresh in milliseconds; unused by the dirty policy. Use an integer with `update_policy = "interval"` |

| Method | Returns | Description |
|---|---|---|
| `poll(cls, context)` | `bool` | Classmethod. Show/hide condition |
| `draw(self, ui)` | `None` | Immediate-mode content |
| `on_bind_model(self, ctx)` | `None` | Bind retained data models before document load |
| `on_mount(self, doc)` | `None` | Called once after the retained document mounts |
| `on_unmount(self, doc)` | `None` | Called before the retained document is destroyed |
| `on_update(self, doc)` | `None \| bool` | Retained update hook. With `update_policy = "interval"` it runs on the interval; with `"dirty"` it runs only after explicit invalidation, scene changes, or update requests. Return `True` to mark content dirty |
| `on_scene_changed(self, doc)` | `None` | Called when the active scene generation changes |
| `capture_chrome(self)` | `dict or None` | Optional panel UI state to persist in the project GUIL chapter |
| `apply_chrome(self, payload)` | `None` | Restore the saved panel UI state |

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
| `RuntimeState.<field>.value` | Read the current app value; publishing is for state producers |
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
RuntimeState.eval_lpips
RuntimeState.scene_generation
RuntimeState.selection_generation
RuntimeState.mode_text
RuntimeState.active_tool
RuntimeState.active_submode
RuntimeState.transform_space
RuntimeState.pivot_mode
RuntimeState.multi_transform_mode
RuntimeState.import_overlay_state
RuntimeState.video_export_overlay_state
RuntimeState.account_state
RuntimeState.gallery_state
RuntimeState.export_progress_state
RuntimeState.mesh2splat_state
RuntimeState.splat_simplify_state
RuntimeState.scripts_generation
RuntimeState.language_generation
RuntimeState.render_settings_generation
```

For FPS use `lf.ui.get_fps()` or `lf.ui.get_ui_fps()`; there is no
`RuntimeState.fps`. Evaluation metrics (`eval_psnr`, `eval_ssim`, `eval_lpips`)
are `None` until an evaluation result is available.

Compatibility names `is_training`, `trainer_state`, `has_trainer`, and
`max_iterations` alias `training_running`, `training_state`, `trainer_loaded`,
and `total_iterations`. `psnr` maps a missing `eval_psnr` to `0.0`.
`training_progress` and `can_start_training` are computed signals.
`has_scene`, `scene_path`, `has_selection`, `selection_count`,
`viewport_width`, `viewport_height`, and `is_headless` are Python compatibility
signals rather than native store fields. For a current poll/snapshot use
`lf.ui.context()` or the native scene/selection/viewport queries.

`AppState`, `AppStore`, and `NativeAppStore` remain as compatibility aliases for older plugins. New plugin code should import `RuntimeState` from `lfs_plugins.ui`.

Old Python UI hooks still compile, but hook registration is deprecated for external plugins. Use retained RML data models plus `RuntimeState` subscriptions for new UI.

### Panel spaces

`MAIN_PANEL_TAB`, `SIDE_PANEL`, `VIEWPORT_OVERLAY`, `SCENE_HEADER`,
`FLOATING`, `BOTTOM_DOCK`, `LEFT_DOCK`, `STATUS_BAR`

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
| `mouse_x`        | `float` | Mouse X in screen pixels                     |
| `mouse_y`        | `float` | Mouse Y in screen pixels                     |
| `mouse_region_x` | `float` | Currently the same coordinate as `mouse_x`                     |
| `mouse_region_y` | `float` | Currently the same coordinate as `mouse_y`                     |
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

For registered operators, mouse-button and motion events fill the mouse
coordinates; `mouse_region_x/y` currently duplicate `mouse_x/y` rather than
subtracting a viewport origin. Subtract `lf.ui.context().viewport_bounds[:2]`
when you need viewport-local coordinates. The adapter populates modifiers on
button/key events; `pressure` and `over_gui` currently keep their defaults
(`1.0` and `False`). Do not use these fields as tablet-pressure or GUI-hit
queries for registered operators.

## Properties

Property update callbacks receive `(owner, None)`; read the assigned property
from the owner. `CollectionProperty` exposes its collection operations on the
descriptor itself (for example `MySettings.items.add()`); instance attribute
access returns the assigned value (default `None`). `PointerProperty.get_instance()` is likewise
a descriptor helper.


```python
from lfs_plugins.props import (
    Property, FloatProperty, IntProperty, BoolProperty,
    StringProperty, EnumProperty, FloatVectorProperty,
    IntVectorProperty, TensorProperty, CollectionProperty,
    PointerProperty, PropertyGroup, PropSubtype,
)
```

### FloatProperty

```text
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

```text
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

```text
BoolProperty(
    default: bool = False,
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### StringProperty

```text
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

```text
EnumProperty(
    items: list[tuple[str, str, str]] | None = None,  # (identifier, label, description)
    default: str = None,     # First item if None
    name: str = "",
    description: str = "",
    update: Callable = None,
)
```

### FloatVectorProperty

```text
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

```text
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

```text
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

```text
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
| `move(from_idx, to_idx)`     | `None`            | Reorder items            |
| `__len__()`          | `int`             | Item count               |
| `__getitem__(index)` | `PropertyGroup`   | Access by index          |
| `__iter__()`         | `Iterator`        | Iterate items            |

### PointerProperty

```text
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

The definitions below summarize dataclass fields; import the implementation
rather than redefining these classes. `field(default_factory=...)` indicates
a new container for each instance.

```python
from lfs_plugins.tool_defs.definition import ToolDef, SubmodeDef, PivotModeDef
```

```text
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
    action_only: bool = False                     # Invoke action without selecting tool
    selected: Callable[[Any], bool] | None = None # Optional selected-state query
    label_key: str = ""                          # Translation key
    description_key: str = ""                    # Translation key
```

| Method                  | Returns | Description                          |
|-------------------------|---------|--------------------------------------|
| `can_activate(context)` | `bool`  | Check if tool can be activated       |
| `to_dict()`             | `dict`  | Convert to dict for C++ interop      |

### SubmodeDef

```text
@dataclass(frozen=True)
class SubmodeDef:
    id: str           # Unique submode ID
    label: str        # Display label
    icon: str         # Icon name
    shortcut: str = ""
    label_key: str = ""
```

### PivotModeDef

```text
@dataclass(frozen=True)
class PivotModeDef:
    id: str           # Unique pivot mode ID
    label: str        # Display label
    icon: str         # Icon name
    label_key: str = ""
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

## Signals

```python
from lfs_plugins.ui.signals import Signal, ComputedSignal, ThrottledSignal, Batch, batch
```

### Signal[T]

```text
Signal(initial_value: T, name: str = "")
```

| Property/Method                          | Returns           | Description                      |
|------------------------------------------|-------------------|----------------------------------|
| `.value`                                 | `T`               | Get/set current value            |
| `.peek()`                                | `T`               | Get without tracking             |
| `.subscribe(callback)`                   | `() -> None`      | Subscribe; returns unsubscribe fn|
| `.subscribe_as(owner, callback)`         | `() -> None`      | Owner-tracked subscription       |

### ComputedSignal[T]

```text
ComputedSignal(compute: Callable[[], T], dependencies: list[Signal])
```

| Property/Method                          | Returns           | Description                      |
|------------------------------------------|-------------------|----------------------------------|
| `.value`                                 | `T`               | Get computed value (lazy)        |
| `.subscribe(callback)`                   | `() -> None`      | Subscribe to changes             |
| `.subscribe_as(owner, callback)`         | `() -> None`      | Owner-tracked subscription       |

### ThrottledSignal[T]

```text
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

```text
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

```text
@dataclass
class CapabilitySchema:
    properties: dict[str, dict[str, Any]] = field(default_factory=dict)
    required: list[str] = field(default_factory=list)
```

### Capability

```text
@dataclass
class Capability:
    name: str
    description: str
    handler: Callable
    schema: CapabilitySchema = field(default_factory=CapabilitySchema)
    plugin_name: Optional[str] = None
    requires_gui: bool = True
```

### PluginContext

```text
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

```text
@dataclass
class SceneContext:
    scene: Any                          # PyScene object
```

| Method                       | Returns | Description                  |
|------------------------------|---------|------------------------------|
| `set_selection_mask(mask)`   | `None`  | Apply selection mask         |

### ViewContext

```text
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

```text
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

```text
@dataclass
class PluginInfo:
    name: str
    version: str
    path: Path
    description: str = ""
    author: str = ""
    entry_point: str = "__init__"
    dependencies: list[str] = field(default_factory=list)
    auto_start: bool = False
    hot_reload: bool = True
    plugin_api: str = ""
    lichtfeld_version: str = ""
    required_features: list[str] = field(default_factory=list)
```

### PluginState

```text
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

### Collapsible

| Method                                              | Returns | Description                    |
|-----------------------------------------------------|---------|--------------------------------|
| `collapsing_header(label, default_open=False)`      | `bool`  | Collapsible section            |

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
| `image(texture_id, size, tint=None)`                   | `None`  | Display image          |
| `image_uv(texture_id, size, uv0, uv1, tint=None)` | `None` | Compatibility form of `image()`; UV arguments are currently ignored |
| `image_button(id, texture_id, size, tint=None)`        | `bool`  | Clickable image        |
| `toolbar_button(id, texture_id, size, selected=False, disabled=False, tooltip='')` | `bool` | Toolbar icon button |

Panel `RmlUILayout` has `image()`, `image_uv()`, and `image_button()`. It does
**not** expose `image_tensor()` or `image_texture()`. Those names exist only
on compatibility `UILayout`, where image drawing is currently inert.

For a tensor preview, keep a `DynamicTexture` on the panel instance and pass
its `id` to `image()` (see [the guide](getting-started.md#displaying-gpu-tensors)).
`image()` and `image_button()` currently ignore `tint`; `image_uv()` also
ignores `uv0`/`uv1`. Use retained RML/RCSS for styling the image container.

---

### DynamicTexture

GPU tensor to UI texture bridge. In the Vulkan viewer this uses an opaque Vulkan UI texture id (`uint64`).

```python
tex = lf.ui.DynamicTexture()          # Empty
tex = lf.ui.DynamicTexture(tensor, plugin_name="my_plugin")  # UI thread only
```

| Method / Property  | Returns              | Description                                    |
|--------------------|----------------------|------------------------------------------------|
| `update(tensor)`   | `None`               | Upload `[H, W, 3\|4]` tensor (auto-converts CPU→CUDA, uint8→float32 normalized to [0,1]) |
| `destroy()`        | `None`               | Release UI texture resources                   |
| `id`               | `int`                | Opaque Vulkan UI texture id (`uint64`)         |
| `width`            | `int`                | Current width in pixels                        |
| `height`           | `int`                | Current height in pixels                       |
| `valid`            | `bool`               | `True` if texture is initialized               |
| `uv1`             | `tuple[float, float]` | Compatibility value `(1.0, 1.0)` in the Vulkan backend        |

Calling `update()` with a different resolution recreates the backend texture.
Texture creation/upload requires the UI thread; use a panel's `draw()` or
`lf.ui.schedule_on_ui_thread(callback)` for work originating on a worker.
A texture created with `DynamicTexture(tensor, plugin_name="my_plugin")` is
tracked by `lf.ui.free_plugin_textures("my_plugin")` during plugin cleanup.
An empty `DynamicTexture()` has no plugin owner: destroy it in `on_unmount()`
or release your reference on unload.

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
| `begin_menu_bar()` / `end_menu_bar()` | `bool` / `None` | Begin/end menu bar |
| `menu_item(label, enabled=True, selected=False)`                    | `bool`  | Menu item                |
| `menu_item_toggle(label, shortcut, selected)` | `(bool, bool)` | `(clicked, new_selected)` on `RmlUILayout` |
| `menu_item_shortcut(label, shortcut, enabled=True)` | `bool`  | Menu item with shortcut  |

### Child Regions

| Method                                                        | Returns        | Description           |
|---------------------------------------------------------------|----------------|-----------------------|
| `begin_child(id, size, border=False)`                   | `bool`         | Start child region    |
| `end_child()`                                                 | `None`         | End child region      |

### Overlay Window State

These methods belong to viewport-overlay `UILayout`; they are not exposed on
panel `RmlUILayout`, whose containing panel already owns position and size.

| Method                                                        | Returns        | Description           |
|---------------------------------------------------------------|----------------|-----------------------|
| `begin_window(title, flags=0)`                                | `bool`         | Start overlay window state |
| `end_window()`                                                | `None`         | End overlay window state |
| `set_next_window_pos(pos, first_use=False)`                   | `None`         | Set window position   |
| `set_next_window_size(size, first_use=False)`                 | `None`         | Set window size       |
| `set_next_window_pos_centered(first_use=False)`               | `None`         | Center next window (main viewport) |

### Drawing (Viewport)

On a viewport-overlay `UILayout`, these primitives enqueue into the active
viewport-scoped `ScreenOverlayRenderer`. Coordinates are absolute screen
coordinates. Outside an active overlay frame they emit no command.

| Method                                                         | Returns | Description            |
|----------------------------------------------------------------|---------|------------------------|
| `draw_line(x0, y0, x1, y1, color, thickness=1)`               | `None`  | Line                   |
| `draw_rect(x0, y0, x1, y1, color, thickness=1)`               | `None`  | Rectangle outline      |
| `draw_rect_filled(x0, y0, x1, y1, color, background=False)`           | `None`  | Filled rectangle       |
| `draw_rect_rounded(x0, y0, x1, y1, color, rounding, thickness=1, background=False)`  | `None`  | Rounded rect outline   |
| `draw_rect_rounded_filled(x0, y0, x1, y1, color, rounding, background=False)`    | `None`  | Filled rounded rect    |
| `draw_circle(x, y, radius, color, segments=32, thickness=1)`  | `None`  | Circle outline         |
| `draw_circle_filled(x, y, radius, color, segments=32)`        | `None`  | Filled circle          |
| `draw_triangle_filled(x0, y0, x1, y1, x2, y2, color, background=False)`  | `None`  | Filled triangle        |
| `draw_text(x, y, text, color, background=False)`                      | `None`  | Text at position       |
| `draw_polyline(points, color, closed=False, thickness=1)`      | `None`  | Polyline               |
| `draw_poly_filled(points, color)`                              | `None`  | Filled polygon         |

### Drawing (Window)

The `draw_window_*` names are compatibility aliases and use the same absolute
screen-coordinate convention as `draw_*`; they do not add a window origin.

| Method                                                         | Returns | Description            |
|----------------------------------------------------------------|---------|------------------------|
| `draw_window_rect_filled(x0, y0, x1, y1, color)`              | `None`  | Filled rectangle       |
| `draw_window_rect(x0, y0, x1, y1, color, thickness=1)`        | `None`  | Rectangle outline      |
| `draw_window_rect_rounded(x0, y0, x1, y1, color, rounding, thickness=1)` | `None` | Rounded outline    |
| `draw_window_rect_rounded_filled(x0, y0, x1, y1, color, rounding)`   | `None`  | Filled rounded rect    |
| `draw_window_line(x0, y0, x1, y1, color, thickness=1)`        | `None`  | Line                   |
| `draw_window_text(x, y, text, color)`                          | `None`  | Text                   |
| `draw_window_triangle_filled(x0, y0, x1, y1, x2, y2, color)` | `None`  | Filled triangle        |

The legacy `background` arguments on applicable `draw_*` calls are accepted
but ignored: there is no separate background draw list. Overlay calls share
one command stream, packed as shapes followed by text/images. Enqueue order is
retained within each batch, but cross-batch call order is not a z-order
guarantee.

### Plots

`plot_lines(label, values, scale_min=0, scale_max=0, size=(0,0))` draws a line
plot on panel `RmlUILayout`. It is inert on overlay `UILayout`.

### Progress & Status

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `progress_bar(fraction, overlay='', width=0, height=0)`       | `None`  | Progress bar             |
| `set_tooltip(text)`                                 | `None`  | Tooltip for last item    |

### State Queries

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `is_item_hovered()`                                 | `bool`  | Last item hovered        |
| `is_item_clicked(button=0)`                         | `bool`  | Last item clicked        |
| `is_item_active()`                                  | `bool`  | Last item active         |
| `is_mouse_double_clicked(button=0)`                 | `bool`  | Double click detected    |
| `is_mouse_dragging(button=0)`                       | `bool`  | Mouse dragging           |
| `get_mouse_wheel()`                                 | `float` | Scroll wheel delta       |
| `get_mouse_delta()`                                 | `tuple` | Mouse delta (dx, dy)     |

### Position / Size

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
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

Panel styling uses RML/RCSS and `RmlSubLayout.enabled`/`active`/`alert`.
`RmlUILayout` has no `get_style_color()`, `set_style_color()`, `get_style_var()`,
or `set_style_var()` methods. The old compatibility style-stack methods do not
provide a panel theme API.


| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `push_item_width(width)` / `pop_item_width()`      | `None`  | Item width stack         |
| `begin_disabled(disabled=True)` / `end_disabled()`  | `None`  | Disable widget region. For composable disabled regions, prefer `SubLayout.enabled` (see Layout Composition below). |

### Pointer

| Method                                              | Returns | Description              |
|-----------------------------------------------------|---------|--------------------------|
| `set_mouse_cursor_hand()`                           | `None`  | Set hand cursor          |

### Specialized Widgets

| Method                                                                            | Returns        | Description           |
|-----------------------------------------------------------------------------------|----------------|-----------------------|
| `template_list(list_type_id, list_id, data, prop_id, active_data, active_prop, rows=5)` | `(int, int)` | Live on `RmlUILayout`; returns `(active_index, item_count)` and writes row selection to `active_data.active_prop`. Compatibility `UILayout` raises `TypeError` outside draw hooks and warns/returns inert values in draw hooks. |

Register a custom list class with `lf.register_uilist(MyList)`. Its `list_id` (or class
name when omitted) is the `list_type_id` passed to `template_list`. The instance
receives `draw_item(layout, data, item, icon, active_data, active_prop, index)` once
per item per draw. `layout` is a live `RmlUILayout` scoped to that row, `icon` is
currently `0`, and the active-selection arguments are the same object and property
passed to `template_list`. Instances persist until unregistered or replaced.
Unregistered types use the ordinary list control.

### Layout Composition

Create composable sub-layouts with automatic widget positioning and state cascading.

| Method                                             | Returns     | Description                     |
|----------------------------------------------------|-------------|---------------------------------|
| `row()`                                            | `RmlSubLayout` | Horizontal layout               |
| `column()`                                         | `RmlSubLayout` | Vertical layout                 |
| `split(factor=0.5)`                                | `RmlSubLayout` | Two-child split; factor controls first/second width ratio |
| `box()`                                            | `RmlSubLayout` | Bordered container              |
| `grid_flow(columns=0, even_columns=True, even_rows=True)` | `RmlSubLayout` | Responsive grid with explicit column/row sizing controls |
| `prop_enum(data, prop_id, value, text='')`          | `bool`      | Enum toggle button              |

Panel layouts return `lf.ui.RmlSubLayout`; compatibility `UILayout` returns
`lf.ui.SubLayout`. Both are context managers, but their method sets differ.
Check the corresponding class in the [UI stubs](../../src/python/stubs/lichtfeld/ui/__init__.pyi)
before calling parent-layout methods on a child. `RmlSubLayout` has fewer
methods; for example, call `image_uv()` on the parent `ui`, not on a row.

A sub-layout is a context manager. Use `with ui.row() as row:` to enter the layout, then call widget methods on `row` instead of `ui`. Sub-layouts nest arbitrarily.

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

## UI Functions

| Function                                    | Returns          | Description                |
|---------------------------------------------|------------------|----------------------------|
| `lf.ui.tr(key)`                             | `str`            | Translate string           |
| `lf.ui.theme()`                             | `Theme`          | Current theme              |
| `lf.ui.context()`                           | `AppContext`     | App context                |
| `lf.ui.schedule_on_ui_thread(callback)` | `None` | Schedule a no-argument callback on the UI thread |
| `lf.ui.request_redraw(delay=0)`                    | `None`           | Request UI redraw          |
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
| `lf.ui.ops.invoke(op_id, /, **kwargs)`         | `OperatorReturnValue` | Invoke operator       |
| `lf.ui.ops.poll(op_id)`                     | `bool`           | Operator poll              |
| `lf.ui.ops.cancel_modal()`                  | `None`           | Cancel modal operator      |
| `lf.ui.get_active_tool()`                   | `str`            | Active tool ID             |
| `lf.ui.get_active_submode()`                | `str`            | Active submode             |
| `lf.ui.set_selection_mode(mode)`            | `None`           | Set selection submode      |
| `lf.ui.get_transform_space()`               | `int`            | Transform space enum index |
| `lf.ui.set_transform_space(space)`          | `None`           | Set transform space index  |
| `lf.ui.get_pivot_mode()` / `set_pivot_mode(mode)` | `int`      | Pivot mode enum index      |
| `lf.ui.get_fps()`                           | `float`          | Fresh view renders in the trailing second; cached/deferred results excluded |
| `lf.ui.get_ui_fps()`                        | `float`          | Successful UI presents in the trailing second; idle-clear frame excluded |
| `lf.ui.get_git_commit()`                    | `str`            | Git commit hash            |
| `lf.ui.is_key_pressed(key, repeat=False)`    | `bool`           | SDL-backed rising edge for the current UI frame; UI thread only, no repeat events |
| `lf.ui.is_key_down(key)`                     | `bool`           | Current SDL keyboard level |

For a coarse CUDA memory number in Python plugin code, use
`lfs_plugins.get_gpu_memory()` from the helper package. It returns PyTorch's
CUDA **allocated bytes**, or `0` when PyTorch/CUDA is unavailable; it does not
measure all native LichtFeld GPU allocations. There is no `lf.ui.get_gpu_memory()`
binding. `lf.diagnostics.collect()` returns best-effort system/CUDA/GPU diagnostics.

## File Dialogs

| Function                                    | Returns          |
|---------------------------------------------|------------------|
| `lf.ui.open_image_dialog(start_dir='')`     | `str`            |
| `lf.ui.open_folder_dialog(title='Select Folder', start_dir='')` | `str` |
| `lf.ui.open_dataset_folder_dialog(default_path='')`        | `str`            |
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

## UI Hooks

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

## pyproject.toml Schema

```toml
[project]
name = ""                    # string, required - Unique plugin identifier
version = ""                 # string, required - Semantic version
description = ""             # string, required
authors = []                 # list[{name, email}], optional - PEP 621 authors
dependencies = []            # list[string], optional - Python packages (PEP 508)

[tool.lichtfeld]
auto_start = false           # bool, optional - Load at startup unless user settings override
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

## Plugin Errors

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

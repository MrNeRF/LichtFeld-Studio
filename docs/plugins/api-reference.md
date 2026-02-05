# Plugin API Reference

Complete API reference for LichtFeld Studio plugins.

---

## Registration

```python
import lichtfeld as lf

lf.register_class(cls)           # Register a Panel or Operator class
lf.unregister_class(cls)         # Unregister a Panel or Operator class
```

---

## Panel

```python
from lfs_plugins.types import Panel
```

| Attribute | Type   | Description            |
|-----------|--------|------------------------|
| `label`   | `str`  | Display name           |
| `space`   | `str`  | Panel space (see below)|
| `order`   | `int`  | Sort order             |

| Method               | Returns | Description                      |
|----------------------|---------|----------------------------------|
| `poll(cls, context)` | `bool`  | Classmethod. Show/hide condition |
| `draw(self, layout)` | `None`  | Render panel via layout API      |

### Panel spaces

`SIDE_PANEL`, `MAIN_PANEL_TAB`, `VIEWPORT_OVERLAY`, `SCENE_HEADER`, `FLOATING`, `DOCKABLE`, `STATUS_BAR`

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
| `key_code`       | `int`   | Raw GLFW key code                              |

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
    group: str = "default"                       # "select", "transform", "paint", "utility"
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

| Method                          | Returns            | Description                       |
|---------------------------------|--------------------|-----------------------------------|
| `plugins_dir`                   | `Path`             | Property. `~/.lichtfeld/plugins/` |
| `discover()`                    | `list[PluginInfo]` | Scan for plugins                  |
| `load(name, on_progress=None)`  | `None`             | Load a plugin                     |
| `unload(name)`                  | `None`             | Unload a plugin                   |
| `install(url, on_progress=None)`| `str`              | Install from GitHub URL           |
| `uninstall(name)`               | `None`             | Remove a plugin                   |
| `update(name, on_progress=None)`| `None`             | Update a plugin                   |
| `get_state(name)`               | `Optional[PluginState]` | Get plugin state             |
| `get_error(name)`               | `Optional[str]`    | Get error message                 |
| `get_traceback(name)`           | `Optional[str]`    | Get error traceback               |

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
    auto_start: bool = True
    hot_reload: bool = True
    min_lichtfeld_version: str = ""
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

---

## Layout API

The `layout` object is passed to `Panel.draw()` and provides all UI widget methods. It wraps ImGui.

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
| `path_input(label, value, folder_mode=True, dialog_title='')` | `(bool, str)` | File/folder picker  |

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

### Images

| Method                                                       | Returns | Description            |
|--------------------------------------------------------------|---------|------------------------|
| `image(texture_id, size, tint=(1,1,1,1))`                   | `None`  | Display image          |
| `image_uv(texture_id, size, uv0, uv1, tint=(1,1,1,1))`     | `None`  | Image with UV coords   |
| `image_button(id, texture_id, size, tint=(1,1,1,1))`        | `bool`  | Clickable image        |
| `toolbar_button(id, tex, size, selected=F, disabled=F, tooltip='')` | `bool` | Toolbar icon button |

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
| `begin_popup_context_item(id='')`                   | `bool`  | Context menu popup       |
| `begin_popup_modal(title)`                          | `bool`  | Modal popup              |
| `open_popup(id)`                                    | `None`  | Trigger popup open       |
| `end_popup()` / `end_popup_modal()`                 | `None`  | End popup                |
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
| `set_next_window_pos_viewport_center()`                       | `None`         | Viewport center       |
| `set_next_window_focus()`                                     | `None`         | Focus next window     |
| `push_window_style()` / `pop_window_style()`                  | `None`         | Window style stack    |
| `push_modal_style()` / `pop_modal_style()`                    | `None`         | Modal style stack     |

### Drawing (Viewport)

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

| Method                                                         | Returns | Description            |
|----------------------------------------------------------------|---------|------------------------|
| `draw_window_rect_filled(x0, y0, x1, y1, color)`              | `None`  | Filled rect to window  |
| `draw_window_rect(x0, y0, x1, y1, color, thickness=1)`        | `None`  | Rect outline to window |
| `draw_window_line(x0, y0, x1, y1, color, thickness=1)`        | `None`  | Line to window         |
| `draw_window_text(x, y, text, color)`                          | `None`  | Text to window         |
| `draw_window_triangle_filled(x0, y0, x1, y1, x2, y2, color)` | `None`  | Triangle to window     |

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
| `crf_curve_preview(label, gamma, toe, shoulder, gamma_r=0, gamma_g=0, gamma_b=0)`| `None`         | Tone curve preview    |
| `chromaticity_diagram(label, rx, ry, gx, gy, bx, by, nx, ny, range=0.5)`         | `(bool, list)` | Chromaticity diagram  |
| `template_list(list_type_id, list_id, data, prop_id, active_data, active_prop, rows=5)` | `(int, int)` | Custom list template |

### Layout Composition

Create composable sub-layouts with automatic widget positioning and state cascading.

| Method                                             | Returns     | Description                     |
|----------------------------------------------------|-------------|---------------------------------|
| `row()`                                            | `SubLayout` | Horizontal layout               |
| `column()`                                         | `SubLayout` | Vertical layout                 |
| `split(factor=0.5)`                                | `SubLayout` | Two-column split                |
| `box()`                                            | `SubLayout` | Bordered container              |
| `grid_flow(columns=0, even_columns=True, even_rows=True)` | `SubLayout` | Responsive grid         |
| `prop_enum(data, prop_id, value, text='')`          | `bool`      | Enum toggle button              |

`SubLayout` is a context manager. Use `with layout.row() as row:` to enter the layout, then call widget methods on `row` instead of `layout`. Sub-layouts nest arbitrarily.

#### SubLayout state properties

| Property  | Type    | Description                                |
|-----------|---------|--------------------------------------------|
| `enabled` | `bool`  | Disabled state (cascades to children)      |
| `active`  | `bool`  | Active state (cascades to children)        |
| `alert`   | `bool`  | One-shot alert styling (red text/bg)       |

#### Example

```python
def draw(self, layout):
    with layout.row() as row:
        row.prop_enum(self, "mode", "fast", "Fast")
        row.prop_enum(self, "mode", "quality", "Quality")

    with layout.box() as box:
        box.heading("Settings")
        box.prop(self, "opacity")

    with layout.column() as col:
        col.enabled = self.is_active
        col.prop(self, "value")
        with col.row() as row:
            row.button("Apply")
            row.button("Cancel")

    with layout.grid_flow(columns=3) as grid:
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
| `duplicate_node(name)`                            | `str`            | Duplicate, returns new name       |
| `merge_group(group_name)`                         | `str`            | Merge group children              |
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
| `world_transform`   | `tuple`              | World-space 4x4 transform        |
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
| `get_selected_node_transform()`       | `list[float]`    | 16 floats, column-major           |
| `set_selected_node_transform(matrix)` | `None`           | Set transform                     |
| `get_selection_center()`              | `list[float]`    | Local space center                |
| `get_selection_world_center()`        | `list[float]`    | World space center                |
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
| `clear_selection()`             | `None`      | Clear gaussian selection     |
| `has_selection()`               | `bool`      | Any gaussians selected       |
| `selection_mask`                | `Tensor`    | Property. Current mask       |
| `set_selection(indices)`        | `None`      | Select by index list         |

### Transforms

| Function                                    | Returns        | Description                      |
|---------------------------------------------|----------------|----------------------------------|
| `get_node_transform(name)`                  | `list[float]`  | 16 floats, column-major          |
| `set_node_transform(name, matrix)`          | `None`         | Set 4x4 transform                |
| `decompose_transform(matrix)`               | `dict`         | See keys below                   |
| `compose_transform(translation, euler, scale)` | `list[float]` | Build 4x4 from components      |

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
| `soft_delete(mask)`   | `Tensor`     | Mark for deletion, returns prev state |
| `undelete(mask)`      | `None`       | Restore deleted gaussians       |
| `apply_deleted()`     | `int`        | Permanently remove, returns count|
| `clear_deleted()`     | `None`       | Clear deletion mask             |
| `deleted`             | `Tensor`     | Property. [N] bool deletion mask|
| `has_deleted_mask()`  | `bool`       | Whether deletion mask exists    |
| `visible_count()`     | `int`        | Number of non-deleted gaussians |

> After calling `soft_delete()`, `undelete()`, or `clear_deleted()`, call `scene.notify_changed()` to update the viewport.

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

### Rendering

| Function                                              | Returns          | Description                |
|-------------------------------------------------------|------------------|----------------------------|
| `get_current_view()`                                  | `ViewInfo`       | Current camera view        |
| `get_viewport_render()`                               | `ViewportRender` | Current viewport image     |
| `capture_viewport()`                                  | `ViewportRender` | Capture for async use      |
| `render_view(rotation, translation, w, h, fov=60, bg=None)` | `Tensor`  | Render from camera         |
| `compute_screen_positions(rotation, translation, w, h, fov=60)` | `Tensor` | [N, 2] screen positions |
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
    format: int,             # 0=PLY, 1=SOG, 2=SPZ, 3=HTML
    path: str,
    node_names: list[str],
    sh_degree: int,
)
lf.save_config_file(path: str)
```

### Logging

| Function           | Description       |
|--------------------|-------------------|
| `lf.log.info(msg)` | Info level        |
| `lf.log.warn(msg)` | Warning level     |
| `lf.log.error(msg)`| Error level       |
| `lf.log.debug(msg)`| Debug level       |

### Undo

```python
lf.undo.push(name: str, undo: Callable, redo: Callable)
```

### UI Functions

| Function                                    | Returns          | Description                |
|---------------------------------------------|------------------|----------------------------|
| `lf.ui.tr(key)`                             | `str`            | Translate string           |
| `lf.ui.theme()`                             | `Theme`          | Current theme              |
| `lf.ui.context()`                           | `AppContext`     | App context                |
| `lf.ui.request_redraw()`                    | `None`           | Request UI redraw          |
| `lf.ui.set_language(lang)`                  | `None`           | Set UI language            |
| `lf.ui.set_panel_enabled(name, enabled)`    | `None`           | Toggle panel               |
| `lf.ui.is_panel_enabled(name)`              | `bool`           | Panel enabled state        |
| `lf.ui.get_panel_names()`                   | `list[str]`      | All panel names            |
| `lf.ui.ops.invoke(op_id)`                   | `None`           | Invoke operator            |
| `lf.ui.ops.cancel_modal()`                  | `None`           | Cancel modal operator      |
| `lf.ui.get_active_tool()`                   | `str or None`    | Active tool ID             |
| `lf.ui.get_active_submode()`                | `str or None`    | Active submode             |
| `lf.ui.set_selection_mode(mode)`            | `None`           | "centers", "rectangle", etc|
| `lf.ui.get_transform_space()`               | `str`            | "local" or "world"        |
| `lf.ui.set_transform_space(space)`          | `None`           | Set transform space        |
| `lf.ui.get_pivot_mode()` / `set_pivot_mode(mode)` | `str`     | Pivot mode                 |
| `lf.ui.get_fps()`                          | `float`          | Current FPS                |
| `lf.ui.get_gpu_memory()`                   | `(int, int)`     | (used, total) GPU memory in bytes |
| `lf.ui.get_git_commit()`                   | `str`            | Git commit hash            |

### File Dialogs

| Function                                    | Returns          |
|---------------------------------------------|------------------|
| `lf.ui.open_file_dialog()`                  | `str or None`    |
| `lf.ui.open_dataset_folder_dialog()`        | `str or None`    |
| `lf.ui.open_ply_file_dialog(initial)`       | `str or None`    |
| `lf.ui.open_checkpoint_file_dialog()`       | `str or None`    |
| `lf.ui.open_json_file_dialog()`             | `str or None`    |
| `lf.ui.open_image_file_dialog()`            | `str or None`    |
| `lf.ui.save_file_dialog()`                  | `str or None`    |
| `lf.ui.save_json_file_dialog()`             | `str or None`    |
| `lf.ui.save_ply_file_dialog()`              | `str or None`    |
| `lf.ui.save_sog_file_dialog()`              | `str or None`    |
| `lf.ui.save_spz_file_dialog()`              | `str or None`    |
| `lf.ui.save_html_file_dialog()`             | `str or None`    |

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

**Available hook points:**

| Panel | Section | Description |
|---|---|---|
| `"rendering"` | `"selection_groups"` | Rendering panel, between settings and tools |

### Tensor API

```python
import lichtfeld as lf
t = lf.Tensor
```

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
| `.max(dim=None)`                    | `Tensor` | Reduce max               |
| `.min(dim=None)`                    | `Tensor` | Reduce min               |
| `.reshape(shape)`                   | `Tensor` | Reshape                  |
| `.view(shape)`                      | `Tensor` | View reshape             |
| `.squeeze(dim=None)`                | `Tensor` | Remove size-1 dims       |
| `.unsqueeze(dim)`                   | `Tensor` | Add size-1 dim           |
| `.transpose(dim0, dim1)`            | `Tensor` | Swap dimensions          |
| `.permute(dims)`                    | `Tensor` | Reorder dimensions       |
| `.flatten(start=0, end=-1)`         | `Tensor` | Flatten range            |

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

---

## plugin.toml Schema

```toml
[plugin]
name = ""                    # string, required - Unique plugin identifier
version = ""                 # string, required - Semantic version
description = ""             # string, optional - Plugin description
author = ""                  # string, optional - Author name
entry_point = "__init__"     # string, optional - Module to load (default: __init__)
min_lichtfeld_version = ""   # string, optional - Minimum LichtFeld version

[dependencies]
packages = []                # list[string], optional - Python packages

[lifecycle]
auto_start = true            # bool, optional - Auto-load on startup (default: true)
hot_reload = true            # bool, optional - Support hot reload (default: true)
```

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

All return OpenGL texture ID (0 on failure). Icons are cached by C++.

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

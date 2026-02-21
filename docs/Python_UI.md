# LichtFeld Studio Python UI Framework

Create custom UI panels and widgets using Python with the ImGui-backed UI system.

---

## Panel System

### Defining a Panel

```python
import lichtfeld as lf
from lfs_plugins.types import Panel

class MyPanel(Panel):
    idname = "my_plugin.panel"
    label = "My Custom Panel"
    space = "FLOATING"      # SIDE_PANEL, FLOATING, VIEWPORT_OVERLAY, DOCKABLE, MAIN_PANEL_TAB, SCENE_HEADER, STATUS_BAR
    order = 100
    options = set()           # e.g. {"DEFAULT_CLOSED"}
    poll_deps = {"SCENE", "SELECTION", "TRAINING"}

    def __init__(self):
        self.value = 0.5
        self.enabled = True

    @classmethod
    def poll(cls, ctx):
        return ctx.has_scene

    def draw(self, ui):
        ui.heading("Settings")
        _, self.enabled = ui.checkbox("Enabled", self.enabled)
        _, self.value = ui.slider_float("Value", self.value, 0.0, 1.0)

        if ui.button("Apply"):
            print("Applied")

lf.register_class(MyPanel)
```

### Panel Spaces

| Space | Description |
|-------|-------------|
| `SIDE_PANEL` | Sidebar panel |
| `FLOATING` | Standalone window |
| `VIEWPORT_OVERLAY` | Draw over viewport |
| `DOCKABLE` | Dockable window |
| `MAIN_PANEL_TAB` | Main tab container |
| `SCENE_HEADER` | Scene panel/header area |
| `STATUS_BAR` | Bottom status bar |

### Panel Management

```python
lf.register_class(MyPanel)
lf.unregister_class(MyPanel)

# Low-level panel API
lf.ui.register_panel(MyPanel)
lf.ui.unregister_panel(MyPanel)
lf.ui.unregister_all_panels()

lf.ui.set_panel_enabled("my_plugin.panel", False)
lf.ui.is_panel_enabled("my_plugin.panel")
lf.ui.get_panel_names("FLOATING")
lf.ui.get_panel("my_plugin.panel")
lf.ui.set_panel_label("my_plugin.panel", "New Label")
lf.ui.set_panel_order("my_plugin.panel", 10)
lf.ui.set_panel_space("my_plugin.panel", "SIDE_PANEL")
```

### Poll Context

Use `lf.ui.context()` for global app state, and in `poll(ctx)` use the provided context.

| Property | Type | Description |
|----------|------|-------------|
| `ctx.has_scene` | bool | Scene is loaded |
| `ctx.scene_generation` | int | Scene generation counter |
| `ctx.has_trainer` | bool | Trainer exists |
| `ctx.is_training` | bool | Training is active |
| `ctx.is_paused` | bool | Training paused |
| `ctx.iteration` | int | Current iteration |
| `ctx.max_iterations` | int | Total iterations |
| `ctx.loss` | float | Current loss |
| `ctx.has_selection` | bool | Selection exists |
| `ctx.num_gaussians` | int | Gaussian count |
| `ctx.selection_submode` | int | Selection submode enum |
| `ctx.pivot_mode` | int | Pivot mode enum |
| `ctx.transform_space` | int | Transform space enum |

---

## UI Widgets

All widgets are called on the `layout`/`ui` object passed to `draw()`.

### Text

```python
ui.label("Simple text")
ui.label_centered("Centered")
ui.heading("Heading")
ui.text_colored("Warning", (1.0, 0.6, 0.2, 1.0))
ui.text_colored_centered("Centered color", (0.3, 0.7, 1.0, 1.0))
ui.text_wrapped("Long wrapped text")
ui.text_disabled("Disabled text")
ui.bullet_text("Bullet")
ui.text_selectable("Copyable text")
```

### Buttons

```python
if ui.button("Click"):
    pass

if ui.button_styled("Primary", "primary", (120, 28)):
    pass

if ui.button_callback("Run", lambda: print("clicked")):
    pass

ui.small_button("Small")
ui.invisible_button("hitbox", (100, 24))
```

### Checkboxes & Radio Buttons

```python
_, self.enabled = ui.checkbox("Enable Feature", self.enabled)

_, self.choice = ui.radio_button("Option A", self.choice, 0)
_, self.choice = ui.radio_button("Option B", self.choice, 1)
```

### Input Fields

```python
_, self.name = ui.input_text("Name", self.name)
_, self.name = ui.input_text_with_hint("Name", "Enter name...", self.name)
_, self.name = ui.input_text_enter("Name", self.name)

_, self.scale = ui.input_float("Scale", self.scale)
_, self.count = ui.input_int("Count", self.count)
_, self.count = ui.input_int_formatted("Count", self.count)
_, self.value = ui.stepper_float("Value", self.value)

_, self.path = ui.path_input("Output", self.path, folder_mode=True)
```

### Sliders

```python
_, self.value = ui.slider_float("Float", self.value, 0.0, 1.0)
_, self.amount = ui.slider_int("Int", self.amount, 0, 100)
_, self.vec2 = ui.slider_float2("Vec2", self.vec2, 0.0, 1.0)
_, self.vec3 = ui.slider_float3("Vec3", self.vec3, -1.0, 1.0)
```

### Drag Inputs

```python
_, self.value = ui.drag_float("Value", self.value, speed=0.01, min=0.0, max=10.0)
_, self.index = ui.drag_int("Index", self.index, speed=1, min=0, max=100)
```

### Color Pickers

```python
_, self.rgb = ui.color_edit3("Color", self.rgb)
_, self.rgba = ui.color_edit4("Color", self.rgba)
ui.color_button("##preview", (1.0, 0.2, 0.2, 1.0), (20, 20))
```

### Dropdowns & Lists

```python
options = ["Option A", "Option B", "Option C"]
_, self.selected = ui.combo("Select", self.selected, options)
_, self.selected = ui.listbox("Items", self.selected, options, height_items=5)

if ui.selectable("Selectable", selected=False):
    pass
```

### Progress Bar

```python
ui.progress_bar(0.75)
ui.progress_bar(0.5, "Loading...", width=200, height=0)
```

---

## Layout

### Spacing & Separators

```python
ui.separator()
ui.spacing()
ui.new_line()
ui.same_line()
ui.same_line(120)
```

### Indentation

```python
ui.indent(20.0)
ui.label("Indented")
ui.unindent(20.0)
```

### Width Control

```python
ui.set_next_item_width(200)
_, value = ui.input_float("Fixed Width", 1.0)

ui.push_item_width(180)
_, value = ui.input_float("Stacked Width", 1.0)
ui.pop_item_width()
```

### Groups

```python
ui.begin_group()
ui.label("Grouped")
ui.button("Together")
ui.end_group()
```

### Collapsible Sections

```python
if ui.collapsing_header("Advanced"):
    ui.label("Hidden by default")

if ui.collapsing_header("Open", default_open=True):
    ui.label("Visible by default")
```

### Tree Nodes

```python
if ui.tree_node("Parent"):
    ui.label("Child")
    if ui.tree_node_ex("Nested", flags="DEFAULT_OPEN"):
        ui.label("Nested child")
        ui.tree_pop()
    ui.tree_pop()
```

### Tables

```python
if ui.begin_table("my_table", 3):
    ui.table_setup_column("A")
    ui.table_setup_column("B")
    ui.table_setup_column("C")
    ui.table_headers_row()

    ui.table_next_row()
    ui.table_next_column(); ui.label("1")
    ui.table_next_column(); ui.label("2")
    ui.table_next_column(); ui.label("3")

    ui.end_table()
```

---

## Interaction

### Tooltips

```python
ui.button("Hover Me")
ui.set_tooltip("This tooltip appears on hover")

if ui.is_item_hovered():
    ui.set_tooltip("Custom tooltip")
```

### Click Detection

```python
ui.button("Click Me")
if ui.is_item_clicked(0):  # 0=left, 1=right, 2=middle
    print("Clicked")
```

### ID Management

```python
for i, item in enumerate(items):
    ui.push_id_int(i)
    if ui.button("Delete"):
        delete_item(i)
    ui.pop_id()
```

---

## UI Hooks

Inject custom UI into existing panels:

```python
import lichtfeld as lf

@lf.ui.hook("training", "status")
def add_custom_info(ui):
    ui.separator()
    ui.label("Custom training info")

def my_hook(ui):
    ui.label("Hook content")

lf.ui.add_hook("rendering", "selection_groups", my_hook, position="append")
lf.ui.remove_hook("rendering", "selection_groups", my_hook)
lf.ui.clear_hooks("rendering", "selection_groups")
lf.ui.clear_hooks("rendering")
lf.ui.clear_all_hooks()

points = lf.ui.get_hook_points()
```

---

## File Dialogs

```python
folder = lf.ui.open_folder_dialog("Select Output Folder", "/tmp")
image = lf.ui.open_image_dialog("/images")
ply = lf.ui.open_ply_file_dialog("/data")
mesh = lf.ui.open_mesh_file_dialog("/data")
ckpt = lf.ui.open_checkpoint_file_dialog()
json_path = lf.ui.open_json_file_dialog()
video = lf.ui.open_video_file_dialog()

save_json = lf.ui.save_json_file_dialog("config.json")
save_ply = lf.ui.save_ply_file_dialog("export.ply")
save_sog = lf.ui.save_sog_file_dialog("export.sog")
save_spz = lf.ui.save_spz_file_dialog("export.spz")
save_html = lf.ui.save_html_file_dialog("viewer.html")
```

All dialog APIs return an empty string when cancelled.

---

## Theme Access

```python
theme = lf.ui.theme()

# Palette values are RGBA tuples
_ = theme.palette.background
_ = theme.palette.surface
_ = theme.palette.primary
_ = theme.palette.secondary
_ = theme.palette.text
_ = theme.palette.warning

# Size values
_ = theme.sizes.window_rounding
_ = theme.sizes.frame_rounding
_ = theme.sizes.window_padding
_ = theme.sizes.item_spacing

# Runtime control
lf.ui.set_theme("dark")   # also: "light"
name = lf.ui.get_theme()
```

---

## Tool Switching

```python
lf.ui.set_tool("selection")
lf.ui.set_tool("translate")
lf.ui.set_tool("rotate")
lf.ui.set_tool("scale")
lf.ui.set_tool("mirror")
lf.ui.set_tool("brush")
lf.ui.set_tool("align")
lf.ui.set_tool("cropbox")
lf.ui.set_tool("none")

# Lower-level operator/tool state
lf.ui.set_active_tool("builtin.select")
active = lf.ui.get_active_tool()
```

---

## Property Widgets

Auto-generate widgets from registered property metadata:

```python
def draw(self, ui):
    changed, value = ui.prop(params, "learning_rate")
    changed, value = ui.prop(params, "iterations", text="Max Iterations")

    if ui.prop_enum(params, "strategy", "default", "Default"):
        pass
```

---

## Complete Example

```python
import lichtfeld as lf
from lfs_plugins.types import Panel

class GaussianFilterPanel(Panel):
    idname = "example.gaussian_filter"
    label = "Gaussian Filter"
    space = "SIDE_PANEL"
    order = 50

    def __init__(self):
        self.opacity_threshold = 0.01
        self.preview = True

    @classmethod
    def poll(cls, ctx):
        return ctx.has_scene and ctx.num_gaussians > 0

    def draw(self, ui):
        ui.heading("Filter Settings")
        _, self.preview = ui.checkbox("Preview", self.preview)
        _, self.opacity_threshold = ui.slider_float("Min Opacity", self.opacity_threshold, 0.0, 1.0)

        if ui.button_styled("Apply", "primary", (120, 28)):
            self.apply_filter()

    def apply_filter(self):
        scene = lf.get_scene()
        if not scene:
            return

        model = scene.training_model()
        if not model:
            return

        opacity = model.get_opacity().squeeze()
        mask = opacity < self.opacity_threshold
        model.soft_delete(mask)
        scene.notify_changed()

lf.register_class(GaussianFilterPanel)
```

---

## Widget Reference

| Category | Widgets |
|----------|---------|
| **Text** | `label`, `label_centered`, `heading`, `text_colored`, `text_wrapped`, `text_disabled`, `bullet_text`, `text_selectable` |
| **Buttons** | `button`, `button_styled`, `button_callback`, `small_button`, `invisible_button` |
| **Toggle** | `checkbox`, `radio_button`, `selectable` |
| **Input** | `input_text`, `input_text_with_hint`, `input_text_enter`, `input_float`, `input_int`, `input_int_formatted`, `stepper_float`, `path_input` |
| **Sliders/Drags** | `slider_float`, `slider_int`, `slider_float2`, `slider_float3`, `drag_float`, `drag_int` |
| **Color** | `color_edit3`, `color_edit4`, `color_button` |
| **Selection** | `combo`, `listbox` |
| **Layout** | `separator`, `spacing`, `same_line`, `new_line`, `indent`, `unindent`, `set_next_item_width`, `begin_group`, `end_group` |
| **Hierarchy** | `collapsing_header`, `tree_node`, `tree_node_ex`, `tree_pop` |
| **Tables** | `begin_table`, `table_setup_column`, `table_headers_row`, `table_next_row`, `table_next_column`, `table_set_column_index`, `end_table` |
| **State/Tooltips** | `progress_bar`, `set_tooltip`, `is_item_hovered`, `is_item_clicked`, `is_item_active` |
| **Style/State** | `push_style_var`, `push_style_color`, `begin_disabled`, `end_disabled`, `push_id`, `pop_id` |

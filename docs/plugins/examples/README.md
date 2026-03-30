# Plugin Examples

Start with these in order:

| Step | File | What it teaches |
|---|---|---|
| 1 | [`01_draw_only.py`](01_draw_only.py) | The smallest useful plugin panel: one class and `draw(ui)` |
| 2 | [`02_status_bar_mixed.py`](02_status_bar_mixed.py) | Add `style`, `height_mode`, `update_interval_ms`, `space = lf.ui.PanelSpace.STATUS_BAR`, and `on_update()` without rewriting `draw(ui)` |
| 3 | [`03_hybrid_plugin/`](03_hybrid_plugin/) | Full retained/hybrid panel with template, RCSS, data model binding, DOM hooks, `on_unmount()`, `on_scene_changed()`, and embedded `draw(ui)` |

`lf.plugins.create("name")` now scaffolds the step-1 panel plus a ready-made `main_panel.rml` / `main_panel.rcss` shell. You can ignore those files until you move to step 3. The CLI command `LichtFeld-Studio plugin create <name>` follows the same UI progression and only adds development helpers such as `.venv` and `.vscode`.

After that, use the focused examples by topic:

| Example | Focus |
|---|---|
| [`hello_world.py`](hello_world.py) | Single-file minimal panel reference |
| [`layout_basics.py`](layout_basics.py) | Rows, columns, boxes, splits, and grid layout |
| [`nested_layout.py`](nested_layout.py) | Deeper layout composition patterns |
| [`grid_enum_demo.py`](grid_enum_demo.py) | Grid flow and enum-style controls |
| [`custom_panel.py`](custom_panel.py) | Widget survey and property-driven UI |
| [`scene_tools.py`](scene_tools.py) | Panels plus properties and scene operators |
| [`modal_operator.py`](modal_operator.py) | Interactive modal operators and event handling |
| [`training_monitor.py`](training_monitor.py) | App state, signals, training callbacks, and reactive UI |
| [`full_plugin/`](full_plugin/) | Multi-file plugin structure with panels, operators, overlays, tools, and capabilities |
| [`scrub_controls_demo/`](scrub_controls_demo/) | Full plugin package demonstrating `lfs_plugins.ScrubFieldController` |

Recommended reading order:

1. Build confidence with `01_draw_only.py`.
2. Learn the unified upgrade path with `02_status_bar_mixed.py`.
3. Study `03_hybrid_plugin/` when you need a custom template or retained DOM.
4. Use the remaining examples as feature references rather than as the first thing you read.

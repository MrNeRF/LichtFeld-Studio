# Layout Composition Patterns

Sub-layouts structure UI elements within Rml im-mode panels using composable
containers. Each container is a context manager that automatically positions
widgets.

## Containers

### Row

Places children horizontally with `horizontal layout` between them.

```python
with layout.row() as row:
    row.button("A")
    row.button("B")
    row.button("C")
```

### Column

Standard vertical stacking (default vertical layout behavior). Useful for applying state to a group of widgets.

```python
with layout.column() as col:
    col.enabled = False
    col.label("All children disabled")
    col.button("Can't click")
```

### Split

Two-column layout. The `factor` argument sets the width ratio of the first
child; the second receives the remainder. The 4dp inter-column gap is removed
from the available width before RmlUi shrinks both percentage bases, preserving
the requested ratio. A split supports two children; extra children are hidden
and produce a one-shot warning.

```python
with layout.split(0.3) as split:
    split.label("Label")        # 30% width
    split.prop(self, "value")   # 70% width
```

### Box

Bordered container with theme-aware background.

```python
with layout.box() as box:
    box.heading("Section")
    box.prop(self, "setting")
```

### GridFlow

Responsive grid. With `even_columns=True`, a positive `columns` value assigns
each child `100 / columns` percent of the row. With `columns=0`, children use a
100dp basis and wrap according to the available width. With
`even_columns=False`, children use their automatic content width.

`even_rows=True` lets cells grow and stretch to the row height;
`even_rows=False` keeps their natural height.

```python
with layout.grid_flow(columns=3) as grid:
    for item in items:
        grid.button(item.name)
```

## Stable Table Rows

Rml im-mode table rows use position identity unless the caller supplies a
stable id. For rows that can be reordered or removed, push an id after
`begin_table()` and before `table_next_row()`, then keep it active while
drawing that row:

```python
if layout.begin_table("jobs", 2):
    for job in jobs:
        layout.push_id(f"##{job.id}")
        layout.table_next_row()
        layout.table_next_column()
        layout.label(job.name)
        layout.table_next_column()
        layout.input_text("Status", job.status)
        layout.pop_id()
    layout.end_table()
```

The hidden `##key` portion is used as the stable token. Without `push_id()`,
row identity is the bare row index, so insertion, removal, or reorder transfers
cell state by position.

## Nesting

Containers nest arbitrarily. Create sub-layouts from sub-layouts:

```python
with layout.box() as box:
    box.heading("Outer")
    with box.row() as row:
        row.button("A")
        with row.column() as col:
            col.label("Nested")
            col.button("B")
```

## State Cascading

State properties cascade from parent to child sub-layouts:

- `enabled` — ANDed: if parent is disabled, children are disabled
- `active` — ANDed: same semantics
- `alert` — one-shot: highlights the next widget with error styling

```python
with layout.column() as outer:
    outer.enabled = False       # everything below is disabled
    outer.prop(self, "a")

    with outer.row() as row:    # inherits disabled state
        row.button("X")        # disabled
        row.button("Y")        # disabled
```

## prop_enum

Toggle buttons that set a string property value. Selected state is styled with the primary theme color.

```python
with layout.row() as row:
    row.prop_enum(self, "mode", "fast", "Fast")
    row.prop_enum(self, "mode", "balanced", "Balanced")
    row.prop_enum(self, "mode", "quality", "Quality")
```

Also available directly on `layout`:

```python
layout.prop_enum(self, "mode", "fast", "Fast")
```

## Alert

Persistent styling that highlights all widgets while active. Cascades to child sub-layouts:

```python
with layout.column() as col:
    col.alert = value > threshold
    col.prop(self, "value")         # red if alert is True
    col.prop(self, "other_value")   # also red (alert persists)
    with col.row() as row:
        row.label("Child")          # also red (inherited from parent)
```

## Method Delegation

`SubLayout` exposes all `UILayout` widget methods. The ~60 most common are explicitly bound for performance. All others delegate via `__getattr__`:

```python
with layout.row() as row:
    row.same_line()         # delegated to UILayout
    row.begin_group()       # delegated to UILayout
    row.end_group()         # delegated to UILayout
```

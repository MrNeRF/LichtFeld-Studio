"""Panel demonstrating all common widget types.

Shows text, buttons, inputs, sliders, checkboxes, combos, colors,
tables, collapsing headers, and property binding.
"""

import lichtfeld as lf
from lfs_plugins.types import Panel
from lfs_plugins.props import (
    PropertyGroup,
    FloatProperty,
    IntProperty,
    BoolProperty,
    StringProperty,
    EnumProperty,
    FloatVectorProperty,
)


class DemoSettings(PropertyGroup):
    opacity = FloatProperty(default=0.8, min=0.0, max=1.0, name="Opacity")
    iterations = IntProperty(default=30000, min=1000, max=100000, name="Iterations")
    enabled = BoolProperty(default=True, name="Enabled")
    label_text = StringProperty(default="My Label", maxlen=64, name="Label")
    mode = EnumProperty(
        items=[
            ("fast", "Fast", "Quick but lower quality"),
            ("balanced", "Balanced", "Good tradeoff"),
            ("quality", "Quality", "Best quality"),
        ],
        name="Mode",
    )
    color = FloatVectorProperty(
        default=(0.2, 0.5, 1.0), size=3, subtype="COLOR", name="Color"
    )


class DemoPanel(Panel):
    label = "Widget Demo"
    space = "SIDE_PANEL"
    order = 150

    def __init__(self):
        self.settings = DemoSettings.get_instance()
        self.counter = 0
        self.selected_tab = 0
        self.items = ["Alpha", "Beta", "Gamma", "Delta"]
        self.selected_item = 0
        self.search_text = ""

    def draw(self, layout):
        # --- Text widgets ---
        if layout.collapsing_header("Text Widgets", default_open=True):
            layout.heading("Heading")
            layout.label("Normal label")
            layout.text_wrapped(
                "This is wrapped text that will flow to multiple lines "
                "when the panel is narrow enough."
            )
            layout.text_colored("Colored text", (1.0, 0.3, 0.3, 1.0))
            layout.text_disabled("Disabled text")
            layout.bullet_text("Bullet point")

        # --- Buttons ---
        if layout.collapsing_header("Buttons", default_open=True):
            if layout.button("Standard Button", (-1, 0)):
                self.counter += 1
            layout.label(f"Clicked {self.counter} times")

            with layout.row() as row:
                row.button_styled("Success", "success", (100, 0))
                row.button_styled("Error", "error", (100, 0))

            if layout.small_button("Small"):
                lf.log.info("Small button clicked")

        # --- Input widgets ---
        if layout.collapsing_header("Input Widgets", default_open=True):
            # Property binding (auto-generates correct widget)
            layout.prop(self.settings, "opacity")
            layout.prop(self.settings, "iterations")
            layout.prop(self.settings, "enabled")
            layout.prop(self.settings, "label_text")
            layout.prop(self.settings, "mode")
            layout.prop(self.settings, "color")

        # --- Manual widgets ---
        if layout.collapsing_header("Manual Widgets", default_open=False):
            changed, self.search_text = layout.input_text_with_hint(
                "##search", "Search...", self.search_text
            )

            changed, self.selected_item = layout.combo(
                "Select Item", self.selected_item, self.items
            )

            layout.separator()
            layout.progress_bar(0.65, "65%")

        # --- Table ---
        if layout.collapsing_header("Table", default_open=False):
            if layout.begin_table("demo_table", 3):
                layout.table_setup_column("Name")
                layout.table_setup_column("Value")
                layout.table_setup_column("Status")
                layout.table_headers_row()

                for i, item in enumerate(self.items):
                    layout.table_next_row()
                    layout.table_next_column()
                    layout.label(item)
                    layout.table_next_column()
                    layout.label(str(i * 10))
                    layout.table_next_column()
                    layout.text_colored(
                        "Active" if i % 2 == 0 else "Idle",
                        (0.3, 1.0, 0.3, 1.0) if i % 2 == 0 else (0.6, 0.6, 0.6, 1.0),
                    )
                layout.end_table()

        # --- Disabled region ---
        if layout.collapsing_header("Conditional UI", default_open=False):
            with layout.column() as col:
                col.enabled = self.settings.enabled
                col.label("This section is disabled when 'Enabled' is unchecked")
                col.button("Disabled Button")


_classes = [DemoPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)

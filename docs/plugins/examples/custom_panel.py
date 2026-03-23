"""Panel demonstrating all common widget types.

Shows text, buttons, inputs, sliders, checkboxes, combos, colors,
tables, collapsing headers, and property binding.
"""

import lichtfeld as lf
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


class DemoPanel(lf.ui.Panel):
    label = "Widget Demo"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 150

    def __init__(self):
        self.settings = DemoSettings.get_instance()
        self.counter = 0
        self.selected_tab = 0
        self.items = ["Alpha", "Beta", "Gamma", "Delta"]
        self.selected_item = 0
        self.search_text = ""

    def draw(self, ui):
        # --- Text widgets ---
        if ui.collapsing_header("Text Widgets", default_open=True):
            ui.heading("Heading")
            ui.label("Normal label")
            ui.text_wrapped(
                "This is wrapped text that will flow to multiple lines "
                "when the panel is narrow enough."
            )
            ui.text_colored("Colored text", (1.0, 0.3, 0.3, 1.0))
            ui.text_disabled("Disabled text")
            ui.bullet_text("Bullet point")

        # --- Buttons ---
        if ui.collapsing_header("Buttons", default_open=True):
            if ui.button("Standard Button", (-1, 0)):
                self.counter += 1
            ui.label(f"Clicked {self.counter} times")

            with ui.row() as row:
                row.button_styled("Success", "success", (100, 0))
                row.button_styled("Error", "error", (100, 0))

            if ui.small_button("Small"):
                lf.log.info("Small button clicked")

        # --- Input widgets ---
        if ui.collapsing_header("Input Widgets", default_open=True):
            # Property binding (auto-generates correct widget)
            ui.prop(self.settings, "opacity")
            ui.prop(self.settings, "iterations")
            ui.prop(self.settings, "enabled")
            ui.prop(self.settings, "label_text")
            ui.prop(self.settings, "mode")
            ui.prop(self.settings, "color")

        # --- Manual widgets ---
        if ui.collapsing_header("Manual Widgets", default_open=False):
            changed, self.search_text = ui.input_text_with_hint(
                "##search", "Search...", self.search_text
            )

            changed, self.selected_item = ui.combo(
                "Select Item", self.selected_item, self.items
            )

            ui.separator()
            ui.progress_bar(0.65, "65%")

        # --- Table ---
        if ui.collapsing_header("Table", default_open=False):
            if ui.begin_table("demo_table", 3):
                ui.table_setup_column("Name")
                ui.table_setup_column("Value")
                ui.table_setup_column("Status")
                ui.table_headers_row()

                for i, item in enumerate(self.items):
                    ui.table_next_row()
                    ui.table_next_column()
                    ui.label(item)
                    ui.table_next_column()
                    ui.label(str(i * 10))
                    ui.table_next_column()
                    ui.text_colored(
                        "Active" if i % 2 == 0 else "Idle",
                        (0.3, 1.0, 0.3, 1.0) if i % 2 == 0 else (0.6, 0.6, 0.6, 1.0),
                    )
                ui.end_table()

        # --- Disabled region ---
        if ui.collapsing_header("Conditional UI", default_open=False):
            with ui.column() as col:
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

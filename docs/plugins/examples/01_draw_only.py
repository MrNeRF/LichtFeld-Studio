"""Step 1: pure draw(ui) panel.

This is the smallest useful plugin panel using the unified Panel API.
"""

import lichtfeld as lf


class DrawOnlyPanel(lf.ui.Panel):
    id = "docs.draw_only_panel"
    label = "Draw Only"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 200

    def __init__(self):
        self._name = "Aurora Bloom"
        self._enabled = True
        self._strength = 0.45
        self._clicks = 0

    def draw(self, ui):
        ui.heading("1. Pure draw(ui)")
        ui.text_disabled("No template, no style, no retained hooks.")

        _, self._name = ui.input_text_with_hint("Preset", "Name this look", self._name)
        _, self._enabled = ui.checkbox("Enabled", self._enabled)
        _, self._strength = ui.slider_float("Strength", self._strength, 0.0, 1.0)
        ui.progress_bar(self._strength, f"{int(self._strength * 100)}%")

        if ui.button_styled(f"Preview ({self._clicks})", "primary"):
            self._clicks += 1
        ui.same_line()
        if ui.button_styled("Reset", "secondary"):
            self._name = "Aurora Bloom"
            self._enabled = True
            self._strength = 0.45
            self._clicks = 0

        ui.separator()
        ui.bullet_text("State lives directly on the panel instance.")
        ui.bullet_text("This is the right starting point for most plugins.")


_classes = [DrawOnlyPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)

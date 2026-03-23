"""Step 2: draw(ui) plus retained shell features.

This example keeps draw(ui) as the content source while adding:
- space = lf.ui.PanelSpace.STATUS_BAR
- style
- height_mode
- update_interval_ms
- on_update()
"""

import lichtfeld as lf


class StatusBarMixedPanel(lf.ui.Panel):
    id = "docs.status_bar_mixed"
    label = "Status Bar Mixed"
    space = lf.ui.PanelSpace.STATUS_BAR
    order = 50
    height_mode = lf.ui.PanelHeightMode.CONTENT
    update_interval_ms = 140
    style = """
body.status-bar-panel {
    padding: 0 12dp;
}

#im-root .im-line {
    align-items: center;
    gap: 8dp;
}

#im-root .im-label {
    color: #f3c96d;
    font-size: 10dp;
    font-weight: bold;
}
"""

    def __init__(self):
        self._progress = 0.22
        self._beats = 0

    def draw(self, ui):
        ui.label("BUILD UP")
        ui.progress_bar(self._progress, f"{int(self._progress * 100)}%")
        ui.same_line()
        ui.text_disabled(f"tick {self._beats:03d}")

    def on_update(self, doc):
        del doc
        self._beats += 1
        self._progress = (self._progress + 0.02) % 1.0
        return True


_classes = [StatusBarMixedPanel]


def on_load():
    for cls in _classes:
        lf.register_class(cls)


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)

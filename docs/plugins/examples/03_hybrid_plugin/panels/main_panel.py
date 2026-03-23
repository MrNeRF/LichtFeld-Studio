"""Step 3: full hybrid panel example."""

from pathlib import Path
import time

import lichtfeld as lf

MODEL_NAME = "hybrid_panel_demo"


class HybridPanel(lf.ui.Panel):
    id = "docs.hybrid_panel"
    label = "Hybrid Panel"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 220
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT
    update_interval_ms = 200

    def __init__(self):
        self._handle = None
        self._mode = "STORY"
        self._project_name = "Signal Bloom"
        self._focus = 0.55
        self._scene_changes = 0
        self._clicks = 0

    def draw(self, ui):
        ui.text_disabled("This immediate block renders into #im-root.")
        _, self._focus = ui.slider_float("Focus", self._focus, 0.0, 1.0)
        if ui.button_styled(f"Ping ({self._clicks})", "primary"):
            self._clicks += 1
            self._dirty("summary")

    def on_bind_model(self, ctx):
        model = ctx.create_data_model(MODEL_NAME)
        if model is None:
            return

        model.bind("project_name", lambda: self._project_name, self._set_project_name)
        model.bind_func("mode", lambda: self._mode)
        model.bind_func("clock", lambda: time.strftime("%H:%M:%S"))
        model.bind_func("scene_changes", lambda: str(self._scene_changes))
        model.bind_func("summary", self._summary)
        model.bind_event("cycle_mode", self._on_cycle_mode)

        self._handle = model.get_handle()

    def on_mount(self, doc):
        info = doc.get_element_by_id("info-pill")
        if info:
            info.add_event_listener("click", lambda _ev: lf.log.info("Hybrid panel info clicked"))

    def on_unmount(self, doc):
        doc.remove_data_model(MODEL_NAME)
        self._handle = None

    def on_update(self, doc):
        del doc
        self._dirty("clock")
        return True

    def on_scene_changed(self, doc):
        del doc
        self._scene_changes += 1
        self._dirty("scene_changes")
        self._dirty("summary")

    def _set_project_name(self, value):
        self._project_name = str(value)
        self._dirty("project_name")
        self._dirty("summary")

    def _summary(self):
        return (
            f"mode={self._mode.lower()} | "
            f"project={self._project_name} | "
            f"focus={self._focus:.2f} | "
            f"clicks={self._clicks}"
        )

    def _on_cycle_mode(self, _handle, _event, _args):
        self._mode = "REVIEW" if self._mode == "STORY" else "STORY"
        self._dirty("mode")
        self._dirty("summary")

    def _dirty(self, name):
        if self._handle:
            self._handle.dirty(name)

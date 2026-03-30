"""Scrub control demo panel."""

from pathlib import Path

import lichtfeld as lf
try:
    from lfs_plugins import ScrubFieldController, ScrubFieldSpec
except ImportError:
    from lfs_plugins.scrub_fields import ScrubFieldController, ScrubFieldSpec


DATA_MODEL = "scrub_controls_demo"

SCRUB_FIELDS = {
    "strength": ScrubFieldSpec(0.0, 2.0, 0.01, "%.2f"),
    "quality": ScrubFieldSpec(0.0, 1.0, 0.01, "%.2f"),
    "threshold": ScrubFieldSpec(0.0, 1.0, 0.01, "%.2f"),
}


class ScrubControlsDemoPanel(lf.ui.Panel):
    id = "docs.scrub_controls_demo.panel"
    label = "Scrub Controls Demo"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 500
    template = str(Path(__file__).resolve().with_name("main_panel.rml"))
    height_mode = lf.ui.PanelHeightMode.CONTENT

    def __init__(self):
        self._strength = 0.65
        self._quality = 0.35
        self._threshold = 0.2
        self._handle = None
        self._scrub_fields = ScrubFieldController(
            SCRUB_FIELDS,
            self._get_scrub_value,
            self._set_scrub_value,
        )

    def on_bind_model(self, ctx):
        model = ctx.create_data_model(DATA_MODEL)
        if model is None:
            return

        model.bind(
            "strength",
            lambda: self._fmt(self._strength),
            lambda value: self._set_scrub_value("strength", value),
        )
        model.bind(
            "quality",
            lambda: self._fmt(self._quality),
            lambda value: self._set_scrub_value("quality", value),
        )
        model.bind(
            "threshold",
            lambda: self._fmt(self._threshold),
            lambda value: self._set_scrub_value("threshold", value),
        )
        model.bind_func("summary", self._summary_text)
        model.bind_event("do_reset", self._on_reset)

        self._handle = model.get_handle()
        self._dirty_fields("strength", "quality", "threshold", "summary")

    def on_mount(self, doc):
        self._scrub_fields.mount(doc)

    def on_update(self, doc):
        del doc
        return self._scrub_fields.sync_all()

    def on_unmount(self, doc):
        doc.remove_data_model(DATA_MODEL)
        self._handle = None
        self._scrub_fields.unmount()

    def _get_scrub_value(self, prop: str) -> float:
        if prop == "strength":
            return float(self._strength)
        if prop == "quality":
            return float(self._quality)
        return float(self._threshold)

    def _set_scrub_value(self, prop: str, value: float) -> None:
        value = float(value)
        if prop == "strength":
            self._strength = max(0.0, min(2.0, value))
        elif prop == "quality":
            self._quality = max(0.0, min(1.0, value))
        elif prop == "threshold":
            self._threshold = max(0.0, min(1.0, value))

        self._dirty_fields(prop, "summary")

    def _on_reset(self, *_):
        self._strength = 0.65
        self._quality = 0.35
        self._threshold = 0.20
        self._dirty_fields("strength", "quality", "threshold", "summary")
        self._scrub_fields.sync_all()

    def _dirty_fields(self, *names: str) -> None:
        if not self._handle:
            return
        for name in names:
            self._handle.dirty(name)

    def _summary_text(self) -> str:
        return (
            f"strength={self._fmt(self._strength)}, "
            f"quality={self._fmt(self._quality)}, "
            f"threshold={self._fmt(self._threshold)}"
        )

    @staticmethod
    def _fmt(value: float) -> str:
        return f"{float(value):.2f}"

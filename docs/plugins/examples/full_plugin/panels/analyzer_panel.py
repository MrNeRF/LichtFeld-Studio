"""Main analyzer panel with property inspection and filtering."""

import lichtfeld as lf
from lfs_plugins.props import PropertyGroup, FloatProperty, EnumProperty, BoolProperty
from lfs_plugins.ui.state import AppState
from lfs_plugins.ui.signals import Signal


class AnalyzerSettings(PropertyGroup):
    property_name = EnumProperty(
        items=[
            ("opacity", "Opacity", "Filter by opacity"),
            ("scale", "Scale", "Filter by average scale"),
        ],
        name="Property",
    )
    threshold = FloatProperty(default=0.1, min=0.0, max=10.0, name="Threshold")
    auto_update = BoolProperty(default=False, name="Auto-update")


class AnalyzerPanel(lf.ui.Panel):
    label = "Gaussian Analyzer"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 45

    def __init__(self):
        self.settings = AnalyzerSettings.get_instance()
        self.result_count = Signal(0, name="result_count")
        self.result_total = Signal(0, name="result_total")

        if self.settings.auto_update:
            AppState.scene_generation.subscribe_as(
                "gaussian_analyzer", lambda _: self._run_analysis()
            )

    def _run_analysis(self):
        from lfs_plugins.capabilities import CapabilityRegistry

        result = CapabilityRegistry.instance().invoke(
            "gaussian_analyzer.analyze",
            {
                "property": self.settings.property_name,
                "threshold": self.settings.threshold,
            },
        )
        if result.get("success"):
            self.result_count.value = result["count"]
            self.result_total.value = result["total"]

    @classmethod
    def poll(cls, context) -> bool:
        return AppState.has_scene.value

    def draw(self, ui):
        ui.heading("Gaussian Analyzer")

        ui.prop(self.settings, "property_name")
        ui.prop(self.settings, "threshold")
        ui.prop(self.settings, "auto_update")

        ui.separator()

        if ui.button("Analyze", (-1, 0)):
            self._run_analysis()

        total = self.result_total.value
        if total > 0:
            count = self.result_count.value
            pct = count / total * 100
            ui.label(f"Below threshold: {count:,} / {total:,} ({pct:.1f}%)")
            ui.progress_bar(count / total)

        ui.separator()

        with ui.row() as row:
            if row.button("Select Filtered"):
                self._select_filtered()
            if row.button_styled("Delete Filtered", "error"):
                self._delete_filtered()

    def _select_filtered(self):
        scene = lf.get_scene()
        if scene is None:
            return
        model = scene.combined_model()
        if model is None:
            return

        data = self._get_property_data(model)
        if data is None:
            return

        mask = data < self.settings.threshold
        scene.set_selection_mask(mask)
        lf.log.info(f"Selected {int(mask.sum().item()):,} gaussians")

    def _delete_filtered(self):
        scene = lf.get_scene()
        if scene is None:
            return
        model = scene.combined_model()
        if model is None:
            return

        data = self._get_property_data(model)
        if data is None:
            return

        mask = data < self.settings.threshold
        count = int(mask.sum().item())
        model.soft_delete(mask)
        removed = model.apply_deleted()
        lf.log.info(f"Deleted {removed} gaussians")

    def _get_property_data(self, model):
        prop = self.settings.property_name
        if prop == "opacity":
            return model.get_opacity().squeeze()
        elif prop == "scale":
            return model.get_scaling().mean(dim=1)
        return None

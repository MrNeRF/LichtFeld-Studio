"""Scene graph manipulation examples.

Demonstrates node operations, transforms, selection, and splat data access.
"""

import lichtfeld as lf
from lfs_plugins.types import Operator
from lfs_plugins.props import FloatProperty, FloatVectorProperty, StringProperty


class DuplicateNodeOp(Operator):
    label = "Duplicate Selected"
    description = "Duplicate the selected node"

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_selection()

    def execute(self, context) -> set:
        names = lf.get_selected_node_names()
        for name in names:
            new_name = lf.get_scene().duplicate_node(name)
            lf.log.info(f"Duplicated '{name}' -> '{new_name}'")
        return {"FINISHED"}


class TranslateNodeOp(Operator):
    label = "Translate Node"
    description = "Move a node by offset"

    offset: tuple = FloatVectorProperty(
        default=(0.0, 1.0, 0.0), size=3, subtype="TRANSLATION",
        name="Offset",
    )

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_selection() and lf.can_transform_selection()

    def execute(self, context) -> set:
        names = lf.get_selected_node_names()
        for name in names:
            matrix = lf.get_node_transform(name)
            if matrix is None:
                continue
            components = lf.decompose_transform(matrix)
            t = components["translation"]
            new_t = [t[0] + self.offset[0], t[1] + self.offset[1], t[2] + self.offset[2]]
            new_matrix = lf.compose_transform(
                new_t, components["euler"], components["scale"]
            )
            lf.set_node_transform(name, new_matrix)
            lf.log.info(f"Translated '{name}' by {self.offset}")
        return {"FINISHED"}


class CreateGroupOp(Operator):
    label = "Create Group"
    description = "Create a new group node and parent selected nodes to it"

    group_name: str = StringProperty(default="New Group", name="Group Name")

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()

    def execute(self, context) -> set:
        scene = lf.get_scene()
        group_id = scene.add_group(self.group_name)

        selected = lf.get_selected_node_names()
        for name in selected:
            node = scene.get_node(name)
            if node:
                scene.reparent(node.id, group_id)

        lf.log.info(f"Created group '{self.group_name}' with {len(selected)} children")
        return {"FINISHED"}


class SplatInfoOp(Operator):
    label = "Splat Info"
    description = "Log information about the scene's gaussian splat data"

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()

    def execute(self, context) -> set:
        scene = lf.get_scene()
        model = scene.combined_model()
        if model is None:
            lf.log.warn("No splat data in scene")
            return {"CANCELLED"}

        lf.log.info(f"Gaussians: {model.num_points:,}")
        lf.log.info(f"SH degree: {model.active_sh_degree} (max {model.max_sh_degree})")
        lf.log.info(f"Scene scale: {model.scene_scale:.4f}")

        means = model.get_means()
        lf.log.info(f"Means shape: {means.shape}, device: {means.device}")

        bbox_min = means.min(dim=0)
        bbox_max = means.max(dim=0)
        lf.log.info(f"Bounding box: min={bbox_min.cpu().numpy()}, max={bbox_max.cpu().numpy()}")

        opacity = model.get_opacity()
        avg_opacity = opacity.mean().item()
        lf.log.info(f"Average opacity: {avg_opacity:.4f}")

        return {"FINISHED"}


class FilterByOpacity(Operator):
    label = "Filter by Opacity"
    description = "Select gaussians below an opacity threshold"

    threshold: float = FloatProperty(default=0.1, min=0.0, max=1.0, name="Threshold")

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()

    def execute(self, context) -> set:
        scene = lf.get_scene()
        model = scene.combined_model()
        if model is None:
            return {"CANCELLED"}

        opacity = model.get_opacity().squeeze()
        mask = opacity < self.threshold
        count = mask.sum().item()

        scene.set_selection_mask(mask)
        lf.log.info(f"Selected {int(count):,} gaussians with opacity < {self.threshold}")
        return {"FINISHED"}


class SceneToolsPanel(lf.ui.Panel):
    label = "Scene Tools"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 120

    def __init__(self):
        self.filter_threshold = 0.1

    @classmethod
    def poll(cls, context) -> bool:
        return lf.has_scene()

    def draw(self, ui):
        scene = lf.get_scene()
        ui.heading("Scene Tools")

        # Scene info
        ui.label(f"Nodes: {len(scene.get_nodes())}")
        ui.label(f"Gaussians: {scene.total_gaussian_count:,}")

        ui.separator()

        # Node operations
        if ui.collapsing_header("Node Operations", default_open=True):
            has_sel = lf.has_selection()

            ui.begin_disabled(not has_sel)
            if ui.button("Duplicate Selected", (-1, 0)):
                lf.ui.ops.invoke(DuplicateNodeOp._class_id())
            ui.end_disabled()

            if ui.button("Create Group", (-1, 0)):
                lf.ui.ops.invoke(CreateGroupOp._class_id())

            if ui.button("Print Splat Info", (-1, 0)):
                lf.ui.ops.invoke(SplatInfoOp._class_id())

        # Selection tools
        if ui.collapsing_header("Selection", default_open=True):
            changed, self.filter_threshold = ui.slider_float(
                "Opacity Threshold##filter", self.filter_threshold, 0.0, 1.0
            )
            if ui.button("Select Low Opacity", (-1, 0)):
                op = FilterByOpacity()
                op.threshold = self.filter_threshold
                op.execute(None)

            if ui.button("Clear Selection", (-1, 0)):
                scene.clear_selection()
                lf.deselect_all()


_classes = [
    DuplicateNodeOp, TranslateNodeOp, CreateGroupOp,
    SplatInfoOp, FilterByOpacity, SceneToolsPanel,
]


def on_load():
    for cls in _classes:
        lf.register_class(cls)


def on_unload():
    for cls in reversed(_classes):
        lf.unregister_class(cls)

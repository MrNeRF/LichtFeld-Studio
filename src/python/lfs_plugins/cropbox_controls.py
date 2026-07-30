# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""CropBox panel - controls for editing crop box properties."""

from typing import Optional

import lichtfeld as lf


def _get_selected_cropbox():
    """Get the CropBox data if a cropbox node is selected."""
    selected = lf.get_selected_node_names()
    if not selected or len(selected) != 1:
        return None

    scene = lf.get_scene()
    if not scene:
        return None

    node = scene.get_node(selected[0])
    if not node or node.type != lf.scene.NodeType.CROPBOX:
        return None

    return node.cropbox()


def draw_cropbox_controls(layout):
    """Draw crop box controls if a cropbox is selected."""
    cropbox = _get_selected_cropbox()
    if not cropbox:
        return

    if not layout.collapsing_header("Crop Box", default_open=True):
        return

    layout.prop(cropbox, "enabled")

    layout.separator()
    layout.label("Bounds")

    min_val = list(cropbox.min)
    max_val = list(cropbox.max)

    _, min_val[0] = layout.input_float("Min X", min_val[0], 0.1, 1.0, "%.3f")
    _, min_val[1] = layout.input_float("Min Y", min_val[1], 0.1, 1.0, "%.3f")
    _, min_val[2] = layout.input_float("Min Z", min_val[2], 0.1, 1.0, "%.3f")

    cropbox.min = tuple(min_val)

    layout.spacing()

    _, max_val[0] = layout.input_float("Max X", max_val[0], 0.1, 1.0, "%.3f")
    _, max_val[1] = layout.input_float("Max Y", max_val[1], 0.1, 1.0, "%.3f")
    _, max_val[2] = layout.input_float("Max Z", max_val[2], 0.1, 1.0, "%.3f")

    cropbox.max = tuple(max_val)

    layout.separator()
    layout.prop(cropbox, "inverse")

    params = lf.optimization_params()
    if params and params.has_params():
        layout.separator()
        layout.label("Training")
        layout.prop(params, "cropbox_lr_scale")
        layout.prop(params, "cropbox_loss_weight")

    layout.separator()
    layout.label("Appearance")
    layout.prop(cropbox, "color")
    layout.prop(cropbox, "line_width")

    layout.separator()

    btn_width = layout.get_content_region_avail_x() * 0.5 - 4
    if layout.button("Apply Crop", (btn_width, 0)):
        lf.ui.apply_cropbox()

    layout.same_line()

    if layout.button("Reset", (btn_width, 0)):
        lf.ui.reset_cropbox()

    if layout.button("Fit to Scene", (-1, 0)):
        lf.ui.fit_cropbox_to_scene(False)


def register():
    lf.ui.add_hook("tools", "transform", draw_cropbox_controls, "append")


def unregister():
    lf.ui.remove_hook("tools", "transform", draw_cropbox_controls)

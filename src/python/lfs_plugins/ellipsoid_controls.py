# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Ellipsoid panel - controls for editing ellipsoid properties."""

import lichtfeld as lf


def _get_selected_ellipsoid():
    """Get the Ellipsoid data if an ellipsoid node is selected."""
    selected = lf.get_selected_node_names()
    if not selected or len(selected) != 1:
        return None

    scene = lf.get_scene()
    if not scene:
        return None

    node = scene.get_node(selected[0])
    if not node or node.type != lf.scene.NodeType.ELLIPSOID:
        return None

    return node.ellipsoid()


def draw_ellipsoid_controls(layout):
    """Draw ellipsoid controls if an ellipsoid is selected."""
    ellipsoid = _get_selected_ellipsoid()
    if not ellipsoid:
        return

    if not layout.collapsing_header("Ellipsoid", default_open=True):
        return

    layout.prop(ellipsoid, "enabled")

    layout.separator()
    layout.label("Radii")

    radii = list(ellipsoid.radii)

    _, radii[0] = layout.input_float("X##radii", radii[0], 0.1, 1.0, "%.3f")
    _, radii[1] = layout.input_float("Y##radii", radii[1], 0.1, 1.0, "%.3f")
    _, radii[2] = layout.input_float("Z##radii", radii[2], 0.1, 1.0, "%.3f")

    radii = [max(0.001, r) for r in radii]
    ellipsoid.radii = tuple(radii)

    layout.separator()
    layout.prop(ellipsoid, "inverse")

    layout.separator()
    layout.label("Appearance")
    layout.prop(ellipsoid, "color")
    layout.prop(ellipsoid, "line_width")

    layout.separator()

    btn_width = layout.get_content_region_avail_x() * 0.5 - 4
    if layout.button("Apply", (btn_width, 0)):
        lf.ui.apply_ellipsoid()

    layout.same_line()

    if layout.button("Reset", (btn_width, 0)):
        lf.ui.reset_ellipsoid()

    if layout.button("Fit to Scene", (-1, 0)):
        lf.ui.fit_ellipsoid_to_scene(False)


def register():
    lf.ui.add_hook("tools", "transform", draw_ellipsoid_controls, "append")


def unregister():
    lf.ui.remove_hook("tools", "transform", draw_ellipsoid_controls)

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Shared source discovery for LOD budgets and splat processing tools."""

def active_splat_node():
    import lichtfeld as lf
    scene = getattr(lf, "get_scene", lambda: None)()
    if scene is None:
        return None, "", 0

    selected_name = str(getattr(lf, "get_selected_node_name", lambda: "")() or "")
    if not selected_name:
        return None, "", 0

    node = scene.get_node(selected_name)
    if node is None:
        return None, "", 0

    node_type_enum = getattr(getattr(lf, "scene", None), "NodeType", None)
    if node_type_enum is not None and getattr(node, "type", None) != node_type_enum.SPLAT:
        return None, "", 0

    try:
        splat = node.splat_data()
    except Exception:
        splat = None
    if splat is None:
        return None, "", 0

    try:
        count = int(splat.visible_count())
    except Exception:
        count = int(getattr(node, "gaussian_count", 0))
    return node, selected_name, count

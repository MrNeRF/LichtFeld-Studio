# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene Graph Panel - virtualized RmlUI tree backed by a data model."""

import math

import lichtfeld as lf

from .types import RmlPanel
from .ui.state import AppState

TREE_ROW_HEIGHT_DP = 20
TREE_HEADER_HEIGHT_DP = 24
TREE_OVERSCAN_ROWS = 10
SCENE_MODEL_NAME = "scene_panel"

SCENE_MUTATION_NODE_ADDED = 1 << 0
SCENE_MUTATION_NODE_REMOVED = 1 << 1
SCENE_MUTATION_NODE_RENAMED = 1 << 2
SCENE_MUTATION_NODE_REPARENTED = 1 << 3
SCENE_MUTATION_TRANSFORM_CHANGED = 1 << 4
SCENE_MUTATION_VISIBILITY_CHANGED = 1 << 5
SCENE_MUTATION_MODEL_CHANGED = 1 << 6
SCENE_MUTATION_SELECTION_CHANGED = 1 << 7
SCENE_MUTATION_CLEARED = 1 << 8

SCENE_STRUCTURAL_MUTATIONS = (
    SCENE_MUTATION_NODE_ADDED |
    SCENE_MUTATION_NODE_REMOVED |
    SCENE_MUTATION_NODE_REPARENTED |
    SCENE_MUTATION_CLEARED
)

NODE_TYPE_ICONS = {
    "SPLAT": "splat",
    "POINTCLOUD": "pointcloud",
    "GROUP": "group",
    "DATASET": "dataset",
    "CAMERA": "camera",
    "CAMERA_GROUP": "camera",
    "CROPBOX": "cropbox",
    "ELLIPSOID": "ellipsoid",
    "MESH": "mesh",
    "KEYFRAME_GROUP": None,
    "KEYFRAME": None,
    "IMAGE_GROUP": None,
    "IMAGE": None,
}

NODE_TYPE_UNICODE = {
    "KEYFRAME_GROUP": "\u25c6",
    "KEYFRAME": "\u25c6",
    "IMAGE_GROUP": "\u25a3",
    "IMAGE": "\u25a3",
}

NODE_TYPE_CSS_CLASS = {
    "SPLAT": "splat",
    "POINTCLOUD": "pointcloud",
    "GROUP": "group",
    "DATASET": "dataset",
    "CAMERA": "camera",
    "CAMERA_GROUP": "camera_group",
    "CROPBOX": "cropbox",
    "ELLIPSOID": "ellipsoid",
    "MESH": "mesh",
    "KEYFRAME_GROUP": "keyframe_group",
    "KEYFRAME": "keyframe",
    "IMAGE_GROUP": "group",
    "IMAGE": "group",
}

NODE_TYPE_ICON_PATH = {
    node_type: f"../icon/scene/{icon_name}.png"
    for node_type, icon_name in NODE_TYPE_ICONS.items()
    if icon_name
}

KI_RETURN = 72
KI_ESCAPE = 81
KI_DELETE = 99
KI_F2 = 108

EASING_TYPES = [
    (0, "scene.keyframe_easing.linear"),
    (1, "scene.keyframe_easing.ease_in"),
    (2, "scene.keyframe_easing.ease_out"),
    (3, "scene.keyframe_easing.ease_in_out"),
]

DRAGGABLE_TYPES = {"SPLAT", "GROUP", "POINTCLOUD", "MESH", "CROPBOX", "ELLIPSOID"}


def tr(key):
    result = lf.ui.tr(key)
    return result if result else key


def _node_type(node):
    return str(node.type).split(".")[-1]


def _is_deletable(node_type, parent_is_dataset):
    return (node_type not in ("CAMERA", "CAMERA_GROUP", "KEYFRAME", "KEYFRAME_GROUP")
            and not parent_is_dataset)


def _can_drag(node_type, parent_is_dataset):
    return node_type in DRAGGABLE_TYPES and not parent_is_dataset


class ScenePanel(RmlPanel):
    idname = "lfs.scene"
    label = "Scene"
    space = "SCENE_HEADER"
    order = 0
    rml_template = "rmlui/scene_tree.rml"
    update_interval_ms = 16

    def __init__(self):
        self.doc = None
        self.container = None
        self.filter_input = None
        self._filter_clear = None
        self._context_menu = None
        self._handle = None

        self._filter_text = ""
        self._node_snapshots = {}
        self._root_ids = []
        self._flat_rows = []
        self._flat_index_by_id = {}
        self._selected_nodes = set()
        self._click_anchor = None
        self._committed_node_order = []
        self._prev_selected = set()
        self._scroll_to_node = None
        self._restore_scroll_top = None
        self._collapsed_ids = set()
        self._rename_node = None
        self._rename_buffer = ""
        self._context_menu_node = None
        self._context_menu_visible = False
        self._context_menu_left = "0px"
        self._context_menu_top = "0px"
        self._drag_source = None
        self._drop_target = None
        self._models_collapsed = False
        self._last_lang = ""
        self._root_count = 0
        self._scene_has_nodes = False
        self._top_spacer_height = "0dp"
        self._bottom_spacer_height = "0dp"
        self._visible_row_capacity = 0
        self._tree_revision = 0
        self._last_render_key = None
        self._last_scroll_top = -1.0
        self._last_view_h = -1.0
        self._last_ui_scale = 0.0
        self._last_invert_masks = False

    def on_bind_model(self, ctx):
        model = ctx.create_data_model(SCENE_MODEL_NAME)
        if model is None:
            return

        model.bind_func("search_placeholder", lambda: tr("scene.search"))
        model.bind_func("show_tree", lambda: self._scene_has_nodes)
        model.bind_func("show_empty_state", lambda: not self._scene_has_nodes)
        model.bind_func("empty_message_primary", lambda: tr("scene.no_data_loaded"))
        model.bind_func("empty_message_secondary", lambda: tr("scene.use_file_menu"))
        model.bind_func("models_collapsed", lambda: self._models_collapsed)
        model.bind_func("models_header_text",
                        lambda: tr("scene.models").format(self._root_count))
        model.bind_func("top_spacer_height", lambda: self._top_spacer_height)
        model.bind_func("bottom_spacer_height", lambda: self._bottom_spacer_height)
        model.bind_func("context_menu_visible", lambda: self._context_menu_visible)
        model.bind_func("context_menu_left", lambda: self._context_menu_left)
        model.bind_func("context_menu_top", lambda: self._context_menu_top)
        model.bind_func("show_filter_clear", lambda: len(self._filter_text) > 0)
        model.bind_record_list("visible_rows")
        model.bind_record_list("context_menu_entries")
        self._handle = model.get_handle()

    def on_load(self, doc):
        self.doc = doc
        self._last_lang = lf.ui.get_current_language()
        self.container = doc.get_element_by_id("tree-container")
        self.filter_input = doc.get_element_by_id("filter-input")
        self._filter_clear = doc.get_element_by_id("filter-clear")
        self._context_menu = doc.get_element_by_id("context-menu")

        if self.filter_input:
            self.filter_input.add_event_listener("change", self._on_filter_change)
        if self._filter_clear:
            self._filter_clear.add_event_listener("click", self._on_filter_clear)

        if self.container:
            self.container.add_event_listener("click", self._on_tree_click)
            self.container.add_event_listener("dblclick", self._on_tree_dblclick)
            self.container.add_event_listener("mousedown", self._on_tree_mousedown)
            self.container.add_event_listener("dragstart", self._on_tree_dragstart)
            self.container.add_event_listener("dragover", self._on_tree_dragover)
            self.container.add_event_listener("dragout", self._on_tree_dragout)
            self.container.add_event_listener("dragdrop", self._on_tree_dragdrop)
            self.container.add_event_listener("dragend", self._on_tree_dragend)

        if self._context_menu:
            self._context_menu.add_event_listener("click", self._on_context_menu_click)

        body = doc.get_element_by_id("body")
        if body:
            body.add_event_listener("keydown", self._on_keydown)
            body.add_event_listener("click", self._on_body_click)

        self._rebuild_tree(force=True)

    def on_scene_changed(self, doc):
        del doc
        mutation_flags = self._scene_mutation_flags()
        if mutation_flags:
            self._hide_context_menu()
        if not self._handle_scene_changed(mutation_flags):
            self._rebuild_tree(force=True)

    def on_update(self, doc):
        dirty = False

        self._capture_rename_buffer()

        cur_lang = lf.ui.get_current_language()
        if cur_lang != self._last_lang:
            self._last_lang = cur_lang
            self._hide_context_menu()
            if self._handle:
                for name in ("search_placeholder",
                             "empty_message_primary",
                             "empty_message_secondary",
                             "models_header_text"):
                    self._handle.dirty(name)
            dirty |= self._rebuild_tree(force=True)

        cur_invert = lf.ui.get_invert_masks()
        if cur_invert != self._last_invert_masks:
            self._last_invert_masks = cur_invert
            dirty |= self._render_tree_window(force=True)

        current = set(lf.get_selected_node_names())
        if current != self._prev_selected:
            self._prev_selected = current
            self._selected_nodes = current
            if current and self._restore_scroll_top is None:
                self._scroll_to_node = next(iter(current))
            dirty |= self._render_tree_window(force=True)

        if self.container:
            view_h = self.container.client_height or self.container.offset_height
            scroll_top = self.container.scroll_top
            ui_scale = self._ui_scale()
            if (self._restore_scroll_top is not None or self._scroll_to_node is not None or
                    abs(scroll_top - self._last_scroll_top) > 0.5 or
                    abs(view_h - self._last_view_h) > 0.5 or
                    abs(ui_scale - self._last_ui_scale) > 0.001):
                dirty |= self._render_tree_window(force=False)

        return dirty

    def _dirty_model(self, *fields):
        if not self._handle:
            return
        for field in fields:
            self._handle.dirty(field)

    def _find_row_from_target(self, target):
        el = target
        while el is not None:
            if el.is_class_set("tree-row"):
                return el
            el = el.parent()
        return None

    def _on_tree_click(self, event):
        target = event.target()
        if target is None:
            return

        if target.has_attribute("data-action"):
            event.stop_propagation()
            action = target.get_attribute("data-action")
            node_name = target.get_attribute("data-node", "")
            self._handle_inline_action(action, node_name)
            return

        if target.is_class_set("expand-toggle"):
            event.stop_propagation()
            target_id = target.get_attribute("data-target", "")
            self._toggle_expand(target_id)
            return

        el = target
        while el is not None and el != self.container:
            if el.is_class_set("section-header"):
                event.stop_propagation()
                self._toggle_models_section()
                return
            el = el.parent()

        row = self._find_row_from_target(target)
        if row:
            event.stop_propagation()
            node_name = row.get_attribute("data-node", "")
            if node_name:
                self._handle_click(node_name)

    def _on_tree_dblclick(self, event):
        target = event.target()
        if target is None:
            return
        if target.has_attribute("data-action") or target.is_class_set("expand-toggle"):
            return
        row = self._find_row_from_target(target)
        if not row:
            return
        event.stop_propagation()
        node_name = row.get_attribute("data-node", "")
        node_type = row.get_attribute("data-type", "")
        if not node_name:
            return
        scene = lf.get_scene()
        if not scene:
            return
        node = scene.get_node(node_name)
        if not node:
            return
        if node_type == "CAMERA":
            from .image_preview_panel import open_camera_preview_by_uid
            open_camera_preview_by_uid(node.camera_uid)
        elif node_type == "KEYFRAME":
            kf = node.keyframe_data()
            if kf:
                lf.ui.go_to_keyframe(kf.keyframe_index)

    def _on_tree_mousedown(self, event):
        button = int(event.get_parameter("button", "0"))
        if button != 1:
            return
        target = event.target()
        if target is None:
            return
        row = self._find_row_from_target(target)
        if not row:
            return
        event.stop_propagation()
        node_name = row.get_attribute("data-node", "")
        if not node_name:
            return
        mouse_x = event.get_parameter("mouse_x", "0")
        mouse_y = event.get_parameter("mouse_y", "0")
        if node_name not in self._selected_nodes:
            self._preserve_scroll_for_local_selection()
            lf.select_node(node_name)
            self._selected_nodes = {node_name}
            self._click_anchor = node_name
            self._prev_selected = set(self._selected_nodes)
            self._render_tree_window(force=True)
        self._show_context_menu(node_name, mouse_x, mouse_y)

    def _on_tree_dragstart(self, event):
        row = self._find_row_from_target(event.target())
        if row:
            self._drag_source = row.get_attribute("data-node", "")
            self._set_drop_target(None)

    def _on_tree_dragend(self, event):
        del event
        self._drag_source = None
        self._set_drop_target(None)

    def _on_tree_dragover(self, event):
        row = self._find_row_from_target(event.target())
        if not row:
            return
        target_name = row.get_attribute("data-node", "")
        if self._drag_source and target_name != self._drag_source:
            self._set_drop_target(target_name)

    def _on_tree_dragout(self, event):
        del event

    def _on_tree_dragdrop(self, event):
        row = self._find_row_from_target(event.target())
        if row and self._drag_source:
            target_name = row.get_attribute("data-node", "")
            if self._drag_source != target_name:
                lf.reparent_node(self._drag_source, target_name)
                self._drag_source = None
                self._set_drop_target(None)

    def _on_context_menu_click(self, event):
        target = event.target()
        if target is None:
            return
        el = target
        while el is not None and el != self._context_menu:
            if el.is_class_set("context-menu-item"):
                action = el.get_attribute("data-action", "")
                if action:
                    event.stop_propagation()
                    self._hide_context_menu()
                    self._execute_action(action)
                return
            el = el.parent()

    def _handle_inline_action(self, action, node_name):
        scene = lf.get_scene()
        if not scene:
            return
        if action == "toggle-vis":
            node = scene.get_node(node_name)
            if node:
                new_visible = not node.visible
                lf.set_node_visibility(node_name, new_visible)
                if self._set_row_visibility_state(node_name, new_visible):
                    self._render_tree_window(force=True)

    def _toggle_expand(self, target_id):
        if not target_id:
            return
        try:
            nid = int(target_id.replace("children-", ""))
        except ValueError:
            return
        if nid in self._collapsed_ids:
            self._collapsed_ids.discard(nid)
        else:
            self._collapsed_ids.add(nid)
        if not self._apply_expand_toggle(nid):
            self._rebuild_tree(force=True)
            return
        self._mark_tree_dirty()
        self._render_tree_window(force=True)

    def _toggle_models_section(self):
        self._models_collapsed = not self._models_collapsed
        self._last_render_key = None
        self._render_tree_window(force=True)

    def _on_keydown(self, event):
        key = int(event.get_parameter("key_identifier", "0"))

        if key == KI_F2:
            if self._selected_nodes and not self._rename_node:
                name = next(iter(self._selected_nodes))
                scene = lf.get_scene()
                if scene:
                    node = scene.get_node(name)
                    if node and _is_deletable(_node_type(node),
                                              self._check_parent_dataset(scene, node)):
                        self._rename_node = name
                        self._rename_buffer = name
                        self._render_tree_window(force=True)
            event.stop_propagation()

        elif key == KI_DELETE:
            if self._rename_node:
                return
            scene = lf.get_scene()
            if scene:
                self._delete_selected(scene)
            event.stop_propagation()

        elif key == KI_ESCAPE:
            if self._rename_node:
                self._rename_node = None
                self._rename_buffer = ""
                self._render_tree_window(force=True)
            self._hide_context_menu()
            event.stop_propagation()

    def _on_body_click(self, event):
        del event
        self._hide_context_menu()

    def _on_filter_change(self, event):
        del event
        if self.filter_input:
            self._filter_text = self.filter_input.get_attribute("value") or ""
        self._dirty_model("show_filter_clear")
        self._rebuild_tree(force=True)

    def _on_filter_clear(self, event):
        del event
        self._filter_text = ""
        if self.filter_input:
            self.filter_input.set_attribute("value", "")
        self._dirty_model("show_filter_clear")
        self._rebuild_tree(force=True)

    def _preserve_scroll_for_local_selection(self):
        self._restore_scroll_top = self.container.scroll_top if self.container else None

    def _ui_scale(self):
        try:
            return max(float(lf.get_ui_scale()), 1.0)
        except (RuntimeError, AttributeError, ValueError, TypeError):
            return 1.0

    def _row_height_px(self):
        return TREE_ROW_HEIGHT_DP * self._ui_scale()

    def _default_header_height_px(self):
        return TREE_HEADER_HEIGHT_DP * self._ui_scale()

    def _current_header_height_px(self):
        if self.doc and self._scene_has_nodes:
            header = self.doc.get_element_by_id("models-header")
            if header:
                return max(header.offset_height, self._default_header_height_px())
        return self._default_header_height_px()

    def _format_row_span_height(self, row_count):
        return f"{max(0, row_count) * TREE_ROW_HEIGHT_DP}dp"

    def _get_active_rename_input(self):
        if not self.container or not self._rename_node:
            return None
        for rename_el in self.container.query_selector_all(".rename-input"):
            row = self._find_row_from_target(rename_el)
            if row and row.get_attribute("data-node", "") == self._rename_node:
                return rename_el
        return None

    def _capture_rename_buffer(self):
        if not self.doc or not self._rename_node:
            return
        rename_el = self._get_active_rename_input()
        if rename_el:
            self._rename_buffer = rename_el.get_attribute(
                "value", self._rename_buffer or self._rename_node)

    def _format_node_label(self, node, node_type):
        label = node.name
        if node_type == "SPLAT" and node.gaussian_count > 0:
            label += f"  ({node.gaussian_count:,})"
        elif node_type == "POINTCLOUD":
            point_cloud = node.point_cloud()
            if point_cloud:
                label += f"  ({point_cloud.size:,})"
        elif node_type == "MESH":
            mesh = node.mesh()
            if mesh:
                label += f"  ({mesh.vertex_count:,}V / {mesh.face_count:,}F)"
        elif node_type == "KEYFRAME":
            keyframe = node.keyframe_data()
            if keyframe:
                label = tr("scene.keyframe_label").format(
                    index=keyframe.keyframe_index + 1, time=keyframe.time)
        return label

    def _make_node_snapshot(self, node):
        node_type = _node_type(node)
        has_mask = bool(node_type == "CAMERA" and getattr(node, "mask_path", ""))
        return {
            "name": node.name,
            "id": node.id,
            "parent_id": node.parent_id,
            "children": tuple(node.children),
            "node_type": node_type,
            "visible": bool(node.visible),
            "has_children": len(node.children) > 0,
            "training_enabled": bool(getattr(node, "training_enabled", True)),
            "label": self._format_node_label(node, node_type),
            "draggable": False,
            "has_mask": has_mask,
        }

    def _capture_scene_snapshot(self, scene):
        nodes = scene.get_nodes()
        snapshots = {}
        for node in nodes:
            snapshots[node.id] = self._make_node_snapshot(node)

        for snapshot in snapshots.values():
            parent = snapshots.get(snapshot["parent_id"])
            snapshot["draggable"] = _can_drag(
                snapshot["node_type"],
                bool(parent and parent["node_type"] == "DATASET"))

        root_ids = [node.id for node in nodes if node.parent_id == -1]
        return snapshots, root_ids

    def _make_row_state(self, snapshot, depth):
        return {
            "name": snapshot["name"],
            "id": snapshot["id"],
            "node_type": snapshot["node_type"],
            "depth": depth,
            "visible": snapshot["visible"],
            "has_children": snapshot["has_children"],
            "collapsed": snapshot["id"] in self._collapsed_ids,
            "draggable": snapshot["draggable"],
            "training_enabled": snapshot["training_enabled"],
            "label": snapshot["label"],
            "has_mask": snapshot["has_mask"],
        }

    def _append_snapshot_rows(self, node_id, depth, rows, filter_text_lower):
        snapshot = self._node_snapshots.get(node_id)
        if not snapshot:
            return

        child_rows = []
        for child_id in snapshot["children"]:
            self._append_snapshot_rows(child_id, depth + 1, child_rows, filter_text_lower)

        if filter_text_lower and filter_text_lower not in snapshot["name"].lower():
            rows.extend(child_rows)
            return

        row = self._make_row_state(snapshot, depth)
        rows.append(row)
        if row["has_children"] and not row["collapsed"]:
            rows.extend(child_rows)

    def _append_visible_subtree_rows(self, node_id, depth, rows):
        snapshot = self._node_snapshots.get(node_id)
        if not snapshot:
            return

        row = self._make_row_state(snapshot, depth)
        rows.append(row)
        if row["has_children"] and not row["collapsed"]:
            for child_id in snapshot["children"]:
                self._append_visible_subtree_rows(child_id, depth + 1, rows)

    def _make_visible_row_state(self, row, absolute_index):
        node_type = row["node_type"]
        type_icon_src = NODE_TYPE_ICON_PATH.get(node_type, "")
        use_type_icon = bool(type_icon_src)
        unicode_icon = NODE_TYPE_UNICODE.get(node_type, "")
        renaming = self._rename_node == row["name"]
        has_mask = row["has_mask"]
        mask_inverted = has_mask and lf.ui.get_invert_masks()
        return {
            "present": True,
            "name": row["name"],
            "id": row["id"],
            "node_type": node_type,
            "selected": row["name"] in self._selected_nodes,
            "even": absolute_index % 2 == 0,
            "has_children": row["has_children"],
            "collapsed": row["collapsed"],
            "visible": row["visible"],
            "label": row["label"],
            "indent": f"{row['depth'] * 16}dp",
            "type_class": NODE_TYPE_CSS_CLASS.get(node_type, ""),
            "use_type_icon": use_type_icon,
            "type_icon_src": type_icon_src,
            "use_unicode_icon": not use_type_icon and bool(unicode_icon),
            "unicode_icon": unicode_icon or "?",
            "expand_glyph": "\u25B6" if row["collapsed"] else "\u25BC",
            "children_target": f"children-{row['id']}",
            "vis_sprite": "icon-visible" if row["visible"] else "icon-hidden",
            "drag_mode": "drag-drop" if row["draggable"] else "none",
            "camera_training_disabled": node_type == "CAMERA" and not row["training_enabled"],
            "renaming": renaming,
            "rename_value": self._rename_buffer if renaming else "",
            "drop_target": self._drop_target == row["name"],
            "has_mask": has_mask,
            "mask_inverted": mask_inverted,
        }

    def _make_placeholder_row(self, absolute_index):
        return {
            "present": False,
            "name": "",
            "id": -1,
            "node_type": "",
            "selected": False,
            "even": absolute_index % 2 == 0,
            "has_children": False,
            "collapsed": False,
            "visible": False,
            "label": "",
            "indent": "0dp",
            "type_class": "",
            "use_type_icon": False,
            "type_icon_src": "",
            "use_unicode_icon": False,
            "unicode_icon": "",
            "expand_glyph": "",
            "children_target": "",
            "vis_sprite": "icon-hidden",
            "drag_mode": "none",
            "camera_training_disabled": False,
            "renaming": False,
            "rename_value": "",
            "drop_target": False,
            "has_mask": False,
            "mask_inverted": False,
        }

    def _set_row_visibility_state(self, node_name, visible):
        for row in self._flat_rows:
            if row["name"] == node_name:
                if row["visible"] == visible:
                    return False
                row["visible"] = visible
                snapshot = self._node_snapshots.get(row["id"])
                if snapshot:
                    snapshot["visible"] = visible
                self._mark_tree_dirty()
                return True
        return False

    def _set_drop_target(self, node_name):
        if node_name == self._drop_target:
            return False
        self._drop_target = node_name
        self._last_render_key = None
        return self._render_tree_window(force=True)

    def _mark_tree_dirty(self):
        self._tree_revision += 1
        self._last_render_key = None

    def _reindex_flat_rows(self):
        self._flat_index_by_id = {
            row["id"]: index
            for index, row in enumerate(self._flat_rows)
        }
        self._committed_node_order = [row["name"] for row in self._flat_rows]

    def _remap_name_state(self, renamed):
        if not renamed:
            return

        self._selected_nodes = {renamed.get(name, name) for name in self._selected_nodes}
        self._prev_selected = {renamed.get(name, name) for name in self._prev_selected}

        if self._click_anchor in renamed:
            self._click_anchor = renamed[self._click_anchor]
        if self._scroll_to_node in renamed:
            self._scroll_to_node = renamed[self._scroll_to_node]
        if self._rename_node in renamed:
            self._rename_node = renamed[self._rename_node]
        if self._context_menu_node in renamed:
            self._context_menu_node = renamed[self._context_menu_node]
        if self._drag_source in renamed:
            self._drag_source = renamed[self._drag_source]
        if self._drop_target in renamed:
            self._drop_target = renamed[self._drop_target]

    def _refresh_flat_rows_from_snapshot(self):
        renamed = {}
        for row in self._flat_rows:
            snapshot = self._node_snapshots.get(row["id"])
            if not snapshot:
                return False
            old_name = row["name"]
            row.update(self._make_row_state(snapshot, row["depth"]))
            if old_name != row["name"]:
                renamed[old_name] = row["name"]

        self._remap_name_state(renamed)
        self._reindex_flat_rows()
        self._mark_tree_dirty()
        return True

    def _refresh_tree_content(self, scene):
        if not self._node_snapshots:
            return False

        snapshots, root_ids = self._capture_scene_snapshot(scene)
        if set(snapshots) != set(self._node_snapshots):
            return False
        if root_ids != self._root_ids:
            return False

        for node_id, snapshot in snapshots.items():
            previous = self._node_snapshots.get(node_id)
            if previous is None:
                return False
            if (previous["parent_id"] != snapshot["parent_id"] or
                    previous["children"] != snapshot["children"] or
                    previous["node_type"] != snapshot["node_type"]):
                return False

        self._node_snapshots = snapshots
        self._root_ids = root_ids
        self._root_count = len(root_ids)
        self._scene_has_nodes = bool(root_ids)
        if not self._refresh_flat_rows_from_snapshot():
            return False
        return self._render_tree_window(force=True)

    def _handle_scene_changed(self, mutation_flags):
        scene = lf.get_scene()
        if scene is None or not scene.has_nodes():
            return False
        if mutation_flags == 0:
            return False
        if mutation_flags in (SCENE_MUTATION_SELECTION_CHANGED,
                              SCENE_MUTATION_TRANSFORM_CHANGED):
            return True
        if mutation_flags & SCENE_STRUCTURAL_MUTATIONS:
            return False
        if self._filter_text and mutation_flags & SCENE_MUTATION_NODE_RENAMED:
            return False
        return self._refresh_tree_content(scene)

    def _apply_expand_toggle(self, node_id):
        if self._filter_text or node_id not in self._flat_index_by_id:
            return False

        row_index = self._flat_index_by_id[node_id]
        row = self._flat_rows[row_index]
        snapshot = self._node_snapshots.get(node_id)
        if not snapshot or not row["has_children"]:
            return False

        row["collapsed"] = node_id in self._collapsed_ids
        if row["collapsed"]:
            end = row_index + 1
            while end < len(self._flat_rows) and self._flat_rows[end]["depth"] > row["depth"]:
                end += 1
            del self._flat_rows[row_index + 1:end]
        else:
            inserted_rows = []
            for child_id in snapshot["children"]:
                self._append_visible_subtree_rows(child_id, row["depth"] + 1, inserted_rows)
            if inserted_rows:
                self._flat_rows[row_index + 1:row_index + 1] = inserted_rows

        self._reindex_flat_rows()
        return True

    def _render_tree_window(self, force=False):
        if not self.container or not self._handle:
            return False

        row_height = self._row_height_px()
        viewport_h = self.container.client_height or self.container.offset_height or row_height
        header_h = self._current_header_height_px() if self._scene_has_nodes else 0.0
        current_scroll_top = self.container.scroll_top
        scroll_top = (self._restore_scroll_top
                      if self._restore_scroll_top is not None else current_scroll_top)

        if self._scroll_to_node and self._scroll_to_node in self._committed_node_order:
            index = self._committed_node_order.index(self._scroll_to_node)
            row_top = header_h + index * row_height
            row_bottom = row_top + row_height
            if row_top < scroll_top:
                scroll_top = row_top
            elif row_bottom > scroll_top + viewport_h:
                scroll_top = row_bottom - viewport_h

        total_rows = len(self._flat_rows)
        if not self._scene_has_nodes:
            start = 0
            end = 0
            scroll_top = 0.0
        else:
            total_content_h = (header_h if self._models_collapsed
                               else header_h + total_rows * row_height)
            max_scroll_top = max(0.0, total_content_h - viewport_h)
            scroll_top = min(max(0.0, scroll_top), max_scroll_top)
            if self._models_collapsed:
                start = 0
                end = 0
            else:
                rows_scroll_top = max(0.0, scroll_top - header_h)
                start = max(0, int(rows_scroll_top // row_height) - TREE_OVERSCAN_ROWS)
                visible_count = max(
                    1, int(math.ceil(viewport_h / row_height)) + TREE_OVERSCAN_ROWS * 2)
                end = min(total_rows, start + visible_count)

        render_key = (
            self._tree_revision,
            tuple(sorted(self._selected_nodes)),
            self._models_collapsed,
            self._rename_node,
            self._rename_buffer,
            self._scene_has_nodes,
            self._drop_target or "",
            start,
            end,
        )

        if not force and render_key == self._last_render_key:
            if abs(scroll_top - current_scroll_top) > 0.5:
                self.container.scroll_top = scroll_top
            self._last_scroll_top = scroll_top
            self._last_view_h = viewport_h
            self._last_ui_scale = self._ui_scale()
            self._restore_scroll_top = None
            self._scroll_to_node = None
            return abs(scroll_top - current_scroll_top) > 0.5

        visible_rows = []
        should_render_rows = self._scene_has_nodes and not self._models_collapsed
        if should_render_rows:
            for absolute_index in range(start, end):
                visible_rows.append(
                    self._make_visible_row_state(self._flat_rows[absolute_index], absolute_index))
            self._top_spacer_height = self._format_row_span_height(start)
            self._bottom_spacer_height = self._format_row_span_height(total_rows - end)
        else:
            self._top_spacer_height = "0dp"
            self._bottom_spacer_height = "0dp"

        self._visible_row_capacity = max(self._visible_row_capacity, len(visible_rows))
        while len(visible_rows) < self._visible_row_capacity:
            visible_rows.append(self._make_placeholder_row(len(visible_rows)))

        self._handle.update_record_list("visible_rows", visible_rows)
        for name in ("show_tree",
                     "show_empty_state",
                     "models_collapsed",
                     "models_header_text",
                     "top_spacer_height",
                     "bottom_spacer_height"):
            self._handle.dirty(name)

        if abs(scroll_top - current_scroll_top) > 0.5 or self._restore_scroll_top is not None:
            self.container.scroll_top = scroll_top

        self._last_render_key = render_key
        self._last_scroll_top = scroll_top
        self._last_view_h = viewport_h
        self._last_ui_scale = self._ui_scale()
        self._restore_scroll_top = None
        self._scroll_to_node = None

        self._setup_rename_input()
        return True

    def _rebuild_tree(self, force=False):
        if not self.container:
            return False

        self._drop_target = None
        scene = lf.get_scene()
        if scene is None or not scene.has_nodes():
            self._scene_has_nodes = False
            self._node_snapshots = {}
            self._root_ids = []
            self._flat_rows = []
            self._flat_index_by_id = {}
            self._committed_node_order = []
            self._root_count = 0
            self._mark_tree_dirty()
            return self._render_tree_window(force=True)

        self._node_snapshots, self._root_ids = self._capture_scene_snapshot(scene)
        self._scene_has_nodes = True
        self._root_count = len(self._root_ids)
        self._selected_nodes = set(lf.get_selected_node_names())
        rows = []
        filter_text_lower = self._filter_text.lower()
        for node_id in self._root_ids:
            self._append_snapshot_rows(node_id, 0, rows, filter_text_lower)

        self._flat_rows = rows
        self._reindex_flat_rows()
        self._mark_tree_dirty()
        return self._render_tree_window(force=True)

    def _setup_rename_input(self):
        if not self._rename_node or not self.doc:
            return
        rename_el = self._get_active_rename_input()
        if rename_el:
            rename_el.focus()
            rename_el.add_event_listener("keydown", self._on_rename_keydown)

    def _on_rename_keydown(self, event):
        key = int(event.get_parameter("key_identifier", "0"))
        if key == KI_RETURN:
            event.stop_propagation()
            self._confirm_rename()
        elif key == KI_ESCAPE:
            event.stop_propagation()
            self._cancel_rename()

    def _confirm_rename(self):
        if not self._rename_node or not self.doc:
            return
        self._capture_rename_buffer()
        new_name = self._rename_buffer or self._rename_node
        if new_name and new_name != self._rename_node:
            lf.rename_node(self._rename_node, new_name)
        self._rename_node = None
        self._rename_buffer = ""
        self._render_tree_window(force=True)

    def _cancel_rename(self):
        self._rename_node = None
        self._rename_buffer = ""
        self._render_tree_window(force=True)

    def _handle_click(self, node_name):
        self._hide_context_menu()
        ctrl = lf.ui.is_ctrl_down()
        shift = lf.ui.is_shift_down()

        if ctrl:
            self._preserve_scroll_for_local_selection()
            if node_name in self._selected_nodes:
                self._selected_nodes.discard(node_name)
                lf.select_nodes(list(self._selected_nodes))
            else:
                lf.add_to_selection(node_name)
                self._selected_nodes.add(node_name)
            self._click_anchor = node_name
        elif shift and self._click_anchor:
            self._preserve_scroll_for_local_selection()
            names = self._get_range(self._click_anchor, node_name)
            lf.select_nodes(names)
            self._selected_nodes = set(names)
        else:
            if self._selected_nodes == {node_name}:
                return
            self._preserve_scroll_for_local_selection()
            lf.select_node(node_name)
            self._selected_nodes = {node_name}
            self._click_anchor = node_name

        self._prev_selected = set(self._selected_nodes)
        self._render_tree_window(force=True)

    def _get_range(self, a, b):
        order = self._committed_node_order
        try:
            ia, ib = order.index(a), order.index(b)
        except ValueError:
            return [b]
        lo, hi = min(ia, ib), max(ia, ib)
        return order[lo:hi + 1]

    def _check_parent_dataset(self, scene, node):
        if node.parent_id != -1:
            parent = scene.get_node_by_id(node.parent_id)
            if parent and _node_type(parent) == "DATASET":
                return True
        return False

    def _scene_mutation_flags(self):
        try:
            return int(lf.consume_scene_mutation_flags())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return 0

    def _show_context_menu(self, node_name, mouse_x="0", mouse_y="0"):
        if not self._context_menu or not self.doc or not self._handle:
            return

        scene = lf.get_scene()
        if not scene:
            return

        node = scene.get_node(node_name)
        if not node:
            return

        node_type = _node_type(node)
        parent_is_dataset = self._check_parent_dataset(scene, node)
        is_del = _is_deletable(node_type, parent_is_dataset)
        draggable = _can_drag(node_type, parent_is_dataset)

        if len(self._selected_nodes) > 1:
            items = self._build_multi_context_items(scene)
        else:
            items = self._build_single_context_items(
                scene, node, node_type, is_del, draggable)

        if not items:
            self._hide_context_menu()
            return

        self._handle.update_record_list("context_menu_entries", items)
        self._context_menu_visible = True

        item_count = sum(not item["is_label"] for item in items)
        label_count = sum(item["is_label"] for item in items)
        sep_count = sum(item["separator_before"] for item in items)
        estimated_h = item_count * 22 + label_count * 20 + sep_count * 5 + 8
        body = self.doc.get_element_by_id("body")
        panel_h = body.scroll_height if body else 600
        my = float(mouse_y)
        if my + estimated_h > panel_h:
            my = max(0, my - estimated_h)

        self._context_menu_left = f"{mouse_x}px"
        self._context_menu_top = f"{my:.0f}px"
        self._context_menu_node = node_name
        self._dirty_model(
            "context_menu_entries",
            "context_menu_visible",
            "context_menu_left",
            "context_menu_top",
        )

    @staticmethod
    def _context_menu_action(label, action, separator_before=False,
                             is_submenu_item=False, is_active=False):
        return {
            "label": label,
            "action": action,
            "is_label": False,
            "separator_before": separator_before,
            "is_submenu_item": is_submenu_item,
            "is_active": is_active,
        }

    @staticmethod
    def _context_menu_label(label, separator_before=False):
        return {
            "label": label,
            "action": "",
            "is_label": True,
            "separator_before": separator_before,
            "is_submenu_item": False,
            "is_active": False,
        }

    def _build_single_context_items(self, scene, node, node_type, is_deletable, can_drag):
        items = []

        if node_type == "CAMERA":
            items.append(self._context_menu_action(
                tr("scene.go_to_camera_view"),
                f"go_to_camera:{node.camera_uid}",
            ))
            items.append(self._context_menu_action(
                tr("scene.disable_for_training") if node.training_enabled
                else tr("scene.enable_for_training"),
                f"disable_train:{node.name}" if node.training_enabled
                else f"enable_train:{node.name}",
                separator_before=True,
            ))
            return items

        if node_type == "KEYFRAME":
            kf = node.keyframe_data()
            if kf:
                items.extend([
                    self._context_menu_action(
                        tr("scene.go_to_keyframe"),
                        f"go_to_kf:{kf.keyframe_index}",
                    ),
                    self._context_menu_action(
                        tr("scene.update_keyframe"),
                        f"update_kf:{kf.keyframe_index}",
                    ),
                    self._context_menu_action(
                        tr("scene.select_in_timeline"),
                        f"select_kf:{kf.keyframe_index}",
                    ),
                    self._context_menu_label(
                        tr("scene.keyframe_easing"),
                        separator_before=True,
                    ),
                ])
                for easing_id, easing_key in EASING_TYPES:
                    items.append(self._context_menu_action(
                        tr(easing_key),
                        f"set_easing:{kf.keyframe_index}:{easing_id}",
                        is_submenu_item=True,
                        is_active=(kf.easing == easing_id),
                    ))

                if kf.keyframe_index > 0:
                    items.append(self._context_menu_action(
                        tr("scene.delete"),
                        f"delete_kf:{kf.keyframe_index}",
                        separator_before=True,
                    ))
            return items

        if node_type == "KEYFRAME_GROUP":
            items.append(self._context_menu_action(
                tr("scene.add_keyframe_scene"),
                "add_kf",
            ))
            return items

        if node_type == "CAMERA_GROUP":
            items.extend([
                self._context_menu_action(
                    tr("scene.enable_all_training"),
                    f"enable_all_train:{node.name}",
                ),
                self._context_menu_action(
                    tr("scene.disable_all_training"),
                    f"disable_all_train:{node.name}",
                ),
            ])
            return items

        if node_type == "DATASET":
            items.append(self._context_menu_action(
                tr("scene.delete"),
                f"delete:{node.name}",
            ))
            return items

        if node_type == "CROPBOX":
            items.extend([
                self._context_menu_action(tr("common.apply"), "apply_cropbox"),
                self._context_menu_action(
                    tr("scene.fit_to_scene"),
                    "fit_cropbox:0",
                    separator_before=True,
                ),
                self._context_menu_action(
                    tr("scene.fit_to_scene_trimmed"),
                    "fit_cropbox:1",
                ),
                self._context_menu_action(
                    tr("scene.reset_crop"),
                    "reset_cropbox",
                ),
                self._context_menu_action(
                    tr("scene.delete"),
                    f"delete:{node.name}",
                    separator_before=True,
                ),
            ])
            return items

        if node_type == "ELLIPSOID":
            items.extend([
                self._context_menu_action(tr("common.apply"), "apply_ellipsoid"),
                self._context_menu_action(
                    tr("scene.fit_to_scene"),
                    "fit_ellipsoid:0",
                    separator_before=True,
                ),
                self._context_menu_action(
                    tr("scene.fit_to_scene_trimmed"),
                    "fit_ellipsoid:1",
                ),
                self._context_menu_action(
                    tr("scene.reset_crop"),
                    "reset_ellipsoid",
                ),
                self._context_menu_action(
                    tr("scene.delete"),
                    f"delete:{node.name}",
                    separator_before=True,
                ),
            ])
            return items

        if node_type == "GROUP" and not AppState.has_trainer.value:
            items.extend([
                self._context_menu_action(
                    tr("scene.add_group_ellipsis"),
                    f"add_group:{node.name}",
                ),
                self._context_menu_action(
                    tr("scene.merge_to_single_ply"),
                    f"merge_group:{node.name}",
                ),
            ])

        if node_type in ("SPLAT", "POINTCLOUD"):
            separator_before = bool(items)
            items.extend([
                self._context_menu_action(
                    tr("scene.add_crop_box"),
                    f"add_cropbox:{node.name}",
                    separator_before=separator_before,
                ),
                self._context_menu_action(
                    tr("scene.add_crop_ellipsoid"),
                    f"add_ellipsoid:{node.name}",
                ),
                self._context_menu_action(
                    tr("scene.save_to_disk"),
                    f"save_node:{node.name}",
                ),
            ])

        if is_deletable:
            items.append(self._context_menu_action(
                tr("scene.rename"),
                f"rename:{node.name}",
                separator_before=bool(items),
            ))

        items.append(self._context_menu_action(
            tr("scene.duplicate"),
            f"duplicate:{node.name}",
        ))

        if can_drag:
            items.extend(self._build_move_to_items(scene, node.name))

        if is_deletable:
            items.append(self._context_menu_action(
                tr("scene.delete"),
                f"delete:{node.name}",
                separator_before=True,
            ))

        return items

    def _build_move_to_items(self, scene, node_name):
        groups = []
        for n in scene.get_nodes():
            if _node_type(n) == "GROUP" and n.name != node_name:
                groups.append(n.name)

        if not groups:
            return []

        items = [
            self._context_menu_label(tr("scene.move_to"), separator_before=True),
            self._context_menu_action(
                tr("scene.move_to_root"),
                f"reparent:{node_name}:",
                is_submenu_item=True,
            ),
        ]
        for group_name in groups:
            items.append(self._context_menu_action(
                group_name,
                f"reparent:{node_name}:{group_name}",
                is_submenu_item=True,
            ))
        return items

    def _build_multi_context_items(self, scene):
        types = set()
        deletable = []
        for name in self._selected_nodes:
            node = scene.get_node(name)
            if not node:
                continue
            ntype = _node_type(node)
            types.add(ntype)
            parent_is_dataset = self._check_parent_dataset(scene, node)
            if _is_deletable(ntype, parent_is_dataset):
                deletable.append(name)

        items = []
        if types == {"CAMERA"} or types == {"CAMERA_GROUP"}:
            items.extend([
                self._context_menu_action(
                    tr("scene.enable_all_training"),
                    "enable_all_selected_train",
                ),
                self._context_menu_action(
                    tr("scene.disable_all_training"),
                    "disable_all_selected_train",
                ),
            ])

        if deletable:
            items.append(self._context_menu_action(
                f"{tr('scene.delete')} ({len(deletable)})",
                "delete_selected",
                separator_before=bool(items),
            ))

        return items

    def _hide_context_menu(self):
        if not self._context_menu_visible and self._context_menu_node is None:
            return
        self._context_menu_visible = False
        self._context_menu_node = None
        self._dirty_model("context_menu_visible")

    def _execute_action(self, action_str):
        if not action_str:
            return

        parts = action_str.split(":", 1)
        action = parts[0]
        arg = parts[1] if len(parts) > 1 else ""

        scene = lf.get_scene()

        if action == "go_to_camera":
            lf.ui.go_to_camera_view(int(arg))
        elif action == "enable_train":
            node = scene.get_node(arg) if scene else None
            if node:
                node.training_enabled = True
                if not self._refresh_tree_content(scene):
                    self._rebuild_tree(force=True)
        elif action == "disable_train":
            node = scene.get_node(arg) if scene else None
            if node:
                node.training_enabled = False
                if not self._refresh_tree_content(scene):
                    self._rebuild_tree(force=True)
        elif action == "go_to_kf":
            lf.ui.go_to_keyframe(int(arg))
        elif action == "update_kf":
            lf.ui.select_keyframe(int(arg))
            lf.ui.update_keyframe()
        elif action == "select_kf":
            lf.ui.select_keyframe(int(arg))
        elif action == "delete_kf":
            lf.ui.delete_keyframe(int(arg))
        elif action == "add_kf":
            lf.ui.add_keyframe()
        elif action == "enable_all_train":
            self._toggle_children_training(scene, arg, True)
        elif action == "disable_all_train":
            self._toggle_children_training(scene, arg, False)
        elif action == "delete":
            lf.remove_node(arg, False)
        elif action == "rename":
            self._rename_node = arg
            self._rename_buffer = arg
            self._render_tree_window(force=True)
        elif action == "duplicate":
            lf.ui.duplicate_node(arg)
        elif action == "add_group":
            lf.add_group(tr("scene.new_group_name"), arg)
        elif action == "merge_group":
            lf.ui.merge_group(arg)
        elif action == "add_cropbox":
            lf.ui.add_cropbox(arg)
        elif action == "add_ellipsoid":
            lf.ui.add_ellipsoid(arg)
        elif action == "save_node":
            lf.ui.save_node_to_disk(arg)
        elif action == "apply_cropbox":
            lf.ui.apply_cropbox()
        elif action == "fit_cropbox":
            lf.ui.fit_cropbox_to_scene(arg == "1")
        elif action == "reset_cropbox":
            lf.ui.reset_cropbox()
        elif action == "apply_ellipsoid":
            lf.ui.apply_ellipsoid()
        elif action == "fit_ellipsoid":
            lf.ui.fit_ellipsoid_to_scene(arg == "1")
        elif action == "reset_ellipsoid":
            lf.ui.reset_ellipsoid()
        elif action == "enable_all_selected_train":
            self._toggle_selected_training(scene, True)
        elif action == "disable_all_selected_train":
            self._toggle_selected_training(scene, False)
        elif action == "delete_selected":
            self._delete_selected(scene)
        elif action == "set_easing":
            easing_parts = arg.split(":")
            if len(easing_parts) == 2:
                lf.ui.set_keyframe_easing(int(easing_parts[0]), int(easing_parts[1]))
        elif action == "reparent":
            reparent_parts = arg.split(":", 1)
            if len(reparent_parts) == 2:
                lf.reparent_node(reparent_parts[0], reparent_parts[1])

    def _toggle_children_training(self, scene, group_name, enabled):
        if not scene:
            return
        node = scene.get_node(group_name)
        if not node:
            return
        for child_id in node.children:
            child = scene.get_node_by_id(child_id)
            if child and _node_type(child) == "CAMERA":
                child.training_enabled = enabled
        if not self._refresh_tree_content(scene):
            self._rebuild_tree(force=True)

    def _toggle_selected_training(self, scene, enabled):
        if not scene:
            return
        for name in self._selected_nodes:
            node = scene.get_node(name)
            if not node:
                continue
            ntype = _node_type(node)
            if ntype == "CAMERA":
                node.training_enabled = enabled
            elif ntype == "CAMERA_GROUP":
                for child_id in node.children:
                    child = scene.get_node_by_id(child_id)
                    if child and _node_type(child) == "CAMERA":
                        child.training_enabled = enabled
        if not self._refresh_tree_content(scene):
            self._rebuild_tree(force=True)

    def _delete_selected(self, scene):
        if not scene:
            return
        for name in list(self._selected_nodes):
            node = scene.get_node(name)
            if not node:
                continue
            ntype = _node_type(node)
            parent_is_dataset = self._check_parent_dataset(scene, node)
            if _is_deletable(ntype, parent_is_dataset):
                lf.remove_node(name, False)

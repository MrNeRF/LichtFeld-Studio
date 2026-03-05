# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene Graph Panel - RmlUI DOM implementation with retained-mode event delegation."""

from pathlib import Path
import lichtfeld as lf

from .types import RmlPanel
from .ui.state import AppState

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

# Fallback Unicode icons for types without PNG
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

# RmlUI key identifiers (Rml::Input::KeyIdentifier)
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


def _type_dot_html(node_type):
    css_cls = NODE_TYPE_CSS_CLASS.get(node_type, "")
    if NODE_TYPE_ICONS.get(node_type):
        return f'<span class="color-dot {css_cls}"></span>'
    unicode_char = NODE_TYPE_UNICODE.get(node_type, "?")
    return f'<span class="node-icon {css_cls}">{unicode_char}</span>'


class ScenePanel(RmlPanel):
    idname = "lfs.scene"
    label = "Scene"
    space = "SCENE_HEADER"
    order = 0
    rml_template = "rmlui/scene_tree.rml"

    def __init__(self):
        self.doc = None
        self.container = None
        self.filter_input = None
        self._filter_text = ""
        self._selected_nodes = set()
        self._click_anchor = None
        self._visible_node_order = []
        self._committed_node_order = []
        self._prev_selected = set()
        self._scroll_to_node = None
        self._collapsed_ids = set()
        self._rename_node = None
        self._rename_buffer = ""
        self._row_index = 0
        self._context_menu = None
        self._context_menu_node = None
        self._last_scene_gen = 0
        self._drag_source = None
        self._models_collapsed = False
        self._last_lang = ""

    def on_load(self, doc):
        self.doc = doc
        self._last_lang = lf.ui.get_current_language()
        self.container = doc.get_element_by_id("tree-container")
        self.filter_input = doc.get_element_by_id("filter-input")
        self._context_menu = doc.get_element_by_id("context-menu")

        if self.filter_input:
            self.filter_input.set_attribute("placeholder", tr("scene.search"))
            self.filter_input.add_event_listener("change", self._on_filter_change)

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

    def on_scene_changed(self, doc):
        self._rebuild_tree()

    def on_update(self, doc):
        cur_lang = lf.ui.get_current_language()
        if cur_lang != self._last_lang:
            self._last_lang = cur_lang
            if self.filter_input:
                self.filter_input.set_attribute("placeholder", tr("scene.search"))
            self._rebuild_tree()

        current = set(lf.get_selected_node_names())
        if current != self._prev_selected:
            self._prev_selected = current
            self._selected_nodes = current
            self._update_selection_display()
            if current:
                self._scroll_to_node = next(iter(current))
                self._do_scroll()

    # -- DOM traversal helpers --

    def _find_row_from_target(self, target):
        el = target
        while el is not None:
            if el.is_class_set("tree-row"):
                return el
            el = el.parent()
        return None

    # -- Delegated event handlers --

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
            self._toggle_expand(target_id, target)
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
            lf.select_node(node_name)
            self._selected_nodes = {node_name}
            self._click_anchor = node_name
            self._update_selection_display()
        self._show_context_menu(node_name, mouse_x, mouse_y)

    def _on_tree_dragstart(self, event):
        row = self._find_row_from_target(event.target())
        if row:
            self._drag_source = row.get_attribute("data-node", "")

    def _on_tree_dragend(self, event):
        self._drag_source = None
        if self.container:
            for row in self.container.query_selector_all(".drop-target"):
                row.set_class("drop-target", False)

    def _on_tree_dragover(self, event):
        row = self._find_row_from_target(event.target())
        if row:
            target_name = row.get_attribute("data-node", "")
            if self._drag_source and target_name != self._drag_source:
                row.set_class("drop-target", True)

    def _on_tree_dragout(self, event):
        row = self._find_row_from_target(event.target())
        if row:
            row.set_class("drop-target", False)

    def _on_tree_dragdrop(self, event):
        row = self._find_row_from_target(event.target())
        if row and self._drag_source:
            target_name = row.get_attribute("data-node", "")
            if self._drag_source != target_name:
                lf.reparent_node(self._drag_source, target_name)
                self._drag_source = None
                self._rebuild_tree()

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

    # -- Inline action handlers --

    def _handle_inline_action(self, action, node_name):
        scene = lf.get_scene()
        if not scene:
            return
        if action == "toggle-vis":
            node = scene.get_node(node_name)
            if node:
                new_visible = not node.visible
                lf.set_node_visibility(node_name, new_visible)
                self._update_vis_icon(node_name, new_visible)

    def _update_vis_icon(self, node_name, visible):
        if not self.container:
            return
        row = self.container.query_selector(f'[data-node="{node_name}"]')
        if not row:
            return
        vis = row.query_selector("[data-action='toggle-vis']")
        if not vis:
            return
        if visible:
            vis.set_class_names("row-icon icon-vis-on")
            vis.set_attribute("sprite", "icon-visible")
        else:
            vis.set_class_names("row-icon icon-vis-off")
            vis.set_attribute("sprite", "icon-hidden")

    def _toggle_expand(self, target_id, toggle_el):
        if not self.doc or not target_id:
            return
        try:
            nid = int(target_id.replace("children-", ""))
        except ValueError:
            return
        children_elem = self.doc.get_element_by_id(target_id)
        if children_elem is None:
            return
        if nid in self._collapsed_ids:
            self._collapsed_ids.discard(nid)
            children_elem.set_class("collapsed", False)
            toggle_el.set_inner_rml("\u25BC")
        else:
            self._collapsed_ids.add(nid)
            children_elem.set_class("collapsed", True)
            toggle_el.set_inner_rml("\u25B6")

    def _toggle_models_section(self):
        if not self.doc:
            return
        content = self.doc.get_element_by_id("models-content")
        header = self.doc.get_element_by_id("models-header")
        if content:
            self._models_collapsed = not self._models_collapsed
            content.set_class("collapsed", self._models_collapsed)
            if header:
                arrow = "\u25BC" if not self._models_collapsed else "\u25B6"
                scene = lf.get_scene()
                count = sum(1 for n in scene.get_nodes() if n.parent_id == -1) if scene else 0
                header.set_inner_rml(f'{arrow} {tr("scene.models").format(count)}')

    # -- Keyboard handling --

    def _on_keydown(self, event):
        key = int(event.get_parameter("key_identifier", "0"))

        if key == KI_F2:
            if self._selected_nodes and not self._rename_node:
                name = next(iter(self._selected_nodes))
                scene = lf.get_scene()
                if scene:
                    node = scene.get_node(name)
                    if node and _is_deletable(_node_type(node), self._check_parent_dataset(scene, node)):
                        self._rename_node = name
                        self._rename_buffer = name
                        self._rebuild_tree()
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
                self._rebuild_tree()
            self._hide_context_menu()
            event.stop_propagation()

    def _on_body_click(self, event):
        self._hide_context_menu()

    def _on_filter_change(self, event):
        if self.filter_input:
            self._filter_text = self.filter_input.get_attribute("value") or ""
        self._rebuild_tree()

    # -- Tree building --

    def _rebuild_tree(self):
        if not self.container:
            return

        scene = lf.get_scene()
        if scene is None or not scene.has_nodes():
            self.container.set_inner_rml(
                '<div class="empty-message">' + tr("scene.no_data_loaded") + '</div>'
                '<div class="empty-message">' + tr("scene.use_file_menu") + '</div>'
            )
            return

        self._selected_nodes = set(lf.get_selected_node_names())
        self._row_index = 0
        self._visible_node_order = []

        nodes = scene.get_nodes()
        root_count = sum(1 for n in nodes if n.parent_id == -1)

        tree_html = ""
        for node in nodes:
            if node.parent_id == -1:
                tree_html += self._build_node_html(scene, node, 0)

        if not nodes:
            tree_html = '<div class="empty-message">' + tr("scene.no_models_loaded") + '</div>'

        arrow = "\u25BC" if not self._models_collapsed else "\u25B6"
        header_text = tr("scene.models").format(root_count)
        collapsed_cls = " collapsed" if self._models_collapsed else ""

        html = (f'<div class="section-header" id="models-header">'
                f'{arrow} {header_text}</div>'
                f'<div id="models-content" class="{collapsed_cls}">{tree_html}</div>')

        self.container.set_inner_rml(html)
        self._committed_node_order = self._visible_node_order

        self._setup_rename_input()
        self._do_scroll()

    def _build_node_html(self, scene, node, depth):
        if self._filter_text:
            filter_lower = self._filter_text.lower()
            if filter_lower not in node.name.lower():
                child_html = ""
                for child_id in node.children:
                    child = scene.get_node_by_id(child_id)
                    if child:
                        child_html += self._build_node_html(scene, child, depth + 1)
                return child_html

        node_type = _node_type(node)
        is_selected = node.name in self._selected_nodes
        has_children = len(node.children) > 0

        parent_is_dataset = self._check_parent_dataset(scene, node)
        draggable = _can_drag(node_type, parent_is_dataset)

        parity = "even" if self._row_index % 2 == 0 else "odd"
        selected_cls = " selected" if is_selected else ""
        self._row_index += 1
        self._visible_node_order.append(node.name)

        drag_attr = ' drag="drag-drop"' if draggable else ""
        indent_px = depth * 16
        indent_style = f' style="padding-left: {indent_px}dp"' if depth > 0 else ""
        row = f'<div class="tree-row {parity}{selected_cls}" data-node="{node.name}" data-id="{node.id}" data-type="{node_type}"{drag_attr}{indent_style}>'
        row += '<span class="row-content">'

        if node.visible:
            row += f'<img class="row-icon icon-vis-on" sprite="icon-visible" data-action="toggle-vis" data-node="{node.name}" />'
        else:
            row += f'<img class="row-icon icon-vis-off" sprite="icon-hidden" data-action="toggle-vis" data-node="{node.name}" />'

        row += _type_dot_html(node_type)

        if has_children:
            collapsed = node.id in self._collapsed_ids
            arrow = "\u25B6" if collapsed else "\u25BC"
            row += f'<span class="expand-toggle" data-target="children-{node.id}">{arrow}</span>'
        else:
            row += '<span class="leaf-spacer"></span>'

        if self._rename_node and node.name == self._rename_node:
            row += f'<input class="rename-input" id="rename-input" type="text" value="{node.name}" />'
        else:
            label = node.name
            if node_type == "SPLAT" and node.gaussian_count > 0:
                label += f"  ({node.gaussian_count:,})"
            elif node_type == "POINTCLOUD":
                pc = node.point_cloud()
                if pc:
                    label += f"  ({pc.size:,})"
            elif node_type == "MESH":
                mesh = node.mesh()
                if mesh:
                    label += f"  ({mesh.vertex_count:,}V / {mesh.face_count:,}F)"
            elif node_type == "KEYFRAME":
                kf = node.keyframe_data()
                if kf:
                    label = tr("scene.keyframe_label").format(index=kf.keyframe_index + 1, time=kf.time)
            row += f'<span class="node-name">{label}</span>'

        row += '</span></div>'

        if has_children:
            collapsed = node.id in self._collapsed_ids
            collapsed_cls = " collapsed" if collapsed else ""
            row += f'<div class="tree-children{collapsed_cls}" id="children-{node.id}">'
            for child_id in node.children:
                child = scene.get_node_by_id(child_id)
                if child:
                    row += self._build_node_html(scene, child, depth + 1)
            row += '</div>'

        return row

    def _setup_rename_input(self):
        if not self._rename_node or not self.doc:
            return
        rename_el = self.doc.get_element_by_id("rename-input")
        if rename_el:
            rename_el.focus()
            rename_el.add_event_listener("keydown", self._on_rename_keydown)

    # -- Rename --

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
        rename_el = self.doc.get_element_by_id("rename-input")
        if rename_el:
            new_name = rename_el.get_attribute("value", self._rename_node)
            if new_name and new_name != self._rename_node:
                lf.rename_node(self._rename_node, new_name)
        self._rename_node = None
        self._rebuild_tree()

    def _cancel_rename(self):
        self._rename_node = None
        self._rebuild_tree()

    # -- Selection --

    def _handle_click(self, node_name):
        self._hide_context_menu()
        ctrl = lf.ui.is_ctrl_down()
        shift = lf.ui.is_shift_down()

        if ctrl:
            if node_name in self._selected_nodes:
                self._selected_nodes.discard(node_name)
                lf.select_nodes(list(self._selected_nodes))
            else:
                lf.add_to_selection(node_name)
                self._selected_nodes.add(node_name)
            self._click_anchor = node_name
        elif shift and self._click_anchor:
            names = self._get_range(self._click_anchor, node_name)
            lf.select_nodes(names)
            self._selected_nodes = set(names)
        else:
            if self._selected_nodes == {node_name}:
                return
            lf.select_node(node_name)
            self._selected_nodes = {node_name}
            self._click_anchor = node_name

        self._update_selection_display()

    def _get_range(self, a, b):
        order = self._committed_node_order
        try:
            ia, ib = order.index(a), order.index(b)
        except ValueError:
            return [b]
        lo, hi = min(ia, ib), max(ia, ib)
        return order[lo:hi + 1]

    def _update_selection_display(self):
        if not self.container:
            return
        rows = self.container.query_selector_all(".tree-row")
        for row in rows:
            name = row.get_attribute("data-node")
            row.set_class("selected", name in self._selected_nodes)

    def _do_scroll(self):
        if not self._scroll_to_node or not self.container:
            return
        row = self.container.query_selector(f'[data-node="{self._scroll_to_node}"]')
        if row:
            row.scroll_into_view(False)
        self._scroll_to_node = None

    def _check_parent_dataset(self, scene, node):
        if node.parent_id != -1:
            parent = scene.get_node_by_id(node.parent_id)
            if parent and _node_type(parent) == "DATASET":
                return True
        return False

    # -- Context menu --

    def _show_context_menu(self, node_name, mouse_x="0", mouse_y="0"):
        if not self._context_menu or not self.doc:
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
            html = self._build_multi_context_html(scene)
        else:
            html = self._build_single_context_html(scene, node, node_type, is_del, draggable)

        self._context_menu.set_inner_rml(html)
        self._context_menu.set_class("visible", True)

        item_count = html.count("context-menu-item")
        label_count = html.count("context-menu-label")
        sep_count = html.count("context-menu-separator")
        estimated_h = item_count * 22 + label_count * 20 + sep_count * 5 + 8
        body = self.doc.get_element_by_id("body")
        panel_h = body.scroll_height if body else 600
        my = float(mouse_y)
        if my + estimated_h > panel_h:
            my = max(0, my - estimated_h)

        self._context_menu.set_property("left", f"{mouse_x}px")
        self._context_menu.set_property("top", f"{my:.0f}px")
        self._context_menu_node = node_name

    def _build_single_context_html(self, scene, node, node_type, is_deletable, can_drag):
        html = ""

        if node_type == "CAMERA":
            html += f'<div class="context-menu-item" data-action="go_to_camera:{node.camera_uid}">{tr("scene.go_to_camera_view")}</div>'
            html += '<div class="context-menu-separator"></div>'
            if node.training_enabled:
                html += f'<div class="context-menu-item" data-action="disable_train:{node.name}">{tr("scene.disable_for_training")}</div>'
            else:
                html += f'<div class="context-menu-item" data-action="enable_train:{node.name}">{tr("scene.enable_for_training")}</div>'
            return html

        if node_type == "KEYFRAME":
            kf = node.keyframe_data()
            if kf:
                html += f'<div class="context-menu-item" data-action="go_to_kf:{kf.keyframe_index}">{tr("scene.go_to_keyframe")}</div>'
                html += f'<div class="context-menu-item" data-action="update_kf:{kf.keyframe_index}">{tr("scene.update_keyframe")}</div>'
                html += f'<div class="context-menu-item" data-action="select_kf:{kf.keyframe_index}">{tr("scene.select_in_timeline")}</div>'

                html += '<div class="context-menu-separator"></div>'
                html += f'<div class="context-menu-label">{tr("scene.keyframe_easing")}</div>'
                for easing_id, easing_key in EASING_TYPES:
                    active = " active" if kf.easing == easing_id else ""
                    html += f'<div class="context-menu-item submenu-item{active}" data-action="set_easing:{kf.keyframe_index}:{easing_id}">{tr(easing_key)}</div>'

                if kf.keyframe_index > 0:
                    html += '<div class="context-menu-separator"></div>'
                    html += f'<div class="context-menu-item" data-action="delete_kf:{kf.keyframe_index}">{tr("scene.delete")}</div>'
            return html

        if node_type == "KEYFRAME_GROUP":
            html += f'<div class="context-menu-item" data-action="add_kf">{tr("scene.add_keyframe_scene")}</div>'
            return html

        if node_type == "CAMERA_GROUP":
            html += f'<div class="context-menu-item" data-action="enable_all_train:{node.name}">{tr("scene.enable_all_training")}</div>'
            html += f'<div class="context-menu-item" data-action="disable_all_train:{node.name}">{tr("scene.disable_all_training")}</div>'
            return html

        if node_type == "DATASET":
            html += f'<div class="context-menu-item" data-action="delete:{node.name}">{tr("scene.delete")}</div>'
            return html

        if node_type == "CROPBOX":
            html += f'<div class="context-menu-item" data-action="apply_cropbox">{tr("common.apply")}</div>'
            html += '<div class="context-menu-separator"></div>'
            html += f'<div class="context-menu-item" data-action="fit_cropbox:0">{tr("scene.fit_to_scene")}</div>'
            html += f'<div class="context-menu-item" data-action="fit_cropbox:1">{tr("scene.fit_to_scene_trimmed")}</div>'
            html += f'<div class="context-menu-item" data-action="reset_cropbox">{tr("scene.reset_crop")}</div>'
            html += '<div class="context-menu-separator"></div>'
            html += f'<div class="context-menu-item" data-action="delete:{node.name}">{tr("scene.delete")}</div>'
            return html

        if node_type == "ELLIPSOID":
            html += f'<div class="context-menu-item" data-action="apply_ellipsoid">{tr("common.apply")}</div>'
            html += '<div class="context-menu-separator"></div>'
            html += f'<div class="context-menu-item" data-action="fit_ellipsoid:0">{tr("scene.fit_to_scene")}</div>'
            html += f'<div class="context-menu-item" data-action="fit_ellipsoid:1">{tr("scene.fit_to_scene_trimmed")}</div>'
            html += f'<div class="context-menu-item" data-action="reset_ellipsoid">{tr("scene.reset_crop")}</div>'
            html += '<div class="context-menu-separator"></div>'
            html += f'<div class="context-menu-item" data-action="delete:{node.name}">{tr("scene.delete")}</div>'
            return html

        if node_type == "GROUP" and not AppState.has_trainer.value:
            html += f'<div class="context-menu-item" data-action="add_group:{node.name}">{tr("scene.add_group_ellipsis")}</div>'
            html += f'<div class="context-menu-item" data-action="merge_group:{node.name}">{tr("scene.merge_to_single_ply")}</div>'
            html += '<div class="context-menu-separator"></div>'

        if node_type in ("SPLAT", "POINTCLOUD"):
            html += f'<div class="context-menu-item" data-action="add_cropbox:{node.name}">{tr("scene.add_crop_box")}</div>'
            html += f'<div class="context-menu-item" data-action="add_ellipsoid:{node.name}">{tr("scene.add_crop_ellipsoid")}</div>'
            html += f'<div class="context-menu-item" data-action="save_node:{node.name}">{tr("scene.save_to_disk")}</div>'
            html += '<div class="context-menu-separator"></div>'

        if is_deletable:
            html += f'<div class="context-menu-item" data-action="rename:{node.name}">{tr("scene.rename")}</div>'

        html += f'<div class="context-menu-item" data-action="duplicate:{node.name}">{tr("scene.duplicate")}</div>'

        if can_drag:
            html += self._build_move_to_items(scene, node.name)

        if is_deletable:
            html += '<div class="context-menu-separator"></div>'
            html += f'<div class="context-menu-item" data-action="delete:{node.name}">{tr("scene.delete")}</div>'

        return html

    def _build_move_to_items(self, scene, node_name):
        groups = []
        for n in scene.get_nodes():
            if _node_type(n) == "GROUP" and n.name != node_name:
                groups.append(n.name)

        if not groups:
            return ""

        html = '<div class="context-menu-separator"></div>'
        html += f'<div class="context-menu-label">{tr("scene.move_to")}</div>'
        html += f'<div class="context-menu-item submenu-item" data-action="reparent:{node_name}:">{tr("scene.move_to_root")}</div>'
        for group_name in groups:
            html += f'<div class="context-menu-item submenu-item" data-action="reparent:{node_name}:{group_name}">{group_name}</div>'
        return html

    def _build_multi_context_html(self, scene):
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

        html = ""
        if types == {"CAMERA"} or types == {"CAMERA_GROUP"}:
            html += f'<div class="context-menu-item" data-action="enable_all_selected_train">{tr("scene.enable_all_training")}</div>'
            html += f'<div class="context-menu-item" data-action="disable_all_selected_train">{tr("scene.disable_all_training")}</div>'

        if deletable:
            if html:
                html += '<div class="context-menu-separator"></div>'
            html += f'<div class="context-menu-item" data-action="delete_selected">{tr("scene.delete")} ({len(deletable)})</div>'

        return html

    def _hide_context_menu(self):
        if self._context_menu:
            self._context_menu.set_class("visible", False)
            self._context_menu_node = None

    # -- Action execution --

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
                self._rebuild_tree()
        elif action == "disable_train":
            node = scene.get_node(arg) if scene else None
            if node:
                node.training_enabled = False
                self._rebuild_tree()
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
            self._rebuild_tree()
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
                self._rebuild_tree()

    # -- Bulk operations --

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
        self._rebuild_tree()

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
        self._rebuild_tree()

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

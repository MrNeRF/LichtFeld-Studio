# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene Graph Panel - RmlUI DOM implementation."""

from pathlib import Path
import lichtfeld as lf

from .types import RmlPanel
from .ui.state import AppState

# Icon images relative to the RML document (rmlui/ directory)
ICON_PATH = "../icon/scene"

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


def _type_icon_html(node_type):
    icon_name = NODE_TYPE_ICONS.get(node_type)
    css_cls = NODE_TYPE_CSS_CLASS.get(node_type, "")
    if icon_name:
        return f'<img class="row-icon icon-type {css_cls}" src="{ICON_PATH}/{icon_name}.png" />'
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
        self._force_open_ids = set()
        self._rename_node = None
        self._rename_buffer = ""
        self._row_index = 0
        self._context_menu = None
        self._context_menu_node = None
        self._last_scene_gen = 0
        self._drag_source = None
        self._models_collapsed = False

    def on_load(self, doc):
        self.doc = doc
        self.container = doc.get_element_by_id("tree-container")
        self.filter_input = doc.get_element_by_id("filter-input")
        self._context_menu = doc.get_element_by_id("context-menu")

        body = doc.get_element_by_id("body")
        if body:
            body.add_event_listener("keydown", self._on_keydown)
            body.add_event_listener("click", self._on_body_click)

    def on_scene_changed(self, doc):
        self._rebuild_tree()

    def on_update(self, doc):
        current = set(lf.get_selected_node_names())
        if current != self._prev_selected:
            self._prev_selected = current
            self._selected_nodes = current
            self._update_selection_display()
            if current:
                self._scroll_to_node = next(iter(current))
                self._do_scroll()

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

        self._attach_click_handlers(scene)
        self._attach_section_header()
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
        is_camera = node_type == "CAMERA"

        parent_is_dataset = self._check_parent_dataset(scene, node)
        draggable = _can_drag(node_type, parent_is_dataset)
        deletable = _is_deletable(node_type, parent_is_dataset)

        parity = "even" if self._row_index % 2 == 0 else "odd"
        selected_cls = " selected" if is_selected else ""
        self._row_index += 1
        self._visible_node_order.append(node.name)

        training_disabled = is_camera and not node.training_enabled
        name_cls = "node-name"
        if training_disabled:
            name_cls += " training-disabled"

        drag_attr = ' drag="drag-drop"' if draggable else ""
        indent_px = depth * 16
        indent_style = f' style="padding-left: {indent_px}dp"' if depth > 0 else ""
        row = f'<div class="tree-row {parity}{selected_cls}" data-node="{node.name}" data-id="{node.id}" data-type="{node_type}"{drag_attr}{indent_style}>'
        row += '<span class="row-content">'

        # 1. Grip icon (draggable nodes)
        if draggable:
            row += f'<img class="row-icon icon-grip" src="{ICON_PATH}/grip.png" />'

        # 2. Visibility icon
        if node.visible:
            row += f'<img class="row-icon icon-vis-on" src="{ICON_PATH}/visible.png" data-action="toggle-vis" data-node="{node.name}" />'
        else:
            row += f'<img class="row-icon icon-vis-off" src="{ICON_PATH}/hidden.png" data-action="toggle-vis" data-node="{node.name}" />'

        # 3. Training toggle (cameras only)
        if is_camera:
            tooltip = tr("scene.training_enabled_tooltip") if node.training_enabled else tr("scene.training_disabled_tooltip")
            if node.training_enabled:
                row += f'<img class="row-icon icon-train-on" src="{ICON_PATH}/camera.png" data-action="toggle-train" data-node="{node.name}" title="{tooltip}" />'
            else:
                row += f'<span class="action-icon train-off" data-action="toggle-train" data-node="{node.name}" title="{tooltip}">\u25cb</span>'

        # 4. Trash icon (deletable nodes)
        if deletable:
            row += f'<img class="row-icon icon-trash" src="{ICON_PATH}/trash.png" data-action="delete-node" data-node="{node.name}" />'

        # 5. Node type icon
        row += _type_icon_html(node_type)

        # 6. Expand toggle + name (arrow appears right before the label, like ImGui tree_node_ex)
        if has_children:
            row += f'<span class="expand-toggle" data-target="children-{node.id}">\u25BC</span>'
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
            row += f'<span class="{name_cls}">{label}</span>'

        row += '</span></div>'

        # Children
        if has_children:
            collapsed_cls = ""
            if node.id in self._force_open_ids:
                self._force_open_ids.discard(node.id)
            row += f'<div class="tree-children{collapsed_cls}" id="children-{node.id}">'
            for child_id in node.children:
                child = scene.get_node_by_id(child_id)
                if child:
                    row += self._build_node_html(scene, child, depth + 1)
            row += '</div>'

        return row

    # -- Event handler attachment --

    def _attach_click_handlers(self, scene):
        if not self.container:
            return

        rows = self.container.query_selector_all(".tree-row")
        for row_elem in rows:
            node_name = row_elem.get_attribute("data-node")
            node_type = row_elem.get_attribute("data-type")
            if not node_name:
                continue

            row_elem.add_event_listener("click", self._make_click_handler(node_name))
            row_elem.add_event_listener("contextmenu", self._make_context_handler(node_name))
            row_elem.add_event_listener("dblclick", self._make_dblclick_handler(node_name, node_type))

            row_elem.add_event_listener("dragover", self._make_dragover_handler(row_elem))
            row_elem.add_event_listener("dragout", self._make_dragout_handler(row_elem))
            row_elem.add_event_listener("dragdrop", self._make_dragdrop_handler(node_name))

            if row_elem.has_attribute("drag"):
                row_elem.add_event_listener("dragstart", self._make_dragstart_handler(node_name))
                row_elem.add_event_listener("dragend", self._on_dragend)

        # Rename input handlers
        if self._rename_node and self.doc:
            rename_el = self.doc.get_element_by_id("rename-input")
            if rename_el:
                rename_el.focus()
                rename_el.add_event_listener("keydown", self._on_rename_keydown)

        toggles = self.container.query_selector_all(".expand-toggle")
        for toggle in toggles:
            target_id = toggle.get_attribute("data-target")
            if target_id:
                toggle.add_event_listener("click", self._make_toggle_handler(target_id, toggle))

        vis_icons = self.container.query_selector_all("[data-action='toggle-vis']")
        for icon in vis_icons:
            name = icon.get_attribute("data-node")
            if name:
                icon.add_event_listener("click", self._make_vis_handler(name))

        train_icons = self.container.query_selector_all("[data-action='toggle-train']")
        for icon in train_icons:
            name = icon.get_attribute("data-node")
            if name:
                icon.add_event_listener("click", self._make_train_handler(name, scene))

        trash_icons = self.container.query_selector_all("[data-action='delete-node']")
        for icon in trash_icons:
            name = icon.get_attribute("data-node")
            if name:
                icon.add_event_listener("click", self._make_trash_handler(name))

    def _attach_section_header(self):
        if not self.doc:
            return
        header = self.doc.get_element_by_id("models-header")
        if header:
            header.add_event_listener("click", self._on_toggle_models)

    # -- Handler factories --

    def _on_toggle_models(self, event):
        event.stop_propagation()
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

    def _make_click_handler(self, node_name):
        def handler(event):
            event.stop_propagation()
            self._handle_click(node_name)
        return handler

    def _make_context_handler(self, node_name):
        def handler(event):
            event.stop_propagation()
            mouse_x = event.get_parameter("mouse_x", "0")
            mouse_y = event.get_parameter("mouse_y", "0")
            if node_name not in self._selected_nodes:
                lf.select_node(node_name)
                self._selected_nodes = {node_name}
                self._click_anchor = node_name
                self._update_selection_display()
            self._show_context_menu(node_name, mouse_x, mouse_y)
        return handler

    def _make_dblclick_handler(self, node_name, node_type):
        def handler(event):
            event.stop_propagation()
            scene = lf.get_scene()
            if not scene:
                return
            node = scene.get_node(node_name)
            if not node:
                return
            if node_type == "CAMERA":
                lf.ui.go_to_camera_view(node.camera_uid)
            elif node_type == "KEYFRAME":
                kf = node.keyframe_data()
                if kf:
                    lf.ui.go_to_keyframe(kf.keyframe_index)
        return handler

    def _make_toggle_handler(self, target_id, toggle_elem):
        def handler(event):
            event.stop_propagation()
            if not self.doc:
                return
            children_elem = self.doc.get_element_by_id(target_id)
            if children_elem is not None:
                now_collapsed = not children_elem.is_class_set("collapsed")
                children_elem.set_class("collapsed", now_collapsed)
                toggle_elem.set_inner_rml("\u25B6" if now_collapsed else "\u25BC")
        return handler

    def _make_vis_handler(self, name):
        def handler(event):
            event.stop_propagation()
            scene = lf.get_scene()
            if scene:
                node = scene.get_node(name)
                if node:
                    lf.set_node_visibility(name, not node.visible)
                    self._rebuild_tree()
        return handler

    def _make_train_handler(self, name, scene):
        def handler(event):
            event.stop_propagation()
            node = scene.get_node(name)
            if node:
                node.training_enabled = not node.training_enabled
                self._rebuild_tree()
        return handler

    def _make_trash_handler(self, name):
        def handler(event):
            event.stop_propagation()
            lf.remove_node(name, False)
        return handler

    # -- Drag-drop --

    def _make_dragstart_handler(self, node_name):
        def handler(event):
            self._drag_source = node_name
        return handler

    def _on_dragend(self, event):
        self._drag_source = None
        if not self.container:
            return
        for row in self.container.query_selector_all(".drop-target"):
            row.set_class("drop-target", False)

    def _make_dragover_handler(self, row_elem):
        def handler(event):
            target_name = row_elem.get_attribute("data-node")
            if self._drag_source and target_name != self._drag_source:
                row_elem.set_class("drop-target", True)
        return handler

    def _make_dragout_handler(self, row_elem):
        def handler(event):
            row_elem.set_class("drop-target", False)
        return handler

    def _make_dragdrop_handler(self, target_name):
        def handler(event):
            if self._drag_source and self._drag_source != target_name:
                lf.reparent_node(self._drag_source, target_name)
                self._drag_source = None
                self._rebuild_tree()
        return handler

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
        self._context_menu.set_property("left", f"{mouse_x}px")
        self._context_menu.set_property("top", f"{mouse_y}px")
        self._context_menu.set_class("visible", True)
        self._context_menu_node = node_name

        items = self._context_menu.query_selector_all(".context-menu-item")
        for item in items:
            item.add_event_listener("click", self._make_menu_action(item.get_attribute("data-action")))

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

    def _make_menu_action(self, action_str):
        def handler(event):
            event.stop_propagation()
            self._hide_context_menu()
            self._execute_action(action_str)
        return handler

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

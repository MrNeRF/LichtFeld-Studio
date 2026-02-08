# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Scene Graph Panel - Python implementation."""

from pathlib import Path
import lichtfeld as lf

from .types import Panel
from .ui.state import AppState

INDENT_SPACING = 14.0
RENAME_INPUT_WIDTH = 150.0
ROW_PADDING = 2.0
ICON_SIZE_BASE = 16.0
ICON_SPACING_BASE = 2.0

SELECTION_COLOR = (0.3, 0.5, 0.8, 0.4)
VISIBLE_TINT = (0.4, 0.9, 0.4, 1.0)
HIDDEN_TINT = (0.6, 0.4, 0.4, 0.7)
TRASH_SELECTED_TINT = (1.0, 0.6, 0.6, 0.9)
TRASH_DEFAULT_TINT = (0.5, 0.5, 0.5, 0.5)
GRIP_ACTIVE_TINT = (0.5, 0.5, 0.5, 0.5)
GRIP_INACTIVE_TINT = (0.0, 0.0, 0.0, 0.0)
MASK_TINT = (0.9, 0.5, 0.6, 0.8)
TRAINING_ENABLED_TINT = (0.4, 0.7, 1.0, 1.0)
TRAINING_DISABLED_TINT = (0.5, 0.5, 0.5, 0.4)
DEFAULT_ICON_COLOR = (0.7, 0.7, 0.7, 1.0)

NODE_TYPE_COLORS = {
    "SPLAT": (0.6, 0.8, 1.0, 0.9),
    "GROUP": (0.7, 0.7, 0.7, 0.8),
    "DATASET": (0.5, 0.7, 1.0, 0.9),
    "CAMERA": (0.5, 0.6, 0.8, 0.6),
    "CAMERA_GROUP": (0.6, 0.7, 0.9, 0.8),
    "CROPBOX": (1.0, 0.7, 0.3, 0.9),
    "ELLIPSOID": (0.3, 0.8, 1.0, 0.9),
    "POINTCLOUD": (0.8, 0.5, 1.0, 0.8),
}

NODE_TYPE_ICON_NAMES = {
    "SPLAT": "splat",
    "GROUP": "group",
    "DATASET": "dataset",
    "CAMERA": "camera",
    "CAMERA_GROUP": "camera",
    "CROPBOX": "cropbox",
    "ELLIPSOID": "ellipsoid",
    "POINTCLOUD": "pointcloud",
}


def tr(key):
    result = lf.ui.tr(key)
    return result if result else key


class ScenePanel(Panel):
    idname = "lfs.scene"
    label = "Scene"
    space = "SCENE_HEADER"
    order = 0

    def __init__(self):
        self._filter_text = ""
        self._selected_nodes = set()
        self._row_index = 0
        self._rename_node = None
        self._rename_buffer = ""
        self._rename_focus = False
        self._click_anchor = None
        self._visible_node_order = []
        self._committed_node_order = []
        self._prev_selected = set()
        self._scroll_to_node = None

    def _is_node_selected(self, node):
        return node.name in self._selected_nodes

    def _open_camera_preview(self, scene, target_cam_uid: int):
        from . import image_preview_panel

        cameras = []
        for node in scene.get_nodes():
            if str(node.type).endswith("CAMERA") and node.image_path:
                cameras.append((node.name, node.image_path, node.mask_path, node.camera_uid))

        cameras.sort(key=lambda c: c[0])

        image_paths = [Path(c[1]) for c in cameras]
        mask_paths = [Path(c[2]) if c[2] else None for c in cameras]

        target_idx = next((i for i, c in enumerate(cameras) if c[3] == target_cam_uid), 0)

        image_preview_panel.open_image_preview(image_paths, mask_paths, target_idx)

    def _get_icon(self, name: str) -> int:
        from . import icon_manager
        return icon_manager.get_scene_icon(name)

    def _draw_row_background(self, layout, is_selected, row_index):
        theme = lf.ui.theme()
        scale = layout.get_dpi_scale()
        _, cursor_y = layout.get_cursor_screen_pos()
        win_x, _ = layout.get_window_pos()
        win_width = layout.get_window_width()
        row_height = layout.get_text_line_height() + ROW_PADDING * scale

        if is_selected:
            color = SELECTION_COLOR
        elif row_index % 2 == 0:
            color = theme.palette.row_even
        else:
            color = theme.palette.row_odd

        layout.draw_rect_filled(win_x, cursor_y, win_x + win_width, cursor_y + row_height, color)

    def draw(self, layout):
        scene = lf.get_scene()
        if scene is None or not scene.has_nodes():
            theme = lf.ui.theme()
            layout.text_colored(tr("scene.no_data_loaded"), theme.palette.text_dim)
            layout.text_colored(tr("scene.use_file_menu"), theme.palette.text_dim)
            return

        self._selected_nodes = set(lf.get_selected_node_names())

        new_nodes = self._selected_nodes - self._prev_selected
        if new_nodes:
            self._scroll_to_node = next(iter(new_nodes))
        self._prev_selected = set(self._selected_nodes)

        theme = lf.ui.theme()
        scale = layout.get_dpi_scale()

        layout.push_style_var_vec2("FramePadding", (4.0 * scale, 2.0 * scale))
        layout.push_style_color("FrameBg", (theme.palette.surface[0], theme.palette.surface[1],
                                            theme.palette.surface[2], 0.5))
        layout.push_item_width(-1)
        _, self._filter_text = layout.input_text_with_hint("##filter", tr("scene.filter"), self._filter_text)
        layout.pop_item_width()
        layout.pop_style_color()
        layout.pop_style_var()

        layout.spacing()

        layout.push_style_var_vec2("FramePadding", (2.0 * scale, 1.0 * scale))
        layout.push_style_var_vec2("ItemSpacing", (4.0 * scale, 0.0))
        layout.push_style_var("IndentSpacing", INDENT_SPACING * scale)

        self._row_index = 0
        self._visible_node_order = []

        if layout.is_window_focused() and self._rename_node is None:
            if lf.ui.is_key_pressed(lf.ui.Key.F2) and self._selected_nodes:
                first_selected = next(iter(self._selected_nodes))
                self._rename_node = first_selected
                self._rename_buffer = first_selected
                self._rename_focus = True
            if lf.ui.is_key_pressed(lf.ui.Key.DELETE) and self._selected_nodes:
                self._delete_selected(scene)

        nodes = scene.get_nodes()
        splat_count = sum(1 for n in nodes if str(n.type).endswith("SPLAT"))

        models_label = tr("scene.models").format(splat_count)
        if layout.collapsing_header(models_label, default_open=True):
            if layout.begin_drag_drop_target():
                payload = layout.accept_drag_drop_payload("SCENE_NODE")
                if payload:
                    lf.reparent_node(payload, "")
                layout.end_drag_drop_target()

            if layout.begin_context_menu("##ModelsMenu"):
                if not AppState.has_trainer.value:
                    if layout.menu_item(tr("scene.add_group")):
                        lf.add_group("New Group", "")
                    layout.separator()
                layout.end_context_menu()

            for node in nodes:
                if node.parent_id == -1:
                    self._draw_node(layout, scene, node, 0, scale)

            if not nodes:
                layout.text_colored(tr("scene.no_models_loaded"), theme.palette.text_dim)

        self._committed_node_order = self._visible_node_order

        layout.pop_style_var()
        layout.pop_style_var()
        layout.pop_style_var()

    def _draw_node(self, layout, scene, node, depth, scale):
        if self._filter_text:
            filter_lower = self._filter_text.lower()
            if filter_lower not in node.name.lower():
                for child_id in node.children:
                    child = scene.get_node_by_id(child_id)
                    if child:
                        self._draw_node(layout, scene, child, depth + 1, scale)
                return

        layout.push_id(str(node.id))

        node_type = str(node.type).split(".")[-1]
        is_visible = node.visible
        has_children = len(node.children) > 0
        is_group = node_type == "GROUP"
        is_camera = node_type == "CAMERA"
        is_camera_group = node_type == "CAMERA_GROUP"
        is_dataset = node_type == "DATASET"
        is_splat = node_type == "SPLAT"
        is_pointcloud = node_type == "POINTCLOUD"

        parent_is_dataset = False
        if node.parent_id != -1:
            parent = scene.get_node_by_id(node.parent_id)
            if parent:
                parent_type = str(parent.type).split(".")[-1]
                parent_is_dataset = parent_type == "DATASET"

        is_deletable = not is_camera and not is_camera_group and not parent_is_dataset
        can_drag = node_type in ("SPLAT", "GROUP", "POINTCLOUD", "CROPBOX", "ELLIPSOID") and not parent_is_dataset

        is_selected = self._is_node_selected(node)
        self._visible_node_order.append(node.name)

        icon_size = ICON_SIZE_BASE * scale
        icon_spacing = ICON_SPACING_BASE * scale
        icon_sz = (icon_size, icon_size)

        self._draw_row_background(layout, is_selected, self._row_index)
        self._row_index += 1

        if self._scroll_to_node == node.name:
            layout.set_scroll_here_y(0.5)
            self._scroll_to_node = None

        if depth > 0:
            layout.indent(depth * INDENT_SPACING * scale)

        grip_tex = self._get_icon("grip")
        if grip_tex:
            grip_tint = GRIP_ACTIVE_TINT if can_drag else GRIP_INACTIVE_TINT
            layout.image(grip_tex, icon_sz, grip_tint)
            layout.same_line(0.0, icon_spacing)

        vis_tex = self._get_icon("visible") if is_visible else self._get_icon("hidden")
        vis_tint = VISIBLE_TINT if is_visible else HIDDEN_TINT
        if vis_tex and layout.image_button(f"##vis_{node.id}", vis_tex, icon_sz, vis_tint):
            lf.set_node_visibility(node.name, not is_visible)
        layout.same_line(0.0, icon_spacing)

        if is_camera:
            train_enabled = node.training_enabled
            train_tex = self._get_icon("camera")
            train_tint = TRAINING_ENABLED_TINT if train_enabled else TRAINING_DISABLED_TINT
            if train_tex and layout.image_button(f"##train_{node.id}", train_tex, icon_sz, train_tint):
                node.training_enabled = not train_enabled
            if layout.is_item_hovered():
                tip = tr("scene.training_enabled_tooltip") if train_enabled else tr("scene.training_disabled_tooltip")
                layout.set_tooltip(tip)
            layout.same_line(0.0, icon_spacing)

        if is_deletable:
            trash_tex = self._get_icon("trash")
            trash_tint = TRASH_SELECTED_TINT if is_selected else TRASH_DEFAULT_TINT
            if trash_tex and layout.image_button(f"##del_{node.id}", trash_tex, icon_sz, trash_tint):
                lf.remove_node(node.name, False)
            if layout.is_item_hovered():
                layout.set_tooltip(tr("scene.delete_node"))
            layout.same_line(0.0, icon_spacing)

        if self._rename_node == node.name:
            if self._rename_focus:
                layout.set_keyboard_focus_here()
                self._rename_focus = False
            layout.push_item_width(RENAME_INPUT_WIDTH * scale)
            entered, self._rename_buffer = layout.input_text_enter("##rename", self._rename_buffer)
            layout.pop_item_width()
            if entered and self._rename_buffer:
                lf.rename_node(node.name, self._rename_buffer)
                self._rename_node = None
            elif lf.ui.is_key_pressed(lf.ui.Key.ESCAPE):
                self._rename_node = None
        else:
            icon_name = NODE_TYPE_ICON_NAMES.get(node_type, "splat")
            type_tex = self._get_icon(icon_name)
            icon_color = NODE_TYPE_COLORS.get(node_type, DEFAULT_ICON_COLOR)
            if type_tex:
                layout.image(type_tex, icon_sz, icon_color)
                layout.same_line(0.0, icon_spacing)

            if is_camera and node.mask_path:
                mask_tex = self._get_icon("mask")
                if mask_tex:
                    inverted = lf.ui.get_invert_masks()
                    uv0 = (1.0, 1.0) if inverted else (0.0, 0.0)
                    uv1 = (0.0, 0.0) if inverted else (1.0, 1.0)
                    layout.image_uv(mask_tex, icon_sz, uv0, uv1, MASK_TINT)
                    layout.same_line(0.0, icon_spacing)

            layout.same_line(0.0, 2.0 * scale)

            training_disabled = is_camera and not node.training_enabled
            if training_disabled:
                theme = lf.ui.theme()
                layout.push_style_color("Text", theme.palette.text_dim)

            label = node.name
            if is_splat:
                label += f"  ({node.gaussian_count:,})"
            elif is_pointcloud:
                pc = node.point_cloud()
                if pc:
                    label += f"  ({pc.size:,})"

            label_x, label_y = layout.get_cursor_screen_pos()

            if has_children:
                flags = "OpenOnArrow|SpanAvailWidth"
                if is_group or is_dataset:
                    flags += "|DefaultOpen"
                if is_selected:
                    flags += "|Selected"
                is_open = layout.tree_node_ex(label, flags)
                node_clicked = layout.is_item_clicked()

                self._draw_context_menu(layout, scene, node, node_type, is_deletable, can_drag)
                self._handle_drag_drop(layout, node, can_drag)

                if node_clicked:
                    self._handle_click(node.name)

                if is_camera and layout.is_item_hovered() and layout.is_mouse_double_clicked(0):
                    if node.image_path:
                        self._open_camera_preview(scene, node.camera_uid)

                if is_open:
                    for child_id in node.children:
                        child = scene.get_node_by_id(child_id)
                        if child:
                            self._draw_node(layout, scene, child, depth + 1, scale)
                    layout.tree_pop()
            else:
                flags = "Leaf|NoTreePushOnOpen|SpanAvailWidth"
                if is_selected:
                    flags += "|Selected"
                layout.tree_node_ex(label, flags)

                self._draw_context_menu(layout, scene, node, node_type, is_deletable, can_drag)
                self._handle_drag_drop(layout, node, can_drag)

                if layout.is_item_clicked():
                    self._handle_click(node.name)

                if is_camera and layout.is_item_hovered() and layout.is_mouse_double_clicked(0):
                    if node.image_path:
                        self._open_camera_preview(scene, node.camera_uid)

            if training_disabled:
                layout.pop_style_color()
                tw, th = layout.calc_text_size(label)
                arrow_indent = layout.get_text_line_height() + 4.0 * scale
                line_y = label_y + th * 0.5
                layout.draw_window_line(label_x + arrow_indent, line_y,
                                        label_x + arrow_indent + tw, line_y,
                                        theme.palette.text_dim, 1.0)

        if depth > 0:
            layout.unindent(depth * INDENT_SPACING * scale)

        layout.pop_id()

    def _handle_drag_drop(self, layout, node, can_drag):
        node_type = str(node.type).split(".")[-1]
        if can_drag and layout.begin_drag_drop_source():
            layout.set_drag_drop_payload("SCENE_NODE", node.name)
            layout.label(tr("scene.move_node").format(node.name))
            layout.end_drag_drop_source()

        if node_type in ("GROUP", "SPLAT", "POINTCLOUD"):
            if layout.begin_drag_drop_target():
                payload = layout.accept_drag_drop_payload("SCENE_NODE")
                if payload and payload != node.name:
                    lf.reparent_node(payload, node.name)
                layout.end_drag_drop_target()

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

    def _get_range(self, a, b):
        order = self._committed_node_order
        try:
            ia, ib = order.index(a), order.index(b)
        except ValueError:
            return [b]
        lo, hi = min(ia, ib), max(ia, ib)
        return order[lo:hi + 1]

    def _delete_selected(self, scene):
        for name in list(self._selected_nodes):
            node = scene.get_node(name)
            if not node:
                continue
            ntype = str(node.type).split(".")[-1]
            parent_is_dataset = False
            if node.parent_id != -1:
                parent = scene.get_node_by_id(node.parent_id)
                if parent and str(parent.type).split(".")[-1] == "DATASET":
                    parent_is_dataset = True
            if ntype not in ("CAMERA", "CAMERA_GROUP") and not parent_is_dataset:
                lf.remove_node(name, False)

    def _draw_context_menu(self, layout, scene, node, node_type, is_deletable, can_drag):
        if not layout.begin_context_menu(f"##ctx_{node.name}"):
            return

        if node.name not in self._selected_nodes:
            lf.select_node(node.name)
            self._selected_nodes = {node.name}
            self._click_anchor = node.name

        if len(self._selected_nodes) > 1:
            self._draw_multi_context_menu(layout, scene)
            layout.end_context_menu()
            return

        if node_type == "CAMERA":
            if layout.menu_item(tr("scene.go_to_camera_view")):
                lf.ui.go_to_camera_view(node.camera_uid)
            layout.separator()
            if node.training_enabled:
                if layout.menu_item(tr("scene.disable_for_training")):
                    node.training_enabled = False
            else:
                if layout.menu_item(tr("scene.enable_for_training")):
                    node.training_enabled = True
            layout.end_context_menu()
            return

        if node_type == "CAMERA_GROUP":
            if layout.menu_item(tr("scene.enable_all_training")):
                for child_id in node.children:
                    child = scene.get_node_by_id(child_id)
                    if child and str(child.type).split(".")[-1] == "CAMERA":
                        child.training_enabled = True
            if layout.menu_item(tr("scene.disable_all_training")):
                for child_id in node.children:
                    child = scene.get_node_by_id(child_id)
                    if child and str(child.type).split(".")[-1] == "CAMERA":
                        child.training_enabled = False
            layout.end_context_menu()
            return

        if node_type == "DATASET":
            if layout.menu_item(tr("scene.delete")):
                lf.remove_node(node.name, False)
            layout.end_context_menu()
            return

        if node_type == "CROPBOX":
            if layout.menu_item(tr("common.apply")):
                lf.ui.apply_cropbox()
            layout.separator()
            if layout.menu_item(tr("scene.fit_to_scene")):
                lf.ui.fit_cropbox_to_scene(False)
            if layout.menu_item(tr("scene.fit_to_scene_trimmed")):
                lf.ui.fit_cropbox_to_scene(True)
            if layout.menu_item(tr("scene.reset_crop")):
                lf.ui.reset_cropbox()
            layout.separator()
            if layout.menu_item(tr("scene.delete")):
                lf.remove_node(node.name, False)
            layout.end_context_menu()
            return

        if node_type == "ELLIPSOID":
            if layout.menu_item(tr("common.apply")):
                lf.ui.apply_ellipsoid()
            layout.separator()
            if layout.menu_item(tr("scene.fit_to_scene")):
                lf.ui.fit_ellipsoid_to_scene(False)
            if layout.menu_item(tr("scene.fit_to_scene_trimmed")):
                lf.ui.fit_ellipsoid_to_scene(True)
            if layout.menu_item(tr("scene.reset_crop")):
                lf.ui.reset_ellipsoid()
            layout.separator()
            if layout.menu_item(tr("scene.delete")):
                lf.remove_node(node.name, False)
            layout.end_context_menu()
            return

        if node_type == "GROUP" and not AppState.has_trainer.value:
            if layout.menu_item(tr("scene.add_group_ellipsis")):
                lf.add_group("New Group", node.name)
            if layout.menu_item(tr("scene.merge_to_single_ply")):
                lf.ui.merge_group(node.name)
            layout.separator()

        if node_type in ("SPLAT", "POINTCLOUD"):
            if layout.menu_item(tr("scene.add_crop_box")):
                lf.ui.add_cropbox(node.name)
            if layout.menu_item(tr("scene.add_crop_ellipsoid")):
                lf.ui.add_ellipsoid(node.name)
            layout.separator()

        if is_deletable:
            if layout.menu_item(tr("scene.rename")):
                self._rename_node = node.name
                self._rename_buffer = node.name
                self._rename_focus = True

        if layout.menu_item(tr("scene.duplicate")):
            lf.ui.duplicate_node(node.name)

        if can_drag:
            if layout.begin_menu(tr("scene.move_to")):
                if layout.menu_item(tr("scene.move_to_root")):
                    lf.reparent_node(node.name, "")
                layout.separator()
                for other in scene.get_nodes():
                    other_type = str(other.type).split(".")[-1]
                    if other_type == "GROUP" and other.name != node.name:
                        if layout.menu_item(other.name):
                            lf.reparent_node(node.name, other.name)
                layout.end_menu()

        layout.separator()

        if is_deletable:
            if layout.menu_item(tr("scene.delete")):
                lf.remove_node(node.name, False)

        layout.end_context_menu()

    def _draw_multi_context_menu(self, layout, scene):
        types = set()
        deletable = []
        for name in self._selected_nodes:
            node = scene.get_node(name)
            if not node:
                continue
            ntype = str(node.type).split(".")[-1]
            types.add(ntype)
            parent_is_dataset = False
            if node.parent_id != -1:
                parent = scene.get_node_by_id(node.parent_id)
                if parent and str(parent.type).split(".")[-1] == "DATASET":
                    parent_is_dataset = True
            if ntype not in ("CAMERA", "CAMERA_GROUP") and not parent_is_dataset:
                deletable.append(name)

        if types == {"CAMERA"}:
            if layout.menu_item(tr("scene.enable_all_training")):
                for name in self._selected_nodes:
                    node = scene.get_node(name)
                    if node:
                        node.training_enabled = True
            if layout.menu_item(tr("scene.disable_all_training")):
                for name in self._selected_nodes:
                    node = scene.get_node(name)
                    if node:
                        node.training_enabled = False
        elif types == {"CAMERA_GROUP"}:
            if layout.menu_item(tr("scene.enable_all_training")):
                for name in self._selected_nodes:
                    grp = scene.get_node(name)
                    if not grp:
                        continue
                    for child_id in grp.children:
                        child = scene.get_node_by_id(child_id)
                        if child and str(child.type).split(".")[-1] == "CAMERA":
                            child.training_enabled = True
            if layout.menu_item(tr("scene.disable_all_training")):
                for name in self._selected_nodes:
                    grp = scene.get_node(name)
                    if not grp:
                        continue
                    for child_id in grp.children:
                        child = scene.get_node_by_id(child_id)
                        if child and str(child.type).split(".")[-1] == "CAMERA":
                            child.training_enabled = False

        if deletable:
            if types in ({"CAMERA"}, {"CAMERA_GROUP"}):
                layout.separator()
            label = f"{tr('scene.delete')} ({len(deletable)})"
            if layout.menu_item(label):
                for name in deletable:
                    lf.remove_node(name, False)

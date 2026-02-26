# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Selection Groups Panel - RmlUI implementation."""

import lichtfeld as lf

from .types import RmlPanel

def _tr(key):
    result = lf.ui.tr(key)
    return result if result else key


class SelectionGroupsPanel(RmlPanel):
    idname = "lfs.selection_groups"
    label = "Selection Groups"
    space = "MAIN_PANEL_TAB"
    order = 110
    rml_template = "rmlui/selection_groups.rml"
    rml_height_mode = "content"

    def __init__(self):
        self._collapsed = False
        self._prev_group_hash = None
        self._color_edit_group_id = None
        self._color_picker_needs_pos = False
        self._context_menu_group_id = None

    def on_load(self, doc):
        self.doc = doc

        header = doc.get_element_by_id("hdr-groups")
        if header:
            header.add_event_listener("click", self._on_toggle_section)

        btn = doc.get_element_by_id("btn-add-group")
        if btn:
            btn.add_event_listener("click", self._on_add_group)

        container = doc.get_element_by_id("groups-list")
        if container:
            container.add_event_listener("click", self._on_group_click)
            container.add_event_listener("mousedown", self._on_group_mousedown)

        ctx_menu = doc.get_element_by_id("context-menu")
        if ctx_menu:
            ctx_menu.add_event_listener("click", self._on_context_click)

        body = doc.get_element_by_id("body")
        if body:
            body.add_event_listener("click", self._on_body_click)

        self._update_labels()

    def on_update(self, doc):
        visible = lf.ui.get_active_tool() == "builtin.select" and lf.get_scene() is not None
        wrap = doc.get_element_by_id("content-wrap")
        if wrap:
            wrap.set_class("hidden", not visible)
        if not visible:
            return

        self._rebuild_groups(doc)

    def on_scene_changed(self, doc):
        self._prev_group_hash = None

    def draw_imgui(self, layout):
        if self._color_picker_needs_pos:
            layout.set_next_window_pos(layout.get_mouse_pos())
            layout.open_popup("##sg_color_picker")
            self._color_picker_needs_pos = False

        if self._color_edit_group_id is not None and layout.begin_popup("##sg_color_picker"):
            scene = lf.get_scene()
            if scene:
                groups = scene.selection_groups()
                group = next((g for g in groups if g.id == self._color_edit_group_id), None)
                if group:
                    changed, new_color = layout.color_picker3("##sg_picker", list(group.color))
                    if changed:
                        scene.set_selection_group_color(group.id, tuple(new_color))
                        self._prev_group_hash = None
            layout.end_popup()
        elif self._color_edit_group_id is not None:
            self._color_edit_group_id = None

    def _update_labels(self):
        if not self.doc:
            return
        title = self.doc.get_element_by_id("title-groups")
        if title:
            title.set_inner_rml(_tr("main_panel.selection_groups"))
        btn = self.doc.get_element_by_id("btn-add-group")
        if btn:
            btn.set_inner_rml(_tr("main_panel.add_group"))

    def _on_toggle_section(self, event):
        self._collapsed = not self._collapsed
        section = self.doc.get_element_by_id("groups-section")
        if section:
            section.set_class("collapsed", self._collapsed)
        arrow = self.doc.get_element_by_id("arrow-groups")
        if arrow:
            arrow.set_inner_rml("\u25B6" if self._collapsed else "\u25BC")

    def _on_add_group(self, event):
        scene = lf.get_scene()
        if scene:
            scene.add_selection_group("", (0.0, 0.0, 0.0))
            self._prev_group_hash = None

    def _compute_group_hash(self, scene):
        groups = scene.selection_groups()
        active_id = scene.active_selection_group
        parts = []
        for g in groups:
            r, gc, b = g.color
            parts.append(f"{g.id}:{g.name}:{g.count}:{g.locked}:{r:.2f}:{gc:.2f}:{b:.2f}")
        return f"{active_id}|{'|'.join(parts)}"

    def _rebuild_groups(self, doc):
        scene = lf.get_scene()
        if not scene:
            return

        scene.update_selection_group_counts()
        group_hash = self._compute_group_hash(scene)
        if group_hash == self._prev_group_hash:
            return
        self._prev_group_hash = group_hash

        groups = scene.selection_groups()
        active_id = scene.active_selection_group

        no_msg = doc.get_element_by_id("no-groups-msg")
        if no_msg:
            if groups:
                no_msg.set_class("hidden", True)
            else:
                no_msg.set_class("hidden", False)
                no_msg.set_inner_rml(_tr("main_panel.no_selection_groups"))

        container = doc.get_element_by_id("groups-list")
        if not container:
            return

        if not groups:
            container.set_inner_rml("")
            return

        parts = []
        for group in groups:
            gid = group.id
            is_active = gid == active_id
            is_locked = group.locked
            r, g, b = [int(c * 255) for c in group.color]
            icon_name = "locked" if is_locked else "unlocked"
            active_cls = " active" if is_active else ""
            label = f"{group.name} ({group.count})"

            parts.append(
                f'<div class="group-row{active_cls}" data-gid="{gid}">'
                f'  <div class="icon-btn lock-btn" data-action="lock" data-gid="{gid}">'
                f'    <img sprite="icon-{icon_name}" />'
                f'  </div>'
                f'  <div class="color-swatch" data-action="color" data-gid="{gid}"'
                f'       style="background-color: rgb({r},{g},{b})"></div>'
                f'  <span class="group-name" data-action="select" data-gid="{gid}">{label}</span>'
                f'</div>'
            )

        container.set_inner_rml("\n".join(parts))

    def _find_action_element(self, element):
        for _ in range(5):
            if element is None:
                return None, None
            action = element.get_attribute("data-action")
            if action:
                gid = element.get_attribute("data-gid", "-1")
                return action, int(gid)
            p = element.parent()
            if p is None:
                return None, None
            element = p
        return None, None

    def _on_group_click(self, event):
        target = event.target()
        if target is None:
            return
        action, gid = self._find_action_element(target)
        if action is None or gid < 0:
            return

        scene = lf.get_scene()
        if not scene:
            return

        if action == "lock":
            groups = scene.selection_groups()
            group = next((g for g in groups if g.id == gid), None)
            if group:
                scene.set_selection_group_locked(gid, not group.locked)
                self._prev_group_hash = None
        elif action == "color":
            self._color_edit_group_id = gid
            self._color_picker_needs_pos = True
        elif action == "select":
            scene.active_selection_group = gid
            self._prev_group_hash = None

    def _on_group_mousedown(self, event):
        if int(event.get_parameter("button", "0")) != 1:
            return
        target = event.target()
        if target is None:
            return
        _, gid = self._find_action_element(target)
        if gid is None or gid < 0:
            return
        self._show_context_menu(gid, event)

    def _show_context_menu(self, gid, event):
        scene = lf.get_scene()
        if not scene:
            return
        groups = scene.selection_groups()
        group = next((g for g in groups if g.id == gid), None)
        if not group:
            return

        self._context_menu_group_id = gid
        lock_label = _tr("selection_group.unlock") if group.locked else _tr("selection_group.lock")

        ctx = self.doc.get_element_by_id("context-menu")
        if not ctx:
            return

        ctx.set_inner_rml(
            f'<div class="context-menu-item" data-ctx-action="lock">{lock_label}</div>'
            f'<div class="context-menu-item" data-ctx-action="clear">{_tr("main_panel.clear")}</div>'
            f'<div class="context-menu-separator"></div>'
            f'<div class="context-menu-item" data-ctx-action="delete">{_tr("common.delete")}</div>'
        )
        mouse_x = event.get_parameter("mouse_x", "0")
        mouse_y = event.get_parameter("mouse_y", "0")
        ctx.set_property("left", f"{mouse_x}px")
        ctx.set_property("top", f"{mouse_y}px")
        ctx.set_class("visible", True)

    def _on_context_click(self, event):
        target = event.target()
        if target is None:
            return
        action = target.get_attribute("data-ctx-action")
        if not action:
            return

        ctx = self.doc.get_element_by_id("context-menu")
        if ctx:
            ctx.set_class("visible", False)

        scene = lf.get_scene()
        gid = self._context_menu_group_id
        if not scene or gid is None:
            return

        if action == "lock":
            groups = scene.selection_groups()
            group = next((g for g in groups if g.id == gid), None)
            if group:
                scene.set_selection_group_locked(gid, not group.locked)
        elif action == "clear":
            scene.clear_selection_group(gid)
        elif action == "delete":
            scene.remove_selection_group(gid)
        self._prev_group_hash = None
        self._context_menu_group_id = None

    def _on_body_click(self, event):
        ctx = self.doc.get_element_by_id("context-menu")
        if ctx:
            ctx.set_class("visible", False)
        self._context_menu_group_id = None


def register():
    lf.ui.register_rml_panel(SelectionGroupsPanel)
    lf.ui.set_panel_parent("lfs.selection_groups", "lfs.rendering")


def unregister():
    lf.ui.set_panel_enabled("lfs.selection_groups", False)

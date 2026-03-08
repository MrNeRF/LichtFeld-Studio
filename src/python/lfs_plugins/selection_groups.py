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
        self._context_menu_group_id = None
        self._picker_click_handled = False

    @classmethod
    def poll(cls, context):
        del context
        return lf.ui.get_active_tool() == "builtin.select" and lf.get_scene() is not None

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

        self._popup_el = doc.get_element_by_id("color-picker-popup")
        if self._popup_el:
            self._popup_el.add_event_listener("click", self._on_popup_click)

        self._picker_el = doc.get_element_by_id("color-picker-el")
        if self._picker_el:
            self._picker_el.add_event_listener("change", self._on_picker_change)

        body = doc.get_element_by_id("body")
        if body:
            body.add_event_listener("click", self._on_body_click)

        section = doc.get_element_by_id("groups-section")
        arrow = doc.get_element_by_id("arrow-groups")
        if section:
            from . import rml_widgets as w
            w.sync_section_state(section, not self._collapsed, header, arrow)

        self._update_labels()

    def on_update(self, doc):
        visible = lf.ui.get_active_tool() == "builtin.select" and lf.get_scene() is not None
        wrap = doc.get_element_by_id("content-wrap")
        if wrap:
            wrap.set_class("hidden", not visible)
        if not visible:
            return

        action = lf.ui.poll_context_menu()
        if action and self._context_menu_group_id is not None:
            self._handle_context_action(action, self._context_menu_group_id)
            self._context_menu_group_id = None

        self._rebuild_groups(doc)

    def on_scene_changed(self, doc):
        self._prev_group_hash = None

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
        header = self.doc.get_element_by_id("hdr-groups")
        section = self.doc.get_element_by_id("groups-section")
        arrow = self.doc.get_element_by_id("arrow-groups")
        if section:
            from . import rml_widgets as w
            w.animate_section_toggle(section, not self._collapsed, arrow, header_element=header)

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
            self._show_color_picker(gid, event)
        elif action == "select":
            scene.active_selection_group = gid
            self._prev_group_hash = None

    def _show_color_picker(self, gid, event):
        if self._color_edit_group_id == gid:
            self._hide_picker()
            return

        self._picker_click_handled = True
        self._color_edit_group_id = gid

        scene = lf.get_scene()
        if not scene:
            return
        groups = scene.selection_groups()
        group = next((g for g in groups if g.id == gid), None)
        if not group or not self._picker_el or not self._popup_el:
            return

        r, g, b = group.color
        self._picker_el.set_attribute("red", str(float(r)))
        self._picker_el.set_attribute("green", str(float(g)))
        self._picker_el.set_attribute("blue", str(float(b)))

        mx = event.get_parameter("mouse_x", "0")
        my = event.get_parameter("mouse_y", "0")
        self._popup_el.set_property("left", f"{mx}px")
        self._popup_el.set_property("top", f"{int(float(my)) + 2}px")
        self._popup_el.set_class("visible", True)

    def _hide_picker(self):
        if self._popup_el:
            self._popup_el.set_class("visible", False)
        self._color_edit_group_id = None

    def _on_picker_change(self, event):
        if self._color_edit_group_id is None:
            return
        scene = lf.get_scene()
        if not scene:
            return

        r = float(event.get_parameter("red", "0"))
        g = float(event.get_parameter("green", "0"))
        b = float(event.get_parameter("blue", "0"))
        scene.set_selection_group_color(self._color_edit_group_id, (r, g, b))
        self._prev_group_hash = None

    def _on_popup_click(self, event):
        event.stop_propagation()

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
        items = [
            {"label": lock_label, "action": "lock"},
            {"label": _tr("main_panel.clear"), "action": "clear"},
            {"label": _tr("common.delete"), "action": "delete", "separator_before": True},
        ]
        sx, sy = lf.ui.get_mouse_screen_pos()
        lf.ui.show_context_menu(items, sx, sy)

    def _handle_context_action(self, action, gid):
        scene = lf.get_scene()
        if not scene:
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

    def _on_body_click(self, event):
        if self._picker_click_handled:
            self._picker_click_handled = False
            return

        self._hide_picker()


def register():
    lf.ui.register_rml_panel(SelectionGroupsPanel)
    lf.ui.set_panel_parent("lfs.selection_groups", "lfs.rendering")


def unregister():
    lf.ui.set_panel_enabled("lfs.selection_groups", False)

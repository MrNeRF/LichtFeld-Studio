# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Selection group list in the Rendering panel's Selection & Overlays section."""

import lichtfeld as lf

from .ui import RuntimeState

GROUPS_LIST_ID = "selection-groups-list"


class SelectionGroupsSection:
    """Binds the selection group list into a host panel's data model.

    The host owns the color picker; ``show_color_picker(gid, event)`` opens it for a group.
    """

    def __init__(self, show_color_picker):
        self._show_color_picker = show_color_picker
        self._handle = None
        self._has_groups = False
        self._prev_group_hash = None
        self._last_scene_generation = None
        self._last_selection_generation = None

    def bind(self, model):
        model.bind_func("label_selection_groups", lambda: lf.ui.tr("main_panel.selection_groups"))
        model.bind_func("label_add_selection_group", lambda: lf.ui.tr("main_panel.add_group"))
        model.bind_func("label_no_selection_groups", lambda: lf.ui.tr("main_panel.no_selection_groups"))
        model.bind_func("show_no_selection_groups", lambda: not self._has_groups)
        model.bind_record_list("selection_groups")
        model.bind_event("selection_group_add", self._on_add_group)

    def attach(self, handle):
        self._handle = handle

    def mount(self, doc):
        container = doc.get_element_by_id(GROUPS_LIST_ID)
        if container:
            container.add_event_listener("click", self._on_group_click)
            container.add_event_listener("mousedown", self._on_group_mousedown)
        self.invalidate()

    def unmount(self):
        self._handle = None

    def invalidate(self):
        self._prev_group_hash = None
        self._last_scene_generation = None
        self._last_selection_generation = None

    def sync(self, visible=True):
        # Counting copies the selection mask to the host, so a hidden list stays stale until shown.
        if not visible:
            self.invalidate()
            return False
        scene_generation = RuntimeState.scene_generation.value
        selection_generation = RuntimeState.selection_generation.value
        if (
            self._prev_group_hash is not None
            and scene_generation == self._last_scene_generation
            and selection_generation == self._last_selection_generation
        ):
            return False
        self._last_scene_generation = scene_generation
        self._last_selection_generation = selection_generation
        return self._rebuild_groups()

    def group_color(self, gid):
        scene = lf.get_scene()
        group = _find_group(scene, gid) if scene else None
        return tuple(group.color) if group else None

    def set_group_color(self, gid, color):
        scene = lf.get_scene()
        if scene:
            scene.set_selection_group_color(gid, color)
            self._mark_groups_changed()

    def _mark_groups_changed(self):
        self._prev_group_hash = None
        self._rebuild_groups()

    def _on_add_group(self, _handle=None, _event=None, _args=None):
        scene = lf.get_scene()
        if scene:
            scene.add_selection_group("", (0.0, 0.0, 0.0))
            self._mark_groups_changed()

    def _set_has_groups(self, has_groups):
        has_groups = bool(has_groups)
        if has_groups == self._has_groups:
            return
        self._has_groups = has_groups
        if self._handle:
            self._handle.dirty("show_no_selection_groups")

    def _rebuild_groups(self):
        scene = lf.get_scene()
        if not self._handle:
            return False
        if not scene:
            if self._prev_group_hash == "":
                return False
            self._prev_group_hash = ""
            self._set_has_groups(False)
            self._handle.update_record_list("selection_groups", [])
            return True

        scene.update_selection_group_counts()
        group_hash = _compute_group_hash(scene)
        if group_hash == self._prev_group_hash:
            return False
        self._prev_group_hash = group_hash

        groups = scene.selection_groups()
        active_id = scene.active_selection_group
        self._set_has_groups(groups)

        records = []
        for group in groups:
            r, g, b = [int(c * 255) for c in group.color]
            records.append({
                "gid": str(group.id),
                "active": group.id == active_id,
                "lock_sprite": f"icon-{'locked' if group.locked else 'unlocked'}",
                "color_css": f"rgb({r},{g},{b})",
                "label": f"{group.name} ({group.count:,})",
            })

        self._handle.update_record_list("selection_groups", records)
        return True

    def _on_group_click(self, event):
        target = event.target()
        if target is None:
            return
        action, gid = _find_action_element(target)
        if action is None or gid < 0:
            return

        scene = lf.get_scene()
        if not scene:
            return

        if action == "lock":
            group = _find_group(scene, gid)
            if group:
                scene.set_selection_group_locked(gid, not group.locked)
                self._mark_groups_changed()
        elif action == "color":
            self._show_color_picker(gid, event)
        elif action == "select":
            scene.active_selection_group = gid
            self._mark_groups_changed()

    def _on_group_mousedown(self, event):
        if int(event.get_parameter("button", "0")) != 1:
            return
        target = event.target()
        if target is None:
            return
        _, gid = _find_action_element(target)
        if gid is None or gid < 0:
            return
        self._show_context_menu(gid)

    def _show_context_menu(self, gid):
        scene = lf.get_scene()
        group = _find_group(scene, gid) if scene else None
        if not group:
            return

        tr = lf.ui.tr
        lock_label = tr("selection_group.unlock") if group.locked else tr("selection_group.lock")
        items = [
            {"label": lock_label, "action": "lock"},
            {"label": tr("main_panel.clear"), "action": "clear"},
            {"label": tr("common.delete"), "action": "delete", "separator_before": True},
        ]
        sx, sy = lf.ui.get_mouse_screen_pos()
        lf.ui.show_context_menu(
            items,
            sx,
            sy,
            lambda action, group_id=gid: self._handle_context_action(action, group_id),
        )

    def _handle_context_action(self, action, gid):
        scene = lf.get_scene()
        if not scene:
            return

        if action == "lock":
            group = _find_group(scene, gid)
            if group:
                scene.set_selection_group_locked(gid, not group.locked)
        elif action == "clear":
            scene.clear_selection_group(gid)
        elif action == "delete":
            scene.remove_selection_group(gid)
        self._mark_groups_changed()


def _find_group(scene, gid):
    return next((g for g in scene.selection_groups() if g.id == gid), None)


def _compute_group_hash(scene):
    parts = []
    for group in scene.selection_groups():
        r, g, b = group.color
        parts.append(f"{group.id}:{group.name}:{group.count}:{group.locked}:{r:.2f}:{g:.2f}:{b:.2f}")
    return f"{scene.active_selection_group}|{'|'.join(parts)}"


def _find_action_element(element):
    while element is not None:
        action = element.get_attribute("data-action")
        if action:
            return action, int(element.get_attribute("data-gid", "-1"))
        element = element.parent()
    return None, None

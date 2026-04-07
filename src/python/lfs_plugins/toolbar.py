# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Viewport toolbars rendered from a retained RmlUI data model."""

from pathlib import Path
from urllib.parse import quote

from .histogram_support import histogram_mode_available
from .tools import ToolRegistry


_TOOLBAR_HIDDEN_STATES = ("running", "paused", "stopping", "completed")
_RML_PATH_SAFE_CHARS = "/:._-~"

_toolbar_controller = None


def _icon_src(icon_name):
    """Build icon src path relative to the RML document in assets/rmlui/."""
    if "." in icon_name:
        return f"../icon/{icon_name}"
    return f"../icon/{icon_name}.png"


def _tool_icon_src(tool_def):
    """Resolve builtin toolbar icons or plugin-provided toolbar icons."""
    plugin_path = getattr(tool_def, "plugin_path", "") or ""
    if plugin_path:
        candidate = Path(plugin_path) / "icons" / f"{tool_def.icon}.png"
        if candidate.exists():
            return quote(candidate.as_posix(), safe=_RML_PATH_SAFE_CHARS)
    return _icon_src(tool_def.icon)


def _tool_selected(tool_def, active_tool_id, context):
    selected_fn = getattr(tool_def, "selected", None)
    if callable(selected_fn):
        try:
            return bool(selected_fn(context))
        except Exception:
            return False
    return active_tool_id == tool_def.id


def _tooltip_text(label, shortcut=""):
    if label and shortcut:
        return f"{label} ({shortcut})"
    return label or ""


def _button_record(button_id, action, value, icon_src, *,
                   tooltip_key="", tooltip_text="", selected=False, enabled=True):
    return {
        "button_id": button_id,
        "action": action,
        "value": value,
        "icon_src": icon_src,
        "tooltip_key": tooltip_key,
        "tooltip_text": tooltip_text,
        "selected": selected,
        "enabled": enabled,
    }


class _GizmoToolbarController:
    _TOOL_LOCALE_KEYS = {
        "builtin.select": "toolbar.selection",
        "builtin.translate": "toolbar.translate",
        "builtin.rotate": "toolbar.rotate",
        "builtin.scale": "toolbar.scale",
        "builtin.mirror": "toolbar.mirror",
        "builtin.brush": "toolbar.painting",
        "builtin.align": "toolbar.align_3point",
    }

    _SUBMODE_LOCALE_KEYS = {
        "builtin.select:centers": "toolbar.brush_selection",
        "builtin.select:rectangle": "toolbar.rect_selection",
        "builtin.select:polygon": "toolbar.polygon_selection",
        "builtin.select:lasso": "toolbar.lasso_selection",
        "builtin.select:rings": "toolbar.ring_selection",
        "builtin.translate:local": "toolbar.local_space",
        "builtin.translate:world": "toolbar.world_space",
        "builtin.rotate:local": "toolbar.local_space",
        "builtin.rotate:world": "toolbar.world_space",
        "builtin.scale:local": "toolbar.local_space",
        "builtin.scale:world": "toolbar.world_space",
        "builtin.mirror:x": "toolbar.mirror_x",
        "builtin.mirror:y": "toolbar.mirror_y",
        "builtin.mirror:z": "toolbar.mirror_z",
    }

    _PIVOT_LOCALE_KEYS = {
        "origin": "toolbar.origin_pivot",
        "bounds": "toolbar.bounds_center_pivot",
    }

    _TRANSFORM_TOOL_IDS = {"builtin.translate", "builtin.rotate", "builtin.scale"}
    _TRANSFORM_SPACE_IDS = {"local": 0, "world": 1}
    _PIVOT_IDS = {"origin": 0, "bounds": 1}

    def __init__(self):
        self.reset()

    def reset(self):
        self._was_hidden = False

    def snapshot(self):
        import lichtfeld as lf
        from .op_context import get_context
        from .ui.state import AppState

        hidden = AppState.trainer_state.value in _TOOLBAR_HIDDEN_STATES
        if hidden:
            if not self._was_hidden:
                ToolRegistry.clear_active()
            self._was_hidden = True
            return {
                "show_gizmo_toolbar": False,
                "show_submode_toolbar": False,
                "show_pivot_toolbar": False,
                "gizmo_buttons": [],
                "submode_buttons": [],
                "pivot_buttons": [],
            }

        self._was_hidden = False

        context = get_context()
        active_tool_id = lf.ui.get_active_tool() or ""
        tool_defs = ToolRegistry.get_all()
        tool_def = ToolRegistry.get(active_tool_id) if active_tool_id else None

        gizmo_buttons = [
            self._tool_button_record(tool_def_item, active_tool_id, context)
            for tool_def_item in tool_defs
        ]
        submode_buttons = self._build_submode_records(active_tool_id, tool_def)
        pivot_buttons = self._build_pivot_records(tool_def)

        return {
            "show_gizmo_toolbar": bool(gizmo_buttons),
            "show_submode_toolbar": bool(submode_buttons),
            "show_pivot_toolbar": bool(pivot_buttons),
            "gizmo_buttons": gizmo_buttons,
            "submode_buttons": submode_buttons,
            "pivot_buttons": pivot_buttons,
        }

    def _tool_button_record(self, tool_def, active_tool_id, context):
        tooltip_key = self._TOOL_LOCALE_KEYS.get(tool_def.id, "")
        return _button_record(
            f"tool-{tool_def.id}",
            "tool",
            tool_def.id,
            _tool_icon_src(tool_def),
            tooltip_key=tooltip_key,
            tooltip_text="" if tooltip_key else _tooltip_text(tool_def.label, tool_def.shortcut),
            selected=_tool_selected(tool_def, active_tool_id, context),
            enabled=tool_def.can_activate(context),
        )

    def _build_submode_records(self, active_tool_id, tool_def):
        import lichtfeld as lf

        if tool_def is None or not tool_def.submodes:
            return []

        current_space = lf.ui.get_transform_space()
        active_submode = lf.ui.get_active_submode()
        is_transform_tool = active_tool_id in self._TRANSFORM_TOOL_IDS
        is_mirror_tool = active_tool_id == "builtin.mirror"

        if not active_submode and not is_transform_tool and not is_mirror_tool:
            active_submode = tool_def.submodes[0].id
            lf.ui.set_selection_mode(active_submode)

        records = []
        for mode in tool_def.submodes:
            tooltip_key = self._SUBMODE_LOCALE_KEYS.get(f"{active_tool_id}:{mode.id}", "")
            selected = False
            if is_transform_tool:
                selected = current_space == self._TRANSFORM_SPACE_IDS.get(mode.id, -1)
            elif not is_mirror_tool:
                selected = active_submode == mode.id

            records.append(
                _button_record(
                    f"sub-{mode.id}",
                    "submode",
                    mode.id,
                    _icon_src(mode.icon) if mode.icon else "",
                    tooltip_key=tooltip_key,
                    tooltip_text="" if tooltip_key else _tooltip_text(mode.label, mode.shortcut),
                    selected=selected,
                )
            )
        return records

    def _build_pivot_records(self, tool_def):
        import lichtfeld as lf

        if tool_def is None or not tool_def.pivot_modes:
            return []

        current_pivot = lf.ui.get_pivot_mode()
        records = []
        for mode in tool_def.pivot_modes:
            tooltip_key = self._PIVOT_LOCALE_KEYS.get(mode.id, "")
            records.append(
                _button_record(
                    f"pivot-{mode.id}",
                    "pivot",
                    mode.id,
                    _icon_src(mode.icon) if mode.icon else "",
                    tooltip_key=tooltip_key,
                    tooltip_text="" if tooltip_key else mode.label,
                    selected=current_pivot == self._PIVOT_IDS.get(mode.id, -1),
                )
            )
        return records

    def dispatch(self, action, value):
        import lichtfeld as lf
        from .op_context import get_context

        if action == "tool":
            tool_def = ToolRegistry.get(value)
            if tool_def is None or not tool_def.can_activate(get_context()):
                return
            if lf.ui.get_active_tool() == value:
                ToolRegistry.clear_active()
            else:
                ToolRegistry.set_active(value)
            return

        if action == "submode":
            active_tool_id = lf.ui.get_active_tool()
            if active_tool_id == "builtin.mirror":
                lf.ui.execute_mirror(value)
            elif active_tool_id in self._TRANSFORM_TOOL_IDS:
                transform_space = self._TRANSFORM_SPACE_IDS.get(value, -1)
                if transform_space >= 0:
                    lf.ui.set_transform_space(transform_space)
            else:
                lf.ui.set_selection_mode(value)
            return

        if action == "pivot":
            pivot_mode = self._PIVOT_IDS.get(value, -1)
            if pivot_mode >= 0:
                lf.ui.set_pivot_mode(pivot_mode)


class _UtilityToolbarController:
    _CAMERA_MODE_SPECS = (
        ("camera-orbit", "orbit", "Orbit Camera"),
        ("camera-fpv", "fpv", "FPV Camera"),
    )
    _RENDER_MODE_SPECS = (
        ("blob", "splats", "toolbar.splat_rendering"),
        ("dots-diagonal", "points", "toolbar.point_cloud"),
        ("ring", "rings", "toolbar.gaussian_rings"),
        ("circle-dot", "centers", "toolbar.center_markers"),
    )

    def reset(self):
        pass

    def snapshot(self):
        import lichtfeld as lf
        from .histogram_support import histogram_mode_available

        try:
            camera_mode = str(lf.get_camera_navigation_mode()).lower()
        except Exception:
            camera_mode = "orbit"
        if camera_mode == "fly":
            camera_mode = "fpv"

        has_render_manager = True
        try:
            mode_map = {
                "splats": lf.RenderMode.SPLATS,
                "points": lf.RenderMode.POINTS,
                "rings": lf.RenderMode.RINGS,
                "centers": lf.RenderMode.CENTERS,
            }
            render_mode = lf.get_render_mode()
        except Exception:
            has_render_manager = False
            mode_map = {}
            render_mode = None

        is_fullscreen = lf.is_fullscreen() if hasattr(lf, "is_fullscreen") else False
        camera_mode_buttons = [
            _button_record(
                f"util-camera-{mode_id}",
                "set_camera_navigation_mode",
                mode_id,
                _icon_src(icon_name),
                tooltip_text=label,
                selected=camera_mode == mode_id,
            )
            for icon_name, mode_id, label in self._CAMERA_MODE_SPECS
        ]
        primary_buttons = [
            _button_record("util-home", "home", "", _icon_src("home"),
                           tooltip_key="toolbar.home"),
            _button_record(
                "util-fullscreen",
                "fullscreen",
                "",
                _icon_src("arrows-minimize" if is_fullscreen else "arrows-maximize"),
                tooltip_key="toolbar.fullscreen",
                selected=is_fullscreen,
            ),
            _button_record("util-toggle-ui", "toggle_ui", "", _icon_src("layout-off"),
                           tooltip_key="toolbar.toggle_ui"),
        ]

        render_mode_buttons = []
        projection_buttons = []
        utility_extra_buttons = []
        utility_bottom_buttons = []
        if has_render_manager:
            for icon_name, mode_id, tooltip_key in self._RENDER_MODE_SPECS:
                render_mode_buttons.append(
                    _button_record(
                        f"util-render-{mode_id}",
                        "set_render_mode",
                        mode_id,
                        _icon_src(icon_name),
                        tooltip_key=tooltip_key,
                        selected=render_mode == mode_map.get(mode_id),
                    )
                )

            is_ortho = lf.is_orthographic()
            projection_buttons.append(
                _button_record(
                    "util-projection",
                    "toggle_projection",
                    "",
                    _icon_src("box" if is_ortho else "perspective"),
                    tooltip_key="toolbar.orthographic" if is_ortho else "toolbar.perspective",
                    selected=is_ortho,
                )
            )

            seq_visible = lf.ui.is_sequencer_visible()
            utility_extra_buttons.append(
                _button_record(
                    "util-sequencer",
                    "toggle_sequencer",
                    "",
                    _icon_src("video"),
                    tooltip_key="toolbar.sequencer",
                    selected=seq_visible,
                )
            )

        if histogram_mode_available(lf.ui.context()):
            utility_bottom_buttons.append(
                _button_record(
                    "util-histogram",
                    "toggle_panel",
                    "lfs.histogram",
                    _icon_src("histogram.png"),
                    tooltip_key="toolbar.histogram",
                    selected=lf.ui.is_panel_enabled("lfs.histogram"),
                )
            )

        return {
            "camera_mode_buttons": camera_mode_buttons,
            "show_render_controls": has_render_manager,
            "primary_buttons": primary_buttons,
            "render_mode_buttons": render_mode_buttons,
            "projection_buttons": projection_buttons,
            "utility_extra_buttons": utility_extra_buttons,
            "utility_bottom_buttons": utility_bottom_buttons,
        }

    def dispatch(self, action, value):
        import lichtfeld as lf

        if action == "set_camera_navigation_mode":
            lf.set_camera_navigation_mode(value)
            return
        if action == "home":
            lf.reset_camera()
            return
        if action == "fullscreen":
            lf.toggle_fullscreen()
            return
        if action == "toggle_ui":
            lf.toggle_ui()
            return
        if action == "toggle_projection":
            lf.set_orthographic(not lf.is_orthographic())
            return
        if action == "toggle_sequencer":
            lf.ui.set_sequencer_visible(not lf.ui.is_sequencer_visible())
            return
        if action == "toggle_panel":
            if value == "lfs.histogram" and not histogram_mode_available(lf.ui.context()):
                lf.ui.set_panel_enabled(value, False)
                return
            lf.ui.set_panel_enabled(value, not lf.ui.is_panel_enabled(value))
            return
        if action != "set_render_mode":
            return

        mode_map = {
            "splats": lf.RenderMode.SPLATS,
            "points": lf.RenderMode.POINTS,
            "rings": lf.RenderMode.RINGS,
            "centers": lf.RenderMode.CENTERS,
        }
        render_mode = mode_map.get(value)
        if render_mode is not None:
            lf.set_render_mode(render_mode)


class _ViewportToolbarController:
    _BOOLEAN_FIELDS = (
        "show_render_controls",
        "show_gizmo_toolbar",
        "show_submode_toolbar",
        "show_pivot_toolbar",
    )
    _RECORD_FIELDS = (
        "camera_mode_buttons",
        "utility_primary_buttons",
        "render_mode_buttons",
        "projection_buttons",
        "utility_extra_buttons",
        "utility_bottom_buttons",
        "gizmo_buttons",
        "submode_buttons",
        "pivot_buttons",
    )

    def __init__(self):
        self._gizmo = _GizmoToolbarController()
        self._utility = _UtilityToolbarController()
        self.reset()

    def reset(self):
        self._handle = None
        self._record_cache = {name: None for name in self._RECORD_FIELDS}
        self._show_render_controls = False
        self._show_gizmo_toolbar = False
        self._show_submode_toolbar = False
        self._show_pivot_toolbar = False
        self._gizmo.reset()
        self._utility.reset()

    def bind_model(self, model):
        for field in self._BOOLEAN_FIELDS:
            model.bind_func(field, lambda name=field: getattr(self, f"_{name}"))
        for field in self._RECORD_FIELDS:
            model.bind_record_list(field)
        model.bind_event("toolbar_action", self._on_toolbar_action)

    def attach_handle(self, handle):
        self._handle = handle
        self._record_cache = {name: None for name in self._RECORD_FIELDS}
        if self._handle:
            self._handle.dirty_all()

    def update(self, doc):
        if doc is None:
            return
        if self._handle is None:
            return

        utility_state = self._utility.snapshot()
        gizmo_state = self._gizmo.snapshot()

        self._sync_flag("show_render_controls", utility_state["show_render_controls"])
        self._sync_flag("show_gizmo_toolbar", gizmo_state["show_gizmo_toolbar"])
        self._sync_flag("show_submode_toolbar", gizmo_state["show_submode_toolbar"])
        self._sync_flag("show_pivot_toolbar", gizmo_state["show_pivot_toolbar"])

        self._sync_records("camera_mode_buttons", utility_state["camera_mode_buttons"])
        self._sync_records("utility_primary_buttons", utility_state["primary_buttons"])
        self._sync_records("render_mode_buttons", utility_state["render_mode_buttons"])
        self._sync_records("projection_buttons", utility_state["projection_buttons"])
        self._sync_records("utility_extra_buttons", utility_state["utility_extra_buttons"])
        self._sync_records("utility_bottom_buttons", utility_state["utility_bottom_buttons"])
        self._sync_records("gizmo_buttons", gizmo_state["gizmo_buttons"])
        self._sync_records("submode_buttons", gizmo_state["submode_buttons"])
        self._sync_records("pivot_buttons", gizmo_state["pivot_buttons"])

    def _sync_flag(self, name, value):
        current = getattr(self, f"_{name}")
        if current == value:
            return
        setattr(self, f"_{name}", value)
        if self._handle:
            self._handle.dirty(name)

    def _sync_records(self, name, records):
        if self._record_cache.get(name) == records:
            return
        self._record_cache[name] = records
        if self._handle:
            self._handle.update_record_list(name, records)

    def _on_toolbar_action(self, _handle, _event, args):
        if not args:
            return
        action = str(args[0])
        value = str(args[1]) if len(args) > 1 else ""
        if action in {"tool", "submode", "pivot"}:
            self._gizmo.dispatch(action, value)
        else:
            self._utility.dispatch(action, value)


def _ensure_controller():
    global _toolbar_controller
    if _toolbar_controller is None:
        _toolbar_controller = _ViewportToolbarController()


def bind_overlay_model(model):
    _ensure_controller()
    _toolbar_controller.bind_model(model)


def attach_overlay_model_handle(handle):
    _ensure_controller()
    _toolbar_controller.attach_handle(handle)


def update_overlay(doc):
    _ensure_controller()
    _toolbar_controller.update(doc)


def reset_overlay_state():
    if _toolbar_controller is not None:
        _toolbar_controller.reset()

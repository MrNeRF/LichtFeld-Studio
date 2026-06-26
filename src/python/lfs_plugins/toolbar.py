# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Viewport toolbars rendered from a retained RmlUI data model."""

from pathlib import Path
from urllib.parse import quote

from .depth_view_controls import DepthViewControlsController
from .histogram_support import histogram_mode_available
from .selection_controls import SelectionControlsController
from .tools import ToolRegistry
from .transform_controls import TransformControlsController
from .ui import RuntimeState
from .viewport_export_controls import ViewportExportControlsController

try:
    from .ui import native_value as _native_store_value
except Exception:
    def _native_store_value(_field, fallback):
        return fallback


_TOOLBAR_HIDDEN_STATES = ("running", "paused", "stopping", "completed")
_RML_PATH_SAFE_CHARS = "/:._-~"
_OVERLAY_DOC_KEY_ATTR = "data-viewport-toolbar-doc-key"

_toolbar_controller = None
_MISSING = object()


def __lfs_after_reload__(runtime):
    from . import overlays

    if overlays._document_controller is not None:
        overlays._document_controller.reset()
    runtime.ui.request_redraw()


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


def _ui_label(key, fallback=""):
    if not key:
        return fallback or ""
    try:
        import lichtfeld as lf

        tr = getattr(lf.ui, "tr", None)
        if not callable(tr):
            return fallback or ""
        value = tr(key)
    except Exception:
        return fallback or ""
    if value and value != key:
        return value
    return fallback or ""


def _keymap_shortcut(action_id, fallback=""):
    if not action_id:
        return fallback or ""
    try:
        import lichtfeld as lf

        keymap = getattr(lf, "keymap", None)
        action_enum = getattr(keymap, "Action", None)
        mode_enum = getattr(keymap, "ToolMode", None)
        if keymap is None or action_enum is None or mode_enum is None:
            return fallback or ""
        action = getattr(action_enum, action_id, None)
        mode = getattr(mode_enum, "GLOBAL", None)
        if action is None or mode is None:
            return fallback or ""
        is_bound = getattr(keymap, "is_bound", None)
        if callable(is_bound) and not is_bound(action, mode):
            return fallback or ""
        describe = getattr(keymap, "get_trigger_description", None)
        if callable(describe):
            return describe(action, mode) or fallback or ""
    except Exception:
        return fallback or ""
    return fallback or ""


def _panel_enabled(panel_id):
    try:
        import lichtfeld as lf

        getter = getattr(getattr(lf, "ui", None), "is_panel_enabled", None)
        if callable(getter):
            return bool(getter(panel_id))
    except Exception:
        pass
    return False


def _button_record(button_id, action, value, icon_src, *,
                   tooltip_key="", tooltip_text="", action_id="",
                   shortcut_text="", selected=False, enabled=True):
    return {
        "button_id": button_id,
        "action": action,
        "value": value,
        "icon_src": icon_src,
        "tooltip_text": _ui_label(tooltip_key, tooltip_text),
        "action_id": action_id,
        "shortcut_text": _keymap_shortcut(action_id, shortcut_text),
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
        "builtin.cropbox": "toolbar.crop_box",
        "builtin.align": "toolbar.align_3point",
    }

    _TOOL_ACTIONS = {
        "builtin.select": "TOOL_SELECT",
        "builtin.translate": "TOOL_TRANSLATE",
        "builtin.rotate": "TOOL_ROTATE",
        "builtin.scale": "TOOL_SCALE",
        "builtin.mirror": "TOOL_MIRROR",
        "builtin.align": "TOOL_ALIGN",
    }

    _SUBMODE_LOCALE_KEYS = {
        "builtin.select:centers": "toolbar.brush_selection",
        "builtin.select:rectangle": "toolbar.rect_selection",
        "builtin.select:polygon": "toolbar.polygon_selection",
        "builtin.select:lasso": "toolbar.lasso_selection",
        "builtin.select:rings": "toolbar.ring_selection",
        "builtin.select:color": "toolbar.color_selection",
        "builtin.select:box": "toolbar.box_selection",
        "builtin.select:sphere": "toolbar.sphere_selection",
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

    _SELECTION_MODE_ACTIONS = {
        "centers": "SELECT_MODE_CENTERS",
        "rectangle": "SELECT_MODE_RECTANGLE",
        "polygon": "SELECT_MODE_POLYGON",
        "lasso": "SELECT_MODE_LASSO",
        "rings": "SELECT_MODE_RINGS",
        "color": "SELECT_MODE_COLOR",
        "box": "SELECT_MODE_BOX",
        "sphere": "SELECT_MODE_SPHERE",
    }

    _PIVOT_LOCALE_KEYS = {
        "origin": "toolbar.origin_pivot",
        "bounds": "toolbar.bounds_center_pivot",
    }

    _TRANSFORM_TOOL_IDS = {"builtin.translate", "builtin.rotate", "builtin.scale"}
    _MIRROR_TOOL_ID = "builtin.mirror"
    _CROP_TOOL_ID = "builtin.cropbox"
    _HORIZONTAL_TOOL_IDS = {"builtin.select", _MIRROR_TOOL_ID, _CROP_TOOL_ID, *_TRANSFORM_TOOL_IDS}
    _TRANSFORM_SPACE_IDS = {"local": 0, "world": 1}
    _PIVOT_IDS = {"origin": 0, "bounds": 1}
    _CROP_OBJECT_SHAPES = ("box", "ellipsoid")
    _CROP_TRANSFORM_GIZMOS = ("translate", "rotate", "scale")
    _SELECTION_VOLUME_MODES = {"box", "sphere"}

    def __init__(self):
        self.reset()

    def reset(self):
        self._was_hidden = False

    def _active_selection_submode(self):
        import lichtfeld as lf

        active_submode = _native_store_value("active_submode", _MISSING)
        if active_submode is _MISSING:
            get_active_submode = getattr(lf.ui, "get_active_submode", None)
            active_submode = get_active_submode() if callable(get_active_submode) else ""
        return active_submode or ""

    def _selection_volume_active(self, active_tool_id):
        return (
            active_tool_id == "builtin.select"
            and self._active_selection_submode() in self._SELECTION_VOLUME_MODES
        )

    def snapshot(self):
        import lichtfeld as lf
        from .op_context import get_context

        hidden = RuntimeState.trainer_state.value in _TOOLBAR_HIDDEN_STATES
        if hidden:
            if not self._was_hidden:
                ToolRegistry.clear_active()
            self._was_hidden = True
            return {
                "show_transform_toolbar": False,
                "show_mirror_toolbar": False,
                "show_crop_toolbar": False,
                "show_selection_volume_gizmos": False,
                "show_transform_space_controls": False,
                "show_transform_pivot_controls": False,
                "selection_group_buttons": [],
                "selection_mode_buttons": [],
                "selection_volume_gizmo_buttons": [],
                "transform_group_buttons": [],
                "transform_tool_buttons": [],
                "mirror_group_buttons": [],
                "crop_group_buttons": [],
                "crop_object_buttons": [],
                "crop_transform_buttons": [],
                "crop_action_buttons": [],
                "gizmo_buttons": [],
                "submode_buttons": [],
                "pivot_buttons": [],
            }

        self._was_hidden = False

        context = get_context()
        active_tool_id = _native_store_value("active_tool", _MISSING)
        if active_tool_id is _MISSING:
            active_tool_id = lf.ui.get_active_tool() or ""
        else:
            active_tool_id = active_tool_id or ""
        selected_getter = getattr(lf, "get_selected_node_names", None)
        try:
            selected_nodes = tuple(selected_getter() or []) if callable(selected_getter) else ()
        except Exception:
            selected_nodes = ()
        tool_defs = ToolRegistry.get_all()
        tool_def = ToolRegistry.get(active_tool_id) if active_tool_id else None
        select_tool_def = ToolRegistry.get("builtin.select")
        mirror_tool_def = ToolRegistry.get(self._MIRROR_TOOL_ID)
        crop_tool_def = ToolRegistry.get(self._CROP_TOOL_ID)

        transform_tool_defs = [
            tool_def_item
            for tool_def_item in tool_defs
            if getattr(tool_def_item, "id", "") in self._TRANSFORM_TOOL_IDS
        ]
        gizmo_buttons = [
            self._tool_button_record(tool_def_item, active_tool_id, context)
            for tool_def_item in tool_defs
            if (
                tool_def_item.id != "builtin.select" and
                tool_def_item.id != self._MIRROR_TOOL_ID and
                tool_def_item.id != self._CROP_TOOL_ID and
                tool_def_item.id not in self._TRANSFORM_TOOL_IDS
            )
        ]
        selection_group_buttons, selection_mode_buttons = self._build_selection_records(
            select_tool_def,
            active_tool_id,
            context,
        )
        transform_group_buttons, transform_tool_buttons = self._build_transform_records(
            transform_tool_defs,
            active_tool_id,
            context,
        )
        mirror_group_buttons = self._build_mirror_records(mirror_tool_def, active_tool_id, context)
        crop_group_buttons = self._build_crop_group_records(crop_tool_def, active_tool_id, context)
        crop_tool_active = active_tool_id == self._CROP_TOOL_ID
        selection_volume_active = self._selection_volume_active(active_tool_id)
        crop_object_buttons = self._build_crop_object_records(active_tool_id) if crop_tool_active else []
        crop_transform_buttons = self._build_crop_transform_records(active_tool_id) if crop_tool_active else []
        crop_action_buttons = self._build_crop_action_records(active_tool_id) if crop_tool_active else []
        selection_volume_gizmo_buttons = self._build_selection_volume_gizmo_records(active_tool_id)
        submode_buttons = self._build_submode_records(active_tool_id, tool_def)
        pivot_buttons = self._build_pivot_records(tool_def)

        return {
            "show_transform_toolbar": (
                active_tool_id in self._TRANSFORM_TOOL_IDS and
                not selected_nodes and
                bool(transform_group_buttons) and
                bool(transform_tool_buttons)
            ),
            "show_mirror_toolbar": active_tool_id == self._MIRROR_TOOL_ID and bool(submode_buttons),
            "show_crop_toolbar": crop_tool_active and bool(crop_object_buttons),
            "show_selection_volume_gizmos": selection_volume_active and bool(selection_volume_gizmo_buttons),
            "show_transform_space_controls": active_tool_id in self._TRANSFORM_TOOL_IDS and bool(submode_buttons),
            "show_transform_pivot_controls": active_tool_id in self._TRANSFORM_TOOL_IDS and bool(pivot_buttons),
            "selection_group_buttons": selection_group_buttons,
            "selection_mode_buttons": selection_mode_buttons,
            "selection_volume_gizmo_buttons": selection_volume_gizmo_buttons,
            "transform_group_buttons": transform_group_buttons,
            "transform_tool_buttons": transform_tool_buttons,
            "mirror_group_buttons": mirror_group_buttons,
            "crop_group_buttons": crop_group_buttons,
            "crop_object_buttons": crop_object_buttons,
            "crop_transform_buttons": crop_transform_buttons,
            "crop_action_buttons": crop_action_buttons,
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
            tooltip_text=tool_def.label,
            action_id=self._TOOL_ACTIONS.get(tool_def.id, ""),
            shortcut_text=tool_def.shortcut,
            selected=_tool_selected(tool_def, active_tool_id, context),
            enabled=tool_def.can_activate(context),
        )

    def _build_selection_records(self, tool_def, active_tool_id, context):
        import lichtfeld as lf

        if tool_def is None or not tool_def.submodes:
            return [], []

        enabled = tool_def.can_activate(context)
        active_submode = _native_store_value("active_submode", _MISSING)
        if active_submode is _MISSING:
            get_active_submode = getattr(lf.ui, "get_active_submode", None)
            active_submode = get_active_submode() if callable(get_active_submode) else ""
        active_submode = active_submode or ""
        if active_tool_id == "builtin.select" and not active_submode:
            active_submode = tool_def.submodes[0].id
            set_selection_mode = getattr(lf.ui, "set_selection_mode", None)
            if callable(set_selection_mode):
                set_selection_mode(active_submode)

        mode_buttons = []
        for mode in tool_def.submodes:
            tooltip_key = self._SUBMODE_LOCALE_KEYS.get(f"builtin.select:{mode.id}", "")
            selected = active_tool_id == "builtin.select" and active_submode == mode.id
            mode_buttons.append(
                _button_record(
                    f"selection-{mode.id}",
                    "selection_mode",
                    mode.id,
                    _icon_src(mode.icon) if mode.icon else "",
                    tooltip_key=tooltip_key,
                    tooltip_text=mode.label,
                    action_id=self._SELECTION_MODE_ACTIONS.get(mode.id, ""),
                    shortcut_text=mode.shortcut,
                    selected=selected,
                    enabled=enabled,
                )
            )

        group_button = _button_record(
            "group-selection",
            "tool",
            "builtin.select",
            _tool_icon_src(tool_def),
            tooltip_key=self._TOOL_LOCALE_KEYS.get(tool_def.id, ""),
            tooltip_text=tool_def.label,
            action_id="TOOL_SELECT",
            shortcut_text=getattr(tool_def, "shortcut", ""),
            selected=active_tool_id == "builtin.select",
            enabled=enabled,
        )
        return [group_button], mode_buttons

    def _build_transform_records(self, tool_defs, active_tool_id, context):
        if not tool_defs:
            return [], []

        tool_buttons = [
            self._tool_button_record(tool_def, active_tool_id, context)
            for tool_def in tool_defs
        ]
        active_button = next((b for b in tool_buttons if b["selected"]), None)
        fallback = next((b for b in tool_buttons if b["enabled"]), tool_buttons[0])
        display_button = active_button or fallback
        group_button = _button_record(
            "group-transform",
            "tool",
            display_button["value"],
            display_button["icon_src"],
            tooltip_text="Transform Tools",
            action_id=display_button["action_id"],
            shortcut_text=display_button["shortcut_text"],
            selected=active_button is not None,
            enabled=any(b["enabled"] for b in tool_buttons),
        )
        return [group_button], tool_buttons

    def _build_mirror_records(self, tool_def, active_tool_id, context):
        if tool_def is None:
            return []
        return [
            _button_record(
                "group-mirror",
                "tool",
                self._MIRROR_TOOL_ID,
                _tool_icon_src(tool_def),
                tooltip_key=self._TOOL_LOCALE_KEYS.get(self._MIRROR_TOOL_ID, ""),
                tooltip_text=tool_def.label,
                action_id=self._TOOL_ACTIONS.get(self._MIRROR_TOOL_ID, ""),
                shortcut_text=getattr(tool_def, "shortcut", ""),
                selected=active_tool_id == self._MIRROR_TOOL_ID,
                enabled=tool_def.can_activate(context),
            )
        ]

    def _build_crop_group_records(self, tool_def, active_tool_id, context):
        if tool_def is None:
            return []
        return [
            _button_record(
                "group-crop",
                "tool",
                self._CROP_TOOL_ID,
                _tool_icon_src(tool_def),
                tooltip_key=self._TOOL_LOCALE_KEYS.get(self._CROP_TOOL_ID, ""),
                tooltip_text=tool_def.label,
                selected=active_tool_id == self._CROP_TOOL_ID,
                enabled=tool_def.can_activate(context),
            )
        ]

    def _build_crop_object_records(self, active_tool_id):
        active = active_tool_id == self._CROP_TOOL_ID
        active_shape = self._active_crop_shape() if active else "box"
        return [
            _button_record(
                "crop-object-box",
                "crop_object",
                "box",
                _icon_src("cropbox"),
                tooltip_key="toolbar.crop_box",
                tooltip_text="Crop Box",
                selected=active and active_shape == "box",
                enabled=active,
            ),
            _button_record(
                "crop-object-ellipsoid",
                "crop_object",
                "ellipsoid",
                _icon_src("sphere"),
                tooltip_key="toolbar.ellipsoid",
                tooltip_text="Ellipsoid",
                selected=active and active_shape == "ellipsoid",
                enabled=active,
            ),
        ]

    def _build_crop_transform_records(self, active_tool_id):
        import lichtfeld as lf

        active = active_tool_id == self._CROP_TOOL_ID
        active_gizmo = lf.ui.get_gizmo_type() if active and hasattr(lf.ui, "get_gizmo_type") else ""
        if active and not active_gizmo:
            active_gizmo = "translate"
        return self._build_gizmo_operation_records(active, active_gizmo)

    def _build_gizmo_operation_records(self, active, active_gizmo):
        specs = (
            ("translate", "translation", "toolbar.translate", "Translate"),
            ("rotate", "rotation", "toolbar.rotate", "Rotate"),
            ("scale", "bounds", "toolbar.scale", "Bounds"),
        )
        return [
            _button_record(
                f"crop-transform-{mode}",
                "crop_transform",
                mode,
                _icon_src(icon),
                tooltip_key=tooltip_key,
                tooltip_text=label,
                selected=active and active_gizmo == mode,
                enabled=active,
            )
            for mode, icon, tooltip_key, label in specs
        ]

    def _build_selection_volume_gizmo_records(self, active_tool_id):
        if not self._selection_volume_active(active_tool_id):
            return []
        import lichtfeld as lf

        active_gizmo = ""
        if hasattr(lf.ui, "get_crop_tool_operation"):
            active_gizmo = lf.ui.get_crop_tool_operation()
        return self._build_gizmo_operation_records(True, active_gizmo or "scale")

    def _build_crop_action_records(self, active_tool_id):
        active = active_tool_id == self._CROP_TOOL_ID
        return [
            _button_record(
                "crop-trim",
                "crop_trim",
                "",
                _icon_src("arrows-minimize"),
                tooltip_key="scene.fit_to_scene_trimmed",
                tooltip_text="Fit to Scene (Trimmed)",
                enabled=active,
            ),
            _button_record(
                "crop-apply",
                "crop_apply",
                "",
                _icon_src("check"),
                tooltip_key="common.apply",
                tooltip_text="Apply",
                enabled=active,
            )
        ]

    def _active_crop_shape(self):
        try:
            import lichtfeld as lf

            get_shape = getattr(lf.ui, "get_crop_tool_shape", None)
            if callable(get_shape):
                shape = get_shape()
                if shape in self._CROP_OBJECT_SHAPES:
                    return shape
        except Exception:
            pass
        return "box"

    def _activate_crop_tool(self, gizmo_type="translate"):
        import lichtfeld as lf

        lf.ui.set_active_operator(self._CROP_TOOL_ID, gizmo_type)

    def _build_submode_records(self, active_tool_id, tool_def):
        import lichtfeld as lf

        if tool_def is None or not tool_def.submodes:
            return []
        if active_tool_id == "builtin.select":
            return []

        current_space = _native_store_value("transform_space", _MISSING)
        if current_space is _MISSING:
            current_space = lf.ui.get_transform_space()
        active_submode = _native_store_value("active_submode", _MISSING)
        if active_submode is _MISSING:
            active_submode = lf.ui.get_active_submode()
        active_submode = active_submode or ""
        is_transform_tool = active_tool_id in self._TRANSFORM_TOOL_IDS
        is_mirror_tool = active_tool_id == self._MIRROR_TOOL_ID

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
                    tooltip_text=mode.label,
                    shortcut_text=mode.shortcut,
                    selected=selected,
                )
            )
        return records

    def _build_pivot_records(self, tool_def):
        import lichtfeld as lf

        if tool_def is None or not tool_def.pivot_modes:
            return []

        current_pivot = _native_store_value("pivot_mode", _MISSING)
        if current_pivot is _MISSING:
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
                    tooltip_text=mode.label,
                    selected=current_pivot == self._PIVOT_IDS.get(mode.id, -1),
                )
            )
        return records

    def dispatch(self, action, value):
        import lichtfeld as lf
        from .op_context import get_context

        if action == "selection_mode":
            tool_def = ToolRegistry.get("builtin.select")
            if tool_def is None or not tool_def.can_activate(get_context()):
                return
            if lf.ui.get_active_tool() != "builtin.select":
                ToolRegistry.set_active("builtin.select")
            set_selection_mode = getattr(lf.ui, "set_selection_mode", None)
            if callable(set_selection_mode):
                set_selection_mode(value)
            return

        if action == "tool":
            tool_def = ToolRegistry.get(value)
            if tool_def is None or not tool_def.can_activate(get_context()):
                return
            if lf.ui.get_active_tool() == value:
                ToolRegistry.clear_active()
            else:
                ToolRegistry.set_active(value)
            return

        if action == "crop_object":
            if value in self._CROP_OBJECT_SHAPES:
                current_gizmo = lf.ui.get_gizmo_type() if hasattr(lf.ui, "get_gizmo_type") else ""
                set_shape = getattr(lf.ui, "set_crop_tool_shape", None)
                if callable(set_shape):
                    set_shape(value)
                self._activate_crop_tool(current_gizmo or "translate")
            return

        if action == "crop_transform":
            if value in self._CROP_TRANSFORM_GIZMOS:
                if self._selection_volume_active(lf.ui.get_active_tool()):
                    set_operation = getattr(lf.ui, "set_crop_tool_operation", None)
                    if callable(set_operation):
                        set_operation(value)
                    return
                self._activate_crop_tool(value)
            return

        if action == "crop_trim":
            fit_crop = getattr(lf.ui, "fit_crop_tool", None)
            if callable(fit_crop):
                fit_crop(True)
            return

        if action == "crop_apply":
            apply_crop = getattr(lf.ui, "apply_crop_tool", None)
            if callable(apply_crop):
                apply_crop()
            return

        if action == "submode":
            active_tool_id = lf.ui.get_active_tool()
            if active_tool_id == self._MIRROR_TOOL_ID:
                lf.ui.execute_mirror(value)
            elif active_tool_id in self._TRANSFORM_TOOL_IDS:
                transform_space = self._TRANSFORM_SPACE_IDS.get(value, -1)
                if transform_space >= 0:
                    lf.ui.set_transform_space(transform_space)
                    try:
                        RuntimeState.transform_space.value = transform_space
                    except Exception:
                        pass
            else:
                lf.ui.set_selection_mode(value)
            return

        if action == "pivot":
            pivot_mode = self._PIVOT_IDS.get(value, -1)
            if pivot_mode >= 0:
                lf.ui.set_pivot_mode(pivot_mode)
                try:
                    RuntimeState.pivot_mode.value = pivot_mode
                except Exception:
                    pass

    def clear_active_horizontal_tool(self):
        import lichtfeld as lf

        active_tool_id = _native_store_value("active_tool", _MISSING)
        if active_tool_id is _MISSING:
            active_tool_id = lf.ui.get_active_tool() or ""
        else:
            active_tool_id = active_tool_id or ""
        if active_tool_id not in self._HORIZONTAL_TOOL_IDS:
            return

        ToolRegistry.clear_active()
        try:
            RuntimeState.active_tool.value = ""
            RuntimeState.active_submode.value = ""
        except Exception:
            pass


class _UtilityToolbarController:
    _INPUT_SETTINGS_PANEL_ID = "lfs.input_settings"
    _PLUGIN_MARKETPLACE_PANEL_ID = "lfs.plugin_marketplace"
    _CAMERA_MODE_SPECS = (
        ("camera-orbit", "orbit", "Orbit Camera"),
        ("world", "trackball", "Free Orbit Camera"),
        ("camera-fpv", "fpv", "Fly Camera"),
    )
    _PRIMARY_ACTIONS = {
        "home": "CAMERA_RESET_HOME",
        "focus_selection": "CAMERA_FOCUS_SELECTION",
    }

    def __init__(self, viewport_export_visible=None):
        self._viewport_export_visible = viewport_export_visible

    def _is_viewport_export_visible(self):
        if not callable(self._viewport_export_visible):
            return False
        try:
            return bool(self._viewport_export_visible())
        except Exception:
            return False

    def snapshot(self):
        import lichtfeld as lf
        from .histogram_support import histogram_mode_available

        try:
            camera_mode = str(lf.get_camera_navigation_mode()).lower()
        except Exception:
            camera_mode = "orbit"
        if camera_mode == "fly":
            camera_mode = "fpv"
        if camera_mode == "turntable":
            camera_mode = "trackball"

        has_render_manager = True
        try:
            lf.get_render_mode()
        except Exception:
            has_render_manager = False

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
                           tooltip_key="toolbar.home",
                           tooltip_text="Home",
                           action_id=self._PRIMARY_ACTIONS["home"]),
            _button_record(
                "util-focus-selection",
                "focus_selection",
                "",
                _icon_src("focus-selection"),
                tooltip_key="toolbar.focus_selection",
                tooltip_text="Focus Selection",
                action_id=self._PRIMARY_ACTIONS["focus_selection"],
            ),
        ]

        utility_extra_buttons = [
            _button_record(
                "util-input-settings",
                "toggle_panel",
                self._INPUT_SETTINGS_PANEL_ID,
                _icon_src("settings"),
                tooltip_key="window.input_settings",
                tooltip_text="Input Settings",
                selected=_panel_enabled(self._INPUT_SETTINGS_PANEL_ID),
            ),
            _button_record(
                "util-viewport-export",
                "toggle_viewport_export",
                "",
                _icon_src("sequencer/export"),
                tooltip_key="toolbar.viewport_export",
                tooltip_text="Viewport Export",
                selected=self._is_viewport_export_visible(),
            ),
            _button_record(
                "util-plugin-marketplace",
                "toggle_panel",
                self._PLUGIN_MARKETPLACE_PANEL_ID,
                _icon_src("puzzle"),
                tooltip_key="menu.tools.plugin_marketplace",
                tooltip_text="Plugin Marketplace",
                selected=_panel_enabled(self._PLUGIN_MARKETPLACE_PANEL_ID),
            )
        ]
        utility_bottom_buttons = []
        if has_render_manager:
            seq_visible = lf.ui.is_sequencer_visible()
            # The sequencer is disabled while training is active (the native
            # SequencerPanel gates on EditorContext::isToolsDisabled). Reflect
            # that in the button so it greys out instead of appearing live but
            # doing nothing on press, matching the editing-tool buttons.
            seq_enabled = RuntimeState.trainer_state.value not in _TOOLBAR_HIDDEN_STATES
            utility_extra_buttons.append(
                _button_record(
                    "util-sequencer",
                    "toggle_sequencer",
                    "",
                    _icon_src("video"),
                    tooltip_key="toolbar.sequencer",
                    tooltip_text="Sequencer",
                    selected=seq_visible,
                    enabled=seq_enabled,
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
                    tooltip_text="Histogram",
                    selected=_panel_enabled("lfs.histogram"),
                )
            )

        return {
            "camera_mode_buttons": camera_mode_buttons,
            "primary_buttons": primary_buttons,
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
        if action == "focus_selection":
            lf.focus_selection()
            return
        if action == "toggle_sequencer":
            if RuntimeState.trainer_state.value in _TOOLBAR_HIDDEN_STATES:
                return
            lf.ui.set_sequencer_visible(not lf.ui.is_sequencer_visible())
            return
        if action == "toggle_panel":
            if value == "lfs.histogram" and not histogram_mode_available(lf.ui.context()):
                lf.ui.set_panel_enabled(value, False)
                return
            lf.ui.set_panel_enabled(value, not _panel_enabled(value))
            return


class _ViewportToolbarController:
    _BOOLEAN_FIELDS = (
        "show_transform_toolbar",
        "show_mirror_toolbar",
        "show_crop_toolbar",
        "show_selection_volume_gizmos",
        "show_transform_space_controls",
        "show_transform_pivot_controls",
    )
    _RECORD_FIELDS = (
        "camera_mode_buttons",
        "utility_primary_buttons",
        "utility_extra_buttons",
        "utility_bottom_buttons",
        "selection_group_buttons",
        "selection_mode_buttons",
        "selection_volume_gizmo_buttons",
        "transform_group_buttons",
        "transform_tool_buttons",
        "mirror_group_buttons",
        "crop_group_buttons",
        "crop_object_buttons",
        "crop_transform_buttons",
        "crop_action_buttons",
        "gizmo_buttons",
        "submode_buttons",
        "pivot_buttons",
    )

    def __init__(self):
        self._gizmo = _GizmoToolbarController()
        self._depth_view_controls = DepthViewControlsController()
        self._viewport_export_controls = ViewportExportControlsController(
            self._on_viewport_export_visibility_changed
        )
        self._utility = _UtilityToolbarController(lambda: self._viewport_export_controls.visible)
        self._selection_controls = SelectionControlsController()
        self._transform_controls = TransformControlsController()
        self.reset()

    def reset(self):
        self._handle = None
        self._mounted_doc_key = None
        self._current_doc = None
        self._next_doc_key = getattr(self, "_next_doc_key", 1)
        self._record_cache = {name: None for name in self._RECORD_FIELDS}
        self._last_toolbar_signature = None
        self._show_transform_toolbar = False
        self._show_mirror_toolbar = False
        self._show_crop_toolbar = False
        self._show_selection_volume_gizmos = False
        self._show_transform_space_controls = False
        self._show_transform_pivot_controls = False
        self._gizmo.reset()
        self._depth_view_controls.unmount()
        self._viewport_export_controls.unmount()
        self._selection_controls.unmount()
        self._transform_controls.unmount()

    def bind_model(self, model):
        for field in self._BOOLEAN_FIELDS:
            model.bind_func(field, lambda name=field: getattr(self, f"_{name}"))
        for field in self._RECORD_FIELDS:
            model.bind_record_list(field)
        model.bind_event("toolbar_action", self._on_toolbar_action)
        self._depth_view_controls.bind_model(model)
        self._viewport_export_controls.bind_model(model)
        self._selection_controls.bind_model(model)
        self._transform_controls.bind_model(model)

    def attach_handle(self, handle):
        self._handle = handle
        self._record_cache = {name: None for name in self._RECORD_FIELDS}
        self._last_toolbar_signature = None
        if self._handle:
            self._handle.dirty_all()

    def update(self, doc):
        dirty_sources = []
        if doc is None:
            return dirty_sources
        if self._handle is None:
            return dirty_sources

        can_update_tool_overlays = hasattr(doc, "get_element_by_id")
        self._current_doc = doc if can_update_tool_overlays else None
        mount_key = self._mount_key(doc) if can_update_tool_overlays else None
        if mount_key is not None and mount_key != self._mounted_doc_key:
            self._mounted_doc_key = mount_key
            self._depth_view_controls.mount(doc)
            self._viewport_export_controls.mount(doc)
            self._record_cache = {name: None for name in self._RECORD_FIELDS}
            self._last_toolbar_signature = None
            self._selection_controls.mount(doc)
            self._transform_controls.mount(doc)
            dirty_sources.append("mount")

        if self._sync_toolbar_state(doc):
            dirty_sources.append("records")
        if can_update_tool_overlays:
            depth_dirty = self._depth_view_controls.update(doc)
            if depth_dirty:
                dirty_sources.append(f"depth_view_controls:{depth_dirty}")
            viewport_export_dirty = self._viewport_export_controls.update(doc)
            if viewport_export_dirty:
                dirty_sources.append(f"viewport_export_controls:{viewport_export_dirty}")
            if self._viewport_export_controls.visible:
                self._hide_tool_overlay(doc, "depth-view-block")
                self._hide_tool_overlay(doc, "selection-block")
                self._hide_tool_overlay(doc, "transform-block")
            elif self._depth_view_controls.visible:
                self._hide_tool_overlay(doc, "selection-block")
                self._hide_tool_overlay(doc, "transform-block")
            else:
                selection_dirty = self._selection_controls.update(doc)
                if selection_dirty:
                    dirty_sources.append(f"selection_controls:{selection_dirty}")
                transform_dirty = self._transform_controls.update(doc)
                if transform_dirty:
                    dirty_sources.append("transform_controls")
        return dirty_sources

    def _hide_tool_overlay(self, doc, element_id):
        element = doc.get_element_by_id(element_id)
        if element:
            element.set_class("hidden", True)

    def _sync_tool_overlays_now(self):
        doc = self._current_doc
        if doc is None or not hasattr(doc, "get_element_by_id"):
            return

        self._depth_view_controls.update(doc)
        self._viewport_export_controls.update(doc)
        if self._viewport_export_controls.visible:
            self._hide_tool_overlay(doc, "depth-view-block")
            self._hide_tool_overlay(doc, "selection-block")
            self._hide_tool_overlay(doc, "transform-block")
        elif self._depth_view_controls.visible:
            self._hide_tool_overlay(doc, "selection-block")
            self._hide_tool_overlay(doc, "transform-block")

        try:
            import lichtfeld as lf

            request_redraw = getattr(lf.ui, "request_redraw", None)
            if callable(request_redraw):
                request_redraw()
        except Exception:
            pass

    def _on_viewport_export_visibility_changed(self):
        self._last_toolbar_signature = None
        self._sync_toolbar_state()
        self._sync_tool_overlays_now()

    def _mount_key(self, doc):
        body = doc.get_element_by_id("overlay-body")
        if body is None:
            return None

        root = doc.get_element_by_id("dm-root") or body
        key = root.get_attribute(_OVERLAY_DOC_KEY_ATTR, "")
        if key:
            return key

        key = str(self._next_doc_key)
        self._next_doc_key += 1
        root.set_attribute(_OVERLAY_DOC_KEY_ATTR, key)
        return key

    def _sync_toolbar_state(self, doc=None):
        if self._handle is None:
            return False
        signature = self._toolbar_signature()
        if signature == self._last_toolbar_signature:
            return False
        self._last_toolbar_signature = signature

        utility_state = self._utility.snapshot()
        gizmo_state = self._gizmo.snapshot()

        dirty = False
        dirty |= self._sync_flag("show_transform_toolbar", gizmo_state["show_transform_toolbar"])
        dirty |= self._sync_flag("show_mirror_toolbar", gizmo_state["show_mirror_toolbar"])
        dirty |= self._sync_flag("show_crop_toolbar", gizmo_state["show_crop_toolbar"])
        dirty |= self._sync_flag("show_selection_volume_gizmos", gizmo_state["show_selection_volume_gizmos"])
        dirty |= self._sync_flag("show_transform_space_controls", gizmo_state["show_transform_space_controls"])
        dirty |= self._sync_flag("show_transform_pivot_controls", gizmo_state["show_transform_pivot_controls"])

        dirty |= self._sync_records("camera_mode_buttons", utility_state["camera_mode_buttons"])
        dirty |= self._sync_records("utility_primary_buttons", utility_state["primary_buttons"])
        dirty |= self._sync_records("utility_extra_buttons", utility_state["utility_extra_buttons"])
        dirty |= self._sync_records("utility_bottom_buttons", utility_state["utility_bottom_buttons"])
        dirty |= self._sync_records("selection_group_buttons", gizmo_state["selection_group_buttons"], doc)
        dirty |= self._sync_records("selection_mode_buttons", gizmo_state["selection_mode_buttons"], doc)
        dirty |= self._sync_records("selection_volume_gizmo_buttons", gizmo_state["selection_volume_gizmo_buttons"])
        dirty |= self._sync_records("transform_group_buttons", gizmo_state["transform_group_buttons"])
        dirty |= self._sync_records("transform_tool_buttons", gizmo_state["transform_tool_buttons"])
        dirty |= self._sync_records("mirror_group_buttons", gizmo_state["mirror_group_buttons"])
        dirty |= self._sync_records("crop_group_buttons", gizmo_state["crop_group_buttons"])
        dirty |= self._sync_records("crop_object_buttons", gizmo_state["crop_object_buttons"])
        dirty |= self._sync_records("crop_transform_buttons", gizmo_state["crop_transform_buttons"])
        dirty |= self._sync_records("crop_action_buttons", gizmo_state["crop_action_buttons"])
        dirty |= self._sync_records("gizmo_buttons", gizmo_state["gizmo_buttons"])
        dirty |= self._sync_records("submode_buttons", gizmo_state["submode_buttons"])
        dirty |= self._sync_records("pivot_buttons", gizmo_state["pivot_buttons"])
        return dirty

    def _sync_flag(self, name, value):
        current = getattr(self, f"_{name}")
        if current == value:
            return False
        setattr(self, f"_{name}", value)
        if self._handle:
            self._handle.dirty(name)
        return True

    def _sync_records(self, name, records, doc=None):
        previous = self._record_cache.get(name)
        if previous == records:
            return False
        if self._patch_selection_record_state(name, previous, records, doc):
            self._record_cache[name] = records
            return True
        self._record_cache[name] = records
        if self._handle:
            self._handle.update_record_list(name, records)
        return True

    def _patch_selection_record_state(self, name, previous, records, doc):
        if doc is None or previous is None:
            return False
        if name not in {"selection_group_buttons", "selection_mode_buttons"}:
            return False
        if len(previous) != len(records):
            return False

        stable_fields = {
            "button_id",
            "action",
            "value",
            "tooltip_text",
            "action_id",
            "enabled",
        }
        for old, new in zip(previous, records):
            for field in stable_fields:
                if old.get(field) != new.get(field):
                    return False

        patched = False
        for record in records:
            button_id = record.get("button_id", "")
            if not button_id:
                return False
            try:
                buttons = doc.query_selector_all(f"#{button_id}")
            except Exception:
                return False
            if not buttons:
                return False

            for button in buttons:
                button.set_class("selected", bool(record.get("selected", False)))
                if name == "selection_group_buttons":
                    img = button.query_selector("img")
                    if img is not None:
                        img.set_attribute("src", record.get("icon_src", ""))
                patched = True

        return patched

    def _toolbar_signature(self):
        import lichtfeld as lf

        try:
            trainer_state = RuntimeState.trainer_state.value
        except Exception:
            trainer_state = ""

        def call(default, getter, *args):
            if not callable(getter):
                return default
            try:
                return getter(*args)
            except Exception:
                return default

        active_tool = _native_store_value("active_tool", _MISSING)
        if active_tool is _MISSING:
            active_tool = call("", getattr(lf.ui, "get_active_tool", None))
        active_tool = active_tool or ""
        active_submode = _native_store_value("active_submode", _MISSING)
        if active_submode is _MISSING:
            active_submode = call("", getattr(lf.ui, "get_active_submode", None))
        active_submode = active_submode or ""
        gizmo_type = call("", getattr(lf.ui, "get_gizmo_type", None))
        crop_shape = call("box", getattr(lf.ui, "get_crop_tool_shape", None))
        crop_operation = call("translate", getattr(lf.ui, "get_crop_tool_operation", None))
        transform_space = _native_store_value("transform_space", _MISSING)
        if transform_space is _MISSING:
            transform_space = call(1, getattr(lf.ui, "get_transform_space", None))
        pivot_mode = _native_store_value("pivot_mode", _MISSING)
        if pivot_mode is _MISSING:
            pivot_mode = call(0, getattr(lf.ui, "get_pivot_mode", None))
        tool_defs = ToolRegistry.get_all()
        tool_ids = tuple(
            (getattr(tool_def, "id", ""), getattr(tool_def, "group", ""))
            for tool_def in tool_defs
        )

        ui_context = call(None, lf.ui.context) if hasattr(lf.ui, "context") else None
        has_scene_getter = getattr(lf, "has_scene", None)
        has_scene = (
            bool(getattr(ui_context, "has_scene", False))
            if ui_context is not None
            else bool(call(False, has_scene_getter)) if callable(has_scene_getter) else False
        )
        num_gaussians = int(getattr(ui_context, "num_gaussians", 0) or 0)
        selected_getter = getattr(lf, "get_selected_node_names", None)
        selected_nodes = tuple(call([], selected_getter) or []) if callable(selected_getter) else ()

        input_settings_enabled = bool(
            call(
                False,
                getattr(lf.ui, "is_panel_enabled", None),
                _UtilityToolbarController._INPUT_SETTINGS_PANEL_ID,
            )
        )
        plugin_marketplace_enabled = bool(
            call(
                False,
                getattr(lf.ui, "is_panel_enabled", None),
                _UtilityToolbarController._PLUGIN_MARKETPLACE_PANEL_ID,
            )
        )
        return (
            trainer_state,
            active_tool,
            active_submode,
            gizmo_type,
            crop_shape,
            crop_operation,
            transform_space,
            pivot_mode,
            has_scene,
            num_gaussians,
            selected_nodes,
            tool_ids,
            str(call("orbit", lf.get_camera_navigation_mode)).lower() if hasattr(lf, "get_camera_navigation_mode") else "orbit",
            self._viewport_export_controls.visible,
            bool(call(False, getattr(lf.ui, "is_sequencer_visible", None))),
            bool(histogram_mode_available(ui_context)) if ui_context is not None else False,
            input_settings_enabled,
            plugin_marketplace_enabled,
            bool(call(False, getattr(lf.ui, "is_panel_enabled", None), "lfs.histogram")),
        )

    def _on_toolbar_action(self, _handle, _event, args):
        if not args:
            return
        action = str(args[0])
        value = str(args[1]) if len(args) > 1 else ""
        if action == "toggle_viewport_export":
            self._gizmo.clear_active_horizontal_tool()
            self._viewport_export_controls.toggle(notify=False)
            self._last_toolbar_signature = None
            self._sync_toolbar_state()
            self._sync_tool_overlays_now()
            return
        if action in {
            "tool",
            "submode",
            "pivot",
            "selection_mode",
            "crop_object",
            "crop_transform",
            "crop_trim",
            "crop_apply",
        }:
            self._viewport_export_controls.close(notify=False)
            self._gizmo.dispatch(action, value)
        else:
            self._utility.dispatch(action, value)
        self._last_toolbar_signature = None
        self._sync_toolbar_state()
        self._sync_tool_overlays_now()


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
    return _toolbar_controller.update(doc)


def reset_overlay_state():
    if _toolbar_controller is not None:
        _toolbar_controller.reset()

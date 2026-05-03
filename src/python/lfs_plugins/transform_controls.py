# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Transform controls controller embedded into the Rendering panel."""

import math
import time
from typing import List

import lichtfeld as lf

from . import rml_widgets as w

TRANSLATE_STEP = 0.01
TRANSLATE_STEP_FAST = 0.1
ROTATE_STEP = 1.0
ROTATE_STEP_FAST = 15.0
SCALE_STEP = 0.01
SCALE_STEP_FAST = 0.1
MIN_SCALE = 0.001
QUAT_EQUIV_EPSILON = 1e-4

STEP_REPEAT_DELAY = 0.4
STEP_REPEAT_INTERVAL = 0.05

_STEP_CONFIG = {
    "pos_x": (TRANSLATE_STEP, TRANSLATE_STEP_FAST),
    "pos_y": (TRANSLATE_STEP, TRANSLATE_STEP_FAST),
    "pos_z": (TRANSLATE_STEP, TRANSLATE_STEP_FAST),
    "rot_x": (ROTATE_STEP, ROTATE_STEP_FAST),
    "rot_y": (ROTATE_STEP, ROTATE_STEP_FAST),
    "rot_z": (ROTATE_STEP, ROTATE_STEP_FAST),
    "scale_u": (SCALE_STEP, SCALE_STEP_FAST),
    "scale_x": (SCALE_STEP, SCALE_STEP_FAST),
    "scale_y": (SCALE_STEP, SCALE_STEP_FAST),
    "scale_z": (SCALE_STEP, SCALE_STEP_FAST),
}

_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_TRANSFORM_TOOL_IDS = ("builtin.translate", "builtin.rotate", "builtin.scale")
_SPACE_LOCAL = 0
_SPACE_WORLD = 1
_PIVOT_ORIGIN = 0
_PIVOT_BOUNDS = 1


def _ui_label(key: str, fallback: str) -> str:
    value = lf.ui.tr(key)
    if value and value != key:
        return value
    return fallback


def _quat_dot(a: List[float], b: List[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def _same_rotation(a: List[float], b: List[float]) -> bool:
    dot = _quat_dot(a, b)
    return abs(abs(dot) - 1.0) < QUAT_EQUIV_EPSILON


class TransformPanelState:
    def __init__(self):
        self.editing_active = False
        self.editing_node_names: List[str] = []
        self.transforms_before_edit: List[List[float]] = []

        self.euler_display = [0.0, 0.0, 0.0]
        self.euler_display_node = ""
        self.euler_display_rotation = [0.0, 0.0, 0.0, 1.0]

        self.multi_editing_active = False
        self.multi_node_names: List[str] = []
        self.multi_transforms_before: List[List[float]] = []
        self.multi_visualizer_world_transforms_before: List[List[float]] = []
        self.pivot_world = [0.0, 0.0, 0.0]
        self.display_translation = [0.0, 0.0, 0.0]
        self.display_euler = [0.0, 0.0, 0.0]
        self.display_scale = [1.0, 1.0, 1.0]

    def reset_single_edit(self):
        self.editing_active = False
        self.editing_node_names = []
        self.transforms_before_edit = []

    def reset_multi_edit(self):
        self.multi_editing_active = False
        self.multi_node_names = []
        self.multi_transforms_before = []
        self.multi_visualizer_world_transforms_before = []


class TransformControlsController:
    def __init__(self):
        self._state = TransformPanelState()
        self._handle = None
        self._doc = None
        self._visible = False
        self._active_tool = ""
        self._selected = []
        self._transform_space = _SPACE_WORLD
        self._pivot_mode = _PIVOT_ORIGIN

        self._trans = [0.0, 0.0, 0.0]
        self._euler = [0.0, 0.0, 0.0]
        self._scale = [1.0, 1.0, 1.0]

        self._step_repeat_prop = None
        self._step_repeat_dir = 0
        self._step_repeat_start = 0.0
        self._step_repeat_last = 0.0

        self._focus_active = False
        self._escape_revert = w.EscapeRevertController()

    def bind_model(self, model):
        model.bind_func("transform_tool_label", self._tool_label)
        model.bind_func(
            "transform_node_name",
            lambda: f"Node: {self._selected[0]}" if self._selected else "",
        )
        model.bind_func(
            "transform_multi_label",
            lambda: f"{len(self._selected)} nodes selected" if self._selected else "",
        )
        model.bind_func(
            "transform_reset_label",
            lambda: "Reset All" if len(self._selected) > 1 else "Reset Transform",
        )
        model.bind_func("transform_is_single", lambda: len(self._selected) == 1)
        model.bind_func("transform_is_multi", lambda: len(self._selected) > 1)
        model.bind_func("transform_show_translate", lambda: self._active_tool == "builtin.translate")
        model.bind_func("transform_show_rotate", lambda: self._active_tool == "builtin.rotate")
        model.bind_func("transform_show_scale", lambda: self._active_tool == "builtin.scale")
        model.bind_func("transform_space_label", self._space_label)
        model.bind_func("transform_pivot_label", self._pivot_label)

        for axis in ("x", "y", "z"):
            idx = _AXIS_INDEX[axis]
            model.bind(
                f"transform_pos_{axis}_str",
                lambda i=idx: f"{self._trans[i]:.3f}",
                lambda v, i=idx: self._set_value("pos", i, v),
            )
            model.bind(
                f"transform_rot_{axis}_str",
                lambda i=idx: f"{self._euler[i]:.1f}",
                lambda v, i=idx: self._set_value("rot", i, v),
            )
            model.bind(
                f"transform_scale_{axis}_str",
                lambda i=idx: f"{self._scale[i]:.3f}",
                lambda v, i=idx: self._set_value("scale", i, v),
            )

        model.bind(
            "transform_scale_u_str",
            lambda: f"{sum(self._scale) / 3.0:.3f}",
            lambda v: self._set_uniform_scale(v),
        )

        model.bind_event("transform_num_step", self._on_num_step)
        model.bind_event("transform_action", self._on_action)

        self._handle = model.get_handle()

    def mount(self, doc):
        self._doc = doc
        self._escape_revert.clear()

        body = doc.get_element_by_id("body")
        if body:
            body.add_event_listener("mouseup", self._on_step_mouseup)

        for input_id in (
            "transform-pos-x",
            "transform-pos-y",
            "transform-pos-z",
            "transform-rot-x",
            "transform-rot-y",
            "transform-rot-z",
            "transform-scale-u",
            "transform-scale-x",
            "transform-scale-y",
            "transform-scale-z",
        ):
            el = doc.get_element_by_id(input_id)
            if el:
                el.add_event_listener("focus", self._on_input_focus)
                self._escape_revert.bind(
                    el,
                    input_id,
                    lambda: True,
                    lambda _snapshot: self._cancel_active_edit(),
                )
                el.add_event_listener("blur", self._on_input_blur)

    def update(self, doc):
        dirty = False
        prev_tool = self._active_tool
        prev_space = self._transform_space
        prev_selected = tuple(self._selected)

        self._active_tool = lf.ui.get_active_tool() or ""
        self._selected = lf.get_selected_node_names() or []
        self._transform_space = self._current_transform_space()
        self._pivot_mode = self._current_pivot_mode()

        if self._active_tool != prev_tool or self._transform_space != prev_space:
            self._commit_active_edit()

        visible = self._active_tool in _TRANSFORM_TOOL_IDS and len(self._selected) > 0
        if visible != self._visible:
            self._visible = visible
            wrap = doc.get_element_by_id("transform-block")
            if wrap:
                wrap.set_class("hidden", not visible)
            dirty = True

        if not visible:
            self._commit_active_edit()
            return dirty

        if tuple(self._selected) != prev_selected:
            self._commit_active_edit()

        if len(self._selected) == 1:
            self._update_single_node()
        else:
            self._update_multi_selection()

        self._process_step_repeat()
        self._dirty_all()
        return True

    def scene_changed(self):
        self._dirty_all()

    def unmount(self):
        self._handle = None
        self._doc = None
        self._visible = False
        self._active_tool = ""
        self._selected = []
        self._escape_revert.clear()
        self._state.reset_single_edit()
        self._state.reset_multi_edit()
        self._step_repeat_prop = None
        self._focus_active = False

    def _tool_label(self):
        labels = {
            "builtin.translate": _ui_label("toolbar.translate", "Move"),
            "builtin.rotate": _ui_label("toolbar.rotate", "Rotate"),
            "builtin.scale": _ui_label("toolbar.scale", "Scale"),
        }
        return labels.get(self._active_tool, "Transform")

    def _space_label(self):
        if self._transform_space == _SPACE_LOCAL:
            return _ui_label("toolbar.local_space", "Local")
        return _ui_label("toolbar.world_space", "World")

    def _pivot_label(self):
        if self._pivot_mode == _PIVOT_BOUNDS:
            return _ui_label("toolbar.bounds_center_pivot", "Bounds")
        return _ui_label("toolbar.origin_pivot", "Origin")

    def _current_transform_space(self) -> int:
        getter = getattr(lf.ui, "get_transform_space", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                return _SPACE_WORLD
        return _SPACE_WORLD

    def _current_pivot_mode(self) -> int:
        getter = getattr(lf.ui, "get_pivot_mode", None)
        if callable(getter):
            try:
                return int(getter())
            except Exception:
                return _PIVOT_ORIGIN
        return _PIVOT_ORIGIN

    def _single_uses_world_space(self) -> bool:
        return self._transform_space == _SPACE_WORLD

    def _single_display_transform(self, node_name: str):
        if self._single_uses_world_space():
            return lf.get_node_visualizer_world_transform(node_name)
        return lf.get_node_transform(node_name)

    def _set_single_display_transform(self, node_name: str, transform):
        if self._single_uses_world_space():
            lf.set_node_visualizer_world_transform(node_name, transform)
        else:
            lf.set_node_transform(node_name, transform)

    def _update_single_node(self):
        node_name = self._selected[0]
        transform = self._single_display_transform(node_name)
        if transform is None:
            return

        if self._state.multi_editing_active:
            self._commit_multi_edit()

        decomp = lf.decompose_transform(transform)
        self._trans = list(decomp["translation"])
        quat = decomp["rotation_quat"]
        self._scale = list(decomp["scale"])

        selection_changed = node_name != self._state.euler_display_node
        external_change = not _same_rotation(quat, self._state.euler_display_rotation)
        if selection_changed or external_change:
            self._state.euler_display = list(decomp["rotation_euler_deg"])
            self._state.euler_display_node = node_name
            self._state.euler_display_rotation = quat.copy()

        self._euler = self._state.euler_display

    def _update_multi_selection(self):
        world_center = lf.get_selection_visualizer_world_center()
        if world_center is None:
            return

        if self._state.editing_active:
            self._commit_single_edit()

        current_center = list(world_center)

        selection_changed = self._state.multi_editing_active and set(self._state.multi_node_names) != set(self._selected)
        if selection_changed:
            self._commit_multi_edit()
            self._state.reset_multi_edit()

        if not self._state.multi_editing_active:
            self._trans = current_center.copy()
            self._euler = [0.0, 0.0, 0.0]
            self._scale = [1.0, 1.0, 1.0]
            self._state.display_translation = current_center.copy()
            self._state.display_euler = [0.0, 0.0, 0.0]
            self._state.display_scale = [1.0, 1.0, 1.0]
        else:
            self._trans = self._state.display_translation
            self._euler = self._state.display_euler
            self._scale = self._state.display_scale

    def _dirty_all(self):
        if not self._handle:
            return
        for axis in ("x", "y", "z"):
            self._handle.dirty(f"transform_pos_{axis}_str")
            self._handle.dirty(f"transform_rot_{axis}_str")
            self._handle.dirty(f"transform_scale_{axis}_str")
        self._handle.dirty("transform_scale_u_str")
        self._handle.dirty("transform_tool_label")
        self._handle.dirty("transform_node_name")
        self._handle.dirty("transform_multi_label")
        self._handle.dirty("transform_reset_label")
        self._handle.dirty("transform_is_single")
        self._handle.dirty("transform_is_multi")
        self._handle.dirty("transform_show_translate")
        self._handle.dirty("transform_show_rotate")
        self._handle.dirty("transform_show_scale")
        self._handle.dirty("transform_space_label")
        self._handle.dirty("transform_pivot_label")

    def _begin_edit(self):
        if len(self._selected) == 1:
            node_name = self._selected[0]
            transform = lf.get_node_transform(node_name)
            if transform is None:
                return
            self._state.editing_active = True
            self._state.editing_node_names = [node_name]
            self._state.transforms_before_edit = [transform]
        else:
            if self._state.multi_editing_active:
                return
            self._state.multi_editing_active = True
            center = lf.get_selection_visualizer_world_center()
            self._state.pivot_world = list(center) if center else [0.0, 0.0, 0.0]
            self._state.multi_node_names = list(self._selected)
            self._state.multi_transforms_before = []
            self._state.multi_visualizer_world_transforms_before = []
            for name in self._selected:
                transform = lf.get_node_transform(name)
                if transform is not None:
                    self._state.multi_transforms_before.append(transform)
                world_transform = lf.get_node_visualizer_world_transform(name)
                if world_transform is not None:
                    self._state.multi_visualizer_world_transforms_before.append(world_transform)

    def _set_value(self, group, idx, value_str):
        try:
            val = float(value_str)
        except ValueError:
            return

        if not self._state.editing_active and not self._state.multi_editing_active:
            self._begin_edit()

        if group == "pos":
            self._trans[idx] = val
        elif group == "rot":
            self._euler[idx] = val
        elif group == "scale":
            self._scale[idx] = max(val, MIN_SCALE)

        if len(self._selected) == 1:
            self._state.display_translation = self._trans
            self._state.display_euler = self._euler
            self._state.display_scale = self._scale
            self._apply_single_transform()
        else:
            self._state.display_translation = self._trans
            self._state.display_euler = self._euler
            self._state.display_scale = self._scale
            self._apply_multi_transform(self._active_tool)

    def _set_uniform_scale(self, value_str):
        try:
            val = max(float(value_str), MIN_SCALE)
        except ValueError:
            return

        if not self._state.editing_active and not self._state.multi_editing_active:
            self._begin_edit()

        self._scale = [val, val, val]

        if len(self._selected) == 1:
            self._state.display_scale = self._scale
            self._apply_single_transform()
        else:
            self._state.display_scale = self._scale
            self._apply_multi_transform(self._active_tool)

    def _apply_single_transform(self):
        if not self._selected:
            return

        node_name = self._selected[0]

        # Always read current transform to ensure we have up-to-date translation and scale
        current_transform = self._single_display_transform(node_name)
        decomp_current = lf.decompose_transform(current_transform) if current_transform else None

        if self._active_tool == "builtin.rotate":
            euler_to_use = list(self._euler)  # COPY to avoid reference issues
            self._state.euler_display = list(self._euler)  # COPY
            # Use current translation and scale from node to ensure we're not using stale values
            trans_to_use = list(decomp_current["translation"]) if decomp_current else list(self._trans)
            scale_to_use = list(decomp_current["scale"]) if decomp_current else list(self._scale)
        else:
            euler_to_use = list(decomp_current["rotation_euler_deg"]) if decomp_current else list(self._euler)
            trans_to_use = list(self._trans)  # COPY
            scale_to_use = list(self._scale)  # COPY

        new_transform = lf.compose_transform(trans_to_use, euler_to_use, scale_to_use)
        self._set_single_display_transform(node_name, new_transform)

        if self._active_tool == "builtin.rotate":
            new_decomp = lf.decompose_transform(new_transform)
            self._state.euler_display_rotation = list(new_decomp["rotation_quat"])  # COPY

    def _apply_multi_transform(self, tool: str):
        if (
            not self._state.multi_node_names
            or len(self._state.multi_visualizer_world_transforms_before) != len(self._state.multi_node_names)
        ):
            return

        pivot = self._state.pivot_world

        for i, name in enumerate(self._state.multi_node_names):
            original = self._state.multi_visualizer_world_transforms_before[i]
            decomp = lf.decompose_transform(original)
            pos = list(decomp["translation"])

            if tool == "builtin.translate":
                delta = [self._state.display_translation[j] - pivot[j] for j in range(3)]
                new_pos = [pos[j] + delta[j] for j in range(3)]
                new_transform = lf.compose_transform(new_pos, decomp["rotation_euler_deg"], decomp["scale"])
                lf.set_node_visualizer_world_transform(name, new_transform)

            elif tool == "builtin.rotate":
                euler_rad = [math.radians(e) for e in self._state.display_euler]
                cx, cy, cz = math.cos(euler_rad[0]), math.cos(euler_rad[1]), math.cos(euler_rad[2])
                sx, sy, sz = math.sin(euler_rad[0]), math.sin(euler_rad[1]), math.sin(euler_rad[2])
                r00, r01, r02 = cy * cz, -cy * sz, sy
                r10, r11, r12 = sx * sy * cz + cx * sz, -sx * sy * sz + cx * cz, -sx * cy
                r20, r21, r22 = -cx * sy * cz + sx * sz, cx * sy * sz + sx * cz, cx * cy

                rel = [pos[j] - pivot[j] for j in range(3)]
                new_rel = [
                    r00 * rel[0] + r01 * rel[1] + r02 * rel[2],
                    r10 * rel[0] + r11 * rel[1] + r12 * rel[2],
                    r20 * rel[0] + r21 * rel[1] + r22 * rel[2],
                ]
                new_pos = [pivot[j] + new_rel[j] for j in range(3)]
                orig_euler = list(decomp["rotation_euler_deg"])
                new_euler = [orig_euler[j] + self._state.display_euler[j] for j in range(3)]
                new_transform = lf.compose_transform(new_pos, new_euler, decomp["scale"])
                lf.set_node_visualizer_world_transform(name, new_transform)

            elif tool == "builtin.scale":
                rel = [pos[j] - pivot[j] for j in range(3)]
                new_rel = [rel[j] * self._state.display_scale[j] for j in range(3)]
                new_pos = [pivot[j] + new_rel[j] for j in range(3)]
                orig_scale = list(decomp["scale"])
                new_scale = [orig_scale[j] * self._state.display_scale[j] for j in range(3)]
                new_transform = lf.compose_transform(new_pos, decomp["rotation_euler_deg"], new_scale)
                lf.set_node_visualizer_world_transform(name, new_transform)

    def _on_num_step(self, handle, event, args):
        del handle, event
        if len(args) < 2:
            return
        prop = str(args[0])
        direction = int(args[1])
        self._apply_step(prop, direction)
        now = time.monotonic()
        self._step_repeat_prop = prop
        self._step_repeat_dir = direction
        self._step_repeat_start = now
        self._step_repeat_last = now

    def _on_step_mouseup(self, event):
        del event
        if self._step_repeat_prop:
            self._step_repeat_prop = None

    def _process_step_repeat(self):
        if not self._step_repeat_prop:
            return
        now = time.monotonic()
        if now - self._step_repeat_start < STEP_REPEAT_DELAY:
            return
        if now - self._step_repeat_last >= STEP_REPEAT_INTERVAL:
            self._apply_step(self._step_repeat_prop, self._step_repeat_dir)
            self._step_repeat_last = now

    def _apply_step(self, prop, direction):
        cfg = _STEP_CONFIG.get(prop)
        if not cfg:
            return

        step, step_fast = cfg
        step_val = step_fast if lf.ui.is_ctrl_down() else step

        if not self._state.editing_active and not self._state.multi_editing_active:
            self._begin_edit()

        group, axis = prop.split("_", 1)
        if group == "pos":
            idx = _AXIS_INDEX.get(axis, -1)
            if idx >= 0:
                self._trans[idx] += step_val * direction
        elif group == "rot":
            idx = _AXIS_INDEX.get(axis, -1)
            if idx >= 0:
                self._euler[idx] += step_val * direction
        elif group == "scale":
            if axis == "u":
                uniform = max(sum(self._scale) / 3.0 + step_val * direction, MIN_SCALE)
                self._scale = [uniform, uniform, uniform]
            else:
                idx = _AXIS_INDEX.get(axis, -1)
                if idx >= 0:
                    self._scale[idx] = max(self._scale[idx] + step_val * direction, MIN_SCALE)

        if len(self._selected) == 1:
            self._state.display_translation = self._trans
            self._state.display_euler = self._euler
            self._state.display_scale = self._scale
            self._apply_single_transform()
        else:
            self._state.display_translation = self._trans
            self._state.display_euler = self._euler
            self._state.display_scale = self._scale
            self._apply_multi_transform(self._active_tool)

    def _on_action(self, handle, event, args):
        del handle, event
        if not args:
            return
        action = str(args[0])
        if action == "reset":
            if len(self._selected) == 1:
                self._reset_single_transform()
            else:
                self._reset_multi_transforms()

    def _on_input_focus(self, event):
        if self._focus_active:
            return
        self._focus_active = True
        target = event.current_target()
        if target is not None:
            target.select()
        self._begin_edit()

    def _on_input_blur(self, event):
        del event
        if not self._focus_active:
            return
        self._focus_active = False
        if self._state.editing_active:
            self._commit_single_edit()
        elif self._state.multi_editing_active:
            self._commit_multi_edit()

    def _cancel_active_edit(self):
        if self._state.editing_active and self._state.editing_node_names and self._state.transforms_before_edit:
            node_name = self._state.editing_node_names[0]
            lf.set_node_transform(node_name, self._state.transforms_before_edit[0])
            self._state.reset_single_edit()
            self._update_single_node()
            return

        if self._state.multi_editing_active and self._state.multi_node_names and self._state.multi_transforms_before:
            for name, transform in zip(self._state.multi_node_names, self._state.multi_transforms_before):
                lf.set_node_transform(name, transform)
            self._state.reset_multi_edit()
            self._update_multi_selection()

    def _commit_active_edit(self):
        if self._state.editing_active:
            self._commit_single_edit()
        if self._state.multi_editing_active:
            self._commit_multi_edit()

    def _commit_single_edit(self):
        if not self._state.editing_node_names or not self._state.transforms_before_edit:
            self._state.reset_single_edit()
            return

        node_name = self._state.editing_node_names[0]
        current = lf.get_node_transform(node_name)
        if current is None:
            self._state.reset_single_edit()
            return

        old = self._state.transforms_before_edit[0]
        if old != current:
            lf.ops.invoke(
                "transform.apply_batch",
                node_names=[node_name],
                old_transforms=[old],
            )

        self._state.reset_single_edit()

    def _commit_multi_edit(self):
        if not self._state.multi_node_names or not self._state.multi_transforms_before:
            self._state.reset_multi_edit()
            return

        any_changed = False
        for i, name in enumerate(self._state.multi_node_names):
            if i >= len(self._state.multi_transforms_before):
                continue
            current = lf.get_node_transform(name)
            if current is not None and current != self._state.multi_transforms_before[i]:
                any_changed = True
                break

        if any_changed:
            lf.ops.invoke(
                "transform.apply_batch",
                node_names=self._state.multi_node_names,
                old_transforms=self._state.multi_transforms_before,
            )

        self._state.reset_multi_edit()

    def _reset_single_transform(self):
        if not self._selected:
            return

        node_name = self._selected[0]
        current = lf.get_node_transform(node_name)
        if current is None:
            return

        identity = lf.compose_transform([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        lf.set_node_transform(node_name, identity)
        lf.ops.invoke(
            "transform.apply_batch",
            node_names=[node_name],
            old_transforms=[current],
        )
        self._state.euler_display = [0.0, 0.0, 0.0]
        self._state.euler_display_rotation = [0.0, 0.0, 0.0, 1.0]

    def _reset_multi_transforms(self):
        if self._state.multi_editing_active:
            self._commit_multi_edit()

        selected = lf.get_selected_node_names()
        if not selected:
            return

        old_transforms = []
        for name in selected:
            transform = lf.get_node_transform(name)
            if transform is not None:
                old_transforms.append(transform)

        if len(old_transforms) != len(selected):
            return

        identity = lf.compose_transform([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        for name in selected:
            lf.set_node_transform(name, identity)

        lf.ops.invoke(
            "transform.apply_batch",
            node_names=selected,
            old_transforms=old_transforms,
        )

# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Transform panel for editing node transforms - RmlUI panel with ImGui draw."""

import math
from typing import List

import lichtfeld as lf

from .types import RmlPanel

TRANSLATE_STEP = 0.01
TRANSLATE_STEP_FAST = 0.1
TRANSLATE_STEP_CTRL = 0.1
TRANSLATE_STEP_CTRL_FAST = 1.0
ROTATE_STEP = 1.0
ROTATE_STEP_FAST = 15.0
SCALE_STEP = 0.01
SCALE_STEP_FAST = 0.1
MIN_SCALE = 0.001
QUAT_EQUIV_EPSILON = 1e-4


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
        self.initial_translation = [0.0, 0.0, 0.0]
        self.initial_scale = [1.0, 1.0, 1.0]

        self.euler_display = [0.0, 0.0, 0.0]
        self.euler_display_node = ""
        self.euler_display_rotation = [0.0, 0.0, 0.0, 1.0]

        self.multi_editing_active = False
        self.multi_node_names: List[str] = []
        self.multi_transforms_before: List[List[float]] = []
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


class TransformControlsPanel(RmlPanel):
    idname = "lfs.transform_controls"
    label = "Transform"
    space = "MAIN_PANEL_TAB"
    order = 120
    rml_template = "rmlui/transform_controls.rml"
    rml_height_mode = "content"

    def __init__(self):
        self._state = TransformPanelState()

    def draw_imgui(self, layout):
        active_tool = lf.ui.get_active_tool()
        if active_tool not in ("builtin.translate", "builtin.rotate", "builtin.scale"):
            return

        selected = lf.get_selected_node_names()
        if not selected:
            return

        tool_labels = {
            "builtin.translate": "Translate",
            "builtin.rotate": "Rotate",
            "builtin.scale": "Scale"
        }

        if not layout.collapsing_header(tool_labels[active_tool], default_open=True):
            return

        if len(selected) == 1:
            self._draw_single_node(layout, selected[0], active_tool)
        else:
            self._draw_multi_selection(layout, selected, active_tool)

    def _draw_single_node(self, layout, node_name: str, tool: str):
        transform = lf.get_node_transform(node_name)
        if transform is None:
            return

        decomp = lf.decompose_transform(transform)
        trans = list(decomp["translation"])
        quat = decomp["rotation_quat"]
        euler_deg = list(decomp["rotation_euler_deg"])
        scale = list(decomp["scale"])

        selection_changed = node_name != self._state.euler_display_node
        external_change = not _same_rotation(quat, self._state.euler_display_rotation)
        if selection_changed or external_change:
            self._state.euler_display = euler_deg.copy()
            self._state.euler_display_node = node_name
            self._state.euler_display_rotation = quat.copy()

        euler = self._state.euler_display

        layout.label(f"Node: {node_name}")
        layout.separator()

        changed = False
        any_active = False

        if tool == "builtin.translate":
            layout.label("Position")
            ch, trans[0] = layout.input_float("X##pos", trans[0], TRANSLATE_STEP, TRANSLATE_STEP_FAST, "%.3f")
            changed |= ch
            any_active |= layout.is_item_active()
            ch, trans[1] = layout.input_float("Y##pos", trans[1], TRANSLATE_STEP, TRANSLATE_STEP_FAST, "%.3f")
            changed |= ch
            any_active |= layout.is_item_active()
            ch, trans[2] = layout.input_float("Z##pos", trans[2], TRANSLATE_STEP, TRANSLATE_STEP_FAST, "%.3f")
            changed |= ch
            any_active |= layout.is_item_active()

        elif tool == "builtin.rotate":
            layout.label("Rotation (degrees)")
            ch, euler[0] = layout.input_float("X##rot", euler[0], ROTATE_STEP, ROTATE_STEP_FAST, "%.1f")
            changed |= ch
            any_active |= layout.is_item_active()
            ch, euler[1] = layout.input_float("Y##rot", euler[1], ROTATE_STEP, ROTATE_STEP_FAST, "%.1f")
            changed |= ch
            any_active |= layout.is_item_active()
            ch, euler[2] = layout.input_float("Z##rot", euler[2], ROTATE_STEP, ROTATE_STEP_FAST, "%.1f")
            changed |= ch
            any_active |= layout.is_item_active()

        elif tool == "builtin.scale":
            layout.label("Scale")
            uniform = (scale[0] + scale[1] + scale[2]) / 3.0
            ch, uniform = layout.input_float("U##scale", uniform, SCALE_STEP, SCALE_STEP_FAST, "%.3f")
            if ch:
                uniform = max(uniform, MIN_SCALE)
                scale = [uniform, uniform, uniform]
                changed = True
            any_active |= layout.is_item_active()
            layout.separator()
            ch, scale[0] = layout.input_float("X##scale", scale[0], SCALE_STEP, SCALE_STEP_FAST, "%.3f")
            changed |= ch
            any_active |= layout.is_item_active()
            ch, scale[1] = layout.input_float("Y##scale", scale[1], SCALE_STEP, SCALE_STEP_FAST, "%.3f")
            changed |= ch
            any_active |= layout.is_item_active()
            ch, scale[2] = layout.input_float("Z##scale", scale[2], SCALE_STEP, SCALE_STEP_FAST, "%.3f")
            changed |= ch
            any_active |= layout.is_item_active()
            scale = [max(s, MIN_SCALE) for s in scale]

        if (any_active or changed) and not self._state.editing_active:
            self._state.editing_active = True
            self._state.editing_node_names = [node_name]
            self._state.transforms_before_edit = [transform]
            self._state.initial_translation = list(decomp["translation"])
            self._state.initial_scale = list(decomp["scale"])

        if tool == "builtin.rotate" and changed:
            self._state.euler_display = euler.copy()

        if changed:
            if tool == "builtin.rotate":
                euler_to_use = euler
            else:
                euler_to_use = decomp["rotation_euler_deg"]

            new_transform = lf.compose_transform(trans, euler_to_use, scale)
            lf.set_node_transform(node_name, new_transform)

            if tool == "builtin.rotate":
                new_decomp = lf.decompose_transform(new_transform)
                self._state.euler_display_rotation = new_decomp["rotation_quat"].copy()

        if not any_active and self._state.editing_active:
            self._commit_single_edit(node_name)

        layout.separator()
        if layout.button("Reset Transform"):
            self._reset_single_transform(node_name, transform)

    def _draw_multi_selection(self, layout, selected: List[str], tool: str):
        world_center = lf.get_selection_world_center()
        if world_center is None:
            return

        current_center = list(world_center)

        if not self._state.multi_editing_active:
            self._state.display_translation = current_center.copy()
            self._state.display_euler = [0.0, 0.0, 0.0]
            self._state.display_scale = [1.0, 1.0, 1.0]

        layout.label(f"{len(selected)} nodes selected")
        layout.label("Space: World")
        layout.separator()

        selection_changed = (self._state.multi_editing_active and
                            set(self._state.multi_node_names) != set(selected))
        if selection_changed:
            self._commit_multi_edit()
            self._state.reset_multi_edit()
            self._state.display_translation = current_center.copy()

        changed = False
        any_active = False

        if tool == "builtin.translate":
            layout.label("Position")
            ch, self._state.display_translation[0] = layout.input_float(
                "X##mpos", self._state.display_translation[0],
                TRANSLATE_STEP, TRANSLATE_STEP_FAST, "%.3f")
            changed |= ch
            any_active |= layout.is_item_active()
            ch, self._state.display_translation[1] = layout.input_float(
                "Y##mpos", self._state.display_translation[1],
                TRANSLATE_STEP, TRANSLATE_STEP_FAST, "%.3f")
            changed |= ch
            any_active |= layout.is_item_active()
            ch, self._state.display_translation[2] = layout.input_float(
                "Z##mpos", self._state.display_translation[2],
                TRANSLATE_STEP, TRANSLATE_STEP_FAST, "%.3f")
            changed |= ch
            any_active |= layout.is_item_active()

        elif tool == "builtin.rotate":
            layout.label("Rotation (degrees)")
            ch, self._state.display_euler[0] = layout.input_float(
                "X##mrot", self._state.display_euler[0],
                ROTATE_STEP, ROTATE_STEP_FAST, "%.1f")
            changed |= ch
            any_active |= layout.is_item_active()
            ch, self._state.display_euler[1] = layout.input_float(
                "Y##mrot", self._state.display_euler[1],
                ROTATE_STEP, ROTATE_STEP_FAST, "%.1f")
            changed |= ch
            any_active |= layout.is_item_active()
            ch, self._state.display_euler[2] = layout.input_float(
                "Z##mrot", self._state.display_euler[2],
                ROTATE_STEP, ROTATE_STEP_FAST, "%.1f")
            changed |= ch
            any_active |= layout.is_item_active()

        elif tool == "builtin.scale":
            layout.label("Scale")
            uniform = sum(self._state.display_scale) / 3.0
            ch, uniform = layout.input_float("U##mscale", uniform, SCALE_STEP, SCALE_STEP_FAST, "%.3f")
            if ch:
                uniform = max(uniform, MIN_SCALE)
                self._state.display_scale = [uniform, uniform, uniform]
                changed = True
            any_active |= layout.is_item_active()
            layout.separator()
            ch, self._state.display_scale[0] = layout.input_float(
                "X##mscale", self._state.display_scale[0],
                SCALE_STEP, SCALE_STEP_FAST, "%.3f")
            changed |= ch
            any_active |= layout.is_item_active()
            ch, self._state.display_scale[1] = layout.input_float(
                "Y##mscale", self._state.display_scale[1],
                SCALE_STEP, SCALE_STEP_FAST, "%.3f")
            changed |= ch
            any_active |= layout.is_item_active()
            ch, self._state.display_scale[2] = layout.input_float(
                "Z##mscale", self._state.display_scale[2],
                SCALE_STEP, SCALE_STEP_FAST, "%.3f")
            changed |= ch
            any_active |= layout.is_item_active()
            self._state.display_scale = [max(s, MIN_SCALE) for s in self._state.display_scale]

        if (any_active or changed) and not self._state.multi_editing_active:
            self._state.multi_editing_active = True
            self._state.pivot_world = current_center.copy()
            self._state.multi_node_names = list(selected)
            self._state.multi_transforms_before = []
            for name in selected:
                t = lf.get_node_transform(name)
                if t is not None:
                    self._state.multi_transforms_before.append(t)

        if changed and self._state.multi_editing_active:
            self._apply_multi_transform(tool)

        if not any_active and self._state.multi_editing_active:
            self._commit_multi_edit()

        layout.separator()
        if layout.button("Reset All"):
            self._reset_multi_transforms()

    def _apply_multi_transform(self, tool: str):
        if not self._state.multi_node_names:
            return

        pivot = self._state.pivot_world

        for i, name in enumerate(self._state.multi_node_names):
            if i >= len(self._state.multi_transforms_before):
                continue

            original = self._state.multi_transforms_before[i]
            decomp = lf.decompose_transform(original)
            pos = list(decomp["translation"])

            if tool == "builtin.translate":
                delta = [
                    self._state.display_translation[j] - pivot[j]
                    for j in range(3)
                ]
                new_pos = [pos[j] + delta[j] for j in range(3)]
                new_transform = lf.compose_transform(
                    new_pos,
                    decomp["rotation_euler_deg"],
                    decomp["scale"]
                )
                lf.set_node_transform(name, new_transform)

            elif tool == "builtin.rotate":
                euler_rad = [math.radians(e) for e in self._state.display_euler]
                cx, cy, cz = math.cos(euler_rad[0]), math.cos(euler_rad[1]), math.cos(euler_rad[2])
                sx, sy, sz = math.sin(euler_rad[0]), math.sin(euler_rad[1]), math.sin(euler_rad[2])
                r00 = cy * cz
                r01 = -cy * sz
                r02 = sy
                r10 = sx * sy * cz + cx * sz
                r11 = -sx * sy * sz + cx * cz
                r12 = -sx * cy
                r20 = -cx * sy * cz + sx * sz
                r21 = cx * sy * sz + sx * cz
                r22 = cx * cy

                rel = [pos[j] - pivot[j] for j in range(3)]
                new_rel = [
                    r00 * rel[0] + r01 * rel[1] + r02 * rel[2],
                    r10 * rel[0] + r11 * rel[1] + r12 * rel[2],
                    r20 * rel[0] + r21 * rel[1] + r22 * rel[2]
                ]
                new_pos = [pivot[j] + new_rel[j] for j in range(3)]

                orig_euler = list(decomp["rotation_euler_deg"])
                new_euler = [orig_euler[j] + self._state.display_euler[j] for j in range(3)]

                new_transform = lf.compose_transform(new_pos, new_euler, decomp["scale"])
                lf.set_node_transform(name, new_transform)

            elif tool == "builtin.scale":
                rel = [pos[j] - pivot[j] for j in range(3)]
                new_rel = [rel[j] * self._state.display_scale[j] for j in range(3)]
                new_pos = [pivot[j] + new_rel[j] for j in range(3)]

                orig_scale = list(decomp["scale"])
                new_scale = [orig_scale[j] * self._state.display_scale[j] for j in range(3)]

                new_transform = lf.compose_transform(
                    new_pos,
                    decomp["rotation_euler_deg"],
                    new_scale
                )
                lf.set_node_transform(name, new_transform)

    def _commit_single_edit(self, node_name: str):
        if not self._state.transforms_before_edit:
            self._state.reset_single_edit()
            return

        current = lf.get_node_transform(node_name)
        if current is None:
            self._state.reset_single_edit()
            return

        old = self._state.transforms_before_edit[0]
        if old != current:
            lf.ops.invoke("transform.apply_batch",
                          node_names=[node_name],
                          old_transforms=[old])

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
            lf.ops.invoke("transform.apply_batch",
                          node_names=self._state.multi_node_names,
                          old_transforms=self._state.multi_transforms_before)

        self._state.reset_multi_edit()

    def _reset_single_transform(self, node_name: str, current_transform: List[float]):
        identity = lf.compose_transform([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        lf.set_node_transform(node_name, identity)
        lf.ops.invoke("transform.apply_batch",
                      node_names=[node_name],
                      old_transforms=[current_transform])
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
            t = lf.get_node_transform(name)
            if t is not None:
                old_transforms.append(t)

        if len(old_transforms) != len(selected):
            return

        identity = lf.compose_transform([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        for name in selected:
            lf.set_node_transform(name, identity)

        lf.ops.invoke("transform.apply_batch",
                      node_names=selected,
                      old_transforms=old_transforms)


def register():
    lf.ui.register_rml_panel(TransformControlsPanel)
    lf.ui.set_panel_parent("lfs.transform_controls", "lfs.rendering")


def unregister():
    lf.ui.set_panel_enabled("lfs.transform_controls", False)

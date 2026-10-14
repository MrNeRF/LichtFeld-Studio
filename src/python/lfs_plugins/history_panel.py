# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compact navigation for the shared undo history."""

from html import escape
import math

import lichtfeld as lf

from .panels import panel_class
from .types import Panel
from .rml_keys import KI_UP, KI_DOWN, KI_HOME, KI_END, KI_RETURN, KI_SPACE

__lfs_panel_classes__ = ["HistoryPanel"]
__lfs_panel_ids__ = ["lfs.history"]

PAGE_SIZE = 25
LABEL_CHARS = 256
DETAIL_LIMIT = 3


# Match identifiers and existing native labels; display strings come from locales.
_OPERATION_PRESENTATION = {}


def _operation_aliases(aliases, key, icon):
    for alias in aliases:
        _OPERATION_PRESENTATION[alias.casefold()] = (key, icon)


_operation_aliases(("transform.translate", "Translate", "Move"), "translate", "translation.png")
_operation_aliases(("transform.rotate", "Rotate"), "rotate", "rotation.png")
_operation_aliases(("transform.scale", "Scale"), "scale", "scaling.png")
_operation_aliases(("transform.set", "Set Transform", "python.set_selected_node_transform",
                    "python.set_node_transform", "python.scene.set_node_transform",
                    "python.set_node_visualizer_world_transform"), "set_transform", "translation.png")
_operation_aliases(("transform.batch", "Apply Batch Transform"), "batch_transform", "translation.png")
_operation_aliases(("transform.align", "Align"), "align", "focus-selection.png")
_operation_aliases(("Mirror", "transform.mirror"), "mirror", "mirror.png")
_operation_aliases(("edit.delete", "Delete", "Delete Selected"), "delete", "scene/trash.png")
_operation_aliases(("select.stroke", "selection.stroke"), "selection_stroke", "selection.png")
_operation_aliases(("selection.box",), "selection_box", "rectangle.png")
_operation_aliases(("selection.sphere",), "selection_sphere", "sphere.png")
_operation_aliases(("selection.rect", "selection.rectangle"), "selection_rectangle", "rectangle.png")
_operation_aliases(("selection.polygon",), "selection_polygon", "polygon.png")
_operation_aliases(("selection.lasso",), "selection_lasso", "lasso.png")
_operation_aliases(("selection.brush",), "selection_brush", "selection.png")
_operation_aliases(("selection.ring",), "selection_ring", "ring.png")
_operation_aliases(("select.all", "selection.all", "selection.all.filtered", "Select All"), "selection_all", "select-all.png")
_operation_aliases(("select.none", "selection.none", "Deselect All"), "selection_none", "deselect.png")
_operation_aliases(("select.invert", "selection.invert", "selection.invert.filtered", "Invert Selection"), "selection_invert", "select-invert.png")
_operation_aliases(("select.grow", "selection.grow", "Grow Selection"), "selection_grow", "selection.png")
_operation_aliases(("select.shrink", "selection.shrink", "Shrink Selection"), "selection_shrink", "selection.png")
_operation_aliases(("select.by_opacity", "selection.by_opacity", "Select by Opacity"), "selection_opacity", "histogram.png")
_operation_aliases(("select.by_scale", "selection.by_scale", "Select by Scale"), "selection_scale", "histogram.png")
_operation_aliases(("selection.by_color", "Select by Color"), "selection_color", "selection.png")
_operation_aliases(("selection.mask",), "selection_mask", "scene/mask.png")
_operation_aliases(("selection.preview",), "selection_preview", "selection.png")
_operation_aliases(("selection.histogram",), "selection_histogram", "histogram.png")
_operation_aliases(("selection.reset_state",), "selection_reset", "deselect.png")
_operation_aliases(("transform.bake", "Bake Transforms"), "transform_bake", "translation.png")
_operation_aliases(("Crop Box",), "crop_apply", "scene/cropbox.png")
_operation_aliases(("Crop Ellipsoid",), "ellipsoid_apply", "scene/ellipsoid.png")
_operation_aliases(("Add Crop Box",), "crop_add", "scene/cropbox.png")
_operation_aliases(("Add Ellipsoid",), "ellipsoid_add", "scene/ellipsoid.png")
_operation_aliases(("Convert to Crop Box",), "crop_convert_box", "scene/cropbox.png")
_operation_aliases(("Convert to Ellipsoid",), "crop_convert_ellipsoid", "scene/ellipsoid.png")
_operation_aliases(("Remove Duplicate Crop Volume",), "crop_remove_duplicate", "scene/trash.png")
_operation_aliases(("cropbox.transform", "Crop Box Transform"), "crop", "scene/cropbox.png")
_operation_aliases(("ellipsoid.transform", "Ellipsoid Transform"), "ellipsoid", "scene/ellipsoid.png")
_operation_aliases(("Set Property",), "property", "settings.png")
_operation_aliases(("Rename Node",), "node_rename", "scene/group.png")
_operation_aliases(("Set Visibility",), "node_visibility", "scene/visible.png")
_operation_aliases(("Set Lock State",), "node_lock", "scene/locked.png")
_operation_aliases(("Set Training State", "Set Training Enabled", "Set Camera Training State"), "node_training", "scene/splat.png")
_operation_aliases(("Add Group", "Add Node"), "node_add", "scene/group.png")
_operation_aliases(("Delete Node", "Remove Node"), "node_delete", "scene/trash.png")
_operation_aliases(("Delete Nodes",), "delete_nodes", "scene/trash.png")
_operation_aliases(("Reparent Node", "Reparent Nodes"), "node_reparent", "scene/group.png")
_operation_aliases(("Group Selected", "Group Selected Nodes", "Group Nodes"), "node_group", "scene/group.png")
_operation_aliases(("Ungroup", "Ungroup Nodes"), "node_ungroup", "scene/group.png")
_operation_aliases(("Duplicate Node", "Duplicate", "edit.duplicate"), "node_duplicate", "scene/group.png")
_operation_aliases(("Merge Group", "Merge Nodes", "Merge into Single Model"), "node_merge", "scene/splat.png")
_operation_aliases(("Paste", "Paste Nodes"), "node_paste", "scene/group.png")
_operation_aliases(("Add Simplified Splat",), "node_simplify", "scene/splat.png")
_operation_aliases(("Replace Node",), "node_replace", "scene/group.png")
_operation_aliases(("Set Camera Split",), "camera_split", "scene/camera.png")
_operation_aliases(("Add Keyframe", "sequencer.add_keyframe"), "keyframe_add", "video.png")
_operation_aliases(("Delete Keyframe", "Remove Keyframe", "sequencer.delete_keyframe"), "keyframe_delete", "video.png")
_operation_aliases(("Update Keyframe", "Edit Keyframe", "sequencer.update_keyframe"), "keyframe_update", "video.png")
_operation_aliases(("Edit Keyframe Time", "sequencer.keyframe_time"), "keyframe_time", "video.png")
_operation_aliases(("Set Keyframe Easing", "sequencer.keyframe_easing"), "keyframe_easing", "video.png")
_operation_aliases(("Edit Clip Duration", "sequencer.clip_duration"), "clip_duration", "video.png")

_SCOPE_ICONS = {
    "selection": "selection.png", "transforms": "translation.png", "transform": "translation.png",
    "topology": "scene/splat.png", "scene_graph": "scene/group.png", "grouped": "scene/group.png",
    "cropbox": "scene/cropbox.png", "ellipsoid": "scene/ellipsoid.png", "sequencer": "video.png",
    "property": "settings.png", "custom": "puzzle.png", "tensor": "histogram.png", "sh": "histogram.png",
}


def _presentation(name, metadata=None):
    metadata = metadata or {}
    # A human label distinguishes graph operations sharing one metadata id.
    found = _OPERATION_PRESENTATION.get(name.casefold())
    if found is None:
        found = _OPERATION_PRESENTATION.get(str(metadata.get("id", "")).casefold())
    if found is not None:
        return found
    identifier = str(metadata.get("id", name)).casefold()
    if identifier.endswith(".visible"):
        return "node_visibility", "scene/visible.png"
    if identifier.endswith(".locked"):
        return "node_lock", "scene/locked.png"
    if identifier.startswith(("selection.group", "selection_group")):
        return "selection_group", "scene/selection-group.png"
    if identifier.startswith(("tensor.", "histogram.", "splat.")):
        return "tensor", "histogram.png"
    if identifier.startswith("sh."):
        return "sh", "histogram.png"
    return None, _SCOPE_ICONS.get(metadata.get("scope"), "puzzle.png" if metadata.get("source") == "python" else "settings.png")


def operation_label(name, metadata=None):
    if " | " in name:
        return " | ".join(operation_label(part) for part in name.split(" | "))
    key, _ = _presentation(name, metadata)
    if key == "batch_transform":
        return lf.ui.tr("history.batch_transform")
    if key:
        label_key = "history.operation." + key
        return lf.ui.tr(label_key)
    return name


def operation_icon(name, metadata=None):
    if " | " in name:
        return "scene/group.png"
    return _presentation(name, metadata)[1]

def row_markup(name, steps, generation, stack="undo", metadata=None):
    # Bound text before RmlUi layout, including tooltips. Retained names stay intact.
    icon = operation_icon(name, metadata)
    name = operation_label(name, metadata) or lf.ui.tr("runtime.untitled_change")
    name = name.replace("\n", " ").replace("\r", " ")
    name = name if len(name) <= LABEL_CHARS else name[:LABEL_CHARS] + "…"
    classes = "history-entry" + (" is-next" if steps == 1 else "")
    return (f'<div class="{classes}" data-steps="{steps}" data-generation="{generation}">'
            f'<div class="history-entry-header">'
            f'<button type="button" class="history-disclosure" data-details="{steps}" '
            f'data-generation="{generation}" title="{escape(lf.ui.tr("history.details"), quote=True)}">'
            f'<img class="history-chevron" src="../icon/dropdown-arrow.png"/></button>'
            f'<button type="button" class="btn btn--ghost history-jump" '
            f'data-steps="{steps}" data-generation="{generation}" data-tooltip="scene.history_note">'
            f'<img class="history-operation-icon" src="../icon/{icon}" title="{escape(name, quote=True)}"/>'
            f'<span class="history-label">{escape(name)}</span>'
            f'<img class="history-icon" src="../icon/{stack}.png"/></button></div>'
            f'<div id="history-detail-{steps}" class="history-details" style="display: none;"></div></div>')


def _vector_text(values):
    return ", ".join(format(float(value), ".6g") for value in values)


def _detail_line(key, before, after):
    caption = lf.ui.tr(key).rstrip(":")
    return (f'<div class="history-detail-line"><span class="history-detail-property">{escape(caption)}</span>'
            f'<span class="history-detail-value">{escape(_vector_text(before))}</span>'
            f'<span class="history-detail-value">{escape(_vector_text(after))}</span></div>')


def _orthogonal(matrix):
    if any(matrix[i] != 0 for i in (3, 7, 11)) or matrix[15] != 1:
        return False
    columns = [matrix[i:i + 3] for i in (0, 4, 8)]
    lengths = [math.sqrt(sum(value * value for value in column)) for column in columns]
    if not all(math.isfinite(length) and length > 1e-8 for length in lengths):
        return False
    return all(abs(sum(a * b for a, b in zip(columns[i], columns[j])))
               <= 1e-5 * lengths[i] * lengths[j] for i, j in ((0, 1), (0, 2), (1, 2)))


def _detail_value_text(value):
    value = str(value or "")
    if value.startswith("@tr:"):
        value = lf.ui.tr(value[4:])
    return value[:1024] + ("…" if len(value) > 1024 else "")


def _field_caption(key, description):
    if key.startswith(("history.", "transform.", "common.")):
        return lf.ui.tr(key)
    if description.get("source") == "property":
        suffix = key.rsplit(".", 1)[-1]
        field = suffix if suffix in {"name", "visible", "locked", "training_enabled", "source_path",
                                    "transform", "enabled", "inverse", "color", "line_width"} else "property"
        label_key = "history.field." + field
        return lf.ui.tr(label_key)
    return key


def details_markup(description, limit=DETAIL_LIMIT):
    tr = lf.ui.tr
    result = ""
    transforms = description.get("transforms", [])
    fields = description.get("fields", [])
    if not transforms and not fields:
        return f'<div class="text-muted">{escape(tr("history.no_details"))}</div>'
    result += (f'<div class="history-detail-line text-muted"><span class="history-detail-property"></span>'
               f'<span class="history-detail-value">{escape(tr("history.before"))}</span>'
               f'<span class="history-detail-value">{escape(tr("history.after"))}</span></div>')
    last_target = None
    for field in fields[:limit * 2]:
        key = field.get("property", "")
        target = field.get("target", "")
        before, after = field.get("before", ""), field.get("after", "")
        if key == "history.field.operation":
            target = operation_label(target)
            before, after = operation_label(before), operation_label(after)
        if target and target != last_target:
            result += f'<div class="history-detail-target">{escape(target[:LABEL_CHARS])}</div>'
        last_target = target
        caption = _field_caption(key, description)[:LABEL_CHARS]
        result += (f'<div class="history-detail-line"><span class="history-detail-property">{escape(caption)}</span>'
                   f'<span class="history-detail-value">{escape(_detail_value_text(before)) or "—"}</span>'
                   f'<span class="history-detail-value">{escape(_detail_value_text(after)) or "—"}</span></div>')
    for transform in transforms[:limit]:
        target = transform["target"][:LABEL_CHARS]
        result += f'<div class="history-detail-target">{escape(target)}</div>'
        before, after = transform["before"], transform["after"]
        if _orthogonal(before) and _orthogonal(after):
            before_trs = lf.decompose_transform(before)
            after_trs = lf.decompose_transform(after)
            for name, key in (("translation", "transform.position"),
                              ("rotation_euler_deg", "history.rotation_degrees"),
                              ("scale", "transform.scale")):
                if before_trs[name] != after_trs[name]:
                    result += _detail_line(key, before_trs[name], after_trs[name])
        else:
            for row in range(4):
                result += _detail_line("history.matrix", before[row::4], after[row::4])
    field_count = description.get("field_count", len(fields))
    transform_count = description.get("transform_count", len(transforms))
    if field_count > limit * 2:
        result += f'<div class="text-muted">{escape(tr("history.more_fields").format(count=field_count - limit * 2))}</div>'
    if transform_count > limit:
        result += f'<div class="text-muted">{escape(tr("history.more_targets").format(count=transform_count - limit))}</div>'
    return result


@panel_class("history")
class HistoryPanel(Panel):
    def __init__(self):
        super().__init__()
        self._doc = None
        self._handle = None
        self._stamp = None
        self._rows_stamp = None
        self._generation = None
        self._names = {"undo": [], "redo": []}
        self._items = {"undo": [], "redo": []}
        self._stack = "undo"
        self._page = 0
        self._feedback = ""
        self._expanded = {}

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("history_panel")
        if model is not None:
            model.bind_func("panel_label", lambda: lf.ui.tr("scene.history"))
            self._handle = model.get_handle()

    def on_mount(self, doc):
        super().on_mount(doc)
        self._doc = doc
        self._stamp = None
        self._rows_stamp = None
        for name in ("undo", "redo", "clear", "undo-tab", "redo-tab", "newer", "older"):
            doc.get_element_by_id("history-" + name).add_event_listener(
                "click", lambda event, action=name: self._action(action))
        rows = doc.get_element_by_id("history-rows")
        rows.add_event_listener("click", self._on_row)
        rows.add_event_listener("keydown", self._on_key)
        self.on_update(doc)

    def on_unmount(self, doc):
        if doc:
            doc.remove_data_model("history_panel")
        self._doc = None
        self._handle = None
        self._stamp = None
        self._rows_stamp = None
        self._generation = None
        self._names = {"undo": [], "redo": []}
        self._items = {"undo": [], "redo": []}
        self._expanded.clear()

    def _disabled(self, name, disabled):
        element = self._doc.get_element_by_id("history-" + name)
        if disabled:
            element.set_attribute("disabled", "disabled")
        else:
            element.remove_attribute("disabled")

    def on_update(self, doc):
        generation = lf.undo.generation()
        language = lf.ui.get_current_language()
        active = lf.undo.has_active_transaction()
        stamp = (generation, language, active, self._stack, self._page, self._feedback)
        if stamp == self._stamp:
            return
        if self._handle and (self._stamp is None or language != self._stamp[1]):
            self._handle.dirty("panel_label")
        if generation != self._generation:
            snapshot = lf.undo.stack()
            self._items = {stack: snapshot[stack] for stack in ("undo", "redo")}
            self._names = {stack: [item["label"] for item in items] for stack, items in self._items.items()}
            self._generation = generation
            self._page = 0
            self._expanded.clear()
        names = self._names[self._stack]
        pages = max(1, (len(names) + PAGE_SIZE - 1) // PAGE_SIZE)
        self._page = min(self._page, pages - 1)
        tr = lf.ui.tr
        for stack in ("undo", "redo"):
            tab = doc.get_element_by_id("history-" + stack + "-tab")
            tab_text = tr('scene.' + stack + '_stack')
            count_text = format(len(self._names[stack]), 'n')
            tab.set_inner_rml(f'<span class="history-tab-label">{escape(tab_text)}</span>'
                              f'<span class="history-tab-count">{count_text}</span>')
            tab.set_class("is-active", stack == self._stack)
            self._disabled(stack, active or not self._names[stack])
        self._disabled("clear", active or not any(self._names.values()))
        self._disabled("newer", self._page == 0)
        self._disabled("older", self._page + 1 >= pages)
        first = self._page * PAGE_SIZE
        rows_stamp = (generation, language, self._stack, self._page)
        if rows_stamp != self._rows_stamp:
            doc.get_element_by_id("history-rows").set_inner_rml("".join(
                row_markup(name, first + i + 1, generation, self._stack, self._items[self._stack][first + i])
                for i, name in enumerate(names[first:first + PAGE_SIZE])))
            doc.get_element_by_id("history-scroll").scroll_top = 0
            self._rows_stamp = rows_stamp
            for steps, description in self._expanded.items():
                self._show_details(steps, description)
        empty = doc.get_element_by_id("history-empty")
        empty.set_inner_rml(escape(tr("scene.no_entries_in_stack")))
        empty.set_property("display", "none" if names else "block")
        page_text = f"{self._page + 1:n} / {pages:n}"
        doc.get_element_by_id("history-page").set_inner_rml(page_text)
        feedback = doc.get_element_by_id("history-feedback")
        feedback.set_inner_rml(escape(self._feedback))
        feedback.set_property("display", "block" if self._feedback else "none")
        for row in doc.get_element_by_id("history-rows").query_selector_all(".history-jump"):
            if active:
                row.set_attribute("disabled", "disabled")
            else:
                row.remove_attribute("disabled")
        self._stamp = (generation, language, active, self._stack, self._page, self._feedback)

    def _refresh(self):
        if self._doc:
            self.on_update(self._doc)
            lf.ui.request_redraw()

    def _jump(self, stack, count):
        if lf.undo.has_active_transaction():
            return
        result = lf.undo.jump(stack, count)
        self._feedback = result.get("error", "") if not result.get("success") else ""
        self._refresh()

    def _action(self, action):
        self._feedback = ""
        if action.endswith("-tab"):
            self._stack = action[:-4]
            self._page = 0
            self._expanded.clear()
        elif action in ("older", "newer"):
            self._expanded.clear()
            self._page = max(0, self._page + (1 if action == "older" else -1))
        elif action == "clear":
            if not lf.undo.has_active_transaction():
                lf.undo.clear()
        elif action in ("undo", "redo"):
            self._jump(action, 1)
        self._refresh()

    def _show_details(self, steps, description):
        detail = self._doc.get_element_by_id(f"history-detail-{steps}")
        if detail:
            detail.set_inner_rml(details_markup(description))
            detail.set_property("display", "block")
            detail.parent().set_class("is-expanded", True)

    def _toggle_details(self, steps, generation):
        if generation != lf.undo.generation():
            self._refresh()
            return
        if steps in self._expanded:
            del self._expanded[steps]
            detail = self._doc.get_element_by_id(f"history-detail-{steps}")
            detail.set_property("display", "none")
            detail.parent().set_class("is-expanded", False)
        else:
            description = lf.undo.details(self._stack, steps, generation)
            if description is None:
                self._refresh()
                return
            # Keep only the summary the panel can display, rather than retaining
            # full descriptions for large compound edits or scenes.
            fields = description.get("fields", [])
            transforms = description.get("transforms", [])
            summary = {
                "source": description.get("source", ""),
                "fields": fields[:DETAIL_LIMIT * 2], "field_count": len(fields),
                "transforms": transforms[:DETAIL_LIMIT], "transform_count": len(transforms),
            }
            self._expanded[steps] = summary
            self._show_details(steps, summary)
        lf.ui.request_redraw()

    def _row_target(self, event):
        target = event.target()
        while target and target.id != "history-rows":
            if target.get_attribute("data-steps", ""):
                return target
            target = target.parent()
        return None

    def _on_row(self, event):
        target = event.target()
        while target and target.id != "history-rows":
            if "history-details" in target.get_attribute("class", "").split():
                event.stop_propagation()
                return
            steps = target.get_attribute("data-details", "")
            if steps:
                self._toggle_details(int(steps), int(target.get_attribute("data-generation", "-1")))
                event.stop_propagation()
                return
            target = target.parent()
        row = self._row_target(event)
        if row is None:
            return
        generation = int(row.get_attribute("data-generation", "-1"))
        if generation != lf.undo.generation():
            self._refresh()
            return
        self._jump(self._stack, int(row.get_attribute("data-steps", "0")))
        event.stop_propagation()

    def _on_key(self, event):
        key = int(event.get_parameter("key_identifier", "0"))
        if key in (KI_RETURN, KI_SPACE):
            self._on_row(event)
            return
        if key not in (KI_UP, KI_DOWN, KI_HOME, KI_END):
            return
        row = self._row_target(event)
        if row is None:
            return
        rows = self._doc.get_element_by_id("history-rows").query_selector_all(".history-jump")
        index = int(row.get_attribute("data-steps", "1")) - 1 - self._page * PAGE_SIZE
        index = (0 if key == KI_HOME else len(rows) - 1 if key == KI_END
                 else max(0, min(len(rows) - 1, index + (1 if key == KI_DOWN else -1))))
        rows[index].focus()
        rows[index].scroll_into_view(False)
        event.stop_propagation()

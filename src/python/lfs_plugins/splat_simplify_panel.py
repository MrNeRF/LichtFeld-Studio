# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Dedicated, reactive splat simplification tool."""
import math
import lichtfeld as lf
from . import rml_widgets as w
from .panels import panel_class
from .scrub_fields import ScrubFieldController, ScrubFieldSpec
from .default_reset import LabelDefaultReset, values_match
from .types import Panel
from .selected_splat import active_splat_node
from .ui import RuntimeState, PanelStateBinding, native_value as _native_store_value

__lfs_panel_classes__ = ["SplatSimplifyPanel"]
__lfs_panel_ids__ = ["lfs.splat_simplify"]
DEFAULT_SIMPLIFY_TARGET_RATIO = 0.5
DEFAULT_SIMPLIFY_LOD_BASE = 2.0
DEFAULT_SIMPLIFY_OPACITY_PRUNE_THRESHOLD = 0.1
SCRUB_FIELD_DEFS = {
    "simplify_target": ScrubFieldSpec(1.0, 1.0, 1.0, "%d", data_type=int),
    "simplify_lod_base": ScrubFieldSpec(0.1, 10.0, 0.1, "%.1f"),
    "simplify_opacity_prune_threshold": ScrubFieldSpec(0.0, 1.0, 0.01, "%.2f"),
}

def _tr_fallback(key, fallback):
    value = lf.ui.tr(key)
    return value if value and value != key else fallback

def _entry_label(text):
    text = str(text).strip()
    return text if text.endswith((":", "：")) else f"{text}:"

@panel_class("splat_simplify")
class SplatSimplifyPanel(Panel):
    def __init__(self):
        self._handle = None
        self._advanced = False
        self._last_label = ""
        self._simplify_target_count = 0
        self._simplify_target_touched = False
        self._simplify_lod_base = DEFAULT_SIMPLIFY_LOD_BASE
        self._simplify_lod_base_touched = False
        self._simplify_opacity_prune_threshold = DEFAULT_SIMPLIFY_OPACITY_PRUNE_THRESHOLD
        self._simplify_source_name = ""
        self._simplify_original_count = 0
        self._simplify_task_active = False
        self._simplify_progress_value = "0"
        self._simplify_progress_stage = ""
        self._simplify_error_text = ""
        self._default_reset = LabelDefaultReset(self._reset_default, self._apply_reset)
        self._scrub_fields = ScrubFieldController(SCRUB_FIELD_DEFS, self._get_scrub_value, self._set_scrub_value)
        self._reactive_binding = PanelStateBinding()

    def _dirty_model(self, *fields):
        if self._handle:
            for field in fields:
                self._handle.dirty(field)

    def _get_scrub_value(self, prop):
        return {"simplify_target": self._compute_simplify_target_count,
                "simplify_lod_base": self._compute_simplify_lod_base,
                "simplify_opacity_prune_threshold": self._compute_simplify_opacity_prune_threshold}[prop]()

    def _set_scrub_value(self, prop, value):
        {"simplify_target": self._set_simplify_target_count,
         "simplify_lod_base": self._set_simplify_lod_base,
         "simplify_opacity_prune_threshold": self._set_simplify_opacity_prune_threshold}[prop](value)

    def _reset_default(self, prop):
        if self._simplify_task_active or self._simplify_original_count <= 0:
            return None
        return {"simplify_target": self._default_simplify_target_count(),
                "simplify_lod_base": DEFAULT_SIMPLIFY_LOD_BASE,
                "simplify_opacity_prune_threshold": DEFAULT_SIMPLIFY_OPACITY_PRUNE_THRESHOLD}.get(prop)

    def _apply_reset(self, prop, value):
        if self._reset_default(prop) is None:
            return False
        self._scrub_fields.prepare_reset(prop)
        self._set_scrub_value(prop, value)
        if prop == "simplify_target":
            self._simplify_target_touched = False
        elif prop == "simplify_lod_base":
            self._simplify_lod_base_touched = False
        self._scrub_fields.sync_all()
        return values_match(self._get_scrub_value(prop), value)

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("splat_simplify")
        if model is None:
            return
        model.bind_func("advanced", lambda: self._advanced)
        model.bind_event("toggle_advanced", self._toggle_advanced)
        model.bind_func("panel_label", lambda: lf.ui.tr("rendering_panel.section_simplify"))
        model.bind("simplify_target",
                   lambda: f"{self._compute_simplify_target_count():,}",
                   lambda v: self._set_simplify_target_count(v))
        model.bind("simplify_lod_base",
                   lambda: f"{self._compute_simplify_lod_base():.1f}",
                   lambda v: self._set_simplify_lod_base(v))
        model.bind("simplify_opacity_prune_threshold",
                   lambda: f"{self._compute_simplify_opacity_prune_threshold():.2f}",
                   lambda v: self._set_simplify_opacity_prune_threshold(v))

        model.bind_func("label_simplify_source",
                         lambda: _entry_label(_tr_fallback("rendering_panel.simplify_source", "Source")))
        model.bind_func("label_simplify_select_source",
                         lambda: _tr_fallback("rendering_panel.simplify_select_source", "Select a splat node"))
        model.bind_func("label_simplify_target",
                         lambda: _entry_label(_tr_fallback("rendering_panel.simplify_target", "Target")))
        model.bind_func("label_simplify_target_stat",
                         lambda: _tr_fallback("rendering_panel.simplify_target", "Target"))
        model.bind_func("label_simplify_lod_base",
                         lambda: _entry_label(_tr_fallback("rendering_panel.simplify_lod_base", "LOD Base")))
        model.bind_func("label_simplify_opacity_prune",
                         lambda: _entry_label(_tr_fallback("rendering_panel.simplify_opacity_prune", "Opacity Prune")))
        model.bind_func("label_simplify_original",
                         lambda: _tr_fallback("rendering_panel.simplify_original", "Original"))
        model.bind_func("label_simplify_output",
                         lambda: _entry_label(_tr_fallback("rendering_panel.simplify_output", "Output")))
        model.bind_func("label_simplify_apply",
                         lambda: _tr_fallback("common.apply", "Apply"))
        model.bind_func("label_simplify_cancel",
                         lambda: _tr_fallback("common.cancel", "Cancel"))
        model.bind_func("simplify_cancel_tooltip", lambda:
                        "tooltip.simplify_cancel" if self._simplify_task_active else "common.cancel")
        model.bind_func("simplify_has_source", lambda: bool(self._simplify_source_name))
        model.bind_func("simplify_source_name", lambda: self._simplify_source_name)
        model.bind_func("simplify_original_count", lambda: f"{self._simplify_original_count:,}")
        model.bind_func("simplify_target_count", lambda: f"{self._compute_simplify_target_count():,}")
        model.bind_func("simplify_output_name", self._simplify_output_name)
        model.bind_func("simplify_can_apply", self._can_run_simplify)
        model.bind_func("simplify_show_progress", lambda: self._simplify_task_active)
        model.bind_func("simplify_progress_value", lambda: self._simplify_progress_value)
        model.bind_func("simplify_progress_pct", self._simplify_progress_pct)
        model.bind_func("simplify_progress_stage", lambda: self._simplify_progress_stage)
        model.bind_func("simplify_show_error", lambda: bool(self._simplify_error_text))
        model.bind_func("simplify_error_text", lambda: self._simplify_error_text)

        model.bind_event("simplify_apply", self._on_simplify_apply)
        model.bind_event("simplify_cancel", self._on_simplify_cancel)
        self._handle = model.get_handle()

    def on_mount(self, doc):
        super().on_mount(doc)
        self._sync_label()
        self._scrub_fields.mount(doc)
        self._default_reset.mount(doc)
        self._refresh_simplify_source(force=True)
        self._sync_simplify_task_state(force=True)
        self._reactive_binding.set_handle(self._handle).watch(
            RuntimeState.scene_generation, RuntimeState.selection_generation,
            RuntimeState.splat_simplify_state, RuntimeState.language_generation, dirty="*")

    def on_update(self, doc):
        self._sync_label()
        changed = self._refresh_simplify_source(force=False)
        changed |= self._sync_simplify_task_state(force=False)
        changed |= self._scrub_fields.sync_all()
        return changed

    def on_unmount(self, doc):
        self._reactive_binding.close()
        self._scrub_fields.unmount()
        self._default_reset.unmount()
        doc.remove_data_model("splat_simplify")
        self._handle = None

    def _active_splat_node(self):
        return active_splat_node()

    def _refresh_simplify_source(self, force: bool) -> bool:
        _node, source_name, source_count = self._active_splat_node()
        changed = force or source_name != self._simplify_source_name or source_count != self._simplify_original_count
        if not changed:
            return False

        self._simplify_source_name = source_name
        self._simplify_original_count = source_count
        if source_count > 0:
            if self._simplify_target_touched and self._simplify_target_count > 0:
                self._simplify_target_count = self._clamp_simplify_target_count(self._simplify_target_count, source_count)
            else:
                self._simplify_target_count = self._default_simplify_target_count(source_count)
            if self._simplify_lod_base_touched and self._simplify_lod_base > 0:
                self._simplify_lod_base = self._clamp_simplify_lod_base(self._simplify_lod_base)
            else:
                self._simplify_lod_base = DEFAULT_SIMPLIFY_LOD_BASE
        elif not self._simplify_target_touched:
            self._simplify_target_count = 0
        if source_count <= 0 and not self._simplify_lod_base_touched:
            self._simplify_lod_base = DEFAULT_SIMPLIFY_LOD_BASE
        self._sync_simplify_scrub_spec()
        self._dirty_model(
            "simplify_has_source",
            "simplify_source_name",
            "simplify_original_count",
            "simplify_target",
            "simplify_target_count",
            "simplify_lod_base",
            "simplify_opacity_prune_threshold",
            "simplify_output_name",
            "simplify_can_apply",
        )
        return True

    def _sync_simplify_scrub_spec(self):
        max_value = max(0, int(self._simplify_original_count))
        target_spec = ScrubFieldSpec(
            1.0 if max_value > 0 else 0.0,
            float(max_value),
            1.0,
            "%d",
            data_type=int,
        )
        lod_spec = ScrubFieldSpec(0.1, 10.0, 0.1, "%.1f")
        self._scrub_fields.set_spec("simplify_target", target_spec)
        self._scrub_fields.set_spec("simplify_lod_base", lod_spec)

    def _default_simplify_target_count(self, original_count=None) -> int:
        source_count = self._simplify_original_count if original_count is None else int(original_count)
        if source_count <= 0:
            return 0
        return max(1, min(source_count, int(math.ceil(source_count * DEFAULT_SIMPLIFY_TARGET_RATIO))))

    def _clamp_simplify_target_count(self, value, original_count=None):
        try:
            parsed = int(round(float(str(value).strip().replace(",", "").replace("_", ""))))
        except (TypeError, ValueError):
            return None
        clamped = max(1, parsed)
        max_count = self._simplify_original_count if original_count is None else int(original_count)
        if max_count > 0:
            clamped = min(clamped, max_count)
        return clamped

    def _compute_simplify_target_count(self) -> int:
        if self._simplify_original_count <= 0:
            return 0
        if self._simplify_target_count <= 0:
            return self._default_simplify_target_count()
        clamped = self._clamp_simplify_target_count(self._simplify_target_count)
        if clamped is None:
            return self._default_simplify_target_count()
        return clamped

    def _compute_simplify_ratio(self) -> float:
        if self._simplify_original_count <= 0:
            return 0.0
        return float(self._compute_simplify_target_count()) / float(self._simplify_original_count)

    def _clamp_simplify_lod_base(self, value):
        try:
            parsed = float(str(value).strip().replace(",", "").replace("_", ""))
        except (TypeError, ValueError):
            return None
        return max(0.1, min(parsed, 10.0))

    def _compute_simplify_lod_base(self) -> float:
        clamped = self._clamp_simplify_lod_base(self._simplify_lod_base)
        return DEFAULT_SIMPLIFY_LOD_BASE if clamped is None else clamped

    def _set_simplify_lod_base(self, value):
        next_value = self._clamp_simplify_lod_base(value)
        if next_value is None or math.isclose(next_value, self._simplify_lod_base, abs_tol=1.0e-9):
            return
        self._simplify_lod_base = next_value
        self._simplify_lod_base_touched = True
        self._dirty_model("simplify_lod_base")

    def _clamp_simplify_opacity_prune_threshold(self, value):
        try:
            parsed = float(str(value).strip().replace(",", "").replace("_", ""))
        except (TypeError, ValueError):
            return None
        return max(0.0, min(parsed, 1.0))

    def _compute_simplify_opacity_prune_threshold(self) -> float:
        clamped = self._clamp_simplify_opacity_prune_threshold(self._simplify_opacity_prune_threshold)
        return DEFAULT_SIMPLIFY_OPACITY_PRUNE_THRESHOLD if clamped is None else clamped

    def _set_simplify_opacity_prune_threshold(self, value):
        next_value = self._clamp_simplify_opacity_prune_threshold(value)
        if next_value is None or math.isclose(next_value, self._simplify_opacity_prune_threshold, abs_tol=1.0e-9):
            return
        self._simplify_opacity_prune_threshold = next_value
        self._dirty_model("simplify_opacity_prune_threshold")

    def _simplify_output_name(self) -> str:
        if not self._simplify_source_name:
            return ""
        return f"{self._simplify_source_name}_{self._compute_simplify_target_count()}"

    def _can_run_simplify(self) -> bool:
        return bool(
            self._simplify_source_name
            and self._simplify_original_count > 0
            and self._compute_simplify_target_count() > 0
            and not self._simplify_task_active
        )

    def _set_simplify_target_count(self, value):
        next_value = self._clamp_simplify_target_count(value)
        if next_value is None:
            return
        if next_value == self._simplify_target_count and self._simplify_target_touched:
            return
        self._simplify_target_count = next_value
        self._simplify_target_touched = True
        self._dirty_model("simplify_target", "simplify_target_count", "simplify_output_name")

    def _simplify_progress_pct(self) -> str:
        try:
            return f"{int(round(float(self._simplify_progress_value) * 100.0))}%"
        except (TypeError, ValueError):
            return "0%"

    def _simplify_task_state(self) -> dict[str, object]:
        state = _native_store_value("splat_simplify_state", None)
        if isinstance(state, dict):
            return state
        return {
            "active": bool(getattr(lf, "is_splat_simplify_active", lambda: False)()),
            "progress": getattr(lf, "get_splat_simplify_progress", lambda: 0.0)(),
            "stage": getattr(lf, "get_splat_simplify_stage", lambda: "")() or "",
            "error": getattr(lf, "get_splat_simplify_error", lambda: "")() or "",
        }

    def _sync_simplify_task_state(self, force: bool) -> bool:
        state = self._simplify_task_state()
        active = bool(state.get("active", False))
        progress = max(0.0, min(1.0, float(state.get("progress", 0.0))))
        progress_value = f"{progress:.4f}".rstrip("0").rstrip(".") or "0"
        stage = str(state.get("stage", "") or "")
        error_text = str(state.get("error", "") or "")

        changed = force or (
            active != self._simplify_task_active or
            progress_value != self._simplify_progress_value or
            stage != self._simplify_progress_stage or
            error_text != self._simplify_error_text
        )
        if not changed:
            return False

        self._simplify_task_active = active
        self._simplify_progress_value = progress_value
        self._simplify_progress_stage = stage
        self._simplify_error_text = error_text
        self._dirty_model(
            "simplify_can_apply",
            "simplify_show_progress",
            "simplify_cancel_tooltip",
            "simplify_progress_value",
            "simplify_progress_pct",
            "simplify_progress_stage",
            "simplify_show_error",
            "simplify_error_text",
        )
        return True

    def _start_simplify(self):
        if not self._can_run_simplify():
            return
        self._simplify_error_text = ""
        self._dirty_model("simplify_show_error", "simplify_error_text")
        lf.simplify_splats(
            self._simplify_source_name,
            ratio=self._compute_simplify_ratio(),
            lod_base=self._compute_simplify_lod_base(),
            opacity_prune_threshold=self._compute_simplify_opacity_prune_threshold(),
        )
        self._sync_simplify_task_state(force=True)

    def _on_simplify_apply(self, _handle=None, _ev=None, _args=None):
        self._start_simplify()

    def _on_simplify_cancel(self, _handle=None, _ev=None, _args=None):
        if not self._simplify_task_state().get("active", False):
            lf.ui.set_panel_enabled(self.id, False)
            return
        cancel = getattr(lf, "cancel_splat_simplify", None)
        if cancel is not None:
            cancel()


    def _toggle_advanced(self, *_):
        self._advanced = not self._advanced
        self._dirty_model("advanced")

    def _sync_label(self):
        label = lf.ui.tr("rendering_panel.section_simplify")
        if label != self._last_label:
            lf.ui.set_panel_label(self.id, label)
            self._last_label = label

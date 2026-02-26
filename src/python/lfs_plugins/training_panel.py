# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Training Panel - RmlUI implementation with hybrid ImGui fallback."""

import os
import time

import lichtfeld as lf

from .types import RmlPanel
from .ui.state import AppState


def tr(key):
    result = lf.ui.tr(key)
    return result if result else key


class IterationRateTracker:
    WINDOW_SECONDS = 5.0

    def __init__(self):
        self.samples = []

    def add_sample(self, iteration):
        now = time.monotonic()
        self.samples.append((iteration, now))
        self.samples = [(i, t) for i, t in self.samples if now - t <= self.WINDOW_SECONDS]

    def get_rate(self):
        if len(self.samples) < 2:
            return 0.0
        oldest = self.samples[0]
        newest = self.samples[-1]
        iter_diff = newest[0] - oldest[0]
        time_diff = newest[1] - oldest[1]
        return iter_diff / time_diff if time_diff > 0 else 0.0

    def clear(self):
        self.samples = []


_rate_tracker = IterationRateTracker()

CTRL_SECTIONS = ["ctrl-ready", "ctrl-running", "ctrl-paused",
                 "ctrl-completed", "ctrl-stopped", "ctrl-error", "ctrl-stopping"]

STATE_TO_CTRL = {
    "ready": "ctrl-ready",
    "running": "ctrl-running",
    "paused": "ctrl-paused",
    "completed": "ctrl-completed",
    "stopped": "ctrl-stopped",
    "error": "ctrl-error",
    "stopping": "ctrl-stopping",
}

BOOL_PROPS = {
    "cb-bilateral_grid": "use_bilateral_grid",
    "cb-invert_masks": "invert_masks",
    "cb-use_alpha_as_mask": "use_alpha_as_mask",
    "cb-sparsity": "enable_sparsity",
    "cb-gut": "gut",
    "cb-undistort": "undistort",
    "cb-mip_filter": "mip_filter",
    "cb-ppisp": "ppisp",
    "cb-ppisp_use_controller": "ppisp_use_controller",
    "cb-ppisp_freeze_gaussians": "ppisp_freeze_gaussians",
    "cb-random": "random",
    "cb-revised_opacity": "revised_opacity",
}

DATASET_BOOL_PROPS = {
    "cb-cpu_cache": "use_cpu_cache",
    "cb-fs_cache": "use_fs_cache",
}

NUM_PROPS_ON_PARAMS = [
    "iterations", "max_cap", "steps_scaler",
    "means_lr", "shs_lr", "opacity_lr", "scaling_lr", "rotation_lr",
    "refine_every", "start_refine", "stop_refine", "grad_threshold",
    "reset_every", "sh_degree_interval",
    "bilateral_grid_x", "bilateral_grid_y", "bilateral_grid_w", "bilateral_grid_lr",
    "opacity_reg", "scale_reg", "tv_loss_weight",
    "init_opacity", "init_scaling", "init_num_pts", "init_extent",
    "min_opacity", "prune_opacity", "grow_scale3d", "grow_scale2d",
    "prune_scale3d", "prune_scale2d", "pause_refine_after_reset",
    "sparsify_steps", "init_rho", "prune_ratio",
    "ppisp_controller_lr",
]

SLIDER_PROPS_ON_PARAMS = [
    "lambda_dssim", "init_opacity", "prune_ratio",
]


def _color_to_hex(c):
    r = int(c[0] * 255)
    g = int(c[1] * 255)
    b = int(c[2] * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def _hex_to_color(h):
    h = h.lstrip("#")
    if len(h) != 6:
        return None
    try:
        r = int(h[0:2], 16) / 255.0
        g = int(h[2:4], 16) / 255.0
        b = int(h[4:6], 16) / 255.0
        return (r, g, b)
    except ValueError:
        return None


class TrainingPanel(RmlPanel):
    idname = "lfs.training"
    label = "Training"
    space = "MAIN_PANEL_TAB"
    order = 20
    rml_template = "rmlui/training.rml"
    rml_height_mode = "content"

    def __init__(self):
        self._checkpoint_saved_time = 0.0
        self._new_save_step = 7000
        self._auto_scaled_for_cameras = 0
        self._last_lang = ""
        self._last_save_steps = []
        self._color_edit_prop = None
        self._color_picker_needs_pos = False
        self._slider_user_vals = {}

    def on_load(self, doc):
        self._last_lang = lf.ui.get_current_language()
        self._populate_labels(doc)
        self._populate_select_options(doc)

        body = doc.get_element_by_id("body")
        if body:
            body.add_event_listener("change", self._on_change)
            body.add_event_listener("click", self._on_click)

    def _populate_labels(self, doc):
        def _set(el_id, text):
            el = doc.get_element_by_id(el_id)
            if el:
                el.set_inner_rml(text)

        _set("text-hdr-basic_params", tr("training.section.basic_params"))
        _set("text-hdr-advanced_params", tr("training.section.advanced_params"))
        _set("text-hdr-dataset", tr("training.section.dataset"))
        _set("text-hdr-optimization", tr("training.section.optimization"))
        _set("text-hdr-bilateral", tr("training.section.bilateral_grid"))
        _set("text-hdr-losses", tr("training.section.losses"))
        _set("text-hdr-init", tr("training.section.initialization"))
        _set("text-hdr-adc", tr("training_panel.pruning_growing"))
        _set("text-hdr-sparsity", tr("training_panel.sparsity"))
        _set("text-hdr-save_steps", tr("training_panel.save_steps"))

        _set("label-strategy", tr("training_params.strategy"))
        _set("label-iterations", tr("training_params.iterations"))
        _set("label-max_cap", tr("training_params.max_gaussians"))
        _set("label-sh_degree", tr("training_params.sh_degree"))
        _set("label-tile_mode", tr("training_params.tile_mode"))
        _set("label-steps_scaler", tr("training_params.steps_scaler"))

        _set("text-bilateral_grid", tr("training_params.bilateral_grid"))
        _set("label-mask_mode", tr("training_params.mask_mode"))
        _set("text-invert_masks", tr("training_params.invert_masks"))
        _set("text-use_alpha_as_mask", tr("training_params.use_alpha_as_mask"))
        _set("text-sparsity", tr("training_params.sparsity"))
        _set("text-gut", tr("training_params.gut"))
        _set("text-undistort", tr("training_params.undistort"))
        _set("text-mip_filter", tr("training_params.mip_filter"))
        _set("text-ppisp", tr("training_params.ppisp"))
        _set("text-ppisp_controller", tr("training_params.ppisp_controller"))
        _set("text-ppisp_auto_step", tr("common.auto"))
        _set("label-ppisp_activation_step", tr("training_params.ppisp_activation_step"))
        _set("label-ppisp_controller_lr", tr("training_params.ppisp_controller_lr"))
        _set("text-ppisp_freeze_gaussians", tr("training_params.ppisp_freeze_gaussians"))
        _set("label-bg_mode", tr("training_params.bg_mode"))
        _set("label-bg_color", tr("training_params.bg_color"))
        _set("label-bg_image", tr("training_params.bg_image"))

        _set("label-dataset_path", tr("training.dataset.path"))
        _set("label-dataset_images", tr("training.dataset.images"))
        _set("label-resize_factor", tr("training.dataset.resize_factor"))
        _set("label-max_width", tr("training.dataset.max_width"))
        _set("text-cpu_cache", tr("training.dataset.cpu_cache"))
        _set("text-fs_cache", tr("training.dataset.fs_cache"))
        _set("label-dataset_output", tr("training.dataset.output"))
        _set("dataset-no-data", tr("training_panel.no_dataset_loaded"))

        _set("label-opt_strategy", tr("training_params.strategy"))
        _set("text-lr-header", tr("training.opt.learning_rates"))
        _set("label-means_lr", tr("training.opt.lr.position"))
        _set("label-shs_lr", tr("training.opt.lr.sh_coeff"))
        _set("label-opacity_lr", tr("training.opt.lr.opacity"))
        _set("label-scaling_lr", tr("training.opt.lr.scaling"))
        _set("label-rotation_lr", tr("training.opt.lr.rotation"))
        _set("text-refinement-header", tr("training.section.refinement"))
        _set("label-refine_every", tr("training.refinement.refine_every"))
        _set("label-start_refine", tr("training.refinement.start_refine"))
        _set("label-stop_refine", tr("training.refinement.stop_refine"))
        _set("label-grad_threshold", tr("training.refinement.gradient_thr"))
        _set("label-reset_every", tr("training.refinement.reset_every"))
        _set("label-sh_degree_interval", tr("training.refinement.sh_upgrade_every"))

        _set("label-bilateral_grid_x", tr("training.bilateral.grid_x"))
        _set("label-bilateral_grid_y", tr("training.bilateral.grid_y"))
        _set("label-bilateral_grid_w", tr("training.bilateral.grid_w"))
        _set("label-bilateral_grid_lr", tr("training.bilateral.learning_rate"))

        _set("label-lambda_dssim", tr("training.losses.lambda_dssim"))
        _set("label-opacity_reg", tr("training.losses.opacity_reg"))
        _set("label-scale_reg", tr("training.losses.scale_reg"))
        _set("label-tv_loss_weight", tr("training.losses.tv_loss_weight"))

        _set("label-init_opacity", tr("training.init.init_opacity"))
        _set("label-init_scaling", tr("training.init.init_scaling"))
        _set("text-random_init", tr("training.init.random_init"))
        _set("label-init_num_pts", tr("training.init.num_points"))
        _set("label-init_extent", tr("training.init.extent"))

        _set("label-min_opacity", tr("training.thresholds.min_opacity"))
        _set("label-prune_opacity", tr("training.thresholds.prune_opacity"))
        _set("label-grow_scale3d", tr("training.thresholds.grow_scale_3d"))
        _set("label-grow_scale2d", tr("training.thresholds.grow_scale_2d"))
        _set("label-prune_scale3d", tr("training.thresholds.prune_scale_3d"))
        _set("label-prune_scale2d", tr("training.thresholds.prune_scale_2d"))
        _set("label-pause_refine_after_reset", tr("training.thresholds.pause_after_reset"))
        _set("text-revised_opacity", tr("training.thresholds.revised_opacity"))

        _set("label-sparsify_steps", tr("training_params.sparsify_steps"))
        _set("label-init_rho", tr("training_params.init_rho"))
        _set("label-prune_ratio", tr("training_params.prune_ratio"))

        _set("msg-no-trainer", tr("training_panel.no_trainer_loaded"))
        _set("msg-no-params", tr("training_panel.parameters_unavailable"))

        _set("status-completed", tr("status.complete"))
        _set("status-stopped", tr("status.stopped"))
        _set("status-error-label", tr("status.error"))
        _set("status-stopping-label", tr("status.stopping"))
        _set("no-save-steps", tr("training_panel.no_save_steps"))
        _set("no-save-steps-ro", tr("training_panel.no_save_steps"))

        _set("btn-save-checkpoint", tr("training_panel.save_checkpoint"))
        _set("checkpoint-saved", tr("training_panel.checkpoint_saved"))
        _set("btn-add-step", tr("common.add"))
        _set("btn-bg-browse", tr("training_params.bg_image_browse"))
        _set("btn-bg-clear", tr("training_params.bg_image_clear"))

    def _populate_select_options(self, doc):
        def _set_option_text(sel_id, values):
            el = doc.get_element_by_id(sel_id)
            if not el:
                return
            rml = ""
            for val, text in values:
                rml += f'<option value="{val}">{text}</option>'
            el.set_inner_rml(rml)

        _set_option_text("sel-strategy", [
            ("mcmc", tr("training.options.strategy.mcmc")),
            ("adc", tr("training.options.strategy.adc")),
        ])
        _set_option_text("sel-tile_mode", [
            ("1", tr("training.options.tile.full")),
            ("2", tr("training.options.tile.half")),
            ("4", tr("training.options.tile.quarter")),
        ])
        _set_option_text("sel-mask_mode", [
            ("0", tr("training.options.mask.none")),
            ("1", tr("training.options.mask.segment")),
            ("2", tr("training.options.mask.ignore")),
            ("3", tr("training.options.mask.alpha_consistent")),
        ])
        _set_option_text("sel-bg_mode", [
            ("0", tr("training.options.bg.color")),
            ("1", tr("training.options.bg.modulation")),
            ("2", tr("training.options.bg.image")),
            ("3", tr("training.options.bg.random")),
        ])
        _set_option_text("sel-resize_factor", [
            ("-1", tr("common.auto")),
            ("1", "1"),
            ("2", "2"),
            ("4", "4"),
            ("8", "8"),
        ])

    def on_update(self, doc):
        if not AppState.has_trainer.value:
            self._set_visible(doc, "msg-no-trainer", True)
            self._set_visible(doc, "main-content", False)
            self._set_visible(doc, "msg-no-params", False)
            return

        params = lf.optimization_params()
        if not params.has_params():
            self._set_visible(doc, "msg-no-trainer", False)
            self._set_visible(doc, "msg-no-params", True)
            self._set_visible(doc, "main-content", False)
            return

        self._set_visible(doc, "msg-no-trainer", False)
        self._set_visible(doc, "msg-no-params", False)
        self._set_visible(doc, "main-content", True)

        cur_lang = lf.ui.get_current_language()
        if cur_lang != self._last_lang:
            self._last_lang = cur_lang
            self._populate_labels(doc)
            self._populate_select_options(doc)

        state = AppState.trainer_state.value
        iteration = AppState.iteration.value

        if state == "ready" and iteration == 0:
            self._try_auto_scale_steps(params)

        self._update_controls(doc, state, iteration)
        self._update_basic_params(doc, state, iteration, params)
        self._update_advanced_params(doc, state, iteration, params)
        self._update_status(doc, state, iteration)

    def _update_controls(self, doc, state, iteration):
        active_ctrl = STATE_TO_CTRL.get(state)
        for section_id in CTRL_SECTIONS:
            self._set_visible(doc, section_id, section_id == active_ctrl)

        if state == "ready":
            btn = doc.get_element_by_id("btn-start")
            if btn:
                label = tr("training_panel.resume_training") if iteration > 0 else tr("training_panel.start_training")
                btn.set_inner_rml(label)
            self._set_visible(doc, "btn-reset-ready", iteration > 0)
            btn_reset = doc.get_element_by_id("btn-reset-ready")
            if btn_reset:
                btn_reset.set_inner_rml(tr("training_panel.reset"))
            btn_clear = doc.get_element_by_id("btn-clear-ready")
            if btn_clear:
                btn_clear.set_inner_rml(tr("training_panel.clear"))

        elif state == "running":
            btn = doc.get_element_by_id("btn-pause")
            if btn:
                btn.set_inner_rml(tr("training_panel.pause"))

        elif state == "paused":
            self._set_btn_text(doc, "btn-resume", tr("training_panel.resume"))
            self._set_btn_text(doc, "btn-reset-paused", tr("training_panel.reset"))
            self._set_btn_text(doc, "btn-stop", tr("training_panel.stop"))

        elif state in ("completed", "stopped"):
            prefix = "completed" if state == "completed" else "stopped"
            self._set_btn_text(doc, f"btn-switch-edit{'' if state == 'completed' else '-stopped'}", tr("training_panel.switch_edit_mode"))
            self._set_btn_text(doc, f"btn-reset-{prefix}", tr("training_panel.reset"))
            self._set_btn_text(doc, f"btn-clear-{prefix}", tr("training_panel.clear"))

        elif state == "error":
            error_msg = lf.trainer_error()
            el = doc.get_element_by_id("error-message")
            if el:
                el.set_inner_rml(error_msg or "")
            self._set_btn_text(doc, "btn-reset-error", tr("training_panel.reset"))
            self._set_btn_text(doc, "btn-clear-error", tr("training_panel.clear"))

        show_checkpoint = state in ("running", "paused")
        self._set_visible(doc, "btn-save-checkpoint", show_checkpoint)
        show_saved = show_checkpoint and (time.time() - self._checkpoint_saved_time < 2.0)
        self._set_visible(doc, "checkpoint-saved", show_saved)

    def _update_basic_params(self, doc, state, iteration, params):
        can_edit = (state == "ready") and (iteration == 0)
        can_edit_live = state in ("ready", "running", "paused")

        self._set_disabled(doc, "struct-params", not can_edit)
        self._set_disabled(doc, "live-params", not can_edit_live)

        el = doc.get_element_by_id("sel-strategy")
        if el:
            el.set_attribute("value", params.strategy)

        self._set_num_value(doc, "num-iterations", int(params.iterations), "%d")
        self._set_num_value(doc, "num-max_cap", params.max_cap, "%d")

        el = doc.get_element_by_id("sel-sh_degree")
        if el:
            el.set_attribute("value", str(params.sh_degree))

        el = doc.get_element_by_id("sel-tile_mode")
        if el:
            el.set_attribute("value", str(params.tile_mode))

        self._set_num_value(doc, "num-steps_scaler", params.steps_scaler, "%.2f")

        for cb_id, prop in BOOL_PROPS.items():
            el = doc.get_element_by_id(cb_id)
            if el:
                val = getattr(params, prop, False)
                if val:
                    el.set_attribute("checked", "")
                else:
                    el.remove_attribute("checked")

        el = doc.get_element_by_id("sel-mask_mode")
        if el:
            el.set_attribute("value", str(params.mask_mode.value))
        self._set_visible(doc, "dep-mask_mode", params.mask_mode.value != 0)

        gut_disabled = params.strategy == "adc"
        row_gut = doc.get_element_by_id("row-gut")
        if row_gut:
            row_gut.set_class("disabled-overlay", gut_disabled)

        self._set_visible(doc, "dep-ppisp", params.ppisp)
        self._set_visible(doc, "dep-ppisp_controller", params.ppisp_use_controller)

        is_auto = params.ppisp_controller_activation_step < 0
        auto_cb = doc.get_element_by_id("cb-ppisp_auto_step")
        if auto_cb:
            if is_auto:
                auto_cb.set_attribute("checked", "")
            else:
                auto_cb.remove_attribute("checked")
        self._set_visible(doc, "dep-ppisp_manual_step", not is_auto)
        if not is_auto:
            self._set_num_value(doc, "num-ppisp_controller_activation_step",
                                params.ppisp_controller_activation_step, "%d")

        el = doc.get_element_by_id("sel-bg_mode")
        if el:
            el.set_attribute("value", str(params.bg_mode.value))

        bg_mode_val = params.bg_mode.value
        self._set_visible(doc, "dep-bg_color", bg_mode_val in (0, 1))
        self._set_visible(doc, "dep-bg_image", bg_mode_val == 2)

        if bg_mode_val in (0, 1):
            c = params.bg_color
            r, g, b = int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
            self._set_text(doc, "rc-bg_color", f"R:{r:>3d}")
            self._set_text(doc, "gc-bg_color", f"G:{g:>3d}")
            self._set_text(doc, "bc-bg_color", f"B:{b:>3d}")
            swatch = doc.get_element_by_id("swatch-bg_color")
            if swatch:
                swatch.set_property("background-color", f"rgb({r},{g},{b})")
            hex_el = doc.get_element_by_id("hex-bg_color")
            if hex_el:
                hex_el.set_attribute("value", _color_to_hex(c))

        if bg_mode_val == 2:
            img_path = params.bg_image_path
            display = os.path.basename(img_path) if img_path else tr("training.value.none")
            self._set_text(doc, "bg-image-path", display)
            self._set_visible(doc, "btn-bg-clear", bool(img_path))

    def _update_advanced_params(self, doc, state, iteration, params):
        can_edit = (state == "ready") and (iteration == 0)
        self._set_disabled(doc, "optimization-params", not can_edit)
        self._set_disabled(doc, "bilateral-params", not can_edit)
        self._set_disabled(doc, "losses-params", not can_edit)
        self._set_disabled(doc, "init-params", not can_edit)
        self._set_disabled(doc, "adc-params", not can_edit)
        self._set_disabled(doc, "sparsity-params", not can_edit)

        self._set_text(doc, "opt-strategy", params.strategy.upper())

        for prop_id in NUM_PROPS_ON_PARAMS:
            val = params.get(prop_id)
            if val is None:
                val = getattr(params, prop_id, None)
            if val is None:
                continue
            el = doc.get_element_by_id(f"num-{prop_id}")
            if el:
                fmt = el.get_attribute("data-fmt") or ""
                if fmt:
                    el.set_attribute("value", fmt % val)

        for prop_id in SLIDER_PROPS_ON_PARAMS:
            val = params.get(prop_id)
            if val is None:
                continue
            user_val = self._slider_user_vals.get(prop_id)
            if user_val is None or abs(float(val) - user_val) > 1e-6:
                el = doc.get_element_by_id(f"slider-{prop_id}")
                if el:
                    el.set_attribute("value", f"{val:.4g}")
                self._slider_user_vals.pop(prop_id, None)
            val_el = doc.get_element_by_id(f"val-{prop_id}")
            if val_el:
                val_el.set_inner_rml(f"{float(val):.3f}")

        self._set_visible(doc, "dep-bilateral", params.use_bilateral_grid)
        self._set_visible(doc, "dep-adc", params.strategy == "adc")
        self._set_visible(doc, "dep-sparsity", params.enable_sparsity)
        self._set_visible(doc, "dep-random", params.random)

        dataset = lf.dataset_params()
        has_dataset = dataset.has_params()
        dataset_can_edit = dataset.can_edit() if has_dataset else False

        self._set_visible(doc, "dataset-content", has_dataset)
        self._set_visible(doc, "dataset-no-data", not has_dataset)

        if has_dataset:
            self._set_disabled(doc, "dataset-content", not dataset_can_edit)

            data_path = dataset.data_path
            self._set_text(doc, "dataset-path",
                           os.path.basename(data_path) if data_path else tr("training.value.none"))

            images = dataset.images
            self._set_text(doc, "dataset-images",
                           images if images else tr("training.value.default"))

            el = doc.get_element_by_id("sel-resize_factor")
            if el:
                el.set_attribute("value", str(dataset.resize_factor))

            self._set_num_value(doc, "num-max_width", dataset.max_width, "%d")

            for cb_id, prop in DATASET_BOOL_PROPS.items():
                el = doc.get_element_by_id(cb_id)
                if el:
                    val = getattr(dataset, prop, False)
                    if val:
                        el.set_attribute("checked", "")
                    else:
                        el.remove_attribute("checked")

            out_path = dataset.output_path
            self._set_text(doc, "dataset-output",
                           os.path.basename(out_path) if out_path else tr("training.value.not_set"))

        self._update_save_steps(doc, params, can_edit)

    def _update_save_steps(self, doc, params, can_edit):
        self._set_visible(doc, "save-steps-edit", can_edit)
        self._set_visible(doc, "save-steps-readonly", not can_edit)

        steps = list(params.save_steps)

        if can_edit:
            if steps != self._last_save_steps:
                self._last_save_steps = steps[:]
                self._rebuild_save_steps_dom(doc, steps)
            self._set_visible(doc, "no-save-steps", not steps)
        else:
            if steps:
                self._set_text(doc, "save-steps-display", ", ".join(str(s) for s in steps))
            self._set_visible(doc, "save-steps-display", bool(steps))
            self._set_visible(doc, "no-save-steps-ro", not steps)

    def _rebuild_save_steps_dom(self, doc, steps):
        container = doc.get_element_by_id("save-steps-list")
        if not container:
            return
        container.set_inner_rml("")
        for i, step in enumerate(steps):
            row = container.append_child("div")
            row.set_class_names("setting-row")
            inp = row.append_child("input")
            inp.set_id(f"step-{i}")
            inp.set_attribute("type", "text")
            inp.set_class_names("number-input")
            inp.set_attribute("value", str(step))
            inp.set_attribute("data-step-idx", str(i))
            btn = row.append_child("button")
            btn.set_class_names("btn btn--secondary")
            btn.set_attribute("data-action", "remove_step")
            btn.set_attribute("data-step-idx", str(i))
            btn.set_inner_rml(tr("common.remove"))

    def _update_status(self, doc, state, iteration):
        state_labels = {
            "idle": tr("training_panel.idle"),
            "ready": tr("status.ready") if iteration == 0 else tr("training_panel.resume"),
            "running": tr("training_panel.running"),
            "paused": tr("status.paused"),
            "stopping": tr("status.stopping"),
            "completed": tr("status.complete"),
            "stopped": tr("status.stopped"),
            "error": tr("status.error"),
        }
        self._set_text(doc, "status-mode",
                        f"{tr('status.mode')}: {state_labels.get(state, tr('status.unknown'))}")

        _rate_tracker.add_sample(iteration)
        rate = _rate_tracker.get_rate()
        self._set_text(doc, "status-iteration",
                        f"{tr('status.iteration')} {iteration:,} ({rate:.1f} {tr('training_panel.iters_per_sec')})")
        self._set_text(doc, "status-gaussians",
                        tr("progress.num_splats") % f"{AppState.num_gaussians.value:,}")

        max_iter = AppState.max_iterations.value
        show_progress = max_iter > 0 and iteration > 0
        self._set_visible(doc, "progress-wrapper", show_progress)
        if show_progress:
            frac = iteration / max_iter
            prog = doc.get_element_by_id("training-progress")
            if prog:
                prog.set_attribute("value", str(frac))
            self._set_text(doc, "progress-text", f"{iteration:,}/{max_iter:,}")

    def draw_imgui(self, layout):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return

        if self._color_picker_needs_pos:
            layout.set_next_window_pos(layout.get_mouse_pos())
            layout.open_popup("##training_color_picker")
            self._color_picker_needs_pos = False

        if self._color_edit_prop and layout.begin_popup("##training_color_picker"):
            prop_id = self._color_edit_prop
            val = getattr(params, prop_id)
            changed, new_color = layout.color_picker3("##picker", val)
            if changed:
                setattr(params, prop_id, new_color)
                rs = lf.get_render_settings()
                if rs and prop_id == "bg_color":
                    rs.set("background_color", new_color)
            layout.end_popup()
        elif self._color_edit_prop:
            self._color_edit_prop = None

        loss_data = lf.loss_buffer()
        if loss_data:
            min_val = min(loss_data)
            max_val = max(loss_data)
            if min_val == max_val:
                min_val -= 1.0
                max_val += 1.0
            else:
                margin = (max_val - min_val) * 0.05
                min_val -= margin
                max_val += margin
            loss_label = f"{tr('status.loss')}: {loss_data[-1]:.4f}"
            layout.plot_lines(loss_label, loss_data, min_val, max_val, (-1, 60))

    def _on_change(self, event):
        target = event.target()
        if not target:
            return

        tag = target.get_attribute("id") or ""
        prop = target.get_attribute("data-prop") or ""

        if tag.startswith("cb-"):
            self._handle_checkbox_change(tag, prop, target)
        elif tag.startswith("sel-"):
            self._handle_select_change(tag, prop, target)
        elif tag.startswith("num-"):
            self._handle_number_change(tag, prop, target)
        elif tag.startswith("slider-"):
            self._handle_slider_change(tag, prop, target)
        elif tag.startswith("hex-"):
            self._handle_hex_change(tag, prop, target)
        elif tag.startswith("step-"):
            self._handle_step_edit(tag, target)

    def _handle_checkbox_change(self, tag, prop, target):
        val = target.has_attribute("checked")

        if tag in DATASET_BOOL_PROPS:
            dataset = lf.dataset_params()
            if dataset.has_params():
                setattr(dataset, DATASET_BOOL_PROPS[tag], val)
            return

        if tag == "cb-ppisp_auto_step":
            params = lf.optimization_params()
            if not params.has_params():
                return
            if val:
                params.ppisp_controller_activation_step = -1
            else:
                params.ppisp_controller_activation_step = max(1, int(params.iterations) - 5000)
            return

        if not prop:
            return
        params = lf.optimization_params()
        if not params.has_params():
            return
        setattr(params, prop, val)

        rs = lf.get_render_settings()
        if rs:
            if prop == "gut":
                rs.set("gut", val)
            elif prop == "mip_filter":
                rs.set("mip_filter", val)
            elif prop == "ppisp":
                rs.set("apply_appearance_correction", val)

    def _handle_select_change(self, tag, prop, target):
        val = target.get_attribute("value")
        if val is None:
            return

        if tag == "sel-strategy":
            params = lf.optimization_params()
            if not params.has_params():
                return
            if val == "adc" and params.gut:
                btn_gut = tr("training.conflict.btn_disable_gut")
                btn_cancel = tr("training.conflict.btn_cancel")

                def _on_conflict(button, _gut=btn_gut):
                    p = lf.optimization_params()
                    if button == _gut:
                        p.gut = False
                        p.set_strategy("adc")

                lf.ui.confirm_dialog(
                    tr("training.error.adc_gut_title"),
                    tr("training.conflict.adc_gut_strategy_message"),
                    [btn_gut, btn_cancel],
                    _on_conflict)
            else:
                params.set_strategy(val)
            return

        if tag == "sel-mask_mode":
            params = lf.optimization_params()
            if params.has_params():
                params.mask_mode = lf.MaskMode(int(val))
            return

        if tag == "sel-bg_mode":
            params = lf.optimization_params()
            if params.has_params():
                params.bg_mode = lf.BackgroundMode(int(val))
            return

        if tag == "sel-sh_degree":
            params = lf.optimization_params()
            if params.has_params():
                params.sh_degree = int(val)
            return

        if tag == "sel-tile_mode":
            params = lf.optimization_params()
            if params.has_params():
                params.tile_mode = int(val)
            return

        if tag == "sel-resize_factor":
            dataset = lf.dataset_params()
            if dataset.has_params():
                dataset.resize_factor = int(val)
            return

    def _handle_number_change(self, tag, prop, target):
        raw = target.get_attribute("value") or ""
        data_type = target.get_attribute("data-type") or "float"
        data_min = target.get_attribute("data-min")
        data_max = target.get_attribute("data-max")

        try:
            if data_type == "int":
                val = int(raw)
            else:
                val = float(raw)
        except (ValueError, TypeError):
            return

        if data_min is not None:
            try:
                min_v = int(data_min) if data_type == "int" else float(data_min)
                val = max(val, min_v)
            except (ValueError, TypeError):
                pass
        if data_max is not None:
            try:
                max_v = int(data_max) if data_type == "int" else float(data_max)
                val = min(val, max_v)
            except (ValueError, TypeError):
                pass

        if not prop:
            return

        if prop == "max_width":
            dataset = lf.dataset_params()
            if dataset.has_params() and 0 < val <= 4096:
                dataset.max_width = val
            return

        if prop == "iterations":
            params = lf.optimization_params()
            if params.has_params() and val > 0:
                params.iterations = val
            return

        if prop == "max_cap":
            params = lf.optimization_params()
            if params.has_params() and val > 0:
                params.max_cap = val
            return

        if prop == "steps_scaler":
            params = lf.optimization_params()
            if params.has_params():
                params.apply_step_scaling(val)
            return

        if prop == "ppisp_controller_activation_step":
            params = lf.optimization_params()
            if params.has_params():
                params.ppisp_controller_activation_step = max(1, int(val))
            return

        if prop == "ppisp_controller_lr":
            params = lf.optimization_params()
            if params.has_params():
                params.ppisp_controller_lr = val
            return

        params = lf.optimization_params()
        if params.has_params():
            if prop in ("means_lr", "shs_lr", "opacity_lr", "scaling_lr", "rotation_lr"):
                setattr(params, prop, val)
            else:
                params.set(prop, val)

    def _handle_slider_change(self, tag, prop, target):
        if not prop:
            return
        try:
            val = float(target.get_attribute("value"))
        except (ValueError, TypeError):
            return
        params = lf.optimization_params()
        if params.has_params():
            params.set(prop, val)
            self._slider_user_vals[prop] = val

    def _handle_hex_change(self, tag, prop, target):
        if not prop:
            return
        hex_val = target.get_attribute("value")
        if not hex_val:
            return
        color = _hex_to_color(hex_val)
        if not color:
            return
        params = lf.optimization_params()
        if params.has_params():
            setattr(params, prop, color)
            rs = lf.get_render_settings()
            if rs and prop == "bg_color":
                rs.set("background_color", color)

    def _handle_step_edit(self, tag, target):
        idx_str = target.get_attribute("data-step-idx")
        if idx_str is None:
            return
        try:
            idx = int(idx_str)
            new_val = int(target.get_attribute("value") or "0")
        except (ValueError, TypeError):
            return
        if new_val <= 0:
            return
        params = lf.optimization_params()
        if not params.has_params():
            return
        steps = list(params.save_steps)
        if 0 <= idx < len(steps):
            old = steps[idx]
            if new_val != old:
                params.remove_save_step(old)
                params.add_save_step(new_val)
                self._last_save_steps = []

    def _on_click(self, event):
        target = event.target()
        if not target:
            return

        body = event.current_target()
        el = target
        while el:
            cls = el.get_class_names()

            if "section-header" in cls:
                section_name = el.get_attribute("data-section")
                if section_name and body:
                    sec_el = body.get_element_by_id(f"sec-{section_name}")
                    if sec_el:
                        is_collapsed = sec_el.is_class_set("collapsed")
                        sec_el.set_class("collapsed", not is_collapsed)
                        arrow = el.query_selector(".section-arrow")
                        if arrow:
                            arrow.set_inner_rml("\u25BC" if is_collapsed else "\u25B6")
                return

            if "color-swatch" in cls:
                prop_id = el.get_attribute("data-prop")
                if prop_id:
                    if self._color_edit_prop == prop_id:
                        self._color_edit_prop = None
                    else:
                        self._color_edit_prop = prop_id
                        self._color_picker_needs_pos = True
                return

            action = el.get_attribute("data-action")
            if action:
                self._dispatch_action(action, el)
                return

            el = el.parent()

    def _dispatch_action(self, action, el):
        if action == "start":
            params = lf.optimization_params()
            error = params.validate() if params.has_params() else ""
            if error:
                btn_mcmc = tr("training.conflict.btn_use_mcmc")
                btn_gut = tr("training.conflict.btn_disable_gut")
                btn_cancel = tr("training.conflict.btn_cancel")

                def _on_start_conflict(button, _mcmc=btn_mcmc, _gut=btn_gut):
                    p = lf.optimization_params()
                    if button == _mcmc:
                        p.set_strategy("mcmc")
                        lf.start_training()
                    elif button == _gut:
                        p.gut = False
                        lf.start_training()

                lf.ui.confirm_dialog(
                    tr("training.error.adc_gut_title"),
                    tr("training.conflict.adc_gut_start_message"),
                    [btn_mcmc, btn_gut, btn_cancel],
                    _on_start_conflict)
            else:
                lf.start_training()

        elif action == "pause":
            lf.pause_training()
        elif action == "resume":
            lf.resume_training()
        elif action == "stop":
            lf.stop_training()
        elif action == "reset":
            lf.reset_training()
        elif action == "clear":
            lf.clear_scene()
        elif action == "switch_edit":
            lf.switch_to_edit_mode()
        elif action == "save_checkpoint":
            lf.save_checkpoint()
            self._checkpoint_saved_time = time.time()
        elif action == "browse_bg":
            selected = lf.ui.open_image_file_dialog("")
            if selected:
                params = lf.optimization_params()
                if params.has_params():
                    params.bg_image_path = selected
        elif action == "clear_bg":
            params = lf.optimization_params()
            if params.has_params():
                params.bg_image_path = ""
        elif action == "add_step":
            params = lf.optimization_params()
            if params.has_params():
                step_el = el.parent()
                if step_el:
                    inp = step_el.query_selector("#num-new_step")
                    if inp:
                        try:
                            val = int(inp.get_attribute("value") or str(self._new_save_step))
                        except (ValueError, TypeError):
                            val = self._new_save_step
                    else:
                        val = self._new_save_step
                else:
                    val = self._new_save_step
                if val > 0:
                    params.add_save_step(val)
                    self._last_save_steps = []
        elif action == "remove_step":
            idx_str = el.get_attribute("data-step-idx")
            if idx_str is not None:
                try:
                    idx = int(idx_str)
                except (ValueError, TypeError):
                    return
                params = lf.optimization_params()
                if params.has_params():
                    steps = list(params.save_steps)
                    if 0 <= idx < len(steps):
                        params.remove_save_step(steps[idx])
                        self._last_save_steps = []

    def _try_auto_scale_steps(self, params):
        scene = lf.get_scene()
        if scene is None:
            return
        camera_count = scene.active_camera_count
        if camera_count == 0 or camera_count == self._auto_scaled_for_cameras:
            return
        self._auto_scaled_for_cameras = camera_count
        params.auto_scale_steps(camera_count)

    def _sync_render_setting(self, name, value):
        rs = lf.get_render_settings()
        if rs:
            rs.set(name, value)

    @staticmethod
    def _set_visible(doc, el_id, visible):
        el = doc.get_element_by_id(el_id)
        if el:
            el.set_class("hidden", not visible)

    @staticmethod
    def _set_disabled(doc, el_id, disabled):
        el = doc.get_element_by_id(el_id)
        if el:
            el.set_class("disabled-overlay", disabled)

    @staticmethod
    def _set_text(doc, el_id, text):
        el = doc.get_element_by_id(el_id)
        if el:
            el.set_inner_rml(text)

    @staticmethod
    def _set_btn_text(doc, el_id, text):
        el = doc.get_element_by_id(el_id)
        if el:
            el.set_inner_rml(text)

    @staticmethod
    def _set_num_value(doc, el_id, value, fmt):
        el = doc.get_element_by_id(el_id)
        if el:
            el.set_attribute("value", fmt % value)

# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Training Panel - RmlUI with native data binding."""

import os
import re
import time

import lichtfeld as lf

from . import rml_widgets as w
from .scrub_fields import ScrubFieldController, ScrubFieldSpec
from .types import Panel
from .ui.state import AppState

__lfs_panel_classes__ = ["TrainingPanel"]
__lfs_panel_ids__ = ["lfs.training"]


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
        self.samples = [
            (i, t) for i, t in self.samples if now - t <= self.WINDOW_SECONDS
        ]

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


def _is_mrnf_strategy(strategy):
    return strategy in ("mrnf", "mnrf", "lfs")


LOCALE_KEYS = {
    "hdr_basic_params": "training.section.basic_params",
    "hdr_advanced_params": "training.section.advanced_params",
    "hdr_dataset": "training.section.dataset",
    "hdr_optimization": "training.section.optimization",
    "hdr_bilateral": "training.section.bilateral_grid",
    "hdr_losses": "training.section.losses",
    "hdr_init": "training.section.initialization",
    "hdr_pruning_growing": "training_panel.pruning_growing",
    "hdr_mrnf": "training_panel.mrnf_params",
    "hdr_sparsity": "training_panel.sparsity",
    "hdr_save_steps": "training_panel.save_eval_steps",
    "strategy": "training_params.strategy",
    "iterations": "training_params.iterations",
    "max_cap": "training_params.max_gaussians",
    "sh_degree": "training_params.sh_degree",
    "tile_mode": "training_params.tile_mode",
    "steps_scaler": "training_params.steps_scaler",
    "bilateral_grid": "training_params.bilateral_grid",
    "mask_mode": "training_params.mask_mode",
    "invert_masks": "training_params.invert_masks",
    "opacity_penalty_weight": "training.masking.penalty_weight",
    "opacity_penalty_power": "training.masking.penalty_power",
    "mask_threshold": "training.masking.threshold",
    "use_alpha_as_mask": "training_params.use_alpha_as_mask",
    "sparsity": "training_params.sparsity",
    "gut": "training_params.gut",
    "undistort": "training_params.undistort",
    "mip_filter": "training_params.mip_filter",
    "ppisp": "training_params.ppisp",
    "ppisp_controller": "training_params.ppisp_controller",
    "ppisp_freeze_from_sidecar": "training_params.ppisp_freeze_from_sidecar",
    "ppisp_sidecar_path": "training_params.ppisp_sidecar_path",
    "ppisp_activation_step": "training_params.ppisp_activation_step",
    "ppisp_controller_lr": "training_params.ppisp_controller_lr",
    "ppisp_freeze_gaussians": "training_params.ppisp_freeze_gaussians",
    "bg_mode": "training_params.bg_mode",
    "bg_color": "training_params.bg_color",
    "bg_image": "training_params.bg_image",
    "dataset_path": "training.dataset.path",
    "dataset_images": "training.dataset.images",
    "resize_factor": "training.dataset.resize_factor",
    "max_width": "training.dataset.max_width",
    "cpu_cache": "training.dataset.cpu_cache",
    "fs_cache": "training.dataset.fs_cache",
    "dataset_output": "training.dataset.output",
    "auto": "common.auto",
    "no_dataset": "training_panel.no_dataset_loaded",
    "opt_strategy": "training_params.strategy",
    "lr_header": "training.opt.learning_rates",
    "means_lr": "training.opt.lr.position",
    "shs_lr": "training.opt.lr.sh_coeff",
    "opacity_lr": "training.opt.lr.opacity",
    "scaling_lr": "training.opt.lr.scaling",
    "rotation_lr": "training.opt.lr.rotation",
    "refinement_header": "training.section.refinement",
    "refine_every": "training.refinement.refine_every",
    "start_refine": "training.refinement.start_refine",
    "stop_refine": "training.refinement.stop_refine",
    "grow_until_iter": "training.refinement.grow_until_iter",
    "grad_threshold": "training.refinement.gradient_thr",
    "reset_every": "training.refinement.reset_every",
    "sh_degree_interval": "training.refinement.sh_upgrade_every",
    "bilateral_grid_x": "training.bilateral.grid_x",
    "bilateral_grid_y": "training.bilateral.grid_y",
    "bilateral_grid_w": "training.bilateral.grid_w",
    "bilateral_grid_lr": "training.bilateral.learning_rate",
    "lambda_dssim": "training.losses.lambda_dssim",
    "opacity_reg": "training.losses.opacity_reg",
    "scale_reg": "training.losses.scale_reg",
    "tv_loss_weight": "training.losses.tv_loss_weight",
    "init_opacity": "training.init.init_opacity",
    "init_scaling": "training.init.init_scaling",
    "random_init": "training.init.random_init",
    "init_num_pts": "training.init.num_points",
    "init_extent": "training.init.extent",
    "min_opacity": "training.thresholds.min_opacity",
    "prune_opacity": "training.thresholds.prune_opacity",
    "grow_scale3d": "training.thresholds.grow_scale_3d",
    "grow_scale2d": "training.thresholds.grow_scale_2d",
    "prune_scale3d": "training.thresholds.prune_scale_3d",
    "prune_scale2d": "training.thresholds.prune_scale_2d",
    "pause_refine_after_reset": "training.thresholds.pause_after_reset",
    "revised_opacity": "training.thresholds.revised_opacity",
    "sparsify_steps": "training_params.sparsify_steps",
    "init_rho": "training_params.init_rho",
    "prune_ratio": "training_params.prune_ratio",
    "no_trainer": "training_panel.no_trainer_loaded",
    "no_params": "training_panel.parameters_unavailable",
    "no_save_steps": "training_panel.no_save_steps",
    "save_checkpoint": "training_panel.save_checkpoint",
    "checkpoint_saved": "training_panel.checkpoint_saved",
    "add": "common.add",
    "remove": "common.remove",
    "bg_browse": "training_params.bg_image_browse",
    "bg_clear": "training_params.bg_image_clear",
    "strategy_mcmc": "training.options.strategy.mcmc",
    "strategy_mrnf": "training.options.strategy.mrnf",
    "strategy_igs_plus": "training.options.strategy.igs_plus",
    "tile_full": "training.options.tile.full",
    "tile_half": "training.options.tile.half",
    "tile_quarter": "training.options.tile.quarter",
    "mask_none": "training.options.mask.none",
    "mask_segment": "training.options.mask.segment",
    "mask_ignore": "training.options.mask.ignore",
    "mask_alpha_consistent": "training.options.mask.alpha_consistent",
    "bg_option_color": "training.options.bg.color",
    "bg_option_modulation": "training.options.bg.modulation",
    "bg_option_image": "training.options.bg.image",
    "bg_option_random": "training.options.bg.random",
    "bg_color_red_prefix": "training_panel.color_red_prefix",
    "bg_color_green_prefix": "training_panel.color_green_prefix",
    "bg_color_blue_prefix": "training_panel.color_blue_prefix",
    "enable_eval": "training_params.enable_eval",
    "test_every": "training.dataset.test_every",
}

STRATEGY_LABEL_KEYS = {
    "mcmc": "training.options.strategy.mcmc",
    "mrnf": "training.options.strategy.mrnf",
    "mnrf": "training.options.strategy.mrnf",
    "lfs": "training.options.strategy.mrnf",
    "igs+": "training.options.strategy.igs_plus",
}

PARAM_BOOL_PROPS = [
    "use_bilateral_grid",
    "invert_masks",
    "use_alpha_as_mask",
    "enable_sparsity",
    "gut",
    "undistort",
    "mip_filter",
    "ppisp",
    "ppisp_use_controller",
    "ppisp_freeze_from_sidecar",
    "ppisp_freeze_gaussians",
    "random",
    "revised_opacity",
    "enable_eval",
]

DATASET_BOOL_PROPS = ["use_cpu_cache", "use_fs_cache"]

# (prop, type, format, min, max)
NUM_PROP_DEFS = [
    # (name, dtype, format, min, max, step)
    ("iterations", int, "%d", 1, None, 100),
    ("max_cap", int, "%d", 1, None, 100000),
    ("steps_scaler", float, "%.2f", 0.01, None, 0.1),
    ("means_lr", float, "%.6f", 0, None, 0.000001),
    ("shs_lr", float, "%.4f", 0, None, 0.0001),
    ("opacity_lr", float, "%.4f", 0, None, 0.001),
    ("scaling_lr", float, "%.4f", 0, None, 0.0001),
    ("rotation_lr", float, "%.4f", 0, None, 0.0001),
    ("refine_every", int, "%d", 1, None, 10),
    ("start_refine", int, "%d", 0, None, 100),
    ("stop_refine", int, "%d", 0, None, 1000),
    ("grow_until_iter", int, "%d", 0, None, 1000),
    ("grad_threshold", float, "%.6f", 0, None, 0.00001),
    ("reset_every", int, "%d", 1, None, 100),
    ("sh_degree_interval", int, "%d", 1, None, 100),
    ("bilateral_grid_x", int, "%d", 1, None, 1),
    ("bilateral_grid_y", int, "%d", 1, None, 1),
    ("bilateral_grid_w", int, "%d", 1, None, 1),
    ("bilateral_grid_lr", float, "%.6f", 0, None, 0.00001),
    ("mask_opacity_penalty_weight", float, "%.3f", 0, None, 0.1),
    ("mask_opacity_penalty_power", float, "%.3f", 0.5, None, 0.1),
    ("mask_threshold", float, "%.3f", 0, 1, 0.05),
    ("opacity_reg", float, "%.4f", 0, None, 0.001),
    ("scale_reg", float, "%.4f", 0, None, 0.001),
    ("tv_loss_weight", float, "%.1f", 0, None, 0.5),
    ("init_scaling", float, "%.3f", 0.001, None, 0.01),
    ("init_num_pts", int, "%d", 1, None, 10000),
    ("init_extent", float, "%.1f", 0.1, None, 0.5),
    ("min_opacity", float, "%.4f", 0, None, 0.001),
    ("prune_opacity", float, "%.4f", 0, None, 0.001),
    ("grow_scale3d", float, "%.4f", 0, None, 0.001),
    ("grow_scale2d", float, "%.3f", 0, None, 0.01),
    ("prune_scale3d", float, "%.3f", 0, None, 0.01),
    ("prune_scale2d", float, "%.3f", 0, None, 0.01),
    ("pause_refine_after_reset", int, "%d", 0, None, 100),
    ("sparsify_steps", int, "%d", 1, None, 1000),
    ("init_rho", float, "%.4f", 0, None, 0.001),
    ("ppisp_controller_lr", float, "%.5f", 0, None, 0.0001),
]

_NUM_PROP_LOOKUP = {
    name: (dtype, fmt, min_v, max_v, step)
    for name, dtype, fmt, min_v, max_v, step in NUM_PROP_DEFS
}

_INT_INPUT_RE = re.compile(r"^\s*[+-]?\d[\d,]*\s*$")

_FLOAT_INPUT_RE = re.compile(
    r"""
    ^\s*
    [+-]?
    (?:
        (?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d*)?
        |
        \.\d+
    )
    (?:[eE][+-]?\d+)?
    \s*$
    """,
    re.VERBOSE,
)


def _fmt_num(val, dtype, fmt):
    if dtype == int:
        return f"{int(val):,}"
    return fmt % val


def _parse_num(val_str, dtype):
    value = str(val_str).strip()
    if dtype == int:
        if not _INT_INPUT_RE.fullmatch(value):
            raise ValueError(f"invalid integer input: {val_str!r}")
        return value.replace(",", "")

    if not _FLOAT_INPUT_RE.fullmatch(value):
        raise ValueError(f"invalid numeric input: {val_str!r}")
    return value.replace(",", "")


def _resolved_ppisp_activation_step(
    params,
):  # Must match OptimizationParameters::resolved_ppisp_controller_activation_step()
    if params is None or not params.has_params():
        return 0
    scaler = max(float(getattr(params, "steps_scaler", 1.0)), 1.0)
    iterations = int(getattr(params, "iterations", 0))
    tail_iters = int(5000.0 * scaler + 0.5)
    return max(0, iterations - tail_iters)


def _display_ppisp_activation_step(params):
    if params is None or not params.has_params():
        return 0
    step = int(getattr(params, "ppisp_controller_activation_step", -1))
    return step if step >= 0 else _resolved_ppisp_activation_step(params)


SLIDER_PROPS = ["lambda_dssim", "init_opacity", "prune_ratio"]

SCRUB_FIELD_DEFS = {
    "lambda_dssim": ScrubFieldSpec(0.0, 1.0, 0.01, "%.3f"),
    "init_opacity": ScrubFieldSpec(0.01, 1.0, 0.01, "%.3f"),
    "prune_ratio": ScrubFieldSpec(0.0, 1.0, 0.01, "%.3f"),
}

RENDER_SYNC = {
    "gut": "gut",
    "mip_filter": "mip_filter",
    "ppisp": "apply_appearance_correction",
}

SECTIONS = [
    "basic_params",
    "advanced_params",
    "dataset",
    "optimization",
    "bilateral",
    "losses",
    "init",
    "pruning_growing",
    "sparsity",
    "save_steps",
]

INITIALLY_COLLAPSED = {
    "advanced_params",
    "dataset",
    "optimization",
    "bilateral",
    "losses",
    "init",
    "pruning_growing",
    "sparsity",
    "save_steps",
}


def _color_to_hex(c):
    return f"#{int(c[0] * 255):02x}{int(c[1] * 255):02x}{int(c[2] * 255):02x}"


def _hex_to_color(h):
    h = h.lstrip("#")
    if len(h) != 6:
        return None
    try:
        return (
            int(h[0:2], 16) / 255.0,
            int(h[2:4], 16) / 255.0,
            int(h[4:6], 16) / 255.0,
        )
    except ValueError:
        return None


class TrainingPanel(Panel):
    id = "lfs.training"
    label = "Training"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 20
    template = "rmlui/training.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    update_interval_ms = 16

    def __init__(self):
        self._handle = None
        self._checkpoint_saved_time = 0.0
        self._new_save_step = 7000
        self._auto_scaled_for_cameras = 0
        self._last_state = ""
        self._last_save_steps = []
        self._color_edit_prop = None
        self._picker_click_handled = False
        self._collapsed = set(INITIALLY_COLLAPSED)
        self._last_iteration = -1
        self._last_num_gaussians = -1
        self._last_progress_frac = -1.0
        self._last_bg_color = None
        self._doc = None
        self._popup_el = None
        self._loss_graph_el = None
        self._step_repeat_prop = None
        self._step_repeat_dir = 0
        self._step_repeat_start = 0.0
        self._step_repeat_last = 0.0
        self._text_bufs = {}
        self._last_checkpoint_saved_visible = False
        self._last_loss_signature = None
        self._psnr_graph_el = None
        self._last_psnr_signature = None
        self._progress_value = "0"
        self._loss_label = ""
        self._loss_tick_max = ""
        self._loss_tick_mid = ""
        self._loss_tick_min = ""
        self._psnr_label = ""
        self._psnr_tick_max = ""
        self._psnr_tick_mid = ""
        self._psnr_tick_min = ""
        self._last_panel_label = ""
        self._escape_revert = w.EscapeRevertController()
        self._scrub_fields = ScrubFieldController(
            SCRUB_FIELD_DEFS,
            self._get_scrub_value,
            self._set_scrub_value,
        )

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("training")
        if model is None:
            return

        p = lf.optimization_params
        d = lf.dataset_params

        self._bind_labels(model)
        self._bind_visibility(model, p, d)
        self._bind_disabled(model, p)
        self._bind_bool_props(model, p)
        self._bind_dataset_bools(model, d)
        self._bind_select_props(model, p, d)
        self._bind_text_props(model, p)
        self._bind_num_props(model, p, d)
        self._bind_slider_props(model, p)
        self._bind_color(model, p)
        self._bind_status(model, p)
        self._bind_display(model, p, d)
        self._bind_events(model)
        self._handle = model.get_handle()
        self._sync_panel_label()

        params = lf.optimization_params()
        if params and params.has_params() and params.enable_eval:
            self._sync_eval_steps_with_save_steps(params)

    def _sync_panel_label(self):
        label = tr("window.training")
        if not label or label == self._last_panel_label:
            return
        if lf.ui.set_panel_label(self.id, label):
            self._last_panel_label = label

    def _bind_labels(self, model):
        for label_id, key in LOCALE_KEYS.items():
            model.bind_func(f"label_{label_id}", lambda k=key: tr(k))

        model.bind_func("label_reset", lambda: tr("training_panel.reset"))
        model.bind_func("label_clear", lambda: tr("training_panel.clear"))
        model.bind_func("label_pause", lambda: tr("training_panel.pause"))
        model.bind_func("label_resume", lambda: tr("training_panel.resume"))
        model.bind_func("label_stop", lambda: tr("training_panel.stop"))
        model.bind_func(
            "label_switch_edit", lambda: tr("training_panel.switch_edit_mode")
        )
        model.bind_func("label_status_completed", lambda: tr("status.complete"))
        model.bind_func("label_status_stopped", lambda: tr("status.stopped"))
        model.bind_func("label_status_error", lambda: tr("status.error"))
        model.bind_func("label_status_stopping", lambda: tr("status.stopping"))
        model.bind_func("label_ppisp_sidecar_clear", lambda: tr("training_panel.clear"))

        def _btn_start():
            it = AppState.iteration.value
            return (
                tr("training_panel.resume_training")
                if it > 0
                else tr("training_panel.start_training")
            )

        model.bind_func("btn_start", _btn_start)

    def _bind_visibility(self, model, p, d):
        def _state():
            return AppState.trainer_state.value

        def _iteration():
            return AppState.iteration.value

        model.bind_func("show_no_trainer", lambda: not AppState.has_trainer.value)
        model.bind_func(
            "show_no_params",
            lambda: AppState.has_trainer.value and not (p() and p().has_params()),
        )
        model.bind_func(
            "show_main",
            lambda: AppState.has_trainer.value and p() is not None and p().has_params(),
        )

        for state_name in [
            "ready",
            "running",
            "paused",
            "completed",
            "stopped",
            "error",
            "stopping",
        ]:
            model.bind_func(
                f"show_ctrl_{state_name}", lambda s=state_name: _state() == s
            )

        model.bind_func(
            "show_reset_ready", lambda: _state() == "ready" and _iteration() > 0
        )
        model.bind_func("show_checkpoint", lambda: _state() in ("running", "paused"))
        model.bind_func(
            "show_checkpoint_saved",
            lambda: (
                _state() in ("running", "paused")
                and time.time() - self._checkpoint_saved_time < 2.0
            ),
        )

        model.bind_func(
            "dep_mask_mode",
            lambda: p() is not None and p().has_params() and p().mask_mode.value != 0,
        )
        model.bind_func(
            "dep_mask_segment",
            lambda: p() is not None and p().has_params() and p().mask_mode.value == 1,
        )
        model.bind_func(
            "dep_ppisp", lambda: p() is not None and p().has_params() and p().ppisp
        )
        model.bind_func(
            "dep_ppisp_frozen_sidecar",
            lambda: (
                p() is not None
                and p().has_params()
                and p().ppisp
                and p().ppisp_freeze_from_sidecar
            ),
        )
        model.bind_func(
            "dep_ppisp_controller",
            lambda: p() is not None and p().has_params() and p().ppisp_use_controller,
        )
        model.bind_func(
            "has_ppisp_sidecar_clear",
            lambda: (
                p() is not None and p().has_params() and bool(p().ppisp_sidecar_path)
            ),
        )
        model.bind_func(
            "dep_bg_color",
            lambda: (
                p() is not None and p().has_params() and p().bg_mode.value in (0, 1)
            ),
        )
        model.bind_func(
            "dep_bg_image",
            lambda: p() is not None and p().has_params() and p().bg_mode.value == 2,
        )
        model.bind_func(
            "has_bg_clear",
            lambda: p() is not None and p().has_params() and bool(p().bg_image_path),
        )
        model.bind_func(
            "dep_bilateral",
            lambda: p() is not None and p().has_params() and p().use_bilateral_grid,
        )
        model.bind_func(
            "dep_mrnf",
            lambda: (
                p() is not None and p().has_params() and _is_mrnf_strategy(p().strategy)
            ),
        )
        model.bind_func(
            "dep_igs",
            lambda: p() is not None and p().has_params() and p().strategy == "igs+",
        )
        model.bind_func(
            "dep_sparsity",
            lambda: p() is not None and p().has_params() and p().enable_sparsity,
        )
        model.bind_func(
            "dep_random", lambda: p() is not None and p().has_params() and p().random
        )
        model.bind_func(
            "dep_eval", lambda: p() is not None and p().has_params() and p().enable_eval
        )
        model.bind_func(
            "dep_gut", lambda: p() is not None and p().has_params() and p().gut
        )
        model.bind_func(
            "show_progress",
            lambda: AppState.max_iterations.value > 0 and _iteration() > 0,
        )
        model.bind_func("has_dataset", lambda: d() is not None and d().has_params())
        model.bind_func(
            "show_dataset_no_data", lambda: d() is None or not d().has_params()
        )

        model.bind_func(
            "save_edit_mode", lambda: _state() == "ready" and _iteration() == 0
        )
        model.bind_func(
            "save_readonly_mode", lambda: _state() != "ready" or _iteration() != 0
        )
        model.bind_func(
            "no_save_steps",
            lambda: (
                _state() == "ready"
                and _iteration() == 0
                and p() is not None
                and p().has_params()
                and not list(p().save_steps)
            ),
        )
        model.bind_func(
            "no_save_steps_ro",
            lambda: (
                (_state() != "ready" or _iteration() != 0)
                and p() is not None
                and p().has_params()
                and not list(p().save_steps)
            ),
        )
        model.bind_func(
            "has_save_steps",
            lambda: p() is not None and p().has_params() and bool(list(p().save_steps)),
        )
        model.bind_string_list("save_steps_list")

    def _bind_disabled(self, model, p):
        def _params_edit_locked():
            return not (
                AppState.trainer_state.value == "ready"
                and AppState.iteration.value == 0
            )

        model.bind_func("struct_disabled", _params_edit_locked)
        model.bind_func("live_disabled", _params_edit_locked)
        model.bind_func("adv_disabled", _params_edit_locked)
        model.bind_func(
            "gut_disabled",
            lambda: p() is not None and p().has_params() and p().strategy == "igs+",
        )
        model.bind_func(
            "dataset_disabled",
            lambda: (
                not (
                    lf.dataset_params() is not None
                    and lf.dataset_params().has_params()
                    and lf.dataset_params().can_edit()
                )
            ),
        )

    def _bind_bool_props(self, model, p):
        for prop in PARAM_BOOL_PROPS:
            model.bind(
                prop,
                lambda pr=prop: (
                    getattr(p(), pr, False) if p() and p().has_params() else False
                ),
                lambda v, pr=prop: self._set_bool_prop(pr, v),
            )

    def _bind_dataset_bools(self, model, d):
        def _set_dataset_bool(v, pr):
            dp = d()
            if dp and dp.has_params():
                try:
                    setattr(dp, pr, v)
                except RuntimeError:
                    pass

        for prop in DATASET_BOOL_PROPS:
            model.bind(
                prop,
                lambda pr=prop: (
                    getattr(d(), pr, False) if d() and d().has_params() else False
                ),
                lambda v, pr=prop: _set_dataset_bool(v, pr),
            )

    def _bind_select_props(self, model, p, d):
        model.bind(
            "strategy",
            lambda: p().strategy if p() and p().has_params() else "mcmc",
            lambda v: self._set_strategy(v),
        )
        model.bind(
            "sh_degree_str",
            lambda: str(p().sh_degree) if p() and p().has_params() else "0",
            lambda v: self._set_int_param("sh_degree", v),
        )
        model.bind(
            "tile_mode_str",
            lambda: str(p().tile_mode) if p() and p().has_params() else "1",
            lambda v: self._set_int_param("tile_mode", v),
        )
        model.bind(
            "mask_mode_str",
            lambda: str(p().mask_mode.value) if p() and p().has_params() else "0",
            lambda v: self._set_mask_mode(v),
        )
        model.bind(
            "bg_mode_str",
            lambda: str(p().bg_mode.value) if p() and p().has_params() else "0",
            lambda v: self._set_bg_mode(v),
        )
        model.bind(
            "resize_factor_str",
            lambda: str(d().resize_factor) if d() and d().has_params() else "-1",
            lambda v: self._set_resize_factor(v),
        )

    def _bind_text_props(self, model, p):
        model.bind(
            "ppisp_sidecar_path",
            lambda: p().ppisp_sidecar_path if p() and p().has_params() else "",
            lambda v: self._set_ppisp_sidecar_path(v),
        )

    def _bind_num_props(self, model, p, d):
        for prop, dtype, fmt, min_v, max_v, _step in NUM_PROP_DEFS:
            key = f"{prop}_str"
            self._text_bufs[key] = None

            def getter(k=key, pr=prop, dt=dtype, f=fmt):
                if self._text_bufs[k] is None:
                    self._text_bufs[k] = (
                        _fmt_num(getattr(p(), pr, 0), dt, f)
                        if p() and p().has_params()
                        else ""
                    )
                return self._text_bufs[k]

            def setter(v, k=key):
                self._text_bufs[k] = str(v)

            model.bind(key, getter, setter)

        self._text_bufs["ppisp_activation_step_str"] = None

        def ppisp_activation_step_getter():
            if self._text_bufs["ppisp_activation_step_str"] is None:
                self._text_bufs["ppisp_activation_step_str"] = (
                    f"{_display_ppisp_activation_step(p()):,}"
                    if p() and p().has_params()
                    else ""
                )
            return self._text_bufs["ppisp_activation_step_str"]

        def ppisp_activation_step_setter(v):
            self._text_bufs["ppisp_activation_step_str"] = str(v)

        model.bind(
            "ppisp_activation_step_str",
            ppisp_activation_step_getter,
            ppisp_activation_step_setter,
        )

        self._text_bufs["max_width_str"] = None

        def max_width_getter():
            if self._text_bufs["max_width_str"] is None:
                self._text_bufs["max_width_str"] = (
                    f"{d().max_width:,}" if d() and d().has_params() else ""
                )
            return self._text_bufs["max_width_str"]

        def max_width_setter(v):
            self._text_bufs["max_width_str"] = str(v)

        model.bind("max_width_str", max_width_getter, max_width_setter)

        self._text_bufs["test_every_str"] = None

        def test_every_getter():
            if self._text_bufs["test_every_str"] is None:
                self._text_bufs["test_every_str"] = (
                    f"{d().test_every:,}" if d() and d().has_params() else "8"
                )
            return self._text_bufs["test_every_str"]

        def test_every_setter(v):
            self._text_bufs["test_every_str"] = str(v)

        model.bind("test_every_str", test_every_getter, test_every_setter)

        self._text_bufs["new_step_str"] = None

        def new_step_getter():
            if self._text_bufs["new_step_str"] is None:
                self._text_bufs["new_step_str"] = f"{self._new_save_step:,}"
            return self._text_bufs["new_step_str"]

        def new_step_setter(v):
            self._text_bufs["new_step_str"] = str(v)

        model.bind("new_step_str", new_step_getter, new_step_setter)

    def _mark_text_buf_dirty(self, key):
        if self._handle:
            self._handle.dirty(key)

    def _capture_number_input_snapshot(self, key):
        canonical = self._canonical_text_buf_value(key)
        if canonical is not None:
            return canonical
        return str(self._text_bufs.get(key, "") or "")

    def _restore_number_input_snapshot(self, key, snapshot):
        self._text_bufs[key] = str(snapshot or "")
        self._mark_text_buf_dirty(key)

    def _capture_ppisp_sidecar_path_snapshot(self):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return ""
        return str(params.ppisp_sidecar_path or "")

    def _restore_ppisp_sidecar_path_snapshot(self, snapshot):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        params.ppisp_sidecar_path = str(snapshot or "")
        if self._handle:
            self._handle.dirty_all()

    def _capture_bg_color_snapshot(self):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return (0.0, 0.0, 0.0)
        return tuple(params.bg_color)

    def _restore_bg_color_snapshot(self, snapshot):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        color = tuple(snapshot or (0.0, 0.0, 0.0))
        params.bg_color = color
        rs = lf.get_render_settings()
        if rs:
            rs.set("background_color", color)
        if self._handle:
            self._handle.dirty_all()

    def _canonical_text_buf_value(self, key):
        p = lf.optimization_params()
        d = lf.dataset_params()

        if key.endswith("_str"):
            prop = key[:-4]
            entry = _NUM_PROP_LOOKUP.get(prop)
            if entry:
                dtype, fmt, _min_v, _max_v, _step = entry
                return (
                    _fmt_num(getattr(p, prop, 0), dtype, fmt)
                    if p and p.has_params()
                    else ""
                )

        if key == "ppisp_activation_step_str":
            if p and p.has_params():
                return f"{_display_ppisp_activation_step(p):,}"
            return ""

        if key == "max_width_str":
            return f"{d.max_width:,}" if d and d.has_params() else ""

        if key == "test_every_str":
            return f"{d.test_every:,}" if d and d.has_params() else "8"

        if key == "new_step_str":
            return f"{self._new_save_step:,}"

        return None

    def _commit_number_input_key(self, key):
        buf_val = self._text_bufs.get(key)
        if buf_val is not None and buf_val.strip() and key.endswith("_str"):
            prop = key[:-4]
            entry = _NUM_PROP_LOOKUP.get(prop)
            if entry:
                dtype, _fmt, min_v, max_v, _step = entry
                self._set_num_prop(prop, buf_val, dtype, min_v, max_v)
            elif prop == "ppisp_activation_step":
                self._set_ppisp_activation_step(buf_val)
            elif prop == "max_width":
                self._set_max_width(buf_val)
            elif prop == "test_every":
                self._set_test_every(buf_val)
            elif prop == "new_step":
                self._set_new_step_val(buf_val)

        canonical = self._canonical_text_buf_value(key)
        if canonical is None:
            return
        if self._text_bufs.get(key) != canonical:
            self._text_bufs[key] = canonical
            self._mark_text_buf_dirty(key)

    def _sync_text_bufs(self):
        p = lf.optimization_params()
        d = lf.dataset_params()
        for prop, dtype, fmt, _min_v, _max_v, _step in NUM_PROP_DEFS:
            key = f"{prop}_str"
            self._text_bufs[key] = (
                _fmt_num(getattr(p, prop, 0), dtype, fmt)
                if p and p.has_params()
                else ""
            )
        if p and p.has_params():
            self._text_bufs["ppisp_activation_step_str"] = (
                f"{_display_ppisp_activation_step(p):,}"
            )
        else:
            self._text_bufs["ppisp_activation_step_str"] = ""
        self._text_bufs["max_width_str"] = (
            f"{d.max_width:,}" if d and d.has_params() else ""
        )
        self._text_bufs["test_every_str"] = (
            f"{d.test_every:,}" if d and d.has_params() else "8"
        )
        self._text_bufs["new_step_str"] = f"{self._new_save_step:,}"

    def _bind_slider_props(self, model, p):
        for prop in SLIDER_PROPS:
            model.bind(
                prop,
                lambda pr=prop: (
                    float(getattr(p(), pr, 0.0)) if p() and p().has_params() else 0.0
                ),
                lambda v, pr=prop: self._set_slider_prop(pr, v),
            )

    def _bind_color(self, model, p):
        def _bg():
            return (
                getattr(p(), "bg_color", (0, 0, 0))
                if p() and p().has_params()
                else (0, 0, 0)
            )

        model.bind_func(
            "bg_color_r",
            lambda: f"{tr('training_panel.color_red_prefix')}{int(_bg()[0] * 255):>3d}",
        )
        model.bind_func(
            "bg_color_g",
            lambda: (
                f"{tr('training_panel.color_green_prefix')}{int(_bg()[1] * 255):>3d}"
            ),
        )
        model.bind_func(
            "bg_color_b",
            lambda: (
                f"{tr('training_panel.color_blue_prefix')}{int(_bg()[2] * 255):>3d}"
            ),
        )
        model.bind(
            "bg_color_hex",
            lambda: _color_to_hex(_bg()),
            lambda v: self._set_bg_color_hex(v),
        )

        model.bind_func(
            "picker_r", lambda: float(_bg()[0]) if self._color_edit_prop else 0.0
        )
        model.bind_func(
            "picker_g", lambda: float(_bg()[1]) if self._color_edit_prop else 0.0
        )
        model.bind_func(
            "picker_b", lambda: float(_bg()[2]) if self._color_edit_prop else 0.0
        )

    def _bind_status(self, model, p):
        def _status_mode():
            state = AppState.trainer_state.value
            it = AppState.iteration.value
            labels = {
                "idle": tr("training_panel.idle"),
                "ready": tr("status.ready") if it == 0 else tr("training_panel.resume"),
                "running": tr("training_panel.running"),
                "paused": tr("status.paused"),
                "stopping": tr("status.stopping"),
                "completed": tr("status.complete"),
                "stopped": tr("status.stopped"),
                "error": tr("status.error"),
            }
            return f"{tr('status.mode')}: {labels.get(state, tr('status.unknown'))}"

        def _status_iteration():
            it = AppState.iteration.value
            _rate_tracker.add_sample(it)
            rate = _rate_tracker.get_rate()
            return f"{tr('status.iteration')} {it:,} ({rate:.1f} {tr('training_panel.iters_per_sec')})"

        def _status_gaussians():
            return tr("progress.num_splats") % f"{AppState.num_gaussians.value:,}"

        def _progress_text():
            it = AppState.iteration.value
            mx = AppState.max_iterations.value
            return f"{it:,}/{mx:,}" if mx > 0 else ""

        def _error_message():
            return lf.trainer_error() or ""

        model.bind_func("status_mode", _status_mode)
        model.bind_func("status_iteration", _status_iteration)
        model.bind_func("status_gaussians", _status_gaussians)
        model.bind_func("progress_text", _progress_text)
        model.bind_func("progress_value", lambda: self._progress_value)
        model.bind_func("loss_label", lambda: self._loss_label)
        model.bind_func("loss_tick_max", lambda: self._loss_tick_max)
        model.bind_func("loss_tick_mid", lambda: self._loss_tick_mid)
        model.bind_func("loss_tick_min", lambda: self._loss_tick_min)
        model.bind_func("psnr_label", lambda: self._psnr_label)
        model.bind_func("psnr_tick_max", lambda: self._psnr_tick_max)
        model.bind_func("psnr_tick_mid", lambda: self._psnr_tick_mid)
        model.bind_func("psnr_tick_min", lambda: self._psnr_tick_min)
        model.bind_func("error_message", _error_message)

        model.bind_func(
            "save_steps_display",
            lambda: (
                ", ".join(f"{s:,}" for s in p().save_steps)
                if p() and p().has_params()
                else ""
            ),
        )

    def _bind_display(self, model, p, d):
        model.bind_func(
            "opt_strategy_display",
            lambda: (
                tr(STRATEGY_LABEL_KEYS.get(p().strategy, ""))
                if p() and p().has_params() and p().strategy in STRATEGY_LABEL_KEYS
                else (p().strategy if p() and p().has_params() else "")
            ),
        )

        model.bind_func(
            "dataset_path_display",
            lambda: (
                os.path.basename(d().data_path)
                if d() and d().has_params() and d().data_path
                else tr("training.value.none")
            ),
        )
        model.bind_func(
            "dataset_images_display",
            lambda: (
                d().images
                if d() and d().has_params() and d().images
                else tr("training.value.default")
            ),
        )
        model.bind_func(
            "dataset_output_display",
            lambda: (
                os.path.basename(d().output_path)
                if d() and d().has_params() and d().output_path
                else tr("training.value.not_set")
            ),
        )
        model.bind_func(
            "bg_image_path_display",
            lambda: (
                os.path.basename(p().bg_image_path)
                if p() and p().has_params() and p().bg_image_path
                else tr("training.value.none")
            ),
        )

    def _bind_events(self, model):
        model.bind_event("toggle_section", self._on_toggle_section)
        model.bind_event("color_click", self._on_color_click)
        model.bind_event("picker_change", self._on_picker_change)
        model.bind_event("action", self._on_action)
        model.bind_event("remove_step", self._on_remove_step_event)
        model.bind_event("num_step", self._on_num_step)

    def on_mount(self, doc):
        self._doc = doc
        self._sync_panel_label()
        self._popup_el = doc.get_element_by_id("color-picker-popup")
        if self._popup_el:
            self._popup_el.add_event_listener("click", self._on_popup_click)
        body = doc.get_element_by_id("body")
        if body:
            body.add_event_listener("click", self._on_body_click)
            body.add_event_listener("mouseup", self._on_step_mouseup)
        for el in doc.query_selector_all("input.number-input"):
            w.bind_select_all_on_focus(el)
            key = el.get_attribute("data-value", "")
            if key:
                self._escape_revert.bind(
                    el,
                    key,
                    lambda k=key: self._capture_number_input_snapshot(k),
                    lambda snapshot, k=key: self._restore_number_input_snapshot(
                        k, snapshot
                    ),
                )
            el.add_event_listener("change", self._on_number_input_change)
            el.add_event_listener("blur", self._on_number_input_blur)
        for el in doc.query_selector_all("input.color-hex"):
            w.bind_select_all_on_focus(el)
            if el.get_attribute("data-value", "") == "bg_color_hex":
                self._escape_revert.bind(
                    el,
                    "bg_color_hex",
                    self._capture_bg_color_snapshot,
                    self._restore_bg_color_snapshot,
                )
        sidecar_input = doc.query_selector('input[data-value="ppisp_sidecar_path"]')
        if sidecar_input:
            w.bind_select_all_on_focus(sidecar_input)
            self._escape_revert.bind(
                sidecar_input,
                "ppisp_sidecar_path",
                self._capture_ppisp_sidecar_path_snapshot,
                self._restore_ppisp_sidecar_path_snapshot,
            )
        self._loss_graph_el = doc.get_element_by_id("loss-graph-el")
        self._psnr_graph_el = doc.get_element_by_id("psnr-graph-el")
        self._scrub_fields.mount(doc)
        self._sync_section_states()

    def on_update(self, doc):
        if not self._handle:
            return False
        self._sync_panel_label()

        dirty = False
        state = AppState.trainer_state.value
        if state != self._last_state:
            self._last_state = state
            if state == "ready":
                _rate_tracker.clear()
            self._sync_text_bufs()
            self._handle.dirty_all()
            dirty = True
        else:
            it = AppState.iteration.value
            if it != self._last_iteration:
                self._last_iteration = it
                self._handle.dirty("status_iteration")
                self._handle.dirty("progress_text")
                self._handle.dirty("show_progress")
                dirty = True

            ng = AppState.num_gaussians.value
            if ng != self._last_num_gaussians:
                self._last_num_gaussians = ng
                self._handle.dirty("status_gaussians")
                dirty = True

            checkpoint_visible = (
                self._checkpoint_saved_time > 0.0
                and time.time() - self._checkpoint_saved_time < 2.0
            )
            if checkpoint_visible != self._last_checkpoint_saved_visible:
                self._last_checkpoint_saved_visible = checkpoint_visible
                self._handle.dirty("show_checkpoint_saved")
                dirty = True

        if state == "ready" and AppState.iteration.value == 0:
            params = lf.optimization_params()
            if params and params.has_params():
                if self._try_auto_scale_steps(params):
                    self._sync_text_bufs()
                    self._handle.dirty_all()
                    dirty = True

        self._update_step_repeat()
        dirty |= self._update_progress()
        dirty |= self._update_save_steps(doc)
        dirty |= self._update_color_swatch(doc)
        dirty |= self._update_loss_graph()
        dirty |= self._update_psnr_graph()
        dirty |= self._scrub_fields.sync_all()
        return dirty

    def _update_progress(self):
        it = AppState.iteration.value
        mx = AppState.max_iterations.value
        frac = it / mx if mx > 0 and it > 0 else 0.0
        if frac != self._last_progress_frac:
            self._last_progress_frac = frac
            self._progress_value = str(frac)
            if self._handle:
                self._handle.dirty("progress_value")
            return True
        return False

    def _update_save_steps(self, doc):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return False

        state = AppState.trainer_state.value
        can_edit = state == "ready" and AppState.iteration.value == 0
        if not can_edit:
            return False

        steps = list(params.save_steps)
        if steps != self._last_save_steps:
            self._last_save_steps = steps[:]
            self._handle.update_string_list(
                "save_steps_list", [f"{s:,}" for s in steps]
            )
            self._handle.dirty("no_save_steps")
            self._handle.dirty("has_save_steps")
            self._handle.dirty("save_steps_display")
            return True
        return False

    def _update_color_swatch(self, doc):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return False
        c = tuple(params.bg_color)
        if c == self._last_bg_color:
            return False
        self._last_bg_color = c
        swatch = doc.get_element_by_id("swatch-bg_color")
        if swatch:
            r, g, b = int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)
            swatch.set_property("background-color", f"rgb({r},{g},{b})")
        return True

    def on_scene_changed(self, doc):
        if self._handle:
            self._sync_text_bufs()
            self._handle.dirty_all()

    def on_unmount(self, doc):
        doc.remove_data_model("training")
        self._handle = None
        self._doc = None
        self._escape_revert.clear()
        self._scrub_fields.unmount()

    def _update_loss_graph(self):
        if not self._loss_graph_el:
            return False
        loss_data = lf.loss_buffer()
        if not loss_data:
            if self._last_loss_signature is None:
                return False
            self._last_loss_signature = None
            lf.push_loss_to_element(self._loss_graph_el, [])
            self._loss_label = ""
            self._loss_tick_max = ""
            self._loss_tick_mid = ""
            self._loss_tick_min = ""
            if self._handle:
                self._handle.dirty("loss_label")
                self._handle.dirty("loss_tick_max")
                self._handle.dirty("loss_tick_mid")
                self._handle.dirty("loss_tick_min")
            return True
        signature = (len(loss_data), float(loss_data[-1]))
        if signature == self._last_loss_signature:
            return False
        self._last_loss_signature = signature
        data_min, data_max = lf.push_loss_to_element(self._loss_graph_el, loss_data)
        self._loss_label = f"{tr('status.loss')}: {loss_data[-1]:.4f}"
        mid = data_min + (data_max - data_min) * 0.5
        tick_values = [data_max, mid, data_min]
        max_abs = max(abs(data_min), abs(data_max))
        fmt = "%.4f" if max_abs < 0.1 else ("%.3f" if max_abs < 1.0 else "%.2f")
        self._loss_tick_max, self._loss_tick_mid, self._loss_tick_min = [
            fmt % val for val in tick_values
        ]
        if self._handle:
            self._handle.dirty("loss_label")
            self._handle.dirty("loss_tick_max")
            self._handle.dirty("loss_tick_mid")
            self._handle.dirty("loss_tick_min")
        return True

    def _update_psnr_graph(self):
        if not self._psnr_graph_el:
            return False
        psnr_data = lf.psnr_buffer()
        if not psnr_data:
            if self._last_psnr_signature is None:
                return False
            self._last_psnr_signature = None
            lf.push_psnr_to_element(self._psnr_graph_el, [])
            self._psnr_label = ""
            self._psnr_tick_max = ""
            self._psnr_tick_mid = ""
            self._psnr_tick_min = ""
            if self._handle:
                self._handle.dirty("psnr_label")
                self._handle.dirty("psnr_tick_max")
                self._handle.dirty("psnr_tick_mid")
                self._handle.dirty("psnr_tick_min")
            return True
        signature = (len(psnr_data), float(psnr_data[-1]))
        if signature == self._last_psnr_signature:
            return False
        self._last_psnr_signature = signature
        data_min, data_max = lf.push_psnr_to_element(self._psnr_graph_el, psnr_data)
        self._psnr_label = f"{tr('status.psnr')}: {psnr_data[-1]:.2f}"
        mid = data_min + (data_max - data_min) * 0.5
        tick_values = [data_max, mid, data_min]
        max_abs = max(abs(data_min), abs(data_max))
        fmt = "%.4f" if max_abs < 0.1 else ("%.3f" if max_abs < 1.0 else "%.2f")
        self._psnr_tick_max, self._psnr_tick_mid, self._psnr_tick_min = [
            fmt % val for val in tick_values
        ]
        if self._handle:
            self._handle.dirty("psnr_label")
            self._handle.dirty("psnr_tick_max")
            self._handle.dirty("psnr_tick_mid")
            self._handle.dirty("psnr_tick_min")
        return True

    def _on_picker_change(self, handle, event, args):
        params = lf.optimization_params()
        if (
            not params
            or not params.has_params()
            or not event
            or not self._color_edit_prop
        ):
            return
        r = float(event.get_parameter("red", "0"))
        g = float(event.get_parameter("green", "0"))
        b = float(event.get_parameter("blue", "0"))
        setattr(params, self._color_edit_prop, (r, g, b))
        rs = lf.get_render_settings()
        if rs and self._color_edit_prop == "bg_color":
            rs.set("background_color", (r, g, b))
        if self._handle:
            self._sync_text_bufs()
            self._handle.dirty_all()

    def _on_popup_click(self, event):
        self._picker_click_handled = True

    def _on_body_click(self, event):
        if hasattr(self, "_picker_click_handled") and self._picker_click_handled:
            self._picker_click_handled = False
            return
        if hasattr(self, "_popup_el") and self._popup_el:
            self._popup_el.set_class("visible", False)
            self._color_edit_prop = None

    # ── Setters ────────────────────────────────────────────

    def _set_bool_prop(self, prop, val):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        if not hasattr(params, prop):
            return
        if prop == "ppisp_freeze_from_sidecar" and val:
            params.ppisp = True
        elif prop == "ppisp" and not val:
            params.ppisp_freeze_from_sidecar = False
        if prop == "enable_eval" and val and not self._clamp_current_test_every_for_eval():
            return
        setattr(params, prop, val)
        if prop == "enable_eval" and val:
            self._sync_eval_steps_with_save_steps(params)
        rs = lf.get_render_settings()
        if rs and prop in RENDER_SYNC:
            rs.set(RENDER_SYNC[prop], val)
        if self._handle:
            self._sync_text_bufs()
            self._handle.dirty_all()

    def _set_ppisp_sidecar_path(self, val):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        params.ppisp_sidecar_path = str(val)
        if self._handle:
            self._handle.dirty_all()

    def _set_strategy(self, val):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        if val == "igs+" and params.gut:
            btn_gut = tr("training.conflict.btn_disable_gut")
            btn_cancel = tr("training.conflict.btn_cancel")

            def _on_conflict(button, _gut=btn_gut, _val=val):
                p = lf.optimization_params()
                if button == _gut:
                    p.gut = False
                    p.set_strategy(_val)
                    if self._handle:
                        self._sync_text_bufs()
                        self._handle.dirty_all()

            lf.ui.confirm_dialog(
                tr("training.error.strategy_gut_title"),
                tr("training.conflict.strategy_gut_strategy_message"),
                [btn_gut, btn_cancel],
                _on_conflict,
            )
        else:
            params.set_strategy(val)
            if self._handle:
                self._sync_text_bufs()
                self._handle.dirty_all()

    def _set_int_param(self, prop, val_str):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        try:
            setattr(params, prop, int(val_str))
        except (ValueError, TypeError):
            pass

    def _set_mask_mode(self, val_str):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        try:
            params.mask_mode = lf.MaskMode(int(val_str))
        except (ValueError, TypeError):
            pass
        if self._handle:
            self._sync_text_bufs()
            self._handle.dirty_all()

    def _set_bg_mode(self, val_str):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        try:
            params.bg_mode = lf.BackgroundMode(int(val_str))
        except (ValueError, TypeError):
            pass
        if self._handle:
            self._sync_text_bufs()
            self._handle.dirty_all()

    def _set_resize_factor(self, val_str):
        d = lf.dataset_params()
        if not d or not d.has_params():
            return
        try:
            d.resize_factor = int(val_str)
        except (ValueError, TypeError, RuntimeError):
            pass

    def _set_num_prop(self, prop, val_str, dtype, min_v, max_v):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return False
        try:
            val = dtype(_parse_num(str(val_str), dtype))
        except (ValueError, TypeError):
            return False
        if min_v is not None:
            val = max(val, dtype(min_v))
        if max_v is not None:
            val = min(val, dtype(max_v))

        try:
            if prop == "steps_scaler":
                params.apply_step_scaling(val)
                if self._handle:
                    self._sync_text_bufs()
                    self._handle.dirty_all()
            else:
                params.set(prop, val)
        except (ValueError, TypeError, RuntimeError):
            return False
        return True

    def _set_ppisp_activation_step(self, val_str):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return False
        try:
            params.ppisp_controller_activation_step = max(
                1, int(_parse_num(str(val_str), int))
            )
        except (ValueError, TypeError):
            return False
        return True

    def _set_max_width(self, val_str):
        d = lf.dataset_params()
        if not d or not d.has_params():
            return False
        try:
            val = int(_parse_num(str(val_str), int))
            if val >= 0:
                d.max_width = val
                return True
        except (ValueError, TypeError, RuntimeError):
            return False
        return False

    def _active_camera_count(self):
        get_scene = getattr(lf, "get_scene", None)
        if not callable(get_scene):
            return None
        try:
            scene = get_scene()
        except (AttributeError, RuntimeError, TypeError):
            return None
        if scene is None:
            return None
        try:
            return max(0, int(getattr(scene, "active_camera_count")))
        except (AttributeError, TypeError, ValueError):
            return None

    def _test_every_max(self):
        camera_count = self._active_camera_count()
        return max(1, camera_count if camera_count is not None else 100)

    def _eval_requires_training_split(self):
        params = lf.optimization_params()
        return bool(
            params and params.has_params() and getattr(params, "enable_eval", False)
        )

    def _coerce_test_every_for_current_eval_split(self, val):
        camera_count = self._active_camera_count()
        if camera_count is not None and camera_count < 2:
            return None
        return max(2, val)

    def _clamp_current_test_every_for_eval(self):
        d = lf.dataset_params()
        if not d or not d.has_params():
            return True
        val = max(1, min(self._test_every_max(), int(getattr(d, "test_every", 8))))
        val = self._coerce_test_every_for_current_eval_split(val)
        if val is None:
            return False
        if getattr(d, "test_every", None) != val:
            try:
                d.test_every = val
            except RuntimeError:
                return False
            self._text_bufs["test_every_str"] = f"{val:,}"
        return True

    def _set_test_every(self, val_str):
        d = lf.dataset_params()
        if not d or not d.has_params():
            return False
        try:
            val = int(_parse_num(str(val_str), int))
            max_val = self._test_every_max()
            if not (1 <= val <= max_val):
                return False
            if self._eval_requires_training_split():
                val = self._coerce_test_every_for_current_eval_split(val)
                if val is None:
                    return False
            d.test_every = val
            return True
        except (ValueError, TypeError, RuntimeError):
            return False
        return False

    def _set_new_step_val(self, val_str):
        try:
            self._new_save_step = max(1, int(_parse_num(str(val_str), int)))
        except (ValueError, TypeError):
            return False
        return True

    def _set_slider_prop(self, prop, val):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        try:
            params.set(prop, float(val))
            if self._handle:
                self._handle.dirty(prop)
        except (ValueError, TypeError):
            pass

    def _get_scrub_value(self, prop):
        params = lf.optimization_params()
        if not params or not params.has_params():
            spec = SCRUB_FIELD_DEFS[prop]
            return spec.min_value
        return float(getattr(params, prop, 0.0))

    def _set_scrub_value(self, prop, value):
        self._set_slider_prop(prop, value)

    def _set_bg_color_hex(self, hex_val):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        color = _hex_to_color(hex_val)
        if color:
            params.bg_color = color
            rs = lf.get_render_settings()
            if rs:
                rs.set("background_color", color)
            if self._handle:
                self._sync_text_bufs()
                self._handle.dirty_all()

    # ── Event handlers ─────────────────────────────────────

    def _on_num_step(self, handle, event, args):
        if len(args) < 2:
            return
        prop = str(args[0])
        direction = int(args[1])
        self._apply_num_step(prop, direction)
        now = time.monotonic()
        self._step_repeat_prop = prop
        self._step_repeat_dir = direction
        self._step_repeat_start = now
        self._step_repeat_last = now

    def _on_number_input_change(self, event):
        if not event.get_bool_parameter("linebreak", False):
            return
        target = event.current_target()
        if target is None:
            return
        self._commit_number_input_key(target.get_attribute("data-value", ""))

    def _on_number_input_blur(self, event):
        target = event.current_target()
        if target is None:
            return
        self._commit_number_input_key(target.get_attribute("data-value", ""))

    def _apply_num_step(self, prop, direction):
        entry = _NUM_PROP_LOOKUP.get(prop)
        if entry:
            params = lf.optimization_params()
            if not params or not params.has_params():
                return
            dtype, fmt, min_v, max_v, step = entry
            current = getattr(params, prop, 0)
            new_val = dtype(current + step * direction)
            if min_v is not None:
                new_val = max(new_val, dtype(min_v))
            if max_v is not None:
                new_val = min(new_val, dtype(max_v))
            self._set_num_prop(prop, str(new_val), dtype, min_v, max_v)
            self._text_bufs[f"{prop}_str"] = _fmt_num(new_val, dtype, fmt)
            if self._handle:
                self._handle.dirty(f"{prop}_str")
            return

        if prop == "ppisp_activation_step":
            params = lf.optimization_params()
            if not params or not params.has_params():
                return
            current = _display_ppisp_activation_step(params)
            new_val = max(1, current + 100 * direction)
            params.ppisp_controller_activation_step = new_val
            self._text_bufs["ppisp_activation_step_str"] = f"{new_val:,}"
            if self._handle:
                self._handle.dirty("ppisp_activation_step_str")
        elif prop == "max_width":
            d = lf.dataset_params()
            if not d or not d.has_params():
                return
            new_val = max(0, d.max_width + 16 * direction)
            d.max_width = new_val
            self._text_bufs["max_width_str"] = f"{new_val:,}"
            if self._handle:
                self._handle.dirty("max_width_str")
        elif prop == "test_every":
            d = lf.dataset_params()
            if not d or not d.has_params():
                return
            max_test_every = self._test_every_max()
            new_val = max(1, min(max_test_every, d.test_every + direction))
            if self._eval_requires_training_split():
                new_val = self._coerce_test_every_for_current_eval_split(new_val)
                if new_val is None:
                    return
            d.test_every = new_val
            self._text_bufs["test_every_str"] = f"{new_val:,}"
            if self._handle:
                self._handle.dirty("test_every_str")
        elif prop == "new_step":
            self._new_save_step = max(1, self._new_save_step + 100 * direction)
            self._text_bufs["new_step_str"] = f"{self._new_save_step:,}"
            if self._handle:
                self._handle.dirty("new_step_str")

    def _on_step_mouseup(self, event):
        self._step_repeat_prop = None

    def _update_step_repeat(self):
        if not self._step_repeat_prop:
            return
        now = time.monotonic()
        if now - self._step_repeat_start < 0.15:
            return
        if now - self._step_repeat_last < 0.01:
            return
        self._step_repeat_last = now
        self._apply_num_step(self._step_repeat_prop, self._step_repeat_dir)

    def _get_section_elements(self, name):
        if not self._doc:
            return None, None, None
        dom_name = name.replace("_", "-")
        header = self._doc.get_element_by_id(f"hdr-{dom_name}")
        arrow = self._doc.get_element_by_id(f"arrow-{dom_name}")
        content = self._doc.get_element_by_id(f"sec-{dom_name}")
        return header, arrow, content

    def _sync_section_states(self):
        for name in SECTIONS:
            header, arrow, content = self._get_section_elements(name)
            if content:
                w.sync_section_state(
                    content, name not in self._collapsed, header, arrow
                )

    def _on_toggle_section(self, handle, event, args):
        del handle, event
        if not args:
            return
        name = str(args[0])
        expanding = name in self._collapsed
        if expanding:
            self._collapsed.discard(name)
        else:
            self._collapsed.add(name)

        header, arrow, content = self._get_section_elements(name)
        if content:
            w.animate_section_toggle(content, expanding, arrow, header_element=header)

    def _on_color_click(self, handle, event, args):
        if not args:
            return
        prop_id = str(args[0])
        if self._color_edit_prop == prop_id:
            self._color_edit_prop = None
            if hasattr(self, "_popup_el") and self._popup_el:
                self._popup_el.set_class("visible", False)
        else:
            self._color_edit_prop = prop_id
            if hasattr(self, "_popup_el") and self._popup_el and event:
                mx = int(float(event.get_parameter("mouse_x", "0")))
                my = int(float(event.get_parameter("mouse_y", "0")))
                left = max(0, mx - 210)
                self._popup_el.set_property("left", f"{left}px")
                self._popup_el.set_property("top", f"{my + 2}px")
                self._popup_el.set_class("visible", True)
                handle.dirty("picker_r")
                handle.dirty("picker_g")
                handle.dirty("picker_b")
            self._picker_click_handled = True

    def _on_action(self, handle, event, args):
        if not args:
            return
        action = str(args[0])

        if action == "start":
            self._action_start()
        elif action == "pause":
            lf.pause_training()
        elif action == "resume":
            lf.resume_training()
        elif action == "stop":
            lf.stop_training()
        elif action == "reset":
            lf.reset_training()
        elif action == "clear":
            lf.new_project()
        elif action == "switch_edit":
            lf.switch_to_edit_mode()
        elif action == "save_checkpoint":
            lf.save_checkpoint()
            self._checkpoint_saved_time = time.time()
        elif action == "browse_bg":
            selected = lf.ui.open_image_file_dialog("")
            if selected:
                params = lf.optimization_params()
                if params and params.has_params():
                    params.bg_image_path = selected
                    if self._handle:
                        self._sync_text_bufs()
                        self._handle.dirty_all()
        elif action == "clear_bg":
            params = lf.optimization_params()
            if params and params.has_params():
                params.bg_image_path = ""
                if self._handle:
                    self._sync_text_bufs()
                    self._handle.dirty_all()
        elif action == "clear_ppisp_sidecar":
            params = lf.optimization_params()
            if params and params.has_params():
                params.ppisp_sidecar_path = ""
                if self._handle:
                    self._handle.dirty_all()
        elif action == "browse_ppisp_sidecar":
            params = lf.optimization_params()
            start_dir = ""
            if params and params.has_params() and params.ppisp_sidecar_path:
                start_dir = params.ppisp_sidecar_path
            selected = lf.ui.open_ppisp_file_dialog(start_dir)
            if selected and params and params.has_params():
                params.ppisp = True
                params.ppisp_freeze_from_sidecar = True
                params.ppisp_sidecar_path = selected
                if self._handle:
                    self._handle.dirty_all()
        elif action == "add_step":
            params = lf.optimization_params()
            if params and params.has_params() and self._new_save_step > 0:
                params.add_save_step(self._new_save_step)
                if params.enable_eval:
                    self._sync_eval_steps_with_save_steps(params)
                self._last_save_steps = []

    def _action_start(self):
        params = lf.optimization_params()

        if params and params.has_params() and params.enable_eval:
            self._sync_eval_steps_with_save_steps(params)

        error = params.validate() if params and params.has_params() else ""
        if error:
            btn_mcmc = tr("training.conflict.btn_use_mcmc")
            btn_gut = tr("training.conflict.btn_disable_gut")
            btn_cancel = tr("training.conflict.btn_cancel")

            def _on_conflict(button, _mcmc=btn_mcmc, _gut=btn_gut):
                p = lf.optimization_params()
                if button == _mcmc:
                    p.set_strategy("mcmc")
                    lf.start_training()
                elif button == _gut:
                    p.gut = False
                    lf.start_training()

            lf.ui.confirm_dialog(
                tr("training.error.strategy_gut_title"),
                tr("training.conflict.strategy_gut_start_message"),
                [btn_mcmc, btn_gut, btn_cancel],
                _on_conflict,
            )
        elif self._should_offer_pc_save():
            self._show_save_pc_dialog()
        else:
            lf.start_training()

    def _should_offer_pc_save(self):
        scene = lf.get_scene()
        if scene is None or not scene.is_valid():
            return False
        return scene.is_point_cloud_modified

    def _show_save_pc_dialog(self):
        btn_save = tr("training.save_pc.btn_save_start")
        btn_skip = tr("training.save_pc.btn_start_without")
        btn_cancel = tr("training.conflict.btn_cancel")

        def _on_result(button, _s=btn_save, _k=btn_skip):
            if button == _s:
                try:
                    self._save_modified_pc()
                except Exception as e:
                    lf.log.error(f"Failed to save point cloud: {e}")
                lf.start_training()
            elif button == _k:
                lf.start_training()

        lf.ui.confirm_dialog(
            tr("training.save_pc.title"),
            tr("training.save_pc.message"),
            [btn_save, btn_skip, btn_cancel],
            _on_result,
        )

    def _save_modified_pc(self):
        d = lf.dataset_params()
        if not d or not d.has_params() or not d.data_path:
            return
        info = lf.detect_dataset_info(d.data_path)
        if not info or not info.sparse_path:
            return
        save_path = os.path.join(str(info.sparse_path), "points3D.ply")
        scene = lf.get_scene()
        if not scene:
            return
        for node in scene.get_nodes():
            if node.type == lf.scene.NodeType.POINTCLOUD:
                pc = node.point_cloud()
                if pc:
                    lf.io.save_point_cloud_ply(pc, save_path)
                    lf.log.info(f"Saved point cloud ({pc.size} points) to {save_path}")
                    scene.is_point_cloud_modified = False
                    return

    def _on_remove_step_event(self, handle, event, args):
        if not args:
            return
        try:
            idx = int(args[0])
        except (ValueError, TypeError):
            return
        self._on_step_remove(idx)

    def _on_step_remove(self, idx):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        steps = list(params.save_steps)
        if 0 <= idx < len(steps):
            step_to_remove = steps[idx]
            params.remove_save_step(step_to_remove)
            if params.enable_eval:
                self._remove_from_eval_steps(params, step_to_remove)
            self._last_save_steps = []

    def _sync_eval_steps_with_save_steps(self, params):
        if not params or not params.has_params():
            return
        save_steps_list = list(params.save_steps)
        params.clear_eval_steps()
        for step in save_steps_list:
            params.add_eval_step(step)

    def _remove_from_eval_steps(self, params, step):
        if not params or not params.has_params():
            return
        params.remove_eval_step(step)

    def _try_auto_scale_steps(self, params):
        scene = lf.get_scene()
        if scene is None:
            return False
        camera_count = scene.active_camera_count
        if camera_count == 0 or camera_count == self._auto_scaled_for_cameras:
            return False
        self._auto_scaled_for_cameras = camera_count
        params.auto_scale_steps(camera_count)
        return True

    def _draw_controls(self, layout, state, iteration):
        if state == "ready":
            label = (
                tr("training_panel.resume_training")
                if iteration > 0
                else tr("training_panel.start_training")
            )
            if layout.button_styled(label, "success", FULL_WIDTH):
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
                        tr("training.error.strategy_gut_title"),
                        tr("training.conflict.strategy_gut_start_message"),
                        [btn_mcmc, btn_gut, btn_cancel],
                        _on_start_conflict,
                    )
                else:
                    lf.start_training()
            if iteration > 0:
                if layout.button_styled(
                    tr("training_panel.reset"), "secondary", FULL_WIDTH
                ):
                    lf.reset_training()
            if layout.button_styled(tr("training_panel.clear"), "error", FULL_WIDTH):
                lf.new_project()

        elif state == "running":
            if layout.button_styled(tr("training_panel.pause"), "warning", FULL_WIDTH):
                lf.pause_training()

        elif state == "paused":
            if layout.button_styled(tr("training_panel.resume"), "success", FULL_WIDTH):
                lf.resume_training()
            if layout.button_styled(
                tr("training_panel.reset"), "secondary", FULL_WIDTH
            ):
                lf.reset_training()
            if layout.button_styled(tr("training_panel.stop"), "error", FULL_WIDTH):
                lf.stop_training()

        elif state in ("completed", "stopped"):
            if state == "completed":
                layout.text_colored(tr("status.complete"), COLOR_SUCCESS)
            else:
                layout.text_colored(tr("status.stopped"), COLOR_MUTED)
            if layout.button_styled(
                tr("training_panel.switch_edit_mode"), "success", FULL_WIDTH
            ):
                lf.switch_to_edit_mode()
            if layout.button_styled(
                tr("training_panel.reset"), "secondary", FULL_WIDTH
            ):
                lf.reset_training()
            if layout.button_styled(tr("training_panel.clear"), "error", FULL_WIDTH):
                lf.new_project()

        elif state == "error":
            layout.text_colored(tr("status.error"), COLOR_ERROR)
            if error_msg := lf.trainer_error():
                layout.text_wrapped(error_msg)
            if layout.button_styled(
                tr("training_panel.reset"), "secondary", FULL_WIDTH
            ):
                lf.reset_training()
            if layout.button_styled(tr("training_panel.clear"), "error", FULL_WIDTH):
                lf.new_project()

        elif state == "stopping":
            layout.text_colored(tr("status.stopping"), COLOR_MUTED)

        if state in ("running", "paused"):
            if layout.button_styled(
                tr("training_panel.save_checkpoint"), "primary", FULL_WIDTH
            ):
                lf.save_checkpoint()
                self._checkpoint_saved_time = time.time()

            if time.time() - self._checkpoint_saved_time < 2.0:
                theme = lf.ui.theme()
                layout.text_colored(
                    tr("training_panel.checkpoint_saved"), theme.palette.success
                )

    def _draw_basic_params(self, layout, state, iteration, params):
        can_edit = (state == "ready") and (iteration == 0)
        can_edit_live = state in ("ready", "running", "paused")

        if layout.begin_table("PyBasicParamsTable", 2):
            layout.table_setup_column(tr("common.column_label"), 120.0)
            layout.table_setup_column(tr("common.column_control"), 0.0)

            # -- Structural params (only before training starts) --
            layout.begin_disabled(not can_edit)

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.strategy"))
            layout.table_next_column()
            layout.push_item_width(-1)
            strategy_items = [
                tr("training.options.strategy.mrnf"),
                tr("training.options.strategy.igs_plus"),
                tr("training.options.strategy.mcmc"),
            ]
            strategy_map = {0: "mrnf", 1: "igs+", 2: "mcmc"}
            strategy_idx = {"mrnf": 0, "mnrf": 0, "lfs": 0, "igs+": 1, "mcmc": 2}.get(
                params.strategy, 0
            )
            changed, new_idx = layout.combo(
                "##py_strategy", strategy_idx, strategy_items
            )
            if changed:
                new_strategy = strategy_map[new_idx]
                if new_strategy == "igs+" and params.gut:
                    btn_gut = tr("training.conflict.btn_disable_gut")
                    btn_cancel = tr("training.conflict.btn_cancel")

                    def _on_strategy_conflict(
                        button, _gut=btn_gut, _strategy=new_strategy
                    ):
                        p = lf.optimization_params()
                        if button == _gut:
                            p.gut = False
                            p.set_strategy(_strategy)

                    lf.ui.confirm_dialog(
                        tr("training.error.strategy_gut_title"),
                        tr("training.conflict.strategy_gut_strategy_message"),
                        [btn_gut, btn_cancel],
                        _on_strategy_conflict,
                    )
                else:
                    params.set_strategy(new_strategy)
            layout.pop_item_width()
            if layout.is_item_hovered():
                tooltip = (
                    tr("training.tooltip.strategy_gut_conflict")
                    if params.gut
                    else tr("training.tooltip.strategy")
                )
                layout.set_tooltip(tooltip)

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.iterations"))
            layout.table_next_column()
            layout.push_item_width(-1)
            changed, new_val = layout.input_int_formatted(
                "##py_iterations", int(params.iterations), 1000, 5000
            )
            if changed and new_val > 0:
                params.iterations = new_val
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.iterations"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.max_gaussians"))
            layout.table_next_column()
            layout.push_item_width(-1)
            changed, new_val = layout.input_int_formatted(
                "##py_max_cap", params.max_cap, 10000, 100000
            )
            if changed and new_val > 0:
                params.max_cap = new_val
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.max_gaussians"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.sh_degree"))
            layout.table_next_column()
            layout.push_item_width(-1)
            changed, new_idx = layout.combo(
                "##py_sh_degree", params.sh_degree, SH_DEGREE_ITEMS
            )
            if changed:
                params.sh_degree = new_idx
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.sh_degree"))

            if params.gut:
                layout.table_next_row()
                layout.table_next_column()
                layout.label(tr("training_params.tile_mode"))
                layout.table_next_column()
                layout.push_item_width(-1)
                tile_idx = {1: 0, 2: 1, 4: 2}.get(params.tile_mode, 0)
                tile_mode_items = [
                    tr("training.options.tile.full"),
                    tr("training.options.tile.half"),
                    tr("training.options.tile.quarter"),
                ]
                changed, new_idx = layout.combo("##py_tile_mode", tile_idx, tile_mode_items)
                if changed:
                    params.tile_mode = [1, 2, 4][new_idx]
                layout.pop_item_width()
                if layout.is_item_hovered():
                    layout.set_tooltip(tr("training.tooltip.tile_mode"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.steps_scaler"))
            layout.table_next_column()
            layout.push_item_width(-1)
            changed, new_val = layout.input_float(
                "##py_steps_scaler", params.steps_scaler, 0.1, 0.5, "%.2f"
            )
            if changed:
                params.apply_step_scaling(new_val)
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.steps_scaler"))

            layout.end_disabled()

            # -- Live-editable params (available during training) --
            layout.begin_disabled(not can_edit_live)

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.bilateral_grid"))
            layout.table_next_column()
            changed, new_val = layout.checkbox(
                "##py_bilateral_grid", params.use_bilateral_grid
            )
            if changed:
                params.use_bilateral_grid = new_val
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.bilateral_grid"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.mask_mode"))
            layout.table_next_column()
            layout.push_item_width(-1)
            mask_idx = params.mask_mode.value
            mask_mode_items = [
                tr("training.options.mask.none"),
                tr("training.options.mask.segment"),
                tr("training.options.mask.ignore"),
                tr("training.options.mask.alpha_consistent"),
            ]
            changed, new_idx = layout.combo("##py_mask_mode", mask_idx, mask_mode_items)
            if changed:
                params.mask_mode = lf.MaskMode(new_idx)
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.mask_mode"))

            if params.mask_mode.value != 0:
                layout.table_next_row()
                layout.table_next_column()
                layout.label(tr("training_params.invert_masks"))
                layout.table_next_column()
                changed, new_val = layout.checkbox(
                    "##py_invert_masks", params.invert_masks
                )
                if changed:
                    params.invert_masks = new_val
                if layout.is_item_hovered():
                    layout.set_tooltip(tr("training.tooltip.invert_masks"))

                layout.table_next_row()
                layout.table_next_column()
                layout.label(tr("training_params.use_alpha_as_mask"))
                layout.table_next_column()
                changed, new_val = layout.checkbox(
                    "##py_use_alpha_as_mask", params.use_alpha_as_mask
                )
                if changed:
                    params.use_alpha_as_mask = new_val
                if layout.is_item_hovered():
                    layout.set_tooltip(tr("training.tooltip.use_alpha_as_mask"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.sparsity"))
            layout.table_next_column()
            changed, new_val = layout.checkbox("##py_sparsity", params.enable_sparsity)
            if changed:
                params.enable_sparsity = new_val
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.sparsity"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.gut"))
            layout.table_next_column()
            gut_disabled = params.strategy == "igs+"
            if gut_disabled:
                layout.begin_disabled(True)
            changed, new_val = layout.checkbox("##py_gut", params.gut)
            if changed:
                params.gut = new_val
                self._sync_render_setting("gut", new_val)
            if gut_disabled:
                layout.end_disabled()
            if layout.is_item_hovered():
                tooltip = (
                    tr("training.tooltip.gut_strategy_conflict")
                    if gut_disabled
                    else tr("training.tooltip.gut")
                )
                layout.set_tooltip(tooltip)

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.undistort"))
            layout.table_next_column()
            changed, new_val = layout.checkbox("##py_undistort", params.undistort)
            if changed:
                params.undistort = new_val
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.undistort"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.mip_filter"))
            layout.table_next_column()
            changed, new_val = layout.checkbox("##py_mip_filter", params.mip_filter)
            if changed:
                params.mip_filter = new_val
                self._sync_render_setting("mip_filter", new_val)
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.mip_filter"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.ppisp"))
            layout.table_next_column()
            changed, new_val = layout.checkbox("##py_ppisp", params.ppisp)
            if changed:
                params.ppisp = new_val
                self._sync_render_setting("apply_appearance_correction", new_val)
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.ppisp"))

            if params.ppisp:
                layout.table_next_row()
                layout.table_next_column()
                layout.label(tr("training_params.ppisp_controller"))
                layout.table_next_column()
                changed, new_val = layout.checkbox(
                    "##py_ppisp_controller", params.ppisp_use_controller
                )
                if changed:
                    params.ppisp_use_controller = new_val
                if layout.is_item_hovered():
                    layout.set_tooltip(tr("training.tooltip.ppisp_controller"))

                if params.ppisp_use_controller:
                    layout.table_next_row()
                    layout.table_next_column()
                    layout.label(tr("training_params.ppisp_activation_step"))
                    layout.table_next_column()
                    is_auto = params.ppisp_controller_activation_step < 0
                    changed, new_auto = layout.checkbox(
                        f"{tr('common.auto')}##py_ppisp_auto_step", is_auto
                    )
                    if changed:
                        params.ppisp_controller_activation_step = (
                            -1 if new_auto else max(1, int(params.iterations) - 5000)
                        )
                    if not is_auto:
                        layout.same_line()
                        layout.push_item_width(-1)
                        changed, new_val = layout.input_int_formatted(
                            "##py_ppisp_ctrl_step",
                            params.ppisp_controller_activation_step,
                            1000,
                            5000,
                        )
                        if changed:
                            params.ppisp_controller_activation_step = max(1, new_val)
                        layout.pop_item_width()
                    if layout.is_item_hovered():
                        layout.set_tooltip(tr("training.tooltip.ppisp_activation_step"))

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.label(tr("training_params.ppisp_controller_lr"))
                    layout.table_next_column()
                    layout.push_item_width(-1)
                    changed, new_val = layout.input_float(
                        "##py_ppisp_ctrl_lr",
                        params.ppisp_controller_lr,
                        0.0001,
                        0.001,
                        "%.5f",
                    )
                    if changed:
                        params.ppisp_controller_lr = new_val
                    layout.pop_item_width()
                    if layout.is_item_hovered():
                        layout.set_tooltip(tr("training.tooltip.ppisp_controller_lr"))

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.label(tr("training_params.ppisp_freeze_gaussians"))
                    layout.table_next_column()
                    changed, new_val = layout.checkbox(
                        "##py_ppisp_freeze", params.ppisp_freeze_gaussians
                    )
                    if changed:
                        params.ppisp_freeze_gaussians = new_val
                    if layout.is_item_hovered():
                        layout.set_tooltip(
                            tr("training.tooltip.ppisp_freeze_gaussians")
                        )

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.bg_mode"))
            layout.table_next_column()
            layout.push_item_width(-1)
            bg_idx = params.bg_mode.value
            bg_mode_items = [
                tr("training.options.bg.color"),
                tr("training.options.bg.modulation"),
                tr("training.options.bg.image"),
                tr("training.options.bg.random"),
            ]
            changed, new_idx = layout.combo("##py_bg_mode", bg_idx, bg_mode_items)
            if changed:
                params.bg_mode = lf.BackgroundMode(new_idx)
            layout.pop_item_width()

            bg_mode_val = params.bg_mode.value
            if bg_mode_val in (0, 1):
                layout.table_next_row()
                layout.table_next_column()
                layout.label(tr("training_params.bg_color"))
                layout.table_next_column()
                layout.push_item_width(-1)
                changed, new_color = layout.color_edit3(
                    "##py_bg_color", params.bg_color
                )
                if changed:
                    params.bg_color = new_color
                    self._sync_render_setting("background_color", new_color)
                layout.pop_item_width()

            if bg_mode_val == 2:
                layout.table_next_row()
                layout.table_next_column()
                layout.label(tr("training_params.bg_image"))
                layout.table_next_column()
                layout.push_item_width(-1)
                img_path = params.bg_image_path
                display = (
                    os.path.basename(img_path)
                    if img_path
                    else tr("training.value.none")
                )
                layout.label(display)
                layout.pop_item_width()

                layout.table_next_row()
                layout.table_next_column()
                layout.table_next_column()
                if layout.button(
                    tr("training_params.bg_image_browse") + "##py_bg_browse"
                ):
                    selected = lf.ui.open_image_file_dialog("")
                    if selected:
                        params.bg_image_path = selected
                layout.same_line()
                if img_path and layout.button(
                    tr("training_params.bg_image_clear") + "##py_bg_clear"
                ):
                    params.bg_image_path = ""

            layout.end_disabled()
            layout.end_table()

    def _draw_advanced_params(self, layout, state, iteration, params):
        can_edit = (state == "ready") and (iteration == 0)
        dataset = lf.dataset_params()
        dataset_can_edit = dataset.can_edit() if dataset.has_params() else False

        if layout.tree_node(tr("training.section.dataset") + "##py"):
            table_open = False
            try:
                if dataset.has_params():
                    table_open = layout.begin_table("PyDatasetTable", 2)
                    if table_open:
                        layout.table_setup_column(tr("common.column_label"), 120.0)
                        layout.table_setup_column(tr("common.column_control"), 0.0)

                        data_path = dataset.data_path
                        self._table_text(
                            layout,
                            tr("training.dataset.path"),
                            os.path.basename(data_path)
                            if data_path
                            else tr("training.value.none"),
                        )

                        images = dataset.images
                        self._table_text(
                            layout,
                            tr("training.dataset.images"),
                            images if images else tr("training.value.default"),
                        )

                        layout.table_next_row()
                        layout.table_next_column()
                        layout.label(tr("training.dataset.resize_factor"))
                        layout.table_next_column()
                        if dataset_can_edit:
                            layout.push_item_width(-1)
                            resize_options = [-1, 1, 2, 4, 8]
                            resize_labels = [tr("common.auto"), "1", "2", "4", "8"]
                            current_idx = (
                                resize_options.index(dataset.resize_factor)
                                if dataset.resize_factor in resize_options
                                else 0
                            )
                            changed, new_idx = layout.combo(
                                "##py_resize_factor", current_idx, resize_labels
                            )
                            if changed:
                                dataset.resize_factor = resize_options[new_idx]
                            layout.pop_item_width()
                        else:
                            layout.label(
                                tr("common.auto")
                                if dataset.resize_factor < 0
                                else str(dataset.resize_factor)
                            )

                        layout.table_next_row()
                        layout.table_next_column()
                        layout.label(tr("training.dataset.max_width"))
                        layout.table_next_column()
                        if dataset_can_edit:
                            layout.push_item_width(-1)
                            changed, new_val = layout.input_int(
                                "##py_max_width", dataset.max_width, 80, 400
                            )
                            if changed and new_val >= 0:
                                dataset.max_width = new_val
                            layout.pop_item_width()
                        else:
                            layout.label(str(dataset.max_width))

                        layout.table_next_row()
                        layout.table_next_column()
                        layout.label(tr("training.dataset.cpu_cache"))
                        layout.table_next_column()
                        if dataset_can_edit:
                            changed, new_val = layout.checkbox(
                                "##py_cpu_cache", dataset.use_cpu_cache
                            )
                            if changed:
                                dataset.use_cpu_cache = new_val
                        else:
                            layout.label(
                                tr("training.status.enabled")
                                if dataset.use_cpu_cache
                                else tr("training.status.disabled")
                            )

                        layout.table_next_row()
                        layout.table_next_column()
                        layout.label(tr("training.dataset.fs_cache"))
                        layout.table_next_column()
                        if dataset_can_edit:
                            changed, new_val = layout.checkbox(
                                "##py_fs_cache", dataset.use_fs_cache
                            )
                            if changed:
                                dataset.use_fs_cache = new_val
                        else:
                            layout.label(
                                tr("training.status.enabled")
                                if dataset.use_fs_cache
                                else tr("training.status.disabled")
                            )

                        out_path = dataset.output_path
                        self._table_text(
                            layout,
                            tr("training.dataset.output"),
                            os.path.basename(out_path)
                            if out_path
                            else tr("training.value.not_set"),
                        )
                else:
                    layout.label(tr("training_panel.no_dataset_loaded"))
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if layout.tree_node(tr("training.section.optimization") + "##py"):
            table_open = False
            try:
                table_open = layout.begin_table("PyOptTable", 2)
                if table_open:
                    layout.table_setup_column(tr("common.column_label"), 120.0)
                    layout.table_setup_column(tr("common.column_control"), 0.0)

                    layout.begin_disabled(not can_edit)
                    self._table_text(
                        layout, tr("training_params.strategy"), params.strategy.upper()
                    )

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.text_colored(
                        tr("training.opt.learning_rates"), (0.6, 0.6, 0.6, 1.0)
                    )
                    layout.table_next_column()

                    self._input_float_row(
                        layout,
                        tr("training.opt.lr.position"),
                        "means_lr",
                        params,
                        params.means_lr,
                        0.000001,
                        0.00001,
                        "%.6f",
                    )
                    self._input_float_row(
                        layout,
                        tr("training.opt.lr.sh_coeff"),
                        "shs_lr",
                        params,
                        params.shs_lr,
                        0.0001,
                        0.001,
                        "%.4f",
                    )
                    self._input_float_row(
                        layout,
                        tr("training.opt.lr.opacity"),
                        "opacity_lr",
                        params,
                        params.opacity_lr,
                        0.001,
                        0.01,
                        "%.4f",
                    )
                    self._input_float_row(
                        layout,
                        tr("training.opt.lr.scaling"),
                        "scaling_lr",
                        params,
                        params.scaling_lr,
                        0.0001,
                        0.001,
                        "%.4f",
                    )
                    self._input_float_row(
                        layout,
                        tr("training.opt.lr.rotation"),
                        "rotation_lr",
                        params,
                        params.rotation_lr,
                        0.0001,
                        0.001,
                        "%.4f",
                    )

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.text_colored(
                        tr("training.section.refinement"), (0.6, 0.6, 0.6, 1.0)
                    )
                    layout.table_next_column()

                    self._input_int_row(
                        layout,
                        tr("training.refinement.refine_every"),
                        "refine_every",
                        params,
                        10,
                        100,
                    )
                    self._input_int_row(
                        layout,
                        tr("training.refinement.start_refine"),
                        "start_refine",
                        params,
                        100,
                        500,
                    )
                    self._input_int_row(
                        layout,
                        tr("training.refinement.stop_refine"),
                        "stop_refine",
                        params,
                        1000,
                        5000,
                    )
                    self._input_float_prop_row(
                        layout,
                        tr("training.refinement.gradient_thr"),
                        "grad_threshold",
                        params,
                        0.00001,
                        0.0001,
                        "%.6f",
                    )
                    self._input_int_row(
                        layout,
                        tr("training.refinement.reset_every"),
                        "reset_every",
                        params,
                        100,
                        1000,
                    )
                    self._input_int_row(
                        layout,
                        tr("training.refinement.sh_upgrade_every"),
                        "sh_degree_interval",
                        params,
                        100,
                        500,
                    )
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if params.use_bilateral_grid and layout.tree_node(
            tr("training.section.bilateral_grid") + "##py"
        ):
            table_open = False
            try:
                table_open = layout.begin_table("PyBilateralTable", 2)
                if table_open:
                    layout.table_setup_column(tr("common.column_label"), 140.0)
                    layout.table_setup_column(tr("common.column_control"), 0.0)
                    layout.begin_disabled(not can_edit)
                    self._table_prop(
                        layout,
                        params,
                        "bilateral_grid_x",
                        tr("training.bilateral.grid_x"),
                    )
                    self._table_prop(
                        layout,
                        params,
                        "bilateral_grid_y",
                        tr("training.bilateral.grid_y"),
                    )
                    self._table_prop(
                        layout,
                        params,
                        "bilateral_grid_w",
                        tr("training.bilateral.grid_w"),
                    )
                    self._table_prop(
                        layout,
                        params,
                        "bilateral_grid_lr",
                        tr("training.bilateral.learning_rate"),
                    )
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if layout.tree_node(tr("training.section.losses") + "##py"):
            table_open = False
            try:
                table_open = layout.begin_table("PyLossTable", 2)
                if table_open:
                    layout.table_setup_column(tr("common.column_label"), 140.0)
                    layout.table_setup_column(tr("common.column_control"), 0.0)
                    layout.begin_disabled(not can_edit)
                    self._slider_float_row(
                        layout,
                        tr("training.losses.lambda_dssim"),
                        "lambda_dssim",
                        params,
                        0.0,
                        1.0,
                    )
                    self._input_float_prop_row(
                        layout,
                        tr("training.losses.opacity_reg"),
                        "opacity_reg",
                        params,
                        0.001,
                        0.01,
                        "%.4f",
                    )
                    self._input_float_prop_row(
                        layout,
                        tr("training.losses.scale_reg"),
                        "scale_reg",
                        params,
                        0.001,
                        0.01,
                        "%.4f",
                    )
                    self._input_float_prop_row(
                        layout,
                        tr("training.losses.tv_loss_weight"),
                        "tv_loss_weight",
                        params,
                        1.0,
                        5.0,
                        "%.1f",
                    )
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if layout.tree_node(tr("training.section.initialization") + "##py"):
            table_open = False
            try:
                table_open = layout.begin_table("PyInitTable", 2)
                if table_open:
                    layout.table_setup_column(tr("common.column_label"), 140.0)
                    layout.table_setup_column(tr("common.column_control"), 0.0)
                    layout.begin_disabled(not can_edit)
                    self._slider_float_row(
                        layout,
                        tr("training.init.init_opacity"),
                        "init_opacity",
                        params,
                        0.01,
                        1.0,
                    )
                    self._input_float_prop_row(
                        layout,
                        tr("training.init.init_scaling"),
                        "init_scaling",
                        params,
                        0.01,
                        0.1,
                        "%.3f",
                    )

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.label(tr("training.init.random_init"))
                    layout.table_next_column()
                    changed, new_val = layout.checkbox("##py_random", params.random)
                    if changed:
                        params.random = new_val

                    if params.random:
                        self._input_int_row(
                            layout,
                            tr("training.init.num_points"),
                            "init_num_pts",
                            params,
                            10000,
                            50000,
                        )
                        self._input_float_prop_row(
                            layout,
                            tr("training.init.extent"),
                            "init_extent",
                            params,
                            0.5,
                            1.0,
                            "%.1f",
                        )
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if params.strategy == "igs+" and layout.tree_node(
            tr("training_panel.pruning_growing") + "##py"
        ):
            table_open = False
            try:
                table_open = layout.begin_table("PyPruningGrowingTable", 2)
                if table_open:
                    layout.table_setup_column(tr("common.column_label"), 140.0)
                    layout.table_setup_column(tr("common.column_control"), 0.0)
                    layout.begin_disabled(not can_edit)
                    self._input_float_prop_row(
                        layout,
                        tr("training.thresholds.min_opacity"),
                        "min_opacity",
                        params,
                        0.001,
                        0.01,
                        "%.4f",
                        min_val=0.0,
                    )
                    self._input_float_prop_row(
                        layout,
                        tr("training.thresholds.prune_opacity"),
                        "prune_opacity",
                        params,
                        0.001,
                        0.01,
                        "%.4f",
                        min_val=0.0,
                    )
                    self._input_float_prop_row(
                        layout,
                        tr("training.thresholds.grow_scale_3d"),
                        "grow_scale3d",
                        params,
                        0.001,
                        0.01,
                        "%.4f",
                        min_val=0.0,
                    )
                    self._input_float_prop_row(
                        layout,
                        tr("training.thresholds.grow_scale_2d"),
                        "grow_scale2d",
                        params,
                        0.01,
                        0.05,
                        "%.3f",
                        min_val=0.0,
                    )
                    self._input_float_prop_row(
                        layout,
                        tr("training.thresholds.prune_scale_3d"),
                        "prune_scale3d",
                        params,
                        0.01,
                        0.1,
                        "%.3f",
                        min_val=0.0,
                    )
                    self._input_float_prop_row(
                        layout,
                        tr("training.thresholds.prune_scale_2d"),
                        "prune_scale2d",
                        params,
                        0.01,
                        0.1,
                        "%.3f",
                        min_val=0.0,
                    )
                    self._input_int_row(
                        layout,
                        tr("training.thresholds.pause_after_reset"),
                        "pause_refine_after_reset",
                        params,
                        100,
                        500,
                    )
                    self._table_prop(
                        layout,
                        params,
                        "revised_opacity",
                        tr("training.thresholds.revised_opacity"),
                    )
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if _is_mrnf_strategy(params.strategy) and layout.tree_node(
            tr("training_panel.mrnf_params") + "##py"
        ):
            table_open = False
            try:
                table_open = layout.begin_table("PyMRNFTable", 2)
                if table_open:
                    layout.table_setup_column(tr("common.column_label"), 140.0)
                    layout.table_setup_column(tr("common.column_control"), 0.0)
                    layout.begin_disabled(not can_edit)
                    self._input_float_prop_row(
                        layout,
                        "Growth Grad Threshold",
                        "growth_grad_threshold",
                        params,
                        0.0001,
                        0.001,
                        "%.5f",
                        min_val=0.0,
                    )
                    self._input_float_prop_row(
                        layout,
                        "Grow Fraction",
                        "grow_fraction",
                        params,
                        0.01,
                        0.05,
                        "%.3f",
                        min_val=0.0,
                        max_val=1.0,
                    )
                    self._input_int_row(
                        layout, "Grow Until Iter", "grow_until_iter", params, 1000, 5000
                    )
                    self._input_float_prop_row(
                        layout,
                        "Opacity Decay",
                        "opacity_decay",
                        params,
                        0.0001,
                        0.001,
                        "%.4f",
                        min_val=0.0,
                    )
                    self._input_float_prop_row(
                        layout,
                        "Scale Decay",
                        "scale_decay",
                        params,
                        0.0001,
                        0.001,
                        "%.4f",
                        min_val=0.0,
                    )
                    self._input_float_prop_row(
                        layout,
                        "Means Noise Weight",
                        "means_noise_weight",
                        params,
                        1.0,
                        10.0,
                        "%.1f",
                        min_val=0.0,
                    )
                    self._input_float_prop_row(
                        layout,
                        "Bounds Percentile",
                        "bounds_percentile",
                        params,
                        0.01,
                        0.05,
                        "%.2f",
                        min_val=0.5,
                        max_val=1.0,
                    )

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.label("Error Map")
                    layout.table_next_column()
                    changed, new_val = layout.checkbox(
                        "##py_use_error_map", params.use_error_map
                    )
                    if changed:
                        params.use_error_map = new_val

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.label("Edge Map")
                    layout.table_next_column()
                    changed, new_val = layout.checkbox(
                        "##py_use_edge_map", params.use_edge_map
                    )
                    if changed:
                        params.use_edge_map = new_val

                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if params.enable_sparsity and layout.tree_node(
            tr("training_panel.sparsity") + "##py"
        ):
            table_open = False
            try:
                table_open = layout.begin_table("PySparsityTable", 2)
                if table_open:
                    layout.table_setup_column(tr("common.column_label"), 140.0)
                    layout.table_setup_column(tr("common.column_control"), 0.0)
                    layout.begin_disabled(not can_edit)
                    self._input_int_row(
                        layout,
                        tr("training_params.sparsify_steps"),
                        "sparsify_steps",
                        params,
                        1000,
                        5000,
                    )
                    self._input_float_prop_row(
                        layout,
                        tr("training_params.init_rho"),
                        "init_rho",
                        params,
                        0.001,
                        0.01,
                        "%.4f",
                    )
                    self._slider_float_row(
                        layout,
                        tr("training_params.prune_ratio"),
                        "prune_ratio",
                        params,
                        0.0,
                        1.0,
                    )
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if layout.tree_node(tr("training_panel.save_steps") + "##py"):
            try:
                self._draw_save_steps(layout, params, can_edit)
            finally:
                layout.tree_pop()

    def _draw_save_steps(self, layout, params, can_edit):
        theme = lf.ui.theme()
        steps = list(params.save_steps)

        if can_edit:
            _, self._new_save_step = layout.input_int_formatted(
                "##py_new_step", self._new_save_step, 100, 1000
            )
            layout.same_line()
            if layout.button(tr("common.add") + "##py_add"):
                if self._new_save_step > 0:
                    params.add_save_step(self._new_save_step)
                    if params.enable_eval:
                        self._sync_eval_steps_with_save_steps(params)

            layout.separator()

            for i, step in enumerate(steps):
                layout.push_id(f"py_step_{i}")
                layout.set_next_item_width(100)
                changed, new_val = layout.input_int_formatted("##step", step, 0, 0)
                if changed and new_val > 0 and new_val != step:
                    params.remove_save_step(step)
                    params.add_save_step(new_val)
                    if params.enable_eval:
                        self._sync_eval_steps_with_save_steps(params)
                layout.same_line()
                if layout.button(tr("common.remove") + "##rm"):
                    params.remove_save_step(step)
                    if params.enable_eval:
                        self._remove_from_eval_steps(params, step)
                layout.pop_id()

            if not steps:
                layout.text_colored(
                    tr("training_panel.no_save_steps"), theme.palette.text_dim
                )
        else:
            if steps:
                layout.label(", ".join(str(s) for s in steps))
            else:
                layout.text_colored(
                    tr("training_panel.no_save_steps"), theme.palette.text_dim
                )

    def _input_int_row(self, layout, label, prop_id, params, step, step_fast):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.push_item_width(-1)
        current_val = params.get(prop_id)
        if current_val is None:
            current_val = 0
        changed, new_val = layout.input_int_formatted(
            f"##py_{prop_id}", int(current_val), step, step_fast
        )
        if changed and new_val >= 0:
            params.set(prop_id, new_val)
        layout.pop_item_width()

    def _input_float_prop_row(
        self, layout, label, prop_id, params, step, step_fast, fmt, min_val=None
    ):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.push_item_width(-1)
        current_val = params.get(prop_id)
        if current_val is None:
            current_val = 0.0
        changed, new_val = layout.input_float(
            f"##py_{prop_id}", float(current_val), step, step_fast, fmt
        )
        if changed:
            if min_val is not None:
                new_val = max(min_val, new_val)
            params.set(prop_id, new_val)
        layout.pop_item_width()

    def _slider_float_row(self, layout, label, prop_id, params, min_val, max_val):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.push_item_width(-1)
        current_val = params.get(prop_id)
        if current_val is None:
            current_val = 0.0
        changed, new_val = layout.slider_float(
            f"##py_{prop_id}", float(current_val), min_val, max_val
        )
        if changed:
            params.set(prop_id, new_val)
        layout.pop_item_width()

    def _input_float_row(
        self, layout, label, prop_id, params, value, step, step_fast, fmt
    ):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.push_item_width(-1)
        changed, new_val = layout.input_float(
            f"##py_{prop_id}", value, step, step_fast, fmt
        )
        if changed:
            setattr(params, prop_id, new_val)
        layout.pop_item_width()

    def _table_prop(self, layout, params, prop_id, label):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.push_item_width(-1)
        layout.push_id(f"py_{prop_id}")
        layout.prop(params, prop_id)
        layout.pop_id()
        layout.pop_item_width()

    def _table_text(self, layout, label, value):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.label(value)

    def _draw_status(self, layout, state, iteration):
        layout.separator()

        state_labels = {
            "idle": tr("training_panel.idle"),
            "ready": tr("status.ready")
            if iteration == 0
            else tr("training_panel.resume"),
            "running": tr("training_panel.running"),
            "paused": tr("status.paused"),
            "stopping": tr("status.stopping"),
            "completed": tr("status.complete"),
            "stopped": tr("status.stopped"),
            "error": tr("status.error"),
        }
        unknown_state = tr("status.unknown")
        layout.label(f"{tr('status.mode')}: {state_labels.get(state, unknown_state)}")

        _rate_tracker.add_sample(iteration)
        rate = _rate_tracker.get_rate()
        layout.label(
            f"{tr('status.iteration')} {iteration:,} ({rate:.1f} {tr('training_panel.iters_per_sec')})"
        )
        layout.label(tr("progress.num_splats") % f"{AppState.num_gaussians.value:,}")

        max_iter = AppState.max_iterations.value
        if max_iter > 0 and iteration > 0:
            layout.progress_bar(iteration / max_iter, f"{iteration:,}/{max_iter:,}")

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

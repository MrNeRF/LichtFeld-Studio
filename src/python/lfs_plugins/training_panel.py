# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Training Panel - RmlUI with native data binding."""

import os
import threading
import time

import lichtfeld as lf

from . import rml_widgets as w
from . import property_view
from .property_view import parse_number as _parse_num
from .scrub_fields import ScrubFieldController, ScrubFieldSpec
from .types import Panel
from .ui import RuntimeState, PanelStateBinding

__lfs_panel_classes__ = ["TrainingPanel"]
__lfs_panel_ids__ = ["lfs.training"]

def tr(key):
    result = lf.ui.tr(key)
    return result if result else key


def tr_fallback(key, fallback):
    result = lf.ui.tr(key)
    return result if result and result != key else fallback


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
    return property_view.canonical_strategy_name(strategy) == "mrnf"


DEPTH_LOSS_MODE_VALUES = ("ssi", "ssi-disparity", "ssi-depth")
DEFAULT_DEPTH_LOSS_MODE = "ssi"


def _depth_loss_mode_or_default(mode):
    mode = str(mode or "")
    return mode if mode in DEPTH_LOSS_MODE_VALUES else DEFAULT_DEPTH_LOSS_MODE


STRATEGY_LABEL_KEYS = {
    "mcmc": "training.options.strategy.mcmc",
    "mrnf": "training.options.strategy.mrnf",
    "mnrf": "training.options.strategy.mrnf",
    "lfs": "training.options.strategy.mrnf",
    "igs+": "training.options.strategy.igs_plus",
}

DATASET_BOOL_PROPS = ["use_cpu_cache", "use_16bit_color"]

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
    "sparsity",
    "save_steps",
    "advanced_registry",
]

INITIALLY_COLLAPSED = {
    "advanced_params",
    "advanced_registry",
    "dataset",
    "optimization",
    "bilateral",
    "losses",
    "init",
    "sparsity",
    "save_steps",
}


BG_COLOR_CHANNELS = (
    ("bg_color_r", 0),
    ("bg_color_g", 1),
    ("bg_color_b", 2),
)
BG_COLOR_CHANNEL_INDEX = dict(BG_COLOR_CHANNELS)
BG_COLOR_HEX_KEY = "bg_color_hex"
BG_COLOR_TEXT_KEYS = tuple(key for key, _index in BG_COLOR_CHANNELS) + (
    BG_COLOR_HEX_KEY,
)


class TrainingPanel(Panel):
    id = "lfs.training"
    label = "Training"
    space = lf.ui.PanelSpace.MAIN_PANEL_TAB
    order = 20
    template = "rmlui/training.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    update_policy = "dirty"

    def __init__(self):
        self._pv_bindings = ()
        self._pv_binding_by_prop = {}
        self._pv_search_query = ""
        self._pv_publish_pending = []
        self._pv_publish_pending_ids = set()
        self._pv_publish_scheduled = False
        self._handle = None
        self._project_saved_time = 0.0
        self._new_save_step = 7000
        self._auto_scaled_for_cameras = 0
        self._auto_scale_steps_locked = True
        self._auto_scale_user_override = False
        self._auto_scale_dataset_path = ""
        self._last_state = ""
        self._last_save_steps = None
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
        self._last_project_saved_visible = False
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
        self._last_language_generation = -1
        self._reactive_binding = PanelStateBinding()
        self._deferred_update_pending = False
        self._deferred_update_deadline = None
        self._deferred_update_generation = 0
        self._escape_revert = w.EscapeRevertController()
        self._scrub_fields = ScrubFieldController(
            SCRUB_FIELD_DEFS,
            self._get_scrub_value,
            self._set_scrub_value,
        )

    def capture_chrome(self):
        return {
            "collapsed": sorted(self._collapsed),
            "property_search": self._pv_search_query,
            "steps_scaling_lock": bool(self._auto_scale_steps_locked),
        }

    def apply_chrome(self, payload):
        self._collapsed = set(INITIALLY_COLLAPSED)
        self._pv_search_query = ""
        self._auto_scale_steps_locked = True
        if not isinstance(payload, dict):
            if self._handle:
                self._handle.dirty_all()
            return
        collapsed = payload.get("collapsed")
        if isinstance(collapsed, (list, tuple)):
            self._collapsed = {str(name) for name in collapsed}
        search = payload.get("property_search")
        if isinstance(search, str):
            self._pv_search_query = search
        if "steps_scaling_lock" in payload:
            self._auto_scale_steps_locked = bool(payload.get("steps_scaling_lock"))
        if self._handle:
            self._handle.dirty_all()

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("training")
        if model is None:
            return

        p = lf.optimization_params
        d = lf.dataset_params

        self._bind_labels(model)
        self._bind_property_search(model)
        self._bind_visibility(model, p, d)
        self._bind_disabled(model, p)
        self._bind_dataset_bools(model, d)
        self._bind_select_props(model, p, d)
        self._bind_text_props(model, p)
        self._bind_bespoke_num_props(model, p, d)
        self._bind_slider_props(model, p)
        self._bind_color(model, p)
        self._bind_status(model, p)
        self._bind_display(model, p, d)
        self._pv_bindings = property_view.bind_sections(
            model,
            p,
            self._text_bufs,
            publisher=self._queue_pv_publish,
            value_setter=self._set_property_view_value,
            search_accessor=lambda: self._pv_search_query,
            visibility_predicate=self._property_view_condition_visible,
        )
        self._pv_binding_by_prop = {
            row["id"]: binding
            for binding in self._pv_bindings
            for row in binding.rows
        }
        self._bind_events(model)
        self._handle = model.get_handle()
        for binding in self._pv_bindings:
            binding.attach_handle(self._handle)
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
        model.bind_func(
            "label_no_trainer", lambda: tr("training_panel.no_trainer_loaded")
        )
        model.bind_func(
            "label_no_params", lambda: tr("training_panel.parameters_unavailable")
        )
        model.bind_func("label_reset", lambda: tr("training_panel.reset"))
        model.bind_func("label_clear", lambda: tr("training_panel.clear"))
        model.bind_func(
            "pv_search_placeholder",
            lambda: tr("training.search.placeholder"),
        )
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
        model.bind_func(
            "label_save_project", lambda: tr("training_panel.save_project")
        )
        model.bind_func(
            "label_project_saved", lambda: tr("training_panel.project_saved")
        )
        model.bind_func("label_strategy", lambda: tr("training_params.strategy"))
        model.bind_func(
            "label_strategy_mrnf", lambda: tr("training.options.strategy.mrnf")
        )
        model.bind_func(
            "label_strategy_igs_plus",
            lambda: tr("training.options.strategy.igs_plus"),
        )
        model.bind_func(
            "label_strategy_mcmc", lambda: tr("training.options.strategy.mcmc")
        )
        model.bind_func("label_sh_degree", lambda: tr("training_params.sh_degree"))
        model.bind_func(
            "label_depth_loss_mode", lambda: tr("training_params.depth_loss_mode")
        )
        model.bind_func(
            "label_depth_loss_ssi", lambda: tr("training.options.depth_loss.ssi")
        )
        model.bind_func(
            "label_depth_loss_ssi_disparity",
            lambda: tr("training.options.depth_loss.ssi_disparity"),
        )
        model.bind_func(
            "label_depth_loss_ssi_depth",
            lambda: tr("training.options.depth_loss.ssi_depth"),
        )
        model.bind_func(
            "label_ppisp_sidecar_path",
            lambda: tr("training_params.ppisp_sidecar_path"),
        )
        model.bind_func(
            "label_ppisp_activation_step",
            lambda: tr("training_params.ppisp_activation_step"),
        )
        model.bind_func("label_ppisp_sidecar_clear", lambda: tr("training_panel.clear"))
        model.bind_func("label_bg_color", lambda: tr("training_params.bg_color"))
        model.bind_func("label_bg_image", lambda: tr("training_params.bg_image"))
        model.bind_func(
            "label_bg_browse", lambda: tr("training_params.bg_image_browse")
        )
        model.bind_func(
            "label_bg_clear", lambda: tr("training_params.bg_image_clear")
        )
        model.bind_func("label_dataset_path", lambda: tr("training.dataset.path"))
        model.bind_func(
            "label_dataset_images", lambda: tr("training.dataset.images")
        )
        model.bind_func(
            "label_resize_factor", lambda: tr("training.dataset.resize_factor")
        )
        model.bind_func("label_max_width", lambda: tr("training.dataset.max_width"))
        model.bind_func("label_cpu_cache", lambda: tr("training.dataset.cpu_cache"))
        model.bind_func(
            "label_use_16bit_color", lambda: tr("training.dataset.use_16bit_color")
        )
        model.bind_func(
            "label_dataset_output", lambda: tr("training.dataset.output")
        )
        model.bind_func("label_auto", lambda: tr("common.auto"))
        model.bind_func(
            "label_no_dataset", lambda: tr("training_panel.no_dataset_loaded")
        )
        model.bind_func("label_opt_strategy", lambda: tr("training_params.strategy"))
        model.bind_func(
            "label_test_every", lambda: tr("training.dataset.test_every")
        )
        model.bind_func(
            "label_lambda_dssim", lambda: tr("training.losses.lambda_dssim")
        )
        model.bind_func(
            "label_init_opacity", lambda: tr("training.init.init_opacity")
        )
        model.bind_func(
            "label_prune_ratio", lambda: tr("training_params.prune_ratio")
        )
        model.bind_func(
            "label_no_save_steps", lambda: tr("training_panel.no_save_steps")
        )
        model.bind_func("label_add", lambda: tr("common.add"))
        model.bind_func("label_remove", lambda: tr("common.remove"))
        model.bind_func(
            "steps_scaling_lock_tooltip", self._step_scaling_lock_tooltip
        )
        model.bind_func("steps_scaling_lock_icon", self._step_scaling_lock_icon)
        model.bind_func(
            "steps_scaling_lock_selected", lambda: self._auto_scale_steps_locked
        )

        def _btn_start():
            it = RuntimeState.iteration.value
            return (
                tr("training_panel.resume_training")
                if it > 0
                else tr("training_panel.start_training")
            )

        model.bind_func("btn_start", _btn_start)

    def _bind_property_search(self, model):
        model.bind(
            "pv_search_query",
            lambda: self._pv_search_query,
            self._set_property_search_query,
        )
        model.bind_func(
            "pv_search_active", lambda: bool(self._pv_search_query.strip())
        )

    def _set_property_search_query(self, value):
        query = str(value or "")
        if query == self._pv_search_query:
            return
        self._pv_search_query = query
        for binding in self._pv_bindings:
            self._queue_pv_publish(binding)

    def _property_view_condition_visible(self, condition_id):
        if str(condition_id) == "has_dataset":
            dataset = lf.dataset_params()
            return bool(dataset and dataset.has_params())

        params = lf.optimization_params()
        if not params or not params.has_params():
            return False

        mask_mode = params.mask_mode.value
        conditions = {
            "dep_mask_mode": mask_mode != 0,
            "dep_mask_segment": mask_mode in (1, 3),
            "dep_mask_threshold": mask_mode not in (0, 3),
            "dep_depth_loss": params.use_depth_loss,
            "dep_normal_loss": params.use_normal_loss,
            "dep_ppisp": params.ppisp,
            "dep_ppisp_controller": params.ppisp and params.ppisp_use_controller,
            "dep_bilateral": params.use_bilateral_grid,
            "dep_mrnf": _is_mrnf_strategy(params.strategy),
            "dep_igs": params.strategy == "igs+",
            "dep_sparsity": params.enable_sparsity,
            "dep_random": params.random,
        }
        return bool(conditions.get(str(condition_id), True))

    def _bind_visibility(self, model, p, d):
        def _state():
            return RuntimeState.trainer_state.value

        def _iteration():
            return RuntimeState.iteration.value

        model.bind_func("show_no_trainer", lambda: not RuntimeState.has_trainer.value)
        model.bind_func(
            "show_no_params",
            lambda: RuntimeState.has_trainer.value and not (p() and p().has_params()),
        )
        model.bind_func(
            "show_main",
            lambda: RuntimeState.has_trainer.value and p() is not None and p().has_params(),
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
        model.bind_func("show_project_save", lambda: _state() in ("running", "paused"))
        model.bind_func(
            "show_project_saved",
            lambda: (
                _state() in ("running", "paused")
                and time.time() - self._project_saved_time < 2.0
            ),
        )

        model.bind_func(
            "dep_mask_mode",
            lambda: p() is not None and p().has_params() and p().mask_mode.value != 0,
        )
        model.bind_func(
            "dep_mask_segment",
            lambda: p() is not None and p().has_params() and (p().mask_mode.value == 1 or p().mask_mode.value == 3),
        )
        model.bind_func(
            "dep_mask_threshold",
            lambda: p() is not None and p().has_params() and p().mask_mode.value != 3,
        )
        model.bind_func(
            "dep_depth_loss",
            lambda: p() is not None and p().has_params() and p().use_depth_loss,
        )
        model.bind_func(
            "dep_normal_loss",
            lambda: p() is not None and p().has_params() and p().use_normal_loss,
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
            "show_progress",
            lambda: RuntimeState.max_iterations.value > 0 and _iteration() > 0,
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
                RuntimeState.trainer_state.value == "ready"
                and RuntimeState.iteration.value == 0
            )

        model.bind_func("struct_disabled", _params_edit_locked)
        model.bind_func("live_disabled", _params_edit_locked)
        model.bind_func("adv_disabled", _params_edit_locked)
        model.bind_func(
            "step_scaling_params_locked",
            lambda: self._auto_scale_steps_locked,
        )
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
            "depth_loss_mode_str",
            lambda: (
                _depth_loss_mode_or_default(p().depth_loss_mode)
                if p() and p().has_params()
                else DEFAULT_DEPTH_LOSS_MODE
            ),
            lambda v: self._set_depth_loss_mode(v),
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

    def _bind_bespoke_num_props(self, model, p, d):
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
        color = tuple(snapshot or (0.0, 0.0, 0.0))
        if not self._set_training_bg_color(color):
            return
        self._sync_bg_color_text_bufs()
        if self._handle:
            self._handle.dirty_all()

    def _canonical_text_buf_value(self, key):
        p = lf.optimization_params()
        d = lf.dataset_params()

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

        if key == BG_COLOR_HEX_KEY:
            if p and p.has_params():
                return w.color_to_hex(p.bg_color)
            return "#000000"

        for channel_key, channel_index in BG_COLOR_CHANNELS:
            if key == channel_key:
                if p and p.has_params():
                    return w.color_channel_text(p.bg_color, channel_index)
                return "0"

        return None

    def _commit_number_input_key(self, key):
        original = self._text_bufs.get(key)
        buf_val = self._text_bufs.get(key)
        if buf_val is not None and buf_val.strip() and key.endswith("_str"):
            prop = key[:-4]
            if prop == "ppisp_activation_step":
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
        if original != canonical:
            self._text_bufs[key] = canonical
            self._mark_text_buf_dirty(key)

    def _commit_bg_color_text_key(self, key):
        buf_val = self._text_bufs.get(key)
        updated = False
        if buf_val is not None and str(buf_val).strip():
            if key == BG_COLOR_HEX_KEY:
                updated = self._set_bg_color_hex(buf_val)
            elif key in BG_COLOR_CHANNEL_INDEX:
                updated = self._set_bg_color_channel(key, buf_val)

        canonical = self._canonical_text_buf_value(key)
        if canonical is None:
            return
        self._text_bufs[key] = canonical
        if updated:
            self._sync_bg_color_text_bufs()
            self._dirty_bg_color_bindings()
        else:
            self._mark_text_buf_dirty(key)

    def _sync_text_bufs(self):
        p = lf.optimization_params()
        d = lf.dataset_params()
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
        self._sync_bg_color_text_bufs(p)
        for binding in self._pv_bindings:
            binding.sync_text_bufs()

    def _sync_bg_color_text_bufs(self, params=None):
        if params is None:
            params = lf.optimization_params()
        color = (
            params.bg_color
            if params and params.has_params()
            else (0.0, 0.0, 0.0)
        )
        for key, channel_index in BG_COLOR_CHANNELS:
            self._text_bufs[key] = w.color_channel_text(color, channel_index)
        self._text_bufs[BG_COLOR_HEX_KEY] = w.color_to_hex(color)

    def _dirty_bg_color_bindings(self):
        if not self._handle:
            return
        for key in BG_COLOR_TEXT_KEYS:
            self._handle.dirty(key)

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

        for key, channel_index in BG_COLOR_CHANNELS:
            self._text_bufs[key] = None

            def getter(k=key, idx=channel_index):
                if self._text_bufs[k] is None:
                    self._text_bufs[k] = w.color_channel_text(_bg(), idx)
                return self._text_bufs[k]

            def setter(v, k=key):
                self._text_bufs[k] = str(v)

            model.bind(key, getter, setter)

        self._text_bufs[BG_COLOR_HEX_KEY] = None

        def hex_getter():
            if self._text_bufs[BG_COLOR_HEX_KEY] is None:
                self._text_bufs[BG_COLOR_HEX_KEY] = w.color_to_hex(_bg())
            return self._text_bufs[BG_COLOR_HEX_KEY]

        def hex_setter(v):
            self._text_bufs[BG_COLOR_HEX_KEY] = str(v)

        model.bind(
            BG_COLOR_HEX_KEY,
            hex_getter,
            hex_setter,
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
            state = RuntimeState.trainer_state.value
            it = RuntimeState.iteration.value
            if state == "stopping" and lf.trainer_saving_model():
                return f"{tr('status.mode')} Saving model..."
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
            return f"{tr('status.mode')} {labels.get(state, tr('status.unknown'))}"

        def _status_iteration():
            it = RuntimeState.iteration.value
            _rate_tracker.add_sample(it)
            rate = _rate_tracker.get_rate()
            return f"{tr('status.iteration')} {it:,} ({rate:.1f} {tr('training_panel.iters_per_sec')})"

        def _status_gaussians():
            return tr("progress.num_splats") % f"{RuntimeState.num_gaussians.value:,}"

        def _progress_text():
            it = RuntimeState.iteration.value
            mx = RuntimeState.max_iterations.value
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
        model.bind_event("pv_step", self._on_num_step)
        model.bind_event("pv_focus", self._on_pv_number_input_focus)
        model.bind_event("pv_change", self._on_pv_number_input_change)
        model.bind_event("pv_blur", self._on_pv_number_input_blur)
        model.bind_event("pv_escape", self._on_pv_number_input_escape)
        model.bind_event("pv_value_change", self._on_pv_value_change)
        model.bind_event("pv_search_clear", self._on_pv_search_clear)
        model.bind_event(
            "toggle_step_scaling_lock", self._on_step_scaling_lock_toggle
        )

    def _step_scaling_lock_tooltip(self):
        if self._auto_scale_steps_locked:
            return tr_fallback(
                "training.tooltip.step_scaling_locked",
                "Auto-scales relevant training parameters.",
            )
        return tr_fallback(
            "training.tooltip.step_scaling_unlocked",
            "Manual: training parameters are not auto-scaled.",
        )

    def _step_scaling_lock_icon(self):
        state = "locked" if self._auto_scale_steps_locked else "unlocked"
        return f"../icon/scene/{state}.png"

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
            if el.get_attribute("data-pv-input", "") == "1":
                # data-for rows may be materialized after on_mount. Their inline
                # data-event handlers provide the equivalent behavior without
                # relying on mount-time DOM discovery.
                continue
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
                el.add_event_listener("change", self._on_bg_color_hex_change)
                el.add_event_listener("blur", self._on_bg_color_hex_blur)
        for el in doc.query_selector_all("input.color-channel"):
            key = el.get_attribute("data-value", "")
            if key not in BG_COLOR_CHANNEL_INDEX:
                continue
            w.bind_select_all_on_focus(el)
            self._escape_revert.bind(
                el,
                key,
                self._capture_bg_color_snapshot,
                self._restore_bg_color_snapshot,
            )
            el.add_event_listener("change", self._on_color_channel_input_change)
            el.add_event_listener("blur", self._on_color_channel_input_blur)
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
        self._subscribe_reactive_state()
        self._request_reactive_update()

    def _subscribe_reactive_state(self):
        if self._reactive_binding.active:
            return

        native_signals = (
            RuntimeState.training_running,
            RuntimeState.training_state,
            RuntimeState.trainer_loaded,
            RuntimeState.iteration,
            RuntimeState.total_iterations,
            RuntimeState.loss,
            RuntimeState.eval_psnr,
            RuntimeState.num_gaussians,
            RuntimeState.scene_generation,
            RuntimeState.language_generation,
        )
        self._reactive_binding.set_handle(self._handle).watch(*native_signals)

    def _sync_auto_scale_markers(self):
        d = lf.dataset_params()
        dataset_path = d.data_path if d and d.has_params() else ""
        if dataset_path != self._auto_scale_dataset_path:
            self._auto_scale_dataset_path = dataset_path
            self._auto_scaled_for_cameras = 0
            self._auto_scale_user_override = False

    def _unsubscribe_reactive_state(self):
        self._reactive_binding.close()

    def _request_reactive_update(self):
        if self._handle:
            w.request_model_update(self._handle)

    def _schedule_deferred_update(self, delay_seconds):
        delay_seconds = max(0.0, float(delay_seconds))
        deadline = time.monotonic() + delay_seconds
        if (
            self._deferred_update_pending
            and self._deferred_update_deadline is not None
            and self._deferred_update_deadline <= deadline
        ):
            return
        self._deferred_update_generation += 1
        self._deferred_update_pending = True
        self._deferred_update_deadline = deadline
        generation = self._deferred_update_generation

        def fire():
            def request_on_ui_thread():
                if generation != self._deferred_update_generation:
                    return
                self._deferred_update_pending = False
                self._deferred_update_deadline = None
                self._request_reactive_update()

            try:
                scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
                if scheduler is None:
                    scheduler = getattr(lf.ui, "_run_on_ui_thread", None)
                if callable(scheduler):
                    scheduler(request_on_ui_thread)
                else:
                    self._deferred_update_pending = False
                    self._deferred_update_deadline = None
            except Exception:
                self._deferred_update_pending = False
                self._deferred_update_deadline = None

        timer = threading.Timer(delay_seconds, fire)
        timer.daemon = True
        timer.start()

    def _cancel_deferred_updates(self):
        self._deferred_update_generation += 1
        self._deferred_update_pending = False
        self._deferred_update_deadline = None

    def _mark_project_saved(self):
        self._project_saved_time = time.time()
        self._last_project_saved_visible = True
        if self._handle:
            self._handle.dirty("show_project_saved")
        self._schedule_deferred_update(2.05)

    def on_update(self, doc):
        if not self._handle:
            return False
        self._sync_panel_label()
        self._sync_auto_scale_markers()

        dirty = self._flush_pv_publish()
        language_generation = RuntimeState.language_generation.value
        if language_generation != self._last_language_generation:
            self._last_language_generation = language_generation
            for binding in self._pv_bindings:
                binding.publish()
            self._handle.dirty_all()
            self._sync_section_states()
            dirty = True
        state = RuntimeState.trainer_state.value
        if state != self._last_state:
            self._last_state = state
            if state == "ready":
                _rate_tracker.clear()
            self._sync_text_bufs()
            self._handle.dirty_all()
            dirty = True
        else:
            it = RuntimeState.iteration.value
            if it != self._last_iteration:
                self._last_iteration = it
                self._handle.dirty("status_iteration")
                self._handle.dirty("progress_text")
                self._handle.dirty("show_progress")
                dirty = True
            if state == "stopping":
                self._handle.dirty("status_mode")

            ng = RuntimeState.num_gaussians.value
            if ng != self._last_num_gaussians:
                self._last_num_gaussians = ng
                self._handle.dirty("status_gaussians")
                dirty = True

            project_saved_visible = (
                self._project_saved_time > 0.0
                and time.time() - self._project_saved_time < 2.0
            )
            if project_saved_visible != self._last_project_saved_visible:
                self._last_project_saved_visible = project_saved_visible
                self._handle.dirty("show_project_saved")
                dirty = True

        if state == "ready" and RuntimeState.iteration.value == 0:
            params = lf.optimization_params()
            if params and params.has_params():
                if self._try_auto_scale_steps(params):
                    self._sync_text_bufs()
                    self._handle.dirty_all()
                    dirty = True

        dirty |= self._update_step_repeat()
        dirty |= self._update_progress()
        dirty |= self._update_save_steps(doc)
        dirty |= self._update_color_swatch(doc)
        dirty |= self._update_loss_graph()
        dirty |= self._update_psnr_graph()
        dirty |= self._scrub_fields.sync_all()
        return dirty

    def _update_progress(self):
        it = RuntimeState.iteration.value
        mx = RuntimeState.max_iterations.value
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

        state = RuntimeState.trainer_state.value
        can_edit = state == "ready" and RuntimeState.iteration.value == 0
        if not can_edit:
            return False

        return self._refresh_save_steps_model(params)

    def _refresh_save_steps_model(self, params=None):
        if params is None:
            params = lf.optimization_params()
        if not self._handle or not params or not params.has_params():
            self._last_save_steps = None
            return False

        steps = list(params.save_steps)
        if self._last_save_steps is None or steps != self._last_save_steps:
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
        self._sync_render_background_to_training(params)
        self._sync_bg_color_text_bufs(params)
        swatch = doc.get_element_by_id("swatch-bg_color")
        if swatch:
            r = w.color_channel_byte(c, 0)
            g = w.color_channel_byte(c, 1)
            b = w.color_channel_byte(c, 2)
            swatch.set_property("background-color", f"rgb({r},{g},{b})")
        if self._handle:
            self._dirty_bg_color_bindings()
        return True

    def on_scene_changed(self, doc):
        self._sync_render_background_to_training()
        if self._handle:
            self._sync_text_bufs()
            self._handle.dirty_all()

    def on_unmount(self, doc):
        self._unsubscribe_reactive_state()
        self._cancel_deferred_updates()
        doc.remove_data_model("training")
        self._handle = None
        self._pv_bindings = ()
        self._pv_binding_by_prop = {}
        self._pv_publish_pending = []
        self._pv_publish_pending_ids.clear()
        self._pv_publish_scheduled = False
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
        if self._color_edit_prop == "bg_color":
            self._set_training_bg_color((r, g, b))
        else:
            setattr(params, self._color_edit_prop, (r, g, b))
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

    def _sync_render_setting(self, prop, val):
        rs = lf.get_render_settings()
        if not rs:
            return
        if prop == "gut":
            rs.set("raster_backend", "3dgut" if val else "3dgs")
            return
        rs.set(prop, val)

    def _set_training_bg_color(self, color):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return False
        normalized = w.normalize_color(color)
        params.bg_color = normalized
        self._sync_render_setting("background_color", normalized)
        return True

    def _sync_render_background_to_training(self, params=None):
        if params is None:
            params = lf.optimization_params()
        if not params or not params.has_params():
            return False
        self._sync_render_setting("background_color", w.normalize_color(params.bg_color))
        return True

    def _set_bool_prop(self, prop, val):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return False
        if not hasattr(params, prop):
            return False
        if prop == "ppisp_freeze_from_sidecar" and val:
            params.ppisp = True
        elif prop == "ppisp" and not val:
            params.ppisp_freeze_from_sidecar = False
        if (
            prop == "enable_eval"
            and val
            and not self._clamp_current_test_every_for_eval()
        ):
            return False
        setattr(params, prop, val)
        if prop == "enable_eval" and val:
            self._sync_eval_steps_with_save_steps(params)
        if prop in RENDER_SYNC:
            self._sync_render_setting(RENDER_SYNC[prop], val)
        if self._handle:
            self._sync_text_bufs()
            self._handle.dirty_all()
        return True

    def _set_ppisp_sidecar_path(self, val):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        params.ppisp_sidecar_path = str(val)
        if self._handle:
            self._handle.dirty_all()

    def _refresh_strategy_values(self):
        if not self._handle:
            return
        self._sync_text_bufs()
        self._handle.dirty_all()
        self._request_reactive_update()

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
                    self._refresh_strategy_values()

            lf.ui.confirm_dialog(
                tr("training.error.strategy_gut_title"),
                tr("training.conflict.strategy_gut_strategy_message"),
                [btn_gut, btn_cancel],
                _on_conflict,
            )
        else:
            params.set_strategy(val)
            self._refresh_strategy_values()

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
            return False
        try:
            params.mask_mode = lf.MaskMode(int(val_str))
        except (ValueError, TypeError):
            return False
        if self._handle:
            self._sync_text_bufs()
            self._handle.dirty_all()
        return True

    def _set_depth_loss_mode(self, val_str):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return
        params.depth_loss_mode = _depth_loss_mode_or_default(val_str)
        if self._handle:
            self._handle.dirty_all()

    def _set_bg_mode(self, val_str):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return False
        try:
            params.bg_mode = lf.BackgroundMode(int(val_str))
        except (ValueError, TypeError):
            return False
        if self._handle:
            self._sync_text_bufs()
            self._handle.dirty_all()
        return True

    def _set_resize_factor(self, val_str):
        d = lf.dataset_params()
        if not d or not d.has_params():
            return
        try:
            d.resize_factor = int(val_str)
        except (ValueError, TypeError, RuntimeError):
            pass

    def _set_property_view_value(self, prop, value):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return False
        if prop in property_view.BOOL_PROPS:
            return self._set_bool_prop(prop, bool(value))
        if prop == "mask_mode":
            return self._set_mask_mode(value)
        if prop == "bg_mode":
            return self._set_bg_mode(value)
        if prop == "iterations":
            return self._set_iterations(params, int(value))
        try:
            params.set(prop, value)
        except (ValueError, TypeError, OverflowError, RuntimeError):
            return False
        return True

    def _set_iterations(self, params, val):
        if val <= 0:
            return False
        if not self._auto_scale_steps_locked:
            params.iterations = val
            return True

        current = max(1, int(getattr(params, "iterations", 0)))
        if current == val:
            return True

        current_scaler = float(getattr(params, "steps_scaler", 1.0))
        if current_scaler <= 0.0:
            current_scaler = 1.0
        next_scaler = current_scaler * (float(val) / float(current))
        params.apply_step_scaling(next_scaler)
        params.iterations = val
        if self._handle:
            self._sync_text_bufs()
            self._handle.dirty_all()
        self._auto_scale_user_override = True
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
        color = w.hex_to_color(hex_val)
        if color is not None and self._set_training_bg_color(color):
            return True
        return False

    def _set_bg_color_channel(self, key, val_str):
        params = lf.optimization_params()
        if not params or not params.has_params():
            return False
        parsed = w.parse_color_channel(val_str)
        if parsed is None:
            return False
        channel_index = BG_COLOR_CHANNEL_INDEX.get(key)
        if channel_index is None:
            return False
        color = list(w.normalize_color(params.bg_color))
        color[channel_index] = parsed
        return self._set_training_bg_color(color)

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
        self._schedule_deferred_update(0.15)

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

    def _on_pv_number_input_focus(self, _handle, event, args):
        if not args:
            return
        binding = self._pv_binding_by_prop.get(str(args[0]))
        if binding is None:
            return
        target = event.current_target()
        if target is not None:
            try:
                target.select()
            except Exception:
                pass
        binding.begin_edit(str(args[0]))

    def _on_pv_number_input_change(self, _handle, event, args):
        if not args:
            return
        prop = str(args[0])
        binding = self._pv_binding_by_prop.get(prop)
        if binding is None:
            return
        binding.update_draft(
            prop,
            args[1] if len(args) > 1 else event.get_parameter("value", ""),
        )
        if event.get_bool_parameter("linebreak", False):
            binding.commit(prop)

    def _on_pv_number_input_blur(self, _handle, _event, args):
        if not args:
            return
        prop = str(args[0])
        binding = self._pv_binding_by_prop.get(prop)
        if binding is None:
            return
        if len(args) > 1:
            binding.update_draft(prop, args[1])
        binding.commit(prop)
        binding.finish_edit(prop)

    def _on_pv_number_input_escape(self, _handle, event, args):
        if not args:
            return
        binding = self._pv_binding_by_prop.get(str(args[0]))
        if binding is not None and binding.cancel_edit(str(args[0])):
            event.stop_propagation()

    def _on_pv_value_change(self, _handle, _event, args):
        if len(args) < 2:
            return
        prop = str(args[0])
        binding = self._pv_binding_by_prop.get(prop)
        if binding is not None:
            binding.set_value(prop, args[1])

    def _on_pv_search_clear(self, *_args):
        self._set_property_search_query("")

    def _queue_pv_publish(self, binding):
        binding_id = id(binding)
        if binding_id not in self._pv_publish_pending_ids:
            self._pv_publish_pending_ids.add(binding_id)
            self._pv_publish_pending.append(binding)
        if self._pv_publish_scheduled:
            return
        self._pv_publish_scheduled = True

        def publish_after_event():
            self._flush_pv_publish()

        scheduler = getattr(lf.ui, "schedule_on_ui_thread", None)
        if scheduler is None:
            scheduler = getattr(lf.ui, "_run_on_ui_thread", None)
        if callable(scheduler):
            scheduler(publish_after_event)
        else:
            self._request_reactive_update()

    def _flush_pv_publish(self):
        if not self._pv_publish_pending:
            self._pv_publish_scheduled = False
            return False
        pending = self._pv_publish_pending
        self._pv_publish_pending = []
        self._pv_publish_pending_ids.clear()
        self._pv_publish_scheduled = False
        active_ids = {id(binding) for binding in self._pv_bindings}
        published = False
        for binding in pending:
            if id(binding) in active_ids:
                binding.publish()
                published = True
        if published:
            self._dirty_property_search_models()
            self._sync_section_states()
        return published

    def _dirty_property_search_models(self):
        if not self._handle:
            return
        for key in (
            "pv_search_query",
            "pv_search_active",
            *property_view.SEARCH_VISIBILITY_MODEL_KEYS,
        ):
            self._handle.dirty(key)

    def _on_color_channel_input_change(self, event):
        if not event.get_bool_parameter("linebreak", False):
            return
        target = event.current_target()
        if target is None:
            return
        self._commit_bg_color_text_key(target.get_attribute("data-value", ""))

    def _on_color_channel_input_blur(self, event):
        target = event.current_target()
        if target is None:
            return
        self._commit_bg_color_text_key(target.get_attribute("data-value", ""))

    def _on_bg_color_hex_change(self, event):
        if not event.get_bool_parameter("linebreak", False):
            return
        self._commit_bg_color_text_key(BG_COLOR_HEX_KEY)

    def _on_bg_color_hex_blur(self, event):
        self._commit_bg_color_text_key(BG_COLOR_HEX_KEY)

    def _on_step_scaling_lock_toggle(self, *_args):
        self._set_auto_scale_steps_locked(not self._auto_scale_steps_locked)

    def _apply_num_step(self, prop, direction):
        binding = self._pv_binding_by_prop.get(prop)
        if binding is not None:
            binding.step(prop, direction)
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
            return False
        now = time.monotonic()
        if now - self._step_repeat_start < 0.15:
            self._schedule_deferred_update(0.15 - (now - self._step_repeat_start))
            return False
        if now - self._step_repeat_last < 0.01:
            self._schedule_deferred_update(0.01 - (now - self._step_repeat_last))
            return False
        self._step_repeat_last = now
        self._apply_num_step(self._step_repeat_prop, self._step_repeat_dir)
        self._schedule_deferred_update(0.01)
        return True

    def _get_section_elements(self, name):
        if not self._doc:
            return None, None, None
        dom_name = name.replace("_", "-")
        header = self._doc.get_element_by_id(f"hdr-{dom_name}")
        arrow = self._doc.get_element_by_id(f"arrow-{dom_name}")
        content = self._doc.get_element_by_id(f"sec-{dom_name}")
        return header, arrow, content

    def _sync_section_states(self):
        search_active = bool(self._pv_search_query.strip())
        for name in SECTIONS:
            header, arrow, content = self._get_section_elements(name)
            if content:
                search_expanded = search_active and property_view.section_is_visible(
                    self._pv_bindings, name
                )
                w.sync_section_state(
                    content,
                    name not in self._collapsed or search_expanded,
                    header,
                    arrow,
                )

    def _on_toggle_section(self, handle, event, args):
        del handle, event
        if not args:
            return
        name = str(args[0])
        if self._pv_search_query.strip() and property_view.section_is_visible(
            self._pv_bindings, name
        ):
            self._sync_section_states()
            return
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
        elif action == "save_project":
            lf.project_save()
            self._mark_project_saved()
        elif action == "browse_bg":
            selected = lf.ui.open_image_dialog("")
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
                self._refresh_save_steps_model(params)

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
            self._refresh_save_steps_model(params)

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
        self._sync_auto_scale_markers()
        if not self._auto_scale_steps_locked or self._auto_scale_user_override:
            return False
        scene = lf.get_scene()
        if scene is None:
            return False
        camera_count = scene.active_camera_count
        if camera_count == 0 or camera_count == self._auto_scaled_for_cameras:
            return False
        self._auto_scaled_for_cameras = camera_count
        params.auto_scale_steps(camera_count)
        return True

    def _set_auto_scale_steps_locked(self, locked):
        locked = bool(locked)
        if self._auto_scale_steps_locked == locked:
            return
        self._auto_scale_steps_locked = locked

        if locked:
            self._auto_scaled_for_cameras = 0
            self._auto_scale_user_override = False
            params = lf.optimization_params()
            if params and params.has_params() and self._try_auto_scale_steps(params):
                self._sync_text_bufs()

        if self._handle:
            self._handle.dirty_all()

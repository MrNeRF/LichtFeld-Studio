# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Training Panel - Python implementation matching C++ panel exactly."""

import os
import time

import lichtfeld as lf

from .types import Panel
from .ui.state import AppState

COLOR_IDLE = (0.5, 0.5, 0.5, 1.0)
COLOR_MUTED = (0.6, 0.6, 0.6, 1.0)
COLOR_SUCCESS = (0.3, 0.9, 0.3, 1.0)
COLOR_ERROR = (0.9, 0.3, 0.3, 1.0)

FULL_WIDTH = (-1, 0)

TILE_MODE_ITEMS = ["1 (Full)", "2 (Half)", "4 (Quarter)"]
SH_DEGREE_ITEMS = ["0", "1", "2", "3"]
MASK_MODE_ITEMS = ["None", "Segment", "Ignore", "Alpha Consistent"]
BG_MODE_ITEMS = ["Color", "Modulation", "Image", "Random"]
STRATEGY_ITEMS = ["MCMC", "ADC"]


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


class TrainingPanel(Panel):
    idname = "lfs.training"
    label = "Training"
    space = "MAIN_PANEL_TAB"
    order = 20

    def __init__(self):
        self._checkpoint_saved_time = 0.0
        self._new_save_step = 7000

    def _sync_render_setting(self, name, value):
        rs = lf.get_render_settings()
        if rs:
            rs.set(name, value)

    def draw(self, layout):
        if not AppState.has_trainer.value:
            layout.text_colored("No trainer loaded", COLOR_IDLE)
            return

        params = lf.optimization_params()
        if not params.has_params():
            layout.text_colored("Parameters not available", COLOR_IDLE)
            return

        state = AppState.trainer_state.value
        iteration = AppState.iteration.value

        self._draw_controls(layout, state, iteration)
        layout.separator()

        if layout.collapsing_header(tr("training.section.basic_params") + "##py", default_open=True):
            self._draw_basic_params(layout, state, iteration, params)

        if layout.collapsing_header(tr("training.section.advanced_params") + "##py", default_open=False):
            self._draw_advanced_params(layout, state, iteration, params)

        self._draw_status(layout, state, iteration)

    def _draw_controls(self, layout, state, iteration):
        if state == "ready":
            label = tr("training_panel.resume_training") if iteration > 0 else tr("training_panel.start_training")
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
                        tr("training.error.adc_gut_title"),
                        tr("training.conflict.adc_gut_start_message"),
                        [btn_mcmc, btn_gut, btn_cancel],
                        _on_start_conflict)
                else:
                    lf.start_training()
            if iteration > 0:
                if layout.button_styled(tr("training_panel.reset"), "secondary", FULL_WIDTH):
                    lf.reset_training()
            if layout.button_styled(tr("training_panel.clear"), "error", FULL_WIDTH):
                lf.clear_scene()

        elif state == "running":
            if layout.button_styled(tr("training_panel.pause"), "warning", FULL_WIDTH):
                lf.pause_training()

        elif state == "paused":
            if layout.button_styled(tr("training_panel.resume"), "success", FULL_WIDTH):
                lf.resume_training()
            if layout.button_styled(tr("training_panel.reset"), "secondary", FULL_WIDTH):
                lf.reset_training()
            if layout.button_styled(tr("training_panel.stop"), "error", FULL_WIDTH):
                lf.stop_training()

        elif state in ("completed", "stopped"):
            if state == "completed":
                layout.text_colored(tr("status.complete"), COLOR_SUCCESS)
            else:
                layout.text_colored(tr("status.stopped"), COLOR_MUTED)
            if layout.button_styled(tr("training_panel.switch_edit_mode"), "success", FULL_WIDTH):
                lf.switch_to_edit_mode()
            if layout.button_styled(tr("training_panel.reset"), "secondary", FULL_WIDTH):
                lf.reset_training()
            if layout.button_styled(tr("training_panel.clear"), "error", FULL_WIDTH):
                lf.clear_scene()

        elif state == "error":
            layout.text_colored(tr("status.error"), COLOR_ERROR)
            if error_msg := lf.trainer_error():
                layout.text_wrapped(error_msg)
            if layout.button_styled(tr("training_panel.reset"), "secondary", FULL_WIDTH):
                lf.reset_training()
            if layout.button_styled(tr("training_panel.clear"), "error", FULL_WIDTH):
                lf.clear_scene()

        elif state == "stopping":
            layout.text_colored(tr("status.stopping"), COLOR_MUTED)

        if state in ("running", "paused"):
            if layout.button_styled(tr("training_panel.save_checkpoint"), "primary", FULL_WIDTH):
                lf.save_checkpoint()
                self._checkpoint_saved_time = time.time()

            if time.time() - self._checkpoint_saved_time < 2.0:
                theme = lf.ui.theme()
                layout.text_colored(tr("training_panel.checkpoint_saved"), theme.palette.success)

    def _draw_basic_params(self, layout, state, iteration, params):
        can_edit = (state == "ready") and (iteration == 0)
        can_edit_live = state in ("ready", "running", "paused")

        if layout.begin_table("PyBasicParamsTable", 2):
            layout.table_setup_column("Label", 120.0)
            layout.table_setup_column("Control", 0.0)

            # -- Structural params (only before training starts) --
            layout.begin_disabled(not can_edit)

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.strategy"))
            layout.table_next_column()
            layout.push_item_width(-1)
            strategy_idx = 0 if params.strategy == "mcmc" else 1
            changed, new_idx = layout.combo("##py_strategy", strategy_idx, STRATEGY_ITEMS)
            if changed:
                if new_idx == 1 and params.gut:
                    btn_gut = tr("training.conflict.btn_disable_gut")
                    btn_cancel = tr("training.conflict.btn_cancel")
                    def _on_strategy_conflict(button, _gut=btn_gut):
                        p = lf.optimization_params()
                        if button == _gut:
                            p.gut = False
                            p.set_strategy("adc")
                    lf.ui.confirm_dialog(
                        tr("training.error.adc_gut_title"),
                        tr("training.conflict.adc_gut_strategy_message"),
                        [btn_gut, btn_cancel],
                        _on_strategy_conflict)
                else:
                    params.set_strategy("mcmc" if new_idx == 0 else "adc")
            layout.pop_item_width()
            if layout.is_item_hovered():
                tooltip = tr("training.tooltip.strategy_gut_conflict") if params.gut else tr("training.tooltip.strategy")
                layout.set_tooltip(tooltip)

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.iterations"))
            layout.table_next_column()
            layout.push_item_width(-1)
            changed, new_val = layout.input_int_formatted("##py_iterations", int(params.iterations), 1000, 5000)
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
            changed, new_val = layout.input_int_formatted("##py_max_cap", params.max_cap, 10000, 100000)
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
            changed, new_idx = layout.combo("##py_sh_degree", params.sh_degree, SH_DEGREE_ITEMS)
            if changed:
                params.sh_degree = new_idx
            layout.pop_item_width()
            if layout.is_item_hovered():
                layout.set_tooltip(tr("training.tooltip.sh_degree"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.tile_mode"))
            layout.table_next_column()
            layout.push_item_width(-1)
            tile_idx = {1: 0, 2: 1, 4: 2}.get(params.tile_mode, 0)
            changed, new_idx = layout.combo("##py_tile_mode", tile_idx, TILE_MODE_ITEMS)
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
            changed, new_val = layout.input_float("##py_steps_scaler", params.steps_scaler, 0.1, 0.5, "%.2f")
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
            changed, new_val = layout.checkbox("##py_bilateral_grid", params.use_bilateral_grid)
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
            changed, new_idx = layout.combo("##py_mask_mode", mask_idx, MASK_MODE_ITEMS)
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
                changed, new_val = layout.checkbox("##py_invert_masks", params.invert_masks)
                if changed:
                    params.invert_masks = new_val
                if layout.is_item_hovered():
                    layout.set_tooltip(tr("training.tooltip.invert_masks"))

                layout.table_next_row()
                layout.table_next_column()
                layout.label(tr("training_params.use_alpha_as_mask"))
                layout.table_next_column()
                changed, new_val = layout.checkbox("##py_use_alpha_as_mask", params.use_alpha_as_mask)
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
            gut_disabled = (params.strategy == "adc")
            if gut_disabled:
                layout.begin_disabled(True)
            changed, new_val = layout.checkbox("##py_gut", params.gut)
            if changed:
                params.gut = new_val
                self._sync_render_setting("gut", new_val)
            if gut_disabled:
                layout.end_disabled()
            if layout.is_item_hovered():
                tooltip = tr("training.tooltip.gut_adc_conflict") if gut_disabled else tr("training.tooltip.gut")
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
                changed, new_val = layout.checkbox("##py_ppisp_controller", params.ppisp_use_controller)
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
                    changed, new_auto = layout.checkbox("Auto##py_ppisp_auto_step", is_auto)
                    if changed:
                        params.ppisp_controller_activation_step = -1 if new_auto else max(1, int(params.iterations) - 5000)
                    if not is_auto:
                        layout.same_line()
                        layout.push_item_width(-1)
                        changed, new_val = layout.input_int_formatted("##py_ppisp_ctrl_step", params.ppisp_controller_activation_step, 1000, 5000)
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
                    changed, new_val = layout.input_float("##py_ppisp_ctrl_lr", params.ppisp_controller_lr, 0.0001, 0.001, "%.5f")
                    if changed:
                        params.ppisp_controller_lr = new_val
                    layout.pop_item_width()
                    if layout.is_item_hovered():
                        layout.set_tooltip(tr("training.tooltip.ppisp_controller_lr"))

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.label(tr("training_params.ppisp_freeze_gaussians"))
                    layout.table_next_column()
                    changed, new_val = layout.checkbox("##py_ppisp_freeze", params.ppisp_freeze_gaussians)
                    if changed:
                        params.ppisp_freeze_gaussians = new_val
                    if layout.is_item_hovered():
                        layout.set_tooltip(tr("training.tooltip.ppisp_freeze_gaussians"))

            layout.table_next_row()
            layout.table_next_column()
            layout.label(tr("training_params.bg_mode"))
            layout.table_next_column()
            layout.push_item_width(-1)
            bg_idx = params.bg_mode.value
            changed, new_idx = layout.combo("##py_bg_mode", bg_idx, BG_MODE_ITEMS)
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
                changed, new_color = layout.color_edit3("##py_bg_color", params.bg_color)
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
                display = os.path.basename(img_path) if img_path else "(none)"
                layout.label(display)
                layout.pop_item_width()

                layout.table_next_row()
                layout.table_next_column()
                layout.table_next_column()
                if layout.button(tr("training_params.bg_image_browse") + "##py_bg_browse"):
                    selected = lf.ui.open_image_file_dialog("")
                    if selected:
                        params.bg_image_path = selected
                layout.same_line()
                if img_path and layout.button(tr("training_params.bg_image_clear") + "##py_bg_clear"):
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
                        layout.table_setup_column("Label", 120.0)
                        layout.table_setup_column("Control", 0.0)

                        data_path = dataset.data_path
                        self._table_text(layout, tr("training.dataset.path"), os.path.basename(data_path) if data_path else "(none)")

                        images = dataset.images
                        self._table_text(layout, tr("training.dataset.images"), images if images else "(default)")

                        layout.table_next_row()
                        layout.table_next_column()
                        layout.label(tr("training.dataset.resize_factor"))
                        layout.table_next_column()
                        if dataset_can_edit:
                            layout.push_item_width(-1)
                            resize_options = [-1, 1, 2, 4, 8]
                            resize_labels = ["Auto", "1", "2", "4", "8"]
                            current_idx = resize_options.index(dataset.resize_factor) if dataset.resize_factor in resize_options else 0
                            changed, new_idx = layout.combo("##py_resize_factor", current_idx, resize_labels)
                            if changed:
                                dataset.resize_factor = resize_options[new_idx]
                            layout.pop_item_width()
                        else:
                            layout.label("Auto" if dataset.resize_factor < 0 else str(dataset.resize_factor))

                        layout.table_next_row()
                        layout.table_next_column()
                        layout.label(tr("training.dataset.max_width"))
                        layout.table_next_column()
                        if dataset_can_edit:
                            layout.push_item_width(-1)
                            changed, new_val = layout.input_int("##py_max_width", dataset.max_width, 80, 400)
                            if changed and 0 < new_val <= 4096:
                                dataset.max_width = new_val
                            layout.pop_item_width()
                        else:
                            layout.label(str(dataset.max_width))

                        layout.table_next_row()
                        layout.table_next_column()
                        layout.label(tr("training.dataset.cpu_cache"))
                        layout.table_next_column()
                        if dataset_can_edit:
                            changed, new_val = layout.checkbox("##py_cpu_cache", dataset.use_cpu_cache)
                            if changed:
                                dataset.use_cpu_cache = new_val
                        else:
                            layout.label(tr("training.status.enabled") if dataset.use_cpu_cache else tr("training.status.disabled"))

                        layout.table_next_row()
                        layout.table_next_column()
                        layout.label(tr("training.dataset.fs_cache"))
                        layout.table_next_column()
                        if dataset_can_edit:
                            changed, new_val = layout.checkbox("##py_fs_cache", dataset.use_fs_cache)
                            if changed:
                                dataset.use_fs_cache = new_val
                        else:
                            layout.label(tr("training.status.enabled") if dataset.use_fs_cache else tr("training.status.disabled"))

                        out_path = dataset.output_path
                        self._table_text(layout, tr("training.dataset.output"),
                                       os.path.basename(out_path) if out_path else "(not set)")
                else:
                    layout.label("No dataset loaded")
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if layout.tree_node(tr("training.section.optimization") + "##py"):
            table_open = False
            try:
                table_open = layout.begin_table("PyOptTable", 2)
                if table_open:
                    layout.table_setup_column("Label", 120.0)
                    layout.table_setup_column("Control", 0.0)

                    layout.begin_disabled(not can_edit)
                    self._table_text(layout, tr("training_params.strategy"), params.strategy.upper())

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.text_colored(tr("training.opt.learning_rates"), (0.6, 0.6, 0.6, 1.0))
                    layout.table_next_column()

                    self._input_float_row(layout, tr("training.opt.lr.position"), "means_lr", params, params.means_lr, 0.000001, 0.00001, "%.6f")
                    self._input_float_row(layout, tr("training.opt.lr.sh_coeff"), "shs_lr", params, params.shs_lr, 0.0001, 0.001, "%.4f")
                    self._input_float_row(layout, tr("training.opt.lr.opacity"), "opacity_lr", params, params.opacity_lr, 0.001, 0.01, "%.4f")
                    self._input_float_row(layout, tr("training.opt.lr.scaling"), "scaling_lr", params, params.scaling_lr, 0.0001, 0.001, "%.4f")
                    self._input_float_row(layout, tr("training.opt.lr.rotation"), "rotation_lr", params, params.rotation_lr, 0.0001, 0.001, "%.4f")

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.text_colored(tr("training.section.refinement"), (0.6, 0.6, 0.6, 1.0))
                    layout.table_next_column()

                    self._input_int_row(layout, tr("training.refinement.refine_every"), "refine_every", params, 10, 100)
                    self._input_int_row(layout, tr("training.refinement.start_refine"), "start_refine", params, 100, 500)
                    self._input_int_row(layout, tr("training.refinement.stop_refine"), "stop_refine", params, 1000, 5000)
                    self._input_float_prop_row(layout, tr("training.refinement.gradient_thr"), "grad_threshold", params, 0.00001, 0.0001, "%.6f")
                    self._input_int_row(layout, tr("training.refinement.reset_every"), "reset_every", params, 100, 1000)
                    self._input_int_row(layout, tr("training.refinement.sh_upgrade_every"), "sh_degree_interval", params, 100, 500)
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if params.use_bilateral_grid and layout.tree_node(tr("training.section.bilateral_grid") + "##py"):
            table_open = False
            try:
                table_open = layout.begin_table("PyBilateralTable", 2)
                if table_open:
                    layout.table_setup_column("Label", 140.0)
                    layout.table_setup_column("Control", 0.0)
                    layout.begin_disabled(not can_edit)
                    self._table_prop(layout, params, "bilateral_grid_x", tr("training.bilateral.grid_x"))
                    self._table_prop(layout, params, "bilateral_grid_y", tr("training.bilateral.grid_y"))
                    self._table_prop(layout, params, "bilateral_grid_w", tr("training.bilateral.grid_w"))
                    self._table_prop(layout, params, "bilateral_grid_lr", tr("training.bilateral.learning_rate"))
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
                    layout.table_setup_column("Label", 140.0)
                    layout.table_setup_column("Control", 0.0)
                    layout.begin_disabled(not can_edit)
                    self._slider_float_row(layout, tr("training.losses.lambda_dssim"), "lambda_dssim", params, 0.0, 1.0)
                    self._input_float_prop_row(layout, tr("training.losses.opacity_reg"), "opacity_reg", params, 0.001, 0.01, "%.4f")
                    self._input_float_prop_row(layout, tr("training.losses.scale_reg"), "scale_reg", params, 0.001, 0.01, "%.4f")
                    self._input_float_prop_row(layout, tr("training.losses.tv_loss_weight"), "tv_loss_weight", params, 1.0, 5.0, "%.1f")
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
                    layout.table_setup_column("Label", 140.0)
                    layout.table_setup_column("Control", 0.0)
                    layout.begin_disabled(not can_edit)
                    self._slider_float_row(layout, tr("training.init.init_opacity"), "init_opacity", params, 0.01, 1.0)
                    self._input_float_prop_row(layout, tr("training.init.init_scaling"), "init_scaling", params, 0.01, 0.1, "%.3f")

                    layout.table_next_row()
                    layout.table_next_column()
                    layout.label(tr("training.init.random_init"))
                    layout.table_next_column()
                    changed, new_val = layout.checkbox("##py_random", params.random)
                    if changed:
                        params.random = new_val

                    if params.random:
                        self._input_int_row(layout, tr("training.init.num_points"), "init_num_pts", params, 10000, 50000)
                        self._input_float_prop_row(layout, tr("training.init.extent"), "init_extent", params, 0.5, 1.0, "%.1f")
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if params.strategy == "adc" and layout.tree_node(tr("training_panel.pruning_growing") + "##py"):
            table_open = False
            try:
                table_open = layout.begin_table("PyADCTable", 2)
                if table_open:
                    layout.table_setup_column("Label", 140.0)
                    layout.table_setup_column("Control", 0.0)
                    layout.begin_disabled(not can_edit)
                    self._input_float_prop_row(layout, tr("training.thresholds.min_opacity"), "min_opacity", params, 0.001, 0.01, "%.4f", min_val=0.0)
                    self._input_float_prop_row(layout, tr("training.thresholds.prune_opacity"), "prune_opacity", params, 0.001, 0.01, "%.4f", min_val=0.0)
                    self._input_float_prop_row(layout, tr("training.thresholds.grow_scale_3d"), "grow_scale3d", params, 0.001, 0.01, "%.4f", min_val=0.0)
                    self._input_float_prop_row(layout, tr("training.thresholds.grow_scale_2d"), "grow_scale2d", params, 0.01, 0.05, "%.3f", min_val=0.0)
                    self._input_float_prop_row(layout, tr("training.thresholds.prune_scale_3d"), "prune_scale3d", params, 0.01, 0.1, "%.3f", min_val=0.0)
                    self._input_float_prop_row(layout, tr("training.thresholds.prune_scale_2d"), "prune_scale2d", params, 0.01, 0.1, "%.3f", min_val=0.0)
                    self._input_int_row(layout, tr("training.thresholds.pause_after_reset"), "pause_refine_after_reset", params, 100, 500)
                    self._table_prop(layout, params, "revised_opacity", tr("training.thresholds.revised_opacity"))
                    layout.end_disabled()
            finally:
                if table_open:
                    layout.end_table()
                layout.tree_pop()

        if params.enable_sparsity and layout.tree_node(tr("training_panel.sparsity") + "##py"):
            table_open = False
            try:
                table_open = layout.begin_table("PySparsityTable", 2)
                if table_open:
                    layout.table_setup_column("Label", 140.0)
                    layout.table_setup_column("Control", 0.0)
                    layout.begin_disabled(not can_edit)
                    self._input_int_row(layout, tr("training_params.sparsify_steps"), "sparsify_steps", params, 1000, 5000)
                    self._input_float_prop_row(layout, tr("training_params.init_rho"), "init_rho", params, 0.001, 0.01, "%.4f")
                    self._slider_float_row(layout, tr("training_params.prune_ratio"), "prune_ratio", params, 0.0, 1.0)
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
            _, self._new_save_step = layout.input_int_formatted("##py_new_step", self._new_save_step, 100, 1000)
            layout.same_line()
            if layout.button(tr("common.add") + "##py_add"):
                if self._new_save_step > 0:
                    params.add_save_step(self._new_save_step)

            layout.separator()

            for i, step in enumerate(steps):
                layout.push_id(f"py_step_{i}")
                layout.set_next_item_width(100)
                changed, new_val = layout.input_int_formatted("##step", step, 0, 0)
                if changed and new_val > 0 and new_val != step:
                    params.remove_save_step(step)
                    params.add_save_step(new_val)
                layout.same_line()
                if layout.button(tr("common.remove") + "##rm"):
                    params.remove_save_step(step)
                layout.pop_id()

            if not steps:
                layout.text_colored(tr("training_panel.no_save_steps"), theme.palette.text_dim)
        else:
            if steps:
                layout.label(", ".join(str(s) for s in steps))
            else:
                layout.text_colored(tr("training_panel.no_save_steps"), theme.palette.text_dim)

    def _input_int_row(self, layout, label, prop_id, params, step, step_fast):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.push_item_width(-1)
        current_val = params.get(prop_id)
        if current_val is None:
            current_val = 0
        changed, new_val = layout.input_int_formatted(f"##py_{prop_id}", int(current_val), step, step_fast)
        if changed and new_val >= 0:
            params.set(prop_id, new_val)
        layout.pop_item_width()

    def _input_float_prop_row(self, layout, label, prop_id, params, step, step_fast, fmt, min_val=None):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.push_item_width(-1)
        current_val = params.get(prop_id)
        if current_val is None:
            current_val = 0.0
        changed, new_val = layout.input_float(f"##py_{prop_id}", float(current_val), step, step_fast, fmt)
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
        changed, new_val = layout.slider_float(f"##py_{prop_id}", float(current_val), min_val, max_val)
        if changed:
            params.set(prop_id, new_val)
        layout.pop_item_width()

    def _input_float_row(self, layout, label, prop_id, params, value, step, step_fast, fmt):
        layout.table_next_row()
        layout.table_next_column()
        layout.label(label)
        layout.table_next_column()
        layout.push_item_width(-1)
        changed, new_val = layout.input_float(f"##py_{prop_id}", value, step, step_fast, fmt)
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
            "ready": tr("status.ready") if iteration == 0 else tr("training_panel.resume"),
            "running": tr("training_panel.running"),
            "paused": tr("status.paused"),
            "stopping": tr("status.stopping"),
            "completed": tr("status.complete"),
            "stopped": tr("status.stopped"),
            "error": tr("status.error"),
        }
        layout.label(f"{tr('status.mode')}: {state_labels.get(state, 'Unknown')}")

        _rate_tracker.add_sample(iteration)
        rate = _rate_tracker.get_rate()
        layout.label(f"{tr('status.iteration')} {iteration:,} ({rate:.1f} iters/sec)")
        layout.label(f"num Splats: {AppState.num_gaussians.value:,}")

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

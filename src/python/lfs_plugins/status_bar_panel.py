# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Status bar panel displaying mode, training progress, GPU memory, and FPS."""

from .types import Panel

import lichtfeld as lf

_ACTIVE_TRAINING_STATES = ("running", "paused")


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.2f}M"
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


def _fmt_time(secs: float) -> str:
    if secs < 0:
        return "--:--"
    total = int(secs)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class StatusBarPanel(Panel):
    idname = "lfs.status_bar"
    label = "##StatusBar"
    space = "STATUS_BAR"
    order = 0

    def draw(self, layout):
        theme = lf.ui.theme()
        p = theme.palette

        mode, mode_color = self._get_mode_info(p)
        show_training = self._should_show_training()

        layout.text_colored(mode, mode_color)

        if show_training:
            self._draw_training_progress(layout, p)
        else:
            self._draw_splat_count(layout, p)

        self._draw_split_view(layout, p)
        self._draw_speed_overlays(layout, p)
        self._draw_right_section(layout, p)

    def _get_mode_info(self, p):
        content_type = lf.ui.get_content_type()

        if content_type == "empty":
            return "Empty", p.text_dim

        if content_type == "splat_files":
            return "Viewer", p.info

        state = lf.trainer_state()
        strategy = lf.trainer_strategy_type()
        gut = lf.trainer_is_gut_enabled()
        method = "GUT" if gut else "3DGS"
        strat_name = "MCMC" if strategy == "mcmc" else "Default"

        if state == "running":
            return f"Training ({strat_name}/{method})", p.warning
        if state == "paused":
            return f"Paused ({strat_name}/{method})", p.text_dim
        if state == "ready":
            current_iter = lf.trainer_current_iteration()
            base = "Resume" if current_iter > 0 else "Ready"
            return f"{base} ({strat_name}/{method})", p.success
        if state == "completed":
            return f"Complete ({strat_name}/{method})", p.success
        if state == "stopped":
            return f"Stopped ({strat_name}/{method})", p.text_dim
        if state == "error":
            return f"Error ({strat_name}/{method})", p.error

        return "Dataset", p.text_dim

    def _should_show_training(self) -> bool:
        content_type = lf.ui.get_content_type()
        if content_type != "dataset":
            return False
        state = lf.trainer_state()
        return state in _ACTIVE_TRAINING_STATES

    def _draw_training_progress(self, layout, p):
        layout.same_line(spacing=20)
        layout.text_colored("|", p.text_dim)
        layout.same_line(spacing=12)

        current_iter = lf.trainer_current_iteration()
        total_iter = lf.trainer_total_iterations()
        progress = current_iter / total_iter if total_iter > 0 else 0.0

        layout.progress_bar(progress, f"{progress * 100:.0f}%", 80)

        layout.same_line(spacing=12)
        layout.text_colored("Step", p.text_dim)
        layout.same_line(spacing=6)
        layout.label(f"{current_iter}/{total_iter}")

        layout.same_line(spacing=12)
        layout.text_colored("|", p.text_dim)
        layout.same_line(spacing=12)

        loss = lf.trainer_current_loss()
        layout.text_colored("Loss", p.text_dim)
        layout.same_line(spacing=6)
        layout.label(f"{loss:.4f}")

        layout.same_line(spacing=12)
        layout.text_colored("|", p.text_dim)
        layout.same_line(spacing=12)

        num_splats = lf.trainer_num_splats()
        max_gaussians = lf.trainer_max_gaussians()
        layout.text_colored("Gaussians", p.text_dim)
        layout.same_line(spacing=6)
        layout.label(f"{_fmt_count(num_splats)}/{_fmt_count(max_gaussians)}")

        layout.same_line(spacing=12)
        layout.text_colored("|", p.text_dim)
        layout.same_line(spacing=12)

        elapsed = lf.trainer_elapsed_seconds()
        eta = lf.trainer_eta_seconds()
        layout.label(_fmt_time(elapsed))
        layout.same_line(spacing=6)
        layout.text_colored("ETA", p.text_dim)
        layout.same_line(spacing=6)
        layout.label(_fmt_time(eta))

    def _draw_splat_count(self, layout, p):
        if lf.ui.get_content_type() == "empty":
            return

        total = lf.get_num_gaussians()
        if total == 0:
            return

        layout.same_line(spacing=20)
        layout.text_colored("|", p.text_dim)
        layout.same_line()
        layout.text_colored(f"{_fmt_count(total)} Gaussians", p.text)

    def _draw_split_view(self, layout, p):
        info = lf.ui.get_split_view_info()
        if not info.get("enabled"):
            return

        layout.same_line(spacing=20)
        layout.text_colored("|", p.text_dim)
        layout.same_line()

        mode = lf.ui.get_split_view_mode()
        if mode == "gt_comparison":
            cam_id = lf.ui.get_current_camera_id()
            layout.text_colored("GT Compare", p.warning)
            layout.same_line(spacing=4)
            layout.text_colored(f"Cam {cam_id}", p.text_dim)
        elif mode == "ply_comparison":
            left = info.get("left_name", "")
            right = info.get("right_name", "")
            layout.text_colored("Split:", p.warning)
            layout.same_line(spacing=4)
            layout.label(f"{left} | {right}")

    def _draw_speed_overlays(self, layout, p):
        wasd_speed, wasd_alpha, zoom_speed, zoom_alpha = lf.ui.get_speed_overlay()

        if wasd_alpha > 0:
            sep_color = (p.text_dim[0], p.text_dim[1], p.text_dim[2], wasd_alpha)
            speed_color = (p.info[0], p.info[1], p.info[2], wasd_alpha)
            layout.same_line(spacing=20)
            layout.text_colored("|", sep_color)
            layout.same_line()
            layout.text_colored(f"WASD: {wasd_speed:.0f}", speed_color)

        if zoom_alpha > 0:
            sep_color = (p.text_dim[0], p.text_dim[1], p.text_dim[2], zoom_alpha)
            speed_color = (p.info[0], p.info[1], p.info[2], zoom_alpha)
            layout.same_line(spacing=20)
            layout.text_colored("|", sep_color)
            layout.same_line()
            layout.text_colored(f"Zoom: {zoom_speed * 10:.0f}", speed_color)

    def _draw_right_section(self, layout, p):
        used, total = lf.ui.get_gpu_memory()
        used_gb = used / 1e9
        total_gb = total / 1e9
        pct = (used_gb / total_gb) * 100 if total_gb > 0 else 0

        if pct < 50:
            mem_color = p.success
        elif pct < 75:
            mem_color = p.warning
        else:
            mem_color = p.error

        fps = lf.ui.get_fps()
        if fps >= 30:
            fps_color = p.success
        elif fps >= 15:
            fps_color = p.warning
        else:
            fps_color = p.error

        git_commit = lf.ui.get_git_commit()

        mem_text = f"{used_gb:.1f}/{total_gb:.1f}GB"
        fps_text = f"{fps:.0f}"

        right_width = 240
        window_w = layout.get_window_width()
        target_x = window_w - right_width

        layout.same_line()
        cur_x, _ = layout.get_cursor_pos()
        if target_x > cur_x:
            layout.set_cursor_pos_x(target_x)

        layout.text_colored("GPU ", p.text_dim)
        layout.same_line(spacing=0)
        layout.text_colored(mem_text, mem_color)

        layout.same_line(spacing=16)
        layout.text_colored(fps_text, fps_color)
        layout.same_line(spacing=0)
        layout.text_colored(" FPS", p.text_dim)

        layout.same_line(spacing=16)
        layout.text_colored(git_commit, p.text_dim)

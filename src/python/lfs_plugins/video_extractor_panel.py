# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Retained RmlUI panel for video frame extraction."""

from __future__ import annotations

import os
import threading

import lichtfeld as lf

from .types import Panel


def _format_time(seconds: float) -> str:
    mins = int(seconds) // 60
    secs = seconds - mins * 60
    return f"{mins}:{secs:05.2f}"


SCALE_VALUES = (0.125, 0.25, 0.5, 1.0, 2.0)


class VideoExtractorPanel(Panel):
    """Floating retained panel for video frame extraction."""

    id = "lfs.video_extractor"
    label = "Video Extractor"
    space = lf.ui.PanelSpace.FLOATING
    order = 11
    template = "rmlui/video_extractor_panel.rml"
    height_mode = lf.ui.PanelHeightMode.CONTENT
    size = (750, 0)
    update_interval_ms = 16

    def __init__(self):
        self._handle = None
        self._video_path = ""
        self._output_path = ""
        self._has_video = False
        self._is_playing = False

        # Timeline / trim
        self._duration = 0.0
        self._current_time = 0.0
        self._trim_start = 0.0
        self._trim_end = 0.0

        # Extraction settings
        self._mode_index = 0  # 0=FPS, 1=INTERVAL
        self._fps_value = 1
        self._interval_value = 1

        # Format settings
        self._format_index = 0  # 0=PNG, 1=JPG
        self._jpg_quality = 95

        # Resolution settings
        self._resolution_mode = 0  # 0=Original, 1=Scale, 2=Custom
        self._scale_index = 3  # 1x
        self._custom_width = 1920
        self._custom_height = 1080

        # Output naming
        self._filename_pattern = "frame_%d"

        # Extraction state
        self._is_extracting = False
        self._progress = 0.0
        self._progress_current = 0
        self._progress_total = 0
        self._show_completion = False
        self._show_error = False
        self._error_text = ""

        # Video player handle
        self._player_open = False
        self._extraction_thread = None

    # ── Data model ────────────────────────────────────────────

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("video_extractor")
        if model is None:
            return

        model.bind_func("panel_label", lambda: lf.ui.tr("video_extractor.title"))
        model.bind_func("show_placeholder", lambda: not self._has_video)
        model.bind_func("has_video", lambda: self._has_video)

        # Transport
        model.bind_func("play_pause_label", lambda: "\u23F8" if self._is_playing else "\u25B6")
        model.bind_func("time_display", self._get_time_display)

        # Timeline
        model.bind("timeline_pos", self._get_timeline_pos, self._set_timeline_pos)
        model.bind("trim_start", self._get_trim_start_slider, self._set_trim_start_slider)
        model.bind("trim_end", self._get_trim_end_slider, self._set_trim_end_slider)
        model.bind_func("trim_start_text", lambda: _format_time(self._trim_start))
        model.bind_func("trim_end_text", lambda: _format_time(self._trim_end))

        # File paths
        model.bind_func("video_path", lambda: self._video_path or "—")
        model.bind_func("output_path", lambda: self._output_path or "—")

        # Extraction settings
        model.bind("mode_index", lambda: str(self._mode_index), self._set_mode_index)
        model.bind_func("show_fps", lambda: self._mode_index == 0)
        model.bind_func("show_interval", lambda: self._mode_index == 1)
        model.bind("fps_value", lambda: str(self._fps_value), self._set_fps_value)
        model.bind_func("fps_text", lambda: str(self._fps_value))
        model.bind("interval_value", lambda: str(self._interval_value), self._set_interval_value)
        model.bind_func("interval_text", lambda: str(self._interval_value))
        model.bind_func("estimated_frames_text", self._get_estimated_frames_text)

        # Format settings
        model.bind("format_index", lambda: str(self._format_index), self._set_format_index)
        model.bind_func("show_quality", lambda: self._format_index == 1)
        model.bind("jpg_quality", lambda: str(self._jpg_quality), self._set_jpg_quality)
        model.bind_func("quality_text", lambda: str(self._jpg_quality))

        # Resolution settings
        model.bind("resolution_mode", lambda: str(self._resolution_mode), self._set_resolution_mode)
        model.bind_func("show_scale", lambda: self._resolution_mode == 1)
        model.bind_func("show_custom_res", lambda: self._resolution_mode == 2)
        model.bind("scale_index", lambda: str(self._scale_index), self._set_scale_index)
        model.bind("custom_width", lambda: str(self._custom_width), self._set_custom_width)
        model.bind("custom_height", lambda: str(self._custom_height), self._set_custom_height)

        # Output naming
        model.bind("filename_pattern", lambda: self._filename_pattern, self._set_filename_pattern)
        model.bind_func("pattern_example", self._get_pattern_example)

        # Action / status
        model.bind_func("show_warning", lambda: bool(self._get_warning_text()))
        model.bind_func("warning_text", self._get_warning_text)
        model.bind_func("show_actions", lambda: not self._is_extracting)
        model.bind_func("can_extract", self._can_extract)
        model.bind_func("is_extracting", lambda: self._is_extracting)
        model.bind_func("progress_value", lambda: f"{self._progress:.4f}")
        model.bind_func("progress_pct", lambda: f"{self._progress * 100:.0f}%")
        model.bind_func("progress_stage", lambda: f"{self._progress_current}/{self._progress_total}")
        model.bind_func("show_completion", lambda: self._show_completion)
        model.bind_func("show_error", lambda: self._show_error)
        model.bind_func("error_text", lambda: self._error_text)

        # Events
        model.bind_event("do_step_back", self._on_step_back)
        model.bind_event("do_play_pause", self._on_play_pause)
        model.bind_event("do_step_fwd", self._on_step_fwd)
        model.bind_event("do_browse_video", self._on_browse_video)
        model.bind_event("do_browse_output", self._on_browse_output)
        model.bind_event("do_start_extract", self._on_start_extract)
        model.bind_event("do_cancel", self._on_cancel)
        model.bind_event("do_cancel_extract", self._on_cancel_extract)

        self._handle = model.get_handle()

    # ── Lifecycle ─────────────────────────────────────────────

    def on_mount(self, doc):
        super().on_mount(doc)
        self._show_completion = False
        self._show_error = False
        self._error_text = ""

    def on_update(self, doc):
        dirty = False

        if self._has_video and self._player_open:
            try:
                updated = lf.io.video_player_update()
                if updated:
                    # Feed frame data to the preview element
                    preview = doc.get_element_by_id("video-preview")
                    if preview is not None:
                        lf.io.feed_preview_element(preview)
                    dirty = True

                current_time = lf.io.video_player_current_time()
                is_playing = lf.io.video_player_is_playing()

                if current_time != self._current_time or is_playing != self._is_playing:
                    self._current_time = current_time
                    self._is_playing = is_playing
                    self._dirty_model(
                        "time_display", "timeline_pos", "play_pause_label"
                    )
                    dirty = True
            except Exception:
                pass

        if self._is_extracting:
            try:
                state = lf.io.video_extract_state()
                progress_current = state.get("current", 0)
                progress_total = state.get("total", 0)
                is_active = state.get("active", False)
                error = state.get("error", "")

                if progress_total > 0:
                    self._progress = progress_current / progress_total
                else:
                    self._progress = 0.0
                self._progress_current = progress_current
                self._progress_total = progress_total

                if not is_active:
                    self._is_extracting = False
                    if error:
                        self._show_error = True
                        self._error_text = error
                    else:
                        self._show_completion = True
                    self._dirty_model()
                    dirty = True
                else:
                    self._dirty_model(
                        "progress_value", "progress_pct", "progress_stage"
                    )
                    dirty = True
            except Exception:
                pass

        return dirty

    def on_unmount(self, doc):
        self._close_video()
        doc.remove_data_model("video_extractor")
        self._handle = None

    # ── Helpers ──────────────────────────────────────────────

    def _dirty_model(self, *fields):
        if not self._handle:
            return
        if not fields:
            self._handle.dirty_all()
            return
        for field in fields:
            self._handle.dirty(field)

    def _get_time_display(self) -> str:
        return f"{_format_time(self._current_time)} / {_format_time(self._duration)}"

    def _get_timeline_pos(self) -> str:
        if self._duration <= 0:
            return "0"
        return str(int(self._current_time / self._duration * 1000))

    def _set_timeline_pos(self, value):
        try:
            pos = max(0, min(1000, int(float(value))))
        except (ValueError, TypeError):
            return
        if self._duration <= 0:
            return
        target_time = pos / 1000.0 * self._duration
        self._current_time = target_time
        try:
            lf.io.video_player_seek(target_time)
        except Exception:
            pass
        self._dirty_model("time_display", "timeline_pos")

    def _get_trim_start_slider(self) -> str:
        if self._duration <= 0:
            return "0"
        return str(int(self._trim_start / self._duration * 1000))

    def _set_trim_start_slider(self, value):
        try:
            pos = max(0, min(1000, int(float(value))))
        except (ValueError, TypeError):
            return
        self._trim_start = pos / 1000.0 * self._duration
        if self._trim_start > self._trim_end:
            self._trim_start = self._trim_end
        self._dirty_model("trim_start", "trim_start_text", "estimated_frames_text")

    def _get_trim_end_slider(self) -> str:
        if self._duration <= 0:
            return "1000"
        return str(int(self._trim_end / self._duration * 1000))

    def _set_trim_end_slider(self, value):
        try:
            pos = max(0, min(1000, int(float(value))))
        except (ValueError, TypeError):
            return
        self._trim_end = pos / 1000.0 * self._duration
        if self._trim_end < self._trim_start:
            self._trim_end = self._trim_start
        self._dirty_model("trim_end", "trim_end_text", "estimated_frames_text")

    def _set_mode_index(self, value):
        try:
            idx = int(float(value))
        except (ValueError, TypeError):
            return
        if idx == self._mode_index:
            return
        self._mode_index = idx
        self._dirty_model("mode_index", "show_fps", "show_interval", "estimated_frames_text")

    def _set_fps_value(self, value):
        try:
            v = max(1, min(60, int(float(value))))
        except (ValueError, TypeError):
            return
        if v == self._fps_value:
            return
        self._fps_value = v
        self._dirty_model("fps_value", "fps_text", "estimated_frames_text")

    def _set_interval_value(self, value):
        try:
            v = max(1, min(100, int(float(value))))
        except (ValueError, TypeError):
            return
        if v == self._interval_value:
            return
        self._interval_value = v
        self._dirty_model("interval_value", "interval_text", "estimated_frames_text")

    def _set_format_index(self, value):
        try:
            idx = int(float(value))
        except (ValueError, TypeError):
            return
        if idx == self._format_index:
            return
        self._format_index = idx
        self._dirty_model("format_index", "show_quality")

    def _set_jpg_quality(self, value):
        try:
            v = max(1, min(100, int(float(value))))
        except (ValueError, TypeError):
            return
        if v == self._jpg_quality:
            return
        self._jpg_quality = v
        self._dirty_model("jpg_quality", "quality_text")

    def _set_resolution_mode(self, value):
        try:
            idx = int(float(value))
        except (ValueError, TypeError):
            return
        if idx == self._resolution_mode:
            return
        self._resolution_mode = idx
        self._dirty_model("resolution_mode", "show_scale", "show_custom_res")

    def _set_scale_index(self, value):
        try:
            idx = int(float(value))
        except (ValueError, TypeError):
            return
        if idx == self._scale_index:
            return
        self._scale_index = idx
        self._dirty_model("scale_index")

    def _set_custom_width(self, value):
        try:
            v = max(1, int(float(value)))
        except (ValueError, TypeError):
            return
        self._custom_width = v
        self._dirty_model("custom_width")

    def _set_custom_height(self, value):
        try:
            v = max(1, int(float(value)))
        except (ValueError, TypeError):
            return
        self._custom_height = v
        self._dirty_model("custom_height")

    def _set_filename_pattern(self, value):
        self._filename_pattern = value or "frame_%d"
        self._dirty_model("filename_pattern", "pattern_example")

    def _get_pattern_example(self) -> str:
        ext = "png" if self._format_index == 0 else "jpg"
        try:
            name = self._filename_pattern.replace("%d", "0001")
        except Exception:
            name = "frame_0001"
        return f"e.g. {name}.{ext}"

    def _get_estimated_frames_text(self) -> str:
        if not self._has_video:
            return ""
        count = self._calculate_estimated_frames()
        return f"~{count} frames"

    def _calculate_estimated_frames(self) -> int:
        if not self._has_video or self._duration <= 0:
            return 0
        trim_duration = self._trim_end - self._trim_start
        if trim_duration <= 0:
            return 0
        if self._mode_index == 0:
            return max(1, int(trim_duration * self._fps_value))
        try:
            video_fps = lf.io.video_player_fps()
        except Exception:
            video_fps = 30.0
        if video_fps <= 0:
            return 0
        return max(1, int((trim_duration * video_fps) / self._interval_value))

    def _get_warning_text(self) -> str:
        if not self._video_path:
            return ""
        if not self._output_path:
            return lf.ui.tr("video_extractor.no_output_dir")
        return ""

    def _can_extract(self) -> bool:
        return bool(
            self._has_video
            and self._video_path
            and self._output_path
            and not self._is_extracting
        )

    def _close_video(self):
        if self._player_open:
            try:
                lf.io.video_player_close()
            except Exception:
                pass
            self._player_open = False

    # ── Event handlers ────────────────────────────────────────

    def _on_browse_video(self, _handle=None, _ev=None, _args=None):
        path = lf.ui.open_video_file_dialog()
        if not path:
            return

        self._video_path = path
        self._show_completion = False
        self._show_error = False
        self._error_text = ""

        try:
            lf.io.video_player_open(path)
            self._player_open = True
            self._has_video = True
            self._duration = lf.io.video_player_duration()
            self._trim_start = 0.0
            self._trim_end = self._duration
            self._current_time = 0.0
            self._is_playing = False

            if not self._output_path:
                base = os.path.splitext(os.path.basename(path))[0]
                parent = os.path.dirname(path)
                self._output_path = os.path.join(parent, f"{base}_frames")
        except Exception as e:
            self._has_video = False
            self._show_error = True
            self._error_text = str(e)

        self._dirty_model()

    def _on_browse_output(self, _handle=None, _ev=None, _args=None):
        path = lf.ui.open_folder_dialog()
        if path:
            self._output_path = path
            self._dirty_model("output_path", "show_warning", "warning_text", "can_extract")

    def _on_step_back(self, _handle=None, _ev=None, _args=None):
        if not self._has_video:
            return
        try:
            lf.io.video_player_step_backward()
        except Exception:
            pass

    def _on_play_pause(self, _handle=None, _ev=None, _args=None):
        if not self._has_video:
            return
        try:
            lf.io.video_player_toggle_play_pause()
        except Exception:
            pass

    def _on_step_fwd(self, _handle=None, _ev=None, _args=None):
        if not self._has_video:
            return
        try:
            lf.io.video_player_step_forward()
        except Exception:
            pass

    def _on_start_extract(self, _handle=None, _ev=None, _args=None):
        if not self._can_extract():
            return

        self._show_completion = False
        self._show_error = False
        self._error_text = ""
        self._is_extracting = True
        self._progress = 0.0
        self._progress_current = 0
        self._progress_total = 0

        scale = SCALE_VALUES[self._scale_index] if self._scale_index < len(SCALE_VALUES) else 1.0

        try:
            lf.io.video_extract_frames(
                video_path=self._video_path,
                output_dir=self._output_path,
                mode=self._mode_index,
                fps=float(self._fps_value),
                frame_interval=self._interval_value,
                format=self._format_index,
                jpg_quality=self._jpg_quality,
                start_time=self._trim_start,
                end_time=self._trim_end,
                resolution_mode=self._resolution_mode,
                scale=float(scale),
                custom_width=self._custom_width,
                custom_height=self._custom_height,
                filename_pattern=self._filename_pattern,
            )
        except Exception as e:
            self._is_extracting = False
            self._show_error = True
            self._error_text = str(e)

        self._dirty_model()

    def _on_cancel(self, _handle=None, _ev=None, _args=None):
        lf.ui.set_panel_enabled("lfs.video_extractor", False)

    def _on_cancel_extract(self, _handle=None, _ev=None, _args=None):
        # Extraction runs in a background thread; there's no clean
        # cancel mechanism in VideoFrameExtractor currently, so we
        # just mark as not extracting and let the thread finish.
        self._is_extracting = False
        self._dirty_model()

# SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Sequencer settings panel - camera animation and video export (RML)."""

import time

import lichtfeld as lf

from .sequencer_section import (
    PRESET_INFO, SPEED_VALUES, SNAP_VALUES, FPS_VALUES,
    VideoPreset, _video_state,
    MIN_WIDTH, MAX_WIDTH, MIN_HEIGHT, MAX_HEIGHT,
)
from .types import RmlPanel


class SequencerSettingsPanel(RmlPanel):
    idname = "lfs.sequencer_settings"
    label = "Sequencer"
    space = "MAIN_PANEL_TAB"
    order = 15
    rml_template = "rmlui/sequencer_settings.rml"
    rml_height_mode = "content"

    def __init__(self):
        self._handle = None
        self._collapsed = set()
        self._show_clear_modal = False
        self._last_has_keyframes = None
        self._step_repeat_prop = None
        self._step_repeat_dir = 0
        self._step_repeat_start = 0.0
        self._step_repeat_last = 0.0

    @classmethod
    def poll(cls, context=None):
        return lf.ui.is_sequencer_visible()

    def on_bind_model(self, ctx):
        model = ctx.create_data_model("sequencer_settings")
        if model is None:
            return

        vs = _video_state

        def _get_speed_idx():
            state = lf.ui.get_sequencer_state()
            if not state:
                return "2"
            for i, val in enumerate(SPEED_VALUES):
                if abs(state.playback_speed - val) < 0.01:
                    return str(i)
            return "2"

        def _set_speed_idx(v):
            state = lf.ui.get_sequencer_state()
            if state:
                idx = int(v)
                if 0 <= idx < len(SPEED_VALUES):
                    state.playback_speed = SPEED_VALUES[idx]
                    lf.ui.set_playback_speed(state.playback_speed)

        def _get_snap_idx():
            state = lf.ui.get_sequencer_state()
            if not state:
                return "1"
            for i, val in enumerate(SNAP_VALUES):
                if abs(state.snap_interval - val) < 0.01:
                    return str(i)
            return "1"

        def _set_snap_idx(v):
            state = lf.ui.get_sequencer_state()
            if state:
                idx = int(v)
                if 0 <= idx < len(SNAP_VALUES):
                    state.snap_interval = SNAP_VALUES[idx]

        def _seq_bool_getter(attr):
            state = lf.ui.get_sequencer_state()
            return getattr(state, attr, False) if state else False

        def _seq_bool_setter(attr, v):
            state = lf.ui.get_sequencer_state()
            if state:
                setattr(state, attr, v)

        def _seq_float_getter(attr, default=0.0):
            state = lf.ui.get_sequencer_state()
            return float(getattr(state, attr, default)) if state else default

        def _seq_float_setter(attr, v):
            state = lf.ui.get_sequencer_state()
            if state:
                setattr(state, attr, float(v))

        model.bind("show_camera_path",
                    lambda: _seq_bool_getter("show_camera_path"),
                    lambda v: _seq_bool_setter("show_camera_path", v))
        model.bind("snap_to_grid",
                    lambda: _seq_bool_getter("snap_to_grid"),
                    lambda v: _seq_bool_setter("snap_to_grid", v))
        model.bind("follow_playback",
                    lambda: _seq_bool_getter("follow_playback"),
                    lambda v: _seq_bool_setter("follow_playback", v))
        model.bind("pip_preview_scale",
                    lambda: _seq_float_getter("pip_preview_scale", 1.0),
                    lambda v: _seq_float_setter("pip_preview_scale", v))
        model.bind("speed_idx", _get_speed_idx, _set_speed_idx)
        model.bind("snap_idx", _get_snap_idx, _set_snap_idx)

        model.bind_func("has_keyframes", lambda: lf.ui.has_keyframes())
        model.bind_func("show_clear_modal", lambda: self._show_clear_modal)

        # Video export bindings
        model.bind("preset_idx",
                    lambda: str(vs.preset),
                    lambda v: self._set_preset(int(v)))
        model.bind_func("is_custom_preset",
                         lambda: vs.preset == int(VideoPreset.CUSTOM))
        model.bind("custom_width",
                    lambda: str(vs.custom_width),
                    lambda v: self._set_custom_dim("width", v))
        model.bind("custom_height",
                    lambda: str(vs.custom_height),
                    lambda v: self._set_custom_dim("height", v))
        model.bind("quality",
                    lambda: float(vs.quality),
                    lambda v: setattr(vs, "quality", int(float(v))))

        def _get_fps_idx():
            if vs.framerate == 24:
                return "0"
            elif vs.framerate == 60:
                return "2"
            return "1"

        def _set_fps_idx(v):
            idx = int(v)
            if 0 <= idx < len(FPS_VALUES):
                vs.framerate = FPS_VALUES[idx]

        model.bind("fps_idx", _get_fps_idx, _set_fps_idx)

        model.bind_func("preset_desc", lambda: PRESET_INFO.get(
            VideoPreset(vs.preset) if 0 <= vs.preset <= 7 else VideoPreset.YOUTUBE_1080P,
            {}).get("desc", ""))

        model.bind_func("label_sequencer", lambda: "Sequencer")

        for sec in ["seq_main", "video_export"]:
            model.bind(f"sec_{sec}_collapsed",
                       lambda n=sec: n in self._collapsed)
            model.bind_func(f"sec_{sec}_arrow",
                            lambda n=sec: "\u25B6" if n in self._collapsed else "\u25BC")

        model.bind_event("toggle_section", self._on_toggle_section)
        model.bind_event("save_path", self._on_save_path)
        model.bind_event("load_path", self._on_load_path)
        model.bind_event("clear_keyframes", self._on_clear_keyframes)
        model.bind_event("clear_confirm", self._on_clear_confirm)
        model.bind_event("clear_cancel", self._on_clear_cancel)
        model.bind_event("export_video", self._on_export_video)
        model.bind_event("num_step", self._on_num_step)

        self._handle = model.get_handle()

    def on_load(self, doc):
        body = doc.get_element_by_id("body")
        if body:
            body.add_event_listener("mouseup", self._on_step_mouseup)

    def on_update(self, doc):
        self._update_step_repeat()
        if self._handle:
            hk = lf.ui.has_keyframes()
            if hk != self._last_has_keyframes:
                self._last_has_keyframes = hk
                self._handle.dirty("has_keyframes")

    def on_scene_changed(self, doc):
        if self._handle:
            self._handle.dirty_all()

    def on_unload(self, doc):
        doc.remove_data_model("sequencer_settings")
        self._handle = None

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

    def _apply_num_step(self, prop, direction):
        vs = _video_state
        if prop == "custom_width":
            vs.custom_width = max(MIN_WIDTH, min(MAX_WIDTH, vs.custom_width + 16 * direction))
            if self._handle:
                self._handle.dirty("custom_width")
        elif prop == "custom_height":
            vs.custom_height = max(MIN_HEIGHT, min(MAX_HEIGHT, vs.custom_height + 16 * direction))
            if self._handle:
                self._handle.dirty("custom_height")

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

    def _set_preset(self, idx):
        vs = _video_state
        vs.preset = idx
        preset = VideoPreset(idx) if 0 <= idx <= 7 else VideoPreset.YOUTUBE_1080P
        if preset != VideoPreset.CUSTOM:
            info = PRESET_INFO[preset]
            vs.framerate = info["fps"]
            vs.quality = info["crf"]
        if self._handle:
            self._handle.dirty_all()

    def _set_custom_dim(self, dim, val):
        vs = _video_state
        try:
            v = int(val)
        except (ValueError, TypeError):
            return
        if dim == "width":
            vs.custom_width = max(MIN_WIDTH, min(MAX_WIDTH, v))
        else:
            vs.custom_height = max(MIN_HEIGHT, min(MAX_HEIGHT, v))

    def _on_toggle_section(self, handle, event, args):
        if not args:
            return
        name = str(args[0])
        if name in self._collapsed:
            self._collapsed.discard(name)
        else:
            self._collapsed.add(name)
        handle.dirty(f"sec_{name}_collapsed")
        handle.dirty(f"sec_{name}_arrow")

    def _on_save_path(self, handle, event, args):
        if not lf.ui.has_keyframes():
            return
        path = lf.ui.save_json_file_dialog("camera_path")
        if path:
            if lf.ui.save_camera_path(path):
                lf.log_info(f"Camera path saved to {path}")
            else:
                lf.log_error(f"Failed to save camera path to {path}")

    def _on_load_path(self, handle, event, args):
        path = lf.ui.open_json_file_dialog()
        if path:
            if lf.ui.load_camera_path(path):
                lf.log_info(f"Camera path loaded from {path}")
            else:
                lf.log_error(f"Failed to load camera path from {path}")

    def _on_clear_keyframes(self, handle, event, args):
        if not lf.ui.has_keyframes():
            return
        self._show_clear_modal = True
        handle.dirty("show_clear_modal")

    def _on_clear_confirm(self, handle, event, args):
        lf.ui.clear_keyframes()
        self._show_clear_modal = False
        handle.dirty("show_clear_modal")
        handle.dirty("has_keyframes")

    def _on_clear_cancel(self, handle, event, args):
        self._show_clear_modal = False
        handle.dirty("show_clear_modal")

    def _on_export_video(self, handle, event, args):
        if not lf.ui.has_keyframes():
            return
        vs = _video_state
        preset = VideoPreset(vs.preset) if 0 <= vs.preset <= 7 else VideoPreset.YOUTUBE_1080P
        info = PRESET_INFO[preset]
        width = vs.custom_width if preset == VideoPreset.CUSTOM else info["width"]
        height = vs.custom_height if preset == VideoPreset.CUSTOM else info["height"]
        lf.ui.export_video(width, height, vs.framerate, vs.quality)

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for viewport selection controls."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from xml.etree import ElementTree
import sys

import pytest


def _install_lf_stub(monkeypatch):
    state = SimpleNamespace(
        active_tool="builtin.select",
        active_submode="rectangle",
        has_scene=True,
        has_selection=True,
        depth_enabled=False,
        depth_near=0.25,
        depth_far=7.5,
        depth_read_error=False,
        depth_width=1.35,
        depth_calls=[],
        # Screen-space window state. The defaults match the controller's own so
        # the first refresh does not look like an external change.
        depth_scale=0.35,
        depth_scale_y=0.35,
        depth_offset_x=0.0,
        depth_offset_y=0.0,
        window_calls=[],
        # Set to make the native window write raise, as a rejected or failed
        # native call would. _apply_depth_window catches and reports it.
        window_write_error=False,
        stage_calls=[],
        undo_available=True,
        redo_available=True,
        undo_calls=0,
        redo_calls=0,
        # Optional callables run by lf.undo.undo()/redo(), standing in for a
        # native undo entry that restores state -- DepthWindowSyncUndoEntry
        # restores the sync flag itself (depth_window_undo_entry.cpp:85).
        undo_effect=None,
        redo_effect=None,
        # Plan-03 split-panel surface. The defaults describe a single viewport,
        # so every pre-existing test sees exactly today's behavior: the chip and
        # the sync toggle are hidden and one shared Size reference is in use.
        split_view_mode="none",
        focused_panel="left",
        depth_sync=False,
        # Set to make set_depth_window_sync a silent no-op, the way the manager
        # ignores the toggle while a depth-window drag is in flight.
        sync_write_ignored=False,
        sync_calls=[],
        # Model the split service's leave: record the native collapse source before
        # resetting observable focus to Left. The plugin then knows which window
        # survived (applyDepthWindowModeTransitionLocked).
        depth_window_collapse_source="left",
        # Model stamped lineage writes: leave collapse, sync copy, project/sync restore
        # and retained-pair discard. Each stamp updates the triple under one lock; sync
        # undo/redo stamps separately from its slot restore. Keep the consumer's cached
        # generation behind to model changes between polls.
        depth_window_collapse_generation=0,
        # WHICH native write stamped the lineage record last
        # (rendering_manager.hpp, DepthWindowLineageKind): 'leave_collapse',
        # 'sync_copy' or 'project_restore'. The default matches the pre-lineage
        # behaviour of every test that only ever models a mode leave.
        depth_window_collapse_kind="leave_collapse",
        # Per-panel window scales for explicit-panel reads (panel='left'/'right').
        panel_scales={},
        # Per-panel near/far, for tests that need DISTINCT canonical values per
        # panel. Empty by default, so the no-panel read projects the global
        # near/far exactly as it always did and no pre-existing test moves.
        panel_ranges={},
        # Explicit-panel writes, in order: (panel, near, far, scale).
        panel_writes=[],
        # WHERE each write landed, in order: 'left', 'right', 'both' (the synced
        # fan-out) or 'global'. The no-panel setter resolves its destination the
        # way updateSettings back-routes -- focused slot when unsynced, both
        # slots when synced, the single global window otherwise -- so a test can
        # assert which panel received a write, not merely that one happened.
        write_targets=[],
    )

    class _SceneStub:
        def has_selection(self):
            return state.has_selection

    class _StageStub:
        def __init__(self, name):
            self._name = name

        def execute(self):
            state.stage_calls.append(self._name)
            return {"ok": True, "error": ""}

    lf_stub = ModuleType("lichtfeld")
    def _set_depth_window_sync(sync):
        state.sync_calls.append(bool(sync))
        if state.sync_write_ignored:
            return state.depth_sync
        # Model setDepthWindowSync: enabling sync with distinct independent slots copies
        # the focused window and stamps lineage in the same critical section. A
        # flag-only stub would miss invalidated-reference regressions.
        if (
            bool(sync)
            and not state.depth_sync
            and state.split_view_mode == "independent_dual"
        ):
            focused = state.focused_panel
            other = "right" if focused == "left" else "left"
            focused_range = state.panel_ranges.get(focused)
            focused_scales = state.panel_scales.get(focused)
            copied = False
            if focused_range is not None and state.panel_ranges.get(other) != focused_range:
                state.panel_ranges[other] = focused_range
                copied = True
            if focused_scales is not None and state.panel_scales.get(other) != focused_scales:
                state.panel_scales[other] = focused_scales
                copied = True
            if copied:
                # The single window the toolbar now displays is the focused
                # panel's, so the projection follows it too.
                if focused_range is not None:
                    state.depth_near, state.depth_far = focused_range
                if focused_scales is not None:
                    state.depth_scale, state.depth_scale_y = focused_scales
                state.depth_window_collapse_source = focused
                state.depth_window_collapse_kind = "sync_copy"
                state.depth_window_collapse_generation += 1
        state.depth_sync = bool(sync)
        return state.depth_sync

    lf_stub.ui = SimpleNamespace(
        get_active_tool=lambda: state.active_tool,
        get_active_submode=lambda: state.active_submode,
        message_dialog=lambda *_args, **_kwargs: None,
        get_split_view_mode=lambda: state.split_view_mode,
        get_focused_split_panel=lambda: state.focused_panel,
        get_depth_window_sync=lambda: state.depth_sync,
        get_depth_window_collapse_source=lambda: state.depth_window_collapse_source,
        get_depth_window_collapse_record=lambda: (
            state.depth_window_collapse_source,
            state.depth_window_collapse_generation,
            state.depth_window_collapse_kind,
        ),
        set_depth_window_sync=_set_depth_window_sync,
    )
    lf_stub.has_scene = lambda: state.has_scene
    lf_stub.get_scene = lambda: _SceneStub() if state.has_scene else None

    def _set_depth_filter_range(enabled, near, far, width):
        state.depth_enabled = bool(enabled)
        state.depth_near = float(near)
        state.depth_far = float(far)
        state.depth_width = float(width)
        state.depth_calls.append((state.depth_enabled, state.depth_near, state.depth_far, state.depth_width))

    def _get_depth_filter_range():
        if state.depth_read_error:
            raise RuntimeError("depth state unavailable")
        return (
            state.depth_enabled,
            state.depth_near,
            state.depth_far,
            state.depth_width,
        )

    def _write_target():
        """Where a NO-PANEL write lands, the way updateSettings routes it."""
        if state.split_view_mode != "independent_dual":
            return "global"
        return "both" if state.depth_sync else state.focused_panel

    def _projected_range():
        """The near/far the no-panel read projects.

        Two independent windows exist only in independent-dual with sync off,
        and then the projection follows the FOCUSED panel. With panel_ranges
        empty -- every pre-existing test -- this is the global pair unchanged.
        """
        if (
            state.split_view_mode == "independent_dual"
            and not state.depth_sync
            and state.focused_panel in state.panel_ranges
        ):
            return state.panel_ranges[state.focused_panel]
        return state.depth_near, state.depth_far

    def _get_depth_filter_window(*, panel=None):
        if state.depth_read_error:
            raise RuntimeError("depth state unavailable")
        near, far = _projected_range()
        if panel in ("left", "right"):
            near, far = state.panel_ranges.get(panel, (near, far))
            # Explicit-panel read: that panel's own stored window. Panels not
            # seeded by a test read back as the displayed one, which is what a
            # freshly entered split view looks like.
            scale_x, scale_y = state.panel_scales.get(
                panel, (state.depth_scale, state.depth_scale_y)
            )
        elif panel in (None, "main"):
            scale_x, scale_y = state.depth_scale, state.depth_scale_y
        else:
            raise ValueError("panel must be 'main', 'left', or 'right'")
        return (
            state.depth_enabled,
            near,
            far,
            scale_x,
            scale_y,
            state.depth_offset_x,
            state.depth_offset_y,
        )

    def _set_depth_filter_window(
        enabled, near, far, scale, offset_x, offset_y, scale_y=None, *, panel=None
    ):
        if state.window_write_error:
            raise RuntimeError("depth window write rejected")
        if panel not in (None, "main", "left", "right"):
            raise ValueError("panel must be 'main', 'left', or 'right'")
        target = _write_target()
        if panel in ("left", "right"):
            # Explicit-panel write: it lands in THAT panel's slot and leaves the
            # displayed window where it is, the way setDepthWindowForPanel does.
            state.panel_scales[panel] = (
                float(scale),
                float(scale if scale_y is None else scale_y),
            )
            state.panel_writes.append((panel, float(near), float(far), float(scale)))
            if panel in state.panel_ranges:
                state.panel_ranges[panel] = (float(near), float(far))
            state.write_targets.append(panel)
            return
        state.depth_enabled = bool(enabled)
        state.depth_near = float(near)
        state.depth_far = float(far)
        state.depth_scale = float(scale)
        state.depth_scale_y = float(scale if scale_y is None else scale_y)
        state.depth_offset_x = float(offset_x)
        state.depth_offset_y = float(offset_y)
        # Back-route the near/far the way the manager does, so a test that
        # installed distinct per-panel values can see WHICH slot a no-panel
        # write moved. With panel_ranges empty this is inert.
        for slot in (("left", "right") if target == "both" else (target,)):
            if slot in state.panel_ranges:
                state.panel_ranges[slot] = (state.depth_near, state.depth_far)
        state.write_targets.append(target)
        state.window_calls.append(
            (
                state.depth_enabled,
                state.depth_near,
                state.depth_far,
                state.depth_scale,
                state.depth_offset_x,
                state.depth_offset_y,
                state.depth_scale_y,
            )
        )
        # The controller prefers this setter over set_depth_filter_range when the
        # binding exposes it, so mirror the near/far half into depth_calls too --
        # the C++ side updates one piece of state either way, and the existing
        # assertions describe that state, not which entry point carried it.
        state.depth_calls.append(
            (state.depth_enabled, state.depth_near, state.depth_far, state.depth_width)
        )

    lf_stub.selection = SimpleNamespace(
        get_depth_filter_range=_get_depth_filter_range,
        set_depth_filter_range=_set_depth_filter_range,
        get_depth_filter_window=_get_depth_filter_window,
        set_depth_filter_window=_set_depth_filter_window,
    )
    lf_stub.pipeline = SimpleNamespace(
        edit=SimpleNamespace(delete_=lambda: _StageStub("edit.delete")),
        select=SimpleNamespace(
            all=lambda: _StageStub("select.all"),
            invert=lambda: _StageStub("select.invert"),
            none=lambda: _StageStub("select.none"),
        ),
    )
    def _undo():
        state.undo_calls += 1
        if state.undo_effect is not None:
            state.undo_effect()
        return True

    def _redo():
        state.redo_calls += 1
        if state.redo_effect is not None:
            state.redo_effect()
        return True

    lf_stub.undo = SimpleNamespace(
        can_undo=lambda: state.undo_available,
        can_redo=lambda: state.redo_available,
        undo=_undo,
        redo=_redo,
    )

    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return state


class _DataModelHandleStub:
    def __init__(self):
        self.dirty_calls = []
        # RmlUi answers a dirtied binding by reading the getter and, for a range
        # input, replaying its position back into the bound setter. The real
        # handle does that on the next model update; tests set this hook to
        # deliver it re-entrantly, from inside the call that dirtied the model.
        self.on_dirty = None

    def dirty(self, name):
        self.dirty_calls.append(name)
        if self.on_dirty is not None:
            self.on_dirty(name)


class _DataModelStub:
    def __init__(self):
        self.bound_binds = {}
        self.bound_funcs = {}
        self.bound_events = {}
        self.handle = _DataModelHandleStub()

    def bind(self, name, getter, setter):
        self.bound_binds[name] = (getter, setter)

    def bind_func(self, name, getter):
        self.bound_funcs[name] = getter

    def bind_event(self, name, callback):
        self.bound_events[name] = callback

    def get_handle(self):
        return self.handle


class _ElementStub:
    def __init__(self):
        self.classes = set()
        self.attributes = {}
        self.listeners = []
        self.select_calls = 0

    def set_class(self, name, active):
        if active:
            self.classes.add(name)
        else:
            self.classes.discard(name)

    def add_event_listener(self, name, callback):
        self.listeners.append((name, callback))

    def get_attribute(self, name, default=""):
        return self.attributes.get(name, default)

    def set_attribute(self, name, value):
        self.attributes[name] = value

    def parent(self):
        return self

    def select(self):
        self.select_calls += 1
        return True

    def emit(self, name, event=None):
        event = event or _InputEventStub()
        for event_name, callback in list(self.listeners):
            if event_name == name:
                callback(event)


class _InputEventStub:
    def __init__(self, *, linebreak=False):
        self._linebreak = linebreak
        self.propagation_stopped = False

    def get_bool_parameter(self, name, default=False):
        if name == "linebreak":
            return self._linebreak
        return default

    def stop_propagation(self):
        self.propagation_stopped = True


class _DocumentStub:
    def __init__(self):
        self.wrap = _ElementStub()
        self.near = _ElementStub()
        self.far = _ElementStub()
        self.scale = _ElementStub()
        self.offset_x = _ElementStub()
        self.offset_y = _ElementStub()
        self.near_slider = _ElementStub()
        self.far_slider = _ElementStub()
        self.scale_slider = _ElementStub()
        self.offset_x_slider = _ElementStub()
        self.offset_y_slider = _ElementStub()
        self._by_id = {
            "selection-block": self.wrap,
            "selection-depth-near": self.near,
            "selection-depth-far": self.far,
            "selection-depth-scale": self.scale,
            "selection-depth-offset-x": self.offset_x,
            "selection-depth-offset-y": self.offset_y,
            "selection-depth-near-slider": self.near_slider,
            "selection-depth-far-slider": self.far_slider,
            "selection-depth-scale-slider": self.scale_slider,
            "selection-depth-offset-x-slider": self.offset_x_slider,
            "selection-depth-offset-y-slider": self.offset_y_slider,
        }

    def get_element_by_id(self, element_id):
        return self._by_id.get(element_id)


@pytest.fixture
def selection_controls_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins.selection_controls", None)
    sys.modules.pop("lfs_plugins", None)
    state = _install_lf_stub(monkeypatch)
    module = import_module("lfs_plugins.selection_controls")
    module.RuntimeState.depth_window_draw_generation._fallback = 0
    module.RuntimeState.depth_window_draw_generation.value = 0
    commit = {"generation": 0, "panel": "left"}
    module.RuntimeState.depth_window_draw_commit._fallback = dict(commit)
    module.RuntimeState.depth_window_draw_commit.value = dict(commit)
    return module, state


def _publish_draw_commit(module, panel):
    """Emit the panel-addressed draw-commit signal the C++ side publishes."""
    generation = int(module.RuntimeState.depth_window_draw_generation.value) + 1
    module.RuntimeState.depth_window_draw_generation.value = generation
    module.RuntimeState.depth_window_draw_commit.value = {
        "generation": generation,
        "panel": panel,
    }


def test_selection_controls_show_for_selection_modes(selection_controls_module):
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    model = _DataModelStub()
    doc = _DocumentStub()

    panel.bind_model(model)
    panel.mount(doc)
    panel.update(doc)

    assert "hidden" not in doc.wrap.classes
    assert "selection_mode_label" not in model.bound_funcs
    assert model.bound_funcs["selection_has_scene"]() is True
    assert model.bound_funcs["selection_has_selection"]() is True
    assert model.bound_funcs["selection_can_undo"]() is True
    assert model.bound_binds["selection_depth_near_str"][0]() == "0.25"
    assert model.bound_binds["selection_depth_far_str"][0]() == "7.50"
    assert model.bound_funcs["selection_depth_near_slider_min"]() == "0.000"
    assert model.bound_funcs["selection_depth_near_slider_max"]() == "7.490"
    assert model.bound_funcs["selection_depth_far_slider_min"]() == "0.260"
    assert model.bound_funcs["selection_depth_far_slider_max"]() == "27.500"

    state.active_submode = "lasso"
    panel.update(doc)

    assert "selection_mode_label" not in model.handle.dirty_calls


def test_selection_depth_fallback_far_defaults_to_6(selection_controls_module):
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    model = _DataModelStub()

    state.depth_read_error = True
    panel.bind_model(model)
    panel.update(_DocumentStub())

    assert model.bound_binds["selection_depth_near_str"][0]() == "0.00"
    assert model.bound_binds["selection_depth_far_str"][0]() == "6.00"
    assert model.bound_funcs["selection_depth_far_slider_max"]() == "26.000"


def test_selection_depth_toggle_and_sliders_use_selection_api(selection_controls_module):
    module, state = selection_controls_module
    # Start disabled so the toggle actually enables. _mounted_panel decays the
    # echo holdoff armed by the first refresh (controller defaults (0, 6) vs
    # stub (0.25, 7.5)) before any slider write.
    panel, model, _doc = _mounted_panel(module, state, enabled=False)

    model.bound_events["selection_action"](None, None, ["toggle_depth"])
    assert state.depth_calls[-1] == (True, 0.25, 7.5, 1.35)

    model.bound_binds["selection_depth_near_value"][1]("1.5")
    assert state.depth_calls[-1] == (True, 1.5, 7.5, 1.35)

    model.bound_binds["selection_depth_far_value"][1]("2.0")
    assert state.depth_calls[-1] == (True, 1.5, 2.0, 1.35)


def test_selection_depth_window_sliders_ignore_rmlui_echo(selection_controls_module):
    """The size and offset sliders need the same echo protection as near/far.

    RmlUi replays a range input's pre-update position into its setter when the
    bound attributes change in the same frame. The controller absorbs that with
    a holdoff, but the holdoff was armed only by a near/far change and the three
    window setters consulted no holdoff at all, so a replayed position silently
    overwrote the size or offset the user had just set.
    """
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    model = _DataModelStub()
    doc = _DocumentStub()

    state.depth_enabled = True
    panel.bind_model(model)
    panel.mount(doc)
    for _ in range(4):
        panel.update(doc)  # let the holdoff from the initial refresh decay
    state.window_calls.clear()

    # The window moves from outside the panel -- a C++ clamp, the selection tool,
    # or an MCP write -- and the panel picks it up on the next update.
    state.depth_scale = 0.5
    state.depth_offset_x = 0.2
    panel.update(doc)
    assert panel._window_scale == pytest.approx(0.5)
    assert panel._offset_x == pytest.approx(0.2)

    # RmlUi now replays each slider's stale position into its setter.
    model.bound_binds["selection_depth_scale_value"][1]("35")
    model.bound_binds["selection_depth_offset_x_value"][1]("0")
    model.bound_binds["selection_depth_offset_y_value"][1]("0")

    # None of it may reach the binding, and the live values must survive.
    assert state.window_calls == []
    assert state.depth_scale == pytest.approx(0.5)
    assert state.depth_offset_x == pytest.approx(0.2)


def test_depth_window_scale_y_only_change_arms_echo_holdoff(selection_controls_module):
    """A native scale_y-only change must arm the echo holdoff (A8).

    Same mounted end-to-end pattern as the stale-write test above: an external
    scale_y change arms the holdoff, and a replayed size-slider position must not
    overwrite the anisotropic state.
    """
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    model = _DataModelStub()
    doc = _DocumentStub()

    state.depth_enabled = True
    panel.bind_model(model)
    panel.mount(doc)
    for _ in range(4):
        panel.update(doc)  # let the holdoff from the initial refresh decay
    state.window_calls.clear()

    state.depth_scale_y = 0.5
    panel.update(doc)
    assert panel._window_scale_y == pytest.approx(0.5)
    assert panel._depth_echo_holdoff > 0

    model.bound_binds["selection_depth_scale_value"][1]("35")
    model.bound_binds["selection_depth_offset_x_value"][1]("0")
    model.bound_binds["selection_depth_offset_y_value"][1]("0")

    assert state.window_calls == []
    assert state.depth_scale == pytest.approx(0.35)
    assert state.depth_scale_y == pytest.approx(0.5)


def test_d9_same_ratio_fresh_draw_reads_100_percent(selection_controls_module):
    """A same-ratio committed draw re-bases the Size readout to 100% (TR-2 / D9)."""
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    assert panel._window_scale == pytest.approx(0.35)
    assert panel._window_scale_y == pytest.approx(0.35)

    state.depth_scale = 0.70
    state.depth_scale_y = 0.70
    module.RuntimeState.depth_window_draw_generation.value += 1
    panel.update(doc)

    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"
    assert panel._ref_scale_x[panel._ref_key()] == pytest.approx(0.70)
    assert panel._ref_scale_y[panel._ref_key()] == pytest.approx(0.70)


def test_d9_pure_scaling_tracks_200_percent_after_release(selection_controls_module):
    """Pure scaling tracks the reference without re-basing (D9).

    Continues after the same-ratio draw re-base in (a): once the reference is
    established, doubling through the size slider must not re-base again. Use
    0.50 scales so 200% still fits under the 1.0 clamp with ref=0.50.
    """
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    state.depth_scale = 0.50
    state.depth_scale_y = 0.50
    module.RuntimeState.depth_window_draw_generation.value += 1
    panel.update(doc)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"
    for _ in range(2):
        panel.update(doc)  # decay the holdoff armed by the external draw sync
    assert panel._depth_echo_holdoff == 0
    ref_x = panel._ref_scale_x[panel._ref_key()]
    ref_y = panel._ref_scale_y[panel._ref_key()]

    model.bound_binds["selection_depth_scale_value"][1]("200")

    assert model.bound_binds["selection_depth_scale_value"][0]() == "200"
    assert panel._ref_scale_x[panel._ref_key()] == pytest.approx(ref_x)
    assert panel._ref_scale_y[panel._ref_key()] == pytest.approx(ref_y)
    assert state.depth_scale == pytest.approx(1.0)
    assert state.depth_scale_y == pytest.approx(1.0)


def test_d9_ratio_change_rebases_to_100_percent(selection_controls_module):
    """An aspect-ratio change re-bases the Size readout to 100% (D9)."""
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    state.depth_scale = 0.70
    state.depth_scale_y = 0.35
    panel.update(doc)

    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"
    assert panel._ref_scale_x[panel._ref_key()] == pytest.approx(0.70)
    assert panel._ref_scale_y[panel._ref_key()] == pytest.approx(0.35)



def test_selection_depth_user_edit_mark_expires(selection_controls_module, monkeypatch):
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    model = _DataModelStub()

    panel.bind_model(model)
    panel.update(_DocumentStub())
    panel._depth_echo_holdoff = 1
    panel._mark_depth_user_edit("near")
    marked_at = module.time.monotonic()
    monkeypatch.setattr(
        module.time,
        "monotonic",
        lambda: marked_at + module._DEPTH_USER_EDIT_MARK_TTL + 0.01,
    )

    model.bound_binds["selection_depth_near_value"][1]("1.5")

    assert state.depth_calls == []
    assert "near" not in panel._depth_user_edit_pending


def test_selection_depth_slider_press_passes_through_echo_holdoff(selection_controls_module):
    """A pressed slider applies during the echo holdoff; an unmarked replay does not.

    The user-edit mark (#1927) and the per-slider entry points (#1932) meet
    here: an external change arms the holdoff, the replayed position is dropped,
    and the position from a slider the user pressed goes through.
    """
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    model = _DataModelStub()
    doc = _DocumentStub()

    state.depth_enabled = True
    panel.bind_model(model)
    panel.mount(doc)
    for _ in range(4):
        panel.update(doc)
    state.depth_calls.clear()

    state.depth_near = 1.0
    panel.update(doc)
    assert panel._depth_echo_holdoff > 0

    model.bound_binds["selection_depth_near_value"][1]("0.5")
    assert state.depth_calls == []

    doc.near_slider.emit("mousedown")
    model.bound_binds["selection_depth_near_value"][1]("1.5")
    assert state.depth_calls[-1][1] == pytest.approx(1.5)


def test_selection_depth_text_fields_commit_like_panel_inputs(selection_controls_module):
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    model = _DataModelStub()
    doc = _DocumentStub()

    state.depth_enabled = True
    panel.bind_model(model)
    panel.mount(doc)
    panel.update(doc)

    near_getter, near_setter = model.bound_binds["selection_depth_near_str"]
    far_getter, far_setter = model.bound_binds["selection_depth_far_str"]

    doc.near.emit("focus")
    near_setter("1")

    assert doc.near.select_calls == 1
    assert near_getter() == "1"
    assert state.depth_calls == []

    doc.near.emit("change", _InputEventStub(linebreak=True))

    assert state.depth_calls[-1] == (True, 1.0, 7.5, 1.35)
    assert near_getter() == "1.00"

    far_setter("9")
    assert far_getter() == "9"

    doc.far.emit("blur")

    assert state.depth_calls[-1] == (True, 1.0, 9.0, 1.35)
    assert far_getter() == "9.00"


def test_selection_depth_text_escape_reverts_pending_edit(selection_controls_module):
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    model = _DataModelStub()
    doc = _DocumentStub()

    panel.bind_model(model)
    panel.mount(doc)
    panel.update(doc)

    near_getter, near_setter = model.bound_binds["selection_depth_near_str"]
    event = _InputEventStub()

    doc.near.emit("focus")
    near_setter("4")
    doc.near.emit("escapecancel", event)

    assert near_getter() == "0.25"
    assert state.depth_calls == []
    assert event.propagation_stopped


def test_selection_depth_text_invalid_commit_reverts(selection_controls_module):
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    model = _DataModelStub()
    doc = _DocumentStub()

    panel.bind_model(model)
    panel.mount(doc)
    panel.update(doc)

    near_getter, near_setter = model.bound_binds["selection_depth_near_str"]

    doc.near.emit("focus")
    near_setter("not-a-number")
    doc.near.emit("blur")

    assert near_getter() == "0.25"
    assert state.depth_calls == []


def test_selection_actions_use_undoable_pipeline_and_history(selection_controls_module):
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    model = _DataModelStub()

    panel.bind_model(model)

    model.bound_events["selection_action"](None, None, ["delete"])
    model.bound_events["selection_action"](None, None, ["select_all"])
    model.bound_events["selection_action"](None, None, ["invert"])
    model.bound_events["selection_action"](None, None, ["unselect"])
    model.bound_events["selection_action"](None, None, ["undo"])
    model.bound_events["selection_action"](None, None, ["redo"])

    assert state.stage_calls == ["edit.delete", "select.all", "select.invert", "select.none"]
    assert state.undo_calls == 1
    assert state.redo_calls == 1


def test_selection_controls_hide_when_selection_tool_is_inactive(selection_controls_module):
    module, state = selection_controls_module
    panel = module.SelectionControlsController()
    doc = _DocumentStub()

    state.active_tool = "builtin.translate"
    panel.mount(doc)
    doc.wrap.classes.discard("hidden")

    panel.update(doc)

    assert "hidden" in doc.wrap.classes


# ---------------------------------------------------------------------------
# Text commits vs the slider echo holdoff.
#
# The holdoff exists to drop the stale position RmlUi replays into a range input
# after its bound attributes change. A typed-and-committed value is a deliberate
# edit and must not be dropped by it. The five text commits therefore reach the
# core setters directly, while the range inputs are bound to _from_slider
# wrappers that keep the check.
# ---------------------------------------------------------------------------

# str binding, doc element, slider binding, typed text, native reader, expected
_DEPTH_TEXT_FIELDS = (
    ("selection_depth_near_str", "near", "selection_depth_near_value",
     "1", lambda s: s.depth_near, 1.0),
    ("selection_depth_far_str", "far", "selection_depth_far_value",
     "9", lambda s: s.depth_far, 9.0),
    ("selection_depth_scale_str", "scale", "selection_depth_scale_value",
     "50", lambda s: s.depth_scale, 0.175),
    ("selection_depth_offset_x_str", "offset_x", "selection_depth_offset_x_value",
     "25", lambda s: s.depth_offset_x, 0.25),
    ("selection_depth_offset_y_str", "offset_y", "selection_depth_offset_y_value",
     "25", lambda s: s.depth_offset_y, 0.25),
)


def _mounted_panel(module, state, *, enabled=True):
    """A bound, mounted, visible panel whose echo holdoff has decayed to zero."""
    panel = module.SelectionControlsController()
    model = _DataModelStub()
    doc = _DocumentStub()
    state.depth_enabled = enabled
    panel.bind_model(model)
    panel.mount(doc)
    for _ in range(4):
        panel.update(doc)
    assert panel._depth_echo_holdoff == 0
    return panel, model, doc


def _arm_holdoff(panel, doc, state):
    """Move the native state from outside the panel, as a clamp or the tool would."""
    state.depth_near = round(state.depth_near + 0.01, 4)
    panel.update(doc)
    assert panel._depth_echo_holdoff > 0


def _commit(doc, element_name, kind):
    element = getattr(doc, element_name)
    if kind == "enter":
        element.emit("change", _InputEventStub(linebreak=True))
    else:
        element.emit("blur")


@pytest.mark.parametrize("kind", ["enter", "blur"])
@pytest.mark.parametrize(
    "str_key,element_name,slider_key,typed,read_native,expected", _DEPTH_TEXT_FIELDS
)
def test_depth_text_commit_applies_while_holdoff_is_armed(
    selection_controls_module, kind, str_key, element_name, slider_key, typed, read_native, expected
):
    """Cases 1 and 4: every field, committed by Enter and by blur."""
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)
    _arm_holdoff(panel, doc, state)

    text_before = model.bound_binds[str_key][0]()
    getattr(doc, element_name).emit("focus")
    model.bound_binds[str_key][1](typed)
    _commit(doc, element_name, kind)

    assert read_native(state) == pytest.approx(expected)
    # _commit_depth_text_key ends in _sync_depth_text_bufs(force=True), which
    # rewrites every buffer from the canonical value. That must now be the
    # committed value, not the one the field held before the edit.
    assert model.bound_binds[str_key][0]() != text_before


@pytest.mark.parametrize(
    "str_key,element_name,slider_key,typed,read_native,expected", _DEPTH_TEXT_FIELDS
)
def test_depth_slider_is_still_rejected_while_holdoff_is_armed(
    selection_controls_module, str_key, element_name, slider_key, typed, read_native, expected
):
    """Case 2: the echo protection this round added must survive."""
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)
    _arm_holdoff(panel, doc, state)

    before = read_native(state)
    model.bound_binds[slider_key][1](typed)

    assert read_native(state) == pytest.approx(before)


@pytest.mark.parametrize("kind", ["enter", "blur"])
@pytest.mark.parametrize(
    "str_key,element_name,slider_key,typed,read_native,expected", _DEPTH_TEXT_FIELDS
)
def test_depth_text_commit_survives_the_slider_echo_it_causes(
    selection_controls_module, kind, str_key, element_name, slider_key, typed, read_native, expected
):
    """Case 3, the one an origin flag alone fails.

    Start with no holdoff. The commit's own _apply_depth_window dirties every
    slider-bound value, and RmlUi answers by replaying the pre-commit slider
    position. Delivered re-entrantly - from inside that dirty, while any bypass
    would still be in scope - it must not overwrite the committed value.
    """
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    stale_position = model.bound_binds[slider_key][0]()
    replayed = []

    def _echo(name):
        if name == slider_key and not replayed:
            replayed.append(name)
            model.bound_binds[slider_key][1](stale_position)

    model.handle.on_dirty = _echo

    getattr(doc, element_name).emit("focus")
    model.bound_binds[str_key][1](typed)
    _commit(doc, element_name, kind)

    assert replayed == [slider_key], "the echo was never delivered"
    assert read_native(state) == pytest.approx(expected)

    # The same echo arriving after the commit returns must also lose.
    model.handle.on_dirty = None
    model.bound_binds[slider_key][1](stale_position)
    assert read_native(state) == pytest.approx(expected)


@pytest.mark.parametrize(
    "str_key,element_name,slider_key,typed,read_native,expected", _DEPTH_TEXT_FIELDS
)
def test_focused_text_field_does_not_authorise_another_fields_slider(
    selection_controls_module, str_key, element_name, slider_key, typed, read_native, expected
):
    """Case 5: authority is per-origin, never per-focus.

    Focus each field in turn and drive every OTHER field's range input.
    """
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)
    _arm_holdoff(panel, doc, state)

    getattr(doc, element_name).emit("focus")
    for other_str, _other_el, other_slider, other_typed, other_read, _exp in _DEPTH_TEXT_FIELDS:
        if other_str == str_key:
            continue
        before = other_read(state)
        model.bound_binds[other_slider][1](other_typed)
        assert other_read(state) == pytest.approx(before), other_slider


@pytest.mark.parametrize(
    "str_key,element_name,slider_key,typed,read_native,expected", _DEPTH_TEXT_FIELDS
)
def test_enter_then_blur_may_write_twice_upstream_compatibility(
    selection_controls_module, str_key, element_name, slider_key, typed, read_native, expected
):
    """Upstream compatibility: Enter leaves the field focused, so blur follows.

    Both events may carry the same canonical value to the native side.
    """
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    element = getattr(doc, element_name)
    element.emit("focus")
    model.bound_binds[str_key][1](typed)
    element.emit("change", _InputEventStub(linebreak=True))

    assert read_native(state) == pytest.approx(expected)
    writes_after_enter = len(state.window_calls)

    element.emit("blur")

    assert len(state.window_calls) == writes_after_enter + 1
    assert read_native(state) == pytest.approx(expected)


@pytest.mark.parametrize("kind", ["enter", "blur"])
@pytest.mark.parametrize(
    "str_key,element_name,slider_key,typed,read_native,expected", _DEPTH_TEXT_FIELDS
)
def test_text_commit_does_not_defeat_the_visible_guard(
    selection_controls_module, kind, str_key, element_name, slider_key, typed, read_native, expected
):
    """Case 7, first half. Isolated: hiding the panel through update() also nulls
    _last_state_key (:298-301), so this clears _visible on its own."""
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    panel._visible = False
    assert panel._last_state_key is not None
    before = len(state.window_calls)

    getattr(doc, element_name).emit("focus")
    model.bound_binds[str_key][1](typed)
    _commit(doc, element_name, kind)

    assert len(state.window_calls) == before
    # and a refused commit must not leave the echo holdoff armed
    assert panel._depth_echo_holdoff == 0


@pytest.mark.parametrize("kind", ["enter", "blur"])
@pytest.mark.parametrize(
    "str_key,element_name,slider_key,typed,read_native,expected", _DEPTH_TEXT_FIELDS
)
def test_text_commit_does_not_defeat_the_state_key_guard(
    selection_controls_module, kind, str_key, element_name, slider_key, typed, read_native, expected
):
    """Case 7, second half: visible, but no update has landed yet."""
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    panel._last_state_key = None
    assert panel._visible is True
    before = len(state.window_calls)

    getattr(doc, element_name).emit("focus")
    model.bound_binds[str_key][1](typed)
    _commit(doc, element_name, kind)

    assert len(state.window_calls) == before
    assert panel._depth_echo_holdoff == 0


def test_depth_uses_the_range_api_when_the_window_api_is_absent(selection_controls_module):
    """The stub exposes the window API, which would otherwise hide the legacy
    fallback in _apply_depth_window from every test in this file."""
    module, state = selection_controls_module
    lf_stub = sys.modules["lichtfeld"]
    del lf_stub.selection.get_depth_filter_window
    del lf_stub.selection.set_depth_filter_window

    panel, model, doc = _mounted_panel(module, state)
    state.depth_calls.clear()
    state.window_calls.clear()

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("1")
    doc.near.emit("change", _InputEventStub(linebreak=True))

    assert state.window_calls == []
    assert state.depth_calls, "the range fallback carried nothing"
    assert state.depth_calls[-1][1] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "str_key,element_name,slider_key,typed,read_native,expected", _DEPTH_TEXT_FIELDS
)
def test_a_refused_commit_is_retried_on_blur(
    selection_controls_module, str_key, element_name, slider_key, typed, read_native, expected
):
    """A native write rejected on Enter is retried when blur follows.

    _apply_depth_window catches and reports the failure, but blur re-enters
    _commit_depth_text_key with the same buffer and may succeed.
    """
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    element = getattr(doc, element_name)
    element.emit("focus")
    model.bound_binds[str_key][1](typed)

    state.window_write_error = True
    element.emit("change", _InputEventStub(linebreak=True))
    assert read_native(state) != pytest.approx(expected), "the write should have failed"

    state.window_write_error = False
    element.emit("blur")

    assert read_native(state) == pytest.approx(expected)


@pytest.mark.parametrize(
    "str_key,element_name,slider_key,typed,read_native,expected", _DEPTH_TEXT_FIELDS
)
def test_escape_after_a_commit_still_reverts(
    selection_controls_module, str_key, element_name, slider_key, typed, read_native, expected
):
    """Escape must keep reverting after an Enter has already committed.

    cancelFocusedElement dispatches escapecancel and then blurs immediately
    (rml_input_utils.hpp), so the revert reaches the native side through the
    blur commit.
    """
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    original = read_native(state)
    element = getattr(doc, element_name)
    element.emit("focus")
    model.bound_binds[str_key][1](typed)
    element.emit("change", _InputEventStub(linebreak=True))
    assert read_native(state) == pytest.approx(expected)

    element.emit("escapecancel", _InputEventStub())
    element.emit("blur")

    assert read_native(state) == pytest.approx(original)


@pytest.mark.parametrize(
    "str_key,element_name,slider_key,typed,read_native,expected", _DEPTH_TEXT_FIELDS
)
def test_depth_text_commit_same_string_in_a_later_session(
    selection_controls_module, str_key, element_name, slider_key, typed, read_native, expected
):
    """Cross-session: a later focus session can commit the same final string."""
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    element = getattr(doc, element_name)
    element.emit("focus")
    model.bound_binds[str_key][1](typed)
    element.emit("change", _InputEventStub(linebreak=True))
    element.emit("blur")
    assert read_native(state) == pytest.approx(expected)
    first_session_writes = len(state.window_calls)
    # The canonical display string session one settled on: session two must
    # reproduce it EXACTLY (typing the raw `typed` string would change the
    # buffer and, under the removed mark, clear the mark as a side effect —
    # which made three of five cases pass even with the mark present).
    canonical = model.bound_binds[str_key][0]()

    # Native state moves away WITHOUT the panel re-synchronizing the buffer:
    # the buffer still holds the canonical string from session one, which is
    # exactly the state the removed mark used to suppress.
    if str_key == "selection_depth_near_str":
        state.depth_near = expected + 1.0
    elif str_key == "selection_depth_far_str":
        state.depth_far = expected + 1.0
    elif str_key == "selection_depth_scale_str":
        bumped = min(expected + 0.1, 1.0)
        state.depth_scale = bumped
        state.depth_scale_y = bumped
    elif str_key == "selection_depth_offset_x_str":
        state.depth_offset_x = expected + 0.1
    else:
        state.depth_offset_y = expected + 0.1

    element.emit("focus")
    model.bound_binds[str_key][1](canonical)
    element.emit("blur")

    # Exactly one new native write, restoring the value the string names.
    assert len(state.window_calls) == first_session_writes + 1
    assert read_native(state) == pytest.approx(expected)




# str binding, doc element, typed text, native reader, base value, value after commit.
# Each typed value renders IDENTICALLY to the base at that field's display
# precision: near/far round to two decimals, scale and the offsets to whole
# percent. That collision is the point of the test below.
_DEPTH_COLLIDING_FIELDS = (
    ("selection_depth_near_str", "near", "0.254", lambda s: s.depth_near, 0.25, 0.254),
    ("selection_depth_far_str", "far", "7.504", lambda s: s.depth_far, 7.5, 7.504),
    ("selection_depth_scale_str", "scale", "100.4", lambda s: s.depth_scale, 0.35, 0.3514),
    ("selection_depth_offset_x_str", "offset_x", "0.4", lambda s: s.depth_offset_x, 0.0, 0.004),
    ("selection_depth_offset_y_str", "offset_y", "0.4", lambda s: s.depth_offset_y, 0.0, 0.004),
)


@pytest.mark.parametrize(
    "str_key,element_name,typed,read_native,base,committed", _DEPTH_COLLIDING_FIELDS
)
def test_escape_reverts_when_the_restored_text_is_unchanged(
    selection_controls_module, str_key, element_name, typed, read_native, base, committed
):
    """Escape must revert even when the restored text is identical.

    Canonical text is rounded, so a native value the user has just committed can
    render exactly like the pre-edit one. The blur after escapecancel must still
    carry the revert to the native side.
    """
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    assert read_native(state) == pytest.approx(base)
    getter = model.bound_binds[str_key][0]
    text_before = getter()

    element = getattr(doc, element_name)
    element.emit("focus")
    model.bound_binds[str_key][1](typed)
    element.emit("change", _InputEventStub(linebreak=True))

    assert read_native(state) == pytest.approx(committed), "the commit did not land"
    assert getter() == text_before, "these values must collide for this test to mean anything"
    writes_after_enter = len(state.window_calls)

    element.emit("escapecancel", _InputEventStub())
    element.emit("blur")

    assert read_native(state) == pytest.approx(base)
    # escapecancel only restores the buffer; the blur carries the single revert
    # write. Pinning the count rejects an implementation that writes in both.
    assert len(state.window_calls) == writes_after_enter + 1


# ---------------------------------------------------------------------------
# Per-panel depth windows (chip, sync toggle, per-panel Size refs)
# ---------------------------------------------------------------------------


def _independent_dual(state, *, focused="left", sync=False):
    state.split_view_mode = "independent_dual"
    state.focused_panel = focused
    state.depth_sync = sync


def test_panel_chip_and_sync_toggle_hidden_outside_independent_dual(
    selection_controls_module,
):
    module, state = selection_controls_module
    _panel, model, _doc = _mounted_panel(module, state)

    assert model.bound_funcs["selection_panel_chip_visible"]() is False
    assert model.bound_funcs["selection_depth_sync_active"]() is False


def test_panel_chip_tracks_the_focused_panel(selection_controls_module):
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    panel, model, doc = _mounted_panel(module, state)

    assert model.bound_funcs["selection_panel_chip_visible"]() is True
    assert model.bound_funcs["selection_panel_chip_label"]() == "L"

    state.focused_panel = "right"
    panel.update(doc)

    assert model.bound_funcs["selection_panel_chip_label"]() == "R"


def test_focus_change_between_identical_windows_still_dirties_the_chip(
    selection_controls_module,
):
    """Polling fix.

    The state tuple used to carry VALUES only, so a focus switch between two
    panels holding the SAME window -- the default right after entering split
    view -- would never dirty anything and the chip would keep reading L.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    panel, model, doc = _mounted_panel(module, state)
    model.handle.dirty_calls.clear()

    state.focused_panel = "right"
    # Deliberately do NOT move the window: the panels are identical here.
    reason = panel.update(doc)

    assert reason is not None
    assert "selection_panel_chip_label" in model.handle.dirty_calls


def test_size_reference_is_per_panel_across_focus_switches(
    selection_controls_module,
):
    """A focus switch must not read as 'a new shape drawn'.

    Draw in L, focus R, focus back to L: L's percentage must be exactly what it
    was, and R must keep its own reference rather than inheriting L's.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    panel, model, doc = _mounted_panel(module, state)

    # A fresh draw in L rebases L's reference to the drawn size.
    state.depth_scale = 0.60
    state.depth_scale_y = 0.60
    panel.update(doc)
    _publish_draw_commit(module, "left")
    panel.update(doc)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"

    # Shrink L against its own reference.
    state.depth_scale = 0.30
    state.depth_scale_y = 0.30
    panel.update(doc)
    left_percent = model.bound_binds["selection_depth_scale_value"][0]()
    assert left_percent == "50"

    # Focus R, which carries a different window and its own untouched reference.
    state.focused_panel = "right"
    state.depth_scale = 0.35
    state.depth_scale_y = 0.35
    panel.update(doc)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"

    # Back to L: the reference survived the round trip untouched.
    state.focused_panel = "left"
    state.depth_scale = 0.30
    state.depth_scale_y = 0.30
    panel.update(doc)
    assert model.bound_binds["selection_depth_scale_value"][0]() == left_percent
    assert panel._ref_scale_x["left"] == pytest.approx(0.60)
    assert panel._ref_scale_x["right"] == pytest.approx(0.35)


_GT_SIZE_REFERENCES = {"left": (0.60, 0.30), "right": (0.40, 0.20)}


def _focus_reference_panel(state, panel):
    state.focused_panel = panel
    state.depth_scale, state.depth_scale_y = state.panel_scales[panel]


def _mounted_gt_references(module, state, *, focused="right"):
    """Give the real controller two drawn baselines, then scale both windows."""
    _independent_dual(state)
    panel, model, doc = _mounted_panel(module, state)
    for key, scales in _GT_SIZE_REFERENCES.items():
        state.panel_scales[key] = scales
        _focus_reference_panel(state, key)
        panel.update(doc)
        _publish_draw_commit(module, key)
        panel.update(doc)
        state.panel_scales[key] = (0.30, 0.15)
        _focus_reference_panel(state, key)
        panel.update(doc)
    _focus_reference_panel(state, focused)
    panel.update(doc)
    return panel, model, doc


def _assert_gt_references_and_size_edits(panel, model, doc, state):
    # Equal native windows still have DIFFERENT reference identities. Comparing
    # restored geometry alone, or only checking 100%, would miss this regression.
    for key, percent in (("left", "50"), ("right", "75")):
        _focus_reference_panel(state, key)
        panel.update(doc)
        assert model.bound_binds["selection_depth_scale_value"][0]() == percent
        assert (panel._ref_scale_x[key], panel._ref_scale_y[key]) == pytest.approx(
            _GT_SIZE_REFERENCES[key]
        )
    for key, scales in _GT_SIZE_REFERENCES.items():
        _focus_reference_panel(state, key)
        panel.update(doc)
        doc.scale.emit("focus")
        model.bound_binds["selection_depth_scale_str"][1]("125")
        doc.scale.emit("change", _InputEventStub(linebreak=True))
        assert state.write_targets[-1] == key
        write = state.window_calls[-1]
        assert (write[3], write[6]) == pytest.approx(tuple(v * 1.25 for v in scales))
        doc.scale.emit("blur")


@pytest.mark.parametrize("focused", ["left", "right"])
@pytest.mark.parametrize(
    "schedule",
    ["direct", "observed", "only_gt", "only_disabled", "coalesced", "hidden",
     "repeated", "repeated_coalesced", "repeated_hidden"],
)
def test_gt_roundtrip_preserves_panel_reference_identity(
    selection_controls_module, focused, schedule
):
    module, state = selection_controls_module
    panel, model, doc = _mounted_gt_references(module, state, focused=focused)
    generation = state.depth_window_collapse_generation
    # Native park/retained Disabled/restore do not invalidate either slot, so
    # these published endpoints have NO lineage stamp. Focus resets to Left.
    modes = ["gt_comparison"]
    if schedule != "direct":
        modes.append("none")
    if schedule.startswith("repeated"):
        modes += ["gt_comparison", "none"]
    hidden = schedule in ("hidden", "repeated_hidden")
    if hidden:
        state.active_tool = "builtin.move"
        panel.update(doc)
    # An external focus change can coalesce with park and its reset to Left.
    state.focused_panel = "right" if focused == "left" else "left"
    for mode in modes:
        state.split_view_mode = mode
        state.focused_panel = "left"
        if hidden or schedule in ("direct", "observed", "repeated") or (
            schedule == "only_gt" and mode == "gt_comparison"
        ) or (schedule == "only_disabled" and mode == "none"):
            panel.update(doc)
    _independent_dual(state)
    _focus_reference_panel(state, "left")
    state.active_tool = "builtin.select"
    panel.update(doc)
    assert state.depth_window_collapse_generation == generation
    _assert_gt_references_and_size_edits(panel, model, doc, state)


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("through_disabled", [False, True])
def test_first_mount_in_gt_baselines_without_inventing_reference_history(
    selection_controls_module, through_disabled, sync
):
    module, state = selection_controls_module
    state.split_view_mode = "gt_comparison"
    state.depth_sync = sync
    state.panel_scales = {"left": (0.30, 0.15), "right": (0.30, 0.15) if sync else (0.20, 0.10)}
    # A global GT write can differ from either retained window.
    state.depth_scale, state.depth_scale_y = (0.80, 0.40)
    panel, model, doc = _mounted_panel(module, state)
    if through_disabled:
        state.split_view_mode = "none"
        panel.update(doc)
    _independent_dual(state, sync=sync)
    for key in ("left", "right"):
        _focus_reference_panel(state, key)
        panel.update(doc)
        assert model.bound_binds["selection_depth_scale_value"][0]() == "100"
        assert (panel._ref_scale_x[key], panel._ref_scale_y[key]) == pytest.approx(
            state.panel_scales[key]
        )


def _mounted_synced_gt_reference(module, state):
    _independent_dual(state, sync=True)
    baseline = (0.60, 0.30)
    parked_scales = (0.30, 0.15)
    state.depth_scale, state.depth_scale_y = baseline
    state.panel_scales = {key: baseline for key in ("left", "right")}
    panel, model, doc = _mounted_panel(module, state)
    state.depth_scale, state.depth_scale_y = parked_scales
    state.panel_scales = {key: parked_scales for key in ("left", "right")}
    panel.update(doc)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "50"
    return panel, model, doc


@pytest.mark.parametrize("schedule", ["direct", "observed", "only_disabled", "coalesced", "hidden", "repeated"])
@pytest.mark.parametrize("gt_scales", [(0.80, 0.20), (0.80, 0.40), (0.30, 0.15)],
                         ids=["changed_aspect", "same_aspect", "noop"])
def test_synced_gt_roundtrip_preserves_shared_size_reference(
    selection_controls_module, schedule, gt_scales
):
    module, state = selection_controls_module
    panel, model, doc = _mounted_synced_gt_reference(module, state)
    baseline = (0.60, 0.30)
    parked_scales = (0.30, 0.15)
    generation = state.depth_window_collapse_generation
    if schedule == "hidden":
        state.active_tool = "builtin.move"
        panel.update(doc)
    # Native park/GT global write retain the synced pair without stamping.
    state.split_view_mode = "gt_comparison"
    state.depth_scale, state.depth_scale_y = gt_scales
    state.panel_scales = {key: gt_scales for key in ("left", "right")}
    if schedule in ("direct", "observed", "hidden", "repeated"):
        panel.update(doc)
    if schedule != "direct":
        state.split_view_mode = "none"
        if schedule != "coalesced":
            panel.update(doc)
    if schedule == "repeated":
        for mode in ("gt_comparison", "none"):
            state.split_view_mode = mode
            panel.update(doc)
    # Restoration replaces GT's live projection with the original synced pair.
    state.panel_scales = {key: parked_scales for key in ("left", "right")}
    state.depth_scale, state.depth_scale_y = parked_scales
    _independent_dual(state, sync=True)
    state.active_tool = "builtin.select"
    panel.update(doc)
    assert state.depth_window_collapse_generation == generation
    assert model.bound_binds["selection_depth_scale_value"][0]() == "50"
    assert (panel._ref_scale_x["shared"], panel._ref_scale_y["shared"]) == pytest.approx(baseline)
    # Disabling sync must seed both panels from that same retained shared base.
    state.depth_sync = False
    panel.update(doc)
    for key in ("left", "right"):
        _focus_reference_panel(state, key)
        panel.update(doc)
        assert model.bound_binds["selection_depth_scale_value"][0]() == "50"
        doc.scale.emit("focus")
        model.bound_binds["selection_depth_scale_str"][1]("125")
        doc.scale.emit("change", _InputEventStub(linebreak=True))
        assert state.write_targets[-1] == key
        write = state.window_calls[-1]
        assert (write[3], write[6]) == pytest.approx((0.75, 0.375))
        doc.scale.emit("blur")


@pytest.mark.parametrize("schedule", ["observed", "coalesced", "hidden"])
@pytest.mark.parametrize("invalidator", ["geometry", "disabled_sync", "scene_reset"])
def test_synced_retained_discard_does_not_restore_shared_size_reference(
    selection_controls_module, schedule, invalidator
):
    module, state = selection_controls_module
    panel, model, doc = _mounted_synced_gt_reference(module, state)
    if schedule == "hidden":
        state.active_tool = "builtin.move"
        panel.update(doc)
    state.split_view_mode = "gt_comparison"
    state.depth_scale, state.depth_scale_y = (0.80, 0.20)
    state.panel_scales = {key: (0.80, 0.20) for key in ("left", "right")}
    if schedule != "coalesced":
        panel.update(doc)
    state.split_view_mode = "none"
    if schedule != "coalesced":
        panel.update(doc)
    changes = {
        "geometry": {"depth_scale": 0.45, "depth_scale_y": 0.225},
        "disabled_sync": {"depth_sync": False},
        "scene_reset": {"has_scene": False},
    }[invalidator]
    _publish_retained_discard(state, changes)
    if schedule != "coalesced":
        panel.update(doc)
    current_scales = (state.depth_scale, state.depth_scale_y)
    state.panel_scales = {key: current_scales for key in ("left", "right")}
    _independent_dual(state, sync=state.depth_sync)
    state.has_scene = True
    state.active_tool = "builtin.select"
    for key in ("left", "right"):
        _focus_reference_panel(state, key)
        panel.update(doc)
        assert model.bound_binds["selection_depth_scale_value"][0]() == "100"
        assert (panel._ref_scale_x[key], panel._ref_scale_y[key]) == pytest.approx(current_scales)
    assert panel._retained_reference_generation is None


@pytest.mark.parametrize("sync", [False, True])
@pytest.mark.parametrize("through_disabled", [False, True])
def test_global_origin_gt_aspect_edit_keeps_current_global_reference(
    selection_controls_module, sync, through_disabled
):
    module, state = selection_controls_module
    state.depth_sync = sync
    state.depth_scale, state.depth_scale_y = (0.60, 0.30)
    panel, model, doc = _mounted_panel(module, state)
    state.depth_scale, state.depth_scale_y = (0.30, 0.15)
    panel.update(doc)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "50"
    state.split_view_mode = "gt_comparison"
    state.depth_scale, state.depth_scale_y = (0.80, 0.20)
    state.panel_scales = {key: (0.80, 0.20) for key in ("left", "right")}
    panel.update(doc)
    if through_disabled:
        state.split_view_mode = "none"
        panel.update(doc)
    _independent_dual(state, sync=sync)
    for key in ("left", "right"):
        _focus_reference_panel(state, key)
        panel.update(doc)
        assert model.bound_binds["selection_depth_scale_value"][0]() == "100"
        assert (panel._ref_scale_x[panel._ref_key()], panel._ref_scale_y[panel._ref_key()]) == pytest.approx((0.80, 0.20))
    assert panel._retained_reference_generation is None


def _visit_retained_disabled(panel, doc, state, schedule):
    if schedule == "hidden":
        state.active_tool = "builtin.move"
        panel.update(doc)
    for mode in ("gt_comparison", "none"):
        state.split_view_mode = mode
        state.focused_panel = "left"
        if schedule in ("observed", "hidden"):
            panel.update(doc)


def _publish_retained_discard(state, changes, *, kind="retained_pair_discard"):
    # A published native result, not an implementation of its setters: native
    # tests must establish WHICH writes stamp. The real consumer must recover
    # from this packet regardless of polling schedule or latest source panel.
    for field, value in changes.items():
        setattr(state, field, value)
    state.depth_window_collapse_source = "right"
    state.depth_window_collapse_kind = kind
    state.depth_window_collapse_generation += 1


@pytest.mark.parametrize("schedule", ["observed", "coalesced", "hidden"])
@pytest.mark.parametrize(
    "changes,kind",
    [
        pytest.param({"depth_scale": 0.25, "depth_scale_y": 0.125}, "project_restore", id="project_reset"),
        pytest.param({"has_scene": False}, "retained_pair_discard", id="scene_reset"),
        pytest.param({"split_view_mode": "ply_comparison"}, "retained_pair_discard", id="other_comparison"),
        pytest.param({"depth_near": 0.50}, "retained_pair_discard", id="near_write"),
        pytest.param({"depth_far": 12.0}, "retained_pair_discard", id="far_write"),
        pytest.param({"depth_scale": 0.45}, "retained_pair_discard", id="scale_x_write"),
        pytest.param({"depth_scale_y": 0.225}, "retained_pair_discard", id="scale_y_write"),
        pytest.param({"depth_offset_x": 0.20}, "retained_pair_discard", id="offset_x_write"),
        pytest.param({"depth_offset_y": -0.20}, "retained_pair_discard", id="offset_y_write"),
        pytest.param({"depth_scale": 0.45, "depth_scale_y": 0.225}, "retained_pair_discard", id="changed_drag_commit"),
        pytest.param({"depth_sync": True}, "retained_pair_discard", id="disabled_sync_change"),
    ],
)
def test_retained_pair_discard_invalidates_size_references(
    selection_controls_module, changes, kind, schedule
):
    module, state = selection_controls_module
    panel, model, doc = _mounted_gt_references(module, state)
    _visit_retained_disabled(panel, doc, state, schedule)
    _publish_retained_discard(state, changes, kind=kind)
    if schedule in ("observed", "hidden"):
        panel.update(doc)
    # With eligibility discarded, native entry seeds both windows from global.
    current_scales = (state.depth_scale, state.depth_scale_y)
    state.panel_scales = {key: current_scales for key in ("left", "right")}
    _independent_dual(state, sync=state.depth_sync)
    state.has_scene = True
    state.active_tool = "builtin.select"
    for key in ("left", "right"):
        _focus_reference_panel(state, key)
        panel.update(doc)
        assert model.bound_binds["selection_depth_scale_value"][0]() == "100"
        assert (panel._ref_scale_x[key], panel._ref_scale_y[key]) == pytest.approx(current_scales)
    assert panel._collapse_generation == state.depth_window_collapse_generation
    assert panel._retained_reference_generation is None
    # A subsequent valid excursion must not resurrect the discarded references.
    state.depth_sync = False
    panel.update(doc)
    _visit_retained_disabled(panel, doc, state, "observed")
    _independent_dual(state)
    panel.update(doc)
    for key in ("left", "right"):
        assert (panel._ref_scale_x[key], panel._ref_scale_y[key]) == pytest.approx(current_scales)


@pytest.mark.parametrize("schedule", ["observed", "coalesced", "hidden"])
@pytest.mark.parametrize(
    "operation",
    ["equal_normalized_write", "enable_only", "viz_only", "same_sync", "refused_write",
     "cancelled_drag", "subthreshold_drag", "unchanged_drag_commit", "gt_global_geometry"],
)
def test_retained_pair_preserving_updates_keep_size_references(
    selection_controls_module, operation, schedule
):
    module, state = selection_controls_module
    panel, model, doc = _mounted_gt_references(module, state)
    generation = state.depth_window_collapse_generation
    original_windows = dict(state.panel_scales)
    _visit_retained_disabled(panel, doc, state, schedule)
    if operation == "enable_only":
        state.depth_enabled = False
    elif operation == "viz_only":
        module.lf.get_render_settings = lambda: SimpleNamespace(depth_filter_viz_mode=2)
    elif operation == "same_sync":
        assert module.lf.ui.set_depth_window_sync(False) is False
    elif operation == "refused_write":
        state.window_write_error = True
        with pytest.raises(RuntimeError, match="rejected"):
            module.lf.selection.set_depth_filter_window(True, 2.0, 20.0, 0.80, 0.2, 0.3)
        state.window_write_error = False
    elif operation in ("cancelled_drag", "subthreshold_drag", "gt_global_geometry"):
        if operation == "gt_global_geometry":
            state.split_view_mode = "gt_comparison"
        # Global preview / GT compatibility writes may even change aspect ratio.
        # Only shared is addressed here; retained per-panel baselines must survive.
        state.depth_scale, state.depth_scale_y = (0.80, 0.35)
        if schedule in ("observed", "hidden"):
            panel.update(doc)
        if operation != "gt_global_geometry":
            state.depth_scale, state.depth_scale_y = (0.30, 0.15)
    else:
        # Equal normalized write and unchanged commit publish unchanged state.
        # Normalization/commit eligibility is exercised by the native tests;
        # this consumer receives only these already-normalized getter values.
        module.lf.selection.set_depth_filter_window(
            state.depth_enabled, state.depth_near, state.depth_far,
            state.depth_scale, state.depth_offset_x, state.depth_offset_y, state.depth_scale_y,
        )
    if schedule in ("observed", "hidden"):
        panel.update(doc)
    assert state.depth_window_collapse_generation == generation
    state.panel_scales = original_windows
    _independent_dual(state)
    _focus_reference_panel(state, "left")
    state.active_tool = "builtin.select"
    panel.update(doc)
    _assert_gt_references_and_size_edits(panel, model, doc, state)


def test_retained_discard_between_record_reads_retries_before_using_references(
    selection_controls_module,
):
    module, state = selection_controls_module
    panel, model, doc = _mounted_gt_references(module, state)
    _visit_retained_disabled(panel, doc, state, "observed")
    settled_record = module.lf.ui.get_depth_window_collapse_record
    reads = []

    def record_with_discard():
        record = settled_record()
        reads.append(record)
        if len(reads) == 1:
            _publish_retained_discard(state, {"depth_scale": 0.45, "depth_scale_y": 0.225})
            state.panel_scales = {key: (0.45, 0.225) for key in ("left", "right")}
            _independent_dual(state)
        return record

    module.lf.ui.get_depth_window_collapse_record = record_with_discard
    panel.update(doc)
    assert len(reads) == 4
    assert panel._context_read_exhausted is False
    assert panel._retained_reference_generation is None
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"
    for key in ("left", "right"):
        assert (panel._ref_scale_x[key], panel._ref_scale_y[key]) == pytest.approx((0.45, 0.225))


def test_retained_discard_exhaustion_consumes_neither_references_nor_stale_size_edit(
    selection_controls_module,
):
    module, state = selection_controls_module
    panel, model, doc = _mounted_gt_references(module, state)
    _visit_retained_disabled(panel, doc, state, "observed")
    generation = panel._retained_reference_generation
    assert generation == state.depth_window_collapse_generation
    doc.scale.emit("focus")
    model.bound_binds["selection_depth_scale_str"][1]("150")
    settled_record = module.lf.ui.get_depth_window_collapse_record
    reads = []

    def storming_record():
        reads.append(1)
        if len(reads) % 2 == 0:
            # Each read spans another park -> Disabled edit -> Independent
            # cycle. Only its destructive edit stamps, and the endpoint can
            # still be identical to the preceding attempt's endpoint.
            _publish_retained_discard(state, {
                "depth_near": state.depth_near + 0.1,
                "depth_scale": 0.45, "depth_scale_y": 0.225,
            })
            state.panel_scales = {key: (0.45, 0.225) for key in ("left", "right")}
            _independent_dual(state)
        return settled_record()

    module.lf.ui.get_depth_window_collapse_record = storming_record
    panel.update(doc)
    assert len(reads) == 2 * module._CONTEXT_READ_ATTEMPTS
    assert panel._context_read_exhausted is True
    assert panel._collapse_generation == generation
    assert panel._retained_reference_generation == generation
    for key, scales in _GT_SIZE_REFERENCES.items():
        assert (panel._ref_scale_x[key], panel._ref_scale_y[key]) == pytest.approx(scales)
    writes_before = len(state.window_calls)
    doc.scale.emit("change", _InputEventStub(linebreak=True))
    assert len(state.window_calls) == writes_before
    assert panel._retained_reference_generation == generation
    assert panel._depth_text_bufs["selection_depth_scale_str"] == "150"
    module.lf.ui.get_depth_window_collapse_record = settled_record
    panel.update(doc)
    assert panel._context_read_exhausted is False
    assert panel._retained_reference_generation is None
    assert panel._collapse_generation == state.depth_window_collapse_generation
    # A deferred live edit can replay the freshly canonicalized 100% as a no-op;
    # it must never apply the stale 150% text or either discarded baseline.
    for write in state.window_calls[writes_before:]:
        assert (write[3], write[6]) == pytest.approx((0.45, 0.225))
    assert (state.depth_scale, state.depth_scale_y) == pytest.approx((0.45, 0.225))
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"
    for key in ("left", "right"):
        assert (panel._ref_scale_x[key], panel._ref_scale_y[key]) == pytest.approx((0.45, 0.225))


def test_first_mount_in_disabled_keeps_shared_to_independent_seed(selection_controls_module):
    module, state = selection_controls_module
    # No prior observed GT/panel references: mode='none' cannot reveal history.
    state.depth_scale, state.depth_scale_y = (0.60, 0.30)
    panel, model, doc = _mounted_panel(module, state)
    state.depth_scale, state.depth_scale_y = (0.30, 0.15)
    panel.update(doc)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "50"
    state.panel_scales = {key: (0.30, 0.15) for key in ("left", "right")}
    _independent_dual(state)
    for key in ("left", "right"):
        _focus_reference_panel(state, key)
        panel.update(doc)
        assert model.bound_binds["selection_depth_scale_value"][0]() == "50"
        assert (panel._ref_scale_x[key], panel._ref_scale_y[key]) == pytest.approx((0.60, 0.30))


@pytest.mark.parametrize("through_disabled", [False, True])
def test_global_origin_gt_keeps_known_shared_size_reference(
    selection_controls_module, through_disabled
):
    module, state = selection_controls_module
    state.depth_scale, state.depth_scale_y = (0.60, 0.30)
    panel, model, doc = _mounted_panel(module, state)
    state.depth_scale, state.depth_scale_y = (0.30, 0.15)
    panel.update(doc)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "50"
    state.split_view_mode = "gt_comparison"
    panel.update(doc)
    if through_disabled:
        state.split_view_mode = "none"
        panel.update(doc)
    # No independent pair was parked. This is ordinary global-window seeding,
    # whose already-observed shared baseline must survive the GT detour.
    state.panel_scales = {key: (0.30, 0.15) for key in ("left", "right")}
    _independent_dual(state)
    for key in ("left", "right"):
        _focus_reference_panel(state, key)
        panel.update(doc)
        assert model.bound_binds["selection_depth_scale_value"][0]() == "50"
        assert (panel._ref_scale_x[key], panel._ref_scale_y[key]) == pytest.approx((0.60, 0.30))


def test_draw_commit_rebases_the_panel_the_signal_names(selection_controls_module):
    """Undoing an R drag while L is focused must not touch L."""
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    panel, _model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20

    # A commit lands on the UNFOCUSED panel while the toolbar displays L. The
    # two windows differ, so rebasing to the DISPLAYED size instead of R's own
    # would land 0.45 rather than 0.80 -- that is the whole distinction.
    state.depth_scale = 0.45
    state.depth_scale_y = 0.45
    state.panel_scales["right"] = (0.80, 0.80)
    panel.update(doc)
    _publish_draw_commit(module, "right")
    panel.update(doc)

    assert panel._ref_scale_x["right"] == pytest.approx(0.80), "R rebased to the displayed window"
    assert panel._ref_scale_y["right"] == pytest.approx(0.80)
    assert panel._ref_scale_x["left"] == pytest.approx(0.60), "L was rebased by an R commit"


def test_sync_off_seeds_both_panel_references_from_the_shared_one(
    selection_controls_module,
):
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=True)
    panel, _model, doc = _mounted_panel(module, state)

    # Under sync the two references collapse to one shared entry.
    assert panel._ref_key() == "shared"
    panel._ref_scale_x["shared"] = 0.55
    panel._ref_scale_y["shared"] = 0.55
    panel._ref_scale_x["left"] = 0.11
    panel._ref_scale_x["right"] = 0.99

    state.depth_sync = False
    panel.update(doc)

    assert panel._ref_key() == "left"
    assert panel._ref_scale_x["left"] == pytest.approx(0.55)
    assert panel._ref_scale_x["right"] == pytest.approx(0.55)


def test_sync_toggle_reflects_the_actual_post_call_state(selection_controls_module):
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    panel, model, doc = _mounted_panel(module, state)

    model.bound_events["selection_action"](model.handle, None, ["toggle_sync"])

    assert state.sync_calls == [True]
    assert model.bound_funcs["selection_depth_sync_active"]() is True
    assert model.bound_funcs["selection_depth_sync_icon"]() == module._SYNC_ICON_ON

    panel.update(doc)
    assert model.bound_funcs["selection_depth_sync_active"]() is True


def test_sync_toggle_reports_the_refused_state_not_the_requested_one(
    selection_controls_module,
):
    """The manager silently ignores the toggle while a drag is in flight.

    The toolbar must re-read the flag rather than assume its request landed --
    otherwise the icon claims a state the manager never entered.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    state.sync_write_ignored = True
    _panel, model, _doc = _mounted_panel(module, state)

    model.bound_events["selection_action"](model.handle, None, ["toggle_sync"])

    assert state.sync_calls == [True], "the request was still made"
    assert state.depth_sync is False, "the stub refused it"
    assert model.bound_funcs["selection_depth_sync_active"]() is False
    assert (
        model.bound_funcs["selection_depth_sync_icon"]()
        == module._SYNC_ICON_OFF["left"]
    )


def test_sync_off_icon_names_the_focused_panel_and_follows_it(
    selection_controls_module,
):
    """The OFF frame is a readout, not a constant.

    With sync off the two split panels hold separate depth windows and the
    sliders address exactly one of them. The OFF frame says which, by filling
    the half of the two-column glyph that stands for the focused panel -- so
    it has to move when focus moves, and it has to be DIRTIED when it moves or
    RmlUi keeps painting the stale frame. Focus is the same source of truth
    the chip beside it reads, so the two are asserted to agree.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    panel, model, doc = _mounted_panel(module, state)

    assert (
        model.bound_funcs["selection_depth_sync_icon"]()
        == module._SYNC_ICON_OFF["left"]
    )
    assert model.bound_funcs["selection_panel_chip_label"]() == "L"
    model.handle.dirty_calls.clear()

    state.focused_panel = "right"
    panel.update(doc)

    assert (
        model.bound_funcs["selection_depth_sync_icon"]()
        == module._SYNC_ICON_OFF["right"]
    ), "the OFF frame did not follow focus to the right panel"
    assert model.bound_funcs["selection_panel_chip_label"]() == "R"
    assert "selection_depth_sync_icon" in model.handle.dirty_calls, (
        "focus moved the OFF frame but never dirtied it, so the button would "
        "keep painting the panel that is no longer focused"
    )
    # The two OFF frames must be DIFFERENT files, and neither may be the ON
    # frame: three states of one button that render alike are no states at all.
    assert (
        len({*module._SYNC_ICON_OFF.values(), module._SYNC_ICON_ON}) == 3
    ), f"the sync toggle's three frames are not three distinct assets: {module._SYNC_ICON_OFF}"


def test_sync_on_uses_one_frame_whatever_the_focus_is(selection_controls_module):
    """With sync ON there is no focused half to point at.

    Both panels share one depth window, so the plain unfilled glyph is correct
    for either focus -- and a focus move must not smuggle a filled half back
    onto a button that is describing a shared window.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=True)
    panel, model, doc = _mounted_panel(module, state)

    assert model.bound_funcs["selection_depth_sync_icon"]() == module._SYNC_ICON_ON

    state.focused_panel = "right"
    panel.update(doc)

    assert model.bound_funcs["selection_depth_sync_icon"]() == module._SYNC_ICON_ON


def test_every_depth_toolbar_icon_names_an_asset_that_exists(selection_controls_module):
    """The frames are file paths, and a missing file renders as nothing at all.

    RmlUi does not fail loudly on an `img src` it cannot open, so a typo in
    any of these names is invisible until someone looks at the button. Each is
    resolved the way the RML resolves it -- relative to the RCSS/RML resource
    directory, whose `../icon/` is the asset folder. The five new icons use the
    canonical 40px exporter; the existing sync-ON icon remains 24px.
    """
    icons, _state = selection_controls_module
    assets = (
        Path(__file__).parent.parent.parent / "src" / "visualizer" / "gui" / "assets"
    )
    names = [
        *icons._VIZ_MODE_ICONS.values(),
        icons._SYNC_ICON_ON,
        *icons._SYNC_ICON_OFF.values(),
    ]
    expected_sizes = {
        "../icon/depth-show.png": (40, 40),
        "../icon/depth-dim.png": (40, 40),
        "../icon/depth-hide.png": (40, 40),
        "../icon/layout-columns-left.png": (40, 40),
        "../icon/layout-columns-right.png": (40, 40),
        "../icon/layout-columns.png": (24, 24),
    }
    assert set(names) == set(expected_sizes), names
    for name in names:
        assert name.startswith("../icon/"), name
        path = assets / name.removeprefix("../")
        assert path.is_file(), f"{name} names no file (looked at {path})"
        header = path.read_bytes()[:24]
        assert header[:8] == b"\x89PNG\r\n\x1a\n", f"{name} is not a PNG"
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
        expected_size = expected_sizes[name]
        assert (width, height) == expected_size, (
            f"{name} is {width}x{height}, expected {expected_size}"
        )
        if expected_size == (40, 40):
            source = path.parent / "src" / path.with_suffix(".svg").name
            assert source.is_file(), f"{name} has no canonical SVG source at {source}"
            assert not path.with_suffix(".svg").exists(), (
                f"{name} still has a duplicate source outside icon/src"
            )
            svg = ElementTree.parse(source).getroot()
            assert svg.get("viewBox") == "0 0 24 24", source
            colors = {
                element.attrib[attr]
                for element in svg.iter()
                for attr in ("stroke", "fill")
                if attr in element.attrib
            }
            assert "currentColor" in colors, f"{source} has no currentColor glyph"
            assert colors <= {"none", "currentColor"}, (
                f"{source} has hardcoded glyph colors: {colors}"
            )
    # The Dim frame must NOT be the file the selection toolbar's invert button
    # uses: they sit four seats apart in the same panel and were identical.
    invert = _read_src(
        "src", "visualizer", "gui", "rmlui", "resources", "viewport_overlay.rml"
    )
    assert '<img src="../icon/select-invert.png" />' in invert, (
        "the invert button no longer uses select-invert.png; this collision "
        "check is now checking nothing"
    )
    assert icons._VIZ_MODE_ICONS[1] != "../icon/select-invert.png", (
        "the viz-mode Dim frame is the invert button's icon again"
    )


def _png_alpha(path):
    """Decode a small RGBA PNG to a list of per-row alpha lists.

    Only what these small icons actually use: 8-bit RGBA, no interlace, the
    five standard row filters. Enough to compare two frames pixel for pixel
    without pulling an image library into the test requirements.
    """
    import struct
    import zlib

    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", path
    idat = b""
    width = height = None
    i = 8
    while i < len(data):
        length = struct.unpack(">I", data[i : i + 4])[0]
        kind = data[i + 4 : i + 8]
        chunk = data[i + 8 : i + 8 + length]
        if kind == b"IHDR":
            width, height, depth, color = struct.unpack(">IIBB", chunk[:10])
            assert (depth, color) == (8, 6), f"{path} is not 8-bit RGBA"
            assert chunk[12] == 0, f"{path} is interlaced"
        elif kind == b"IDAT":
            idat += chunk
        i += 12 + length
    raw = zlib.decompress(idat)
    bpp = 4
    stride = width * bpp
    rows = []
    previous = bytearray(stride)
    offset = 0
    for _ in range(height):
        filter_type = raw[offset]
        offset += 1
        line = bytearray(raw[offset : offset + stride])
        offset += stride
        for x in range(stride):
            left = line[x - bpp] if x >= bpp else 0
            up = previous[x]
            up_left = previous[x - bpp] if x >= bpp else 0
            if filter_type == 1:
                line[x] = (line[x] + left) & 0xFF
            elif filter_type == 2:
                line[x] = (line[x] + up) & 0xFF
            elif filter_type == 3:
                line[x] = (line[x] + (left + up) // 2) & 0xFF
            elif filter_type == 4:
                estimate = left + up - up_left
                d_left = abs(estimate - left)
                d_up = abs(estimate - up)
                d_up_left = abs(estimate - up_left)
                if d_left <= d_up and d_left <= d_up_left:
                    nearest = left
                elif d_up <= d_up_left:
                    nearest = up
                else:
                    nearest = up_left
                line[x] = (line[x] + nearest) & 0xFF
        rows.append([line[x * 4 + 3] for x in range(width)])
        previous = line
    return rows


def test_the_three_viz_frames_are_one_family_around_one_unchanging_box(
    selection_controls_module,
):
    """Off / Dim / Hide differ by COMPOSITION only, never by the box itself.

    The cycle's three frames all draw the same solid rounded square -- the
    depth window -- and say what happens outside it: a dotted frame (Off,
    everything outside still shown), a ring of specks (Dim), nothing at all
    (Hide). If one frame's box were rasterised at a different stroke weight or
    alpha the set would read as three unrelated glyphs instead of one control
    changing state, which is exactly the failure the eye / eye-slash pair had.

    The box occupies x=8..16 y=8..16 in the 24-unit source grid. At the 40px
    export size, a 20x20 crop at (10, 10) covers the same source region from
    (6, 6) to (18, 18), containing its stroke but none of the outside decoration.
    """
    icons, _state = selection_controls_module
    assets = (
        Path(__file__).parent.parent.parent / "src" / "visualizer" / "gui" / "assets"
    )

    frames = icons._VIZ_MODE_ICONS
    assert sorted(frames) == [0, 1, 2], frames
    assert len(set(frames.values())) == 3, (
        f"the viz cycle's three frames are not three distinct assets: {frames}"
    )
    for mode, name in frames.items():
        assert name.startswith("../icon/depth-"), (
            f"viz frame {mode} is {name}, which is outside the depth-* icon "
            "family the three frames must share"
        )
        assert not name.startswith("../icon/scene/"), (
            f"viz frame {mode} is back on a scene/ eye icon ({name}); the eye "
            "says nothing about the depth window"
        )

    boxes = {}
    for mode, name in frames.items():
        rows = _png_alpha(assets / name.removeprefix("../"))
        assert len(rows) == 40 and len(rows[0]) == 40, name
        boxes[mode] = tuple(tuple(row[10:30]) for row in rows[10:30])

    assert boxes[0] == boxes[1] == boxes[2], (
        "the three viz frames do not draw the same box: their 20x20 centres "
        "differ, so one frame's stroke weight or alpha is off and the frames "
        "differ by more than composition"
    )
    # And the box is really there -- an all-transparent centre would satisfy
    # the equality above while drawing nothing.
    assert max(max(row) for row in boxes[1]) == 255, (
        "the shared box has no full-strength stroke; it is not the same solid "
        "box the Dim frame established"
    )

    # Off and Hide are still distinguishable from Dim OUTSIDE that centre --
    # that is the whole information content of the cycle.
    outside = {}
    for mode, name in frames.items():
        rows = _png_alpha(assets / name.removeprefix("../"))
        total = sum(sum(row) for row in rows)
        outside[mode] = total - sum(sum(row) for row in boxes[mode])
    assert outside[2] == 0, (
        f"the Hide frame draws {outside[2]} of ink outside the box; it is the "
        "bare box and nothing else"
    )
    assert outside[0] > outside[1] > 0, (
        "the Off frame's dotted border must carry more ink than the Dim "
        f"frame's specks, and Dim more than nothing: {outside}"
    )


def test_focus_change_cancels_a_depth_text_edit_started_on_the_other_panel(
    selection_controls_module,
):
    """The edit-origin guard.

    Blur commits the buffer before it clears the edit state, so an edit typed
    against panel L that survives a switch to R would land L's number on R.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    panel, model, doc = _mounted_panel(module, state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")
    assert "selection_depth_near_str" in panel._editing_depth_text

    state.focused_panel = "right"
    panel.update(doc)

    # The edit is RETARGETED, not dropped -- the field is still
    # DOM-focused (no blur was ever observed), so it stays a live edit whose
    # origin is now the panel on screen.
    assert "selection_depth_near_str" in panel._editing_depth_text
    assert panel._depth_text_edit_panel["selection_depth_near_str"] == "right"
    before = state.depth_near
    canonical = float(f"{state.depth_near:.2f}")

    # Blur must actually write Right's canonical payload after retargeting, never Left's
    # text. Checking only unchanged final state would also pass the old blanket-swallow
    # behavior.
    writes_before = len(state.window_calls)
    doc.near.emit("blur")

    assert len(state.window_calls) == writes_before + 1, (
        "the foreign blur was swallowed instead of writing the canonical value"
    )
    assert state.window_calls[-1][1] == pytest.approx(canonical)
    assert state.depth_near == pytest.approx(before)
    assert state.depth_near != pytest.approx(3.75)


def test_focus_change_keeps_an_edit_started_on_the_panel_now_focused(
    selection_controls_module,
):
    """The guard cancels FOREIGN edits only; a same-panel refresh must not
    throw away what the user is typing."""
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    panel, model, doc = _mounted_panel(module, state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")

    # A refresh that does not change the focused panel.
    panel.update(doc)

    assert "selection_depth_near_str" in panel._editing_depth_text
    assert model.bound_binds["selection_depth_near_str"][0]() == "3.75"


def test_toolbar_sync_on_collapses_the_focused_panels_reference(
    selection_controls_module,
):
    """The toggle consumes the sync edge before update() can see it, so it must
    reconcile itself. Sync ON takes the focused panel's reference as shared instead of
    retaining a stale one.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    panel, model, _doc = _mounted_panel(module, state)

    panel._ref_scale_x["shared"] = 0.35
    panel._ref_scale_y["shared"] = 0.35
    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20

    model.bound_events["selection_action"](model.handle, None, ["toggle_sync"])

    assert panel._ref_key() == "shared"
    assert panel._ref_scale_x["shared"] == pytest.approx(0.60)
    assert panel._ref_scale_y["shared"] == pytest.approx(0.60)


def test_toolbar_sync_on_collapses_from_right_when_right_is_focused(
    selection_controls_module,
):
    module, state = selection_controls_module
    _independent_dual(state, focused="right", sync=False)
    panel, model, _doc = _mounted_panel(module, state)

    panel._ref_scale_x["shared"] = 0.35
    panel._ref_scale_y["shared"] = 0.35
    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20

    model.bound_events["selection_action"](model.handle, None, ["toggle_sync"])

    assert panel._ref_scale_x["shared"] == pytest.approx(0.20)
    assert panel._ref_scale_y["shared"] == pytest.approx(0.20)


def test_toolbar_sync_off_seeds_both_panel_references(selection_controls_module):
    """The same edge in the other direction, driven from the TOOLBAR."""
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=True)
    panel, model, _doc = _mounted_panel(module, state)

    panel._ref_scale_x["shared"] = 0.55
    panel._ref_scale_y["shared"] = 0.55
    panel._ref_scale_x["left"] = 0.11
    panel._ref_scale_y["left"] = 0.11
    panel._ref_scale_x["right"] = 0.99
    panel._ref_scale_y["right"] = 0.99

    model.bound_events["selection_action"](model.handle, None, ["toggle_sync"])

    assert panel._ref_key() == "left"
    assert panel._ref_scale_x["left"] == pytest.approx(0.55)
    assert panel._ref_scale_x["right"] == pytest.approx(0.55)
    assert panel._ref_scale_y["left"] == pytest.approx(0.55)
    assert panel._ref_scale_y["right"] == pytest.approx(0.55)


def test_toolbar_toggle_reads_the_actual_flag_before_toggling(
    selection_controls_module,
):
    """Refresh before toggling: an external flag change since the 100 ms poll makes an
    inverted cache a no-op request. Assert both reconciliations by value: pending ON
    takes focused Left's .60, not stale shared .35 or Right's .20; the click then turns
    OFF and seeds both panels from shared .60.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    panel, model, _doc = _mounted_panel(module, state)
    assert panel._depth_sync is False

    panel._ref_scale_x["shared"] = 0.35
    panel._ref_scale_y["shared"] = 0.35
    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20

    # An external channel flips it ON; no poll has run since.
    state.depth_sync = True

    model.bound_events["selection_action"](model.handle, None, ["toggle_sync"])

    assert state.sync_calls == [False], (
        "the click re-requested the value the manager already held"
    )
    assert state.depth_sync is False
    assert panel._depth_sync is False

    assert panel._ref_scale_x["shared"] == pytest.approx(0.60), (
        "the pending ON edge was not reconciled from the focused panel"
    )
    assert panel._ref_scale_y["shared"] == pytest.approx(0.60)
    assert panel._ref_key() == "left"
    assert panel._ref_scale_x["left"] == pytest.approx(0.60)
    assert panel._ref_scale_x["right"] == pytest.approx(0.60), (
        "the subsequent OFF edge did not seed both panels from the shared entry"
    )
    assert panel._ref_scale_y["right"] == pytest.approx(0.60)


def test_coalesced_focus_and_sync_edge_collapses_from_the_new_focus(
    selection_controls_module,
):
    """Regression.

    An external focus change L->R followed by sync ON, both before the next
    poll: setDepthWindowSync copied the panel focused AT SET TIME
    (rendering_manager.cpp:1270), i.e. Right. Reconciling from the CACHED Left
    would seed the shared reference from the wrong panel.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    panel, _model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20

    # Both edges land in the SAME refresh; the mode never changes.
    state.focused_panel = "right"
    state.depth_sync = True
    panel.update(doc)

    assert panel._depth_sync is True
    assert panel._ref_key() == "shared"
    assert panel._ref_scale_x["shared"] == pytest.approx(0.20), (
        "the collapse used the cached focus (Left) instead of the freshly read Right"
    )
    assert panel._ref_scale_y["shared"] == pytest.approx(0.20)


def test_leaving_independent_dual_collapses_the_pre_transition_focus(
    selection_controls_module,
):
    """Regression.

    The split service resets focus to Left when leaving independent-dual
    (split_view_service.cpp:214), so the post-transition focus is ALWAYS Left.
    Reading it would collapse Left's reference while the manager collapsed
    Right's window. The collapse must use the pre-transition panel.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="right", sync=False)
    panel, _model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20

    # Leave the mode exactly the way the service does it: mode off AND focus
    # reset to Left in the same frame, with the manager recording the panel it
    # actually collapsed.
    state.split_view_mode = "none"
    state.focused_panel = "left"
    state.depth_window_collapse_source = "right"
    # One collapse, exactly the one this poll observes.
    state.depth_window_collapse_generation = 1
    panel.update(doc)

    assert panel._ref_key() == "shared"
    assert panel._ref_scale_x["shared"] == pytest.approx(0.20), (
        "the collapse used the post-transition focus (Left) instead of Right"
    )
    assert panel._ref_scale_y["shared"] == pytest.approx(0.20)


def test_reedit_after_a_cancel_still_commits_on_linebreak(selection_controls_module):
    """Regression: blur, THEN a fresh edit of the same field.

    A linebreak commit does not blur (rml_widgets.py:159), so a cancel must
    leave nothing behind that could eat the later legitimate write.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    panel, model, doc = _mounted_panel(module, state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")

    # Focus moves to R: the L-origin edit is retargeted and the field reverted.
    state.focused_panel = "right"
    panel.update(doc)

    # The foreign blur arrives. The cancel reverted the field in
    # place, so this commit is not swallowed -- it writes the canonical value,
    # which is idempotent. The stale 3.75 never reaches the native side.
    # Prove the write happened AND its payload.
    writes_before = len(state.window_calls)
    before = state.depth_near
    canonical = float(f"{state.depth_near:.2f}")
    doc.near.emit("blur")
    assert len(state.window_calls) == writes_before + 1, (
        "the foreign blur was swallowed instead of writing the canonical value"
    )
    assert state.window_calls[-1][1] == pytest.approx(canonical)
    assert state.depth_near == pytest.approx(before)
    assert state.depth_near != pytest.approx(3.75)

    # Now the user edits the SAME field on the panel now focused and commits
    # with a linebreak, which never blurs. That write must land.
    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("4.25")
    doc.near.emit("change", _InputEventStub(linebreak=True))

    assert state.depth_near == pytest.approx(4.25)
    assert len(state.window_calls) > writes_before


def test_commit_lands_when_focus_changed_without_blur_or_a_new_focus_event(
    selection_controls_module,
):
    """Operator/MCP/project-restore focus changes send no pointer blur or refocus. After
    the retained field is reverted in place, continued typing and Enter must commit
    without a leftover refusal flag.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    panel, model, doc = _mounted_panel(module, state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")

    # Focus moves to R with NO blur and NO second focus event.
    state.focused_panel = "right"
    panel.update(doc)

    # The cancel reverted the field to the panel now on screen.
    canonical = f"{state.depth_near:.2f}"
    assert panel._depth_text_bufs["selection_depth_near_str"] == canonical

    # The user carries on typing in the same, still-DOM-focused field.
    model.bound_binds["selection_depth_near_str"][1]("4.25")

    # Poll between typing and Enter, as the overlay does every 100 ms. If retargeting
    # drops live-edit membership, _sync_depth_text_bufs overwrites 4.25 with canonical
    # text here.
    panel.update(doc)
    assert panel._depth_text_bufs["selection_depth_near_str"] == "4.25"

    doc.near.emit("change", _InputEventStub(linebreak=True))

    assert state.depth_near == pytest.approx(4.25)


def test_a_second_focus_change_retargets_the_retained_field_again(
    selection_controls_module,
):
    """Keep the retained edit registered after retargeting. A second external focus
    change must revert it again and update its origin to the displayed panel.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    panel, model, doc = _mounted_panel(module, state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")

    state.focused_panel = "right"
    panel.update(doc)
    assert panel._depth_text_edit_panel["selection_depth_near_str"] == "right"

    # The user types again on the retained field, then focus moves BACK.
    model.bound_binds["selection_depth_near_str"][1]("9.50")
    state.focused_panel = "left"
    panel.update(doc)

    assert "selection_depth_near_str" in panel._editing_depth_text, (
        "the second focus change found no live edit to guard"
    )
    assert panel._depth_text_edit_panel["selection_depth_near_str"] == "left"
    canonical = f"{state.depth_near:.2f}"
    assert panel._depth_text_bufs["selection_depth_near_str"] == canonical

    # And the twice-stale 9.50 can no longer reach the native side.
    before = state.depth_near
    doc.near.emit("blur")
    assert state.depth_near == pytest.approx(before)
    assert state.depth_near != pytest.approx(9.50)


def test_a_toolbar_action_retargets_the_active_edit_whose_edge_it_consumes(
    selection_controls_module,
):
    """_on_action refreshes context itself. It must also retarget the edit when it
    consumes a focus edge, or the following blur writes Left's stale text into Right.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    # Use DISTINCT per-panel canonical values, so "the retarget read
    # the right panel" is provable rather than order-insensitive.
    state.panel_ranges = {"left": (1.00, 9.00), "right": (2.50, 8.00)}
    panel, model, doc = _mounted_panel(module, state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")
    assert panel._depth_text_edit_panel["selection_depth_near_str"] == "left"
    assert panel._depth_text_bufs["selection_depth_near_str"] == "3.75"

    # Focus moves externally, and a toolbar action -- NOT a poll -- is what next
    # reads the native context.
    state.focused_panel = "right"
    model.bound_events["selection_action"](model.handle, None, ["undo"])

    assert panel._depth_text_edit_panel["selection_depth_near_str"] == "right", (
        "the toolbar action consumed the focus edge without retargeting the edit"
    )
    canonical = "2.50"
    assert panel._depth_text_bufs["selection_depth_near_str"] == canonical, (
        "the field reverted to Left's canonical value, not the panel now on screen"
    )

    writes_before = len(state.window_calls)
    doc.near.emit("blur")

    assert len(state.window_calls) == writes_before + 1
    assert state.window_calls[-1][1] == pytest.approx(float(canonical))
    # WHICH panel received it, not merely that a write happened.
    assert state.write_targets[-1] == "right"
    assert state.panel_ranges["right"][0] == pytest.approx(2.50)
    assert state.panel_ranges["left"][0] == pytest.approx(1.00), (
        "the blur wrote into the panel the edit STARTED on"
    )
    assert state.panel_ranges["right"][0] != pytest.approx(3.75)


def test_the_sync_toggle_retargets_the_active_edit_whose_edge_it_consumes(
    selection_controls_module,
):
    """The same defect through the toggle's two refreshes."""
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    panel, model, doc = _mounted_panel(module, state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")

    state.focused_panel = "right"
    model.bound_events["selection_action"](model.handle, None, ["toggle_sync"])

    assert state.depth_sync is True
    assert panel._depth_text_edit_panel["selection_depth_near_str"] == "right", (
        "the sync toggle consumed the focus edge without retargeting the edit"
    )
    # Sync ON leaves ONE window, so the canonical value is the global one.
    canonical = f"{state.depth_near:.2f}"
    assert panel._depth_text_bufs["selection_depth_near_str"] == canonical
    assert canonical != "3.75"

    before = state.depth_near
    writes_before = len(state.window_calls)
    doc.near.emit("blur")

    # The blur must actually EMIT the canonical write, and it
    # must fan out to both slots the way a synced write does.
    assert len(state.window_calls) == writes_before + 1, (
        "the blur was swallowed instead of writing the canonical value"
    )
    assert state.window_calls[-1][1] == pytest.approx(float(canonical))
    assert state.write_targets[-1] == "both", (
        "a synced write must fan out to both slots, not address one panel"
    )
    assert state.depth_near == pytest.approx(before)
    assert state.depth_near != pytest.approx(3.75)


def test_a_commit_before_the_next_poll_retargets_the_active_edit(
    selection_controls_module,
):
    """Enter or blur may precede any poll of a focus change. Validate the edit origin
    against fresh context at commit time and write canonical text on a mismatch, never
    the stale-origin buffer.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    state.panel_ranges = {"left": (1.00, 9.00), "right": (2.50, 8.00)}
    panel, model, doc = _mounted_panel(module, state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")

    canonical = 2.50

    # No update(), no toolbar action: focus changes and the user hits Enter.
    state.focused_panel = "right"
    writes_before = len(state.window_calls)
    doc.near.emit("change", _InputEventStub(linebreak=True))

    assert panel._depth_text_edit_panel["selection_depth_near_str"] == "right"
    assert len(state.window_calls) == writes_before + 1
    assert state.window_calls[-1][1] == pytest.approx(canonical), (
        "the commit wrote the stale-origin text into the panel now focused"
    )
    assert state.write_targets[-1] == "right"
    assert state.panel_ranges["right"][0] == pytest.approx(canonical)
    assert state.panel_ranges["left"][0] == pytest.approx(1.00), (
        "the commit landed in the panel the edit started on"
    )
    assert state.panel_ranges["right"][0] != pytest.approx(3.75)


def test_a_retyped_stale_string_after_a_retarget_still_commits(
    selection_controls_module,
):
    """After retargeting reverts a field, explicitly retyping the same stale string is a
    legitimate edit. No text-keyed refusal may survive to reject it.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    panel, model, doc = _mounted_panel(module, state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")
    assert f"{state.depth_near:.2f}" != "3.75", (
        "the stale string must differ from canonical for this test to mean anything"
    )

    state.focused_panel = "right"
    panel.update(doc)
    assert panel._depth_text_bufs["selection_depth_near_str"] == f"{state.depth_near:.2f}"

    # The user types the same number again, on the panel now on screen.
    model.bound_binds["selection_depth_near_str"][1]("3.75")
    doc.near.emit("change", _InputEventStub(linebreak=True))

    assert state.depth_near == pytest.approx(3.75)
    assert state.window_calls[-1][1] == pytest.approx(3.75)


def test_coalesced_focus_change_and_mode_leave_uses_the_native_collapse_source(
    selection_controls_module,
):
    """With cached Left focus, native focus moves Right and leaves independent-dual
    before a poll. The manager collapses Right but service focus resets Left; use the
    native collapse source, since cached Left's .60 describes the wrong surviving
    window.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    panel, _model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20
    assert panel._focused_panel == "left"

    # Focus L->R, then the mode leaves -- one poll sees both.
    state.split_view_mode = "none"
    state.focused_panel = "left"  # the service's reset
    state.depth_window_collapse_source = "right"
    # One collapse, exactly the one this poll observes.
    state.depth_window_collapse_generation = 1
    panel.update(doc)

    assert panel._ref_key() == "shared"
    assert panel._ref_scale_x["shared"] == pytest.approx(0.20), (
        "the collapse seeded from the stale cached focus (Left) instead of the "
        "panel the manager actually collapsed (Right)"
    )
    assert panel._ref_scale_y["shared"] == pytest.approx(0.20)


def test_focus_then_sync_on_then_mode_leave_uses_the_native_collapse_source(
    selection_controls_module,
):
    """Focus, sync ON and mode leave coalesce into one poll. The final leave determines
    the surviving window, so its recorded source wins.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    panel, _model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20

    state.depth_sync = True
    state.split_view_mode = "none"
    state.focused_panel = "left"
    state.depth_window_collapse_source = "right"
    # One collapse, exactly the one this poll observes.
    state.depth_window_collapse_generation = 1
    panel.update(doc)

    assert panel._ref_key() == "shared"
    assert panel._ref_scale_x["shared"] == pytest.approx(0.20)
    assert panel._ref_scale_y["shared"] == pytest.approx(0.20)


def test_a_hidden_leave_enter_leave_cycle_resyncs_from_the_native_window(
    selection_controls_module,
):
    """Start with references L=.60/R=.20 and Right focused. A hidden leave collapses
    Right; re-entry seeds both slots from .20 and resets focus Left; another leave
    records Left. Two collapse generations expose the missed cycle. Fresh-baseline from
    the surviving native .20 instead of cached Left's obsolete .60.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="right", sync=False)
    panel, _model, doc = _mounted_panel(module, state)
    assert panel._collapse_generation == 0

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20

    # The whole leave -> enter -> leave cycle lands between two polls.
    state.split_view_mode = "none"
    state.focused_panel = "left"
    state.depth_scale = 0.20
    state.depth_scale_y = 0.20
    state.depth_window_collapse_source = "left"
    state.depth_window_collapse_generation = 2
    panel.update(doc)

    assert panel._ref_key() == "shared"
    assert panel._ref_scale_x["shared"] != pytest.approx(0.60), (
        "the shared reference was seeded from the final source's cached entry, "
        "which the first collapse of the cycle had already invalidated"
    )
    assert panel._ref_scale_x["shared"] == pytest.approx(0.20)
    assert panel._ref_scale_y["shared"] == pytest.approx(0.20)


def test_a_hidden_cycle_resync_trusts_no_cached_entry_at_all(
    selection_controls_module,
):
    """Use surviving native .20 against cached .60/.45, so neither cache entry can
    accidentally satisfy the hidden-cycle check. Only a fresh native baseline produces
    the expected reference.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="right", sync=False)
    panel, _model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.45
    panel._ref_scale_y["right"] = 0.45
    panel._ref_scale_x["shared"] = 0.90
    panel._ref_scale_y["shared"] = 0.90

    state.split_view_mode = "none"
    state.focused_panel = "left"
    state.depth_scale = 0.20
    state.depth_scale_y = 0.20
    state.depth_window_collapse_source = "left"
    state.depth_window_collapse_generation = 3
    panel.update(doc)

    assert panel._ref_scale_x["shared"] == pytest.approx(0.20)
    assert panel._ref_scale_y["shared"] == pytest.approx(0.20)


def test_a_single_observed_leave_still_uses_the_collapse_source(
    selection_controls_module,
):
    """One observed leave and one collapse preserve the named source's reference.

    This ordinary path must not discard a valid cached reference through unnecessary
    fresh-baseline recovery.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="right", sync=False)
    panel, _model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20

    state.split_view_mode = "none"
    state.focused_panel = "left"
    # The native window really is Right's, but a DIFFERENT scale from the
    # reference: a fresh-baseline resync here would wrongly discard the
    # reference the user's shape is measured against.
    state.depth_scale = 0.31
    state.depth_scale_y = 0.31
    state.depth_window_collapse_source = "right"
    state.depth_window_collapse_generation = 1
    panel.update(doc)

    assert panel._ref_scale_x["shared"] == pytest.approx(0.20), (
        "an ordinary single leave was treated as a missed-boundary resync"
    )


def test_a_coalesced_focus_change_and_leave_never_commits_stale_origin_text(
    selection_controls_module,
):
    """Edit Left, focus Right and leave independent-dual before a poll. Right's window
    survives but service focus resets Left, hiding the change from a focus-only guard.
    The mode boundary must revert and retarget the field; stale 3.75 must never reach
    native state.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    state.panel_ranges = {"left": (1.00, 9.00), "right": (2.50, 8.00)}
    panel, model, doc = _mounted_panel(module, state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")
    assert panel._depth_text_edit_panel["selection_depth_near_str"] == "left"

    # Native: focus L -> R, then the leave. The manager folded Right's window
    # into the single global one; the service reset the focus to Left.
    state.split_view_mode = "none"
    state.focused_panel = "left"
    state.depth_near, state.depth_far = 2.50, 8.00
    state.depth_window_collapse_source = "right"
    state.depth_window_collapse_generation = 1
    panel.update(doc)

    # The field now shows the POST-transition canonical value, not 3.75.
    assert panel._depth_text_bufs["selection_depth_near_str"] == "2.50"

    writes_before = len(state.window_calls)
    doc.near.emit("change", _InputEventStub(linebreak=True))

    assert len(state.window_calls) == writes_before + 1
    assert state.window_calls[-1][1] == pytest.approx(2.50), (
        "the stale-origin 3.75 landed in the window collapsed from Right"
    )
    assert state.write_targets[-1] == "global"
    assert state.depth_near == pytest.approx(2.50)
    assert state.depth_near != pytest.approx(3.75)


def test_a_coalesced_focus_sync_on_and_leave_never_commits_stale_origin_text(
    selection_controls_module,
):
    """The same reproduction through the focus -> sync-ON -> leave ordering."""
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    state.panel_ranges = {"left": (1.00, 9.00), "right": (2.50, 8.00)}
    panel, model, doc = _mounted_panel(module, state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")

    state.depth_sync = True
    state.split_view_mode = "none"
    state.focused_panel = "left"
    state.depth_near, state.depth_far = 2.50, 8.00
    state.depth_window_collapse_source = "right"
    state.depth_window_collapse_generation = 1
    panel.update(doc)

    assert panel._depth_text_bufs["selection_depth_near_str"] == "2.50"

    writes_before = len(state.window_calls)
    doc.near.emit("blur")

    assert len(state.window_calls) == writes_before + 1
    assert state.window_calls[-1][1] == pytest.approx(2.50)
    assert state.depth_near != pytest.approx(3.75)


def test_a_hidden_leave_then_reenter_reconciles_although_the_endpoint_is_unchanged(
    selection_controls_module,
):
    """With L=.60/R=.20 and Right focused, a hidden leave collapses .20; re-entry seeds
    both slots and resets focus Left. The endpoint remains independent-dual, so only
    lineage exposes the lost Left reference. Keeping .60 would show 33% and mis-scale
    later Size edits.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="right", sync=False)
    panel, _model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20

    # The hidden leave -> enter cycle. Endpoint: still independent-dual.
    state.focused_panel = "left"
    state.depth_scale = 0.20
    state.depth_scale_y = 0.20
    state.depth_window_collapse_source = "right"
    state.depth_window_collapse_kind = "leave_collapse"
    state.depth_window_collapse_generation = 1
    panel.update(doc)

    assert panel._split_mode == "independent_dual"
    assert panel._ref_key() == "left"
    assert panel._ref_scale_x["left"] == pytest.approx(0.20), (
        "the stale pre-collapse reference survived a cycle the endpoint hid"
    )
    assert panel._ref_scale_x["right"] == pytest.approx(0.20)
    assert panel._ref_scale_x["shared"] == pytest.approx(0.20)
    assert panel._scale_percent() == 100


def test_a_hidden_sync_cycle_resyncs_from_the_native_window(
    selection_controls_module,
):
    """Hidden ON/OFF/focus/ON performs two copies while the poll sees one
    unsynced-to-synced edge. No cached reference survives both; fresh-baseline shared
    from the native window.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="right", sync=False)
    panel, _model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.45
    panel._ref_scale_y["right"] = 0.45

    state.depth_sync = True
    state.focused_panel = "left"
    state.depth_scale = 0.20
    state.depth_scale_y = 0.20
    state.depth_window_collapse_source = "left"
    state.depth_window_collapse_kind = "sync_copy"
    state.depth_window_collapse_generation = 2
    panel.update(doc)

    assert panel._ref_key() == "shared"
    assert panel._ref_scale_x["shared"] == pytest.approx(0.20), (
        "the shared reference was seeded from a cached entry that two hidden "
        "sync copies had already invalidated"
    )
    assert panel._ref_scale_y["shared"] == pytest.approx(0.20)


def test_a_single_sync_copy_seeds_from_the_recorded_source_panel(
    selection_controls_module,
):
    """Exactly one sync copy preserves its source panel's cached reference. Give native
    state a third value so fresh-baselining cannot accidentally pass this control.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="right", sync=False)
    panel, _model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20

    state.depth_sync = True
    state.depth_scale = 0.90
    state.depth_scale_y = 0.90
    state.depth_window_collapse_source = "right"
    state.depth_window_collapse_kind = "sync_copy"
    state.depth_window_collapse_generation = 1
    panel.update(doc)

    assert panel._ref_scale_x["shared"] == pytest.approx(0.20)
    assert panel._ref_scale_x["left"] == pytest.approx(0.20)
    assert panel._ref_scale_x["right"] == pytest.approx(0.20)


def test_a_project_restore_while_independent_fresh_baselines_the_references(
    selection_controls_module,
):
    """Project restore seeds both slots but can leave mode, focus and sync unchanged.
    Its lineage stamp must fresh-baseline all references from restored state rather than
    the previous project.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    panel, _model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.45
    panel._ref_scale_y["right"] = 0.45
    panel._ref_scale_x["shared"] = 0.90
    panel._ref_scale_y["shared"] = 0.90

    state.depth_scale = 0.35
    state.depth_scale_y = 0.35
    state.depth_window_collapse_source = "left"
    state.depth_window_collapse_kind = "project_restore"
    state.depth_window_collapse_generation = 1
    panel.update(doc)

    assert panel._split_mode == "independent_dual"
    assert panel._depth_sync is False
    for key in ("shared", "left", "right"):
        assert panel._ref_scale_x[key] == pytest.approx(0.35), key
        assert panel._ref_scale_y[key] == pytest.approx(0.35), key


def test_a_hidden_sync_undo_redo_pair_fresh_baselines_the_references(
    selection_controls_module,
):
    """Sync undo/redo restores absolute windows and invalidates cached references. Both
    can occur between polls with no endpoint edge; project_restore lineage stamps are
    the only witness. The kind denotes fresh-baseline semantics, not necessarily a
    project load.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=True)
    state.depth_window_collapse_generation = 5
    state.depth_window_collapse_kind = "sync_copy"
    panel, _model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.45
    panel._ref_scale_y["right"] = 0.45
    panel._ref_scale_x["shared"] = 0.90
    panel._ref_scale_y["shared"] = 0.90

    # The hidden undo, then the hidden redo -- two stamps, flag back where it
    # was.
    state.depth_scale = 0.35
    state.depth_scale_y = 0.35
    state.depth_window_collapse_kind = "project_restore"
    state.depth_window_collapse_generation = 7
    panel.update(doc)

    assert panel._depth_sync is True, "the pair left the flag where it started"
    for key in ("shared", "left", "right"):
        assert panel._ref_scale_x[key] == pytest.approx(0.35), key
        assert panel._ref_scale_y[key] == pytest.approx(0.35), key


def test_a_sync_undo_restoring_distinct_slots_baselines_each_panel_from_its_own(
    selection_controls_module,
):
    """A single sync undo stamps project_restore and restores L=.60/R=.20.

    The endpoint stays independent and unsynced. Baseline each panel from its own slot
    and shared from the focused projection. Using Right's .20 for Left would show 300%
    and mis-scale later edits; both panels must show 100%.
    """
    module, state = selection_controls_module
    # The state a sync undo of {L=.60, R=.20} leaves behind, with Right focused.
    _independent_dual(state, focused="right", sync=False)
    state.panel_scales = {"left": (0.60, 0.60), "right": (0.20, 0.20)}
    state.depth_scale = 0.20
    state.depth_scale_y = 0.20
    panel, model, doc = _mounted_panel(module, state)

    # Everything cached predates the restore, including a shared entry that
    # matches NEITHER slot.
    panel._ref_scale_x["left"] = 0.45
    panel._ref_scale_y["left"] = 0.45
    panel._ref_scale_x["right"] = 0.45
    panel._ref_scale_y["right"] = 0.45
    panel._ref_scale_x["shared"] = 0.90
    panel._ref_scale_y["shared"] = 0.90

    # ONE stamp: the undo itself. No mode edge, no sync edge, no focus edge.
    state.depth_window_collapse_source = "right"
    state.depth_window_collapse_kind = "project_restore"
    state.depth_window_collapse_generation += 1
    panel.update(doc)

    assert panel._ref_scale_x["left"] == pytest.approx(0.60)
    assert panel._ref_scale_y["left"] == pytest.approx(0.60)
    assert panel._ref_scale_x["right"] == pytest.approx(0.20)
    assert panel._ref_scale_y["right"] == pytest.approx(0.20)
    # The shared entry is the focused projection, which is Right's window here.
    assert panel._ref_scale_x["shared"] == pytest.approx(0.20)
    assert panel._ref_scale_y["shared"] == pytest.approx(0.20)

    # What the user actually sees: 100% in BOTH panels.
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"
    state.focused_panel = "left"
    state.depth_scale = 0.60
    state.depth_scale_y = 0.60
    panel.update(doc)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"


def test_a_stamp_landing_between_the_endpoint_and_record_reads_is_revalidated(
    selection_controls_module,
):
    """A sync undo between endpoint and lineage reads restores L=.60/R=.20, sync=false
    and project_restore while the cached endpoint still says sync=true. Consuming that
    mix seeds all references from focused .20 and leaves Left at 300% after the next
    poll. Read record/endpoint/record and retry the whole set if generation changes.
    """
    module, state = selection_controls_module
    # Cached endpoint: independent-dual and SYNCED, Right focused.
    _independent_dual(state, focused="right", sync=True)
    state.depth_scale = 0.20
    state.depth_scale_y = 0.20
    panel, model, doc = _mounted_panel(module, state)

    # Everything cached predates the restore, shared matching NEITHER slot.
    for key, value in (("left", 0.45), ("right", 0.45), ("shared", 0.90)):
        panel._ref_scale_x[key] = value
        panel._ref_scale_y[key] = value

    # The sync undo lands EXACTLY between the endpoint read and the record
    # read: get_depth_window_sync returns the stale pre-restore true, and the
    # restore + stamp happen immediately afterwards.
    landed = []

    def _torn_sync():
        stale = state.depth_sync
        if not landed:
            landed.append(True)
            state.panel_scales = {"left": (0.60, 0.60), "right": (0.20, 0.20)}
            state.depth_sync = False
            state.depth_window_collapse_source = "right"
            state.depth_window_collapse_kind = "project_restore"
            state.depth_window_collapse_generation += 1
        return stale

    module.lf.ui.get_depth_window_sync = _torn_sync
    panel.update(doc)

    # The revalidation retried, so the whole set describes the post-restore
    # instant: each panel entry is baselined from ITS OWN slot.
    assert panel._depth_sync is False
    assert panel._ref_scale_x["left"] == pytest.approx(0.60)
    assert panel._ref_scale_y["left"] == pytest.approx(0.60)
    assert panel._ref_scale_x["right"] == pytest.approx(0.20)
    assert panel._ref_scale_y["right"] == pytest.approx(0.20)
    assert panel._ref_scale_x["shared"] == pytest.approx(0.20)
    assert panel._ref_scale_y["shared"] == pytest.approx(0.20)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"

    # And the NEXT poll's sync-OFF edge must not corrupt them: the edge was
    # already consumed by the refresh that reconciled the stamp.
    panel.update(doc)
    assert panel._ref_scale_x["left"] == pytest.approx(0.60)
    assert panel._ref_scale_x["right"] == pytest.approx(0.20)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"
    state.focused_panel = "left"
    state.depth_scale = 0.60
    state.depth_scale_y = 0.60
    panel.update(doc)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"


def test_an_exhausted_revalidation_consumes_nothing_and_heals_next_poll(
    selection_controls_module,
):
    """Land sync Undo/Redo/Undo between endpoint and second-record reads, one stamped
    absolute restore per attempt. The last torn set says sync=true while native state is
    distinct L=.60/R=.20, sync=false; consuming it would leave Left at 300%. Exhaustion
    must consume no endpoint, lineage, references or retarget. The next stable poll sees
    the full delta and baselines each panel from its own slot.
    """
    module, state = selection_controls_module
    # LIVE (and therefore cached) start: independent-dual, sync ON over two
    # EQUAL windows -- a genuinely synced state. Starting here is what makes the
    # storm's first Undo (to distinct/off) a real transition rather than a
    # re-assignment of the values already in place, so all three legs below move
    # the world.
    _independent_dual(state, focused="right", sync=True)
    state.panel_scales = {"left": (0.20, 0.20), "right": (0.20, 0.20)}
    state.depth_scale = 0.20
    state.depth_scale_y = 0.20
    panel, model, doc = _mounted_panel(module, state)

    # Everything cached predates the storm, shared matching NEITHER slot.
    for key, value in (("left", 0.45), ("right", 0.45), ("shared", 0.90)):
        panel._ref_scale_x[key] = value
        panel._ref_scale_y[key] = value
    consumed = panel._collapse_generation

    reads = []
    settled_record = module.lf.ui.get_depth_window_collapse_record

    def _storming_record():
        # THREE mutations, one per attempt, each landing between that attempt's
        # endpoint read and its second record read -- the Undo -> Redo -> Undo
        # the user is holding down. Each leg actually rewrites both absolute
        # windows and the sync flag, the way a restore does, and stamps once.
        reads.append(1)
        if len(reads) % 2 == 0:
            leg = len(reads) // 2
            if leg % 2 == 1:
                # Undo: sync off, the two distinct windows restored.
                state.panel_scales = {"left": (0.60, 0.60), "right": (0.20, 0.20)}
                state.depth_sync = False
            else:
                # Redo: the sync copy back on, both windows equal again.
                state.panel_scales = {"left": (0.20, 0.20), "right": (0.20, 0.20)}
                state.depth_sync = True
            state.depth_window_collapse_source = "right"
            state.depth_window_collapse_kind = "project_restore"
            state.depth_window_collapse_generation += 1
        return (
            state.depth_window_collapse_source,
            state.depth_window_collapse_generation,
            state.depth_window_collapse_kind,
        )

    module.lf.ui.get_depth_window_collapse_record = _storming_record
    panel.update(doc)

    # Bounded: two reads per attempt, three attempts, and then it gives up.
    assert len(reads) == 2 * module._CONTEXT_READ_ATTEMPTS
    # Three legs, so the pending delta is 3 -- SMALLER than the six stamps a
    # read-count-driven implementation would have produced.
    assert state.depth_window_collapse_generation - consumed == 3
    # Undo/Redo/Undo went equal/on -> distinct/off -> equal/on -> distinct/off. The last
    # torn endpoint still says sync=true over native L=.60/R=.20, sync=false. No
    # endpoint, generation or reference may be consumed; the entire cache stays
    # pre-storm.
    assert state.depth_sync is False
    assert state.panel_scales == {"left": (0.60, 0.60), "right": (0.20, 0.20)}
    assert panel._depth_sync is True, (
        "the exhausted tick consumed the torn endpoint instead of deferring it"
    )
    assert panel._split_mode == "independent_dual"
    assert panel._focused_panel == "right"
    for key, value in (("left", 0.45), ("right", 0.45), ("shared", 0.90)):
        assert panel._ref_scale_x[key] == pytest.approx(value)
        assert panel._ref_scale_y[key] == pytest.approx(value)
    assert panel._collapse_generation == consumed

    # The storm ends. The world it left behind: sync off, two distinct
    # absolute windows, the restore stamp still unconsumed.
    module.lf.ui.get_depth_window_collapse_record = settled_record
    assert state.depth_sync is False
    panel.update(doc)

    # The pending delta is seen against a stable set, the deferred sync-OFF edge
    # is consumed there, and each panel entry is baselined from ITS OWN slot.
    assert panel._depth_sync is False
    assert panel._collapse_generation == state.depth_window_collapse_generation
    assert panel._ref_scale_x["left"] == pytest.approx(0.60)
    assert panel._ref_scale_y["left"] == pytest.approx(0.60)
    assert panel._ref_scale_x["right"] == pytest.approx(0.20)
    assert panel._ref_scale_y["right"] == pytest.approx(0.20)
    assert panel._ref_scale_x["shared"] == pytest.approx(0.20)
    assert panel._ref_scale_y["shared"] == pytest.approx(0.20)

    # What the user actually sees: 100% in BOTH panels.
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"
    state.focused_panel = "left"
    state.depth_scale = 0.60
    state.depth_scale_y = 0.60
    panel.update(doc)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"


def test_an_exhausted_leave_neither_canonicalizes_nor_commits_the_size_field(
    selection_controls_module,
):
    """Consuming an endpoint without its lineage can retarget Size against a stale
    reference: native .20/shared .90 displays 22%, then a second exhausted commit writes
    it. An exhausted refresh must preserve mode, edit buffer and references, and defer
    the commit without writing.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="right", sync=False)
    state.panel_scales = {"left": (0.60, 0.60), "right": (0.20, 0.20)}
    state.depth_scale = 0.20
    state.depth_scale_y = 0.20
    panel, model, doc = _mounted_panel(module, state)

    # The shared entry is the stale one -- .90 against a native .20 reads as
    # 22%. The focused panel's own entry is correct, so as long as nothing
    # consumes the leave the field must keep reading 100%.
    panel._ref_scale_x["left"] = panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = panel._ref_scale_y["right"] = 0.20
    panel._ref_scale_x["shared"] = panel._ref_scale_y["shared"] = 0.90

    # The user is mid-edit in the Size field.
    doc.scale.emit("focus")
    model.bound_binds["selection_depth_scale_str"][1]("150")
    assert panel._depth_text_edit_panel["selection_depth_scale_str"] == "right"

    reads = []
    settled_record = module.lf.ui.get_depth_window_collapse_record

    def _storming_record():
        reads.append(1)
        if len(reads) == 1:
            # The LEAVE every endpoint read below observes: Right's window
            # folds into the single global one and focus resets to Left.
            state.split_view_mode = "disabled"
            state.focused_panel = "left"
            state.depth_window_collapse_source = "right"
            state.depth_window_collapse_kind = "leave_collapse"
            state.depth_window_collapse_generation += 1
        elif len(reads) % 2 == 0:
            # A further stamp between this attempt's endpoint read and its
            # second record read, so no attempt ever sees a stable set.
            state.depth_window_collapse_kind = "project_restore"
            state.depth_window_collapse_generation += 1
        return (
            state.depth_window_collapse_source,
            state.depth_window_collapse_generation,
            state.depth_window_collapse_kind,
        )

    module.lf.ui.get_depth_window_collapse_record = _storming_record
    panel.update(doc)

    assert len(reads) == 2 * module._CONTEXT_READ_ATTEMPTS
    # Nothing consumed: the leave is still pending, so _ref_key() still names
    # the focused panel and the canonical text is NOT computed from the stale
    # shared .90.
    canonical = panel._canonical_depth_text_value("selection_depth_scale_str")
    assert canonical != "22%", (
        "the exhausted tick canonicalized Size from the unreconciled reference"
    )
    assert canonical == "100%"
    assert panel._split_mode == "independent_dual"
    assert panel._focused_panel == "right"
    # The guard never fired, so the user's keystrokes are untouched.
    assert panel._depth_text_bufs["selection_depth_scale_str"] == "150"
    assert "selection_depth_scale_str" in panel._editing_depth_text

    # The user hits Enter while the storm is still running. The commit's own
    # validating refresh exhausts too, so it must write NOTHING.
    writes_before = len(state.window_calls)
    panel_writes_before = len(state.panel_writes)
    scales_before = dict(state.panel_scales)
    doc.scale.emit("change", _InputEventStub(linebreak=True))

    assert len(state.window_calls) == writes_before, (
        "a commit was applied against a torn snapshot"
    )
    assert len(state.panel_writes) == panel_writes_before
    assert state.panel_scales == scales_before
    assert state.depth_scale == pytest.approx(0.20)
    # Deferred, not dropped: the buffer and the live-edit state both survive.
    assert panel._depth_text_bufs["selection_depth_scale_str"] == "150"
    assert "selection_depth_scale_str" in panel._editing_depth_text

    # The storm stops. The next poll reads a stable world, consumes the leave,
    # and the single global window baselines the shared entry from itself.
    module.lf.ui.get_depth_window_collapse_record = settled_record
    panel.update(doc)

    assert panel._split_mode == "disabled"
    assert panel._ref_scale_x["shared"] == pytest.approx(0.20)
    assert panel._ref_scale_y["shared"] == pytest.approx(0.20)
    assert panel._canonical_depth_text_value("selection_depth_scale_str") == "100%"
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"

    # The stable poll retries the deferred commit without another Enter. Its
    # mode-boundary retarget first restores canonical text, so the write is an
    # idempotent 100% of the reconciled reference, never the torn snapshot's 22%.
    assert panel._depth_text_bufs["selection_depth_scale_str"] == "100%"
    assert len(state.window_calls) == writes_before + 1
    assert state.depth_scale == pytest.approx(0.20)
    assert "selection_depth_scale_str" not in panel._deferred_depth_commits
    # The edit is still live -- no blur ever fired -- and a further poll retries
    # nothing, because the deferral is cleared once the write lands.
    assert "selection_depth_scale_str" in panel._editing_depth_text
    panel.update(doc)
    assert len(state.window_calls) == writes_before + 1


def _tearing_record(state):
    """Fail every second record read at a fixed native generation. Each failed read
    reports None, so retries never obtain comparable generations. This isolates
    exhaustion from phantom lineage changes that could independently trigger
    retargeting.
    """
    reads = []

    def _read():
        reads.append(1)
        if len(reads) % 2 == 0:
            raise RuntimeError("collapse record unavailable")
        return (
            state.depth_window_collapse_source,
            state.depth_window_collapse_generation,
            state.depth_window_collapse_kind,
        )

    return _read


def test_an_escape_revert_survives_a_blur_that_lands_during_exhaustion(
    selection_controls_module,
):
    """Enter changes native .25 to 1.0; Escape restores the .25 buffer, then
    cancelFocusedElement blurs during exhaustion. The widget ends the edit even though
    its revert write deferred. Keep a blurred record with reverted text and origin,
    retire live-edit membership, and apply that frozen revert on the first stable poll
    so native state cannot remain 1.0.
    """
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    original = state.depth_near
    assert original == pytest.approx(0.25)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("1.00")
    doc.near.emit("change", _InputEventStub(linebreak=True))
    assert state.depth_near == pytest.approx(1.00), "the Enter commit did not land"
    writes_after_enter = len(state.window_calls)

    settled_record = module.lf.ui.get_depth_window_collapse_record
    doc.near.emit("escapecancel", _InputEventStub())
    assert panel._depth_text_bufs["selection_depth_near_str"] == "0.25"

    # The storm starts before the revert's blur reaches the commit.
    module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)
    doc.near.emit("blur")

    # Deferred, so nothing was written against the torn snapshot...
    assert len(state.window_calls) == writes_after_enter
    assert state.depth_near == pytest.approx(1.00)
    # ...and nothing was dropped either. The blur is COMPLETE, so the edit is
    # retired exactly as an undeferred blur retires it; what carries the revert
    # is the deferral record, which froze the reverted text and the panel the
    # edit was aimed at.
    assert panel._depth_text_bufs["selection_depth_near_str"] == "0.25"
    assert "selection_depth_near_str" not in panel._editing_depth_text
    record = panel._deferred_depth_commits["selection_depth_near_str"]
    assert record["kind"] == "blurred"
    assert record["payload"] == "0.25"
    assert record["panel"] == "left"

    # A poll that is still storming retries and defers again -- still no write,
    # and the frozen record is untouched.
    panel.update(doc)
    assert len(state.window_calls) == writes_after_enter
    assert state.depth_near == pytest.approx(1.00)
    assert panel._deferred_depth_commits["selection_depth_near_str"]["payload"] == "0.25"

    # The storm clears. The first stable poll resolves the deferred revert.
    module.lf.ui.get_depth_window_collapse_record = settled_record
    panel.update(doc)

    assert state.depth_near == pytest.approx(original), (
        "the Escape revert was lost to the exhausted blur"
    )
    assert len(state.window_calls) == writes_after_enter + 1
    # The blur is honoured once its commit lands: the edit is retired, and no
    # later poll rewrites anything.
    assert "selection_depth_near_str" not in panel._editing_depth_text
    assert "selection_depth_near_str" not in panel._deferred_depth_commits
    assert "selection_depth_near_str" not in panel._depth_text_edit_panel
    panel.update(doc)
    assert len(state.window_calls) == writes_after_enter + 1
    assert state.depth_near == pytest.approx(original)


def test_a_plain_blur_commit_deferred_by_exhaustion_lands_on_a_stable_poll(
    selection_controls_module,
):
    """The same guarantee for a typed value, with no Escape involved.

    A blur is the ordinary way a text edit commits. If exhaustion swallowed it
    the user's number would vanish silently, which is the identical defect.
    """
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    writes_before = len(state.window_calls)
    settled_record = module.lf.ui.get_depth_window_collapse_record

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("0.75")
    module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)
    doc.near.emit("blur")

    assert len(state.window_calls) == writes_before
    assert state.depth_near == pytest.approx(0.25)
    assert panel._depth_text_bufs["selection_depth_near_str"] == "0.75"
    # The blur completed the edit, so the live-edit state is retired and the
    # typed value is held frozen in the deferral record instead.
    assert "selection_depth_near_str" not in panel._editing_depth_text
    assert panel._deferred_depth_commits["selection_depth_near_str"]["payload"] == "0.75"

    module.lf.ui.get_depth_window_collapse_record = settled_record
    panel.update(doc)

    assert state.depth_near == pytest.approx(0.75), "the blur commit was dropped"
    assert len(state.window_calls) == writes_before + 1
    assert "selection_depth_near_str" not in panel._editing_depth_text
    assert "selection_depth_near_str" not in panel._deferred_depth_commits


def test_two_deferred_keys_both_land_on_the_same_flush(selection_controls_module):
    """Defer Near .75 and Far 8.25. Committing either key forces text synchronization,
    which must preserve the other pending buffer even though it is no longer live. Both
    intended values must land.
    """
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    settled_record = module.lf.ui.get_depth_window_collapse_record
    module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("0.75")
    doc.near.emit("change", _InputEventStub(linebreak=True))
    doc.far.emit("focus")
    model.bound_binds["selection_depth_far_str"][1]("8.25")
    doc.far.emit("change", _InputEventStub(linebreak=True))

    # Both commits exhausted, so nothing was written and both are recorded.
    assert state.depth_near == pytest.approx(0.25)
    assert state.depth_far == pytest.approx(7.5)
    assert set(panel._deferred_depth_commits) == {
        "selection_depth_near_str",
        "selection_depth_far_str",
    }

    module.lf.ui.get_depth_window_collapse_record = settled_record
    panel.update(doc)

    assert state.depth_far == pytest.approx(8.25), "the Far deferral was dropped"
    assert state.depth_near == pytest.approx(0.75), (
        "the Far commit's forced sync canonicalized the still-deferred Near"
    )
    assert panel._deferred_depth_commits == {}


def test_the_flush_replays_two_deferred_blurs_in_the_order_the_user_made_them(
    selection_controls_module,
):
    """From Near .90/Far 1.00, defer Near .20 then Far .50. Alphabetical Far-first
    replay would clamp .50 to .91 against the old Near and destroy intent. The result
    must be the user's .20/.50 pair.
    """
    module, state = selection_controls_module
    state.depth_near = 0.90
    state.depth_far = 1.00
    panel, model, doc = _mounted_panel(module, state)

    settled_record = module.lf.ui.get_depth_window_collapse_record
    module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("0.20")
    doc.near.emit("blur")
    doc.far.emit("focus")
    model.bound_binds["selection_depth_far_str"][1]("0.50")
    doc.far.emit("blur")

    # Both blurs completed against the storm, so nothing was written and both
    # records are frozen -- in the order the user made them.
    assert state.depth_near == pytest.approx(0.90)
    assert state.depth_far == pytest.approx(1.00)
    assert list(panel._deferred_depth_commits) == [
        "selection_depth_near_str",
        "selection_depth_far_str",
    ]
    assert panel._deferred_depth_commits["selection_depth_near_str"]["kind"] == "blurred"
    assert panel._deferred_depth_commits["selection_depth_far_str"]["kind"] == "blurred"

    module.lf.ui.get_depth_window_collapse_record = settled_record
    panel.update(doc)

    assert state.depth_near == pytest.approx(0.20)
    assert state.depth_far == pytest.approx(0.50), (
        "the flush replayed Far before Near, so Far was clamped against the "
        "stale native Near instead of the one the user had just set"
    )
    assert panel._deferred_depth_commits == {}


def test_the_flush_orders_a_blurred_deferral_before_a_later_live_one(
    selection_controls_module,
):
    """Resolve Near .20/Far .50 with blurred Near and Enter-committed, live Far.

    Read frozen Near and Far's current buffer. Both intended values must land; serial
    replay by field name would clamp Far against the stale native Near.
    """
    module, state = selection_controls_module
    state.depth_near = 0.90
    state.depth_far = 1.00
    panel, model, doc = _mounted_panel(module, state)

    settled_record = module.lf.ui.get_depth_window_collapse_record
    module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("0.20")
    doc.near.emit("blur")
    doc.far.emit("focus")
    model.bound_binds["selection_depth_far_str"][1]("0.50")
    doc.far.emit("change", _InputEventStub(linebreak=True))

    assert state.depth_near == pytest.approx(0.90)
    assert state.depth_far == pytest.approx(1.00)
    assert list(panel._deferred_depth_commits) == [
        "selection_depth_near_str",
        "selection_depth_far_str",
    ]
    assert panel._deferred_depth_commits["selection_depth_near_str"]["kind"] == "blurred"
    assert panel._deferred_depth_commits["selection_depth_far_str"]["kind"] == "live"

    module.lf.ui.get_depth_window_collapse_record = settled_record
    panel.update(doc)

    assert state.depth_near == pytest.approx(0.20)
    assert state.depth_far == pytest.approx(0.50), (
        "the live Far record was replayed ahead of the earlier blurred Near"
    )
    assert panel._deferred_depth_commits == {}


def test_a_superseded_and_re_armed_near_still_lands_with_the_far_between_them(
    selection_controls_module,
):
    """From .90/1.00, blur-defer Near .20 then Far .50; refocus and re-arm Near, moving
    its surviving record behind Far. Even chronological Far/Near replay clamps .50 to
    .91. Resolve the target's final Near/Far together and write once, regardless of
    record order.
    """
    module, state = selection_controls_module
    state.depth_near = 0.90
    state.depth_far = 1.00
    panel, model, doc = _mounted_panel(module, state)

    settled_record = module.lf.ui.get_depth_window_collapse_record
    module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("0.20")
    doc.near.emit("blur")
    doc.far.emit("focus")
    model.bound_binds["selection_depth_far_str"][1]("0.50")
    doc.far.emit("blur")

    # The user comes back to Near. Re-focusing supersedes the pending record.
    doc.near.emit("focus")
    assert list(panel._deferred_depth_commits) == ["selection_depth_far_str"], (
        "re-focusing did not supersede the pending Near record"
    )

    # They re-commit the same value and leave, so Near is re-armed -- at the
    # TAIL of the registry, behind the Far edit they made in between.
    model.bound_binds["selection_depth_near_str"][1]("0.20")
    doc.near.emit("change", _InputEventStub(linebreak=True))
    doc.near.emit("blur")
    assert list(panel._deferred_depth_commits) == [
        "selection_depth_far_str",
        "selection_depth_near_str",
    ], "the re-armed Near did not land at the registry tail"

    # Nothing was written against the storm.
    assert state.depth_near == pytest.approx(0.90)
    assert state.depth_far == pytest.approx(1.00)

    module.lf.ui.get_depth_window_collapse_record = settled_record
    panel.update(doc)

    assert state.depth_near == pytest.approx(0.20)
    assert state.depth_far == pytest.approx(0.50), (
        "the flush replayed the pair serially, so Far was clamped against the "
        "stale native Near instead of the one the user had just set"
    )
    assert panel._deferred_depth_commits == {}


def test_the_flush_resolves_each_write_target_as_its_own_pair(
    selection_controls_module,
):
    """Both blurs belong to Left but focus is Right at flush time. Far-first serial
    panel writes would clamp against Left's old Near and lose .50. Resolve and write the
    pair once to Left; Right must remain untouched.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    state.panel_ranges["left"] = (0.90, 1.00)
    state.panel_ranges["right"] = (2.00, 9.00)
    panel, model, doc = _mounted_panel(module, state)

    settled_record = module.lf.ui.get_depth_window_collapse_record
    module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)

    doc.far.emit("focus")
    model.bound_binds["selection_depth_far_str"][1]("0.50")
    doc.far.emit("blur")
    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("0.20")
    doc.near.emit("blur")

    assert panel._deferred_depth_commits["selection_depth_near_str"]["panel"] == "left"
    assert panel._deferred_depth_commits["selection_depth_far_str"]["panel"] == "left"

    # Focus moves away before the deferral resolves, so Left is now a window a
    # no-panel write would MISS.
    state.focused_panel = "right"
    module.lf.ui.get_depth_window_collapse_record = settled_record
    writes_before = len(state.panel_writes)
    panel.update(doc)

    assert panel._focused_panel == "right"
    # Grouping is by the window the write will land in: a blur frozen to a panel
    # a no-panel write would now MISS is addressed by panel, while a live record
    # -- still aimed at the context the user is editing in -- is not.
    assert panel._deferred_write_target({"kind": "blurred", "panel": "left"}) == "left"
    assert panel._deferred_write_target({"kind": "blurred", "panel": "right"}) is None
    assert panel._deferred_write_target({"kind": "live"}) is None
    assert state.panel_ranges["left"][0] == pytest.approx(0.20)
    assert state.panel_ranges["left"][1] == pytest.approx(0.50), (
        "the panel-addressed flush replayed the pair serially, so Far was "
        "clamped against Left's stale native Near"
    )
    assert state.panel_ranges["right"] == (2.00, 9.00), "the other panel was written"
    assert state.write_targets[-1] == "left"
    assert len(state.panel_writes) == writes_before + 1, (
        "the resolved pair was not written as ONE combined update"
    )
    assert panel._deferred_depth_commits == {}


def test_a_mid_flush_retarget_regroups_instead_of_writing_the_stale_target(
    selection_controls_module,
):
    """Start focused Left, L=(.90,1.00), R=(2.00,9.00), with blurred Near .20 frozen to
    Left and Far still live. They initially share the displayed target. If Far's
    revalidation sees focus move Right, both its canonical buffer and the displayed
    target change. Regroup after that read: write frozen Near explicitly to Left and
    retargeted Far to Right, never the combined stale pair to Right.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    state.panel_ranges["left"] = (0.90, 1.00)
    state.panel_ranges["right"] = (2.00, 9.00)
    panel, model, doc = _mounted_panel(module, state)

    settled_record = module.lf.ui.get_depth_window_collapse_record
    module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("0.20")
    doc.near.emit("blur")
    doc.far.emit("focus")
    model.bound_binds["selection_depth_far_str"][1]("0.50")
    doc.far.emit("change", _InputEventStub(linebreak=True))

    assert panel._deferred_depth_commits["selection_depth_near_str"]["kind"] == "blurred"
    assert panel._deferred_depth_commits["selection_depth_near_str"]["panel"] == "left"
    assert panel._deferred_depth_commits["selection_depth_far_str"]["kind"] == "live"
    assert state.panel_ranges["left"] == (0.90, 1.00)
    assert state.panel_ranges["right"] == (2.00, 9.00)

    module.lf.ui.get_depth_window_collapse_record = settled_record

    # The focus change is delivered with no pointer input, so the poll's own
    # refresh still sees Left and only the flush's INNER revalidation -- the
    # second context read of this update() -- observes the move.
    original_refresh = panel._refresh_panel_context
    reads = []

    def _refresh_moving_focus_on_the_second_read():
        reads.append(1)
        if len(reads) == 2:
            state.focused_panel = "right"
        return original_refresh()

    panel._refresh_panel_context = _refresh_moving_focus_on_the_second_read
    panel.update(doc)
    panel._refresh_panel_context = original_refresh

    assert len(reads) >= 2, "the live member never revalidated inside the flush"
    assert panel._focused_panel == "right"
    assert state.panel_ranges["right"] == (
        pytest.approx(2.00),
        pytest.approx(9.00),
    ), (
        "the combined write used the target computed BEFORE the revalidation, "
        "so the frozen Left intent contaminated Right"
    )
    assert state.panel_ranges["left"] == (
        pytest.approx(0.20),
        pytest.approx(1.00),
    ), "the frozen Near did not land on the panel it was typed into"
    assert panel._deferred_depth_commits == {}


def test_a_mixed_target_pair_writes_each_field_to_its_own_panel(
    selection_controls_module,
):
    """With stable Right focus, blurred Near belongs to Left and live Far to Right.
    Resolve each against its own counterpart: Right Far .50 clamps above its Near 2.00
    to 2.01, independently of Left Near .20.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    state.panel_ranges["left"] = (0.90, 1.00)
    state.panel_ranges["right"] = (2.00, 9.00)
    panel, model, doc = _mounted_panel(module, state)

    settled_record = module.lf.ui.get_depth_window_collapse_record
    module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("0.20")
    doc.near.emit("blur")

    # The user moves to the other panel and the plugin observes it -- a bare
    # context refresh, not a poll, so the frozen Near is not flushed yet.
    state.focused_panel = "right"
    module.lf.ui.get_depth_window_collapse_record = settled_record
    panel._refresh_panel_context()
    assert panel._focused_panel == "right"
    assert list(panel._deferred_depth_commits) == ["selection_depth_near_str"]

    # The storm resumes and the Far edit they now make on Right defers LIVE.
    module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)
    doc.far.emit("focus")
    model.bound_binds["selection_depth_far_str"][1]("0.50")
    doc.far.emit("change", _InputEventStub(linebreak=True))

    assert panel._deferred_depth_commits["selection_depth_near_str"]["panel"] == "left"
    assert panel._deferred_depth_commits["selection_depth_far_str"]["kind"] == "live"

    module.lf.ui.get_depth_window_collapse_record = settled_record
    panel.update(doc)

    assert state.panel_ranges["left"] == (
        pytest.approx(0.20),
        pytest.approx(1.00),
    ), "the frozen Near did not reach its own panel"
    assert state.panel_ranges["right"] == (
        pytest.approx(2.00),
        pytest.approx(2.01),
    ), (
        "Far was not clamped against RIGHT's own Near -- the two targets were "
        "resolved as one pair"
    )
    assert panel._deferred_depth_commits == {}


def test_a_blurred_deferral_survives_a_focus_move_and_lands_on_its_own_panel(
    selection_controls_module,
):
    """Blur-defer Left Near .75, then focus Right. A completed blur must retain its
    payload and origin outside live-edit retargeting; otherwise the flush writes Right's
    canonical value and strands Left at .25. The intended write must reach Left only.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    state.panel_ranges["left"] = (0.25, 7.5)
    state.panel_ranges["right"] = (2.0, 9.0)
    panel, model, doc = _mounted_panel(module, state)

    settled_record = module.lf.ui.get_depth_window_collapse_record
    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("0.75")
    module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)
    doc.near.emit("blur")

    # Focus moves to the other panel before the deferral resolves.
    state.focused_panel = "right"
    module.lf.ui.get_depth_window_collapse_record = settled_record
    panel.update(doc)

    assert panel._focused_panel == "right"
    assert state.panel_ranges["left"][0] == pytest.approx(0.75), (
        "the completed blur was retargeted to the panel focus moved to"
    )
    assert state.panel_ranges["left"][1] == pytest.approx(7.5)
    assert state.panel_ranges["right"] == (2.0, 9.0), "the other panel was written"
    assert state.write_targets[-1] == "left"
    assert panel._deferred_depth_commits == {}
    # And the mechanism that made it possible: the blur FROZE the payload and
    # the origin panel into the record, and retired the live-edit state, so the
    # transition guard above found no live edit to retarget.
    assert "selection_depth_near_str" not in panel._editing_depth_text
    assert "selection_depth_near_str" not in panel._depth_text_edit_panel


def test_refocusing_a_blurred_deferral_supersedes_it_and_spares_the_new_edit(
    selection_controls_module,
):
    """After blur-deferring .75, refocus and type .90. Drop the old record without
    committing or retiring the new edit, so Enter can apply .90 and later polling cannot
    overwrite newly typed 1.10.
    """
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    settled_record = module.lf.ui.get_depth_window_collapse_record
    writes_before = len(state.window_calls)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("0.75")
    module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)
    doc.near.emit("blur")
    assert panel._deferred_depth_commits["selection_depth_near_str"]["kind"] == "blurred"

    # The user comes back to the field and types something else.
    module.lf.ui.get_depth_window_collapse_record = settled_record
    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("0.90")

    panel.update(doc)

    assert state.depth_near == pytest.approx(0.25), "the superseded 0.75 was written"
    assert len(state.window_calls) == writes_before
    assert "selection_depth_near_str" in panel._editing_depth_text, (
        "the flush retired the edit the user is in the middle of"
    )
    assert panel._depth_text_bufs["selection_depth_near_str"] == "0.90"

    # The new value commits on the user's OWN Enter...
    doc.near.emit("change", _InputEventStub(linebreak=True))
    assert state.depth_near == pytest.approx(0.90)

    # ...and what they type after it is not clobbered by the next poll.
    model.bound_binds["selection_depth_near_str"][1]("1.10")
    panel.update(doc)
    assert panel._depth_text_bufs["selection_depth_near_str"] == "1.10"
    assert state.depth_near == pytest.approx(0.90)
    # The mechanism: re-focusing dropped the stale record, so the flush had
    # nothing of the old session left to act on.
    assert "selection_depth_near_str" not in panel._deferred_depth_commits


def test_a_sync_toggle_during_exhaustion_is_dropped_not_written_from_the_cache(
    selection_controls_module,
):
    """If exhausted refresh leaves cached sync=false while native sync=true, inverting
    the cache sends a no-op TRUE request. Drop the click without a setter call until a
    later poll can refresh the actual state.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    panel, model, doc = _mounted_panel(module, state)
    assert panel._depth_sync is False

    # The flag moves externally (undo elsewhere, MCP, a project restore) and a
    # storm starts before the next poll can observe it.
    state.depth_sync = True
    settled_record = module.lf.ui.get_depth_window_collapse_record
    module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)

    model.bound_events["selection_action"](model.handle, None, ["toggle_sync"])

    assert state.sync_calls == [], (
        "the toggle was written from the stale cache after an exhausted read"
    )
    assert state.depth_sync is True, "native state was not touched"
    # The cache is untouched too -- an exhausted tick consumes nothing.
    assert panel._depth_sync is False

    # The button tells the truth again at the next stable poll.
    module.lf.ui.get_depth_window_collapse_record = settled_record
    panel.update(doc)
    assert panel._depth_sync is True
    assert model.bound_funcs["selection_depth_sync_active"]() is True
    assert model.bound_funcs["selection_depth_sync_icon"]() == module._SYNC_ICON_ON
    assert state.sync_calls == []


def test_a_leave_between_the_record_reads_seeds_from_the_post_leave_source(
    selection_controls_module,
):
    """An older source-only binding returns generation=None on both reads and ends retry
    after one attempt. Keep the post-endpoint source: a Right leave between reads leaves
    the first source at Left, but no later generation or endpoint edge can repair a
    baseline seeded from it.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    state.panel_scales = {"left": (0.60, 0.60), "right": (0.20, 0.20)}
    state.depth_scale = 0.60
    state.depth_scale_y = 0.60
    state.depth_window_collapse_source = "left"
    # An older module: no lineage record at all, only the source.
    del module.lf.ui.get_depth_window_collapse_record
    panel, model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20
    panel._ref_scale_x["shared"] = 0.90
    panel._ref_scale_y["shared"] = 0.90

    landed = []

    def _torn_source():
        # The FIRST read is pre-leave and names Left; the leave -- which folds
        # RIGHT's window into the single remaining one and resets the
        # observable focus to Left -- lands immediately afterwards, so the
        # endpoint and the second record read are both post-leave.
        stale = state.depth_window_collapse_source
        if not landed:
            landed.append(True)
            state.split_view_mode = "none"
            state.focused_panel = "left"
            state.depth_window_collapse_source = "right"
            state.depth_scale = 0.20
            state.depth_scale_y = 0.20
        return stale

    module.lf.ui.get_depth_window_collapse_source = _torn_source
    panel.update(doc)

    assert panel._split_mode == "none"
    # Seeded from RIGHT, the panel whose window actually survived.
    assert panel._ref_scale_x["shared"] == pytest.approx(0.20)
    assert panel._ref_scale_y["shared"] == pytest.approx(0.20)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"

    # And it stays right: nothing later can repair a wrong seed here.
    panel.update(doc)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "100"


def test_a_cycle_hidden_by_invisible_controls_reconciles_on_resume(
    selection_controls_module,
):
    """Hidden controls return before refresh. Advance the consumed generation only on
    reconciliation, so the first visible refresh still sees the entire hidden-cycle
    delta.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="right", sync=False)
    panel, _model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20
    consumed = panel._collapse_generation

    # The controls go away (another tool becomes active), and the whole cycle
    # happens while they are hidden.
    state.active_tool = "builtin.move"
    panel.update(doc)
    state.focused_panel = "left"
    state.depth_scale = 0.20
    state.depth_scale_y = 0.20
    state.depth_window_collapse_source = "right"
    state.depth_window_collapse_kind = "leave_collapse"
    state.depth_window_collapse_generation = 3
    panel.update(doc)
    assert panel._collapse_generation == consumed, (
        "a hidden refresh consumed the generation without reconciling"
    )

    # Resume.
    state.active_tool = "builtin.select"
    panel.update(doc)

    assert panel._ref_scale_x["left"] == pytest.approx(0.20)
    assert panel._ref_scale_x["right"] == pytest.approx(0.20)
    assert panel._ref_scale_x["shared"] == pytest.approx(0.20)


@pytest.mark.parametrize("generation", [0, None, -1], ids=["unchanged", "unavailable", "reset"])
def test_a_no_op_mode_change_retargets_only_for_lineage_invalidation(
    selection_controls_module, generation,
):
    """A mode no-op preserves drafts unless lineage invalidates their target."""
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)
    panel._ref_scale_x["shared"] = 0.70
    panel._ref_scale_y["shared"] = 0.70
    expected_near = state.depth_near if generation == -1 else 3.75

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")

    state.split_view_mode = "ply_comparison"
    state.depth_window_collapse_generation = generation
    writes_before = len(state.window_calls)
    panel.update(doc)

    assert panel._split_mode == "ply_comparison"
    assert panel._depth_text_bufs["selection_depth_near_str"] == str(expected_near)
    assert model.bound_binds["selection_depth_scale_value"][0]() == "50"
    assert len(state.window_calls) == writes_before

    writes_before = len(state.window_calls)
    doc.near.emit("change", _InputEventStub(linebreak=True))

    assert len(state.window_calls) == writes_before + 1
    assert state.window_calls[-1][1] == pytest.approx(expected_near)
    assert state.depth_near == pytest.approx(expected_near)


def test_the_sync_toggle_pre_reads_the_focused_canonical_and_stays_idempotent(
    selection_controls_module,
):
    """Model the production sync copy and stamp with distinct panel ranges. Refresh the
    edit to Right's focused canonical value before copying Right into both slots; later
    blur must re-emit that same value, never typed text or Left's canonical.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    state.panel_ranges = {"left": (1.00, 9.00), "right": (2.50, 8.00)}
    panel, model, doc = _mounted_panel(module, state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")
    assert panel._depth_text_edit_panel["selection_depth_near_str"] == "left"

    state.focused_panel = "right"
    model.bound_events["selection_action"](model.handle, None, ["toggle_sync"])

    assert state.depth_sync is True
    # The copy really happened, and it was stamped.
    assert state.panel_ranges["left"] == state.panel_ranges["right"] == (2.50, 8.00)
    assert state.depth_window_collapse_kind == "sync_copy"
    assert state.depth_window_collapse_generation == 1
    assert state.depth_window_collapse_source == "right"
    # The pre-read was the FOCUSED panel's canonical, not the edit's origin.
    assert panel._depth_text_edit_panel["selection_depth_near_str"] == "right"
    assert panel._depth_text_bufs["selection_depth_near_str"] == "2.50"

    writes_before = len(state.window_calls)
    doc.near.emit("blur")

    assert len(state.window_calls) == writes_before + 1
    assert state.window_calls[-1][1] == pytest.approx(2.50)
    assert state.write_targets[-1] == "both", (
        "a synced write must fan out to both slots, not address one panel"
    )
    # Idempotent: the write put back exactly what the copy had already left.
    assert state.panel_ranges["left"] == state.panel_ranges["right"] == (2.50, 8.00)
    assert state.depth_near == pytest.approx(2.50)
    assert state.depth_near != pytest.approx(3.75)


def test_toolbar_undo_of_the_sync_flag_reconciles_the_references(
    selection_controls_module,
):
    """Refresh must reconcile native sync changes outside the toggle path.

    This flag-only stub starts L=.60/R=.20, toggles ON, then simulates Undo by clearing
    sync. Refresh must copy shared .60 to both references. Native absolute restore and
    lineage recovery are covered by the restore tests.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=False)
    panel, model, doc = _mounted_panel(module, state)
    del doc

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20

    model.bound_events["selection_action"](model.handle, None, ["toggle_sync"])
    assert panel._depth_sync is True
    assert panel._ref_scale_x["shared"] == pytest.approx(0.60)

    # Toolbar Undo. The native entry restores the flag; no toggle path runs.
    state.undo_effect = lambda: setattr(state, "depth_sync", False)
    model.bound_events["selection_action"](model.handle, None, ["undo"])

    assert state.undo_calls == 1
    assert panel._depth_sync is False
    assert panel._ref_key() == "left"
    assert panel._ref_scale_x["left"] == pytest.approx(0.60)
    assert panel._ref_scale_x["right"] == pytest.approx(0.60), (
        "Undo restored the flag but the references were never reconciled"
    )
    assert panel._ref_scale_y["left"] == pytest.approx(0.60)
    assert panel._ref_scale_y["right"] == pytest.approx(0.60)


def test_toolbar_redo_of_the_sync_flag_reconciles_the_references(
    selection_controls_module,
):
    """The same edge in the other direction, through the redo channel."""
    module, state = selection_controls_module
    _independent_dual(state, focused="right", sync=False)
    panel, model, doc = _mounted_panel(module, state)
    del doc

    panel._ref_scale_x["left"] = 0.60
    panel._ref_scale_y["left"] = 0.60
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20

    # Redo re-applies a sync-ON: the FOCUSED panel's reference collapses.
    state.redo_effect = lambda: setattr(state, "depth_sync", True)
    model.bound_events["selection_action"](model.handle, None, ["redo"])

    assert state.redo_calls == 1
    assert panel._depth_sync is True
    assert panel._ref_key() == "shared"
    assert panel._ref_scale_x["shared"] == pytest.approx(0.20)
    assert panel._ref_scale_y["shared"] == pytest.approx(0.20)


def test_a_sync_edge_reconciles_exactly_once_whatever_delivers_it(
    selection_controls_module,
):
    """The invariant behind the shared refresh helper: no double-fire.

    After the toggle path has reconciled, the refresh at the end of the same
    dispatch -- and every later update() -- must see no edge and leave the
    references alone.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left", sync=True)
    panel, model, doc = _mounted_panel(module, state)

    panel._ref_scale_x["shared"] = 0.55
    panel._ref_scale_y["shared"] = 0.55

    # Sync OFF seeds both panels from shared.
    model.bound_events["selection_action"](model.handle, None, ["toggle_sync"])
    assert panel._ref_scale_x["left"] == pytest.approx(0.55)
    assert panel._ref_scale_x["right"] == pytest.approx(0.55)

    # Diverge the two entries the way two independent drags would, then poll.
    # A second reconciliation would re-seed them from shared and wipe this.
    panel._ref_scale_x["right"] = 0.20
    panel._ref_scale_y["right"] = 0.20
    for _ in range(3):
        panel.update(doc)

    assert panel._ref_scale_x["right"] == pytest.approx(0.20)
    assert panel._ref_scale_y["right"] == pytest.approx(0.20)
    assert panel._ref_scale_x["left"] == pytest.approx(0.55)


@pytest.mark.parametrize("panel_token", ["left", "right"])
def test_explicit_panel_writes_route_to_that_panels_slot(
    selection_controls_module, panel_token
):
    """Test gap: a VALID panel= SETTER routing check.

    An explicit-panel write must land in that panel's slot -- readable back
    through the panel getter -- and must NOT move the displayed window.
    """
    import lichtfeld as lf_stub

    _module, state = selection_controls_module
    other = "right" if panel_token == "left" else "left"
    state.depth_scale = 0.35
    state.depth_scale_y = 0.35
    state.panel_scales[other] = (0.22, 0.22)

    lf_stub.selection.set_depth_filter_window(
        True, 1.0, 9.0, 0.72, 0.0, 0.0, 0.72, panel=panel_token
    )

    # Read back through the panel getter.
    window = lf_stub.selection.get_depth_filter_window(panel=panel_token)
    assert window[3] == pytest.approx(0.72)
    assert window[4] == pytest.approx(0.72)

    # Manager-visible effect: that slot moved, the other slot and the displayed
    # projection window did not.
    assert state.panel_scales[panel_token] == pytest.approx((0.72, 0.72))
    assert state.panel_scales[other] == pytest.approx((0.22, 0.22))
    assert state.depth_scale == pytest.approx(0.35)
    assert state.panel_writes[-1][0] == panel_token


# ---------------------------------------------------------------------------
# The native bindings themselves (real lichtfeld module)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("panel", ["", "middle", "Left", "LEFT", "both", "0"])
def test_depth_filter_window_rejects_unknown_panel_tokens(lf, panel):
    with pytest.raises(ValueError):
        lf.selection.get_depth_filter_window(panel=panel)
    with pytest.raises(ValueError):
        lf.selection.set_depth_filter_window(False, panel=panel)


def test_panel_is_keyword_only_on_the_depth_window_calls(lf):
    """The positional signature is frozen: panel must never be reachable by
    position, or an eighth positional argument would silently change meaning."""
    with pytest.raises(TypeError):
        lf.selection.get_depth_filter_window("left")
    with pytest.raises(TypeError):
        lf.selection.set_depth_filter_window(
            False, 0.0, 100.0, 0.35, 0.0, 0.0, None, "left"
        )


def test_legacy_depth_calls_gained_no_panel_argument(lf):
    """The compat surface stays projection-only."""
    with pytest.raises(TypeError):
        lf.selection.set_depth_filter_range(False, 0.0, 100.0, 50.0, panel="left")
    with pytest.raises(TypeError):
        lf.selection.get_depth_filter_range(panel="left")
    with pytest.raises(TypeError):
        lf.selection.get_depth_filter(panel="left")
    with pytest.raises(TypeError):
        lf.selection.set_depth_filter(False, 100.0, 1.35, 0.0, panel="left")


def test_depth_window_sync_accessors_round_trip(lf):
    before = lf.ui.get_depth_window_sync()
    assert isinstance(before, bool)

    # With no rendering manager in this headless process the setter reports the
    # flag's ACTUAL state, which is the contract that matters: callers must
    # never assume the requested value landed.
    reported = lf.ui.set_depth_window_sync(not before)
    assert isinstance(reported, bool)
    assert reported == lf.ui.get_depth_window_sync()

    lf.ui.set_depth_window_sync(before)
    assert lf.ui.get_depth_window_sync() == before


@pytest.mark.parametrize("panel", ["left", "right"])
def test_native_draw_commit_store_field_round_trips_its_panel(lf, panel):
    """Test gap: exercise the panel-tagged draw-commit binding
    (py_store.cpp) end to end -- dict in, same dict out, panel preserved."""
    store = lf.ui.store
    before = store.get("depth_window_draw_commit")
    try:
        store.set("depth_window_draw_commit", {"generation": 41, "panel": panel})
        after = store.get("depth_window_draw_commit")

        assert isinstance(after, dict)
        assert after["generation"] == 41
        assert after["panel"] == panel
    finally:
        store.set("depth_window_draw_commit", before)


def test_a_deferred_pair_stops_the_whole_flush_instead_of_freeing_its_siblings(
    selection_controls_module,
):
    """Cache Left while native focus moves Right, then exhaust the read that would
    expose it. If a deferred Near/Far pair fails to stop the whole flush, frozen Left
    Size can use the stale displayed target and write Right. Require zero writes, zero
    consumption and all three records retained; no sibling may proceed on the failed
    context.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    state.panel_ranges["left"] = (0.90, 1.00)
    state.panel_ranges["right"] = (2.00, 9.00)
    panel, model, doc = _mounted_panel(module, state)

    settled_record = module.lf.ui.get_depth_window_collapse_record
    tearing_record = _tearing_record(state)
    module.lf.ui.get_depth_window_collapse_record = tearing_record

    # Near and Size are BLURRED and frozen to Left; Far is still LIVE. All
    # three defer against the storm, and with focus on Left every one of them
    # groups under the target None.
    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("0.20")
    doc.near.emit("blur")
    doc.scale.emit("focus")
    model.bound_binds["selection_depth_scale_str"][1]("150")
    doc.scale.emit("blur")
    doc.far.emit("focus")
    model.bound_binds["selection_depth_far_str"][1]("0.50")
    doc.far.emit("change", _InputEventStub(linebreak=True))

    assert panel._deferred_depth_commits["selection_depth_near_str"]["kind"] == "blurred"
    assert panel._deferred_depth_commits["selection_depth_scale_str"]["kind"] == "blurred"
    assert panel._deferred_depth_commits["selection_depth_scale_str"]["panel"] == "left"
    assert panel._deferred_depth_commits["selection_depth_far_str"]["kind"] == "live"

    # The poll's OWN refresh succeeds and still sees Left, so the flush is
    # entered with a valid context. Native focus then moves to Right and the
    # storm resumes, so the pair's inner revalidation -- the flush's own read --
    # is the one that comes back exhausted, never observing the move.
    module.lf.ui.get_depth_window_collapse_record = settled_record
    original_refresh = panel._refresh_panel_context
    reads = []

    def _refresh_tearing_from_the_second_read():
        reads.append(1)
        if len(reads) == 2:
            state.focused_panel = "right"
            module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)
        return original_refresh()

    panel._refresh_panel_context = _refresh_tearing_from_the_second_read
    writes_before = list(state.write_targets)
    panel_writes_before = list(state.panel_writes)
    panel.update(doc)
    panel._refresh_panel_context = original_refresh
    module.lf.ui.get_depth_window_collapse_record = settled_record

    assert len(reads) >= 2, "the live member never revalidated inside the flush"
    assert panel._context_read_exhausted is True
    assert panel._focused_panel == "left", "the exhausted read must not move the cache"

    assert state.write_targets == writes_before, (
        "a write was issued on a tick whose context read was exhausted"
    )
    assert state.panel_writes == panel_writes_before
    assert state.panel_ranges["right"] == (2.00, 9.00), (
        "the frozen-to-Left Size record rode a no-panel write into RIGHT"
    )
    assert state.panel_ranges["left"] == (0.90, 1.00)
    assert set(panel._deferred_depth_commits) == {
        "selection_depth_near_str",
        "selection_depth_far_str",
        "selection_depth_scale_str",
    }, "a sibling record was consumed after the pair deferred"


def test_a_second_retarget_on_an_unpaired_record_defers_it_instead_of_writing(
    selection_controls_module,
):
    """A second retarget must leave the live record pending after one regroup.

    Start L=(.90,1.00), R=(2.00,9.00), focused Left, with blurred Near .20 frozen to
    Left and Far live. Moving Right during pair validation splits them; moving back Left
    during Far's single-record validation exhausts the regroup budget. Only the explicit
    Left Near write lands. Far returns RETARGETED and stays pending, rather than writing
    after that second move.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    state.panel_ranges["left"] = (0.90, 1.00)
    state.panel_ranges["right"] = (2.00, 9.00)
    panel, model, doc = _mounted_panel(module, state)

    settled_record = module.lf.ui.get_depth_window_collapse_record
    module.lf.ui.get_depth_window_collapse_record = _tearing_record(state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("0.20")
    doc.near.emit("blur")
    doc.far.emit("focus")
    model.bound_binds["selection_depth_far_str"][1]("0.50")
    doc.far.emit("change", _InputEventStub(linebreak=True))

    assert panel._deferred_depth_commits["selection_depth_near_str"]["panel"] == "left"
    assert panel._deferred_depth_commits["selection_depth_far_str"]["kind"] == "live"

    module.lf.ui.get_depth_window_collapse_record = settled_record
    original_refresh = panel._refresh_panel_context
    reads = []

    def _refresh_moving_focus_twice():
        reads.append(1)
        if len(reads) == 2:
            state.focused_panel = "right"
        elif len(reads) == 3:
            state.focused_panel = "left"
        return original_refresh()

    panel._refresh_panel_context = _refresh_moving_focus_twice
    writes_before = len(state.write_targets)
    panel.update(doc)
    panel._refresh_panel_context = original_refresh

    assert len(reads) >= 3, "the restarted single-record pass never revalidated"

    assert state.write_targets[writes_before:] == ["left"], (
        "the twice-retargeted live record was written anyway"
    )
    assert state.panel_ranges["left"] == (
        pytest.approx(0.20),
        pytest.approx(1.00),
    ), "the frozen Near did not reach the panel it was typed into"
    assert state.panel_ranges["right"] == (2.00, 9.00), "the other panel was written"
    assert list(panel._deferred_depth_commits) == ["selection_depth_far_str"], (
        "the second retarget was consumed instead of left deferred"
    )


def _read_src(*parts: str) -> str:
    return (Path(__file__).parent.parent.parent.joinpath(*parts)).read_text(
        encoding="utf-8"
    )

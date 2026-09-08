# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for viewport selection controls."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from xml.etree import ElementTree
import re
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
        # The manager's record of which panel the last independent-dual collapse
        # folded (rendering_manager.cpp, the collapse branch of
        # applyDepthWindowModeTransitionLocked). A test that leaves the mode sets
        # this the way the native transition would, because the split service
        # resets the observable focus to Left in the same transition and the
        # plugin has no other way to learn the source panel.
        depth_window_collapse_source="left",
        # How many slot-invalidating writes the manager has stamped: ALL FOUR
        # lineage writes bump it, not only the independent-dual collapses --
        # the sync-ON copy and the two fresh-baseline restores do too
        # (rendering_manager.cpp, stampDepthWindowLineageLocked bumps it beside
        # the source and the kind, under the same lock). A test that models one
        # of those writes bumps it the way the native side would; leaving it
        # BEHIND the writes it describes is how a test models a cycle the
        # poller slept through.
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
        # MODEL the production copy (rendering_manager.cpp,
        # setDepthWindowSync): turning sync ON in independent-dual with
        # DIFFERING slots copies the focused panel's window over the other and
        # stamps the lineage record inside the same critical section. Flipping
        # only the flag would let a test pass while the reference the copy
        # invalidated was never reconciled.
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


def test_the_eye_icons_the_viz_cycle_dropped_are_still_used_elsewhere(
    selection_controls_module,
):
    """The cycle stopped using the eye pair; the rest of the app did not.

    `scene/visible.png` and `scene/hidden.png` are the app's general
    show/hide glyphs. Replacing them in this one cycle must not read as
    permission to retire them.
    """
    icons, _state = selection_controls_module
    assert not any(
        name.startswith("../icon/scene/") for name in icons._VIZ_MODE_ICONS.values()
    )
    preview = _read_src(
        "src", "visualizer", "gui", "rmlui", "resources", "image_preview.rml"
    )
    assert "../icon/scene/visible.png" in preview
    panel = _read_src("src", "python", "lfs_plugins", "image_preview_panel.py")
    assert "../icon/scene/visible.png" in panel
    assert "../icon/scene/hidden.png" in panel


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

    # The blur that the host still delivers must not carry L's number.
    # The cancel reverts the field in place instead of fencing
    # the commit off, so the blur writes R's own canonical value -- a no-change
    # write. Assert that the write ACTUALLY HAPPENED and
    # carried the canonical payload, not merely that the final value is
    # unchanged -- the weak form passed under the old blanket-swallow code.
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
    """Regression.

    The toggle path refreshes the cached sync flag itself, so update() can never
    see the edge; the reconciliation has to run on the toggle path. Sync ON
    means one window again, collapsed from the FOCUSED panel -- so the shared
    reference must come from that panel's entry, not stay stale.
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
    """Regression.

    `not self._depth_sync` computed from the CACHE is only a toggle while the
    cache agrees with the manager. If the flag moved externally since the last
    100ms poll, the click re-requests the value the manager already holds and
    nothing toggles. The toggle path must refresh first.

    That refresh is not merely a read. It consumes a
    PENDING external sync edge, so the click drives TWO reconciliations, and the
    test has to pin both by VALUE:

      * the pending ON is reconciled first -- one window again, collapsed from
        the focused panel, so the shared reference becomes Left's .60 (not the
        stale .35, and not Right's .20);
      * the toggle then turns sync OFF, which seeds BOTH panel entries from that
        shared one, so Right's .20 is replaced by .60 as well.
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
    """Regression: the real ordering.

    An operator/MCP/project-restore focus change sends no pointer input
    (depth_window_ops.cpp:430 focuses the panel directly), so the retained
    input is never blurred AND never re-focused. Continued typing plus Enter
    must still commit: the cancel reverted the field in place, so there is
    nothing left to fence off.
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

    # An INTERVENING poll. The overlay updates every
    # 100ms (rml_viewport_overlay.hpp:215), so a real user's keystroke and
    # their Enter are always separated by several of these. If the retarget
    # dropped the key from _editing_depth_text, _sync_depth_text_bufs() would
    # treat the still-focused field as unedited and overwrite 4.25 with the
    # canonical text right here.
    panel.update(doc)
    assert panel._depth_text_bufs["selection_depth_near_str"] == "4.25"

    doc.near.emit("change", _InputEventStub(linebreak=True))

    assert state.depth_near == pytest.approx(4.25)


def test_a_second_focus_change_retargets_the_retained_field_again(
    selection_controls_module,
):
    """Regression: the guard must survive to fire twice.

    With the edit-origin entry removed on the first cancel, a SECOND external
    focus change sees no live edit and runs no guard at all. Retargeting keeps
    the key registered, so each change reverts the field again and the origin
    always names the panel on screen.
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


def test_untouched_field_commits_the_canonical_value_after_a_cancel(
    selection_controls_module,
):
    """Regression: the harmless half of the revert.

    Cancel, then a blur with nothing typed since: the commit LANDS but writes
    the canonical value of the panel now on screen, never panel A's text.
    """
    module, state = selection_controls_module
    _independent_dual(state, focused="left")
    panel, model, doc = _mounted_panel(module, state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")
    state.focused_panel = "right"
    panel.update(doc)

    before = state.depth_near
    doc.near.emit("blur")

    assert state.depth_near == pytest.approx(before)
    assert state.depth_near != pytest.approx(3.75)


def test_a_toolbar_action_retargets_the_active_edit_whose_edge_it_consumes(
    selection_controls_module,
):
    """Regression: update() is not the only edge consumer.

    `_on_action` refreshes the panel context after every toolbar action. While
    that refresh was the only place the new focus was adopted and the guard
    lived in update(), the action swallowed the focus edge and the edit kept its
    stale Left origin -- so the blur that followed wrote Left's text into Right.
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
    """Regression: the commit path's own validation.

    Enter or blur can arrive before ANY refresh has observed the focus change,
    so no channel had a chance to retarget. `_commit_depth_text_key` therefore
    validates the recorded origin against a fresh context read of its own; on a
    mismatch the retarget rule applies and the canonical value is what gets
    written, never the stale-origin text.
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
    """Regression: the legitimate-retype case.

    The retarget reverts the field, but the user may legitimately want the very
    number they had typed before it. Retyping the EXACT stale string must land
    natively: nothing keyed on the text itself survives the revert, so there is
    no collision left to mis-refuse it.
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
    """Regression: collapse provenance across a coalesced leave.

    Cached focus Left. Native focus moves to Right and the mode then LEAVES
    independent-dual, both before one poll. The manager collapsed RIGHT's window
    (it captures the pre-transition focus), while the split service reset the
    observable focus to Left in the same transition. The plugin's cached
    "previous" panel is Left and cannot recover Right, so it reads the manager's
    record of what it actually collapsed; seeding shared from Left's .60 would
    put the displayed window and the Size reference on different panels.
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
    """The same provenance loss through the focus -> sync-ON -> leave ordering.

    Three native edges coalesce into one poll. The leave is the last of them and
    is what fixed the final window, so its recorded source is the one that must
    win.
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
    """Regression: a cycle that runs entirely between two polls.

    Cached {L=.60, R=.20} with Right focused. Natively: the first leave collapses
    Right's .20; the re-entry seeds BOTH slots from it and resets focus to Left;
    the final leave collapses Left -- which is now .20 -- and records source
    Left. Python sees only independent -> none, so endpoint identity hands it
    "Left" and its cached Left reference is still the .60 that stopped existing
    at the first collapse. The collapse GENERATION is what exposes the cycle:
    two collapses for the one transition observed. Nothing cached can be
    replayed at that point, so the shared reference is re-seeded from the window
    that actually survived -- fresh-baseline semantics, as if a new shape had
    just been drawn.
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
    """The same rule, with the surviving window matching NEITHER cached entry.

    That cycle happens to end on a value one cached entry also holds, so it
    cannot tell "re-seeded from Right's cache" from "re-seeded from the native
    window". Here the native window that survived is .20 while the cached
    entries are .60 and .45, so only a fresh read of the native state produces
    the asserted value.
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
    """The other half of the rule: an EXACT delta keeps the plain behaviour.

    One observed leave, one recorded collapse. Nothing was missed, so the cached
    reference of the panel the manager named is still valid and must win --
    the resync is for missed boundaries only, never for the ordinary path.
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
    """Regression: the reproduction below.

    Edit Left, native focus moves to Right, then independent-dual is left, all
    before one poll. The manager collapsed RIGHT's window into the single
    remaining one and the split service reset the observable focus to Left, so
    the edit's recorded origin (Left) EQUALS the post-transition focus and the
    focus-equality guard sees nothing. The mode boundary is what exposes it: the
    field addresses a different window than the one the text was typed against,
    so the edit is reverted to the post-transition canonical value and
    retargeted. The typed 3.75 must never reach the native side.
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
    """Regression: the reopen ordering.

    Cached {L=.60, R=.20} with Right focused. Natively the mode is left (Right's
    .20 collapses), then re-entered (both slots seeded from .20, focus reset to
    Left) -- all between two polls, so the poll sees independent-dual before and
    after and NEITHER endpoint predicate fires. Only the lineage generation
    witnesses it. Left's cached .60 describes a window that stopped existing at
    the collapse; leaving it in place makes the Size field read 33% and every
    later Size edit measure against a shape that is gone.
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
    """Regression: sync-ON is a collapse path too.

    Hidden ON -> OFF -> focus change -> ON. Two native copies happened, so the
    delta is two while the poll observes a single unsynced -> synced edge. No
    cached entry survived both copies, so the shared reference fresh-baselines
    from the window that actually exists.
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
    """The other half: ONE sync copy is replayable from the recorded source.

    The native side copied the source panel's window over the other, so that
    panel's cached reference is still the one describing the surviving window --
    and a fresh-baseline resync here would needlessly discard it. The native
    window is deliberately a THIRD value, so only the cached source entry can
    produce the asserted result.
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
    """Regression: a restore is a reference-lifetime discontinuity.

    restoreDepthWindowStateFromProject() seeds BOTH slots from the restored
    projection. Beginning and ending independent and unsynced, it presents NO
    mode edge, NO sync edge and NO focus edge -- the lineage stamp is the only
    witness. Every cached entry describes the previous project's windows, so all
    of them fresh-baseline from the restored state.
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
    """Regression: the sync UNDO/REDO restore is a lineage producer too.

    DepthWindowSyncUndoEntry restores two possibly-differing ABSOLUTE window
    snapshots, so no cached per-panel reference survives it. Undo and redo BOTH
    between two polls leaves the sync flag exactly where it started: there is no
    flag edge, no mode edge and no focus edge, and the generation stamps (kind
    'project_restore', meaning fresh-baseline required rather than literally a
    project load) are the only witness the poller gets.
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
    """Regression: a fresh baseline is PER PANEL, not per projection.

    A SINGLE sync undo (delta == 1, kind 'project_restore') restores two
    DIFFERING absolute windows at once and ends independent-and-unsynced, so
    there are two live windows and no edge of any kind to observe. Baselining
    all three entries from the focused projection would leave Left's reference
    describing RIGHT's window: focusing Left would then report .60 against a
    .20 reference -- 300% -- and every later Size edit would scale from it.

    Each panel entry must therefore baseline from ITS OWN slot
    (get_depth_filter_window(panel=...)), leaving both panels reading 100%,
    while the shared entry -- the one consulted in every other endpoint --
    takes the focused projection.
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
    """Regression: a torn endpoint/lineage read must not stick.

    The endpoint (mode/focus/sync) and the lineage record are separate native
    reads. A sync undo landing BETWEEN them -- restoring {L=.60, R=.20,
    sync=false} and stamping 'project_restore' -- used to be processed against
    the stale sync=true, so the fresh baseline took its single-window branch
    and seeded all three entries from the focused .20. The NEXT poll then saw
    the sync-OFF edge and copied that shared .20 into both panel entries, so
    Left reported 300% and never healed.

    _refresh_panel_context now reads the record, then the endpoint, then the
    record AGAIN, and retries the whole set when the generation moved.
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
    """Regression: an exhausted tick is a NO-OP, not a partial one.

    If the generation moves on EVERY re-read the loop cannot win, and the set
    it holds is torn: the endpoint can be pre-write while the record is
    post-write. The real storm is a toolbar Undo -> Redo -> Undo of the sync
    flag, three restores that each rewrite BOTH absolute windows and stamp
    'project_restore'. Landed one per attempt, between the endpoint read and
    the second record read, they leave the last attempt holding an endpoint
    that says sync=TRUE while the world it describes is already back to
    sync=false with {L=.60, R=.20}. Consuming that set seeds every entry from
    the focused .20 and the next poll's sync-OFF edge copies it into both
    panels, so Left reports 300% forever.

    So the exhausted tick consumes NOTHING -- not the endpoint, not the lineage
    record, no reconciliation, no retarget -- and the whole delta stays pending
    for the next poll, which re-reads a stable world and baselines each panel
    entry from its own slot.
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
    # Three REAL transitions landed: equal/on -> distinct/off -> equal/on ->
    # distinct/off. The last attempt's endpoint was read after leg 2, so it says
    # sync=TRUE over a world that leg 3 had already put back to sync=false with
    # {L=.60, R=.20} -- the interleaving that used to corrupt. Nothing from that
    # torn set was consumed: not the endpoint, not the generation, not one
    # reference entry; every cached field is still the pre-storm one.
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
    """Regression: the active-edit half of the inert-tick rule.

    Caching the endpoint on an exhausted tick while deferring the lineage is
    NOT a safe halfway house. The mode/focus retarget guard fires on the cached
    edge and canonicalizes the Size field -- but the reconciliation that would
    have re-seeded the references was skipped, so the canonical text is
    computed from a stale reference. With native scale .20 against a stale
    shared reference of .90 the field reads 22%, and a commit landing during a
    SECOND exhausted refresh then writes that 22% into native state, which no
    later poll can undo.

    So the exhausted tick touches nothing: the mode is not cached, the guard
    does not fire, the buffer the user is typing is left alone, and a commit
    whose validating refresh comes back exhausted DEFERS rather than writing.
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

    # And the deferred commit resolved against that stable world -- the poll
    # above retried it (_flush_deferred_depth_commits), no second Enter needed.
    # The stable poll DID observe the mode boundary, so the settled retarget
    # rule reverted the buffer to that panel's canonical text first: the commit
    # wrote 100% of the reconciled reference, an idempotent no-change write,
    # never the 22% the torn snapshot would have produced.
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
    """A record getter that exhausts the revalidation loop at a STILL generation.

    Every second read fails, and a failed read reports generation None
    (_depth_window_collapse_record), so no attempt ever gets a comparable pair
    -- while the native generation never moves at all. That is the read storm
    under reproduction, and it keeps the exhaustion free of any phantom
    lineage delta that would fire the retarget guard for unrelated reasons.
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
    """Regression: exhaustion may delay a revert, never drop it.

    Reproduction: native .25, Enter commits 1.0, Escape restores the
    buffer to .25, and the blur that cancelFocusedElement fires immediately
    afterwards -- the blur that CARRIES the revert -- lands mid-storm. Its
    commit defers, but the widget calls on_blur unconditionally
    (rml_widgets.bind_committed_text_input), so forgetting the key there would
    strand the revert: native would stay 1.0 with nothing left to retry.

    The blur is COMPLETE, so the live-edit membership is retired exactly as an
    undeferred blur retires it; what carries the revert is the deferral record,
    which freezes the reverted text and the origin panel. The first stable poll
    retries the write from that frozen record.
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
    """Regression: one key's commit must not eat another's buffer.

    Reproduction: Near 0.75 and Far 8.25 both deferred by the storm. The
    flush walks every deferred key, but the first commit to land ends in
    _sync_depth_text_bufs(force=True), and a forced sync that only skipped LIVE
    edits canonicalized the still-deferred sibling -- Near finished at its
    native 0.25 with the record retired. BOTH values are the user's, and both
    must land.
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
    """Regression: the flush must not reorder the user's edits.

    Reproduction, from native Near 0.90 / Far 1.00: blur-defer Near 0.20,
    then blur-defer Far 0.50. Near and Far are order-dependent, because each
    write clamps against the counterpart AS IT STANDS at that moment. Replayed
    in the user's own order the pair lands (0.20, 0.50). Replayed Far-first --
    which is what a `sorted()` walk of the record keys does, since
    "selection_depth_far_str" sorts before "selection_depth_near_str" -- Far is
    clamped up against the still-native 0.90 to 0.91, and the 0.50 the user
    typed is destroyed before Near has moved out of its way.
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
    """Regression: the same ordering law across the two record kinds.

    Identical sequence to the blurred/blurred case, except the Far edit is still
    LIVE (committed with Enter, field still focused) when the flush runs. The
    two kinds take different write paths -- frozen payload versus a re-read of
    the live buffer -- and both must be replayed at the position the user's
    event took, not at the position their key name sorts to.
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
    """Regression: no replay ORDER is safe for the clamped pair.

    Reproduction, from native Near 0.90 / Far 1.00: blur-defer Near 0.20,
    blur-defer Far 0.50, then re-focus Near -- which SUPERSEDES and drops its
    record -- and re-arm it with the same 0.20. The record is reinserted at the
    registry tail, so the insertion order is now [Far, Near] and is honestly
    chronological: Near's surviving record really was made last. Replaying that
    order serially still clamps Far up against the untouched native Near 0.90 to
    0.91 and destroys the 0.50.

    Paired resolution removes the dependence on order entirely: the target's
    final intended (Near, Far) is resolved first and written as ONE update, so
    the clamp sees the user's own Near, whichever record arrived last.
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
    """Regression: pairing is PER TARGET, panel path included.

    Both blurs are frozen to Left, and focus has moved to Right by the time the
    flush runs, so the pair is written through the panel= setter overload rather
    than the displayed window. That path clamps Far against the panel's OWN
    native Near, so serial replay destroys the 0.50 there exactly as it does on
    the focused path -- here the user happens to edit Far first, which is enough
    on its own. Resolved as a pair, the two fields reach Left in a single
    panel-addressed write and Right is never touched.
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
    """Regression: no write may use a PRE-revalidation target.

    Reproduction, on two independent windows with focus on Left and
    Left=(0.90,1.00), Right=(2.00,9.00): Near is blurred and frozen to Left at
    0.20 while Far is still LIVE. The flush groups both under the target None,
    because a no-panel write reaches Left at that moment and a live record is
    always aimed at the context the user is editing in. Then the live member's
    own revalidation -- the first read to observe it -- sees focus move L to R.

    That single read invalidates the grouping twice over. The live Far is
    retargeted to Right (its buffer becomes Right's canonical text), and the
    target named None now means Right, so the combined write carried the frozen
    Left intent into a panel the user never touched: Right became (0.20, 9.00),
    Left kept (0.90, 1.00), and BOTH records were consumed by a write that was
    not theirs.

    Re-deriving identity and target after the revalidation splits the pair back
    apart: the frozen Near is panel-addressed to Left, and the retargeted Far
    follows its context to Right on its own.
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
    """Regression: genuinely mixed targets, no mid-flush move.

    Near is blurred and frozen to Left while Far is LIVE on the focused panel,
    Right -- the two records really do belong to different windows, and the
    context does not move during the flush. Each has to take its own write and
    clamp against its OWN panel's counterpart: Far's 0.50 is below Right's
    native Near 2.00 and must ride up to 2.01, which it cannot do if the pair
    was resolved together against Left's 0.20.
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
    """Regression: a COMPLETED blur is not retargetable.

    Reproduction: Left Near 0.75 deferred by the storm, focus then moves
    to Right. Holding the blurred edit in _editing_depth_text made the focus
    transition treat it as live, so _cancel_foreign_depth_text_edits replaced
    both its payload and its target -- the flush wrote Right's canonical value
    to Right and Left stayed 0.25. The user's finished intent must land on the
    panel they typed it into, whatever focus did afterwards.
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
    """Regression: the latest intent wins, cleanly.

    Reproduction: defer 0.75 by blurring mid-storm, then re-focus the
    field and type 0.90. With the stale blurred record still in place the flush
    committed and then retired the edit the user was in the middle of, so the
    1.10 they typed next was overwritten back by the following poll. Re-focusing
    supersedes the pending record instead, and the new edit owns its lifecycle.
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
    """Regression: `not self._depth_sync` is only a toggle when
    the refresh actually refreshed.

    An exhausted refresh consumes nothing, so the cache stays at its stale
    pre-refresh value. With the cache false and the manager already true,
    `not cache` requests TRUE -- a no-change write that loses the user's click
    while pretending to have served it. The click is dropped instead, exactly
    as the manager drops one refused mid-drag.
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
    """Regression: the generation-None fallback keeps the SECOND read.

    An older binding exposes only the single-value source getter, so both
    record reads report a generation of None and the revalidation loop ends on
    its first attempt. It must still retain the POST-endpoint read: if a
    Right-panel leave lands between the two reads, the endpoint is already
    post-leave while the first record still names Left, and the observed-leave
    rule would seed the shared reference from Left's window -- permanently,
    because the next poll sees neither a generation nor an endpoint edge.
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
    """Regression: ordinary panel/tool suspension must not swallow the edge.

    update() returns before refreshing anything while the controls are hidden,
    so the last-consumed generation only advances when a refresh actually
    reconciles. The FIRST refresh after the controls come back therefore still
    sees the whole accumulated delta.
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


def test_a_no_op_mode_change_keeps_a_legitimate_typed_buffer(
    selection_controls_module,
):
    """Regression: mirror the NATIVE boundary predicate.

    applyDepthWindowModeTransitionLocked treats Disabled <-> PLYComparison as a
    complete depth-window no-op: no collapse, no seed, not even an epoch bump.
    The field the user is typing into still addresses exactly the same window,
    so cancelling the edit would destroy a legitimate buffer for nothing.
    """
    module, state = selection_controls_module
    panel, model, doc = _mounted_panel(module, state)

    doc.near.emit("focus")
    model.bound_binds["selection_depth_near_str"][1]("3.75")

    state.split_view_mode = "ply_comparison"
    panel.update(doc)

    assert panel._split_mode == "ply_comparison"
    assert panel._depth_text_bufs["selection_depth_near_str"] == "3.75", (
        "a no-op mode change replaced a legitimate buffer with canonical text"
    )

    writes_before = len(state.window_calls)
    doc.near.emit("change", _InputEventStub(linebreak=True))

    assert len(state.window_calls) == writes_before + 1
    assert state.window_calls[-1][1] == pytest.approx(3.75)
    assert state.depth_near == pytest.approx(3.75)


def test_the_sync_toggle_pre_reads_the_focused_canonical_and_stays_idempotent(
    selection_controls_module,
):
    """Regression: the cross-path test with a MODELLED copy.

    Distinct per-panel ranges, and a sync setter that performs the production
    copy (focused slot over the other, plus the lineage stamp) rather than
    merely flipping the flag. The specification is that the
    focus refresh reverts the edit to RIGHT's canonical (the focused-canonical
    pre-read, taken before the copy), sync then copies that same Right window to
    both slots, and the later blur re-emits that value -- idempotent, never the
    typed text and never Left's canonical.
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
    """Regression.

    DepthWindowSyncUndoEntry restores the sync flag natively
    (depth_window_undo_entry.cpp:85). The plugin observes that edge only
    through its own refresh, so the refresh -- not the toggle path -- has to be
    what reconciles. Example: {L=.60, R=.20}, toggle ON, Undo, and both
    references must be seeded from the shared .60.
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


def test_get_depth_filter_window_keeps_its_seven_tuple_shape(lf):
    window = lf.selection.get_depth_filter_window()

    assert isinstance(window, tuple)
    assert len(window) == 7
    enabled, near, far, scale_x, scale_y, offset_x, offset_y = window
    assert isinstance(enabled, bool)
    for value in (near, far, scale_x, scale_y, offset_x, offset_y):
        assert isinstance(value, float)


@pytest.mark.parametrize("panel", [None, "main", "left", "right"])
def test_depth_filter_window_accepts_every_panel_token(lf, panel):
    window = lf.selection.get_depth_filter_window(panel=panel)

    assert len(window) == 7


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


def test_focused_split_panel_getter_reports_a_panel_token(lf):
    assert lf.ui.get_focused_split_panel() in ("left", "right")


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
    """Regression: DEFERRED must stop the ENTIRE flush.

    Probe: focus is cached at Left while native focus has already moved
    to Right, and the read that would reveal it comes back EXHAUSTED. The
    Near/Far pair correctly reports DEFERRED and is retained -- but the caller
    only ever branched on RETARGETED, so control fell straight through to the
    group's remaining records. The frozen Size record was consumed, and because
    the CACHED focus still said Left its origin panel looked like the focused
    one, so it took an ordinary no-panel write -- which native routing delivered
    to RIGHT. A record frozen to Left wrote into the panel the user never
    touched, on a tick whose only context read had already failed.

    An exhausted read is a statement about the WORLD, not about one record. No
    record may write or be consumed while it holds. Zero writes, zero
    consumption, all three records still pending.
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
    """Regression: unpaired records join the regroup protocol.

    A two-change attack on two independent windows, focus Left,
    L=(0.90,1.00) / R=(2.00,9.00). Near is blurred and frozen to Left at 0.20,
    Far is LIVE. Focus moves L to R on the pair's revalidation -- the pair
    reports RETARGETED and the flush regroups, which splits the two records into
    two SINGLE-record groups: the frozen Near now needs a panel-addressed write
    to Left, and the live Far follows the context. Focus then moves back R to L
    on the Far single's own revalidation.

    A single record used to revalidate and then write with no way to report the
    move, so that second retarget was consumed: writes landed twice on Left and
    the registry emptied. The single path now returns the same three outcomes as
    the pair, the restart budget is already spent, and so the second retarget
    DEFERS. Only the write whose target still matches its own derivation lands:
    the panel-addressed Near into Left.
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


def _viewport_overlay_key_block() -> str:
    """The overlay's keyboard-forwarding block, gate line included.

    Substring searches over the whole file cannot tell live code from dead
    code: the select branch of the Escape handler was present, matched every
    grep, and was unreachable because the gate above it excluded selects. These
    tests read the block the branch actually lives in.
    """
    overlay = _read_src("src", "visualizer", "gui", "rml_viewport_overlay.cpp")
    start = overlay.index("if (auto* focused = rml_context_->GetFocusElement())")
    end = overlay.index("ProcessTextInput(static_cast", start)
    return overlay[start:end]


def test_viewport_overlay_escape_branch_is_reachable_for_every_cancel_target():
    """Escape in a depth text field must reach the plugin's revert path.

    The revert itself is plugin-side (`_restore_depth_text_snapshot`, bound
    through `rml_widgets.EscapeRevertController` on the custom `escapecancel`
    event), and it is only reachable if the HOST dispatches that event -- and
    only if the gate ABOVE the dispatch admits the focused element in the first
    place. `publishOverlayTextFocus` alone does not: `wantsTextInput` excludes
    a normal `<select>`, so gating on it exclusively made the select half of
    the branch dead code. The decision itself is pinned in C++ against real
    RmlUi elements (`OverlayEscapeContractTest`); this pins that this host
    still routes through it.

    COVERAGE NOTE -- the two ENDS of this path are behavior-tested, the middle is
    not. The plugin end is driven for real in this file (`escapecancel` is
    emitted on live elements and the revert is observed); the shared rule is
    driven for real in `OverlayEscapeContractTest`. What no headless process can
    drive is the C++ host between them, so this reads the gate line above the
    branch -- which is exactly the thing a substring search over the whole file
    could not see, and exactly what let the select branch sit there dead.
    """
    block = _viewport_overlay_key_block()

    assert "shouldCancelOnEscape" in block, (
        "the overlay no longer uses the shared Escape-cancel rule, so its "
        "behavior can drift from the sidebar host's again"
    )
    assert "rml_input::cancelFocusedElement" in block, (
        "the viewport overlay does not dispatch escapecancel, so the plugin's "
        "Escape revert stays dead in this host"
    )
    assert "isSelectRelatedElement" in block, (
        "the gate does not admit a focused select, so the Escape branch is "
        "unreachable for the overlay's depth/GT/export dropdowns"
    )
    gate = block[: block.index("SDL_SCANCODE_ESCAPE")]
    assert "publishOverlayTextFocus" in gate and "select_focus" in gate, (
        "Escape handling is no longer nested under a gate that admits both "
        "text focus and select focus"
    )


def test_overlay_press_focus_is_classified_from_the_press_coordinates():
    """Toolbar chrome is not a viewport, and neither is the left dock.

    A press on an interactive overlay control must not re-point the focused
    split panel at whatever panel the control happens to be drawn over: the
    action the control fires (the sync toggle, for one) reads the focused
    panel, and would otherwise read a focus its own press had just moved. The
    press that DISMISSES a text field is the opposite case -- it moves focus,
    after the blur has committed -- but only when it landed inside the viewport
    rectangle, because the overlay's bounds are stretched over the left dock.

    All of it has to be decided from the coordinates SDL reported on the
    button-down: a frame's events are delivered together, so `mouse_x/mouse_y`
    already carry any motion queued behind the press.

    COVERAGE NOTE -- what this test is and is not. It is a WIRING assertion over
    the live call site, not a behavior test: nothing in this process can drive a
    C++ GUI frame. The behavior it guards is pinned elsewhere and only there --
    `DepthWindowOverlayPressFocusTest` in `tests/test_depth_window_panels_interaction.cpp`
    covers the decision (truth table over the real struct), the capture and copy
    (synthetic SDL events through `FrameInputBuffer`), and the ownership record.
    Nothing reachable from Python covers them, so this stays a wiring check
    rather than being deleted or dressed up as more.
    """
    overlay_hpp = _read_src("src", "visualizer", "gui", "rml_viewport_overlay.hpp")
    gui_manager_cpp = _read_src("src", "visualizer", "gui", "gui_manager.cpp")
    frame_buffer_hpp = _read_src(
        "src", "visualizer", "input", "frame_input_buffer.hpp"
    )

    # The coordinates and the ownership verdict live on ONE canonical event in
    # `FrameInputBuffer::mouse_button_events`
    # (src/visualizer/input/frame_input_buffer.hpp), recorded at BUTTON_DOWN.
    # R10 replaced the per-button press-record array with that ordered event
    # vector; the verdict became the event's own `gui_owned` field, so the two
    # still cannot be copied apart.
    assert "mouse_button_events.push_back" in frame_buffer_hpp and (
        "SDL_EVENT_MOUSE_BUTTON_DOWN" in frame_buffer_hpp
    ), "the input buffer no longer records where the press landed"
    assert ".x = event.button.x," in frame_buffer_hpp, (
        "the press coordinates are no longer captured from the BUTTON_DOWN event"
    )
    assert ".gui_owned" in frame_buffer_hpp, (
        "the ownership verdict no longer rides on the press event that carries "
        "the coordinates, so the two can be taken from different presses"
    )
    assert "lastPress" in _read_src("src", "visualizer", "gui", "panel_layout.hpp")

    for symbol in (
        "OverlayPressFocusInputs",
        "pointInsideViewport",
        "press_inside_viewport",
    ):
        assert symbol in overlay_hpp, f"{symbol} is gone from the press rule"

    start = gui_manager_cpp.index("overlayPressMayFocusPanel(")
    end = gui_manager_cpp.index("setFocusedSplitPanel", start)
    focus_block = gui_manager_cpp[start:end]

    assert "pointInsideViewport(" in focus_block, (
        "the focus block admits presses without checking they landed in the "
        "viewport, so a left-dock click can refocus a split panel"
    )
    assert "viewport_layout_.pos" in focus_block and "viewport_layout_.size" in (
        focus_block
    ), "containment is not being tested against the viewport rectangle"
    assert "overlay_press_point" in focus_block, (
        "resolveViewerPanel is no longer given the press coordinates"
    )
    assert "viewport_overlay_input.mouse_x" not in focus_block, (
        "the focus block fell back to the frame's latest cursor position, "
        "which coalesced motion has already moved off the pressed target"
    )
    assert "overlay_press->gui_owned" in focus_block, (
        "the focus block no longer consults the GUI's event-time verdict, so a "
        "press on the left-dock resize strip -- whose hitbox straddles the "
        "viewport's left boundary and therefore passes containment -- can "
        "refocus a split panel while a depth field is focused"
    )

    # The rev-2 shapes the C++ tests cannot pin (they re-model the fold in
    # test-local code, so they stay green if the PRODUCTION loop regresses).
    # Slice the live sequential fold: every left DOWN gets its own verdict in
    # SDL order, and a refusal continues instead of erasing earlier focus.
    seq_start = gui_manager_cpp.index("const auto& overlay_left_presses")
    seq_end = gui_manager_cpp.index("const bool has_python_overlay_hooks", seq_start)
    seq_block = gui_manager_cpp[seq_start:seq_end]

    assert (
        "for (const auto& overlay_event : viewport_overlay_input.mouse_button_events)"
        in seq_block
    ), (
        "the focus fold no longer walks the canonical per-event vector, so it "
        "cannot decide each press in SDL order"
    )
    skip_at = seq_block.index("if (!overlay_event.down || overlay_event.button != 0)")
    assert "continue" in seq_block[skip_at : skip_at + 120], (
        "non-DOWN / non-left events are no longer skipped inside the fold"
    )
    assert "overlay_left_presses[overlay_press_index++]" in seq_block, (
        "the fold no longer consumes leftPressClassifications() in lockstep "
        "with the event vector, so verdicts and presses can pair off wrong"
    )
    guard_start = seq_block.index("if (!overlayPressMayFocusPanel({")
    guard_end = seq_block.index("if (auto* const rendering", guard_start)
    assert "continue" in seq_block[guard_start:guard_end], (
        "a refused admission no longer continues, so a later chrome press can "
        "erase the focus an earlier viewport press legitimately set"
    )
    assert seq_block.index("setFocusedSplitPanel", guard_end) > guard_start, (
        "focus is no longer set after the admission guard inside the loop"
    )
    assert "lastPress(0)" not in seq_block, (
        "the fold regressed to the rejected one-press-per-frame lastPress "
        "summary instead of per-event verdicts"
    )

    # The aggregate fallback may fire ONLY when the frame carried no canonical
    # events at all; keying it on 'nothing was replayed' fabricates transitions
    # at the final cursor from events that were merely unowned.
    overlay_cpp = _read_src("src", "visualizer", "gui", "rml_viewport_overlay.cpp")
    fb_start = overlay_cpp.index("// AGGREGATE FALLBACK")
    fb_end = overlay_cpp.index("if (input.mouse_wheel", fb_start)
    fb_block = overlay_cpp[fb_start:fb_end]
    assert "if (input.mouse_button_events.empty() &&" in fb_block, (
        "the aggregate fallback is no longer gated on an empty canonical "
        "event vector"
    )
    assert "!replayed_button_events" not in fb_block, (
        "the fallback regressed to 'nothing was replayed', which synthesizes "
        "button events at the final cursor when real events were skipped as "
        "unowned"
    )


def _rcss_value(block: str, prop: str) -> float:
    """First `prop: <n>dp;` (or `%`) declaration in an RCSS rule body."""
    match = re.search(rf"{prop}\s*:\s*([0-9.]+)\s*(?:dp|%)", block)
    assert match, f"no {prop} declaration in:\n{block}"
    return float(match.group(1))


def _rcss_rule(source: str, selector: str) -> str:
    start = source.index(selector)
    open_brace = source.index("{", start)
    return source[open_brace : source.index("}", open_brace)]


def _rcss_bodies(source: str, selector: str) -> list[tuple[int, str]]:
    """Every rule body whose selector line is EXACTLY `selector`, with its offset.

    Anchored so `#selection-block .viewport-selection-depth-axis` does not also
    match `... > .number-input`, and so a selector can legitimately appear more
    than once (this file declares the axis twice).
    """
    bodies: list[tuple[int, str]] = []
    for match in re.finditer(
        rf"(?m)^[ \t]*{re.escape(selector)}[ \t]*\{{", source
    ):
        open_brace = source.index("{", match.start())
        bodies.append((match.start(), source[open_brace : source.index("}", open_brace)]))
    return bodies


def _specificity(selector: str) -> tuple[int, int, int]:
    """(ids, classes, tags) -- enough for the selectors this file uses."""
    return (
        selector.count("#"),
        selector.count("."),
        len(re.findall(r"(?:^|[\s>])([a-zA-Z][\w-]*)", selector)),
    )


def _cascade_value(source: str, selectors: list[str], prop: str) -> float:
    """`prop` for an element matched by ALL of `selectors`, resolved as RmlUi does.

    Reading the first rule that happens to mention a property is what made the
    old derivation wrong: this file declares `.viewport-selection-depth-axis`
    generically AND under `#selection-block`, and the ID-qualified rule is the
    one that applies. Highest specificity wins; a tie goes to the later rule.
    """
    best: tuple[tuple[int, int, int, int], float] | None = None
    for selector in selectors:
        spec = _specificity(selector)
        for offset, body in _rcss_bodies(source, selector):
            match = re.search(
                rf"(?m)^\s*{re.escape(prop)}\s*:\s*([0-9.]+)\s*(?:dp|%)?\s*;", body
            )
            if not match:
                continue
            key = (*spec, offset)
            if best is None or key > best[0]:
                best = (key, float(match.group(1)))
    assert best is not None, f"no {prop} declaration for any of {selectors}"
    return best[1]


def _media_spans(source: str) -> list[tuple[float, int, int]]:
    """Every `@media (max-width: Ndp) { ... }` block as (N, body_start, body_end).

    The single brace-matching parser in this file. `_media_blocks` slices bodies
    out of it; the clearance test needs the OFFSETS instead, so that a rule can
    be told apart from one that merely reads alike outside a block.
    """
    spans: list[tuple[float, int, int]] = []
    for match in re.finditer(r"@media \(max-width:\s*([0-9.]+)dp\)\s*\{", source):
        depth = 0
        for index in range(match.end() - 1, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((float(match.group(1)), match.end(), index))
                    break
    return spans


def _media_blocks(source: str) -> list[tuple[float, str]]:
    """Every `@media (max-width: Ndp) { ... }` block, as (N, body), in file order."""
    return [
        (width, source[start:end]) for width, start, end in _media_spans(source)
    ]


def _depth_breakpoints(source: str) -> dict[str, float]:
    """The two depth-row breakpoints, found by what each block DOES.

    Keyed by behavior rather than by number so the derivations below cannot go
    stale against a moved breakpoint -- which is exactly how the 1150dp wrap
    number survived long after it stopped covering the single row.

    The depth toolbar has EXACTLY THREE width states -- one line, two lines,
    hidden: a layout that cannot be fitted is hidden outright rather than
    restacked into more lines. A block that stacks the Size/X/Y group
    one axis per line is a fourth state and is rejected here, at the one place
    every depth-row test goes through.
    """
    found: dict[str, float] = {}
    for width, body in _media_blocks(source):
        if "display: none" in body and "viewport-selection-depth-fields" in body:
            found["hide"] = width
        elif "depth-axis-group-sxy" in body:
            found["stack"] = width
        elif "flex-wrap: wrap" in body and "padding-right" in body:
            found["wrap"] = width
    assert "stack" not in found, (
        f"a stacked Size/X/Y tier was reintroduced at {found['stack']:g}dp. The "
        f"depth toolbar has exactly three states -- one line, two lines, hidden "
        f"-- and a width that cannot hold two lines must hide the row, not stack "
        f"it"
    )
    assert set(found) == {"hide", "wrap"}, (
        f"expected a hide and a wrap breakpoint, found {sorted(found)}"
    )
    assert found["hide"] < found["wrap"], (
        f"the depth-row breakpoints are out of order: {found}"
    )
    return found


def test_depth_row_never_over_constrains_its_widest_line_at_any_width():
    """No window width may leave the depth row wider than the panel it sits in.

    A reported failure at a width where the row was `nowrap` but the panel was
    far too narrow for it: the row needs 914dp of content and the
    old 1150dp wrap breakpoint only guaranteed 705.9dp, so 1151-1470dp rendered
    a single squeezed line with labels and value boxes overprinting.

    The two breakpoints are NOT keyed on different measures -- RmlUi resolves
    `max-width` against the context dimensions, and the panel is 65% of that
    same context in every split mode -- so the fix is arithmetic on one measure.
    There are exactly three states -- one line, two lines, hidden -- and this
    walks the two VISIBLE ones, asserting the widest line of
    each fits at the NARROWEST width where that state is still active, which is
    the boundary below it (the boundary itself is the conservative worst case).
    Below the hide breakpoint there is no line to fit.
    """
    rcss = _read_src(
        "src", "visualizer", "gui", "rmlui", "resources", "viewport_overlay.rcss"
    )
    breakpoints = _depth_breakpoints(rcss)
    wrap_body = dict(_media_blocks(rcss))[breakpoints["wrap"]]
    hide_body = dict(_media_blocks(rcss))[breakpoints["hide"]]

    axis_selectors = [
        "#selection-block .viewport-selection-depth-axis",
        ".viewport-selection-depth-axis",
    ]
    label = _cascade_value(
        rcss,
        [".viewport-selection-depth-axis > .viewport-transform-axis-label"],
        "min-width",
    )
    axis_gap = _cascade_value(rcss, axis_selectors, "gap")
    slider = _cascade_value(
        rcss,
        [
            "#selection-block .depth-axis-group .viewport-selection-slider",
            "#selection-block .viewport-selection-slider",
            ".viewport-selection-slider",
        ],
        "min-width",
    )
    number = _cascade_value(
        rcss,
        [
            "#selection-block .viewport-selection-depth-axis > "
            ".number-input.viewport-selection-depth-input",
            ".viewport-selection-depth-axis > "
            ".number-input.viewport-selection-depth-input",
        ],
        "min-width",
    )
    size_number = _cascade_value(
        rcss,
        [
            "#selection-block .depth-axis-size > "
            ".number-input.viewport-selection-depth-input"
        ],
        "min-width",
    )
    button = _cascade_value(
        rcss,
        [
            "#transform-block .viewport-transform-option,\n"
            "#transform-block .viewport-transform-action,\n"
            "#selection-block .viewport-transform-option,\n"
            "#selection-block .viewport-transform-action,\n"
            "#depth-view-block .viewport-transform-action"
        ],
        "min-width",
    )
    chip = _rcss_value(
        _rcss_rule(rcss, "#selection-block .viewport-selection-depth-panel-chip"),
        "width",
    )
    row_gap = _rcss_value(
        _rcss_rule(rcss, "#selection-block .viewport-selection-depth-fields"), "gap"
    )
    group_gap = _rcss_value(
        _rcss_rule(rcss, "#selection-block .depth-axis-group"), "gap"
    )
    # Two-line tier: the row reserves space for the pinned buttons on BOTH
    # sides, equally. The right reserve is what the buttons stand in; the left
    # one is its mirror, and only mirrored paddings leave the row's content box
    # -- and therefore both centred lines -- concentric with the row itself.
    # Both sides are read, and their equality asserted, because an asymmetric
    # reserve is exactly the off-centre row this mirroring removes.
    wrap_fields = _rcss_rule(
        wrap_body, "#selection-block .viewport-selection-depth-fields"
    )
    flank = _rcss_value(wrap_fields, "padding-right")
    assert flank == _rcss_value(wrap_fields, "padding-left"), (
        "the depth row's two reserves are no longer equal, so its content box "
        "is off-centre in the row and both wrapped lines sit off-centre in the "
        "panel with the pinned buttons at an edge of their own"
    )
    # Both paddings shrink the content box that the `flex-basis: 100%` lines
    # resolve against, so BOTH are charged to every line.
    reserve = 2.0 * flank
    # The two buttons are stacked ONE PER LINE, not side by side, so each
    # flank has to hold a SINGLE button -- which is what halves the reserve
    # and buys the two-line band back. The split is what makes that legal, so
    # it is asserted here rather than assumed: the shared rule puts both on
    # the same right edge with no horizontal offset between them, and each
    # button then takes its own vertical anchor. If both ever landed on one
    # line again they would overprint each other, and a one-button flank
    # would be too narrow for the pair.
    shared_pin = _rcss_rule(
        wrap_body,
        "#selection-block .viewport-selection-depth-fields > "
        ".viewport-transform-action",
    )
    # `right: 0` is unitless, so this is a text match rather than _rcss_value.
    assert re.search(r"right\s*:\s*0\s*;", shared_pin), (
        f"the pinned depth-row buttons no longer share one right edge:\n"
        f"{shared_pin}"
    )
    anchors = {}
    for name, selector in (
        ("sync", "viewport-selection-sync-action"),
        ("viz", "viewport-selection-viz-action"),
    ):
        rule = _rcss_rule(
            wrap_body,
            f"#selection-block .viewport-selection-depth-fields > .{selector}",
        )
        declared = [side for side in ("top", "bottom") if f"{side}:" in rule]
        assert len(declared) == 1, (
            f"the {name} button declares {declared or 'no'} vertical anchor in "
            f"the two-line tier; it must declare exactly one, which is what "
            f"puts it on one wrapped line and not the other"
        )
        anchors[name] = declared[0]
        assert "right" not in rule, (
            f"the {name} button reintroduced a horizontal offset -- the two "
            f"buttons are separated by LINE now, not by a sideways dodge, and "
            f"an offset one would sit outside the {flank}dp flank"
        )
    assert set(anchors.values()) == {"top", "bottom"}, (
        f"both pinned buttons anchor to the same edge ({anchors}), so they "
        f"ride the same wrapped line and overprint each other"
    )
    assert flank >= button, (
        f"the {flank}dp reserve cannot hold the {button}dp button pinned into "
        f"it -- it would overprint the axis beside it"
    )
    panel = _rcss_rule(rcss, "#selection-block .viewport-selection-panel")
    fraction = _rcss_value(panel, "width") / 100.0
    overlay = _rcss_rule(rcss, "\n.viewport-selection-overlay")
    inset = _rcss_value(overlay, "left") + _rcss_value(overlay, "right")

    plain_axis = label + axis_gap + slider + axis_gap + number
    size_axis = label + axis_gap + slider + axis_gap + size_number
    near_far = chip + group_gap + plain_axis + group_gap + plain_axis
    size_x_y = size_axis + group_gap + plain_axis + group_gap + plain_axis
    # nowrap: both groups plus the two action buttons share ONE line, and the
    # buttons are in flow there (the absolute pinning lives in the wrap block).
    single_row = near_far + row_gap + size_x_y + row_gap + button + row_gap + button

    def content(width: float) -> float:
        return fraction * (width - inset)

    # Tier 1 -- one line, active above the wrap breakpoint. Worst case: the
    # breakpoint itself.
    assert single_row <= content(breakpoints["wrap"]), (
        f"the single-line depth row needs {single_row}dp but only "
        f"{content(breakpoints['wrap']):.1f}dp is available at the "
        f"{breakpoints['wrap']:g}dp wrap breakpoint -- widths just above it "
        "render one over-constrained line"
    )
    # Tier 2 -- two lines, active down to the hide breakpoint. BOTH lines must
    # fit there: the Size/X/Y line is the wider at 500dp, and the Near/Far line
    # is the one the `top: 0` pinned buttons share, so neither may overflow.
    # This is the assertion the deleted stacked tier used to absorb: with three
    # states the two-line tier now runs all the way down to the hide number, so
    # the hide number is what has to be big enough. The chip is a flow item at
    # the head of the Near/Far line, so it is inside `near_far` already.
    widest_two_line = max(near_far, size_x_y)
    assert widest_two_line <= content(breakpoints["hide"]) - reserve, (
        f"the two-line tier's widest line needs {widest_two_line}dp but only "
        f"{content(breakpoints['hide']) - reserve:.1f}dp is available at the "
        f"{breakpoints['hide']:g}dp hide breakpoint -- widths just above it "
        f"render a Size/X/Y line that overflows the panel (hide "
        f"instead of stacking)"
    )
    # Tier 3 -- hidden. There is no line, and there must be no line: nothing
    # below the hide breakpoint may lay the row out again.
    assert "display: none" in _rcss_rule(
        hide_body, "#selection-block .viewport-selection-depth-fields"
    ), "the hide tier no longer hides the depth row"
    # ...and no media block anywhere may stack the Size/X/Y group. This is the
    # rejected fourth state; `_depth_breakpoints` refuses a block that declares
    # it, and this refuses the rules themselves wherever they are written.
    for width, body in _media_blocks(rcss):
        assert "depth-axis-group-sxy" not in body, (
            f"the {width:g}dp media block styles the Size/X/Y group -- a stacked "
            f"tier is the rejected fourth state"
        )
        assert "viewport-selection-depth-axis" not in body, (
            f"the {width:g}dp media block styles an individual depth AXIS. The "
            f"two visible states wrap whole GROUPS (`.depth-axis-group`); a rule "
            f"that reaches a single axis is how a stacked tier gets rebuilt "
            f"under the three-state rule"
        )
    # Neither breakpoint may drift far above what it needs: wrapping or hiding
    # earlier than necessary costs a usable layout.
    assert breakpoints["wrap"] - (single_row / fraction + inset) < 60.0
    assert breakpoints["hide"] - ((widest_two_line + reserve) / fraction + inset) < 60.0


def test_depth_row_hide_breakpoint_clears_the_pinned_toolbar_buttons():
    """The top line must fit before the depth row is allowed to show.

    Below the wrap breakpoint the Near/Far group is the FIRST flex line and the
    sync toggle is absolutely pinned to it (the viz-mode button is pinned to
    the second line). Both lines are laid out in the SAME content box, so the
    row's mirrored padding is charged to both and either line can collide with
    the button beside it. Moving the panel chip into the Near/Far group added
    its width plus a gap to that line, so the breakpoint is arithmetic, not
    taste: this recomputes it from the file.

    Every value is read through the EFFECTIVE cascade -- the highest-specificity
    matching rule -- because this file declares the axis twice with different
    numbers, and reading the first one seen understates the row.
    """
    rcss = _read_src(
        "src", "visualizer", "gui", "rmlui", "resources", "viewport_overlay.rcss"
    )

    narrow = rcss[rcss.index(f"@media (max-width: {_depth_breakpoints(rcss)['wrap']:g}dp)") :]
    axis_selectors = [
        "#selection-block .viewport-selection-depth-axis",
        ".viewport-selection-depth-axis",
    ]
    chip = _rcss_value(
        _rcss_rule(rcss, "#selection-block .viewport-selection-depth-panel-chip"),
        "width",
    )
    label = _cascade_value(
        rcss,
        [".viewport-selection-depth-axis > .viewport-transform-axis-label"],
        "min-width",
    )
    axis_gap = _cascade_value(rcss, axis_selectors, "gap")
    # The generic rule declares gap 3dp; the ID-qualified one 6dp. If the
    # resolver ever falls back to the generic value the whole derivation is
    # understated and the breakpoint silently stops protecting the buttons.
    assert axis_gap != _rcss_value(
        _rcss_bodies(rcss, ".viewport-selection-depth-axis")[0][1], "gap"
    ), "the cascade resolver read the generic axis rule, not the one that applies"

    # The axis SHRINKS in the effective cascade (`#selection-block` sets
    # flex-shrink: 1, overriding the generic 0), so its floor is its children's
    # min-widths, not their preferred widths.
    assert (
        _cascade_value(rcss, axis_selectors, "flex-shrink") == 1
    ), "the axis no longer shrinks; its minimum is now its preferred size"
    slider = _cascade_value(
        rcss,
        [
            "#selection-block .depth-axis-group .viewport-selection-slider",
            "#selection-block .viewport-selection-slider",
            ".viewport-selection-slider",
        ],
        "min-width",
    )
    number = _cascade_value(
        rcss,
        [
            "#selection-block .viewport-selection-depth-axis > "
            ".number-input.viewport-selection-depth-input",
            ".viewport-selection-depth-axis > "
            ".number-input.viewport-selection-depth-input",
        ],
        "min-width",
    )
    group_gap = _rcss_value(_rcss_rule(rcss, "#selection-block .depth-axis-group"), "gap")
    # The row reserves the buttons' width on the right and mirrors it on the
    # left so the wrapped lines stay concentric with the row. Padding shrinks
    # the content box the lines resolve against, so BOTH sides are charged.
    button_reserve = 2.0 * _rcss_value(
        _rcss_rule(narrow, "#selection-block .viewport-selection-depth-fields"),
        "padding-right",
    )
    panel = _rcss_rule(rcss, "#selection-block .viewport-selection-panel")
    panel_fraction = _rcss_value(panel, "width") / 100.0
    # RCSS defaults to `box-sizing: content-box`, and neither the panel rule nor
    # the base `.viewport-transform-panel` rule overrides it -- so `width: 65%`
    # already IS the content width and the panel's own padding must NOT be
    # subtracted from it a second time. Subtracting it here would overstate
    # the requirement by 2 * 6dp. If the panel ever becomes
    # border-box the derivation changes, so that is asserted rather than assumed.
    for rule in (panel, _rcss_rule(rcss, "\n.viewport-transform-panel")):
        assert "box-sizing" not in rule, (
            "the selection panel is no longer content-box; `width: 65%` is now a "
            "border-box width and its padding must be subtracted again"
        )
    overlay = _rcss_rule(rcss, "\n.viewport-selection-overlay")
    band_inset = _rcss_value(overlay, "left") + _rcss_value(overlay, "right")

    axis = label + axis_gap + slider + axis_gap + number
    # The chip is a flow item at the head of this line, so its width is part
    # of the line rather than of the reserve.
    top_line = chip + group_gap + axis + group_gap + axis

    # L(M) = panel_fraction * (M - band_inset) - button_reserve
    #      = 0.65 * (M - 64) - 72
    # -> the top line (324dp) first fits at M = 673.2dp; the 960dp breakpoint
    #    clears it with room to spare, because the binding line is the other
    #    one (Size/X/Y at 500dp, M = 944.0dp).
    required = (top_line + button_reserve) / panel_fraction + band_inset

    hide = float(
        re.search(
            r"@media \(max-width: ([0-9.]+)dp\)\s*\{\s*"
            r"#selection-block \.viewport-selection-depth-fields\s*\{\s*display:\s*none",
            rcss,
        ).group(1)
    )

    assert hide >= required, (
        f"the depth row shows down to {hide}dp but its top line needs "
        f"{top_line}dp of content width, which only arrives at {required:.1f}dp "
        f"-- between the two the Near/Far row runs under the pinned sync and "
        f"viz buttons"
    )
    # No drift guard here any more. Under the three-state rule the hide
    # breakpoint is NOT set by this line: the two-line tier now runs all the way
    # down to it, so the binding requirement is the 500dp Size/X/Y line
    # (944.0dp), not this 324dp top line (673.2dp). The top line is a FLOOR that
    # the real number must clear, which is what is asserted above; the drift
    # guard against the binding requirement lives in
    # test_depth_row_never_over_constrains_its_widest_line_at_any_width. Keeping
    # a `< 60` guard here would demand a breakpoint that reintroduces the
    # overflowing Size/X/Y band that rule removed.


def _rcss_signed(block: str, prop: str) -> float:
    """Like `_rcss_value`, but accepts a NEGATIVE offset.

    `_rcss_value`'s pattern is unsigned, so it silently refuses `top: -3dp`
    instead of reading it -- and the two split buttons are placed with exactly
    that.
    """
    match = re.search(rf"{prop}\s*:\s*(-?[0-9.]+)\s*dp", block)
    assert match, f"no {prop} declaration in:\n{block}"
    return float(match.group(1))


def test_the_two_pinned_depth_buttons_ride_separate_lines_without_overlapping():
    """One toolbar button per wrapped line, each clear of the other and of the panel.

    The two-line row is 24 + row gap 10 + 24 = 58dp of content and a button is
    30dp, so the two buttons need 60dp of vertical room they do not have: with
    both flush (`top: 0` and `bottom: 0`) they would overlap by 2dp on a shared
    right edge. Each is therefore centred on its OWN line instead, which is
    what makes one button per line fit at all -- and one button per line is
    what halves the row's reserve and hands the two-line band back.

    Neither button is in flow, so this changes no panel height; what it does
    change is that each button now overhangs its line's outer edge by 3dp, and
    that overhang has to land inside the panel rather than on its border.

    Known limits, as everywhere in this file: this reads declared RCSS. It does
    not run RmlUi's layout, so it cannot prove the offsets resolve against the
    box this derivation assumes.
    """
    rcss = _read_src(
        "src", "visualizer", "gui", "rmlui", "resources", "viewport_overlay.rcss"
    )
    wrap_body = dict(_media_blocks(rcss))[_depth_breakpoints(rcss)["wrap"]]

    button_h = _cascade_value(
        rcss,
        [
            "#transform-block .viewport-transform-option,\n"
            "#transform-block .viewport-transform-action,\n"
            "#selection-block .viewport-transform-option,\n"
            "#selection-block .viewport-transform-action,\n"
            "#depth-view-block .viewport-transform-action"
        ],
        "min-height",
    )
    line_h = _cascade_value(
        rcss,
        [
            ".viewport-selection-depth-axis > "
            ".number-input.viewport-selection-depth-input"
        ],
        "min-height",
    )
    row_gap = _rcss_value(
        _rcss_rule(rcss, "#selection-block .viewport-selection-depth-fields"), "gap"
    )
    rows_h = line_h + row_gap + line_h

    top_off = _rcss_signed(
        _rcss_rule(
            wrap_body,
            "#selection-block .viewport-selection-depth-fields > "
            ".viewport-selection-sync-action",
        ),
        "top",
    )
    bottom_off = _rcss_signed(
        _rcss_rule(
            wrap_body,
            "#selection-block .viewport-selection-depth-fields > "
            ".viewport-selection-viz-action",
        ),
        "bottom",
    )

    # y measured down from the row's content-box top.
    sync_top, sync_bottom = top_off, top_off + button_h
    viz_bottom = rows_h - bottom_off
    viz_top = viz_bottom - button_h

    assert sync_bottom <= viz_top, (
        f"the two pinned depth-row buttons overlap by {sync_bottom - viz_top:g}dp "
        f"on the same right edge: the sync toggle spans {sync_top:g}..{sync_bottom:g}dp "
        f"and the viz-mode button {viz_top:g}..{viz_bottom:g}dp of a {rows_h:g}dp row"
    )
    # ...and each sits on the line it belongs to, centred: the top button on
    # the Near/Far line (0..line_h), the bottom one on the Size/X/Y line.
    assert sync_top + button_h / 2.0 == line_h / 2.0, (
        f"the sync toggle's centre is {sync_top + button_h / 2.0:g}dp, not the "
        f"{line_h / 2.0:g}dp centre of the Near/Far line it is pinned to"
    )
    assert viz_top + button_h / 2.0 == rows_h - line_h / 2.0, (
        f"the viz-mode button's centre is {viz_top + button_h / 2.0:g}dp, not the "
        f"{rows_h - line_h / 2.0:g}dp centre of the Size/X/Y line it is pinned to"
    )
    # The overhang at each end must stay inside the panel. Above the row sits
    # the panel's own flex gap; below it, the panel's bottom padding before its
    # border. A button reaching past either would paint on the panel's edge.
    panel_body = _rcss_rule(rcss, "#selection-block .viewport-selection-panel")
    assert -sync_top <= _rcss_value(panel_body, "gap"), (
        f"the sync toggle overhangs the row by {-sync_top:g}dp into a "
        f"{_rcss_value(panel_body, 'gap'):g}dp panel gap -- it would reach the icon row"
    )
    assert -bottom_off <= _rcss_value(panel_body, "padding"), (
        f"the viz-mode button overhangs the row by {-bottom_off:g}dp into a "
        f"{_rcss_value(panel_body, 'padding'):g}dp panel padding -- it would reach the border"
    )


def test_depth_panel_never_covers_the_per_viewport_controls():
    """The selection panel's box must stop above each viewport's own controls.

    Making the depth row wrap makes the PANEL
    taller, and the panel is `z-index: 9` with `pointer-events: auto` while
    every viewport's top-right control group (`.viewport-gizmo-controls`) is
    `z-index: 8`. Wherever the panel's box reaches down into that group, the
    panel paints over it AND swallows its clicks. The row's own side reserve
    protects only the panel's internal pinned buttons; it is no defence here.

    Horizontal separation cannot be relied on: the panel is centred and 65%
    wide inside a band that spans the whole context, and in independent-dual
    split the primary viewport's group rides the divider, so at some divider
    position it is always inside the panel's band. The clearance therefore has
    to be vertical, and this pins it per tier by the same declared arithmetic
    the width test uses.

    Known limits, unchanged from the width test: this reads RCSS text. It does
    not run a live DOM, a media-aware cascade, RmlUi flex layout, z-ordering or
    hit testing. It asserts that the DECLARED geometry leaves clearance; only
    a hands-on check confirms RmlUi agrees.
    """
    rcss = _read_src(
        "src", "visualizer", "gui", "rmlui", "resources", "viewport_overlay.rcss"
    )
    breakpoints = _depth_breakpoints(rcss)
    # (max-width, start, end) for every media block, so a rule can be told
    # apart from one that merely reads alike outside a block.
    media_spans = _media_spans(rcss)

    overlay_top = _rcss_value(
        _rcss_rule(rcss, "\n.viewport-selection-overlay"), "top"
    )
    panel_body = _rcss_rule(rcss, "#selection-block .viewport-selection-panel")
    pad_y = _rcss_value(panel_body, "padding")
    panel_gap = _rcss_value(panel_body, "gap")
    border = _rcss_value(
        _rcss_bodies(rcss, ".viewport-transform-panel")[0][1], "border-width"
    )
    # The icon row is a 30dp button inside a group that adds its 1dp border on
    # both sides -- 32dp, not 30dp.
    button_h = _cascade_value(
        rcss,
        [
            "#transform-block .viewport-transform-option,\n"
            "#transform-block .viewport-transform-action,\n"
            "#selection-block .viewport-transform-option,\n"
            "#selection-block .viewport-transform-action,\n"
            "#depth-view-block .viewport-transform-action"
        ],
        "min-height",
    )
    group_border = _rcss_value(
        _rcss_rule(rcss, ".viewport-selection-tool-group"), "border-width"
    )
    icon_row = button_h + 2 * group_border
    # A depth line is as tall as its tallest child, the value box.
    axis_h = _cascade_value(
        rcss,
        [
            ".viewport-selection-depth-axis > "
            ".number-input.viewport-selection-depth-input"
        ],
        "min-height",
    )
    row_gap = _rcss_value(
        _rcss_rule(rcss, "#selection-block .viewport-selection-depth-fields"), "gap"
    )

    def panel_bottom(fields: float) -> float:
        chrome = overlay_top + 2 * border + 2 * pad_y + icon_row
        if fields == 0.0:
            return chrome
        return chrome + panel_gap + fields

    def controls_top(width: float) -> float:
        """`.viewport-gizmo-controls` top as it resolves AT `width`.

        The blocks are `max-width`, so at a narrow width several of them apply
        at once and the last one in source order wins -- a tier that declares no
        override inherits the tier above it, not the stock value. Modelling that
        is what makes a DELETED override visible as an overlap rather than as a
        missing declaration.
        """
        applicable = [
            (offset, body)
            for offset, body in _rcss_bodies(rcss, ".viewport-gizmo-controls")
            if all(
                width <= media_width
                for media_width, start, end in media_spans
                if start <= offset < end
            )
        ]
        assert applicable, "the per-viewport control group lost its `top` anchor"
        return _rcss_value(applicable[-1][1], "top")

    stock = controls_top(breakpoints["wrap"] + 1.0)
    tiers = [
        # (name, widest width at which the tier is active, fields column height)
        # ABOVE the wrap breakpoint the sync/viz buttons are still in normal
        # flow -- only the wrap block pins them with `position: absolute`
        # (viewport_overlay.rml:781, .rcss:1045). The fields row is a flex row
        # with `align-items: center`, so its height is its TALLEST child: the
        # 30dp button, not the 24dp axis. Supplying axis_h here understated the
        # panel by 6dp and made a real 3dp overlap read as green.
        ("single-line", breakpoints["wrap"] + 1.0, max(axis_h, button_h)),
        ("two-line", breakpoints["wrap"], axis_h + row_gap + axis_h),
        # No stacked tier: hidden is the state below two lines,
        # so there is no third visible panel height to clear.
        ("hidden", breakpoints["hide"], 0.0),
    ]
    for name, width, fields in tiers:
        bottom = panel_bottom(fields)
        top = controls_top(width)
        assert bottom <= top, (
            f"in the {name} tier the selection panel's bottom is {bottom:g}dp but "
            f"the per-viewport control group starts at {top:g}dp -- the panel "
            f"(z-index 9, pointer-events auto) overlaps that group "
            f"(z-index 8) by {bottom - top:g}dp and intercepts its clicks"
        )

    # The narrow tiers are max-width, so every push also applies below it. The
    # hide tier must put the group back where it belongs, or it sits low for a
    # panel that is no longer there.
    assert controls_top(breakpoints["hide"]) == stock, (
        f"the hide tier leaves the per-viewport controls at "
        f"{controls_top(breakpoints['hide']):g}dp instead of restoring "
        f"the stock {stock:g}dp, but the depth row is hidden there"
    )
    # ...and there must be no DANGLING relocation. Three states means exactly
    # two in-media `.viewport-gizmo-controls` overrides: the two-line push and
    # the hide restore. A leftover override for a tier that no longer exists
    # (the deleted 188dp four-line push is the concrete case) would sit between
    # them in source order and silently win below its own breakpoint.
    in_media = [
        (offset, body)
        for offset, body in _rcss_bodies(rcss, ".viewport-gizmo-controls")
        if any(start <= offset < end for _, start, end in media_spans)
    ]
    assert len(in_media) == 2, (
        f"{len(in_media)} width-tier relocations of the per-viewport controls, "
        f"expected exactly 2 (the two-line push and the hide restore). A "
        f"relocation for a tier that no longer exists is a dangling override "
        f"(there is no four-line tier and no 188dp push)"
    )
    assert {_rcss_value(body, "top") for _, body in in_media} == {
        controls_top(breakpoints["wrap"]),
        stock,
    }, "the two surviving relocations are not the two-line push and the restore"
    # ...and no tier may push them further than its own panel needs.
    for name, width, fields in tiers:
        assert controls_top(width) - panel_bottom(fields) < 40.0 or fields == 0.0, (
            f"the {name} tier pushes the per-viewport controls to "
            f"{controls_top(width):g}dp for a panel that ends at "
            f"{panel_bottom(fields):g}dp"
        )

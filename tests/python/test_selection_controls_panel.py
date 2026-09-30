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
        active_view=1,
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
        # native undo entry that restores state.
        undo_effect=None,
        redo_effect=None,
        split_view_mode="none",
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
    lf_stub.ui = SimpleNamespace(
        get_active_tool=lambda: state.active_tool,
        get_active_view_id=lambda: state.active_view,
        get_active_submode=lambda: state.active_submode,
        message_dialog=lambda *_args, **_kwargs: None,
        get_split_view_mode=lambda: state.split_view_mode,
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

    def _get_depth_filter_window():
        if state.depth_read_error:
            raise RuntimeError("depth state unavailable")
        near, far = state.depth_near, state.depth_far
        scale_x, scale_y = state.depth_scale, state.depth_scale_y
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
        enabled, near, far, scale, offset_x, offset_y, scale_y=None
    ):
        if state.window_write_error:
            raise RuntimeError("depth window write rejected")
        state.depth_enabled = bool(enabled)
        state.depth_near = float(near)
        state.depth_far = float(far)
        state.depth_scale = float(scale)
        state.depth_scale_y = float(scale if scale_y is None else scale_y)
        state.depth_offset_x = float(offset_x)
        state.depth_offset_y = float(offset_y)
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


_GT_SIZE_REFERENCES = {"left": (0.60, 0.30), "right": (0.40, 0.20)}


def test_every_depth_toolbar_icon_names_an_asset_that_exists(selection_controls_module):
    """The frames are file paths, and a missing file renders as nothing at all.

    RmlUi does not fail loudly on an `img src` it cannot open, so a typo in
    any of these names is invisible until someone looks at the button. Each is
    resolved the way the RML resolves it -- relative to the RCSS/RML resource
    directory, whose `../icon/` is the asset folder. The icons use the canonical 40px exporter.
    """
    icons, _state = selection_controls_module
    assets = (
        Path(__file__).parent.parent.parent / "src" / "visualizer" / "gui" / "assets"
    )
    names = [
        *icons._VIZ_MODE_ICONS.values(),
    ]
    expected_sizes = {
        "../icon/depth-show.png": (40, 40),
        "../icon/depth-dim.png": (40, 40),
        "../icon/depth-hide.png": (40, 40),
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




# ---------------------------------------------------------------------------
# The native bindings themselves (real lichtfeld module)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("view", [1, 7])
def test_native_draw_commit_store_field_round_trips_its_view(lf, view):
    """Draw commits preserve the owning view through the native store."""
    store = lf.ui.store
    before = store.get("depth_window_draw_commit")
    try:
        store.set("depth_window_draw_commit", {"generation": 41, "view": view, "scale_x": 0.4, "scale_y": 0.6})
        after = store.get("depth_window_draw_commit")

        assert isinstance(after, dict)
        assert after["generation"] == 41
        assert after["view"] == view
    finally:
        store.set("depth_window_draw_commit", before)


def _read_src(*parts: str) -> str:
    return (Path(__file__).parent.parent.parent.joinpath(*parts)).read_text(
        encoding="utf-8"
    )

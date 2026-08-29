# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for viewport selection controls."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
        get_active_submode=lambda: state.active_submode,
        message_dialog=lambda *_args, **_kwargs: None,
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
        return (
            state.depth_enabled,
            state.depth_near,
            state.depth_far,
            state.depth_scale,
            state.depth_offset_x,
            state.depth_offset_y,
        )

    def _set_depth_filter_window(enabled, near, far, scale, offset_x, offset_y):
        if state.window_write_error:
            raise RuntimeError("depth window write rejected")
        state.depth_enabled = bool(enabled)
        state.depth_near = float(near)
        state.depth_far = float(far)
        state.depth_scale = float(scale)
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
    lf_stub.undo = SimpleNamespace(
        can_undo=lambda: state.undo_available,
        can_redo=lambda: state.redo_available,
        undo=lambda: setattr(state, "undo_calls", state.undo_calls + 1) or True,
        redo=lambda: setattr(state, "redo_calls", state.redo_calls + 1) or True,
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
        self._by_id = {
            "selection-block": self.wrap,
            "selection-depth-near": self.near,
            "selection-depth-far": self.far,
            "selection-depth-scale": self.scale,
            "selection-depth-offset-x": self.offset_x,
            "selection-depth-offset-y": self.offset_y,
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
     "50", lambda s: s.depth_scale, 0.5),
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
        state.depth_scale = min(expected + 0.1, 1.0)
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
    ("selection_depth_scale_str", "scale", "35.4", lambda s: s.depth_scale, 0.35, 0.354),
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

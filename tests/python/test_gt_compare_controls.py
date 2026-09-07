# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for requested versus published GT 1:1 state."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


class _HandleStub:
    def __init__(self):
        self.dirty_calls = []

    def dirty(self, name):
        self.dirty_calls.append(name)


class _ModelStub:
    def __init__(self):
        self.bound_funcs = {}
        self.bound_events = {}
        self.bound_binds = {}
        self.handle = _HandleStub()

    def bind_func(self, name, getter):
        self.bound_funcs[name] = getter

    def bind_event(self, name, callback):
        self.bound_events[name] = callback

    def bind(self, name, getter, setter):
        self.bound_binds[name] = (getter, setter)

    def get_handle(self):
        return self.handle


class _ElementStub:
    def __init__(self):
        self.classes = set()

    def set_class(self, name, enabled):
        if enabled:
            self.classes.add(name)
        else:
            self.classes.discard(name)


class _DocumentStub:
    def __init__(self):
        self.element = _ElementStub()

    def get_element_by_id(self, element_id):
        if element_id == "gt-compare-mode-block":
            return self.element
        return None


@pytest.fixture
def controller_environment(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) in sys.path:
        sys.path.remove(str(source_python))
    sys.path.insert(0, str(source_python))

    state = SimpleNamespace(
        split_mode="gt_comparison",
        mode="rgb",
        requested=False,
        available=True,
        active=False,
        requested_writes=[],
        error="",
        retries=0,
    )

    def set_requested(enabled):
        state.requested = bool(enabled)
        state.requested_writes.append(state.requested)
        state.error = ""

    def retry():
        state.retries += 1
        state.error = ""

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        get_split_view_mode=lambda: state.split_mode,
        get_gt_comparison_mode=lambda: state.mode,
        set_gt_comparison_mode=lambda mode: setattr(state, "mode", mode),
        get_gt_comparison_actual_size=lambda: state.requested,
        set_gt_comparison_actual_size=set_requested,
        is_gt_comparison_actual_size_available=lambda: state.available,
        is_gt_comparison_actual_size_active=lambda: state.active,
        get_gt_comparison_actual_size_error=lambda: state.error,
        retry_gt_comparison_actual_size=retry,
        tr=lambda key: key,
    )
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)

    ui_stub = ModuleType("lfs_plugins.ui")
    ui_stub.RuntimeState = SimpleNamespace(
        language_generation=SimpleNamespace(value=0),
    )
    monkeypatch.setitem(sys.modules, "lfs_plugins.ui", ui_stub)

    sys.modules.pop("lfs_plugins", None)
    sys.modules.pop("lfs_plugins.gt_compare_controls", None)
    module = import_module("lfs_plugins.gt_compare_controls")
    return module, state


@pytest.mark.parametrize(
    (
        "requested",
        "available",
        "active",
        "transitioning",
        "tooltip",
    ),
    (
        (
            True,
            True,
            False,
            True,
            "1:1 was requested, but a Fit preview is currently displayed while native-resolution content loads or recovers.",
        ),
        (
            True,
            True,
            True,
            False,
            "1:1 is active: one image pixel maps to one physical display pixel.",
        ),
        (
            True,
            False,
            False,
            False,
            "1:1 is unavailable for this camera or comparison mode. Distorted images require usable saved undistortion calibration.",
        ),
        (
            False,
            True,
            True,
            True,
            "1:1 is active: one image pixel maps to one physical display pixel.",
        ),
    ),
)
def test_controller_exposes_requested_available_active_and_transitioning(
    controller_environment,
    requested,
    available,
    active,
    transitioning,
    tooltip,
):
    module, state = controller_environment
    state.requested = requested
    state.available = available
    state.active = active

    controller = module.GTCompareControlsController()
    model = _ModelStub()
    document = _DocumentStub()
    controller.bind_model(model)
    controller.mount(document)
    assert controller.update(document) is not None

    assert model.bound_funcs["gt_compare_actual_size"]() is requested
    assert model.bound_funcs["gt_compare_actual_size_available"]() is available
    assert model.bound_funcs["gt_compare_actual_size_active"]() is active
    assert model.bound_funcs["gt_compare_actual_size_transitioning"]() is transitioning
    assert model.bound_funcs["gt_compare_actual_size_tooltip"]() == tooltip


def test_toggle_changes_requested_state_not_published_state(controller_environment):
    module, state = controller_environment
    controller = module.GTCompareControlsController()
    model = _ModelStub()
    document = _DocumentStub()
    controller.bind_model(model)
    controller.mount(document)
    controller.update(document)

    model.bound_events["gt_compare_toggle_actual_size"]()

    assert state.requested_writes == [True]
    assert state.active is False
    assert model.bound_funcs["gt_compare_actual_size"]() is True
    assert model.bound_funcs["gt_compare_actual_size_active"]() is False
    assert model.bound_funcs["gt_compare_actual_size_transitioning"]() is True
    assert model.bound_funcs["gt_compare_actual_size_loading"]() is True
    assert model.bound_funcs["gt_compare_actual_size_status"]() == "Preparing 1:1…"

    model.bound_events["gt_compare_toggle_actual_size"]()
    assert state.requested_writes == [True, False]
    assert not model.bound_funcs["gt_compare_actual_size_loading"]()
    assert model.bound_funcs["gt_compare_actual_size_status"]() == ""


@pytest.mark.parametrize("recovery", ("automatic", "retry", "cancel", "unavailable", "camera_change"))
def test_error_feedback_and_recovery(controller_environment, recovery):
    module, state = controller_environment
    state.requested = True
    state.error = "native decode failed"
    controller = module.GTCompareControlsController()
    model = _ModelStub()
    doc = _DocumentStub()
    controller.bind_model(model)
    controller.update(doc)
    assert model.bound_funcs["gt_compare_actual_size_failed"]()
    assert not model.bound_funcs["gt_compare_actual_size_loading"]()
    assert "Retrying" in model.bound_funcs["gt_compare_actual_size_status"]()
    assert "native decode failed" in model.bound_funcs["gt_compare_actual_size_tooltip"]()
    assert controller.update(doc) is None
    assert model.bound_funcs["gt_compare_actual_size_failed"]()

    if recovery == "retry":
        model.bound_events["gt_compare_retry_actual_size"]()
        assert state.retries == 1
        assert state.requested_writes == []
    elif recovery == "cancel":
        model.bound_events["gt_compare_toggle_actual_size"]()
    elif recovery == "unavailable":
        state.available = False
    else:
        state.error = ""
        state.active = recovery == "automatic"
    controller.update(doc)
    assert not model.bound_funcs["gt_compare_actual_size_failed"]()
    assert model.bound_funcs["gt_compare_actual_size_loading"]() is (recovery in {"retry", "camera_change"})
    assert model.bound_funcs["gt_compare_actual_size_active"]() is (recovery == "automatic")


@pytest.mark.parametrize("requested,available", ((False, True), (True, False)))
def test_retry_ignores_disabled_or_unavailable_request(controller_environment, requested, available):
    module, state = controller_environment
    state.requested, state.available = requested, available
    controller = module.GTCompareControlsController()
    controller._retry_actual_size()
    assert state.retries == 0


def test_cancel_hides_loading_before_fit_frame_publishes(controller_environment):
    module, state = controller_environment
    state.requested = state.active = True
    controller = module.GTCompareControlsController()
    model = _ModelStub()
    controller.bind_model(model)
    controller._toggle_actual_size()
    assert model.bound_funcs["gt_compare_actual_size_active"]()
    assert not model.bound_funcs["gt_compare_actual_size_loading"]()
    assert model.bound_funcs["gt_compare_actual_size_status"]() == ""


def test_rml_selection_uses_published_state_and_exposes_transition():
    project_root = Path(__file__).parent.parent.parent
    source = (
        project_root
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "viewport_overlay.rml"
    ).read_text(encoding="utf-8")

    assert 'data-class-selected="gt_compare_actual_size_active"' in source
    assert 'data-class-transitioning="gt_compare_actual_size_transitioning"' in source
    assert 'data-attr-title="gt_compare_actual_size_tooltip"' in source
    assert 'data-class-selected="gt_compare_actual_size"' not in source

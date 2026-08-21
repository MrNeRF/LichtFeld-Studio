# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for retained training panel status bindings."""

from importlib import import_module
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


def _install_lf_stub(monkeypatch):
    panel_space = SimpleNamespace(
        SIDE_PANEL="SIDE_PANEL",
        FLOATING="FLOATING",
        VIEWPORT_OVERLAY="VIEWPORT_OVERLAY",
        MAIN_PANEL_TAB="MAIN_PANEL_TAB",
        SCENE_HEADER="SCENE_HEADER",
        STATUS_BAR="STATUS_BAR",
    )
    panel_height_mode = SimpleNamespace(FILL="fill", CONTENT="content")
    panel_option = SimpleNamespace(DEFAULT_CLOSED="DEFAULT_CLOSED", HIDE_HEADER="HIDE_HEADER")
    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=panel_space,
        PanelHeightMode=panel_height_mode,
        PanelOption=panel_option,
        tr=lambda key: key,
    )
    lf_stub.optimization_params = lambda: None
    lf_stub.dataset_params = lambda: None
    lf_stub.get_scene = lambda: None
    lf_stub.start_training = lambda: None
    lf_stub.training_start_overwrite_conflict = lambda: None
    lf_stub.loss_buffer = lambda: []
    lf_stub.push_loss_to_element = lambda _element, _data: (0.0, 0.0)
    lf_stub.get_render_settings = lambda: None
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return lf_stub


def test_bundled_locales_define_training_panel_strategy_and_color_keys():
    project_root = Path(__file__).parent.parent.parent
    locale_dir = project_root / "src" / "visualizer" / "gui" / "resources" / "locales"

    for locale_path in locale_dir.glob("*.json"):
        data = json.loads(locale_path.read_text())
        assert data["training"]["options.strategy.igs_plus"] == "IGS+"
        assert "refinement.grow_until_iter" in data["training"]
        assert "tooltip.grow_until_iter" in data["training"]
        assert data["training"]["overwrite.btn_save_as_start"]
        assert data["training_panel"]["color_red_prefix"] == "R:"
        assert data["training_panel"]["color_green_prefix"] == "G:"
        assert data["training_panel"]["color_blue_prefix"] == "B:"


@pytest.fixture
def training_panel_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))
    sys.modules.pop("lfs_plugins.training_panel", None)
    sys.modules.pop("lfs_plugins.training_confirm", None)
    sys.modules.pop("lfs_plugins", None)
    _install_lf_stub(monkeypatch)
    return import_module("lfs_plugins.training_panel")


class _HandleStub:
    def __init__(self):
        self.dirty_fields = []
        self.dirty_all_count = 0
        self.request_update_count = 0

    def dirty(self, name):
        self.dirty_fields.append(name)

    def dirty_all(self):
        self.dirty_all_count += 1

    def request_update(self):
        self.request_update_count += 1


def _make_signal(value):
    return SimpleNamespace(value=value)


class _ParamsStub:
    def __init__(self):
        self.iterations = 1234
        self.means_lr = 0.25
        self.steps_scaler = 1.0
        self.start_refine = 500
        self.stop_refine = 15000
        self.grow_until_iter = 15000
        self.refine_every = 100
        self.reset_every = 3000
        self.sh_degree_interval = 1000
        self.ppisp_controller_activation_step = 5678
        self.enable_eval = False
        self.save_steps = [7000]
        self.eval_steps = []
        self.bg_color = (0.0, 0.0, 0.0)
        self.bg_image_path = ""
        self.auto_scaled_for = None

    def has_params(self):
        return True

    def apply_step_scaling(self, value):
        scale = value / self.steps_scaler if self.steps_scaler else value
        self.steps_scaler = value
        self.iterations = int(self.iterations * scale)
        self.start_refine = int(self.start_refine * scale)
        self.stop_refine = int(self.stop_refine * scale)
        self.grow_until_iter = int(self.grow_until_iter * scale)

    def auto_scale_steps(self, camera_count):
        self.auto_scaled_for = camera_count

    def set(self, prop, value):
        setattr(self, prop, value)

    def clear_eval_steps(self):
        self.eval_steps.clear()

    def add_eval_step(self, step):
        self.eval_steps.append(step)


class _StrategyParamsStub:
    def __init__(self):
        self._strategy = "mrnf"
        self._slots = {
                "mrnf": {
                "max_cap": 5_000_000,
                "means_lr": 2e-5,
                "scaling_lr": 0.007,
                "lambda_dssim": 0.2,
                "init_opacity": 0.5,
                "prune_ratio": 0.6,
                "sh_degree": 3,
                "depth_loss_mode": "ssi",
                "ppisp_controller_activation_step": 25_000,
                "bg_color": (0.0, 0.0, 0.0),
                "save_steps": [7_000, 30_000],
                "use_edge_map": True,
            },
            "igs+": {
                "max_cap": 4_000_000,
                "means_lr": 1.6e-5,
                "scaling_lr": 0.02,
                "lambda_dssim": 0.35,
                "init_opacity": 0.1,
                "prune_ratio": 0.4,
                "sh_degree": 2,
                "depth_loss_mode": "ssi-depth",
                "ppisp_controller_activation_step": 12_000,
                "bg_color": (0.1, 0.2, 0.3),
                "save_steps": [5_000],
                "use_edge_map": False,
            },
        }
        self.gut = False

    def has_params(self):
        return True

    @property
    def strategy(self):
        return self._strategy

    def set_strategy(self, strategy):
        self._strategy = strategy

    def get(self, prop):
        return self._slots[self._strategy][prop]

    def __getattr__(self, prop):
        try:
            return self._slots[self._strategy][prop]
        except KeyError as exc:
            raise AttributeError(prop) from exc


class _DatasetStub:
    def __init__(self):
        self.data_path = "/data/scene_a"
        self.max_width = 2048
        self.test_every = 8

    def has_params(self):
        return True


class _ModelStub:
    def __init__(self):
        self.bindings = {}

    def bind(self, name, getter, setter):
        self.bindings[name] = (getter, setter)

    def bind_func(self, name, getter):
        self.bindings[name] = (getter, None)

    def bind_string_list(self, name):
        self.bindings[name] = (None, None)


def test_strategy_switch_resyncs_generated_rows_and_requests_panel_update(
    training_panel_module, monkeypatch
):
    params = _StrategyParamsStub()
    dataset = _DatasetStub()
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()

    monkeypatch.setattr(
        training_panel_module.lf,
        "optimization_params",
        lambda: params,
    )
    monkeypatch.setattr(
        training_panel_module.lf,
        "dataset_params",
        lambda: dataset,
    )
    monkeypatch.setattr(
        training_panel_module.lf.ui,
        "schedule_on_ui_thread",
        lambda _callback: None,
        raising=False,
    )

    def row(prop_id, *, is_int=False, precision=6, strategies=()):
        return {
            "id": prop_id,
            "kind": "number",
            "label_key": "",
            "tooltip_key": "",
            "precision": precision,
            "step": 1,
            "min": 0,
            "max": 10_000_000,
            "is_int": is_int,
            "name": prop_id,
            "items": [],
            "strategies": strategies,
        }

    binding = training_panel_module.property_view.SectionBinding(
        "strategy_values",
        [
            row("max_cap", is_int=True, precision=0),
            row("means_lr"),
            row("scaling_lr"),
            {
                **row("use_edge_map"),
                "kind": "checkbox",
                "strategies": ("mrnf",),
            },
        ],
        lambda: params,
        panel._text_bufs,
        panel._queue_pv_publish,
    )
    panel._pv_bindings = (binding,)
    model = _ModelStub()
    panel._bind_select_props(model, lambda: params, lambda: dataset)

    assert "use_edge_map" in {record["id"] for record in binding._records()}
    panel._set_strategy("igs+")

    assert params.strategy == "igs+"
    assert "use_edge_map" not in {
        record["id"] for record in binding._records()
    }
    assert panel._text_bufs[binding.input_key("max_cap")] == "4,000,000"
    assert panel._text_bufs[binding.input_key("means_lr")] == "0.000016"
    assert panel._text_bufs[binding.input_key("scaling_lr")] == "0.020000"
    assert panel._get_scrub_value("lambda_dssim") == pytest.approx(0.35)
    assert panel._get_scrub_value("init_opacity") == pytest.approx(0.1)
    assert panel._get_scrub_value("prune_ratio") == pytest.approx(0.4)
    assert model.bindings["sh_degree_str"][0]() == "2"
    assert model.bindings["depth_loss_mode_str"][0]() == "ssi-depth"
    assert panel._text_bufs["ppisp_activation_step_str"] == "12,000"
    assert panel._text_bufs[training_panel_module.BG_COLOR_HEX_KEY] == (
        training_panel_module.w.color_to_hex(params.bg_color)
    )
    assert params.save_steps == [5_000]
    assert panel._handle.dirty_all_count == 1
    assert panel._handle.request_update_count == 1


def test_auto_scale_marker_survives_reset_but_not_dataset_change(
    training_panel_module, monkeypatch
):
    dataset = _DatasetStub()
    scene = SimpleNamespace(active_camera_count=600)
    monkeypatch.setattr(training_panel_module.lf, "dataset_params", lambda: dataset)
    monkeypatch.setattr(training_panel_module.lf, "get_scene", lambda: scene)
    panel = training_panel_module.TrainingPanel()
    params = _ParamsStub()

    assert panel._try_auto_scale_steps(params) is True
    assert params.auto_scaled_for == 600

    # Reset re-enters with the same dataset path.
    params.auto_scaled_for = None
    assert panel._try_auto_scale_steps(params) is False
    assert params.auto_scaled_for is None

    dataset.data_path = "/data/scene_b"
    assert panel._try_auto_scale_steps(params) is True
    assert params.auto_scaled_for == 600


def test_auto_scale_user_override_survives_camera_change_until_relocked(
    training_panel_module, monkeypatch
):
    dataset = _DatasetStub()
    scene = SimpleNamespace(active_camera_count=900)
    monkeypatch.setattr(training_panel_module.lf, "dataset_params", lambda: dataset)
    monkeypatch.setattr(training_panel_module.lf, "get_scene", lambda: scene)
    params = _ParamsStub()
    monkeypatch.setattr(training_panel_module.lf, "optimization_params", lambda: params)
    panel = training_panel_module.TrainingPanel()

    assert panel._try_auto_scale_steps(params) is True
    assert params.auto_scaled_for == 900

    assert panel._set_iterations(params, 50000) is True
    assert params.iterations == 50000

    scene.active_camera_count = 800
    assert panel._try_auto_scale_steps(params) is False
    assert params.auto_scaled_for == 900
    assert params.iterations == 50000

    panel._set_auto_scale_steps_locked(False)
    panel._set_auto_scale_steps_locked(True)
    assert params.auto_scaled_for == 800


def test_training_panel_progress_updates_bound_value(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()

    monkeypatch.setattr(
        training_panel_module,
        "RuntimeState",
        SimpleNamespace(
            iteration=_make_signal(25),
            max_iterations=_make_signal(100),
        ),
    )

    assert panel._update_progress() is True
    assert panel._progress_value == "0.25"
    assert panel._handle.dirty_fields == ["progress_value"]


def test_training_panel_uses_dirty_update_policy(training_panel_module):
    assert training_panel_module.TrainingPanel.update_policy == "dirty"
    assert "update_interval_ms" not in training_panel_module.TrainingPanel.__dict__


def test_training_panel_store_update_requests_panel_update(training_panel_module):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()

    panel._subscribe_reactive_state()
    try:
        training_panel_module.RuntimeState.iteration.value += 1

        assert panel._handle.request_update_count == 1
        assert panel._handle.dirty_all_count == 0
    finally:
        panel._unsubscribe_reactive_state()
        training_panel_module.RuntimeState.iteration._fallback = 0


def test_training_panel_language_update_requests_panel_update(training_panel_module):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()

    panel._subscribe_reactive_state()
    try:
        training_panel_module.RuntimeState.language_generation.value += 1

        assert panel._handle.request_update_count == 1
        assert panel._handle.dirty_all_count == 0
    finally:
        panel._unsubscribe_reactive_state()
        training_panel_module.RuntimeState.language_generation._fallback = 0


def test_training_panel_project_saved_dirties_field(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    scheduled = []
    monkeypatch.setattr(panel, "_schedule_deferred_update", lambda delay: scheduled.append(delay))

    panel._mark_project_saved()

    assert panel._last_project_saved_visible is True
    assert panel._handle.dirty_fields == ["show_project_saved"]
    assert scheduled == [2.05]


def test_training_panel_deferred_update_keeps_earliest_timer(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    timers = []
    scheduled_callbacks = []

    class _TimerStub:
        def __init__(self, delay, callback):
            self.delay = delay
            self.callback = callback
            timers.append(self)

        def start(self):
            pass

    monkeypatch.setattr(training_panel_module.threading, "Timer", _TimerStub)
    monkeypatch.setattr(
        training_panel_module.lf.ui,
        "schedule_on_ui_thread",
        scheduled_callbacks.append,
        raising=False,
    )

    panel._schedule_deferred_update(1.0)
    panel._schedule_deferred_update(2.0)
    panel._schedule_deferred_update(0.5)

    assert [timer.delay for timer in timers] == [1.0, 0.5]

    timers[0].callback()
    scheduled_callbacks.pop(0)()
    assert panel._handle.request_update_count == 0

    timers[1].callback()
    scheduled_callbacks.pop(0)()
    assert panel._handle.request_update_count == 1


def test_training_panel_loss_graph_updates_bound_labels(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    panel._loss_graph_el = object()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            loss_buffer=lambda: [1.0, 0.5, 0.25],
            push_loss_to_element=lambda _element, _data: (0.25, 1.0),
            ui=SimpleNamespace(tr=lambda key: key),
        ),
    )

    assert panel._update_loss_graph() is True
    assert panel._loss_label == "status.loss: 0.2500"
    assert panel._loss_tick_max == "1.00"
    assert panel._loss_tick_mid == "0.62"
    assert panel._loss_tick_min == "0.25"
    assert panel._handle.dirty_fields == [
        "loss_label",
        "loss_tick_max",
        "loss_tick_mid",
        "loss_tick_min",
    ]


def test_training_panel_loss_graph_clears_bound_labels(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    panel._loss_graph_el = object()
    panel._last_loss_signature = (3, 0.25)
    panel._loss_label = "status.loss: 0.2500"
    panel._loss_tick_max = "1.00"
    panel._loss_tick_mid = "0.62"
    panel._loss_tick_min = "0.25"

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            loss_buffer=lambda: [],
            push_loss_to_element=lambda _element, _data: (0.0, 0.0),
            ui=SimpleNamespace(tr=lambda key: key),
        ),
    )

    assert panel._update_loss_graph() is True
    assert panel._loss_label == ""
    assert panel._loss_tick_max == ""
    assert panel._loss_tick_mid == ""
    assert panel._loss_tick_min == ""
    assert panel._handle.dirty_fields == [
        "loss_label",
        "loss_tick_max",
        "loss_tick_mid",
        "loss_tick_min",
    ]


def test_numeric_parser_normalizes_integer_commas_and_keeps_float_validation(training_panel_module):
    assert training_panel_module._parse_num("1,234", int) == "1234"
    assert training_panel_module._parse_num("1,5", int) == "15"
    assert training_panel_module._parse_num("1,234.5", float) == "1234.5"

    with pytest.raises(ValueError):
        training_panel_module._parse_num("0,0001", float)

    with pytest.raises(ValueError):
        training_panel_module._parse_num("1,5", float)


def test_max_width_zero_disables_cap(training_panel_module, monkeypatch):
    dataset = _DatasetStub()
    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(dataset_params=lambda: dataset),
    )

    panel = training_panel_module.TrainingPanel()

    assert panel._set_max_width("0") is True
    assert dataset.max_width == 0


def test_max_width_step_clamps_at_zero(training_panel_module, monkeypatch):
    dataset = _DatasetStub()
    dataset.max_width = 8
    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(dataset_params=lambda: dataset),
    )

    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()

    panel._apply_num_step("max_width", -1)

    assert dataset.max_width == 0
    assert panel._text_bufs["max_width_str"] == "0"
    assert panel._handle.dirty_fields == ["max_width_str"]


@pytest.mark.parametrize(
    ("binding_name", "expected_text"),
    [
        ("ppisp_activation_step_str", "5,678"),
        ("max_width_str", "2,048"),
        ("new_step_str", "7,000"),
    ],
)
def test_cleared_numeric_fields_restore_model_value(training_panel_module, monkeypatch, binding_name, expected_text):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    model = _ModelStub()
    params = _ParamsStub()
    dataset = _DatasetStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
        ),
    )

    panel._bind_bespoke_num_props(model, lambda: params, lambda: dataset)

    getter, setter = model.bindings[binding_name]
    setter("")
    assert getter() == ""

    panel._commit_number_input_key(binding_name)
    assert getter() == expected_text


@pytest.mark.parametrize(
    ("binding_name", "input_text", "expected_text"),
    [
        ("ppisp_activation_step_str", "300000", "300,000"),
        ("max_width_str", "3000", "3,000"),
        ("new_step_str", "300000", "300,000"),
    ],
)
def test_committed_numeric_fields_reformat_and_dirty(
    training_panel_module, monkeypatch, binding_name, input_text, expected_text
):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    model = _ModelStub()
    params = _ParamsStub()
    dataset = _DatasetStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
        ),
    )

    panel._bind_bespoke_num_props(model, lambda: params, lambda: dataset)

    getter, setter = model.bindings[binding_name]
    setter(input_text)
    assert getter() == input_text
    assert panel._handle.dirty_fields == []

    panel._commit_number_input_key(binding_name)
    assert getter() == expected_text
    assert panel._handle.dirty_fields == [binding_name]


@pytest.mark.parametrize(
    ("binding_name", "input_text", "expected_text"),
    [
        ("ppisp_activation_step_str", "56,7800", "567,800"),
        ("max_width_str", "2,0,4,8", "2,048"),
        ("new_step_str", "70,0000", "700,000"),
    ],
)
def test_integer_fields_strip_arbitrary_commas_and_reformat(
    training_panel_module, monkeypatch, binding_name, input_text, expected_text
):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    model = _ModelStub()
    params = _ParamsStub()
    dataset = _DatasetStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
        ),
    )

    panel._bind_bespoke_num_props(model, lambda: params, lambda: dataset)

    getter, setter = model.bindings[binding_name]
    setter(input_text)

    panel._commit_number_input_key(binding_name)
    assert getter() == expected_text
    assert panel._handle.dirty_fields == [binding_name]


@pytest.mark.parametrize(
    ("binding_name", "buffer_text", "expected_text"),
    [
        ("ppisp_activation_step_str", "56x7800", "5,678"),
        ("max_width_str", "20x48", "2,048"),
        ("new_step_str", "70x000", "7,000"),
    ],
)
def test_invalid_numeric_commit_restores_canonical_value(
    training_panel_module, monkeypatch, binding_name, buffer_text, expected_text
):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    params = _ParamsStub()
    dataset = _DatasetStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
        ),
    )

    panel._text_bufs[binding_name] = buffer_text
    panel._commit_number_input_key(binding_name)

    assert panel._text_bufs[binding_name] == expected_text
    assert panel._handle.dirty_fields == [binding_name]


def test_locked_iterations_rescale_dependent_property_view_buffers(
    training_panel_module, monkeypatch
):
    """Issue #970: an iterations edit refreshes every auto-scaled row buffer."""
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    params = _ParamsStub()
    params.iterations = 30000
    params.start_refine = 500
    params.stop_refine = 15000
    params.grow_until_iter = 15000
    params.steps_scaler = 1.0
    dataset = _DatasetStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
        ),
    )

    def row(prop_id):
        return {
            "id": prop_id,
            "kind": "number",
            "label_key": "",
            "tooltip_key": "",
            "precision": 0,
            "step": 100,
            "min": 0,
            "max": 100000,
            "is_int": True,
            "name": prop_id,
            "items": [],
        }

    queued = []
    binding = training_panel_module.property_view.SectionBinding(
        "dependent_steps",
        [row("iterations"), row("grow_until_iter")],
        lambda: params,
        panel._text_bufs,
        queued.append,
    )
    panel._pv_bindings = (binding,)
    panel._pv_binding_by_prop = {
        row_meta["id"]: binding for row_meta in binding.rows
    }

    assert panel._set_iterations(params, 60000) is True
    assert params.steps_scaler == 2.0
    assert params.iterations == 60000
    assert params.grow_until_iter == 30000
    assert panel._text_bufs[binding.input_key("iterations")] == "60,000"
    assert panel._text_bufs[binding.input_key("grow_until_iter")] == "30,000"
    assert queued == [binding]
    assert panel._handle.dirty_all_count >= 1


def test_legacy_negative_ppisp_activation_step_displays_resolved_value(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    model = _ModelStub()
    params = _ParamsStub()
    params.iterations = 60000
    params.steps_scaler = 2.0
    params.ppisp_controller_activation_step = -1
    dataset = _DatasetStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
        ),
    )

    panel._bind_bespoke_num_props(model, lambda: params, lambda: dataset)

    getter, _setter = model.bindings["ppisp_activation_step_str"]
    assert getter() == "50,000"


def test_eval_test_every_one_clamps_to_preserve_training_split(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    params = _ParamsStub()
    params.enable_eval = True
    dataset = _DatasetStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
            get_scene=lambda: SimpleNamespace(active_camera_count=5),
        ),
    )

    assert panel._set_test_every("1") is True
    assert dataset.test_every == 2


def test_eval_test_every_stepper_keeps_lower_bound_at_two(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    params = _ParamsStub()
    params.enable_eval = True
    dataset = _DatasetStub()
    dataset.test_every = 2

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
            get_scene=lambda: SimpleNamespace(active_camera_count=5),
        ),
    )

    panel._apply_num_step("test_every", -1)

    assert dataset.test_every == 2
    assert panel._text_bufs["test_every_str"] == "2"
    assert panel._handle.dirty_fields == ["test_every_str"]


def test_enabling_eval_clamps_existing_bad_test_every(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    params = _ParamsStub()
    dataset = _DatasetStub()
    dataset.test_every = 1

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
            get_render_settings=lambda: None,
            get_scene=lambda: SimpleNamespace(active_camera_count=5),
        ),
    )

    panel._set_bool_prop("enable_eval", True)

    assert params.enable_eval is True
    assert dataset.test_every == 2
    assert panel._text_bufs["test_every_str"] == "2"
    assert params.eval_steps == params.save_steps


def test_enabling_eval_rejects_single_camera_split(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    params = _ParamsStub()
    dataset = _DatasetStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
            get_render_settings=lambda: None,
            get_scene=lambda: SimpleNamespace(active_camera_count=1),
        ),
    )

    panel._set_bool_prop("enable_eval", True)

    assert params.enable_eval is False


def test_training_rml_no_longer_includes_ppisp_auto_toggle():
    project_root = Path(__file__).parent.parent.parent
    training_rml = project_root / "src" / "visualizer" / "gui" / "rmlui" / "resources" / "training.rml"
    content = training_rml.read_text()

    assert "ppisp_auto_step" not in content
    assert "label_ppisp_auto" not in content


def test_training_rml_exposes_mrnf_grow_until_iter():
    project_root = Path(__file__).parent.parent.parent
    training_rml = project_root / "src" / "visualizer" / "gui" / "rmlui" / "resources" / "training.rml"
    content = training_rml.read_text()

    assert 'data-for="row : pv_refinement_grow_rows" data-if="dep_mrnf"' in content
    assert 'data-value="row.text"' in content
    assert 'data-event-mousedown="pv_step(row.id, -1)"' in content


def test_set_bool_prop_hasattr_guard(training_panel_module, monkeypatch):
    """Issue #972: _set_bool_prop must not crash on missing attributes."""
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    params = _ParamsStub()

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            get_render_settings=lambda: None,
        ),
    )

    panel._set_bool_prop("nonexistent_property", True)
    assert not hasattr(params, "nonexistent_property")


def test_browse_background_image_uses_current_image_dialog(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    params = _ParamsStub()
    dataset = _DatasetStub()
    selected_path = "/tmp/background.png"
    calls = []

    def open_image_dialog(start_dir):
        calls.append(start_dir)
        return selected_path

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: dataset,
            ui=SimpleNamespace(open_image_dialog=open_image_dialog),
        ),
    )

    panel._on_action(None, None, ["browse_bg"])

    assert calls == [""]
    assert params.bg_image_path == selected_path
    assert panel._handle.dirty_all_count == 1


def test_training_panel_no_longer_uses_removed_image_dialog_alias():
    project_root = Path(__file__).parent.parent.parent
    training_panel = project_root / "src" / "python" / "lfs_plugins" / "training_panel.py"

    assert "open_image_file_dialog" not in training_panel.read_text()


def test_save_steps_editable_in_active_trainer_states(training_panel_module):
    """Issue #1648: save steps stay editable after checkpoint resume and while running."""
    panel = training_panel_module.TrainingPanel()
    model = _ModelStub()
    params = _ParamsStub()
    dataset = _DatasetStub()
    runtime = training_panel_module.RuntimeState

    panel._bind_visibility(model, lambda: params, lambda: dataset)
    save_edit_mode = model.bindings["save_edit_mode"][0]
    save_readonly_mode = model.bindings["save_readonly_mode"][0]

    try:
        runtime.trainer_state.value = "ready"
        runtime.iteration.value = 0
        assert save_edit_mode() is True
        assert save_readonly_mode() is False

        runtime.iteration.value = 15000
        assert save_edit_mode() is True

        runtime.trainer_state.value = "paused"
        assert save_edit_mode() is True

        runtime.trainer_state.value = "running"
        assert save_edit_mode() is True

        runtime.trainer_state.value = "finished"
        assert save_edit_mode() is False
        assert save_readonly_mode() is True
    finally:
        runtime.iteration._fallback = 0
        runtime.training_state._fallback = "idle"


OVERWRITE_BTN = "training.overwrite.btn_overwrite_start"
SAVE_AS_BTN = "training.overwrite.btn_save_as_start"
CANCEL_BTN = "training.conflict.btn_cancel"


def _overwrite_dialog_harness(training_panel_module, monkeypatch):
    dialogs = []
    starts = []
    save_as_calls = []
    scheduled = []
    state = SimpleNamespace(has_path=False, save_as_result=True)

    def confirm_dialog(title, message, buttons, callback=None):
        dialogs.append((title, message, list(buttons), callback))

    def project_save_as(path="", wait=False):
        save_as_calls.append((path, wait))
        return state.save_as_result

    monkeypatch.setattr(
        training_panel_module.lf.ui, "confirm_dialog", confirm_dialog, raising=False
    )
    monkeypatch.setattr(
        training_panel_module.lf.ui,
        "schedule_on_ui_thread",
        scheduled.append,
        raising=False,
    )
    monkeypatch.setattr(
        training_panel_module.lf, "project_save_as", project_save_as, raising=False
    )
    monkeypatch.setattr(
        training_panel_module.lf,
        "project_has_path",
        lambda: state.has_path,
        raising=False,
    )
    monkeypatch.setattr(
        training_panel_module.lf, "start_training", lambda: starts.append(True)
    )
    monkeypatch.setattr(training_panel_module.lf, "optimization_params", lambda: None)
    monkeypatch.setattr(training_panel_module.lf, "get_scene", lambda: None)

    panel = training_panel_module.TrainingPanel()
    return panel, dialogs, starts, save_as_calls, scheduled, state


@pytest.mark.parametrize("conflict", [7000, -1])
def test_overwrite_dialog_offers_save_as_between_overwrite_and_cancel(
    training_panel_module, monkeypatch, conflict
):
    panel, dialogs, _starts, _save_as_calls, _scheduled, _state = _overwrite_dialog_harness(
        training_panel_module, monkeypatch
    )

    panel._show_overwrite_dialog(conflict)

    assert len(dialogs) == 1
    title, _message, buttons, _callback = dialogs[0]
    assert buttons == [OVERWRITE_BTN, SAVE_AS_BTN, CANCEL_BTN]
    if conflict >= 0:
        assert title == "training.overwrite.title"
    else:
        assert title == "training.overwrite.existing_title"


def test_overwrite_save_as_routes_through_project_save_as_and_starts_after_bind(
    training_panel_module, monkeypatch
):
    panel, dialogs, starts, save_as_calls, scheduled, state = _overwrite_dialog_harness(
        training_panel_module, monkeypatch
    )
    panel._show_overwrite_dialog(12)
    _title, _message, _buttons, callback = dialogs[0]

    state.save_as_result = True
    state.has_path = True
    callback(SAVE_AS_BTN)

    assert save_as_calls == [("", True)]
    assert starts == [True]
    assert scheduled == []


def test_overwrite_save_as_waits_for_fire_and_forget_save_to_bind(
    training_panel_module, monkeypatch
):
    panel, dialogs, starts, save_as_calls, scheduled, state = _overwrite_dialog_harness(
        training_panel_module, monkeypatch
    )
    panel._show_overwrite_dialog(-1)
    _title, _message, _buttons, callback = dialogs[0]

    state.save_as_result = True
    state.has_path = False
    callback(SAVE_AS_BTN)

    assert save_as_calls == [("", True)]
    assert starts == []
    assert len(scheduled) == 1

    state.has_path = True
    scheduled[0]()
    assert starts == [True]


def test_overwrite_save_as_native_dialog_cancel_starts_nothing(
    training_panel_module, monkeypatch
):
    panel, dialogs, starts, save_as_calls, scheduled, state = _overwrite_dialog_harness(
        training_panel_module, monkeypatch
    )
    panel._show_overwrite_dialog(3)
    _title, _message, _buttons, callback = dialogs[0]

    state.save_as_result = False
    state.has_path = False
    callback(SAVE_AS_BTN)

    assert save_as_calls == [("", True)]
    assert starts == []
    assert scheduled == []


def test_overwrite_save_as_native_cancel_on_titled_project_starts_nothing(
    training_panel_module, monkeypatch
):
    panel, dialogs, starts, save_as_calls, scheduled, state = _overwrite_dialog_harness(
        training_panel_module, monkeypatch
    )
    panel._show_overwrite_dialog(4000)
    _title, _message, _buttons, callback = dialogs[0]

    state.save_as_result = False
    state.has_path = True
    callback(SAVE_AS_BTN)

    assert save_as_calls == [("", True)]
    assert starts == []
    assert scheduled == []


def test_overwrite_save_as_accepts_path_only_save_as_stub(
    training_panel_module, monkeypatch
):
    panel, dialogs, starts, save_as_calls, _scheduled, state = _overwrite_dialog_harness(
        training_panel_module, monkeypatch
    )
    path_only_calls = []

    def project_save_as(path=""):
        path_only_calls.append(path)
        return True

    monkeypatch.setattr(
        training_panel_module.lf, "project_save_as", project_save_as, raising=False
    )
    panel._show_overwrite_dialog(-1)
    _title, _message, _buttons, callback = dialogs[0]

    state.has_path = True
    callback(SAVE_AS_BTN)

    assert path_only_calls == [""]
    assert save_as_calls == []
    assert starts == [True]


def test_overwrite_dialog_cancel_button_starts_nothing(
    training_panel_module, monkeypatch
):
    panel, dialogs, starts, save_as_calls, _scheduled, _state = _overwrite_dialog_harness(
        training_panel_module, monkeypatch
    )
    panel._show_overwrite_dialog(-1)
    _title, _message, _buttons, callback = dialogs[0]

    callback(CANCEL_BTN)
    callback("")

    assert save_as_calls == []
    assert starts == []


def test_overwrite_and_start_still_starts_without_save_as(
    training_panel_module, monkeypatch
):
    panel, dialogs, starts, save_as_calls, _scheduled, _state = _overwrite_dialog_harness(
        training_panel_module, monkeypatch
    )
    panel._show_overwrite_dialog(9)
    _title, _message, _buttons, callback = dialogs[0]

    callback(OVERWRITE_BTN)

    assert save_as_calls == []
    assert starts == [True]


def test_action_start_opens_overwrite_dialog_instead_of_starting(
    training_panel_module, monkeypatch
):
    panel, dialogs, starts, save_as_calls, _scheduled, _state = _overwrite_dialog_harness(
        training_panel_module, monkeypatch
    )
    monkeypatch.setattr(
        training_panel_module.lf,
        "training_start_overwrite_conflict",
        lambda: 12,
    )

    panel._action_start()

    assert len(dialogs) == 1
    _title, _message, buttons, _callback = dialogs[0]
    assert buttons == [OVERWRITE_BTN, SAVE_AS_BTN, CANCEL_BTN]
    assert save_as_calls == []
    assert starts == []

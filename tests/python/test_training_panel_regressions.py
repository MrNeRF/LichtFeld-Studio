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
        property_group_info=lambda _group: {"properties": []},
        confirm_dialog=lambda *args, **kwargs: None,
    )
    lf_stub.optimization_params = lambda: None
    lf_stub.dataset_params = lambda: None
    lf_stub.get_scene = lambda: None
    lf_stub.start_training = lambda: None
    lf_stub.training_start_overwrite_conflict = lambda: None
    lf_stub.loss_buffer = lambda: []
    lf_stub.push_loss_to_element = lambda _element, _data: (0.0, 0.0)
    lf_stub.get_render_settings = lambda: None
    lf_stub.detect_dataset_info = lambda _path: None
    lf_stub.log = SimpleNamespace(error=lambda *_a, **_k: None, info=lambda *_a, **_k: None)
    lf_stub.io = SimpleNamespace(save_point_cloud_ply=lambda *_a, **_k: None)
    lf_stub.scene = SimpleNamespace(NodeType=SimpleNamespace(POINTCLOUD="POINTCLOUD"))
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)
    return lf_stub


def test_bundled_locales_define_training_panel_strategy_and_color_keys():
    project_root = Path(__file__).parent.parent.parent
    locale_dir = project_root / "src" / "visualizer" / "gui" / "resources" / "locales"

    for locale_path in locale_dir.glob("*.json"):
        data = json.loads(locale_path.read_text(encoding="utf-8"))
        assert data["training"]["options.strategy.igs_plus"] == "IGS+"
        assert "refinement.grow_until_iter" in data["training"]
        assert "tooltip.grow_until_iter" in data["training"]
        assert data["training"]["overwrite.btn_save_as_start"]
        assert data["training"]["save_pc.message_project"]
        assert data["training_panel"]["color_red_prefix"] == "R:"
        assert data["training_panel"]["color_green_prefix"] == "G:"
        assert data["training_panel"]["color_blue_prefix"] == "B:"


@pytest.mark.parametrize("enabled", [False, True])
def test_sparsity_toggle_controls_popspa_rows_and_search(
    training_panel_module, monkeypatch, enabled
):
    module = training_panel_module
    params = SimpleNamespace(
        has_params=lambda: True, mask_mode=SimpleNamespace(value=0),
        use_depth_loss=False, use_normal_loss=False, ppisp=False,
        use_exposure_correction=False, ppisp_use_controller=False,
        use_bilateral_grid=False, strategy="mcmc", random=False,
        enable_sparsity=enabled, enable_eval=False, eval_mask="", undistort=False,
    )
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)
    panel = module.TrainingPanel()
    assert panel._property_view_condition_visible("dep_sparsity") is enabled
    view = module.property_view
    sparsity_run = next(run for run in view.BASIC_RUNS if run.id == "sparsity")
    assert sparsity_run.prop_ids == view.POPSPA_PROPS
    assert sparsity_run.visibility_condition_id == "dep_sparsity"
    assert not set(view.POPSPA_PROPS) & set(view.PROP_VISIBILITY_CONDITION_IDS)
    # Search hides the whole run while sparsity is off, like ordinary rendering.
    rows = [
        dict(id=prop_id, kind="number", label_key="", name=prop_id,
             tooltip_key="", precision=0, text="0", strategies=())
        for prop_id in ("popspa_target_count", "popspa_sparsify_steps")
    ]
    binding = view.SectionBinding(
        "sparsity", rows, lambda: params, {}, lambda _binding: None,
        search_accessor=lambda: "s",
        visibility_condition_id="dep_sparsity",
        visibility_predicate=panel._property_view_condition_visible,
    )
    visible = {row["id"] for row in binding._records()}
    assert visible == ({"popspa_target_count", "popspa_sparsify_steps"} if enabled else set())


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
        self.eval_mask = ""
        self.eval_mask_report = ""
        self.save_steps = [7000]
        self.eval_steps = []
        self.bg_color = (0.0, 0.0, 0.0)
        self.bg_image_path = ""
        self.auto_scaled_for = None
        self.undistort = False
        self.gut = False
        self.strategy = "mrnf"

    def has_params(self):
        return True

    def validate(self):
        return ""

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
        self.eval_mask = ""
        self._slots = {
                "mrnf": {
                "max_cap": 5_000_000,
                "means_lr": 2e-5,
                "scaling_lr": 0.007,
                "lambda_dssim": 0.2,
                "init_opacity": 0.5,
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


def test_paused_iteration_limit_rejects_current_step_without_scaling(
    training_panel_module, monkeypatch
):
    params = _ParamsStub()
    params.iterations = 100
    params.steps_scaler = 2.0
    params.grow_until_iter = 50
    monkeypatch.setattr(
        training_panel_module,
        "RuntimeState",
        SimpleNamespace(
            training_state=SimpleNamespace(value="paused"),
            iteration=SimpleNamespace(value=100),
        ),
    )
    panel = training_panel_module.TrainingPanel()

    assert panel._set_iterations(params, 99) is False
    assert params.iterations == 100
    assert params.steps_scaler == 2.0
    assert params.grow_until_iter == 50


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


# Catches the panel replacing eval steps given with --eval-steps by the save steps.
def test_enabling_eval_keeps_requested_eval_steps(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    params = _ParamsStub()
    params.eval_steps = [300, 4000]
    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            dataset_params=lambda: _DatasetStub(),
            get_render_settings=lambda: None,
            get_scene=lambda: SimpleNamespace(active_camera_count=5),
        ),
    )

    panel._set_bool_prop("enable_eval", True)

    assert params.eval_steps == [300, 4000, 7000]


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


def test_training_rml_makes_every_panel_section_collapsible(training_panel_module):
    project_root = Path(__file__).parent.parent.parent
    training_rml = project_root / "src" / "visualizer" / "gui" / "rmlui" / "resources" / "training.rml"
    content = training_rml.read_text()

    panel_blocks = {"basic_params", "advanced_params"}
    for name in training_panel_module.SECTIONS:
        if name in panel_blocks:
            continue
        dom_name = name.replace("_", "-")
        assert f'id="hdr-{dom_name}" data-event-click="toggle_section(\'{name}\')"' in content
        assert f'id="arrow-{dom_name}"' in content
        assert f'class="section-content collapsed" id="sec-{dom_name}"' in content


def test_training_panel_restores_new_sections_collapsed(training_panel_module):
    panel = training_panel_module.TrainingPanel.__new__(training_panel_module.TrainingPanel)
    panel._handle = None
    panel.apply_chrome({"expanded": ["losses", "retired_section"]})

    assert panel._collapsed == training_panel_module.INITIALLY_COLLAPSED - {"losses"}
    assert panel.capture_chrome()["expanded"] == ["basic_params", "losses"]


def test_training_rml_exposes_mrnf_grow_until_iter():
    project_root = Path(__file__).parent.parent.parent
    training_rml = project_root / "src" / "visualizer" / "gui" / "rmlui" / "resources" / "training.rml"
    content = training_rml.read_text()

    assert 'data-for="row : pv_refinement_grow_rows" data-if="dep_mrnf"' in content
    assert 'data-value="row.text"' in content
    assert 'data-event-mousedown="pv_step(row.id, -1)"' in content


def test_training_panel_keeps_controls_and_search_outside_scroll_region():
    project_root = Path(__file__).parent.parent.parent
    resources = project_root / "src" / "visualizer" / "gui" / "rmlui" / "resources"
    rml = (resources / "training.rml").read_text()
    rcss = (resources / "training.rcss").read_text()
    panel_source = (project_root / "src" / "python" / "lfs_plugins" / "training_panel.py").read_text()

    controls = rml.index('id="controls"')
    search = rml.index('id="training-search-container"')
    telemetry = rml.index('class="section-gap training-telemetry"')
    scroll_start = rml.index('class="training-scroll-region"')
    parameters = rml.index('class="training-panel-title"')
    scroll_end = rml.index("<!-- /training-scroll-region -->")
    color_picker = rml.index('id="color-picker-popup"')

    assert controls < telemetry < search < scroll_start < parameters < scroll_end < color_picker
    assert 'class="section-gap training-telemetry" data-if="show_training_telemetry"' in rml
    assert 'id="training-search-icon" src="../icon/scene/search.png"' in rml
    assert 'id="training-search-input" type="text"' in rml
    assert '<img src="../icon/scene/x.png" />' in rml
    assert ".training-panel-layout" in rcss
    assert ".training-scroll-region" in rcss
    assert ".training-telemetry" in rcss
    assert "#training-search-input" in rcss
    assert "background-color: transparent" in rcss
    assert "border-width: 0" in rcss
    assert "overflow-y: auto" in rcss
    assert ".training-scroll-region scrollbarvertical" not in rcss
    assert "padding-bottom: 6dp" in rcss
    assert "height_mode = lf.ui.PanelHeightMode.FILL" in panel_source


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


# Catches the mesh resolution row showing for splat or point inits, or a file probe on every redraw.
def test_mesh_resolution_row_follows_the_init_file(training_panel_module, monkeypatch):
    dataset = SimpleNamespace(init_path="", has_params=lambda: True)
    probes = []

    def is_mesh_file(path):
        probes.append(path)
        return path.endswith(".obj")

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(dataset_params=lambda: dataset, io=SimpleNamespace(is_mesh_file=is_mesh_file)),
    )
    panel = training_panel_module.TrainingPanel()

    assert panel._property_view_condition_visible("dep_init_mesh") is False
    dataset.init_path = "/scan/model.obj"
    assert panel._property_view_condition_visible("dep_init_mesh") is True
    assert panel._property_view_condition_visible("dep_init_mesh") is True
    dataset.init_path = "/scan/points.ply"
    assert panel._property_view_condition_visible("dep_init_mesh") is False
    assert probes == ["/scan/model.obj", "/scan/points.ply"]

    training_rml = Path(__file__).parent.parent.parent / "src/visualizer/gui/rmlui/resources/training.rml"
    content = training_rml.read_text()
    guard = content.index('<div data-if="dep_init_mesh">')
    assert guard < content.index("pv_init_mesh_rows") < content.index("</div>", guard + 1)


# Catches a cleared path leaving pixel sets listed or a browse action ignoring its start directory.
def test_eval_mesh_browser_sets_path_and_clear_resets_the_report(
    training_panel_module, monkeypatch, tmp_path
):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    params = _ParamsStub()
    current_dir = tmp_path / "current"
    current_dir.mkdir()
    params.eval_mask = str(current_dir / "mask.obj")
    selected_path = str(tmp_path / "new" / "mask.ply")
    params.eval_mask_report = "inverted,mask"
    calls = []

    def open_mesh_file_dialog(start_dir):
        calls.append(start_dir)
        return selected_path

    monkeypatch.setattr(
        training_panel_module,
        "lf",
        SimpleNamespace(
            optimization_params=lambda: params,
            ui=SimpleNamespace(open_mesh_file_dialog=open_mesh_file_dialog),
        ),
    )

    panel._on_action(None, None, ["browse_eval_mask"])
    assert calls == [str(current_dir)]
    assert params.eval_mask == selected_path

    panel._on_action(None, None, ["clear_eval_mask"])
    assert params.eval_mask == ""
    assert params.eval_mask_report == ""
    assert panel._handle.dirty_all_count == 2


# Catches the headline select or the also-score checkboxes writing a list that loses the headline, keeps the headline
# as an extra set, or stores the plain mask as a report.
def test_eval_report_headline_and_also_scored_sets(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    params = _ParamsStub()
    params.eval_mask = "cropbox"
    monkeypatch.setattr(training_panel_module, "lf", SimpleNamespace(optimization_params=lambda: params))

    panel._set_eval_also_scored("inverted", True)
    assert params.eval_mask_report == "mask,inverted"
    panel._set_eval_also_scored("unmasked", True)
    assert params.eval_mask_report == "mask,unmasked,inverted"
    panel._set_eval_report("inverted", training_panel_module._eval_report_sets(params)[1:])
    assert params.eval_mask_report == "inverted,unmasked"
    panel._set_eval_also_scored("mask", True)
    assert params.eval_mask_report == "inverted,mask,unmasked"
    assert training_panel_module._eval_report_sets(params)[0] == "inverted"
    panel._set_eval_report("mask", ["unmasked"])
    panel._set_eval_also_scored("unmasked", False)
    assert params.eval_mask_report == ""


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


def test_training_telemetry_is_reserved_from_start_until_clear(training_panel_module):
    panel = training_panel_module.TrainingPanel()
    model = _ModelStub()
    params = _ParamsStub()
    dataset = _DatasetStub()
    runtime = training_panel_module.RuntimeState

    panel._bind_visibility(model, lambda: params, lambda: dataset)
    show_telemetry = model.bindings["show_training_telemetry"][0]

    try:
        runtime.trainer_state.value = "ready"
        runtime.iteration.value = 0
        assert show_telemetry() is False

        runtime.trainer_state.value = "running"
        assert show_telemetry() is True

        runtime.trainer_state.value = "ready"
        runtime.iteration.value = 1
        assert show_telemetry() is True

        runtime.iteration.value = 0
        assert show_telemetry() is False
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
    fresh_save_as_calls = []
    scheduled = []
    state = SimpleNamespace(has_path=False, save_as_result=True)

    def confirm_dialog(title, message, buttons, callback=None):
        dialogs.append((title, message, list(buttons), callback))

    def project_save_as(path="", wait=False):
        save_as_calls.append((path, wait))
        return state.save_as_result

    def project_save_as_for_training_start(path="", wait=False):
        fresh_save_as_calls.append((path, wait))
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
        "project_save_as_for_training_start",
        project_save_as_for_training_start,
        raising=False,
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
    return panel, dialogs, starts, save_as_calls, fresh_save_as_calls, scheduled, state


@pytest.mark.parametrize("conflict", [7000, -1])
def test_overwrite_dialog_offers_save_as_between_overwrite_and_cancel(
    training_panel_module, monkeypatch, conflict
):
    panel, dialogs, _starts, _save_as_calls, _fresh_save_as_calls, _scheduled, _state = _overwrite_dialog_harness(
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


def test_overwrite_save_as_uses_fresh_run_save_and_starts_after_bind(
    training_panel_module, monkeypatch
):
    panel, dialogs, starts, save_as_calls, fresh_save_as_calls, scheduled, state = _overwrite_dialog_harness(
        training_panel_module, monkeypatch
    )
    panel._show_overwrite_dialog(12)
    _title, _message, _buttons, callback = dialogs[0]

    state.save_as_result = True
    state.has_path = True
    callback(SAVE_AS_BTN)

    assert save_as_calls == [("", True)]
    assert fresh_save_as_calls == [("", True)]
    assert starts == [True]
    assert scheduled == []


def test_overwrite_save_as_waits_for_fire_and_forget_save_to_bind(
    training_panel_module, monkeypatch
):
    panel, dialogs, starts, save_as_calls, _fresh_save_as_calls, scheduled, state = _overwrite_dialog_harness(
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
    panel, dialogs, starts, save_as_calls, _fresh_save_as_calls, scheduled, state = _overwrite_dialog_harness(
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
    panel, dialogs, starts, save_as_calls, _fresh_save_as_calls, scheduled, state = _overwrite_dialog_harness(
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
    panel, dialogs, starts, save_as_calls, _fresh_save_as_calls, _scheduled, state = _overwrite_dialog_harness(
        training_panel_module, monkeypatch
    )
    path_only_calls = []

    def project_save_as(path=""):
        path_only_calls.append(path)
        return True

    monkeypatch.setattr(
        training_panel_module.lf, "project_save_as", project_save_as, raising=False
    )
    monkeypatch.setattr(
        training_panel_module.lf,
        "project_save_as_for_training_start",
        None,
        raising=False,
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
    panel, dialogs, starts, save_as_calls, _fresh_save_as_calls, _scheduled, _state = _overwrite_dialog_harness(
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
    panel, dialogs, starts, save_as_calls, _fresh_save_as_calls, _scheduled, _state = _overwrite_dialog_harness(
        training_panel_module, monkeypatch
    )
    panel._show_overwrite_dialog(9)
    _title, _message, _buttons, callback = dialogs[0]

    callback(OVERWRITE_BTN)

    assert save_as_calls == []
    assert starts == [True]


def test_finished_run_overwrite_resets_before_the_new_run_starts(
    training_panel_module, monkeypatch
):
    """A completed run is a new training, so Overwrite must leave the finished trainer first."""
    panel, dialogs, _starts, _save_as_calls, _fresh_save_as_calls, _scheduled, _state = _overwrite_dialog_harness(
        training_panel_module, monkeypatch
    )
    runtime = training_panel_module.RuntimeState
    order = []

    def reset_training():
        order.append(("reset", runtime.trainer_state.value))
        runtime.trainer_state.value = "ready"
        runtime.iteration.value = 0

    def start_training():
        order.append(("start", runtime.trainer_state.value))

    monkeypatch.setattr(
        training_panel_module.lf,
        "training_start_overwrite_conflict",
        lambda: 40,
    )
    monkeypatch.setattr(
        training_panel_module.lf, "reset_training", reset_training, raising=False
    )
    monkeypatch.setattr(training_panel_module.lf, "start_training", start_training)

    try:
        runtime.trainer_state.value = "completed"
        runtime.iteration.value = 40
        panel._action_start()

        assert len(dialogs) == 1
        title, _message, buttons, callback = dialogs[0]
        assert title == "training.overwrite.title"
        assert buttons[0] == OVERWRITE_BTN
        assert order == []

        callback(OVERWRITE_BTN)
        assert order == [("reset", "completed"), ("start", "ready")]
    finally:
        runtime.iteration._fallback = 0
        runtime.training_state._fallback = "idle"


def test_action_start_opens_overwrite_dialog_instead_of_starting(
    training_panel_module, monkeypatch
):
    panel, dialogs, starts, save_as_calls, _fresh_save_as_calls, _scheduled, _state = _overwrite_dialog_harness(
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


def test_reset_prompts_when_dirty_then_resets_on_continue(
    training_panel_module, monkeypatch
):
    dialogs = []
    resets = []
    monkeypatch.setattr(
        training_panel_module.lf.ui,
        "confirm_dialog",
        lambda title, message, buttons, callback=None: dialogs.append(
            (title, message, list(buttons), callback)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        training_panel_module.lf, "project_is_dirty", lambda: True, raising=False
    )
    monkeypatch.setattr(
        training_panel_module.lf, "project_has_path", lambda: False, raising=False
    )
    monkeypatch.setattr(
        training_panel_module.lf, "is_training_active", lambda: False, raising=False
    )
    monkeypatch.setattr(
        training_panel_module.lf,
        "reset_training",
        lambda: resets.append(True),
        raising=False,
    )

    panel = training_panel_module.TrainingPanel()
    panel._on_action(None, None, ["reset"])

    assert resets == []
    assert len(dialogs) == 1
    title, message, buttons, callback = dialogs[0]
    assert title == "training_panel.reset"
    assert message == "exit_popup.unsaved_warning"
    assert buttons == [
        "menu.file.save_project_as",
        "unsaved_work.continue_without_saving",
        "common.cancel",
    ]

    callback("common.cancel")
    assert resets == []

    callback("unsaved_work.continue_without_saving")
    assert resets == [True]


def test_reset_when_dirty_and_training_does_not_ask_stop(
    training_panel_module, monkeypatch
):
    dialogs = []
    resets = []
    monkeypatch.setattr(
        training_panel_module.lf.ui,
        "confirm_dialog",
        lambda title, message, buttons, callback=None: dialogs.append(
            (title, message, list(buttons), callback)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        training_panel_module.lf, "project_is_dirty", lambda: True, raising=False
    )
    monkeypatch.setattr(
        training_panel_module.lf, "project_has_path", lambda: False, raising=False
    )
    monkeypatch.setattr(
        training_panel_module.lf, "is_training_active", lambda: True, raising=False
    )
    monkeypatch.setattr(
        training_panel_module.lf,
        "reset_training",
        lambda: resets.append(True),
        raising=False,
    )

    panel = training_panel_module.TrainingPanel()
    panel._on_action(None, None, ["reset"])

    assert resets == []
    assert len(dialogs) == 1
    title, message, buttons, callback = dialogs[0]
    assert title == "training_panel.reset"
    assert message == "exit_popup.unsaved_warning"
    assert buttons == [
        "menu.file.save_project_as",
        "unsaved_work.continue_without_saving",
        "common.cancel",
    ]

    callback("unsaved_work.continue_without_saving")
    assert resets == [True]
    assert len(dialogs) == 1


def test_reset_when_training_and_clean_runs_immediately(
    training_panel_module, monkeypatch
):
    dialogs = []
    resets = []
    monkeypatch.setattr(
        training_panel_module.lf.ui,
        "confirm_dialog",
        lambda title, message, buttons, callback=None: dialogs.append(
            (title, message, list(buttons), callback)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        training_panel_module.lf, "project_is_dirty", lambda: False, raising=False
    )
    monkeypatch.setattr(
        training_panel_module.lf, "is_training_active", lambda: True, raising=False
    )
    monkeypatch.setattr(
        training_panel_module.lf,
        "reset_training",
        lambda: resets.append(True),
        raising=False,
    )

    panel = training_panel_module.TrainingPanel()
    panel._on_action(None, None, ["reset"])

    assert dialogs == []
    assert resets == [True]


def test_reset_without_dirty_or_training_runs_immediately(
    training_panel_module, monkeypatch
):
    resets = []
    monkeypatch.setattr(
        training_panel_module.lf, "project_is_dirty", lambda: False, raising=False
    )
    monkeypatch.setattr(
        training_panel_module.lf, "is_training_active", lambda: False, raising=False
    )
    monkeypatch.setattr(
        training_panel_module.lf,
        "reset_training",
        lambda: resets.append(True),
        raising=False,
    )

    panel = training_panel_module.TrainingPanel()
    panel._on_action(None, None, ["reset"])

    assert resets == [True]


def _stub_stored_session(training_panel_module, monkeypatch, **overrides):
    state = {
        "available": True,
        "iteration": 0,
        "max_iterations": 0,
        "strategy": "mrnf",
        "completed": False,
        "hydrated": False,
        "restoring": False,
        "error": "",
    }
    state.update(overrides)
    monkeypatch.setattr(
        training_panel_module.lf,
        "project_training_session_state",
        lambda: dict(state),
        raising=False,
    )
    return state


def test_completed_stored_session_shows_complete_mode_and_buttons(
    training_panel_module, monkeypatch
):
    _stub_stored_session(
        training_panel_module,
        monkeypatch,
        iteration=30000,
        max_iterations=30000,
        completed=True,
    )
    panel = training_panel_module.TrainingPanel()
    model = _ModelStub()
    params = _ParamsStub()
    dataset = _DatasetStub()
    runtime = training_panel_module.RuntimeState

    panel._bind_visibility(model, lambda: params, lambda: dataset)
    panel._bind_status(model, lambda: params)
    try:
        runtime.has_trainer.value = False
        runtime.trainer_state.value = "idle"
        runtime.iteration.value = 30000
        runtime.max_iterations.value = 30000

        assert model.bindings["show_ctrl_completed"][0]() is True
        assert model.bindings["show_ctrl_paused"][0]() is False
        assert model.bindings["show_ctrl_ready"][0]() is False
        assert "status.complete" in model.bindings["status_mode"][0]()
        assert "30,000/30,000" in model.bindings["progress_text"][0]()
        assert "session_at_iteration" not in model.bindings["status_mode"][0]()
    finally:
        runtime.has_trainer._fallback = False
        runtime.training_state._fallback = "idle"
        runtime.iteration._fallback = 0
        runtime.total_iterations._fallback = 0


def test_paused_stored_session_shows_paused_mode_and_resume(
    training_panel_module, monkeypatch
):
    _stub_stored_session(
        training_panel_module,
        monkeypatch,
        iteration=7000,
        max_iterations=30000,
        completed=False,
    )
    panel = training_panel_module.TrainingPanel()
    model = _ModelStub()
    params = _ParamsStub()
    dataset = _DatasetStub()
    runtime = training_panel_module.RuntimeState

    panel._bind_visibility(model, lambda: params, lambda: dataset)
    panel._bind_status(model, lambda: params)
    try:
        runtime.has_trainer.value = False
        runtime.trainer_state.value = "idle"
        runtime.iteration.value = 7000
        runtime.max_iterations.value = 30000

        assert model.bindings["show_ctrl_paused"][0]() is True
        assert model.bindings["show_ctrl_completed"][0]() is False
        assert model.bindings["show_ctrl_ready"][0]() is False
        assert "status.paused" in model.bindings["status_mode"][0]()
        assert "7,000/30,000" in model.bindings["progress_text"][0]()
        assert "session_at_iteration" not in model.bindings["status_mode"][0]()
    finally:
        runtime.has_trainer._fallback = False
        runtime.training_state._fallback = "idle"
        runtime.iteration._fallback = 0
        runtime.total_iterations._fallback = 0


def test_late_restore_failure_notifies_panel_and_keeps_resume_disabled(
    training_panel_module, monkeypatch
):
    session = {
        "available": True,
        "iteration": 7000,
        "max_iterations": 30000,
        "strategy": "mrnf",
        "completed": False,
        "hydrated": False,
        "restoring": True,
        "error": "",
    }
    monkeypatch.setattr(training_panel_module, "_training_session_state", lambda: session)
    panel = training_panel_module.TrainingPanel()
    panel._handle = _HandleStub()
    model = _ModelStub()
    runtime = training_panel_module.RuntimeState
    elapsed = [0.0]
    scheduled = []
    monkeypatch.setattr(
        training_panel_module, "time",
        SimpleNamespace(monotonic=lambda: elapsed[0], time=lambda: elapsed[0]),
    )
    monkeypatch.setattr(panel, "_schedule_deferred_update", scheduled.append)
    for name in "_sync_auto_scale_markers _sync_text_bufs _update_step_repeat " \
                "_update_progress _update_save_steps _update_color_swatch " \
                "_update_loss_graph _update_psnr_graph".split():
        monkeypatch.setattr(panel, name, lambda *_args: False)
    monkeypatch.setattr(panel._scrub_fields, "sync_all", lambda: False)
    panel._bind_visibility(model, lambda: _ParamsStub(), lambda: _DatasetStub())
    panel._subscribe_reactive_state()

    try:
        runtime.has_trainer.value = True
        runtime.trainer_state.value = "paused"
        panel._handle.request_update_count = 0
        panel.on_update(None)
        assert scheduled == []
        assert model.bindings["show_ctrl_paused"][0]() is False
        generation = runtime.stored_session_generation.value

        # Simulate a restore that finishes well beyond the old 30-second cap.
        elapsed[0] = 60.0
        session.update(restoring=False, error="Project trainer restore failed")
        runtime.stored_session_generation.value = generation + 1

        assert panel._handle.request_update_count == 1
        assert model.bindings["show_ctrl_paused"][0]() is False
        assert model.bindings["show_ctrl_error"][0]() is True
    finally:
        panel._unsubscribe_reactive_state()
        runtime.has_trainer._fallback = False
        runtime.training_state._fallback = "idle"
        runtime.iteration._fallback = 0
        runtime.total_iterations._fallback = 0


def test_stopped_stored_session_keeps_stopped_mode_and_edit_controls(
    training_panel_module, monkeypatch
):
    _stub_stored_session(
        training_panel_module,
        monkeypatch,
        iteration=7000,
        max_iterations=30000,
        completed=False,
    )
    panel = training_panel_module.TrainingPanel()
    model = _ModelStub()
    params = _ParamsStub()
    dataset = _DatasetStub()
    runtime = training_panel_module.RuntimeState

    panel._bind_visibility(model, lambda: params, lambda: dataset)
    panel._bind_status(model, lambda: params)
    try:
        runtime.has_trainer.value = False
        runtime.trainer_state.value = "stopped"
        runtime.iteration.value = 7000
        runtime.max_iterations.value = 30000

        assert model.bindings["show_ctrl_paused"][0]() is False
        assert model.bindings["show_ctrl_stopped"][0]() is True
        assert model.bindings["show_ctrl_completed"][0]() is False
        assert model.bindings["show_ctrl_ready"][0]() is False
        assert "status.stopped" in model.bindings["status_mode"][0]()
        assert "7,000/30,000" in model.bindings["progress_text"][0]()
        assert "session_at_iteration" not in model.bindings["status_mode"][0]()
    finally:
        runtime.has_trainer._fallback = False
        runtime.training_state._fallback = "idle"
        runtime.iteration._fallback = 0
        runtime.total_iterations._fallback = 0


def test_resume_on_stored_session_restores_before_resuming(
    training_panel_module, monkeypatch
):
    _stub_stored_session(
        training_panel_module,
        monkeypatch,
        iteration=7000,
        max_iterations=30000,
        completed=False,
    )
    calls = []
    monkeypatch.setattr(
        training_panel_module.lf,
        "restore_training_session",
        lambda then_start=False: calls.append(("restore", then_start)),
        raising=False,
    )
    monkeypatch.setattr(
        training_panel_module.lf,
        "resume_training",
        lambda: calls.append(("resume",)),
        raising=False,
    )
    runtime = training_panel_module.RuntimeState
    try:
        runtime.has_trainer.value = False
        runtime.trainer_state.value = "paused"
        panel = training_panel_module.TrainingPanel()
        panel._on_action(None, None, ["resume"])
        assert calls[0] == ("restore", True)
        assert ("resume",) in calls
    finally:
        runtime.has_trainer._fallback = False
        runtime.training_state._fallback = "idle"


def test_save_modified_pc_writes_to_bound_project(training_panel_module, monkeypatch):
    project_saves = []
    detect_calls = []
    ply_calls = []
    scene = SimpleNamespace(is_point_cloud_modified=True)

    def save_titled():
        project_saves.append(True)
        return True

    monkeypatch.setattr(training_panel_module, "_project_has_path", lambda: True)
    monkeypatch.setattr(training_panel_module, "_save_titled_project", save_titled)
    monkeypatch.setattr(training_panel_module.lf, "get_scene", lambda: scene)
    monkeypatch.setattr(
        training_panel_module.lf,
        "detect_dataset_info",
        lambda *args, **kwargs: detect_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        training_panel_module.lf,
        "io",
        SimpleNamespace(
            save_point_cloud_ply=lambda *args, **kwargs: ply_calls.append((args, kwargs))
        ),
    )

    training_panel_module.TrainingPanel()._save_modified_pc()

    assert project_saves == [True]
    assert detect_calls == []
    assert ply_calls == []
    assert scene.is_point_cloud_modified is False


def test_save_modified_pc_bound_project_failed_save_skips_dataset(
    training_panel_module, monkeypatch
):
    ply_calls = []
    scene = SimpleNamespace(is_point_cloud_modified=True)

    monkeypatch.setattr(training_panel_module, "_project_has_path", lambda: True)
    monkeypatch.setattr(training_panel_module, "_save_titled_project", lambda: False)
    monkeypatch.setattr(training_panel_module.lf, "get_scene", lambda: scene)
    monkeypatch.setattr(
        training_panel_module.lf,
        "io",
        SimpleNamespace(
            save_point_cloud_ply=lambda *args, **kwargs: ply_calls.append((args, kwargs))
        ),
    )

    training_panel_module.TrainingPanel()._save_modified_pc()

    assert ply_calls == []
    assert scene.is_point_cloud_modified is True


def test_save_modified_pc_unbound_writes_dataset_ply(training_panel_module, monkeypatch):
    ply_calls = []
    pc = SimpleNamespace(size=12)
    node = SimpleNamespace(
        type=training_panel_module.lf.scene.NodeType.POINTCLOUD,
        point_cloud=lambda: pc,
    )
    scene = SimpleNamespace(
        is_point_cloud_modified=True,
        get_nodes=lambda: [node],
    )
    dataset = SimpleNamespace(data_path="/data/scene_a", has_params=lambda: True)
    info = SimpleNamespace(sparse_path="/data/scene_a/sparse/0")

    monkeypatch.setattr(training_panel_module, "_project_has_path", lambda: False)
    monkeypatch.setattr(training_panel_module.lf, "dataset_params", lambda: dataset)
    monkeypatch.setattr(training_panel_module.lf, "detect_dataset_info", lambda _path: info)
    monkeypatch.setattr(training_panel_module.lf, "get_scene", lambda: scene)
    monkeypatch.setattr(
        training_panel_module.lf,
        "io",
        SimpleNamespace(
            save_point_cloud_ply=lambda cloud, path: ply_calls.append((cloud, path))
        ),
    )

    training_panel_module.TrainingPanel()._save_modified_pc()

    assert [(cloud, Path(path)) for cloud, path in ply_calls] == [
        (pc, Path("/data/scene_a/sparse/0/points3D.ply"))
    ]
    assert scene.is_point_cloud_modified is False


@pytest.mark.parametrize(
    ("bound", "expected_message"),
    [
        (True, "training.save_pc.message_project"),
        (False, "training.save_pc.message"),
    ],
)
def test_show_save_pc_dialog_message_depends_on_project_binding(
    training_panel_module, monkeypatch, bound, expected_message
):
    dialogs = []

    monkeypatch.setattr(training_panel_module, "_project_has_path", lambda: bound)
    monkeypatch.setattr(
        training_panel_module.lf.ui,
        "confirm_dialog",
        lambda title, message, buttons, callback=None: dialogs.append(
            (title, message, list(buttons), callback)
        ),
        raising=False,
    )

    training_panel_module.TrainingPanel()._show_save_pc_dialog()

    assert len(dialogs) == 1
    title, message, buttons, _callback = dialogs[0]
    assert title == "training.save_pc.title"
    assert message == expected_message
    assert buttons == [
        "training.save_pc.btn_save_start",
        "training.save_pc.btn_start_without",
        "training.conflict.btn_cancel",
    ]


@pytest.mark.parametrize(
    "prop,initial,changed,method,args",
    [
        ("strategy", "mrnf", "mcmc", "_set_strategy", ()),
        ("sh_degree", 3, 2, "_set_int_param", ("sh_degree",)),
        ("depth_loss_mode", "ssi", "ssi-disparity", "_set_depth_loss_mode", ()),
        ("lambda_dssim", 0.25, 0.5, "_set_slider_prop", ("lambda_dssim",)),
    ],
)
def test_restored_control_echo_is_not_an_edit(
    training_panel_module, monkeypatch, prop, initial, changed, method, args
):
    class Params:
        def __init__(self):
            object.__setattr__(self, "writes", [])
            object.__setattr__(self, "gut", False)
            object.__setattr__(self, prop, initial)

        def __setattr__(self, name, value):
            self.writes.append((name, value))
            object.__setattr__(self, name, value)

        def has_params(self):
            return True

        def set(self, name, value):
            setattr(self, name, value)

        def set_strategy(self, value):
            self.strategy = value

    params = Params()
    monkeypatch.setattr(training_panel_module.lf, "optimization_params", lambda: params)
    panel = training_panel_module.TrainingPanel()
    setter = getattr(panel, method)
    setter(*args, initial)
    assert params.writes == []
    setter(*args, changed)
    assert params.writes == [(prop, changed)]
    assert getattr(params, prop) == changed
    setter(*args, changed)
    assert params.writes == [(prop, changed)]


@pytest.mark.parametrize(
    ("prop", "saved", "control_echo"),
    [
        ("init_opacity", 0.5, 0.4999999701976776),
        ("lambda_dssim", 0.2, 0.20000000298023224),
    ],
)
def test_slider_float_echo_does_not_edit_saved_value(
    training_panel_module, monkeypatch, prop, saved, control_echo
):
    class Params:
        def __init__(self):
            self.value = saved
            self.writes = []

        def has_params(self):
            return True

        def __getattr__(self, name):
            if name == prop:
                return self.value
            raise AttributeError(name)

        def set(self, name, value):
            self.writes.append((name, value))
            self.value = value

    params = Params()
    monkeypatch.setattr(training_panel_module.lf, "optimization_params", lambda: params)
    panel = training_panel_module.TrainingPanel()
    panel._set_slider_prop(prop, control_echo)
    assert params.writes == []
    assert params.value == saved

    panel._set_slider_prop(prop, saved + 0.01)
    assert params.writes == [(prop, saved + 0.01)]

    panel._set_slider_prop(prop, saved + 0.010002)
    assert params.writes[-1] == (prop, saved + 0.010002)


@pytest.mark.parametrize("completed", [False, True])
@pytest.mark.parametrize("runtime_iteration", [0, 1776])
def test_stored_session_iteration_does_not_report_a_live_rate(
    training_panel_module, monkeypatch, completed, runtime_iteration
):
    module = training_panel_module
    runtime = SimpleNamespace(iteration=_make_signal(0), has_trainer=_make_signal(False))
    monkeypatch.setattr(module, "RuntimeState", runtime)
    now = [10.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    panel = module.TrainingPanel()
    model = _ModelStub()
    panel._bind_status(model, lambda: None)
    status = model.bindings["status_iteration"][0]
    status()  # The panel is evaluated before a project is opened.
    session = _stub_stored_session(
        module, monkeypatch, iteration=1776, completed=completed
    )
    runtime.iteration.value = runtime_iteration
    now[0] += 0.4

    assert status() == "status.iteration 1,776"
    assert module._rate_tracker.samples == []

    # Hydrating and resuming starts a fresh measurement at the stored counter.
    session["hydrated"] = True
    runtime.has_trainer.value = True
    runtime.iteration.value = 1776
    assert "(0.0 training_panel.iters_per_sec)" in status()
    now[0] += 1.0
    runtime.iteration.value += 373
    assert "(373.0 training_panel.iters_per_sec)" in status()


def test_live_iteration_rate_matches_existing_window(training_panel_module, monkeypatch):
    module = training_panel_module
    runtime = SimpleNamespace(iteration=_make_signal(0), has_trainer=_make_signal(True))
    monkeypatch.setattr(module, "RuntimeState", runtime)
    # Saving a live run also exposes a stored session. It must not hide live rates.
    _stub_stored_session(module, monkeypatch, iteration=373, hydrated=False)
    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    model = _ModelStub()
    module.TrainingPanel()._bind_status(model, lambda: None)
    status = model.bindings["status_iteration"][0]
    samples = []
    for elapsed, iteration in [(0, 0), (1, 373), (2, 746), (3, 746), (6, 1119)]:
        now[0] = elapsed
        runtime.iteration.value = iteration
        # Reference the existing five-second estimator, including paused updates.
        shown_iteration = iteration or 373
        samples.append((shown_iteration, elapsed))
        samples = [(i, t) for i, t in samples if elapsed - t <= 5.0]
        dt = samples[-1][1] - samples[0][1]
        expected = (samples[-1][0] - samples[0][0]) / dt if dt > 0 else 0.0
        assert status() == (
            f"status.iteration {shown_iteration:,} ({expected:.1f} training_panel.iters_per_sec)"
        )


@pytest.mark.parametrize("key", ["messages.err_distorted_images", "messages.err_non_pinhole"])
def test_camera_error_resolves_locale_when_presented(training_panel_module, monkeypatch, key):
    panel = training_panel_module.TrainingPanel()
    model = _ModelStub()
    panel._bind_status(model, lambda: None)
    monkeypatch.setattr(training_panel_module.lf, "trainer_error", lambda: "Readable English guidance", raising=False)
    monkeypatch.setattr(training_panel_module.lf, "trainer_error_key", lambda: key, raising=False)
    get_message = model.bindings["error_message"][0]
    for translation in ("Aktivieren Sie GUT im Trainingspanel.", "Activez GUT dans le panneau Entraînement."):
        monkeypatch.setattr(training_panel_module.lf.ui, "tr", lambda value: translation if value == key else value)
        assert get_message() == translation
    monkeypatch.setattr(training_panel_module.lf, "trainer_error", lambda: "Other diagnostic")
    monkeypatch.setattr(training_panel_module.lf, "trainer_error_key", lambda: None)
    assert get_message() == "Other diagnostic"
    monkeypatch.setattr(training_panel_module.lf, "trainer_error_key", lambda: "unknown.key")
    assert get_message() == "Other diagnostic"
    monkeypatch.setattr(training_panel_module.lf, "trainer_error", lambda: None)

    def unexpected_key_lookup():
        pytest.fail("No localization lookup is needed without an error")

    monkeypatch.setattr(training_panel_module.lf, "trainer_error_key", unexpected_key_lookup)
    assert get_message() == ""


def test_camera_guidance_locales_use_gui_options():
    locale_dir = Path(__file__).resolve().parents[2] / "src/visualizer/gui/resources/locales"
    locales = list(locale_dir.glob("*.json"))
    assert len(locales) == 10
    for path in locales:
        messages = json.loads(path.read_text(encoding="utf-8"))["messages"]
        for key in ("err_distorted_images", "err_non_pinhole"):
            assert "--gut" not in messages[key], path.name
            assert "--undistort" not in messages[key], path.name
            assert "GUT" in messages[key], path.name


def test_trainer_error_api_without_manager(lf):
    assert lf.trainer_error() is None
    assert lf.trainer_error_key() is None


@pytest.fixture
def ready_panel_settings(training_panel_module, monkeypatch):
    module = training_panel_module
    callbacks = {}
    monkeypatch.setattr(module.lf.ui, "schedule_on_ui_thread", lambda callback: callback(), raising=False)
    next_token = iter(range(1000))

    def subscribe(path, callback):
        token = next(next_token)
        callbacks[token] = (path, callback)
        return token

    monkeypatch.setattr(module.lf, "on_property_change", subscribe, raising=False)
    monkeypatch.setattr(module.lf, "unsubscribe_property_change", callbacks.pop, raising=False)
    monkeypatch.setattr(module.lf.ui, "property_group_info", lambda group: {
        "properties": [{"id": name} for name in (
            ("iterations", "max_cap", "enable_eval", "bg_mode", "strategy", "bg_color", "eval_mask", "ppisp_controller_activation_step", "steps_scaler")
            if group == "optimization" else ("max_width", "test_every")
        )]
    })
    params = _ParamsStub()
    params.max_cap = 5_000_000
    params.iterations = 30_000
    dataset = _DatasetStub()
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)
    monkeypatch.setattr(module.lf, "dataset_params", lambda: dataset)
    monkeypatch.setattr(module, "RuntimeState", SimpleNamespace(**{
        name: _make_signal(value) for name, value in {
            "trainer_state": "ready", "language_generation": 0,
            "iteration": 0, "num_gaussians": 0,
        }.items()
    }))
    # Use the real native-signal facade for subscription lifetime tests.
    native_state = import_module("lfs_plugins.ui").RuntimeState
    for name in ("training_running", "training_state", "trainer_loaded", "total_iterations",
                 "loss", "eval_psnr", "scene_generation", "stored_session_generation"):
        setattr(module.RuntimeState, name, getattr(native_state, name))
    for name in ("iteration", "num_gaussians", "language_generation"):
        setattr(module.RuntimeState, name, getattr(native_state, name))
    panel = module.TrainingPanel()
    panel._handle = _HandleStub()
    rows = [{"id": name, "kind": "number", "is_int": True, "precision": 0,
             "strategies": (), "name": name, "label_key": "", "tooltip_key": "",
             "step": 1, "min": 0, "max": 10_000_000, "items": []}
            for name in ("iterations", "max_cap")]
    binding = module.property_view.SectionBinding(
        "limits", rows, lambda: params, panel._text_bufs, panel._queue_pv_publish)
    panel._pv_bindings = (binding,)
    for name in ("_sync_section_states", "_update_step_repeat", "_update_progress",
                 "_update_save_steps", "_update_color_swatch", "_update_loss_graph",
                 "_update_psnr_graph", "_flush_pv_publish"):
        monkeypatch.setattr(panel, name, lambda *_args: False)
    monkeypatch.setattr(panel._scrub_fields, "sync_all", lambda: False)
    panel.on_update(None)
    panel.on_update(None)
    panel._handle = _HandleStub()
    panel._subscribe_reactive_state()

    def change(group, name, value):
        target = params if group == "optimization" else dataset
        old = getattr(target, name, None)
        setattr(target, name, value)
        for path, callback in list(callbacks.values()):
            if path == f"{group}.{name}":
                callback(old, value)

    yield panel, params, dataset, binding, change, callbacks
    panel._unsubscribe_reactive_state()


def test_external_settings_refresh_ready_panel_before_start(ready_panel_settings):
    panel, params, dataset, binding, change, _ = ready_panel_settings
    change("optimization", "iterations", 600)
    change("optimization", "max_cap", 100_000)
    change("optimization", "bg_mode", "random")
    assert panel._handle.request_update_count == 1
    assert panel.on_update(None)
    assert panel._text_bufs[binding.input_key("iterations")] == "600"
    assert panel._text_bufs[binding.input_key("max_cap")] == "100,000"
    assert panel._handle.dirty_all_count == 1
    assert params.iterations == 600
    assert dataset.test_every == 8


def test_external_dataset_settings_refresh_text_fields(ready_panel_settings):
    panel, _, _, _, change, _ = ready_panel_settings
    change("dataset", "max_width", 1024)
    change("dataset", "test_every", 16)
    assert panel._handle.request_update_count == 1
    panel.on_update(None)
    assert panel._text_bufs["max_width_str"] == "1,024"
    assert panel._text_bufs["test_every_str"] == "16"


def test_settings_subscriptions_close_and_remount_once(ready_panel_settings):
    panel, _, _, _, change, callbacks = ready_panel_settings
    count = len(callbacks)
    assert count == 11
    panel._subscribe_reactive_state()
    assert len(callbacks) == count
    panel._unsubscribe_reactive_state()
    assert not callbacks
    change("optimization", "iterations", 600)
    assert panel._handle.request_update_count == 0
    panel._subscribe_reactive_state()
    assert len(callbacks) == count
    change("optimization", "iterations", 700)
    assert panel._handle.request_update_count == 1


def test_idle_updates_preserve_draft_and_do_not_republish(ready_panel_settings):
    panel, _, _, binding, _, _ = ready_panel_settings
    key = binding.input_key("iterations")
    panel._text_bufs[key] = "123"
    for _ in range(3):
        assert panel.on_update(None) is False
    assert panel._text_bufs[key] == "123"
    assert panel._handle.dirty_all_count == 0
    assert panel._handle.request_update_count == 0


def test_settings_notification_defers_ui_work(
    ready_panel_settings, training_panel_module, monkeypatch
):
    panel, _, _, _, change, _ = ready_panel_settings
    scheduled = []
    monkeypatch.setattr(training_panel_module.lf.ui, "schedule_on_ui_thread", scheduled.append)
    change("optimization", "iterations", 600)
    change("optimization", "max_cap", 100_000)
    assert len(scheduled) == 1
    assert panel._handle.request_update_count == 0
    assert panel._handle.dirty_all_count == 0
    scheduled.pop()()
    assert panel._handle.request_update_count == 1
    panel.on_update(None)
    assert panel._handle.dirty_all_count == 1


@pytest.mark.parametrize("changed_property", ["max_cap", "iterations"])
def test_external_change_preserves_active_numeric_draft(ready_panel_settings, changed_property):
    panel, params, _, binding, change, _ = ready_panel_settings
    binding.begin_edit("iterations")
    binding.update_draft("iterations", "123")
    change("optimization", changed_property, 100_000)
    panel.on_update(None)
    assert panel._text_bufs[binding.input_key("iterations")] == "123"
    if changed_property == "max_cap":
        assert panel._text_bufs[binding.input_key("max_cap")] == "100,000"
    binding.commit("iterations")
    binding.finish_edit("iterations")
    assert params.iterations == 123
    change("optimization", "iterations", 700)
    panel.on_update(None)
    assert panel._text_bufs[binding.input_key("iterations")] == "700"


class _DraftInput:
    def __init__(self):
        self.listeners = {}

    def add_event_listener(self, name, callback):
        self.listeners[name] = callback

    def parent(self):
        return self


@pytest.mark.parametrize("key,group,prop,value", [
    ("max_width_str", "dataset", "max_width", 1024),
    ("test_every_str", "dataset", "test_every", 16),
    ("eval_mask_spec_str", "optimization", "eval_mask", "changed"),
    ("ppisp_activation_step_str", "optimization", "ppisp_controller_activation_step", 2000),
    ("bg_color_hex", "optimization", "bg_color", (0.5, 0.25, 1.0)),
    ("bg_color_r", "optimization", "bg_color", (0.5, 0.25, 1.0)),
])
def test_external_change_preserves_manual_draft(
    ready_panel_settings, training_panel_module, key, group, prop, value
):
    panel, _, _, _, change, _ = ready_panel_settings
    element = _DraftInput()
    panel._escape_revert.bind(element, key,
                             lambda: panel._canonical_text_buf_value(key),
                             lambda snapshot: panel._restore_number_input_snapshot(key, snapshot))
    element.listeners["focus"](None)
    panel._text_bufs[key] = "123"
    change("optimization", "max_cap", 100_000)
    panel.on_update(None)
    assert panel._text_bufs[key] == "123"
    change(group, prop, value)
    panel.on_update(None)
    if prop == "bg_color":
        training_panel_module.TrainingPanel._update_color_swatch(
            panel, SimpleNamespace(get_element_by_id=lambda _id: None))
    assert panel._text_bufs[key] == "123"
    element.listeners["blur"](None)
    change(group, prop, value)
    panel.on_update(None)
    assert panel._text_bufs[key] == panel._canonical_text_buf_value(key)


def test_unrelated_notification_does_not_reset_unedited_buffers(ready_panel_settings):
    panel, _, _, binding, change, _ = ready_panel_settings
    key = binding.input_key("iterations")
    panel._text_bufs[key] = "030000"
    panel._text_bufs["new_step_str"] = "123"
    change("dataset", "max_width", 1024)
    panel.on_update(None)
    assert panel._text_bufs[key] == "030000"
    assert panel._text_bufs["new_step_str"] == "123"
    assert panel._text_bufs["max_width_str"] == "1,024"


def test_pending_settings_callback_does_not_cross_unmount(
    ready_panel_settings, training_panel_module, monkeypatch
):
    panel, _, _, _, change, callbacks = ready_panel_settings
    scheduled = []
    monkeypatch.setattr(training_panel_module.lf.ui, "schedule_on_ui_thread", scheduled.append)
    stale_callback = next(callback for path, callback in callbacks.values()
                          if path == "optimization.iterations")
    change("optimization", "iterations", 600)
    assert len(scheduled) == 1
    old_handle = panel._handle
    panel.on_unmount(SimpleNamespace(remove_data_model=lambda _name: None))
    scheduled[0]()
    stale_callback(600, 700)
    assert old_handle.request_update_count == 0
    assert not panel._params_refresh_pending
    panel._handle = _HandleStub()
    panel._subscribe_reactive_state()
    scheduled[0]()
    stale_callback(700, 800)
    assert panel._handle.request_update_count == 0
    assert not panel._params_refresh_pending
    change("optimization", "iterations", 900)
    assert len(scheduled) == 2
    scheduled[1]()
    assert panel._handle.request_update_count == 1


@pytest.mark.parametrize("prop,value,expected", [("iterations", 40000, "35,000"), ("steps_scaler", 2.0, "20,000")])
def test_external_change_refreshes_derived_activation_step(ready_panel_settings, prop, value, expected):
    panel, params, _, _, change, _ = ready_panel_settings
    params.ppisp_controller_activation_step = -1
    change("optimization", prop, value)
    panel.on_update(None)
    assert panel._text_bufs["ppisp_activation_step_str"] == expected


@pytest.mark.parametrize("locked", [False, True])
def test_label_reset_respects_coupled_step_lock(training_panel_module, monkeypatch, locked):
    module = training_panel_module
    panel = module.TrainingPanel()
    panel._auto_scale_steps_locked = locked
    monkeypatch.setattr(panel, "_can_edit_configuration", lambda: True)
    params = SimpleNamespace(has_params=lambda: True, strategy="mrnf",
        prop_info=lambda prop: {"readonly": False}, builtin_default=lambda prop: 100)
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)
    prop = next(run.prop_ids[0] for run in module.property_view.RUNS
                if run.disabled_condition_id == "step_scaling_params_locked")
    assert panel._reset_default(prop) == (None if locked else 100)


@pytest.mark.parametrize("state,iteration,editable", [
    ("ready", 0, True),
    ("ready", 100, False),
    ("running", 100, False),
    ("paused", 100, False),
    ("completed", 100, True),
    ("stopped", 100, True),
    ("error", 100, True),
])
def test_configuration_editability_follows_run_state(
    training_panel_module, monkeypatch, state, iteration, editable
):
    module = training_panel_module
    monkeypatch.setattr(module, "RuntimeState", SimpleNamespace(
        trainer_state=_make_signal(state), iteration=_make_signal(iteration)))
    params = SimpleNamespace(has_params=lambda: True, strategy="mrnf",
        prop_info=lambda prop: {"readonly": False}, builtin_default=lambda prop: 100)
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)
    panel = module.TrainingPanel()
    model = _ModelStub()
    panel._bind_disabled(model, lambda: params)

    assert panel._can_edit_configuration() is editable
    for binding in ("struct_disabled", "live_disabled", "adv_disabled"):
        assert model.bindings[binding][0]() is (not editable)
    assert panel._reset_default("means_lr") == (100 if editable else None)


def test_label_reset_rejects_training_runtime_and_forced_method_fields(training_panel_module, monkeypatch):
    module = training_panel_module
    panel = module.TrainingPanel()
    params = SimpleNamespace(has_params=lambda: True, strategy="igs+",
        prop_info=lambda prop: {"readonly": False}, builtin_default=lambda prop: False)
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)
    monkeypatch.setattr(panel, "_can_edit_configuration", lambda: False)
    monkeypatch.setattr(module, "_trainer_run_is_finished", lambda: False)
    assert panel._reset_default("iterations") is None
    monkeypatch.setattr(panel, "_can_edit_configuration", lambda: True)
    assert panel._reset_default("gut") is None
    assert panel._reset_default("strategy") is None


def test_label_reset_uses_iterations_scaling_path(training_panel_module, monkeypatch):
    module = training_panel_module
    panel = module.TrainingPanel()
    panel._auto_scale_steps_locked = True
    calls = []
    params = SimpleNamespace(has_params=lambda: True, strategy="mrnf", iterations=60000, steps_scaler=2.,
        apply_step_scaling=lambda value: calls.append(value))
    params.get = lambda prop: getattr(params, prop)
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)
    monkeypatch.setattr(panel, "_reset_default", lambda prop: 30000)
    monkeypatch.setattr(panel, "_sync_text_bufs", lambda: None)
    assert panel._apply_reset("iterations", 30000)
    assert calls == [1.0] and params.iterations == 30000


def test_distorted_images_shows_prompt_and_handles_undistort(training_panel_module, monkeypatch):
    module = training_panel_module
    panel = module.TrainingPanel()
    panel._handle = _HandleStub()

    params = _ParamsStub()
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)

    scene = SimpleNamespace(
        has_active_distorted_cameras=lambda: True,
        can_undistort_active_cameras=lambda: True,
    )
    monkeypatch.setattr(module.lf, "get_scene", lambda: scene)

    started = []
    monkeypatch.setattr(module.lf, "start_training", lambda: started.append(True))

    dialog_opened = []
    def mock_confirm_dialog(title, message, buttons, callback, style="info"):
        dialog_opened.append((title, message, buttons, callback, style))
    monkeypatch.setattr(module.lf.ui, "confirm_dialog", mock_confirm_dialog)

    panel._check_distortion_and_start()

    assert len(dialog_opened) == 1
    title, message, buttons, callback, style = dialog_opened[0]
    assert style == "warning"
    assert len(buttons) == 3
    btn_undistort, btn_gut, btn_cancel = buttons

    assert len(started) == 0
    callback(btn_undistort)
    assert params.undistort is True
    assert panel._handle.dirty_all_count == 1
    assert len(started) == 1


def test_distorted_images_shows_prompt_and_handles_gut(training_panel_module, monkeypatch):
    module = training_panel_module
    panel = module.TrainingPanel()
    panel._handle = _HandleStub()

    params = _ParamsStub()
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)

    scene = SimpleNamespace(
        has_active_distorted_cameras=lambda: True,
        can_undistort_active_cameras=lambda: True,
    )
    monkeypatch.setattr(module.lf, "get_scene", lambda: scene)

    started = []
    monkeypatch.setattr(module.lf, "start_training", lambda: started.append(True))

    dialog_opened = []
    def mock_confirm_dialog(title, message, buttons, callback, style="info"):
        dialog_opened.append((title, message, buttons, callback, style))
    monkeypatch.setattr(module.lf.ui, "confirm_dialog", mock_confirm_dialog)

    panel._check_distortion_and_start()

    assert len(dialog_opened) == 1
    _, _, buttons, callback, _ = dialog_opened[0]
    btn_undistort, btn_gut, btn_cancel = buttons

    callback(btn_gut)
    assert params.gut is True
    assert panel._handle.dirty_all_count == 1
    assert len(started) == 1


def test_distorted_images_shows_prompt_and_handles_cancel(training_panel_module, monkeypatch):
    module = training_panel_module
    panel = module.TrainingPanel()
    panel._handle = _HandleStub()

    params = _ParamsStub()
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)

    scene = SimpleNamespace(
        has_active_distorted_cameras=lambda: True,
        can_undistort_active_cameras=lambda: True,
    )
    monkeypatch.setattr(module.lf, "get_scene", lambda: scene)

    started = []
    monkeypatch.setattr(module.lf, "start_training", lambda: started.append(True))

    dialog_opened = []
    def mock_confirm_dialog(title, message, buttons, callback, style="info"):
        dialog_opened.append((title, message, buttons, callback, style))
    monkeypatch.setattr(module.lf.ui, "confirm_dialog", mock_confirm_dialog)

    panel._check_distortion_and_start()

    assert len(dialog_opened) == 1
    _, _, buttons, callback, _ = dialog_opened[0]
    btn_undistort, btn_gut, btn_cancel = buttons

    callback(btn_cancel)
    assert params.undistort is False
    assert params.gut is False
    assert len(started) == 0


def test_distorted_images_equirectangular_offers_only_gut(training_panel_module, monkeypatch):
    module = training_panel_module
    panel = module.TrainingPanel()
    panel._handle = _HandleStub()

    params = _ParamsStub()
    params.strategy = "mrnf"
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)

    scene = SimpleNamespace(
        has_active_distorted_cameras=lambda: True,
        can_undistort_active_cameras=lambda: False,
    )
    monkeypatch.setattr(module.lf, "get_scene", lambda: scene)

    started = []
    monkeypatch.setattr(module.lf, "start_training", lambda: started.append(True))

    dialog_opened = []
    def mock_confirm_dialog(title, message, buttons, callback, style="info"):
        dialog_opened.append((title, message, buttons, callback, style))
    monkeypatch.setattr(module.lf.ui, "confirm_dialog", mock_confirm_dialog)

    panel._check_distortion_and_start()

    assert len(dialog_opened) == 1
    _, _, buttons, callback, _ = dialog_opened[0]
    assert len(buttons) == 2
    btn_gut, btn_cancel = buttons

    callback(btn_gut)
    assert params.gut is True
    assert panel._handle.dirty_all_count == 1
    assert len(started) == 1


def test_distorted_images_igs_strategy_offers_only_undistort(training_panel_module, monkeypatch):
    module = training_panel_module
    panel = module.TrainingPanel()
    panel._handle = _HandleStub()

    params = _ParamsStub()
    params.strategy = "igs+"
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)

    scene = SimpleNamespace(
        has_active_distorted_cameras=lambda: True,
        can_undistort_active_cameras=lambda: True,
    )
    monkeypatch.setattr(module.lf, "get_scene", lambda: scene)

    started = []
    monkeypatch.setattr(module.lf, "start_training", lambda: started.append(True))

    dialog_opened = []
    def mock_confirm_dialog(title, message, buttons, callback, style="info"):
        dialog_opened.append((title, message, buttons, callback, style))
    monkeypatch.setattr(module.lf.ui, "confirm_dialog", mock_confirm_dialog)

    panel._check_distortion_and_start()

    assert len(dialog_opened) == 1
    _, _, buttons, callback, _ = dialog_opened[0]
    assert len(buttons) == 2
    btn_undistort, btn_cancel = buttons

    callback(btn_undistort)
    assert params.undistort is True
    assert panel._handle.dirty_all_count == 1
    assert len(started) == 1


def test_distorted_images_skipped_if_already_undistorted_or_gut(training_panel_module, monkeypatch):
    module = training_panel_module
    panel = module.TrainingPanel()
    panel._handle = _HandleStub()

    scene = SimpleNamespace(
        has_active_distorted_cameras=lambda: True,
        can_undistort_active_cameras=lambda: True,
    )
    monkeypatch.setattr(module.lf, "get_scene", lambda: scene)

    dialog_opened = []
    monkeypatch.setattr(module.lf.ui, "confirm_dialog", lambda *args, **kwargs: dialog_opened.append(args))

    # Case 1: undistort is True
    started = []
    monkeypatch.setattr(module.lf, "start_training", lambda: started.append(True))
    params = _ParamsStub()
    params.undistort = True
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)
    panel._check_distortion_and_start()
    assert len(dialog_opened) == 0
    assert len(started) == 1

    # Case 2: gut is True
    started.clear()
    params = _ParamsStub()
    params.gut = True
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)
    panel._check_distortion_and_start()
    assert len(dialog_opened) == 0
    assert len(started) == 1

    # Case 3: no distorted cameras
    started.clear()
    scene.has_active_distorted_cameras = lambda: False
    params = _ParamsStub()
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)
    panel._check_distortion_and_start()
    assert len(dialog_opened) == 0
    assert len(started) == 1

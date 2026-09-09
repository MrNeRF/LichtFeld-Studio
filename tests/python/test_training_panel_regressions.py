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
    lf_stub.training_backends = lambda: [
        {"id": "3dgs", "label": "3DGS"}, {"id": "3dgut", "label": "3DGUT"},
    ]
    lf_stub.dataset_params = lambda: None
    lf_stub.get_scene = lambda: None
    lf_stub.start_training = lambda: None
    lf_stub.trainer_saving_model = lambda: False
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


def test_start_feedback_tracks_actual_conflicts_without_mutating(training_panel_module, monkeypatch):
    module = training_panel_module
    panel = module.TrainingPanel()
    params = SimpleNamespace(
        has_params=lambda: True, validate=lambda: "invalid combination",
        raster_backend="3dgut", strategy="igs+", undistort=True, mip_filter=False,
        use_depth_loss=True, use_normal_loss=False,
        backend_capabilities=dict.fromkeys(
            ("igs_plus", "mip_filter", "depth_supervision", "normal_supervision"), "unsupported"),
    )
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)
    monkeypatch.setattr(module.RuntimeState.trainer_state, "value", "ready")
    params.backend_capabilities["undistort"] = "supported"
    dirty = []
    panel._handle = SimpleNamespace(dirty=dirty.append)
    before = vars(params).copy()
    assert panel._start_error() == "invalid combination"
    notice = panel._backend_notice(selected_only=True)
    assert notice.startswith("Not available with 3DGUT: ")
    assert "IGS+" in notice and "use_depth_loss" in notice
    assert "undistort" not in notice
    assert "mip_filter" not in notice and "use_normal_loss" not in notice
    assert panel._sync_start_feedback()
    assert set(dirty) == {"start_error", "start_blocked", "start_conflicts"}
    assert not panel._sync_start_feedback()
    assert vars(params) == before
    params.validate = lambda: ""
    assert panel._sync_start_feedback()
    assert panel._start_error() == ""
    params.backend_capabilities = dict.fromkeys(params.backend_capabilities, "supported")
    assert panel._backend_notice() == ""


def test_backend_notice_uses_descriptor_label_and_localized_template(training_panel_module, monkeypatch):
    module = training_panel_module
    params = SimpleNamespace(
        has_params=lambda: True, raster_backend="test_backend",
        mip_filter=False, backend_capabilities={"mip_filter": "unsupported"},
    )
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)
    monkeypatch.setattr(module.lf, "training_backends", lambda: [
        {"id": "test_backend", "label": "Test Renderer"},
    ])
    translations = {
        "training.backend_unsupported": "{backend} unavailable: {features}",
        "training_params.mip_filter": "Mip Filter:",
    }
    monkeypatch.setattr(module.lf.ui, "tr", lambda key: translations.get(key, key))
    panel = module.TrainingPanel()
    assert panel._backend_notice() == "Test Renderer unavailable: Mip Filter"
    assert panel._backend_notice(selected_only=True) == ""
    params.mip_filter = True
    assert panel._backend_notice(selected_only=True) == "Test Renderer unavailable: Mip Filter"
    params.backend_capabilities["mip_filter"] = "supported"
    assert panel._backend_notice() == ""


def test_backend_notice_locale_placeholders():
    from string import Formatter

    locales = Path(__file__).resolve().parents[2] / "src/visualizer/gui/resources/locales"
    for path in locales.glob("*.json"):
        catalog = json.loads(path.read_text(encoding="utf-8"))
        if "training" not in catalog:
            continue
        template = catalog["training"]["backend_unsupported"]
        fields = {field for _, field, _, _ in Formatter().parse(template) if field is not None}
        assert fields == {"backend", "features"}, path.name


@pytest.mark.parametrize("state", ["paused", "running", "idle", "completed", "error"])
def test_next_run_validation_does_not_gate_resume(training_panel_module, monkeypatch, state):
    panel = training_panel_module.TrainingPanel()
    monkeypatch.setattr(training_panel_module.RuntimeState.trainer_state, "value", state)
    monkeypatch.setattr(panel, "_validation_error", lambda: pytest.fail("Not a new Start"))
    assert panel._start_error() == ""


def test_invalid_start_is_rejected_before_overwrite_consent(training_panel_module, monkeypatch):
    module = training_panel_module
    panel = module.TrainingPanel()
    monkeypatch.setattr(module, "_restore_stored_session_if_needed", lambda **_kw: False)
    monkeypatch.setattr(module, "_training_session_state", lambda: {})
    monkeypatch.setattr(panel, "_validation_error", lambda: "invalid numeric parameter")
    monkeypatch.setattr(module.lf, "training_start_overwrite_conflict",
                        lambda: pytest.fail("Must reject before consent"))
    errors = []
    monkeypatch.setattr(module.lf.ui, "message_dialog", lambda *args, **kwargs: errors.append(args), raising=False)
    panel._action_start()
    assert errors[0][1] == "invalid numeric parameter"


def test_only_start_is_disabled_and_feedback_is_outside_search():
    from xml.etree import ElementTree as ET
    root = ET.parse(Path(__file__).parents[2] / "src/visualizer/gui/rmlui/resources/training.rml")
    controls = root.find(".//*[@id='controls']")
    buttons = {node.get("data-event-click"): node for node in controls.iter("button")}
    assert buttons["action('start')"].get("data-attrif-disabled") == "start_blocked"
    assert buttons["action('resume')"].get("data-attrif-disabled") is None
    feedback = controls.find(".//*[@data-if='start_blocked']")
    assert "{{start_error}}" in "".join(feedback.itertext())
    assert "{{start_conflicts}}" in "".join(feedback.itertext())


@pytest.mark.parametrize("state,iteration,actions", [
    ("ready", 0, ["start", "clear"]),
    ("ready", 12, ["start", "reset", "clear"]),
    ("starting", 0, ["pause", "stop"]),
    ("running", 12, ["pause", "save_project"]),
    ("paused", 12, ["resume", "reset", "stop", "save_project"]),
    ("completed", 12, ["switch_edit", "reset", "clear"]),
    ("stopped", 12, ["switch_edit", "reset", "clear"]),
    ("error", 12, ["reset", "clear"]),
    ("stopping", 12, []),
])
def test_compact_toolbar_preserves_visible_actions_by_state(
    training_panel_module, monkeypatch, state, iteration, actions
):
    from xml.etree import ElementTree as ET
    module = training_panel_module
    monkeypatch.setattr(module, "_training_session_state", lambda: {})
    monkeypatch.setattr(module.RuntimeState.trainer_state, "value", state)
    monkeypatch.setattr(module.RuntimeState.iteration, "value", iteration)
    panel = module.TrainingPanel()
    model = _ModelStub()
    panel._bind_visibility(model, lambda: None, lambda: None)
    root = ET.parse(Path(__file__).parents[2] / "src/visualizer/gui/rmlui/resources/training.rml")
    controls = root.find(".//*[@id='controls']")
    visible = []

    def visit(node):
        condition = node.get("data-if")
        # Error text is independent of the action visibility matrix.
        if condition == "start_blocked":
            return
        if condition and not model.bindings[condition][0]():
            return
        if node.tag == "button":
            visible.append(node.get("data-event-click"))
        for child in node:
            visit(child)

    visit(controls)
    assert visible == [f"action('{action}')" for action in actions]


def test_compact_toolbar_icons_labels_and_tooltips_are_retained():
    from xml.etree import ElementTree as ET
    project = Path(__file__).parents[2]
    root = ET.parse(project / "src/visualizer/gui/rmlui/resources/training.rml")
    controls = root.find(".//*[@id='controls']")
    for button in controls.iter("button"):
        classes = button.get("class", "").split()
        assert "btn--full" not in classes
        assert button.get("data-tooltip")
        assert "training-toolbar-action" in classes
        assert button.find("img") is not None
        assert "{{" in "".join(button.itertext())
        for icon in button.iter("img"):
            path = icon.get("src").removeprefix("../")
            assert (project / "src/visualizer/gui/assets" / path).is_file()
    assert controls.find(".//*[@data-if='show_project_saved']") is not None
    assert controls.find(".//*[@data-if='show_ctrl_error']") is not None


def test_toolbar_status_shares_nonwrapping_row_with_actions_outside_telemetry():
    from xml.etree import ElementTree as ET
    project = Path(__file__).parents[2]
    root = ET.parse(project / "src/visualizer/gui/rmlui/resources/training.rml")
    controls = root.find(".//*[@id='controls']")
    toolbar = controls.find("div[@id='training-toolbar']")
    assert toolbar.get("class") == "training-toolbar"
    header = toolbar.find("div[@id='training-controls-header']")
    assert header is not None and header.get("data-if") is None
    assert toolbar[-1] is header
    badges = {badge.get("data-if"): badge for badge in header.findall("span")}
    ready_label = badges["show_ctrl_ready"].find("span[@class='training-status-badge-label']")
    assert ready_label.text == "@tr:status.ready"
    assert all(
        badge.find("span[@class='training-status-badge-label']") is not None
        for badge in badges.values()
    )
    assert "is-ready" in badges["show_ctrl_ready"].get("class")
    assert "is-active" in badges["show_ctrl_running"].get("class")
    assert "is-paused" in badges["show_ctrl_paused"].get("class")
    assert "is-complete" in badges["show_ctrl_completed"].get("class")
    assert "is-error" in badges["show_ctrl_error"].get("class")
    assert set(badges) == {
        "show_ctrl_ready", "show_ctrl_starting", "show_ctrl_running",
        "show_ctrl_paused", "show_ctrl_completed", "show_ctrl_stopped",
        "show_ctrl_error", "show_ctrl_stopping", "show_ctrl_restoring", "show_ctrl_saving",
    }
    assert sum(node.text == "{{status_mode}}" for node in root.iter()) == 0
    for group in toolbar.findall("div"):
        for row in group.findall("div[@class='training-action-row']"):
            assert all(child.tag == "button" for child in row)
    css = (project / "src/visualizer/gui/rmlui/resources/training.rcss").read_text()
    for selector in ("#controls", ".training-actions-group", ".training-action-row"):
        block = css.split(selector + " {", 1)[1].split("}", 1)[0]
        assert "width: 100%;" in block
    row = css.split(".training-action-row {", 1)[1].split("}", 1)[0]
    assert "display: flex;" in row
    assert "flex-wrap: nowrap;" in row
    button = css.split(".training-toolbar-action {", 1)[1].split("}", 1)[0]
    assert "display: inline-flex;" in button
    assert "flex: 0 0 auto;" in button
    assert "overflow: hidden" not in button
    assert "\n    width: 100%;" not in button


@pytest.mark.parametrize("iteration,key", [(0, "training.action_start"), (12, "training_panel.resume")])
def test_toolbar_uses_short_primary_labels(training_panel_module, monkeypatch, iteration, key):
    module = training_panel_module
    monkeypatch.setattr(module, "_training_session_state", lambda: {})
    monkeypatch.setattr(module.RuntimeState.iteration, "value", iteration)
    model = _ModelStub()
    module.TrainingPanel()._bind_labels(model)
    assert model.bindings["btn_start"][0]() == key
    assert model.bindings["label_toolbar_edit"][0]() == "common.edit"


def test_toolbar_starting_status_is_not_unknown(training_panel_module, monkeypatch):
    module = training_panel_module
    monkeypatch.setattr(module, "_training_session_state", lambda: {})
    monkeypatch.setattr(module.RuntimeState.trainer_state, "value", "starting")
    model = _ModelStub()
    module.TrainingPanel()._bind_status(model, lambda: None)
    assert "runtime.task_starting" in model.bindings["status_mode"][0]()


def test_restore_failure_keeps_detail_below_error_badge(training_panel_module, monkeypatch):
    module = training_panel_module
    monkeypatch.setattr(module, "_training_session_state", lambda: {"error": "bad checkpoint"})
    monkeypatch.setattr(module.RuntimeState.has_trainer, "value", False)
    model = _ModelStub()
    panel = module.TrainingPanel()
    panel._bind_status(model, lambda: None)
    panel._bind_visibility(model, lambda: None, lambda: None)
    assert "status.error" in model.bindings["status_mode"][0]()
    assert "bad checkpoint" in model.bindings["error_message"][0]()
    assert model.bindings["show_ctrl_error"][0]()
    assert not model.bindings["show_ctrl_paused"][0]()
    monkeypatch.setattr(module.RuntimeState.has_trainer, "value", True)
    monkeypatch.setattr(module.RuntimeState.trainer_state, "value", "error")
    monkeypatch.setattr(module.lf, "trainer_error", lambda: "new training error", raising=False)
    assert model.bindings["error_message"][0]() == "new training error"
    monkeypatch.setattr(module, "_training_session_state", lambda: {"restoring": True})
    assert model.bindings["show_ctrl_restoring"][0]()
    assert not model.bindings["show_ctrl_error"][0]()


@pytest.mark.parametrize("scale", [1.0, 1.5, 2.0])
@pytest.mark.parametrize("state,actions", [
    ("ready", ("start", "clear")),
    ("completed", ("switch_edit", "reset", "clear")),
    ("paused", ("resume", "reset", "stop", "save_project")),
])
def test_toolbar_fit_uses_measured_width_and_can_restore_captions(training_panel_module, monkeypatch, scale, state, actions):
    module = training_panel_module
    panel = module.TrainingPanel()
    scheduled = []
    monkeypatch.setattr(panel, "_schedule_deferred_update", scheduled.append)
    monkeypatch.setattr(module, "_training_session_state", lambda: {})
    monkeypatch.setattr(module.RuntimeState.trainer_state, "value", state)
    monkeypatch.setattr(module.RuntimeState.has_trainer, "value", True)
    monkeypatch.setattr(module.RuntimeState.iteration, "value", 0)
    classes = set()
    toolbar = SimpleNamespace(
        client_width=1000 * scale,
        is_class_set=lambda name: name in classes,
        set_class=lambda name, enabled: classes.add(name) if enabled else classes.discard(name),
    )
    elements = {
        "training-toolbar": toolbar,
        "training-controls-header": SimpleNamespace(absolute_width=80 * scale),
        "measure-action-gap": SimpleNamespace(absolute_width=4 * scale),
        "measure-status-gap": SimpleNamespace(absolute_width=6 * scale),
    }
    elements.update({"measure-" + action: SimpleNamespace(absolute_width=90 * scale) for action in actions})
    panel._doc = SimpleNamespace(get_element_by_id=elements.get)
    required = (90 * len(actions) + 4 * (len(actions) - 1) + 80 + 6) * scale
    toolbar.client_width = required - 1
    assert panel._sync_toolbar_fit()
    assert "is-compact" in classes
    assert scheduled == [0.01]
    assert not panel._sync_toolbar_fit()
    assert scheduled == [0.01]
    toolbar.client_width = required + 1
    assert panel._sync_toolbar_fit()
    assert "is-compact" not in classes
    elements["measure-" + actions[0]].absolute_width += 20 * scale
    assert panel._sync_toolbar_fit()
    assert "is-compact" in classes


def test_sparsity_is_a_collapsible_advanced_group():
    from xml.etree import ElementTree as ET
    root = ET.parse(Path(__file__).parents[2] / "src/visualizer/gui/rmlui/resources/training.rml")
    advanced = root.find(".//*[@id='sec-advanced-params']")
    header = advanced.find(".//*[@id='hdr-sparsity']")
    assert header.get("data-event-click") == "toggle_section('sparsity')"
    assert header.find(".//*[@id='arrow-sparsity']") is not None
    content = advanced.find(".//*[@id='sec-sparsity']")
    assert "collapsed" in content.get("class")
    assert content.find(".//*[@data-for='row : pv_basic_sparsity_toggle_rows']") is None
    assert advanced.find(".//*[@id='advanced-feature-activations']//*[@data-for='row : pv_basic_sparsity_toggle_rows']") is not None
    assert content.find(".//*[@data-if='dep_sparsity']") is not None
    assert content.find(".//*[@data-for='row : pv_sparsity_rows']") is not None
    assert content.find(".//*[@id='sec-save-steps']") is None


def test_advanced_has_single_real_activation_for_each_optional_feature():
    from xml.etree import ElementTree as ET
    root = ET.parse(Path(__file__).parents[2] / "src/visualizer/gui/rmlui/resources/training.rml")
    activations = root.find(".//*[@id='advanced-feature-activations']")
    assert activations.get("data-class-disabled-overlay") == "live_disabled"
    for run in ("basic_depth_toggle", "basic_normal_toggle", "basic_bilateral_toggle",
                "basic_ppisp_toggle", "basic_sparsity_toggle", "dataset_eval", "feature_random"):
        rows = root.findall(f".//*[@data-for='row : pv_{run}_rows']")
        assert len(rows) == 1
        assert activations.find(f".//*[@data-for='row : pv_{run}_rows']") is rows[0]
        checkbox = rows[0].find("input")
        assert checkbox.get("data-checked") == "row.checked"
        assert checkbox.get("data-event-click") == "pv_value_change(row.id, !row.checked)"
    advanced = root.find(".//*[@id='sec-advanced-params']")
    for section in ("depth", "normal", "background", "ppisp", "bilateral", "evaluation", "random-init", "dataset", "sparsity", "optimization"):
        assert advanced.find(f".//*[@id='sec-{section}']") is not None
    assert root.find(".//*[@id='hdr-advanced-params']").get("data-event-click") == "toggle_section('advanced_params')"


def test_save_project_is_in_the_same_row_as_pause_and_resume():
    from xml.etree import ElementTree as ET
    root = ET.parse(Path(__file__).parents[2] / "src/visualizer/gui/rmlui/resources/training.rml")
    for state in ("running", "paused"):
        group = root.find(
            f".//div[@data-if='show_ctrl_{state}']"
        )
        row = group.find("div[@class='training-action-row']")
        assert row.find("button[@data-if='show_project_save']") is not None
    controls = root.find(".//*[@id='controls']")
    assert controls.find("button[@data-if='show_project_save']") is None


def test_bundled_locales_define_training_panel_strategy_and_color_keys():
    project_root = Path(__file__).parent.parent.parent
    locale_dir = project_root / "src" / "visualizer" / "gui" / "resources" / "locales"

    for locale_path in locale_dir.glob("*.json"):
        data = json.loads(locale_path.read_text())
        assert data["training"]["options.strategy.igs_plus"] == "IGS+"
        assert data["training"]["start_fix_settings"]
        assert data["training"]["action_start"]
        assert data["training"]["action_stop"]
        assert data["training"]["action_save"]
        assert data["training"]["status_restoring"]
        assert data["training"]["status_saving"]
        assert "refinement.grow_until_iter" in data["training"]
        assert "tooltip.grow_until_iter" in data["training"]
        assert data["training"]["overwrite.btn_save_as_start"]
        assert data["training"]["save_pc.message_project"]
        conflict_template = data["training"]["backend_conflict.message"]
        assert "{backend}" in conflict_template
        assert "{feature}" in conflict_template
        assert "{fallback_backend}" in conflict_template
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


@pytest.mark.parametrize(
    ("conflict", "feature"),
    [
        ("igs_plus", "IGS+"),
        ("mip_filter", "Mip Filter"),
        ("depth_supervision", "Depth Loss"),
        ("normal_supervision", "Normal Loss"),
    ],
)
def test_backend_conflict_messages_share_one_localized_template(
    training_panel_module, monkeypatch, conflict, feature
):
    translations = {
        "training.backend_conflict.message": (
            "{backend} conflicts with {feature}; choose {fallback_backend}."
        ),
        "training.options.strategy.igs_plus": "IGS+",
        "training_params.mip_filter": "Mip Filter:",
        "training_params.use_depth_loss": "Depth Loss:",
        "training_params.use_normal_loss": "Normal Loss:",
    }
    monkeypatch.setattr(
        training_panel_module, "tr", lambda key: translations.get(key, key)
    )

    assert training_panel_module._localized_backend_conflict_message(conflict) == (
        f"3DGUT conflicts with {feature}; choose 3DGS."
    )


def test_backend_conflict_template_accepts_descriptor_context_for_future_backends(
    training_panel_module, monkeypatch
):
    monkeypatch.setattr(
        training_panel_module,
        "tr",
        lambda key: (
            "{backend} conflicts with {feature}; choose {fallback_backend}."
            if key == "training.backend_conflict.message"
            else key
        ),
    )

    assert training_panel_module._localized_backend_conflict_message(
        "temporal_filter",
        backend="FutureGS",
        feature_fallback="Temporal Filter",
        fallback_backend="3DGS",
    ) == "FutureGS conflicts with Temporal Filter; choose 3DGS."


@pytest.mark.parametrize(
    ("error", "conflict", "conflict_message", "expected_title", "expected_message"),
    [
        (
            "3DGUT cannot be used with Depth Loss. Change this setting or select 3DGS.",
            "depth_supervision",
            "3DGUT cannot be used with Depth Loss. Change this setting or select 3DGS.",
            "training.error.strategy_gut_title",
            "training.backend_conflict.message",
        ),
        (
            "iterations must be within [1, 2147483647]",
            "depth_supervision",
            "3DGUT cannot be used with Depth Loss. Change this setting or select 3DGS.",
            "status.error",
            "iterations must be within [1, 2147483647]",
        ),
        (
            "exposure correction replaces the standalone bilateral grid and PPISP options",
            "",
            "",
            "status.error",
            "exposure correction replaces the standalone bilateral grid and PPISP options",
        ),
    ],
)
def test_start_validation_localizes_typed_backend_conflicts_without_masking_other_errors(
    training_panel_module,
    monkeypatch,
    error,
    conflict,
    conflict_message,
    expected_title,
    expected_message,
):
    panel = training_panel_module.TrainingPanel()
    params = SimpleNamespace(
        has_params=lambda: True,
        validate=lambda: error,
        backend_conflict=conflict,
        backend_conflict_message=conflict_message,
        gut=True,
    )
    starts, dialogs, offers = [], [], []
    monkeypatch.setattr(training_panel_module.lf, "optimization_params", lambda: params)
    monkeypatch.setattr(training_panel_module.lf, "start_training", lambda: starts.append(True))
    monkeypatch.setattr(panel, "_should_offer_pc_save", lambda: offers.append(True))
    monkeypatch.setattr(
        training_panel_module.lf.ui,
        "message_dialog",
        lambda *args, **kwargs: dialogs.append((args, kwargs)),
        raising=False,
    )

    panel._start_after_consent()

    assert len(dialogs) == 1
    (title, message), kwargs = dialogs[0]
    assert title == expected_title
    assert message == expected_message
    assert kwargs == {"style": "error"}
    assert starts == []
    assert offers == []
    assert params.gut is True


@pytest.mark.parametrize(
    ("button", "expected_strategy", "expected_gut", "expected_viewer_update"),
    [
        ("training.conflict.btn_use_mcmc", "mcmc", True, []),
        (
            "training.conflict.btn_disable_gut",
            "igs+",
            False,
            [("raster_backend", "3dgs")],
        ),
    ],
)
def test_start_igs_gut_conflict_keeps_one_click_resolution(
    training_panel_module,
    monkeypatch,
    button,
    expected_strategy,
    expected_gut,
    expected_viewer_update,
):
    class Params:
        strategy = "igs+"
        gut = True

        def has_params(self):
            return True

        @property
        def backend_conflict(self):
            return "igs_plus" if self.strategy == "igs+" and self.gut else ""

        @property
        def backend_conflict_message(self):
            return (
                "3DGUT cannot be used with IGS+. Change this setting or select 3DGS."
                if self.backend_conflict
                else ""
            )

        def validate(self):
            return self.backend_conflict_message

        def set_strategy(self, strategy):
            self.departed_preset_gut = self.gut
            self.strategy = strategy

    panel = training_panel_module.TrainingPanel()
    params = Params()
    dialogs, starts, updates = [], [], []
    monkeypatch.setattr(training_panel_module.lf, "optimization_params", lambda: params)
    monkeypatch.setattr(training_panel_module.lf, "start_training", lambda: starts.append(True))
    monkeypatch.setattr(panel, "_should_offer_pc_save", lambda: False)
    monkeypatch.setattr(panel, "_refresh_strategy_values", lambda: None)
    monkeypatch.setattr(
        training_panel_module.lf,
        "get_render_settings",
        lambda: SimpleNamespace(set=lambda *args: updates.append(args)),
    )
    monkeypatch.setattr(
        training_panel_module.lf.ui,
        "confirm_dialog",
        lambda *args: dialogs.append(args),
        raising=False,
    )

    panel._start_after_consent()
    assert len(dialogs) == 1
    assert dialogs[0][1] == "training.backend_conflict.message"
    dialogs[0][3](button)

    assert params.strategy == expected_strategy
    assert params.gut is expected_gut
    assert updates == expected_viewer_update
    assert starts == [True]
    if expected_strategy == "mcmc":
        assert params.departed_preset_gut is False


def test_native_backend_rollback_republishes_bound_controls(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    params = SimpleNamespace(
        has_params=lambda: True, strategy="mcmc", gut=True,
        mip_filter=True, use_depth_loss=False, use_normal_loss=False,
    )
    published, dirty = [], []
    panel._pv_bindings = [SimpleNamespace(publish=lambda: published.append(True), sync_text_bufs=lambda **_: False)]
    panel._handle = SimpleNamespace(dirty_all=lambda: dirty.append(True))
    monkeypatch.setattr(panel, "_sync_text_bufs", lambda **_kwargs: None)
    monkeypatch.setattr(training_panel_module.lf, "optimization_params", lambda: params)
    assert panel._refresh_native_backend_controls()
    assert not panel._refresh_native_backend_controls()
    params.mip_filter = False
    assert panel._refresh_native_backend_controls()
    assert len(published) == 2
    assert len(dirty) == 2


def test_native_rollback_between_publication_and_update_is_not_missed(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    params = SimpleNamespace(
        has_params=lambda: True, strategy="mcmc", gut=True,
        mip_filter=False, use_depth_loss=False, use_normal_loss=False,
    )
    published = []
    binding = SimpleNamespace(publish=lambda: published.append(params.mip_filter), sync_text_bufs=lambda **_: False)
    panel._pv_bindings = [binding]
    panel._handle = SimpleNamespace(dirty_all=lambda: None)
    monkeypatch.setattr(panel, "_sync_text_bufs", lambda **_kwargs: None)
    monkeypatch.setattr(panel, "_dirty_property_search_models", lambda: None)
    monkeypatch.setattr(panel, "_sync_section_states", lambda: None)
    monkeypatch.setattr(training_panel_module.lf, "optimization_params", lambda: params)
    assert panel._refresh_native_backend_controls()
    params.mip_filter = True
    panel._pv_publish_pending = [binding]
    assert panel._flush_pv_publish()
    params.mip_filter = False
    assert panel._refresh_native_backend_controls()
    assert published == [False, True, False]


@pytest.mark.parametrize("rollback_timing", ["before_refresh", "after_refresh"])
@pytest.mark.parametrize("prop,initial,draft,is_int", [
    ("max_cap", 5_000_000, "4000000", True),
    ("iterations", 30_000, "20000", True),
    ("means_lr", 0.00016, "0.00012", False),
])
def test_numeric_draft_survives_refresh_and_publication_settles(
    training_panel_module, monkeypatch, prop, initial, draft, is_int, rollback_timing
):
    module = training_panel_module
    panel = module.TrainingPanel()
    params = SimpleNamespace(
        has_params=lambda: True, strategy="mrnf", gut=False, mip_filter=False,
        use_depth_loss=False, use_normal_loss=False, ppisp_controller_activation_step=0,
        bg_color=(0.0, 0.0, 0.0),
    )
    setattr(params, prop, initial)
    params.set = lambda name, value: setattr(params, name, value)
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)
    monkeypatch.setattr(module.lf, "dataset_params", lambda: None)
    scheduled = []
    monkeypatch.setattr(module.lf.ui, "schedule_on_ui_thread", scheduled.append, raising=False)
    monkeypatch.setattr(panel, "_dirty_property_search_models", lambda: None)
    monkeypatch.setattr(panel, "_sync_section_states", lambda: None)
    records = {}
    panel._handle = SimpleNamespace(
        dirty_all=lambda: None,
        update_record_list=lambda name, values: records.update({name: values}),
    )
    row = dict(id=prop, kind="number", name=prop, label_key="", tooltip_key="",
               is_int=is_int, precision=0 if is_int else 6, step=1 if is_int else 0.00001,
               min=0, max=None, items=[])
    binding = module.property_view.SectionBinding(
        "basic_struct", [row], lambda: params, panel._text_bufs, panel._queue_pv_publish)
    binding.attach_handle(panel._handle)
    panel._pv_bindings = (binding,)
    panel._pv_binding_by_prop = {prop: binding}
    assert panel._refresh_native_backend_controls()
    assert not panel._pv_publish_pending
    binding.begin_edit(prop)
    change = SimpleNamespace(get_bool_parameter=lambda *_: False)
    panel._on_pv_number_input_change(None, change, [prop, draft])
    # Exercise the real queue -> flush -> native-refresh -> buffer-sync chain.
    panel._queue_pv_publish(binding)
    assert panel._flush_pv_publish()
    assert panel._refresh_native_backend_controls()
    assert records[binding.model_key][0]["text"] == draft
    assert getattr(params, prop) == initial
    assert not panel._pv_publish_pending
    assert not panel._flush_pv_publish()
    assert not panel._refresh_native_backend_controls()
    panel._on_pv_number_input_blur(None, None, [prop, draft])
    assert getattr(params, prop) == pytest.approx(float(draft))
    assert panel._flush_pv_publish()
    if rollback_timing == "after_refresh":
        assert panel._refresh_native_backend_controls()
        assert not panel._refresh_native_backend_controls()
    # A worker may roll back before OR after the first post-edit refresh.
    setattr(params, prop, initial)
    assert panel._refresh_native_backend_controls()
    assert records[binding.model_key][0]["text"] == binding.canonical_text(prop)
    assert not panel._pv_publish_pending
    assert not panel._refresh_native_backend_controls()


def test_saving_badge_tracks_transition_without_restarting_training(training_panel_module, monkeypatch):
    module = training_panel_module
    panel = module.TrainingPanel()
    model = _ModelStub()
    monkeypatch.setattr(module, "_training_session_state", lambda: {})
    monkeypatch.setattr(module.RuntimeState.has_trainer, "value", True)
    monkeypatch.setattr(module.RuntimeState.trainer_state, "value", "stopping")
    saving = {"value": False}
    monkeypatch.setattr(module.lf, "trainer_saving_model", lambda: saving["value"])
    dirty, scheduled = [], []
    panel._handle = SimpleNamespace(dirty=dirty.append)
    monkeypatch.setattr(panel, "_schedule_deferred_update", scheduled.append)
    panel._bind_visibility(model, lambda: None, lambda: None)
    panel._bind_status(model, lambda: None)
    assert not panel._sync_saving_status()
    assert scheduled == [0.1]
    assert model.bindings["show_ctrl_stopping"][0]()
    assert not model.bindings["show_ctrl_saving"][0]()
    saving["value"] = True
    assert panel._sync_saving_status()
    assert set(dirty) == {"status_mode", "show_ctrl_stopping", "show_ctrl_saving"}
    assert model.bindings["show_ctrl_saving"][0]()
    assert not model.bindings["show_ctrl_stopping"][0]()
    assert "training.status_saving" in model.bindings["status_mode"][0]()
    dirty.clear()
    assert not panel._sync_saving_status()
    assert not dirty
    monkeypatch.setattr(module.RuntimeState.trainer_state, "value", "completed")
    scheduled.clear()
    assert panel._sync_saving_status()
    assert not scheduled
    assert not model.bindings["show_ctrl_saving"][0]()
    assert model.bindings["show_ctrl_completed"][0]()


@pytest.mark.parametrize("offer_save", [False, True])
def test_valid_start_preserves_point_cloud_save_flow(
    training_panel_module, monkeypatch, offer_save
):
    panel = training_panel_module.TrainingPanel()
    actions = []
    params = SimpleNamespace(has_params=lambda: True, validate=lambda: "")
    monkeypatch.setattr(training_panel_module.lf, "optimization_params", lambda: params)
    monkeypatch.setattr(training_panel_module.lf, "start_training", lambda: actions.append("start"))
    monkeypatch.setattr(panel, "_should_offer_pc_save", lambda: offer_save)
    monkeypatch.setattr(panel, "_show_save_pc_dialog", lambda: actions.append("save"))
    panel._start_after_consent()
    assert actions == (["save"] if offer_save else ["start"])


def test_igs_conflict_resolution_synchronizes_viewer(training_panel_module, monkeypatch):
    panel = training_panel_module.TrainingPanel()
    params = _StrategyParamsStub()
    params.gut = True
    dialogs, updates = [], []
    monkeypatch.setattr(training_panel_module.lf, "optimization_params", lambda: params)
    monkeypatch.setattr(training_panel_module.lf, "get_render_settings",
                        lambda: SimpleNamespace(set=lambda *args: updates.append(args)))
    monkeypatch.setattr(training_panel_module.lf.ui, "confirm_dialog",
                        lambda *args: dialogs.append(args), raising=False)
    monkeypatch.setattr(panel, "_refresh_strategy_values", lambda: None)
    panel._set_strategy("igs+")
    dialogs[0][3]("training.conflict.btn_cancel")
    assert params.gut is True
    assert updates == []
    dialogs[0][3]("training.conflict.btn_disable_gut")
    assert params.gut is False
    assert params.strategy == "igs+"
    assert updates == [("raster_backend", "3dgs")]


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


def _configuration_panel(module, monkeypatch):
    panel = module.TrainingPanel()
    params = SimpleNamespace(
        has_params=lambda: True, raster_backend="3dgs",
        use_exposure_correction=False, use_bilateral_grid=False, ppisp=False,
        ppisp_use_controller=False, ppisp_freeze_from_sidecar=False,
        ppisp_controller_lr=0.003, ppisp_sidecar_path="saved.ppisp",
    )
    params.set = lambda key, value: setattr(params, key, value)
    monkeypatch.setattr(module.lf, "optimization_params", lambda: params)
    monkeypatch.setattr(panel, "_can_edit_configuration", lambda: True)
    monkeypatch.setattr(panel, "_refresh_strategy_values", lambda: None)
    return panel, params


def test_backend_selector_uses_available_descriptors_and_syncs_viewer(training_panel_module, monkeypatch):
    panel, params = _configuration_panel(training_panel_module, monkeypatch)
    updates = []
    monkeypatch.setattr(training_panel_module.lf, "training_backends", lambda: [
        {"id": "3dgs", "label": "3DGS", "viewer_backend": "3dgs"},
        {"id": "3dgut", "label": "3DGUT", "viewer_backend": "3dgut"},
    ], raising=False)
    monkeypatch.setattr(training_panel_module.lf, "get_render_settings",
                        lambda: SimpleNamespace(set=lambda *args: updates.append(args)))
    panel._set_training_backend("3dgut")
    assert params.raster_backend == "3dgut"
    assert updates == [("raster_backend", "3dgut")]
    panel._set_training_backend("nonexistent")
    assert params.raster_backend == "3dgut"
    assert len(updates) == 1
    panel._set_training_backend("3dgs")
    assert updates[-1] == ("raster_backend", "3dgs")
    monkeypatch.setattr(panel, "_can_edit_configuration", lambda: False)
    panel._set_training_backend("3dgut")
    assert params.raster_backend == "3dgs"
    assert len(updates) == 2


def test_advanced_enable_flags_are_real_and_preserve_tuning(training_panel_module, monkeypatch):
    panel, params = _configuration_panel(training_panel_module, monkeypatch)
    flags = ("use_depth_loss", "use_normal_loss", "enable_sparsity", "random", "enable_eval",
             "use_bilateral_grid", "ppisp", "use_exposure_correction",
             "ppisp_use_controller", "ppisp_freeze_from_sidecar")
    declared = {prop for spec in training_panel_module.property_view.SECTIONS for run in spec.runs for prop in run.prop_ids}
    assert set(flags) <= declared
    aliases = {}
    def set_flag(prop, value):
        setattr(params, aliases.get(prop, prop), value)
        return True
    panel._pv_binding_by_prop = {prop: SimpleNamespace(set_value=set_flag) for prop in flags}
    monkeypatch.setattr(panel, "_sync_section_states", lambda: None)
    for prop in flags:
        panel._on_pv_value_change(None, None, [prop, True])
        assert getattr(params, aliases.get(prop, prop)) is True
        panel._on_pv_value_change(None, None, [prop, False])
        assert getattr(params, aliases.get(prop, prop)) is False
    panel._on_pv_value_change(None, None, ["use_bilateral_grid", True])
    panel._on_pv_value_change(None, None, ["ppisp", True])
    panel._on_pv_value_change(None, None, ["ppisp_use_controller", True])
    panel._on_pv_value_change(None, None, ["ppisp_freeze_from_sidecar", True])
    assert params.use_bilateral_grid and params.ppisp
    panel._on_pv_value_change(None, None, ["use_exposure_correction", True])
    assert params.use_exposure_correction
    assert not params.ppisp and not params.use_bilateral_grid
    assert not params.ppisp_use_controller and not params.ppisp_freeze_from_sidecar
    assert params.ppisp_controller_lr == 0.003
    assert params.ppisp_sidecar_path == "saved.ppisp"
    assert "advanced_params" not in panel._collapsed


def test_rejected_enable_does_not_change_other_features(training_panel_module, monkeypatch):
    panel, params = _configuration_panel(training_panel_module, monkeypatch)
    params.use_exposure_correction = True
    panel._pv_binding_by_prop = {
        "ppisp": SimpleNamespace(set_value=lambda *_: False),
        "use_exposure_correction": SimpleNamespace(set_value=lambda *_: pytest.fail("Rejected toggle changed another flag")),
    }
    panel._on_pv_value_change(None, None, ["ppisp", True])
    assert params.use_exposure_correction


@pytest.mark.parametrize("query,section,field", [
    ("3DGS", "basic_params", "backend"),
    ("strategy", "basic_params", "strategy"),
    ("SH_degree", "optimization", "sh_degree"),
    ("bg_image", "background", "background_fields"),
    ("ppisp", "ppisp", "appearance"),
    ("resize_factor", "dataset", "dataset_fields"),
    ("save_steps", "save_steps", "save_steps"),
])
def test_search_keeps_bespoke_controls_reachable(training_panel_module, query, section, field):
    panel = training_panel_module.TrainingPanel()
    panel._pv_search_query = query
    assert panel._bespoke_matches(field)
    assert panel._bespoke_section_visible(section)
    assert training_panel_module.property_view.section_is_visible(
        (), section, panel._bespoke_section_visible)
    panel._pv_search_query = "no-such-setting"
    assert not panel._bespoke_matches(field)
    assert not panel._bespoke_section_visible(section)


def test_redesigned_rml_preserves_locks_and_groups_all_controls():
    from xml.etree import ElementTree
    root = Path(__file__).resolve().parents[2]
    document = ElementTree.parse(root / "src/visualizer/gui/rmlui/resources/training.rml").getroot()
    ids = [node.attrib["id"] for node in document.iter() if "id" in node.attrib]
    assert len(ids) == len(set(ids))
    by_id = {node.attrib["id"]: node for node in document.iter() if "id" in node.attrib}
    assert by_id["training-backend"].attrib["data-value"] == "training_backend"
    assert "appearance-mode" not in by_id
    assert by_id["advanced-feature-activations"] is not None
    assert any(node.attrib.get("data-event-click") == "toggle_step_scaling_lock"
               for node in document.iter("button"))
    assert any(node.attrib.get("data-class-disabled-overlay") == "step_scaling_params_locked"
               for node in document.iter("div"))
    for section, runs in {
        "camera": ("basic_exposure_correction", "basic_undistort", "basic_mip_filter"),
        "masking": ("basic_live_start", "mask_invert", "mask_threshold", "mask_alpha", "mask_penalties"),
        "depth": ("basic_depth_weight",),
        "normal": ("basic_normal_weights",),
        "background": ("bg_mode",),
        "ppisp": ("appearance_tuning",),
        "bilateral": ("bilateral",),
        "random-init": ("init_random",),
    }.items():
        mounted = {node.attrib.get("data-for") for node in by_id[f"sec-{section}"].iter()}
        assert all(f"row : pv_{run}_rows" in mounted for run in runs)

    for run, condition in {
        "basic_mip_filter": "gut_mip_filter_disabled",
        "basic_depth_toggle": "gut_depth_supervision_disabled",
        "basic_normal_toggle": "gut_normal_supervision_disabled",
    }.items():
        rows = [node for node in document.iter("div")
                if node.attrib.get("data-for") == f"row : pv_{run}_rows"]
        assert len(rows) == 1
        assert rows[0].attrib.get("data-class-disabled-overlay") == condition


@pytest.mark.parametrize("state,iteration,editable", [
    ("ready", 0, True), ("ready", 20, False), ("running", 20, False),
    ("paused", 20, False), ("starting", 0, False), ("completed", 20, False),
    ("error", 0, False), ("stopping", 20, False),
])
def test_new_selectors_follow_training_edit_lock(training_panel_module, monkeypatch, state, iteration, editable):
    monkeypatch.setattr(training_panel_module, "RuntimeState", SimpleNamespace(
        trainer_state=SimpleNamespace(value=state),
        iteration=SimpleNamespace(value=iteration),
    ))
    assert training_panel_module.TrainingPanel._can_edit_configuration() is editable


def test_bespoke_search_models_are_bound(training_panel_module):
    panel = training_panel_module.TrainingPanel()
    model = _ModelStub()
    panel._bind_property_search(model)
    panel._pv_search_query = "3dgut"
    assert model.bindings["pv_show_backend"][0]()
    assert not model.bindings["pv_show_sh_degree"][0]()


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


def test_loaded_feature_flags_drive_detail_visibility(training_panel_module):
    params = SimpleNamespace(has_params=lambda: True, use_depth_loss=True,
                             use_normal_loss=True, ppisp=True, use_bilateral_grid=True,
                             use_exposure_correction=False, enable_sparsity=True,
                             enable_eval=True, random=True)
    model = _ModelStub()
    panel = training_panel_module.TrainingPanel()
    panel._bind_visibility(model, lambda: params, lambda: None)
    for condition, prop in (("dep_depth_loss", "use_depth_loss"),
                            ("dep_normal_loss", "use_normal_loss"),
                            ("dep_ppisp", "ppisp"), ("dep_bilateral", "use_bilateral_grid"),
                            ("dep_sparsity", "enable_sparsity"), ("dep_eval", "enable_eval"),
                            ("dep_random", "random")):
        getter = model.bindings[condition][0]
        assert getter() is True
        setattr(params, prop, False)
        assert getter() is False


def test_backend_disabled_conditions_prevent_new_conflicts_but_allow_correction(
    training_panel_module,
):
    params = SimpleNamespace(
        has_params=lambda: True,
        strategy="mcmc",
        gut=True,
        undistort=False,
        mip_filter=False,
        use_depth_loss=True,
        use_normal_loss=False,
    )
    model = _ModelStub()
    panel = training_panel_module.TrainingPanel()
    panel._bind_disabled(model, lambda: params)

    def disabled(name):
        return model.bindings[name][0]()

    assert disabled("gut_disabled") is False
    assert disabled("gut_depth_supervision_disabled") is False
    assert disabled("gut_normal_supervision_disabled") is True
    assert disabled("gut_mip_filter_disabled") is True
    assert "gut_undistort_disabled" not in model.bindings

    params.gut = False
    assert disabled("gut_disabled") is True
    params.use_depth_loss = False
    params.undistort = True
    assert disabled("gut_disabled") is False
    params.strategy = "igs+"
    assert disabled("gut_disabled") is True


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

    assert controls < search < telemetry < scroll_start < parameters < scroll_end < color_picker
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
    assert ".training-scroll-region scrollbarvertical" in rcss
    assert "padding-bottom: 6dp" in rcss
    assert "width: 4dp" in rcss
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
        assert calls == [("restore", True)]
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

    assert len(ply_calls) == 1
    assert ply_calls[0][0] is pc
    assert Path(ply_calls[0][1]) == Path(info.sparse_path) / "points3D.ply"
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

def test_compact_toolbar_preserves_icons_tooltips_and_accessible_names():
    from xml.etree import ElementTree as ET

    root = Path(__file__).resolve().parents[2]
    document = ET.parse(root / "src/visualizer/gui/rmlui/resources/training.rml")
    toolbar = document.find(".//*[@class='training-toolbar']")
    for button in toolbar.iter("button"):
        assert button.find("img") is not None
        assert button.get("data-tooltip")
        assert button.get("data-attr-aria-label")
        action = button.get("data-event-click").split("'")[1]
        probe = document.find(f".//*[@id='measure-{action}']")
        assert probe.find("span").text == button.find("span").text
        assert probe.find("img").get("src") == button.find("img").get("src")
        assert {c for c in probe.get("class").split() if c.startswith("training-toolbar")} == {
            c for c in button.get("class").split() if c.startswith("training-toolbar")}
    css = (root / "src/visualizer/gui/rmlui/resources/training.rcss").read_text()
    assert "@media (max-width:" not in css
    assert ".training-toolbar.is-compact .training-toolbar-action span" in css
    assert "width: 28dp" in css
    assert document.find(".//*[@id='training-toolbar-measure']").get("aria-hidden") == "true"
    ready = toolbar.find("div[@data-if=\'show_ctrl_ready\']")
    assert ready.find(".//button[@data-if='show_reset_ready']") is not None
    paused = toolbar.find("div[@data-if=\'show_ctrl_paused\']")
    assert paused.find(".//button[@data-if='show_project_save']") is not None
    toolbar_css = css.split(".training-toolbar {", 1)[1].split("}", 1)[0]
    assert "flex-wrap: nowrap" in toolbar_css


def test_enabled_features_have_independent_parameter_sections():
    from xml.etree import ElementTree as ET

    root = Path(__file__).resolve().parents[2]
    document = ET.parse(root / "src/visualizer/gui/rmlui/resources/training.rml")
    advanced = document.find(".//*[@id='sec-advanced-params']")
    sections = {name: advanced.find(f".//*[@id='sec-{name}']") for name in (
        "depth", "normal", "ppisp", "bilateral", "exposure", "evaluation", "random-init", "sparsity")}
    for name, section in sections.items():
        assert section is not None, name
        for other in sections:
            if name != other:
                assert section.find(f".//*[@id='sec-{other}']") is None
    assert sections["depth"].find(".//*[@data-for='row : pv_basic_normal_weights_rows']") is None
    assert sections["normal"].find(".//*[@data-for='row : pv_basic_depth_weight_rows']") is None
    assert sections["evaluation"].find(".//*[@data-value='test_every_str']") is not None
    assert sections["random-init"].find(".//*[@data-for='row : pv_init_random_rows']") is not None
    for name in ("ppisp", "bilateral"):
        parent = next(node for node in advanced.iter() if sections[name] in list(node))
        assert "!dep_exposure_correction" in parent.get("data-if")
    for run in ("ppisp_exif", "appearance_tuning", "bilateral", "exposure_grid_start"):
        assert sections["exposure"].find(f".//*[@data-for='row : pv_{run}_rows']") is not None

def test_error_details_do_not_participate_in_toolbar_layout():
    from xml.etree import ElementTree as ET

    root = Path(__file__).resolve().parents[2]
    document = ET.parse(root / "src/visualizer/gui/rmlui/resources/training.rml")
    controls = document.find(".//*[@id='controls']")
    toolbar = controls.find("div[@class='training-toolbar']")
    assert all(node.text != "{{error_message}}" for node in toolbar.iter())
    feedback = controls.find("div[@id='training-error-feedback']")
    assert feedback is not None
    assert feedback.get("data-if") == "show_ctrl_error"
    assert feedback.get("class") == "training-start-feedback"
    assert feedback.find("span").text == "{{error_message}}"
    assert list(controls).index(feedback) > list(controls).index(toolbar)
    error_actions = toolbar.find("div[@data-if='show_ctrl_error']")
    assert len(error_actions.findall(".//button")) == 2
    assert toolbar.find(".//*[@class='training-status-badge is-error']") is not None


def test_advanced_title_keeps_arrow_inside_its_box():
    root = Path(__file__).resolve().parents[2]
    css = (root / "src/visualizer/gui/rmlui/resources/training.rcss").read_text()
    title = css.split("#hdr-advanced-params {", 1)[1].split("}", 1)[0]
    assert "margin: 8dp 0 0" in title
    assert "padding: 5dp 6dp" in title
    assert "box-sizing: border-box" in title
    arrow = css.split("#hdr-advanced-params > .section-arrow {", 1)[1].split("}", 1)[0]
    assert "flex: 0 0 10dp" in arrow

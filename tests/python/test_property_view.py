# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contracts for registry-backed retained-mode property rows."""

import json
import math
import re
from pathlib import Path

import pytest

from lfs_plugins import property_view


ROOT = Path(__file__).resolve().parents[2]
LOCALES = ROOT / "src" / "visualizer" / "gui" / "resources" / "locales"
TRAINING_RML = (
    ROOT
    / "src"
    / "visualizer"
    / "gui"
    / "rmlui"
    / "resources"
    / "training.rml"
)


# id: (locale key, tooltip key, precision, step, registry min, registry max, int)
EXPECTED_NUMBER_ROWS = {
    "iterations": (
        "training_params.iterations",
        "training.tooltip.iterations",
        0,
        100,
        1,
        1_000_000,
        True,
    ),
    "max_cap": (
        "training_params.max_gaussians",
        "training.tooltip.max_gaussians",
        0,
        100_000,
        1_000,
        200_000_000,
        True,
    ),
    "steps_scaler": (
        "training_params.steps_scaler",
        "training.tooltip.steps_scaler",
        2,
        0.1,
        0.0,
        10.0,
        False,
    ),
    "means_lr": (
        "training.opt.lr.position",
        "training.tooltip.lr_position",
        6,
        0.000001,
        0.0,
        0.001,
        False,
    ),
    "shs_lr": (
        "training.opt.lr.sh_coeff",
        "training.tooltip.lr_sh_coeff",
        4,
        0.0001,
        0.0,
        0.1,
        False,
    ),
    "opacity_lr": (
        "training.opt.lr.opacity",
        "training.tooltip.lr_opacity",
        4,
        0.001,
        0.0,
        1.0,
        False,
    ),
    "scaling_lr": (
        "training.opt.lr.scaling",
        "training.tooltip.lr_scaling",
        4,
        0.0001,
        0.0,
        0.1,
        False,
    ),
    "rotation_lr": (
        "training.opt.lr.rotation",
        "training.tooltip.lr_rotation",
        4,
        0.0001,
        0.0,
        0.1,
        False,
    ),
    "refine_every": (
        "training.refinement.refine_every",
        "training.tooltip.refine_every",
        0,
        10,
        1,
        1_000,
        True,
    ),
    "start_refine": (
        "training.refinement.start_refine",
        "training.tooltip.start_refine",
        0,
        100,
        0,
        10_000,
        True,
    ),
    "stop_refine": (
        "training.refinement.stop_refine",
        "training.tooltip.stop_refine",
        0,
        1_000,
        0,
        100_000,
        True,
    ),
    "grow_until_iter": (
        "training.refinement.grow_until_iter",
        "training.tooltip.grow_until_iter",
        0,
        1_000,
        0,
        100_000,
        True,
    ),
    "reset_every": (
        "training.refinement.reset_every",
        "training.tooltip.reset_every",
        0,
        100,
        100,
        10_000,
        True,
    ),
    "sh_degree_interval": (
        "training.refinement.sh_upgrade_every",
        "training.tooltip.sh_upgrade_every",
        0,
        100,
        100,
        10_000,
        True,
    ),
    "bilateral_grid_x": (
        "training.bilateral.grid_x",
        "training.tooltip.bilateral_grid_x",
        0,
        1,
        4,
        64,
        True,
    ),
    "bilateral_grid_y": (
        "training.bilateral.grid_y",
        "training.tooltip.bilateral_grid_y",
        0,
        1,
        4,
        64,
        True,
    ),
    "bilateral_grid_w": (
        "training.bilateral.grid_w",
        "training.tooltip.bilateral_grid_w",
        0,
        1,
        2,
        32,
        True,
    ),
    "bilateral_grid_lr": (
        "training.bilateral.learning_rate",
        "training.tooltip.bilateral_grid_lr",
        6,
        0.00001,
        0.0,
        0.1,
        False,
    ),
    "mask_opacity_penalty_weight": (
        "training.masking.penalty_weight",
        "training.tooltip.penalty_weight",
        3,
        0.1,
        0.0,
        10.0,
        False,
    ),
    "mask_opacity_penalty_power": (
        "training.masking.penalty_power",
        "training.tooltip.penalty_power",
        3,
        0.1,
        0.5,
        4.0,
        False,
    ),
    "mask_threshold": (
        "training.masking.threshold",
        "training.tooltip.mask_threshold",
        3,
        0.05,
        0.0,
        1.0,
        False,
    ),
    "depth_loss_weight": (
        "training_params.depth_loss_weight",
        "training.tooltip.depth_loss_weight",
        3,
        0.1,
        0.0,
        100.0,
        False,
    ),
    "normal_loss_weight": (
        "training_params.normal_loss_weight",
        "training.tooltip.normal_loss_weight",
        3,
        0.01,
        0.0,
        100.0,
        False,
    ),
    "normal_consistency_weight": (
        "training_params.normal_consistency_weight",
        "training.tooltip.normal_consistency_weight",
        3,
        0.01,
        0.0,
        100.0,
        False,
    ),
    "normal_flatten_weight": (
        "training_params.normal_flatten_weight",
        "training.tooltip.normal_flatten_weight",
        3,
        0.1,
        0.0,
        1_000.0,
        False,
    ),
    "normal_start_fraction": (
        "training_params.normal_start_fraction",
        "training.tooltip.normal_start_fraction",
        3,
        0.01,
        0.0,
        1.0,
        False,
    ),
    "normal_end_fraction": (
        "training_params.normal_end_fraction",
        "training.tooltip.normal_end_fraction",
        3,
        0.01,
        0.0,
        1.0,
        False,
    ),
    "opacity_reg": (
        "training.losses.opacity_reg",
        "training.tooltip.opacity_reg",
        4,
        0.001,
        0.0,
        1.0,
        False,
    ),
    "scale_reg": (
        "training.losses.scale_reg",
        "training.tooltip.scale_reg",
        4,
        0.001,
        0.0,
        1.0,
        False,
    ),
    "tv_loss_weight": (
        "training.losses.tv_loss_weight",
        "training.tooltip.tv_loss_weight",
        1,
        0.5,
        0.0,
        100.0,
        False,
    ),
    "init_scaling": (
        "training.init.init_scaling",
        "training.tooltip.init_scaling",
        3,
        0.01,
        0.0,
        1.0,
        False,
    ),
    "init_num_pts": (
        "training.init.num_points",
        "training.tooltip.num_points",
        0,
        10_000,
        1_000,
        1_000_000,
        True,
    ),
    "init_extent": (
        "training.init.extent",
        "training.tooltip.extent",
        1,
        0.5,
        0.1,
        10.0,
        False,
    ),
    "min_opacity": (
        "training.thresholds.min_opacity",
        "training.tooltip.min_opacity",
        4,
        0.001,
        0.0,
        math.inf,
        False,
    ),
    "prune_opacity": (
        "training.thresholds.prune_opacity",
        "training.tooltip.prune_opacity",
        4,
        0.001,
        0.0,
        math.inf,
        False,
    ),
    "sparsify_steps": (
        "training_params.sparsify_steps",
        "training.tooltip.sparsify_steps",
        0,
        1_000,
        1_000,
        50_000,
        True,
    ),
    "init_rho": (
        "training_params.init_rho",
        "training.tooltip.init_rho",
        4,
        0.001,
        0.0,
        0.01,
        False,
    ),
    "ppisp_controller_lr": (
        "training_params.ppisp_controller_lr",
        "training.tooltip.ppisp_controller_lr",
        5,
        0.0001,
        0.00001,
        0.1,
        False,
    ),
}


EXPECTED_CHECKBOX_ROWS = {
    "use_bilateral_grid": (
        "training_params.bilateral_grid",
        "training.tooltip.bilateral_grid",
    ),
    "invert_masks": (
        "training_params.invert_masks",
        "training.tooltip.invert_masks",
    ),
    "use_alpha_as_mask": (
        "training_params.use_alpha_as_mask",
        "training.tooltip.use_alpha_as_mask",
    ),
    "use_depth_loss": (
        "training_params.use_depth_loss",
        "training.tooltip.use_depth_loss",
    ),
    "use_normal_loss": (
        "training_params.use_normal_loss",
        "training.tooltip.use_normal_loss",
    ),
    "normal_auto_generate": (
        "training_params.normal_auto_generate",
        "training.tooltip.normal_auto_generate",
    ),
    "enable_sparsity": (
        "training_params.sparsity",
        "training.tooltip.sparsity",
    ),
    "gut": ("training_params.gut", "training.tooltip.gut"),
    "undistort": ("training_params.undistort", "training.tooltip.undistort"),
    "mip_filter": ("training_params.mip_filter", "training.tooltip.mip_filter"),
    "ppisp": ("training_params.ppisp", "training.tooltip.ppisp"),
    "ppisp_exposure_from_exif": (
        "training_params.ppisp_exposure_from_exif",
        "training.tooltip.ppisp_exposure_from_exif",
    ),
    "ppisp_use_controller": (
        "training_params.ppisp_controller",
        "training.tooltip.ppisp_controller",
    ),
    "ppisp_freeze_from_sidecar": (
        "training_params.ppisp_freeze_from_sidecar",
        "training.tooltip.ppisp_freeze_from_sidecar",
    ),
    "ppisp_freeze_gaussians": (
        "training_params.ppisp_freeze_gaussians",
        "training.tooltip.ppisp_freeze_gaussians",
    ),
    "random": ("training.init.random_init", "training.tooltip.random_init"),
    "enable_eval": (
        "training_params.enable_eval",
        "training.tooltip.enable_eval",
    ),
    "background_improvements": (
        "training_params.background_improvements",
        "training.tooltip.background_improvements",
    ),
}


EXPECTED_SELECT_ROWS = {
    "mask_mode": (
        "training_params.mask_mode",
        "training.tooltip.mask_mode",
        (
            (0, "training.options.mask.none"),
            (1, "training.options.mask.segment"),
            (2, "training.options.mask.ignore"),
            (3, "training.options.mask.segment_and_ignore"),
            (4, "training.options.mask.alpha_consistent"),
        ),
    ),
    "bg_mode": (
        "training_params.bg_mode",
        "training.tooltip.bg_modulation",
        (
            (0, "training.options.bg.color"),
            (1, "training.options.bg.modulation"),
            (2, "training.options.bg.image"),
            (3, "training.options.bg.random"),
        ),
    ),
    "normal_loss_space": (
        "training.advanced.normal_loss_space",
        "training.tooltip.normal_loss_space",
        (
            (0, "training.options.normal_loss_space.auto"),
            (1, "training.options.normal_loss_space.camera_opencv"),
            (2, "training.options.normal_loss_space.camera_opengl"),
            (3, "training.options.normal_loss_space.world"),
        ),
    ),
}

EXPECTED_ADVANCED_IDS = (
    "means_lr_end",
    "scaling_lr_end",
    "cropbox_lr_scale",
    "cropbox_loss_weight",
    "morton_reorder_interval",
    "min_opacity",
    "growth_grad_threshold",
    "grow_fraction",
    "opacity_decay",
    "scale_decay",
    "means_noise_weight",
    "bounds_percentile",
    "use_error_map",
    "use_edge_map",
    "far_scene_min_fraction",
    "growth_ratio_rank",
    "growth_ratio_pow",
    "fill_pacing_iter",
    "far_seed_dose",
    "ppisp_lr",
    "ppisp_reg_weight",
    "ppisp_warmup_steps",
)


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            yield from _flatten(nested, next_prefix)
    else:
        yield prefix, value


def _all_rows(lf):
    return property_view.build_rows(
        lf.ui.property_group_info("optimization"),
        property_view.MIGRATED_PROP_IDS,
        lf.optimization_params,
    )


def test_full_migration_inventory_and_schema_are_exact(lf):
    assert property_view.NUMBER_PROPS == tuple(EXPECTED_NUMBER_ROWS)
    assert property_view.BOOL_PROPS == tuple(EXPECTED_CHECKBOX_ROWS)
    assert property_view.SELECT_PROPS == tuple(EXPECTED_SELECT_ROWS)
    assert len(property_view.MIGRATED_PROP_IDS) == 59
    assert len(set(property_view.MIGRATED_PROP_IDS)) == 59

    group_info = lf.ui.property_group_info("optimization")
    resolved_runs = property_view.resolve_runs(group_info)
    rendered = tuple(prop for run in resolved_runs for prop in run.prop_ids)
    assert len(rendered) == len(set(rendered)) == 79
    assert set(rendered) == (
        set(property_view.MIGRATED_PROP_IDS) | set(EXPECTED_ADVANCED_IDS)
    ) - set(property_view.BESPOKE_OR_HIDDEN)


def test_auto_advanced_roster_and_exclusions_follow_declaration_order(lf):
    group_info = lf.ui.property_group_info("optimization")
    assert property_view.auto_advanced_prop_ids(group_info) == EXPECTED_ADVANCED_IDS
    assert "background_improvements" not in EXPECTED_ADVANCED_IDS
    assert "background_improvements" in {
        prop_id for run in property_view.BASIC_RUNS for prop_id in run.prop_ids
    }

    properties = {meta["id"]: meta for meta in group_info["properties"]}
    for prop_id in EXPECTED_ADVANCED_IDS:
        assert properties[prop_id]["advanced"] is True
    assert set(property_view.BESPOKE_OR_HIDDEN) == {
        "sh_degree",
        "lambda_dssim",
        "init_opacity",
        "depth_loss_mode",
        "strategy",
        "ppisp_controller_activation_step",
        "bg_modulation",
        "headless",
        "prune_ratio",
        "steps_scaler",
    }

    future_group = {
        "properties": [
            {"id": "future_scalar", "type": "float"},
            {"id": "future_enum", "type": "enum"},
            {"id": "future_string", "type": "string"},
            {"id": "headless", "type": "bool"},
        ]
    }
    assert property_view.auto_advanced_prop_ids(future_group) == (
        "future_scalar",
        "future_enum",
    )


def test_strategy_applicability_filters_auto_rows_and_search(lf):
    group_info = lf.ui.property_group_info("optimization")
    properties = {meta["id"]: meta for meta in group_info["properties"]}
    known_mrnf_only = {
        "means_lr_end",
        "scaling_lr_end",
        "growth_grad_threshold",
        "grow_fraction",
        "grow_until_iter",
        "opacity_decay",
        "scale_decay",
        "means_noise_weight",
        "bounds_percentile",
        "use_error_map",
        "use_edge_map",
        "background_improvements",
        "far_scene_min_fraction",
        "growth_ratio_rank",
        "growth_ratio_pow",
        "fill_pacing_iter",
        "far_seed_dose",
    }
    auto_mrnf_only = known_mrnf_only - {"grow_until_iter", "background_improvements"}
    for prop_id in known_mrnf_only:
        assert properties[prop_id]["strategies"] == ["mrnf"]

    params = {
        "strategy": "mcmc",
    }
    rows = property_view.build_rows(
        group_info,
        property_view.auto_advanced_prop_ids(group_info),
        lambda: params,
    )
    binding = property_view.SectionBinding(
        "strategy_filter",
        rows,
        lambda: params,
        {},
        lambda _binding: None,
        search_accessor=lambda: query["value"],
    )

    query = {"value": ""}
    assert not ({record["id"] for record in binding._records()} & auto_mrnf_only)
    assert "min_opacity" in {record["id"] for record in binding._records()}
    query["value"] = "edge"
    assert binding._records() == []

    params["strategy"] = "mnrf"
    query["value"] = ""
    assert auto_mrnf_only <= {
        record["id"] for record in binding._records()
    }
    assert "min_opacity" not in {record["id"] for record in binding._records()}
    query["value"] = "edge"
    assert [record["id"] for record in binding._records()] == ["use_edge_map"]

    curated = property_view.SectionBinding(
        "curated_strategy_filter",
        property_view.build_rows(
            group_info,
            ("grow_until_iter",),
            lambda: params,
        ),
        lambda: params,
        {},
        lambda _binding: None,
    )
    params["strategy"] = "mcmc"
    assert curated._records() == []
    params["strategy"] = "mrnf"
    assert [record["id"] for record in curated._records()] == ["grow_until_iter"]


def test_ppisp_advanced_declarations_are_exact(lf):
    group_info = lf.ui.property_group_info("optimization")
    params = lf.optimization_params()
    rows = {
        row["id"]: row
        for row in property_view.build_rows(
            group_info,
            ("ppisp_lr", "ppisp_reg_weight", "ppisp_warmup_steps"),
            lf.optimization_params,
        )
    }
    expected = {
        "ppisp_lr": (
            "training_params.ppisp_lr",
            "training.tooltip.ppisp_lr",
            5,
            0.0001,
            0.0,
            0.1,
            False,
        ),
        "ppisp_reg_weight": (
            "training_params.ppisp_reg",
            "training.tooltip.ppisp_reg",
            5,
            0.0001,
            0.0,
            0.1,
            False,
        ),
        "ppisp_warmup_steps": (
            "training_params.ppisp_warmup",
            "training.tooltip.ppisp_warmup",
            0,
            100,
            0,
            100_000,
            True,
        ),
    }
    for prop_id, values in expected.items():
        label, tooltip, precision, step, minimum, maximum, is_int = values
        row = rows[prop_id]
        assert row["kind"] == "number"
        assert row["label_key"] == label
        assert row["tooltip_key"] == tooltip
        assert row["precision"] == precision
        assert row["step"] == pytest.approx(step)
        assert row["min"] == pytest.approx(minimum)
        assert row["max"] == pytest.approx(maximum)
        assert row["is_int"] is is_int
        assert params.prop_info(prop_id)["advanced"] is True


def test_all_number_rows_match_registry_declarations(lf):
    rows = {row["id"]: row for row in _all_rows(lf)}
    params = lf.optimization_params()

    for prop_id, expected in EXPECTED_NUMBER_ROWS.items():
        label, tooltip, precision, step, min_value, max_value, is_int = expected
        row = rows[prop_id]
        assert row["kind"] == "number"
        assert row["label_key"] == label
        assert row["tooltip_key"] == tooltip
        assert row["precision"] == precision
        assert row["step"] == pytest.approx(step)
        assert row["min"] == pytest.approx(min_value)
        assert row["max"] == pytest.approx(max_value)
        assert row["is_int"] is is_int

        prop_info = params.prop_info(prop_id)
        assert prop_info["locale_key"] == label
        assert prop_info["tooltip_key"] == tooltip
        assert prop_info["precision"] == precision
        assert prop_info["step"] == pytest.approx(step)
        if prop_id in property_view.LEARNING_RATES:
            assert prop_info["live_update"] is True


def test_checkbox_and_select_rows_match_registry_declarations(lf):
    rows = {row["id"]: row for row in _all_rows(lf)}
    params = lf.optimization_params()

    for prop_id, (label, tooltip) in EXPECTED_CHECKBOX_ROWS.items():
        row = rows[prop_id]
        assert row["kind"] == "checkbox"
        assert row["label_key"] == label
        assert row["tooltip_key"] == tooltip
        if prop_id in {"use_bilateral_grid", "random", "undistort", "background_improvements"}:
            assert params.prop_info(prop_id)["needs_restart"] is True

    for prop_id, (label, tooltip, expected_items) in EXPECTED_SELECT_ROWS.items():
        row = rows[prop_id]
        assert row["kind"] == "select"
        assert row["label_key"] == label
        assert row["tooltip_key"] == tooltip
        assert tuple(
            (item["value"], item["locale_key"]) for item in row["items"]
        ) == expected_items


def test_number_formatting_matches_training_panel_rules():
    assert property_view.format_number(2e-5, precision=6) == "0.000020"
    assert property_view.format_number(0.0025, precision=4) == "0.0025"
    assert property_view.format_number(1234567, is_int=True) == "1,234,567"


def _number_row():
    return {
        "id": "amount",
        "kind": "number",
        "label_key": "",
        "tooltip_key": "",
        "precision": 2,
        "step": 0.1,
        "min": 0.0,
        "max": 1.0,
        "is_int": False,
        "name": "Amount",
        "items": [],
    }


def test_parse_clamp_and_invalid_commit_behavior():
    assert property_view.parse_clamped_number(
        "2.0", is_int=False, min_value=0.0, max_value=1.0
    ) == pytest.approx(1.0)
    assert property_view.parse_clamped_number(
        "-10", is_int=True, min_value=1, max_value=100
    ) == 1
    with pytest.raises(ValueError):
        property_view.parse_clamped_number(
            "0,0001", is_int=False, min_value=0.0, max_value=1.0
        )

    params = {"amount": 0.25}
    buffers = {}
    queued = []
    binding = property_view.SectionBinding(
        "test",
        [_number_row()],
        lambda: params,
        buffers,
        queued.append,
    )

    binding.update_draft("amount", "2.5")
    assert binding.commit("amount") is True
    assert params["amount"] == pytest.approx(1.0)
    assert buffers[binding.input_key("amount")] == "1.00"

    binding.update_draft("amount", "not-a-number")
    assert binding.commit("amount") is False
    assert params["amount"] == pytest.approx(1.0)
    assert buffers[binding.input_key("amount")] == "1.00"

    binding.begin_edit("amount")
    binding.update_draft("amount", "0.50")
    assert binding.cancel_edit("amount") is True
    assert buffers[binding.input_key("amount")] == "1.00"


class _RecordHandle:
    def __init__(self):
        self.updates = []

    def update_record_list(self, key, records):
        self.updates.append((key, records))


def test_event_reachable_updates_use_the_deferred_publisher():
    params = {"amount": 0.25, "enabled": False, "mode": 0}
    queued = []
    rows = [
        _number_row(),
        {
            "id": "enabled",
            "kind": "checkbox",
            "label_key": "",
            "tooltip_key": "",
            "precision": None,
            "step": 1,
            "min": None,
            "max": None,
            "is_int": False,
            "name": "Enabled",
            "items": [],
        },
        {
            "id": "mode",
            "kind": "select",
            "label_key": "",
            "tooltip_key": "",
            "precision": None,
            "step": 1,
            "min": None,
            "max": None,
            "is_int": False,
            "name": "Mode",
            "items": [
                {
                    "name": "Zero",
                    "value": 0,
                    "locale_key": "",
                    "tooltip_key": "",
                },
                {
                    "name": "One",
                    "value": 1,
                    "locale_key": "",
                    "tooltip_key": "",
                },
            ],
        },
    ]
    binding = property_view.SectionBinding(
        "deferred", rows, lambda: params, {}, queued.append
    )
    handle = _RecordHandle()
    binding.attach_handle(handle)
    assert len(handle.updates) == 2

    def assert_queued(action):
        queued.clear()
        update_count = len(handle.updates)
        action()
        assert queued == [binding]
        assert len(handle.updates) == update_count

    binding.update_draft("amount", "0.5")
    assert_queued(lambda: binding.commit("amount"))
    binding.begin_edit("amount")
    assert_queued(lambda: binding.cancel_edit("amount"))
    assert_queued(lambda: binding.step("amount", 1))
    assert_queued(lambda: binding.set_value("enabled", True))
    assert_queued(lambda: binding.set_value("mode", 1))
    assert_queued(binding.sync_text_bufs)

    with pytest.raises(TypeError, match="deferred publisher"):
        property_view.SectionBinding("unsafe", [_number_row()], params, {}, None)


def test_search_filters_localized_labels_ids_and_runtime_conditions():
    query = {"value": ""}
    condition = {"visible": True}
    rows = [
        _number_row(),
        {
            **_number_row(),
            "id": "position_rate",
            "name": "Localized Position Rate",
        },
    ]
    binding = property_view.SectionBinding(
        "search",
        rows,
        {"amount": 0.25, "position_rate": 0.5},
        {},
        lambda _binding: None,
        search_accessor=lambda: query["value"],
        visibility_condition_id="dep_test",
        visibility_predicate=lambda _condition: condition["visible"],
    )

    assert [record["id"] for record in binding._records()] == [
        "amount",
        "position_rate",
    ]
    query["value"] = "POSITION"
    assert [record["id"] for record in binding._records()] == [
        "position_rate"
    ]
    query["value"] = "AmOuNt"
    assert [record["id"] for record in binding._records()] == ["amount"]
    condition["visible"] = False
    assert binding._records() == []
    assert binding.is_visible() is False
    query["value"] = ""
    assert binding.is_visible() is True


def test_search_changes_queue_all_property_runs_without_direct_publish():
    from lfs_plugins.training_panel import TrainingPanel

    panel = object.__new__(TrainingPanel)
    panel._pv_search_query = ""
    panel._pv_bindings = (object(), object())
    queued = []
    panel._queue_pv_publish = queued.append

    panel._set_property_search_query("opacity")
    assert panel._pv_search_query == "opacity"
    assert queued == list(panel._pv_bindings)


def test_search_auto_expand_does_not_mutate_collapse_state(monkeypatch):
    from lfs_plugins.training_panel import TrainingPanel

    panel = object.__new__(TrainingPanel)
    panel._pv_search_query = "opacity"
    panel._pv_bindings = ()
    panel._collapsed = {"losses"}
    panel._sync_section_states = lambda: None
    monkeypatch.setattr(property_view, "section_is_visible", lambda *_args: True)

    panel._on_toggle_section(None, None, ["losses"])
    assert panel._collapsed == {"losses"}


def test_option_records_support_multiple_auto_placed_enums():
    def enum_row(prop_id):
        return {
            "id": prop_id,
            "kind": "select",
            "label_key": "",
            "tooltip_key": "",
            "precision": None,
            "step": 1,
            "min": None,
            "max": None,
            "is_int": False,
            "name": prop_id,
            "items": [
                {
                    "name": "First",
                    "value": 0,
                    "locale_key": "",
                    "tooltip_key": "",
                }
            ],
        }

    binding = property_view.SectionBinding(
        "enums",
        [enum_row("first_mode"), enum_row("second_mode")],
        {"first_mode": 0, "second_mode": 0},
        {},
        lambda _binding: None,
    )
    assert [item["prop_id"] for item in binding._option_records()] == [
        "first_mode",
        "second_mode",
    ]


def test_all_declaration_and_schema_locale_keys_resolve_in_every_locale(lf):
    group_info = lf.ui.property_group_info("optimization")
    keys = set()
    for meta in group_info["properties"]:
        for field in ("locale_key", "tooltip_key"):
            if meta.get(field):
                keys.add(meta[field])
        for item in meta.get("items", []):
            if item.get("locale_key"):
                keys.add(item["locale_key"])
    keys.update(section.header_locale_key for section in property_view.SECTIONS)
    keys.update(property_view._ENUM_OPTION_TOOLTIP_KEYS.values())
    keys.add("training.search.placeholder")

    assert keys
    locale_paths = sorted(LOCALES.glob("*.json"))
    assert len(locale_paths) == 10
    for locale_path in locale_paths:
        localized = dict(
            _flatten(json.loads(locale_path.read_text(encoding="utf-8")))
        )
        for key in keys:
            assert key in localized, f"{locale_path.name}: missing {key}"
            assert str(localized[key]).strip(), f"{locale_path.name}: empty {key}"


def test_training_rml_mounts_every_run_with_writable_records():
    rml = TRAINING_RML.read_text(encoding="utf-8")
    positions = []
    for run in property_view.RUNS:
        marker = f'data-for="row : pv_{run.id}_rows"'
        assert marker in rml
        positions.append(rml.index(marker))
        if any(prop in property_view.SELECT_PROPS for prop in run.prop_ids):
            assert f'data-for="item : pv_{run.id}_options"' in rml
    assert positions == sorted(positions)

    assert 'data-value="row.text"' in rml
    assert 'data-checked="row.checked"' in rml
    assert 'data-value="row.value"' in rml
    assert 'data-if="item.prop_id == row.id"' in rml
    assert 'data-event-click="pv_value_change(row.id, !row.checked)"' in rml
    assert 'data-event-change="pv_value_change(row.id, ev.value)"' in rml
    assert 'data-pv-input="1"' in rml
    assert 'data-attr-data-pv-id="row.id"' in rml
    assert 'data-attr-data-tooltip="row.tooltip_key"' in rml
    assert 'data-event-focus="pv_focus(row.id)"' in rml
    assert 'data-event-change="pv_change(row.id, ev.value)"' in rml
    assert 'data-event-blur="pv_blur(row.id, row.text)"' in rml
    assert 'data-event-escapecancel="pv_escape(row.id)"' in rml
    assert re.search(r'(?<!data-attr-)data-tooltip="row\.tooltip_key"', rml) is None

    assert 'data-if="row.id == \'iterations\'"' in rml
    assert 'data-class-steps-scale-lock-row="row.id == \'iterations\'"' in rml
    assert "steps_scaler" not in rml
    assert 'data-value="pv_search_query"' in rml
    assert 'data-event-click="pv_search_clear"' in rml
    assert 'id="sec-advanced-registry"' in rml
    assert rml.index('id="sec-save-steps"') < rml.index('id="sec-advanced-registry"')

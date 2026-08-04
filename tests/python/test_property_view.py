# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Contracts for registry-backed retained-mode property rows."""

import json
import re
from pathlib import Path

import pytest

from lfs_plugins import property_view
from lfs_plugins.environment import flag as environment_flag


ROOT = Path(__file__).resolve().parents[2]
LOCALES = ROOT / "src" / "visualizer" / "gui" / "resources" / "locales"


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, nested in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else key
            yield from _flatten(nested, next_prefix)
    else:
        yield prefix, value


def test_learning_rate_rows_match_registry_declarations(lf):
    group_info = lf.ui.property_group_info("optimization")
    rows = property_view.build_rows(
        group_info,
        property_view.LEARNING_RATES,
        lf.optimization_params,
    )

    expected = [
        (
            "means_lr",
            "training.opt.lr.position",
            "training.tooltip.lr_position",
            6,
            1e-6,
            0.0,
            0.001,
        ),
        (
            "shs_lr",
            "training.opt.lr.sh_coeff",
            "training.tooltip.lr_sh_coeff",
            4,
            1e-4,
            0.0,
            0.1,
        ),
        (
            "opacity_lr",
            "training.opt.lr.opacity",
            "training.tooltip.lr_opacity",
            4,
            0.001,
            0.0,
            1.0,
        ),
        (
            "scaling_lr",
            "training.opt.lr.scaling",
            "training.tooltip.lr_scaling",
            4,
            1e-4,
            0.0,
            0.1,
        ),
        (
            "rotation_lr",
            "training.opt.lr.rotation",
            "training.tooltip.lr_rotation",
            4,
            1e-4,
            0.0,
            0.1,
        ),
    ]

    assert len(rows) == len(expected)
    params = lf.optimization_params()
    for row, declaration in zip(rows, expected):
        prop_id, label, tooltip, precision, step, min_value, max_value = declaration
        assert row["id"] == prop_id
        assert row["kind"] == "number"
        assert row["label_key"] == label
        assert row["tooltip_key"] == tooltip
        assert row["precision"] == precision
        assert row["step"] == pytest.approx(step)
        assert row["min"] == pytest.approx(min_value)
        assert row["max"] == pytest.approx(max_value)
        assert row["is_int"] is False

        prop_info = params.prop_info(prop_id)
        assert prop_info["locale_key"] == label
        assert prop_info["tooltip_key"] == tooltip
        assert prop_info["precision"] == precision
        assert prop_info["step"] == pytest.approx(step)
        assert prop_info["live_update"] is True


def test_number_formatting_matches_training_panel_rules():
    assert property_view.format_number(2e-5, precision=6) == "0.000020"
    assert property_view.format_number(0.0025, precision=4) == "0.0025"
    assert property_view.format_number(1234567, is_int=True) == "1,234,567"


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
    binding = property_view.SectionBinding(
        "test",
        [
            {
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
        ],
        lambda: params,
        buffers,
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


def test_optimization_registry_locale_keys_resolve_in_every_locale(lf):
    group_info = lf.ui.property_group_info("optimization")
    keys = set()
    for meta in group_info["properties"]:
        for field in ("locale_key", "tooltip_key"):
            if meta.get(field):
                keys.add(meta[field])
        for item in meta.get("items", []):
            if item.get("locale_key"):
                keys.add(item["locale_key"])

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


@pytest.mark.parametrize("value", ["1", "TRUE", " yes ", "On"])
def test_environment_flag_uses_native_truthy_spellings(monkeypatch, value):
    monkeypatch.setenv("LFS_TEST_PROPERTY_VIEW_FLAG", value)
    assert environment_flag("LFS_TEST_PROPERTY_VIEW_FLAG") is True


@pytest.mark.parametrize("value", [None, "0", "FALSE", " no ", "Off"])
def test_environment_flag_defaults_off(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("LFS_TEST_PROPERTY_VIEW_FLAG", raising=False)
    else:
        monkeypatch.setenv("LFS_TEST_PROPERTY_VIEW_FLAG", value)
    assert environment_flag("LFS_TEST_PROPERTY_VIEW_FLAG") is False


def test_training_rml_uses_writable_record_text_and_dynamic_tooltip_key():
    rml = (
        ROOT
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "training.rml"
    ).read_text(encoding="utf-8")
    assert 'data-for="row : pv_lr_rows"' in rml
    assert 'data-value="row.text"' in rml
    assert 'data-pv-input="1"' in rml
    assert 'data-attr-data-pv-id="row.id"' in rml
    assert 'data-attr-data-tooltip="row.tooltip_key"' in rml
    assert 'data-event-focus="pv_focus(row.id)"' in rml
    assert 'data-event-change="pv_change(row.id, ev.value)"' in rml
    assert 'data-event-blur="pv_blur(row.id, row.text)"' in rml
    assert 'data-event-escapecancel="pv_escape(row.id)"' in rml
    assert re.search(r'(?<!data-attr-)data-tooltip="row\.tooltip_key"', rml) is None

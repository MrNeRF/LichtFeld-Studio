# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for histogram localization bundles."""

import json
from pathlib import Path


def test_histogram_locale_bundles_define_all_new_metric_keys():
    project_root = Path(__file__).parent.parent.parent
    locale_dir = project_root / "src" / "visualizer" / "gui" / "resources" / "locales"
    required_metrics = {"position_x", "position_y", "position_z", "volume", "anisotropy", "erank", "world_distance"}

    for locale_path in sorted(locale_dir.glob("*.json")):
        data = json.loads(locale_path.read_text())
        metric_block = data["histogram"]["metric"]

        for metric_name in required_metrics:
            assert metric_name in metric_block, f"{locale_path.name} missing histogram.metric.{metric_name}"
            assert "label" in metric_block[metric_name], f"{locale_path.name} missing {metric_name}.label"
            assert "description" in metric_block[metric_name], f"{locale_path.name} missing {metric_name}.description"


def test_erank_locales_are_not_left_as_english_placeholders():
    project_root = Path(__file__).parent.parent.parent
    locale_dir = project_root / "src" / "visualizer" / "gui" / "resources" / "locales"

    en_metric = json.loads((locale_dir / "en.json").read_text())["histogram"]["metric"]["erank"]

    for locale_path in sorted(locale_dir.glob("*.json")):
        if locale_path.name == "en.json":
            continue

        metric = json.loads(locale_path.read_text())["histogram"]["metric"]["erank"]
        assert metric["label"] != en_metric["label"], f"{locale_path.name} erank label still uses English placeholder"
        assert metric["description"] != en_metric["description"], (
            f"{locale_path.name} erank description still uses English placeholder"
        )


def test_histogram_compare_locales_define_required_keys():
    project_root = Path(__file__).parent.parent.parent
    locale_dir = project_root / "src" / "visualizer" / "gui" / "resources" / "locales"

    for locale_path in sorted(locale_dir.glob("*.json")):
        data = json.loads(locale_path.read_text())
        assert "common" in data, f"{locale_path.name} missing common block"
        assert "close" in data["common"], f"{locale_path.name} missing common.close"

        histogram = data["histogram"]
        assert "bins" in histogram, f"{locale_path.name} missing histogram.bins"
        assert "compare_x_bins" in histogram, f"{locale_path.name} missing histogram.compare_x_bins"
        assert "compare_y_bins" in histogram, f"{locale_path.name} missing histogram.compare_y_bins"
        assert "compare_with" in histogram, f"{locale_path.name} missing histogram.compare_with"
        assert "compare_bin_count" in histogram, f"{locale_path.name} missing histogram.compare_bin_count"
        assert "{x_count}" in histogram["compare_bin_count"], f"{locale_path.name} compare_bin_count missing x_count"
        assert "{y_count}" in histogram["compare_bin_count"], f"{locale_path.name} compare_bin_count missing y_count"
        assert "compare" in histogram, f"{locale_path.name} missing histogram.compare"

        compare = histogram["compare"]
        assert "off" in compare, f"{locale_path.name} missing histogram.compare.off"
        assert "summary" in compare, f"{locale_path.name} missing histogram.compare.summary"
        assert "range_value" in compare, f"{locale_path.name} missing histogram.compare.range_value"
        assert "status_selection" in compare, f"{locale_path.name} missing histogram.compare.status_selection"
        assert "bin_tooltip" in compare, f"{locale_path.name} missing histogram.compare.bin_tooltip"
        assert "empty" in compare, f"{locale_path.name} missing histogram.compare.empty"

        empty = compare["empty"]
        assert "title" in empty, f"{locale_path.name} missing histogram.compare.empty.title"
        assert "message" in empty, f"{locale_path.name} missing histogram.compare.empty.message"
        assert "metric_unavailable" in empty, f"{locale_path.name} missing histogram.compare.empty.metric_unavailable"
        assert "no_visible_values" in empty, f"{locale_path.name} missing histogram.compare.empty.no_visible_values"

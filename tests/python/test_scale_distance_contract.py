# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Source contracts for scale-distance bindings, stubs and action labels."""

import ast
import json
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[2]
UI_SOURCE = ROOT / "src/python/lfs/py_ui.cpp"


@pytest.mark.parametrize(("binding", "entry"), [
    ("set_tool", r'\{"scale_distance",\s*ToolType::ScaleDistance\}'),
    ("get_active_tool", r'case vis::ToolType::ScaleDistance:\s*return "builtin.scale_distance";'),
    ("is_tool_available", r'\{"builtin.scale_distance",\s*vis::ToolType::ScaleDistance\}'),
    ("set_active_tool", r'\{"builtin.scale_distance",\s*vis::ToolType::ScaleDistance\}'),
])
def test_scale_distance_native_tool_maps(binding, entry):
    definitions = re.split(r'\bm\.def\s*\(', UI_SOURCE.read_text(encoding="utf-8"))
    definition = next(part for part in definitions if part.lstrip().startswith(f'"{binding}"'))
    assert re.search(entry, definition), f"{binding} must recognize scale distance"


def test_scale_distance_points_have_array_converter_and_list_stub():
    assert "#include <nanobind/stl/array.h>" in UI_SOURCE.read_text(encoding="utf-8")
    stub = ROOT / "src/python/stubs/lichtfeld/ui/__init__.pyi"
    tree = ast.parse(stub.read_text(encoding="utf-8"))
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "get_scale_distance_points")
    assert ast.unparse(function.returns) == "list[list[float]]"


@pytest.mark.parametrize("locale", ["de", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "zh"])
def test_scale_distance_keymap_action_is_localized(locale):
    path = ROOT / "src/visualizer/gui/resources/locales" / f"{locale}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    label = data["input_settings"]["action.tool_scale_distance"]
    assert isinstance(label, str) and label.strip()
    if locale != "en":
        assert label != "Scale Distance Tool"

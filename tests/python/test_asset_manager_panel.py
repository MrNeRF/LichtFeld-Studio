# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Regression tests for Asset Manager panel record formatting and selection."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import json
import sys

import pytest


def _install_lf_stub(monkeypatch):
    class _LogStub:
        def __init__(self):
            self.messages = []

        def info(self, message):
            self.messages.append(("info", message))

        def warn(self, message):
            self.messages.append(("warn", message))

        def error(self, message):
            self.messages.append(("error", message))

    lf_stub = ModuleType("lichtfeld")
    lf_stub.ui = SimpleNamespace(
        PanelSpace=SimpleNamespace(FLOATING="FLOATING"),
        PanelHeightMode=SimpleNamespace(FILL="FILL"),
    )
    lf_stub.log = _LogStub()
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)


@pytest.fixture
def asset_manager_panel_module(monkeypatch):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    sys.modules.pop("lfs_plugins.asset_manager_panel", None)
    sys.modules.pop("lfs_plugins", None)
    _install_lf_stub(monkeypatch)
    return import_module("lfs_plugins.asset_manager_panel")


class _HandleStub:
    def __init__(self):
        self.records = {}
        self.dirty_fields = []

    def update_record_list(self, name, rows):
        self.records[name] = rows

    def dirty(self, name):
        self.dirty_fields.append(name)

    def dirty_all(self):
        self.dirty_fields.append("__all__")


class _ElementStub:
    def __init__(self, attrs=None, parent=None, tag_name="div"):
        self._attrs = attrs or {}
        self._parent = parent
        self.tag_name = tag_name

    def get_attribute(self, name, default=""):
        return self._attrs.get(name, default)

    def has_attribute(self, name):
        return name in self._attrs

    def parent(self):
        return self._parent


class _EventStub:
    def __init__(self, current_target=None, target=None, bool_params=None):
        self._current_target = current_target
        self._target = target or current_target
        self._bool_params = bool_params or {}
        self.stopped = False

    def current_target(self):
        return self._current_target

    def target(self):
        return self._target

    def get_bool_parameter(self, key, default=False):
        return self._bool_params.get(key, default)

    def stop_propagation(self):
        self.stopped = True


def _make_asset():
    return {
        "id": "a1",
        "name": "bicycle",
        "type": "dataset",
        "role": "source_dataset",
        "absolute_path": "/tmp/bicycle",
        "path": "/tmp/bicycle",
        "file_size_bytes": 4206437268,
        "tags": ["outdoor", "benchmark"],
        "exists": True,
        "dataset_metadata": {
            "image_count": 194,
            "has_masks": False,
            "mask_count": 0,
            "sparse_model": True,
            "camera_count": 1,
            "database_present": False,
            "image_root": "images",
        },
        "geometry_metadata": {},
        "video_metadata": {},
        "project_id": "p1",
        "scene_id": "s1",
        "run_id": None,
        "created_at": "2026-02-15T21:52:45.881056",
        "modified_at": "2026-04-28T14:48:57.606369",
        "is_favorite": False,
    }


def test_asset_rows_expose_scalar_tag_label(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    row = panel._format_asset_for_ui(_make_asset())

    assert row["tags_label"] == "outdoor, benchmark"
    assert "tags" not in row


def test_asset_card_title_uses_asset_path_leaf(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    asset = _make_asset()
    asset["name"] = "truck"
    asset["absolute_path"] = "/data/tandt/truck/train"
    fields = panel._get_asset_display_fields(
        asset,
        project_name="tandt",
        scene_name="truck",
        run_name="",
    )

    assert fields["display_name"] == "train"
    assert fields["display_subtitle"] == "truck"


def test_asset_manager_rml_uses_text_interpolation_for_display_values():
    project_root = Path(__file__).parent.parent.parent
    rml_path = (
        project_root
        / "src"
        / "visualizer"
        / "gui"
        / "rmlui"
        / "resources"
        / "asset_manager.rml"
    )
    rml = rml_path.read_text(encoding="utf-8")

    display_data_value_lines = [
        line.strip()
        for line in rml.splitlines()
        if ("<span" in line or "<div" in line) and "data-value=" in line
    ]

    assert display_data_value_lines == []
    assert "{{asset.display_name}}" in rml
    assert "{{selected_asset_name}}" in rml
    assert "{{selected_asset_dataset_image_count}}" in rml
    assert "{{selected_asset_iterations}}" in rml
    assert 'data-if="selected_asset_has_training_provenance"' in rml


def test_asset_selection_dirties_info_fields(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={"a1": _make_asset()},
        projects={"p1": {"id": "p1", "name": "Imported Datasets", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "bicycle", "project_id": "p1"}},
        runs={},
        tags={},
        collections={},
    )

    panel.toggle_asset_selection(None, None, ["a1"])

    assert panel.get_selected_asset_name() == "bicycle"
    assert panel.get_selected_asset_path() == "/tmp/bicycle"
    assert panel.get_selected_asset_dataset_image_count() == "194"
    assert "selected_asset_path" in panel._handle.dirty_fields
    assert "show_selection_asset" in panel._handle.dirty_fields


def test_asset_selection_resolves_asset_id_from_clicked_element(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={"a1": _make_asset()},
        projects={"p1": {"id": "p1", "name": "Imported Datasets", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "bicycle", "project_id": "p1"}},
        runs={},
        tags={},
        collections={},
    )

    event = _EventStub(current_target=_ElementStub({"data-asset-id": "a1"}))
    panel.toggle_asset_selection(None, event, [])

    assert panel.get_selected_asset_name() == "bicycle"
    assert panel.get_selected_count() == 1


def test_dom_card_click_selects_asset_from_stable_parent(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = SimpleNamespace(
        assets={"a1": _make_asset()},
        projects={"p1": {"id": "p1", "name": "Imported Datasets", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "bicycle", "project_id": "p1"}},
        runs={},
        tags={},
        collections={},
    )

    container = _ElementStub({"id": "asset-popup-content"})
    card = _ElementStub(
        {"data-asset-id": "a1", "data-asset-action": "select"},
        parent=container,
    )
    child = _ElementStub(parent=card)
    event = _EventStub(current_target=container, target=child)

    panel._on_asset_manager_click(event)

    assert panel.get_selected_asset_name() == "bicycle"
    assert panel.get_selected_count() == 1
    assert event.stopped is True


def test_dom_card_ctrl_click_adds_to_multi_selection(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    first = _make_asset()
    second = _make_asset()
    second["id"] = "a2"
    second["name"] = "garden"
    second["absolute_path"] = "/tmp/garden"
    panel._asset_index = SimpleNamespace(
        assets={"a1": first, "a2": second},
        projects={"p1": {"id": "p1", "name": "Imported Datasets", "scene_ids": ["s1"]}},
        scenes={"s1": {"id": "s1", "name": "bicycle", "project_id": "p1"}},
        runs={},
        tags={},
        collections={},
    )

    container = _ElementStub({"id": "asset-popup-content"})
    card_a1 = _ElementStub(
        {"data-asset-id": "a1", "data-asset-action": "select"},
        parent=container,
    )
    card_a2 = _ElementStub(
        {"data-asset-id": "a2", "data-asset-action": "select"},
        parent=container,
    )

    panel._on_asset_manager_click(_EventStub(current_target=container, target=card_a1))
    panel._on_asset_manager_click(
        _EventStub(
            current_target=container,
            target=card_a2,
            bool_params={"ctrl_key": True},
        )
    )

    assert panel.get_selected_count() == 2
    assert panel.get_selection_type() == "multiple"


def test_dataset_remove_deletes_catalog_json_entry(asset_manager_panel_module, tmp_path):
    index = asset_manager_panel_module.AssetIndex(
        library_path=tmp_path / "library.json"
    )
    project = index.create_project("truck")
    scene = index.create_scene(project.id, "train")
    asset = index.create_asset(
        project_id=project.id,
        scene_id=scene.id,
        run_id=None,
        name="train",
        type="dataset",
        role="source_dataset",
        path="/tmp/train",
        absolute_path="/tmp/train",
        exists=True,
    )
    index.update_scene(scene.id, dataset_asset_id=asset.id)

    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    panel._asset_index = index
    panel._selected_asset_ids = {asset.id}
    panel._selection_type = "asset"

    panel.on_remove_from_catalog(None, None, [])

    data = json.loads((tmp_path / "library.json").read_text(encoding="utf-8"))
    assert asset.id not in data["assets"]
    assert scene.id not in data["scenes"]
    assert project.id not in data["projects"]
    assert panel.get_selected_count() == 0


def test_trained_asset_parameters_resolve_from_run(asset_manager_panel_module):
    panel = asset_manager_panel_module.AssetManagerPanel()
    panel._handle = _HandleStub()
    asset = _make_asset()
    asset["id"] = "trained-1"
    asset["name"] = "model_final.ply"
    asset["type"] = "ply"
    asset["role"] = "trained_output"
    asset["run_id"] = "r1"
    asset["dataset_metadata"] = {}
    asset["geometry_metadata"] = {"gaussian_count": 1234}

    dataset = _make_asset()
    dataset["id"] = "dataset-1"
    dataset["scene_id"] = "s1"

    panel._asset_index = SimpleNamespace(
        assets={"trained-1": asset, "dataset-1": dataset},
        projects={"p1": {"id": "p1", "name": "Imported Datasets", "scene_ids": ["s1"]}},
        scenes={
            "s1": {
                "id": "s1",
                "name": "bicycle",
                "project_id": "p1",
                "dataset_asset_id": "dataset-1",
            }
        },
        runs={
            "r1": {
                "id": "r1",
                "scene_id": "s1",
                "project_id": "p1",
                "created_at": "2026-04-28T14:00:00",
                "completed_at": "2026-04-28T14:30:00",
                "parameters": {
                    "strategy": "mcmc",
                    "iterations": 30000,
                    "sh_degree": 3,
                    "optimizer": "mcmc",
                    "steps_scaler": 1.0,
                    "learning_rate": 0.00016,
                },
                "metrics": {"final_gaussians": 1234},
            }
        },
        tags={},
        collections={},
    )

    panel.toggle_asset_selection(None, None, ["trained-1"])

    assert panel.get_selected_asset_source_dataset() == "bicycle"
    assert panel.get_selected_asset_iterations() == "30000"
    assert panel.get_selected_asset_sh_degree() == "3"
    assert panel.get_selected_asset_optimizer() == "mcmc"
    assert panel.get_selected_asset_learning_rate() == "0.000160"
    assert panel.get_selected_asset_gaussian_count() == "1,234"
    assert panel.get_selected_asset_image_count() == "194"
    assert panel.get_selected_asset_strategy() == "mcmc"
    assert panel.get_selected_asset_steps_scaler() == "1.00"

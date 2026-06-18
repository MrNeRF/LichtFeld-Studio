# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for the pure-Python splat LOD hierarchy builder."""

from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
import json
import sys

import pytest


class _FakeSplatData:
    def __init__(self, name: str, count: int):
        self.name = name
        self.num_points = count
        self._count = count

    def visible_count(self) -> int:
        return self._count


class _FakeMergeTree:
    def __init__(
        self,
        *,
        leaf_count: int,
        target_count: int,
        post_prune_count: int,
        requested_ratio: float,
        final_roots: list[int],
        pruned_leaf_ids: list[int],
        merge_children: list[list[int]],
        merge_pass: list[int],
    ):
        self._leaf_count = leaf_count
        self.target_count = target_count
        self.post_prune_count = post_prune_count
        self.requested_ratio = requested_ratio
        self.final_roots = final_roots
        self.pruned_leaf_ids = pruned_leaf_ids
        self.merge_children = merge_children
        self.merge_left = [c[0] for c in merge_children]
        self.merge_right = [c[1] for c in merge_children]
        self.merge_pass = merge_pass

    def leaf_count(self) -> int:
        return self._leaf_count

    def merge_count(self) -> int:
        return len(self.merge_children)


class _FakeSimplifyResult:
    def __init__(self, splat_data: _FakeSplatData, merge_tree: _FakeMergeTree):
        self.splat_data = splat_data
        self.merge_tree = merge_tree


def _import_hierarchy_module(monkeypatch, simplify_fn, save_fn):
    project_root = Path(__file__).parent.parent.parent
    source_python = project_root / "src" / "python"
    if str(source_python) not in sys.path:
        sys.path.insert(0, str(source_python))

    fake_io = SimpleNamespace(save_ply=save_fn)
    scene_state = SimpleNamespace(selected_name="", nodes={})

    class _SceneStub:
        def is_valid(self):
            return True

        def get_node(self, name):
            return scene_state.nodes.get(name)

    lf_stub = ModuleType("lichtfeld")
    lf_stub.io = fake_io
    lf_stub.simplify_splat_data_with_history = simplify_fn
    lf_stub.get_scene = lambda: _SceneStub()
    lf_stub.get_selected_node_name = lambda: scene_state.selected_name
    monkeypatch.setitem(sys.modules, "lichtfeld", lf_stub)

    sys.modules.pop("lfs_splat_lod_hierarchy", None)
    return import_module("lfs_splat_lod_hierarchy"), scene_state


def test_build_splat_lod_hierarchy_tracks_global_node_ids_across_levels(monkeypatch):
    source = _FakeSplatData("lod0", 4)
    lod1 = _FakeSplatData("lod1", 3)
    lod2 = _FakeSplatData("lod2", 1)

    calls = []

    def _simplify(source_splat, *, ratio, lod_base, opacity_prune_threshold, progress=None):
        calls.append(
            {
                "source": source_splat.name,
                "ratio": ratio,
                "lod_base": lod_base,
                "opacity_prune_threshold": opacity_prune_threshold,
            }
        )
        if source_splat is source:
            return _FakeSimplifyResult(
                lod1,
                _FakeMergeTree(
                    leaf_count=4,
                    target_count=3,
                    post_prune_count=4,
                    requested_ratio=0.75,
                    final_roots=[2, 3, 4],
                    pruned_leaf_ids=[],
                    merge_children=[[0, 1]],
                    merge_pass=[0],
                ),
            )
        if source_splat is lod1:
            return _FakeSimplifyResult(
                lod2,
                _FakeMergeTree(
                    leaf_count=3,
                    target_count=3,
                    post_prune_count=2,
                    requested_ratio=1.0,
                    final_roots=[3],
                    pruned_leaf_ids=[1],
                    merge_children=[[0, 2]],
                    merge_pass=[0],
                ),
            )
        raise AssertionError(f"Unexpected simplify source: {source_splat.name}")

    module, _scene_state = _import_hierarchy_module(monkeypatch, _simplify, lambda *args, **kwargs: None)
    hierarchy = module.build_splat_lod_hierarchy(source, ratio=0.75, max_levels=3)

    assert [level.row_node_ids for level in hierarchy.levels] == [
        [0, 1, 2, 3],
        [2, 3, 4],
        [5],
    ]
    assert hierarchy.merge_node_ids == [4, 5]
    assert hierarchy.merge_children == [[0, 1], [2, 4]]
    assert hierarchy.created_lod == [1, 2]
    assert hierarchy.created_pass == [0, 0]
    assert hierarchy.children(4) == [0, 1]
    assert hierarchy.children(5) == [2, 4]
    assert hierarchy.children(0) is None
    assert hierarchy.levels[2].pruned_input_node_ids == [3]

    payload = hierarchy.to_dict()
    assert payload["format"] == "lichtfeld.splat_lod_hierarchy/v1"
    assert payload["levels"][1]["new_merge_node_ids"] == [4]
    assert payload["levels"][2]["pruned_input_node_ids"] == [3]
    assert payload["merge_nodes"]["children"] == [[0, 1], [2, 4]]
    assert calls == [
        {
            "source": "lod0",
            "ratio": pytest.approx(0.75),
            "lod_base": pytest.approx(2.0),
            "opacity_prune_threshold": pytest.approx(0.1),
        },
        {
            "source": "lod1",
            "ratio": pytest.approx(1.0),
            "lod_base": pytest.approx(2.0),
            "opacity_prune_threshold": pytest.approx(0.1),
        },
    ]


def test_hierarchy_save_writes_sidecar_and_node_id_attributes(monkeypatch, tmp_path, numpy):
    source = _FakeSplatData("lod0", 4)
    lod1 = _FakeSplatData("lod1", 2)

    saved_plys = []

    def _simplify(source_splat, *, ratio, lod_base, opacity_prune_threshold, progress=None):
        assert source_splat is source
        return _FakeSimplifyResult(
            lod1,
            _FakeMergeTree(
                leaf_count=4,
                target_count=2,
                post_prune_count=4,
                requested_ratio=0.5,
                final_roots=[4, 5],
                pruned_leaf_ids=[],
                merge_children=[[0, 1], [2, 3]],
                merge_pass=[0, 0],
            ),
        )

    def _save_ply(data, path, binary=True, progress=None, extra_attributes=None):
        saved_plys.append(
            {
                "name": data.name,
                "path": str(path),
                "binary": binary,
                "extra_attributes": extra_attributes,
            }
        )
        Path(path).write_text("ply\n", encoding="utf-8")

    module, _scene_state = _import_hierarchy_module(monkeypatch, _simplify, _save_ply)
    hierarchy = module.build_splat_lod_hierarchy(source, ratio=0.5, max_levels=2)
    manifest = hierarchy.save(tmp_path, base_name="bike", node_id_attribute="node_id")

    assert manifest["ply_paths"] == [
        str(tmp_path / "bike_lod0.ply"),
        str(tmp_path / "bike_lod1.ply"),
    ]
    assert manifest["json_path"] == str(tmp_path / "bike_hierarchy.json")
    assert Path(manifest["json_path"]).exists()

    with open(manifest["json_path"], "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["levels"][0]["ply_path"] == str(tmp_path / "bike_lod0.ply")
    assert payload["levels"][1]["ply_path"] == str(tmp_path / "bike_lod1.ply")
    assert saved_plys[0]["name"] == "lod0"
    assert saved_plys[1]["name"] == "lod1"
    numpy.testing.assert_array_equal(saved_plys[0]["extra_attributes"]["node_id"], numpy.array([0, 1, 2, 3], dtype=numpy.float32))
    numpy.testing.assert_array_equal(saved_plys[1]["extra_attributes"]["node_id"], numpy.array([4, 5], dtype=numpy.float32))


def test_lichtfeld_module_exposes_build_splat_lod_hierarchy(lf):
    assert hasattr(lf, "build_splat_lod_hierarchy")


def test_build_splat_lod_hierarchy_resolves_source_by_scene_node_name(monkeypatch):
    source = _FakeSplatData("lod0", 4)
    node = SimpleNamespace(name="SourceNode", id=17, splat_data=lambda: source)

    def _simplify(source_splat, *, ratio, lod_base, opacity_prune_threshold, progress=None):
        assert source_splat is source
        return _FakeSimplifyResult(
            _FakeSplatData("lod1", 2),
            _FakeMergeTree(
                leaf_count=4,
                target_count=2,
                post_prune_count=4,
                requested_ratio=0.5,
                final_roots=[4, 5],
                pruned_leaf_ids=[],
                merge_children=[[0, 1], [2, 3]],
                merge_pass=[0, 0],
            ),
        )

    module, scene_state = _import_hierarchy_module(monkeypatch, _simplify, lambda *args, **kwargs: None)
    scene_state.nodes["SourceNode"] = node

    hierarchy = module.build_splat_lod_hierarchy("SourceNode", ratio=0.5, max_levels=2)

    assert hierarchy.source_node_name == "SourceNode"
    assert hierarchy.source_node_id == 17
    assert hierarchy.levels[0].row_node_ids == [0, 1, 2, 3]
    assert hierarchy.levels[1].row_node_ids == [4, 5]


def test_build_splat_lod_hierarchy_uses_selected_scene_node_when_source_is_omitted(monkeypatch):
    source = _FakeSplatData("lod0", 2)
    node = SimpleNamespace(name="SelectedNode", id=9, splat_data=lambda: source)

    def _simplify(source_splat, *, ratio, lod_base, opacity_prune_threshold, progress=None):
        assert source_splat is source
        return _FakeSimplifyResult(
            _FakeSplatData("lod1", 1),
            _FakeMergeTree(
                leaf_count=2,
                target_count=1,
                post_prune_count=2,
                requested_ratio=0.5,
                final_roots=[2],
                pruned_leaf_ids=[],
                merge_children=[[0, 1]],
                merge_pass=[0],
            ),
        )

    module, scene_state = _import_hierarchy_module(monkeypatch, _simplify, lambda *args, **kwargs: None)
    scene_state.selected_name = "SelectedNode"
    scene_state.nodes["SelectedNode"] = node

    hierarchy = module.build_splat_lod_hierarchy(ratio=0.5, max_levels=2)

    assert hierarchy.source_node_name == "SelectedNode"
    assert hierarchy.source_node_id == 9
    assert hierarchy.final_row_node_ids == [2]

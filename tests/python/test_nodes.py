# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

import json

import pytest


def _geometry(lf, numpy):
    tensor = lambda value: lf.Tensor.from_numpy(numpy.ascontiguousarray(value))
    count = 2
    splats = lf.nodes.Splats(
        tensor(numpy.zeros((count, 3), dtype=numpy.float32)),
        tensor(numpy.array([[0.2, 0.3, 0.4], [0.7, 0.8, 0.9]], dtype=numpy.float32)),
        tensor(numpy.zeros((count, 1, 3), dtype=numpy.float32)),
        tensor(numpy.zeros((count, 3), dtype=numpy.float32)),
        tensor(numpy.tile(numpy.array([[1, 0, 0, 0]], dtype=numpy.float32), (count, 1))),
        tensor(numpy.zeros((count, 1), dtype=numpy.float32)),
    )
    return lf.nodes.Geometry(splats=splats)


def _insert_between(tree, node, output="Result"):
    group_input, group_output = tree.nodes[:2]
    assert tree.unlink(group_input, "Geometry", group_output, "Geometry")
    tree.link(group_input, "Geometry", node, "Geometry")
    tree.link(node, output, group_output, "Geometry")


def test_types_and_tree_json_round_trip(lf):
    descriptors = lf.nodes.node_types()
    ids = {item["id"] for item in descriptors}
    assert "lfs.colour_correct" in ids
    assert "lfs.posterize" in ids
    colour_correct = next(item for item in descriptors if item["id"] == "lfs.colour_correct")
    exposure = next(item for item in colour_correct["inputs"] if item["identifier"] == "Exposure")
    assert exposure["default"] == 0.0
    assert exposure["soft_min"] == -10.0
    assert exposure["soft_max"] == 10.0

    tree = lf.nodes.new_tree("Round trip")
    tree.add_input("Strength", "float", 0.5, min=0.0, max=1.0)
    restored = lf.nodes.load_tree(tree.to_json())
    payload = json.loads(restored.to_json())
    assert payload["name"] == "Round trip"
    assert payload["interface"]["inputs"][1]["identifier"] == "Strength"
    assert payload["interface"]["inputs"][1]["min"] == 0.0
    assert payload["interface"]["inputs"][1]["max"] == 1.0


def test_builtin_python_posterize_evaluates(lf, numpy):
    tree = lf.nodes.new_tree("Posterize test")
    posterize = tree.add_node("lfs.posterize", "Posterize")
    _insert_between(tree, posterize, "Geometry")

    result = lf.nodes.evaluate_tree(tree, _geometry(lf, numpy))

    assert result.splats.sh0.tolist() == [
        [0.0, 0.25, 0.25],
        [0.5, 0.75, 0.75],
    ]


def test_python_node_hot_reload_and_error_containment(lf, numpy):
    class PassThrough(lf.nodes.Node):
        id = "tests.pass_through"
        label = "Pass Through"
        category = "Test"
        inputs = [
            lf.nodes.Input("Geometry", "geometry"),
            lf.nodes.Input("Selection", "float", 1.0, field=True),
        ]
        outputs = [lf.nodes.Output("Result", "geometry")]
        properties = [lf.nodes.Property("Mode", "string", "pass")]

        def execute(self, ctx):
            assert ctx.field("Selection", ctx.input("Geometry").splats).shape == (2,)
            assert ctx.prop("Mode") == "pass"
            return {"Result": ctx.input("Geometry")}

    assert lf.nodes.register_node(PassThrough) == PassThrough.id
    tree = lf.nodes.new_tree("Python node")
    node = tree.add_node(PassThrough.id, "Python", location=(200.0, 10.0))
    assert tuple(node.location) == (200.0, 10.0)
    assert tree.input_node.type_id == "lfs.group_input"
    assert tree.output_node.type_id == "lfs.group_output"
    _insert_between(tree, node)
    result = lf.nodes.evaluate_tree(tree, _geometry(lf, numpy))
    assert result.splats.means.shape == (2, 3)

    class Broken(PassThrough):
        id = PassThrough.id

        def execute(self, ctx):
            raise RuntimeError("contained python failure")

    lf.nodes.register_node(Broken)
    with pytest.raises(ValueError, match="contained python failure"):
        lf.nodes.evaluate_tree(tree, _geometry(lf, numpy))
    assert node.error == "RuntimeError: contained python failure"
    assert lf.nodes.unregister_nodes_for_module(PassThrough.__module__) == 1
    assert PassThrough.id not in {item["id"] for item in lf.nodes.node_types()}


def test_scene_modifier_api_skips_without_scene(lf):
    with pytest.raises(RuntimeError, match="scene manager is unavailable"):
        lf.nodes.evaluate("missing")

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import importlib.util
from pathlib import Path

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


def test_builtin_help_contract(lf):
    root = Path(__file__).resolve().parents[2]
    english = json.loads((root / "src/visualizer/gui/resources/locales/en.json").read_text())["nodes"]
    for node in lf.nodes.node_types():
        if node["id"][4:] not in english:
            continue
        assert 0 < len(node["description"]) <= 110, node["id"]
        for banned in ("log-scale", "geometric mean", "field", "tensor", "domain"):
            assert banned not in node["description"].lower(), (node["id"], banned)
        assert 2 <= len(node["help"].splitlines()) <= 4, node["id"]
        text = english[node["id"][4:]]
        assert node["description"] == text["description"]
        assert node["help"] == text["help"]
        for collection in ("inputs", "outputs", "properties"):
            for declaration in node[collection]:
                message = declaration["description"]
                assert message == text[collection][declaration["identifier"]]
                # This supplied model text is intentionally verbatim, at 139 characters.
                limit = 139 if (node["id"], declaration["identifier"]) == ("lfs.remove_floaters", "Isolation Radius") else 120
                assert 0 < len(message) <= limit, (node["id"], declaration["identifier"])


def test_builtin_help_translations_are_complete_and_compact():
    root = Path(__file__).resolve().parents[2]
    for locale in (root / "src/visualizer/gui/resources/locales").glob("*.json"):
        for node_id, node in json.loads(locale.read_text(encoding="utf-8"))["nodes"].items():
            assert node["label"].strip(), (locale.name, node_id)
            assert 0 < len(node["description"]) <= 110, (locale.name, node_id)
            assert 2 <= len(node["help"].splitlines()) <= 4, (locale.name, node_id)
            for collection in ("inputs", "outputs", "properties"):
                for identifier, message in node[collection].items():
                    limit = 139 if (locale.stem, node_id, identifier) == ("en", "remove_floaters", "Isolation Radius") else 120
                    assert 0 < len(message) <= limit, (locale.name, node_id, identifier)


def test_python_help_declarations_round_trip(lf):
    class HelpNode(lf.nodes.Node):
        id = "tests.help"
        description = "Plugin description, unchanged."
        help = "First line.\nSecond line."
        inputs = [lf.nodes.Input("Amount", "float", 1.0, description="Your input help.")]
        outputs = [lf.nodes.Output("Result", "float", description="Your output help.")]
        properties = [lf.nodes.Property("Mode", "enum", "first", ["first", "second"], description="Your setting help.")]

        def execute(self, ctx):
            return {"Result": ctx.input("Amount")}

    lf.nodes.register_node(HelpNode)
    try:
        descriptor = next(node for node in lf.nodes.node_types() if node["id"] == HelpNode.id)
        assert descriptor["description"] == HelpNode.description
        assert descriptor["help"] == HelpNode.help
        assert descriptor["inputs"][0]["description"] == "Your input help."
        assert descriptor["outputs"][0]["description"] == "Your output help."
        assert descriptor["properties"][0]["description"] == "Your setting help."
    finally:
        lf.nodes.unregister_node(HelpNode.id)


def test_generated_node_reference_is_current(lf):
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("node_reference", root / "tools/generate_node_reference.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    descriptors = {node["id"]: node for node in lf.nodes.node_types() if node["id"].startswith("lfs.")}
    for node in descriptors.values():
        path = reference.DESTINATION / (node["id"] + ".md")
        assert path.read_text(encoding="utf-8") == reference.render_node(node), node["id"]
    # Host-only descriptors are checked against the live MCP registry in C++.
    # Include their page headings when checking the complete index headlessly.
    for page in reference.DESTINATION.glob("lfs.*.md"):
        if page.stem not in descriptors:
            lines = page.read_text(encoding="utf-8").splitlines()
            descriptors[page.stem] = dict(id=page.stem, label=lines[1][2:],
                                           description=lines[3], category=lines[5].split(" · ", 1)[1])
    assert (reference.DESTINATION / "index.md").read_text(encoding="utf-8") == reference.render_index(descriptors.values())


def test_builtin_python_posterize_evaluates(lf, numpy):
    tree = lf.nodes.new_tree("Posterize test")
    posterize = tree.add_node("lfs.posterize", "Posterize")
    _insert_between(tree, posterize, "Geometry")

    result = lf.nodes.evaluate_tree(tree, _geometry(lf, numpy))

    original = numpy.asarray(_geometry(lf, numpy).splats.sh0.tolist())
    rgb = numpy.clip(0.5 + 0.28209479177387814 * original, 0, 1)
    expected = (numpy.round(rgb * 3) / 3 - 0.5) / 0.28209479177387814
    numpy.testing.assert_allclose(result.splats.sh0.tolist(), expected, atol=1e-6)


@pytest.mark.parametrize("weight", [0.0, 0.25, 1.0])
def test_posterize_selection_blends_rgb_and_fades_higher_sh(lf, numpy, weight):
    tree = lf.nodes.new_tree("Posterize selection")
    node = tree.add_node("lfs.posterize")
    node.set_input("Selection", weight)
    node.set_input("Levels", 3)
    _insert_between(tree, node, "Geometry")
    geometry = _geometry(lf, numpy)
    shn = lf.Tensor.from_numpy(numpy.ones((2, 1, 3), dtype=numpy.float32))
    geometry = geometry.replace(splats=geometry.splats.replace(shN=shn))
    original = numpy.asarray(geometry.splats.sh0.tolist())
    result = lf.nodes.evaluate_tree(tree, geometry)
    rgb = numpy.clip(0.5 + 0.28209479177387814 * original, 0, 1)
    quantized = (numpy.round(rgb * 2) / 2 - 0.5) / 0.28209479177387814
    numpy.testing.assert_allclose(result.splats.sh0.tolist(), original + (quantized - original) * weight, atol=1e-6)
    numpy.testing.assert_allclose(result.splats.shN.tolist(), 1.0 - weight, atol=1e-6)


def test_graph_names_are_unique_on_create_rename_and_import(lf):
    first = lf.nodes.new_tree("Autumn Lawn")
    second = lf.nodes.new_tree("Autumn Lawn")
    assert (first.name, second.name) == ("Autumn Lawn", "Autumn Lawn 2")
    third = lf.nodes.new_tree("Other")
    third.name = "Autumn Lawn"
    assert third.name == "Autumn Lawn 3"
    data = json.loads(first.to_json())
    data.pop("uuid")
    imported = lf.nodes.load_tree(json.dumps(data))
    assert imported.name == "Autumn Lawn 4"


def test_python_node_declarations_require_lists_and_execute(lf):
    class AttributeNode:
        id = "tests.invalid_declarations"
        Geometry = lf.nodes.Input("Geometry", "geometry")

        def execute(self, ctx):
            return {}

    with pytest.raises(TypeError, match="Attribute-style"):
        lf.nodes.register_node(AttributeNode)

    class OldCallback:
        id = AttributeNode.id
        inputs = []
        outputs = []

        def evaluate(self, ctx):
            return {}

    with pytest.raises(TypeError, match="execute"):
        lf.nodes.register_node(OldCallback)

    class InvalidList:
        id = AttributeNode.id
        inputs = (lf.nodes.Input("Geometry", "geometry"),)

        def execute(self, ctx):
            return {}

    with pytest.raises(TypeError, match="must be lists"):
        lf.nodes.register_node(InvalidList)
    with pytest.raises(TypeError):
        lf.nodes.Input("geometry")
    with pytest.raises(TypeError):
        lf.nodes.Output("geometry")


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

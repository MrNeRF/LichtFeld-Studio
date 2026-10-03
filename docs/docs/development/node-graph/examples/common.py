"""Helpers shared by the node-graph examples.

Each example module defines SCENE, TITLE, a `build(target)` that creates its
node tree and adds it as a modifier on the scene node `target`, and a
`check(target)` that returns numbers proving the result. Run one from the
Python console with `run_example(module)`.
"""
import math

import lichtfeld as lf

N = lf.nodes

# Captured scenes are not axis aligned. These come from the COLMAP cameras of
# the Mip-NeRF 360 captures: `up` is the mean camera up vector and `focus` the
# point all cameras look at (the vase on the garden table, the bike).
SCENES = {
    "garden": {"up": (-0.002, -0.882, -0.471), "focus": (0.415, 1.509, 1.097)},
    "bicycle": {"up": (-0.016, -0.973, -0.231), "focus": (0.425, 0.387, 0.281)},
}


def unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def offset(point, direction, distance):
    return tuple(p + d * distance for p, d in zip(point, direction))


def new_graph(name):
    """A tree with Group Input and Group Output, not yet linked."""
    tree = N.new_tree(name)
    return tree, tree.input_node, tree.output_node


def add(tree, type_id, x=0.0, y=0.0, props=None, **inputs):
    """Add a node; keyword inputs use the socket label with spaces as `_`."""
    node = tree.add_node(type_id, location=(x, y))
    for key, value in inputs.items():
        node.set_input(key.replace("_", " "), value)
    for key, value in (props or {}).items():
        node.set_property(key, value)
    return node


def link(tree, a, out, b, inp):
    tree.link(a, out, b, inp)


def horizontal_axes(scene):
    """Two unit vectors spanning the ground plane: `side` (the world X axis
    projected onto the ground) and `across`, perpendicular to both."""
    up = unit(SCENES[scene]["up"])
    d = up[0]
    side = unit((1.0 - d * up[0], -d * up[1], -d * up[2]))
    across = (up[1] * side[2] - up[2] * side[1], up[2] * side[0] - up[0] * side[2],
              up[0] * side[1] - up[1] * side[0])
    return side, across


def along_field(tree, scene, direction, x=-600.0, y=-700.0):
    """Value output = signed distance from the focus along `direction`."""
    pos = add(tree, "lfs.position", x, y)
    rel = add(tree, "lfs.vector_math", x + 180, y, props={"operation": "subtract"}, B=SCENES[scene]["focus"])
    dot = add(tree, "lfs.vector_math", x + 360, y, props={"operation": "dot"}, B=unit(direction))
    link(tree, pos, "Position", rel, "A")
    link(tree, rel, "Vector", dot, "A")
    return dot


def height_field(tree, scene, x=-600.0, y=-300.0):
    """Value output = signed height above the scene focus along its up axis."""
    s = SCENES[scene]
    pos = add(tree, "lfs.position", x, y)
    rel = add(tree, "lfs.vector_math", x + 180, y, props={"operation": "subtract"}, B=s["focus"])
    dot = add(tree, "lfs.vector_math", x + 360, y, props={"operation": "dot"}, B=unit(s["up"]))
    link(tree, pos, "Position", rel, "A")
    link(tree, rel, "Vector", dot, "A")
    return dot


def horizontal_distance_field(tree, scene, x=-600.0, y=-500.0):
    """Value output = distance from the focus measured in the ground plane."""
    s = SCENES[scene]
    up = unit(s["up"])
    pos = add(tree, "lfs.position", x, y)
    rel = add(tree, "lfs.vector_math", x + 180, y, props={"operation": "subtract"}, B=s["focus"])
    h = add(tree, "lfs.vector_math", x + 360, y + 80, props={"operation": "dot"}, B=up)
    along = add(tree, "lfs.vector_math", x + 540, y + 80, props={"operation": "scale"}, A=up)
    flat = add(tree, "lfs.vector_math", x + 720, y, props={"operation": "subtract"})
    length = add(tree, "lfs.vector_math", x + 900, y, props={"operation": "length"})
    link(tree, pos, "Position", rel, "A")
    link(tree, rel, "Vector", h, "A")
    link(tree, h, "Value", along, "Scale")
    link(tree, rel, "Vector", flat, "A")
    link(tree, along, "Vector", flat, "B")
    link(tree, flat, "Vector", length, "A")
    return length


def splat_count(target):
    geo = N.evaluated(target)
    return 0 if geo is None or geo.splats is None else int(geo.splats.means.shape[0])


def stored_count(target):
    node = lf.get_scene().get_node(target)
    return int(node.splat_data().num_points) if node is not None else 0


def clear_examples():
    """Remove every modifier and tree so the next example starts clean."""
    scene = lf.get_scene()
    for node in scene.get_nodes():
        for modifier in list(N.modifiers(node.name)):
            N.remove_modifier(node.name, modifier.name)
    for tree in list(N.trees()):
        N.remove_tree(tree.uuid)

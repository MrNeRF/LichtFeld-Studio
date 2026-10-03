"""Set dressing: copy the vase and stand the copy further along the table.

Separate the vase (everything above the table top near its axis), move the copy along the table top
and join it back with the original scene.
"""
from common import *

SCENE = "garden"
TITLE = "Duplicate the vase"


def vase_selection(t, x=-400, y=-220):
    """The vase and its dried flowers: above the table top, within 0.4 of the
    vase axis."""
    near = add(t, "lfs.compare", x, y, props={"operation": "less_than"}, B=0.4)
    link(t, horizontal_distance_field(t, SCENE, x - 1100, y), "Value", near, "A")
    above = add(t, "lfs.compare", x, y - 200, props={"operation": "greater_than"}, B=0.03)
    link(t, height_field(t, SCENE, x - 600, y - 200), "Value", above, "A")
    both = add(t, "lfs.boolean_math", x + 180, y - 100, props={"operation": "and"})
    link(t, near, "Result", both, "A")
    link(t, above, "Result", both, "B")
    return both


def build(target):
    t, gin, gout = new_graph(TITLE)
    s = SCENES[SCENE]
    vase = vase_selection(t)
    take = add(t, "lfs.separate_geometry", -200, 0)
    _, across = horizontal_axes(SCENE)
    move = add(t, "lfs.transform_geometry", 0, 0, Translation=tuple(0.8 * a for a in across))
    join = add(t, "lfs.join_geometry", 200, 100)
    link(t, gin, "Geometry", take, "Geometry")
    link(t, vase, "Result", take, "Selection")
    link(t, take, "Selection", move, "Geometry")
    link(t, gin, "Geometry", join, "Geometry")
    link(t, move, "Geometry", join, "Geometry")
    link(t, join, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    before, after = stored_count(target), splat_count(target)
    return {"before": before, "after": after, "copied": after - before, "ok": 2000 < after - before < 100000}

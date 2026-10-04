"""Snow globe: keep only what lies inside a mesh.

Any closed mesh can be a selection volume. A sphere mesh, scaled around the
bike, keeps the inside and drops the rest - the same works with a hand-made
mesh for shapes a box or ellipsoid cannot describe.
"""
from common import *

SCENE = "bicycle"
TITLE = "Mesh-shaped cut-out"
MESH = "sphere"


def build(target):
    t, gin, gout = new_graph(TITLE)
    s = SCENES[SCENE]
    info = add(t, "lfs.object_info", -600, -220, props={"object": MESH, "transform_space": "original"})
    place = add(t, "lfs.transform_geometry", -400, -220, Scale=1.9,
                Translation=offset(s["focus"], unit(s["up"]), -0.2))
    inside = add(t, "lfs.inside_mesh", -200, -220)
    keep = add(t, "lfs.separate_geometry", 0, 0)
    link(t, info, "Geometry", place, "Geometry")
    link(t, place, "Geometry", inside, "Mesh")
    link(t, gin, "Geometry", keep, "Geometry")
    link(t, inside, "Selection", keep, "Selection")
    link(t, keep, "Selection", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    before, after = stored_count(target), splat_count(target)
    return {"before": before, "after": after, "ok": 0.02 * before < after < 0.6 * before}

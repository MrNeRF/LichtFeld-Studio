"""Show the SfM point cloud as splats to compare it with the trained result.

The sparse COLMAP points are imported as a point-cloud node. A graph on the
garden node replaces what it shows with those points converted to small round
Gaussians (radius picked from the cloud's density) - the trained splats stay
stored underneath and come back when the modifier is switched off.
"""
from common import *

SCENE = "garden"
TITLE = "SfM points as splats"
POINTS = "garden_sparse"


def build(target):
    t, gin, gout = new_graph(TITLE)
    info = add(t, "lfs.object_info", -300, 0, props={"object": POINTS, "transform_space": "original"})
    splats = add(t, "lfs.points_to_splats", 0, 0, Radius=0.0, Opacity=0.95)
    link(t, info, "Geometry", splats, "Geometry")
    link(t, splats, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    after = splat_count(target)
    return {"splats_from_points": after, "ok": after > 100000}

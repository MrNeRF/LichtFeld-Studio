"""Remove floaters only below the ground and up in the sky.

Floaters gather where the capture has nothing to hold them: under the ground
plane and in the empty sky. Restricting the clean-up to those two bands keeps
the sparse, large background Gaussians of the scene itself intact.
"""
from common import *

SCENE = "bicycle"
TITLE = "Floaters below ground and in the sky"
CAMERA = {"distance": 9.0, "height": 4.0}


def build(target):
    t, gin, gout = new_graph(TITLE)
    h = height_field(t, SCENE)
    below = add(t, "lfs.compare", -60, -260, props={"operation": "less_than"}, B=-1.6)
    above = add(t, "lfs.compare", -60, -380, props={"operation": "greater_than"}, B=4.0)
    band = add(t, "lfs.boolean_math", 120, -320, props={"operation": "or"})
    clean = add(t, "lfs.remove_floaters", 300, 0, Min_Opacity=0.1, Isolation_Radius=3.0,
                Min_Neighbours=2, props={"relative_to_size": True})
    link(t, h, "Value", below, "A")
    link(t, h, "Value", above, "A")
    link(t, below, "Result", band, "A")
    link(t, above, "Result", band, "B")
    link(t, gin, "Geometry", clean, "Geometry")
    link(t, band, "Result", clean, "Selection")
    link(t, clean, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    before, after = stored_count(target), splat_count(target)
    return {"before": before, "after": after, "removed": before - after,
            "ok": 0 < before - after < 0.2 * before}

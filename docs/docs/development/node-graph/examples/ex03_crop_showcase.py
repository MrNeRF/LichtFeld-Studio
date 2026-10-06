"""Crop the garden to the table for a clean showcase.

Keeps everything within a sphere around the table and drops the rest, the
usual first step before publishing an object-centred capture.
"""
from common import *

SCENE = "garden"
TITLE = "Crop to the table"


def build(target):
    t, gin, gout = new_graph(TITLE)
    s = SCENES[SCENE]
    sphere = add(t, "lfs.ellipsoid_selection", -300, -200, Centre=offset(s["focus"], unit(s["up"]), -0.6),
                 Radii=(2.2, 2.2, 2.2))
    keep = add(t, "lfs.separate_geometry", 0, 0)
    link(t, gin, "Geometry", keep, "Geometry")
    link(t, sphere, "Selection", keep, "Selection")
    link(t, keep, "Selection", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    before, after = stored_count(target), splat_count(target)
    return {"before": before, "after": after, "kept_fraction": round(after / before, 3), "ok": 0.05 < after / before < 0.6}

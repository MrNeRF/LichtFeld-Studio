"""Borrow a small bouquet from the vase and scatter it onto the lawn.

Separate the source once, centre it, and shrink it to a little tuft. A very
sparse sample of green lawn splats supplies the anchors. Instance on Points
rotates the geometry and its view-dependent colour together before joining
the copies back into the garden. Increase the anchor fraction cautiously.
"""
from common import *
from ex15_duplicate_vase import vase_selection

SCENE = "garden"
TITLE = "Scatter flowers on the lawn"


def build(target):
    t, gin, gout = new_graph(TITLE)
    cut = add(t, "lfs.separate_geometry", -150, 100)
    link(t, gin, "Geometry", cut, "Geometry")
    link(t, vase_selection(t, -600, 450), "Result", cut, "Selection")
    centre = add(t, "lfs.transform_geometry", 90, 100,
                 Translation=tuple(-v for v in SCENES[SCENE]["focus"]))
    link(t, cut, "Selection", centre, "Geometry")
    greens = add(t, "lfs.hsv_range", -650, -500, Hue=0.25, Hue_Range=0.07,
                 Saturation_Min=0.3, Value_Min=0.08)
    lawn = add(t, "lfs.separate_geometry", -400, -280)
    link(t, gin, "Geometry", lawn, "Geometry")
    link(t, greens, "Selection", lawn, "Selection")
    # Take regularly spaced indices in the already-filtered lawn, without
    # relying on Decimate's minimum fraction or creating millions of copies.
    index = add(t, "lfs.index", -650, -740)
    stride = add(t, "lfs.math", -430, -740, props={"operation": "divide"}, B=50000.0)
    fraction = add(t, "lfs.math", -210, -740, props={"operation": "fraction"})
    anchors = add(t, "lfs.compare", 10, -740, props={"operation": "less_than"}, B=0.00001)
    sparse = add(t, "lfs.separate_geometry", -150, -280)
    link(t, index, "Index", stride, "A")
    link(t, stride, "Value", fraction, "A")
    link(t, fraction, "Value", anchors, "A")
    link(t, anchors, "Result", sparse, "Selection")
    link(t, lawn, "Selection", sparse, "Geometry")
    scatter = add(t, "lfs.instance_on_points", 350, -80, Scale=(0.17, 0.17, 0.17), Rotation=(0, 0, 12))
    link(t, sparse, "Selection", scatter, "Points")
    link(t, centre, "Geometry", scatter, "Instance")
    join = add(t, "lfs.join_geometry", 620, 0)
    link(t, gin, "Geometry", join, "Geometry")
    link(t, scatter, "Geometry", join, "Geometry")
    link(t, join, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    before, after = stored_count(target), splat_count(target)
    return {"before": before, "after": after, "scattered_splats": after - before,
            "ok": 0 < after - before < 500000}

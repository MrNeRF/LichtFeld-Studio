"""Sharpen only the centrepiece.

Shrinking Gaussians crisps up edges; keep coverage raises their opacity so the
surface does not thin out. A feathered ellipsoid limits it to the vase and
table top.
"""
from common import *

SCENE = "garden"
TITLE = "Sharpen the centrepiece"


def build(target):
    t, gin, gout = new_graph(TITLE)
    s = SCENES[SCENE]
    subject = add(t, "lfs.ellipsoid_selection", -300, -220, Centre=offset(s["focus"], unit(s["up"]), 0.3),
                  Radii=(1.0, 1.0, 1.0), Falloff=0.4)
    sharpen = add(t, "lfs.sharpen", 0, 0, Amount=0.3, props={"keep_coverage": True})
    link(t, gin, "Geometry", sharpen, "Geometry")
    link(t, subject, "Selection", sharpen, "Selection")
    link(t, sharpen, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    before = lf.get_scene().get_node(target).splat_data().scaling_raw.mean().item()
    after = N.evaluated(target).splats.scaling.mean().item()
    return {"mean_log_scale_before": round(before, 4), "after": round(after, 4), "ok": after < before}

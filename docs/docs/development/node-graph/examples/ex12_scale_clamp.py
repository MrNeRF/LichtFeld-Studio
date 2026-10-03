"""Fix needle and pancake artifacts by limiting how stretched a Gaussian can be.

Extremely elongated Gaussians show up as streaks when the view moves off the
training cameras. Limiting the largest/smallest axis ratio to 8 keeps each
footprint but removes the streaks.
"""
import math

from common import *

SCENE = "bicycle"
TITLE = "Scale clamp"
ASPECT = 8.0


def build(target):
    t, gin, gout = new_graph(TITLE)
    clamp = add(t, "lfs.scale_clamp", 0, 0, Max_Aspect=ASPECT)
    link(t, gin, "Geometry", clamp, "Geometry")
    link(t, clamp, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def spread(scaling):
    return float((scaling.max(1) - scaling.min(1)).max().item())


def check(target):
    before = spread(lf.get_scene().get_node(target).splat_data().scaling_raw)
    after = spread(N.evaluated(target).splats.scaling)
    return {"max_log_aspect_before": round(before, 3), "after": round(after, 3),
            "limit": round(math.log(ASPECT), 3), "ok": after <= math.log(ASPECT) + 1e-3 < before}

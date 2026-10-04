"""Prepare a lightweight web version: SH degree 0 and half the Gaussians.

Degree-3 spherical harmonics are 45 of the 59 floats per Gaussian. Dropping
them and decimating by importance (opacity × size) gives roughly an 8x
smaller file that still reads well in a browser viewer.
"""
from common import *

SCENE = "bicycle"
TITLE = "Web export"


def build(target):
    t, gin, gout = new_graph(TITLE)
    degree = add(t, "lfs.set_sh_degree", 0, 0, Degree=0)
    decimate = add(t, "lfs.decimate", 200, 0, Keep_Fraction=0.5)
    link(t, gin, "Geometry", degree, "Geometry")
    link(t, degree, "Geometry", decimate, "Geometry")
    link(t, decimate, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    s = N.evaluated(target).splats
    before, after = stored_count(target), splat_count(target)
    floats_before = 59 * before
    floats_after = (14 + 3 * int(s.shN.shape[1])) * after
    return {"before": before, "after": after, "shN_coefficients": int(s.shN.shape[1]),
            "size_ratio": round(floats_before / max(floats_after, 1), 2),
            "ok": int(s.shN.shape[1]) == 0 and abs(after - before // 2) <= 0.01 * before}

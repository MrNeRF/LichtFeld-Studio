"""Grade the whole capture: deeper blacks, more contrast and saturation, warmer light.

Captures come out flat and slightly cool. A lift of the black point, more
contrast and saturation and a warm white balance give the photographic look. The affine part of the grade is applied
to every spherical-harmonic band, so the view-dependent colour is graded too.
"""
from common import *

SCENE = "garden"
TITLE = "Colour grade"


def build(target):
    t, gin, gout = new_graph(TITLE)
    grade = add(t, "lfs.colour_correct", 0, 0, Black_Point=0.03, Contrast=1.15, Saturation=1.25,
                Temperature=0.07, Exposure=0.1)
    link(t, gin, "Geometry", grade, "Geometry")
    link(t, grade, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def mean_rgb(splats):
    rgb = splats.sh0 * 0.28209479 + 0.5
    return [round(float(rgb[:, c].mean().item()), 4) for c in range(3)]


def check(target):
    stored = lf.get_scene().get_node(target).splat_data()
    before = [round(float((stored.sh0_raw.reshape([-1, 3])[:, c] * 0.28209479 + 0.5).mean().item()), 4) for c in range(3)]
    after = mean_rgb(N.evaluated(target).splats)
    return {"mean_rgb_before": before, "mean_rgb_after": after,
            "ok": after[0] - after[2] > before[0] - before[2] - 0.05}

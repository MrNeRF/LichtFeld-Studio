"""Turn the lawn autumn-coloured.

HSV Range picks the grass by the colour it is (a band of greens, not one exact
green), then a hue shift on that selection only turns it golden. The soft edge
of the selection keeps the transition into the flower beds natural.
"""
from common import *

SCENE = "garden"
TITLE = "Autumn grass"


def build(target):
    t, gin, gout = new_graph(TITLE)
    greens = add(t, "lfs.hsv_range", -300, -220, Hue=0.25, Hue_Range=0.07, Hue_Softness=0.04,
                 Saturation_Min=0.25, Saturation_Max=1.0, Value_Min=0.08, Value_Max=1.0, Softness=0.08)
    grade = add(t, "lfs.colour_correct", 0, 0, Hue_Shift=-38.0, Saturation=1.1)
    link(t, gin, "Geometry", grade, "Geometry")
    link(t, greens, "Selection", grade, "Selection")
    link(t, grade, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    rgb = N.evaluated(target).splats.sh0 * 0.28209479 + 0.5
    stored = lf.get_scene().get_node(target).splat_data().sh0_raw.reshape([-1, 3]) * 0.28209479 + 0.5
    green_before = float((stored[:, 1] - stored[:, 0]).mean().item())
    green_after = float((rgb[:, 1] - rgb[:, 0]).mean().item())
    return {"green_minus_red_before": round(green_before, 4), "after": round(green_after, 4),
            "ok": green_after < green_before}

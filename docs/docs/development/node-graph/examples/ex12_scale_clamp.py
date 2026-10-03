"""Shorten needle-shaped Gaussians without thickening flat surface splats.

Needles have one long axis and two short axes, and show up as streaks when the
view moves off the training cameras. Trained surfaces instead use flat discs
with two long axes and one short axis, so Scale Clamp leaves those alone by
default.
"""
import math

from common import *

SCENE = "bicycle"
TITLE = "Scale clamp"
ASPECT = 16.0


def build(target):
    t, gin, gout = new_graph(TITLE)
    clamp = add(t, "lfs.scale_clamp", 0, 0, Max_Aspect=ASPECT)
    link(t, gin, "Geometry", clamp, "Geometry")
    link(t, clamp, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    before = lf.get_scene().get_node(target).splat_data().scaling_raw
    after = N.evaluated(target).splats.scaling
    limit = math.log(ASPECT)

    def longest_over_middle(log_scale):
        ordered = log_scale.sort(1)[0]
        return ordered[:, 2] - ordered[:, 1]

    compliant = (longest_over_middle(before) <= limit).to("float32")
    unchanged = ((after - before).abs().sum(1) == 0).to("float32")
    expected_untouched = int(compliant.sum().item())
    untouched = int((compliant * unchanged).sum().item())
    maximum = float(longest_over_middle(after).max().item())
    return {"max_longest_middle_log_ratio": round(maximum, 3), "limit": round(limit, 3),
            "untouched": untouched, "expected_untouched": expected_untouched,
            "ok": maximum <= limit + 1e-4 and untouched == expected_untouched}

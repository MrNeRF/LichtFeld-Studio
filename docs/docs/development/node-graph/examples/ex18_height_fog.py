"""Add low-lying morning fog.

Height above the ground drives how strongly each Gaussian is pushed towards a
pale fog colour: dense near the lawn, gone at table height.
"""
from common import *

SCENE = "garden"
TITLE = "Height fog"


def build(target):
    t, gin, gout = new_graph(TITLE)
    h = height_field(t, SCENE)
    density = add(t, "lfs.map_range", -60, -300, From_Min=-1.45, From_Max=0.2, To_Min=0.5, To_Max=0.0,
                  props={"clamp": True})
    fog = add(t, "lfs.set_colour", 200, 0, Colour=(0.78, 0.82, 0.88), props={"clear_view_dependent": True})
    link(t, h, "Value", density, "Value")
    link(t, gin, "Geometry", fog, "Geometry")
    link(t, density, "Result", fog, "Selection")
    link(t, fog, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    rgb = N.evaluated(target).splats.sh0 * 0.28209479 + 0.5
    stored = lf.get_scene().get_node(target).splat_data().sh0_raw.reshape([-1, 3]) * 0.28209479 + 0.5
    before, after = stored.mean().item(), rgb.mean().item()
    return {"mean_brightness_before": round(before, 4), "after": round(after, 4), "ok": after > before}

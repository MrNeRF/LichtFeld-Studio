"""Give the lawn patches of autumn colour instead of one flat tint.

Noise supplies a slow, mottled pattern. The ramp turns that pattern into moss,
ochre and rust. A soft green-colour selection and a ground-height test keep the
table and upper flower beds out of the treatment.
"""
from common import *

SCENE = "garden"
TITLE = "Noise-driven autumn lawn"


def build(target):
    t, gin, gout = new_graph(TITLE)
    greens = add(t, "lfs.hsv_range", -650, -220, Hue=0.25, Hue_Range=0.08,
                 Hue_Softness=0.04, Saturation_Min=0.2, Value_Min=0.05, Softness=0.08)
    height = height_field(t, SCENE, -1250, -450)
    ground = add(t, "lfs.compare", -650, -450, props={"operation": "less_than"}, B=0.05)
    selection = add(t, "lfs.math", -400, -230, props={"operation": "multiply"})
    noise = add(t, "lfs.noise_texture", -650, 150, Scale=1.4, Detail=2.5, Roughness=0.55, Seed=7.0)
    ramp = add(t, "lfs.colour_ramp", -380, 150, props={"stops": [
        [0.0, 0.08, 0.13, 0.015, 1], [0.35, 0.28, 0.2, 0.025, 1],
        [0.6, 0.75, 0.27, 0.025, 1], [1.0, 0.32, 0.045, 0.008, 1]]})
    colour = add(t, "lfs.set_colour", -60, 0)
    link(t, height, "Value", ground, "A")
    link(t, greens, "Selection", selection, "A")
    link(t, ground, "Result", selection, "B")
    link(t, noise, "Fac", ramp, "Fac")
    link(t, ramp, "Colour", colour, "Colour")
    link(t, selection, "Value", colour, "Selection")
    link(t, gin, "Geometry", colour, "Geometry")
    link(t, colour, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    result = N.evaluated(target).splats
    before = lf.get_scene().get_node(target).splat_data().sh0_raw.reshape([-1, 3])
    delta = float((result.sh0 - before).abs().mean().item())
    warm = float((result.sh0[:, 0] - result.sh0[:, 1]).mean().item())
    original = float((before[:, 0] - before[:, 1]).mean().item())
    return {"mean_colour_change": delta, "red_minus_green_before": original,
            "red_minus_green_after": warm, "ok": delta > 0.005 and warm > original}

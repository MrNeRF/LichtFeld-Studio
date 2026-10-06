"""Colour pop: everything grey except the centrepiece.

A feathered ellipsoid around the vase, inverted, drives a full desaturation,
so the colour fades out with distance instead of stopping at a hard edge.
"""
from common import *

SCENE = "garden"
TITLE = "Colour pop"


def build(target):
    t, gin, gout = new_graph(TITLE)
    s = SCENES[SCENE]
    subject = add(t, "lfs.ellipsoid_selection", -400, -220, Centre=offset(s["focus"], unit(s["up"]), 0.25),
                  Radii=(0.9, 0.9, 0.9), Falloff=0.6)
    invert = add(t, "lfs.math", -200, -220, props={"operation": "subtract"}, A=1.0)
    grey = add(t, "lfs.colour_correct", 0, 0, Saturation=0.0)
    link(t, subject, "Selection", invert, "B")
    link(t, gin, "Geometry", grey, "Geometry")
    link(t, invert, "Value", grey, "Selection")
    link(t, grey, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    rgb = N.evaluated(target).splats.sh0 * 0.28209479 + 0.5
    chroma = rgb.max(1) - rgb.min(1)
    grey_fraction = float((chroma < 0.01).to("float32").mean().item())
    return {"grey_fraction": round(grey_fraction, 3), "ok": 0.5 < grey_fraction < 0.99}

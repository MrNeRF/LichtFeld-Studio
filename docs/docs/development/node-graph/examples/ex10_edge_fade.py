"""Fade the capture out softly at its edges for web embeds.

Unbounded captures end in a ragged wall of half-reconstructed Gaussians. Fading
opacity with distance from the subject turns that into a soft vignette.
"""
from common import *

SCENE = "garden"
TITLE = "Soft edge fade"
CAMERA = {"distance": 6.0, "height": 3.2, "azimuth": 20.0}


def build(target):
    t, gin, gout = new_graph(TITLE)
    d = horizontal_distance_field(t, SCENE)
    fade = add(t, "lfs.map_range", 200, -300, From_Min=4.5, From_Max=9.0, To_Min=1.0, To_Max=0.0,
               props={"clamp": True})
    opacity = add(t, "lfs.opacity", 200, -450)
    scaled = add(t, "lfs.math", 400, -360, props={"operation": "multiply"})
    set_opacity = add(t, "lfs.set_opacity", 600, 0)
    link(t, d, "Value", fade, "Value")
    link(t, fade, "Result", scaled, "A")
    link(t, opacity, "Opacity", scaled, "B")
    link(t, gin, "Geometry", set_opacity, "Geometry")
    link(t, scaled, "Value", set_opacity, "Opacity")
    link(t, set_opacity, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    after = N.evaluated(target).splats.opacity.sigmoid().mean().item()
    before = lf.get_scene().get_node(target).splat_data().opacity_raw.sigmoid().mean().item()
    return {"mean_opacity_before": round(before, 4), "after": round(after, 4), "ok": after < before}

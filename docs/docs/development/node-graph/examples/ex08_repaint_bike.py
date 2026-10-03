"""Repaint the bike, keeping its shading.

Recolour with "keep shading" carries each splat's own brightness into the new
colour, so the frame stays lit and shaded instead of becoming a flat cut-out,
(the white tubes are found by colour inside the bench-and-bike region, minus
the bench seat in front),
and fades the view-dependent colour that still describes the old paint.
"""
from common import *
from ex04_remove_bike import subject_selection

SCENE = "bicycle"
TITLE = "Repaint the bike"


def build(target):
    t, gin, gout = new_graph(TITLE)
    subject = subject_selection(t)
    side, _ = horizontal_axes(SCENE)
    depth = along_field(t, SCENE, side, -900, -700)
    not_seat = add(t, "lfs.compare", -300, -700, props={"operation": "less_than"}, B=0.15)
    above_seat = add(t, "lfs.compare", -300, -860, props={"operation": "greater_than"}, B=-0.6)
    link(t, height_field(t, SCENE, -900, -860), "Value", above_seat, "A")
    frame_zone = add(t, "lfs.boolean_math", -120, -780, props={"operation": "and"})
    link(t, not_seat, "Result", frame_zone, "A")
    link(t, above_seat, "Result", frame_zone, "B")
    bike = add(t, "lfs.boolean_math", -120, -500, props={"operation": "and"})
    link(t, subject, "Result", bike, "A")
    link(t, depth, "Value", not_seat, "A")
    link(t, frame_zone, "Result", bike, "B")
    bright = add(t, "lfs.hsv_range", -300, -420, Hue=0.0, Hue_Range=0.5, Saturation_Min=0.0, Saturation_Max=0.35,
                 Value_Min=0.55, Value_Max=1.0, Softness=0.08)
    frame = add(t, "lfs.math", 0, -300, props={"operation": "multiply"})
    paint = add(t, "lfs.recolour", 200, 0, Colour=(0.85, 0.12, 0.1), Weight=1.0,
                props={"keep_shading": True, "fade_view_dependent": True})
    link(t, bike, "Result", frame, "A")
    link(t, bright, "Selection", frame, "B")
    link(t, gin, "Geometry", paint, "Geometry")
    link(t, frame, "Value", paint, "Selection")
    link(t, paint, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    rgb = N.evaluated(target).splats.sh0 * 0.28209479 + 0.5
    redness = ((rgb[:, 0] - rgb[:, 2]) > 0.3).to("float32").sum().item()
    stored = lf.get_scene().get_node(target).splat_data().sh0_raw.reshape([-1, 3]) * 0.28209479 + 0.5
    red_before = ((stored[:, 0] - stored[:, 2]) > 0.3).to("float32").sum().item()
    return {"strongly_red_before": int(red_before), "after": int(redness), "ok": redness > red_before + 500}

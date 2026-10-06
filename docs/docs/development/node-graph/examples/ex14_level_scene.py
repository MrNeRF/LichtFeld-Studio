"""Level the scene and put the subject at the origin.

COLMAP leaves captures tilted and off-centre, which makes every later step
(placing objects, exporting to an engine, orbit cameras) awkward. One rotation
maps the capture's up axis to the viewer's up (data -Y; the viewer shows data
with Y and Z flipped) and a translation moves the table to (0,0,0).
"""
import math

from common import *

SCENE = "garden"
TITLE = "Level and centre"
FRAME_AFTER = {"up": (0.0, -1.0, 0.0), "focus": (0.0, 0.0, 0.0)}


def leveling(scene):
    """Rotation about X (degrees) taking the scene up to -Y, and the translation
    that then moves the focus to the origin."""
    ux, uy, uz = unit(SCENES[scene]["up"])
    angle = math.atan2(uz, -uy)
    c, s = math.cos(angle), math.sin(angle)
    fx, fy, fz = SCENES[scene]["focus"]
    rotated = (fx, fy * c - fz * s, fy * s + fz * c)
    return math.degrees(angle), tuple(-v for v in rotated)


def build(target):
    t, gin, gout = new_graph(TITLE)
    degrees, translation = leveling(SCENE)
    level = add(t, "lfs.transform_geometry", 0, 0, Rotation=(degrees, 0.0, 0.0), Translation=translation)
    link(t, gin, "Geometry", level, "Geometry")
    link(t, level, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    means = N.evaluated(target).splats.means
    near = ((means * means).sum(1) < 0.25).to("float32")
    centre = [(means[:, c] * near).sum().item() / max(near.sum().item(), 1) for c in range(3)]
    return {"mean_of_points_near_origin": [round(v, 3) for v in centre], "points_near_origin": int(near.sum().item()),
            "ok": near.sum().item() > 1000}

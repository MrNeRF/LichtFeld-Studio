"""Reveal the garden from the table outwards over two seconds.

The comparison radius is a normal sequencer track: scrub from 0 to 2 seconds
to watch the reveal. Scene Time also raises the opacity as the scene appears.
For a video, add two camera keyframes at those times and export the sequence;
the export waits for the node result at every frame before rendering it.
"""
from common import *

SCENE = "garden"
TITLE = "Animated garden reveal"
ANIMATED = True
_tree = None


def build(target):
    global _tree
    t, gin, gout = new_graph(TITLE)
    pos = add(t, "lfs.position", -680, -180)
    distance = add(t, "lfs.distance", -480, -180, Point=SCENES[SCENE]["focus"])
    radius = add(t, "lfs.compare", -250, -180, props={"operation": "less_than"}, B=0.2)
    radius.name = "Reveal radius"
    radius.keyframe_insert("B", time=0.0, value=0.2, easing=3)
    radius.keyframe_insert("B", time=2.0, value=8.0)
    take = add(t, "lfs.separate_geometry", 0, 0)
    clock = add(t, "lfs.scene_time", -470, 200)
    fade = add(t, "lfs.map_range", -240, 200, From_Min=0.0, From_Max=2.0, To_Min=0.35, To_Max=1.0)
    opacity = add(t, "lfs.set_opacity", 250, 0)
    link(t, pos, "Position", distance, "Vector")
    link(t, distance, "Distance", radius, "A")
    link(t, radius, "Result", take, "Selection")
    link(t, gin, "Geometry", take, "Geometry")
    link(t, clock, "Seconds", fade, "Value")
    link(t, fade, "Result", opacity, "Opacity")
    link(t, take, "Selection", opacity, "Geometry")
    link(t, opacity, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    _tree = t
    return t


def check(target):
    stored = lf.get_scene().get_node(target).splat_data()
    source = N.Geometry(points=N.Points(stored.means_raw, stored.sh0_raw.reshape([-1, 3])))
    counts = [int(N.evaluate_tree(_tree, source, time=time).points.positions.shape[0]) for time in (0.0, 1.0, 2.0)]
    return {"counts_at_0_1_2_seconds": counts, "ok": 0 < counts[0] < counts[1] < counts[2]}

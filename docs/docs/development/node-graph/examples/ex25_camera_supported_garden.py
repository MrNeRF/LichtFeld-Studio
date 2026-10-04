"""Keep what the garden cameras actually saw.

Load the garden camera poses as well as its PLY before running this example.
Coverage counts frustum hits, not occlusion: a surface behind another surface
can still count as covered. Three views is a gentle first cleanup threshold.
"""
from common import *

SCENE = "garden"
TITLE = "Keep what the cameras saw"
REQUIRES_CAMERAS = True


def build(target):
    if lf.get_scene().active_camera_count == 0:
        raise ValueError("Load the garden dataset cameras before this example")
    t, gin, gout = new_graph(TITLE)
    coverage = add(t, "lfs.camera_coverage", -430, -150)
    unsupported = add(t, "lfs.compare", -180, -150, props={"operation": "less_than"}, B=3.0)
    delete = add(t, "lfs.delete_geometry", 60, 0)
    link(t, coverage, "Count", unsupported, "A")
    link(t, unsupported, "Result", delete, "Selection")
    link(t, gin, "Geometry", delete, "Geometry")
    link(t, delete, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    before, after = stored_count(target), splat_count(target)
    return {"cameras": lf.get_scene().active_camera_count, "before": before,
            "after": after, "removed": before - after, "ok": 0 < after < before}

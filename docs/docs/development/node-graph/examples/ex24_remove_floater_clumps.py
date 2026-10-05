"""Clear the little islands floating above the bicycle capture.

Unlike an opacity threshold, Remove Clumps asks whether a splat belongs to a
large connected patch using each splat's local spacing. Restrict it to the sky
so the sparse spokes and leaves near the bike are left alone. Toggle Delete off
to inspect its Selection output.
"""
from common import *

SCENE = "bicycle"
TITLE = "Remove floater clumps"
CAMERA = {"distance": 7.0, "height": 1.7, "azimuth": 180.0, "target_lift": 3.5}


def build(target):
    t, gin, gout = new_graph(TITLE)
    height = height_field(t, SCENE, -600, -180)
    sky = add(t, "lfs.compare", 0, -180, props={"operation": "greater_than"}, B=4.0)
    clean = add(t, "lfs.remove_clumps", 220, 0, Radius=2.0, Min_Size=16)
    link(t, height, "Value", sky, "A")
    link(t, sky, "Result", clean, "Selection")
    link(t, gin, "Geometry", clean, "Geometry")
    link(t, clean, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    before, after = stored_count(target), splat_count(target)
    focus, up = SCENES[SCENE]["focus"], unit(SCENES[SCENE]["up"])
    def non_sky_count(means):
        # The eligibility mask must preserve every element below the sky plane.
        height = sum((means[:, axis] - focus[axis]) * up[axis] for axis in range(3))
        return int((height <= 4.0).sum().item())
    ground_before = non_sky_count(lf.get_scene().get_node(target).splat_data().means_raw)
    ground_after = non_sky_count(N.evaluated(target).splats.means)
    return {"before": before, "after": after, "removed": before - after,
            "non_sky_before": ground_before, "non_sky_after": ground_after,
            "ok": 0 < before - after < before * 0.25 and ground_before == ground_after}

"""Extract the bench with the bike as a standalone asset.

The same selection as the clean plate, but keeping the inside, then removing
the stray specks a hard cut leaves at its border - ready to export and drop
into another scene.
"""
from common import *
from ex04_remove_bike import subject_selection

SCENE = "bicycle"
TITLE = "Extract bench and bike"


def build(target):
    t, gin, gout = new_graph(TITLE)
    subject = subject_selection(t)
    keep = add(t, "lfs.separate_geometry", 200, 0)
    tidy = add(t, "lfs.remove_floaters", 400, 0, Min_Opacity=0.05, Isolation_Radius=3.0, Min_Neighbours=3,
               props={"relative_to_size": True})
    link(t, gin, "Geometry", keep, "Geometry")
    link(t, subject, "Result", keep, "Selection")
    link(t, keep, "Selection", tidy, "Geometry")
    link(t, tidy, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    before, after = stored_count(target), splat_count(target)
    return {"before": before, "after": after, "ok": 1000 < after < 0.3 * before}

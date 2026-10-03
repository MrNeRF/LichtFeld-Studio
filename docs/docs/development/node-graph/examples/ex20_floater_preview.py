"""Floater QA: see what a clean-up would remove before committing to it.

With preview on, Remove Floaters shows only its candidates, preserving their
colour, opacity and shape, so thresholds can be tuned by eye. Isolation
is judged relative to each Gaussian's own size, so the large, sparse
background is not mistaken for floaters.
"""
from common import *

SCENE = "garden"
TITLE = "Floater preview"


def build(target):
    t, gin, gout = new_graph(TITLE)
    clean = add(t, "lfs.remove_floaters", 0, 0, Min_Opacity=0.05, Isolation_Radius=3.0, Min_Neighbours=3,
                props={"relative_to_size": True, "preview": True})
    link(t, gin, "Geometry", clean, "Geometry")
    link(t, clean, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    candidates, stored = splat_count(target), stored_count(target)
    return {"would_remove": candidates, "of": stored, "ok": 0 < candidates < 0.1 * stored}

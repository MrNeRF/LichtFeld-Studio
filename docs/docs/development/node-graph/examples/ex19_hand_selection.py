"""Delete something you selected by hand - non-destructively.

Select with the viewport tools (brush, lasso, box) as usual, hide the
modifiers, press Capture on a Stored Selection node, show them again. The
selection is now part of the graph: the deletion can be switched off, edited
or undone at any time, and the stored splats are untouched. Here the vase on
the table is selected programmatically to stand in for a brush stroke.
"""
from common import *

SCENE = "garden"
TITLE = "Delete a hand selection"


def select_vase(target):
    """Stand-in for a brush stroke over the vase: above the table top and
    within 0.4 of the vase axis."""
    s = SCENES[SCENE]
    up, f = unit(s["up"]), s["focus"]
    means = lf.get_scene().get_node(target).splat_data().means_raw
    rel = [means[:, c] - f[c] for c in range(3)]
    h = sum(rel[c] * up[c] for c in range(3))
    flat2 = sum((rel[c] - h * up[c]) * (rel[c] - h * up[c]) for c in range(3))
    mask = ((h > 0.03).to("float32") * (flat2 < 0.4 ** 2).to("float32")).to("uint8")
    lf.get_scene().set_selection_mask(mask)
    return int(mask.to("float32").sum().item())


def means_of(target):
    return lf.get_scene().get_node(target).splat_data().means_raw


def build(target):
    t, gin, gout = new_graph(TITLE)
    stored = add(t, "lfs.stored_selection", -300, -200)
    delete = add(t, "lfs.delete_geometry", 0, 0)
    link(t, gin, "Geometry", delete, "Geometry")
    link(t, stored, "Selection", delete, "Selection")
    link(t, delete, "Geometry", gout, "Geometry")
    modifier = N.add_modifier(target, t)
    modifier.show_viewport = False
    select_vase(target)
    modifier.capture_selection(stored.name)
    modifier.show_viewport = True
    lf.get_scene().set_selection_mask((means_of(target)[:, 0] > 1e9).to("uint8"))
    return t


def check(target):
    before, after = stored_count(target), splat_count(target)
    return {"before": before, "after": after, "deleted": before - after, "ok": 1000 < before - after < 100000}

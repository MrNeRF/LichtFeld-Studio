"""Strip faint, oversized haze Gaussians.

Large, nearly transparent Gaussians are what makes a capture look milky when
you leave the original camera path. They carry almost no detail, so removing
them sharpens the whole scene at little cost.
"""
from common import *

SCENE = "garden"
TITLE = "Remove background haze"


def build(target):
    t, gin, gout = new_graph(TITLE)
    opacity = add(t, "lfs.opacity", -600, -200)
    faint = add(t, "lfs.compare", -420, -200, props={"operation": "less_than"}, B=0.15)
    scale = add(t, "lfs.scale", -600, -360)
    size = add(t, "lfs.vector_math", -420, -360, props={"operation": "length"})
    big = add(t, "lfs.compare", -240, -360, props={"operation": "greater_than"}, B=0.25)
    haze = add(t, "lfs.boolean_math", -60, -260, props={"operation": "and"})
    delete = add(t, "lfs.delete_geometry", 200, 0)
    link(t, opacity, "Opacity", faint, "A")
    link(t, scale, "Scale", size, "A")
    link(t, size, "Value", big, "A")
    link(t, faint, "Result", haze, "A")
    link(t, big, "Result", haze, "B")
    link(t, gin, "Geometry", delete, "Geometry")
    link(t, haze, "Result", delete, "Selection")
    link(t, delete, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    before, after = stored_count(target), splat_count(target)
    return {"before": before, "after": after, "removed": before - after, "ok": 0 < before - after < 0.3 * before}

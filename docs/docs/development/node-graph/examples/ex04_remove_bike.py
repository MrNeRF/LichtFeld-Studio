"""Remove the bench and the bike: a clean background plate.

A clean plate is what you composite new objects into. The bench and the bike
leaning on it are selected as a cylinder - horizontal distance from the
subject and a height band - so the lawn underneath stays.
"""
from common import *

SCENE = "bicycle"
TITLE = "Clean plate: remove bench and bike"


def band(t, value_node, low, high, x, y):
    """Bool output: low < value < high."""
    above = add(t, "lfs.compare", x, y, props={"operation": "greater_than"}, B=low)
    below = add(t, "lfs.compare", x, y - 120, props={"operation": "less_than"}, B=high)
    both = add(t, "lfs.boolean_math", x + 180, y - 60, props={"operation": "and"})
    link(t, value_node, "Value", above, "A")
    link(t, value_node, "Value", below, "A")
    link(t, above, "Result", both, "A")
    link(t, below, "Result", both, "B")
    return both


def subject_selection(t, x=-300, y=-220):
    """Bench and bike: a cylinder around them, from just above the lawn to
    above the handlebars."""
    near = add(t, "lfs.compare", x, y, props={"operation": "less_than"}, B=1.85)
    link(t, horizontal_distance_field(t, SCENE, x - 1100, y), "Value", near, "A")
    height = band(t, height_field(t, SCENE, x - 600, y - 300), -1.05, 0.7, x, y - 300)
    both = add(t, "lfs.boolean_math", x + 360, y - 150, props={"operation": "and"})
    link(t, near, "Result", both, "A")
    link(t, height, "Result", both, "B")
    return both


def build(target):
    t, gin, gout = new_graph(TITLE)
    subject = subject_selection(t)
    delete = add(t, "lfs.delete_geometry", 200, 0)
    link(t, gin, "Geometry", delete, "Geometry")
    link(t, subject, "Result", delete, "Selection")
    link(t, delete, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    before, after = stored_count(target), splat_count(target)
    return {"before": before, "after": after, "removed": before - after, "ok": 0 < before - after < 0.3 * before}

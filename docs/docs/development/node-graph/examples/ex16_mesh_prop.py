"""Put a mesh prop on the table as Gaussians.

A torus imported as a mesh is converted to splats, scaled and stood upright in
the capture's tilted frame, and joined into the garden next to the vase. The
mesh node stays the editable source; the splats follow it.
"""
import math

from common import *

SCENE = "garden"
TITLE = "Mesh prop on the table"
MESH = "torus"


def build(target):
    t, gin, gout = new_graph(TITLE)
    s = SCENES[SCENE]
    ux, uy, uz = unit(SCENES[SCENE]["up"])
    degrees = math.degrees(math.atan2(uz, uy))  # Rx(angle) maps +Y onto `up`
    up = unit(s["up"])
    _, across = horizontal_axes(SCENE)
    spot = offset(offset(s["focus"], up, 0.08), across, -0.75)
    info = add(t, "lfs.object_info", -600, -200, props={"object": MESH, "transform_space": "original"})
    to_splats = add(t, "lfs.mesh_to_splats", -400, -200)
    place = add(t, "lfs.transform_geometry", -200, -200, Rotation=(degrees, 0.0, 0.0), Scale=0.2,
                Translation=spot)
    join = add(t, "lfs.join_geometry", 100, 0)
    link(t, info, "Geometry", to_splats, "Geometry")
    link(t, to_splats, "Geometry", place, "Geometry")
    link(t, gin, "Geometry", join, "Geometry")
    link(t, place, "Geometry", join, "Geometry")
    link(t, join, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    before, after = stored_count(target), splat_count(target)
    return {"before": before, "after": after, "prop_splats": after - before, "ok": after > before}

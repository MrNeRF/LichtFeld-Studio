"""A modifier on a mesh: cut a model in half to show its cross-section.

Node graphs are not only for splats. On a mesh node the same Position and
Compare fields select vertices, and Delete Geometry removes every face that
touches them.
"""
from common import *

SCENE = "mesh"
TITLE = "Clip a mesh"
MESH = "torus"
VISIBLE = ["torus"]
FRAME = {"focus": (0.0, 0.0, 0.0), "up": (0.0, -1.0, 0.0)}
CAMERA = {"distance": 4.6, "height": 3.0}


def build(target):
    t, gin, gout = new_graph(TITLE)
    pos = add(t, "lfs.position", -500, -200)
    xyz = add(t, "lfs.separate_xyz", -320, -200)
    front = add(t, "lfs.compare", -140, -200, props={"operation": "greater_than"}, B=0.0)
    delete = add(t, "lfs.delete_geometry", 100, 0)
    link(t, pos, "Position", xyz, "Vector")
    link(t, xyz, "Z", front, "A")
    link(t, gin, "Geometry", delete, "Geometry")
    link(t, front, "Result", delete, "Selection")
    link(t, delete, "Geometry", gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    mesh = N.evaluated(target).mesh
    faces = int(mesh.indices.shape[0])
    return {"faces_after": faces, "ok": 0 < faces}

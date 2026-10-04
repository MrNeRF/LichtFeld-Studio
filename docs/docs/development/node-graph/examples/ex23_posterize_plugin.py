"""A node written in Python: posterize the bike scene.

`lfs.posterize` is registered by a Python plug-in through `lichtfeld.nodes`
and runs on `lf.Tensor`, so it works on every GPU backend like the built-in
nodes. Selection-aware stylisation in a few lines of Python.
"""
from common import *

SCENE = "bicycle"
TITLE = "Posterize (Python node)"


def build(target):
    t, gin, gout = new_graph(TITLE)
    poster = add(t, "lfs.posterize", 0, 0, Levels=5)
    out = [o["identifier"] for o in next(x for x in N.node_types() if x["id"] == "lfs.posterize")["outputs"]][0]
    link(t, gin, "Geometry", poster, "Geometry")
    link(t, poster, out, gout, "Geometry")
    N.add_modifier(target, t)
    return t


def check(target):
    rgb = N.evaluated(target).splats.sh0 * 0.28209479 + 0.5
    q = rgb * 4.0
    off_grid = ((q - q.round()).abs() > 1e-3).to("float32").mean().item()
    return {"fraction_off_5_levels": round(off_grid, 4), "ok": off_grid < 0.01}

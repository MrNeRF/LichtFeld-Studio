# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Generated geometry and composited-window reads for viewer integration tests."""

import base64
import io
import json
import math
import struct
import time

from test_api_camera_project_runtime import _editor
from test_render_capture_camera import _capture
from test_render_on_demand_idle import _call, _tool


def background_splats(path):
    """A dense generated sheet behind the mesh keeps both split panes renderable."""
    fields = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
    fields += ["opacity", "scale_0", "scale_1", "scale_2",
               "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 961\n"
    header += "".join(f"property float {field}\n" for field in fields) + "end_header\n"
    with path.open("wb") as output:
        output.write(header.encode())
        for y in range(31):
            for x in range(31):
                values = [(x - 15) / 5., (y - 15) / 5., -4., 0., 0., 1.]
                values += [(color - .5) / .28209479177387814 for color in [.2, .3, .5]]
                values += [10.] + [math.log(.22)] * 3 + [1., 0., 0., 0.]
                output.write(struct.pack("<17f", *values))

MESH_CODE = '"""Small generated geometry for viewer QA; execute in the app\'s Python editor."""\nimport lichtfeld as lf\nimport math\nfrom array import array\n\ncorners = [[-1, 0, -1], [1, 0, -1], [0, 0, 1], [0, 2, 0]]\nfaces = [(0, 3, 1), (1, 3, 2), (2, 3, 0), (0, 1, 2)]\npalette = [(1, .1, .1, 1), (.1, 1, .1, 1), (.1, .1, 1, 1), (.7, .7, .7, 1)]\nverts, norms, cols = [], [], []\nfor face, color in zip(faces, palette):\n    tri = [corners[i] for i in face]\n    a = [tri[1][i]-tri[0][i] for i in range(3)]\n    b = [tri[2][i]-tri[0][i] for i in range(3)]\n    normal = [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]\n    length = math.sqrt(sum(x*x for x in normal));normal = [x/length for x in normal]\n    verts.extend(tri); norms.extend([normal] * 3); cols.extend([color] * 3)\nfloor = [[-3, -.05, -3], [-3, -.05, 3], [3, -.05, 3], [-3, -.05, -3], [3, -.05, 3], [3, -.05, -3]]\nverts.extend(floor); norms.extend([[0, 1, 0]] * 6); cols.extend([[.55, .55, .55, 1]] * 6)\nif globals().get(\'QA_FLOOR_FIRST\', False):\n    verts = verts[-6:] + verts[:-6]\n    norms = norms[-6:] + norms[:-6]\n    cols = cols[-6:] + cols[:-6]\ndef tensor(values, dtype):\n    flat = array(dtype, [v for row in values for v in row])\n    shaped = memoryview(flat).cast(\'B\').cast(dtype, (len(values), len(values[0])))\n    return lf.Tensor.from_numpy(shaped)\nnode = lf.get_scene().add_mesh(globals().get(\'QA_MESH_NAME\', \'QA_mesh\'), tensor(verts, \'f\'),\n    tensor([list(range(i,i+3)) for i in range(0,len(verts),3)], \'i\'),\n    tensor(cols, \'f\'), tensor(norms, \'f\'))\nlf.set_node_visualizer_world_transform(globals().get(\'QA_MESH_NAME\', \'QA_mesh\'),\n    [1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.], record_history=False)\nprint(\'Generated mesh node\', node, \'vertices\', len(verts))\n'


def setup_mesh(endpoint):
    # Mesh vertices are pinned into visualizer space below. A loaded splat keeps
    # an identity dataset transform, and the visualizer flips that to diag(1,-1,-1),
    # so a sheet authored at z=-4 is drawn at visualizer z=+4, between the usual
    # camera and the mesh. Pin splat nodes first so authored z stays put.
    _editor(endpoint, """
import lichtfeld as lf
identity = [1.,0.,0.,0., 0.,1.,0.,0., 0.,0.,1.,0., 0.,0.,0.,1.]
for node in lf.get_scene().get_nodes():
    if node.splat_data() is not None:
        lf.set_node_visualizer_world_transform(node.name, identity, record_history=False)
""")
    _editor(endpoint, MESH_CODE)
    _tool(endpoint, "render_settings_set", {
        "show_grid": False, "show_coord_axes": False, "show_pivot": False, "render_scale": 1.,
    })
    _editor(endpoint, "settings=lf.get_render_settings(); settings.mesh_backface_culling=False; settings.mesh_wireframe=False; settings.mesh_shadow_enabled=False")


def viewport_image(endpoint, directory, name, require_green=True):
    from PIL import Image

    path = directory / "bounds.json"
    _editor(endpoint, f"import json; open({str(path)!r}, 'w').write(json.dumps(lf.selection.get_viewport_bounds()))")
    x, y, width, height = map(round, json.loads(path.read_text()))
    for index in range(8):
        time.sleep(.3)
        _capture(endpoint, {})
        result = _call(endpoint, "tools/call", {"name": "render_capture_window", "arguments": {}})
        content = next(item for item in result["content"] if item["type"] == "image")
        window = Image.open(io.BytesIO(base64.b64decode(content["data"]))).convert("RGB")
        image = window.crop((x, y, x + width, y + height))
        core = image.crop((80, 160, image.width - 35, image.height - 50))
        green_pixels = sum(g > 2 * max(r, b) for r, g, b in core.getdata())
        if (not require_green or green_pixels > 100) and index >= 2:
            break
    window.save(directory / (name + "-window.png"))
    image.save(directory / (name + ".png"))
    if require_green:
        assert green_pixels > 100, "Generated mesh did not reach the viewport"
    return image


def difference(first, second):
    from PIL import ImageChops, ImageStat

    assert first.size == second.size
    return max(ImageStat.Stat(ImageChops.difference(first, second)).mean)

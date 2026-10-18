# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""A mixed comparison uses voxel rendering only for the point-cloud side."""

from pathlib import Path
import os
import base64
import io
import time

import pytest

from test_api_camera_project_runtime import _editor, camera_runtime
from test_mesh_split_depth_runtime import _front_splats
from test_render_on_demand_idle import _tool, _call

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def _cloud(path):
    points = [(x * .15, y * .15, 0.) for x in range(-6, 7) for y in range(-6, 7)]
    path.write_text('ply\nformat ascii 1.0\nelement vertex ' + str(len(points)) + '\n'
                    'property float x\nproperty float y\nproperty float z\n'
                    'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n' +
                    ''.join(f'{x} {y} {z} 0 255 0\n' for x, y, z in points))


def _window(endpoint):
    from PIL import Image

    # Captures consume the last presented frame; allow the queued settings
    # update to pass through the rotating output slots before measuring.
    for _ in range(4):
        result = _call(endpoint, 'tools/call', {'name': 'render_capture_window', 'arguments': {}})
    block = next(item for item in result['content'] if item['type'] == 'image')
    return Image.open(io.BytesIO(base64.b64decode(block['data']))).convert('RGB')


def _color_counts(image):
    # Avoid the toolbar and orientation gizmo at the top of the viewport.
    pixels = image.crop((90, 180, image.width - 35, image.height - 55)).getdata()
    red = green = 0
    for r, g, b in pixels:
        red += r > 100 and r > 2 * max(g, b)
        green += g > 240 and r < 20 and b < 20
    return red, green


@pytest.mark.parametrize('backend', ['3dgs', '3dgut'])
@pytest.mark.parametrize('cloud_first', [True, False])
def test_mixed_comparison_respects_voxel_size(tmp_path, backend, cloud_first, camera_runtime):
    pictures = Path(os.environ.get('LFS_TEST_ARTIFACT_DIR', tmp_path)) / (backend + ('-cloud-first' if cloud_first else '-splat-first'))
    pictures.mkdir(parents=True, exist_ok=True)
    cloud = tmp_path / 'green_cloud.ply'
    splat = tmp_path / 'red_splat.ply'
    _cloud(cloud)
    _front_splats(splat, z=0.)
    endpoint = camera_runtime
    paths = [str(path) for path in ((cloud, splat) if cloud_first else (splat, cloud))]
    _editor(endpoint, f"import lichtfeld as lf; lf.load_files({paths!r}, discard_changes=True)")
    job = _tool(endpoint, 'runtime_job_wait', {'job_id': 'import.dataset', 'timeout_ms': 30000})
    assert job.get('success'), job
    _editor(endpoint, '''
import lichtfeld as lf
identity=[1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.]
for node in lf.get_scene().get_nodes():
    if node.name in ('green_cloud', 'red_splat'):
        lf.set_node_visualizer_world_transform(node.name, identity, record_history=False)
lf.select_nodes([])
assert lf.get_scene().get_node('green_cloud') is not None
assert lf.get_scene().get_node('red_splat') is not None
''')
    _tool(endpoint, 'render_settings_set', {
        'point_cloud_mode': False, 'split_view_mode': 1, 'split_position': .5,
        'show_grid': False, 'show_camera_frustums': False, 'show_coord_axes': False,
        'raster_backend': backend, 'voxel_size': .015, 'render_scale': 1.,
    })
    _tool(endpoint, 'camera_set_view', {'eye': [0., 0., 6.], 'target': [0., 0., 0.]})
    small = _window(endpoint)
    small.save(pictures / 'small.png')
    _tool(endpoint, 'render_settings_set', {'voxel_size': .12})
    large = _window(endpoint)
    large.save(pictures / 'large.png')
    red_small, green_small = _color_counts(small)
    red_large, green_large = _color_counts(large)
    print(f"{backend}: voxel .015 -> .12; green {green_small} -> {green_large}; red {red_small} -> {red_large}")
    assert green_small > 20, (red_small, green_small)
    assert green_large > green_small * 3, (green_small, green_large)
    assert red_small > 1000
    assert red_large == pytest.approx(red_small, rel=.01), (red_small, red_large)
    _editor(endpoint, '''
assert lf.get_scene().get_node('green_cloud').point_cloud() is not None
assert lf.get_scene().get_node('green_cloud').splat_data() is None
assert lf.get_scene().get_node('red_splat').splat_data() is not None
''')
    project = tmp_path / 'mixed.licht'
    _tool(endpoint, 'project_save_as', {'path': str(project)})
    _tool(endpoint, 'project_open', {'path': str(project), 'discard_changes': True})
    deadline = time.monotonic() + 20
    while _tool(endpoint, 'project_get_info').get('hydration_state') != 'complete':
        assert time.monotonic() < deadline, 'mixed project did not finish hydrating'
        time.sleep(.05)
    _editor(endpoint, "assert lf.get_scene().get_node('green_cloud').point_cloud() is not None; "
                      "assert lf.get_scene().get_node('red_splat').splat_data() is not None")

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
    _editor(endpoint, "assert lf.get_scene().get_node('green_cloud').splat_data() is not None")
    # Normal view keeps the previous editable Gaussian representation. Separate
    # the models so an opaque red sheet cannot hide the green cloud.
    _editor(endpoint, """
cloud_transform = identity.copy(); cloud_transform[13] = -1.5
splat_transform = identity.copy(); splat_transform[13] = 1.5
lf.set_node_visualizer_world_transform('green_cloud', cloud_transform, record_history=False)
lf.set_node_visualizer_world_transform('red_splat', splat_transform, record_history=False)
""")
    _tool(endpoint, 'render_settings_set', {
        'split_view_mode': 0, 'point_cloud_mode': False, 'show_grid': False,
        'show_camera_frustums': False, 'show_coord_axes': False,
        'raster_backend': backend, 'render_scale': 1.,
    })
    _tool(endpoint, 'camera_set_view', {'eye': [0., 0., 10.], 'target': [0., 0., 0.]})
    normal = _window(endpoint)
    normal.save(pictures / 'normal.png')
    pixels = normal.crop((90, 180, normal.width - 35, normal.height - 55)).getdata()
    normal_green = sum(g > 80 and g > 2 * max(r, b) for r, g, b in pixels)
    assert normal_green > 20, normal_green
    assert _color_counts(normal)[0] > 1000
    # Both existing screen-selection tools must still include all raw points.
    _tool(endpoint, 'selection_brush', {'x': 0., 'y': 0., 'radius': 4 * max(normal.size), 'camera_index': -1})
    assert _tool(endpoint, 'selection_get', {'max_indices': 0})['selected_count'] == 1130
    _editor(endpoint, "lf.get_scene().clear_selection()")
    assert _tool(endpoint, 'selection_get', {'max_indices': 0})['selected_count'] == 0
    _tool(endpoint, 'selection_lasso', {'camera_index': -1, 'points': [
        [-normal.width, -normal.height], [2 * normal.width, -normal.height],
        [2 * normal.width, 2 * normal.height], [-normal.width, 2 * normal.height],
    ]})
    assert _tool(endpoint, 'selection_get', {'max_indices': 0})['selected_count'] == 1130
    _editor(endpoint, "lf.get_scene().clear_selection()")
    assert _tool(endpoint, 'selection_get', {'max_indices': 0})['selected_count'] == 0
    _editor(endpoint, """
for name in ('green_cloud', 'red_splat'):
    lf.set_node_visualizer_world_transform(name, identity, record_history=False)
""")
    _tool(endpoint, 'render_settings_set', {
        'point_cloud_mode': False, 'split_view_mode': 1, 'split_position': .5,
        'show_grid': False, 'show_camera_frustums': False, 'show_coord_axes': False,
        'raster_backend': backend, 'voxel_size': .015, 'render_scale': 1.,
    })
    _tool(endpoint, 'camera_set_view', {'eye': [0., 0., 6.], 'target': [0., 0., 0.]})
    small = _window(endpoint)
    small.save(pictures / 'small.png')
    expected_left, expected_right = (('green_cloud', 'red_splat') if cloud_first
                                     else ('red_splat', 'green_cloud'))
    _editor(endpoint, f"info = lf.ui.get_split_view_info(); "
                      f"assert (info['left_name'], info['right_name']) == {(expected_left, expected_right)!r}, info")

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
    # The native point path must show the selection before destructive edits.
    _tool(endpoint, 'ui_tool_invoke', {'tool_id': 'builtin.select', 'submode_id': 'rectangle'})
    cloud_offset = 0 if cloud_first else 961
    _editor(endpoint, f"lf.select_node('green_cloud'); "
                      f"lf.get_scene().set_selection(list(range({cloud_offset}, {cloud_offset + 169})))")
    assert _tool(endpoint, 'selection_get', {'max_indices': 0})['selected_count'] == 169
    highlighted = _window(endpoint)
    highlighted.save(pictures / 'selected-points.png')
    assert _color_counts(highlighted)[1] < _color_counts(large)[1] / 2
    _editor(endpoint, "lf.get_scene().clear_selection(); lf.select_nodes([])")
    _tool(endpoint, 'ui_tool_clear_active', {'clear_submode': True})

    _editor(endpoint, '''
assert lf.get_scene().get_node('green_cloud').point_cloud() is None
assert lf.get_scene().get_node('green_cloud').splat_data() is not None
assert lf.get_scene().get_node('red_splat').splat_data() is not None
''')
    exported = tmp_path / 'cloud.spz'
    _editor(endpoint, f"lf.io.save_spz(lf.get_scene().get_node('green_cloud').splat_data(), {str(exported)!r})")
    assert exported.is_file() and exported.stat().st_size > 0
    exported_sog = tmp_path / 'cloud.sog'
    _editor(endpoint, f"lf.io.save_sog(lf.get_scene().get_node('green_cloud').splat_data(), {str(exported_sog)!r}, use_gpu=False, kmeans_iterations=1)")
    assert exported_sog.is_file() and exported_sog.stat().st_size > 0
    project = tmp_path / 'mixed.licht'
    _tool(endpoint, 'project_save_as', {'path': str(project)})
    _tool(endpoint, 'project_open', {'path': str(project), 'discard_changes': True})
    deadline = time.monotonic() + 20
    while _tool(endpoint, 'project_get_info').get('hydration_state') != 'complete':
        assert time.monotonic() < deadline, 'mixed project did not finish hydrating'
        time.sleep(.05)
    _editor(endpoint, "assert lf.get_scene().get_node('green_cloud').splat_data() is not None; "
                      "assert lf.get_scene().get_node('red_splat').splat_data() is not None")

    # The raw-source hint survives embedded project payloads and hydration.
    _tool(endpoint, 'render_settings_set', {'voxel_size': .015})
    reopened_small = _color_counts(_window(endpoint))[1]
    _tool(endpoint, 'render_settings_set', {'voxel_size': .12})
    reopened_large = _color_counts(_window(endpoint))[1]
    assert reopened_large > reopened_small * 3, (reopened_small, reopened_large)

    # Group order must take precedence over model insertion/render-slot order.
    _editor(endpoint, f"""
scene = lf.get_scene()
first = scene.add_group('First comparison group')
second = scene.add_group('Second comparison group')
assert scene.reparent(scene.get_node({expected_right!r}).id, first)
assert scene.reparent(scene.get_node({expected_left!r}).id, second)
""")
    _window(endpoint).save(pictures / 'reordered.png')
    _editor(endpoint, f"info = lf.ui.get_split_view_info(); "
                      f"assert (info['left_name'], info['right_name']) == {(expected_right, expected_left)!r}, info")


def test_native_single_cloud_comparison_falls_back_to_normal_view(tmp_path, camera_runtime):
    cloud = tmp_path / 'native_green.ply'
    _cloud(cloud)
    endpoint = camera_runtime
    _editor(endpoint, f"""
import lichtfeld as lf
means, colors = lf.io.load_point_cloud({str(cloud)!r})
assert lf.get_scene().add_point_cloud('Native cloud', means, colors) >= 0
""")
    _tool(endpoint, 'render_settings_set', {
        'split_view_mode': 1, 'point_cloud_mode': False, 'voxel_size': .12,
        'show_grid': False, 'show_coord_axes': False, 'show_camera_frustums': False,
    })
    _tool(endpoint, 'camera_set_view', {'eye': [0., 0., 6.], 'target': [0., 0., 0.]})
    image = _window(endpoint)
    image.save(tmp_path / 'native-single-cloud.png')
    assert _color_counts(image)[1] > 1000
    _editor(endpoint, "assert not lf.ui.get_split_view_info()['enabled']")
    _editor(endpoint, "assert lf.get_scene().get_node('Native cloud').point_cloud() is not None")


@pytest.mark.parametrize('selected_name', ['left_splat', 'right_splat'])
def test_splat_point_mode_comparison_keeps_selection_visible(tmp_path, selected_name, camera_runtime):
    paths = [tmp_path / 'left_splat.ply', tmp_path / 'right_splat.ply']
    for path in paths:
        _front_splats(path, z=0.)
    endpoint = camera_runtime
    _editor(endpoint, f"import lichtfeld as lf; lf.load_files({[str(p) for p in paths]!r}, discard_changes=True)")
    assert _tool(endpoint, 'runtime_job_wait', {'job_id': 'import.dataset', 'timeout_ms': 30000}).get('success')
    _editor(endpoint, """
identity=[1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.,0.,0.,0.,0.,1.]
for name in ('left_splat', 'right_splat'):
    lf.set_node_visualizer_world_transform(name, identity, record_history=False)
lf.select_nodes([])
""")
    _tool(endpoint, 'render_settings_set', {
        'split_view_mode': 1, 'point_cloud_mode': True, 'split_position': .5,
        'voxel_size': .04, 'show_grid': False, 'show_camera_frustums': False,
        'show_coord_axes': False, 'render_scale': 1.,
    })
    _tool(endpoint, 'camera_set_view', {'eye': [0., 0., 6.], 'target': [0., 0., 0.]})
    _tool(endpoint, 'ui_tool_invoke', {'tool_id': 'builtin.select', 'submode_id': 'rectangle'})
    baseline = _window(endpoint)
    baseline.save(tmp_path / 'point-mode-before.png')
    assert _color_counts(baseline)[0] > 1000
    offset = 0 if selected_name == 'left_splat' else 961
    _editor(endpoint, f"lf.select_node({selected_name!r}); lf.get_scene().set_selection(list(range({offset}, {offset + 961})))")
    assert _tool(endpoint, 'selection_get', {'max_indices': 0})['selected_count'] == 961
    selected = _window(endpoint)
    selected.save(tmp_path / 'point-mode-selected.png')
    # The untouched panel stays red. The default selection tint is light red,
    # so count its raised green/blue channels rather than a hue change.
    red_before = _color_counts(baseline)[0]
    red_after = _color_counts(selected)[0]
    assert red_after > red_before * .2, (red_before, red_after)
    def highlighted_pixels(image):
        return sum(r > 180 and 25 < g < 120 and 25 < b < 120 and abs(g - b) < 10
                   for r, g, b in image.crop((90, 180, image.width - 35, image.height - 55)).getdata())
    assert highlighted_pixels(selected) > highlighted_pixels(baseline) + 1000

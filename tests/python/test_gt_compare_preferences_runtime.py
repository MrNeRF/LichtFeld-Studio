# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Native GT repeat preferences must clamp and persist in an isolated GUI."""

import json
import os

import pytest

from test_api_camera_project_runtime import camera_runtime, _editor

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_gt_repeat_preferences_persist_and_clamp(camera_runtime, tmp_path):
    _editor(camera_runtime, """
import lichtfeld as lf
assert lf.ui.get_gt_comparison_key_repeat() is True
assert lf.ui.get_gt_comparison_repeat_rate() == 5.0
lf.ui.set_gt_comparison_key_repeat(False)
lf.ui.set_gt_comparison_repeat_rate(200)
assert lf.ui.get_gt_comparison_repeat_rate() == 20.0
lf.ui.set_gt_comparison_repeat_rate(-2)
assert lf.ui.get_gt_comparison_repeat_rate() == 1.0
lf.ui.set_gt_comparison_repeat_rate(float('nan'))
assert lf.ui.get_gt_comparison_repeat_rate() == 5.0
lf.ui.set_gt_comparison_repeat_rate(7)
lf.ui.set_gt_comparison_repeat_rate(12, persist=False)
assert lf.ui.get_gt_comparison_repeat_rate() == 12.0
""")
    path = tmp_path / "home/.lichtfeld/config/preferences.json"
    preferences = json.loads(path.read_text())
    assert preferences["gt_comparison_key_repeat"] is False
    assert preferences["gt_comparison_repeat_rate"] == 7.0
    _editor(camera_runtime, "import lichtfeld as lf\nlf.ui.set_gt_comparison_repeat_rate(12)")
    assert json.loads(path.read_text())["gt_comparison_repeat_rate"] == 12.0


@pytest.mark.skipif(os.environ.get('LFS_TEST_WAYLAND_KEYS') != '1',
                    reason='requires an interactive Hyprland session and wtype')
def test_gt_repeat_held_arrows(camera_runtime, tmp_path):
    import struct
    import subprocess
    import time
    import zlib

    from test_render_on_demand_idle import _tool

    dataset = tmp_path / 'dataset'
    images = dataset / 'images'
    sparse = dataset / 'sparse/0'
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)

    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))

    png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', 64, 64, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress((b'\0' + bytes([90, 110, 130]) * 64) * 64))
           + chunk(b'IEND', b''))
    for index in range(32):
        (images / f'{index}.png').write_bytes(png)
    (sparse / 'cameras.txt').write_text('1 PINHOLE 64 64 50 50 32 32\n')
    (sparse / 'images.txt').write_text(''.join(
        f'{i + 1} 1 0 0 0 {i * .001} 0 4 1 {i}.png\n\n' for i in range(32)))
    (sparse / 'points3D.txt').write_text(''.join(
        f'{y * 10 + x + 1} {(x - 4.5) * .1} {(y - 4.5) * .1} 0 200 100 50 0\n'
        for y in range(10) for x in range(10)))
    _tool(camera_runtime, 'scene_load_dataset', {
        'path': str(dataset), 'output_path': str(tmp_path / 'output'),
        'min_track_length': 0, 'max_iterations': 3,
    })
    job = _tool(camera_runtime, 'runtime_job_wait', {
        'job_id': 'import.dataset', 'until': 'inactive', 'timeout_ms': 20000,
    })
    assert not job.get('timed_out') and job.get('status') != 'failed', job
    splat = tmp_path / 'splat.ply'
    fields = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2',
              'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    with splat.open('wb') as handle:
        handle.write(('ply\nformat binary_little_endian 1.0\nelement vertex 100\n'
                      + ''.join(f'property float {field}\n' for field in fields)
                      + 'end_header\n').encode())
        for y in range(10):
            for x in range(10):
                handle.write(struct.pack('<17f', (x - 4.5) * .1, (y - 4.5) * .1, 0,
                                         0, 0, 0, 1, .2, .1, 3, -3, -3, -3, 1, 0, 0, 0))
    _editor(camera_runtime, f"""
import lichtfeld as lf
import os
s = lf.io.load({str(splat)!r}).splat_data
lf.get_scene().add_splat('synthetic', s.means_raw, s.sh0_raw, s.shN_raw,
                        s.scaling_raw, s.rotation_raw, s.opacity_raw, s.active_sh_degree, s.scene_scale)
lf.ui.go_to_camera_view(0)
lf.ui.toggle_gt_comparison()
assert lf.ui.is_gt_comparison_active()
print(os.getpid())
""")
    pid = int(_editor(camera_runtime, 'print(os.getpid())').strip())
    desktop_env = os.environ.copy()
    if not desktop_env.get('HYPRLAND_INSTANCE_SIGNATURE'):
        instances = json.loads(subprocess.check_output(['hyprctl', 'instances', '-j']))
        desktop_env['HYPRLAND_INSTANCE_SIGNATURE'] = next(
            item['instance'] for item in instances if item['wl_socket'] == desktop_env['WAYLAND_DISPLAY'])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        clients = json.loads(subprocess.check_output(['hyprctl', 'clients', '-j'], env=desktop_env))
        if any(client['pid'] == pid for client in clients):
            break
        time.sleep(.1)
    else:
        pytest.fail(f'test window {pid} did not map: {clients}')
    subprocess.run(['hyprctl', 'dispatch', f'hl.dsp.focus({{window = "pid:{pid}"}})'],
                   env=desktop_env, check=True, capture_output=True)
    time.sleep(.2)

    def current():
        return int(_editor(camera_runtime, 'print(lf.ui.get_current_camera_id())').strip())

    # Keep one virtual keyboard alive. Recreating it between holds can cause
    # compositor focus changes that do not occur with a physical keyboard.
    cases = [
        ('Right', 1100, 'lf.ui.set_gt_comparison_key_repeat(False)'),
        ('Right', 1100, 'lf.ui.set_gt_comparison_key_repeat(True)\nlf.ui.set_gt_comparison_repeat_rate(2)'),
        ('Right', 1100, 'lf.ui.set_gt_comparison_repeat_rate(8)'),
        ('Left', 1100, ''),
        ('Right', 80, ''),
        ('Right', 1100, 'lf.ui.toggle_gt_comparison()\nassert not lf.ui.is_gt_comparison_active()'),
        ('Right', 1100, 'lf.ui.toggle_gt_comparison()\nassert lf.ui.is_gt_comparison_active()'),
    ]
    args = ['wtype']
    for key, duration, _ in cases:
        args += ['-s', '700', '-P', key, '-s', str(duration), '-p', key, '-s', '700']
    _editor(camera_runtime, cases[0][2])
    subprocess.run(['hyprctl', 'dispatch', f'hl.dsp.focus({{window = "pid:{pid}"}})'],
                   env=desktop_env, check=True, capture_output=True)
    started = time.monotonic()
    keyboard = subprocess.Popen(args)
    elapsed = 0.0
    advances = []
    try:
        for index, (key, duration, configure) in enumerate(cases):
            if index and configure:
                _editor(camera_runtime, configure)
            before = current()
            if index == 6:
                # Switching comparison off during a hold must not stop browsing.
                time.sleep(max(0, started + elapsed + 1.3 - time.monotonic()))
                _editor(camera_runtime, 'lf.ui.toggle_gt_comparison()\nassert not lf.ui.is_gt_comparison_active()')
            elapsed += .7 + duration / 1000
            time.sleep(max(0, started + elapsed + .15 - time.monotonic()))
            after = current()
            time.sleep(.2)
            assert current() == after, 'navigation continued after release'
            advances.append((after - before) % 32 if key == 'Right' else (before - after) % 32)
            elapsed += .7
        keyboard.wait(timeout=3)
    finally:
        if keyboard.poll() is None:
            keyboard.terminate()
            keyboard.wait(timeout=3)
    print('camera advances:', advances, flush=True)
    assert advances[0] == 1
    assert 2 <= advances[1] <= 4
    assert advances[2] > advances[1]
    assert 5 <= advances[2] <= 9
    assert 5 <= advances[3] <= 9
    assert advances[4] == 1
    assert 5 <= advances[5] <= 9
    assert 5 <= advances[6] <= 9

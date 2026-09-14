# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

import benchmark_camera_pose_matrix as matrix


def test_plan_pairs_only_differ_in_pose_switch_and_output(tmp_path):
    runs = matrix.plan(Path('app.exe'), Path('bicycle'), Path('statue'), tmp_path)
    assert len(runs) == 12
    assert len({r['folder'] for r in runs}) == 12
    for off, on in zip(runs[::2], runs[1::2]):
        assert off['strategy'] == on['strategy']
        assert '--refine-camera-poses' not in off['command']
        assert on['command'][-5:] == ['--refine-camera-poses', '--camera-pose-start-step', '500',
                                     '--camera-pose-end-percent', '50']
        def normalized(run):
            command = run['command'][:]
            for flag in ('-o', '--log-file'):
                command[command.index(flag) + 1] = 'OUTPUT'
            return command
        assert normalized(off) == normalized(on)[:-5]
        assert ('--undistort' in off['command']) == (off['dataset'] == 'statue')
        assert off['command'][off['command'].index('--sh-degree') + 1] == '0'
    assert {r['strategy'] for r in runs} == {'mrnf', 'mcmc', 'igs+'}


def write_result(folder, support='69/69', freeze=True, error=False):
    folder.mkdir(parents=True, exist_ok=True)
    log = f'Camera pose SfM guard: {support} movable cameras\nHeadless training completed\n'
    if freeze:
        log += 'Camera pose refinement frozen at iteration 5000 (scheduled stop 5000)\n'
    if error:
        log += '[error] failed\n'
    log += '[warning] image rounded\n'
    (folder / 'training.log').write_text(log, encoding='utf-8')
    (folder / 'metrics.csv').write_text('iteration,psnr,ssim,lpips,num_gaussians\n10000,23,0.6,0.3,100\n')
    views = folder / 'eval_step_10000'
    views.mkdir()
    (views / 'per_image_metrics.csv').write_text('image_name,psnr,ssim,lpips\na.jpg,23,0.6,0.3\n')


def test_result_and_comparison(tmp_path):
    write_result(tmp_path / 'on')
    on = matrix.read_result(tmp_path / 'on', 'on')
    assert on['warnings'] == ['image rounded']
    assert on['support'] == {'protected': 69, 'movable': 69}
    on['seconds'] = 12
    off = copy.deepcopy(on)
    off['seconds'] = 10
    off['metrics']['psnr'] -= 1
    off['views']['a.jpg']['psnr'] -= 1
    result = matrix.compare(off, on)
    assert result['psnr_delta'] == 1
    assert result['psnr_views_improved'] == 1
    assert result['overhead_percent'] == pytest.approx(20)
    off['views']['different.jpg'] = off['views'].pop('a.jpg')
    with pytest.raises(ValueError, match='views differ'):
        matrix.compare(off, on)


@pytest.mark.parametrize('kwargs,reason', [
    ({'support': '0/69'}, 'geometric support'),
    ({'freeze': False}, 'pose freeze'),
    ({'error': True}, 'failed training'),
])
def test_invalid_run_rejected(tmp_path, kwargs, reason):
    write_result(tmp_path / 'on', **kwargs)
    with pytest.raises(ValueError, match=reason):
        matrix.read_result(tmp_path / 'on', 'on')


@pytest.mark.parametrize('rows', [
    'a.jpg,nan,0.6,0.3\n',
    'a.jpg,23,0.6,0.3\na.jpg,23,0.6,0.3\n',
    '',
])
def test_invalid_view_metrics_rejected(tmp_path, rows):
    write_result(tmp_path / 'on')
    (tmp_path / 'on/eval_step_10000/per_image_metrics.csv').write_text('image_name,psnr,ssim,lpips\n' + rows)
    with pytest.raises(ValueError):
        matrix.read_result(tmp_path / 'on', 'on')


def test_live_output_and_child_error(tmp_path, capsys):
    with pytest.raises(subprocess.CalledProcessError) as failure:
        matrix.run_child([sys.executable, '-c',
                          "print('detail'); print('Training [50%]'); print('[warning] note'); raise SystemExit(3)"],
                         tmp_path / 'console.log', 'case')
    assert failure.value.returncode == 3
    assert 'case: Training [50%]' in capsys.readouterr().out
    assert 'detail' in (tmp_path / 'console.log').read_text()


def test_full_driver_with_mock_training(tmp_path):
    executable = tmp_path / 'app.exe'
    executable.write_bytes(b'not executable')
    for dataset, images in (('bicycle', 'images_4'), ('statue', 'images')):
        (tmp_path / dataset / images).mkdir(parents=True)
        (tmp_path / dataset / 'sparse').mkdir()
    args = ['benchmark', '--bicycle', str(tmp_path / 'bicycle'), '--statue', str(tmp_path / 'statue'),
            '--executable', str(executable), '--output', str(tmp_path / 'results')]
    def fake_training(command, console, label):
        folder = Path(command[command.index('-o') + 1])
        write_result(folder)
    with patch.object(sys, 'argv', args + ['--dry-run']), patch.object(matrix, 'run_child') as child:
        matrix.main()
        child.assert_not_called()
        assert not (tmp_path / 'results').exists()
    with patch.object(sys, 'argv', args), patch.object(matrix, 'run_child', side_effect=fake_training) as child, \
            patch.object(matrix.time, 'perf_counter', side_effect=range(24)):
        matrix.main()
        assert child.call_count == 12
    summary = json.loads((tmp_path / 'results/summary.json').read_text())
    assert len(summary) == 6
    with (tmp_path / 'results/summary.csv').open() as file:
        assert len(list(csv.DictReader(file))) == 6
    with patch.object(sys, 'argv', args), patch.object(matrix, 'run_child') as child:
        with pytest.raises(SystemExit):
            matrix.main()
        child.assert_not_called()

# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Compare actual generic/profile dispatch in fresh HIP processes, without Torch.

The fixture directory must be new and outside tracked sources. --expect-plane
requires the validated device; otherwise an unselected profile is reported.
"""
import argparse
import os
from pathlib import Path
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('executable',type=Path)
p.add_argument('module',type=Path)
p.add_argument('fixture',type=Path)
p.add_argument('--expect-plane',action='store_true')
a=p.parse_args()
root=a.fixture.resolve()
root.mkdir(parents=True,exist_ok=False)
logs={}
for mode,disable,profiles in (('generic','1','0'),('selected','0','0'),('profiles-disabled','0','1')):
    directory=root/mode
    directory.mkdir()
    env=dict(os.environ,LFS_HIP_DISABLE_PLANE_SSIM=disable,
             LFS_HIP_DISABLE_ARCH_PROFILES=profiles,LFS_HIP_VALIDATION_PHOTO_DISPATCH='1')
    result=subprocess.run([str(a.executable.resolve()),str(a.module.resolve()),
                           '--plane-dump='+str(directory)],
                          env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          timeout=180)
    (root/(mode+'.log')).write_bytes(result.stdout)
    if result.returncode:
        raise RuntimeError(f'{mode} contract failed: {result.stdout.decode(errors="replace")}')
    logs[mode]=result.stdout
assert b'plane=1' not in logs['generic'],'Opt-out failed in the native module'
assert b'plane=1' not in logs['profiles-disabled'],'Global architecture-profile opt-out failed'
if a.expect_plane:
    assert logs['selected'].count(b'plane=1')==4,'Expected four large Float32 plane dispatches'
else:
    print('Native plane dispatch selected:',b'plane=1' in logs['selected'])
files=sorted((root/'generic').glob('*.bin'))
assert len(files)==8,'Missing whole-gradient cases'
for first in files:
    for mode in ('selected','profiles-disabled'):
        second=root/mode/first.name
        assert first.stat().st_size==second.stat().st_size,'Gradient extent changed'
        # Includes loss, workspace size, count and every FP32 gradient.
        with first.open('rb') as old,second.open('rb') as new:
            while True:
                block=old.read(1024*1024)
                assert block==new.read(len(block)),f'Whole-gradient mismatch: {first.name}, {mode}'
                if not block:break
    print('PASS exact loss/workspace/gradient:',first.name)
print('PASS fresh-process HIP Float32 selective dispatch and UInt8 preservation')

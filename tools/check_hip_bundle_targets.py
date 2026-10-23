#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Inspect the generated build graph; does not claim execution on other GPUs."""
import argparse
from pathlib import Path
import re
import subprocess

PORTABLE_TARGETS = frozenset('gfx1010 gfx1011 gfx1012 gfx1030 gfx1031 gfx1032 gfx1033 gfx1034 gfx1035 gfx1036 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1152 gfx1153 gfx1200 gfx1201 gfx900 gfx906 gfx908 gfx90a gfx942 gfx950'.split())


def gpu_commands(build, ninja, target):
    output = subprocess.run([ninja, '-C', str(build), '-t', 'commands', target],
                            check=True, capture_output=True, text=True).stdout
    result = []
    for command in output.splitlines():
        batch = re.fullmatch(r'"?([^"\r\n]+\.bat)"?(?:\s+\S+)?', command)
        if batch:
            command = (build / batch.group(1).replace('\\', '/')).read_text(encoding='utf-8')
        if '--offload-arch=' in command:
            result.append(command)
    if not result:
        raise RuntimeError(f'No GPU commands found for {target}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--ninja', default='ninja')
    args = parser.parse_args()
    for target, expected in [('lfs_hip_training_build', PORTABLE_TARGETS),
                             ('lfs_hip_runtime_build', PORTABLE_TARGETS),
                             ('lfs_hip_gfx1150_profile', frozenset(['gfx1150']))]:
        commands = gpu_commands(args.build.resolve(), args.ninja, target)
        for command in commands:
            actual = frozenset(re.findall(r'--offload-arch=([a-z0-9-]+)', command))
            if actual != expected:
                raise RuntimeError(f'{target}: wrong image set {sorted(actual)}, expected {sorted(expected)}')
            if '--offload-compress' not in command:
                raise RuntimeError(f'{target}: an uncompressed GPU unit remains')
        print(f'PASS {target}: {len(commands)} GPU commands, {len(expected)} targets each, compressed')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Run shared SDK-independent contracts using an existing build's compiler/GLM."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--application-build', type=Path, required=True)
    args = parser.parse_args()
    application = args.application_build.resolve()
    cache = {}
    for line in (application / 'CMakeCache.txt').read_text(encoding='utf-8').splitlines():
        if not line or line.startswith(('#', '//')) or '=' not in line:
            continue
        key, value = line.split('=', 1)
        cache[key.split(':', 1)[0]] = value
    required = ('VCPKG_INSTALLED_DIR', 'VCPKG_TARGET_TRIPLET', 'CMAKE_CXX_COMPILER', 'Python_EXECUTABLE')
    missing = [key for key in required if not cache.get(key)]
    if missing:
        raise RuntimeError('Missing application configuration: ' + ', '.join(missing))
    include = Path(cache['VCPKG_INSTALLED_DIR']) / cache['VCPKG_TARGET_TRIPLET'] / 'include'
    if not (include / 'glm/glm.hpp').is_file():
        raise RuntimeError('Application GLM headers are missing: ' + str(include))
    root = Path(__file__).resolve().parents[1]
    build = application / 'gpu-policy-cpu'
    env = dict(os.environ)
    for key in ('PYTHONHOME', 'PYTHONPATH'):
        env.pop(key, None)
    env['PYTHONNOUSERSITE'] = '1'
    python_root = build / 'python'
    python = python_root / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.is_file():
        subprocess.run([cache['Python_EXECUTABLE'], '-m', 'venv', str(python_root)], env=env, check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', '--disable-pip-version-check',
                    '--retries', '3', 'pytest>=7,<10'], env=env, check=True)
    command = ['cmake', '-S', str(root / 'tests/gpu_policy'), '-B', str(build), '-G', 'Ninja',
               '-DCMAKE_BUILD_TYPE=' + (cache.get('CMAKE_BUILD_TYPE') or 'Release'),
               '-DCMAKE_CXX_COMPILER=' + cache['CMAKE_CXX_COMPILER'],
               '-DLFS_CPU_DEPENDENCY_INCLUDE=' + str(include),
               '-DLFS_CPU_PYTHON_UI_TESTS=ON', '-DPython3_EXECUTABLE=' + str(python)]
    for key in ('CMAKE_OSX_DEPLOYMENT_TARGET', 'CMAKE_OSX_ARCHITECTURES'):
        if cache.get(key):
            command.append('-D' + key + '=' + cache[key])
    subprocess.run(command, env=env, check=True)
    subprocess.run(['cmake', '--build', str(build), '--parallel', '2'], env=env, check=True)
    subprocess.run(['ctest', '--test-dir', str(build), '--output-on-failure', '--no-tests=error'],
                   env=env, check=True)


if __name__ == '__main__':
    main()

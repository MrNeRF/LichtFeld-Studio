#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Manual HIP build jobs: compile coverage only, never GPU validation."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-type', choices=['Release', 'Debug'], required=True)
    parser.add_argument('--cuda', choices=['ON', 'OFF'], required=True)
    args = parser.parse_args()
    sdk = Path(os.environ['LFS_HIP_ROOT']).resolve()
    root = Path(__file__).resolve().parents[1]
    build = root / 'build-hip-ci'
    command = ['cmake', '-S', str(root), '-B', str(build), '-G', 'Ninja',
               '-DCMAKE_BUILD_TYPE=' + args.build_type, '-DLFS_ENABLE_CUDA=' + args.cuda,
               '-DBUILD_PORTABLE=ON',
               '-DLFS_HIP_RUNTIME=ON', '-DLFS_TENSOR_HIP=ON', '-DLFS_HIP_TRAINING_RUNTIME=ON',
               '-DLFS_HIP_PORTABLE=ON', '-DLFS_HIP_ARCH_PROFILES=ON',
               '-DLFS_HIP_OFFLOAD_COMPRESSION=ON',
               '-DLFS_HIP_ROOT=' + str(sdk), '-DLFS_TENSOR_VULKAN=ON',
               '-DLFS_FAST_COMPILE=OFF', '-DBUILD_TESTS=OFF', '-DBUILD_PYTHON_STUBS=OFF',
               '-DLFS_DEV_IMPORT_SOURCE_PYTHON=OFF', '-DLFS_DEV_IMPORT_SOURCE_RESOURCES=OFF',
               '-DLFS_ENABLE_CUDA_DEVICE_DEBUG=OFF']
    subprocess.run([sys.executable, str(root / 'tools/ci_retry_configure.py'), '--'] + command,
                   check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', 'LichtFeld-Studio',
                    '--parallel', '2'], check=True)
    # These checks require neither a GPU nor a loaded AMD driver.
    for name, flags in [
        ('tuned', ['-DLFS_HIP_ARCHITECTURE=gfx1150']),
        ('generic', ['-DLFS_HIP_ARCHITECTURE=gfx1103']),
        ('stale', ['-DLFS_HIP_ARCHITECTURE=gfx1150', '-DLFS_HIP_VALIDATED_TUNING=OFF',
                   '-DLFS_HIP_FAST_PACKED_GRADIENTS=ON']),
    ]:
        subprocess.run(['cmake', '-S', str(root / 'tests/hip/profile_policy'),
                        '-B', str(build / ('policy-' + name)), '-G', 'Ninja'] + flags, check=True)
    policy = build / 'module-policy'
    subprocess.run(['cmake', '-S', str(root / 'tests/hip/module_policy'), '-B', str(policy),
                    '-G', 'Ninja'], check=True)
    subprocess.run(['cmake', '--build', str(policy), '--parallel', '2'], check=True)
    subprocess.run(['ctest', '--test-dir', str(policy), '--output-on-failure', '--no-tests=error'], check=True)
    subprocess.run([sys.executable, str(root / 'tools/check_hip_bundle_targets.py'), '--build', str(build)], check=True)
    subprocess.run(['cmake', '-DTEST_ROOT=' + str(build / 'runtime-symlinks'), '-P',
                    str(root / 'tests/cmake/hip_runtime_symlinks.cmake')], check=True)
    if os.name == 'nt':
        runtime_policy = build / 'runtime-package-policy'
        subprocess.run(['cmake', '-S', str(root / 'tests/hip/runtime_package'), '-B', str(runtime_policy),
                        '-G', 'Ninja', '-DTEST_SDK=' + str(sdk)], check=True)
        subprocess.run(['cmake', '--build', str(runtime_policy), '--parallel', '2'], check=True)
        subprocess.run(['ctest', '--test-dir', str(runtime_policy), '--output-on-failure', '--no-tests=error'], check=True)
    print('PASS compilation and CPU policy checks; GPU execution is not validated by this job.')


if __name__ == '__main__':
    main()

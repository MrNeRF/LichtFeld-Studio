#!/usr/bin/env python3
"""Compile the real asset resolver in temporary bundle/development layouts."""
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform == 'darwin', 'Requires macOS executable path APIs')
class PortableResourcesTest(unittest.TestCase):
    def test_bundle_isolation_and_development_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            bindir = root / 'Test.app/Contents/MacOS'
            bindir.mkdir(parents=True)
            for portable in (False, True):
                executable = bindir / ('portable' if portable else 'development')
                command = ['xcrun', 'clang++', '-std=c++23',
                           '-I' + str(ROOT / 'src/visualizer'),
                           '-I' + str(ROOT / 'src/core/include'), '-I' + str(ROOT / 'src'),
                           '-DVISUALIZER_ASSET_PATH="' + str(root / 'development-assets') + '"',
                           '-DLFS_PYTHON_EXECUTABLE="' + str(root / 'development-assets/python3') + '"']
                if portable:
                    command.append('-DLFS_MACOS_PORTABLE_APP=1')
                command.extend([str(ROOT / 'tests/test_macos_portable_resource_paths.cpp'), '-o', str(executable)])
                subprocess.run(command, check=True, capture_output=True)
                subprocess.run([str(executable)], check=True)


if __name__ == '__main__':
    unittest.main()

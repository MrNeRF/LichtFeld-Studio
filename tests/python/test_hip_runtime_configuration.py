# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU-only HIP target configuration checks; no SDK, driver or GPU required."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CMAKE = os.environ.get("CMAKE_COMMAND") or shutil.which("cmake")


class HipConfiguration(unittest.TestCase):
    def configure(self, settings, expected=None):
        self.assertIsNotNone(CMAKE, "CMake is required for configuration tests")
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "targets.cmake"
            script.write_text(
                'cmake_minimum_required(VERSION 3.30)\n'
                f'include("{(ROOT / "cmake/HipArchitectures.cmake").as_posix()}")\n'
                + settings + '\nlfs_hip_architectures()\n'
                'message(STATUS "TARGETS=${LFS_HIP_COMPILED_ARCHITECTURES}")\n'
                'message(STATUS "FLAGS=${LFS_HIP_OFFLOAD_FLAGS}")\n',
                encoding="utf-8")
            result = subprocess.run([CMAKE, "-P", str(script)], capture_output=True,
                                    text=True, timeout=30)
        output = result.stdout + result.stderr
        if expected is None:
            self.assertNotEqual(result.returncode, 0, output)
        else:
            self.assertEqual(result.returncode, 0, output)
            self.assertIn("TARGETS=" + expected, output)
            for architecture in expected.split(","):
                self.assertIn("--offload-arch=" + architecture, output)
        return output

    def test_explicit_target(self):
        self.configure('set(LFS_HIP_ARCHITECTURE gfx1150)', 'gfx1150')

    def test_multiple_targets_deduplicate_and_override_single(self):
        self.configure('set(LFS_HIP_ARCHITECTURE gfx1100)\n'
                       'set(LFS_HIP_ARCHITECTURES "gfx1150;gfx942;gfx1150")',
                       'gfx1150,gfx942')

    def test_missing_target_is_rejected(self):
        self.assertIn("Set LFS_HIP_ARCHITECTURE", self.configure(""))

    def test_unknown_target_is_rejected(self):
        self.assertIn("Unsupported HIP target", self.configure(
            'set(LFS_HIP_ARCHITECTURE gfx999999)'))

    def test_explicit_list_overrides_portable_default(self):
        self.configure('set(LFS_HIP_PORTABLE ON)\n'
                       'set(LFS_HIP_ARCHITECTURES "gfx1150;gfx942")', 'gfx1150,gfx942')

    def test_requested_compression_requires_supported_compiler(self):
        self.assertIn("HIP offload compression requires", self.configure(
            'set(LFS_HIP_ARCHITECTURE gfx1150)\n'
            'set(LFS_HIP_ROOT "${CMAKE_CURRENT_LIST_DIR}/missing-sdk")\n'
            'set(LFS_HIP_OFFLOAD_COMPRESSION ON)'))


if __name__ == "__main__":
    unittest.main()

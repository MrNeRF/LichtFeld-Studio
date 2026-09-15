# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise ABI generation in CMake script mode, without configuring or building C++."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CMAKE = os.environ.get("CMAKE_COMMAND") or shutil.which("cmake")


@unittest.skipUnless(CMAKE, "CMake is required for ABI script tests")
class CoreAbiStampTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "source"
        self.build = self.root / "build with spaces"
        core = self.source / "src/core"
        core.mkdir(parents=True)
        (core / "CMakeLists.txt").write_text("# fixture\n", encoding="utf-8")
        self.core_file = core / "sample.cpp"
        self.core_file.write_text("int value = 1;\n", encoding="utf-8")
        self.metadata = {
            "CMAKE_SOURCE_DIR": self.source.as_posix(),
            "CMAKE_BINARY_DIR": self.build.as_posix(),
            "PROJECT_VERSION": "1.0",
            "GIT_COMMIT_HASH_SHORT": "fixture",
            "CMAKE_SYSTEM_NAME": "Windows",
            "CMAKE_SIZEOF_VOID_P": "8",
            "CMAKE_CXX_COMPILER_ID": "MSVC",
            "CMAKE_CXX_COMPILER_VERSION": "19.44",
            "CMAKE_CUDA_COMPILER_ID": "NVIDIA",
            "CMAKE_CUDA_COMPILER_VERSION": "12.8",
            "CMAKE_CUDA_ARCHITECTURES": "86;89",
            "VCPKG_TARGET_TRIPLET": "x64-windows-release",
        }

    def run_script(self, body, generator="Ninja"):
        script = self.root / "fixture.cmake"
        prelude = "cmake_minimum_required(VERSION 3.30)\n"
        prelude += "".join(f'set({key} "{value}")\n' for key, value in self.metadata.items())
        script.write_text(prelude + body, encoding="utf-8")
        result = subprocess.run(
            [CMAKE, "-G", generator, "-P", str(script)],
            capture_output=True, text=True,
        )
        self.assertEqual(0, result.returncode, result.stdout + result.stderr)

    def initialize(self, configuration="Release", generator="Ninja"):
        self.run_script(
            # Mimic dependencies populating this variable for single-config Ninja.
            'set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo")\n'
            f'set(CMAKE_BUILD_TYPE "{configuration}")\n'
            f'include("{ROOT.as_posix()}/cmake/InitializeCoreAbiStamp.cmake")\n'
            'lfs_initialize_core_abi_stamps()\n', generator,
        )

    def header(self, configuration):
        return self.build / "include/core-abi" / configuration / "lfs_core_abi_stamp.h"

    def test_single_config_seeds_only_actual_configuration_and_removes_legacy(self):
        for configuration in ("Debug", "Release", "RelWithDebInfo", "MinSizeRel", ""):
            with self.subTest(configuration=configuration):
                self.build = self.root / (configuration or "unnamed")
                self.metadata["CMAKE_BINARY_DIR"] = self.build.as_posix()
                legacy = self.build / "include/lfs_core_abi_stamp.h"
                legacy.parent.mkdir(parents=True)
                legacy.write_text("stale", encoding="utf-8")
                self.initialize(configuration)
                self.assertFalse(legacy.exists())
                self.assertEqual([self.header(configuration)],
                                 list((self.build / "include").rglob("lfs_core_abi_stamp.h")))

    def test_multi_config_seeds_distinct_headers_before_any_build(self):
        self.initialize(generator="Ninja Multi-Config")
        stamps = {self.header(config).read_text(encoding="utf-8")
                  for config in ("Debug", "Release", "RelWithDebInfo")}
        self.assertEqual(3, len(stamps))

    def test_noop_keeps_timestamp_and_core_edit_changes_stamp(self):
        self.initialize()
        header = self.header("Release")
        before = header.read_bytes()
        os.utime(header, (1_000_000_000, 1_000_000_000))
        timestamp = header.stat().st_mtime_ns
        self.initialize()
        self.assertEqual(timestamp, header.stat().st_mtime_ns)
        self.core_file.write_text("int value = 2;\n", encoding="utf-8")
        self.initialize()
        self.assertNotEqual(before, header.read_bytes())

    def test_build_refresh_matches_configure_seed(self):
        self.initialize()
        header = self.header("Release")
        initial = header.read_bytes()
        # Independently supply the same metadata as the production build target.
        values = {
            "LFS_SOURCE_DIR": self.source.as_posix(),
            "LFS_CORE_ABI_TEMPLATE": f"{ROOT.as_posix()}/cmake/lfs_core_abi_stamp.h.in",
            "LFS_CORE_ABI_HEADER": header.as_posix(),
            "LFS_PROJECT_VERSION": "1.0",
            "LFS_CONFIGURED_GIT_COMMIT_HASH_SHORT": "fixture",
            "LFS_BUILD_CONFIG": "Release",
            "LFS_SYSTEM_NAME": "Windows", "LFS_SIZEOF_VOID_P": "8",
            "LFS_CXX_COMPILER_ID": "MSVC", "LFS_CXX_COMPILER_VERSION": "19.44",
            "LFS_CUDA_COMPILER_ID": "NVIDIA", "LFS_CUDA_COMPILER_VERSION": "12.8",
            "LFS_CUDA_ARCHITECTURES": "86;89", "LFS_VCPKG_TARGET_TRIPLET": "x64-windows-release",
        }
        body = "".join(f'set({key} "{value}")\n' for key, value in values.items())
        body += f'include("{ROOT.as_posix()}/cmake/GenerateCoreAbiStamp.cmake")\n'
        self.run_script(body)
        self.assertEqual(initial, header.read_bytes())
        self.core_file.write_text("int value = 3;\n", encoding="utf-8")
        self.run_script(body)
        self.assertNotEqual(initial, header.read_bytes())


if __name__ == "__main__":
    unittest.main()

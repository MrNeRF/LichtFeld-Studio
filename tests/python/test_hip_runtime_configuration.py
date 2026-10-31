# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""CPU-only HIP target configuration checks; no SDK, driver or GPU required."""
import os
import argparse
import importlib.util
from unittest import mock
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
CMAKE = os.environ.get("CMAKE_COMMAND") or shutil.which("cmake")


class HipConfiguration(unittest.TestCase):
    def configure(self, settings, expected=None, *,
                  module="cmake/HipArchitectures.cmake", invoke="lfs_hip_architectures()"):
        self.assertIsNotNone(CMAKE, "CMake is required for configuration tests")
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary) / "targets.cmake"
            script.write_text(
                'cmake_minimum_required(VERSION 3.30)\n'
                + settings + '\n'
                f'include("{(ROOT / module).as_posix()}")\n'
                + invoke + '\n'
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

    def configure_backend_registry(self, directory, settings, prelude=""):
        self.assertIsNotNone(CMAKE, "CMake is required for configuration tests")
        (directory / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.30)\n'
            'project(BackendRegistryContracts LANGUAGES NONE)\n'
            + settings + '\n' + prelude + '\n'
            f'include("{(ROOT / "cmake/GpuBackendRegistry.cmake").as_posix()}")\n'
            'lfs_generate_backend_registry()\n'
            'lfs_generate_backend_registry()\n'
            'get_target_property(includes lfs_backend_configuration INTERFACE_INCLUDE_DIRECTORIES)\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/backend_configuration_includes.txt" "${includes}")\n',
            encoding="utf-8")
        build = directory / "build"
        result = subprocess.run([CMAKE, "-S", str(directory), "-B", str(build), "-G", "Ninja"],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return build

    def test_backend_registry_matches_each_build_configuration(self):
        """Registration and public inventory must share the same CMake source."""
        self.assertIsNotNone(CMAKE, "CMake is required for configuration tests")
        names = ("CUDA", "Vulkan", "Metal", "HIP")
        flags = ("LFS_HAS_CUDA", "LFS_TENSOR_VULKAN", "LFS_TENSOR_METAL", "LFS_TENSOR_HIP")
        for mask in range(16):
            enabled = [name for index, name in enumerate(names) if mask & (1 << index)]
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary).resolve()
                settings = "\n".join(f"set({flag} {'ON' if mask & (1 << index) else 'OFF'})"
                                     for index, flag in enumerate(flags))
                build = self.configure_backend_registry(directory, settings)
                self.assertEqual((build / "backend_configuration_includes.txt").read_text(encoding="utf-8"),
                                 (build / "include").as_posix())
                header = (build / "include/core/gpu_backend_config.hpp").read_text(encoding="utf-8")
                source = (build / "generated_backend_registration.cpp").read_text(encoding="utf-8")
                compiled = ", ".join(f"GpuBackend::{name}" for name in enabled)
                self.assertIn(f"kCompiledGpuBackends{{{compiled}}}", header)
                for name in names:
                    self.assertEqual(f"register_{name.lower()}_backend(registry);" in source,
                                     name in enabled)
                defaults = (["CUDA"] if "CUDA" in enabled else
                            [name for name in ("Metal", "Vulkan") if name in enabled] or
                            (["HIP"] if "HIP" in enabled else ["CUDA"]))
                expected = ", ".join(f"GpuBackend::{name}" for name in defaults)
                self.assertIn(f"kDefaultGpuBackends{{{expected}}}", header)

    def test_backend_registry_preserves_an_existing_configuration_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve()
            build = self.configure_backend_registry(
                directory, "set(LFS_TENSOR_HIP ON)",
                'add_library(lfs_backend_configuration INTERFACE)\n'
                'target_include_directories(lfs_backend_configuration INTERFACE '
                '"${CMAKE_CURRENT_SOURCE_DIR}/existing")')
            self.assertEqual((build / "backend_configuration_includes.txt").read_text(encoding="utf-8"),
                             (directory / "existing").as_posix())

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

    def test_cross_compilation_is_rejected_before_sdk_discovery(self):
        self.assertIn("does not support cross-compilation", self.configure(
            'set(CMAKE_CROSSCOMPILING TRUE)\nset(CMAKE_SYSTEM_NAME Linux)',
            module="src/core/tensor/backend/hip/CMakeLists.txt", invoke=""))

    def test_unsupported_platform_is_rejected_before_sdk_discovery(self):
        self.assertIn("supports Windows and Linux", self.configure(
            'set(CMAKE_SYSTEM_NAME FreeBSD)',
            module="src/core/tensor/backend/hip/CMakeLists.txt", invoke=""))

    def test_native_cli_rejects_invalid_ordinals_before_loading(self):
        for device in (-1, 2147483648, 4294967296):
            with self.subTest(device=device):
                result = subprocess.run([
                    sys.executable, str(ROOT / "tests/hip/test_native_runtime.py"),
                    "missing-module.dll", "--device", str(device),
                    "--runtime-dir", "missing-sdk-directory"],
                    capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("ordinal in 0..2147483647", result.stderr)

    def test_module_revision_is_checked_before_operator_exports(self):
        script = ROOT / "tests/hip/test_native_runtime.py"
        spec = importlib.util.spec_from_file_location("native_abi_contract", script)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        native.options = argparse.Namespace(module=Path("unused-module"), faults=False)
        for revision in (1, 2, 3, 4, 5):
            older_module = argparse.Namespace(lfs_hip_runtime_abi=lambda: revision)
            with self.subTest(revision=revision), mock.patch.object(native.c, "CDLL", return_value=older_module):
                with self.assertRaisesRegex(RuntimeError, f"ABI mismatch: expected 6, got {revision}"):
                    native.RuntimeContract.setUpClass()

    def test_abi_only_requires_the_native_module(self):
        result = subprocess.run([
            sys.executable, str(ROOT / "tests/hip/test_native_runtime.py"),
            "missing-module.dll", "--abi-only"], capture_output=True, text=True, timeout=30)
        # A setUpClass failure can also produce a distinct no-tests exit status.
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("setUpClass", result.stderr)
        self.assertIn("FAILED", result.stderr)

    def test_tensor_cli_rejects_gpu_fault_mode_in_cpu_mode_before_loading(self):
        result = subprocess.run([
            sys.executable, str(ROOT / "tests/hip/test_tensor_operations.py"),
            "missing-module.dll", "--abi-only", "--faults"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("mutually exclusive", result.stderr)

    def test_cpu_abi_mode_rejects_gpu_fault_mode_before_loading(self):
        result = subprocess.run([
            sys.executable, str(ROOT / "tests/hip/test_native_runtime.py"),
            "missing-module.dll", "--abi-only", "--faults"],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2)
        self.assertIn("mutually exclusive", result.stderr)


if __name__ == "__main__":
    unittest.main()

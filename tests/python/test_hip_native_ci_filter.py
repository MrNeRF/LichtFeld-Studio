# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Exercise CI selection against real Git history, without an SDK or GPU."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/hip_native_ci_filter.py"
spec = importlib.util.spec_from_file_location("hip_ci_filter", SCRIPT)
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)


class HipCiFilter(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name)
        self.git("init", "-q")
        self.git("config", "user.name", "CI Contract")
        self.git("config", "user.email", "ci@example.invalid")
        self.write("initial.txt")
        self.base = self.commit()

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.repo, text=True).strip()

    def write(self, name, text="content"):
        path = self.repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def commit(self):
        self.git("add", "-A")
        self.git("commit", "-qm", "fixture")
        return self.git("rev-parse", "HEAD")

    def test_unrelated_source_changes_do_not_require_sdk(self):
        self.write("src/visualizer/example.cpp")
        self.assertFalse(selector.needs_compile(self.repo, "pull_request", self.base, self.commit()))

    def test_direct_and_supporting_changes_require_sdk(self):
        for path in ("src/core/tensor/tensor_upload.cpp", "src/core/tensor/expression_program.cpp",
                     "src/core/include/core/tensor.hpp", "src/core/CMakeLists.txt",
                     "src/python/lfs/py_tensor.cpp", "tests/TensorOperationContracts.cmake",
                     "src/core/tensor/backend/hip/runtime.hip", "cmake/hip-native/ci-requirements.txt",
                     "tests/hip/test_native_runtime.py", "cmake/HipArchitectures.cmake",
                     "cmake/HipOffloadCompression.cmake", ".github/workflows/ubuntu.yml",
                     "tools/hip_native_ci_filter.py", "tests/python/test_hip_native_ci_filter.py",
                     "tests/python/test_hip_runtime_configuration.py",
                     "tests/python/test_hip_pointwise_generation.py",
                     "cmake/hip-host/CMakeLists.txt", "tests/python/test_hip_host_contracts.py",
                     "src/core/include/core/detail/pointwise_ops.def",
                     "src/core/include/core/detail/pointwise_lowering.hpp",
                     "src/core/include/core/detail/tensor_functors.hpp",
                     "src/core/tensor/backend/vulkan/shaders/inference.slang",
                     "src/core/cuda/undistort/distortion_model.cuh",
                     "src/core/tensor/backend/metal/expression_msl.cpp",
                     "src/io/cuda/image_format_kernels.cu"):
            with self.subTest(path=path):
                self.write(path)
                head = self.commit()
                self.assertTrue(selector.needs_compile(self.repo, "pull_request", self.base, head))
                self.base = head

    def test_deleted_and_renamed_away_paths_require_sdk(self):
        old = "src/core/tensor/backend/hip/runtime.hip"
        self.write(old)
        base = self.commit()
        self.git("mv", old, "moved.hip")
        self.assertTrue(selector.needs_compile(self.repo, "push", base, self.commit()))
        self.write(old)
        base = self.commit()
        (self.repo / old).unlink()
        self.assertTrue(selector.needs_compile(self.repo, "push", base, self.commit()))

    def test_pull_request_uses_merge_base_not_new_base_changes(self):
        self.git("checkout", "-qb", "topic")
        self.write("unrelated.cpp")
        head = self.commit()
        self.git("checkout", "-qb", "base-advanced", self.base)
        self.write("cmake/HipArchitectures.cmake")
        base = self.commit()
        self.assertFalse(selector.needs_compile(self.repo, "pull_request", base, head))
        self.assertTrue(selector.needs_compile(self.repo, "push", base, head))

    def test_manual_and_new_branch_push_always_run(self):
        self.assertTrue(selector.needs_compile(self.repo, "workflow_dispatch", "", ""))
        self.assertTrue(selector.needs_compile(self.repo, "push", "0" * 40, self.base))

    def test_bad_revisions_and_missing_history_fail_instead_of_skipping(self):
        for event, base in (("pull_request", ""), ("push", "--output=bad"), ("unknown", self.base)):
            with self.subTest(event=event, base=base), self.assertRaises(ValueError):
                selector.needs_compile(self.repo, event, base, self.base)
        with self.assertRaises(subprocess.CalledProcessError):
            selector.needs_compile(self.repo, "push", "f" * 40, self.base)

    def test_cli_writes_job_output(self):
        output = self.repo / "job-output"
        subprocess.run([sys.executable, str(SCRIPT), "--event", "push", "--base", self.base,
                        "--head", self.base, "--repo", str(self.repo), "--output", str(output)],
                       check=True, capture_output=True, text=True, timeout=30)
        self.assertEqual(output.read_text(encoding="utf-8"), "changed=false\n")


if __name__ == "__main__":
    unittest.main()

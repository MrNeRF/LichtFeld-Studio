#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Neutral trainer sources must not include kernel or CUDA headers."""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import unittest
from pathlib import Path

import check_backend_neutrality as gate


class NeutralTrainerIncludeTest(unittest.TestCase):
    def test_neutral_trainer_file_rejects_kernel_and_cuda_headers(self) -> None:
        relative = "src/training/trainer.cpp"
        self.assertTrue(gate.neutral_trainer_file(relative))
        for header in (
            "kernels/depth_loss.hpp",
            "training/kernels/mrnf_kernels.hpp",
            "lfs/kernels/ssim.cuh",
            "lfs/training/ops/geometry_cuda.hpp",
            "cuda_runtime.h",
            "cuda.h",
        ):
            self.assertIsNotNone(gate.TRAINER_KERNEL_INCLUDE.search(header), header)

    def test_cuda_implementation_files_are_not_neutral(self) -> None:
        for relative in (
            "src/training/kernels/depth_loss.cu",
            "src/training/ops/geometry_cuda.cpp",
            "src/training/include/lfs/training/ops/geometry_cuda.hpp",
            "src/training/rasterization/fast_rasterizer_cuda.cpp",
            "src/training/rasterization/fastgs/rasterization/src/forward.cu",
            "src/training/rasterization/gsplat/Rasterization.cpp",
        ):
            self.assertFalse(gate.neutral_trainer_file(relative), relative)

    def test_vulkan_implementation_files_are_not_neutral(self) -> None:
        for relative in (
            "src/training/ops/session_vulkan.cpp",
            "src/training/ops/table_vulkan.cpp",
            "src/training/include/lfs/training/ops/session_vulkan.hpp",
            "src/training/vulkan/pair_sort.cpp",
            "src/training/vulkan/shaders/pair_sort.slang",
        ):
            self.assertFalse(gate.neutral_trainer_file(relative), relative)

    def test_scan_reports_a_neutral_include(self) -> None:
        path = Path(__file__).resolve().parent.parent / "src/training/trainer.hpp"
        # The live header is part of the gate. A synthetic copy proves the rule
        # fires even when the tree is clean.
        source = path.read_text(encoding="utf-8")
        self.assertNotIn("kernels/depth_loss.hpp", source)
        forbidden = path.with_name("trainer_cuda_include_probe.hpp")
        forbidden.write_text('#include "kernels/depth_loss.hpp"\n', encoding="utf-8")
        try:
            findings = gate.scan(forbidden)
        finally:
            forbidden.unlink()
        self.assertIn((1, "trainer-cuda-include", "kernels/depth_loss.hpp"), findings)

    def test_directory_scan_includes_objective_cpp(self) -> None:
        # Metal viewer code is Objective-C++; a directory scan must not skip it.
        probe = Path(__file__).resolve().parent.parent / "src/visualizer/rendering/neutrality_probe"
        probe.mkdir()
        try:
            (probe / "probe.mm").write_text("bool f(B b) { return b == core::GpuBackend::Metal; }\n", encoding="utf-8")
            argv, sys.argv = sys.argv, ["check_backend_neutrality.py", "--root", str(probe)]
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    failed = gate.main()
            finally:
                sys.argv = argv
        finally:
            shutil.rmtree(probe)
        self.assertTrue(failed, output.getvalue())
        self.assertIn("probe.mm:1: [backend-branch]", output.getvalue())


if __name__ == "__main__":
    unittest.main()

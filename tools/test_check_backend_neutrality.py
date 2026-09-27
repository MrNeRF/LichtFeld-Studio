#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Neutral trainer sources must not include kernel or CUDA headers."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()

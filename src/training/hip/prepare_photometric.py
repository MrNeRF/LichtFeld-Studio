# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later
"""Adapt this checkout's unmasked fused L1/SSIM kernels for private HIP use.

CUDA sources, floating-point formulas, half partials and convolution order stay
unchanged. Extraction fails when the source seam changes rather than silently
generating a different implementation.
"""
import argparse
from pathlib import Path


def prepare(source: Path, output: Path) -> None:
    text = source.read_text(encoding="utf-8")
    prefix = text[text.index("namespace {\n") + len("namespace {\n"):
                  text.index("    // ------------------------------------------\n    // Forward Kernel: Fused SSIM")]
    kernels = text[text.index("    // Fused L1+SSIM Forward Kernel\n"):
                   text.index("    // Normalize only the masked samples;")]
    for name in ("fusedL1SSIMForwardCUDA", "fusedL1SSIMBackwardCUDA"):
        if kernels.count("void " + name + "(") != 1:
            raise ValueError(f"Unexpected photometric kernel seam: {name}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text[:text.index('#include "core/cuda_error.hpp"')] + '''#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_cooperative_groups.h>
#include <cstdint>
#include <type_traits>
namespace lfs_hip_photo {
namespace cg = cooperative_groups;
''' + prefix + kernels + '''
inline void forward(int h, int w, const float* image, const float* target,
                    __half* partials, float* map, hipStream_t stream) {
    const size_t count = size_t(h) * w * 3;
    hipLaunchKernelGGL(fusedL1SSIMForwardCUDA<float>, dim3((w + 15) / 16, (h + 15) / 16), dim3(16, 16), 0, stream,
                      h, w, 3, 0.01f * 0.01f, 0.03f * 0.03f, image, target,
                      partials, partials + count, partials + count * 2, map, nullptr);
}
inline void backward(int h, int w, float weight, bool valid, const float* image,
                     const float* target, const __half* partials, float* gradient, hipStream_t stream) {
    const size_t count = size_t(h) * w * 3;
    const size_t normalized = size_t(valid && h > 10 ? h - 10 : h) * (valid && w > 10 ? w - 10 : w) * 3;
    hipLaunchKernelGGL(fusedL1SSIMBackwardCUDA<float>, dim3((w + 15) / 16, (h + 15) / 16), dim3(16, 16), 0, stream,
                      weight, h, w, 3, 0.01f * 0.01f, 0.03f * 0.03f, 1.0f / normalized, valid,
                      image, target, gradient, partials, partials + count, partials + count * 2);
}
} // namespace lfs_hip_photo
#undef BLOCK_X
#undef BLOCK_Y
#undef HALO
#undef SHARED_X
#undef SHARED_Y
#undef CONV_X
#undef CONV_Y
''', encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)

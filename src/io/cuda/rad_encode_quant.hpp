/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <memory>
#include <span>

namespace lfs::io::cuda {

    // One chunk's pack-domain splat arrays (host pointers, layouts as in
    // RadStreamChunkSource). Scales/orientation are absent on purpose: their
    // encoders go through libm (log, acos, sin) and stay on the CPU so file
    // bytes remain bit-identical.
    struct RadEncodeQuantChunkIn {
        std::uint32_t count = 0;
        const float* means = nullptr; // [count*3]
        const float* alpha = nullptr; // [count]
        const float* rgb = nullptr;   // [count*3]
        const float* shN = nullptr;   // [count*sh_coeffs*3], optional
    };

    // Quantized property planes matching the CPU PropertyEncoder Auto profile
    // bit for bit (center f32_lebytes, alpha r8/f16, rgb r8_delta, shN s8).
    // Pointers reference quantizer-owned pinned memory and stay valid until
    // the next quantize_batch call on the same instance.
    struct RadEncodeQuantChunkOut {
        const std::uint8_t* center = nullptr; // 12*count
        const std::uint8_t* alpha = nullptr;  // f16: 2*count, r8: count
        bool alpha_f16 = false;
        float alpha_min = 0.0f;            // r8 only
        float alpha_max = 0.0f;            // r8 only
        const std::uint8_t* rgb = nullptr; // 3*count
        float rgb_min = 0.0f;
        float rgb_max = 0.0f;
        const std::uint8_t* sh[3] = {}; // s8 band planes: 9/15/21 dims * count
        float sh_max_abs[3] = {0.0f, 0.0f, 0.0f};
    };

    [[nodiscard]] bool rad_encode_gpu_available();

    class RadEncodeGpuQuantizer {
    public:
        RadEncodeGpuQuantizer();
        ~RadEncodeGpuQuantizer();
        RadEncodeGpuQuantizer(const RadEncodeGpuQuantizer&) = delete;
        RadEncodeGpuQuantizer& operator=(const RadEncodeGpuQuantizer&) = delete;

        // False on any CUDA failure; the instance is then permanently failed
        // and callers fall back to the CPU encoders.
        [[nodiscard]] bool quantize_batch(std::span<const RadEncodeQuantChunkIn> chunks,
                                          int sh_coeffs,
                                          bool lod_tree,
                                          std::span<RadEncodeQuantChunkOut> out);

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::io::cuda

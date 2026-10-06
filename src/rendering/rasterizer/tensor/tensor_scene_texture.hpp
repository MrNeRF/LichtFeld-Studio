/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/gpu_kernel_module.hpp"
#include "core/tensor.hpp"

#include <array>
#include <cstdint>
#include <memory>

namespace lfs::rendering {
    struct alignas(16) SceneTextureParameters {
        std::array<uint32_t, 4> extents{}; // input width/height, output width/height
        std::array<uint32_t, 4> layout{};  // source width, Float32 color, flip Y, temporal
        std::array<uint32_t, 4> pitches{}; // color/depth/motion/output byte row strides
        std::array<float, 4> depth{};      // near, far, orthographic, source depth row stride
        std::array<float, 4> jitter{};     // input pixel offset for current alpha coverage
    };

    // Single-source conversion between scene tensors and padded texture byte layouts.
    class TensorSceneTextureKernels {
    public:
        explicit TensorSceneTextureKernels(core::GpuBackend backend) : backend_(backend) {}
        lfs::Result<void> dispatch(bool unpack, const SceneTextureParameters& parameters,
                                   const std::array<core::Tensor*, 8>& tensors);

    private:
        core::GpuBackend backend_;
        std::unique_ptr<core::GpuKernelModule> module_;
    };
} // namespace lfs::rendering

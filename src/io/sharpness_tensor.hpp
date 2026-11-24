// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "core/gpu_kernel_module.hpp"
#include "core/tensor.hpp"
#include "media/sharpness_score.hpp"
namespace lfs::io {
    class TensorSharpnessScorer {
    public:
        std::optional<double> score(const core::Tensor& rgb, int width, int height, SharpnessAlgorithm algorithm);

    private:
        std::unique_ptr<core::GpuKernelModule> module_;
        core::Tensor partials_;
    };
} // namespace lfs::io

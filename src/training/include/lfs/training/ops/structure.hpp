/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "lfs/training/ops/types.hpp"

namespace lfs::training::kernels {

    // The separable passes run over row bands so the intermediate stays a few MiB at any image size.
    struct RidgeWorkspace {
        lfs::core::Tensor horizontal;
        lfs::core::Tensor reduction;
        size_t band_bytes = size_t{16} << 20;
        [[nodiscard]] size_t band_rows(size_t height, size_t width) const;
        void ensure_size(size_t band_rows, size_t width);
    };

    float structure_base_denominator(const lfs::core::Tensor& base_weight, int height, int width, bool valid_padding);

    void ridge_structure_map(const lfs::core::Tensor& image,
                             lfs::core::Tensor& output,
                             RidgeWorkspace& workspace);

    void structure_photometric_weight(const lfs::core::Tensor& structure,
                                      const lfs::core::Tensor& base_weight,
                                      lfs::core::Tensor& output,
                                      float gain, bool valid_padding);

    void structure_densification_weight(lfs::core::Tensor& error,
                                        const lfs::core::Tensor& structure,
                                        float gain);

} // namespace lfs::training::kernels

namespace lfs::training::kernels {

    inline constexpr int GRADIENT_LOSS_START_STEP = 2000;
    inline constexpr float GRADIENT_LOSS_EPSILON = 0.001f;

    struct GradientResidualWorkspace {
        lfs::core::Tensor partial;
        lfs::core::Tensor totals;
        lfs::core::Tensor loss;
        void ensure_allocated();
    };

    lfs::core::Tensor gradient_residual_loss_gradient(
        const lfs::core::Tensor& image,
        const lfs::core::Tensor& target,
        const lfs::core::Tensor& pixel_weight,
        lfs::core::Tensor& gradient,
        float weight,
        GradientResidualWorkspace& workspace);

} // namespace lfs::training::kernels

namespace lfs::gpu_ops {
    struct StructureOps {
        void (*ridge)(In, Out, training::kernels::RidgeWorkspace&);
        void (*photometric_weight)(In, In, Out, float, bool);
        void (*densification_weight)(Out, In, float);
        Tensor (*gradient_residual)(In, In, In, Out, float, training::kernels::GradientResidualWorkspace&);
    };
} // namespace lfs::gpu_ops

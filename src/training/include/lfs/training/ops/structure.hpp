/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include "lfs/training/ops/domain_types.hpp"
#include "lfs/training/ops/types.hpp"

#include <unordered_map>

namespace lfs::training::kernels {

    using MaskPhotoMode = lfs::gpu_ops::MaskPhotoMode;

    // The separable passes run over row bands so the intermediate stays a few MiB at any image size.
    struct RidgeWorkspace {
        lfs::core::Tensor horizontal;
        lfs::core::Tensor reduction;
        size_t band_bytes = size_t{16} << 20;
        [[nodiscard]] size_t band_rows(size_t height, size_t width) const;
        void ensure_size(size_t band_rows, size_t width, bool masked = false);
    };

    float structure_base_denominator(const lfs::core::Tensor& base_weight, int height, int width, bool valid_padding);

    // structure_base_denominator reads its base weight's sum back from the GPU. A camera's base weight is made from
    // its mask and crop box weight, which rarely change, so this keeps each camera's denominator together with what
    // its weight was made from, Key, and reads it back again only when that changed.
    template <typename Key>
    class StructureDenominatorCache {
    public:
        float get(const int camera, const Key& inputs, const lfs::core::Tensor& base_weight, const int height,
                  const int width, const bool valid_padding) {
            const auto found = entries_.find(camera);
            if (found != entries_.end() && found->second.inputs == inputs && found->second.height == height &&
                found->second.width == width && found->second.valid_padding == valid_padding)
                return found->second.denominator;
            const float denominator = structure_base_denominator(base_weight, height, width, valid_padding);
            entries_.insert_or_assign(camera, Entry{inputs, height, width, valid_padding, denominator});
            return denominator;
        }

        void clear() { entries_.clear(); }

    private:
        struct Entry {
            Key inputs;
            int height;
            int width;
            bool valid_padding;
            float denominator;
        };
        std::unordered_map<int, Entry> entries_;
    };

    void ridge_structure_map(const lfs::core::Tensor& image,
                             lfs::core::Tensor& output,
                             RidgeWorkspace& workspace);
    void ridge_structure_map(const lfs::core::Tensor& image,
                             lfs::core::Tensor& output,
                             RidgeWorkspace& workspace,
                             const lfs::core::Tensor& photometric_mask,
                             lfs::gpu_ops::MaskPhotoMode mask_mode);

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
        void (*ridge)(In, Out, training::kernels::RidgeWorkspace&, In, MaskPhotoMode);
        void (*photometric_weight)(In, In, Out, float, bool);
        void (*densification_weight)(Out, In, float);
        Tensor (*gradient_residual)(In, In, In, Out, float, training::kernels::GradientResidualWorkspace&);
    };
} // namespace lfs::gpu_ops

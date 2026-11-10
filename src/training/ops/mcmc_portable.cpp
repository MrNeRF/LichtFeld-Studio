/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "lfs/training/ops/mcmc.hpp"

#include <limits>

namespace lfs::training {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;

    bool portable_mcmc_sample_rows(const Tensor& weights, const float min_weight, const Tensor& excluded,
                                   Tensor& indices, const uint64_t seed) {
        const size_t samples = indices.numel();
        if (samples == 0 || weights.numel() == 0)
            return false;
        Tensor sampleable = weights.reshape({-1});
        if (min_weight != -std::numeric_limits<float>::infinity())
            sampleable = sampleable.clamp_min(min_weight);
        if (excluded.is_valid())
            sampleable = sampleable.masked_fill(excluded.reshape({-1}).to(DataType::Bool), 0.0f);
        if (!(sampleable.sum().item<float>() > 0.0f))
            return false;
        indices.copy_from(Tensor::multinomial(sampleable, static_cast<int>(samples), true, seed));
        return true;
    }

    void portable_mcmc_relocate_copies(const gpu_ops::McmcOps& ops, const Tensor& sources, const Tensor& copies,
                                       const int64_t first_copy, const Tensor& ratios, const float min_opacity,
                                       Tensor& raw_scales, Tensor& raw_opacity) {
        const size_t n = sources.numel();
        if (n == 0)
            return;
        // The copies still hold their sources' values, so the source rows are read.
        const Tensor opacity = raw_opacity.index_select(0, sources).reshape({-1}).sigmoid();
        const Tensor scales = raw_scales.index_select(0, sources).exp();
        Tensor new_opacity = Tensor::empty({n}, Device::GPU, DataType::Float32);
        Tensor new_scales = Tensor::empty({n, 3}, Device::GPU, DataType::Float32);
        ops.relocate(opacity, scales, ratios, new_opacity, new_scales, min_opacity);
        Tensor new_opacity_raw = new_opacity.clamp(min_opacity, 1.0f - 1e-7f).logit(1e-7f);
        if (raw_opacity.ndim() == 2)
            new_opacity_raw = new_opacity_raw.unsqueeze(-1);
        const Tensor new_scales_raw = new_scales.log();
        ops.update_rows(sources, new_scales_raw, new_opacity_raw, raw_scales, raw_opacity);
        if (copies.is_valid()) {
            ops.update_rows(copies, new_scales_raw, new_opacity_raw, raw_scales, raw_opacity);
            return;
        }
        const auto first = static_cast<size_t>(first_copy);
        auto copy_scales = raw_scales.slice(0, first, first + n);
        copy_scales.copy_from(new_scales_raw);
        auto copy_opacity = raw_opacity.slice(0, first, first + n);
        copy_opacity.copy_from(new_opacity_raw);
    }
} // namespace lfs::training

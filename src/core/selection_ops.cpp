/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/selection_ops.hpp"
#include "core/tensor.hpp"
#include "core/tensor_spatial.hpp"

#include <cassert>
#include <cstdint>

namespace lfs::core {

    namespace {
        constexpr float kShC0 = 0.28209479177387814f;

        Tensor empty_group_mask(const Tensor& input) {
            return Tensor::empty_like(input, TensorShape{0}, DataType::UInt8);
        }

        Tensor group_mask(const Tensor& mask, const uint8_t group_id) {
            Tensor out = mask.to(DataType::UInt8);
            out.masked_fill_(mask, static_cast<float>(group_id));
            return out;
        }
    } // namespace

    Tensor selection_grow(const Tensor& mask, const Tensor& means, const float radius, const uint8_t group_id) {
        LFS_ASSERT_MSG(mask.dtype() == DataType::UInt8, "selection_grow requires a UInt8 mask");
        const auto neighbors = radius_neighbors(means, mask, radius);
        auto result = mask.clone();
        result.masked_fill_(neighbors.logical_and(mask.eq(0.0f)), static_cast<float>(group_id));
        return result;
    }

    Tensor selection_shrink(const Tensor& mask, const Tensor& means, const float radius) {
        LFS_ASSERT_MSG(mask.dtype() == DataType::UInt8, "selection_shrink requires a UInt8 mask");
        const auto neighbors = radius_neighbors(means, mask.eq(0.0f), radius);
        auto result = mask.clone();
        result.masked_fill_(neighbors, 0.0f);
        return result;
    }

    Tensor select_by_opacity(const Tensor& opacity_raw, const float min_opacity,
                             const float max_opacity, const uint8_t group_id) {
        assert(opacity_raw.dtype() == DataType::Float32);
        if (opacity_raw.numel() == 0) {
            return empty_group_mask(opacity_raw);
        }
        const Tensor activated = opacity_raw.flatten().sigmoid();
        return group_mask((activated >= min_opacity).logical_and(activated <= max_opacity), group_id);
    }

    Tensor select_by_scale(const Tensor& scale_raw, const float max_scale, const uint8_t group_id) {
        assert(scale_raw.dtype() == DataType::Float32);
        assert(scale_raw.ndim() == 2 && scale_raw.size(1) == 3);
        if (scale_raw.size(0) == 0) {
            return empty_group_mask(scale_raw);
        }
        const Tensor largest = scale_raw.exp().max(1);
        return group_mask(largest <= max_scale, group_id);
    }

    Tensor select_by_color(const Tensor& sh0, const float ref_r, const float ref_g, const float ref_b,
                           const float threshold, const uint8_t group_id) {
        assert(sh0.dtype() == DataType::Float32);
        const size_t n = sh0.size(0);
        if (n == 0) {
            return empty_group_mask(sh0);
        }
        const Tensor decoded = sh0.reshape({static_cast<int>(n), 3}).mul(kShC0).add(0.5f).clamp(0.0f, 1.0f);
        const auto channel_matches = [&](const size_t channel, const float reference) {
            return (decoded.slice(1, channel, channel + 1) - reference).abs() <= threshold;
        };
        const Tensor mask = channel_matches(0, ref_r)
                                .logical_and(channel_matches(1, ref_g))
                                .logical_and(channel_matches(2, ref_b))
                                .flatten();
        return group_mask(mask, group_id);
    }

} // namespace lfs::core

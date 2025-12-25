/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data_mirror.hpp"
#include "core/logger.hpp"
#include "core/splat_data.hpp"
#include <mutex>

namespace lfs::core {

    namespace {

        // Sign multipliers for position reflection per axis
        constexpr float POS_MULT[3][3] = {
            {-1.0f, 1.0f, 1.0f},  // X
            {1.0f, -1.0f, 1.0f},  // Y
            {1.0f, 1.0f, -1.0f}   // Z
        };

        // Quaternion sign multipliers per axis (w,x,y,z)
        constexpr float QUAT_MULT[3][4] = {
            {1.0f, 1.0f, -1.0f, -1.0f},  // X
            {1.0f, -1.0f, 1.0f, -1.0f},  // Y
            {1.0f, -1.0f, -1.0f, 1.0f}   // Z
        };

        // SH coefficient multipliers per axis (15 coeffs for degrees 1-3, excluding DC)
        constexpr float SH_MULT[3][15] = {
            {1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, -1},  // X
            {-1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, 1},  // Y
            {1, -1, 1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, 1}   // Z
        };

        // Cumulative SH coefficient counts per degree (excluding DC)
        constexpr int SH_COEFF_COUNT[4] = {0, 3, 8, 15};

        struct MirrorCache {
            Tensor pos_mult[3];
            Tensor quat_mult[3];
            Tensor sh_mult[3][3];  // [axis][degree 1-3] - no degree 0 (empty)
            Device device = Device::CPU;
            bool valid = false;
        };

        std::mutex g_cache_mutex;
        MirrorCache g_cache;

        void ensure_cache(const Device device) {
            std::lock_guard lock(g_cache_mutex);

            if (g_cache.valid && g_cache.device == device)
                return;

            for (int a = 0; a < 3; ++a) {
                g_cache.pos_mult[a] = Tensor::from_vector(
                    {POS_MULT[a][0], POS_MULT[a][1], POS_MULT[a][2]}, {1, 3}, device);
                g_cache.quat_mult[a] = Tensor::from_vector(
                    {QUAT_MULT[a][0], QUAT_MULT[a][1], QUAT_MULT[a][2], QUAT_MULT[a][3]}, {1, 4}, device);

                // Only degrees 1-3 (degree 0 has no coeffs in shN)
                for (int d = 1; d <= 3; ++d) {
                    const int n = SH_COEFF_COUNT[d];
                    std::vector<float> v(n);
                    for (int i = 0; i < n; ++i)
                        v[i] = SH_MULT[a][i];
                    g_cache.sh_mult[a][d - 1] = Tensor::from_vector(v, {1, static_cast<size_t>(n), 1}, device);
                }
            }
            g_cache.device = device;
            g_cache.valid = true;
        }

    } // namespace

    glm::vec3 compute_selection_center(const SplatData& splat_data, const Tensor& selection_mask) {
        const auto& means = splat_data.means();
        if (!means.is_valid() || means.size(0) == 0)
            return glm::vec3(0.0f);

        const auto selected = selection_mask.ne(0);
        const int count = selected.sum_scalar();
        if (count == 0)
            return glm::vec3(0.0f);

        // Masked sum on GPU, only transfer 3 floats
        const auto mask_f = selected.to(DataType::Float32).unsqueeze(1);
        const auto masked = means * mask_f;
        const auto sum = masked.sum({0}, false).to(Device::CPU).contiguous();
        const auto* s = static_cast<const float*>(sum.data_ptr());
        const float inv = 1.0f / static_cast<float>(count);

        return {s[0] * inv, s[1] * inv, s[2] * inv};
    }

    void mirror_gaussians(SplatData& splat_data,
                          const Tensor& selection_mask,
                          const MirrorAxis axis,
                          const glm::vec3& center) {
        LOG_TIMER("mirror_gaussians");

        auto& means = splat_data.means();
        if (!means.is_valid() || means.size(0) == 0)
            return;

        const int a = static_cast<int>(axis);
        const auto device = means.device();
        ensure_cache(device);

        const auto selected = selection_mask.ne(0);
        if (selected.sum_scalar() == 0)
            return;

        auto indices = selected.nonzero();
        if (indices.ndim() == 2)
            indices = indices.squeeze(1);
        if (indices.dtype() != DataType::Int32)
            indices = indices.to(DataType::Int32);

        // Position: p' = p * mult + offset
        {
            const auto sel = means.index_select(0, indices);
            const float off = 2.0f * center[a];
            const auto offset = Tensor::from_vector(
                {a == 0 ? off : 0.0f, a == 1 ? off : 0.0f, a == 2 ? off : 0.0f}, {1, 3}, device);
            means.index_copy_(0, indices, sel * g_cache.pos_mult[a] + offset);
        }

        // Quaternion
        if (auto& rot = splat_data.rotation_raw(); rot.is_valid() && rot.size(0) > 0) {
            rot.index_copy_(0, indices, rot.index_select(0, indices) * g_cache.quat_mult[a]);
        }

        // SH coefficients (degrees 1-3 only, shN excludes DC)
        if (auto& shN = splat_data.shN(); shN.is_valid() && shN.size(0) > 0 && shN.size(1) > 0) {
            const int degree = static_cast<int>(std::sqrt(shN.size(1) + 1)) - 1;
            if (degree >= 1 && degree <= 3) {
                shN.index_copy_(0, indices, shN.index_select(0, indices) * g_cache.sh_mult[a][degree - 1]);
            }
        }
    }

} // namespace lfs::core

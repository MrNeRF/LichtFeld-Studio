/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/types.hpp"

#include <cstddef>
#include <cstring>

namespace lfs::gpu_ops {

    // POPSpa scores live in aligned UInt8 [N, 8] rows holding one nonnegative sum each.
    // Float64 stores an IEEE double. Float32Pair stores the unevaluated sum hi + lo of two
    // floats, with hi the float nearest the sum, for devices without FP64 arithmetic.
    enum class PopScoreEncoding { Float64,
                                  Float32Pair };

    // Host conversion of one 8-byte score row.
    inline double decode_pop_score(const void* row, const PopScoreEncoding encoding) {
        if (encoding == PopScoreEncoding::Float64) {
            double value;
            std::memcpy(&value, row, sizeof(value));
            return value;
        }
        float pair[2];
        std::memcpy(pair, row, sizeof(pair));
        return static_cast<double>(pair[0]) + static_cast<double>(pair[1]);
    }

    inline void encode_pop_score(const double value, void* row, const PopScoreEncoding encoding) {
        if (encoding == PopScoreEncoding::Float64) {
            std::memcpy(row, &value, sizeof(value));
            return;
        }
        const float hi = static_cast<float>(value);
        const float pair[2] = {hi, static_cast<float>(value - static_cast<double>(hi))};
        std::memcpy(row, pair, sizeof(pair));
    }

    // Masks are bool [N]; every frozen row is active. Selection keeps all frozen rows,
    // then the highest-priority remaining active rows; equal values prefer lower row ids.
    struct PopspaOps {
        PopScoreEncoding score_encoding = PopScoreEncoding::Float64;
        // keep: Int64 [k], ascending row ids.
        void (*keep_indices)(In scores, In active, In frozen, size_t k, Out keep);
        // Proxy z = sigmoid(raw) + u on the k rows selected by that value, 0 elsewhere.
        // With update_dual, then u += sigmoid(raw) - z on active, non-frozen rows.
        void (*project)(In raw_opacity, Out z, Out u, In active, In frozen, size_t k, bool update_dual);
        // On active, non-frozen rows, with a = sigmoid(raw) and d = a - z + u:
        // grad += rho * d * a * (1 - a) and loss[0] += 0.5 * rho * d^2.
        void (*penalty)(In raw_opacity, In z, In u, In active, In frozen, Out grad, Out loss, float rho);
        // Effective-rank needle plus the smallest-axis scale, averaged over active_count
        // rows, added to loss[0]; gradients go to grad_log_scales except on frozen rows.
        // active and frozen may be empty.
        void (*effective_rank)(In log_scales, Out grad_log_scales, In active, In frozen, Out loss,
                               size_t active_count, float rank_weight, float thin_weight, float epsilon);
        // Float32Pair only: the exact FP64 rows of scores into float64 ([N, 8]) on the device,
        // so checkpoints stream from storage the snapshot service can stage.
        void (*export_scores)(In scores, Out float64) = nullptr;
    };

} // namespace lfs::gpu_ops

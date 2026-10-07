/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace lfs::training {
    struct GumbelTopKScratch;
} // namespace lfs::training

namespace lfs::training::mrnf_strategy {
    struct MRNFBounds {
        float center[3] = {};
        float extent[3] = {};
        float median_size = 0.f;
        float max_extent = 0.f;
    };

} // namespace lfs::training::mrnf_strategy

namespace lfs::gpu_ops {

    struct Bounds {
        float center[3] = {};
        float extent[3] = {};
        float median_size = 0.f;
        float max_extent = 0.f;
    };

    struct MrnfNoiseParams {
        uint64_t seed = 0;
        float lr_mean = 0.f;
        float noise_weight = 0.f;
        float median_scale = 0.f;
    };

    struct DecayParams {
        float opacity_decay = 0.f;
        float scale_decay = 0.f;
        float train_t = 0.f;
        Tensor rendered_count;
    };

    struct GumbelParams {
        uint64_t seed = 0;
        size_t known_nnz = 0;
        bool compact_sparse = true;
    };

    // Scratch stays the caller's GumbelTopKScratch.
    // A null Gumbel scratch keeps the launcher's temporary-allocation path.
    struct MrnfOps {
        void (*noise)(
            Out means, In raw_opacity, In visibility, In frozen,
            const MrnfNoiseParams&);

        void (*decay)(
            Out raw_opacity, Out log_scales, In frozen,
            const DecayParams&);

        Bounds (*percentile_bounds)(In means, float percentile);

        void (*gumbel)(
            lfs::training::GumbelTopKScratch* scratch, In weights, Out indices,
            const GumbelParams&);

        // Folds densification row 1 into the max and clears only that row.
        void (*fold_error)(Out weight_max, Out densification);

        size_t (*compact_bool_indices)(In mask, Out indices, size_t count);
        void (*prune_bounds)(In means, In scale_max, Out mask, std::array<float, 3> center, float maximum, float log_maximum);
        void (*replace_parent_weights)(In opacity, In visibility, In active, In trainable, In edge, Out weights);
    };

} // namespace lfs::gpu_ops

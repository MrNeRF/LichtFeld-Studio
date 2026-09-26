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

namespace lfs::gpu_ops {

    struct Bounds {
        float center[3] = {};
        float extent[3] = {};
        float median_size = 0.f;
        float max_extent = 0.f;
    };

    struct ScalarValidity {
        float value = 0.f;
        bool valid = false;
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
        float far_decay_scale = 1.f;
        float train_t = 0.f;
    };

    struct GumbelParams {
        uint64_t seed = 0;
        size_t known_nnz = 0;
        bool compact_sparse = true;
    };

    struct ProjectParams {
        HW image;
        Intrinsics intrinsics;
        float near_plane = 0.01f;
    };

    // Scratch stays the caller's GumbelTopKScratch.
    // A null Gumbel scratch keeps the launcher's temporary-allocation path.
    struct MrnfOps {
        void (*noise)(
            Out means, In raw_opacity, In visibility, In frozen,
            const MrnfNoiseParams&);

        void (*decay)(
            Out raw_opacity, Out log_scales, In frozen, In far_mask,
            const DecayParams&);

        Bounds (*percentile_bounds)(In means, float percentile);
        ScalarValidity (*median_extent)(In raw_scales);

        void (*gumbel)(
            lfs::training::GumbelTopKScratch* scratch, In weights, Out indices,
            const GumbelParams&);

        // Zeros the first two densification rows. An invalid ratio_max is absent.
        void (*fold)(
            Out visibility, Out weight_max, Out densification,
            Out ratio_max, float ratio_power);

        // Folds densification row 1 into the max and clears only that row.
        void (*fold_error)(Out weight_max, Out densification);

        void (*project_centers)(
            In means, In view, Out means2d, Out radii, const ProjectParams&);

        void (*gather_center_error)(
            In means2d, In radii, In error, Out scores);

        void (*far_mask)(
            In means, Out mask, std::array<float, 3> center, float radius);

        void (*mean_abs_error)(In predicted, In target, Out error);
        void (*seed_weights)(In error, In alpha, Out weights);

        void (*gather_seeds)(
            In indices, In target, In alpha, In depth,
            Out rgb, Out sampled_alpha, Out sampled_depth);

        float (*sorted_median)(In values);
        void (*starvation_weights)(Out weights, In visibility, float median);
    };

} // namespace lfs::gpu_ops

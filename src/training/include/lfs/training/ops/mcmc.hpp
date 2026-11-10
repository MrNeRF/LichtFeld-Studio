/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/types.hpp"

#include <cstdint>

namespace lfs::gpu_ops {

    enum class SampleDomain { AliveIndices,
                              All };

    // SH-rest rows use their own packed-storage path.
    struct McmcRows {
        Out means, sh0, raw_scales, raw_quats, raw_opacity;
    };

    struct McmcOps {
        void (*initialize)(int n_max);
        void (*relocate)(In opacity, In scales, In ratios,
                         Out new_opacity, Out new_scales, float min_opacity);
        void (*noise)(In raw_opacity, In raw_scales, In raw_quats, In frozen,
                      Out means, uint64_t seed, float lr);
        void (*copy_rows)(In source_indices, In destination_indices, const McmcRows&);
        // Both input and output scales are logarithmic, as in the live caller.
        void (*update_rows)(In indices, In raw_scales, In raw_opacity,
                            Out raw_scales_out, Out raw_opacity_out);
        void (*sample)(In weights, In opacity, In raw_scales, In alive_indices,
                       Out indices, Out sampled_opacity, Out sampled_scales,
                       SampleDomain, uint64_t seed);
        // Fold row 1 into the max and zero both densification rows.
        void (*fold_error)(Out error_max, Out densification);
        // Draws indices.numel() rows with probability max(weight, min_weight);
        // nonzero rows of `excluded` (optional, [N]) are never drawn. False,
        // with nothing drawn, when no row has a positive weight.
        bool (*sample_rows)(In weights, float min_weight, In excluded, Out indices, uint64_t seed);
        // Each copy row already holds its source row. Splits the source's
        // opacity and scale over its copies (Equation 9) and writes them, raw,
        // to the copy and to the source. Without `copies` the copy of
        // sources[i] is row first_copy + i.
        void (*relocate_copies)(In sources, In copies, int64_t first_copy, In ratios, float min_opacity,
                                Out raw_scales, Out raw_opacity);
    };

} // namespace lfs::gpu_ops

namespace lfs::training {
    // sample_rows and relocate_copies built from tensor ops and a table's
    // relocate/update_rows, for backends without dedicated kernels.
    bool portable_mcmc_sample_rows(gpu_ops::In weights, float min_weight, gpu_ops::In excluded,
                                   gpu_ops::Out indices, uint64_t seed);
    void portable_mcmc_relocate_copies(const gpu_ops::McmcOps& ops, gpu_ops::In sources, gpu_ops::In copies,
                                       int64_t first_copy, gpu_ops::In ratios, float min_opacity,
                                       gpu_ops::Out raw_scales, gpu_ops::Out raw_opacity);

    template <const gpu_ops::McmcOps& (*Table)()>
    void relocate_copies_via(gpu_ops::In sources, gpu_ops::In copies, const int64_t first_copy, gpu_ops::In ratios,
                             const float min_opacity, gpu_ops::Out raw_scales, gpu_ops::Out raw_opacity) {
        portable_mcmc_relocate_copies(Table(), sources, copies, first_copy, ratios, min_opacity, raw_scales,
                                      raw_opacity);
    }
} // namespace lfs::training

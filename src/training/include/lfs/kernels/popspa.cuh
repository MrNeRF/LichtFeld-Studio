/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

namespace lfs::training::kernels {
    // All buffers live on the device and all work uses the supplied stream.
    void launch_popspa_keep_indices(const double* scores, const bool* active,
                                    const bool* frozen, size_t n, size_t k,
                                    int64_t* scratch_indices, int64_t* keep_indices,
                                    cudaStream_t stream);
    void launch_popspa_project(const float* raw_opacity, float* z, float* u,
                               const bool* active, const bool* frozen, size_t n,
                               size_t k, double* scratch_scores, int64_t* scratch_indices,
                               int64_t* keep_indices, bool update_dual, cudaStream_t stream);
    void launch_popspa_penalty(const float* raw_opacity, const float* z, const float* u,
                               const bool* active, const bool* frozen, float* grad,
                               float* loss, size_t n, float rho, cudaStream_t stream);
} // namespace lfs::training::kernels

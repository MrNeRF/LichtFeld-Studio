/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once
#include <cstddef>
#include <cuda_runtime_api.h>
namespace lfs::training::kernels {
    void launch_effective_rank_regularization(const float* log_scales, float* grad,
                                              const bool* active, const bool* frozen,
                                              float* loss, size_t n, size_t active_count,
                                              float rank_weight, float thin_weight,
                                              float epsilon, cudaStream_t stream);
}

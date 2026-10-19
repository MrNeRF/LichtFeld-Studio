/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "core/tensor.hpp"

namespace lfs::core::cuda {

    /// MRNF initial scales on the current stream. Every row of `scaling` [N, 3] receives, in all three
    /// columns, log(clamp((d1 + d2) / 4, 1e-3, max_scale)) where d1 and d2 are the exact distances to the
    /// two nearest other points of `means` [N, 3] and max_scale is a tenth of the median per-axis extent
    /// of the central 75% of finite coordinates (at least 1e-3). Requires N >= 3.
    LFS_CUDA_API void mrnf_knn_log_scales(const Tensor& means, Tensor& scaling);

} // namespace lfs::core::cuda

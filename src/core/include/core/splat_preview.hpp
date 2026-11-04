/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

namespace lfs::core {
    class SplatData;
    class Tensor;

    // Owned canonical Float32 CPU coefficients [selected rows, (degree+1)^2-1, 3].
    // Gather directly from resident storage, without decoding the entire model.
    // Call at a model read boundary; rows must be rank-1 CPU Int32 or Int64.
    LFS_CORE_API Tensor splat_sh_preview(const SplatData& model, const Tensor& rows, int degree);
} // namespace lfs::core

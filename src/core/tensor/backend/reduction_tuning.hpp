/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <cstddef>
namespace lfs::core::internal {
    enum class ReductionTuningPath { None,
                                     StridedFast,
                                     Transpose,
                                     Column,
                                     Default };
    // Preserve the existing build's tuning/test hooks without SDK headers in
    // shared dispatch. CMake selects the native hooks or the standard policy.
    ReductionTuningPath reduction_tuning_override() noexcept;
    bool prefer_strided_reduction(size_t outer, size_t reduced, size_t inner) noexcept;
    void record_reduction_tuning_path(ReductionTuningPath) noexcept;
} // namespace lfs::core::internal

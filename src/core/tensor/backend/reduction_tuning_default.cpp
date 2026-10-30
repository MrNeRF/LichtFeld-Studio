/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "reduction_tuning.hpp"
namespace lfs::core::internal {
    ReductionTuningPath reduction_tuning_override() noexcept { return ReductionTuningPath::None; }
    bool prefer_strided_reduction(size_t, size_t, size_t) noexcept { return true; }
    void record_reduction_tuning_path(ReductionTuningPath) noexcept {}
} // namespace lfs::core::internal

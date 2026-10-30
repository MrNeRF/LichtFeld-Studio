/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../reduction_tuning.hpp"
#include "kernels/tensor_ops.hpp"
namespace lfs::core::internal {
    static_assert(int(ReductionTuningPath::None) == int(tensor_ops::ReducePathForTesting::None));
    static_assert(int(ReductionTuningPath::StridedFast) == int(tensor_ops::ReducePathForTesting::StridedFast));
    static_assert(int(ReductionTuningPath::Transpose) == int(tensor_ops::ReducePathForTesting::Transpose));
    static_assert(int(ReductionTuningPath::Column) == int(tensor_ops::ReducePathForTesting::Column));
    static_assert(int(ReductionTuningPath::Default) == int(tensor_ops::ReducePathForTesting::Default));
    ReductionTuningPath reduction_tuning_override() noexcept { return static_cast<ReductionTuningPath>(tensor_ops::reduce_path_override_for_testing()); }
    bool prefer_strided_reduction(size_t outer, size_t reduced, size_t inner) noexcept { return tensor_ops::should_prefer_strided_over_transpose(outer, reduced, inner); }
    void record_reduction_tuning_path(ReductionTuningPath path) noexcept { tensor_ops::set_reduce_last_path_for_testing(static_cast<tensor_ops::ReducePathForTesting>(path)); }
} // namespace lfs::core::internal

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cuda_runtime.h>
#include <limits>

namespace lfs::core {

    // Start values for max/min reductions. numeric_limits<T>::infinity() is zero
    // for integer types, so those start from their extreme values instead.
    template <typename T>
    __host__ __device__ constexpr T max_reduce_identity() {
        static_assert(std::numeric_limits<T>::is_specialized, "max reduction needs numeric_limits<T>");
        if constexpr (std::numeric_limits<T>::has_infinity)
            return -std::numeric_limits<T>::infinity();
        else
            return std::numeric_limits<T>::lowest();
    }

    template <typename T>
    __host__ __device__ constexpr T min_reduce_identity() {
        static_assert(std::numeric_limits<T>::is_specialized, "min reduction needs numeric_limits<T>");
        if constexpr (std::numeric_limits<T>::has_infinity)
            return std::numeric_limits<T>::infinity();
        else
            return std::numeric_limits<T>::max();
    }

} // namespace lfs::core

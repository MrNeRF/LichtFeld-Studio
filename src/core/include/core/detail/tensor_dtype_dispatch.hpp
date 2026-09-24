/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/assert.hpp"
#include "core/detail/tensor_half.hpp"
#include "core/tensor_fwd.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace lfs::core::detail {

#if defined(__CUDACC__)
#define LFS_TENSOR_DTYPE_HD __host__ __device__
#else
#define LFS_TENSOR_DTYPE_HD
#endif

    template <typename Function>
    void dispatch_dtype(const DataType dtype, Function&& function) {
        switch (dtype) {
        case DataType::Float32:
            std::forward<Function>(function).template operator()<float>();
            return;
        case DataType::Float16:
            std::forward<Function>(function).template operator()<detail::tensor_half_t>();
            return;
        case DataType::Int32:
            std::forward<Function>(function).template operator()<int32_t>();
            return;
        case DataType::Int64:
            std::forward<Function>(function).template operator()<int64_t>();
            return;
        case DataType::UInt8:
            std::forward<Function>(function).template operator()<uint8_t>();
            return;
        case DataType::Bool:
            std::forward<Function>(function).template operator()<bool>();
            return;
        case DataType::UInt32:
            std::forward<Function>(function).template operator()<uint32_t>();
            return;
        }

        LFS_ASSERT_MSG(false,
                       "unsupported dtype reached exhaustive tensor dispatch");
    }

    inline void require_scalar_representable(const DataType dtype,
                                             const float value,
                                             const std::string_view operation) {
        const auto failure = [&](const std::string_view reason) {
            LFS_ASSERT_MSG(false, std::string(operation) + ": " + std::string(reason));
        };

        switch (dtype) {
        case DataType::Float32:
        case DataType::Bool:
            return;
        case DataType::Float16:
            if (std::isfinite(value) && std::abs(value) > 65504.0f) {
                failure("scalar is outside the Float16 range");
            }
            return;
        case DataType::Int32:
            if (!std::isfinite(value) ||
                static_cast<double>(value) <
                    static_cast<double>(std::numeric_limits<int32_t>::lowest()) ||
                static_cast<double>(value) >
                    static_cast<double>(std::numeric_limits<int32_t>::max())) {
                failure("scalar is outside the Int32 range");
            }
            return;
        case DataType::Int64:
            if (!std::isfinite(value) ||
                static_cast<long double>(value) <
                    static_cast<long double>(std::numeric_limits<int64_t>::lowest()) ||
                static_cast<long double>(value) >
                    static_cast<long double>(std::numeric_limits<int64_t>::max())) {
                failure("scalar is outside the Int64 range");
            }
            return;
        case DataType::UInt8:
            if (!std::isfinite(value) || value < 0.0f || value > 255.0f) {
                failure("scalar is outside the UInt8 range");
            }
            return;
        case DataType::UInt32:
            if (!std::isfinite(value) || value < 0.0f ||
                static_cast<double>(value) >
                    static_cast<double>(std::numeric_limits<uint32_t>::max())) {
                failure("scalar is outside the UInt32 range");
            }
            return;
        }

        failure("unsupported dtype");
    }

    inline bool is_exact_int32_scalar(const float value) {
        return std::isfinite(value) && std::trunc(value) == value &&
               static_cast<double>(value) >=
                   static_cast<double>(std::numeric_limits<int32_t>::lowest()) &&
               static_cast<double>(value) <=
                   static_cast<double>(std::numeric_limits<int32_t>::max());
    }

    LFS_TENSOR_DTYPE_HD inline uint8_t torch_uint8_cast(const float value) {
#ifdef __CUDA_ARCH__
        if (!isfinite(value)) {
            return 0;
        }
        float wrapped = fmodf(truncf(value), 256.0f);
#else
        if (!std::isfinite(value)) {
            return 0;
        }
        float wrapped = std::fmod(std::trunc(value), 256.0f);
#endif
        if (wrapped < 0.0f) {
            wrapped += 256.0f;
        }
        return static_cast<uint8_t>(wrapped);
    }

    LFS_TENSOR_DTYPE_HD inline uint8_t torch_uint8_cast(const detail::tensor_half_t value) {
        return torch_uint8_cast(detail::tensor_half_to_float(value));
    }

    template <typename T>
        requires std::is_integral_v<T>
    LFS_TENSOR_DTYPE_HD inline uint8_t torch_uint8_cast(const T value) {
        return static_cast<uint8_t>(value);
    }

#undef LFS_TENSOR_DTYPE_HD

} // namespace lfs::core::detail

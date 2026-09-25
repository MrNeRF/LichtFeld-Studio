/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "core/detail/tensor_half.hpp"
#include "gpu_backend_ops.hpp"

#include <cstdint>
#include <cstring>

namespace lfs::core::internal {

    // Scalar operands as the integer, float and fill bit patterns GPU kernels consume.
    inline uint64_t scalar_integer(const ScalarOperand scalar) {
        switch (scalar.kind) {
        case ScalarKind::Float: return 0;
        case ScalarKind::Int32:
            return static_cast<uint64_t>(static_cast<int64_t>(scalar.value.int32_value));
        case ScalarKind::Int64: return static_cast<uint64_t>(scalar.value.int64_value);
        case ScalarKind::Bool: return scalar.value.bool_value ? 1 : 0;
        }
        return 0;
    }

    inline float scalar_float(const ScalarOperand scalar) {
        switch (scalar.kind) {
        case ScalarKind::Float: return scalar.value.float_value;
        case ScalarKind::Int32: return static_cast<float>(scalar.value.int32_value);
        case ScalarKind::Int64: return static_cast<float>(scalar.value.int64_value);
        case ScalarKind::Bool: return scalar.value.bool_value ? 1.0f : 0.0f;
        }
        return 0.0f;
    }

    inline uint64_t fill_pattern(const DataType dtype, const ScalarOperand value) {
        uint64_t pattern = 0;
        switch (dtype) {
        case DataType::Float32: {
            const float converted = scalar_float(value);
            std::memcpy(&pattern, &converted, sizeof(converted));
            return pattern;
        }
        case DataType::Float16: {
            const detail::tensor_half_t converted = detail::tensor_float_to_half(scalar_float(value));
            std::memcpy(&pattern, &converted, sizeof(converted));
            return pattern;
        }
        case DataType::Int32: {
            const int32_t converted = static_cast<int32_t>(scalar_integer(value));
            std::memcpy(&pattern, &converted, sizeof(converted));
            return pattern;
        }
        case DataType::UInt32: {
            const uint32_t converted = static_cast<uint32_t>(scalar_integer(value));
            std::memcpy(&pattern, &converted, sizeof(converted));
            return pattern;
        }
        case DataType::Int64: return scalar_integer(value);
        case DataType::Bool: return scalar_float(value) != 0.0f ? 1 : 0;
        case DataType::UInt8: return static_cast<uint8_t>(scalar_integer(value));
        }
        return 0;
    }

} // namespace lfs::core::internal

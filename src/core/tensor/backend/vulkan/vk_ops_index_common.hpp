/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "../../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "vk_ops_common.hpp"

#include <cuda_fp16.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <utility>

// Helpers shared by the index, mask and sort adapters.
namespace lfs::core::internal::vk_index {

    // Element codes of index.slang and mask.slang follow DataType.
    inline uint32_t shader_dtype(const DataType dtype) {
        switch (dtype) {
        case DataType::Float32: return 0;
        case DataType::Float16: return 1;
        case DataType::Int32: return 2;
        case DataType::Int64: return 3;
        case DataType::UInt8: return 4;
        case DataType::Bool: return 5;
        default: break;
        }
        LFS_ASSERT_MSG(false, "Vulkan index operation received an unsupported dtype");
        return 0;
    }

    inline std::array<uint32_t, MAX_TENSOR_RANK> shader_dims(const StridedLayout& layout) {
        LFS_ASSERT_MSG(layout.rank <= MAX_TENSOR_RANK,
                       "Vulkan layout rank exceeds MAX_TENSOR_RANK");
        std::array<uint32_t, MAX_TENSOR_RANK> result{};
        for (size_t i = 0; i < layout.rank; ++i) {
            result[i] = vk::checked_u32(layout.dims[i], "Vulkan dimension exceeds uint32");
        }
        return result;
    }

    // Bit pattern of a scalar in the element type, split into two 32-bit words.
    inline std::pair<uint32_t, uint32_t> fill_bits(const DataType dtype, const ScalarOperand value) {
        float as_float = 0.0f;
        int64_t as_integer = 0;
        switch (value.kind) {
        case ScalarKind::Float:
            as_float = value.value.float_value;
            as_integer = static_cast<int64_t>(value.value.float_value);
            break;
        case ScalarKind::Int32:
            as_float = static_cast<float>(value.value.int32_value);
            as_integer = value.value.int32_value;
            break;
        case ScalarKind::Int64:
            as_float = static_cast<float>(value.value.int64_value);
            as_integer = value.value.int64_value;
            break;
        case ScalarKind::Bool:
            as_float = value.value.bool_value ? 1.0f : 0.0f;
            as_integer = value.value.bool_value ? 1 : 0;
            break;
        }
        switch (dtype) {
        case DataType::Float32: {
            uint32_t bits = 0;
            std::memcpy(&bits, &as_float, sizeof(bits));
            return {bits, 0};
        }
        case DataType::Float16: {
            const __half converted = __float2half_rn(as_float);
            uint16_t bits = 0;
            std::memcpy(&bits, &converted, sizeof(bits));
            return {bits, 0};
        }
        case DataType::Int32:
        case DataType::Int64:
        case DataType::UInt8:
        case DataType::Bool: {
            const auto bits = static_cast<uint64_t>(as_integer);
            return {static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32)};
        }
        default: break;
        }
        LFS_ASSERT_MSG(false, "unsupported Vulkan fill dtype");
        return {0, 0};
    }

} // namespace lfs::core::internal::vk_index

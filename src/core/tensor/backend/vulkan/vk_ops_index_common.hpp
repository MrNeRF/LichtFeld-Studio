/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "../../internal/tensor_impl.hpp"
#include "../scalar_operand.hpp"
#include "core/assert.hpp"
#include "vk_ops_common.hpp"

#include <array>
#include <cstdint>

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

} // namespace lfs::core::internal::vk_index

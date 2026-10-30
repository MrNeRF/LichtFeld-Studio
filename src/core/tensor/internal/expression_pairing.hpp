/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "expression_runtime.hpp"
#include "tensor_impl.hpp"
#include <span>
#include <stdexcept>
namespace lfs::core::internal {
    struct ExpressionPairingRequest {
        const ExpressionProgram& program;
        std::span<const ExprInput> inputs;
        std::span<const std::array<int32_t, ExpressionProgram::max_rank>> strides;
    };
    inline uint64_t descriptor_expression_address(StorageRef storage) {
        if (!storage.meta || !storage.meta->gpu_descriptor.base_address)
            throw std::invalid_argument("expression storage has no device address");
        return storage.meta->gpu_descriptor.base_address + storage.byte_offset;
    }
} // namespace lfs::core::internal

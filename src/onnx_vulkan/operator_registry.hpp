/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "model.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace lfs::onnx_vulkan::detail {

    struct AttributeSchema {
        std::string_view name;
        AttributeType type;
        bool required = false;
    };

    struct OperatorSchema {
        std::string_view name;
        std::int64_t since_version = 1;
        std::int64_t through_version = 14;
        std::size_t minimum_inputs = 0;
        std::size_t maximum_inputs = 0;
        std::size_t minimum_outputs = 1;
        std::size_t maximum_outputs = 1;
        std::span<const AttributeSchema> attributes;
    };

    [[nodiscard]] const OperatorSchema* find_operator_schema(std::string_view name,
                                                              std::int64_t opset) noexcept;
    [[nodiscard]] std::span<const OperatorSchema> operator_registry() noexcept;

} // namespace lfs::onnx_vulkan::detail

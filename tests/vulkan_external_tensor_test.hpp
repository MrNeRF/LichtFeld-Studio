/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor.hpp"
#include "window/vulkan_context.hpp"

#include <cstddef>
#include <exception>
#include <expected>
#include <string>
#include <utility>

namespace lfs::test {

    // A CUDA tensor the Vulkan viewer renderer can bind, allocated as the app's
    // makeGraphicsExternalTensor does, for fixtures that own a bare VulkanContext.
    inline std::expected<core::Tensor, std::string> makeVulkanExternalTensor(vis::VulkanContext& context,
                                                                             core::TensorShape shape,
                                                                             const core::DataType dtype,
                                                                             const std::size_t capacity,
                                                                             const char* const label) {
        try {
            auto tensor = context.tensorInterop().empty(std::move(shape), dtype, core::GpuBackend::CUDA, capacity);
            if (label)
                tensor.set_name(label);
            return tensor;
        } catch (const std::exception& error) {
            return std::unexpected(std::string(error.what()));
        }
    }

} // namespace lfs::test

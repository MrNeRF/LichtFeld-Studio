/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace lfs::training::vulkan {
    struct EmbeddedShader {
        std::string_view name;
        std::span<const uint32_t> words;
    };

    std::span<const EmbeddedShader> embedded_training_shaders();
} // namespace lfs::training::vulkan

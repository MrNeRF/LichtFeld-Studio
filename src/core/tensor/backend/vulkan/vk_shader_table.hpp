/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string_view>

namespace lfs::core::internal {

    // One compiled compute module with the facts the loader checks, all read
    // from the finalized SPIR-V at build time by lfs_tensor_spirv_finalize.
    struct EmbeddedShader {
        std::string_view name;
        std::span<const uint32_t> code;
        std::string_view entry_point;
        std::array<uint32_t, 3> local_size;
        uint32_t push_constant_size;
        std::span<const std::string_view> capabilities;
        std::span<const uint32_t> float_widths;
        std::span<const uint32_t> signed_zero_inf_nan_preserve;
    };

    [[nodiscard]] LFS_CORE_API std::span<const EmbeddedShader> embedded_shaders();
    [[nodiscard]] LFS_CORE_API const EmbeddedShader* find_embedded_shader(std::string_view name);

} // namespace lfs::core::internal

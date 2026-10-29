/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <iterator>
#include <string_view>

namespace lfs::rendering {
    enum class CustomKind { None,
                            Host,
                            Spirv };
    // Survivor projection follows custom cull; 3DGUT projection precedes custom rasters.
    enum class CustomChain { None,
                             Survivors,
                             Gut };
    struct ShaderVariants {
        std::string_view program, banded, reveal, custom, custom_banded;
        CustomKind custom_kind;
        CustomChain chain;
        std::string_view customVariantLabel(bool banded) const {
            return banded && !custom_banded.empty() ? custom_banded : custom;
        }
    };
    inline constexpr ShaderVariants shader_variants[] = {
#include "shader_variants.inc"
    };
    inline const ShaderVariants* shaderVariants(std::string_view name) {
        const auto found = std::ranges::find(shader_variants, name, &ShaderVariants::program);
        return found == std::end(shader_variants) ? nullptr : found;
    }
} // namespace lfs::rendering

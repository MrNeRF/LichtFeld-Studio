/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <string>

namespace lfs::core {

    struct Material {
        glm::vec4 base_color{1.0f, 1.0f, 1.0f, 1.0f};
        glm::vec3 emissive{0.0f};
        float metallic = 0.0f;
        float roughness = 1.0f;
        float ao = 1.0f;

        uint32_t albedo_tex = 0;
        uint32_t normal_tex = 0;
        uint32_t metallic_roughness_tex = 0;
        uint32_t emissive_tex = 0;
        uint32_t ao_tex = 0;

        std::string albedo_tex_path;
        std::string normal_tex_path;
        std::string metallic_roughness_tex_path;

        bool double_sided = false;
        std::string name;

        bool has_albedo_texture() const { return albedo_tex != 0; }
        bool has_normal_texture() const { return normal_tex != 0; }
        bool has_metallic_roughness_texture() const { return metallic_roughness_tex != 0; }
    };

} // namespace lfs::core

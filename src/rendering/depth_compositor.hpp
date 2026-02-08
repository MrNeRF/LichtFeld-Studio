/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>

namespace lfs::rendering {

    class DepthCompositor {
    public:
        Result<void> initialize();
        bool isInitialized() const { return initialized_; }

        Result<void> composite(GLuint splat_color_tex, GLuint splat_depth_tex,
                               GLuint mesh_color_tex, GLuint mesh_depth_tex,
                               float near_plane, float far_plane,
                               bool flip_splat_y = true,
                               const glm::vec2& splat_texcoord_scale = glm::vec2(1.0f),
                               bool splat_depth_is_ndc = false);

    private:
        ManagedShader shader_;
        VAO vao_;
        VBO vbo_;
        bool initialized_ = false;
    };

} // namespace lfs::rendering

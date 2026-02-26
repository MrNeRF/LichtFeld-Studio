/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "grid_renderer.hpp"
#include "core/logger.hpp"
#include "gl_state_guard.hpp"
#include "shader_paths.hpp"
#include <cmath>

namespace lfs::rendering {

    namespace {
        constexpr int QUAD_VERTEX_COUNT = 4;
    }

    Result<void> RenderInfiniteGrid::init() {
        if (initialized_)
            return {};

        LOG_TIMER("RenderInfiniteGrid::init");

        auto shader_result = load_shader("infinite_grid", "infinite_grid.vert", "infinite_grid.frag", false);
        if (!shader_result) {
            LOG_ERROR("Failed to load grid shader: {}", shader_result.error().what());
            return std::unexpected(shader_result.error().what());
        }
        shader_ = std::move(*shader_result);

        auto vao_result = create_vao();
        if (!vao_result) {
            LOG_ERROR("Failed to create VAO: {}", vao_result.error());
            return std::unexpected(vao_result.error());
        }

        auto vbo_result = create_vbo();
        if (!vbo_result) {
            LOG_ERROR("Failed to create VBO: {}", vbo_result.error());
            return std::unexpected(vbo_result.error());
        }
        vbo_ = std::move(*vbo_result);

        constexpr float VERTICES[] = {-1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f};
        const std::span<const float> vertices_span(VERTICES);

        VAOBuilder builder(std::move(*vao_result));
        builder.attachVBO(vbo_, vertices_span, GL_STATIC_DRAW)
            .setAttribute({.index = 0,
                           .size = 2,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = 2 * sizeof(float),
                           .offset = nullptr,
                           .divisor = 0});
        vao_ = builder.build();

        initialized_ = true;
        return {};
    }

    void RenderInfiniteGrid::computeFrustumPerspective(const glm::mat4& view_inv, const float fov_y, const float aspect,
                                                       glm::vec3& near_origin, glm::vec3& far_origin,
                                                       glm::vec3& far_x, glm::vec3& far_y) const {
        const glm::vec3 cam_pos = glm::vec3(view_inv[3]);
        const glm::vec3 cam_right = glm::vec3(view_inv[0]);
        const glm::vec3 cam_up = glm::vec3(view_inv[1]);
        const glm::vec3 cam_forward = -glm::vec3(view_inv[2]);

        near_origin = cam_pos;

        const float half_height = std::tan(fov_y * 0.5f);
        const float half_width = half_height * aspect;

        const glm::vec3 far_center = cam_pos + cam_forward;
        const glm::vec3 right_offset = cam_right * half_width;
        const glm::vec3 up_offset = cam_up * half_height;

        const glm::vec3 far_bl = far_center - right_offset - up_offset;
        const glm::vec3 far_br = far_center + right_offset - up_offset;
        const glm::vec3 far_tl = far_center - right_offset + up_offset;

        far_origin = far_bl;
        far_x = far_br - far_bl;
        far_y = far_tl - far_bl;
    }

    void RenderInfiniteGrid::computeFrustumOrthographic(const glm::mat4& view_inv, const float half_width, const float half_height,
                                                        glm::vec3& near_origin, glm::vec3& near_x, glm::vec3& near_y,
                                                        glm::vec3& far_origin, glm::vec3& far_x, glm::vec3& far_y) const {
        const glm::vec3 cam_pos = glm::vec3(view_inv[3]);
        const glm::vec3 cam_right = glm::vec3(view_inv[0]);
        const glm::vec3 cam_up = glm::vec3(view_inv[1]);
        const glm::vec3 cam_forward = -glm::vec3(view_inv[2]);

        // Orthographic: parallel rays with identical near/far plane dimensions
        const glm::vec3 right_offset = cam_right * half_width;
        const glm::vec3 up_offset = cam_up * half_height;

        // Extended ray range ensures grid intersection from any camera position
        constexpr float RAY_NEAR = -1000.0f;
        constexpr float RAY_FAR = 1000.0f;

        const glm::vec3 near_center = cam_pos + cam_forward * RAY_NEAR;
        near_origin = near_center - right_offset - up_offset;
        near_x = right_offset * 2.0f;
        near_y = up_offset * 2.0f;

        const glm::vec3 far_center = cam_pos + cam_forward * RAY_FAR;
        far_origin = far_center - right_offset - up_offset;
        far_x = right_offset * 2.0f;
        far_y = up_offset * 2.0f;
    }

    Result<void> RenderInfiniteGrid::render(const glm::mat4& view, const glm::mat4& projection,
                                            const bool orthographic) {
        if (!initialized_ || !shader_.valid())
            return std::unexpected("Grid renderer not initialized");

        const glm::mat4 view_inv = glm::inverse(view);
        const glm::vec3 view_position = glm::vec3(view_inv[3]);

        glm::vec3 near_origin, near_x, near_y, far_origin, far_x, far_y;

        if (orthographic) {
            // Extract half-extents from orthographic projection matrix
            const float half_width = 1.0f / projection[0][0];
            const float half_height = 1.0f / std::abs(projection[1][1]);
            computeFrustumOrthographic(view_inv, half_width, half_height,
                                       near_origin, near_x, near_y, far_origin, far_x, far_y);
        } else {
            // Perspective: rays converge to camera position
            const float fov_y = 2.0f * std::atan(1.0f / std::abs(projection[1][1]));
            const float aspect = std::abs(projection[1][1] / projection[0][0]);
            computeFrustumPerspective(view_inv, fov_y, aspect, near_origin, far_origin, far_x, far_y);
            near_x = glm::vec3{0.0f};
            near_y = glm::vec3{0.0f};
        }

        const glm::mat4 view_proj = projection * view;

        GLStateGuard state_guard;

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glDepthMask(GL_TRUE);

        ShaderScope s(shader_);

        if (auto r = s->set("near_origin", near_origin); !r)
            return r;
        if (auto r = s->set("near_x", near_x); !r)
            return r;
        if (auto r = s->set("near_y", near_y); !r)
            return r;
        if (auto r = s->set("far_origin", far_origin); !r)
            return r;
        if (auto r = s->set("far_x", far_x); !r)
            return r;
        if (auto r = s->set("far_y", far_y); !r)
            return r;
        if (auto r = s->set("view_position", view_position); !r)
            return r;
        if (auto r = s->set("matrix_viewProjection", view_proj); !r)
            return r;
        if (auto r = s->set("plane", static_cast<int>(plane_)); !r)
            return r;
        if (auto r = s->set("opacity", opacity_); !r)
            return r;

        VAOBinder vao_bind(vao_);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, QUAD_VERTEX_COUNT);

        return {};
    }

} // namespace lfs::rendering

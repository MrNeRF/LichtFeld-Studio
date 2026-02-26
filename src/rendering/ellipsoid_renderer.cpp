/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "ellipsoid_renderer.hpp"
#include "core/logger.hpp"
#include "gl_state_guard.hpp"
#include "shader_paths.hpp"
#include <glm/gtc/constants.hpp>

namespace lfs::rendering {

    EllipsoidRenderer::EllipsoidRenderer() {
        createSphereGeometry();
    }

    void EllipsoidRenderer::createSphereGeometry() {
        vertices_.clear();
        indices_.clear();

        for (int lat = 0; lat <= LAT_SEGMENTS; ++lat) {
            const float theta = static_cast<float>(lat) / LAT_SEGMENTS * glm::pi<float>();
            const float sin_theta = std::sin(theta);
            const float cos_theta = std::cos(theta);

            for (int lon = 0; lon <= LON_SEGMENTS; ++lon) {
                const float phi = static_cast<float>(lon) / LON_SEGMENTS * 2.0f * glm::pi<float>();
                const float sin_phi = std::sin(phi);
                const float cos_phi = std::cos(phi);

                vertices_.emplace_back(sin_theta * cos_phi, cos_theta, sin_theta * sin_phi);
            }
        }

        for (int lat = 0; lat < LAT_SEGMENTS; lat += 2) {
            for (int lon = 0; lon < LON_SEGMENTS; ++lon) {
                const unsigned int curr = lat * (LON_SEGMENTS + 1) + lon;
                indices_.push_back(curr);
                indices_.push_back(curr + 1);
            }
        }

        for (int lon = 0; lon < LON_SEGMENTS; lon += 2) {
            for (int lat = 0; lat < LAT_SEGMENTS; ++lat) {
                const unsigned int curr = lat * (LON_SEGMENTS + 1) + lon;
                indices_.push_back(curr);
                indices_.push_back(curr + LON_SEGMENTS + 1);
            }
        }
    }

    Result<void> EllipsoidRenderer::init() {
        if (initialized_)
            return {};

        LOG_INFO("Initializing ellipsoid renderer");

        auto result = load_shader("ellipsoid", "ellipsoid.vert", "ellipsoid.frag", false);
        if (!result) {
            LOG_ERROR("Failed to load ellipsoid shader: {}", result.error().what());
            return std::unexpected(result.error().what());
        }
        shader_ = std::move(*result);

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

        auto ebo_result = create_vbo();
        if (!ebo_result) {
            LOG_ERROR("Failed to create EBO: {}", ebo_result.error());
            return std::unexpected(ebo_result.error());
        }
        ebo_ = std::move(*ebo_result);

        VAOBuilder builder(std::move(*vao_result));
        builder.attachVBO(vbo_)
            .setAttribute({.index = 0,
                           .size = 3,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = sizeof(glm::vec3),
                           .offset = nullptr,
                           .divisor = 0})
            .attachEBO(ebo_);

        vao_ = builder.build();
        initialized_ = true;

        if (auto setup_result = setupVertexData(); !setup_result) {
            initialized_ = false;
            return setup_result;
        }

        LOG_INFO("Ellipsoid renderer initialized");
        return {};
    }

    Result<void> EllipsoidRenderer::setupVertexData() {
        if (!initialized_ || !vao_)
            return std::unexpected("Ellipsoid renderer not initialized");

        BufferBinder<GL_ARRAY_BUFFER> vbo_bind(vbo_);
        upload_buffer(GL_ARRAY_BUFFER, std::span(vertices_), GL_STATIC_DRAW);

        BufferBinder<GL_ELEMENT_ARRAY_BUFFER> ebo_bind(ebo_);
        upload_buffer(GL_ELEMENT_ARRAY_BUFFER, std::span(indices_), GL_STATIC_DRAW);

        return {};
    }

    Result<void> EllipsoidRenderer::render(const glm::mat4& view, const glm::mat4& projection) {
        if (!initialized_ || !shader_.valid() || !vao_)
            return std::unexpected("Ellipsoid renderer not initialized");

        GLStateGuard state_guard;

        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        GLLineGuard line_guard(line_width_);
        ShaderScope s(shader_);

        const glm::mat4 scale_matrix = glm::scale(glm::mat4(1.0f), radii_);
        const glm::mat4 model = transform_ * scale_matrix;
        const glm::mat4 mvp = projection * view * model;

        const glm::mat4 inv_view = glm::inverse(view);
        const glm::vec3 camera_pos = glm::vec3(inv_view[3]);
        const glm::vec3 ellipsoid_center = glm::vec3(transform_[3]);
        const glm::vec3 view_dir = glm::normalize(ellipsoid_center - camera_pos);

        const glm::mat3 normal_matrix = glm::transpose(glm::inverse(glm::mat3(model)));
        const glm::vec3 local_view_dir = glm::normalize(glm::inverse(normal_matrix) * view_dir);

        if (auto result = s->set("u_mvp", mvp); !result)
            return result;
        if (auto result = s->set("u_color", color_); !result)
            return result;
        if (auto result = s->set("u_view_dir", local_view_dir); !result)
            return result;

        VAOBinder vao_bind(vao_);
        glDrawElements(GL_LINES, static_cast<GLsizei>(indices_.size()), GL_UNSIGNED_INT, nullptr);

        return {};
    }

} // namespace lfs::rendering

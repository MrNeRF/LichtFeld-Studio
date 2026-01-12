/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "bbox_renderer.hpp"
#include "core/logger.hpp"
#include "gl_state_guard.hpp"
#include "shader_paths.hpp"

namespace lfs::rendering {

    RenderBoundingBox::RenderBoundingBox() : color_(1.0f, 1.0f, 0.0f), // Yellow by default
                                             line_width_(2.0f),
                                             initialized_(false) {
        // Initialize vertices vector with 8 vertices
        vertices_.resize(8);

        // Initialize indices vector with line indices
        indices_.assign(cube_line_indices_, cube_line_indices_ + 24);

        LOG_DEBUG("RenderBoundingBox created with default color (yellow) and line width {}", line_width_);
    }

    void RenderBoundingBox::setBounds(const glm::vec3& min, const glm::vec3& max) {
        LOG_TRACE("Setting bounding box bounds: min=({}, {}, {}), max=({}, {}, {})",
                  min.x, min.y, min.z, max.x, max.y, max.z);

        // Call base class implementation
        BoundingBox::setBounds(min, max);
        createCubeGeometry();

        if (isInitialized()) {
            if (auto result = setupVertexData(); !result) {
                LOG_ERROR("Failed to setup vertex data for render bounding box: {}", result.error());
            }
        }
    }

    Result<void> RenderBoundingBox::init() {
        if (isInitialized())
            return {};

        LOG_TIMER("RenderBoundingBox::init");
        LOG_INFO("Initializing bounding box renderer");

        // Create shader for bounding box rendering
        auto result = load_shader("bounding_box", "bounding_box.vert", "bounding_box.frag", false);
        if (!result) {
            LOG_ERROR("Failed to load bounding box shader: {}", result.error().what());
            return std::unexpected(result.error().what());
        }
        shader_ = std::move(*result);

        // Create OpenGL objects using RAII
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

        auto ebo_result = create_vbo(); // EBO is also a buffer
        if (!ebo_result) {
            LOG_ERROR("Failed to create EBO: {}", ebo_result.error());
            return std::unexpected(ebo_result.error());
        }
        ebo_ = std::move(*ebo_result);

        // Build VAO using VAOBuilder
        VAOBuilder builder(std::move(*vao_result));

        // We'll set up the structure now, data will come in setupVertexData
        builder.attachVBO(vbo_) // Attach without data initially
            .setAttribute({.index = 0,
                           .size = 3,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = sizeof(glm::vec3),
                           .offset = nullptr,
                           .divisor = 0})
            .attachEBO(ebo_); // Attach EBO without data initially

        vao_ = builder.build();

        initialized_ = true;

        // Initialize cube geometry with default bounds if not already set
        if (min_bounds_ == glm::vec3(0.0f) && max_bounds_ == glm::vec3(0.0f)) {
            setBounds(glm::vec3(-1.0f), glm::vec3(1.0f));
        } else {
            createCubeGeometry();
            if (auto result = setupVertexData(); !result) {
                initialized_ = false;
                LOG_ERROR("Failed to setup vertex data: {}", result.error());
                return result;
            }
        }

        LOG_INFO("Bounding box renderer initialized successfully");
        return {};
    }

    void RenderBoundingBox::createCubeGeometry() {
        LOG_TIMER_TRACE("RenderBoundingBox::createCubeGeometry");

        // Create 8 vertices of the bounding box cube
        vertices_[0] = glm::vec3(min_bounds_.x, min_bounds_.y, min_bounds_.z); // 0: min corner
        vertices_[1] = glm::vec3(max_bounds_.x, min_bounds_.y, min_bounds_.z); // 1: +x
        vertices_[2] = glm::vec3(max_bounds_.x, max_bounds_.y, min_bounds_.z); // 2: +x+y
        vertices_[3] = glm::vec3(min_bounds_.x, max_bounds_.y, min_bounds_.z); // 3: +y
        vertices_[4] = glm::vec3(min_bounds_.x, min_bounds_.y, max_bounds_.z); // 4: +z
        vertices_[5] = glm::vec3(max_bounds_.x, min_bounds_.y, max_bounds_.z); // 5: +x+z
        vertices_[6] = glm::vec3(max_bounds_.x, max_bounds_.y, max_bounds_.z); // 6: max corner
        vertices_[7] = glm::vec3(min_bounds_.x, max_bounds_.y, max_bounds_.z); // 7: +y+z

        LOG_TRACE("Created cube geometry with {} vertices", vertices_.size());
    }

    Result<void> RenderBoundingBox::setupVertexData() {
        if (!initialized_ || !vao_) {
            LOG_ERROR("Bounding box not initialized");
            return std::unexpected("Bounding box not initialized");
        }

        LOG_TIMER_TRACE("RenderBoundingBox::setupVertexData");

        // Upload vertex data
        BufferBinder<GL_ARRAY_BUFFER> vbo_bind(vbo_);
        upload_buffer(GL_ARRAY_BUFFER, std::span(vertices_), GL_DYNAMIC_DRAW);

        // Upload index data
        BufferBinder<GL_ELEMENT_ARRAY_BUFFER> ebo_bind(ebo_);
        upload_buffer(GL_ELEMENT_ARRAY_BUFFER, std::span(indices_), GL_STATIC_DRAW);

        LOG_TRACE("Uploaded {} vertices and {} indices", vertices_.size(), indices_.size());
        return {};
    }

    Result<void> RenderBoundingBox::render(const glm::mat4& view, const glm::mat4& projection) {
        if (!initialized_ || !shader_.valid() || !vao_) {
            LOG_ERROR("Bounding box renderer not initialized");
            return std::unexpected("Bounding box renderer not initialized");
        }

        LOG_TIMER_TRACE("RenderBoundingBox::render");

        // Save current state that we'll modify
        GLboolean depth_test_enabled = glIsEnabled(GL_DEPTH_TEST);
        GLboolean depth_mask;
        glGetBooleanv(GL_DEPTH_WRITEMASK, &depth_mask);
        GLboolean blend_enabled = glIsEnabled(GL_BLEND);
        GLint blend_src, blend_dst;
        if (blend_enabled) {
            glGetIntegerv(GL_BLEND_SRC_RGB, &blend_src);
            glGetIntegerv(GL_BLEND_DST_RGB, &blend_dst);
        }

        // Set state for wireframe rendering:
        // - Enable depth test so box respects depth
        // - Disable depth writing so wireframe doesn't occlude things behind it
        // - Enable blending for potential transparency
        glEnable(GL_DEPTH_TEST);
        glDepthMask(GL_FALSE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Use GLLineGuard for line width management
        GLLineGuard line_guard(line_width_);

        ShaderScope s(shader_);

        const glm::mat4 box2world = use_mat4_transform_ ? box2world_mat4_ : world2BBox_.inv().toMat4();
        const glm::mat4 mvp = projection * view * box2world;

        const glm::mat4 inv_view = glm::inverse(view);
        const glm::vec3 camera_pos = glm::vec3(inv_view[3]);
        const glm::vec3 local_center = (min_bounds_ + max_bounds_) * 0.5f;
        const glm::vec3 world_center = glm::vec3(box2world * glm::vec4(local_center, 1.0f));
        const glm::vec3 view_dir = glm::normalize(world_center - camera_pos);

        const glm::mat4 world2box = use_mat4_transform_ ? glm::inverse(box2world_mat4_) : world2BBox_.toMat4();
        const glm::vec3 local_view_dir = glm::mat3(world2box) * view_dir;

        LOG_TRACE("Rendering bounding box with color ({}, {}, {})", color_.r, color_.g, color_.b);

        if (auto result = s->set("u_mvp", mvp); !result)
            return result;
        if (auto result = s->set("u_color", color_); !result)
            return result;
        if (auto result = s->set("u_box_center", local_center); !result)
            return result;
        if (auto result = s->set("u_view_dir", local_view_dir); !result)
            return result;

        // Bind VAO and draw
        VAOBinder vao_bind(vao_);
        glDrawElements(GL_LINES, static_cast<GLsizei>(indices_.size()), GL_UNSIGNED_INT, 0);

        // Restore state
        glDepthMask(depth_mask);
        if (!depth_test_enabled) {
            glDisable(GL_DEPTH_TEST);
        }
        if (!blend_enabled) {
            glDisable(GL_BLEND);
        } else {
            glBlendFunc(blend_src, blend_dst);
        }

        return {};
    }

    const unsigned int RenderBoundingBox::cube_line_indices_[24] = {
        // Bottom face edges
        0, 1, 1, 2, 2, 3, 3, 0,
        // Top face edges
        4, 5, 5, 6, 6, 7, 7, 4,
        // Vertical edges
        0, 4, 1, 5, 2, 6, 3, 7};

} // namespace lfs::rendering
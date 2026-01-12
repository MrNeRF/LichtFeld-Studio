/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "axes_renderer.hpp"
#include "core/logger.hpp"
#include "gl_state_guard.hpp"
#include "shader_paths.hpp"
#include <format>

namespace lfs::rendering {

    // Standard coordinate axes colors (RGB convention)
    const glm::vec3 RenderCoordinateAxes::X_AXIS_COLOR = glm::vec3(1.0f, 0.0f, 0.0f); // Red
    const glm::vec3 RenderCoordinateAxes::Y_AXIS_COLOR = glm::vec3(0.0f, 1.0f, 0.0f); // Green
    const glm::vec3 RenderCoordinateAxes::Z_AXIS_COLOR = glm::vec3(0.0f, 0.0f, 1.0f); // Blue

    RenderCoordinateAxes::RenderCoordinateAxes() : size_(2.0f),
                                                   line_width_(3.0f),
                                                   initialized_(false) {
        // All axes visible by default
        axis_visible_[0] = true; // X
        axis_visible_[1] = true; // Y
        axis_visible_[2] = true; // Z

        // Reserve space for 6 vertices (2 per axis: origin + endpoint)
        vertices_.reserve(6);

        LOG_DEBUG("RenderCoordinateAxes created with size {} and line width {}", size_, line_width_);
    }

    void RenderCoordinateAxes::setSize(float size) {
        LOG_TRACE("Setting axes size to {}", size);
        size_ = size;
        createAxesGeometry();

        if (isInitialized()) {
            if (auto result = setupVertexData(); !result) {
                LOG_ERROR("Failed to setup vertex data for coordinate axes: {}", result.error());
            }
        }
    }

    void RenderCoordinateAxes::setAxisVisible(int axis, bool visible) {
        if (axis >= 0 && axis < 3) {
            LOG_TRACE("Setting axis {} visibility to {}", axis, visible);
            axis_visible_[axis] = visible;
            createAxesGeometry();

            if (isInitialized()) {
                if (auto result = setupVertexData(); !result) {
                    LOG_ERROR("Failed to setup vertex data for coordinate axes: {}", result.error());
                }
            }
        } else {
            LOG_WARN("Invalid axis index: {}", axis);
        }
    }

    bool RenderCoordinateAxes::isAxisVisible(int axis) const {
        if (axis >= 0 && axis < 3) {
            return axis_visible_[axis];
        }
        return false;
    }

    Result<void> RenderCoordinateAxes::init() {
        if (isInitialized())
            return {};

        LOG_TIMER("RenderCoordinateAxes::init");
        LOG_INFO("Initializing coordinate axes renderer");

        // Create shader for coordinate axes rendering (with geometry shader for equirectangular seam culling)
        auto result = load_shader_with_geometry("coordinate_axes", "coordinate_axes.vert",
                                                "coordinate_axes.geom", "coordinate_axes.frag", false);
        if (!result) {
            LOG_ERROR("Failed to load coordinate axes shader: {}", result.error().what());
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

        // Build VAO using VAOBuilder
        VAOBuilder builder(std::move(*vao_result));

        // Setup vertex attributes (data will be filled in setupVertexData)
        builder.attachVBO(vbo_) // Attach without data initially
            .setAttribute({.index = 0,
                           .size = 3,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = sizeof(AxisVertex),
                           .offset = (void*)offsetof(AxisVertex, position),
                           .divisor = 0})
            .setAttribute({.index = 1,
                           .size = 3,
                           .type = GL_FLOAT,
                           .normalized = GL_FALSE,
                           .stride = sizeof(AxisVertex),
                           .offset = (void*)offsetof(AxisVertex, color),
                           .divisor = 0});

        vao_ = builder.build();

        initialized_ = true;

        // Initialize axes geometry
        createAxesGeometry();

        if (auto setup_result = setupVertexData(); !setup_result) {
            initialized_ = false;
            LOG_ERROR("Failed to setup vertex data: {}", setup_result.error());
            return setup_result;
        }

        LOG_INFO("Coordinate axes renderer initialized successfully");
        return {};
    }

    void RenderCoordinateAxes::createAxesGeometry() {
        LOG_TIMER_TRACE("RenderCoordinateAxes::createAxesGeometry");

        vertices_.clear();

        int visible_count = 0;

        // X-axis (Red)
        if (axis_visible_[0]) {
            vertices_.push_back({glm::vec3(0.0f, 0.0f, 0.0f), X_AXIS_COLOR});  // Origin
            vertices_.push_back({glm::vec3(size_, 0.0f, 0.0f), X_AXIS_COLOR}); // X endpoint
            visible_count++;
        }

        // Y-axis (Green)
        if (axis_visible_[1]) {
            vertices_.push_back({glm::vec3(0.0f, 0.0f, 0.0f), Y_AXIS_COLOR});  // Origin
            vertices_.push_back({glm::vec3(0.0f, size_, 0.0f), Y_AXIS_COLOR}); // Y endpoint
            visible_count++;
        }

        // Z-axis (Blue)
        if (axis_visible_[2]) {
            vertices_.push_back({glm::vec3(0.0f, 0.0f, 0.0f), Z_AXIS_COLOR});  // Origin
            vertices_.push_back({glm::vec3(0.0f, 0.0f, size_), Z_AXIS_COLOR}); // Z endpoint
            visible_count++;
        }

        LOG_TRACE("Created axes geometry with {} visible axes, {} vertices total", visible_count, vertices_.size());
    }

    Result<void> RenderCoordinateAxes::setupVertexData() {
        if (!initialized_ || !vao_ || vertices_.empty())
            return {}; // Nothing to setup if no visible axes

        LOG_TIMER_TRACE("RenderCoordinateAxes::setupVertexData");

        // Upload vertex data
        BufferBinder<GL_ARRAY_BUFFER> vbo_bind(vbo_);
        upload_buffer(GL_ARRAY_BUFFER, std::span(vertices_), GL_DYNAMIC_DRAW);

        LOG_TRACE("Uploaded {} vertices for coordinate axes", vertices_.size());
        return {};
    }

    Result<void> RenderCoordinateAxes::render(const glm::mat4& view, const glm::mat4& projection, const bool equirectangular) {
        if (!initialized_ || !shader_.valid() || !vao_ || vertices_.empty())
            return {};

        LOG_TIMER_TRACE("RenderCoordinateAxes::render");

        const GLboolean depth_test_enabled = glIsEnabled(GL_DEPTH_TEST);
        glDisable(GL_DEPTH_TEST);

        GLLineGuard line_guard(line_width_);
        ShaderScope s(shader_);

        const glm::mat4 mvp = projection * view;
        if (auto result = s->set("u_mvp", mvp); !result) {
            if (depth_test_enabled)
                glEnable(GL_DEPTH_TEST);
            return result;
        }
        s->set("u_view", view);
        s->set("u_equirectangular", equirectangular);

        VAOBinder vao_bind(vao_);
        glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(vertices_.size()));

        if (depth_test_enabled)
            glEnable(GL_DEPTH_TEST);

        return {};
    }

} // namespace lfs::rendering
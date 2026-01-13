/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "rendering/rendering.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>
#include <vector>

namespace lfs::rendering {
    class RenderCoordinateAxes : public ICoordinateAxes {
    public:
        RenderCoordinateAxes();
        ~RenderCoordinateAxes() override = default;

        // ICoordinateAxes interface implementation
        void setSize(float size) override;
        [[nodiscard]] float getSize() const { return size_; }

        // Initialize OpenGL resources - now returns Result
        Result<void> init();

        // Check if initialized
        [[nodiscard]] bool isInitialized() const { return initialized_; }

        // Set line width for axes
        void setLineWidth(float width) { line_width_ = width; }

        // Enable/disable individual axes
        void setAxisVisible(int axis, bool visible) override; // 0=X, 1=Y, 2=Z
        [[nodiscard]] bool isAxisVisible(int axis) const override;

        Result<void> render(const glm::mat4& view, const glm::mat4& projection, bool equirectangular = false);

    private:
        void createAxesGeometry();
        Result<void> setupVertexData();

        // OpenGL resources using RAII
        ManagedShader shader_;
        VAO vao_;
        VBO vbo_;

        // Axes properties
        float size_;
        float line_width_;
        bool initialized_;
        bool axis_visible_[3]{}; // X, Y, Z visibility

        // Standard colors for coordinate axes
        static const glm::vec3 X_AXIS_COLOR; // Red
        static const glm::vec3 Y_AXIS_COLOR; // Green
        static const glm::vec3 Z_AXIS_COLOR; // Blue

        // Geometry data - vertices with colors
        struct AxisVertex {
            glm::vec3 position;
            glm::vec3 color;
        };

        std::vector<AxisVertex> vertices_;
    };
} // namespace lfs::rendering

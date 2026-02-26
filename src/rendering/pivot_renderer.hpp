/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "rendering/rendering.hpp"
#include "shader_manager.hpp"
#include <glm/glm.hpp>

namespace lfs::rendering {

    class RenderPivotPoint {
    public:
        RenderPivotPoint() = default;
        ~RenderPivotPoint() = default;

        Result<void> init();
        [[nodiscard]] bool isInitialized() const { return initialized_; }

        void setPosition(const glm::vec3& position) { pivot_position_ = position; }
        void setSize(const float size) { screen_size_ = size; }
        void setColor(const glm::vec3& color) { color_ = color; }
        void setOpacity(const float opacity) { opacity_ = opacity; }

        Result<void> render(const glm::mat4& view, const glm::mat4& projection);

    private:
        static constexpr float DEFAULT_SCREEN_SIZE = 50.0f;
        static constexpr glm::vec3 DEFAULT_COLOR{0.26f, 0.59f, 0.98f};

        ManagedShader shader_;
        VAO vao_;

        glm::vec3 pivot_position_{0.0f};
        glm::vec3 color_{DEFAULT_COLOR};
        float screen_size_{DEFAULT_SCREEN_SIZE};
        float opacity_{1.0f};
        bool initialized_{false};
    };

} // namespace lfs::rendering

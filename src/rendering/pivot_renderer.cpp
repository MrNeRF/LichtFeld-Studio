/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "pivot_renderer.hpp"
#include "core/logger.hpp"
#include "gl_state_guard.hpp"
#include "shader_paths.hpp"

namespace lfs::rendering {

    Result<void> RenderPivotPoint::init() {
        if (initialized_) {
            return {};
        }

        auto result = load_shader("pivot_point", "pivot_point.vert", "pivot_point.frag", false);
        if (!result) {
            LOG_ERROR("Failed to load pivot shader: {}", result.error().what());
            return std::unexpected(result.error().what());
        }
        shader_ = std::move(*result);

        auto vao_result = create_vao();
        if (!vao_result) {
            return std::unexpected(vao_result.error());
        }
        vao_ = std::move(*vao_result);

        initialized_ = true;
        LOG_DEBUG("Pivot renderer initialized");
        return {};
    }

    Result<void> RenderPivotPoint::render(const glm::mat4& view, const glm::mat4& projection) {
        if (!initialized_ || !shader_.valid()) {
            return std::unexpected("Pivot renderer not initialized");
        }

        GLStateGuard state_guard;

        GLint viewport[4];
        glGetIntegerv(GL_VIEWPORT, viewport);
        const glm::vec2 viewport_size(static_cast<float>(viewport[2]), static_cast<float>(viewport[3]));

        glDisable(GL_DEPTH_TEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        ShaderScope s(shader_);

        if (auto r = s->set("u_view", view); !r)
            return r;
        if (auto r = s->set("u_projection", projection); !r)
            return r;
        if (auto r = s->set("u_pivot_pos", pivot_position_); !r)
            return r;
        if (auto r = s->set("u_screen_size", screen_size_); !r)
            return r;
        if (auto r = s->set("u_viewport_size", viewport_size); !r)
            return r;
        if (auto r = s->set("u_color", color_); !r)
            return r;
        if (auto r = s->set("u_opacity", opacity_); !r)
            return r;

        VAOBinder vao_bind(vao_);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        return {};
    }

} // namespace lfs::rendering

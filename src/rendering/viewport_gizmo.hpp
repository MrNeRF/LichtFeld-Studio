/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gl_resources.hpp"
#include "shader_manager.hpp"
#include <array>
#include <cstddef>
#include <glm/glm.hpp>
#include <memory>
#include <optional>

namespace lfs::rendering {

    class TextRenderer; // Forward declaration

    enum class GizmoAxis { X = 0,
                           Y = 1,
                           Z = 2 };

    struct GizmoHitResult {
        GizmoAxis axis;
        bool negative;
    };

    class ViewportGizmo {
    public:
        ViewportGizmo();
        ~ViewportGizmo();

        Result<void> initialize();
        void shutdown();

        Result<void> render(const glm::mat3& camera_rotation,
                            const glm::vec2& viewport_pos,
                            const glm::vec2& viewport_size);

        [[nodiscard]] std::optional<GizmoHitResult> hitTest(const glm::vec2& click_pos,
                                                            const glm::vec2& viewport_pos,
                                                            const glm::vec2& viewport_size) const;

        [[nodiscard]] static glm::mat3 getAxisViewRotation(GizmoAxis axis, bool negative = false);

        void setSize(int size) { size_ = size; }
        void setMargins(int x, int y) {
            margin_x_ = x;
            margin_y_ = y;
        }
        [[nodiscard]] int getSize() const { return size_; }
        [[nodiscard]] int getMarginX() const { return margin_x_; }
        [[nodiscard]] int getMarginY() const { return margin_y_; }

        void setHoveredAxis(std::optional<GizmoAxis> axis, bool negative = false) {
            hovered_axis_ = axis;
            hovered_negative_ = negative;
        }
        [[nodiscard]] std::optional<GizmoAxis> getHoveredAxis() const { return hovered_axis_; }

    private:
        Result<void> generateGeometry();
        Result<void> createShaders();

        VAO vao_;
        VBO vbo_;
        ManagedShader shader_;
        std::unique_ptr<TextRenderer> text_renderer_;

        int cylinder_vertex_count_ = 0;
        int sphere_vertex_start_ = 0;
        int sphere_vertex_count_ = 0;
        int ring_vertex_start_ = 0;
        int ring_vertex_count_ = 0;

        int size_ = 95;
        int margin_x_ = 10;
        int margin_y_ = 10;
        bool initialized_ = false;
        std::optional<GizmoAxis> hovered_axis_;
        bool hovered_negative_ = false;

        struct HitInfo {
            glm::vec2 screen_pos{0.0f};
            float radius = 0.0f;
            bool visible = false;
        };
        struct HitCache {
            glm::vec2 viewport_pos{0.0f};
            glm::vec2 viewport_size{0.0f};
            std::array<HitInfo, 3> sphere_hits{};
            std::array<HitInfo, 3> ring_hits{};
        };
        mutable HitInfo sphere_hits_[3];
        mutable HitInfo ring_hits_[3];
        mutable std::array<HitCache, 2> hit_caches_{};
        mutable size_t hit_cache_count_ = 0;

        static constexpr glm::vec3 AXIS_COLORS[3] = {
            {0.89f, 0.15f, 0.21f},
            {0.54f, 0.86f, 0.20f},
            {0.17f, 0.48f, 0.87f}};
    };

} // namespace lfs::rendering

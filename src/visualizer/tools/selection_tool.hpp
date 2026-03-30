/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "tool_base.hpp"
#include "tools/selection_operation.hpp"
#include <algorithm>
#include <glm/glm.hpp>

namespace lfs::vis::input {
    class InputBindings;
}

namespace lfs::vis::tools {

    class SelectionTool : public ToolBase {
    public:
        SelectionTool();
        ~SelectionTool() override = default;

        [[nodiscard]] std::string_view getName() const override { return "Selection Tool"; }
        [[nodiscard]] std::string_view getDescription() const override { return "Paint to select Gaussians"; }

        bool initialize(const ToolContext& ctx) override;
        void shutdown() override;
        void update(const ToolContext& ctx) override;
        void renderUI(const lfs::vis::gui::UIContext& ui_ctx, bool* p_open) override;

        [[nodiscard]] float getBrushRadius() const { return brush_radius_; }
        void setBrushRadius(float radius) { brush_radius_ = std::clamp(radius, 1.0f, 500.0f); }

        void onSelectionModeChanged();

        // Depth filter
        [[nodiscard]] bool isDepthFilterEnabled() const { return depth_filter_enabled_; }
        void setDepthFilterEnabled(bool enabled);
        void toggleDepthFilter() { setDepthFilterEnabled(!depth_filter_enabled_); }
        void adjustDepthFar(float scale);
        void syncDepthFilterToCamera(const Viewport& viewport);

        // Crop filter (use scene crop box/ellipsoid as selection filter)
        [[nodiscard]] bool isCropFilterEnabled() const { return crop_filter_enabled_; }
        void setCropFilterEnabled(bool enabled);
        void toggleCropFilter() { setCropFilterEnabled(!crop_filter_enabled_); }

        // Input bindings
        void setInputBindings(const input::InputBindings* bindings) { input_bindings_ = bindings; }

    protected:
        void onEnabledChanged(bool enabled) override;

    private:
        struct RenderModeSnapshot {
            bool valid = false;
            bool point_cloud_mode = false;
            bool show_rings = false;
            bool show_center_markers = false;
        };

        glm::vec2 last_mouse_pos_{0.0f};
        float brush_radius_ = 20.0f;
        const ToolContext* tool_context_ = nullptr;

        // Determine operation from modifier keys
        SelectionOp getOpFromModifiers(int mods) const;

        // Depth filter
        bool depth_filter_enabled_ = false;
        float depth_near_ = 0.0f;
        float depth_far_ = DEFAULT_DEPTH_FAR;
        float frustum_half_width_ = DEFAULT_FRUSTUM_HALF_WIDTH;
        RenderModeSnapshot depth_filter_render_mode_snapshot_;

        // Crop filter
        bool crop_filter_enabled_ = false;

        static constexpr float DEPTH_MIN = 0.01f;
        static constexpr float DEPTH_MAX = 1000.0f;
        static constexpr float DEFAULT_DEPTH_FAR = 5.3f;
        static constexpr float DEFAULT_FRUSTUM_HALF_WIDTH = 1.35f;

        void drawDepthFrustum(const ToolContext& ctx) const;
        void applySelectionFilterSettings(const ToolContext& ctx) const;
        void clearSelectionRenderState(const ToolContext& ctx) const;
        void syncDepthFilterRenderMode(const ToolContext& ctx);

        // Input bindings
        const input::InputBindings* input_bindings_ = nullptr;
    };

} // namespace lfs::vis::tools

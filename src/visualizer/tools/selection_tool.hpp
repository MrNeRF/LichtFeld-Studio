/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "tool_base.hpp"
#include <algorithm>
#include <glm/glm.hpp>

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
        [[nodiscard]] float getDepthNear() const { return depth_near_; }
        [[nodiscard]] float getDepthFar() const { return depth_far_; }
        [[nodiscard]] float getDepthFrustumHalfWidth() const { return frustum_half_width_; }
        void setDepthFilterEnabled(bool enabled);
        void setDepthFilterRange(bool enabled, float depth_near, float depth_far, float frustum_half_width);
        void toggleDepthFilter() { setDepthFilterEnabled(!depth_filter_enabled_); }
        void adjustDepthFar(float scale);
        void syncDepthFilterToCamera(const Viewport& viewport);

        // Crop filter (use scene crop box/ellipsoid as selection filter)
        [[nodiscard]] bool isCropFilterEnabled() const { return crop_filter_enabled_; }
        void setCropFilterEnabled(bool enabled);
        void toggleCropFilter() { setCropFilterEnabled(!crop_filter_enabled_); }

    protected:
        void onEnabledChanged(bool enabled) override;

    private:
        glm::vec2 last_mouse_pos_{0.0f};
        float brush_radius_ = 20.0f;
        const ToolContext* tool_context_ = nullptr;

        // Depth filter
        bool depth_filter_enabled_ = false;
        float depth_near_ = 0.0f;
        float depth_far_ = DEFAULT_DEPTH_FAR;
        float frustum_half_width_ = DEFAULT_FRUSTUM_HALF_WIDTH;

        // Crop filter
        bool crop_filter_enabled_ = false;

        static constexpr float DEPTH_MIN = 0.01f;
        static constexpr float DEPTH_MAX = 1000.0f;
        static constexpr float DEFAULT_DEPTH_FAR = 5.3f;
        static constexpr float DEFAULT_FRUSTUM_HALF_WIDTH = 1.35f;

        void applySelectionFilterSettings(const ToolContext& ctx) const;
        void clearSelectionRenderState(const ToolContext& ctx) const;
    };

} // namespace lfs::vis::tools

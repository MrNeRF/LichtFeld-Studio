/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"
#include "internal/viewport.hpp"
#include "rendering_types.hpp"
#include <mutex>
#include <optional>

namespace lfs::vis {
    class GTTextureCache;
    class SceneManager;
    struct FrameResources;
} // namespace lfs::vis

namespace lfs::core {
    class Camera;
} // namespace lfs::core

namespace lfs::vis {

    namespace detail {
        [[nodiscard]] LFS_VIS_API std::optional<GTRenderCamera> buildGTRenderCamera(
            const lfs::core::Camera& cam,
            glm::ivec2 render_size,
            const glm::mat4& scene_transform);
    } // namespace detail

    class LFS_VIS_API SplitViewService {
    public:
        struct ModeChangeResult {
            SplitViewMode previous_mode = SplitViewMode::Disabled;
            SplitViewMode current_mode = SplitViewMode::Disabled;
            bool mode_changed = false;
            bool clear_viewport_output = false;
            std::optional<bool> restore_equirectangular;
        };

        [[nodiscard]] std::optional<glm::ivec2> gtContentDimensions() const;
        [[nodiscard]] const std::optional<GTComparisonContext>& gtContext() const { return gt_context_; }
        [[nodiscard]] bool isActive(const RenderSettings& settings) const;
        [[nodiscard]] bool isGTComparisonActive(const RenderSettings& settings) const;
        [[nodiscard]] bool isIndependentDualActive(const RenderSettings& settings) const;
        [[nodiscard]] std::optional<std::array<SplitViewPanelLayout, 2>>
        panelLayouts(const RenderSettings& settings, int total_width) const;
        [[nodiscard]] std::optional<int> dividerPixel(const RenderSettings& settings, int total_width) const;

        [[nodiscard]] ModeChangeResult toggleMode(RenderSettings& settings,
                                                  SplitViewMode target_mode,
                                                  const Viewport* primary_viewport = nullptr);
        [[nodiscard]] ModeChangeResult handleSceneLoaded(RenderSettings& settings);
        [[nodiscard]] ModeChangeResult handleSceneCleared(RenderSettings& settings);
        [[nodiscard]] ModeChangeResult handlePLYRemoved(RenderSettings& settings, SceneManager* scene_manager);
        void advanceSplitOffset(RenderSettings& settings);
        [[nodiscard]] SplitViewInfo getInfo() const;
        // Focused panel and the secondary viewport are main-thread-owned state used by
        // input, frame planning, and rendering on the UI thread. Only current_info_ is
        // mutex-protected because it is read from UI/status-bar code outside that path.
        void setFocusedPanel(SplitViewPanelId panel) { focused_panel_ = panel; }
        [[nodiscard]] SplitViewPanelId focusedPanel() const { return focused_panel_; }
        [[nodiscard]] Viewport& secondaryViewport() { return secondary_viewport_; }
        [[nodiscard]] const Viewport& secondaryViewport() const { return secondary_viewport_; }
        void updateInfo(const FrameResources& resources);
        void prepareGTComparisonContext(SceneManager* scene_manager,
                                        const RenderSettings& settings,
                                        int current_camera_id,
                                        bool has_renderable_content,
                                        bool has_viewport_output,
                                        GTTextureCache& texture_cache,
                                        bool& request_viewport_prerender);

    private:
        enum class GTExitBehavior {
            PreserveCurrent,
            RestorePrevious
        };

        [[nodiscard]] bool hasValidGTContext() const;
        [[nodiscard]] ModeChangeResult transitionToMode(RenderSettings& settings,
                                                        SplitViewMode target_mode,
                                                        const Viewport* primary_viewport,
                                                        GTExitBehavior gt_exit_behavior);
        void clear();
        void clearGTContext();

        mutable std::mutex info_mutex_;
        SplitViewInfo current_info_;
        std::optional<GTComparisonContext> gt_context_;
        bool pre_gt_equirectangular_ = false;
        SplitViewPanelId focused_panel_ = SplitViewPanelId::Left;
        Viewport secondary_viewport_;
    };

} // namespace lfs::vis

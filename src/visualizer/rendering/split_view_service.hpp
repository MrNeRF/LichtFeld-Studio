/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering_types.hpp"
#include "core/export.hpp"
#include <mutex>
#include <optional>

namespace lfs::vis {
    class GTTextureCache;
    class SceneManager;
    struct FrameResources;

    class LFS_VIS_API SplitViewService {
    public:
        struct GTToggleResult {
            bool enabled = false;
            std::optional<bool> restore_equirectangular;
        };

        [[nodiscard]] std::optional<glm::ivec2> gtContentDimensions() const;
        [[nodiscard]] const std::optional<GTComparisonContext>& gtContext() const { return gt_context_; }

        [[nodiscard]] bool togglePLYComparison(RenderSettings& settings);
        [[nodiscard]] GTToggleResult toggleGTComparison(RenderSettings& settings);
        void handleSceneLoaded(RenderSettings& settings);
        void handleSceneCleared(RenderSettings& settings);
        [[nodiscard]] bool handlePLYRemoved(RenderSettings& settings, SceneManager* scene_manager);
        void advanceSplitOffset(RenderSettings& settings);
        [[nodiscard]] SplitViewInfo getInfo() const;
        void updateInfo(const FrameResources& resources);
        void prepareGTComparisonContext(SceneManager* scene_manager,
                                        const RenderSettings& settings,
                                        int current_camera_id,
                                        bool has_renderable_content,
                                        bool has_viewport_output,
                                        GTTextureCache& texture_cache,
                                        bool& request_viewport_prerender);

    private:
        [[nodiscard]] bool hasValidGTContext() const;
        void clear();
        void clearGTContext();

        mutable std::mutex info_mutex_;
        SplitViewInfo current_info_;
        std::optional<GTComparisonContext> gt_context_;
        bool pre_gt_equirectangular_ = false;
    };

} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "dirty_flags.hpp"
#include "framerate_controller.hpp"
#include "render_pass_graph.hpp"
#include "split_view_service.hpp"
#include "viewport_artifact_service.hpp"
#include "viewport_interaction_context.hpp"
#include "viewport_overlay_service.hpp"
#include <chrono>
#include <optional>

namespace lfs::core {
    class SplatData;
}

namespace lfs::rendering {
    class RenderingEngine;
}

namespace lfs::vis {
    class SceneManager;

    class RenderFrameCoordinator {
    public:
        struct Context {
            const Viewport& viewport;
            const ViewportRegion* viewport_region = nullptr;
            SceneManager* scene_manager = nullptr;
            const lfs::core::SplatData* model = nullptr;
            bool render_lock_held = false;
            const RenderSettings& settings;
            std::array<int, 2> grid_planes{{1, 1}};
            DirtyMask frame_dirty = 0;
            float selection_flash_intensity = 0.0f;
            int current_camera_id = -1;
            int hovered_camera_id = -1;
        };

        struct Dependencies {
            lfs::rendering::RenderingEngine& engine;
            RenderPassGraph& pass_graph;
            FramerateController& framerate_controller;
            ViewportArtifactService& viewport_artifacts;
            ViewportInteractionContext& viewport_interaction_context;
            ViewportOverlayService& viewport_overlay;
            SplitViewService& split_view_service;
            uint64_t& render_count;
        };

        struct Result {
            DirtyMask additional_dirty = 0;
            std::optional<std::chrono::steady_clock::time_point> pivot_animation_end;
        };

        explicit RenderFrameCoordinator(Dependencies dependencies)
            : dependencies_(dependencies) {}

        [[nodiscard]] Result execute(const Context& context);

    private:
        Dependencies dependencies_;
    };

} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "dirty_flags.hpp"
#include "rendering_manager.hpp"
#include <rendering/rendering.hpp>

namespace lfs::vis {

    class SceneManager;

    struct FrameContext {
        const RenderingManager::RenderContext& ctx;
        SceneManager* scene_manager;
        const lfs::core::SplatData* model;
        RenderSettings settings;
        glm::ivec2 render_size;
        glm::ivec2 viewport_pos;
        DirtyMask frame_dirty = 0;
    };

    struct FrameResources {
        lfs::rendering::RenderResult cached_result;
        glm::ivec2 cached_result_size{0};
        bool render_texture_valid = false;
        unsigned int cached_render_texture = 0;
        bool splats_presented = false;
        bool split_view_executed = false;
        std::optional<GTComparisonContext> gt_context;
        int gt_context_camera_id = -1;

        GTTextureCache* gt_texture_cache = nullptr;
        unsigned long long* d_hovered_depth_id = nullptr;
        std::unique_ptr<lfs::core::PointCloud>* cached_filtered_pc = nullptr;
    };

    class RenderPass {
    public:
        virtual ~RenderPass() = default;
        [[nodiscard]] virtual const char* name() const = 0;
        [[nodiscard]] virtual DirtyMask sensitivity() const = 0;

        [[nodiscard]] virtual bool shouldExecute(DirtyMask frame_dirty, const FrameContext& ctx) const {
            return (frame_dirty & sensitivity()) != 0;
        }

        virtual void execute(lfs::rendering::RenderingEngine& engine,
                             const FrameContext& ctx,
                             FrameResources& res) = 0;
    };

} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor_fwd.hpp"

#include <glm/glm.hpp>
#include <memory>

namespace lfs::vis {
    struct SplitViewCpuPanelDesc {
        std::shared_ptr<const lfs::core::Tensor> image;
        float start_position = 0.0f;
        float end_position = 1.0f;
        bool normalize_x_to_panel = false;
        bool flip_y = false;
        glm::vec2 uv_scale{1.0f};
        glm::vec2 uv_clamp_max{1.0f};
        glm::vec2 texcoord_scale{1.0f};
        glm::vec2 texcoord_offset{0.0f};
        bool spatial_filter = false;
    };

    struct SplitViewCpuDesc {
        bool loss_visualization = false;
        SplitViewCpuPanelDesc left;
        SplitViewCpuPanelDesc right;
        float split_position = 0.5f;
        glm::ivec4 content_rect{0, 0, 0, 0};
        glm::ivec2 coordinate_extent{0, 0};
        glm::vec3 background{0.0f};
    };
} // namespace lfs::vis

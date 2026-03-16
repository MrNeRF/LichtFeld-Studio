/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "render_pass.hpp"

namespace lfs::vis {

    // Centralized compatibility adapter for the frozen RenderRequest API.
    // New visualizer-side work should build FrameView/GpuFrame contracts first
    // and only translate to RenderRequest here while legacy entry points remain.
    [[nodiscard]] lfs::rendering::RenderRequest buildLegacyGaussianRenderRequest(
        const FrameContext& ctx, glm::ivec2 render_size);

    [[nodiscard]] lfs::rendering::RenderRequest buildLegacyPointCloudRenderRequest(
        const FrameContext& ctx, const std::vector<glm::mat4>& model_transforms);

} // namespace lfs::vis

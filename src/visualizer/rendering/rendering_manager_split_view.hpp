/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "internal/viewport.hpp"
#include "rendering_types.hpp"
#include "split_view_cpu_desc.hpp"
#include "viewport_interaction_context.hpp"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace lfs::core {
    class Camera;
    class Tensor;
} // namespace lfs::core

namespace lfs::rendering {
    struct FrameMetadata;
    struct ViewportRenderRequest;
} // namespace lfs::rendering

namespace lfs::vis {
    struct FrameContext;
    struct SplitCompositeContentRect {
        int x = 0;
        int y = 0;
        int width = 0;
        int height = 0;
    };

    [[nodiscard]] glm::ivec2 gtComparisonPreviewSize(
        const lfs::core::Camera& camera, glm::ivec2 viewport_size);
    [[nodiscard]] std::shared_ptr<lfs::core::Tensor> makeGTComparePlaceholderTensor(
        glm::ivec2 size, glm::vec3 tint);
    [[nodiscard]] std::shared_ptr<lfs::core::Tensor> makeNormalDisplayFromDepthTensor(
        const lfs::core::Tensor& depth,
        const lfs::rendering::CameraIntrinsics& intrinsics);
    [[nodiscard]] std::vector<ViewportInteractionPanel> buildSplitViewInteractionPanels(
        const Viewport& viewport, const RenderSettings& settings,
        glm::vec2 screen_viewport_pos, glm::vec2 screen_viewport_size);
    [[nodiscard]] lfs::rendering::ViewportRenderRequest buildPlyComparisonRenderRequest(
        const FrameContext& context, glm::ivec2 panel_size,
        const Viewport& source_viewport, SplitViewPanelId panel_id,
        glm::ivec2 full_size);
    [[nodiscard]] SplitCompositeContentRect resolveSplitCompositeContentRect(
        glm::ivec2 output_size, bool letterbox, glm::ivec2 content_size);
    [[nodiscard]] lfs::rendering::FrameMetadata makeSplitMetadata(
        const lfs::rendering::FrameMetadata& left,
        const lfs::rendering::FrameMetadata& right,
        float split_position);
    [[nodiscard]] SplitViewInfo makeGTSplitViewInfo(
        GTComparisonMode mode, const std::string& image_name);
    [[nodiscard]] LFS_VIS_API std::shared_ptr<lfs::core::Tensor> composeSplitViewCpuImage(
        const SplitViewCpuDesc& params, glm::ivec2 output_size);
} // namespace lfs::vis

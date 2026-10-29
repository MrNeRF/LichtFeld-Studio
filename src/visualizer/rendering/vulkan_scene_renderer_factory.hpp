/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "scene_renderer_factory.hpp"
namespace lfs::vis {
    std::unique_ptr<SceneRenderer> createVulkanSceneRenderer(VulkanContext&);
    std::unique_ptr<PointSceneRenderer> createVulkanPointSceneRenderer(VulkanContext&);
    class VksplatViewportRenderer;
    // The VkSplat renderer behind a scene renderer createVulkanSceneRenderer made; null for any other.
    [[nodiscard]] LFS_VIS_API VksplatViewportRenderer* vksplatRendererOf(SceneRenderer& renderer) noexcept;
} // namespace lfs::vis

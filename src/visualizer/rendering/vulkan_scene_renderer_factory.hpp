/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "scene_renderer_factory.hpp"
namespace lfs::vis {
    std::unique_ptr<SceneRenderer> createVulkanSceneRenderer(VulkanContext&);
    std::unique_ptr<PointSceneRenderer> createVulkanPointSceneRenderer(VulkanContext&);
} // namespace lfs::vis

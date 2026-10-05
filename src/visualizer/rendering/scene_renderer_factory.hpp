/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "scene_renderer.hpp"
namespace lfs::vis {
    class GraphicsContext;
    // Bind compositor transport once; scene rendering calls are API independent.
    // The compositor context must outlive both renderer instances.
    LFS_VIS_API std::unique_ptr<SceneRenderer> createSceneRenderer(GraphicsContext&);
    LFS_VIS_API std::unique_ptr<PointSceneRenderer> createPointSceneRenderer(GraphicsContext&);
    LFS_VIS_API void preloadSceneRenderer();
} // namespace lfs::vis

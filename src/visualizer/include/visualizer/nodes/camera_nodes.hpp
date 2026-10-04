/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/nodes/evaluator.hpp"
namespace lfs::core {
    class Scene;
}
namespace lfs::vis {
    LFS_VIS_API void registerCameraNodes(lfs::nodes::NodeTypeRegistry& registry);
    LFS_VIS_API std::vector<lfs::nodes::EvaluationCamera> captureNodeCameras(const core::Scene& scene);
} // namespace lfs::vis

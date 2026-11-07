/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include <atomic>
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <vector>

namespace lfs::vis {

    // CPU input is prepared by the protocol reader. AsyncTaskManager owns the
    // upload worker and joins it before the renderer/context is destroyed.
    struct ExternalPreviewUpdate {
        std::unique_ptr<core::SplatData> model;
        core::Tensor camera_rows;
        std::vector<core::Scene::ExternalPreviewCamera> cameras;
        glm::mat4 transform{1.0f};
        glm::vec3 centroid{0.0f};
        std::string error;
        std::atomic_bool ready{false};
        std::atomic_bool invalidated{false};
        bool submitted = false;
        bool applied = false;
        bool replace_cameras = false;
    };

} // namespace lfs::vis

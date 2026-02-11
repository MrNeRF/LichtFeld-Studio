/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <functional>
#include <string>

#include <glm/glm.hpp>

namespace lfs::core {

    struct Mesh2SplatOptions {
        static constexpr int kMinResolution = 16;

        int resolution_target = 1024;
        float sigma = 0.65f;
        glm::vec3 light_dir{0.0f, 0.0f, 1.0f};
        float light_intensity = 0.7f;
        float ambient = 0.4f;
    };

    using Mesh2SplatProgressCallback = std::function<bool(float progress, const std::string& stage)>;

} // namespace lfs::core

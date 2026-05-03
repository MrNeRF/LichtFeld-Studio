/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "rendering/coordinate_conventions.hpp"
#include "visualizer/ipc/view_context.hpp"

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

namespace lfs::app {

    inline float ortho_view_extent_world(const vis::ViewInfo& info) {
        if (!info.orthographic || info.ortho_scale <= 0.0f)
            return 0.0f;
        return static_cast<float>(info.height) / info.ortho_scale;
    }

    inline nlohmann::json view_info_json(const vis::ViewInfo& info) {
        using json = nlohmann::json;

        const json rotation = json::array({
            json::array({info.rotation[0], info.rotation[1], info.rotation[2]}),
            json::array({info.rotation[3], info.rotation[4], info.rotation[5]}),
            json::array({info.rotation[6], info.rotation[7], info.rotation[8]}),
        });
        const glm::mat3 rotation_matrix(
            glm::vec3(info.rotation[0], info.rotation[3], info.rotation[6]),
            glm::vec3(info.rotation[1], info.rotation[4], info.rotation[7]),
            glm::vec3(info.rotation[2], info.rotation[5], info.rotation[8]));
        const glm::vec3 forward = rendering::cameraForward(rotation_matrix);

        return json{
            {"success", true},
            {"camera", {
                           {"eye", json::array({info.translation[0], info.translation[1], info.translation[2]})},
                           {"target", json::array({info.pivot[0], info.pivot[1], info.pivot[2]})},
                           {"pivot", json::array({info.pivot[0], info.pivot[1], info.pivot[2]})},
                           {"up", json::array({info.rotation[1], info.rotation[4], info.rotation[7]})},
                           {"forward", json::array({forward.x, forward.y, forward.z})},
                           {"rotation_matrix", rotation},
                           {"width", info.width},
                           {"height", info.height},
                           {"fov_degrees", info.fov},
                           {"orthographic", info.orthographic},
                           {"ortho_scale", info.ortho_scale},
                           {"ortho_view_extent_world", ortho_view_extent_world(info)},
                       }},
        };
    }

} // namespace lfs::app

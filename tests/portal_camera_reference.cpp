/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// Cross-repository executable reference for the portal's camera-path player.
#include "sequencer/interpolation.hpp"
#include <iostream>
#include <nlohmann/json.hpp>

int main() {
    using namespace lfs::sequencer;
    nlohmann::json input;
    std::cin >> input;
    std::vector<Keyframe> keys;
    for (const auto& value : input["path"]["keyframes"]) {
        const auto& p = value["position"];
        const auto& q = value["rotation"];
        keys.push_back(Keyframe{.time = value["time"], .position = {p[0], p[1], p[2]}, .rotation = glm::normalize(glm::quat(q[0], q[1], q[2], q[3])), .focal_length_mm = value["focal_length_mm"], .easing = static_cast<EasingType>(value["easing"].get<int>())});
    }
    if (input["path"]["loopMode"] == "loop" && keys.size() > 1) {
        Keyframe close = keys.front();
        close.time = input["path"]["duration"];
        keys.push_back(close);
    }
    auto output = nlohmann::json::array();
    for (const auto& time : input["times"]) {
        const auto state = interpolateSpline(keys, time);
        output.push_back({{"position", {state.position.x, state.position.y, state.position.z}},
                          {"rotation", {state.rotation.w, state.rotation.x, state.rotation.y, state.rotation.z}},
                          {"focalLength", state.focal_length_mm}});
    }
    std::cout << output.dump() << '\n';
}

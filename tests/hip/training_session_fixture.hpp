/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "../../src/training/hip/training_api.h"
#include "training_fixture.hpp"
namespace hip_fixture {
    struct Fixture {
        std::array<std::vector<float>, 5> values{std::vector<float>(n * 3), std::vector<float>(n * 4),
                                                 std::vector<float>(n * 3), std::vector<float>(n * 3), std::vector<float>(n)};
        std::vector<float> target;
        Fixture() {
            auto& means = values[0];
            auto& quats = values[1];
            auto& scales = values[2];
            auto& sh0 = values[3];
            auto& opacities = values[4];
            std::vector<float> colors(n * 3);
            for (unsigned g = 0; g < n; ++g) {
                means[g * 3] = (int(g % 3) - 1) * 0.65f;
                means[g * 3 + 1] = (int((g / 3) % 2) * 2 - 1) * 0.45f;
                means[g * 3 + 2] = 3 + 0.005f * g;
                quats[g * 4] = 1;
                quats[g * 4 + 1] = 0.08f * (g % 6);
                quats[g * 4 + 2] = 0.03f * (g % 6);
                for (unsigned c = 0; c < 3; ++c) {
                    scales[g * 3 + c] = std::log(0.22f + 0.06f * c + 0.01f * (g % 6));
                    colors[g * 3 + c] = 0.2f + float((g * 3 + c * 5) % 7) * 0.1f;
                    sh0[g * 3 + c] = (colors[g * 3 + c] - 0.5f) / sh_c0;
                }
                opacities[g] = -3.5f + 0.1f * (g % 6);
            }
            target = cpu_render(means, quats, scales, colors, opacities);
        }
        void perturb() {
            for (unsigned g = 0; g < n; ++g) {
                values[0][g * 3] += 0.07f;
                values[1][g * 4 + 2] += 0.04f;
                values[2][g * 3 + 1] += 0.08f;
                for (unsigned c = 0; c < 2; ++c) {
                    const float color = sh_c0 * values[3][g * 3 + c] + 0.5f;
                    values[3][g * 3 + c] = (color * (c == 0 ? 0.6f : 0.7f) - 0.5f) / sh_c0;
                }
                values[4][g] -= 0.25f;
            }
        }
        LfsHipTrainModel model() const { return {values[0].data(), values[1].data(), values[2].data(), values[3].data(), values[4].data()}; }
        LfsHipTrainFrame frame() const {
            LfsHipTrainFrame frame{};
            frame.struct_bytes = sizeof(frame);
            frame.width = width;
            frame.height = height;
            for (unsigned i = 0; i < 4; ++i)
                frame.world_to_camera[i * 5] = 1;
            frame.fx = frame.fy = 32;
            frame.cx = width / 2.0f;
            frame.cy = height / 2.0f;
            frame.near_plane = 0.01f;
            frame.far_plane = 1000;
            frame.target_chw = target.data();
            return frame;
        }
    };
} // namespace hip_fixture

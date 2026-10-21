/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
namespace hip_fixture {
    inline constexpr unsigned n = 263, width = 35, height = 29, pixels = width * height, channels = pixels * 3;
    inline constexpr float sh_c0 = 0.28209479177387814f;
    inline std::vector<float> cpu_render(const std::vector<float>& means, const std::vector<float>& quats,
                                         const std::vector<float>& scales, const std::vector<float>& colors,
                                         const std::vector<float>& opacities) {
        std::vector<float> image(channels);
        for (unsigned y = 0; y < height; ++y)
            for (unsigned x = 0; x < width; ++x) {
                const double ray[3] = {(x + 0.5 - width / 2.0) / 32, (y + 0.5 - height / 2.0) / 32, 1};
                double transmission = 1;
                for (unsigned g = 0; g < n; ++g) {
                    double q[4], norm = 0;
                    for (unsigned k = 0; k < 4; ++k)
                        norm += double(quats[g * 4 + k]) * quats[g * 4 + k];
                    for (unsigned k = 0; k < 4; ++k)
                        q[k] = quats[g * 4 + k] / std::sqrt(norm);
                    const auto [w, a, b, c] = std::array<double, 4>{q[0], q[1], q[2], q[3]};
                    const double rotation[3][3] = {
                        {1 - 2 * (b * b + c * c), 2 * (a * b - w * c), 2 * (a * c + w * b)},
                        {2 * (a * b + w * c), 1 - 2 * (a * a + c * c), 2 * (b * c - w * a)},
                        {2 * (a * c - w * b), 2 * (b * c + w * a), 1 - 2 * (a * a + b * b)}};
                    double origin[3]{}, direction[3]{}, direction_norm = 0;
                    for (unsigned row = 0; row < 3; ++row) {
                        for (unsigned col = 0; col < 3; ++col) {
                            origin[row] -= rotation[col][row] * means[g * 3 + col];
                            direction[row] += rotation[col][row] * ray[col];
                        }
                        const double scale = std::exp(scales[g * 3 + row]);
                        origin[row] /= scale;
                        direction[row] /= scale;
                        direction_norm += direction[row] * direction[row];
                    }
                    for (auto& value : direction)
                        value /= std::sqrt(direction_norm);
                    double distance = 0;
                    for (unsigned k = 0; k < 3; ++k) {
                        const double cross = direction[(k + 1) % 3] * origin[(k + 2) % 3] - direction[(k + 2) % 3] * origin[(k + 1) % 3];
                        distance += cross * cross;
                    }
                    const double alpha = std::min(0.999, std::exp(-distance / 2) / (1 + std::exp(-opacities[g])));
                    if (alpha < 1.0 / 255)
                        continue;
                    const double next = transmission * (1 - alpha);
                    if (next <= 1e-4)
                        break;
                    for (unsigned k = 0; k < 3; ++k)
                        image[k * pixels + y * width + x] += float(colors[g * 3 + k] * alpha * transmission);
                    transmission = next;
                }
            }
        return image;
    }
} // namespace hip_fixture

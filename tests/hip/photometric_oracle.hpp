/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <array>
#include <cmath>
#include <span>
#include <vector>

namespace photo_oracle {
    struct Result {
        double loss = 0;
        std::vector<double> gradient;
    };
    // Independent double-precision 2D convolution and chain rule, without the
    // device kernel's separable/shared-memory traversal or half intermediates.
    inline Result evaluate(unsigned width, unsigned height, std::span<const float> image,
                           std::span<const float> target, double weight, bool valid, bool gradient = true) {
        const size_t pixels = size_t(width) * height;
        Result result;
        if (gradient)
            result.gradient.resize(pixels * 3);
        constexpr std::array<float, 11> gaussian{0.001028380123898387f, 0.0075987582094967365f, 0.036000773310661316f,
                                                 0.10936068743467331f, 0.21300552785396576f, 0.26601171493530273f,
                                                 0.21300552785396576f, 0.10936068743467331f, 0.036000773310661316f,
                                                 0.0075987582094967365f, 0.001028380123898387f};
        const int x0 = valid && width > 10 ? 5 : 0, y0 = valid && height > 10 ? 5 : 0;
        const int x1 = int(width) - x0, y1 = int(height) - y0;
        const double normalization = 1.0 / ((x1 - x0) * (y1 - y0) * 3);
        for (unsigned c = 0; c < 3; ++c)
            for (int y = y0; y < y1; ++y)
                for (int x = x0; x < x1; ++x) {
                    const size_t center = c * pixels + y * width + x;
                    const double delta = double(image[center]) - target[center];
                    result.loss += (1 - weight) * std::abs(delta) * normalization;
                    if (gradient)
                        result.gradient[center] += (1 - weight) * ((delta > 0) - (delta < 0)) * normalization;
                    if (weight == 0)
                        continue;
                    double mx = 0, my = 0, xx = 0, yy = 0, xy = 0;
                    for (int dy = -5; dy <= 5; ++dy)
                        for (int dx = -5; dx <= 5; ++dx) {
                            const int sx = x + dx, sy = y + dy;
                            if (sx < 0 || sy < 0 || sx >= int(width) || sy >= int(height))
                                continue;
                            const double k = double(gaussian[dx + 5]) * gaussian[dy + 5];
                            const size_t index = c * pixels + sy * width + sx;
                            const double a = image[index], b = target[index];
                            mx += k * a;
                            my += k * b;
                            xx += k * a * a;
                            yy += k * b * b;
                            xy += k * a * b;
                        }
                    const double a = mx * mx + my * my + 0.01f * 0.01f;
                    const double b = xx - mx * mx + yy - my * my + 0.03f * 0.03f;
                    const double d = 2 * (xy - mx * my) + 0.03f * 0.03f;
                    const double e = 2 * mx * my + 0.01f * 0.01f;
                    result.loss += weight * (1 - e * d / (a * b)) * normalization;
                    if (!gradient)
                        continue;
                    const double dm = 2 * my * (d - e) / (a * b) - 2 * mx * e * d / (a * a * b) + 2 * mx * e * d / (a * b * b);
                    const double dv = -e * d / (a * b * b), dc = 2 * e / (a * b);
                    for (int dy = -5; dy <= 5; ++dy)
                        for (int dx = -5; dx <= 5; ++dx) {
                            const int sx = x + dx, sy = y + dy;
                            if (sx < 0 || sy < 0 || sx >= int(width) || sy >= int(height))
                                continue;
                            const size_t index = c * pixels + sy * width + sx;
                            const double k = double(gaussian[dx + 5]) * gaussian[dy + 5];
                            result.gradient[index] -= weight * normalization * k * (dm + 2 * image[index] * dv + target[index] * dc);
                        }
                }
        return result;
    }
} // namespace photo_oracle

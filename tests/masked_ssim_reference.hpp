/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/tensor.hpp"
#include "lfs/kernels/ssim.cuh"

#include <array>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

// Independent double-precision, direct 2D convolution and analytic derivative.
// Outside-image zero padding remains part of the window; zero-weight samples
// inside the image do not contribute to either moments or their derivatives.
inline std::pair<float, lfs::core::Tensor> masked_ssim_reference(
    const lfs::core::Tensor& prediction, const lfs::core::Tensor& target,
    const lfs::core::Tensor& mask, float ssim_weight) {
    using namespace lfs::core;
    const auto shape = prediction.ndim() == 3 ? prediction.unsqueeze(0).shape() : prediction.shape();
    const int n = shape[0], channels = shape[1], h = shape[2], w = shape[3];
    const int pixels = h * w;
    const auto x = prediction.cpu().to_vector();
    const auto y = target.cpu().to_vector();
    const auto m = mask.to(DataType::Float32).cpu().to_vector();
    std::array<double, 11> gaussian{};
    double mass = 0.0;
    for (int k = -5; k <= 5; ++k) {
        gaussian[k + 5] = std::exp(-static_cast<double>(k * k) / 4.5);
        mass += gaussian[k + 5];
    }
    for (auto& value : gaussian)
        value /= mass;
    const double denominator = std::accumulate(m.begin(), m.end(), 0.0) * n * channels +
                               lfs::training::kernels::SSIM_EPSILON;
    double loss = 0.0;
    std::vector<double> grad(x.size(), 0.0);
    for (int bc = 0; bc < n * channels; ++bc) {
        for (int cy = 0; cy < h; ++cy) {
            for (int cx = 0; cx < w; ++cx) {
                const int center = cy * w + cx;
                if (m[center] == 0.0f)
                    continue;
                double norm = 0.0, mx = 0.0, my = 0.0, xx = 0.0, yy = 0.0, xy = 0.0;
                for (int dy = -5; dy <= 5; ++dy) {
                    for (int dx = -5; dx <= 5; ++dx) {
                        const int px = cx + dx, py = cy + dy;
                        const double weight = gaussian[dx + 5] * gaussian[dy + 5];
                        if (px < 0 || px >= w || py < 0 || py >= h) {
                            norm += weight;
                            continue;
                        }
                        const int p = py * w + px, i = bc * pixels + p;
                        const double q = weight * m[p];
                        norm += q;
                        mx += q * x[i];
                        my += q * y[i];
                        xx += q * x[i] * x[i];
                        yy += q * y[i] * y[i];
                        xy += q * x[i] * y[i];
                    }
                }
                mx /= norm;
                my /= norm;
                xx /= norm;
                yy /= norm;
                xy /= norm;
                const double a = mx * mx + my * my + 0.0001;
                const double b = xx - mx * mx + yy - my * my + 0.0009;
                const double c = 2.0 * mx * my + 0.0001;
                const double d = 2.0 * (xy - mx * my) + 0.0009;
                const double chain = m[center] / denominator;
                const int index = bc * pixels + center;
                const double delta = x[index] - y[index];
                loss += chain * ((1.0 - ssim_weight) * std::abs(delta) + ssim_weight * (1.0 - c * d / (a * b)));
                grad[index] += chain * (1.0 - ssim_weight) * ((delta > 0) - (delta < 0));
                const double dmu = 2.0 * my * d / (a * b) - 2.0 * my * c / (a * b) -
                                   2.0 * mx * c * d / (a * a * b) + 2.0 * mx * c * d / (a * b * b);
                const double dxx = -c * d / (a * b * b);
                const double dxy = 2.0 * c / (a * b);
                for (int dy = -5; dy <= 5; ++dy) {
                    for (int dx = -5; dx <= 5; ++dx) {
                        const int px = cx + dx, py = cy + dy;
                        if (px < 0 || px >= w || py < 0 || py >= h)
                            continue;
                        const int p = py * w + px, i = bc * pixels + p;
                        const double q = gaussian[dx + 5] * gaussian[dy + 5] * m[p] / norm;
                        grad[i] -= ssim_weight * chain * q * (dmu + 2.0 * x[i] * dxx + y[i] * dxy);
                    }
                }
            }
        }
    }
    const std::vector<float> result(grad.begin(), grad.end());
    return {static_cast<float>(loss), Tensor::from_vector(result, shape, Device::CUDA)};
}

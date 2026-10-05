/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "io/apple_reframe_conversion.hpp"
#include <iostream>
#include <limits>

int main() {
    using namespace lfs::io::reframe;
    auto require = [](bool condition) { if (!condition) throw std::runtime_error("Reframe conversion contract failed"); };
    require(std::abs(std::exp(logScale(0.025f)) - 0.025f) < 1e-6f);
    require(std::abs(opacityLogit(0.5f)) < 1e-6f);
    constexpr float c0 = 0.28209479177387814f;
    require(std::abs(viewerSh0((0.21404114f - 0.5f) / c0)) < 1e-5f);
    require(std::abs(viewerSh0(-0.5f / c0) * c0 + 0.5f) < 1e-6f);
    require(std::abs(viewerSh0(0.5f / c0) * c0 + 0.5f - 1.0f) < 1e-6f);
    require(std::abs(fitVerticalFov(0.5f, 1.5f) - 90.0f) < 1e-5f);
    require(std::abs(fitVerticalFov(2.0f, 1.0f) - 126.8699f) < 1e-4f);
    for (float alpha : {0.0f, 0.25f, 0.9f, 1.0f}) {
        const auto logit = opacityLogit(alpha);
        require(std::isfinite(logit));
        require(std::abs(1.0f / (1.0f + std::exp(-logit)) - alpha) < 2e-6f);
    }
    auto q = normalizedRotation({2, 0, 0, 0});
    require(q[0] == 1 && q[1] == 0 && q[2] == 0 && q[3] == 0);
    int rejected = 0;
    for (float invalid : {0.0f, -1.0f, std::numeric_limits<float>::infinity()}) {
        try {
            (void)logScale(invalid);
        } catch (const std::runtime_error&) { ++rejected; }
    }
    try {
        (void)opacityLogit(1.1f);
    } catch (const std::runtime_error&) { ++rejected; }
    try {
        (void)normalizedRotation({0, 0, 0, 0});
    } catch (const std::runtime_error&) { ++rejected; }
    try {
        (void)normalizedRotation({1, 0, std::numeric_limits<float>::quiet_NaN(), 0});
    } catch (const std::runtime_error&) { ++rejected; }
    require(rejected == 6);
    std::cout << "Reframe conversion contracts passed\n";
}

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <algorithm>
#include <cmath>
#include <nlohmann/json.hpp>
#include <vector>
namespace lfs::nodes {
    struct CurvePoint {
        float x;
        float y;
        float tangent = 0;
    };

    inline std::vector<CurvePoint> curve_control_points(const nlohmann::json& data) {
        std::vector<CurvePoint> result;
        if (data.is_array()) {
            for (const auto& point : data) {
                if (point.is_array() && point.size() >= 2 && point[0].is_number() && point[1].is_number()) {
                    const float x = point[0].get<float>();
                    const float y = point[1].get<float>();
                    if (std::isfinite(x) && std::isfinite(y))
                        result.push_back({x, y});
                }
            }
        }
        if (result.empty())
            result = {{0, 0}, {1, 1}};
        std::ranges::sort(result, {}, &CurvePoint::x);
        result.erase(std::unique(result.begin(), result.end(), [](const auto& a, const auto& b) {
                         return std::abs(a.x - b.x) <= 1e-7f;
                     }),
                     result.end());
        if (result.size() == 1)
            result.push_back({result.front().x + 1, result.front().y});

        const size_t n = result.size();
        std::vector<float> h(n - 1), delta(n - 1);
        for (size_t i = 0; i + 1 < n; ++i) {
            h[i] = result[i + 1].x - result[i].x;
            delta[i] = (result[i + 1].y - result[i].y) / h[i];
        }
        if (n == 2) {
            result[0].tangent = result[1].tangent = delta[0];
            return result;
        }
        const auto endpoint = [](float h0, float h1, float d0, float d1) {
            float m = ((2 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
            if (m * d0 <= 0)
                return 0.0f;
            if (d0 * d1 < 0 && std::abs(m) > std::abs(3 * d0))
                return 3 * d0;
            return m;
        };
        result.front().tangent = endpoint(h[0], h[1], delta[0], delta[1]);
        result.back().tangent = endpoint(h[n - 2], h[n - 3], delta[n - 2], delta[n - 3]);
        for (size_t i = 1; i + 1 < n; ++i) {
            if (delta[i - 1] * delta[i] <= 0) {
                result[i].tangent = 0;
            } else {
                const float w1 = 2 * h[i] + h[i - 1];
                const float w2 = h[i] + 2 * h[i - 1];
                result[i].tangent = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
            }
        }
        return result;
    }

    inline float sample_curve(const std::vector<CurvePoint>& points, float x) {
        x = std::clamp(x, points.front().x, points.back().x);
        auto end = std::upper_bound(points.begin(), points.end(), x, [](float value, const CurvePoint& p) { return value < p.x; });
        const size_t i = std::clamp<size_t>(size_t(end - points.begin()), 1, points.size() - 1) - 1;
        const auto& a = points[i];
        const auto& b = points[i + 1];
        const float h = b.x - a.x, t = (x - a.x) / h, t2 = t * t, t3 = t2 * t;
        return (2 * t3 - 3 * t2 + 1) * a.y + (t3 - 2 * t2 + t) * h * a.tangent + (3 * t2 - 2 * t3) * b.y + (t3 - t2) * h * b.tangent;
    }
} // namespace lfs::nodes

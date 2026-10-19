/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <glm/glm.hpp>
#include <limits>
#include <vector>

namespace lfs::vis {
    // Conservative coverage in render pixels, including line antialiasing and
    // the neighbouring texels read by the overlays' bilinear depth sampler.
    class FrustumDepthCoverage {
    public:
        // Matches HIGS_DEPTH_SAMPLE_TILE_SIZE in the rasterizer configuration.
        static constexpr int kTileSize = 4;
        FrustumDepthCoverage(glm::ivec2 size, float margin)
            : size_(size), mask_size_((size + kTileSize - 1) / kTileSize), margin_(margin), words_((size_t(mask_size_.x) * mask_size_.y + 31u) / 32u, 0u) {}

        void rectangle(glm::vec2 lo, glm::vec2 hi) {
            writeRectangle(lo - margin_, hi + margin_);
        }

        void quad(const std::array<glm::vec2, 4>& points) {
            float ymin = float(size_.y), ymax = 0;
            for (auto p : points) {
                ymin = std::min(ymin, p.y);
                ymax = std::max(ymax, p.y);
            }
            const int begin = int(std::floor(std::clamp(ymin - margin_, 0.0f, float(size_.y))));
            const int end = int(std::ceil(std::clamp(ymax + margin_, 0.0f, float(size_.y))));
            for (int y = begin; y < end; ++y) {
                const float low = y - margin_, high = y + 1.0f + margin_;
                float xmin = std::numeric_limits<float>::max(), xmax = -xmin;
                for (size_t i = 0; i < points.size(); ++i) {
                    const auto a = points[i], b = points[(i + 1) % points.size()];
                    if (std::max(a.y, b.y) < low || std::min(a.y, b.y) > high)
                        continue;
                    float x0 = a.x, x1 = b.x;
                    if (std::abs(b.y - a.y) > 1e-10f) {
                        x0 = glm::mix(a.x, b.x, std::clamp((low - a.y) / (b.y - a.y), 0.0f, 1.0f));
                        x1 = glm::mix(a.x, b.x, std::clamp((high - a.y) / (b.y - a.y), 0.0f, 1.0f));
                    }
                    xmin = std::min(xmin, std::min(x0, x1));
                    xmax = std::max(xmax, std::max(x0, x1));
                }
                if (xmin <= xmax)
                    writeRectangle({xmin - margin_, y}, {xmax + margin_, y + 1});
            }
        }

        void line(glm::vec2 a, glm::vec2 b) {
            // Clip before stepping: a near-plane crossing can project far
            // beyond the target and must not turn into unbounded CPU work.
            const glm::vec2 d = b - a;
            float begin = 0.0f, end = 1.0f;
            for (int axis = 0; axis < 2; ++axis) {
                if (std::abs(d[axis]) < 1e-10f) {
                    if (a[axis] < -margin_ || a[axis] > size_[axis] + margin_)
                        return;
                } else {
                    float t0 = (-margin_ - a[axis]) / d[axis];
                    float t1 = (size_[axis] + margin_ - a[axis]) / d[axis];
                    if (t0 > t1)
                        std::swap(t0, t1);
                    begin = std::max(begin, t0);
                    end = std::min(end, t1);
                }
            }
            if (begin > end)
                return;
            b = a + d * end;
            a += d * begin;
            const glm::vec2 extent = glm::abs(b - a);
            const int steps = std::max(1, int(std::ceil(std::max(extent.x, extent.y) / 8.0f)));
            for (int i = 0; i < steps; ++i) {
                const auto p = glm::mix(a, b, float(i) / steps);
                const auto q = glm::mix(a, b, float(i + 1) / steps);
                rectangle(glm::min(p, q), glm::max(p, q));
            }
        }

        std::vector<uint32_t> take() { return std::move(words_); }

    private:
        void writeRectangle(glm::vec2 lo, glm::vec2 hi) {
            lo = glm::clamp(lo, glm::vec2(0), glm::vec2(size_));
            hi = glm::clamp(hi, glm::vec2(0), glm::vec2(size_));
            if (lo.x >= hi.x || lo.y >= hi.y)
                return;
            lo /= float(kTileSize);
            hi /= float(kTileSize);
            const int x0 = int(std::floor(lo.x)), x1 = int(std::ceil(hi.x));
            for (int y = int(std::floor(lo.y)); y < int(std::ceil(hi.y)); ++y) {
                size_t bit = size_t(y) * mask_size_.x + x0;
                const size_t end = size_t(y) * mask_size_.x + x1;
                while (bit < end) {
                    const unsigned count = unsigned(std::min(end - bit, 32u - bit % 32u));
                    words_[bit / 32u] |= (0xffffffffu >> (32u - count)) << (bit % 32u);
                    bit += count;
                }
            }
        }

        glm::ivec2 size_;
        glm::ivec2 mask_size_;
        float margin_;
        std::vector<uint32_t> words_;
    };
} // namespace lfs::vis

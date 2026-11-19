/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

namespace lfs::training::morton {

    // Morton codes are taken over a contracted space: p = (x - centre) / radius, and p * (2 - 1/|p|) / |p| beyond
    // the unit ball, so every point lands in the ball of radius 2 and the grid spans [-2, 2]^3. The bulk of the
    // scene keeps the inner half of the grid at a resolution set by its own extent: a few distant points must not
    // stretch a global bounding box and collapse the rest of the scene into a handful of cells.
    inline constexpr float kContractedHalfExtent = 2.0f;
    inline constexpr std::size_t kContractionSampleRows = 65536;

    struct ContractionFrame {
        float centre[3] = {0.0f, 0.0f, 0.0f};
        float radius = 1.0f;
    };

    // Every backend samples rows 0, s, 2s, ... of the means (at most kContractionSampleRows of them).
    [[nodiscard]] inline std::size_t contraction_sample_stride(const std::size_t rows) {
        return std::max<std::size_t>(1, rows / kContractionSampleRows);
    }

    // Centre (per-axis median) and radius (median distance from it) of the sampled xyz rows.
    [[nodiscard]] inline ContractionFrame contraction_frame(const std::span<const float> sample_xyz) {
        const std::size_t m = sample_xyz.size() / 3;
        // Non-finite rows would break the selection's ordering; they get code 0 either way.
        std::vector<float> host;
        host.reserve(m * 3);
        for (std::size_t j = 0; j < m; ++j) {
            const float* p = sample_xyz.data() + j * 3;
            if (std::isfinite(p[0]) && std::isfinite(p[1]) && std::isfinite(p[2]))
                host.insert(host.end(), p, p + 3);
        }
        const std::size_t finite = host.size() / 3;
        ContractionFrame frame;
        if (finite == 0)
            return frame;
        std::vector<float> values(finite);
        const auto median = [&] {
            const auto mid = values.begin() + static_cast<std::ptrdiff_t>(finite / 2);
            std::nth_element(values.begin(), mid, values.end());
            return *mid;
        };
        for (int k = 0; k < 3; ++k) {
            for (std::size_t j = 0; j < finite; ++j)
                values[j] = host[j * 3 + k];
            frame.centre[k] = median();
        }
        for (std::size_t j = 0; j < finite; ++j) {
            const float* p = host.data() + j * 3;
            const float dx = p[0] - frame.centre[0], dy = p[1] - frame.centre[1], dz = p[2] - frame.centre[2];
            values[j] = std::sqrt(dx * dx + dy * dy + dz * dz);
        }
        const float radius = median();
        if (std::isfinite(radius) && radius > 0.0f)
            frame.radius = radius;
        return frame;
    }

} // namespace lfs::training::morton

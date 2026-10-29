/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <span>
#include <string_view>

namespace lfs::preprocessing::detail {

    struct RecoveryQuality {
        float iou = 0, object_score = 0, area = 0;
        bool empty = true;

        std::string_view rejection(float min_iou, float max_area) const {
            if (!(object_score > 0))
                return "object_absent";
            if (!(iou >= min_iou) || !std::isfinite(iou))
                return "low_iou";
            if (!(area <= max_area) || !std::isfinite(area))
                return "oversized";
            return empty || !(area > 0) ? "empty_prediction" : "";
        }
    };

    // Independent state for each sequence and propagation direction.
    struct MaskRecovery {
        int unreliable_frames = 0;

        void accept() {
            unreliable_frames = 0;
        }

        std::string_view observe(const RecoveryQuality& quality,
                                 float min_iou, float max_area, int patience) {
            if (quality.rejection(min_iou, max_area).empty()) {
                accept();
                return "";
            }
            unreliable_frames = std::min(unreliable_frames + 1, patience);
            if (quality.empty)
                return "empty_prediction";
            return unreliable_frames >= patience ? "no_reliable_frame" : "";
        }
    };

    inline std::optional<std::array<float, 4>> recovery_box(std::span<const float> mask,
                                                            int side, int width, int height) {
        if (side <= 0 || mask.size() != size_t(side) * side)
            return std::nullopt;
        int x0 = side, y0 = side, x1 = -1, y1 = -1;
        for (int y = 0; y < side; ++y)
            for (int x = 0; x < side; ++x)
                if (mask[size_t(y) * side + x] > 0.5f) {
                    x0 = std::min(x0, x);
                    y0 = std::min(y0, y);
                    x1 = std::max(x1, x);
                    y1 = std::max(y1, y);
                }
        if (x1 < x0)
            return std::nullopt;
        return std::array{float(x0) * width / side, float(y0) * height / side,
                          float(x1 + 1) * width / side, float(y1 + 1) * height / side};
    }

} // namespace lfs::preprocessing::detail

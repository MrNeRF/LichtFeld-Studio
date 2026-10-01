/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/camera.hpp"
#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>

namespace lfs::training {
    struct HoldoutAppearanceFrame {
        const lfs::core::Camera* camera;
        bool training;
    };

    struct HoldoutAppearanceSelection {
        int left_uid;
        int right_uid;
        int camera_id;
        float fraction;
    };

    inline std::optional<HoldoutAppearanceSelection> select_holdout_appearance(
        std::vector<HoldoutAppearanceFrame> capture, const lfs::core::Camera& query) {
        std::sort(capture.begin(), capture.end(), [](const auto& a, const auto& b) {
            return a.camera->image_name() < b.camera->image_name();
        });
        const auto position = std::lower_bound(capture.begin(), capture.end(), query.image_name(),
                                               [](const auto& frame, const auto& name) { return frame.camera->image_name() < name; });
        if (position == capture.end() || position->camera->image_name() != query.image_name() || position->training)
            return std::nullopt;
        const auto index = position - capture.begin();
        auto left = index - 1;
        auto right = index + 1;
        const auto size = static_cast<std::ptrdiff_t>(capture.size());
        while (left >= 0 && !capture[left].training)
            --left;
        while (right < size && !capture[right].training)
            ++right;
        if (left < 0 && right == size)
            return std::nullopt;
        if (left < 0)
            left = right;
        if (right == size)
            right = left;
        const float fraction = left == right ? 0.0f : static_cast<float>(index - left) / static_cast<float>(right - left);
        const auto nearest = index - left <= right - index ? left : right;
        return HoldoutAppearanceSelection{capture[left].camera->uid(), capture[right].camera->uid(), capture[nearest].camera->camera_id(), fraction};
    }
} // namespace lfs::training

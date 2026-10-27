/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera_types.h"
#include <algorithm>
#include <array>
#include <span>

namespace lfs::core {

    // Callers validate or truncate inputs to the supported lengths for their API.
    template <typename T>
    inline int packCameraDistortion(CameraModelType model, std::span<const float> radial,
                                    std::span<const float> tangential, std::span<T, 12> output) {
        std::fill(output.begin(), output.end(), T{0});
        if (model != CameraModelType::PINHOLE && model != CameraModelType::FISHEYE && model != CameraModelType::THIN_PRISM_FISHEYE)
            return 0;
        const bool rational = model == CameraModelType::PINHOLE && radial.size() == 6 &&
                              (radial[3] != 0.0f || radial[4] != 0.0f || radial[5] != 0.0f);
        const auto radial_count = model == CameraModelType::PINHOLE && !rational
                                      ? std::min<std::size_t>(radial.size(), 3)
                                      : radial.size();
        std::copy_n(radial.begin(), radial_count, output.begin());
        int count = static_cast<int>(radial_count);
        // COLMAP supplies p1, p2, sx1, sy1, sx2, sy2.
        constexpr std::array<int, 6> prism_slots = {4, 5, 6, 8, 7, 9};
        if (model != CameraModelType::FISHEYE) {
            for (std::size_t i = 0; i < tangential.size(); ++i) {
                const int slot = model == CameraModelType::PINHOLE ? (rational ? 6 : 3) + static_cast<int>(i) : prism_slots[i];
                output[slot] = tangential[i];
                count = std::max(count, slot + 1);
            }
        }
        return count;
    }

} // namespace lfs::core

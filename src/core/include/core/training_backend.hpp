/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <array>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace lfs::core::param {
    enum class RasterBackendId { FastGS,
                                 ThreeDGUT };

    struct TrainingBackendCapabilities {
        bool igs_plus;
        bool undistort;
        bool mip_filter;
        bool depth_supervision;
        bool normal_supervision;
    };

    struct TrainingBackendDescriptor {
        RasterBackendId id;
        std::string_view wire_name;
        std::string_view label;
        std::string_view viewer_name;
        TrainingBackendCapabilities capabilities;
    };

    // Only verified training restrictions belong here. Mask/appearance availability
    // must not be inferred from the supervision capabilities.
    inline constexpr std::array kTrainingBackends{
        TrainingBackendDescriptor{RasterBackendId::FastGS, "fastgs", "FastGS", "3dgs", {true, true, true, true, true}},
        TrainingBackendDescriptor{RasterBackendId::ThreeDGUT, "3dgut", "3DGUT", "3dgut", {false, false, false, false, false}},
    };

    [[nodiscard]] inline std::optional<RasterBackendId> parse_training_backend(std::string_view name) {
        for (const auto& backend : kTrainingBackends)
            if (backend.wire_name == name)
                return backend.id;
        return std::nullopt;
    }

    [[nodiscard]] inline const TrainingBackendDescriptor& training_backend_descriptor(RasterBackendId id) {
        for (const auto& backend : kTrainingBackends)
            if (backend.id == id)
                return backend;
        throw std::invalid_argument("Unknown training raster backend");
    }
} // namespace lfs::core::param

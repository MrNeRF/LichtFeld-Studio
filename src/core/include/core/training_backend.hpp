/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <array>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace lfs::core::param {
    enum class RasterBackendId {
        ThreeDGS,
        ThreeDGUT,
    };

    enum class TrainingFeatureSupport {
        Supported,
        Unsupported,
    };

    [[nodiscard]] inline constexpr std::string_view training_feature_support_name(
        const TrainingFeatureSupport support) {
        switch (support) {
        case TrainingFeatureSupport::Supported:
            return "supported";
        case TrainingFeatureSupport::Unsupported:
            return "unsupported";
        }
        throw std::invalid_argument("Unknown training feature support state");
    }

    [[nodiscard]] inline constexpr bool is_training_feature_unsupported(
        const TrainingFeatureSupport support) noexcept {
        return support == TrainingFeatureSupport::Unsupported;
    }

    struct TrainingBackendCapabilities {
        TrainingFeatureSupport mcmc = TrainingFeatureSupport::Unsupported;
        TrainingFeatureSupport mrnf = TrainingFeatureSupport::Unsupported;
        TrainingFeatureSupport igs_plus = TrainingFeatureSupport::Unsupported;
        TrainingFeatureSupport undistort = TrainingFeatureSupport::Unsupported;
        TrainingFeatureSupport mip_filter = TrainingFeatureSupport::Unsupported;
        TrainingFeatureSupport depth_supervision = TrainingFeatureSupport::Unsupported;
        TrainingFeatureSupport normal_supervision = TrainingFeatureSupport::Unsupported;
        TrainingFeatureSupport masking = TrainingFeatureSupport::Unsupported;
        TrainingFeatureSupport segmentation = TrainingFeatureSupport::Unsupported;
        TrainingFeatureSupport background_modes = TrainingFeatureSupport::Unsupported;
        TrainingFeatureSupport background_improvements = TrainingFeatureSupport::Unsupported;
        TrainingFeatureSupport exposure_correction = TrainingFeatureSupport::Unsupported;
        TrainingFeatureSupport bilateral_grid = TrainingFeatureSupport::Unsupported;
        TrainingFeatureSupport ppisp = TrainingFeatureSupport::Unsupported;
        TrainingFeatureSupport sparsity = TrainingFeatureSupport::Unsupported;
    };

    struct TrainingBackendDescriptor {
        RasterBackendId id;
        std::string_view wire_name;
        std::string_view label;
        std::string_view viewer_name;
        std::string_view description;
        TrainingBackendCapabilities capabilities;
    };

    // High-level feature groups belong here. Unsupported entries are hard backend
    // conflicts. Descriptors must explicitly assess every feature before exposure.
    inline constexpr std::array kTrainingBackends{
        TrainingBackendDescriptor{
            .id = RasterBackendId::ThreeDGS,
            .wire_name = "3dgs",
            .label = "3DGS",
            .viewer_name = "3dgs",
            .description = "3DGS: Original 3D Gaussian Splatting based on EWA projection",
            .capabilities = {
                .mcmc = TrainingFeatureSupport::Supported,
                .mrnf = TrainingFeatureSupport::Supported,
                .igs_plus = TrainingFeatureSupport::Supported,
                .undistort = TrainingFeatureSupport::Supported,
                .mip_filter = TrainingFeatureSupport::Supported,
                .depth_supervision = TrainingFeatureSupport::Supported,
                .normal_supervision = TrainingFeatureSupport::Supported,
                .masking = TrainingFeatureSupport::Supported,
                .segmentation = TrainingFeatureSupport::Supported,
                .background_modes = TrainingFeatureSupport::Supported,
                .background_improvements = TrainingFeatureSupport::Supported,
                .exposure_correction = TrainingFeatureSupport::Supported,
                .bilateral_grid = TrainingFeatureSupport::Supported,
                .ppisp = TrainingFeatureSupport::Supported,
                .sparsity = TrainingFeatureSupport::Supported,
            },
        },
        TrainingBackendDescriptor{
            .id = RasterBackendId::ThreeDGUT,
            .wire_name = "3dgut",
            .label = "3DGUT",
            .viewer_name = "3dgut",
            .description =
                "3DGUT: Gaussian rasterization using the Unscented Transform for nonlinear "
                "and distorted camera models",
            .capabilities = {
                .mcmc = TrainingFeatureSupport::Supported,
                .mrnf = TrainingFeatureSupport::Supported,
                .igs_plus = TrainingFeatureSupport::Unsupported,
                .undistort = TrainingFeatureSupport::Supported,
                .mip_filter = TrainingFeatureSupport::Unsupported,
                .depth_supervision = TrainingFeatureSupport::Unsupported,
                .normal_supervision = TrainingFeatureSupport::Unsupported,
                .masking = TrainingFeatureSupport::Supported,
                .segmentation = TrainingFeatureSupport::Supported,
                .background_modes = TrainingFeatureSupport::Supported,
                .background_improvements = TrainingFeatureSupport::Supported,
                .exposure_correction = TrainingFeatureSupport::Supported,
                .bilateral_grid = TrainingFeatureSupport::Supported,
                .ppisp = TrainingFeatureSupport::Supported,
                .sparsity = TrainingFeatureSupport::Supported,
            },
        },
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

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "rendering_types.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <glm/glm.hpp>

namespace lfs::vis::detail {

    struct GTComparisonSourceKey {
        int camera_uid = -1;
        std::filesystem::path image_path;

        friend bool operator==(const GTComparisonSourceKey&,
                               const GTComparisonSourceKey&) = default;
    };

    struct GTComparisonCrop {
        glm::ivec2 origin{0, 0};
        glm::ivec2 extent{0, 0};

        [[nodiscard]] bool valid() const {
            return extent.x > 0 && extent.y > 0;
        }

        friend bool operator==(const GTComparisonCrop&,
                               const GTComparisonCrop&) = default;
    };

    struct GTComparisonTileKey {
        std::uint64_t source_generation = 0;
        glm::ivec2 full_extent{0, 0};
        glm::ivec2 framebuffer_extent{0, 0};
        GTComparisonCrop crop{};
        bool distorted = false;

        friend bool operator==(const GTComparisonTileKey&,
                               const GTComparisonTileKey&) = default;
    };

    [[nodiscard]] inline GTComparisonCrop clampGTComparisonCrop(
        const glm::ivec2 full_extent,
        const glm::ivec2 viewport_extent,
        const glm::ivec2 requested_origin) {
        if (full_extent.x <= 0 || full_extent.y <= 0 ||
            viewport_extent.x <= 0 || viewport_extent.y <= 0) {
            return {};
        }
        const glm::ivec2 extent{
            std::min(full_extent.x, viewport_extent.x),
            std::min(full_extent.y, viewport_extent.y)};
        const glm::ivec2 max_origin{
            std::max(full_extent.x - extent.x, 0),
            std::max(full_extent.y - extent.y, 0)};
        return {
            .origin = {
                std::clamp(requested_origin.x, 0, max_origin.x),
                std::clamp(requested_origin.y, 0, max_origin.y)},
            .extent = extent};
    }

    [[nodiscard]] inline GTComparisonCrop centerGTComparisonCrop(
        const glm::ivec2 full_extent,
        const glm::ivec2 viewport_extent) {
        if (full_extent.x <= 0 || full_extent.y <= 0 ||
            viewport_extent.x <= 0 || viewport_extent.y <= 0) {
            return {};
        }
        const glm::ivec2 extent{
            std::min(full_extent.x, viewport_extent.x),
            std::min(full_extent.y, viewport_extent.y)};
        return clampGTComparisonCrop(
            full_extent,
            viewport_extent,
            {(full_extent.x - extent.x) / 2,
             (full_extent.y - extent.y) / 2});
    }

    [[nodiscard]] inline bool isGTComparisonCropValidForViewport(
        const glm::ivec2 full_extent,
        const glm::ivec2 viewport_extent,
        const GTComparisonCrop crop) {
        if (full_extent.x <= 0 || full_extent.y <= 0 ||
            viewport_extent.x <= 0 || viewport_extent.y <= 0) {
            return false;
        }
        const glm::ivec2 expected_extent{
            std::min(full_extent.x, viewport_extent.x),
            std::min(full_extent.y, viewport_extent.y)};
        const glm::ivec2 max_origin{
            full_extent.x - expected_extent.x,
            full_extent.y - expected_extent.y};
        return crop.extent == expected_extent &&
               crop.origin.x >= 0 && crop.origin.y >= 0 &&
               crop.origin.x <= max_origin.x && crop.origin.y <= max_origin.y;
    }

    [[nodiscard]] inline GTComparisonCrop resizeGTComparisonCropPreservingCenter(
        const glm::ivec2 full_extent,
        const glm::ivec2 viewport_extent,
        const GTComparisonCrop previous_crop) {
        if (full_extent.x <= 0 || full_extent.y <= 0 ||
            viewport_extent.x <= 0 || viewport_extent.y <= 0 ||
            previous_crop.origin.x < 0 || previous_crop.origin.y < 0 ||
            previous_crop.extent.x <= 0 || previous_crop.extent.y <= 0 ||
            previous_crop.extent.x > full_extent.x ||
            previous_crop.extent.y > full_extent.y ||
            previous_crop.origin.x > full_extent.x - previous_crop.extent.x ||
            previous_crop.origin.y > full_extent.y - previous_crop.extent.y) {
            return centerGTComparisonCrop(full_extent, viewport_extent);
        }

        const glm::ivec2 next_extent{
            std::min(full_extent.x, viewport_extent.x),
            std::min(full_extent.y, viewport_extent.y)};
        const auto preserve_axis = [](const int origin,
                                      const int old_extent,
                                      const int new_extent) {
            return static_cast<int>(std::lround(
                static_cast<double>(origin) +
                (static_cast<double>(old_extent) -
                 static_cast<double>(new_extent)) /
                    2.0));
        };
        return clampGTComparisonCrop(
            full_extent,
            viewport_extent,
            {preserve_axis(
                 previous_crop.origin.x, previous_crop.extent.x, next_extent.x),
             preserve_axis(
                 previous_crop.origin.y, previous_crop.extent.y, next_extent.y)});
    }

    // x, y, width, height in top-left physical framebuffer coordinates.
    [[nodiscard]] inline glm::ivec4 centeredGTComparisonContentRect(
        const glm::ivec2 framebuffer_extent,
        const glm::ivec2 content_extent) {
        if (framebuffer_extent.x <= 0 || framebuffer_extent.y <= 0 ||
            content_extent.x <= 0 || content_extent.y <= 0) {
            return {0, 0, 0, 0};
        }
        const glm::ivec2 bounded{
            std::min(framebuffer_extent.x, content_extent.x),
            std::min(framebuffer_extent.y, content_extent.y)};
        return {
            (framebuffer_extent.x - bounded.x) / 2,
            (framebuffer_extent.y - bounded.y) / 2,
            bounded.x,
            bounded.y};
    }

    // Map both physical edges independently so odd HiDPI letterbox widths do
    // not accumulate a second rounding convention.
    [[nodiscard]] inline glm::vec4 physicalToLogicalContentRect(
        const glm::ivec4 physical_rect,
        const glm::ivec2 physical_extent,
        const glm::ivec2 logical_extent) {
        if (physical_extent.x <= 0 || physical_extent.y <= 0 ||
            logical_extent.x <= 0 || logical_extent.y <= 0 ||
            physical_rect.z <= 0 || physical_rect.w <= 0) {
            return {0.0f, 0.0f, 0.0f, 0.0f};
        }
        const double sx = static_cast<double>(logical_extent.x) /
                          static_cast<double>(physical_extent.x);
        const double sy = static_cast<double>(logical_extent.y) /
                          static_cast<double>(physical_extent.y);
        const double left = static_cast<double>(physical_rect.x) * sx;
        const double top = static_cast<double>(physical_rect.y) * sy;
        const double right = static_cast<double>(physical_rect.x + physical_rect.z) * sx;
        const double bottom = static_cast<double>(physical_rect.y + physical_rect.w) * sy;
        return {
            static_cast<float>(left),
            static_cast<float>(top),
            static_cast<float>(right - left),
            static_cast<float>(bottom - top)};
    }

    [[nodiscard]] inline glm::dvec2 physicalScaleForExtents(
        const glm::ivec2 logical_extent,
        const glm::ivec2 physical_extent) {
        return {
            logical_extent.x > 0 && physical_extent.x > 0
                ? static_cast<double>(physical_extent.x) /
                      static_cast<double>(logical_extent.x)
                : 1.0,
            logical_extent.y > 0 && physical_extent.y > 0
                ? static_cast<double>(physical_extent.y) /
                      static_cast<double>(logical_extent.y)
                : 1.0};
    }

    [[nodiscard]] inline glm::ivec2 roundedPhysicalDrag(
        const glm::dvec2 logical_displacement,
        const glm::dvec2 captured_scale) {
        const glm::dvec2 physical = logical_displacement * captured_scale;
        return {
            static_cast<int>(std::lround(physical.x)),
            static_cast<int>(std::lround(physical.y))};
    }

    [[nodiscard]] inline bool isGTComparisonActualSizeCameraModelSupported(
        const lfs::core::CameraModelType model) {
        return model == lfs::core::CameraModelType::PINHOLE ||
               model == lfs::core::CameraModelType::FISHEYE ||
               model == lfs::core::CameraModelType::THIN_PRISM_FISHEYE;
    }

    [[nodiscard]] inline bool isGTComparisonActualSizeAvailable(
        const lfs::core::Camera& camera,
        const GTComparisonMode mode) {
        return mode == GTComparisonMode::RGB &&
               camera.has_image() && !camera.image_path().empty() &&
               isGTComparisonActualSizeCameraModelSupported(camera.camera_model_type()) &&
               (!camera.has_distortion() || camera.is_undistort_precomputed());
    }

} // namespace lfs::vis::detail

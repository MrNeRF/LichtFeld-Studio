/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace lfs::geometry::apriltag {

    enum class AprilTagFamily : std::uint8_t {
        Tag36h11 = 0,
        Tag25h9 = 1,
        Tag16h5 = 2,
        Tag36h10 = 3,
        TagStandard41h12 = 4,
        TagCircle21h7 = 5,
    };

    [[nodiscard]] std::string_view aprilTagFamilyName(AprilTagFamily family) noexcept;
    [[nodiscard]] std::optional<AprilTagFamily> parseAprilTagFamily(std::string_view name) noexcept;

    struct AprilTagDetectorConfig {
        AprilTagFamily family = AprilTagFamily::Tag36h11;
        float quad_decimate = 2.0f;
        float quad_sigma = 0.0f;
        bool refine_edges = true;
        double decode_sharpening = 0.25;
        int max_hamming = 1;
        float min_decision_margin = 25.0f;
        int num_threads = 1;
    };

    struct TagDetection2D {
        int id = -1;
        int hamming = 0;
        float decision_margin = 0.0f;
        glm::dvec2 center{0.0, 0.0};
    };

    struct GrayscaleBitmap {
        int width = 0;
        int height = 0;
        std::vector<std::uint8_t> pixels;
    };

    [[nodiscard]] std::optional<GrayscaleBitmap> renderAprilTagBitmap(
        AprilTagFamily family,
        int tag_id,
        int scale_px = 8,
        int quiet_zone_modules = 2);

    class AprilTagDetector {
    public:
        explicit AprilTagDetector(const AprilTagDetectorConfig& config = {});
        ~AprilTagDetector();

        AprilTagDetector(const AprilTagDetector&) = delete;
        AprilTagDetector& operator=(const AprilTagDetector&) = delete;

        [[nodiscard]] std::vector<TagDetection2D> detectGrayscale(
            const std::uint8_t* data,
            int width,
            int height,
            int row_stride_bytes = 0) const;

        [[nodiscard]] std::vector<TagDetection2D> detectInterleaved(
            const std::uint8_t* data,
            int width,
            int height,
            int channels,
            int row_stride_bytes = 0) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::geometry::apriltag

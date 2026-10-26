/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "geometry/apriltag_detector.hpp"

#include <apriltag/apriltag.h>
#include <apriltag/common/image_u8.h>
#include <apriltag/tag16h5.h>
#include <apriltag/tag25h9.h>
#include <apriltag/tag36h10.h>
#include <apriltag/tag36h11.h>
#include <apriltag/tagCircle21h7.h>
#include <apriltag/tagStandard41h12.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <mutex>
#include <string>

namespace lfs::geometry::apriltag {

    namespace {

        struct Family {
            AprilTagFamily id;
            std::string_view name;
            apriltag_family_t* (*create)();
            void (*destroy)(apriltag_family_t*);
        };

        constexpr Family families[] = {
            {AprilTagFamily::Tag36h11, "tag36h11", tag36h11_create, tag36h11_destroy},
            {AprilTagFamily::Tag25h9, "tag25h9", tag25h9_create, tag25h9_destroy},
            {AprilTagFamily::Tag16h5, "tag16h5", tag16h5_create, tag16h5_destroy},
            {AprilTagFamily::Tag36h10, "tag36h10", tag36h10_create, tag36h10_destroy},
            {AprilTagFamily::TagStandard41h12, "tagStandard41h12", tagStandard41h12_create, tagStandard41h12_destroy},
            {AprilTagFamily::TagCircle21h7, "tagCircle21h7", tagCircle21h7_create, tagCircle21h7_destroy},
        };

        const Family& familyInfo(const AprilTagFamily family) {
            for (const auto& entry : families) {
                if (entry.id == family)
                    return entry;
            }
            return families[0];
        }

    } // namespace

    std::string_view aprilTagFamilyName(const AprilTagFamily family) noexcept {
        return familyInfo(family).name;
    }

    std::optional<AprilTagFamily> parseAprilTagFamily(const std::string_view name) noexcept {
        std::string normalized;
        normalized.reserve(name.size());
        for (const char ch : name) {
            if (ch != '_' && ch != '-' && ch != ' ') {
                normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
            }
        }

        if (normalized == "tag36h11" || normalized == "36h11") {
            return AprilTagFamily::Tag36h11;
        }
        if (normalized == "tag25h9" || normalized == "25h9") {
            return AprilTagFamily::Tag25h9;
        }
        if (normalized == "tag16h5" || normalized == "16h5") {
            return AprilTagFamily::Tag16h5;
        }
        if (normalized == "tag36h10" || normalized == "36h10") {
            return AprilTagFamily::Tag36h10;
        }
        if (normalized == "tagstandard41h12" || normalized == "standard41h12" || normalized == "41h12") {
            return AprilTagFamily::TagStandard41h12;
        }
        if (normalized == "tagcircle21h7" || normalized == "circle21h7" || normalized == "21h7") {
            return AprilTagFamily::TagCircle21h7;
        }
        return std::nullopt;
    }

    std::optional<GrayscaleBitmap> renderAprilTagBitmap(
        const AprilTagFamily family,
        const int tag_id,
        const int scale_px,
        const int quiet_zone_modules) {
        if (tag_id < 0 || scale_px <= 0 || quiet_zone_modules < 0) {
            return std::nullopt;
        }

        apriltag_family_t* const tf = familyInfo(family).create();
        if (!tf) {
            return std::nullopt;
        }

        if (static_cast<std::uint32_t>(tag_id) >= tf->ncodes) {
            familyInfo(family).destroy(tf);
            return std::nullopt;
        }

        image_u8_t* const base_img = apriltag_to_image(tf, static_cast<std::uint32_t>(tag_id));
        familyInfo(family).destroy(tf);
        if (!base_img) {
            return std::nullopt;
        }

        const int base_w = base_img->width;
        const int base_h = base_img->height;
        const int out_w = (base_w + 2 * quiet_zone_modules) * scale_px;
        const int out_h = (base_h + 2 * quiet_zone_modules) * scale_px;

        GrayscaleBitmap result;
        result.width = out_w;
        result.height = out_h;
        result.pixels.assign(static_cast<std::size_t>(out_w) * static_cast<std::size_t>(out_h), 255U);

        for (int by = 0; by < base_h; ++by) {
            const std::uint8_t* const src_row = base_img->buf + by * base_img->stride;
            for (int bx = 0; bx < base_w; ++bx) {
                const std::uint8_t val = src_row[bx];
                const int start_y = (by + quiet_zone_modules) * scale_px;
                const int start_x = (bx + quiet_zone_modules) * scale_px;
                for (int dy = 0; dy < scale_px; ++dy) {
                    std::uint8_t* const dst_row =
                        result.pixels.data() + static_cast<std::size_t>(start_y + dy) * static_cast<std::size_t>(out_w);
                    for (int dx = 0; dx < scale_px; ++dx) {
                        dst_row[start_x + dx] = val;
                    }
                }
            }
        }

        image_u8_destroy(base_img);
        return result;
    }

    struct AprilTagDetector::Impl {
        AprilTagDetectorConfig config;
        apriltag_family_t* family = nullptr;
        apriltag_detector_t* detector = nullptr;
        mutable std::mutex mutex;

        explicit Impl(const AprilTagDetectorConfig& cfg)
            : config(cfg) {
            family = familyInfo(config.family).create();
            detector = apriltag_detector_create();
            if (detector && family) {
                const int bits_corrected = std::clamp(config.max_hamming, 0, 2);
                apriltag_detector_add_family_bits(detector, family, bits_corrected);
                detector->quad_decimate = std::max(0.0f, config.quad_decimate);
                detector->quad_sigma = config.quad_sigma;
                detector->refine_edges = config.refine_edges;
                detector->decode_sharpening = config.decode_sharpening;
                detector->nthreads = std::max(1, config.num_threads);
            }
        }

        ~Impl() {
            if (detector) {
                apriltag_detector_destroy(detector);
            }
            if (family) {
                familyInfo(config.family).destroy(family);
            }
        }
    };

    AprilTagDetector::AprilTagDetector(const AprilTagDetectorConfig& config)
        : impl_(std::make_unique<Impl>(config)) {
    }

    AprilTagDetector::~AprilTagDetector() = default;

    std::vector<TagDetection2D> AprilTagDetector::detectGrayscale(
        const std::uint8_t* const data,
        const int width,
        const int height,
        const int row_stride_bytes) const {
        if (!data || width <= 0 || height <= 0 || !impl_ || !impl_->detector) {
            return {};
        }

        const int src_stride = row_stride_bytes > 0 ? row_stride_bytes : width;
        if (src_stride < width) {
            return {};
        }

        image_u8_t* const img = image_u8_create(width, height);
        if (!img) {
            return {};
        }

        for (int y = 0; y < height; ++y) {
            std::memcpy(
                img->buf + static_cast<std::size_t>(y) * static_cast<std::size_t>(img->stride),
                data + static_cast<std::size_t>(y) * static_cast<std::size_t>(src_stride),
                static_cast<std::size_t>(width));
        }

        std::vector<TagDetection2D> results;
        {
            std::lock_guard<std::mutex> lock(impl_->mutex);
            zarray_t* const detections = apriltag_detector_detect(impl_->detector, img);
            if (detections) {
                const int count = zarray_size(detections);
                results.reserve(static_cast<std::size_t>(count));
                for (int i = 0; i < count; ++i) {
                    apriltag_detection_t* det = nullptr;
                    zarray_get(detections, i, &det);
                    if (!det) {
                        continue;
                    }
                    if (det->hamming > impl_->config.max_hamming ||
                        det->decision_margin < impl_->config.min_decision_margin) {
                        continue;
                    }

                    TagDetection2D item;
                    item.id = det->id;
                    item.hamming = det->hamming;
                    item.decision_margin = det->decision_margin;
                    item.center = glm::dvec2(det->c[0], det->c[1]);
                    results.push_back(item);
                }
                apriltag_detections_destroy(detections);
            }
        }

        image_u8_destroy(img);
        std::sort(results.begin(), results.end(), [](const TagDetection2D& a, const TagDetection2D& b) {
            if (a.id != b.id) {
                return a.id < b.id;
            }
            return a.decision_margin > b.decision_margin;
        });
        return results;
    }

    std::vector<TagDetection2D> AprilTagDetector::detectInterleaved(
        const std::uint8_t* const data,
        const int width,
        const int height,
        const int channels,
        const int row_stride_bytes) const {
        if (!data || width <= 0 || height <= 0 || channels <= 0) {
            return {};
        }

        if (channels == 1) {
            return detectGrayscale(data, width, height, row_stride_bytes);
        }

        const int src_stride = row_stride_bytes > 0 ? row_stride_bytes : width * channels;
        if (src_stride < width * channels) {
            return {};
        }

        std::vector<std::uint8_t> gray(static_cast<std::size_t>(width) * static_cast<std::size_t>(height));
        for (int y = 0; y < height; ++y) {
            const std::uint8_t* const src_row =
                data + static_cast<std::size_t>(y) * static_cast<std::size_t>(src_stride);
            std::uint8_t* const dst_row =
                gray.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(width);

            if (channels >= 3) {
                for (int x = 0; x < width; ++x) {
                    const std::uint8_t r = src_row[x * channels + 0];
                    const std::uint8_t g = src_row[x * channels + 1];
                    const std::uint8_t b = src_row[x * channels + 2];
                    dst_row[x] = static_cast<std::uint8_t>((77U * r + 150U * g + 29U * b + 128U) >> 8U);
                }
            } else {
                for (int x = 0; x < width; ++x) {
                    dst_row[x] = src_row[x * channels];
                }
            }
        }

        return detectGrayscale(gray.data(), width, height, width);
    }

} // namespace lfs::geometry::apriltag

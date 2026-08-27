/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/point_cloud.hpp"
#include "io/loader.hpp"
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <glm/glm.hpp>
#include <memory>
#include <random>
#include <span>
#include <vector>

namespace lfs::io {

    inline bool detect_camera_alpha(const std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
                                    const CancelCallback& cancel_requested = nullptr) {
        bool images_have_alpha = false;
        size_t alpha_count = 0;
        for (size_t i = 0; i < cameras.size(); ++i) {
            if (cancel_requested && (i % 64) == 0 && cancel_requested()) {
                throw LoadCancelledError("Image alpha probe cancelled");
            }

            const auto& cam = cameras[i];
            if (!cam->has_image())
                continue;
            auto ext = cam->image_path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg")
                continue;
            try {
                auto [w, h, c] = lfs::core::get_image_info(cam->image_path());
                if (c == 4) {
                    cam->set_has_alpha(true);
                    images_have_alpha = true;
                    ++alpha_count;
                }
            } catch (const std::exception& e) {
                LOG_DEBUG("Failed to probe alpha for '{}': {}", cam->image_name(), e.what());
            }
        }
        if (alpha_count > 0) {
            LOG_INFO("Alpha channel detected in {}/{} images", alpha_count, cameras.size());
        }
        return images_have_alpha;
    }

    // Centralizes the scene in-place based on the given mode.
    // Shifts both cameras and point cloud by the computed center.
    struct CentralizationResult {
        lfs::core::Tensor scene_center;
        std::optional<ImportGeoreference> georeference;
    };

    inline glm::dvec3 geometric_median_f64(
        const std::span<const glm::dvec3> points) {
        if (points.empty()) {
            return {};
        }
        if (points.size() == 1) {
            return points.front();
        }
        if (points.size() == 2) {
            return (points[0] + points[1]) * 0.5;
        }
        glm::dvec3 estimate{};
        for (const auto& point : points) {
            estimate += point;
        }
        estimate /= static_cast<double>(points.size());
        constexpr double EPSILON = 1e-12;
        constexpr double TOLERANCE = 1e-9;
        for (int iteration = 0; iteration < 100;
             ++iteration) {
            glm::dvec3 weighted{};
            double weights = 0.0;
            for (const auto& point : points) {
                const double distance =
                    glm::length(estimate - point);
                if (distance < EPSILON) {
                    continue;
                }
                const double weight = 1.0 / distance;
                weighted += point * weight;
                weights += weight;
            }
            if (weights < EPSILON) {
                break;
            }
            const glm::dvec3 next = weighted / weights;
            if (glm::length(next - estimate) < TOLERANCE) {
                estimate = next;
                break;
            }
            estimate = next;
        }
        return estimate;
    }

    // The recorded origin is the removed offset: world = local + world_origin.
    inline CentralizationResult centralize_scene(
        std::vector<std::shared_ptr<lfs::core::Camera>>& cameras,
        std::shared_ptr<lfs::core::PointCloud>& point_cloud,
        CentralizeDataset mode,
        lfs::core::Tensor initial_scene_center) {

        lfs::core::Tensor center;

        auto build_pts = [&]() {
            auto positions = point_cloud->means.cpu().contiguous();
            const int64_t N = static_cast<int64_t>(positions.shape()[0]);
            auto pos_acc = positions.accessor<float, 2>();
            constexpr int64_t MAX_SAMPLES = 50000;
            std::vector<glm::dvec3> pts;
            if (N <= MAX_SAMPLES) {
                pts.resize(static_cast<size_t>(N));
                for (int64_t i = 0; i < N; ++i)
                    pts[static_cast<size_t>(i)] = {
                        static_cast<double>(pos_acc(i, 0)),
                        static_cast<double>(pos_acc(i, 1)),
                        static_cast<double>(pos_acc(i, 2))};
            } else {
                pts.resize(MAX_SAMPLES);
                std::mt19937 rng(33550336);
                std::uniform_int_distribution<int64_t> dist(0, N - 1);
                for (int64_t i = 0; i < MAX_SAMPLES; ++i) {
                    int64_t idx = dist(rng);
                    pts[static_cast<size_t>(i)] = {
                        static_cast<double>(pos_acc(idx, 0)),
                        static_cast<double>(pos_acc(idx, 1)),
                        static_cast<double>(pos_acc(idx, 2))};
                }
            }
            return pts;
        };

        auto build_camera_pts = [&]() {
            std::vector<glm::dvec3> pts;
            pts.reserve(cameras.size());
            for (const auto& cam : cameras) {
                auto pos = cam->cam_position().cpu();
                const float* p = pos.ptr<float>();
                pts.push_back({static_cast<double>(p[0]),
                               static_cast<double>(p[1]),
                               static_cast<double>(p[2])});
            }
            return pts;
        };

        std::optional<std::array<double, 3>>
            removed_origin;
        switch (mode) {
        case CentralizeDataset::ByPointCloud:
            if (point_cloud && point_cloud->size() > 0) {
                auto pts = build_pts();
                const auto med = geometric_median_f64(pts);
                removed_origin =
                    std::array{med.x, med.y, med.z};
                std::vector<float> center_data = {
                    static_cast<float>(med.x),
                    static_cast<float>(med.y),
                    static_cast<float>(med.z)};
                center = lfs::core::Tensor::from_vector(center_data, {3}, lfs::core::Device::CPU);
                LOG_INFO("Centralizing by point cloud (geometric median): center=[{:.3f}, {:.3f}, {:.3f}]",
                         med.x, med.y, med.z);
            }
            break;
        case CentralizeDataset::ByCameras:
            if (!cameras.empty()) {
                auto cam_pts = build_camera_pts();
                const auto med =
                    geometric_median_f64(cam_pts);
                removed_origin =
                    std::array{med.x, med.y, med.z};
                std::vector<float> center_data = {
                    static_cast<float>(med.x),
                    static_cast<float>(med.y),
                    static_cast<float>(med.z)};
                center = lfs::core::Tensor::from_vector(center_data, {3}, lfs::core::Device::CPU);
                LOG_INFO("Centralizing by cameras (geometric median): center=[{:.3f}, {:.3f}, {:.3f}]",
                         med.x, med.y, med.z);
            }
            break;
        default:
            break;
        }

        if (center.is_valid()) {
            if (point_cloud && point_cloud->size() > 0 && point_cloud->means.is_valid()) {
                auto center_dev = center.to(point_cloud->means.device());
                point_cloud->means = point_cloud->means - center_dev;
            }

            auto neg_center = center.neg();
            for (auto& cam : cameras)
                cam->translate(neg_center);

            assert(removed_origin);
            return CentralizationResult{
                .scene_center =
                    lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
                .georeference =
                    ImportGeoreference{
                        .crs = std::nullopt,
                        .world_origin = *removed_origin,
                        .world_unit_scale = 1.0,
                        .world_origin_provenance =
                            mode == CentralizeDataset::ByPointCloud
                                ? ImportWorldOriginProvenance::
                                      CentralizeByPointCloud
                                : ImportWorldOriginProvenance::
                                      CentralizeByCameras,
                    },
            };
        }

        return CentralizationResult{
            .scene_center = std::move(initial_scene_center),
            .georeference = std::nullopt,
        };
    }

} // namespace lfs::io

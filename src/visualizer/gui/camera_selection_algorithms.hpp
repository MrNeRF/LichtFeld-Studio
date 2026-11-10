/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <glm/glm.hpp>
#include <nanoflann.hpp>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <vector>

namespace lfs::vis::gui::camera_selection {

    template <class Apply>
    bool apply_if_ready(const bool cancelled, const bool failed, Apply apply) {
        if (cancelled || failed)
            return false;
        apply();
        return true;
    }

    template <class Cancelled>
    std::vector<size_t> every_n(const size_t count, const int n, Cancelled cancelled) {
        std::vector<size_t> selected;
        const auto step = static_cast<size_t>(std::max(1, n));
        for (size_t i = 0; i < count; i += step) {
            if (cancelled())
                break;
            selected.push_back(i);
        }
        return selected;
    }

    template <class Generator>
    std::vector<size_t> random(const size_t count, const float percentage, Generator& generator) {
        std::vector<size_t> selected(count);
        std::iota(selected.begin(), selected.end(), size_t{0});
        std::shuffle(selected.begin(), selected.end(), generator);
        const auto target = static_cast<size_t>(std::clamp(
            std::floor(static_cast<double>(std::clamp(percentage, 0.0F, 100.0F)) * count / 100.0 + 0.5),
            0.0, static_cast<double>(count)));
        selected.resize(target);
        return selected;
    }

    struct CameraSelectionGridCell {
        int x = 0, y = 0, z = 0;
        bool operator==(const CameraSelectionGridCell&) const = default;
    };
    struct CameraSelectionGridCellHash {
        size_t operator()(const CameraSelectionGridCell& cell) const noexcept {
            size_t value = std::hash<int>{}(cell.x);
            value ^= std::hash<int>{}(cell.y) + 0x9e3779b9 + (value << 6U) + (value >> 2U);
            value ^= std::hash<int>{}(cell.z) + 0x9e3779b9 + (value << 6U) + (value >> 2U);
            return value;
        }
    };
    struct CameraSelectionPointCloud {
        const float* points = nullptr;
        size_t count = 0;
        [[nodiscard]] size_t kdtree_get_point_count() const { return count; }
        [[nodiscard]] float kdtree_get_pt(const size_t index, const size_t dimension) const { return points[index * 3 + dimension]; }
        template <class BBox>
        bool kdtree_get_bbox(BBox&) const { return false; }
    };
    using CameraSelectionKDTree = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, CameraSelectionPointCloud>, CameraSelectionPointCloud, 3>;

    template <class Snapshot>
    float camera_selection_overlap(const Snapshot& first,
                                   const Snapshot& second,
                                   const float spacing) {
        const float dot = std::clamp(glm::dot(first.forward, second.forward), -1.0F, 1.0F);
        const float angular = std::max(0.0F, 1.0F - std::acos(dot) / std::max(first.fov + second.fov, 1.e-6F));
        const float distance = glm::distance(first.center, second.center);
        const float normalized = distance / std::max(1.5F * spacing, 1.e-6F);
        return 100.0F * angular / (1.0F + normalized * normalized);
    }

    template <class Cameras, class Cancelled, class Report>
    std::vector<size_t> distinct(const Cameras& cameras, const float overlap, Cancelled cancelled, Report report) {
        std::vector<size_t> selected;
        const size_t count = cameras.size();
        std::vector<float> positions;
        std::vector<size_t> pose_indices;
        positions.reserve(count * 3);
        pose_indices.reserve(count);
        for (size_t index = 0; index < count; ++index)
            if (cameras[index].pose_valid) {
                positions.insert(positions.end(), {cameras[index].center.x, cameras[index].center.y, cameras[index].center.z});
                pose_indices.push_back(index);
            }
        CameraSelectionPointCloud cloud{positions.data(), pose_indices.size()};
        CameraSelectionKDTree tree(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
        tree.buildIndex();
        std::vector<float> distances;
        for (size_t point = 0; point < pose_indices.size(); ++point) {
            if (cancelled())
                break;
            std::array<size_t, 2> indices{};
            std::array<float, 2> squared{};
            nanoflann::KNNResultSet<float> result(2);
            result.init(indices.data(), squared.data());
            tree.findNeighbors(result, positions.data() + point * 3, nanoflann::SearchParameters(0));
            if (result.size() == 2 && squared[1] > 1.e-18F)
                distances.push_back(std::sqrt(squared[1]));
            if ((point & 63U) == 0U)
                report(0.4F * static_cast<float>(point) / std::max<size_t>(pose_indices.size(), 1));
        }
        std::sort(distances.begin(), distances.end());
        const float spacing = distances.empty() ? 1.0F : std::max(distances[distances.size() / 2], 1.e-6F);
        const float threshold = std::clamp(overlap, 0.0F, 100.0F);
        if (threshold <= 0.0F) {
            if (const auto found = std::ranges::find_if(cameras, [](const auto& camera) { return camera.pose_valid; }); found != cameras.end())
                selected.push_back(static_cast<size_t>(std::distance(cameras.begin(), found)));
        } else {
            const float radius = threshold <= 0.0F ? 0.0F : 1.5F * spacing * std::sqrt(std::max(0.0F, 100.0F / threshold - 1.0F));
            const float cell_size = std::max(radius, 1.e-6F);
            std::unordered_map<CameraSelectionGridCell, std::vector<size_t>, CameraSelectionGridCellHash> grid;
            const auto cell_for = [cell_size](const glm::vec3 point) { return CameraSelectionGridCell{static_cast<int>(std::floor(point.x / cell_size)), static_cast<int>(std::floor(point.y / cell_size)), static_cast<int>(std::floor(point.z / cell_size))}; };
            const int cell_radius = radius > 0.0F ? 1 : 0;
            for (size_t index = 0; index < count; ++index) {
                if (cancelled())
                    break;
                const auto& candidate = cameras[index];
                if (!candidate.pose_valid)
                    continue;
                const auto cell = cell_for(candidate.center);
                bool keep = true;
                for (int x = -cell_radius; keep && x <= cell_radius; ++x)
                    for (int y = -cell_radius; keep && y <= cell_radius; ++y)
                        for (int z = -cell_radius; keep && z <= cell_radius; ++z)
                            if (const auto found = grid.find({cell.x + x, cell.y + y, cell.z + z}); found != grid.end())
                                for (const auto prior : found->second)
                                    if (camera_selection_overlap(candidate, cameras[prior], spacing) >= threshold) {
                                        keep = false;
                                        break;
                                    }
                if (keep) {
                    selected.push_back(index);
                    grid[cell].push_back(index);
                }
                if ((index & 63U) == 0U)
                    report(0.4F + 0.5F * static_cast<float>(index) / std::max<size_t>(count, 1));
            }
        }
        return selected;
    }

    template <class Snapshot>
    float crop_coverage(const Snapshot& camera, const std::vector<glm::vec3>& points,
                        const glm::mat4& world_to_local, const glm::vec3 box_min,
                        const glm::vec3 box_max,
                        const std::optional<bool> center_is_inside = std::nullopt,
                        const bool is_polygon_prism = false) {
        if (!camera.pose_valid || points.empty())
            return 0.0F;
        const glm::vec3 local_center = glm::vec3(world_to_local * glm::vec4(camera.center, 1.0F));
        const bool box_contains_center = glm::all(glm::greaterThanEqual(local_center, box_min)) &&
                                         glm::all(glm::lessThanEqual(local_center, box_max));
        if (center_is_inside.value_or(box_contains_center))
            return 100.0F;
        constexpr float near_z = 1.e-6F;
        std::vector<glm::vec3> view;
        view.reserve(points.size());
        for (const auto& point : points)
            view.push_back(glm::vec3(camera.world_to_camera * glm::vec4(point, 1.0F)));
        float min_x = camera.width, max_x = 0.0F, min_y = camera.height, max_y = 0.0F;
        bool projected = false;
        const auto project = [&](const glm::vec3& p) {
            const float x = camera.focal_x * p.x / p.z + camera.center_x;
            const float y = camera.focal_y * p.y / p.z + camera.center_y;
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
            projected = true;
        };
        const auto clip_edge = [&](const size_t i, const size_t j) {
            const glm::vec3& a = view[i];
            const glm::vec3& b = view[j];
            if ((a.z < near_z) == (b.z < near_z))
                return;
            const float t = (near_z - a.z) / (b.z - a.z);
            project(glm::vec3(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y), near_z));
        };
        for (const auto& p : view)
            if (p.z >= near_z)
                project(p);
        if (!is_polygon_prism && view.size() == 8) {
            for (size_t i = 0; i < 8; ++i)
                for (size_t bit : {1ULL, 2ULL, 4ULL}) {
                    const size_t j = i ^ bit;
                    if (j > i)
                        clip_edge(i, j);
                }
        } else if (is_polygon_prism && view.size() >= 6 && (view.size() % 2 == 0)) {
            const size_t n = view.size() / 2;
            for (size_t k = 0; k < n; ++k) {
                const size_t next = (k + 1) % n;
                clip_edge(2 * k, 2 * k + 1);
                clip_edge(2 * k, 2 * next);
                clip_edge(2 * k + 1, 2 * next + 1);
            }
        } else {
            for (size_t i = 0; i < view.size(); ++i)
                for (size_t j = i + 1; j < view.size(); ++j)
                    clip_edge(i, j);
        }
        if (!projected)
            return 0.0F;
        const float width = std::max(0.0F, std::min(camera.width, max_x) - std::max(0.0F, min_x));
        const float height = std::max(0.0F, std::min(camera.height, max_y) - std::max(0.0F, min_y));
        return 100.0F * width * height / std::max(camera.width * camera.height, 1.e-6F);
    }
} // namespace lfs::vis::gui::camera_selection

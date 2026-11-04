/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/scene_partitioner.hpp"
#include "core/camera.hpp"
#include "core/image_io.hpp"
#include "core/point_cloud.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <format>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>

namespace lfs::core {
namespace {
    constexpr float SPARSE_VIEW_COVERAGE = 0.02f;

    glm::mat3 tile_rotation(const glm::vec3& degrees) {
        return glm::mat3(glm::eulerAngleXYZ(glm::radians(degrees.x),
            glm::radians(degrees.y), glm::radians(degrees.z)));
    }

    bool contains(const glm::vec3& p, const glm::vec3& low, const glm::vec3& high) {
        return p.x >= low.x && p.x <= high.x && p.y >= low.y && p.y <= high.y &&
               p.z >= low.z && p.z <= high.z;
    }

    glm::vec2 project_observation_pixel(const TileCameraEvidence& view,
                                        const glm::vec3& world) {
        const float invalid = std::numeric_limits<float>::quiet_NaN();
        if (!view.pinhole || view.fx <= 0.0f || view.fy <= 0.0f)
            return {invalid, invalid};
        const glm::vec3 delta = world - view.center;
        const auto& m = view.world_to_camera;
        const glm::vec3 camera{m[0]*delta.x + m[1]*delta.y + m[2]*delta.z,
                               m[4]*delta.x + m[5]*delta.y + m[6]*delta.z,
                               m[8]*delta.x + m[9]*delta.y + m[10]*delta.z};
        if (!std::isfinite(camera.z) || camera.z <= 1e-4f)
            return {invalid, invalid};
        return {view.fx * camera.x / camera.z + view.cx,
                view.fy * camera.y / camera.z + view.cy};
    }

    float projected_coverage(const TileCameraEvidence& view, const glm::vec3& center,
                             const glm::mat3& inverse_rotation, const glm::vec3& half) {
        if (!view.pinhole || view.width <= 0 || view.height <= 0 ||
            !std::isfinite(view.fx) || !std::isfinite(view.fy) ||
            !std::isfinite(view.cx) || !std::isfinite(view.cy) ||
            view.fx <= 0.0f || view.fy <= 0.0f) return 0.0f;
        const glm::vec3 origin = inverse_rotation * (view.center - center);
        if (contains(origin, -half, half)) return 1.0f;
        constexpr float near_plane = 1e-4f;
        std::array<glm::vec3, 8> camera_corners;
        const auto& m = view.world_to_camera;
        const glm::mat3 rotation = glm::transpose(inverse_rotation);
        for (int i = 0; i < 8; ++i) {
            const glm::vec3 offset{(i & 1) ? half.x : -half.x,
                                   (i & 2) ? half.y : -half.y,
                                   (i & 4) ? half.z : -half.z};
            const glm::vec3 world = center + rotation * offset - view.center;
            camera_corners[i] = {m[0]*world.x + m[1]*world.y + m[2]*world.z,
                                 m[4]*world.x + m[5]*world.y + m[6]*world.z,
                                 m[8]*world.x + m[9]*world.y + m[10]*world.z};
        }
        std::array<glm::vec2, 20> visible_pixels;
        size_t visible_count = 0;
        const auto append_projection = [&](const glm::vec3& p) {
            const glm::vec2 pixel{view.fx * p.x / p.z + view.cx,
                                  view.fy * p.y / p.z + view.cy};
            if (std::isfinite(pixel.x) && std::isfinite(pixel.y))
                visible_pixels[visible_count++] = pixel;
        };
        for (int i = 0; i < 8; ++i) {
            if (camera_corners[i].z >= near_plane)
                append_projection(camera_corners[i]);
            for (int axis = 0; axis < 3; ++axis) {
                const int other = i ^ (1 << axis);
                if (other <= i) continue;
                const glm::vec3& a = camera_corners[i];
                const glm::vec3& b = camera_corners[other];
                if ((a.z < near_plane) == (b.z < near_plane)) continue;
                const float t = (near_plane - a.z) / (b.z - a.z);
                append_projection(a + t * (b - a));
            }
        }
        if (visible_count < 3) return 0.0f;
        glm::vec2 projected_min = visible_pixels[0], projected_max = visible_pixels[0];
        for (size_t i = 1; i < visible_count; ++i) {
            projected_min = glm::min(projected_min, visible_pixels[i]);
            projected_max = glm::max(projected_max, visible_pixels[i]);
        }
        if (projected_max.x <= 0.0f || projected_min.x >= view.width ||
            projected_max.y <= 0.0f || projected_min.y >= view.height)
            return 0.0f;
        std::vector<glm::vec2> projected(visible_pixels.begin(),
                                         visible_pixels.begin() + visible_count);
        std::sort(projected.begin(), projected.end(), [](const glm::vec2& a, const glm::vec2& b) {
            return a.x == b.x ? a.y < b.y : a.x < b.x;
        });
        projected.erase(std::unique(projected.begin(), projected.end(),
                                    [](const glm::vec2& a, const glm::vec2& b) {
                                        return a.x == b.x && a.y == b.y;
                                    }), projected.end());
        const auto cross = [](const glm::vec2& a, const glm::vec2& b, const glm::vec2& c) {
            return static_cast<double>(b.x - a.x) * (c.y - a.y) -
                   static_cast<double>(b.y - a.y) * (c.x - a.x);
        };
        std::vector<glm::vec2> hull;
        hull.reserve(projected.size() * 2);
        for (const auto& p : projected) {
            while (hull.size() >= 2 && cross(hull[hull.size()-2], hull.back(), p) <= 0.0)
                hull.pop_back();
            hull.push_back(p);
        }
        const size_t lower_size = hull.size();
        for (auto it = projected.rbegin() + 1; it != projected.rend(); ++it) {
            while (hull.size() > lower_size && cross(hull[hull.size()-2], hull.back(), *it) <= 0.0)
                hull.pop_back();
            hull.push_back(*it);
        }
        if (hull.size() < 4) return 0.0f;
        hull.pop_back(); // The closing vertex duplicates the first one.
        const auto clip = [&](auto inside, auto intersection) {
            std::vector<glm::vec2> output;
            output.reserve(hull.size() + 4);
            for (size_t i = 0; i < hull.size(); ++i) {
                const auto& a = hull[i];
                const auto& b = hull[(i + 1) % hull.size()];
                const bool a_inside = inside(a), b_inside = inside(b);
                if (a_inside != b_inside) output.push_back(intersection(a, b));
                if (b_inside) output.push_back(b);
            }
            hull = std::move(output);
        };
        const auto x_intersection = [](const glm::vec2& a, const glm::vec2& b, const float x) {
            return glm::vec2{x, a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x)};
        };
        const auto y_intersection = [](const glm::vec2& a, const glm::vec2& b, const float y) {
            return glm::vec2{a.x + (b.x - a.x) * (y - a.y) / (b.y - a.y), y};
        };
        clip([](const glm::vec2& p) { return p.x >= 0.0f; },
             [&](const glm::vec2& a, const glm::vec2& b) { return x_intersection(a, b, 0.0f); });
        clip([&](const glm::vec2& p) { return p.x <= view.width; },
             [&](const glm::vec2& a, const glm::vec2& b) {
                 return x_intersection(a, b, static_cast<float>(view.width));
             });
        clip([](const glm::vec2& p) { return p.y >= 0.0f; },
             [&](const glm::vec2& a, const glm::vec2& b) { return y_intersection(a, b, 0.0f); });
        clip([&](const glm::vec2& p) { return p.y <= view.height; },
             [&](const glm::vec2& a, const glm::vec2& b) {
                 return y_intersection(a, b, static_cast<float>(view.height));
             });
        if (hull.size() < 3) return 0.0f;
        double twice_area = 0.0;
        for (size_t i = 0; i < hull.size(); ++i) {
            const auto& a = hull[i];
            const auto& b = hull[(i + 1) % hull.size()];
            twice_area += static_cast<double>(a.x) * b.y - static_cast<double>(a.y) * b.x;
        }
        const double image_area = static_cast<double>(view.width) * view.height;
        return static_cast<float>(std::clamp(std::abs(twice_area) / (2.0 * image_area), 0.0, 1.0));
    }

    struct Rect { glm::vec2 low, high; };

    struct PlanningCamera {
        const TileCameraEvidence* view;
        glm::vec3 local_center;
        std::vector<glm::vec3> local_observations;
        std::vector<glm::vec2> observation_pixels;
        double effective_pixels = 0.0;
    };

    struct PlanContext {
        const SceneBounds& bounds;
        const SceneBounds& outer_bounds;
        const std::vector<glm::vec2>& samples;
        const std::vector<PlanningCamera>& cameras;
        const std::vector<TileCameraEvidence>& selection_cameras;
        glm::mat3 rotation;
        glm::mat3 inverse_rotation;
        float overlap_ratio;
        bool all_cameras_for_training;
    };

    struct PlanEstimate {
        TilePlanDiagnostics diagnostics;
        std::vector<int> supporting_uids;
    };

    TileSpec tile_from_rect(const Rect& rect, const PlanContext& context,
                            const int index) {
        glm::vec2 low = rect.low, high = rect.high;
        const auto& bounds = context.bounds;
        const auto& outer = context.outer_bounds;
        for (int axis = 0; axis < 2; ++axis) {
            const float width = high[axis] - low[axis];
            const float margin = width * context.overlap_ratio * 0.5f;
            low[axis] -= rect.low[axis] <= bounds.min[axis] + 1e-5f
                ? std::min(std::max(bounds.min[axis] - outer.min[axis], width * 0.1f), width * 0.5f)
                : margin;
            high[axis] += rect.high[axis] >= bounds.max[axis] - 1e-5f
                ? std::min(std::max(outer.max[axis] - bounds.max[axis], width * 0.1f), width * 0.5f)
                : margin;
        }
        const glm::vec3 extent = bounds.max - bounds.min;
        const glm::vec3 local_center{(low.x + high.x) * 0.5f,
                                     (low.y + high.y) * 0.5f,
                                     (bounds.min.z + bounds.max.z) * 0.5f};
        const glm::vec3 half{(high.x - low.x) * 0.5f,
                             (high.y - low.y) * 0.5f, extent.z * 0.5f};
        const glm::vec3 center = context.rotation * local_center;
        return {center - half, center + half, bounds.rotation_degrees,
                index, 0, std::format("tile_{}_0", index)};
    }

    int spatial_cell(const glm::vec2& point, const Rect& rect) {
        constexpr int side = 4;
        const glm::vec2 extent = rect.high - rect.low;
        const int x = std::clamp(static_cast<int>(side * (point.x - rect.low.x) / extent.x), 0, side - 1);
        const int y = std::clamp(static_cast<int>(side * (point.y - rect.low.y) / extent.y), 0, side - 1);
        return y * side + x;
    }

    PlanEstimate estimate_rect(const Rect& rect, const PlanContext& context) {
        PlanEstimate result;
        std::array<bool, 16> occupied{};
        std::array<int, 16> observed_by_cameras{};
        const auto tile = tile_from_rect(rect, context, 0);
        const glm::vec3 center = (tile.min + tile.max) * 0.5f;
        const glm::vec3 half = (tile.max - tile.min) * 0.5f;
        const glm::vec3 local_center = context.inverse_rotation * center;
        TileCameraSelectionDiagnostics camera_diagnostics;
        const auto selected_uids = select_tile_camera_uids(
            tile, context.selection_cameras, &camera_diagnostics);
        result.supporting_uids = camera_diagnostics.supported_uids;
        result.diagnostics.supporting_cameras =
            camera_diagnostics.track_supported_views + camera_diagnostics.airspace_supported_views;
        result.diagnostics.estimated_selected_cameras = context.all_cameras_for_training
            ? static_cast<int>(context.cameras.size()) : static_cast<int>(selected_uids.size());
        const float crop_epsilon = std::max(rect.high.x - rect.low.x,
                                            rect.high.y - rect.low.y) * 1e-5f;
        for (const auto& point : context.samples) {
            if (point.x >= local_center.x - half.x - crop_epsilon &&
                point.x <= local_center.x + half.x + crop_epsilon &&
                point.y >= local_center.y - half.y - crop_epsilon &&
                point.y <= local_center.y + half.y + crop_epsilon)
                ++result.diagnostics.sampled_crop_points;
            if (point.x >= rect.low.x &&
                (point.x < rect.high.x || rect.high.x == context.bounds.max.x) &&
                point.y >= rect.low.y &&
                (point.y < rect.high.y || rect.high.y == context.bounds.max.y)) {
                ++result.diagnostics.sampled_points;
                occupied[spatial_cell(point, rect)] = true;
            }
        }
        double selected_pixels = 0.0;
        double all_pixels = 0.0;
        int valid_images = 0;
        int valid_selected_images = 0;
        std::vector<std::array<bool, 16>> camera_cells(context.cameras.size());
        constexpr int image_grid_side = 16;
        std::vector<std::array<std::uint64_t, 4>> image_masks(
            context.cameras.size(), std::array<std::uint64_t, 4>{});
        std::vector<int> image_observation_counts(context.cameras.size(), 0);
        for (size_t view_index = 0; view_index < context.cameras.size(); ++view_index) {
            const auto& planning_camera = context.cameras[view_index];
            const auto& view = *planning_camera.view;
            if (planning_camera.effective_pixels > 0.0) {
                all_pixels += planning_camera.effective_pixels;
                ++valid_images;
            }
            std::array<bool, 16> seen_cells{};
            // Planning uses bounded SfM samples; the final allowlist uses full tracks.
            for (size_t observation_index = 0;
                 observation_index < planning_camera.local_observations.size();
                 ++observation_index) {
                const glm::vec3& point = planning_camera.local_observations[observation_index];
                if (point.x >= rect.low.x &&
                    (point.x < rect.high.x || rect.high.x == context.bounds.max.x) &&
                    point.y >= rect.low.y &&
                    (point.y < rect.high.y || rect.high.y == context.bounds.max.y) &&
                    point.z >= context.bounds.min.z && point.z <= context.bounds.max.z)
                    seen_cells[spatial_cell({point.x, point.y}, rect)] = true;
                if (contains(point, local_center - half, local_center + half)) {
                    const glm::vec2 pixel = planning_camera.observation_pixels[observation_index];
                    if (view.width > 0 && view.height > 0 &&
                        std::isfinite(pixel.x) && std::isfinite(pixel.y) &&
                        pixel.x >= 0.0f && pixel.x < view.width &&
                        pixel.y >= 0.0f && pixel.y < view.height) {
                        const int x = std::clamp(static_cast<int>(image_grid_side * pixel.x / view.width),
                                                 0, image_grid_side - 1);
                        const int y = std::clamp(static_cast<int>(image_grid_side * pixel.y / view.height),
                                                 0, image_grid_side - 1);
                        const int cell = y * image_grid_side + x;
                        image_masks[view_index][cell / 64] |= std::uint64_t{1} << (cell % 64);
                        ++image_observation_counts[view_index];
                    }
                }
            }
            camera_cells[view_index] = seen_cells;
        }
        for (size_t i = 0; i < context.cameras.size(); ++i) {
            if (!context.all_cameras_for_training &&
                !std::binary_search(selected_uids.begin(), selected_uids.end(),
                                    context.cameras[i].view->uid)) continue;
            if (image_observation_counts[i] >= 3) {
                ++result.diagnostics.image_evidence_views;
                for (const std::uint64_t word : image_masks[i])
                    result.diagnostics.observed_image_cells += std::popcount(word);
            }
            for (int cell = 0; cell < 16; ++cell)
                observed_by_cameras[cell] += camera_cells[i][cell];
            if (context.cameras[i].effective_pixels > 0.0) {
                selected_pixels += context.cameras[i].effective_pixels;
                ++valid_selected_images;
            }
        }
        for (int cell = 0; cell < 16; ++cell) {
            if (!occupied[cell]) continue;
            ++result.diagnostics.occupied_cells;
            result.diagnostics.covered_cells += observed_by_cameras[cell] >= 2;
        }
        const double mean_pixels = valid_selected_images > 0
            ? selected_pixels / valid_selected_images
            : valid_images > 0 ? all_pixels / valid_images : 1'000'000.0;
        result.diagnostics.mean_image_megapixels = static_cast<float>(mean_pixels / 1'000'000.0);
        if (result.diagnostics.image_evidence_views > 0)
            result.diagnostics.mean_image_coverage = static_cast<float>(
                result.diagnostics.observed_image_cells /
                (256.0 * result.diagnostics.image_evidence_views));
        // This is a relative planning proxy, not a runtime or VRAM prediction.
        const double point_floor = std::max<size_t>(1, context.samples.size()) * 0.05;
        const size_t estimated_points = context.all_cameras_for_training
            ? context.samples.size() : result.diagnostics.sampled_crop_points;
        result.diagnostics.estimated_work_units =
            (estimated_points + point_floor) *
            result.diagnostics.mean_image_megapixels;
        return result;
    }

    template <class Func>
    void each_observation(const TileCameraEvidence& view, Func&& visit) {
        if (view.source) {
            for (const auto& p : view.source->sfm_observations())
                visit(glm::vec3{p.x, p.y, p.z});
        } else {
            for (const auto& p : view.observations) visit(p);
        }
    }

    struct PointKey {
        std::uint32_t x, y, z;
        bool operator==(const PointKey&) const = default;
    };

    PointKey point_key(const glm::vec3& point) {
        const auto bits = [](float value) {
            return std::bit_cast<std::uint32_t>(value == 0.0f ? 0.0f : value);
        };
        return {bits(point.x), bits(point.y), bits(point.z)};
    }

    struct PointKeyHash {
        size_t operator()(const PointKey& key) const noexcept {
            size_t hash = key.x;
            hash ^= static_cast<size_t>(key.y) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            hash ^= static_cast<size_t>(key.z) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            return hash;
        }
    };
}

std::vector<TileCameraEvidence> collect_tile_camera_evidence(
    const std::vector<std::shared_ptr<Camera>>& cameras) {
    std::vector<TileCameraEvidence> result;
    result.reserve(cameras.size());
    for (const auto& camera : cameras) {
        if (!camera || !camera->has_image()) continue;
        TileCameraEvidence view;
        view.uid = camera->uid();
        view.image_name = camera->image_name();
        view.source = camera;
        const auto center = camera->cam_position().cpu().contiguous();
        const auto matrix = camera->world_view_transform().cpu().contiguous();
        if (center.numel() < 3 || matrix.numel() < 16) continue;
        const auto* cp = center.ptr<float>();
        view.center = {cp[0], cp[1], cp[2]};
        if (!std::isfinite(view.center.x) || !std::isfinite(view.center.y) ||
            !std::isfinite(view.center.z)) continue;
        std::copy_n(matrix.ptr<float>(), 16, view.world_to_camera.begin());
        view.fx = camera->focal_x(); view.fy = camera->focal_y();
        view.cx = camera->center_x(); view.cy = camera->center_y();
        view.width = camera->camera_width(); view.height = camera->camera_height();
        view.pinhole = camera->camera_model_type() == CameraModelType::PINHOLE &&
                       camera->radial_distortion().numel() == 0 &&
                       camera->tangential_distortion().numel() == 0;
        const auto& observations = camera->sfm_observations();
        const size_t step = std::max<size_t>(1, (observations.size() + 2047) / 2048);
        view.observations.reserve(std::min<size_t>(observations.size(), 2048));
        view.observation_pixels.reserve(std::min<size_t>(observations.size(), 2048));
        for (size_t i = 0; i < observations.size(); i += step) {
            const auto& p = observations[i];
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
                view.observations.emplace_back(p.x, p.y, p.z);
                view.observation_pixels.emplace_back(p.u, p.v);
            }
        }
        result.push_back(std::move(view));
    }
    return result;
}

std::vector<TileSpec> generate_adaptive_tiles(
    const SceneBounds& bounds, int count, float overlap_ratio,
    const SceneBounds& outer_bounds, const PointCloud& points,
    const std::vector<TileCameraEvidence>& cameras,
    std::vector<TilePlanDiagnostics>* diagnostics,
    const AdaptiveTileOptions options) {
    if (count < 1 || count > 100 || !std::isfinite(overlap_ratio) ||
        overlap_ratio < 0.0f || overlap_ratio > 0.3f)
        throw std::invalid_argument("Adaptive tiling needs 1 to 100 tiles and overlap from 0 to 0.3");
    const glm::vec3 extent = bounds.max - bounds.min;
    if (extent.x <= 0.0f || extent.y <= 0.0f || extent.z <= 0.0f ||
        !std::isfinite(extent.x) || !std::isfinite(extent.y) || !std::isfinite(extent.z) ||
        outer_bounds.rotation_degrees.x != bounds.rotation_degrees.x ||
        outer_bounds.rotation_degrees.y != bounds.rotation_degrees.y ||
        outer_bounds.rotation_degrees.z != bounds.rotation_degrees.z)
        throw std::invalid_argument("Adaptive tiling needs valid, aligned bounds");
    for (int axis = 0; axis < 3; ++axis)
        if (!std::isfinite(bounds.min[axis]) || !std::isfinite(bounds.max[axis]) ||
            !std::isfinite(outer_bounds.min[axis]) || !std::isfinite(outer_bounds.max[axis]) ||
            outer_bounds.min[axis] > outer_bounds.max[axis])
            throw std::invalid_argument("Adaptive tiling needs finite scene bounds");
    const glm::mat3 R = tile_rotation(bounds.rotation_degrees);
    const glm::mat3 inv = glm::transpose(R);
    std::vector<glm::vec2> samples;
    const auto has_spatial_support = [&](const std::vector<glm::vec2>& values) {
        if (values.size() < 2) return false;
        glm::vec2 low = values.front(), high = values.front();
        for (const auto& value : values) {
            low = glm::min(low, value);
            high = glm::max(high, value);
        }
        return high.x - low.x > extent.x * 1e-6f ||
               high.y - low.y > extent.y * 1e-6f;
    };
    if (points.means.is_valid() && points.means.ndim() == 2 && points.means.size(1) >= 3) {
        const auto means = points.means.cpu().contiguous();
        const auto* values = means.ptr<float>();
        const size_t stride = means.size(1);
        const size_t step = std::max<size_t>(1, (means.size(0) + 9999) / 10000);
        for (size_t i = 0; i < means.size(0); i += step) {
            const glm::vec3 p{values[i*stride], values[i*stride+1], values[i*stride+2]};
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            const auto local = inv * p;
            if (local.x >= bounds.min.x && local.x <= bounds.max.x &&
                local.y >= bounds.min.y && local.y <= bounds.max.y &&
                local.z >= bounds.min.z && local.z <= bounds.max.z)
                samples.emplace_back(local.x, local.y);
        }
    }
    // Sparse point density estimates content complexity. Tracks are a fallback
    // when a dataset has no point cloud; repeated observations must not skew it.
    if (!has_spatial_support(samples)) {
        samples.clear();
        for (const auto& view : cameras) {
            const size_t step = std::max<size_t>(1, (view.observations.size() + 63) / 64);
            for (size_t i = 0; i < view.observations.size(); i += step) {
                const auto p = inv * view.observations[i];
                if (p.x >= bounds.min.x && p.x <= bounds.max.x &&
                    p.y >= bounds.min.y && p.y <= bounds.max.y &&
                    p.z >= bounds.min.z && p.z <= bounds.max.z)
                    samples.emplace_back(p.x, p.y);
            }
        }
        std::sort(samples.begin(), samples.end(), [](const glm::vec2& a, const glm::vec2& b) {
            return a.x == b.x ? a.y < b.y : a.x < b.x;
        });
        samples.erase(std::unique(samples.begin(), samples.end(),
                                  [](const glm::vec2& a, const glm::vec2& b) {
                                      return a.x == b.x && a.y == b.y;
                                  }), samples.end());
    }
    if (!has_spatial_support(samples)) {
        samples.clear();
        for (const auto& view : cameras) {
            const auto p = inv * view.center;
            if (std::isfinite(p.x) && std::isfinite(p.y) &&
                p.x >= bounds.min.x && p.x <= bounds.max.x &&
                p.y >= bounds.min.y && p.y <= bounds.max.y)
                samples.emplace_back(p.x, p.y);
        }
    }
    std::vector<PlanningCamera> planning_cameras;
    planning_cameras.reserve(cameras.size());
    std::vector<TileCameraEvidence> selection_cameras;
    selection_cameras.reserve(cameras.size());
    for (const auto& view : cameras) {
        PlanningCamera camera{};
        camera.view = &view;
        camera.local_center = inv * view.center;
        TileCameraEvidence sampled_view;
        sampled_view.uid = view.uid;
        sampled_view.center = view.center;
        sampled_view.world_to_camera = view.world_to_camera;
        sampled_view.fx = view.fx; sampled_view.fy = view.fy;
        sampled_view.cx = view.cx; sampled_view.cy = view.cy;
        sampled_view.width = view.width; sampled_view.height = view.height;
        sampled_view.pinhole = view.pinhole;
        if (view.width > 0 && view.height > 0) {
            const auto [width, height] = resized_image_dimensions(
                view.width, view.height, options.resize_factor, options.max_width);
            camera.effective_pixels = static_cast<double>(width) * height;
        }
        const size_t step = std::max<size_t>(1, (view.observations.size() + 127) / 128);
        camera.local_observations.reserve(std::min<size_t>(view.observations.size(), 128));
        camera.observation_pixels.reserve(std::min<size_t>(view.observations.size(), 128));
        for (size_t i = 0; i < view.observations.size(); i += step) {
            const glm::vec3 local = inv * view.observations[i];
            if (std::isfinite(local.x) && std::isfinite(local.y) && std::isfinite(local.z)) {
                camera.local_observations.push_back(local);
                const glm::vec2 pixel = i < view.observation_pixels.size()
                    ? view.observation_pixels[i]
                    : project_observation_pixel(view, view.observations[i]);
                camera.observation_pixels.push_back(pixel);
                sampled_view.observations.push_back(view.observations[i]);
                sampled_view.observation_pixels.push_back(pixel);
            }
        }
        planning_cameras.push_back(std::move(camera));
        selection_cameras.push_back(std::move(sampled_view));
    }
    const PlanContext context{bounds, outer_bounds, samples, planning_cameras, selection_cameras, R, inv,
                              overlap_ratio, options.all_cameras_for_training};
    std::vector<Rect> leaves{{{bounds.min.x, bounds.min.y}, {bounds.max.x, bounds.max.y}}};
    std::vector<PlanEstimate> estimates{estimate_rect(leaves.front(), context)};
    std::vector<bool> unsplittable{false};
    while (static_cast<int>(leaves.size()) < count) {
        size_t leaf_index = leaves.size();
        double highest_work = -1.0;
        for (size_t i = 0; i < leaves.size(); ++i) {
            const double priority = options.all_cameras_for_training
                ? static_cast<double>(estimates[i].diagnostics.sampled_points)
                : estimates[i].diagnostics.estimated_work_units;
            if (!unsplittable[i] && priority > highest_work) {
                highest_work = priority;
                leaf_index = i;
            }
        }
        if (leaf_index == leaves.size())
            throw std::runtime_error("Requested tile count exceeds scene support");

        const Rect parent = leaves[leaf_index];
        const glm::vec2 width = parent.high - parent.low;
        int best_axis = -1, best_tier = -1, best_shape_tier = -1, best_shared = -1;
        size_t best_imbalance = std::numeric_limits<size_t>::max();
        float best_cut = 0.0f;
        double best_score = std::numeric_limits<double>::infinity();
        PlanEstimate best_left, best_right;
        const double parent_aspect = std::max(width.x / width.y, width.y / width.x);
        const double acceptable_aspect = std::max(2.5, parent_aspect * 1.25);
        for (int axis = 0; axis < 2; ++axis) {
            if (width[axis] <= extent[axis] * 0.01f) continue;
            std::vector<float> point_positions, camera_positions;
            for (const auto& point : samples)
                if (point.x >= parent.low.x &&
                    (point.x < parent.high.x || parent.high.x == bounds.max.x) &&
                    point.y >= parent.low.y &&
                    (point.y < parent.high.y || parent.high.y == bounds.max.y))
                    point_positions.push_back(point[axis]);
            if (point_positions.size() < 2) continue;
            for (const auto& camera : planning_cameras) {
                const glm::vec3 position = camera.local_center;
                if (position.x >= parent.low.x &&
                    (position.x < parent.high.x || parent.high.x == bounds.max.x) &&
                    position.y >= parent.low.y &&
                    (position.y < parent.high.y || parent.high.y == bounds.max.y))
                    camera_positions.push_back(position[axis]);
            }
            std::vector<float> cuts{parent.low[axis] + width[axis] * 0.5f};
            const auto add_quantiles = [&](std::vector<float>& positions) {
                if (positions.size() < 4) return;
                std::sort(positions.begin(), positions.end());
                for (const float q : {0.25f, 0.5f, 0.75f})
                    cuts.push_back(positions[static_cast<size_t>(q * (positions.size() - 1))]);
            };
            add_quantiles(point_positions);
            add_quantiles(camera_positions);
            std::sort(cuts.begin(), cuts.end());
            cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
            for (const float cut : cuts) {
                if (cut - parent.low[axis] < width[axis] * 0.05f ||
                    parent.high[axis] - cut < width[axis] * 0.05f)
                    continue;
                Rect left = parent, right = parent;
                left.high[axis] = cut;
                right.low[axis] = cut;
                auto left_estimate = estimate_rect(left, context);
                auto right_estimate = estimate_rect(right, context);
                if (left_estimate.diagnostics.sampled_points == 0 ||
                    right_estimate.diagnostics.sampled_points == 0)
                    continue;
                std::vector<int> shared;
                std::set_intersection(left_estimate.supporting_uids.begin(),
                                      left_estimate.supporting_uids.end(),
                                      right_estimate.supporting_uids.begin(),
                                      right_estimate.supporting_uids.end(),
                                      std::back_inserter(shared));
                const int shared_count = static_cast<int>(shared.size());
                const int min_support = std::min(left_estimate.diagnostics.supporting_cameras,
                                                 right_estimate.diagnostics.supporting_cameras);
                const auto coverage = [](const PlanEstimate& estimate) {
                    const auto& d = estimate.diagnostics;
                    return d.occupied_cells > 0
                        ? static_cast<float>(d.covered_cells) / d.occupied_cells : 0.0f;
                };
                const float min_coverage = std::min(coverage(left_estimate),
                                                    coverage(right_estimate));
                const auto& left_diag = left_estimate.diagnostics;
                const auto& right_diag = right_estimate.diagnostics;
                const bool has_image_evidence =
                    estimates[leaf_index].diagnostics.image_evidence_views > 0;
                const float min_image_coverage = std::min(left_diag.mean_image_coverage,
                                                          right_diag.mean_image_coverage);
                const int tier = min_support >= 2 &&
                                 (shared_count >= 1 || options.all_cameras_for_training) ? 1 : 0;
                const auto aspect = [](const Rect& rect) {
                    const glm::vec2 size = rect.high - rect.low;
                    return std::max(static_cast<double>(size.x / size.y),
                                    static_cast<double>(size.y / size.x));
                };
                const double child_aspect = std::max(aspect(left), aspect(right));
                // A thin strip can look cheap when the view/point proxy is tied.
                // Prefer cuts that do not worsen the parent's shape substantially;
                // if every supported cut is elongated, still allow the best one.
                const int shape_tier = child_aspect <= acceptable_aspect ? 1 : 0;
                const double work = std::max(left_estimate.diagnostics.estimated_work_units,
                                             right_estimate.diagnostics.estimated_work_units);
                // Compact cells reduce overlap and avoid full-scene slivers.
                const double shape_penalty = 1.0 + 0.2 *
                    std::max(0.0, child_aspect - parent_aspect);
                // Sparse image-cell occupancy is a view-space support proxy,
                // rather than a dense visibility mask or a pixel-loss estimate.
                const double image_retention = has_image_evidence
                    ? std::clamp(static_cast<double>(min_image_coverage) /
                        estimates[leaf_index].diagnostics.mean_image_coverage, 0.0, 1.0)
                    : 1.0;
                const double image_penalty = 1.0 + 0.5 * (1.0 - image_retention);
                const double score = work * (1.0 + 0.5 * (1.0 - min_coverage)) *
                                     image_penalty * shape_penalty;
                const size_t left_points = left_estimate.diagnostics.sampled_points;
                const size_t right_points = right_estimate.diagnostics.sampled_points;
                const size_t imbalance = left_points > right_points
                    ? left_points - right_points : right_points - left_points;
                if (tier > best_tier ||
                    (tier == best_tier && shape_tier > best_shape_tier) ||
                    (tier == best_tier && shape_tier == best_shape_tier &&
                     score < best_score - 1e-6) ||
                    (tier == best_tier && shape_tier == best_shape_tier &&
                     std::abs(score - best_score) <= 1e-6 &&
                     (imbalance < best_imbalance ||
                      (imbalance == best_imbalance && shared_count > best_shared)))) {
                    best_axis = axis;
                    best_cut = cut;
                    best_tier = tier;
                    best_shape_tier = shape_tier;
                    best_score = score;
                    best_shared = shared_count;
                    best_imbalance = imbalance;
                    best_left = std::move(left_estimate);
                    best_right = std::move(right_estimate);
                }
            }
        }
        if (best_axis < 0) {
            unsplittable[leaf_index] = true;
            continue;
        }
        Rect right = parent;
        leaves[leaf_index].high[best_axis] = best_cut;
        right.low[best_axis] = best_cut;
        leaves.push_back(right);
        estimates[leaf_index] = std::move(best_left);
        estimates.push_back(std::move(best_right));
        unsplittable[leaf_index] = false;
        unsplittable.push_back(false);
    }
    std::vector<size_t> order(leaves.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](const size_t a, const size_t b) {
        return leaves[a].low.y == leaves[b].low.y
            ? leaves[a].low.x < leaves[b].low.x
            : leaves[a].low.y < leaves[b].low.y;
    });
    std::vector<TileSpec> tiles;
    tiles.reserve(leaves.size());
    if (diagnostics) {
        diagnostics->clear();
        diagnostics->reserve(order.size());
    }
    for (size_t i = 0; i < order.size(); ++i) {
        tiles.push_back(tile_from_rect(leaves[order[i]], context, static_cast<int>(i)));
        if (diagnostics) diagnostics->push_back(estimates[order[i]].diagnostics);
    }
    return tiles;
}

std::vector<int> select_tile_camera_uids(
    const TileSpec& tile, const std::vector<TileCameraEvidence>& cameras,
    TileCameraSelectionDiagnostics* diagnostics) {
    if (diagnostics) *diagnostics = {};
    if (cameras.empty()) return {};
    const glm::mat3 inverse = glm::transpose(tile_rotation(tile.rotation_degrees));
    const glm::vec3 center = (tile.min + tile.max) * 0.5f;
    const glm::vec3 half = (tile.max - tile.min) * 0.5f;
    if (half.x <= 0.0f || half.y <= 0.0f || half.z <= 0.0f ||
        !std::isfinite(half.x) || !std::isfinite(half.y) || !std::isfinite(half.z))
        throw std::invalid_argument("Camera selection needs a non-empty tile");

    constexpr int image_side = 16;
    constexpr int spatial_cells = 32; // 4 x 4 x 2 in the tile's local frame
    struct ViewSupport {
        int uid = -1;
        int inside_tracks = 0;
        int image_cells = 0;
        std::bitset<spatial_cells> cells;
        float projected_area = 0.0f;
        float distance = 0.0f;
        double score = 0.0;
    };
    std::vector<ViewSupport> support;
    support.reserve(cameras.size());
    int best_tracks = 0, best_image_cells = 0;
    std::array<int, spatial_cells> available{};
    for (const auto& view : cameras) {
        ViewSupport current;
        current.uid = view.uid;
        current.distance = glm::length(view.center - center);
        if (!std::isfinite(current.distance))
            current.distance = std::numeric_limits<float>::infinity();
        current.projected_area = projected_coverage(view, center, inverse, half);
        std::array<int, spatial_cells> tracks_per_cell{};
        std::bitset<image_side * image_side> image_cells;
        const auto visit = [&](const glm::vec3& world, glm::vec2 pixel) {
            if (!std::isfinite(world.x) || !std::isfinite(world.y) || !std::isfinite(world.z))
                return;
            const glm::vec3 local = inverse * (world - center);
            if (!contains(local, -half, half)) return;
            ++current.inside_tracks;
            const glm::vec3 unit = (local + half) / (2.0f * half);
            const int x = std::clamp(static_cast<int>(unit.x * 4.0f), 0, 3);
            const int y = std::clamp(static_cast<int>(unit.y * 4.0f), 0, 3);
            const int z = std::clamp(static_cast<int>(unit.z * 2.0f), 0, 1);
            ++tracks_per_cell[(z * 4 + y) * 4 + x];
            if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y))
                pixel = project_observation_pixel(view, world);
            if (view.width > 0 && view.height > 0 &&
                std::isfinite(pixel.x) && std::isfinite(pixel.y) &&
                pixel.x >= 0.0f && pixel.x < view.width &&
                pixel.y >= 0.0f && pixel.y < view.height) {
                const int ix = std::clamp(static_cast<int>(image_side * pixel.x / view.width), 0, image_side - 1);
                const int iy = std::clamp(static_cast<int>(image_side * pixel.y / view.height), 0, image_side - 1);
                image_cells.set(iy * image_side + ix);
            }
        };
        if (view.source) {
            for (const auto& observation : view.source->sfm_observations())
                visit({observation.x, observation.y, observation.z},
                      {observation.u, observation.v});
        } else {
            for (size_t index = 0; index < view.observations.size(); ++index) {
                const float invalid = std::numeric_limits<float>::quiet_NaN();
                visit(view.observations[index],
                      index < view.observation_pixels.size()
                          ? view.observation_pixels[index] : glm::vec2{invalid});
            }
        }
        for (int cell = 0; cell < spatial_cells; ++cell)
            if (tracks_per_cell[cell] >= 2) {
                current.cells.set(cell);
                ++available[cell];
            }
        current.image_cells = static_cast<int>(image_cells.count());
        current.score = current.inside_tracks + 8.0 * current.image_cells +
                        16.0 * current.cells.count();
        best_tracks = std::max(best_tracks, current.inside_tracks);
        best_image_cells = std::max(best_image_cells, current.image_cells);
        support.push_back(current);
    }

    const int track_floor = std::max(6, static_cast<int>(std::ceil(best_tracks * 0.05)));
    const int image_floor = std::min(best_image_cells,
        std::max(2, static_cast<int>(std::ceil(best_image_cells * 0.10))));
    std::vector<bool> chosen(cameras.size(), false);
    std::array<int, spatial_cells> covered{};
    int evidence_views = 0, observed_cells = 0;
    size_t selected_count = 0;
    const auto add = [&](const size_t index) {
        if (chosen[index]) return;
        chosen[index] = true;
        ++selected_count;
        for (int cell = 0; cell < spatial_cells; ++cell)
            covered[cell] += support[index].cells.test(cell);
    };
    for (size_t index = 0; index < support.size(); ++index) {
        const auto& view = support[index];
        if (view.inside_tracks >= 2 && view.cells.any()) ++evidence_views;
        const bool enough_image_evidence = best_image_cells == 0 ||
            view.image_cells >= image_floor ||
            (view.image_cells == 0 && view.inside_tracks >= best_tracks * 0.25);
        const bool track_supported = view.inside_tracks >= track_floor && enough_image_evidence;
        // Project the full tile volume, including empty air above the SfM points.
        // Surface tracks alone can miss views needed to suppress floaters there.
        const bool airspace_supported = view.projected_area >= TILE_AIRSPACE_VIEW_THRESHOLD;
        if (diagnostics) {
            diagnostics->track_supported_views += track_supported;
            diagnostics->airspace_supported_views += !track_supported && airspace_supported;
            if (track_supported || airspace_supported)
                diagnostics->supported_uids.push_back(view.uid);
        }
        if (track_supported || airspace_supported)
            add(index);
    }
    for (const int count : available) observed_cells += count > 0;

    // Rescue tile regions seen by the SfM cameras but not yet covered twice.
    // This is bounded by the 32 coarse cells rather than by camera count.
    for (;;) {
        size_t best = support.size();
        int best_gain = 0;
        for (size_t index = 0; index < support.size(); ++index) {
            if (chosen[index] || support[index].inside_tracks < 2) continue;
            int gain = 0;
            for (int cell = 0; cell < spatial_cells; ++cell) {
                const int target = std::min(2, available[cell]);
                if (support[index].cells.test(cell) && covered[cell] < target)
                    gain += covered[cell] == 0 ? 2 : 1;
            }
            if (gain > best_gain ||
                (gain == best_gain && gain > 0 && best < support.size() &&
                 (support[index].score > support[best].score ||
                  (support[index].score == support[best].score &&
                   support[index].uid < support[best].uid)))) {
                best = index;
                best_gain = gain;
            }
        }
        if (best_gain == 0) break;
        add(best);
    }

    const auto ranked = [&](const size_t a, const size_t b) {
        if (support[a].score != support[b].score) return support[a].score > support[b].score;
        if (support[a].projected_area != support[b].projected_area)
            return support[a].projected_area > support[b].projected_area;
        if (support[a].distance != support[b].distance)
            return support[a].distance < support[b].distance;
        return support[a].uid < support[b].uid;
    };
    std::vector<size_t> order(support.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), ranked);
    const size_t minimum_evidence_views = std::min<size_t>(4, static_cast<size_t>(evidence_views));
    for (const size_t index : order) {
        if (selected_count >= minimum_evidence_views)
            break;
        if (support[index].inside_tracks >= 2 && support[index].cells.any()) add(index);
    }

    const bool sparse = evidence_views < 4 || observed_cells < 2 || best_image_cells < 2;
    if (sparse) {
        const size_t floor = std::min<size_t>(8, support.size());
        for (const size_t index : order) {
            if (selected_count >= floor) break;
            if (support[index].inside_tracks >= 2 ||
                support[index].projected_area >= SPARSE_VIEW_COVERAGE)
                add(index);
        }
        // The remaining floor is a last resort for datasets with no usable tracks.
        std::vector<size_t> nearest = order;
        std::sort(nearest.begin(), nearest.end(), [&](const size_t a, const size_t b) {
            return support[a].distance == support[b].distance
                ? support[a].uid < support[b].uid
                : support[a].distance < support[b].distance;
        });
        for (const size_t index : nearest) {
            if (selected_count >= floor) break;
            add(index);
        }
    }

    if (diagnostics) {
        std::sort(diagnostics->supported_uids.begin(), diagnostics->supported_uids.end());
        diagnostics->supported_uids.erase(
            std::unique(diagnostics->supported_uids.begin(), diagnostics->supported_uids.end()),
            diagnostics->supported_uids.end());
        diagnostics->evidence_views = evidence_views;
        diagnostics->observed_cells = observed_cells;
        diagnostics->sparse_fallback = sparse;
        for (int cell = 0; cell < spatial_cells; ++cell) {
            if (available[cell] == 0) continue;
            diagnostics->covered_cells += covered[cell] >= 1;
            if (available[cell] >= 2) {
                ++diagnostics->multi_view_cells;
                diagnostics->multi_view_covered_cells += covered[cell] >= 2;
            }
        }
    }
    std::vector<int> selected;
    for (size_t index = 0; index < chosen.size(); ++index)
        if (chosen[index]) selected.push_back(support[index].uid);
    std::sort(selected.begin(), selected.end());
    return selected;
}

TileVisibilityMask estimate_tile_visibility_mask(
    const TileSpec& tile, const TileCameraEvidence& camera,
    const int width, const int height) {
    TileVisibilityMask result;
    if (width <= 0 || height <= 0 || width > 256 || height > 256 ||
        camera.width <= 0 || camera.height <= 0 || !camera.pinhole)
        return result;

    const glm::mat3 inverse = glm::transpose(tile_rotation(tile.rotation_degrees));
    const glm::vec3 center = (tile.min + tile.max) * 0.5f;
    const glm::vec3 half = (tile.max - tile.min) * 0.5f;
    if (half.x <= 0.0f || half.y <= 0.0f || half.z <= 0.0f)
        return result;

    struct Sample { glm::vec2 pixel; bool inside; };
    std::vector<Sample> samples;
    samples.reserve(std::min<size_t>(camera.observations.size(), 512));
    int inside_count = 0, outside_count = 0;
    const size_t step = std::max<size_t>(1, (camera.observations.size() + 511) / 512);
    for (size_t i = 0; i < camera.observations.size(); i += step) {
        const auto& point = camera.observations[i];
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z))
            continue;
        const glm::vec2 pixel = i < camera.observation_pixels.size()
            ? camera.observation_pixels[i] : project_observation_pixel(camera, point);
        if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y) ||
            pixel.x < 0.0f || pixel.y < 0.0f ||
            pixel.x >= camera.width || pixel.y >= camera.height)
            continue;
        const bool inside = contains(inverse * (point - center), -half, half);
        samples.push_back({{pixel.x / camera.width, pixel.y / camera.height}, inside});
        inside_count += inside;
        outside_count += !inside;
    }
    // Sparse or one-sided tracks cannot establish a reliable image boundary.
    if (inside_count < 8 || outside_count < 8)
        return result;

    result.width = width;
    result.height = height;
    result.weights.resize(static_cast<size_t>(width) * height, 1.0f);
    bool has_excluded_region = false;
    constexpr float radius_squared = 0.15f * 0.15f;
    constexpr float sigma_squared = 0.07f * 0.07f;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const glm::vec2 pixel{(x + 0.5f) / width, (y + 0.5f) / height};
            float inside_support = 0.0f, outside_support = 0.0f;
            int nearby_outside = 0;
            for (const auto& sample : samples) {
                const glm::vec2 delta = pixel - sample.pixel;
                const float distance_squared = glm::dot(delta, delta);
                if (distance_squared > radius_squared) continue;
                const float support = std::exp(-0.5f * distance_squared / sigma_squared);
                if (sample.inside) inside_support += support;
                else {
                    outside_support += support;
                    ++nearby_outside;
                }
            }
            if (nearby_outside >= 2 && outside_support >= 0.8f &&
                outside_support > inside_support * 2.0f) {
                const float in_fraction = inside_support / (inside_support + outside_support);
                result.weights[static_cast<size_t>(y) * width + x] =
                    std::clamp(0.15f + 1.7f * in_fraction, 0.15f, 1.0f);
                has_excluded_region = true;
            }
        }
    }
    if (!has_excluded_region) return {};
    return result;
}

std::vector<int> select_tile_initial_point_indices(
    const PointCloud& points, const TileSpec& tile,
    const std::vector<TileCameraEvidence>& selected_cameras) {
    if (!points.means.is_valid() || points.means.ndim() != 2 || points.means.size(1) < 3)
        return {};
    const auto means = points.means.cpu().contiguous();
    const size_t count = means.size(0), stride = means.size(1);
    if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Tile point selection exceeds supported index count");
    const auto* values = means.ptr<float>();
    const glm::mat3 inverse_rotation = glm::transpose(tile_rotation(tile.rotation_degrees));
    const glm::vec3 center = (tile.min + tile.max) * 0.5f;
    const glm::vec3 half = (tile.max - tile.min) * 0.5f;

    std::unordered_set<PointKey, PointKeyHash> observed;
    for (const auto& view : selected_cameras)
        each_observation(view, [&](const glm::vec3& point) {
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
                observed.insert(point_key(point));
        });

    std::vector<int> selected;
    selected.reserve(std::min<size_t>(count, 100000));
    for (size_t index = 0; index < count; ++index) {
        const glm::vec3 point{values[index*stride], values[index*stride+1], values[index*stride+2]};
        if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) continue;
        if (contains(inverse_rotation * (point - center), -half, half) ||
            observed.contains(point_key(point)))
            selected.push_back(static_cast<int>(index));
    }
    // A mostly empty cell still needs enough initial points for a stable run.
    if (selected.size() < std::min<size_t>(128, count)) {
        std::vector<std::pair<float, int>> distances;
        distances.reserve(count);
        for (size_t index = 0; index < count; ++index) {
            const glm::vec3 point{values[index*stride], values[index*stride+1], values[index*stride+2]};
            if (std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z))
                distances.emplace_back(glm::length(point - center), static_cast<int>(index));
        }
        const size_t nearest_count = std::min<size_t>(128, distances.size());
        if (nearest_count < distances.size())
            std::nth_element(distances.begin(), distances.begin() + nearest_count, distances.end());
        for (size_t i = 0; i < nearest_count; ++i)
            selected.push_back(distances[i].second);
        std::sort(selected.begin(), selected.end());
        selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    }
    return selected;
}
} // namespace lfs::core

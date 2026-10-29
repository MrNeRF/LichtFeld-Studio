/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/scene_partitioner.hpp"
#include "core/point_cloud.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#include <glm/vec2.hpp>
#include <limits>
#include <stdexcept>

namespace lfs::core {
    namespace {
        std::vector<glm::vec3> positions(const PointCloud& pc) {
            std::vector<glm::vec3> result;
            if (!pc.means.is_valid() || pc.means.ndim() != 2 || pc.means.size(1) < 3)
                return result;
            const auto means = pc.means.cpu().contiguous();
            const auto* values = means.ptr<float>();
            const size_t stride = means.size(1);
            result.reserve(means.size(0));
            for (size_t n = 0; n < means.size(0); ++n) {
                const glm::vec3 p{values[n * stride], values[n * stride + 1], values[n * stride + 2]};
                if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
                    result.push_back(p);
            }
            return result;
        }

        float percentile(std::vector<float>& values, const float fraction) {
            const size_t trim = std::min((values.size() - 1) / 2,
                                         static_cast<size_t>(std::round(
                                             std::min(fraction, 1.0f - fraction) * values.size())));
            const size_t index = fraction <= 0.5f ? trim : values.size() - 1 - trim;
            std::nth_element(values.begin(), values.begin() + index, values.end());
            return values[index];
        }
    } // namespace

    float compute_dominant_orientation_pca(const PointCloud& pc) {
        const auto points = positions(pc);
        if (points.size() < 2)
            return 0.0f;
        std::vector<float> xs, ys;
        xs.reserve(points.size()); ys.reserve(points.size());
        for (const auto& p : points) { xs.push_back(p.x); ys.push_back(p.y); }
        const float min_x = percentile(xs, 0.02f), max_x = percentile(xs, 0.98f);
        const float min_y = percentile(ys, 0.02f), max_y = percentile(ys, 0.98f);
        double x = 0.0, y = 0.0;
        size_t count = 0;
        for (const auto& p : points) {
            if (p.x < min_x || p.x > max_x || p.y < min_y || p.y > max_y) continue;
            x += p.x; y += p.y; ++count;
        }
        if (count < 2) return 0.0f;
        x /= static_cast<double>(count);
        y /= static_cast<double>(count);
        double xx = 0.0, xy = 0.0, yy = 0.0;
        for (const auto& p : points) {
            if (p.x < min_x || p.x > max_x || p.y < min_y || p.y > max_y) continue;
            const double dx = p.x - x, dy = p.y - y;
            xx += dx * dx; xy += dx * dy; yy += dy * dy;
        }
        if (std::abs(xx - yy) + std::abs(xy) < 1e-12)
            return 0.0f;
        return static_cast<float>(0.5 * std::atan2(2.0 * xy, xx - yy) * 180.0 / 3.14159265358979323846);
    }

    SceneBounds compute_robust_scene_bounds(const PointCloud& pc,
                                             const std::vector<glm::vec3>& camera_centers,
                                             const float quantile, const float angle_degrees) {
        if (!std::isfinite(quantile) || quantile < 0.0f || quantile >= 0.5f ||
            !std::isfinite(angle_degrees))
            throw std::invalid_argument("Invalid scene quantile or orientation");
        auto points = positions(pc);
        // A one-point sparse cloud cannot describe the scene footprint. Camera
        // positions provide a useful footprint for these degenerate datasets.
        if (points.size() < 2 && !camera_centers.empty()) points = camera_centers;
        if (points.empty())
            throw std::invalid_argument("Auto-tiling needs sparse points or camera centers");

        const float angle = angle_degrees * 3.14159265358979323846f / 180.0f;
        const float c = std::cos(angle), s = std::sin(angle);
        std::vector<float> axes[3];
        for (auto& axis : axes) axis.reserve(points.size());
        for (const auto& p : points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            axes[0].push_back(c * p.x + s * p.y);
            axes[1].push_back(-s * p.x + c * p.y);
            axes[2].push_back(p.z);
        }
        if (axes[0].empty())
            throw std::invalid_argument("Auto-tiling has no finite positions");
        SceneBounds bounds;
        bounds.angle_degrees = angle_degrees;
        for (int axis = 0; axis < 3; ++axis) {
            bounds.min[axis] = percentile(axes[axis], quantile);
            bounds.max[axis] = percentile(axes[axis], 1.0f - quantile);
            if (bounds.max[axis] <= bounds.min[axis]) {
                bounds.min[axis] -= 0.5f;
                bounds.max[axis] += 0.5f;
            }
        }
        return bounds;
    }

    std::vector<TileSpec> generate_tile_grid(const SceneBounds& bounds, const TileGridConfig& config) {
        if (!std::isfinite(config.overlap_ratio) || config.overlap_ratio < 0.0f ||
            config.overlap_ratio > 0.3f || !std::isfinite(config.target_size) ||
            config.target_size < 0.0f)
            throw std::invalid_argument("Tile size or overlap is invalid");
        const glm::vec3 extent = bounds.max - bounds.min;
        if (extent.x <= 0.0f || extent.y <= 0.0f || extent.z <= 0.0f)
            throw std::invalid_argument("Scene bounds must have positive extent");
        const double required_columns = config.target_size > 0.0f
                                            ? std::ceil(static_cast<double>(extent.x) / config.target_size) : config.columns;
        const double required_rows = config.target_size > 0.0f
                                         ? std::ceil(static_cast<double>(extent.y) / config.target_size) : config.rows;
        if (!std::isfinite(required_columns) || !std::isfinite(required_rows) ||
            required_columns < 1.0 || required_rows < 1.0 ||
            required_columns > 100.0 || required_rows > 100.0 ||
            required_columns * required_rows > 1000.0)
            throw std::invalid_argument("Tile grid must contain between 1 and 1000 cells");
        const int columns = static_cast<int>(required_columns);
        const int rows = static_cast<int>(required_rows);
        if (columns < 1 || rows < 1 || columns > 100 || rows > 100 || columns * rows > 1000)
            throw std::invalid_argument("Tile grid must contain between 1 and 1000 cells");
        const glm::vec2 cell{extent.x / columns, extent.y / rows};
        const glm::vec2 margin = cell * config.overlap_ratio * 0.5f;
        const float angle = bounds.angle_degrees * 3.14159265358979323846f / 180.0f;
        const float c = std::cos(angle), s = std::sin(angle);
        std::vector<TileSpec> tiles;
        tiles.reserve(columns * rows);
        for (int j = 0; j < rows; ++j) {
            for (int i = 0; i < columns; ++i) {
                const glm::vec2 low{bounds.min.x + cell.x * i - (i ? margin.x : 0.0f),
                                    bounds.min.y + cell.y * j - (j ? margin.y : 0.0f)};
                const glm::vec2 high{bounds.min.x + cell.x * (i + 1) + (i + 1 < columns ? margin.x : 0.0f),
                                     bounds.min.y + cell.y * (j + 1) + (j + 1 < rows ? margin.y : 0.0f)};
                const glm::vec2 center = (low + high) * 0.5f;
                const glm::vec2 world{c * center.x - s * center.y, s * center.x + c * center.y};
                const glm::vec3 half{(high.x - low.x) * 0.5f, (high.y - low.y) * 0.5f,
                                     extent.z * 0.5f};
                const glm::vec3 world_center{world.x, world.y, (bounds.min.z + bounds.max.z) * 0.5f};
                tiles.push_back({world_center - half, world_center + half,
                                 {0.0f, 0.0f, bounds.angle_degrees}, i, j,
                                 std::format("tile_{}_{}", i, j)});
            }
        }
        return tiles;
    }
} // namespace lfs::core

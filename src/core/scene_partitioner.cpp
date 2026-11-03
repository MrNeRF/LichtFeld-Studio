/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/scene_partitioner.hpp"
#include "core/lod_merge_math.hpp"
#include "core/point_cloud.hpp"
#include <algorithm>
#include <cmath>
#include <format>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>
#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>
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

        std::vector<glm::vec3> bounds_positions(const PointCloud& pc,
                                                 const std::vector<glm::vec3>& camera_centers,
                                                 const BoundsSource source) {
            // Camera-only tiling should not copy the potentially large sparse cloud.
            if (source == BoundsSource::Cameras && camera_centers.size() >= 2)
                return camera_centers;
            const auto pc_points = positions(pc);
            std::vector<glm::vec3> pts;
            if (source == BoundsSource::Both) {
                pts = pc_points;
                pts.insert(pts.end(), camera_centers.begin(), camera_centers.end());
            } else if (source == BoundsSource::Points) {
                pts = pc_points;
            } else if (source == BoundsSource::Cameras) {
                pts = camera_centers;
            }
            if (pts.size() < 2) {
                if (!camera_centers.empty()) pts = camera_centers;
                else if (!pc_points.empty()) pts = pc_points;
            }
            return pts;
        }

        float percentile(std::vector<float>& values, const float fraction) {
            const size_t trim = std::min((values.size() - 1) / 2,
                                         static_cast<size_t>(std::round(
                                             std::min(fraction, 1.0f - fraction) * values.size())));
            const size_t index = fraction <= 0.5f ? trim : values.size() - 1 - trim;
            std::nth_element(values.begin(), values.begin() + index, values.end());
            return values[index];
        }

        float compute_dominant_orientation_pca_from_points(const std::vector<glm::vec3>& points,
                                                           const int first_axis, const int second_axis,
                                                           const float quantile = 0.02f) {
            if (points.size() < 2)
                return 0.0f;
            std::vector<float> xs, ys;
            xs.reserve(points.size()); ys.reserve(points.size());
            for (const auto& p : points) {
                if (std::isfinite(p[first_axis]) && std::isfinite(p[second_axis])) {
                    xs.push_back(p[first_axis]); ys.push_back(p[second_axis]);
                }
            }
            if (xs.size() < 2)
                return 0.0f;

            float min_x = -std::numeric_limits<float>::infinity(), max_x = std::numeric_limits<float>::infinity();
            float min_y = -std::numeric_limits<float>::infinity(), max_y = std::numeric_limits<float>::infinity();
            if (quantile > 0.0f) {
                min_x = percentile(xs, quantile); max_x = percentile(xs, 1.0f - quantile);
                min_y = percentile(ys, quantile); max_y = percentile(ys, 1.0f - quantile);
            }
            double x = 0.0, y = 0.0;
            size_t count = 0;
            for (const auto& p : points) {
                const float px = p[first_axis], py = p[second_axis];
                if (!std::isfinite(px) || !std::isfinite(py)) continue;
                if (quantile > 0.0f && (px < min_x || px > max_x || py < min_y || py > max_y)) continue;
                x += px; y += py; ++count;
            }
            if (count < 2) return 0.0f;
            x /= static_cast<double>(count);
            y /= static_cast<double>(count);
            double xx = 0.0, xy = 0.0, yy = 0.0;
            for (const auto& p : points) {
                const float px = p[first_axis], py = p[second_axis];
                if (!std::isfinite(px) || !std::isfinite(py)) continue;
                if (quantile > 0.0f && (px < min_x || px > max_x || py < min_y || py > max_y)) continue;
                const double dx = px - x, dy = py - y;
                xx += dx * dx; xy += dx * dy; yy += dy * dy;
            }
            if (std::abs(xx - yy) + std::abs(xy) < 1e-12)
                return 0.0f;
            return static_cast<float>(0.5 * std::atan2(2.0 * xy, xx - yy) * 180.0 / 3.14159265358979323846);
        }
    } // namespace

    float compute_dominant_orientation_pca(const PointCloud& pc) {
        return compute_dominant_orientation_pca_from_points(positions(pc), 0, 1);
    }

    glm::vec3 compute_dominant_orientation_3d(const std::vector<glm::vec3>& points, const float quantile) {
        if (points.size() < 3)
            return glm::vec3(0.0f);

        std::vector<float> xs, ys, zs;
        xs.reserve(points.size()); ys.reserve(points.size()); zs.reserve(points.size());
        for (const auto& p : points) {
            if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
                xs.push_back(p.x); ys.push_back(p.y); zs.push_back(p.z);
            }
        }
        if (xs.size() < 3)
            return glm::vec3(0.0f);

        float min_x = -std::numeric_limits<float>::infinity(), max_x = std::numeric_limits<float>::infinity();
        float min_y = -std::numeric_limits<float>::infinity(), max_y = std::numeric_limits<float>::infinity();
        float min_z = -std::numeric_limits<float>::infinity(), max_z = std::numeric_limits<float>::infinity();
        if (quantile > 0.0f) {
            min_x = percentile(xs, quantile); max_x = percentile(xs, 1.0f - quantile);
            min_y = percentile(ys, quantile); max_y = percentile(ys, 1.0f - quantile);
            min_z = percentile(zs, quantile); max_z = percentile(zs, 1.0f - quantile);
        }

        double mx = 0.0, my = 0.0, mz = 0.0;
        size_t count = 0;
        for (const auto& p : points) {
            if (quantile > 0.0f && (p.x < min_x || p.x > max_x || p.y < min_y || p.y > max_y || p.z < min_z || p.z > max_z))
                continue;
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                continue;
            mx += p.x; my += p.y; mz += p.z;
            ++count;
        }
        bool use_trimmed_points = quantile > 0.0f;
        if (count < 3) {
            use_trimmed_points = false;
            mx = 0.0; my = 0.0; mz = 0.0; count = 0;
            for (const auto& p : points) {
                if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
                    mx += p.x; my += p.y; mz += p.z;
                    ++count;
                }
            }
            if (count < 3) return glm::vec3(0.0f);
        }
        mx /= static_cast<double>(count);
        my /= static_cast<double>(count);
        mz /= static_cast<double>(count);

        double cxx = 0.0, cxy = 0.0, cxz = 0.0;
        double cyy = 0.0, cyz = 0.0, czz = 0.0;
        for (const auto& p : points) {
            if (use_trimmed_points && (p.x < min_x || p.x > max_x || p.y < min_y || p.y > max_y || p.z < min_z || p.z > max_z))
                continue;
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
                continue;
            const double dx = p.x - mx, dy = p.y - my, dz = p.z - mz;
            cxx += dx * dx; cxy += dx * dy; cxz += dx * dz;
            cyy += dy * dy; cyz += dy * dz;
            czz += dz * dz;
        }
        const double inv_n = 1.0 / static_cast<double>(count);
        const std::array<float, 9> cov = {
            static_cast<float>(cxx * inv_n), static_cast<float>(cxy * inv_n), static_cast<float>(cxz * inv_n),
            static_cast<float>(cxy * inv_n), static_cast<float>(cyy * inv_n), static_cast<float>(cyz * inv_n),
            static_cast<float>(cxz * inv_n), static_cast<float>(cyz * inv_n), static_cast<float>(czz * inv_n)
        };

        const auto eig = lodmath::eigen_symmetric_3x3_jacobi(cov);

        glm::vec3 v0{eig.vectors[0], eig.vectors[3], eig.vectors[6]};
        glm::vec3 v1{eig.vectors[1], eig.vectors[4], eig.vectors[7]};
        glm::vec3 v2{eig.vectors[2], eig.vectors[5], eig.vectors[8]};

        // Align canonical orientation preserving det(R) = +1
        if (v2.z < 0.0f || (v2.z == 0.0f && v2.y < 0.0f)) {
            v2 = -v2;
            v1 = -v1;
        }
        if (v0.x < 0.0f || (v0.x == 0.0f && v0.y < 0.0f)) {
            v0 = -v0;
            v1 = -v1;
        }

        glm::mat4 R{1.0f};
        R[0] = glm::vec4(v0, 0.0f);
        R[1] = glm::vec4(v1, 0.0f);
        R[2] = glm::vec4(v2, 0.0f);

        float rx = 0.0f, ry = 0.0f, rz = 0.0f;
        glm::extractEulerAngleXYZ(R, rx, ry, rz);
        if (!std::isfinite(rx) || !std::isfinite(ry) || !std::isfinite(rz))
            return glm::vec3(0.0f);
        return glm::vec3(glm::degrees(rx), glm::degrees(ry), glm::degrees(rz));
    }

    SceneBounds compute_robust_scene_bounds(const PointCloud& pc,
                                             const std::vector<glm::vec3>& camera_centers,
                                             const float quantile, const float angle_degrees) {
        if (!std::isfinite(quantile) || quantile < 0.0f || quantile >= 0.5f ||
            !std::isfinite(angle_degrees))
            throw std::invalid_argument("Invalid scene quantile or orientation");
        auto points = positions(pc);
        if (points.size() < 2 && !camera_centers.empty()) points = camera_centers;
        if (points.empty())
            throw std::invalid_argument("Auto-tiling needs sparse points or camera centers");

        const glm::mat4 R = glm::eulerAngleXYZ(0.0f, 0.0f, glm::radians(angle_degrees));
        const glm::mat3 R_inv = glm::transpose(glm::mat3(R));
        std::vector<float> axes[3];
        for (auto& axis : axes) axis.reserve(points.size());
        for (const auto& p : points) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            const glm::vec3 local_p = R_inv * p;
            axes[0].push_back(local_p.x);
            axes[1].push_back(local_p.y);
            axes[2].push_back(local_p.z);
        }
        if (axes[0].empty())
            throw std::invalid_argument("Auto-tiling has no finite positions");
        SceneBounds bounds;
        bounds.rotation_degrees = glm::vec3(0.0f, 0.0f, angle_degrees);
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

    SceneBounds compute_robust_scene_bounds_obb(
        const PointCloud& pc, const std::vector<glm::vec3>& camera_centers,
        const float quantile,
        const OrientationMode mode,
        const BoundsSource source) {
        if (!std::isfinite(quantile) || quantile < 0.0f || quantile >= 0.5f)
            throw std::invalid_argument("Invalid scene quantile");

        const auto pts = bounds_positions(pc, camera_centers, source);
        if (pts.empty())
            throw std::invalid_argument("Auto-tiling needs sparse points or camera centers");

        glm::vec3 rotation_degrees{0.0f};
        if (mode == OrientationMode::Obb3D) {
            rotation_degrees = compute_dominant_orientation_3d(pts, quantile);
        } else if (mode == OrientationMode::PlanarYaw) {
            const float yaw = compute_dominant_orientation_pca_from_points(pts, 0, 1, quantile);
            rotation_degrees = glm::vec3(0.0f, 0.0f, yaw);
        } else if (mode == OrientationMode::PlanarPitch) {
            const float pitch = compute_dominant_orientation_pca_from_points(pts, 0, 2, quantile);
            rotation_degrees = glm::vec3(0.0f, -pitch, 0.0f);
        } else if (mode == OrientationMode::PlanarRoll) {
            const float roll = compute_dominant_orientation_pca_from_points(pts, 1, 2, quantile);
            rotation_degrees = glm::vec3(roll, 0.0f, 0.0f);
        } else if (mode == OrientationMode::AxisAligned) {
            rotation_degrees = glm::vec3(0.0f);
        }

        const glm::mat4 R = glm::eulerAngleXYZ(glm::radians(rotation_degrees.x),
                                               glm::radians(rotation_degrees.y),
                                               glm::radians(rotation_degrees.z));
        const glm::mat3 R_inv = glm::transpose(glm::mat3(R));

        std::vector<float> axes[3];
        for (auto& axis : axes) axis.reserve(pts.size());
        for (const auto& p : pts) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            const glm::vec3 local_p = R_inv * p;
            axes[0].push_back(local_p.x);
            axes[1].push_back(local_p.y);
            axes[2].push_back(local_p.z);
        }
        if (axes[0].empty())
            throw std::invalid_argument("Auto-tiling has no finite positions");

        SceneBounds bounds;
        bounds.rotation_degrees = rotation_degrees;
        bounds.angle_degrees = rotation_degrees.z;
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

    SceneBounds compute_outer_scene_bounds(const PointCloud& pc,
                                           const std::vector<glm::vec3>& camera_centers,
                                           const SceneBounds& partition_bounds,
                                           const float quantile,
                                           const BoundsSource source) {
        if (!std::isfinite(quantile) || quantile < 0.0f || quantile >= 0.5f)
            throw std::invalid_argument("Invalid scene quantile");
        const auto pts = bounds_positions(pc, camera_centers, source);
        if (pts.empty())
            throw std::invalid_argument("Auto-tiling needs sparse points or camera centers");
        const glm::mat4 R = glm::eulerAngleXYZ(glm::radians(partition_bounds.rotation_degrees.x),
                                               glm::radians(partition_bounds.rotation_degrees.y),
                                               glm::radians(partition_bounds.rotation_degrees.z));
        const glm::mat3 R_inv = glm::transpose(glm::mat3(R));
        std::vector<float> axes[3];
        for (auto& axis : axes) axis.reserve(pts.size());
        for (const auto& p : pts) {
            if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) continue;
            const auto local = R_inv * p;
            for (int axis = 0; axis < 3; ++axis) axes[axis].push_back(local[axis]);
        }
        if (axes[0].empty())
            throw std::invalid_argument("Auto-tiling has no finite positions");
        SceneBounds outer = partition_bounds;
        const float outer_quantile = quantile * 0.25f;
        for (int axis = 0; axis < 3; ++axis) {
            outer.min[axis] = percentile(axes[axis], outer_quantile);
            outer.max[axis] = percentile(axes[axis], 1.0f - outer_quantile);
        }
        return outer;
    }

    static std::vector<TileSpec> generate_tile_grid_impl(const SceneBounds& bounds, const TileGridConfig& config,
                                                         const SceneBounds* outer_bounds) {
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
        if (outer_bounds) {
            for (int axis = 0; axis < 3; ++axis) {
                if (!std::isfinite(outer_bounds->min[axis]) || !std::isfinite(outer_bounds->max[axis]) ||
                    outer_bounds->min[axis] > outer_bounds->max[axis] ||
                    outer_bounds->rotation_degrees[axis] != bounds.rotation_degrees[axis])
                    throw std::invalid_argument("Outer bounds must be finite and share the grid orientation");
            }
        }
        const glm::vec2 cell{extent.x / columns, extent.y / rows};
        const glm::vec2 margin = cell * config.overlap_ratio * 0.5f;
        // The baseline extends beyond the existing partition padding even when the looser
        // percentile falls inside it. Cap the extension in case that percentile contains noise.
        const glm::vec2 outer_low = outer_bounds
                                        ? glm::max(glm::vec2{bounds.min.x - outer_bounds->min.x,
                                                             bounds.min.y - outer_bounds->min.y}, glm::vec2(0.0f))
                                        : glm::vec2(0.0f);
        const glm::vec2 outer_high = outer_bounds
                                         ? glm::max(glm::vec2{outer_bounds->max.x - bounds.max.x,
                                                              outer_bounds->max.y - bounds.max.y}, glm::vec2(0.0f))
                                         : glm::vec2(0.0f);
        const glm::vec2 extension_low = outer_bounds
                                            ? glm::min(glm::max(outer_low, cell * 0.1f), cell * 0.5f)
                                            : glm::vec2(0.0f);
        const glm::vec2 extension_high = outer_bounds
                                             ? glm::min(glm::max(outer_high, cell * 0.1f), cell * 0.5f)
                                             : glm::vec2(0.0f);
        const glm::mat4 R = glm::eulerAngleXYZ(glm::radians(bounds.rotation_degrees.x),
                                               glm::radians(bounds.rotation_degrees.y),
                                               glm::radians(bounds.rotation_degrees.z));
        std::vector<TileSpec> tiles;
        tiles.reserve(columns * rows);
        for (int j = 0; j < rows; ++j) {
            for (int i = 0; i < columns; ++i) {
                const glm::vec2 low{bounds.min.x + cell.x * i - (i ? margin.x : extension_low.x),
                                    bounds.min.y + cell.y * j - (j ? margin.y : extension_low.y)};
                const glm::vec2 high{bounds.min.x + cell.x * (i + 1) + (i + 1 < columns ? margin.x : extension_high.x),
                                     bounds.min.y + cell.y * (j + 1) + (j + 1 < rows ? margin.y : extension_high.y)};
                const glm::vec3 local_center{(low.x + high.x) * 0.5f,
                                             (low.y + high.y) * 0.5f,
                                             (bounds.min.z + bounds.max.z) * 0.5f};
                const glm::vec3 half{(high.x - low.x) * 0.5f,
                                     (high.y - low.y) * 0.5f,
                                     extent.z * 0.5f};
                const glm::vec3 world_center = glm::vec3(R * glm::vec4(local_center, 1.0f));
                tiles.push_back({world_center - half, world_center + half,
                                 bounds.rotation_degrees, i, j,
                                 std::format("tile_{}_{}", i, j)});
            }
        }
        return tiles;
    }

    std::vector<TileSpec> generate_tile_grid(const SceneBounds& bounds, const TileGridConfig& config) {
        return generate_tile_grid_impl(bounds, config, nullptr);
    }

    std::vector<TileSpec> generate_tile_grid(const SceneBounds& bounds, const TileGridConfig& config,
                                              const SceneBounds& outer_bounds) {
        return generate_tile_grid_impl(bounds, config, &outer_bounds);
    }
} // namespace lfs::core

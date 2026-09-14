/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "bounded_pose_optimizer.hpp"
#include "sparse_reprojection_guard.hpp"
#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>

namespace lfs::training::camera_pose {
    using SparsePointPosition = std::array<double, 3>;

    struct SparseTrackMeasurement {
        std::uint64_t point_id = std::numeric_limits<std::uint64_t>::max();
        SparsePointPosition source{};
        int camera_uid = -1;
        bool training = false;
        Matrix4 pose = identity_transform();
        ReprojectionCalibration calibration{};
        double u = 0, v = 0;
    };

    struct SparsePointTrack {
        std::uint64_t point_id;
        SparsePointPosition source;
        std::vector<SparseTrackMeasurement> measurements;
    };

    // One dataset per graph. Do not infer correspondence from nearby positions:
    // two different tracks can occupy the same location. Evaluation/disabled
    // views are excluded by the caller's training membership, before grouping.
    inline std::vector<SparsePointTrack> build_sparse_point_tracks(std::span<const SparseTrackMeasurement> input) {
        std::map<std::uint64_t, SparsePointTrack> tracks;
        std::set<std::uint64_t> invalid;
        for (const auto& m : input) {
            if (!m.training || m.point_id == std::numeric_limits<std::uint64_t>::max())
                continue;
            auto [it, inserted] = tracks.try_emplace(m.point_id, SparsePointTrack{m.point_id, m.source, {}});
            auto& track = it->second;
            (void)inserted;
            for (int axis = 0; axis < 3; ++axis)
                if (!std::isfinite(m.source[axis]) || m.source[axis] != track.source[axis])
                    invalid.insert(m.point_id);
            if (std::any_of(track.measurements.begin(), track.measurements.end(),
                            [&](const auto& other) { return m.camera_uid == other.camera_uid; }))
                invalid.insert(m.point_id);
            track.measurements.push_back(m);
        }
        std::vector<SparsePointTrack> result;
        for (auto& [id, track] : tracks) {
            if (invalid.contains(id) || track.measurements.size() < 3)
                continue;
            std::sort(track.measurements.begin(), track.measurements.end(),
                      [](const auto& a, const auto& b) { return a.camera_uid < b.camera_uid; });
            result.push_back(std::move(track));
        }
        return result;
    }

    struct SparsePointProposal {
        SparsePointPosition position;
        double source_cost, candidate_cost;
    };

    // Point block for a future joint pose/structure transaction. Produces only a
    // proposal; it never rewrites COLMAP, Camera, Gaussian means or optimizer
    // moments. All poses and measured pixels stay fixed for this solve.
    inline std::optional<SparsePointProposal> propose_sparse_point(
        const SparsePointTrack& track, const double max_displacement,
        const double huber_pixels = 2.0) {
        if (track.point_id == std::numeric_limits<std::uint64_t>::max() ||
            track.measurements.size() < 3 || !std::isfinite(max_displacement) || max_displacement <= 0 ||
            !std::isfinite(huber_pixels) || huber_pixels <= 0)
            return std::nullopt;
        std::set<int> cameras;
        for (const auto& m : track.measurements) {
            const auto& k = m.calibration;
            if (!m.training || m.point_id != track.point_id || !cameras.insert(m.camera_uid).second ||
                k.width <= 0 || k.height <= 0 || k.fx <= 0 || k.fy <= 0 ||
                !std::isfinite(k.fx) || !std::isfinite(k.fy) || !std::isfinite(k.cx) || !std::isfinite(k.cy) ||
                !std::isfinite(m.u) || !std::isfinite(m.v) ||
                m.u < 0 || m.v < 0 || m.u >= k.width || m.v >= k.height || m.source != track.source)
                return std::nullopt;
            try {
                (void)BoundedPoseOptimizer(m.camera_uid, m.pose, BoundedPoseConfig{});
            } catch (const std::invalid_argument&) {
                return std::nullopt;
            }
        }
        using Matrix3 = std::array<SparsePointPosition, 3>;
        auto evaluate = [&](const SparsePointPosition& point, Matrix3* h, SparsePointPosition* g) {
            double cost = 0;
            for (const auto& m : track.measurements) {
                const auto& p = m.pose;
                const auto& k = m.calibration;
                const double x = p[0] * point[0] + p[1] * point[1] + p[2] * point[2] + p[3];
                const double y = p[4] * point[0] + p[5] * point[1] + p[6] * point[2] + p[7];
                const double z = p[8] * point[0] + p[9] * point[1] + p[10] * point[2] + p[11];
                if (!std::isfinite(z) || z <= 0)
                    return std::numeric_limits<double>::infinity();
                const double scale = std::max(k.width, k.height);
                const double ru = (k.fx * x / z + k.cx - m.u) / scale;
                const double rv = (k.fy * y / z + k.cy - m.v) / scale;
                const double r = std::hypot(ru, rv), delta = huber_pixels / scale;
                if (!std::isfinite(r))
                    return std::numeric_limits<double>::infinity();
                cost += r <= delta ? 0.5 * r * r : delta * (r - 0.5 * delta);
                if (!h)
                    continue;
                const double weight = r <= delta ? 1.0 : delta / r;
                SparsePointPosition ju{}, jv{};
                for (int a = 0; a < 3; ++a) {
                    ju[a] = k.fx / (scale * z) * (p[a] - x / z * p[8 + a]);
                    jv[a] = k.fy / (scale * z) * (p[4 + a] - y / z * p[8 + a]);
                    (*g)[a] += weight * (ju[a] * ru + jv[a] * rv);
                    for (int b = 0; b <= a; ++b) {
                        (*h)[a][b] += weight * (ju[a] * ju[b] + jv[a] * jv[b]);
                        (*h)[b][a] = (*h)[a][b];
                    }
                }
            }
            return cost;
        };
        SparsePointPosition position = track.source;
        const double source_cost = evaluate(position, nullptr, nullptr);
        if (!std::isfinite(source_cost))
            return std::nullopt;
        double cost = source_cost;
        for (int iteration = 0; iteration < 8; ++iteration) {
            Matrix3 h{}, l{};
            SparsePointPosition g{}, scale{}, step{};
            evaluate(position, &h, &g);
            for (int a = 0; a < 3; ++a) {
                if (!std::isfinite(h[a][a]) || h[a][a] <= 0)
                    return std::nullopt;
                scale[a] = std::sqrt(h[a][a]);
            }
            // Reject unobservable depth rather than inventing a damping prior.
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b <= a; ++b) {
                    double value = h[a][b] / (scale[a] * scale[b]);
                    for (int c = 0; c < b; ++c)
                        value -= l[a][c] * l[b][c];
                    if (!std::isfinite(value) || (a == b && value <= 1e-8))
                        return std::nullopt;
                    l[a][b] = a == b ? std::sqrt(value) : value / l[b][b];
                }
                double value = -g[a] / scale[a];
                for (int b = 0; b < a; ++b)
                    value -= l[a][b] * step[b];
                step[a] = value / l[a][a];
            }
            for (int a = 2; a >= 0; --a) {
                for (int b = a + 1; b < 3; ++b)
                    step[a] -= l[b][a] * step[b];
                step[a] /= l[a][a];
            }
            for (int a = 0; a < 3; ++a)
                step[a] /= scale[a];
            bool accepted = false;
            for (int backtrack = 0; backtrack < 8; ++backtrack) {
                const double alpha = std::ldexp(1.0, -backtrack);
                SparsePointPosition candidate{};
                double distance2 = 0;
                for (int a = 0; a < 3; ++a) {
                    candidate[a] = position[a] + alpha * step[a];
                    distance2 += std::pow(candidate[a] - track.source[a], 2);
                }
                if (!std::isfinite(distance2) || std::sqrt(distance2) > max_displacement)
                    continue;
                const double next_cost = evaluate(candidate, nullptr, nullptr);
                if (std::isfinite(next_cost) && next_cost < cost) {
                    position = candidate;
                    cost = next_cost;
                    accepted = true;
                    break;
                }
            }
            if (!accepted)
                break;
        }
        if (!(cost < source_cost))
            return std::nullopt;
        return SparsePointProposal{position, source_cost, cost};
    }
} // namespace lfs::training::camera_pose

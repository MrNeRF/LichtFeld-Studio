/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "sparse_point_refinement.hpp"

namespace lfs::training::camera_pose {
    namespace joint_detail {
        template <size_t N>
        using Vector = std::array<double, N>;
        template <size_t N>
        using Matrix = std::array<Vector<N>, N>;

        // Column normalization makes the observability check independent of
        // world units. Do not turn singular geometry into evidence by damping it.
        template <size_t N>
        struct Factor {
            Matrix<N> l{};
            Vector<N> scale{};
            bool compute(const Matrix<N>& h) {
                for (size_t i = 0; i < N; ++i) {
                    if (!std::isfinite(h[i][i]) || h[i][i] <= 0)
                        return false;
                    scale[i] = std::sqrt(h[i][i]);
                }
                for (size_t i = 0; i < N; ++i)
                    for (size_t j = 0; j <= i; ++j) {
                        double v = h[i][j] / (scale[i] * scale[j]);
                        for (size_t k = 0; k < j; ++k)
                            v -= l[i][k] * l[j][k];
                        if (!std::isfinite(v) || (i == j && v <= 1e-8))
                            return false;
                        l[i][j] = i == j ? std::sqrt(v) : v / l[j][j];
                    }
                return true;
            }
            Vector<N> solve(Vector<N> rhs) const {
                for (size_t i = 0; i < N; ++i) {
                    rhs[i] /= scale[i];
                    for (size_t j = 0; j < i; ++j)
                        rhs[i] -= l[i][j] * rhs[j];
                    rhs[i] /= l[i][i];
                }
                for (size_t i = N; i-- > 0;) {
                    for (size_t j = i + 1; j < N; ++j)
                        rhs[i] -= l[j][i] * rhs[j];
                    rhs[i] /= l[i][i];
                }
                for (size_t i = 0; i < N; ++i)
                    rhs[i] /= scale[i];
                return rhs;
            }
        };
    } // namespace joint_detail

    // Precondition the COMBINED gradient, not the geometric gradient alone.
    // The identity term is damping in scene-normalized tangent coordinates,
    // not an additional objective or a claim of recovered observability.
    // Points remain fixed, so this uses the camera block, not a Schur elimination.
    inline std::optional<Twist> propose_combined_pose(
        const SparsePoseObjective& geometry, const Twist& gradient,
        double weight, double scene_scale) {
        if (!std::isfinite(weight) || weight <= 0 || !std::isfinite(scene_scale) || scene_scale <= 0)
            return std::nullopt;
        joint_detail::Matrix<6> h{};
        joint_detail::Vector<6> rhs{};
        for (size_t a = 0; a < 6; ++a) {
            const double sa = a < 3 ? scene_scale : 1.0;
            rhs[a] = -gradient[a] * sa;
            if (!std::isfinite(rhs[a]))
                return std::nullopt;
            for (size_t b = 0; b < 6; ++b) {
                const double sb = b < 3 ? scene_scale : 1.0;
                h[a][b] = weight * geometry.curvature[a][b] * sa * sb + (a == b ? 1.0 : 0.0);
                if (!std::isfinite(h[a][b]))
                    return std::nullopt;
            }
        }
        joint_detail::Factor<6> factor;
        if (!factor.compute(h))
            return std::nullopt;
        const auto step = factor.solve(rhs);
        Twist proposal{};
        double slope = 0;
        for (size_t a = 0; a < 6; ++a) {
            proposal[a] = static_cast<float>(step[a] * (a < 3 ? scene_scale : 1.0));
            if (!std::isfinite(proposal[a]))
                return std::nullopt;
            slope += proposal[a] * gradient[a];
        }
        return std::isfinite(slope) && slope < 0 ? std::optional<Twist>(proposal) : std::nullopt;
    }

    // One active camera, all incident points, all other cameras fixed. Eliminate
    // point increments from the robust GN system before solving the camera block:
    // S = A - E C^-1 E^T, rhs = -g_camera + E C^-1 g_point.
    // Unlike a fixed-point PnP proposal this accounts for structure motion.
    // This is NOT multi-camera BA. Nonlinear point bounds, image descent and
    // geometry acceptance still belong to the caller's transaction.
    inline std::optional<Twist> propose_joint_pose(
        int uid, const Matrix4& pose, std::span<const SparsePointTrack> tracks,
        std::span<const SparsePointPosition> positions, double scene_scale,
        std::optional<Twist> photometric_gradient = {}, double geometry_weight = 1e-4) {
        using namespace joint_detail;
        if (uid < 0 || tracks.size() < 12 || tracks.size() != positions.size() ||
            !std::isfinite(scene_scale) || scene_scale <= 0 ||
            (photometric_gradient && (!std::isfinite(geometry_weight) || geometry_weight <= 0)))
            return std::nullopt;
        try {
            (void)BoundedPoseOptimizer(uid, pose, BoundedPoseConfig{});
        } catch (const std::invalid_argument&) {
            return std::nullopt;
        }
        Matrix<6> reduced{};
        Vector<6> rhs{};
        for (size_t t = 0; t < tracks.size(); ++t) {
            const auto& track = tracks[t];
            const auto& point = positions[t];
            if (track.measurements.size() < 3 ||
                track.point_id == std::numeric_limits<std::uint64_t>::max() ||
                !std::all_of(point.begin(), point.end(), [](double v) { return std::isfinite(v); }))
                return std::nullopt;
            Matrix<3> c{};
            Vector<3> gp{};
            Matrix<6> a{};
            Vector<6> gc{};
            std::array<Vector<3>, 6> e{};
            std::set<int> seen;
            for (const auto& m : track.measurements) {
                const auto& p = m.camera_uid == uid ? pose : m.pose;
                const auto& k = m.calibration;
                if (!m.training || m.point_id != track.point_id || m.source != track.source ||
                    !seen.insert(m.camera_uid).second || k.width <= 0 || k.height <= 0 ||
                    !std::isfinite(k.fx) || !std::isfinite(k.fy) || k.fx <= 0 || k.fy <= 0 ||
                    !std::isfinite(k.cx) || !std::isfinite(k.cy) || !std::isfinite(m.u) || !std::isfinite(m.v) ||
                    m.u < 0 || m.v < 0 || m.u >= k.width || m.v >= k.height)
                    return std::nullopt;
                try {
                    (void)BoundedPoseOptimizer(m.camera_uid, p, BoundedPoseConfig{});
                } catch (const std::invalid_argument&) {
                    return std::nullopt;
                }
                const double x = p[0] * point[0] + p[1] * point[1] + p[2] * point[2] + p[3];
                const double y = p[4] * point[0] + p[5] * point[1] + p[6] * point[2] + p[7];
                const double z = p[8] * point[0] + p[9] * point[1] + p[10] * point[2] + p[11];
                if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || z <= 0)
                    return std::nullopt;
                const double n = std::max(k.width, k.height) / (photometric_gradient ? 1600.0 : 1.0);
                const double ru = (k.fx * x / z + k.cx - m.u) / n;
                const double rv = (k.fy * y / z + k.cy - m.v) / n;
                const double r = std::hypot(ru, rv), delta = photometric_gradient ? 1.0 : 2.0 / n;
                if (!std::isfinite(r))
                    return std::nullopt;
                const double w = r <= delta ? 1.0 : delta / r;
                const double u = k.fx / (n * z), v = k.fy / (n * z);
                Vector<3> pu{}, pv{};
                for (int i = 0; i < 3; ++i) {
                    pu[i] = u * (p[i] - x / z * p[8 + i]);
                    pv[i] = v * (p[4 + i] - y / z * p[8 + i]);
                    gp[i] += w * (pu[i] * ru + pv[i] * rv);
                }
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        c[i][j] += w * (pu[i] * pu[j] + pv[i] * pv[j]);
                if (m.camera_uid != uid)
                    continue;
                const Vector<6> cu{u * scene_scale, 0, -u * x / z * scene_scale,
                                   -u * x * y / z, u * (z + x * x / z), -u * y};
                const Vector<6> cv{0, v * scene_scale, -v * y / z * scene_scale,
                                   -v * (z + y * y / z), v * x * y / z, v * x};
                for (int i = 0; i < 6; ++i) {
                    gc[i] += w * (cu[i] * ru + cv[i] * rv);
                    for (int j = 0; j < 6; ++j)
                        a[i][j] += w * (cu[i] * cu[j] + cv[i] * cv[j]);
                    for (int j = 0; j < 3; ++j)
                        e[i][j] += w * (cu[i] * pu[j] + cv[i] * pv[j]);
                }
            }
            if (!seen.contains(uid))
                return std::nullopt;
            Factor<3> factor;
            if (!factor.compute(c))
                return std::nullopt;
            const auto cg = factor.solve(gp);
            std::array<Vector<3>, 6> ce{};
            for (int i = 0; i < 6; ++i)
                ce[i] = factor.solve(e[i]);
            for (int i = 0; i < 6; ++i) {
                rhs[i] -= gc[i];
                for (int k = 0; k < 3; ++k)
                    rhs[i] += e[i][k] * cg[k];
                for (int j = 0; j <= i; ++j) {
                    double value = a[i][j];
                    for (int k = 0; k < 3; ++k)
                        value -= e[i][k] * ce[j][k];
                    reduced[i][j] += value;
                    reduced[j][i] = reduced[i][j];
                }
            }
        }
        // Joint mode retains the Schur point response, then adds the image
        // gradient to the same reduced system. Unit damping is expressed in
        // scene-normalized tangent coordinates, not added to the loss.
        if (photometric_gradient) {
            for (size_t i = 0; i < 6; ++i) {
                const double image_gradient = (*photometric_gradient)[i] * (i < 3 ? scene_scale : 1.0);
                if (!std::isfinite(image_gradient))
                    return std::nullopt;
                rhs[i] = geometry_weight * rhs[i] - image_gradient;
                for (size_t j = 0; j < 6; ++j)
                    reduced[i][j] = geometry_weight * reduced[i][j] + (i == j ? 1.0 : 0.0);
            }
        }
        Factor<6> factor;
        if (!factor.compute(reduced))
            return std::nullopt;
        const auto solution = factor.solve(rhs);
        Twist step{};
        for (int i = 0; i < 6; ++i) {
            const double value = solution[i] * (i < 3 ? scene_scale : 1.0);
            if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
                return std::nullopt;
            step[i] = static_cast<float>(value);
        }
        return step;
    }
} // namespace lfs::training::camera_pose

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "se3.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace lfs::training::camera_pose {
    struct ReprojectionObservation {
        double u, v, x, y, z;
    };

    struct ReprojectionCalibration {
        double fx, fy, cx, cy;
        int width, height;
    };

    // Fixed sparse geometry, independent of the evolving Gaussian model. Select
    // support once from the imported pose; candidates cannot discard observations
    // by moving them behind the camera or changing their residual ranking.
    class SparseReprojectionGuard {
    public:
        SparseReprojectionGuard() = default;
        SparseReprojectionGuard(const Matrix4& source, ReprojectionCalibration calibration,
                                std::span<const ReprojectionObservation> observations)
            : calibration_(calibration) {
            if (calibration.width <= 0 || calibration.height <= 0 ||
                !std::isfinite(calibration.fx) || !std::isfinite(calibration.fy) ||
                !std::isfinite(calibration.cx) || !std::isfinite(calibration.cy) ||
                calibration.fx <= 0 || calibration.fy <= 0)
                return;
            normalization_ = static_cast<double>(std::max(calibration.width, calibration.height));
            std::vector<double> errors;
            for (const auto& observation : observations) {
                if (!std::isfinite(observation.u) || !std::isfinite(observation.v) ||
                    observation.u < 0 || observation.u >= calibration.width ||
                    observation.v < 0 || observation.v >= calibration.height)
                    continue;
                const double error = residual(source, observation);
                if (!std::isfinite(error))
                    continue;
                observations_.push_back(observation);
                errors.push_back(error);
            }
            if (errors.size() < MIN_OBSERVATIONS) {
                observations_.clear();
                return;
            }
            auto middle = errors.begin() + errors.size() / 2;
            std::nth_element(errors.begin(), middle, errors.end());
            // Reject gross source outliers, keeping the same inlier set for all
            // later proposals. The pixel floor only affects support selection.
            const double cutoff = std::max(4.0 * *middle, 4.0 / normalization_);
            std::erase_if(observations_, [&](const auto& observation) {
                return residual(source, observation) > cutoff;
            });
            double min_u = calibration.width, min_v = calibration.height;
            double max_u = 0, max_v = 0;
            for (const auto& observation : observations_) {
                min_u = std::min(min_u, observation.u);
                max_u = std::max(max_u, observation.u);
                min_v = std::min(min_v, observation.v);
                max_v = std::max(max_v, observation.v);
            }
            if (observations_.size() < MIN_OBSERVATIONS ||
                max_u - min_u < 0.1 * calibration.width ||
                max_v - min_v < 0.1 * calibration.height) {
                observations_.clear();
                return;
            }
            source_error_ = error(source);
            if (!std::isfinite(source_error_))
                observations_.clear();
        }

        [[nodiscard]] bool active() const noexcept { return !observations_.empty(); }
        [[nodiscard]] std::size_t observation_count() const noexcept { return observations_.size(); }
        [[nodiscard]] double source_error() const noexcept { return source_error_; }

        // RMS in image-size-normalized pixels. The ceiling always refers to the
        // immutable source, including after checkpoint resume; it cannot ratchet
        // outwards across visits. Allow only float pose/projection roundoff.
        [[nodiscard]] bool allows(const Matrix4& candidate) const noexcept {
            if (!active())
                return true;
            constexpr double ROUNDING_ALLOWANCE = 8.0 * std::numeric_limits<float>::epsilon();
            const double candidate_error = error(candidate);
            return std::isfinite(candidate_error) &&
                   candidate_error <= source_error_ + ROUNDING_ALLOWANCE;
        }

        [[nodiscard]] double error(const Matrix4& pose) const noexcept {
            if (!active())
                return std::numeric_limits<double>::infinity();
            double squared_sum = 0;
            for (const auto& observation : observations_) {
                const double value = residual(pose, observation);
                if (!std::isfinite(value))
                    return std::numeric_limits<double>::infinity();
                squared_sum += value * value;
            }
            return std::sqrt(squared_sum / static_cast<double>(observations_.size()));
        }

        // Gauss-Newton proposal from the fixed observations, in the same fresh
        // left tangent as the photometric gradient. It is only a proposal:
        // the caller must still enforce photometric descent and pose bounds.
        [[nodiscard]] std::optional<Twist> proposal(const Matrix4& pose, double scene_scale) const noexcept {
            if (!active() || !std::isfinite(scene_scale) || scene_scale <= 0)
                return std::nullopt;
            using Vector = std::array<double, 6>;
            using Matrix = std::array<Vector, 6>;
            Matrix h{};
            Vector g{};
            for (const auto& p : observations_) {
                const double x = pose[0] * p.x + pose[1] * p.y + pose[2] * p.z + pose[3];
                const double y = pose[4] * p.x + pose[5] * p.y + pose[6] * p.z + pose[7];
                const double z = pose[8] * p.x + pose[9] * p.y + pose[10] * p.z + pose[11];
                if (!std::isfinite(z) || z <= 0)
                    return std::nullopt;
                const double a = calibration_.fx / (normalization_ * z);
                const double b = calibration_.fy / (normalization_ * z);
                const Vector ju{a * scene_scale, 0, -a * x / z * scene_scale,
                                -a * x * y / z, a * (z + x * x / z), -a * y};
                const Vector jv{0, b * scene_scale, -b * y / z * scene_scale,
                                -b * (z + y * y / z), b * x * y / z, b * x};
                const double ru = (calibration_.fx * x / z + calibration_.cx - p.u) / normalization_;
                const double rv = (calibration_.fy * y / z + calibration_.cy - p.v) / normalization_;
                for (int i = 0; i < 6; ++i) {
                    g[i] += ju[i] * ru + jv[i] * rv;
                    for (int j = 0; j < 6; ++j)
                        h[i][j] += ju[i] * ju[j] + jv[i] * jv[j];
                }
            }
            // Normalize the columns before Cholesky so the rank check is not
            // tied to scene units or image resolution. Reject weak geometry;
            // regularization must not manufacture an observable pose direction.
            Vector diagonal{}, rhs{}, solution{};
            Matrix l{};
            for (int i = 0; i < 6; ++i) {
                if (!std::isfinite(h[i][i]) || h[i][i] <= 0 || !std::isfinite(g[i]))
                    return std::nullopt;
                diagonal[i] = std::sqrt(h[i][i]);
                rhs[i] = -g[i] / diagonal[i];
            }
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j <= i; ++j) {
                    double value = h[i][j] / (diagonal[i] * diagonal[j]);
                    for (int k = 0; k < j; ++k)
                        value -= l[i][k] * l[j][k];
                    if (!std::isfinite(value))
                        return std::nullopt;
                    if (i == j) {
                        if (value <= 1e-8)
                            return std::nullopt;
                        l[i][j] = std::sqrt(value);
                    } else {
                        l[i][j] = value / l[j][j];
                    }
                }
            }
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < i; ++j)
                    rhs[i] -= l[i][j] * rhs[j];
                rhs[i] /= l[i][i];
            }
            for (int i = 5; i >= 0; --i) {
                double value = rhs[i];
                for (int j = i + 1; j < 6; ++j)
                    value -= l[j][i] * solution[j];
                solution[i] = value / l[i][i];
            }
            Twist step{};
            for (int i = 0; i < 6; ++i) {
                const double value = solution[i] / diagonal[i] * (i < 3 ? scene_scale : 1.0);
                if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<float>::max())
                    return std::nullopt;
                step[i] = static_cast<float>(value);
            }
            return step;
        }

    private:
        [[nodiscard]] double residual(const Matrix4& pose, const ReprojectionObservation& point) const noexcept {
            const double x = pose[0] * point.x + pose[1] * point.y + pose[2] * point.z + pose[3];
            const double y = pose[4] * point.x + pose[5] * point.y + pose[6] * point.z + pose[7];
            const double z = pose[8] * point.x + pose[9] * point.y + pose[10] * point.z + pose[11];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || z <= 0)
                return std::numeric_limits<double>::infinity();
            const double du = (calibration_.fx * (x / z) + calibration_.cx - point.u) / normalization_;
            const double dv = (calibration_.fy * (y / z) + calibration_.cy - point.v) / normalization_;
            return std::hypot(du, dv);
        }

        static constexpr std::size_t MIN_OBSERVATIONS = 12;
        ReprojectionCalibration calibration_{};
        double normalization_ = 1;
        double source_error_ = std::numeric_limits<double>::infinity();
        std::vector<ReprojectionObservation> observations_;
    };
} // namespace lfs::training::camera_pose

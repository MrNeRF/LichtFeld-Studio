/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "bounded_pose_optimizer.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace lfs::training::camera_pose {
    namespace {
        using Vector = std::array<double, 6>;
        double dot(const Vector& a, const Vector& b) {
            return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
        }
        std::array<double, 3> center(const Matrix4& pose) {
            std::array<double, 3> result{};
            for (int j = 0; j < 3; ++j)
                for (int i = 0; i < 3; ++i)
                    result[j] -= static_cast<double>(pose[4 * i + j]) * pose[4 * i + 3];
            return result;
        }
        double center_distance(const Matrix4& a, const Matrix4& b) {
            const auto ca = center(a), cb = center(b);
            double squared = 0;
            for (int i = 0; i < 3; ++i)
                squared += (ca[i] - cb[i]) * (ca[i] - cb[i]);
            return std::sqrt(squared);
        }
        double rotation_distance(const Matrix4& a, const Matrix4& b) {
            std::array<double, 9> r{};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    for (int k = 0; k < 3; ++k)
                        r[3 * i + j] += static_cast<double>(a[4 * i + k]) * b[4 * j + k];
            const double x = r[7] - r[5], y = r[2] - r[6], z = r[3] - r[1];
            return std::atan2(0.5 * std::sqrt(x * x + y * y + z * z), 0.5 * (r[0] + r[4] + r[8] - 1));
        }
        bool rigid(const Matrix4& pose) {
            for (const auto v : pose)
                if (!std::isfinite(v))
                    return false;
            if (pose[12] != 0 || pose[13] != 0 || pose[14] != 0 || pose[15] != 1)
                return false;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    double value = 0;
                    for (int k = 0; k < 3; ++k)
                        value += static_cast<double>(pose[4 * i + k]) * pose[4 * j + k];
                    if (std::abs(value - (i == j ? 1.0 : 0.0)) > 1e-3)
                        return false;
                }
            const double determinant = pose[0] * (pose[5] * pose[10] - pose[6] * pose[9]) -
                                       pose[1] * (pose[4] * pose[10] - pose[6] * pose[8]) + pose[2] * (pose[4] * pose[9] - pose[5] * pose[8]);
            return determinant > 0;
        }
    } // namespace

    BoundedPoseOptimizer::BoundedPoseOptimizer(int uid, const Matrix4& source,
                                               BoundedPoseConfig config, PoseRole role)
        : config_(config), role_(role) {
        for (const double value : {config.scene_scale, config.max_center_fraction,
                                   config.max_rotation_radians, config.step_center_fraction, config.step_rotation_radians})
            if (!std::isfinite(value) || value <= 0)
                throw std::invalid_argument("Pose scales and limits must be finite and positive");
        for (const double value : {config.center_prior, config.rotation_prior, config.min_relative_improvement})
            if (!std::isfinite(value) || value < 0)
                throw std::invalid_argument("Pose priors and improvement must be finite and nonnegative");
        if (!std::isfinite(config.scene_scale * config.scene_scale) || config.scene_scale * config.scene_scale == 0 ||
            !std::isfinite(config.scene_scale * config.max_center_fraction) ||
            !std::isfinite(config.scene_scale * config.step_center_fraction) ||
            uid < 0 || !rigid(source) || config.max_backtracks < 1 || config.max_backtracks > 16 ||
            config.min_relative_improvement >= 1 || config.max_rotation_radians >= 3.141592653589793 ||
            (role != PoseRole::Train && role != PoseRole::Anchor && role != PoseRole::Evaluation))
            throw std::invalid_argument("Invalid pose controller configuration or source pose");
        state_.uid = uid;
        state_.source = state_.current = source;
        clear_history();
    }

    PoseSnapshot BoundedPoseOptimizer::snapshot() const noexcept { return state_; }

    void BoundedPoseOptimizer::clear_history() noexcept {
        history_valid_ = false;
        inverse_hessian_ = {};
        for (int i = 0; i < 6; ++i)
            inverse_hessian_[i][i] = 1;
    }

    void BoundedPoseOptimizer::set_frozen(bool frozen) noexcept {
        frozen_ = frozen;
        clear_history();
    }

    void BoundedPoseOptimizer::reset() noexcept {
        state_.current = state_.source;
        ++state_.revision; // Do not make old baseline evaluations valid again.
        state_.accepted_steps = state_.rejected_steps = 0;
        state_.center_displacement = state_.rotation_displacement = 0;
        model_revision_ = 0;
        clear_history();
    }

    double BoundedPoseOptimizer::prior(const Matrix4& pose, Vector* gradient) const noexcept {
        const auto current_center = center(pose), source_center = center(state_.source);
        std::array<double, 3> delta{};
        double loss = 0;
        const double weight = config_.center_prior / (config_.scene_scale * config_.scene_scale);
        for (int i = 0; i < 3; ++i) {
            delta[i] = current_center[i] - source_center[i];
            loss += weight * delta[i] * delta[i];
        }
        Matrix4 rotation_gradient{};
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                const double difference = static_cast<double>(pose[4 * i + j]) - state_.source[4 * i + j];
                loss += 0.5 * config_.rotation_prior * difference * difference;
                rotation_gradient[4 * i + j] = static_cast<float>(config_.rotation_prior * difference);
            }
        if (gradient) {
            const auto rotation_tangent = left_increment_gradient(pose, rotation_gradient);
            for (int i = 0; i < 3; ++i) {
                (*gradient)[i] = 0;
                // dC/dv = -R^T; a pure left rotation keeps C fixed.
                for (int j = 0; j < 3; ++j)
                    (*gradient)[i] -= 2 * weight * pose[4 * i + j] * delta[j];
                (*gradient)[i + 3] = rotation_tangent[i + 3];
            }
        }
        return loss;
    }

    PoseStepResult BoundedPoseOptimizer::step(const PoseEvaluation& baseline,
                                              const std::function<double(const Matrix4&)>& candidate_loss) {
        auto working = *this;
        auto result = working.step_impl(baseline, candidate_loss);
        *this = std::move(working);
        return result;
    }

    PoseStepResult BoundedPoseOptimizer::step_impl(const PoseEvaluation& baseline,
                                                   const std::function<double(const Matrix4&)>& candidate_loss) {
        PoseStepResult result{PoseStepStatus::Rejected, 0, baseline.image_loss};
        if (frozen_ || role_ != PoseRole::Train) {
            result.status = PoseStepStatus::Frozen;
            return result;
        }
        if (baseline.uid != state_.uid || baseline.pose_revision != state_.revision ||
            baseline.model_revision < model_revision_) {
            result.status = PoseStepStatus::Stale;
            return result;
        }
        if (!candidate_loss || !std::isfinite(baseline.image_loss) || baseline.image_loss < 0 ||
            !std::all_of(baseline.image_gradient.begin(), baseline.image_gradient.end(), [](float v) { return std::isfinite(v); })) {
            result.status = PoseStepStatus::InvalidInput;
            return result;
        }
        Vector gradient{};
        const double objective = baseline.image_loss + prior(state_.current, &gradient);
        for (int i = 0; i < 6; ++i)
            gradient[i] = (gradient[i] + baseline.image_gradient[i]) * (i < 3 ? config_.scene_scale : 1.0);
        if (!std::isfinite(objective) || !std::all_of(gradient.begin(), gradient.end(), [](double v) { return std::isfinite(v); })) {
            result.status = PoseStepStatus::InvalidInput;
            return result;
        }
        if (model_revision_ != baseline.model_revision)
            clear_history();
        model_revision_ = baseline.model_revision;
        // Safeguarded inverse BFGS in the scaled left-tangent coordinates.
        // This is a direction approximation, not an exact SE(3) Hessian.
        // History is reusable ONLY while the model/objective stays frozen.
        if (history_valid_) {
            Vector y{}, hy{};
            for (int i = 0; i < 6; ++i)
                y[i] = gradient[i] - previous_gradient_[i];
            const double sy = dot(previous_step_, y);
            if (sy > 1e-12 && sy > 1e-6 * std::sqrt(dot(previous_step_, previous_step_) * dot(y, y))) {
                for (int i = 0; i < 6; ++i)
                    for (int j = 0; j < 6; ++j)
                        hy[i] += inverse_hessian_[i][j] * y[j];
                const double coefficient = (sy + dot(y, hy)) / (sy * sy);
                for (int i = 0; i < 6; ++i)
                    for (int j = 0; j < 6; ++j)
                        inverse_hessian_[i][j] += coefficient * previous_step_[i] * previous_step_[j] -
                                                  (hy[i] * previous_step_[j] + previous_step_[i] * hy[j]) / sy;
            } else {
                clear_history();
            }
        }
        Vector direction{};
        for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j)
                direction[i] -= inverse_hessian_[i][j] * gradient[j];
        double slope = dot(direction, gradient);
        if (!std::isfinite(slope) || slope >= 0) {
            clear_history();
            for (int i = 0; i < 6; ++i)
                direction[i] = -gradient[i];
            slope = -dot(gradient, gradient);
        }
        if (!std::isfinite(slope) || slope >= -1e-24) {
            result.status = PoseStepStatus::NoDescent;
            return result;
        }
        const double translation_norm = std::sqrt(direction[0] * direction[0] + direction[1] * direction[1] + direction[2] * direction[2]);
        const double rotation_norm = std::sqrt(direction[3] * direction[3] + direction[4] * direction[4] + direction[5] * direction[5]);
        double scale = std::min({1.0, config_.step_center_fraction / std::max(translation_norm, 1e-30),
                                 config_.step_rotation_radians / std::max(rotation_norm, 1e-30)});
        for (int attempt = 0; attempt < config_.max_backtracks; ++attempt, scale *= 0.5) {
            Twist increment{};
            for (int i = 0; i < 6; ++i)
                increment[i] = static_cast<float>(scale * direction[i] * (i < 3 ? config_.scene_scale : 1.0));
            const auto candidate = apply_left_increment(increment, state_.current);
            const double displacement = center_distance(candidate, state_.source);
            const double rotation = rotation_distance(candidate, state_.source);
            if (!rigid(candidate) || displacement > config_.max_center_fraction * config_.scene_scale ||
                rotation > config_.max_rotation_radians || candidate == state_.current)
                continue;
            ++result.evaluations;
            const double loss = candidate_loss(candidate);
            if (!std::isfinite(loss) || loss < 0)
                continue;
            if (loss < baseline.image_loss * (1 - config_.min_relative_improvement) &&
                loss + prior(candidate) <= objective + 1e-4 * scale * slope) {
                previous_gradient_ = gradient;
                for (int i = 0; i < 6; ++i)
                    previous_step_[i] = scale * direction[i];
                history_valid_ = true;
                state_.current = candidate;
                state_.center_displacement = displacement;
                state_.rotation_displacement = rotation;
                ++state_.revision;
                ++state_.accepted_steps;
                result.status = PoseStepStatus::Accepted;
                result.image_loss = loss;
                return result;
            }
        }
        clear_history();
        ++state_.rejected_steps;
        return result;
    }
} // namespace lfs::training::camera_pose

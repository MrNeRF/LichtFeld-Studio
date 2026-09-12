/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <array>

namespace lfs::training::camera_pose {

    // Row-major homogeneous transform. Twist order is [vx, vy, vz, wx, wy, wz].
    using Matrix4 = std::array<float, 16>;
    using Twist = std::array<float, 6>;

    [[nodiscard]] Matrix4 identity_transform() noexcept;

    // Exponential map from se(3) to SE(3), including the SO(3) left Jacobian
    // that maps the translational part of the twist.
    [[nodiscard]] Matrix4 exp_se3(const Twist& twist) noexcept;

    [[nodiscard]] Matrix4 multiply(const Matrix4& lhs, const Matrix4& rhs) noexcept;

    // Pose-refinement convention: the incremental camera-space motion is
    // applied on the left of the immutable source world-to-camera transform.
    [[nodiscard]] Matrix4 apply_left_increment(
        const Twist& increment,
        const Matrix4& source_world_to_camera) noexcept;

    // Converts dL/dT into the tangent gradient for a fresh left increment
    // Exp(d_xi) * T evaluated at d_xi = 0. This is the gradient used by a
    // retraction-based optimizer; it is not d/d_xi Exp(xi) at non-zero xi.
    [[nodiscard]] Twist left_increment_gradient(
        const Matrix4& world_to_camera,
        const Matrix4& transform_gradient) noexcept;

} // namespace lfs::training::camera_pose

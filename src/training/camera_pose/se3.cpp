/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "se3.hpp"

#include <array>
#include <cmath>

namespace lfs::training::camera_pose {
    namespace {
        using Matrix3d = std::array<double, 9>;

        Matrix3d multiply3(const Matrix3d& lhs, const Matrix3d& rhs) noexcept {
            Matrix3d result{};
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    for (int inner = 0; inner < 3; ++inner) {
                        result[row * 3 + column] +=
                            lhs[row * 3 + inner] * rhs[inner * 3 + column];
                    }
                }
            }
            return result;
        }
    } // namespace

    Matrix4 identity_transform() noexcept {
        return {
            1.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 1.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 1.0f};
    }

    Matrix4 exp_se3(const Twist& twist) noexcept {
        const double wx = twist[3];
        const double wy = twist[4];
        const double wz = twist[5];
        const double theta_squared = wx * wx + wy * wy + wz * wz;

        double a;
        double b;
        double c;
        if (theta_squared < 1.0e-8) {
            const double theta_fourth = theta_squared * theta_squared;
            a = 1.0 - theta_squared / 6.0 + theta_fourth / 120.0;
            b = 0.5 - theta_squared / 24.0 + theta_fourth / 720.0;
            c = 1.0 / 6.0 - theta_squared / 120.0 + theta_fourth / 5040.0;
        } else {
            const double theta = std::sqrt(theta_squared);
            a = std::sin(theta) / theta;
            b = (1.0 - std::cos(theta)) / theta_squared;
            c = (theta - std::sin(theta)) / (theta_squared * theta);
        }

        const Matrix3d omega{
            0.0, -wz, wy,
            wz, 0.0, -wx,
            -wy, wx, 0.0};
        const Matrix3d omega_squared = multiply3(omega, omega);

        Matrix3d rotation{};
        Matrix3d left_jacobian{};
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                const int index = row * 3 + column;
                const double identity = row == column ? 1.0 : 0.0;
                rotation[index] = identity + a * omega[index] + b * omega_squared[index];
                left_jacobian[index] = identity + b * omega[index] + c * omega_squared[index];
            }
        }

        std::array<double, 3> translation{};
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                translation[row] += left_jacobian[row * 3 + column] * twist[column];
            }
        }

        Matrix4 result = identity_transform();
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                result[row * 4 + column] = static_cast<float>(rotation[row * 3 + column]);
            }
            result[row * 4 + 3] = static_cast<float>(translation[row]);
        }
        return result;
    }

    Matrix4 multiply(const Matrix4& lhs, const Matrix4& rhs) noexcept {
        Matrix4 result{};
        for (int row = 0; row < 4; ++row) {
            for (int column = 0; column < 4; ++column) {
                for (int inner = 0; inner < 4; ++inner) {
                    result[row * 4 + column] +=
                        lhs[row * 4 + inner] * rhs[inner * 4 + column];
                }
            }
        }
        return result;
    }

    Matrix4 apply_left_increment(
        const Twist& increment,
        const Matrix4& source_world_to_camera) noexcept {
        return multiply(exp_se3(increment), source_world_to_camera);
    }

    Twist left_increment_gradient(
        const Matrix4& world_to_camera,
        const Matrix4& transform_gradient) noexcept {
        Twist result{};

        // A pure left translation changes only the homogeneous translation
        // column because the last row of an SE(3) transform is [0, 0, 0, 1].
        result[0] = transform_gradient[3];
        result[1] = transform_gradient[7];
        result[2] = transform_gradient[11];

        // d/dw Exp(w)T at w=0 is hat(e_i)T. Include all four columns so
        // rotation of the existing translation contributes to the torque.
        for (int column = 0; column < 4; ++column) {
            result[3] +=
                -transform_gradient[4 + column] * world_to_camera[8 + column] +
                transform_gradient[8 + column] * world_to_camera[4 + column];
            result[4] +=
                transform_gradient[column] * world_to_camera[8 + column] -
                transform_gradient[8 + column] * world_to_camera[column];
            result[5] +=
                -transform_gradient[column] * world_to_camera[4 + column] +
                transform_gradient[4 + column] * world_to_camera[column];
        }
        return result;
    }

} // namespace lfs::training::camera_pose

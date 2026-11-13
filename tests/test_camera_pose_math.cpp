/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "kernels/camera_pose_math.hpp"

#include <array>
#include <cmath>
#include <gtest/gtest.h>

namespace {

    namespace cp = lfs::training::camera_pose;
    using Mat = std::array<double, 16>;
    using Off = std::array<double, 6>;

    Mat start_pose() {
        const double phi[3] = {0.4, -1.1, 0.7};
        double R[9];
        cp::exp_so3(phi, R);
        return {R[0], R[1], R[2], 0.8, R[3], R[4], R[5], -2.5, R[6], R[7], R[8], 3.1, 0, 0, 0, 1};
    }

    // A smooth loss of the pose the rasterizer is given, with its gradient as the rasterizer reports it:
    // W, b and the centre c treated as independent inputs.
    struct TestLoss {
        double value(const double w2c[16], const double c[3]) const {
            double l = 0.0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j)
                    l += (0.3 + 0.1 * i - 0.2 * j) * std::sin(w2c[i * 4 + j]);
                l += (0.5 - 0.3 * i) * w2c[i * 4 + 3] * w2c[i * 4 + 3];
                l += (0.2 + 0.4 * i) * c[i] + 0.05 * c[i] * c[i];
            }
            return l;
        }
        void grad(const double w2c[16], const double c[3], double g_w2c[16], double g_c[3]) const {
            for (int k = 0; k < 16; ++k)
                g_w2c[k] = 0.0;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j)
                    g_w2c[i * 4 + j] = (0.3 + 0.1 * i - 0.2 * j) * std::cos(w2c[i * 4 + j]);
                g_w2c[i * 4 + 3] = 2.0 * (0.5 - 0.3 * i) * w2c[i * 4 + 3];
                g_c[i] = (0.2 + 0.4 * i) + 0.1 * c[i];
            }
        }
    };

    double loss_at(const Mat& w2c0, Off off) {
        double w2c[16], c[3];
        cp::compose(w2c0.data(), off.data(), w2c, c);
        return TestLoss{}.value(w2c, c);
    }

    void expect_gradient_matches(const Off& off) {
        const Mat w2c0 = start_pose();
        double w2c[16], c[3], g_w2c[16], g_c[3], analytic[6];
        cp::compose(w2c0.data(), off.data(), w2c, c);
        TestLoss{}.grad(w2c, c, g_w2c, g_c);
        cp::offset_grad(w2c0.data(), off.data(), g_w2c, g_c, analytic);
        constexpr double h = 1e-6;
        for (int k = 0; k < 6; ++k) {
            Off plus = off, minus = off;
            plus[k] += h;
            minus[k] -= h;
            const double numeric = (loss_at(w2c0, plus) - loss_at(w2c0, minus)) / (2.0 * h);
            EXPECT_NEAR(analytic[k], numeric, 1e-7 * std::max(1.0, std::abs(numeric))) << "component " << k;
        }
    }

} // namespace

TEST(CameraPoseMath, ZeroOffsetIsTheStartPoseExactly) {
    const Mat w2c0 = start_pose();
    const Off zero{};
    double w2c[16], c[3];
    cp::compose(w2c0.data(), zero.data(), w2c, c);
    for (int k = 0; k < 16; ++k)
        EXPECT_EQ(w2c[k], w2c0[k]) << k;
    for (int j = 0; j < 3; ++j)
        EXPECT_NEAR(c[j], -(w2c0[j] * w2c0[3] + w2c0[4 + j] * w2c0[7] + w2c0[8 + j] * w2c0[11]), 1e-12);
}

TEST(CameraPoseMath, OffsetTurnsAndMovesTheCameraRigidly) {
    const Mat w2c0 = start_pose();
    const Off off{0.03, -0.02, 0.05, 0.4, -0.1, 0.25};
    double w2c[16], c[3], c0[3];
    cp::compose(w2c0.data(), off.data(), w2c, c);
    const Off zero{};
    double unused[16];
    cp::compose(w2c0.data(), zero.data(), unused, c0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double dot = 0.0;
            for (int k = 0; k < 3; ++k)
                dot += w2c[i * 4 + k] * w2c[j * 4 + k];
            EXPECT_NEAR(dot, i == j ? 1.0 : 0.0, 1e-12);
        }
    // |d| is the centre's move and |r| the turn from the start orientation.
    double move2 = 0.0;
    for (int j = 0; j < 3; ++j)
        move2 += (c[j] - c0[j]) * (c[j] - c0[j]);
    EXPECT_NEAR(std::sqrt(move2), std::sqrt(off[3] * off[3] + off[4] * off[4] + off[5] * off[5]), 1e-12);
    double trace = 0.0; // trace(W W0^T) = 1 + 2 cos(turn)
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k)
            trace += w2c[i * 4 + k] * w2c0[i * 4 + k];
    EXPECT_NEAR(std::acos((trace - 1.0) / 2.0), std::sqrt(off[0] * off[0] + off[1] * off[1] + off[2] * off[2]), 1e-9);
}

TEST(CameraPoseMath, OffsetGradientMatchesFiniteDifferencesAtZero) {
    expect_gradient_matches(Off{});
}

TEST(CameraPoseMath, OffsetGradientMatchesFiniteDifferencesNearTheSeriesThreshold) {
    expect_gradient_matches(Off{4e-4, -3e-4, 5e-4, 1e-3, 2e-3, -1e-3});
}

TEST(CameraPoseMath, OffsetGradientMatchesFiniteDifferencesAtALargeOffset) {
    expect_gradient_matches(Off{0.35, -0.2, 0.5, 0.7, -0.4, 0.3});
}

TEST(CameraPoseMath, SinglePrecisionFollowsDoublePrecision) {
    const Mat w2c0 = start_pose();
    const Off off{0.01, -0.02, 0.015, 0.03, 0.01, -0.02};
    double w2c[16], c[3], g_w2c[16], g_c[3], expected[6];
    cp::compose(w2c0.data(), off.data(), w2c, c);
    TestLoss{}.grad(w2c, c, g_w2c, g_c);
    cp::offset_grad(w2c0.data(), off.data(), g_w2c, g_c, expected);

    float w2c0_f[16], off_f[6], g_w2c_f[16], g_c_f[3], actual[6];
    for (int k = 0; k < 16; ++k) {
        w2c0_f[k] = static_cast<float>(w2c0[k]);
        g_w2c_f[k] = static_cast<float>(g_w2c[k]);
    }
    for (int k = 0; k < 6; ++k)
        off_f[k] = static_cast<float>(off[k]);
    for (int k = 0; k < 3; ++k)
        g_c_f[k] = static_cast<float>(g_c[k]);
    cp::offset_grad(w2c0_f, off_f, g_w2c_f, g_c_f, actual);
    for (int k = 0; k < 6; ++k)
        EXPECT_NEAR(actual[k], expected[k], 1e-5 * std::max(1.0, std::abs(expected[k]))) << k;
}

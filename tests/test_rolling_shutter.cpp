/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "components/rolling_shutter.hpp"
#include "core/camera.hpp"
#include "core/tensor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

namespace {

    using lfs::training::attach_rolling_shutter;
    using lfs::training::parse_rolling_shutter;
    using lfs::training::rolling_shutter_window_poses;

    constexpr const char* kSidecar = R"({
        "version": 1,
        "model": "linear rolling shutter about the reference pose",
        "readout_axis": "rows",
        "cameras": [{"camera_id": 1, "readout_s": 0.03}, {"camera_id": 2, "readout_s": -0.02}],
        "images": [
            {"image_id": 7, "name": "cam1/a.jpg", "camera_id": 1,
             "rotation_rad": [0.01, -0.02, 0.005], "velocity_m_s": [1.0, 2.0, -3.0],
             "translation_m": [0.03, 0.06, -0.09]},
            {"image_id": 8, "name": "cam2/b.jpg",
             "rotation_rad": [0.0, 0.0, 0.0], "velocity_m_s": [5.0, 0.0, 0.0],
             "translation_m": [-0.1, 0.0, 0.0]}
        ]
    })";

    std::string replaced(std::string text, const std::string& from, const std::string& to) {
        text.replace(text.find(from), from.size(), to);
        return text;
    }

    std::shared_ptr<lfs::core::Camera> make_camera(const std::string& name, const int camera_id) {
        return std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
            100.0f, 100.0f, 32.0f, 32.0f,
            lfs::core::Tensor(), lfs::core::Tensor(), lfs::core::CameraModelType::FISHEYE,
            name, std::filesystem::path{}, std::filesystem::path{}, 64, 64, 0, camera_id);
    }

    using Pose = std::array<float, 16>;

    std::array<double, 3> transform_point(const Pose& pose, const std::array<double, 3>& x) {
        std::array<double, 3> out{};
        for (int r = 0; r < 3; ++r)
            out[r] = pose[r * 4] * x[0] + pose[r * 4 + 1] * x[1] + pose[r * 4 + 2] * x[2] + pose[r * 4 + 3];
        return out;
    }

    // GeosunPre's first-order model: (I + s [w]x) (R X + t - s d).
    std::array<double, 3> first_order(const lfs::core::RollingShutterMotion& motion,
                                      const std::array<float, 9>& R, const std::array<float, 3>& t,
                                      const double s, const std::array<double, 3>& x) {
        std::array<double, 3> p{};
        for (int r = 0; r < 3; ++r)
            p[r] = R[r * 3] * x[0] + R[r * 3 + 1] * x[1] + R[r * 3 + 2] * x[2] + t[r] - s * motion.displacement[r];
        const std::array<double, 3> w = {s * motion.rotation[0], s * motion.rotation[1], s * motion.rotation[2]};
        return {p[0] + w[1] * p[2] - w[2] * p[1],
                p[1] + w[2] * p[0] - w[0] * p[2],
                p[2] + w[0] * p[1] - w[1] * p[0]};
    }

    double angle_between(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        const double dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        const double na = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        const double nb = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]);
        return std::acos(std::clamp(dot / (na * nb), -1.0, 1.0));
    }

    double norm(const std::array<double, 3>& a) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }

    std::array<double, 3> cross(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    }

    std::array<double, 3> as_double(const std::array<float, 3>& a) { return {a[0], a[1], a[2]}; }

    using Quat = std::array<double, 4>; // w, x, y, z

    Quat to_quat(const Pose& m) {
        const double w = std::sqrt(std::max(0.0, 1.0 + m[0] + m[5] + m[10])) / 2.0;
        return {w, (m[9] - m[6]) / (4 * w), (m[2] - m[8]) / (4 * w), (m[4] - m[1]) / (4 * w)};
    }

    // gsplat's RollingShutterParameters + interpolate_shutter_pose: slerp the rotation,
    // lerp the translation.
    Pose gsplat_interpolate(const lfs::training::RollingShutterWindowPoses& poses, const double t) {
        const Quat a = to_quat(poses.first);
        Quat b = to_quat(poses.last);
        double dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
        if (dot < 0) {
            for (auto& v : b)
                v = -v;
            dot = -dot;
        }
        const double theta = std::acos(std::min(1.0, dot));
        Quat q{};
        for (int i = 0; i < 4; ++i)
            q[i] = theta < 1e-12 ? a[i]
                                 : (std::sin((1 - t) * theta) * a[i] + std::sin(t * theta) * b[i]) / std::sin(theta);
        const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        const double qw = q[0] / n, qx = q[1] / n, qy = q[2] / n, qz = q[3] / n;
        Pose out{};
        const double m[9] = {1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw),
                             2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw),
                             2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)};
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c)
                out[r * 4 + c] = static_cast<float>(m[r * 3 + c]);
            out[r * 4 + 3] = static_cast<float>((1 - t) * poses.first[r * 4 + 3] + t * poses.last[r * 4 + 3]);
        }
        out[15] = 1.0f;
        return out;
    }

    lfs::core::RollingShutterMotion sample_motion() {
        lfs::core::RollingShutterMotion motion;
        motion.rotation = {0.01f, -0.02f, 0.005f};
        motion.displacement = {0.03f, 0.06f, -0.09f};
        return motion;
    }

    // A rotation about z by 0.3 rad and a translation, as COLMAP would store them.
    constexpr std::array<float, 9> kR = {0.9553365f, -0.2955202f, 0.0f,
                                         0.2955202f, 0.9553365f, 0.0f,
                                         0.0f, 0.0f, 1.0f};
    constexpr std::array<float, 3> kT = {0.5f, -1.0f, 2.0f};

} // namespace

TEST(RollingShutterSidecar, ParsesVersionOne) {
    const auto table = parse_rolling_shutter(kSidecar);
    ASSERT_TRUE(table) << table.error();
    EXPECT_TRUE(table->readout_rows);
    EXPECT_DOUBLE_EQ(table->readout_s.at(1), 0.03);
    EXPECT_DOUBLE_EQ(table->readout_s.at(2), -0.02);
    ASSERT_EQ(table->images.size(), 2u);
    EXPECT_EQ(table->images.at("cam1/a.jpg").camera_id, 1);
    EXPECT_EQ(table->images.at("cam2/b.jpg").camera_id, -1);
    EXPECT_DOUBLE_EQ(table->images.at("cam1/a.jpg").rotation[1], -0.02);

    const auto columns = parse_rolling_shutter(replaced(kSidecar, "\"rows\"", "\"columns\""));
    ASSERT_TRUE(columns) << columns.error();
    EXPECT_FALSE(columns->readout_rows);
}

TEST(RollingShutterSidecar, RejectsUnsupportedOrInconsistentFiles) {
    EXPECT_FALSE(parse_rolling_shutter(replaced(kSidecar, "\"version\": 1", "\"version\": 2")));
    EXPECT_FALSE(parse_rolling_shutter(replaced(kSidecar, "about the reference pose", "between two poses")));
    EXPECT_FALSE(parse_rolling_shutter(replaced(kSidecar, "\"rows\"", "\"diagonal\"")));
    EXPECT_FALSE(parse_rolling_shutter(replaced(kSidecar, "[0.01, -0.02, 0.005]", "[0.06, 0.0, 0.0]")));
    EXPECT_FALSE(parse_rolling_shutter(replaced(kSidecar, "\"readout_s\": 0.03", "\"readout_s\": 0.08")));
    EXPECT_FALSE(parse_rolling_shutter(replaced(kSidecar, "\"cam2/b.jpg\"", "\"cam1/a.jpg\"")));
    EXPECT_FALSE(parse_rolling_shutter(replaced(kSidecar, "\"camera_id\": 1,", "\"camera_id\": 3,")));
    EXPECT_FALSE(parse_rolling_shutter(replaced(kSidecar, "[0.01, -0.02, 0.005]", "[0.01, -0.02]")));
    EXPECT_FALSE(parse_rolling_shutter("not json"));
}

TEST(RollingShutterSidecar, AttachesMotionByImageName) {
    const auto table = parse_rolling_shutter(kSidecar);
    ASSERT_TRUE(table) << table.error();
    const std::vector cameras = {make_camera("cam1/a.jpg", 1), make_camera("cam2/b.jpg", 2)};
    ASSERT_TRUE(attach_rolling_shutter(*table, cameras));

    const auto& a = cameras[0]->rolling_shutter();
    ASSERT_TRUE(a);
    EXPECT_TRUE(a->readout_rows);
    EXPECT_FLOAT_EQ(a->rotation[2], 0.005f);
    EXPECT_FLOAT_EQ(a->displacement[0], 0.03f);
    EXPECT_FLOAT_EQ(a->displacement[2], -0.09f);
    // The sign of readout_s carries the scan order into the displacement.
    const auto& b = cameras[1]->rolling_shutter();
    ASSERT_TRUE(b);
    EXPECT_FLOAT_EQ(b->displacement[0], -0.1f);

    lfs::training::clear_rolling_shutter(cameras);
    EXPECT_FALSE(cameras[0]->rolling_shutter());
}

TEST(RollingShutterSidecar, RejectsCamerasTheSidecarDoesNotDescribe) {
    const auto table = parse_rolling_shutter(kSidecar);
    ASSERT_TRUE(table) << table.error();
    EXPECT_FALSE(attach_rolling_shutter(*table, std::vector{make_camera("cam1/missing.jpg", 1)}));
    EXPECT_FALSE(attach_rolling_shutter(*table, std::vector{make_camera("cam1/a.jpg", 2)}));
    EXPECT_FALSE(attach_rolling_shutter(*table, std::vector{make_camera("cam2/b.jpg", 9)}));

    const auto inconsistent = parse_rolling_shutter(replaced(kSidecar, "[-0.1, 0.0, 0.0]", "[0.1, 0.0, 0.0]"));
    ASSERT_TRUE(inconsistent) << inconsistent.error();
    EXPECT_FALSE(attach_rolling_shutter(*inconsistent, std::vector{make_camera("cam2/b.jpg", 2)}));
}

TEST(RollingShutterPoses, ZeroMotionKeepsTheReferencePose) {
    const auto poses = rolling_shutter_window_poses({}, kR, kT, 480, 0, 480);
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            EXPECT_FLOAT_EQ(poses.first[r * 4 + c], kR[r * 3 + c]);
            EXPECT_FLOAT_EQ(poses.last[r * 4 + c], kR[r * 3 + c]);
        }
        EXPECT_FLOAT_EQ(poses.first[r * 4 + 3], kT[r]);
        EXPECT_FLOAT_EQ(poses.last[r * 4 + 3], kT[r]);
    }
}

TEST(RollingShutterPoses, GsplatInterpolationFollowsTheFirstOrderModel) {
    const auto motion = sample_motion();
    constexpr int N = 480;
    const auto poses = rolling_shutter_window_poses(motion, kR, kT, N, 0, N);
    const double w = std::sqrt(0.01 * 0.01 + 0.02 * 0.02 + 0.005 * 0.005);
    const auto wd = cross(as_double(motion.rotation), as_double(motion.displacement));
    for (const auto& x : {std::array<double, 3>{1.0, -2.0, 6.0}, std::array<double, 3>{-3.0, 1.0, 1.5}}) {
        for (int line = 0; line < N; line += 17) {
            const double s = (line + 0.5) / N - 0.5;
            const auto ours = transform_point(gsplat_interpolate(poses, static_cast<double>(line) / (N - 1)), x);
            const auto reference = first_order(motion, kR, kT, s, x);
            // The rotation differs by (s |w|)^2 / 2 and the translation by
            // |s^2 - 1/8| |w x d| <= |w x d| / 8.
            const double bound = 0.5 * (s * w) * (s * w) + norm(wd) / 8.0 / norm(reference);
            EXPECT_LT(angle_between(ours, reference), bound + 1e-6) << "line " << line;
        }
    }
}

TEST(RollingShutterPoses, WindowEndsMatchTheirFullImageLines) {
    const auto motion = sample_motion();
    constexpr int N = 480;
    const auto window = rolling_shutter_window_poses(motion, kR, kT, N, 160, 32);
    const auto first = rolling_shutter_window_poses(motion, kR, kT, N, 160, 1).first;
    const auto last = rolling_shutter_window_poses(motion, kR, kT, N, 191, 1).first;
    for (int i = 0; i < 16; ++i) {
        EXPECT_FLOAT_EQ(window.first[i], first[i]);
        EXPECT_FLOAT_EQ(window.last[i], last[i]);
    }
}

TEST(RollingShutterPoses, FullWindowIsSymmetricAboutTheReference) {
    // gsplat projects splat centres at the midpoint of the two poses; for the whole
    // image that is the reference pose up to second order.
    const auto motion = sample_motion();
    const auto poses = rolling_shutter_window_poses(motion, kR, kT, 480, 0, 480);
    for (int r = 0; r < 3; ++r)
        EXPECT_NEAR(0.5 * (poses.first[r * 4 + 3] + poses.last[r * 4 + 3]), kT[r], 1e-3);
}

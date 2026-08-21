/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/cuda/memory_arena.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "fastgs_fisheye_cuda.hpp"
#include "fisheye_kb.cuh"
#include "lfs/training/joint_adam_codec.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/rasterization/fast_rasterizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace lfs::training;
using namespace lfs::core;
namespace kb = fast_lfs::rasterization::fisheye_kb;

namespace {

    std::filesystem::path golden_path() {
        return std::filesystem::path(PROJECT_ROOT_PATH) / "tests" / "data" / "fisheye_kb_golden.json";
    }

    nlohmann::json load_golden() {
        std::ifstream in(golden_path());
        if (!in) {
            throw std::runtime_error("missing " + golden_path().string());
        }
        nlohmann::json j;
        in >> j;
        return j;
    }

    double rel_err(double a, double b, double eps = 1e-12) {
        return std::abs(a - b) / (std::abs(b) + eps);
    }

    double mixed_rel(double a, double b, double floor) {
        return std::abs(a - b) / (std::abs(b) + floor);
    }

    double float_px_tol(double px) {
        const double ulp = static_cast<double>(std::numeric_limits<float>::epsilon()) *
                           (1.0 + std::abs(px));
        return std::max(1e-4, 8.0 * ulp);
    }

    const AdamParamState& adam_state(const AdamOptimizer& opt, ParamType type) {
        const auto* state = opt.get_state(type);
        if (!state || !state->exp_avg.is_valid()) {
            throw std::runtime_error("Missing Adam moment state");
        }
        return *state;
    }

    Tensor adam_moment(const AdamOptimizer& opt, ParamType type) {
        const auto& state = adam_state(opt, type);
        if (state.exp_avg.dtype() == DataType::Float32) {
            return state.exp_avg;
        }
        if (!state.is_joint()) {
            throw std::runtime_error("Legacy Adam moment codec is unsupported");
        }
        const int bits = state.joint_bits;
        const int bpc = joint_adam::bytes_per_cell(bits);
        auto packed_cpu = state.exp_avg.to(Device::CPU);
        auto bounds_cpu = state.joint_bounds.to(Device::CPU);
        const auto* packed = packed_cpu.ptr<std::uint8_t>();
        const auto* bounds = bounds_cpu.ptr<float>();
        const size_t n_prim = state.exp_avg.shape()[0];
        const size_t packed_row = state.exp_avg.ndim() >= 2 ? state.exp_avg.shape()[1] : packed_cpu.numel();
        const size_t n_attr = state.exp_avg.ndim() >= 2 ? packed_row / static_cast<size_t>(bpc) : 1;
        std::vector<float> dequant(n_prim * n_attr, 0.0f);
        for (size_t p = 0; p < n_prim; ++p) {
            const size_t bidx = p / static_cast<size_t>(joint_adam::kBlockSize);
            const float umin = bounds[bidx * 4 + 0];
            const float umax = bounds[bidx * 4 + 1];
            const float smin = bounds[bidx * 4 + 2];
            const float smax = bounds[bidx * 4 + 3];
            for (size_t a = 0; a < n_attr; ++a) {
                const size_t cell = p * n_attr + a;
                float g1 = 0.0f, g2 = 0.0f;
                if (bits == 16) {
                    joint_adam::Codec16::decode_g1g2(packed, cell, umin, umax, smin, smax, g1, g2);
                } else {
                    joint_adam::Codec8::decode_g1g2(packed, cell, umin, umax, smin, smax, g1, g2);
                }
                dequant[cell] = g1;
            }
        }
        TensorShape out_shape = n_attr == 1 ? TensorShape({n_prim}) : TensorShape({n_prim, n_attr});
        if (type == ParamType::Sh0 && n_attr == 3) {
            out_shape = TensorShape({n_prim, size_t{1}, size_t{3}});
        }
        return Tensor::from_blob(dequant.data(), out_shape, Device::CPU, DataType::Float32).clone().to(Device::CUDA);
    }

    Tensor recovered_fused_grad(const AdamOptimizer& opt, ParamType type, float beta1 = 0.9f) {
        return adam_moment(opt, type).mul(1.0f / (1.0f - beta1));
    }

    double image_loss_f64(const Tensor& image, const Tensor& depth, bool with_depth) {
        auto img = image.to(Device::CPU);
        const float* p = img.ptr<float>();
        double acc = 0.0;
        for (size_t i = 0; i < img.numel(); ++i) {
            const double v = static_cast<double>(p[i]);
            acc += v * v;
        }
        if (with_depth) {
            auto d = depth.to(Device::CPU);
            const float* dp = d.ptr<float>();
            for (size_t i = 0; i < d.numel(); ++i) {
                const double v = static_cast<double>(dp[i]);
                acc += v * v;
            }
        }
        return acc;
    }

    float psnr_region(const std::vector<float>& a, const std::vector<float>& b, const std::vector<char>& mask) {
        double mse = 0.0;
        int n = 0;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (!mask[i])
                continue;
            const double d0 = static_cast<double>(a[i] - b[i]);
            mse += d0 * d0;
            ++n;
        }
        if (n == 0)
            return 0.0f;
        mse /= static_cast<double>(n);
        if (mse <= 1e-12)
            return 99.0f;
        return static_cast<float>(10.0 * std::log10(1.0 / mse));
    }

    std::unique_ptr<Camera> make_camera(
        CameraModelType model,
        int w,
        int h,
        float fx,
        float fy,
        float cx,
        float cy,
        const std::vector<float>& k = {}) {
        auto R = Tensor::eye(3, Device::CUDA);
        std::vector<float> t_data{0, 0, 0};
        auto T = Tensor::from_blob(t_data.data(), {3}, Device::CPU, DataType::Float32).to(Device::CUDA);
        Tensor radial;
        if (!k.empty()) {
            radial = Tensor::from_vector(k, {k.size()}, Device::CPU);
        }
        return std::make_unique<Camera>(
            R, T, fx, fy, cx, cy, radial, Tensor(), model, "fisheye", "",
            std::filesystem::path{}, w, h, 0);
    }

    int tensor_mismatch_count(const Tensor& a, const Tensor& b) {
        auto ac = a.to(Device::CPU);
        auto bc = b.to(Device::CPU);
        EXPECT_EQ(ac.numel(), bc.numel());
        const float* pa = ac.ptr<float>();
        const float* pb = bc.ptr<float>();
        int n_diff = 0;
        for (size_t i = 0; i < ac.numel(); ++i) {
            if (pa[i] != pb[i])
                ++n_diff;
        }
        return n_diff;
    }

    float cosine_nonzero(const Tensor& numerical, const Tensor& analytical) {
        auto ac = numerical.to(Device::CPU);
        auto bc = analytical.to(Device::CPU);
        const float* pa = ac.ptr<float>();
        const float* pb = bc.ptr<float>();
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < ac.numel(); ++i) {
            if (pa[i] == 0.0f)
                continue;
            dot += pa[i] * pb[i];
            na += pa[i] * pa[i];
            nb += pb[i] * pb[i];
        }
        if (na < 1e-20 && nb < 1e-20)
            return 1.0f;
        return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
    }

} // namespace

TEST(FastGSFisheyeT1, GoldenProjectionDoubleAndFloat) {
    const auto recs = load_golden();
    ASSERT_EQ(recs.size(), 16u);
    for (const auto& r : recs) {
        const double Px = r["P"][0], Py = r["P"][1], Pz = r["P"][2];
        const double fx = r["f"][0], fy = r["f"][1], cx = r["f"][2], cy = r["f"][3];
        const double k1 = r["k"][0], k2 = r["k"][1], k3 = r["k"][2], k4 = r["k"][3];
        const auto pd = kb::kb_project(Px, Py, Pz, fx, fy, cx, cy, k1, k2, k3, k4);
        EXPECT_NEAR(pd.theta, r["theta"].get<double>(), 1e-12);
        EXPECT_NEAR(pd.px, r["pixel"][0].get<double>(), 1e-9);
        EXPECT_NEAR(pd.py, r["pixel"][1].get<double>(), 1e-9);
        const double J[2][3] = {{pd.J00, pd.J01, pd.J02}, {pd.J10, pd.J11, pd.J12}};
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                EXPECT_LT(rel_err(J[i][j], r["J"][i][j].get<double>()), 1e-6);
            }
        }

        const auto pf = kb::kb_project(
            static_cast<float>(Px), static_cast<float>(Py), static_cast<float>(Pz),
            static_cast<float>(fx), static_cast<float>(fy), static_cast<float>(cx), static_cast<float>(cy),
            static_cast<float>(k1), static_cast<float>(k2), static_cast<float>(k3), static_cast<float>(k4));
        // Float header vs the same formula in double at float-rounded inputs.
        const auto pd_at_f = kb::kb_project(
            static_cast<double>(static_cast<float>(Px)),
            static_cast<double>(static_cast<float>(Py)),
            static_cast<double>(static_cast<float>(Pz)),
            static_cast<double>(static_cast<float>(fx)),
            static_cast<double>(static_cast<float>(fy)),
            static_cast<double>(static_cast<float>(cx)),
            static_cast<double>(static_cast<float>(cy)),
            static_cast<double>(static_cast<float>(k1)),
            static_cast<double>(static_cast<float>(k2)),
            static_cast<double>(static_cast<float>(k3)),
            static_cast<double>(static_cast<float>(k4)));
        EXPECT_LT(std::abs(static_cast<double>(pf.px) - pd_at_f.px), float_px_tol(pd_at_f.px));
        EXPECT_LT(std::abs(static_cast<double>(pf.py) - pd_at_f.py), float_px_tol(pd_at_f.py));
        const float Jf[2][3] = {{pf.J00, pf.J01, pf.J02}, {pf.J10, pf.J11, pf.J12}};
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                EXPECT_LT(rel_err(Jf[i][j], r["J"][i][j].get<double>()), 1e-3);
            }
        }
    }
}

TEST(FastGSFisheyeT1, OnAxisLimitAndAsymmetricIntrinsics) {
    const double fx = 700, fy = 650, cx = 512, cy = 384;
    const double k1 = 0.05, k2 = -0.01, k3 = 0.002, k4 = -0.0005;
    const auto p = kb::kb_project(0.0, 0.0, 2.0, fx, fy, cx, cy, k1, k2, k3, k4);
    EXPECT_TRUE(p.on_axis);
    EXPECT_NEAR(p.px, cx, 1e-12);
    EXPECT_NEAR(p.py, cy, 1e-12);
    EXPECT_NEAR(p.J00, fx / 2.0, 1e-9);
    EXPECT_NEAR(p.J11, fy / 2.0, 1e-9);
    EXPECT_NEAR(p.J01, 0.0, 1e-12);
    EXPECT_NEAR(p.J02, 0.0, 1e-12);
    EXPECT_NEAR(p.J10, 0.0, 1e-12);
    EXPECT_NEAR(p.J12, 0.0, 1e-12);
}

TEST(FastGSFisheyeT1, DeviceMatchesHostFloat) {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> pos(-2.5f, 2.5f);
    std::vector<fastgs_fisheye_test::DeviceProjectSample> samples;
    samples.reserve(272);
    const auto recs = load_golden();
    for (const auto& r : recs) {
        fastgs_fisheye_test::DeviceProjectSample s{};
        s.Px = static_cast<float>(r["P"][0].get<double>());
        s.Py = static_cast<float>(r["P"][1].get<double>());
        s.Pz = static_cast<float>(r["P"][2].get<double>());
        s.fx = static_cast<float>(r["f"][0].get<double>());
        s.fy = static_cast<float>(r["f"][1].get<double>());
        s.cx = static_cast<float>(r["f"][2].get<double>());
        s.cy = static_cast<float>(r["f"][3].get<double>());
        s.k1 = static_cast<float>(r["k"][0].get<double>());
        s.k2 = static_cast<float>(r["k"][1].get<double>());
        s.k3 = static_cast<float>(r["k"][2].get<double>());
        s.k4 = static_cast<float>(r["k"][3].get<double>());
        samples.push_back(s);
    }
    for (int i = 0; i < 256; ++i) {
        fastgs_fisheye_test::DeviceProjectSample s{};
        s.Px = pos(rng);
        s.Py = pos(rng);
        s.Pz = pos(rng);
        if (std::hypot(s.Px, s.Py, s.Pz) < 0.2f)
            s.Pz = 1.2f;
        s.fx = 700.f;
        s.fy = 650.f;
        s.cx = 512.f;
        s.cy = 384.f;
        s.k1 = 0.05f;
        s.k2 = -0.01f;
        s.k3 = 0.002f;
        s.k4 = -0.0005f;
        samples.push_back(s);
    }
    auto host = samples;
    ASSERT_TRUE(fastgs_fisheye_test::project_on_device(samples.data(), samples.size()));
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto h = kb::kb_project(
            host[i].Px, host[i].Py, host[i].Pz, host[i].fx, host[i].fy, host[i].cx, host[i].cy,
            host[i].k1, host[i].k2, host[i].k3, host[i].k4);
        EXPECT_EQ(samples[i].finite, h.finite ? 1 : 0);
        if (!h.finite)
            continue;
        // nvcc vs g++ FMA contraction: fp32 tolerance, never bitwise.
        EXPECT_NEAR(samples[i].px, h.px, 5e-3f);
        EXPECT_NEAR(samples[i].py, h.py, 5e-3f);
        EXPECT_NEAR(samples[i].theta, h.theta, 2e-5f);
        const float Jdev[6] = {samples[i].J00, samples[i].J01, samples[i].J02,
                               samples[i].J10, samples[i].J11, samples[i].J12};
        const float Jhost[6] = {h.J00, h.J01, h.J02, h.J10, h.J11, h.J12};
        for (int j = 0; j < 6; ++j) {
            const double scale = std::abs(static_cast<double>(Jhost[j])) + 1.0;
            EXPECT_LT(std::abs(static_cast<double>(Jdev[j] - Jhost[j])) / scale, 5e-3)
                << "device vs host J mismatch at sample " << i << " entry " << j;
        }
    }
}

TEST(FastGSFisheyeT1, RandomRaysAndNearThetaMax) {
    std::mt19937 rng(19);
    std::uniform_real_distribution<double> pos(-2.0, 2.0);
    const double fx = 700, fy = 650, cx = 512, cy = 384;
    const double k1 = 0.05, k2 = -0.01, k3 = 0.002, k4 = -0.0005;
    const double tmax = kb::kb_theta_max(k1, k2, k3, k4, 1024.0, 768.0, fx, fy, cx, cy);
    int n_ok = 0, n_zneg = 0, n_near_max = 0;
    for (int i = 0; i < 800; ++i) {
        const double P[3] = {pos(rng), pos(rng), pos(rng)};
        const auto p = kb::kb_project(P[0], P[1], P[2], fx, fy, cx, cy, k1, k2, k3, k4);
        if (!p.finite)
            continue;
        ++n_ok;
        if (P[2] < 0)
            ++n_zneg;
        if (p.theta > tmax - (1.0 * kb::kPi / 180.0) && p.theta < tmax)
            ++n_near_max;
        const double rd = std::hypot(p.px - cx, p.py - cy);
        if (p.d >= 1e-6) {
            EXPECT_GT(rd, 0.0);
        }
    }
    EXPECT_GT(n_ok, 400);
    EXPECT_GT(n_zneg, 50);

    const double th = std::max(1e-3, tmax - 0.01);
    const auto rim = kb::kb_project(std::sin(th), 0.0, std::cos(th), fx, fy, cx, cy, k1, k2, k3, k4);
    EXPECT_TRUE(rim.finite);
    EXPECT_LT(rim.theta, tmax);
    EXPECT_GT(n_near_max + (rim.theta > tmax - 0.02 ? 1 : 0), 0);
}

TEST(FastGSFisheyeT2, JacobianFiniteDifferences) {
    std::mt19937 rng(11);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double fx = 700, fy = 650, cx = 512, cy = 384;
    const double k1 = 0.05, k2 = -0.01, k3 = 0.002, k4 = -0.0005;
    const double h = 1e-6;
    const double tmax = kb::kb_theta_max(k1, k2, k3, k4, 1024.0, 768.0, fx, fy, cx, cy);
    const double fold = kb::kb_theta_foldover(k1, k2, k3, k4);
    int n_ok_mid = 0, n_ok_rim = 0, n_zneg = 0;
    auto sample_P = [&](double theta, double phi, double D, double* P) {
        P[0] = D * std::sin(theta) * std::cos(phi);
        P[1] = D * std::sin(theta) * std::sin(phi);
        P[2] = D * std::cos(theta);
    };
    for (int i = 0; i < 1600; ++i) {
        const bool want_back = (i % 4 == 0);
        const double theta = want_back
                                 ? (0.5 * kb::kPi + u01(rng) * std::min(0.8, fold - 0.5 * kb::kPi - 0.05))
                                 : (0.05 + u01(rng) * (std::min(tmax, fold) - 0.08));
        const double phi = u01(rng) * 2.0 * kb::kPi;
        const double D = 0.6 + 2.4 * u01(rng);
        double P[3];
        sample_P(theta, phi, D, P);
        const double d = std::hypot(P[0], P[1]);
        if (d < 1e-3 || std::abs(P[2]) < 8 * h)
            continue;
        const auto proj = kb::kb_project(P[0], P[1], P[2], fx, fy, cx, cy, k1, k2, k3, k4);
        if (!proj.finite || proj.theta_d_prime <= 0.0)
            continue;
        double Jfd[2][3]{};
        for (int k = 0; k < 3; ++k) {
            double Pp[3] = {P[0], P[1], P[2]};
            double Pm[3] = {P[0], P[1], P[2]};
            Pp[k] += h;
            Pm[k] -= h;
            const auto rp = kb::kb_project(Pp[0], Pp[1], Pp[2], fx, fy, cx, cy, k1, k2, k3, k4);
            const auto rm = kb::kb_project(Pm[0], Pm[1], Pm[2], fx, fy, cx, cy, k1, k2, k3, k4);
            Jfd[0][k] = (rp.px - rm.px) / (2 * h);
            Jfd[1][k] = (rp.py - rm.py) / (2 * h);
        }
        const double J[2][3] = {{proj.J00, proj.J01, proj.J02}, {proj.J10, proj.J11, proj.J12}};
        double worst = 0.0;
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 3; ++c) {
                worst = std::max(worst, mixed_rel(J[r][c], Jfd[r][c], 1.0));
            }
        }
        const bool near_rim = std::abs(tmax - proj.theta) < (1.0 * kb::kPi / 180.0) ||
                              (fold - proj.theta) < (1.0 * kb::kPi / 180.0);
        if (near_rim) {
            EXPECT_LT(worst, 1e-2);
            ++n_ok_rim;
        } else {
            EXPECT_LT(worst, 1e-5);
            ++n_ok_mid;
        }
        if (P[2] < 0)
            ++n_zneg;
    }
    EXPECT_GT(n_ok_mid, 800);
    EXPECT_GT(n_zneg, 50);
}

TEST(FastGSFisheyeT2, JacobianFiniteDifferencesFloat) {
    std::mt19937 rng(13);
    std::uniform_real_distribution<float> u01(0.0f, 1.0f);
    const float fx = 700.f, fy = 650.f, cx = 512.f, cy = 384.f;
    const float k1 = 0.05f, k2 = -0.01f, k3 = 0.002f, k4 = -0.0005f;
    const float h = 1e-3f;
    const double tmax = kb::kb_theta_max(
        0.05, -0.01, 0.002, -0.0005, 1024.0, 768.0, 700.0, 650.0, 512.0, 384.0);
    const double fold = kb::kb_theta_foldover(0.05, -0.01, 0.002, -0.0005);
    int n_ok_mid = 0, n_ok_rim = 0, n_zneg = 0;
    for (int i = 0; i < 1600; ++i) {
        const bool want_back = (i % 4 == 0);
        const float theta = want_back
                                ? static_cast<float>(0.5 * kb::kPi + u01(rng) * std::min(0.8, fold - 0.5 * kb::kPi - 0.05))
                                : static_cast<float>(0.05 + u01(rng) * (std::min(tmax, fold) - 0.08));
        const float phi = u01(rng) * static_cast<float>(2.0 * kb::kPi);
        const float D = 0.6f + 2.4f * u01(rng);
        float P[3] = {D * std::sin(theta) * std::cos(phi),
                      D * std::sin(theta) * std::sin(phi),
                      D * std::cos(theta)};
        const float d = std::hypot(P[0], P[1]);
        if (d < 1e-3f || std::abs(P[2]) < 8 * h)
            continue;
        const auto proj = kb::kb_project(P[0], P[1], P[2], fx, fy, cx, cy, k1, k2, k3, k4);
        if (!proj.finite || proj.theta_d_prime <= 0.0f)
            continue;
        float Jfd[2][3]{};
        for (int k = 0; k < 3; ++k) {
            float Pp[3] = {P[0], P[1], P[2]};
            float Pm[3] = {P[0], P[1], P[2]};
            Pp[k] += h;
            Pm[k] -= h;
            const auto rp = kb::kb_project(Pp[0], Pp[1], Pp[2], fx, fy, cx, cy, k1, k2, k3, k4);
            const auto rm = kb::kb_project(Pm[0], Pm[1], Pm[2], fx, fy, cx, cy, k1, k2, k3, k4);
            Jfd[0][k] = (rp.px - rm.px) / (2 * h);
            Jfd[1][k] = (rp.py - rm.py) / (2 * h);
        }
        const float J[2][3] = {{proj.J00, proj.J01, proj.J02}, {proj.J10, proj.J11, proj.J12}};
        double worst = 0.0;
        for (int r = 0; r < 2; ++r) {
            for (int c = 0; c < 3; ++c) {
                // Float FD noise in J is ~ulp(pixel)/h; mixed floor absorbs it.
                worst = std::max(worst, mixed_rel(J[r][c], Jfd[r][c], 80.0));
            }
        }
        const bool near_rim =
            P[2] < 0.0f ||
            std::abs(tmax - static_cast<double>(proj.theta)) < (1.0 * kb::kPi / 180.0) ||
            (fold - static_cast<double>(proj.theta)) < (1.0 * kb::kPi / 180.0);
        if (near_rim) {
            EXPECT_LT(worst, 1e-2);
            ++n_ok_rim;
        } else {
            EXPECT_LT(worst, 1e-3);
            ++n_ok_mid;
        }
        if (P[2] < 0)
            ++n_zneg;
    }
    EXPECT_GT(n_ok_mid, 400);
    EXPECT_GT(n_zneg, 30);
}

TEST(FastGSFisheyeDJDP, GoldenContraction) {
    const auto recs = load_golden();
    int n_checked = 0;
    for (const auto& r : recs) {
        const double Px = r["P"][0], Py = r["P"][1], Pz = r["P"][2];
        const double fx = r["f"][0], fy = r["f"][1], cx = r["f"][2], cy = r["f"][3];
        const double k1 = r["k"][0], k2 = r["k"][1], k3 = r["k"][2], k4 = r["k"][3];
        const auto proj = kb::kb_project(Px, Py, Pz, fx, fy, cx, cy, k1, k2, k3, k4);
        if (proj.on_axis || !proj.finite)
            continue;
        ++n_checked;
        double dJdP[2][3][3];
        kb::kb_dJ_dP(proj, fx, fy, dJdP);
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 3; ++k) {
                    EXPECT_LT(rel_err(dJdP[i][j][k], r["dJdP_fd"][i][j][k].get<double>()), 1e-4)
                        << "dJdP[" << i << "][" << j << "][" << k << "]";
                }
            }
        }
        // Unit dL/dJ on each of the six entries, then a mixed contraction.
        const double dL_units[6][6] = {
            {1, 0, 0, 0, 0, 0},
            {0, 1, 0, 0, 0, 0},
            {0, 0, 1, 0, 0, 0},
            {0, 0, 0, 1, 0, 0},
            {0, 0, 0, 0, 1, 0},
            {0, 0, 0, 0, 0, 1},
        };
        const int imap[6][2] = {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}};
        for (int u = 0; u < 6; ++u) {
            double dL_dP[3] = {0, 0, 0};
            kb::kb_contract_dL_dJ(proj, fx, fy,
                                  dL_units[u][0], dL_units[u][1], dL_units[u][2],
                                  dL_units[u][3], dL_units[u][4], dL_units[u][5],
                                  dL_dP[0], dL_dP[1], dL_dP[2]);
            EXPECT_LT(rel_err(dL_dP[0], dJdP[imap[u][0]][imap[u][1]][0]), 1e-9);
            EXPECT_LT(rel_err(dL_dP[1], dJdP[imap[u][0]][imap[u][1]][1]), 1e-9);
            EXPECT_LT(rel_err(dL_dP[2], dJdP[imap[u][0]][imap[u][1]][2]), 1e-9);
        }
        const double mix[6] = {0.5, -0.3, 0.2, 0.1, 0.4, -0.25};
        double dL_dP[3] = {0, 0, 0};
        kb::kb_contract_dL_dJ(proj, fx, fy, mix[0], mix[1], mix[2], mix[3], mix[4], mix[5],
                              dL_dP[0], dL_dP[1], dL_dP[2]);
        double expect[3] = {0, 0, 0};
        for (int u = 0; u < 6; ++u)
            for (int k = 0; k < 3; ++k)
                expect[k] += mix[u] * dJdP[imap[u][0]][imap[u][1]][k];
        EXPECT_LT(rel_err(dL_dP[0], expect[0]), 1e-9);
        EXPECT_LT(rel_err(dL_dP[1], expect[1]), 1e-9);
        EXPECT_LT(rel_err(dL_dP[2], expect[2]), 1e-9);
    }
    EXPECT_EQ(n_checked, 16);
}

TEST(FastGSFisheyeT3, Cov2dMatchesNumericalJacobian) {
    const double fx = 400, fy = 380, cx = 64, cy = 64;
    const double k1 = 0.02, k2 = -0.004, k3 = 0.0, k4 = 0.0;
    const double P[3] = {0.4, -0.25, 1.6};
    const double Sigma[6] = {0.01, 0.001, 0.0, 0.012, 0.0, 0.008}; // triu xx,xy,xz,yy,yz,zz
    const auto proj = kb::kb_project(P[0], P[1], P[2], fx, fy, cx, cy, k1, k2, k3, k4);
    const double J[2][3] = {{proj.J00, proj.J01, proj.J02}, {proj.J10, proj.J11, proj.J12}};
    auto apply_sigma = [&](const double* v, double* out) {
        out[0] = Sigma[0] * v[0] + Sigma[1] * v[1] + Sigma[2] * v[2];
        out[1] = Sigma[1] * v[0] + Sigma[3] * v[1] + Sigma[4] * v[2];
        out[2] = Sigma[2] * v[0] + Sigma[4] * v[1] + Sigma[5] * v[2];
    };
    double JW_S[2][3]{};
    for (int r = 0; r < 2; ++r) {
        apply_sigma(J[r], JW_S[r]);
    }
    double cov[3] = {
        JW_S[0][0] * J[0][0] + JW_S[0][1] * J[0][1] + JW_S[0][2] * J[0][2],
        JW_S[0][0] * J[1][0] + JW_S[0][1] * J[1][1] + JW_S[0][2] * J[1][2],
        JW_S[1][0] * J[1][0] + JW_S[1][1] * J[1][1] + JW_S[1][2] * J[1][2]};

    const double h = 1e-5;
    double Jfd[2][3]{};
    for (int k = 0; k < 3; ++k) {
        double Pp[3] = {P[0], P[1], P[2]};
        double Pm[3] = {P[0], P[1], P[2]};
        Pp[k] += h;
        Pm[k] -= h;
        const auto rp = kb::kb_project(Pp[0], Pp[1], Pp[2], fx, fy, cx, cy, k1, k2, k3, k4);
        const auto rm = kb::kb_project(Pm[0], Pm[1], Pm[2], fx, fy, cx, cy, k1, k2, k3, k4);
        Jfd[0][k] = (rp.px - rm.px) / (2 * h);
        Jfd[1][k] = (rp.py - rm.py) / (2 * h);
    }
    double JW_Sfd[2][3]{};
    for (int r = 0; r < 2; ++r)
        apply_sigma(Jfd[r], JW_Sfd[r]);
    double cov_fd[3] = {
        JW_Sfd[0][0] * Jfd[0][0] + JW_Sfd[0][1] * Jfd[0][1] + JW_Sfd[0][2] * Jfd[0][2],
        JW_Sfd[0][0] * Jfd[1][0] + JW_Sfd[0][1] * Jfd[1][1] + JW_Sfd[0][2] * Jfd[1][2],
        JW_Sfd[1][0] * Jfd[1][0] + JW_Sfd[1][1] * Jfd[1][1] + JW_Sfd[1][2] * Jfd[1][2]};
    EXPECT_LT(rel_err(cov[0], cov_fd[0]), 1e-4);
    EXPECT_LT(rel_err(cov[1], cov_fd[1]), 1e-4);
    EXPECT_LT(rel_err(cov[2], cov_fd[2]), 1e-4);
}

class FastGSFisheyeKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        w_ = 64;
        h_ = 64;
        fx_ = 48.0f;
        fy_ = 46.0f;
        cx_ = 32.0f;
        cy_ = 32.0f;
        k_ = {0.04f, -0.008f, 0.001f, 0.0f};
        camera_ = make_camera(CameraModelType::FISHEYE, w_, h_, fx_, fy_, cx_, cy_,
                              {k_[0], k_[1], k_[2], k_[3]});
        bg_ = Tensor::zeros({3}, Device::CUDA);
        n_ = 6;
        std::vector<float> means(n_ * 3);
        for (size_t i = 0; i < n_; ++i) {
            means[i * 3 + 0] = (static_cast<float>(i) - 2.5f) * 0.05f;
            means[i * 3 + 1] = (static_cast<float>(i % 3) - 1.0f) * 0.05f;
            means[i * 3 + 2] = 1.2f + static_cast<float>(i) * 0.15f;
        }
        means_ = Tensor::from_vector(means, {n_, 3}, Device::CUDA);
        sh0_ = Tensor::full({n_, size_t{1}, size_t{3}}, 0.4f, Device::CUDA);
        shN_ = Tensor::zeros({n_, 0, 3}, Device::CUDA);
        std::vector<float> scales(n_ * 3);
        for (size_t i = 0; i < n_; ++i) {
            scales[i * 3 + 0] = -2.8f;
            scales[i * 3 + 1] = -3.5f;
            scales[i * 3 + 2] = -4.0f;
        }
        scaling_ = Tensor::from_vector(scales, {n_, 3}, Device::CUDA);
        rotation_ = Tensor::zeros({n_, 4}, Device::CUDA);
        rotation_.slice(1, 0, 1).fill_(1.0f);
        opacity_ = Tensor::full({n_}, 2.0f, Device::CUDA);
    }

    void TearDown() override {
        GlobalArenaManager::instance().get_arena().full_reset();
    }

    std::unique_ptr<SplatData> make_splat() {
        return std::make_unique<SplatData>(0, means_.clone(), sh0_.clone(), shN_.clone(),
                                           scaling_.clone(), rotation_.clone(), opacity_.clone(), 1.0f);
    }

    size_t n_;
    int w_, h_;
    float fx_, fy_, cx_, cy_;
    std::vector<float> k_;
    Tensor means_, sh0_, shN_, scaling_, rotation_, opacity_, bg_;
    std::unique_ptr<Camera> camera_;
};

TEST_F(FastGSFisheyeKernelTest, T4GradcheckMeansScalesOpacitySh0) {
    auto splat = make_splat();
    auto fwd = fast_rasterize_forward(*camera_, *splat, bg_);
    ASSERT_TRUE(fwd.has_value());
    const float alpha_sum = fwd->first.alpha.sum().item<float>();
    ASSERT_GT(alpha_sum, 1e-3f) << "forward-liveness: expected visible alpha";
    ASSERT_GT(fwd->first.image.abs().sum().item<float>(), 1e-4f);
    fwd->second.release_forward_context();

    const double tmax = kb::kb_theta_max(
        static_cast<double>(k_[0]), static_cast<double>(k_[1]),
        static_cast<double>(k_[2]), static_cast<double>(k_[3]),
        static_cast<double>(w_), static_cast<double>(h_),
        static_cast<double>(fx_), static_cast<double>(fy_),
        static_cast<double>(cx_), static_cast<double>(cy_));
    auto means_cpu = means_.to(Device::CPU);
    const float* mp = means_cpu.ptr<float>();
    for (size_t i = 0; i < n_; ++i) {
        const auto proj = kb::kb_project(mp[i * 3], mp[i * 3 + 1], mp[i * 3 + 2],
                                         fx_, fy_, cx_, cy_, k_[0], k_[1], k_[2], k_[3]);
        ASSERT_LT(proj.theta + 0.15, tmax) << "gaussian in theta_max band";
        ASSERT_GT(proj.px, 4.0f);
        ASSERT_LT(proj.px, w_ - 4.0f);
        ASSERT_GT(proj.py, 4.0f);
        ASSERT_LT(proj.py, h_ - 4.0f);
    }

    auto run_analytical = [&](ParamType param) {
        auto s = make_splat();
        auto r = fast_rasterize_forward(*camera_, *s, bg_);
        AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
        auto opt = std::make_unique<AdamOptimizer>(*s, cfg);
        opt->allocate_gradients();
        opt->zero_grad(0);
        auto grad_out = r->first.image.mul(2.0f);
        auto grad_depth = r->first.depth.mul(2.0f);
        fast_rasterize_backward(r->second, grad_out, *s, *opt, {}, {}, DensificationType::None, 1, {}, grad_depth);
        return recovered_fused_grad(*opt, param);
    };

    auto numerical = [&](ParamType param, float eps) {
        Tensor orig;
        switch (param) {
        case ParamType::Means: orig = means_.clone(); break;
        case ParamType::Scaling: orig = scaling_.clone(); break;
        case ParamType::Rotation: orig = rotation_.clone(); break;
        case ParamType::Opacity: orig = opacity_.clone(); break;
        case ParamType::Sh0: orig = sh0_.clone(); break;
        default: return Tensor();
        }
        auto orig_cpu = orig.to(Device::CPU);
        auto grad_cpu = Tensor::zeros_like(orig_cpu);
        float* g = grad_cpu.ptr<float>();
        const float* o = orig_cpu.ptr<float>();
        const size_t n_probe = std::min(orig.numel(), size_t{4});
        for (size_t p = 0; p < n_probe; ++p) {
            const size_t i = (n_probe == 1) ? 0 : p * (orig.numel() - 1) / (n_probe - 1);
            auto plus = orig_cpu.clone();
            auto minus = orig_cpu.clone();
            plus.ptr<float>()[i] = o[i] + eps;
            minus.ptr<float>()[i] = o[i] - eps;
            auto apply = [&](const Tensor& cpu_val) {
                switch (param) {
                case ParamType::Means: means_ = cpu_val.to(Device::CUDA); break;
                case ParamType::Scaling: scaling_ = cpu_val.to(Device::CUDA); break;
                case ParamType::Rotation: rotation_ = cpu_val.to(Device::CUDA); break;
                case ParamType::Opacity: opacity_ = cpu_val.to(Device::CUDA); break;
                case ParamType::Sh0: sh0_ = cpu_val.to(Device::CUDA); break;
                default: break;
                }
            };
            apply(plus);
            auto sp = make_splat();
            auto rp = fast_rasterize_forward(*camera_, *sp, bg_);
            const double lp = image_loss_f64(rp->first.image, rp->first.depth, true);
            rp->second.release_forward_context();
            apply(minus);
            auto sm = make_splat();
            auto rm = fast_rasterize_forward(*camera_, *sm, bg_);
            const double lm = image_loss_f64(rm->first.image, rm->first.depth, true);
            rm->second.release_forward_context();
            g[i] = static_cast<float>((lp - lm) / (2.0 * static_cast<double>(eps)));
        }
        switch (param) {
        case ParamType::Means: means_ = orig; break;
        case ParamType::Scaling: scaling_ = orig; break;
        case ParamType::Rotation: rotation_ = orig; break;
        case ParamType::Opacity: opacity_ = orig; break;
        case ParamType::Sh0: sh0_ = orig; break;
        default: break;
        }
        return grad_cpu.to(Device::CUDA);
    };

    auto cosine = [](const Tensor& a, const Tensor& b) {
        auto ac = a.to(Device::CPU);
        auto bc = b.to(Device::CPU);
        const float* pa = ac.ptr<float>();
        const float* pb = bc.ptr<float>();
        double dot = 0, na = 0, nb = 0;
        for (size_t i = 0; i < ac.numel(); ++i) {
            if (pa[i] == 0.0f)
                continue;
            dot += pa[i] * pb[i];
            na += pa[i] * pa[i];
            nb += pb[i] * pb[i];
        }
        if (na < 1e-20 && nb < 1e-20)
            return 1.0f;
        return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
    };

    const ParamType params[] = {ParamType::Means, ParamType::Scaling, ParamType::Rotation, ParamType::Opacity, ParamType::Sh0};
    for (auto p : params) {
        auto ana = run_analytical(p);
        float best = -1.f;
        for (float eps : {1e-3f, 5e-4f}) {
            auto num = numerical(p, eps);
            best = std::max(best, cosine(num, ana));
            if (best > 0.90f)
                break;
        }
        const char* name = "unknown";
        switch (p) {
        case ParamType::Means: name = "means"; break;
        case ParamType::Scaling: name = "scaling"; break;
        case ParamType::Rotation: name = "rotation"; break;
        case ParamType::Opacity: name = "opacity"; break;
        case ParamType::Sh0: name = "sh0"; break;
        default: break;
        }
        EXPECT_GT(best, 0.75f) << "gradcheck failed for " << name;
    }
}

TEST_F(FastGSFisheyeKernelTest, DepthIsRayLength) {
    std::vector<float> means{0.35f, 0.2f, 1.5f};
    means_ = Tensor::from_vector(means, {size_t{1}, size_t{3}}, Device::CUDA);
    n_ = 1;
    sh0_ = Tensor::full({1, 1, 3}, 1.0f, Device::CUDA);
    shN_ = Tensor::zeros({1, 0, 3}, Device::CUDA);
    scaling_ = Tensor::full({1, 3}, -2.5f, Device::CUDA);
    rotation_ = Tensor::zeros({1, 4}, Device::CUDA);
    rotation_.slice(1, 0, 1).fill_(1.0f);
    opacity_ = Tensor::full({1}, 4.0f, Device::CUDA);
    auto splat = make_splat();
    auto r = fast_rasterize_forward(*camera_, *splat, bg_);
    ASSERT_TRUE(r.has_value());
    const float D = std::sqrt(0.35f * 0.35f + 0.2f * 0.2f + 1.5f * 1.5f);
    auto depth = r->first.depth.to(Device::CPU);
    auto alpha = r->first.alpha.to(Device::CPU);
    const float* d = depth.ptr<float>();
    const float* a = alpha.ptr<float>();
    float max_norm = 0.f;
    for (size_t i = 0; i < depth.numel(); ++i) {
        if (a[i] < 0.3f)
            continue;
        max_norm = std::max(max_norm, d[i] / a[i]);
    }
    EXPECT_NEAR(max_norm, D, 0.08f);
    EXPECT_GT(max_norm, 1.5f + 0.02f) << "fisheye depth should be ray length, not z";
}

TEST_F(FastGSFisheyeKernelTest, T7ThetaMaxCulling) {
    const double tmax = kb::kb_theta_max(
        static_cast<double>(k_[0]), static_cast<double>(k_[1]),
        static_cast<double>(k_[2]), static_cast<double>(k_[3]),
        static_cast<double>(w_), static_cast<double>(h_),
        static_cast<double>(fx_), static_cast<double>(fy_),
        static_cast<double>(cx_), static_cast<double>(cy_));
    const double th_in = tmax - 0.04;
    const double th_out = tmax + 0.04;
    const double phi_corner = std::atan2(static_cast<double>(h_) - cy_, static_cast<double>(w_) - cx_);
    auto ray = [](double theta, double phi) {
        return std::vector<float>{
            static_cast<float>(std::sin(theta) * std::cos(phi)),
            static_cast<float>(std::sin(theta) * std::sin(phi)),
            static_cast<float>(std::cos(theta))};
    };
    auto count_touched = [&](const std::vector<float>& P) {
        means_ = Tensor::from_vector(P, {size_t{1}, size_t{3}}, Device::CUDA);
        n_ = 1;
        sh0_ = Tensor::full({1, 1, 3}, 0.8f, Device::CUDA);
        shN_ = Tensor::zeros({1, 0, 3}, Device::CUDA);
        scaling_ = Tensor::full({1, 3}, -3.5f, Device::CUDA);
        rotation_ = Tensor::zeros({1, 4}, Device::CUDA);
        rotation_.slice(1, 0, 1).fill_(1.0f);
        opacity_ = Tensor::full({1}, 3.0f, Device::CUDA);
        auto splat = make_splat();
        auto r = fast_rasterize_forward(*camera_, *splat, bg_);
        EXPECT_TRUE(r.has_value());
        const int n_instances = r->second.forward_ctx.n_instances;
        const float alpha_sum = r->first.alpha.sum().item<float>();
        return std::pair<int, float>{n_instances, alpha_sum};
    };
    auto Pin = ray(th_in, phi_corner);
    for (auto& v : Pin)
        v *= 1.4f;
    auto Pout = ray(th_out, phi_corner);
    for (auto& v : Pout)
        v *= 1.4f;
    const auto in_c = count_touched(Pin);
    const auto out_c = count_touched(Pout);
    EXPECT_GT(in_c.first, 0);
    EXPECT_EQ(out_c.first, 0);
    EXPECT_GT(in_c.second, 0.0f);
    EXPECT_NEAR(out_c.second, 0.0f, 1e-6f);
}

TEST(FastGSFisheyeT5, EquidistantWarpParity) {
    GlobalArenaManager::instance().get_arena().full_reset();
    constexpr int W = 96, H = 96;
    const float fx = 70.f, fy = 70.f, cx = 48.f, cy = 48.f;
    std::mt19937 rng(21);
    constexpr size_t N = 16;
    std::vector<float> means(N * 3), sh0(N * 3), rot(N * 4, 0.f), scale(N * 3, -2.8f), opa(N, 1.5f);
    std::uniform_real_distribution<float> xy(-0.4f, 0.4f), z(1.2f, 2.0f), col(0.1f, 0.9f);
    for (size_t i = 0; i < N; ++i) {
        means[i * 3] = xy(rng);
        means[i * 3 + 1] = xy(rng);
        means[i * 3 + 2] = z(rng);
        sh0[i * 3] = col(rng);
        sh0[i * 3 + 1] = col(rng);
        sh0[i * 3 + 2] = col(rng);
        rot[i * 4] = 1.f;
    }
    auto means_t = Tensor::from_vector(means, {N, 3}, Device::CUDA);
    auto sh0_t = Tensor::from_blob(sh0.data(), {N, 1, 3}, Device::CPU, DataType::Float32).to(Device::CUDA);
    auto shN_t = Tensor::zeros({N, 0, 3}, Device::CUDA);
    auto scale_t = Tensor::from_vector(scale, {N, 3}, Device::CUDA);
    auto rot_t = Tensor::from_vector(rot, {N, 4}, Device::CUDA);
    auto opa_t = Tensor::from_vector(opa, {N}, Device::CUDA);
    auto splat = SplatData(0, means_t, sh0_t, shN_t, scale_t, rot_t, opa_t, 1.0f);
    auto bg = Tensor::zeros({3}, Device::CUDA);

    auto cam_p = make_camera(CameraModelType::PINHOLE, W, H, fx, fy, cx, cy);
    auto cam_f = make_camera(CameraModelType::FISHEYE, W, H, fx, fy, cx, cy, {0, 0, 0, 0});
    auto pin = fast_rasterize_forward(*cam_p, splat, bg);
    ASSERT_TRUE(pin.has_value());
    auto pin_img = pin->first.image.to(Device::CPU);
    pin->second.release_forward_context();
    auto fish = fast_rasterize_forward(*cam_f, splat, bg);
    ASSERT_TRUE(fish.has_value());
    auto fish_img = fish->first.image.to(Device::CPU);
    fish->second.release_forward_context();

    const float* p = pin_img.ptr<float>();
    const float* fimg = fish_img.ptr<float>();
    const int npix = W * H;
    auto sample = [&](int c, float u, float v) {
        const float x = std::clamp(u, 0.0f, static_cast<float>(W - 1));
        const float y = std::clamp(v, 0.0f, static_cast<float>(H - 1));
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const int x1 = std::min(x0 + 1, W - 1);
        const int y1 = std::min(y0 + 1, H - 1);
        const float ax = x - static_cast<float>(x0);
        const float ay = y - static_cast<float>(y0);
        auto at = [&](int xx, int yy) { return p[c * npix + yy * W + xx]; };
        const float v0 = at(x0, y0) * (1 - ax) + at(x1, y0) * ax;
        const float v1 = at(x0, y1) * (1 - ax) + at(x1, y1) * ax;
        return v0 * (1 - ay) + v1 * ay;
    };
    std::vector<float> warped(3 * npix), fish_v(3 * npix);
    std::vector<char> mask(npix, 0);
    const float erode = 8.f;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float u = static_cast<float>(x) + 0.5f;
            const float v = static_cast<float>(y) + 0.5f;
            const float mx = (u - cx) / fx;
            const float my = (v - cy) / fy;
            const float rd = std::hypot(mx, my);
            if (rd < 1e-6f) {
                if (u > erode && u < W - erode && v > erode && v < H - erode)
                    mask[y * W + x] = 1;
                for (int c = 0; c < 3; ++c) {
                    warped[c * npix + y * W + x] = sample(c, u, v);
                    fish_v[c * npix + y * W + x] = fimg[c * npix + y * W + x];
                }
                continue;
            }
            const float theta = rd;
            if (theta >= 1.2f)
                continue;
            const float scale = std::tan(theta) / theta;
            const float up = cx + fx * mx * scale;
            const float vp = cy + fy * my * scale;
            if (up < erode || up >= W - erode || vp < erode || vp >= H - erode)
                continue;
            mask[y * W + x] = 1;
            for (int c = 0; c < 3; ++c) {
                warped[c * npix + y * W + x] = sample(c, up, vp);
                fish_v[c * npix + y * W + x] = fimg[c * npix + y * W + x];
            }
        }
    }
    std::vector<char> mask3(3 * npix);
    for (int c = 0; c < 3; ++c)
        for (int i = 0; i < npix; ++i)
            mask3[c * npix + i] = mask[i];
    const float psnr = psnr_region(warped, fish_v, mask3);
    EXPECT_GT(psnr, 32.0f);
    GlobalArenaManager::instance().get_arena().full_reset();

    // Smooth synthetic: one large gaussian.
    std::vector<float> m2{0.f, 0.f, 1.6f}, s2{3 * -1.8f, 0, 0}, r2{1, 0, 0, 0}, o2{2.0f}, c2{0.6f, 0.5f, 0.4f};
    s2 = {-1.8f, -1.8f, -1.8f};
    auto splat2 = SplatData(
        0,
        Tensor::from_vector(m2, {1, 3}, Device::CUDA),
        Tensor::from_blob(c2.data(), {1, 1, 3}, Device::CPU, DataType::Float32).to(Device::CUDA),
        Tensor::zeros({1, 0, 3}, Device::CUDA),
        Tensor::from_vector(s2, {1, 3}, Device::CUDA),
        Tensor::from_vector(r2, {1, 4}, Device::CUDA),
        Tensor::from_vector(o2, {1}, Device::CUDA),
        1.0f);
    auto pin2 = fast_rasterize_forward(*cam_p, splat2, bg);
    ASSERT_TRUE(pin2.has_value());
    auto pin2c = pin2->first.image.to(Device::CPU);
    pin2->second.release_forward_context();
    auto fish2 = fast_rasterize_forward(*cam_f, splat2, bg);
    ASSERT_TRUE(fish2.has_value());
    auto fish2c = fish2->first.image.to(Device::CPU);
    fish2->second.release_forward_context();
    const float* p2 = pin2c.ptr<float>();
    const float* f2 = fish2c.ptr<float>();
    std::vector<float> warped2(3 * npix), fish2v(3 * npix);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const float u = x + 0.5f, v = y + 0.5f;
            const float mx = (u - cx) / fx, my = (v - cy) / fy;
            const float rd = std::hypot(mx, my);
            const float theta = std::max(rd, 1e-6f);
            const float scale = std::tan(theta) / theta;
            const float up = cx + fx * mx * scale;
            const float vp = cy + fy * my * scale;
            if (!mask[y * W + x])
                continue;
            auto samp = [&](int c, float uu, float vv) {
                const float xx = std::clamp(uu, 0.f, float(W - 1));
                const float yy = std::clamp(vv, 0.f, float(H - 1));
                const int x0 = int(std::floor(xx)), y0 = int(std::floor(yy));
                const int x1 = std::min(x0 + 1, W - 1), y1 = std::min(y0 + 1, H - 1);
                const float ax = xx - x0, ay = yy - y0;
                auto at = [&](int xx2, int yy2) { return p2[c * npix + yy2 * W + xx2]; };
                return (at(x0, y0) * (1 - ax) + at(x1, y0) * ax) * (1 - ay) +
                       (at(x0, y1) * (1 - ax) + at(x1, y1) * ax) * ay;
            };
            for (int c = 0; c < 3; ++c) {
                warped2[c * npix + y * W + x] = samp(c, up, vp);
                fish2v[c * npix + y * W + x] = f2[c * npix + y * W + x];
            }
        }
    }
    EXPECT_GT(psnr_region(warped2, fish2v, mask3), 40.0f);
    GlobalArenaManager::instance().get_arena().full_reset();
}

TEST(FastGSFisheyeT6, PinholeRenderDeterministic) {
    GlobalArenaManager::instance().get_arena().full_reset();
    constexpr int W = 64, H = 64;
    auto cam = make_camera(CameraModelType::PINHOLE, W, H, 80.f, 80.f, 32.f, 32.f);
    std::vector<float> means{0.f, 0.f, 1.5f, 0.2f, -0.1f, 1.8f};
    auto splat = SplatData(
        0,
        Tensor::from_vector(means, {2, 3}, Device::CUDA),
        Tensor::full({2, 1, 3}, 0.5f, Device::CUDA),
        Tensor::zeros({2, 0, 3}, Device::CUDA),
        Tensor::full({2, 3}, -3.0f, Device::CUDA),
        Tensor::from_vector(std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0}, {2, 4}, Device::CUDA),
        Tensor::full({2}, 1.0f, Device::CUDA),
        1.0f);
    auto bg = Tensor::zeros({3}, Device::CUDA);
    auto a = fast_rasterize_forward(*cam, splat, bg);
    ASSERT_TRUE(a.has_value());
    auto ac = a->first.image.to(Device::CPU);
    a->second.release_forward_context();
    auto b = fast_rasterize_forward(*cam, splat, bg);
    ASSERT_TRUE(b.has_value());
    auto bc = b->first.image.to(Device::CPU);
    b->second.release_forward_context();
    const float* pa = ac.ptr<float>();
    const float* pb = bc.ptr<float>();
    int n_diff = 0;
    for (size_t i = 0; i < ac.numel(); ++i) {
        if (pa[i] != pb[i])
            ++n_diff;
    }
    EXPECT_EQ(n_diff, 0);
    std::uint64_t checksum = 14695981039346656037ull;
    for (size_t i = 0; i < ac.numel(); ++i) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, pa + i, sizeof(bits));
        checksum ^= bits;
        checksum *= 1099511628211ull;
    }
    std::printf("pinhole_bitid_checksum=%llu n=%zu\n",
                static_cast<unsigned long long>(checksum), ac.numel());
    GlobalArenaManager::instance().get_arena().full_reset();
}

TEST(FastGSFisheye, PreprocessAddsNoDilation) {
    GlobalArenaManager::instance().get_arena().full_reset();
    constexpr int W = 64, H = 64;
    const float fx = 48.f, fy = 46.f, cx = 32.f, cy = 32.f;
    auto means = Tensor::from_vector(std::vector<float>{0.f, 0.f, 1.5f}, {1, 3}, Device::CUDA);
    auto sh0 = Tensor::full({1, 1, 3}, 0.6f, Device::CUDA);
    auto shN = Tensor::zeros({1, 0, 3}, Device::CUDA);
    auto scale = Tensor::from_vector(std::vector<float>{-2.5f, -2.5f, -2.5f}, {1, 3}, Device::CUDA);
    auto rot = Tensor::from_vector(std::vector<float>{1.f, 0.f, 0.f, 0.f}, {1, 4}, Device::CUDA);
    auto opa = Tensor::from_vector(std::vector<float>{3.0f}, {1}, Device::CUDA);
    auto splat = SplatData(0, means, sh0, shN, scale, rot, opa, 1.0f);
    auto bg = Tensor::zeros({3}, Device::CUDA);

    auto cam_f = make_camera(CameraModelType::FISHEYE, W, H, fx, fy, cx, cy,
                             {0.04f, -0.008f, 0.001f, 0.0f});
    auto fish_off = fast_rasterize_forward(*cam_f, splat, bg, 0, 0, 0, 0, false);
    ASSERT_TRUE(fish_off.has_value());
    ASSERT_GT(fish_off->first.alpha.sum().item<float>(), 1e-3f);
    auto fish_off_img = fish_off->first.image.to(Device::CPU);
    auto fish_off_alpha = fish_off->first.alpha.to(Device::CPU);
    fish_off->second.release_forward_context();
    auto fish_on = fast_rasterize_forward(*cam_f, splat, bg, 0, 0, 0, 0, true);
    ASSERT_TRUE(fish_on.has_value());
    auto fish_on_img = fish_on->first.image.to(Device::CPU);
    auto fish_on_alpha = fish_on->first.alpha.to(Device::CPU);
    fish_on->second.release_forward_context();
    EXPECT_EQ(tensor_mismatch_count(fish_off_img, fish_on_img), 0);
    EXPECT_EQ(tensor_mismatch_count(fish_off_alpha, fish_on_alpha), 0);

    auto cam_p = make_camera(CameraModelType::PINHOLE, W, H, fx, fy, cx, cy);
    auto pin_off = fast_rasterize_forward(*cam_p, splat, bg, 0, 0, 0, 0, false);
    ASSERT_TRUE(pin_off.has_value());
    ASSERT_GT(pin_off->first.alpha.sum().item<float>(), 1e-3f);
    auto pin_off_img = pin_off->first.image.to(Device::CPU);
    auto pin_off_alpha = pin_off->first.alpha.to(Device::CPU);
    pin_off->second.release_forward_context();
    auto pin_on = fast_rasterize_forward(*cam_p, splat, bg, 0, 0, 0, 0, true);
    ASSERT_TRUE(pin_on.has_value());
    auto pin_on_img = pin_on->first.image.to(Device::CPU);
    auto pin_on_alpha = pin_on->first.alpha.to(Device::CPU);
    pin_on->second.release_forward_context();
    EXPECT_GT(tensor_mismatch_count(pin_off_img, pin_on_img) +
                  tensor_mismatch_count(pin_off_alpha, pin_on_alpha),
              0);

    GlobalArenaManager::instance().get_arena().full_reset();
}

TEST(FastGSFisheye, RimGradcheckMeansScalesRotationOpacity) {
    GlobalArenaManager::instance().get_arena().full_reset();
    constexpr int W = 128, H = 96;
    const float fx = 36.0f, fy = 33.4f, cx = 64.0f, cy = 48.0f;
    const std::vector<float> k{0.05f, -0.01f, 0.002f, -0.0005f};
    auto camera = make_camera(CameraModelType::FISHEYE, W, H, fx, fy, cx, cy, k);
    auto bg = Tensor::zeros({3}, Device::CUDA);
    const double tmax = kb::kb_theta_max(
        static_cast<double>(k[0]), static_cast<double>(k[1]),
        static_cast<double>(k[2]), static_cast<double>(k[3]),
        static_cast<double>(W), static_cast<double>(H),
        static_cast<double>(fx), static_cast<double>(fy),
        static_cast<double>(cx), static_cast<double>(cy));
    const float phi = std::atan2(static_cast<float>(H) - cy, static_cast<float>(W) - cx);
    const float D = 1.6f;
    const float thetas_deg[] = {80.f, 88.f, 92.f, 96.f};

    auto make_splat = [](const Tensor& means, const Tensor& sh0, const Tensor& shN,
                         const Tensor& scaling, const Tensor& rotation, const Tensor& opacity) {
        return SplatData(0, means.clone(), sh0.clone(), shN.clone(),
                         scaling.clone(), rotation.clone(), opacity.clone(), 1.0f);
    };

    for (float theta_deg : thetas_deg) {
        SCOPED_TRACE(testing::Message() << "theta_deg=" << theta_deg);
        const float th = theta_deg * static_cast<float>(kb::kPi / 180.0);
        std::vector<float> P{
            D * std::sin(th) * std::cos(phi),
            D * std::sin(th) * std::sin(phi),
            D * std::cos(th)};
        if (theta_deg > 90.f) {
            ASSERT_LT(P[2], 0.0f);
        } else {
            ASSERT_GT(P[2], 0.0f);
        }
        const auto proj = kb::kb_project(P[0], P[1], P[2], fx, fy, cx, cy, k[0], k[1], k[2], k[3]);
        ASSERT_TRUE(proj.finite);
        ASSERT_LT(static_cast<double>(proj.theta), tmax);
        ASSERT_GT(proj.px, 2.0f);
        ASSERT_LT(proj.px, static_cast<float>(W) - 2.0f);
        ASSERT_GT(proj.py, 2.0f);
        ASSERT_LT(proj.py, static_cast<float>(H) - 2.0f);

        Tensor means = Tensor::from_vector(P, {size_t{1}, size_t{3}}, Device::CUDA);
        Tensor sh0 = Tensor::full({1, 1, 3}, 0.5f, Device::CUDA);
        Tensor shN = Tensor::zeros({1, 0, 3}, Device::CUDA);
        Tensor scaling = Tensor::full({1, 3}, -3.6f, Device::CUDA);
        Tensor rotation = Tensor::zeros({1, 4}, Device::CUDA);
        rotation.slice(1, 0, 1).fill_(1.0f);
        Tensor opacity = Tensor::full({1}, 2.5f, Device::CUDA);

        {
            auto splat = make_splat(means, sh0, shN, scaling, rotation, opacity);
            auto fwd = fast_rasterize_forward(*camera, splat, bg);
            ASSERT_TRUE(fwd.has_value());
            ASSERT_GT(fwd->first.alpha.sum().item<float>(), 1e-3f)
                << "rim gaussian at " << theta_deg << " deg produced no alpha";
            fwd->second.release_forward_context();
        }

        auto run_analytical = [&](ParamType param) {
            auto s = make_splat(means, sh0, shN, scaling, rotation, opacity);
            auto r = fast_rasterize_forward(*camera, s, bg);
            AdamConfig cfg{.lr = 0.001f, .beta1 = 0.9, .beta2 = 0.999, .eps = 1e-15};
            auto opt = std::make_unique<AdamOptimizer>(s, cfg);
            opt->allocate_gradients();
            opt->zero_grad(0);
            auto grad_out = r->first.image.mul(2.0f);
            auto grad_depth = r->first.depth.mul(2.0f);
            fast_rasterize_backward(r->second, grad_out, s, *opt, {}, {}, DensificationType::None, 1, {}, grad_depth);
            return recovered_fused_grad(*opt, param);
        };

        auto numerical = [&](ParamType param, float eps) {
            Tensor orig;
            switch (param) {
            case ParamType::Means: orig = means.clone(); break;
            case ParamType::Scaling: orig = scaling.clone(); break;
            case ParamType::Rotation: orig = rotation.clone(); break;
            case ParamType::Opacity: orig = opacity.clone(); break;
            default: return Tensor();
            }
            auto orig_cpu = orig.to(Device::CPU);
            auto grad_cpu = Tensor::zeros_like(orig_cpu);
            float* g = grad_cpu.ptr<float>();
            const float* o = orig_cpu.ptr<float>();
            const size_t n_probe = std::min(orig.numel(), size_t{4});
            for (size_t p = 0; p < n_probe; ++p) {
                const size_t i = (n_probe == 1) ? 0 : p * (orig.numel() - 1) / (n_probe - 1);
                auto plus = orig_cpu.clone();
                auto minus = orig_cpu.clone();
                plus.ptr<float>()[i] = o[i] + eps;
                minus.ptr<float>()[i] = o[i] - eps;
                auto apply = [&](const Tensor& cpu_val) {
                    switch (param) {
                    case ParamType::Means: means = cpu_val.to(Device::CUDA); break;
                    case ParamType::Scaling: scaling = cpu_val.to(Device::CUDA); break;
                    case ParamType::Rotation: rotation = cpu_val.to(Device::CUDA); break;
                    case ParamType::Opacity: opacity = cpu_val.to(Device::CUDA); break;
                    default: break;
                    }
                };
                apply(plus);
                auto sp = make_splat(means, sh0, shN, scaling, rotation, opacity);
                auto rp = fast_rasterize_forward(*camera, sp, bg);
                const double lp = image_loss_f64(rp->first.image, rp->first.depth, true);
                rp->second.release_forward_context();
                apply(minus);
                auto sm = make_splat(means, sh0, shN, scaling, rotation, opacity);
                auto rm = fast_rasterize_forward(*camera, sm, bg);
                const double lm = image_loss_f64(rm->first.image, rm->first.depth, true);
                rm->second.release_forward_context();
                g[i] = static_cast<float>((lp - lm) / (2.0 * static_cast<double>(eps)));
            }
            switch (param) {
            case ParamType::Means: means = orig; break;
            case ParamType::Scaling: scaling = orig; break;
            case ParamType::Rotation: rotation = orig; break;
            case ParamType::Opacity: opacity = orig; break;
            default: break;
            }
            return grad_cpu.to(Device::CUDA);
        };

        const ParamType params[] = {ParamType::Means, ParamType::Scaling, ParamType::Rotation, ParamType::Opacity};
        for (auto p : params) {
            auto ana = run_analytical(p);
            float best = -1.f;
            for (float eps : {1e-3f, 5e-4f}) {
                auto num = numerical(p, eps);
                best = std::max(best, cosine_nonzero(num, ana));
                if (best > 0.90f)
                    break;
            }
            const char* name = "unknown";
            switch (p) {
            case ParamType::Means: name = "means"; break;
            case ParamType::Scaling: name = "scaling"; break;
            case ParamType::Rotation: name = "rotation"; break;
            case ParamType::Opacity: name = "opacity"; break;
            default: break;
            }
            EXPECT_GT(best, 0.75f) << "rim gradcheck failed for " << name << " at " << theta_deg << " deg";
        }
    }
    GlobalArenaManager::instance().get_arena().full_reset();
}

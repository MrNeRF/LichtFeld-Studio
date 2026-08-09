/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * joint (u, log_s) Adam codec unit tests.
 *
 * Host-side encode/decode, zero fixed point, bounds reduction, and
 * optimizer-step trajectory error vs fp32 reference (joint <= legacy).
 */

#include "lfs/training/joint_adam_codec.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <random>
#include <vector>

using namespace lfs::training::joint_adam;

namespace {

    // Legacy uint8 m + sqrt(v) codec (matches adam_kernels.cuh) for SNR comparison.
    constexpr uint8_t kLegacyZeroPoint = 128;

    float legacy_dequant_m(const uint8_t q, const float scale) {
        return scale == 0.0f ? 0.0f : (static_cast<int>(q) - static_cast<int>(kLegacyZeroPoint)) * scale;
    }
    uint8_t legacy_quantize_m(const float value, const float scale) {
        if (scale == 0.0f)
            return kLegacyZeroPoint;
        const int q = static_cast<int>(std::round(value / scale)) + static_cast<int>(kLegacyZeroPoint);
        return static_cast<uint8_t>(std::min(255, std::max(0, q)));
    }
    float legacy_dequant_sqrt_v(const uint8_t q, const float scale) {
        return scale == 0.0f ? 0.0f : static_cast<float>(q) * scale;
    }
    uint8_t legacy_quantize_sqrt_v(const float v, const float scale) {
        if (scale == 0.0f)
            return 0;
        const float s = std::sqrt(std::max(v, 0.0f));
        int q = static_cast<int>(std::round(s / scale));
        if (v > 0.0f && q == 0)
            q = 1;
        return static_cast<uint8_t>(std::min(255, std::max(0, q)));
    }

    struct Moments {
        std::vector<float> m;
        std::vector<float> v;
    };

    void adam_step_fp32(Moments& st, const std::vector<float>& grad,
                        float beta1, float beta2, float lr, float eps, int step) {
        const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
        const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
        const float step_size = lr / bc1;
        const float bc2_sqrt_rcp = 1.0f / std::sqrt(bc2);
        for (size_t i = 0; i < st.m.size(); ++i) {
            st.m[i] = beta1 * st.m[i] + (1.0f - beta1) * grad[i];
            st.v[i] = beta2 * st.v[i] + (1.0f - beta2) * grad[i] * grad[i];
            // param update not stored — we track moment fidelity only
            (void)step_size;
            (void)bc2_sqrt_rcp;
            (void)eps;
        }
    }

    // One encode/decode roundtrip of moments with joint codec (block = all cells).
    template <int BITS>
    Moments joint_roundtrip(const Moments& in) {
        using C = Codec<BITS>;
        const size_t n = in.m.size();
        float bounds[4];
        C::reduce_bounds(in.m.data(), in.v.data(), n, bounds);
        std::vector<uint8_t> packed(n * C::kBytesPerCell, 0);
        for (size_t i = 0; i < n; ++i) {
            C::encode_g1g2(packed.data(), i, in.m[i], in.v[i],
                           bounds[0], bounds[1], bounds[2], bounds[3]);
        }
        Moments out;
        out.m.resize(n);
        out.v.resize(n);
        for (size_t i = 0; i < n; ++i) {
            C::decode_g1g2(packed.data(), i, bounds[0], bounds[1], bounds[2], bounds[3],
                           out.m[i], out.v[i]);
        }
        return out;
    }

    Moments legacy_roundtrip(const Moments& in) {
        const size_t n = in.m.size();
        float max_abs_m = 0.0f, max_v = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            max_abs_m = std::max(max_abs_m, std::abs(in.m[i]));
            max_v = std::max(max_v, std::max(in.v[i], 0.0f));
        }
        const float m_scale = max_abs_m > 0.0f ? max_abs_m / 127.0f : 0.0f;
        const float v_scale = max_v > 0.0f ? std::sqrt(max_v) / 255.0f : 0.0f;
        std::vector<uint8_t> mq(n), vq(n);
        for (size_t i = 0; i < n; ++i) {
            mq[i] = legacy_quantize_m(in.m[i], m_scale);
            vq[i] = legacy_quantize_sqrt_v(in.v[i], v_scale);
        }
        Moments out;
        out.m.resize(n);
        out.v.resize(n);
        for (size_t i = 0; i < n; ++i) {
            out.m[i] = legacy_dequant_m(mq[i], m_scale);
            const float s = legacy_dequant_sqrt_v(vq[i], v_scale);
            out.v[i] = s * s;
        }
        return out;
    }

    double mse(const std::vector<float>& a, const std::vector<float>& b) {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            s += d * d;
        }
        return s / static_cast<double>(a.size());
    }

} // namespace

TEST(JointAdamCodecTest, ZeroFixedPointDecodesToZero) {
    constexpr size_t n = 16;
    std::vector<uint8_t> packed16(n * Codec16::kBytesPerCell, 0);
    std::vector<uint8_t> packed8(n * Codec8::kBytesPerCell, 0);
    for (size_t i = 0; i < n; ++i) {
        float g1 = 1.0f, g2 = 1.0f;
        Codec16::decode_g1g2(packed16.data(), i, 0, 0, 0, 0, g1, g2);
        EXPECT_FLOAT_EQ(g1, 0.0f);
        EXPECT_FLOAT_EQ(g2, 0.0f);
        Codec8::decode_g1g2(packed8.data(), i, 0, 0, 0, 0, g1, g2);
        EXPECT_FLOAT_EQ(g1, 0.0f);
        EXPECT_FLOAT_EQ(g2, 0.0f);
    }
}

TEST(JointAdamCodecTest, EndpointExactRoundtrip) {
    // Single cell: encode endpoints of a range, decode must hit lo/hi exactly for u,log_s.
    float g1s[2] = {-0.5f, 1.25f};
    float g2s[2] = {1e-10f, 4.0f};
    float bounds[4];
    Codec16::reduce_bounds(g1s, g2s, 2, bounds);

    std::vector<uint8_t> packed(2 * Codec16::kBytesPerCell, 0);
    Codec16::encode_g1g2(packed.data(), 0, g1s[0], g2s[0], bounds[0], bounds[1], bounds[2], bounds[3]);
    Codec16::encode_g1g2(packed.data(), 1, g1s[1], g2s[1], bounds[0], bounds[1], bounds[2], bounds[3]);

    float u0, s0, u1, s1;
    Codec16::decode_us(packed.data(), 0, bounds[0], bounds[1], bounds[2], bounds[3], u0, s0);
    Codec16::decode_us(packed.data(), 1, bounds[0], bounds[1], bounds[2], bounds[3], u1, s1);

    // Endpoint-exact: min/max of decoded (u,log_s) must recover bounds.
    EXPECT_NEAR(std::min(u0, u1), bounds[0], 1e-6f);
    EXPECT_NEAR(std::max(u0, u1), bounds[1], 1e-6f);
    EXPECT_NEAR(std::min(s0, s1), bounds[2], 1e-6f);
    EXPECT_NEAR(std::max(s0, s1), bounds[3], 1e-6f);
}

TEST(JointAdamCodecTest, RoundtripErrorBounded) {
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 0.1f);
    std::uniform_real_distribution<float> uv(1e-12f, 1e-2f);

    Moments ref;
    ref.m.resize(64);
    ref.v.resize(64);
    for (size_t i = 0; i < 64; ++i) {
        ref.m[i] = nd(rng);
        ref.v[i] = uv(rng);
    }

    const auto j16 = joint_roundtrip<16>(ref);
    const auto j8 = joint_roundtrip<8>(ref);
    const double mse_m16 = mse(ref.m, j16.m);
    const double mse_m8 = mse(ref.m, j8.m);
    // 16-bit should be tight; 8-bit coarser but finite.
    EXPECT_LT(mse_m16, 1e-6);
    EXPECT_LT(mse_m8, 1e-2);
    EXPECT_LT(mse_m16, mse_m8);
}

TEST(JointAdamCodecTest, BoundsReductionMatchesManual) {
    const float g1[] = {0.1f, -0.2f, 0.0f};
    const float g2[] = {1e-8f, 1e-4f, 0.0f};
    float bounds[4];
    Codec16::reduce_bounds(g1, g2, 3, bounds);

    float umin = 1e30f, umax = -1e30f, smin = 1e30f, smax = -1e30f;
    for (int i = 0; i < 3; ++i) {
        float u, s;
        Codec16::g1g2_to_us(g1[i], g2[i], u, s);
        umin = std::min(umin, u);
        umax = std::max(umax, u);
        smin = std::min(smin, s);
        smax = std::max(smax, s);
    }
    EXPECT_FLOAT_EQ(bounds[0], umin);
    EXPECT_FLOAT_EQ(bounds[1], umax);
    EXPECT_FLOAT_EQ(bounds[2], smin);
    EXPECT_FLOAT_EQ(bounds[3], smax);
}

TEST(JointAdamCodecTest, TrajectoryErrorNotWorseThanLegacy) {
    // Synthetic Adam moment trajectory: joint 16-bit roundtrip each step vs legacy.
    // Compare MSE of moments vs fp32 reference after K steps; joint must be <= legacy * 1.05.
    constexpr int kSteps = 50;
    constexpr size_t kN = 32;
    constexpr float beta1 = 0.9f, beta2 = 0.999f, lr = 1e-3f, eps = 1e-15f;

    std::mt19937 rng(7);
    std::normal_distribution<float> gd(0.0f, 0.05f);

    Moments ref, joint_st, legacy_st;
    ref.m.assign(kN, 0.0f);
    ref.v.assign(kN, 0.0f);
    joint_st = ref;
    legacy_st = ref;

    double joint_err = 0.0, legacy_err = 0.0;
    for (int step = 1; step <= kSteps; ++step) {
        std::vector<float> grad(kN);
        for (size_t i = 0; i < kN; ++i)
            grad[i] = gd(rng);

        adam_step_fp32(ref, grad, beta1, beta2, lr, eps, step);
        adam_step_fp32(joint_st, grad, beta1, beta2, lr, eps, step);
        adam_step_fp32(legacy_st, grad, beta1, beta2, lr, eps, step);

        // Requantize each path after the step (codec noise).
        joint_st = joint_roundtrip<16>(joint_st);
        legacy_st = legacy_roundtrip(legacy_st);

        joint_err += mse(ref.m, joint_st.m) + mse(ref.v, joint_st.v);
        legacy_err += mse(ref.m, legacy_st.m) + mse(ref.v, legacy_st.v);
    }

    // Joint (u,log_s) is designed for better small-v SNR; require not worse than legacy.
    EXPECT_LE(joint_err, legacy_err * 1.05 + 1e-12)
        << "joint_err=" << joint_err << " legacy_err=" << legacy_err;
}

TEST(JointAdamCodecTest, OptimBpsConstantsMatchFootprint) {
    EXPECT_EQ(kOptimBpsLegacySh3, 172u);
    EXPECT_EQ(kOptimBpsJointSh3, 152u); // LFS swizzled SH pad: 48 cells × 2 B
    EXPECT_LT(kOptimBpsJointSh3, kOptimBpsLegacySh3);
}

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/splat_simplify.hpp"
#include "core/tensor.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::SplatData;
using lfs::core::SplatSimplifyOptions;
using lfs::core::Tensor;

namespace {

    constexpr double kTwoPiPow1p5 = 15.749609945722419;
    constexpr double kEpsCov = 1e-8;
    constexpr double kMinScale = 1e-12;
    constexpr double kMinQuatNorm = 1e-12;

    struct RefInput {
        std::vector<double> means;
        std::vector<double> scaling_raw;
        std::vector<double> rotation_raw;
        std::vector<double> opacity_raw;
        std::vector<double> appearance;
        int app_dim = 0;

        [[nodiscard]] size_t count() const { return means.size() / 3; }
    };

    struct RefMerge {
        std::array<double, 3> mean{};
        std::array<double, 9> sigma{};
        double opacity = 0.0;
        std::vector<double> appearance;
    };

    [[nodiscard]] double sigmoid(const double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    [[nodiscard]] double quat_norm(const double qw, const double qx, const double qy, const double qz) {
        return std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
    }

    void quat_to_rotmat(const double qw, const double qx, const double qy, const double qz, std::array<double, 9>& out) {
        const double xx = qx * qx;
        const double yy = qy * qy;
        const double zz = qz * qz;
        const double wx = qw * qx;
        const double wy = qw * qy;
        const double wz = qw * qz;
        const double xy = qx * qy;
        const double xz = qx * qz;
        const double yz = qy * qz;

        out[0] = 1.0 - 2.0 * (yy + zz);
        out[1] = 2.0 * (xy - wz);
        out[2] = 2.0 * (xz + wy);
        out[3] = 2.0 * (xy + wz);
        out[4] = 1.0 - 2.0 * (xx + zz);
        out[5] = 2.0 * (yz - wx);
        out[6] = 2.0 * (xz - wy);
        out[7] = 2.0 * (yz + wx);
        out[8] = 1.0 - 2.0 * (xx + yy);
    }

    void sigma_from_rot_var(const std::array<double, 9>& R,
                            const double vx,
                            const double vy,
                            const double vz,
                            std::array<double, 9>& out) {
        const double r00 = R[0], r01 = R[1], r02 = R[2];
        const double r10 = R[3], r11 = R[4], r12 = R[5];
        const double r20 = R[6], r21 = R[7], r22 = R[8];
        out[0] = r00 * r00 * vx + r01 * r01 * vy + r02 * r02 * vz;
        out[1] = r00 * r10 * vx + r01 * r11 * vy + r02 * r12 * vz;
        out[2] = r00 * r20 * vx + r01 * r21 * vy + r02 * r22 * vz;
        out[3] = out[1];
        out[4] = r10 * r10 * vx + r11 * r11 * vy + r12 * r12 * vz;
        out[5] = r10 * r20 * vx + r11 * r21 * vy + r12 * r22 * vz;
        out[6] = out[2];
        out[7] = out[5];
        out[8] = r20 * r20 * vx + r21 * r21 * vy + r22 * r22 * vz;
    }

    [[nodiscard]] RefMerge reference_moment_match(const RefInput& input, const int i, const int j) {
        const size_t i3 = static_cast<size_t>(i) * 3;
        const size_t j3 = static_cast<size_t>(j) * 3;
        const size_t i4 = static_cast<size_t>(i) * 4;
        const size_t j4 = static_cast<size_t>(j) * 4;

        const double sxi = std::max(std::exp(input.scaling_raw[i3 + 0]), kMinScale);
        const double syi = std::max(std::exp(input.scaling_raw[i3 + 1]), kMinScale);
        const double szi = std::max(std::exp(input.scaling_raw[i3 + 2]), kMinScale);
        const double sxj = std::max(std::exp(input.scaling_raw[j3 + 0]), kMinScale);
        const double syj = std::max(std::exp(input.scaling_raw[j3 + 1]), kMinScale);
        const double szj = std::max(std::exp(input.scaling_raw[j3 + 2]), kMinScale);

        const double alpha_i = sigmoid(input.opacity_raw[i]);
        const double alpha_j = sigmoid(input.opacity_raw[j]);
        const double wi = kTwoPiPow1p5 * alpha_i * sxi * syi * szi + 1e-12;
        const double wj = kTwoPiPow1p5 * alpha_j * sxj * syj * szj + 1e-12;
        const double W = std::max(wi + wj, 1e-12);

        RefMerge merge;
        merge.mean = {
            (wi * input.means[i3 + 0] + wj * input.means[j3 + 0]) / W,
            (wi * input.means[i3 + 1] + wj * input.means[j3 + 1]) / W,
            (wi * input.means[i3 + 2] + wj * input.means[j3 + 2]) / W,
        };

        double qwi = input.rotation_raw[i4 + 0];
        double qxi = input.rotation_raw[i4 + 1];
        double qyi = input.rotation_raw[i4 + 2];
        double qzi = input.rotation_raw[i4 + 3];
        double qwj = input.rotation_raw[j4 + 0];
        double qxj = input.rotation_raw[j4 + 1];
        double qyj = input.rotation_raw[j4 + 2];
        double qzj = input.rotation_raw[j4 + 3];

        const double inv_i = 1.0 / std::max(quat_norm(qwi, qxi, qyi, qzi), kMinQuatNorm);
        const double inv_j = 1.0 / std::max(quat_norm(qwj, qxj, qyj, qzj), kMinQuatNorm);
        qwi *= inv_i;
        qxi *= inv_i;
        qyi *= inv_i;
        qzi *= inv_i;
        qwj *= inv_j;
        qxj *= inv_j;
        qyj *= inv_j;
        qzj *= inv_j;

        std::array<double, 9> Ri{};
        std::array<double, 9> Rj{};
        quat_to_rotmat(qwi, qxi, qyi, qzi, Ri);
        quat_to_rotmat(qwj, qxj, qyj, qzj, Rj);

        std::array<double, 9> sig_i{};
        std::array<double, 9> sig_j{};
        sigma_from_rot_var(Ri, sxi * sxi, syi * syi, szi * szi, sig_i);
        sigma_from_rot_var(Rj, sxj * sxj, syj * syj, szj * szj, sig_j);

        const double dix = input.means[i3 + 0] - merge.mean[0];
        const double diy = input.means[i3 + 1] - merge.mean[1];
        const double diz = input.means[i3 + 2] - merge.mean[2];
        const double djx = input.means[j3 + 0] - merge.mean[0];
        const double djy = input.means[j3 + 1] - merge.mean[1];
        const double djz = input.means[j3 + 2] - merge.mean[2];

        for (int a = 0; a < 9; ++a)
            merge.sigma[a] = (wi * sig_i[a] + wj * sig_j[a]) / W;
        merge.sigma[0] += (wi * dix * dix + wj * djx * djx) / W;
        merge.sigma[1] += (wi * dix * diy + wj * djx * djy) / W;
        merge.sigma[2] += (wi * dix * diz + wj * djx * djz) / W;
        merge.sigma[3] += (wi * diy * dix + wj * djy * djx) / W;
        merge.sigma[4] += (wi * diy * diy + wj * djy * djy) / W;
        merge.sigma[5] += (wi * diy * diz + wj * djy * djz) / W;
        merge.sigma[6] += (wi * diz * dix + wj * djz * djx) / W;
        merge.sigma[7] += (wi * diz * diy + wj * djz * djy) / W;
        merge.sigma[8] += (wi * diz * diz + wj * djz * djz) / W;
        merge.sigma[1] = merge.sigma[3] = 0.5 * (merge.sigma[1] + merge.sigma[3]);
        merge.sigma[2] = merge.sigma[6] = 0.5 * (merge.sigma[2] + merge.sigma[6]);
        merge.sigma[5] = merge.sigma[7] = 0.5 * (merge.sigma[5] + merge.sigma[7]);
        merge.sigma[0] += kEpsCov;
        merge.sigma[4] += kEpsCov;
        merge.sigma[8] += kEpsCov;

        merge.opacity = alpha_i + alpha_j - alpha_i * alpha_j;
        merge.appearance.resize(static_cast<size_t>(input.app_dim));
        const size_t ai = static_cast<size_t>(i) * input.app_dim;
        const size_t aj = static_cast<size_t>(j) * input.app_dim;
        for (int k = 0; k < input.app_dim; ++k)
            merge.appearance[static_cast<size_t>(k)] = (wi * input.appearance[ai + k] + wj * input.appearance[aj + k]) / W;
        return merge;
    }

    [[nodiscard]] std::unique_ptr<SplatData> make_test_splat(const RefInput& input, const int max_sh_degree = 1) {
        const size_t count = input.count();
        const int shn_coeffs = max_sh_degree > 0 ? (max_sh_degree + 1) * (max_sh_degree + 1) - 1 : 0;

        std::vector<float> means(input.means.begin(), input.means.end());
        std::vector<float> scaling(input.scaling_raw.begin(), input.scaling_raw.end());
        std::vector<float> rotation(input.rotation_raw.begin(), input.rotation_raw.end());
        std::vector<float> opacity(input.opacity_raw.begin(), input.opacity_raw.end());

        std::vector<float> sh0(count * 3, 0.0f);
        std::vector<float> shN(count * static_cast<size_t>(shn_coeffs) * 3, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            for (int c = 0; c < 3 && c < input.app_dim; ++c)
                sh0[i * 3 + static_cast<size_t>(c)] = static_cast<float>(input.appearance[i * static_cast<size_t>(input.app_dim) + static_cast<size_t>(c)]);
            for (int tail = 0; tail < shn_coeffs * 3; ++tail) {
                const int app_idx = 3 + tail;
                if (app_idx >= input.app_dim)
                    continue;
                const int coeff = tail / 3;
                const int channel = tail % 3;
                shN[i * static_cast<size_t>(shn_coeffs) * 3 + static_cast<size_t>(coeff) * 3 + static_cast<size_t>(channel)] =
                    static_cast<float>(input.appearance[i * static_cast<size_t>(input.app_dim) + static_cast<size_t>(app_idx)]);
            }
        }

        auto result = std::make_unique<SplatData>(
            max_sh_degree,
            Tensor::from_vector(means, {count, size_t{3}}, Device::CUDA).to(DataType::Float32),
            Tensor::from_vector(sh0, {count, size_t{1}, size_t{3}}, Device::CUDA).to(DataType::Float32),
            shn_coeffs > 0
                ? Tensor::from_vector(shN, {count, static_cast<size_t>(shn_coeffs), size_t{3}}, Device::CUDA).to(DataType::Float32)
                : Tensor{},
            Tensor::from_vector(scaling, {count, size_t{3}}, Device::CUDA).to(DataType::Float32),
            Tensor::from_vector(rotation, {count, size_t{4}}, Device::CUDA).to(DataType::Float32),
            Tensor::from_vector(opacity, {count, size_t{1}}, Device::CUDA).to(DataType::Float32),
            1.0f);
        result->set_active_sh_degree(max_sh_degree);
        result->set_max_sh_degree(max_sh_degree);
        return result;
    }

    [[nodiscard]] std::vector<float> row_values(const Tensor& tensor, const size_t row) {
        const auto cpu = tensor.cpu().contiguous();
        const auto flat = cpu.to_vector();
        const size_t stride = flat.size() / static_cast<size_t>(cpu.size(0));
        return std::vector<float>(flat.begin() + static_cast<ptrdiff_t>(row * stride),
                                  flat.begin() + static_cast<ptrdiff_t>((row + 1) * stride));
    }

    [[nodiscard]] std::vector<float> appearance_row(const SplatData& splat, const size_t row) {
        std::vector<float> result = row_values(splat.sh0_raw().reshape({static_cast<int>(splat.size()), 3}), row);
        if (splat.shN_raw().is_valid()) {
            auto tail = row_values(
                splat.shN_raw().reshape({static_cast<int>(splat.size()), static_cast<int>(splat.shN_raw().size(1) * 3)}),
                row);
            result.insert(result.end(), tail.begin(), tail.end());
        }
        return result;
    }

    [[nodiscard]] std::array<double, 9> covariance_from_output_row(const SplatData& splat, const size_t row) {
        const auto scaling = row_values(splat.scaling_raw(), row);
        const auto rotation = row_values(splat.rotation_raw(), row);

        std::array<double, 9> R{};
        quat_to_rotmat(rotation[0], rotation[1], rotation[2], rotation[3], R);
        std::array<double, 9> sigma{};
        sigma_from_rot_var(
            R,
            std::exp(static_cast<double>(scaling[0])) * std::exp(static_cast<double>(scaling[0])),
            std::exp(static_cast<double>(scaling[1])) * std::exp(static_cast<double>(scaling[1])),
            std::exp(static_cast<double>(scaling[2])) * std::exp(static_cast<double>(scaling[2])),
            sigma);
        return sigma;
    }

    [[nodiscard]] std::array<double, 3> mean_from_output_row(const SplatData& splat, const size_t row) {
        const auto means = row_values(splat.means_raw(), row);
        return {means[0], means[1], means[2]};
    }

    [[nodiscard]] double opacity_from_output_row(const SplatData& splat, const size_t row) {
        const auto opacity = row_values(splat.opacity_raw(), row);
        return sigmoid(opacity[0]);
    }

    [[nodiscard]] double euclidean_distance(const std::array<double, 3>& a, const std::array<double, 3>& b) {
        const double dx = a[0] - b[0];
        const double dy = a[1] - b[1];
        const double dz = a[2] - b[2];
        return std::sqrt(dx * dx + dy * dy + dz * dz);
    }

    void expect_vec3_near(const std::array<double, 3>& actual, const std::array<double, 3>& expected, const double tol) {
        for (size_t i = 0; i < 3; ++i)
            EXPECT_NEAR(actual[i], expected[i], tol);
    }

    void expect_mat3_near(const std::array<double, 9>& actual, const std::array<double, 9>& expected, const double tol) {
        for (size_t i = 0; i < 9; ++i)
            EXPECT_NEAR(actual[i], expected[i], tol);
    }

} // namespace

TEST(SplatSimplify, TwoPointMergeMatchesReferenceMomentMatching) {
    RefInput input{
        .means = {
            0.0,
            0.0,
            0.0,
            0.4,
            -0.2,
            0.1,
        },
        .scaling_raw = {
            std::log(0.20),
            std::log(0.10),
            std::log(0.15),
            std::log(0.30),
            std::log(0.25),
            std::log(0.18),
        },
        .rotation_raw = {
            1.0,
            0.0,
            0.0,
            0.0,
            0.9238795,
            0.0,
            0.3826834,
            0.0,
        },
        .opacity_raw = {
            1.2,
            0.7,
        },
        .appearance = {
            0.10,
            0.20,
            0.30,
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.60,
            0.50,
            0.40,
            0.11,
            0.12,
            0.13,
            0.14,
            0.15,
            0.16,
            0.17,
            0.18,
            0.19,
        },
        .app_dim = 12,
    };

    auto source = make_test_splat(input);
    const auto before_means = source->means_raw().cpu().to_vector();

    SplatSimplifyOptions options;
    options.ratio = 0.5f;
    options.knn_k = 16;

    auto result = lfs::core::simplify_splats(*source, options, {});
    ASSERT_TRUE(result) << result.error();
    ASSERT_NE(*result, nullptr);
    ASSERT_EQ((*result)->size(), 1u);
    ASSERT_EQ(source->size(), 2u);
    EXPECT_EQ(source->means_raw().cpu().to_vector(), before_means);

    const RefMerge expected = reference_moment_match(input, 0, 1);
    expect_vec3_near(mean_from_output_row(**result, 0), expected.mean, 5e-4);
    expect_mat3_near(covariance_from_output_row(**result, 0), expected.sigma, 8e-4);
    EXPECT_NEAR(opacity_from_output_row(**result, 0), expected.opacity, 5e-5);

    const auto appearance = appearance_row(**result, 0);
    ASSERT_EQ(appearance.size(), expected.appearance.size());
    for (size_t i = 0; i < appearance.size(); ++i)
        EXPECT_NEAR(appearance[i], expected.appearance[i], 5e-4);
}

TEST(SplatSimplify, RandomizedTwoPointMergeMatchesReferenceMomentMatching) {
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> mean_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> scale_dist(std::log(0.02), std::log(0.8));
    std::uniform_real_distribution<double> quat_dist(-1.0, 1.0);
    std::uniform_real_distribution<double> opacity_dist(-3.0, 3.0);
    std::uniform_real_distribution<double> appearance_dist(-0.5, 0.5);

    for (int iter = 0; iter < 64; ++iter) {
        RefInput input;
        input.app_dim = 12;
        input.means.resize(6);
        input.scaling_raw.resize(6);
        input.rotation_raw.resize(8);
        input.opacity_raw.resize(2);
        input.appearance.resize(24);

        for (double& v : input.means)
            v = mean_dist(rng);
        for (double& v : input.scaling_raw)
            v = scale_dist(rng);
        for (double& v : input.opacity_raw)
            v = opacity_dist(rng);
        for (double& v : input.appearance)
            v = appearance_dist(rng);

        for (int q = 0; q < 2; ++q) {
            double qw = quat_dist(rng);
            double qx = quat_dist(rng);
            double qy = quat_dist(rng);
            double qz = quat_dist(rng);
            const double norm = std::max(std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz), 1e-6);
            input.rotation_raw[q * 4 + 0] = qw / norm;
            input.rotation_raw[q * 4 + 1] = qx / norm;
            input.rotation_raw[q * 4 + 2] = qy / norm;
            input.rotation_raw[q * 4 + 3] = qz / norm;
        }

        auto source = make_test_splat(input);
        SplatSimplifyOptions options;
        options.ratio = 0.5f;
        options.knn_k = 16;
        options.opacity_prune_threshold = 0.0f;

        auto result = lfs::core::simplify_splats(*source, options, {});
        ASSERT_TRUE(result) << result.error();
        ASSERT_NE(*result, nullptr);
        ASSERT_EQ((*result)->size(), 1u);

        const RefMerge expected = reference_moment_match(input, 0, 1);
        const auto actual_mean = mean_from_output_row(**result, 0);
        const auto actual_cov = covariance_from_output_row(**result, 0);
        const auto actual_opacity = opacity_from_output_row(**result, 0);
        const auto actual_appearance = appearance_row(**result, 0);

        for (size_t c = 0; c < 3; ++c)
            EXPECT_NEAR(actual_mean[c], expected.mean[c], 2e-3) << "iter=" << iter << " mean[" << c << "]";
        for (size_t c = 0; c < 9; ++c)
            EXPECT_NEAR(actual_cov[c], expected.sigma[c], 3e-3) << "iter=" << iter << " cov[" << c << "]";
        EXPECT_NEAR(actual_opacity, expected.opacity, 1e-4) << "iter=" << iter;
        ASSERT_EQ(actual_appearance.size(), expected.appearance.size());
        for (size_t c = 0; c < actual_appearance.size(); ++c) {
            EXPECT_NEAR(actual_appearance[c], expected.appearance[c], 2e-3)
                << "iter=" << iter << " appearance[" << c << "]";
        }
    }
}

TEST(SplatSimplify, NoOpSimplifyPreservesAppearanceInPlyPropertyOrder) {
    RefInput input{
        .means = {
            0.0,
            0.0,
            0.0,
            1.0,
            2.0,
            3.0,
        },
        .scaling_raw = {
            std::log(0.10),
            std::log(0.20),
            std::log(0.30),
            std::log(0.40),
            std::log(0.50),
            std::log(0.60),
        },
        .rotation_raw = {
            1.0,
            0.0,
            0.0,
            0.0,
            0.9238795,
            0.0,
            0.3826834,
            0.0,
        },
        .opacity_raw = {
            0.1,
            0.2,
        },
        .appearance = {
            10.0,
            20.0,
            30.0,
            101.0,
            102.0,
            103.0,
            201.0,
            202.0,
            203.0,
            301.0,
            302.0,
            303.0,
            11.0,
            21.0,
            31.0,
            111.0,
            112.0,
            113.0,
            211.0,
            212.0,
            213.0,
            311.0,
            312.0,
            313.0,
        },
        .app_dim = 12,
    };

    auto source = make_test_splat(input);
    SplatSimplifyOptions options;
    options.ratio = 1.0f;
    options.knn_k = 16;

    auto result = lfs::core::simplify_splats(*source, options, {});
    ASSERT_TRUE(result) << result.error();
    ASSERT_EQ((*result)->size(), input.count());

    const auto row0 = appearance_row(**result, 0);
    const auto row1 = appearance_row(**result, 1);
    ASSERT_EQ(row0.size(), 12u);
    ASSERT_EQ(row1.size(), 12u);
    for (size_t i = 0; i < row0.size(); ++i) {
        EXPECT_FLOAT_EQ(row0[i], static_cast<float>(input.appearance[i]));
        EXPECT_FLOAT_EQ(row1[i], static_cast<float>(input.appearance[12 + i]));
    }
}

TEST(SplatSimplify, ThreePointSelectionChoosesClosestPairWhenAllPairsAreCandidates) {
    RefInput input{
        .means = {
            0.00,
            0.00,
            0.00,
            0.03,
            0.01,
            0.00,
            2.50,
            0.50,
            0.20,
        },
        .scaling_raw = {
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.11),
            std::log(0.10),
            std::log(0.09),
            std::log(0.15),
            std::log(0.14),
            std::log(0.16),
        },
        .rotation_raw = {
            1.0,
            0.0,
            0.0,
            0.0,
            0.9807853,
            0.0,
            0.1950903,
            0.0,
            0.9659258,
            0.0,
            0.2588190,
            0.0,
        },
        .opacity_raw = {
            0.8,
            0.75,
            0.9,
        },
        .appearance = {
            0.15,
            0.16,
            0.17,
            0.16,
            0.15,
            0.18,
            0.95,
            0.05,
            0.10,
        },
        .app_dim = 3,
    };

    const std::array<double, 3> mean0 = {input.means[0], input.means[1], input.means[2]};
    const std::array<double, 3> mean1 = {input.means[3], input.means[4], input.means[5]};
    const std::array<double, 3> mean2 = {input.means[6], input.means[7], input.means[8]};
    EXPECT_LT(euclidean_distance(mean0, mean1), euclidean_distance(mean0, mean2));
    EXPECT_LT(euclidean_distance(mean0, mean1), euclidean_distance(mean1, mean2));

    auto source = make_test_splat(input, 0);
    SplatSimplifyOptions options;
    options.ratio = 0.5f;
    options.knn_k = 16;

    auto result = lfs::core::simplify_splats(*source, options, {});
    ASSERT_TRUE(result) << result.error();
    ASSERT_EQ((*result)->size(), 2u);

    const RefMerge expected_merge = reference_moment_match(input, 0, 1);
    const std::array<double, 3> expected_keep = {
        input.means[6],
        input.means[7],
        input.means[8],
    };

    const auto output_mean0 = mean_from_output_row(**result, 0);
    const auto output_mean1 = mean_from_output_row(**result, 1);
    const bool first_is_keep = euclidean_distance(output_mean0, expected_keep) < euclidean_distance(output_mean1, expected_keep);
    const auto& keep_row = first_is_keep ? output_mean0 : output_mean1;
    const auto& merge_row = first_is_keep ? output_mean1 : output_mean0;

    expect_vec3_near(keep_row, expected_keep, 1e-5);
    expect_vec3_near(merge_row, expected_merge.mean, 5e-4);
}

TEST(SplatSimplify, ThreePointSelectionUsesMeansEvenForRotatedAnisotropicGaussians) {
    RefInput input{
        .means = {
            0.0976270065,
            0.4303787351,
            0.2055267543,
            0.0897663683,
            -0.1526903957,
            0.2917882204,
            -0.1248255745,
            0.7835460305,
            0.9273255467,
        },
        .scaling_raw = {
            std::log(0.1982781291),
            std::log(0.5071074367),
            std::log(0.2770543098),
            std::log(0.3031591177),
            std::log(0.6899558306),
            std::log(0.0966540575),
            std::log(0.1002986953),
            std::log(0.0859922841),
            std::log(0.5571201444),
        },
        .rotation_raw = {
            0.2761918604,
            0.2076273263,
            0.9296838641,
            -0.1276587844,
            0.1122946069,
            -0.3063565493,
            -0.9157347083,
            0.2344471812,
            0.2953715622,
            -0.2535923719,
            0.7755585909,
            -0.4969461560,
        },
        .opacity_raw = {
            std::log(0.2140923440 / (1.0 - 0.2140923440)),
            std::log(0.6632266045 / (1.0 - 0.6632266045)),
            std::log(0.6590718031 / (1.0 - 0.6590718031)),
        },
        .appearance = {
            0.1169339940,
            0.4437480867,
            0.1818203032,
            -0.1404920965,
            -0.0629680455,
            0.1976311952,
            -0.4397745430,
            0.1667667180,
            0.1706378758,
            -0.2896174490,
            -0.3710736930,
            -0.1845716536,
            -0.1362892240,
            0.0701967701,
            -0.0613984875,
            0.4883738458,
            -0.3979551792,
            -0.2911232412,
        },
        .app_dim = 6,
    };

    const std::array<double, 3> mean0 = {input.means[0], input.means[1], input.means[2]};
    const std::array<double, 3> mean1 = {input.means[3], input.means[4], input.means[5]};
    const std::array<double, 3> mean2 = {input.means[6], input.means[7], input.means[8]};
    EXPECT_LT(euclidean_distance(mean0, mean1), euclidean_distance(mean0, mean2));
    EXPECT_LT(euclidean_distance(mean0, mean1), euclidean_distance(mean1, mean2));

    auto source = make_test_splat(input, 1);
    SplatSimplifyOptions options;
    options.ratio = 0.5f;
    options.knn_k = 16;

    auto result = lfs::core::simplify_splats(*source, options, {});
    ASSERT_TRUE(result) << result.error();
    ASSERT_EQ((*result)->size(), 2u);

    const RefMerge expected_merge = reference_moment_match(input, 0, 1);
    const std::array<double, 3> expected_keep = {
        input.means[6],
        input.means[7],
        input.means[8],
    };

    const auto output_mean0 = mean_from_output_row(**result, 0);
    const auto output_mean1 = mean_from_output_row(**result, 1);
    const bool first_is_keep = euclidean_distance(output_mean0, expected_keep) < euclidean_distance(output_mean1, expected_keep);
    const auto& keep_row = first_is_keep ? output_mean0 : output_mean1;
    const auto& merge_row = first_is_keep ? output_mean1 : output_mean0;

    expect_vec3_near(keep_row, expected_keep, 1e-5);
    expect_vec3_near(merge_row, expected_merge.mean, 5e-4);
}

TEST(SplatSimplify, IgnoresAppearanceWhenChoosingClosestPair) {
    RefInput input{
        .means = {
            0.00,
            0.00,
            0.00,
            0.02,
            0.00,
            0.00,
            1.00,
            0.00,
            0.00,
        },
        .scaling_raw = {
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
        },
        .rotation_raw = {
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        },
        .opacity_raw = {
            std::log(0.5 / (1.0 - 0.5)),
            std::log(0.5 / (1.0 - 0.5)),
            std::log(0.5 / (1.0 - 0.5)),
        },
        .appearance = {
            0.0,
            0.0,
            0.0,
            2.0,
            2.0,
            2.0,
            -1.0,
            -1.0,
            -1.0,
        },
        .app_dim = 3,
    };

    auto source = make_test_splat(input, 0);
    SplatSimplifyOptions options;
    options.ratio = 0.5f;
    options.knn_k = 16;

    auto result = lfs::core::simplify_splats(*source, options, {});
    ASSERT_TRUE(result) << result.error();
    ASSERT_EQ((*result)->size(), 2u);

    const RefMerge expected_merge = reference_moment_match(input, 0, 1);
    const std::array<double, 3> expected_keep = {
        input.means[6],
        input.means[7],
        input.means[8],
    };

    const std::array<double, 3> input_mean0 = {input.means[0], input.means[1], input.means[2]};
    const std::array<double, 3> input_mean1 = {input.means[3], input.means[4], input.means[5]};
    const std::array<double, 3> input_mean2 = {input.means[6], input.means[7], input.means[8]};
    EXPECT_LT(euclidean_distance(input_mean0, input_mean1), euclidean_distance(input_mean0, input_mean2));
    EXPECT_LT(euclidean_distance(input_mean0, input_mean1), euclidean_distance(input_mean1, input_mean2));

    const auto output_mean0 = mean_from_output_row(**result, 0);
    const auto output_mean1 = mean_from_output_row(**result, 1);
    const bool first_is_keep = euclidean_distance(output_mean0, expected_keep) < euclidean_distance(output_mean1, expected_keep);
    const auto& keep_row = first_is_keep ? output_mean0 : output_mean1;
    const auto& merge_row = first_is_keep ? output_mean1 : output_mean0;

    expect_vec3_near(keep_row, expected_keep, 1e-5);
    expect_vec3_near(merge_row, expected_merge.mean, 5e-4);
}

TEST(SplatSimplify, AllowsOpacityPruneToFinishBelowTarget) {
    RefInput input{
        .means = {
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            3.0,
            0.0,
            0.0,
        },
        .scaling_raw = {
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
        },
        .rotation_raw = {
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        },
        .opacity_raw = {
            std::log(0.02 / (1.0 - 0.02)),
            std::log(0.05 / (1.0 - 0.05)),
            std::log(0.8 / (1.0 - 0.8)),
            std::log(0.9 / (1.0 - 0.9)),
        },
        .appearance = {
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        },
        .app_dim = 3,
    };

    auto source = make_test_splat(input, 0);
    SplatSimplifyOptions options;
    options.ratio = 0.75f;
    options.knn_k = 16;

    auto result = lfs::core::simplify_splats(*source, options, {});
    ASSERT_TRUE(result) << result.error();
    ASSERT_EQ((*result)->size(), 2u);
}

TEST(SplatSimplify, UsesNativeBackendProgressStages) {
    RefInput input{
        .means = {
            0.00,
            0.00,
            0.00,
            0.02,
            0.00,
            0.00,
            1.00,
            0.00,
            0.00,
            1.02,
            0.00,
            0.00,
        },
        .scaling_raw = {
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
            std::log(0.10),
        },
        .rotation_raw = {
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
        },
        .opacity_raw = {
            0.8,
            0.8,
            0.8,
            0.8,
        },
        .appearance = {
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        },
        .app_dim = 3,
    };

    auto source = make_test_splat(input, 0);
    SplatSimplifyOptions options;
    options.ratio = 0.5f;
    options.knn_k = 16;

    std::vector<std::string> stages;
    auto result = lfs::core::simplify_splats(
        *source,
        options,
        [&stages](const float /*progress*/, const std::string& stage) {
            stages.push_back(stage);
            return true;
        });
    ASSERT_TRUE(result) << result.error();
    ASSERT_EQ((*result)->size(), 2u);
    ASSERT_TRUE(std::find(stages.begin(), stages.end(), "Pruning opacity") != stages.end());
    ASSERT_TRUE(std::find(stages.begin(), stages.end(), "Pass 1: building kNN graph") != stages.end());
    ASSERT_TRUE(std::find(stages.begin(), stages.end(), "Pass 1: computing edge costs") != stages.end());
    ASSERT_TRUE(std::find(stages.begin(), stages.end(), "Pass 1: selecting pairs") != stages.end());
    ASSERT_TRUE(std::find_if(stages.begin(),
                             stages.end(),
                             [](const std::string& stage) {
                                 return stage.starts_with("Pass 1: merging ");
                             }) != stages.end());
    ASSERT_TRUE(std::find(stages.begin(), stages.end(), "Complete") != stages.end());
}

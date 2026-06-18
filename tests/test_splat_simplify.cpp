/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/splat_data.hpp"
#include "core/splat_simplify.hpp"
#include "core/splat_simplify_history.hpp"
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

    [[nodiscard]] RefMerge reference_moment_match(const RefInput& input, const std::vector<int>& indices) {
        double total_weight = 0.0;
        std::array<double, 3> weighted_mean = {0.0, 0.0, 0.0};

        // First pass: compute weights and weighted mean
        for (int idx : indices) {
            const size_t i3 = static_cast<size_t>(idx) * 3;
            const double sxi = std::max(std::exp(input.scaling_raw[i3 + 0]), kMinScale);
            const double syi = std::max(std::exp(input.scaling_raw[i3 + 1]), kMinScale);
            const double szi = std::max(std::exp(input.scaling_raw[i3 + 2]), kMinScale);
            const double alpha_i = sigmoid(input.opacity_raw[idx]);
            const double wi = sxi * syi * szi * alpha_i + 1e-12;
            total_weight += wi;

            weighted_mean[0] += wi * input.means[i3 + 0];
            weighted_mean[1] += wi * input.means[i3 + 1];
            weighted_mean[2] += wi * input.means[i3 + 2];
        }

        RefMerge merge;
        merge.mean = {
            weighted_mean[0] / total_weight,
            weighted_mean[1] / total_weight,
            weighted_mean[2] / total_weight,
        };

        // Compute covariance
        std::fill(merge.sigma.begin(), merge.sigma.end(), 0.0);
        for (int idx : indices) {
            const size_t i3 = static_cast<size_t>(idx) * 3;
            const size_t i4 = static_cast<size_t>(idx) * 4;

            const double sxi = std::max(std::exp(input.scaling_raw[i3 + 0]), kMinScale);
            const double syi = std::max(std::exp(input.scaling_raw[i3 + 1]), kMinScale);
            const double szi = std::max(std::exp(input.scaling_raw[i3 + 2]), kMinScale);
            const double alpha_i = sigmoid(input.opacity_raw[idx]);
            const double wi = sxi * syi * szi * alpha_i + 1e-12;

            double qwi = input.rotation_raw[i4 + 0];
            double qxi = input.rotation_raw[i4 + 1];
            double qyi = input.rotation_raw[i4 + 2];
            double qzi = input.rotation_raw[i4 + 3];
            const double inv = 1.0 / std::max(quat_norm(qwi, qxi, qyi, qzi), kMinQuatNorm);
            qwi *= inv;
            qxi *= inv;
            qyi *= inv;
            qzi *= inv;

            std::array<double, 9> Ri{};
            quat_to_rotmat(qwi, qxi, qyi, qzi, Ri);

            std::array<double, 9> sig_i{};
            sigma_from_rot_var(Ri, sxi * sxi, syi * syi, szi * szi, sig_i);

            const double dix = input.means[i3 + 0] - merge.mean[0];
            const double diy = input.means[i3 + 1] - merge.mean[1];
            const double diz = input.means[i3 + 2] - merge.mean[2];

            for (int a = 0; a < 9; ++a)
                merge.sigma[a] += wi * sig_i[a];

            merge.sigma[0] += wi * dix * dix;
            merge.sigma[1] += wi * dix * diy;
            merge.sigma[2] += wi * dix * diz;
            merge.sigma[3] += wi * diy * dix;
            merge.sigma[4] += wi * diy * diy;
            merge.sigma[5] += wi * diy * diz;
            merge.sigma[6] += wi * diz * dix;
            merge.sigma[7] += wi * diz * diy;
            merge.sigma[8] += wi * diz * diz;
        }

        for (int a = 0; a < 9; ++a)
            merge.sigma[a] /= total_weight;

        merge.sigma[1] = merge.sigma[3] = 0.5 * (merge.sigma[1] + merge.sigma[3]);
        merge.sigma[2] = merge.sigma[6] = 0.5 * (merge.sigma[2] + merge.sigma[6]);
        merge.sigma[5] = merge.sigma[7] = 0.5 * (merge.sigma[5] + merge.sigma[7]);
        merge.sigma[0] += kEpsCov;
        merge.sigma[4] += kEpsCov;
        merge.sigma[8] += kEpsCov;

        // Opacity: 1 - prod(1 - alpha_i)
        double one_minus_alpha_prod = 1.0;
        for (int idx : indices) {
            const double alpha_i = sigmoid(input.opacity_raw[idx]);
            one_minus_alpha_prod *= (1.0 - alpha_i);
        }
        merge.opacity = 1.0 - one_minus_alpha_prod;

        // Appearance: weighted average
        merge.appearance.resize(static_cast<size_t>(input.app_dim));
        std::fill(merge.appearance.begin(), merge.appearance.end(), 0.0);
        for (int idx : indices) {
            const size_t i3 = static_cast<size_t>(idx) * 3;
            const double sxi = std::max(std::exp(input.scaling_raw[i3 + 0]), kMinScale);
            const double syi = std::max(std::exp(input.scaling_raw[i3 + 1]), kMinScale);
            const double szi = std::max(std::exp(input.scaling_raw[i3 + 2]), kMinScale);
            const double alpha_i = sigmoid(input.opacity_raw[idx]);
            const double wi = sxi * syi * szi * alpha_i + 1e-12;
            const size_t ai = static_cast<size_t>(idx) * static_cast<size_t>(input.app_dim);
            for (int k = 0; k < input.app_dim; ++k)
                merge.appearance[static_cast<size_t>(k)] += wi * input.appearance[ai + k];
        }
        for (size_t k = 0; k < merge.appearance.size(); ++k)
            merge.appearance[k] /= total_weight;

        return merge;
    }

    [[nodiscard]] RefMerge reference_moment_match(const RefInput& input, const int i, const int j) {
        return reference_moment_match(input, std::vector<int>{i, j});
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
        if (splat.max_sh_coeffs_rest() > 0) {
            const Tensor shN = splat.shN_canonical();
            auto tail = row_values(
                shN.reshape({static_cast<int>(splat.size()), static_cast<int>(shN.size(1) * 3)}),
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
    options.ratio = 0.5;

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

TEST(SplatSimplify, RandomizedTwoPointMergeMatchesReference) {
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
        options.ratio = 0.5;
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

TEST(SplatSimplify, NoOpSimplifyPreservesAppearance) {
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
    options.ratio = 1.0;

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

TEST(SplatSimplify, FourPointVoxelMergeGroupsSpatiallyClosePoints) {
    RefInput input{
        .means = {
            0.00,
            0.00,
            0.00,
            0.02,
            0.00,
            0.00,
            2.00,
            0.00,
            0.00,
            2.02,
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
            std::log(0.5 / (1.0 - 0.5)),
            std::log(0.5 / (1.0 - 0.5)),
            std::log(0.5 / (1.0 - 0.5)),
            std::log(0.5 / (1.0 - 0.5)),
        },
        .appearance = {
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            3.0,
            3.0,
            3.0,
        },
        .app_dim = 3,
    };

    auto source = make_test_splat(input, 0);
    SplatSimplifyOptions options;
    options.ratio = 0.5;

    auto result = lfs::core::simplify_splats(*source, options, {});
    ASSERT_TRUE(result) << result.error();
    ASSERT_EQ((*result)->size(), 2u);

    const RefMerge expected_merge_01 = reference_moment_match(input, 0, 1);
    const RefMerge expected_merge_23 = reference_moment_match(input, 2, 3);

    const auto output_mean0 = mean_from_output_row(**result, 0);
    const auto output_mean1 = mean_from_output_row(**result, 1);

    // Determine which output row corresponds to which expected merge by mean proximity
    const bool first_is_01 = euclidean_distance(output_mean0, expected_merge_01.mean) < euclidean_distance(output_mean0, expected_merge_23.mean);
    const auto& expected0 = first_is_01 ? expected_merge_01 : expected_merge_23;
    const auto& expected1 = first_is_01 ? expected_merge_23 : expected_merge_01;
    const size_t row0 = first_is_01 ? 0 : 1;
    const size_t row1 = first_is_01 ? 1 : 0;

    expect_vec3_near(mean_from_output_row(**result, row0), expected0.mean, 5e-4);
    expect_mat3_near(covariance_from_output_row(**result, row0), expected0.sigma, 8e-4);
    EXPECT_NEAR(opacity_from_output_row(**result, row0), expected0.opacity, 5e-5);

    expect_vec3_near(mean_from_output_row(**result, row1), expected1.mean, 5e-4);
    expect_mat3_near(covariance_from_output_row(**result, row1), expected1.sigma, 8e-4);
    EXPECT_NEAR(opacity_from_output_row(**result, row1), expected1.opacity, 5e-5);

    const auto app0 = appearance_row(**result, row0);
    const auto app1 = appearance_row(**result, row1);
    ASSERT_EQ(app0.size(), expected0.appearance.size());
    ASSERT_EQ(app1.size(), expected1.appearance.size());
    for (size_t i = 0; i < app0.size(); ++i)
        EXPECT_NEAR(app0[i], expected0.appearance[i], 5e-4);
    for (size_t i = 0; i < app1.size(); ++i)
        EXPECT_NEAR(app1[i], expected1.appearance[i], 5e-4);
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
    options.ratio = 0.75;

    auto result = lfs::core::simplify_splats(*source, options, {});
    ASSERT_TRUE(result) << result.error();
    ASSERT_EQ((*result)->size(), 2u);
}

TEST(SplatSimplify, HistoryTracksMultiPassVoxelMergeTree) {
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
    options.ratio = 0.25;
    options.opacity_prune_threshold = 0.0f;

    auto result = lfs::core::simplify_splats_with_history(*source, options, {});
    ASSERT_TRUE(result) << result.error();
    ASSERT_NE(result->splat, nullptr);
    ASSERT_EQ(result->splat->size(), 1u);

    const auto& tree = result->merge_tree;
    EXPECT_EQ(tree.leaf_count(), 4);
    EXPECT_EQ(tree.target_count, 1);
    EXPECT_EQ(tree.post_prune_count, 4);
    EXPECT_TRUE(tree.pruned_leaf_ids.empty());
    EXPECT_EQ(tree.merge_count(), 3);
    EXPECT_EQ(tree.merge_pass, (std::vector<int32_t>{0, 0, 1}));
    ASSERT_EQ(tree.merge_left.size(), 3u);
    ASSERT_EQ(tree.merge_right.size(), 3u);
    std::vector<std::pair<int32_t, int32_t>> first_pass_pairs = {
        {tree.merge_left[0], tree.merge_right[0]},
        {tree.merge_left[1], tree.merge_right[1]},
    };
    std::sort(first_pass_pairs.begin(), first_pass_pairs.end());
    EXPECT_EQ(first_pass_pairs, (std::vector<std::pair<int32_t, int32_t>>{{0, 1}, {2, 3}}));
    EXPECT_EQ(tree.merge_left[2], 4);
    EXPECT_EQ(tree.merge_right[2], 5);
    EXPECT_EQ(tree.final_roots, (std::vector<int32_t>{6}));
}

TEST(SplatSimplify, HistoryTracksOpacityPrunedLeaves) {
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
            std::log(0.01 / (1.0 - 0.01)),
            std::log(0.02 / (1.0 - 0.02)),
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
    options.ratio = 0.75;

    auto result = lfs::core::simplify_splats_with_history(*source, options, {});
    ASSERT_TRUE(result) << result.error();
    ASSERT_NE(result->splat, nullptr);
    ASSERT_EQ(result->splat->size(), 2u);

    const auto& tree = result->merge_tree;
    EXPECT_EQ(tree.leaf_count(), 4);
    EXPECT_EQ(tree.target_count, 3);
    EXPECT_EQ(tree.post_prune_count, 2);
    EXPECT_EQ(tree.merge_count(), 0);
    EXPECT_EQ(tree.pruned_leaf_ids, (std::vector<int32_t>{0, 1}));
    EXPECT_EQ(tree.final_roots, (std::vector<int32_t>{2, 3}));
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
    options.ratio = 0.5;

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
    ASSERT_TRUE(std::find_if(stages.begin(),
                             stages.end(),
                             [](const std::string& stage) {
                                 return stage.starts_with("Pass 1: building voxel grid");
                             }) != stages.end());
    ASSERT_TRUE(std::find_if(stages.begin(),
                             stages.end(),
                             [](const std::string& stage) {
                                 return stage.starts_with("Pass 1: merging ");
                             }) != stages.end());
    ASSERT_TRUE(std::find(stages.begin(), stages.end(), "Complete") != stages.end());
}

TEST(SplatSimplify, VolumeBasedWeightsProduceDifferentResultThanAreaBased) {
    // Two splats with very different isotropic scales (0.5 vs 0.1) and same opacity.
    // Volume-based weights: 0.5^3 : 0.1^3 = 125 : 1
    // Area-based weights (sx*sy): 0.5^2 : 0.1^2 = 25 : 1
    // The merged center should be closer to the large splat than area-based would predict.
    RefInput input{
        .means = {
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
        },
        .scaling_raw = {
            std::log(0.5),
            std::log(0.5),
            std::log(0.5),
            std::log(0.1),
            std::log(0.1),
            std::log(0.1),
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
        },
        .opacity_raw = {
            std::log(0.5 / (1.0 - 0.5)),
            std::log(0.5 / (1.0 - 0.5)),
        },
        .appearance = {
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
    options.ratio = 0.5;

    auto result = lfs::core::simplify_splats(*source, options, {});
    ASSERT_TRUE(result) << result.error();
    ASSERT_EQ((*result)->size(), 1u);

    const auto actual_mean = mean_from_output_row(**result, 0);

    // Compute what the mean would be with area-based weights (alpha * sx * sy)
    const double alpha = 0.5;
    const double w0_area = alpha * 0.5 * 0.5;
    const double w1_area = alpha * 0.1 * 0.1;
    const double area_based_mean_x = (w0_area * 0.0 + w1_area * 1.0) / (w0_area + w1_area);

    // Volume-based should pull the mean closer to the large splat (x=0) than area-based
    EXPECT_LT(std::abs(actual_mean[0] - 0.0), std::abs(area_based_mean_x - 0.0));
    EXPECT_LT(actual_mean[0], area_based_mean_x);

    // Also verify it is very close to the large splat (within ~0.02)
    EXPECT_LT(actual_mean[0], 0.02);
}

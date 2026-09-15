/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/splat_data.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <gtest/gtest.h>
#include <random>

using namespace lfs::core;

namespace {
    glm::dmat3 covariance(const float* scales, const float* q) {
        const auto rotation = glm::mat3_cast(glm::normalize(glm::dquat(q[0], q[1], q[2], q[3])));
        glm::dmat3 diagonal(0.0);
        for (int i = 0; i < 3; ++i)
            diagonal[i][i] = std::exp(2.0 * scales[i]);
        return rotation * diagonal * glm::transpose(rotation);
    }

    SplatData fixture(Device device) {
        constexpr size_t n = 257; // Exercise the final partial CUDA block.
        std::vector<float> means(n * 3), scales(n * 3), rotations(n * 4);
        std::mt19937 random(913);
        std::uniform_real_distribution<float> position(-2, 2), scale(-12, 4), angle(-3, 3);
        for (size_t i = 0; i < n; ++i) {
            const auto q = glm::quat(glm::vec3(angle(random), angle(random), angle(random)));
            for (int j = 0; j < 3; ++j) {
                means[3 * i + j] = position(random);
                scales[3 * i + j] = scale(random);
            }
            rotations[4 * i] = q.w;
            rotations[4 * i + 1] = q.x;
            rotations[4 * i + 2] = q.y;
            rotations[4 * i + 3] = q.z;
        }
        return SplatData(0, Tensor::from_vector(means, {n, 3}, device),
                         Tensor::zeros({n, 1, 3}, device), {},
                         Tensor::from_vector(scales, {n, 3}, device),
                         Tensor::from_vector(rotations, {n, 4}, device),
                         Tensor::zeros({n, 1}, device), 1.0f);
    }

    void check_transform(SplatData data, const glm::mat4& matrix, const SplatData* reference = nullptr) {
        const auto& original = reference ? *reference : data;
        auto old_means = original.means_raw().cpu().contiguous();
        auto old_scales = original.scaling_raw().cpu().contiguous();
        auto old_rotations = original.rotation_raw().cpu().contiguous();
        transform(data, matrix);
        const auto means = data.means_raw().cpu().contiguous();
        const auto scales = data.scaling_raw().cpu().contiguous();
        const auto rotations = data.rotation_raw().cpu().contiguous();
        const glm::dmat3 a(matrix);
        const double log_volume = std::log(std::abs(glm::determinant(a)));
        for (size_t i = 0; i < data.size(); ++i) {
            const auto expected = a * covariance(old_scales.ptr<float>() + 3 * i, old_rotations.ptr<float>() + 4 * i) * glm::transpose(a);
            const auto actual = covariance(scales.ptr<float>() + 3 * i, rotations.ptr<float>() + 4 * i);
            double error = 0, norm = 0, old_log_volume = 0, new_log_volume = 0;
            for (int c = 0; c < 3; ++c) {
                for (int r = 0; r < 3; ++r) {
                    error += std::pow(actual[c][r] - expected[c][r], 2);
                    norm += expected[c][r] * expected[c][r];
                }
                old_log_volume += old_scales.ptr<float>()[3 * i + c];
                new_log_volume += scales.ptr<float>()[3 * i + c];
            }
            EXPECT_LT(std::sqrt(error / norm), 3e-6) << "splat " << i;
            // Also constrain the short axes, which a covariance-relative norm
            // alone can miss in a highly anisotropic splat.
            EXPECT_NEAR(new_log_volume - old_log_volume, log_volume, 4e-5) << "splat " << i;
            const float* p = old_means.ptr<float>() + 3 * i;
            const glm::dvec4 expected_mean = glm::dmat4(matrix) * glm::dvec4(p[0], p[1], p[2], 1);
            for (int c = 0; c < 3; ++c)
                EXPECT_NEAR(means.ptr<float>()[3 * i + c], expected_mean[c], 2e-6);
        }
    }
} // namespace

class SplatAffineTransform : public ::testing::TestWithParam<Device> {};

TEST_P(SplatAffineTransform, PreservesFullCovarianceAndVolume) {
    const auto rotation = glm::rotate(glm::mat4(1), 0.71f, glm::normalize(glm::vec3(1, 2, -1)));
    const auto translation = glm::translate(glm::mat4(1), glm::vec3(0.2f, -0.5f, 1.0f));
    glm::mat4 shear(1);
    shear[1][0] = 0.7f;
    shear[2][1] = -0.4f;
    for (const auto& matrix : {translation, translation * rotation,
                               translation * rotation * glm::scale(glm::mat4(1), glm::vec3(1.7f)),
                               translation * rotation * glm::scale(glm::mat4(1), glm::vec3(2, 0.6f, 1.4f)),
                               translation * shear * rotation,
                               translation * rotation * glm::scale(glm::mat4(1), glm::vec3(-1, 2, 0.4f))}) {
        check_transform(fixture(GetParam()), matrix);
    }
}

TEST_P(SplatAffineTransform, ZeroExtentDoesNotProduceNaN) {
    auto data = fixture(GetParam());
    data.scaling_raw() = Tensor::from_vector(std::vector<float>(data.size() * 3, -INFINITY),
                                             {data.size(), 3}, GetParam());
    transform(data, glm::scale(glm::mat4(1), glm::vec3(2, 0.6f, 1.4f)));
    const auto scales = data.scaling_raw().cpu();
    const auto rotations = data.rotation_raw().cpu();
    for (size_t i = 0; i < scales.numel(); ++i)
        EXPECT_EQ(scales.ptr<float>()[i], -INFINITY);
    for (size_t i = 0; i < rotations.numel(); ++i)
        EXPECT_TRUE(std::isfinite(rotations.ptr<float>()[i]));
}

INSTANTIATE_TEST_SUITE_P(CpuAndCuda, SplatAffineTransform, ::testing::Values(Device::CPU, Device::CUDA));

TEST(SplatAffineTransformCuda, NonblockingProducerPreservesGeometry) {
    const auto reference = fixture(Device::CPU);
    const auto source = fixture(Device::CUDA);
    cudaStream_t stream;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    auto data = source.clone_async(stream);
    // The oracle is already on CPU; no readback may synchronize the producer
    // before transform consumes its buffers.
    check_transform(std::move(data), glm::scale(glm::mat4(1), glm::vec3(2, 0.6f, 1.4f)), &reference);
    EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    CudaMemoryPool::instance().release_stream(stream);
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

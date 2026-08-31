/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/cuda/memory_arena.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "rasterization/edge_compute/rasterization/include/edge_rasterization_api.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::Tensor;

    struct DeviceScene {
        Tensor means;
        Tensor scales;
        Tensor rotations;
        Tensor opacities;
        Tensor w2c;
        Tensor pixel_weights;
    };

    DeviceScene make_random_scene(const int n, const int width, const int height) {
        std::mt19937 generator(0x9e3779b9u);
        std::uniform_real_distribution<float> xy(-1.6f, 1.6f);
        std::uniform_real_distribution<float> z(-2.0f, 8.0f);
        std::uniform_real_distribution<float> scale(-3.0f, -1.0f);
        std::uniform_real_distribution<float> opacity(-3.0f, 3.0f);
        std::normal_distribution<float> rotation(0.0f, 1.0f);
        std::vector<float> means(static_cast<size_t>(n) * 3);
        std::vector<float> scales(static_cast<size_t>(n) * 3);
        std::vector<float> rotations(static_cast<size_t>(n) * 4);
        std::vector<float> opacities(n);
        for (int i = 0; i < n; ++i) {
            means[i * 3 + 0] = xy(generator);
            means[i * 3 + 1] = xy(generator);
            means[i * 3 + 2] = z(generator);
            for (int j = 0; j < 3; ++j)
                scales[i * 3 + j] = scale(generator);
            float norm_sq = 0.0f;
            for (int j = 0; j < 4; ++j) {
                rotations[i * 4 + j] = rotation(generator);
                norm_sq += rotations[i * 4 + j] * rotations[i * 4 + j];
            }
            const float inv_norm = 1.0f / std::sqrt(norm_sq);
            for (int j = 0; j < 4; ++j)
                rotations[i * 4 + j] *= inv_norm;
            opacities[i] = opacity(generator);
        }

        std::vector<float> w2c(16, 0.0f);
        w2c[0] = w2c[5] = w2c[10] = w2c[15] = 1.0f;
        std::vector<float> weights(static_cast<size_t>(width) * height);
        // One active pixel keeps the legacy atomic reduction deterministic,
        // while the full tiled image still exercises visibility compaction,
        // tile/depth sorting, and random culling.
        std::fill(weights.begin(), weights.end(), 0.0f);
        weights[(static_cast<size_t>(height) / 2) * width + width / 2] = 1.0f;

        return {
            Tensor::from_vector(means, {static_cast<size_t>(n), 3}, Device::CUDA),
            Tensor::from_vector(scales, {static_cast<size_t>(n), 3}, Device::CUDA),
            Tensor::from_vector(rotations, {static_cast<size_t>(n), 4}, Device::CUDA),
            Tensor::from_vector(opacities, {static_cast<size_t>(n)}, Device::CUDA),
            Tensor::from_vector(w2c, {4, 4}, Device::CUDA),
            Tensor::from_vector(weights, {static_cast<size_t>(height), static_cast<size_t>(width)}, Device::CUDA)};
    }

    Tensor run_edge(const DeviceScene& scene, const int n, const int width, const int height,
                    const bool mip_filter, const bool legacy) {
        Tensor scores = Tensor::zeros({static_cast<size_t>(n)}, Device::CUDA, DataType::Float32);
        auto result = legacy
                          ? edge_compute::rasterization::edge_forward_raw_legacy_for_testing(
                                scene.means.ptr<float>(), scene.scales.ptr<float>(),
                                scene.rotations.ptr<float>(), scene.opacities.ptr<float>(),
                                scene.w2c.ptr<float>(), n, width, height,
                                80.0f, 80.0f, width * 0.5f, height * 0.5f,
                                0.01f, 100.0f, mip_filter,
                                scene.pixel_weights.ptr<float>(), scores.ptr<float>())
                          : edge_compute::rasterization::edge_forward_raw(
                                scene.means.ptr<float>(), scene.scales.ptr<float>(),
                                scene.rotations.ptr<float>(), scene.opacities.ptr<float>(),
                                scene.w2c.ptr<float>(), n, width, height,
                                80.0f, 80.0f, width * 0.5f, height * 0.5f,
                                0.01f, 100.0f, mip_filter,
                                scene.pixel_weights.ptr<float>(), scores.ptr<float>());
        if (!result.success)
            throw std::runtime_error(result.error_message != nullptr ? result.error_message : "unknown EDGE failure");
        auto& arena = lfs::core::GlobalArenaManager::instance().get_arena();
        arena.end_frame(result.frame_id, lfs::core::getCurrentCUDAStream());
        return scores;
    }

} // namespace

TEST(EdgeRasterizerTest, PackedVisibleCountPathIsBitExactAgainstLegacy) {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count == 0)
        GTEST_SKIP() << "CUDA device not available";

    constexpr int n = 4099;
    constexpr int width = 97;
    constexpr int height = 83;
    const auto scene = make_random_scene(n, width, height);
    for (const bool mip_filter : {false, true}) {
        const Tensor old_scores = run_edge(scene, n, width, height, mip_filter, true).to(Device::CPU);
        const Tensor new_scores = run_edge(scene, n, width, height, mip_filter, false).to(Device::CPU);
        ASSERT_EQ(old_scores.numel(), new_scores.numel());
        const auto* old_ptr = old_scores.ptr<float>();
        const auto* new_ptr = new_scores.ptr<float>();
        size_t mismatch_count = 0;
        size_t first_mismatch = 0;
        for (size_t i = 0; i < old_scores.numel(); ++i) {
            if (std::memcmp(old_ptr + i, new_ptr + i, sizeof(float)) != 0) {
                if (mismatch_count++ == 0)
                    first_mismatch = i;
            }
        }
        ASSERT_EQ(mismatch_count, 0u)
            << "EDGE score mismatch for mip_filter=" << mip_filter
            << " count=" << mismatch_count << " first=" << first_mismatch
            << " old=" << old_ptr[first_mismatch] << " new=" << new_ptr[first_mismatch];
    }
}

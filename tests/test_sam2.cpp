/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "core/nn/models/sam2.hpp"
#include "core/nn/ops.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <optional>
#include <vector>

namespace {
    using namespace lfs::core;
    const auto photo = std::filesystem::path(PROJECT_ROOT_PATH) / "tests/data/nn/lpips_crop_a.png";
    struct TestStream {
        cudaStream_t value = nullptr;
        TestStream() { EXPECT_EQ(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking), cudaSuccess); }
        ~TestStream() {
            cudaStreamSynchronize(value);
            CudaMemoryPool::instance().release_stream(value);
            cudaStreamDestroy(value);
        }
    };

} // namespace

TEST(Sam2Morphology, RealImageComponentsAndDiskDilation) {
    ::testing::Test::RecordProperty("catches", "Wrong component polarity, connectivity, size cutoff or square dilation changes real-image mask pixels");
    TestStream stream;
    CUDAStreamGuard guard(stream.value);
    auto [pixels, width, height, channels] = load_image(photo);
    ASSERT_EQ(channels, 3);
    std::vector<float> scores(size_t(width) * height);
    for (size_t i = 0; i < scores.size(); ++i)
        scores[i] = float(pixels[i * 3]) - 127.5f;
    free_image(pixels);
    const auto tensor = Tensor::from_vector(scores, {1, 1, size_t(height), size_t(width)}, Device::CPU).to(Device::CUDA, stream.value);
    for (const bool foreground : {false, true}) {
        std::vector<bool> visited(scores.size());
        auto expected = scores;
        size_t changed = 0;
        for (size_t first = 0; first < scores.size(); ++first) {
            if (visited[first] || (scores[first] > 0) != foreground)
                continue;
            std::vector<size_t> component{first};
            visited[first] = true;
            for (size_t cursor = 0; cursor < component.size(); ++cursor) {
                const int x = component[cursor] % width, y = component[cursor] / width;
                for (int dy = -1; dy <= 1; ++dy)
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int nx = x + dx, ny = y + dy;
                        if (nx < 0 || ny < 0 || nx >= width || ny >= height)
                            continue;
                        const size_t next = size_t(ny) * width + nx;
                        if (!visited[next] && (scores[next] > 0) == foreground) {
                            visited[next] = true;
                            component.push_back(next);
                        }
                    }
            }
            if (component.size() <= 8)
                for (const auto index : component) {
                    expected[index] = foreground ? -0.1f : 0.1f;
                    ++changed;
                }
        }
        ASSERT_GT(changed, 0u);
        const auto actual = (foreground ? nn::remove_small_foreground(tensor, -0.1f) : nn::fill_small_background(tensor, 0.1f)).to(Device::CPU).to_vector();
        EXPECT_EQ(actual, expected);
    }
    const auto actual = nn::dilate_binary_disk(tensor.gt(0).to(DataType::Float32), 3).to(Device::CPU).to_vector();
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            bool expected = false;
            for (int dy = -3; dy <= 3; ++dy)
                for (int dx = -3; dx <= 3; ++dx) {
                    const int nx = x + dx, ny = y + dy;
                    if (dx * dx + dy * dy <= 9 && nx >= 0 && ny >= 0 && nx < width && ny < height)
                        expected |= scores[size_t(ny) * width + nx] > 0;
                }
            ASSERT_EQ(actual[size_t(y) * width + x], float(expected)) << x << ',' << y;
        }
}

TEST(Sam2Predict, MasksDoNotDependOnEarlierPrompts) {
    ::testing::Test::RecordProperty("catches", "Decoder scratch shared with a later predict's activations changing its masks, as a workspace "
                                               "taken from the activation arena did after the Float16 encoder");
    const auto weights = [] {
        if (const char* env = std::getenv("LFS_SAM2_WEIGHTS"); env && env[0])
            return std::filesystem::path(env);
        const char* home = std::getenv("HOME");
        return home ? std::filesystem::path(home) / ".lichtfeld/onnx/sam2.1-hiera-base-plus.lfw" : std::filesystem::path{};
    }();
    if (weights.empty() || !std::filesystem::is_regular_file(weights))
        GTEST_SKIP() << "SAM2 weights absent: " << weights;
    TestStream stream;
    CUDAStreamGuard guard(stream.value);
    auto [pixels, width, height, channels] = load_image(photo);
    ASSERT_EQ(channels, 3);
    auto cpu = Tensor::empty({size_t(height), size_t(width), 3}, Device::CPU, DataType::UInt8);
    std::memcpy(cpu.data_ptr(), pixels, cpu.bytes());
    free_image(pixels);
    const auto image = cpu.to(Device::CUDA, stream.value).to(DataType::Float32).div(255.0f).permute({2, 0, 1}).contiguous().unsqueeze(0);
    const float w = float(width), h = float(height);
    const std::vector<nn::models::Sam2PointPrompt> one{{0.5f * w, 0.55f * h, 1}};
    const std::array<float, 4> box{0.3f * w, 0.3f * h, 0.7f * w, 0.8f * h};
    const std::vector<nn::models::Sam2PointPrompt> four{
        {0.35f * w, 0.4f * h, 1},
        {0.65f * w, 0.4f * h, 1},
        {0.4f * w, 0.7f * h, 1},
        {0.6f * w, 0.7f * h, 1}};
    const std::array<float, 4> wide{0.2f * w, 0.2f * h, 0.8f * w, 0.9f * h};
    for (const auto encoder : {DataType::Float32, DataType::Float16}) {
        SCOPED_TRACE(encoder == DataType::Float16 ? "Float16 encoder" : "Float32 encoder");
        auto model = nn::models::Sam2::load(weights, Device::CUDA, DataType::Float32);
        ASSERT_TRUE(model.has_value());
        if (encoder == DataType::Float16)
            model->set_encoder_dtype(DataType::Float16);
        ASSERT_TRUE(model->set_image(image).has_value());
        const auto alone = model->predict(one, box, true);
        ASSERT_TRUE(alone.has_value());
        const auto expected = alone->low_res_logits.to(Device::CPU).contiguous().to_vector();
        ASSERT_TRUE(model->predict(four, wide, true).has_value());
        ASSERT_TRUE(model->predict(four, std::nullopt, true).has_value());
        const auto again = model->predict(one, box, true);
        ASSERT_TRUE(again.has_value());
        const auto actual = again->low_res_logits.to(Device::CPU).contiguous().to_vector();
        ASSERT_EQ(actual.size(), expected.size());
        float maximum_error = 0;
        for (size_t i = 0; i < actual.size(); ++i)
            maximum_error = std::max(maximum_error, std::abs(actual[i] - expected[i]));
        EXPECT_LE(maximum_error, 1e-3f);
    }
}

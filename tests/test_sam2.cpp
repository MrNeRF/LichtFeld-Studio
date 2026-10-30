/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "core/nn/ops.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "core/tensor/internal/memory_pool.hpp"
#include <filesystem>
#include <gtest/gtest.h>

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

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <gtest/gtest.h>

#include "core/cuda/lanczos_resize/lanczos_resize.hpp"
#include "core/tensor.hpp"
#include "cuda_backend_test.hpp"
#include "io/nvcodec_image_loader.hpp"
#include "kernels/densification_kernels.hpp"
#include "kernels/image_kernels.hpp"
#include "metrics/eval_mask.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <vector>

using namespace lfs::core;
using namespace lfs::training::kernels;

class ImageKernelsTest : public lfs::test::CudaBackendTest {};

TEST_F(ImageKernelsTest, FusedCannyUInt8MatchesNormalizedFloatInput) {
    constexpr int C = 3;
    constexpr int H = 40;
    constexpr int W = 37;

    std::vector<float> normalized_data(C * H * W);
    std::vector<float> byte_data(C * H * W);
    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                const int idx = (c * H + y) * W + x;
                const int byte_value = (c * 53 + y * 7 + x * 11) % 256;
                normalized_data[idx] = static_cast<float>(byte_value) * (1.0f / 255.0f);
                byte_data[idx] = static_cast<float>(byte_value);
            }
        }
    }

    auto float_input = Tensor::from_vector(normalized_data, TensorShape({C, H, W}), Device::GPU);
    auto uint8_input = Tensor::from_vector(byte_data, TensorShape({C, H, W}), Device::GPU)
                           .to(DataType::UInt8);
    auto float_output = Tensor::zeros({H, W}, Device::GPU, DataType::Float32);
    auto uint8_output = Tensor::zeros({H, W}, Device::GPU, DataType::Float32);

    launch_fused_canny_edge_filter_chw(
        float_input.ptr<float>(),
        float_output.ptr<float>(),
        H,
        W);
    auto err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    launch_fused_canny_edge_filter_chw(
        uint8_input.ptr<uint8_t>(),
        uint8_output.ptr<float>(),
        H,
        W);
    err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    const auto float_cpu = float_output.cpu();
    const auto uint8_cpu = uint8_output.cpu();
    const float* float_ptr = float_cpu.ptr<float>();
    const float* uint8_ptr = uint8_cpu.ptr<float>();

    float max_abs_diff = 0.0f;
    for (int i = 0; i < H * W; ++i) {
        max_abs_diff = std::max(max_abs_diff, std::abs(float_ptr[i] - uint8_ptr[i]));
    }

    EXPECT_LT(max_abs_diff, 1e-5f);
}

TEST_F(ImageKernelsTest, EdgeWeightFloatPositiveMedianPreservesNonQuantizedValues) {
    const std::vector<float> values{0.0f, 0.2f, 0.3f, 0.46f};
    auto weights = Tensor::from_vector(
        values, TensorShape({2, 2}), Device::GPU);
    launch_normalize_by_positive_median(weights.ptr<float>(), weights.numel(), weights.stream());

    const auto result = weights.cpu();
    const auto* ptr = result.ptr<float>();
    EXPECT_FLOAT_EQ(ptr[0], 0.0f);
    EXPECT_NEAR(ptr[1], 2.0f / 3.0f, 1.0e-6f);
    EXPECT_FLOAT_EQ(ptr[2], 1.0f);
    EXPECT_NEAR(ptr[3], 23.0f / 15.0f, 1.0e-6f);
    // 0.46 / 0.3 is not representable by the rejected u8/16 cache path.
    EXPECT_GT(std::abs(ptr[3] - std::round(ptr[3] * 16.0f) / 16.0f), 0.01f);
}

namespace {
    uint32_t radix_order_key(const float v) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
    }

    std::filesystem::path first_dataset_jpeg() {
        std::error_code ec;
        for (const auto& dataset : std::filesystem::directory_iterator(
                 std::filesystem::path(PROJECT_ROOT_PATH) / "data", ec)) {
            std::vector<std::filesystem::path> jpegs;
            for (const auto& entry : std::filesystem::directory_iterator(dataset.path() / "images_4", ec)) {
                auto ext = entry.path().extension().string();
                std::ranges::transform(ext, ext.begin(), [](unsigned char c) { return std::tolower(c); });
                if (ext == ".jpg" || ext == ".jpeg")
                    jpegs.push_back(entry.path());
            }
            if (!jpegs.empty())
                return std::ranges::min(jpegs);
        }
        return {};
    }
} // namespace

// Production input: the Canny edge map of a real training image. A radix select
// that picked a neighboring order statistic would move every value by far more
// than the 2 ulp the GPU division may differ from the host.
TEST_F(ImageKernelsTest, EdgeWeightPositiveMedianMatchesSortOnRealImage) {
    const auto path = first_dataset_jpeg();
    if (path.empty()) {
        GTEST_SKIP() << "no dataset with images_4 JPEGs under data/";
    }
    std::ifstream file(path, std::ios::binary);
    const std::vector<uint8_t> jpeg{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    ASSERT_FALSE(jpeg.empty());
    std::unique_ptr<lfs::io::NvCodecImageLoader> loader;
    try {
        loader = std::make_unique<lfs::io::NvCodecImageLoader>(lfs::io::NvCodecImageLoader::Options{});
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nvImageCodec unavailable: " << e.what();
    }
    const auto decoded = loader->decode_jpeg_batch_from_spans({{jpeg.data(), jpeg.size()}}, nullptr, true, true);
    ASSERT_EQ(decoded.size(), 1u);
    const auto& image = decoded[0];
    ASSERT_EQ(image.ndim(), 3u);
    ASSERT_EQ(image.shape()[0], 3u);
    const int H = static_cast<int>(image.shape()[1]);
    const int W = static_cast<int>(image.shape()[2]);

    auto edges = Tensor::zeros({static_cast<size_t>(H), static_cast<size_t>(W)}, Device::CUDA, DataType::Float32);
    launch_fused_canny_edge_filter_chw(image.ptr<uint8_t>(), edges.ptr<float>(), H, W);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto raw = edges.cpu().to_vector();

    launch_normalize_by_positive_median(edges.ptr<float>(), edges.numel());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto got = edges.cpu().to_vector();

    std::vector<float> positives;
    for (const float v : raw) {
        if (v > 0.0f)
            positives.push_back(v);
    }
    ASSERT_GT(positives.size(), 1000u) << "a real photo must have edges";
    std::ranges::sort(positives, {}, radix_order_key);
    const float median = std::max(positives[positives.size() / 2], 1e-9f);
    ASSERT_EQ(got.size(), raw.size());
    for (size_t i = 0; i < raw.size(); ++i) {
        const float expected = std::isnan(raw[i]) ? 0.0f : raw[i] / median;
        const auto a = radix_order_key(got[i]);
        const auto b = radix_order_key(expected);
        ASSERT_LE(a > b ? a - b : b - a, 2u) << "i=" << i << " got=" << got[i] << " expected=" << expected;
    }
}

TEST_F(ImageKernelsTest, LanczosRgbAndGrayscaleUseBoundedCoefficientBuffers) {
    auto rgb = Tensor::full({4, 6, 3}, 255.0f, Device::GPU, DataType::UInt8);
    auto grayscale = Tensor::full({4, 6}, 255.0f, Device::GPU, DataType::UInt8);

    const auto rgb_output = lanczos_resize(rgb, 3, 5, 2, nullptr);
    const auto grayscale_output = lanczos_resize_grayscale(grayscale, 3, 5, 2, nullptr);

    ASSERT_TRUE(rgb_output.is_valid());
    ASSERT_EQ(rgb_output.shape(), TensorShape({3, 3, 5}));
    ASSERT_TRUE(grayscale_output.is_valid());
    ASSERT_EQ(grayscale_output.shape(), TensorShape({3, 5}));
    EXPECT_TRUE(rgb_output.isfinite().all().item<bool>());
    EXPECT_TRUE(grayscale_output.isfinite().all().item<bool>());
}

// Catches an interleaved kernel that strides the input by a fixed 3 channels: for
// 1, 2 and 4 channels it would read neighbouring pixels instead of its own plane.
TEST_F(ImageKernelsTest, LanczosInterleavedChannelsMatchPerPlaneGrayscale) {
    constexpr int SOURCE_WIDTH = 19;
    constexpr int SOURCE_HEIGHT = 13;
    constexpr int OUTPUT_WIDTH = 7;
    constexpr int OUTPUT_HEIGHT = 5;
    for (int channels = 1; channels <= 4; ++channels) {
        SCOPED_TRACE(channels);
        std::vector<float> source(static_cast<size_t>(SOURCE_WIDTH) * SOURCE_HEIGHT * channels);
        for (size_t index = 0; index < source.size(); ++index)
            source[index] = static_cast<float>((index * 37 + channels * 11) % 997) / 996.0f;
        const auto hwc = Tensor::from_blob(
                             source.data(), TensorShape({SOURCE_HEIGHT, SOURCE_WIDTH, static_cast<size_t>(channels)}),
                             Device::CPU, DataType::Float32)
                             .to(Device::CUDA);

        const auto interleaved = lanczos_resize(hwc, OUTPUT_HEIGHT, OUTPUT_WIDTH, 2, nullptr);
        ASSERT_TRUE(interleaved.is_valid());
        ASSERT_EQ(interleaved.shape(), TensorShape({static_cast<size_t>(channels), OUTPUT_HEIGHT, OUTPUT_WIDTH}));
        const auto got = interleaved.cpu().to_vector();

        for (int channel = 0; channel < channels; ++channel) {
            std::vector<float> plane_values(static_cast<size_t>(SOURCE_WIDTH) * SOURCE_HEIGHT);
            for (size_t pixel = 0; pixel < plane_values.size(); ++pixel)
                plane_values[pixel] = source[pixel * channels + channel];
            const auto plane = Tensor::from_blob(plane_values.data(), TensorShape({SOURCE_HEIGHT, SOURCE_WIDTH}),
                                                 Device::CPU, DataType::Float32)
                                   .to(Device::CUDA);
            const auto expected = lanczos_resize_grayscale(plane, OUTPUT_HEIGHT, OUTPUT_WIDTH, 2, nullptr).cpu().to_vector();
            const size_t offset = static_cast<size_t>(channel) * OUTPUT_WIDTH * OUTPUT_HEIGHT;
            for (size_t index = 0; index < expected.size(); ++index)
                EXPECT_NEAR(got[offset + index], expected[index], 1e-6f) << "channel=" << channel << " index=" << index;
        }
    }
}

TEST_F(ImageKernelsTest, LanczosRejectsNonPositiveOutputExtentBeforeAllocation) {
    const auto input = Tensor::zeros({2, 2, 3}, Device::GPU, DataType::UInt8);
    EXPECT_FALSE(lanczos_resize(input, 0, 2, 2, nullptr).is_valid());
    EXPECT_FALSE(lanczos_resize(input, 2, -1, 2, nullptr).is_valid());
}

// Fails if the target keeps the colour stored under transparent pixels (the render shows the background there),
// if the per-pixel background is ignored, or if uint8 targets are not normalised.
TEST_F(ImageKernelsTest, CompositeOverBackgroundShowsBackgroundWhereTransparent) {
    const std::vector<float> rgb{0.2f, 0.9f, 0.5f, 0.4f, 0.1f, 0.3f, 0.7f, 0.6f, 0.8f, 0.0f, 1.0f, 0.25f};
    const std::vector<float> alpha{1.0f, 0.0f, 0.5f, 0.25f};
    const std::vector<float> color{0.1f, 0.2f, 0.3f};
    std::vector<float> backdrop(12);
    for (size_t i = 0; i < backdrop.size(); ++i)
        backdrop[i] = 0.05f * static_cast<float>(i);
    const auto rgb_gpu = Tensor::from_vector(rgb, {3, 2, 2}, Device::CUDA);
    const auto alpha_gpu = Tensor::from_vector(alpha, {2, 2}, Device::CUDA);

    const auto solid = lfs::training::composite_over_background(rgb_gpu, alpha_gpu, Tensor::from_vector(color, {3}, Device::CUDA))
                           .cpu()
                           .to_vector();
    const auto image = lfs::training::composite_over_background(rgb_gpu, alpha_gpu, Tensor::from_vector(backdrop, {3, 2, 2}, Device::CUDA))
                           .cpu()
                           .to_vector();
    for (size_t c = 0; c < 3; ++c)
        for (size_t p = 0; p < 4; ++p) {
            const size_t i = c * 4 + p;
            EXPECT_NEAR(solid[i], rgb[i] * alpha[p] + color[c] * (1.0f - alpha[p]), 1e-6f) << i;
            EXPECT_NEAR(image[i], rgb[i] * alpha[p] + backdrop[i] * (1.0f - alpha[p]), 1e-6f) << i;
        }

    std::vector<uint8_t> bytes(rgb.size());
    for (size_t i = 0; i < rgb.size(); ++i)
        bytes[i] = static_cast<uint8_t>(std::lround(rgb[i] * 255.0f));
    auto bytes_gpu = Tensor::empty({3, 2, 2}, Device::CUDA, DataType::UInt8);
    ASSERT_EQ(cudaMemcpy(bytes_gpu.ptr<uint8_t>(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice), cudaSuccess);
    const auto from_bytes = lfs::training::composite_over_background(bytes_gpu, alpha_gpu, Tensor::from_vector(color, {3}, Device::CUDA))
                                .cpu()
                                .to_vector();
    for (size_t i = 0; i < from_bytes.size(); ++i)
        EXPECT_NEAR(from_bytes[i], bytes[i] / 255.0f * alpha[i % 4] + color[i / 4] * (1.0f - alpha[i % 4]), 1e-6f) << i;
}

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "core/tensor.hpp"
#include "training/metrics/flip.cuh"

#include <filesystem>
#include <gtest/gtest.h>
#include <torch/torch.h>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;

namespace {
    Tensor load_chw(const std::filesystem::path& path) {
        auto [data, width, height, channels] = lfs::core::load_image(path);
        EXPECT_NE(data, nullptr) << path;
        EXPECT_GE(channels, 3);
        const auto hwc = Tensor::from_blob(data, {static_cast<size_t>(height), static_cast<size_t>(width), static_cast<size_t>(channels)},
                                           Device::CPU, DataType::UInt8)
                             .to(DataType::Float32)
                             .div(255.0f);
        auto chw = hwc.slice(2, 0, 3).permute({2, 0, 1}).contiguous().to(Device::CUDA);
        lfs::core::free_image(data);
        return chw;
    }
} // namespace

// Catches drift from the published LDR-FLIP: the expected mean is the float64 mean of the reference
// implementation's error map for the same decoded pixels (its own reported mean is a float32 running sum).
TEST(FlipMetric, MatchesTheReferenceImplementationOnRealImages) {
    if (!torch::cuda::is_available())
        GTEST_SKIP() << "CUDA not available";
    const auto dir = std::filesystem::path(TEST_DATA_DIR) / "bicycle" / "images_8";
    const auto reference = load_chw(dir / "_DSC8679.JPG");
    const auto test = load_chw(dir / "_DSC8680.JPG");
    const auto error = lfs::training::flip_error_map(reference, test);
    ASSERT_EQ(error.shape(), lfs::core::TensorShape({reference.shape()[1], reference.shape()[2]}));
    EXPECT_NEAR(error.mean().item<float>(), 0.72386001f, 2e-6f);
    EXPECT_GE(error.min().item<float>(), 0.0f);
    EXPECT_LE(error.max().item<float>(), 1.0f);

    EXPECT_EQ(lfs::training::flip_error_map(reference, reference).max().item<float>(), 0.0f);
    const auto image = lfs::training::flip_error_image(Tensor::zeros({2, 2}, Device::CUDA)).cpu().to_vector_uint8();
    EXPECT_EQ(image, (std::vector<uint8_t>{0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4}));
}

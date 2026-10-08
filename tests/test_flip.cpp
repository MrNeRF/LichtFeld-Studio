/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/image_io.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "cuda_backend_test.hpp"
#include "training/metrics/flip.hpp"

#include <filesystem>
#include <format>
#include <gtest/gtest.h>

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
        auto chw = hwc.slice(2, 0, 3).permute({2, 0, 1}).contiguous().to(Device::GPU);
        lfs::core::free_image(data);
        return chw;
    }
} // namespace

// Catches drift from the published LDR-FLIP: the expected mean is the float64 mean of the reference
// implementation's error map for the same decoded pixels (its own reported mean is a float32 running sum).
TEST(FlipMetric, MatchesTheReferenceImplementationOnRealImages) {
    if (!lfs::core::gpu_backend_available(lfs::core::default_gpu_backend()))
        GTEST_SKIP() << "Selected GPU backend not available";
    const auto dir = std::filesystem::path(TEST_DATA_DIR) / "bicycle" / "images_8";
    if (!std::filesystem::is_regular_file(dir / "_DSC8679.JPG") ||
        !std::filesystem::is_regular_file(dir / "_DSC8680.JPG"))
        GTEST_SKIP() << "bicycle images_8 reference pair is absent: " << dir;
    const auto reference = load_chw(dir / "_DSC8679.JPG");
    const auto test = load_chw(dir / "_DSC8680.JPG");
    const auto error = lfs::training::flip_error_map(reference, test);
    ASSERT_EQ(error.shape(), lfs::core::TensorShape({reference.shape()[1], reference.shape()[2]}));
    const float mean = error.mean().item<float>();
    RecordProperty("mean_error", std::format("{:.9g}", mean));
    EXPECT_NEAR(mean, 0.72386001f, 2e-6f);
    EXPECT_GE(error.min().item<float>(), 0.0f);
    EXPECT_LE(error.max().item<float>(), 1.0f);

    EXPECT_EQ(lfs::training::flip_error_map(reference, reference).max().item<float>(), 0.0f);
    const auto image = lfs::training::flip_error_image(Tensor::zeros({2, 2}, Device::GPU)).cpu().to_vector_uint8();
    EXPECT_EQ(image, (std::vector<uint8_t>{0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4}));
}

TEST(FlipMetric, NativeBackendsMatchCudaPerPixel) {
    using lfs::core::GpuBackend;
    if (!lfs::core::gpu_backend_available(GpuBackend::CUDA))
        GTEST_SKIP() << "CUDA reference unavailable";
    const auto dir = std::filesystem::path(TEST_DATA_DIR) / "bicycle" / "images_8";
    if (!std::filesystem::is_regular_file(dir / "_DSC8679.JPG") ||
        !std::filesystem::is_regular_file(dir / "_DSC8680.JPG"))
        GTEST_SKIP() << "bicycle images_8 reference pair is absent: " << dir;
    const auto capture = [&](GpuBackend backend) {
        const lfs::test::DefaultGpuBackendForTesting scope(backend);
        EXPECT_TRUE(scope.switched());
        const auto reference = load_chw(dir / "_DSC8679.JPG");
        const auto test = load_chw(dir / "_DSC8680.JPG");
        return lfs::training::flip_error_map(reference, test).cpu().contiguous();
    };
    const auto expected = capture(GpuBackend::CUDA);
    for (const auto backend : {GpuBackend::Vulkan, GpuBackend::Metal}) {
        if (!lfs::core::gpu_backend_available(backend))
            continue;
        SCOPED_TRACE(static_cast<int>(backend));
        const auto actual = capture(backend);
        ASSERT_EQ(actual.shape(), expected.shape());
        const float maximum = (actual - expected).abs().max().item<float>();
        RecordProperty(backend == GpuBackend::Vulkan ? "vulkan_max_error" : "metal_max_error", std::format("{:.9g}", maximum));
        EXPECT_LE(maximum, 5e-5f);
    }
}

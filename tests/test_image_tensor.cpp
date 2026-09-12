/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "rendering/image_layout.hpp"
#include "rendering/image_tensor.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
    using lfs::core::DataType;
    using lfs::core::Device;
    using lfs::core::gpu_backend_available;
    using lfs::core::gpu_backend_of;
    using lfs::core::GpuBackend;
    using lfs::core::GpuBackendScope;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;
    using lfs::rendering::detectImageLayout;
    using lfs::rendering::flipImageVertical;
    using lfs::rendering::ImageLayout;
    using lfs::rendering::prepareImageRgba8;

    Tensor cpu_u8(const std::vector<std::uint8_t>& bytes, const TensorShape& shape) {
        Tensor tensor = Tensor::empty(shape, Device::CPU, DataType::UInt8);
        if (!bytes.empty()) {
            std::memcpy(tensor.ptr<std::uint8_t>(), bytes.data(), bytes.size());
        }
        return tensor;
    }

    Tensor cpu_f32(const std::vector<float>& values, const TensorShape& shape) {
        return Tensor::from_vector(values, shape, Device::CPU);
    }

    Tensor to_backend(const Tensor& host, const GpuBackend backend) {
        GpuBackendScope scope(backend);
        return host.to(Device::GPU);
    }

    std::vector<GpuBackend> gpu_backends() {
        std::vector<GpuBackend> backends;
        if (gpu_backend_available(GpuBackend::CUDA)) {
            backends.push_back(GpuBackend::CUDA);
        }
        if (gpu_backend_available(GpuBackend::Vulkan)) {
            backends.push_back(GpuBackend::Vulkan);
        }
        return backends;
    }

    const char* backend_name(const GpuBackend backend) {
        return backend == GpuBackend::CUDA ? "cuda" : "vulkan";
    }

    GpuBackend other_backend(const GpuBackend backend) {
        return backend == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA;
    }

    void expect_bytes(const Tensor& tensor, const std::vector<std::uint8_t>& expected) {
        ASSERT_TRUE(tensor.is_valid());
        EXPECT_EQ(tensor.dtype(), DataType::UInt8);
        EXPECT_EQ(tensor.ndim(), 3u);
        EXPECT_EQ(tensor.size(2), 4u);
        EXPECT_TRUE(tensor.is_contiguous());
        const auto got = tensor.to_vector_uint8();
        ASSERT_EQ(got.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(got[i], expected[i]) << "index=" << i;
        }
    }

    // 2x2 HWC RGB: unique pixels so CHW/HWC/flip mistakes are visible.
    const std::vector<float> kRgbHwcF32{
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        0.0f,
        1.0f,
        1.0f,
        1.0f,
        0.0f,
    };
    const std::vector<std::uint8_t> kRgbHwcRgba{
        255,
        0,
        0,
        255,
        0,
        255,
        0,
        255,
        0,
        0,
        255,
        255,
        255,
        255,
        0,
        255,
    };
    const std::vector<std::uint8_t> kRgbHwcRgbaFlipped{
        0,
        0,
        255,
        255,
        255,
        255,
        0,
        255,
        255,
        0,
        0,
        255,
        0,
        255,
        0,
        255,
    };

    const std::vector<std::uint8_t> kGrayHwcU8{10, 20, 30, 40};
    const std::vector<std::uint8_t> kGrayHwcRgba{
        10,
        10,
        10,
        255,
        20,
        20,
        20,
        255,
        30,
        30,
        30,
        255,
        40,
        40,
        40,
        255,
    };

    const std::vector<std::uint8_t> kRgbaHwcU8{
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    };
} // namespace

TEST(ImageLayout, DetectsAmbiguousCubeAsChw) {
    const Tensor cube = cpu_f32(std::vector<float>(27, 0.0f), {3, 3, 3});
    EXPECT_EQ(detectImageLayout(cube), ImageLayout::CHW);
}

TEST(ImageTensorPrepare, CpuUint8HwcRgbIdentityAlphaFill) {
    const Tensor input = cpu_u8(
        {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0}, {2, 2, 3});
    const Tensor out = prepareImageRgba8(input, false);
    EXPECT_EQ(out.device(), Device::CPU);
    EXPECT_FALSE(gpu_backend_of(out).has_value());
    expect_bytes(out, kRgbHwcRgba);
}

TEST(ImageTensorPrepare, CpuGrayFillsRgbAndOpaqueAlpha) {
    const Tensor input = cpu_u8(kGrayHwcU8, {2, 2, 1});
    expect_bytes(prepareImageRgba8(input, false), kGrayHwcRgba);
}

TEST(ImageTensorPrepare, CpuRgbaPreservesAlpha) {
    const Tensor input = cpu_u8(kRgbaHwcU8, {2, 2, 4});
    expect_bytes(prepareImageRgba8(input, false), kRgbaHwcU8);
}

TEST(ImageTensorPrepare, CpuFloatClampNearestAndFlip) {
    const Tensor input = cpu_f32(
        {-1.0f, 0.5f, 2.0f, 0.0f, 1.0f, 0.004f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f},
        {2, 2, 3});
    // Harmonized nearest (toByte / image export): clamp*255+0.5 trunc.
    // 0.5 -> 128 (was 127 under the old CPU trunc fallback), 0.004*255+0.5 -> 1.
    const std::vector<std::uint8_t> expected{
        0,
        0,
        255,
        255,
        255,
        255,
        0,
        255,
        0,
        128,
        255,
        255,
        0,
        255,
        1,
        255,
    };
    expect_bytes(prepareImageRgba8(input, true), expected);
}

TEST(ImageTensorPrepare, CpuChwMatchesHwc) {
    // CHW of the 2x2 RGB test image.
    const std::vector<float> chw{
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        1.0f,
        0.0f,
        0.0f,
        1.0f,
        0.0f,
    };
    const Tensor input = cpu_f32(chw, {3, 2, 2});
    EXPECT_EQ(detectImageLayout(input), ImageLayout::CHW);
    expect_bytes(prepareImageRgba8(input, false), kRgbHwcRgba);
}

TEST(ImageTensorPrepare, CpuAmbiguousCubeUsesChw) {
    std::vector<float> values(27);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < 3; ++y) {
            for (int x = 0; x < 3; ++x) {
                values[static_cast<size_t>((c * 3 + y) * 3 + x)] =
                    static_cast<float>(c + 1) * 0.25f;
            }
        }
    }
    const Tensor input = cpu_f32(values, {3, 3, 3});
    const Tensor out = prepareImageRgba8(input, false);
    ASSERT_TRUE(out.is_valid());
    const auto bytes = out.to_vector_uint8();
    ASSERT_EQ(bytes.size(), 3u * 3u * 4u);
    // CHW means channel 0 is R. First pixel RGB is nearest of (0.25, 0.50, 0.75)*255.
    EXPECT_EQ(bytes[0], 64);
    EXPECT_EQ(bytes[1], 128);
    EXPECT_EQ(bytes[2], 191);
    EXPECT_EQ(bytes[3], 255);
}

TEST(ImageTensorPrepare, CpuNoncontiguousChwView) {
    const Tensor hwc = cpu_f32(kRgbHwcF32, {2, 2, 3});
    const Tensor chw_view = hwc.permute({2, 0, 1});
    EXPECT_FALSE(chw_view.is_contiguous());
    EXPECT_EQ(detectImageLayout(chw_view), ImageLayout::CHW);
    expect_bytes(prepareImageRgba8(chw_view, false), kRgbHwcRgba);
}

TEST(ImageTensorPrepare, CpuSmallInputStaysOnHostWhenGpuIsActive) {
    const Tensor input = cpu_u8(kGrayHwcU8, {2, 2, 1});
    for (const GpuBackend backend : gpu_backends()) {
        SCOPED_TRACE(backend_name(backend));
        GpuBackendScope scope(backend);
        const Tensor out = prepareImageRgba8(input, false);
        EXPECT_EQ(out.device(), Device::CPU);
        EXPECT_FALSE(gpu_backend_of(out).has_value());
        expect_bytes(out, kGrayHwcRgba);
    }
}

TEST(ImageTensorPrepare, InvalidInputsReturnEmpty) {
    EXPECT_FALSE(prepareImageRgba8(Tensor{}, false).is_valid());
    EXPECT_FALSE(prepareImageRgba8(cpu_f32({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}), false).is_valid());
    EXPECT_FALSE(prepareImageRgba8(cpu_u8(std::vector<std::uint8_t>(8, 1), {2, 2, 2}), false)
                     .is_valid());
    EXPECT_FALSE(prepareImageRgba8(cpu_u8(std::vector<std::uint8_t>(20, 1), {2, 2, 5}), false)
                     .is_valid());
}

TEST(ImageTensorPrepare, GpuBackendsMatchCpuReference) {
    const auto backends = gpu_backends();
    if (backends.empty()) {
        GTEST_SKIP() << "no GPU tensor backend";
    }
    const Tensor cpu_rgb = cpu_f32(kRgbHwcF32, {2, 2, 3});
    const Tensor cpu_gray = cpu_u8(kGrayHwcU8, {2, 2, 1});
    const Tensor cpu_rgba = cpu_u8(kRgbaHwcU8, {2, 2, 4});
    const Tensor cpu_chw = cpu_f32(
        {1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f}, {3, 2, 2});
    for (const GpuBackend backend : backends) {
        SCOPED_TRACE(backend_name(backend));
        const Tensor rgb = to_backend(cpu_rgb, backend);
        const Tensor out_rgb = prepareImageRgba8(rgb, false);
        EXPECT_EQ(out_rgb.device(), Device::GPU);
        EXPECT_EQ(gpu_backend_of(out_rgb), backend);
        expect_bytes(out_rgb, kRgbHwcRgba);

        const Tensor flipped = prepareImageRgba8(rgb, true);
        EXPECT_EQ(gpu_backend_of(flipped), backend);
        expect_bytes(flipped, kRgbHwcRgbaFlipped);

        const Tensor gray = to_backend(cpu_gray, backend);
        expect_bytes(prepareImageRgba8(gray, false), kGrayHwcRgba);

        const Tensor rgba = to_backend(cpu_rgba, backend);
        expect_bytes(prepareImageRgba8(rgba, false), kRgbaHwcU8);

        const Tensor chw = to_backend(cpu_chw, backend);
        expect_bytes(prepareImageRgba8(chw, false), kRgbHwcRgba);

        const Tensor view = chw.permute({1, 2, 0});
        EXPECT_FALSE(view.is_contiguous());
        expect_bytes(prepareImageRgba8(view, false), kRgbHwcRgba);
    }
}

TEST(ImageTensorPrepare, OppositeActiveBackendKeepsSource) {
    const auto backends = gpu_backends();
    if (backends.size() < 2) {
        GTEST_SKIP() << "need both CUDA and Vulkan tensor backends";
    }
    const Tensor host = cpu_f32(kRgbHwcF32, {2, 2, 3});
    for (const GpuBackend source : backends) {
        const GpuBackend active = other_backend(source);
        if (!gpu_backend_available(active)) {
            continue;
        }
        SCOPED_TRACE(std::string(backend_name(source)) + "_under_" + backend_name(active));
        const Tensor gpu = to_backend(host, source);
        GpuBackendScope scope(active);
        const Tensor out = prepareImageRgba8(gpu, true);
        EXPECT_EQ(gpu_backend_of(out), source);
        expect_bytes(out, kRgbHwcRgbaFlipped);
    }
}

TEST(ImageLayoutFlip, OppositeActiveBackendKeepsSource) {
    const auto backends = gpu_backends();
    if (backends.size() < 2) {
        GTEST_SKIP() << "need both CUDA and Vulkan tensor backends";
    }
    const Tensor host = cpu_f32(kRgbHwcF32, {2, 2, 3});
    for (const GpuBackend source : backends) {
        const GpuBackend active = other_backend(source);
        if (!gpu_backend_available(active)) {
            continue;
        }
        SCOPED_TRACE(std::string(backend_name(source)) + "_under_" + backend_name(active));
        const Tensor gpu = to_backend(host, source);
        GpuBackendScope scope(active);
        const Tensor flipped = flipImageVertical(gpu, ImageLayout::HWC);
        EXPECT_EQ(gpu_backend_of(flipped), source);
        const Tensor prepared = prepareImageRgba8(flipped, false);
        expect_bytes(prepared, kRgbHwcRgbaFlipped);
    }
}

TEST(ImageTensorPrepare, CpuNanInfMatchFminFmaxToByte) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float pos_inf = std::numeric_limits<float>::infinity();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const Tensor input = cpu_f32(
        {nan, pos_inf, neg_inf, 0.5f, 1.0f, 0.0f, -0.25f, 1.25f, 0.499f, 0.501f, 0.0f, 1.0f},
        {2, 2, 3});
    const Tensor out = prepareImageRgba8(input, false);
    ASSERT_TRUE(out.is_valid());
    const auto bytes = out.to_vector_uint8();
    ASSERT_EQ(bytes.size(), 16u);
    // fminf(fmaxf(NaN, 0), 1) -> 0; +Inf -> 1; -Inf -> 0; 0.5 -> 128.
    EXPECT_EQ(bytes[0], 0);
    EXPECT_EQ(bytes[1], 255);
    EXPECT_EQ(bytes[2], 0);
    EXPECT_EQ(bytes[3], 255);
    EXPECT_EQ(bytes[4], 128);
    EXPECT_EQ(bytes[5], 255);
    EXPECT_EQ(bytes[6], 0);
    EXPECT_EQ(bytes[7], 255);
    EXPECT_EQ(bytes[8], 0);
    EXPECT_EQ(bytes[9], 255);
    EXPECT_EQ(bytes[10], 127);
    EXPECT_EQ(bytes[11], 255);
    EXPECT_EQ(bytes[12], 128);
    EXPECT_EQ(bytes[13], 0);
    EXPECT_EQ(bytes[14], 255);
    EXPECT_EQ(bytes[15], 255);
}

TEST(ImageTensorPrepare, GpuNearestMatchesCpuReference) {
    const auto backends = gpu_backends();
    if (backends.empty()) {
        GTEST_SKIP() << "no GPU tensor backend";
    }
    const Tensor cpu_input = cpu_f32(
        {-1.0f, 0.5f, 2.0f, 0.0f, 1.0f, 0.004f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f},
        {2, 2, 3});
    const Tensor cpu_out = prepareImageRgba8(cpu_input, false);
    const auto expected = cpu_out.to_vector_uint8();
    for (const GpuBackend backend : backends) {
        SCOPED_TRACE(backend_name(backend));
        const Tensor gpu = to_backend(cpu_input, backend);
        const Tensor out = prepareImageRgba8(gpu, false);
        EXPECT_EQ(gpu_backend_of(out), backend);
        expect_bytes(out, expected);
    }
}

TEST(ImageTensorPrepare, GpuNanInfMatchCpuReference) {
    const auto backends = gpu_backends();
    if (backends.empty()) {
        GTEST_SKIP() << "no GPU tensor backend";
    }
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float pos_inf = std::numeric_limits<float>::infinity();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const Tensor cpu_input = cpu_f32({nan, pos_inf, neg_inf, 0.5f}, {4, 1, 1});
    const Tensor cpu_out = prepareImageRgba8(cpu_input, false);
    const auto expected = cpu_out.to_vector_uint8();
    ASSERT_EQ(expected.size(), 4u);
    EXPECT_EQ(expected[0], 0);
    EXPECT_EQ(expected[1], 255);
    EXPECT_EQ(expected[2], 0);
    EXPECT_EQ(expected[3], 128);
    for (const GpuBackend backend : backends) {
        SCOPED_TRACE(backend_name(backend));
        const Tensor gpu = to_backend(cpu_input, backend);
        const Tensor out = prepareImageRgba8(gpu, false);
        EXPECT_EQ(gpu_backend_of(out), backend);
        expect_bytes(out, expected);
    }
}

// Native Vulkan / CUDA UI upload contract (enforced in vulkan_ui_texture.cpp):
//
// Direct CUDA (before prepareImageRgba8): CUDA-backend Float32 or UInt8 with 1/3/4
// channels, including gray. copyTensorToSurface uses toByte nearest. Float16 is not
// packed to RGBA8 before that attempt. Vulkan-backend tensors never take this path
// and never import through CUDA.
//
// Native Vulkan buffer->image: adopted same device, packed HWC UInt8 RGBA. Same
// queue-family copies on graphics with tracked fragment->transfer barriers. Cross-
// family exclusive tensor buffers copy on compute only after a graphics release
// semaphore; compute signals a binary semaphore that graphics waits on. Mixed
// timeline wait + binary signal provides a dummy 0 signal value so
// signalSemaphoreValueCount matches signalSemaphoreCount. Packed RGBA8 that aliases
// the caller is cloned so later mutation cannot race the GPU copy. Failed submits
// after compute work is accepted quarantine in-flight command buffers / semaphores /
// keep_alive (AMB-4) instead of freeing them. Other devices / families / failures
// fall back to host staging of the already-packed bytes.

TEST(ImageTensorPrepare, ChwByteFlipDoesNotModifySource) {
    const std::vector<std::uint8_t> pixels{
        255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0};
    const auto check = [&](const Tensor& hwc) {
        const auto before = hwc.to_vector_uint8();
        const auto chw = hwc.permute({2, 0, 1});
        const auto rgba = prepareImageRgba8(chw, true);
        expect_bytes(rgba, kRgbHwcRgbaFlipped);
        EXPECT_EQ(hwc.to_vector_uint8(), before);
    };
    const auto host = cpu_u8(pixels, {2, 2, 3});
    check(host);
    for (const auto backend : gpu_backends()) {
        SCOPED_TRACE(backend_name(backend));
        check(to_backend(host, backend));
    }
}

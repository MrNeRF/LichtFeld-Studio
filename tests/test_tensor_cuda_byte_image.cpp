/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <optional>
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

    Tensor cpu_u8(const std::vector<std::uint8_t>& bytes, const TensorShape& shape) {
        Tensor tensor = Tensor::empty(shape, Device::CPU, DataType::UInt8);
        if (!bytes.empty()) {
            std::memcpy(tensor.ptr<std::uint8_t>(), bytes.data(), bytes.size());
        }
        return tensor;
    }

    Tensor to_cuda(const Tensor& host) {
        return host.to(Device::GPU);
    }

    void expect_uint8_eq(const Tensor& actual, const std::vector<std::uint8_t>& expected) {
        ASSERT_TRUE(actual.is_valid());
        EXPECT_EQ(actual.dtype(), DataType::UInt8);
        const auto got = actual.to_vector_uint8();
        ASSERT_EQ(got.size(), expected.size());
        for (size_t i = 0; i < expected.size(); ++i) {
            EXPECT_EQ(got[i], expected[i]) << "index=" << i;
        }
    }

    // 2x2 HWC RGB / gray fixtures matching the image-prepare failure case.
    const std::vector<std::uint8_t> kRgbHwc{
        255,
        0,
        0,
        0,
        255,
        0,
        0,
        0,
        255,
        255,
        255,
        0,
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
    const std::vector<std::uint8_t> kGrayHwc{10, 20, 30, 40};
    const std::vector<std::uint8_t> kGrayHwcRgb{
        10,
        10,
        10,
        20,
        20,
        20,
        30,
        30,
        30,
        40,
        40,
        40,
    };
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
} // namespace

class TensorCudaByteImage : public ::testing::Test {
protected:
    void SetUp() override {
        if (!gpu_backend_available(GpuBackend::CUDA)) {
            GTEST_SKIP() << "CUDA tensor backend required";
            return;
        }
        cuda_scope_.emplace(GpuBackend::CUDA);
    }

    std::optional<GpuBackendScope> cuda_scope_;
};

TEST_F(TensorCudaByteImage, UInt8CatLastDimRgbPlusOpaqueAlpha) {
    const Tensor rgb = to_cuda(cpu_u8(kRgbHwc, {2, 2, 3}));
    const Tensor alpha = Tensor::full_like(rgb.slice(2, 0, 1), 255.0f);
    const Tensor out = Tensor::cat({rgb, alpha}, 2);

    EXPECT_EQ(out.device(), Device::GPU);
    EXPECT_EQ(gpu_backend_of(out), GpuBackend::CUDA);
    ASSERT_EQ(out.ndim(), 3u);
    EXPECT_EQ(out.size(0), 2u);
    EXPECT_EQ(out.size(1), 2u);
    EXPECT_EQ(out.size(2), 4u);
    expect_uint8_eq(out, kRgbHwcRgba);

    const auto bytes = out.to_vector_uint8();
    for (size_t pixel = 0; pixel < 4; ++pixel) {
        EXPECT_EQ(bytes[pixel * 4 + 3], 255) << "alpha pixel=" << pixel;
    }
}

TEST_F(TensorCudaByteImage, UInt8CatLastDimGrayReplicateThenAlpha) {
    const Tensor gray = to_cuda(cpu_u8(kGrayHwc, {2, 2, 1}));
    const Tensor rgb = Tensor::cat({gray, gray, gray}, 2);
    expect_uint8_eq(rgb, kGrayHwcRgb);

    const Tensor alpha = Tensor::full_like(gray, 255.0f);
    const Tensor out = Tensor::cat({rgb, alpha}, 2);
    expect_uint8_eq(out, kGrayHwcRgba);
}

TEST_F(TensorCudaByteImage, UInt8FullLikeWritesByte255) {
    const Tensor gray = to_cuda(cpu_u8(kGrayHwc, {2, 2, 1}));
    const Tensor alpha = Tensor::full_like(gray, 255.0f);
    EXPECT_EQ(alpha.dtype(), DataType::UInt8);
    expect_uint8_eq(alpha, {255, 255, 255, 255});
}

TEST_F(TensorCudaByteImage, UInt8IndexSelectFlipsHwcRows) {
    const Tensor rgb = to_cuda(cpu_u8(kRgbHwc, {2, 2, 3}));
    const Tensor indices = Tensor::from_vector(std::vector<int>{1, 0}, {2}, Device::GPU);
    const Tensor flipped = rgb.index_select(0, indices).contiguous();

    EXPECT_EQ(flipped.dtype(), DataType::UInt8);
    expect_uint8_eq(flipped, {0, 0, 255, 255, 255, 0, 255, 0, 0, 0, 255, 0});
}

TEST_F(TensorCudaByteImage, UInt8FlipThenCatAlpha) {
    const Tensor rgb = to_cuda(cpu_u8(kRgbHwc, {2, 2, 3}));
    const Tensor indices = Tensor::from_vector(std::vector<int>{1, 0}, {2}, Device::GPU);
    const Tensor flipped = rgb.index_select(0, indices).contiguous();
    const Tensor alpha = Tensor::full_like(flipped.slice(2, 0, 1), 255.0f);
    const Tensor out = Tensor::cat({flipped, alpha}, 2);
    expect_uint8_eq(out, kRgbHwcRgbaFlipped);
}

TEST_F(TensorCudaByteImage, FloatToUint8ThenCatAlpha) {
    const Tensor rgb_f32 = Tensor::from_vector(
        std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f},
        {2, 2, 3}, Device::GPU);
    const Tensor rgb = (rgb_f32.clamp(0.0f, 1.0f) * 255.0f).to(DataType::UInt8);
    const Tensor alpha = Tensor::full_like(rgb.slice(2, 0, 1), 255.0f);
    const Tensor out = Tensor::cat({rgb, alpha}, 2);
    expect_uint8_eq(out, kRgbHwcRgba);
}

TEST_F(TensorCudaByteImage, UInt8CatMiddleDimInterleavesRows) {
    const Tensor a = to_cuda(cpu_u8({1, 2, 3, 4, 5, 6}, {2, 1, 3}));
    const Tensor b = to_cuda(cpu_u8({7, 8, 9, 10, 11, 12}, {2, 1, 3}));
    const Tensor out = Tensor::cat({a, b}, 1);
    ASSERT_EQ(out.size(0), 2u);
    ASSERT_EQ(out.size(1), 2u);
    ASSERT_EQ(out.size(2), 3u);
    expect_uint8_eq(out, {1, 2, 3, 7, 8, 9, 4, 5, 6, 10, 11, 12});
}

TEST_F(TensorCudaByteImage, Float32CatLastDimRgbPlusAlphaUnchanged) {
    const Tensor rgb = Tensor::from_vector(
        std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f},
        {2, 2, 3}, Device::GPU);
    const Tensor alpha = Tensor::full_like(rgb.slice(2, 0, 1), 1.0f);
    const Tensor out = Tensor::cat({rgb, alpha}, 2);
    ASSERT_EQ(out.dtype(), DataType::Float32);
    ASSERT_EQ(out.size(2), 4u);
    const auto values = out.to_vector();
    const std::vector<float> expected{
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
        1.0f,
        1.0f,
        1.0f,
        0.0f,
        1.0f,
    };
    ASSERT_EQ(values.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_FLOAT_EQ(values[i], expected[i]) << "index=" << i;
    }
}

TEST_F(TensorCudaByteImage, CatPreservesRawBitsAtEveryElementWidth) {
    for (const auto dtype : {DataType::Bool, DataType::UInt8, DataType::Float16,
                             DataType::Int32, DataType::UInt32, DataType::Float32,
                             DataType::Int64}) {
        for (const int dim : {1, 2}) {
            SCOPED_TRACE(static_cast<int>(dtype));
            SCOPED_TRACE(dim);
            Tensor host = Tensor::empty(dim == 1 ? TensorShape{2, 1, 3}
                                                 : TensorShape{2, 3, 1},
                                        Device::CPU, dtype);
            auto* raw = static_cast<uint8_t*>(host.data_ptr());
            for (size_t i = 0; i < host.bytes(); ++i)
                raw[i] = dtype == DataType::Bool ? i % 2 : (i * 73 + 19) % 256;
            const size_t group_bytes = dim == 1 ? host.bytes() / 2 : host.bytes() / 6;
            std::vector<uint8_t> expected;
            for (size_t start = 0; start < host.bytes(); start += group_bytes) {
                expected.insert(expected.end(), raw + start, raw + start + group_bytes);
                expected.insert(expected.end(), raw + start, raw + start + group_bytes);
            }
            const Tensor input = host.to(Device::GPU);
            const Tensor actual = Tensor::cat({input, input}, dim).cpu();
            ASSERT_EQ(actual.bytes(), expected.size());
            EXPECT_EQ(std::memcmp(actual.data_ptr(), expected.data(), expected.size()), 0);
        }
    }
}

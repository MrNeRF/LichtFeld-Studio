/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;

static cudaError_t sync_and_check() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        return err;
    return cudaGetLastError();
}

static void assert_cuda_ok(const char* step) {
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << step << ": " << cudaGetErrorString(err);
}

class ExpandedTensorOpsTest : public ::testing::Test {
protected:
    void SetUp() override { cudaGetLastError(); }
    void TearDown() override {
        cudaError_t err = sync_and_check();
        EXPECT_EQ(err, cudaSuccess) << "Residual: " << cudaGetErrorString(err);
    }
};

// ===== Stack dim=2 correctness (the root cause bug) =====

TEST_F(ExpandedTensorOpsTest, StackDim2_Correctness) {
    const size_t H = 4, W = 3;
    auto r = Tensor::full({H, W}, 0.25f, Device::CUDA);
    auto g = Tensor::full({H, W}, 0.50f, Device::CUDA);
    auto b = Tensor::full({H, W}, 0.75f, Device::CUDA);

    auto stacked = Tensor::stack({r, g, b}, 2);
    assert_cuda_ok("stack dim=2");

    ASSERT_EQ(stacked.size(0), static_cast<int>(H));
    ASSERT_EQ(stacked.size(1), static_cast<int>(W));
    ASSERT_EQ(stacked.size(2), 3);

    auto cpu = stacked.cpu();
    auto* p = cpu.ptr<float>();
    for (size_t row = 0; row < H; ++row) {
        for (size_t col = 0; col < W; ++col) {
            size_t base = (row * W + col) * 3;
            EXPECT_FLOAT_EQ(p[base + 0], 0.25f)
                << "R at [" << row << "," << col << "]";
            EXPECT_FLOAT_EQ(p[base + 1], 0.50f)
                << "G at [" << row << "," << col << "]";
            EXPECT_FLOAT_EQ(p[base + 2], 0.75f)
                << "B at [" << row << "," << col << "]";
        }
    }
}

TEST_F(ExpandedTensorOpsTest, StackDim2_256x256) {
    auto r = Tensor::full({256, 256}, 0.1f, Device::CUDA);
    auto g = Tensor::full({256, 256}, 0.5f, Device::CUDA);
    auto b = Tensor::full({256, 256}, 0.9f, Device::CUDA);

    auto stacked = Tensor::stack({r, g, b}, 2);
    assert_cuda_ok("stack dim=2 256x256");

    ASSERT_EQ(stacked.ndim(), 3u);
    ASSERT_EQ(stacked.size(2), 3);

    auto cpu = stacked.cpu();
    auto* p = cpu.ptr<float>();
    // Spot-check corners
    EXPECT_FLOAT_EQ(p[0], 0.1f);
    EXPECT_FLOAT_EQ(p[1], 0.5f);
    EXPECT_FLOAT_EQ(p[2], 0.9f);
    size_t last = (256 * 256 - 1) * 3;
    EXPECT_FLOAT_EQ(p[last + 0], 0.1f);
    EXPECT_FLOAT_EQ(p[last + 1], 0.5f);
    EXPECT_FLOAT_EQ(p[last + 2], 0.9f);
}

TEST_F(ExpandedTensorOpsTest, StackDim1_Correctness) {
    auto a = Tensor::full({3, 4}, 1.0f, Device::CUDA);
    auto b = Tensor::full({3, 4}, 2.0f, Device::CUDA);

    auto stacked = Tensor::stack({a, b}, 1);
    assert_cuda_ok("stack dim=1");

    ASSERT_EQ(stacked.size(0), 3);
    ASSERT_EQ(stacked.size(1), 2);
    ASSERT_EQ(stacked.size(2), 4);

    auto cpu = stacked.cpu();
    auto* p = cpu.ptr<float>();
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col < 4; ++col) {
            EXPECT_FLOAT_EQ(p[row * 2 * 4 + 0 * 4 + col], 1.0f);
            EXPECT_FLOAT_EQ(p[row * 2 * 4 + 1 * 4 + col], 2.0f);
        }
    }
}

// ===== Full pipeline: expanded tensor arithmetic + stack + interop scatter =====

TEST_F(ExpandedTensorOpsTest, PlasmaPatternFullPipeline) {
    const size_t H = 256, W = 256;

    auto y = Tensor::linspace(-1.0f, 1.0f, H, Device::CUDA)
                 .reshape({static_cast<int>(H), 1})
                 .expand({static_cast<int>(H), static_cast<int>(W)});
    assert_cuda_ok("create y grid");

    auto x = Tensor::linspace(-1.0f, 1.0f, W, Device::CUDA)
                 .reshape({1, static_cast<int>(W)})
                 .expand({static_cast<int>(H), static_cast<int>(W)});
    assert_cuda_ok("create x grid");

    float freq = 3.0f;
    float t = 0.05f;

    auto r = (x * freq + t).sin() * 0.5f + 0.5f;
    assert_cuda_ok("compute r");

    auto g = (y * freq + t * 1.3f).sin() * 0.5f + 0.5f;
    assert_cuda_ok("compute g");

    auto b = ((x + y) * freq + t * 0.7f).sin() * 0.5f + 0.5f;
    assert_cuda_ok("compute b");

    auto image = Tensor::stack({r, g, b}, 2);
    assert_cuda_ok("stack RGB");

    ASSERT_EQ(image.size(0), static_cast<int>(H));
    ASSERT_EQ(image.size(1), static_cast<int>(W));
    ASSERT_EQ(image.size(2), 3);

    // Interop scatter path
    auto rgba = Tensor::empty({H, W, 4}, Device::CUDA);
    rgba.slice(2, 0, 3).copy_from(image);
    assert_cuda_ok("scatter RGB into RGBA");

    rgba.slice(2, 3, 4).fill_(1.0f);
    assert_cuda_ok("fill alpha");

    auto uint8_img = (rgba.clamp(0.0f, 1.0f) * 255.0f).to(DataType::UInt8);
    assert_cuda_ok("convert to uint8");

    auto cpu = uint8_img.cpu();
    auto* p = cpu.ptr<uint8_t>();
    // All alpha channels should be 255
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(p[i * 4 + 3], 255) << "alpha at pixel " << i;
    }
}

// ===== Stack with various dim values =====

TEST_F(ExpandedTensorOpsTest, StackDim0) {
    auto a = Tensor::full({2, 3}, 1.0f, Device::CUDA);
    auto b = Tensor::full({2, 3}, 2.0f, Device::CUDA);
    auto c = Tensor::full({2, 3}, 3.0f, Device::CUDA);

    auto stacked = Tensor::stack({a, b, c}, 0);
    assert_cuda_ok("stack dim=0");

    ASSERT_EQ(stacked.size(0), 3);
    ASSERT_EQ(stacked.size(1), 2);
    ASSERT_EQ(stacked.size(2), 3);

    auto cpu = stacked.cpu();
    auto* p = cpu.ptr<float>();
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[6], 2.0f);
    EXPECT_FLOAT_EQ(p[12], 3.0f);
}

// ===== Scatter into RGBA (from strided fill test, re-validated with new stack) =====

TEST_F(ExpandedTensorOpsTest, ScatterToRGBA) {
    auto rgb = Tensor::full({256, 256, 3}, 0.5f, Device::CUDA);
    auto rgba = Tensor::zeros({256, 256, 4}, Device::CUDA);

    rgba.slice(2, 0, 3).copy_from(rgb);
    assert_cuda_ok("scatter RGB to RGBA");

    rgba.slice(2, 3, 4).fill_(1.0f);
    assert_cuda_ok("fill alpha");

    auto cpu = rgba.cpu();
    auto* p = cpu.ptr<float>();
    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(p[i * 4 + 0], 0.5f);
        EXPECT_FLOAT_EQ(p[i * 4 + 1], 0.5f);
        EXPECT_FLOAT_EQ(p[i * 4 + 2], 0.5f);
        EXPECT_FLOAT_EQ(p[i * 4 + 3], 1.0f);
    }
}

// Raw ptr() on a stride-0 expand view must materialize dense storage because
// callers hand ptr()+numel() to
// flat memcpy/kernels (e.g. SplatData init scaling upload) and read N*expand
// elements from a physically smaller buffer otherwise.
TEST(ExpandedTensorOps, RawPtrOnExpandViewMaterializesDense) {
    using namespace lfs::core;
    auto base = Tensor::from_vector(std::vector<float>{1.f, 2.f, 3.f},
                                    TensorShape({3, 1}), Device::CPU);
    auto expanded = base.expand(std::vector<int>{3, 4});
    ASSERT_EQ(expanded.numel(), 12u);
    ASSERT_TRUE(expanded.has_zero_stride());
    const float* p = expanded.ptr<float>(); // must be safe for flat reads
    ASSERT_NE(p, nullptr);
    EXPECT_FALSE(expanded.has_zero_stride());
    EXPECT_TRUE(expanded.is_contiguous());
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 4; ++c)
            EXPECT_FLOAT_EQ(p[r * 4 + c], static_cast<float>(r + 1))
                << "flat read r=" << r << " c=" << c;
}

// Same escape contract for data_ptr() + bytes() (serialization / memcpy paths).
TEST(ExpandedTensorOps, DataPtrOnExpandViewMaterializesDense) {
    using namespace lfs::core;
    auto base = Tensor::from_vector(std::vector<float>{5.f, 6.f},
                                    TensorShape({2, 1}), Device::CPU);
    auto expanded = base.expand(std::vector<int>{2, 3});
    ASSERT_TRUE(expanded.has_zero_stride());
    ASSERT_EQ(expanded.bytes(), 6u * sizeof(float));
    const auto* p = static_cast<const float*>(expanded.data_ptr());
    ASSERT_NE(p, nullptr);
    EXPECT_FALSE(expanded.has_zero_stride());
    EXPECT_FLOAT_EQ(p[0], 5.f);
    EXPECT_FLOAT_EQ(p[1], 5.f);
    EXPECT_FLOAT_EQ(p[2], 5.f);
    EXPECT_FLOAT_EQ(p[3], 6.f);
    EXPECT_FLOAT_EQ(p[4], 6.f);
    EXPECT_FLOAT_EQ(p[5], 6.f);
}

// Mirror SplatData scaling init: knn distances → clamp_min/sqrt/mul/log →
// unsqueeze → expand → raw ptr() flat upload. Must be safe for flat reads.
TEST(ExpandedTensorOps, SplatDataScalingExpandPtrMaterializes) {
    using namespace lfs::core;
    constexpr size_t N = 128;
    auto nn_dist = Tensor::full({N}, 0.01f, Device::CPU).clamp_min(1e-7f);
    auto scaling = nn_dist.sqrt()
                       .mul(0.1f)
                       .log()
                       .unsqueeze(-1)
                       .expand(std::vector<int>{static_cast<int>(N), 3});
    ASSERT_EQ(scaling.numel(), N * 3);
    const float* p = scaling.ptr<float>();
    ASSERT_NE(p, nullptr);
    EXPECT_FALSE(scaling.has_zero_stride());
    EXPECT_TRUE(scaling.is_contiguous());
    EXPECT_FLOAT_EQ(p[0], p[1]);
    EXPECT_FLOAT_EQ(p[1], p[2]);
}

// SH degree 0 creates shN shape [N,0,3]: contiguous empty, row-major leading
// stride is 0 (product of a zero-size dim). Must NOT be treated as expand view
// and must allow ptr() escape (numel==0 flat upload is a no-op).
TEST(ExpandedTensorOps, EmptyZeroDimTensorPtrIsSafe) {
    using namespace lfs::core;
    auto shN = Tensor::zeros({54275, 0, 3}, Device::CPU);
    ASSERT_EQ(shN.numel(), 0u);
    EXPECT_TRUE(shN.is_contiguous());
    // Not an expand view — size>1 dim with stride 0 only from empty product.
    EXPECT_FALSE(shN.has_zero_stride());
    EXPECT_NO_THROW({
        (void)shN.ptr<float>();
        (void)shN.data_ptr();
    });
}

// A CPU-tagged handle whose storage is CUDA device
// memory must be rejected at raw-pointer escape (would produce cudaMemcpy
// invalid argument with src/dst in the same address region).
TEST(ExpandedTensorOps, CpuTaggedDeviceStorageRejectedOnPtr) {
    using namespace lfs::core;
    auto device_src = Tensor::from_vector(std::vector<float>{1.f, 2.f, 3.f, 4.f},
                                          TensorShape({4}), Device::CUDA);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // Non-owning blob: deliberately lie about device.
    auto mismatched = Tensor::from_blob(device_src.data_ptr(),
                                        TensorShape({4}),
                                        Device::CPU,
                                        DataType::Float32);
    ASSERT_TRUE(mismatched.is_valid());
    EXPECT_EQ(mismatched.device(), Device::CPU);

    EXPECT_THROW(
        {
            (void)mismatched.ptr<float>();
        },
        TensorError);
}

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

using namespace lfs::core;

// Helper: sync GPU and return any pending CUDA error
static cudaError_t sync_and_check() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        return err;
    return cudaGetLastError();
}

class StridedFillTest : public ::testing::Test {
protected:
    void SetUp() override {
        cudaGetLastError(); // clear stale errors
    }

    void TearDown() override {
        cudaError_t err = sync_and_check();
        EXPECT_EQ(err, cudaSuccess)
            << "Residual CUDA error: " << cudaGetErrorString(err);
    }
};

// ===== DynamicTexture exact repro: [H,W,4].slice(2,3,4).fill_(1.0f) =====

TEST_F(StridedFillTest, Slice3D_AlphaChannel_256x256) {
    auto t = Tensor::zeros({256, 256, 4}, Device::CUDA);
    auto alpha = t.slice(2, 3, 4); // [256, 256, 1]
    ASSERT_EQ(alpha.ndim(), 3u);
    ASSERT_EQ(alpha.size(0), 256u);
    ASSERT_EQ(alpha.size(1), 256u);
    ASSERT_EQ(alpha.size(2), 1u);
    ASSERT_FALSE(alpha.is_contiguous());

    alpha.fill_(1.0f);
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << "fill_ on 3D slice failed: " << cudaGetErrorString(err);

    auto cpu = t.cpu();
    auto* p = cpu.ptr<float>();
    // Every 4th element (alpha channel) should be 1.0, others 0.0
    for (int i = 0; i < 256 * 256; ++i) {
        ASSERT_FLOAT_EQ(p[i * 4 + 0], 0.0f) << "R should be 0 at pixel " << i;
        ASSERT_FLOAT_EQ(p[i * 4 + 1], 0.0f) << "G should be 0 at pixel " << i;
        ASSERT_FLOAT_EQ(p[i * 4 + 2], 0.0f) << "B should be 0 at pixel " << i;
        ASSERT_FLOAT_EQ(p[i * 4 + 3], 1.0f) << "A should be 1 at pixel " << i;
    }
}

TEST_F(StridedFillTest, Slice3D_AlphaChannel_512x512) {
    auto t = Tensor::zeros({512, 512, 4}, Device::CUDA);
    t.slice(2, 3, 4).fill_(1.0f);
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto cpu = t.cpu();
    auto* p = cpu.ptr<float>();
    for (int i = 0; i < 512 * 512; ++i) {
        ASSERT_FLOAT_EQ(p[i * 4 + 3], 1.0f) << "alpha mismatch at " << i;
    }
}

// ===== RGB channel slice fills =====

TEST_F(StridedFillTest, Slice3D_RGBChannels) {
    auto t = Tensor::zeros({64, 64, 4}, Device::CUDA);

    t.slice(2, 0, 1).fill_(0.25f); // R
    t.slice(2, 1, 2).fill_(0.50f); // G
    t.slice(2, 2, 3).fill_(0.75f); // B
    t.slice(2, 3, 4).fill_(1.00f); // A

    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto cpu = t.cpu();
    auto* p = cpu.ptr<float>();
    for (int i = 0; i < 64 * 64; ++i) {
        EXPECT_FLOAT_EQ(p[i * 4 + 0], 0.25f);
        EXPECT_FLOAT_EQ(p[i * 4 + 1], 0.50f);
        EXPECT_FLOAT_EQ(p[i * 4 + 2], 0.75f);
        EXPECT_FLOAT_EQ(p[i * 4 + 3], 1.00f);
    }
}

// ===== 3D channel slice with copy_ (the other interop path) =====

TEST_F(StridedFillTest, Slice3D_CopyToRGBChannels) {
    auto rgba = Tensor::zeros({128, 128, 4}, Device::CUDA);
    auto rgb = Tensor::full({128, 128, 3}, 0.5f, Device::CUDA);

    rgba.slice(2, 0, 3).copy_(rgb);
    rgba.slice(2, 3, 4).fill_(1.0f);

    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto cpu = rgba.cpu();
    auto* p = cpu.ptr<float>();
    for (int i = 0; i < 128 * 128; ++i) {
        EXPECT_FLOAT_EQ(p[i * 4 + 0], 0.5f);
        EXPECT_FLOAT_EQ(p[i * 4 + 1], 0.5f);
        EXPECT_FLOAT_EQ(p[i * 4 + 2], 0.5f);
        EXPECT_FLOAT_EQ(p[i * 4 + 3], 1.0f);
    }
}

// ===== 2D strided fills (column slices) =====

TEST_F(StridedFillTest, Slice2D_ColumnFill) {
    auto t = Tensor::zeros({100, 4}, Device::CUDA);
    t.slice(1, 3, 4).fill_(7.0f); // Last column
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto cpu = t.cpu();
    auto* p = cpu.ptr<float>();
    for (int r = 0; r < 100; ++r) {
        EXPECT_FLOAT_EQ(p[r * 4 + 0], 0.0f);
        EXPECT_FLOAT_EQ(p[r * 4 + 1], 0.0f);
        EXPECT_FLOAT_EQ(p[r * 4 + 2], 0.0f);
        EXPECT_FLOAT_EQ(p[r * 4 + 3], 7.0f);
    }
}

// ===== 1D strided fills (row from a 2D tensor) =====

TEST_F(StridedFillTest, Slice1D_RowFill) {
    auto t = Tensor::zeros({4, 256}, Device::CUDA);
    t.slice(0, 2, 3).fill_(3.0f); // 3rd row -> [1, 256] which squeezes to [256]
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto cpu = t.cpu();
    auto* p = cpu.ptr<float>();
    for (int c = 0; c < 256; ++c) {
        EXPECT_FLOAT_EQ(p[0 * 256 + c], 0.0f);
        EXPECT_FLOAT_EQ(p[1 * 256 + c], 0.0f);
        EXPECT_FLOAT_EQ(p[2 * 256 + c], 3.0f);
        EXPECT_FLOAT_EQ(p[3 * 256 + c], 0.0f);
    }
}

// ===== 4D strided fills =====

TEST_F(StridedFillTest, Slice4D_BatchChannelFill) {
    auto t = Tensor::zeros({2, 3, 8, 8}, Device::CUDA);
    t.slice(1, 2, 3).fill_(5.0f); // channel 2 for all batches
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto cpu = t.cpu();
    auto* p = cpu.ptr<float>();
    for (int b = 0; b < 2; ++b) {
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < 8; ++h) {
                for (int w = 0; w < 8; ++w) {
                    size_t idx = b * 3 * 8 * 8 + c * 8 * 8 + h * 8 + w;
                    float expected = (c == 2) ? 5.0f : 0.0f;
                    EXPECT_FLOAT_EQ(p[idx], expected)
                        << "at [" << b << "," << c << "," << h << "," << w << "]";
                }
            }
        }
    }
}

// ===== Edge cases =====

TEST_F(StridedFillTest, Fill_SingleElement3D) {
    auto t = Tensor::zeros({1, 1, 4}, Device::CUDA);
    t.slice(2, 3, 4).fill_(1.0f);
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto cpu = t.cpu();
    auto* p = cpu.ptr<float>();
    EXPECT_FLOAT_EQ(p[0], 0.0f);
    EXPECT_FLOAT_EQ(p[1], 0.0f);
    EXPECT_FLOAT_EQ(p[2], 0.0f);
    EXPECT_FLOAT_EQ(p[3], 1.0f);
}

TEST_F(StridedFillTest, Fill_LargeTensor3D) {
    auto t = Tensor::zeros({1024, 1024, 4}, Device::CUDA);
    t.slice(2, 3, 4).fill_(1.0f);
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    // Spot-check a few pixels
    auto cpu = t.cpu();
    auto* p = cpu.ptr<float>();
    for (int i : {0, 1000, 500000, 1024 * 1024 - 1}) {
        EXPECT_FLOAT_EQ(p[i * 4 + 3], 1.0f) << "alpha mismatch at pixel " << i;
    }
}

// ===== Expanded tensor fill (from DynamicTexture coordinate grid pattern) =====

TEST_F(StridedFillTest, ExpandedView_Fill) {
    auto col = Tensor::full({256, 1}, 0.0f, Device::CUDA);
    auto expanded = col.expand({256, 256});
    // expanded has stride 0 on dim 1 — fill should respect this
    // Note: filling an expanded tensor writes to the same memory multiple times
    // This should NOT crash, even if semantically odd
    expanded.fill_(2.0f);
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
}

// ===== Repeated fill (DynamicTexture updates every frame) =====

TEST_F(StridedFillTest, RepeatedFill_NoErrorAccumulation) {
    auto t = Tensor::zeros({256, 256, 4}, Device::CUDA);
    for (int frame = 0; frame < 100; ++frame) {
        t.slice(2, 0, 3).fill_(0.5f);
        t.slice(2, 3, 4).fill_(1.0f);
    }
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto cpu = t.cpu();
    auto* p = cpu.ptr<float>();
    EXPECT_FLOAT_EQ(p[0], 0.5f);
    EXPECT_FLOAT_EQ(p[1], 0.5f);
    EXPECT_FLOAT_EQ(p[2], 0.5f);
    EXPECT_FLOAT_EQ(p[3], 1.0f);
}

// ===== Int32 strided fill =====

TEST_F(StridedFillTest, Int32_Slice3D_Fill) {
    auto t = Tensor::zeros({64, 64, 4}, Device::CUDA, DataType::Int32);
    t.slice(2, 2, 3).fill_(42.0f);
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto cpu = t.cpu();
    auto* p = cpu.ptr<int>();
    for (int i = 0; i < 64 * 64; ++i) {
        EXPECT_EQ(p[i * 4 + 2], 42);
    }
}

// ===== Bool strided fill =====

TEST_F(StridedFillTest, Bool_Slice3D_Fill) {
    auto t = Tensor::zeros({32, 32, 4}, Device::CUDA, DataType::Bool);
    t.slice(2, 1, 2).fill_(1.0f);
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto cpu = t.cpu();
    auto* p = cpu.ptr<unsigned char>();
    for (int i = 0; i < 32 * 32; ++i) {
        EXPECT_EQ(p[i * 4 + 0], 0);
        EXPECT_EQ(p[i * 4 + 1], 1);
        EXPECT_EQ(p[i * 4 + 2], 0);
        EXPECT_EQ(p[i * 4 + 3], 0);
    }
}

// ===== Non-contiguous 2D general (not column slice) =====

TEST_F(StridedFillTest, Slice2D_GeneralNonContiguous) {
    auto t = Tensor::zeros({100, 100}, Device::CUDA);
    auto sub = t.slice(0, 10, 50).slice(1, 10, 50); // [40, 40] non-contiguous
    ASSERT_FALSE(sub.is_contiguous());
    sub.fill_(9.0f);
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto cpu = t.cpu();
    auto* p = cpu.ptr<float>();
    for (int r = 0; r < 100; ++r) {
        for (int c = 0; c < 100; ++c) {
            float expected = (r >= 10 && r < 50 && c >= 10 && c < 50) ? 9.0f : 0.0f;
            EXPECT_FLOAT_EQ(p[r * 100 + c], expected) << "at [" << r << "," << c << "]";
        }
    }
}

// ===== Contiguous fill (should NOT hit strided path) =====

TEST_F(StridedFillTest, Contiguous3D_Fill) {
    auto t = Tensor::empty({256, 256, 3}, Device::CUDA);
    ASSERT_TRUE(t.is_contiguous());
    t.fill_(0.42f);
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto cpu = t.cpu();
    auto* p = cpu.ptr<float>();
    EXPECT_FLOAT_EQ(p[0], 0.42f);
    EXPECT_FLOAT_EQ(p[256 * 256 * 3 - 1], 0.42f);
}

// ===== Strided fill with larger-than-65535-blocks =====

TEST_F(StridedFillTest, LargeGrid_3D_Fill) {
    // Forces 2D grid path: 2048*2048*1 = 4M elements / 256 = 16384 blocks
    auto t = Tensor::zeros({2048, 2048, 4}, Device::CUDA);
    t.slice(2, 0, 1).fill_(1.0f);
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto cpu = t.cpu();
    auto* p = cpu.ptr<float>();
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 0.0f);
}

TEST_F(StridedFillTest, VeryLargeGrid_3D_Fill) {
    // Forces 2D grid path: 4096*4096*1 = 16M elements / 256 = 65536 blocks (exactly at boundary)
    auto t = Tensor::zeros({4096, 4096, 4}, Device::CUDA);
    t.slice(2, 3, 4).fill_(1.0f);
    cudaError_t err = sync_and_check();
    ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

    auto cpu = t.cpu();
    auto* p = cpu.ptr<float>();
    for (int i : {0, 4096 * 2048, 4096 * 4096 - 1}) {
        EXPECT_FLOAT_EQ(p[i * 4 + 3], 1.0f) << "alpha at pixel " << i;
    }
}

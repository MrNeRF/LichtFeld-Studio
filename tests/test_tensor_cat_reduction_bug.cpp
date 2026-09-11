/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Comprehensive tests to expose ALL bugs in tensor operations:
 * - cat() + reduction (min/max/sum)
 * - slice() + reduction
 * - storage_offset handling
 * - contiguous() on sliced tensors
 * - Different tensor sizes (small, medium, large)
 * - Different dimensions (1D, 2D, 3D)
 * - Different slice patterns (row, column, range)
 */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <numeric>
#include <random>
#include <torch/torch.h>
#include <vector>

using namespace lfs::core;

class TensorCatReductionBugTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available for testing";
        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }

    // Helper to compare our tensor min/max with CPU ground truth
    void verifyMinMax(const Tensor& t, const std::string& name) {
        auto cpu = t.to(Device::CPU);
        const float* data = cpu.ptr<float>();
        size_t n = cpu.numel();

        float cpu_min = data[0], cpu_max = data[0];
        for (size_t i = 1; i < n; ++i) {
            cpu_min = std::min(cpu_min, data[i]);
            cpu_max = std::max(cpu_max, data[i]);
        }

        float gpu_min = t.min().item();
        float gpu_max = t.max().item();

        EXPECT_NEAR(gpu_min, cpu_min, 0.01f) << name << ": GPU min != CPU min";
        EXPECT_NEAR(gpu_max, cpu_max, 0.01f) << name << ": GPU max != CPU max";
    }

    // Helper to compare with torch
    void verifyVsTorch(const Tensor& our, const torch::Tensor& torch_t, const std::string& name) {
        float our_min = our.min().item();
        float our_max = our.max().item();
        float torch_min = torch_t.min().item<float>();
        float torch_max = torch_t.max().item<float>();

        EXPECT_NEAR(our_min, torch_min, 0.01f) << name << ": Our min != Torch min";
        EXPECT_NEAR(our_max, torch_max, 0.01f) << name << ": Our max != Torch max";
    }
};

// ============================================================================
// PART 1: Basic cat + reduction tests at different scales
// ============================================================================

TEST_F(TensorCatReductionBugTest, Cat_1D_Small) {
    std::vector<float> d1 = {100.0f, 200.0f, 300.0f};
    std::vector<float> d2 = {1.0f, 2.0f, 3.0f};

    auto t1 = Tensor::from_vector(d1, {3}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {3}, Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    verifyMinMax(cat, "Cat_1D_Small");
    EXPECT_FLOAT_EQ(cat.min().item(), 1.0f);
    EXPECT_FLOAT_EQ(cat.max().item(), 300.0f);
}

TEST_F(TensorCatReductionBugTest, Cat_1D_Medium_1000) {
    const size_t N = 1000;
    std::vector<float> d1(N), d2(N);
    for (size_t i = 0; i < N; ++i) {
        d1[i] = 100.0f + static_cast<float>(i);  // [100, 1099]
        d2[i] = -500.0f + static_cast<float>(i); // [-500, 499]
    }

    auto t1 = Tensor::from_vector(d1, {N}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {N}, Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    verifyMinMax(cat, "Cat_1D_Medium_1000");
    EXPECT_FLOAT_EQ(cat.min().item(), -500.0f);
    EXPECT_FLOAT_EQ(cat.max().item(), 1099.0f);
}

TEST_F(TensorCatReductionBugTest, Cat_1D_Large_100K) {
    const size_t N = 100000;
    std::vector<float> d1(N), d2(N);
    for (size_t i = 0; i < N; ++i) {
        d1[i] = 1000.0f + static_cast<float>(i % 1000);  // [1000, 1999]
        d2[i] = -1000.0f + static_cast<float>(i % 1000); // [-1000, -1]
    }

    auto t1 = Tensor::from_vector(d1, {N}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {N}, Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    verifyMinMax(cat, "Cat_1D_Large_100K");
    EXPECT_FLOAT_EQ(cat.min().item(), -1000.0f);
    EXPECT_FLOAT_EQ(cat.max().item(), 1999.0f);
}

TEST_F(TensorCatReductionBugTest, Cat_1D_VeryLarge_1M) {
    const size_t N = 1000000;
    auto cpu1 = Tensor::zeros({N}, Device::CPU);
    auto cpu2 = Tensor::zeros({N}, Device::CPU);
    float* p1 = cpu1.ptr<float>();
    float* p2 = cpu2.ptr<float>();
    for (size_t i = 0; i < N; ++i) {
        p1[i] = 50.0f + static_cast<float>(i % 50);   // [50, 99]
        p2[i] = -100.0f + static_cast<float>(i % 50); // [-100, -51]
    }

    auto t1 = cpu1.to(Device::CUDA);
    auto t2 = cpu2.to(Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    verifyMinMax(cat, "Cat_1D_VeryLarge_1M");
    EXPECT_FLOAT_EQ(cat.min().item(), -100.0f);
    EXPECT_FLOAT_EQ(cat.max().item(), 99.0f);
}

TEST_F(TensorCatReductionBugTest, Cat_1D_Huge_2_6M) {
    // Exact size from the failing merge: 2 x 1,314,907 gaussians
    const size_t N = 1314907;
    auto cpu1 = Tensor::zeros({N}, Device::CPU);
    auto cpu2 = Tensor::zeros({N}, Device::CPU);
    float* p1 = cpu1.ptr<float>();
    float* p2 = cpu2.ptr<float>();
    for (size_t i = 0; i < N; ++i) {
        p1[i] = 31.0f + static_cast<float>(i % 36);  // [31, 66]
        p2[i] = -14.0f + static_cast<float>(i % 36); // [-14, 21]
    }

    auto t1 = cpu1.to(Device::CUDA);
    auto t2 = cpu2.to(Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    verifyMinMax(cat, "Cat_1D_Huge_2_6M");
    EXPECT_FLOAT_EQ(cat.min().item(), -14.0f);
    EXPECT_FLOAT_EQ(cat.max().item(), 66.0f);
}

// ============================================================================
// PART 2: cat + slice + reduction (the exact failing pattern)
// ============================================================================

TEST_F(TensorCatReductionBugTest, Cat_2D_SliceColumn_Small) {
    std::vector<float> d1 = {100.0f, 1.0f, 200.0f, 2.0f, 300.0f, 3.0f}; // [3, 2]
    std::vector<float> d2 = {-50.0f, 4.0f, -60.0f, 5.0f};               // [2, 2]

    auto t1 = Tensor::from_vector(d1, {3, 2}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {2, 2}, Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0); // [5, 2]

    ASSERT_EQ(cat.size(0), 5);
    ASSERT_EQ(cat.size(1), 2);

    // Slice column 0
    auto col0 = cat.slice(1, 0, 1).squeeze(1);
    verifyMinMax(col0, "Cat_2D_SliceColumn_Small col0");
    EXPECT_FLOAT_EQ(col0.min().item(), -60.0f);
    EXPECT_FLOAT_EQ(col0.max().item(), 300.0f);

    // Slice column 1
    auto col1 = cat.slice(1, 1, 2).squeeze(1);
    verifyMinMax(col1, "Cat_2D_SliceColumn_Small col1");
    EXPECT_FLOAT_EQ(col1.min().item(), 1.0f);
    EXPECT_FLOAT_EQ(col1.max().item(), 5.0f);
}

TEST_F(TensorCatReductionBugTest, Cat_2D_SliceColumn_Medium) {
    const size_t N1 = 1000, N2 = 500;
    auto cpu1 = Tensor::zeros({N1, 3}, Device::CPU);
    auto cpu2 = Tensor::zeros({N2, 3}, Device::CPU);
    float* p1 = cpu1.ptr<float>();
    float* p2 = cpu2.ptr<float>();

    for (size_t i = 0; i < N1; ++i) {
        p1[i * 3 + 0] = 100.0f + static_cast<float>(i % 100); // x: [100, 199]
        p1[i * 3 + 1] = 0.0f;
        p1[i * 3 + 2] = 0.0f;
    }
    for (size_t i = 0; i < N2; ++i) {
        p2[i * 3 + 0] = -200.0f + static_cast<float>(i % 100); // x: [-200, -101]
        p2[i * 3 + 1] = 0.0f;
        p2[i * 3 + 2] = 0.0f;
    }

    auto t1 = cpu1.to(Device::CUDA);
    auto t2 = cpu2.to(Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    auto col0 = cat.slice(1, 0, 1).squeeze(1);
    verifyMinMax(col0, "Cat_2D_SliceColumn_Medium");
    EXPECT_FLOAT_EQ(col0.min().item(), -200.0f);
    EXPECT_FLOAT_EQ(col0.max().item(), 199.0f);
}

TEST_F(TensorCatReductionBugTest, Cat_2D_SliceColumn_Large_Exact) {
    // EXACT reproduction of the failing merge scenario
    const size_t N = 1314907;
    auto cpu1 = Tensor::zeros({N, 3}, Device::CPU);
    auto cpu2 = Tensor::zeros({N, 3}, Device::CPU);
    float* p1 = cpu1.ptr<float>();
    float* p2 = cpu2.ptr<float>();

    // Splat 1: translated, x in [31, 66]
    for (size_t i = 0; i < N; ++i) {
        p1[i * 3 + 0] = 31.0f + static_cast<float>(i % 36);
        p1[i * 3 + 1] = -35.0f + static_cast<float>(i % 37);
        p1[i * 3 + 2] = -16.0f + static_cast<float>(i % 28);
    }
    // Splat 2: at origin, x in [-14, 21]
    for (size_t i = 0; i < N; ++i) {
        p2[i * 3 + 0] = -14.0f + static_cast<float>(i % 36);
        p2[i * 3 + 1] = -35.0f + static_cast<float>(i % 37);
        p2[i * 3 + 2] = -16.0f + static_cast<float>(i % 28);
    }

    auto t1 = cpu1.to(Device::CUDA);
    auto t2 = cpu2.to(Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    // This is the EXACT pattern from compute_bounds
    for (int i = 0; i < 3; ++i) {
        auto col = cat.slice(1, i, i + 1).squeeze(1);
        verifyMinMax(col, "Cat_2D_SliceColumn_Large col" + std::to_string(i));
    }

    auto x_col = cat.slice(1, 0, 1).squeeze(1);
    EXPECT_NEAR(x_col.min().item(), -14.0f, 0.01f) << "X min should be -14";
    EXPECT_NEAR(x_col.max().item(), 66.0f, 0.01f) << "X max should be 66";
}

// ============================================================================
// PART 3: slice + reduction (without cat)
// ============================================================================

TEST_F(TensorCatReductionBugTest, Slice_1D_Range_Small) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 100.0f, 200.0f, 300.0f};
    auto t = Tensor::from_vector(data, {6}, Device::CUDA);

    auto slice1 = t.slice(0, 0, 3); // [1, 2, 3]
    auto slice2 = t.slice(0, 3, 6); // [100, 200, 300]

    verifyMinMax(slice1, "Slice_1D_Range first");
    verifyMinMax(slice2, "Slice_1D_Range second");

    EXPECT_FLOAT_EQ(slice1.min().item(), 1.0f);
    EXPECT_FLOAT_EQ(slice1.max().item(), 3.0f);
    EXPECT_FLOAT_EQ(slice2.min().item(), 100.0f);
    EXPECT_FLOAT_EQ(slice2.max().item(), 300.0f);
}

TEST_F(TensorCatReductionBugTest, Slice_1D_Range_Large) {
    const size_t N = 100000;
    auto cpu = Tensor::zeros({N}, Device::CPU);
    float* p = cpu.ptr<float>();
    for (size_t i = 0; i < N / 2; ++i) {
        p[i] = 1000.0f + static_cast<float>(i); // [1000, 50999]
    }
    for (size_t i = N / 2; i < N; ++i) {
        p[i] = -1000.0f - static_cast<float>(i - N / 2); // [-1000, -50999]
    }

    auto t = cpu.to(Device::CUDA);

    auto slice1 = t.slice(0, 0, N / 2);
    auto slice2 = t.slice(0, N / 2, N);

    verifyMinMax(slice1, "Slice_1D_Range_Large first");
    verifyMinMax(slice2, "Slice_1D_Range_Large second");

    EXPECT_FLOAT_EQ(slice1.min().item(), 1000.0f);
    EXPECT_FLOAT_EQ(slice1.max().item(), 50999.0f);
    EXPECT_FLOAT_EQ(slice2.min().item(), -50999.0f);
    EXPECT_FLOAT_EQ(slice2.max().item(), -1000.0f);
}

TEST_F(TensorCatReductionBugTest, Slice_2D_Column_Small) {
    std::vector<float> data = {
        1.0f, 10.0f, 100.0f,
        2.0f, 20.0f, 200.0f,
        3.0f, 30.0f, 300.0f,
        4.0f, 40.0f, 400.0f};
    auto t = Tensor::from_vector(data, {4, 3}, Device::CUDA);

    for (int c = 0; c < 3; ++c) {
        auto col = t.slice(1, c, c + 1).squeeze(1);
        verifyMinMax(col, "Slice_2D_Column_Small col" + std::to_string(c));
    }

    auto col0 = t.slice(1, 0, 1).squeeze(1);
    auto col1 = t.slice(1, 1, 2).squeeze(1);
    auto col2 = t.slice(1, 2, 3).squeeze(1);

    EXPECT_FLOAT_EQ(col0.min().item(), 1.0f);
    EXPECT_FLOAT_EQ(col0.max().item(), 4.0f);
    EXPECT_FLOAT_EQ(col1.min().item(), 10.0f);
    EXPECT_FLOAT_EQ(col1.max().item(), 40.0f);
    EXPECT_FLOAT_EQ(col2.min().item(), 100.0f);
    EXPECT_FLOAT_EQ(col2.max().item(), 400.0f);
}

TEST_F(TensorCatReductionBugTest, Slice_2D_Column_Large) {
    const size_t N = 100000;
    auto cpu = Tensor::zeros({N, 3}, Device::CPU);
    float* p = cpu.ptr<float>();
    for (size_t i = 0; i < N; ++i) {
        p[i * 3 + 0] = static_cast<float>(i);  // [0, N-1]
        p[i * 3 + 1] = -static_cast<float>(i); // [-(N-1), 0]
        p[i * 3 + 2] = 1000.0f;                // constant
    }

    auto t = cpu.to(Device::CUDA);

    auto col0 = t.slice(1, 0, 1).squeeze(1);
    auto col1 = t.slice(1, 1, 2).squeeze(1);
    auto col2 = t.slice(1, 2, 3).squeeze(1);

    verifyMinMax(col0, "Slice_2D_Column_Large col0");
    verifyMinMax(col1, "Slice_2D_Column_Large col1");
    verifyMinMax(col2, "Slice_2D_Column_Large col2");

    EXPECT_FLOAT_EQ(col0.min().item(), 0.0f);
    EXPECT_FLOAT_EQ(col0.max().item(), static_cast<float>(N - 1));
    EXPECT_FLOAT_EQ(col1.min().item(), -static_cast<float>(N - 1));
    EXPECT_FLOAT_EQ(col1.max().item(), 0.0f);
    EXPECT_FLOAT_EQ(col2.min().item(), 1000.0f);
    EXPECT_FLOAT_EQ(col2.max().item(), 1000.0f);
}

TEST_F(TensorCatReductionBugTest, Slice_2D_Row_Small) {
    std::vector<float> data = {
        1.0f, 2.0f, 3.0f,
        100.0f, 200.0f, 300.0f,
        10.0f, 20.0f, 30.0f};
    auto t = Tensor::from_vector(data, {3, 3}, Device::CUDA);

    auto row0 = t.slice(0, 0, 1).squeeze(0);
    auto row1 = t.slice(0, 1, 2).squeeze(0);
    auto row2 = t.slice(0, 2, 3).squeeze(0);

    verifyMinMax(row0, "Slice_2D_Row row0");
    verifyMinMax(row1, "Slice_2D_Row row1");
    verifyMinMax(row2, "Slice_2D_Row row2");

    EXPECT_FLOAT_EQ(row0.min().item(), 1.0f);
    EXPECT_FLOAT_EQ(row0.max().item(), 3.0f);
    EXPECT_FLOAT_EQ(row1.min().item(), 100.0f);
    EXPECT_FLOAT_EQ(row1.max().item(), 300.0f);
}

// ============================================================================
// PART 4: storage_offset edge cases
// ============================================================================

TEST_F(TensorCatReductionBugTest, StorageOffset_SliceMiddle) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 100.0f, 200.0f, 300.0f, 10.0f, 20.0f, 30.0f};
    auto t = Tensor::from_vector(data, {9}, Device::CUDA);

    // Slice middle portion
    auto middle = t.slice(0, 3, 6); // Should be [100, 200, 300]

    EXPECT_GT(middle.storage_offset(), 0) << "Middle slice should have non-zero storage_offset";

    verifyMinMax(middle, "StorageOffset_SliceMiddle");
    EXPECT_FLOAT_EQ(middle.min().item(), 100.0f);
    EXPECT_FLOAT_EQ(middle.max().item(), 300.0f);
}

TEST_F(TensorCatReductionBugTest, StorageOffset_SliceEnd) {
    std::vector<float> data = {1000.0f, 2000.0f, 3000.0f, 1.0f, 2.0f, 3.0f};
    auto t = Tensor::from_vector(data, {6}, Device::CUDA);

    auto end = t.slice(0, 3, 6); // Should be [1, 2, 3]

    verifyMinMax(end, "StorageOffset_SliceEnd");
    EXPECT_FLOAT_EQ(end.min().item(), 1.0f);
    EXPECT_FLOAT_EQ(end.max().item(), 3.0f);
}

TEST_F(TensorCatReductionBugTest, StorageOffset_CatThenSlice) {
    std::vector<float> d1 = {10.0f, 20.0f, 30.0f};
    std::vector<float> d2 = {1.0f, 2.0f, 3.0f};

    auto t1 = Tensor::from_vector(d1, {3}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {3}, Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    // Cat should be contiguous with offset 0
    EXPECT_TRUE(cat.is_contiguous());
    EXPECT_EQ(cat.storage_offset(), 0);

    // Slice the second half (should be [1, 2, 3])
    auto slice = cat.slice(0, 3, 6);

    verifyMinMax(slice, "StorageOffset_CatThenSlice");
    EXPECT_FLOAT_EQ(slice.min().item(), 1.0f);
    EXPECT_FLOAT_EQ(slice.max().item(), 3.0f);
}

TEST_F(TensorCatReductionBugTest, StorageOffset_2D_ColumnSlice) {
    // Column slices have stride issues, not just offset
    std::vector<float> data = {
        1.0f, 100.0f,
        2.0f, 200.0f,
        3.0f, 300.0f};
    auto t = Tensor::from_vector(data, {3, 2}, Device::CUDA);

    auto col0 = t.slice(1, 0, 1).squeeze(1); // [1, 2, 3]
    auto col1 = t.slice(1, 1, 2).squeeze(1); // [100, 200, 300]

    verifyMinMax(col0, "StorageOffset_2D col0");
    verifyMinMax(col1, "StorageOffset_2D col1");

    EXPECT_FLOAT_EQ(col0.min().item(), 1.0f);
    EXPECT_FLOAT_EQ(col0.max().item(), 3.0f);
    EXPECT_FLOAT_EQ(col1.min().item(), 100.0f);
    EXPECT_FLOAT_EQ(col1.max().item(), 300.0f);
}

// ============================================================================
// PART 5: contiguous() correctness
// ============================================================================

TEST_F(TensorCatReductionBugTest, Contiguous_ColumnSlice_Small) {
    std::vector<float> data = {
        1.0f, 10.0f, 100.0f,
        2.0f, 20.0f, 200.0f,
        3.0f, 30.0f, 300.0f};
    auto t = Tensor::from_vector(data, {3, 3}, Device::CUDA);

    auto col0 = t.slice(1, 0, 1).squeeze(1);

    // Make contiguous
    auto col0_contig = col0.contiguous();
    EXPECT_TRUE(col0_contig.is_contiguous());

    // Verify data after contiguous()
    auto cpu = col0_contig.to(Device::CPU);
    const float* p = cpu.ptr<float>();
    EXPECT_FLOAT_EQ(p[0], 1.0f);
    EXPECT_FLOAT_EQ(p[1], 2.0f);
    EXPECT_FLOAT_EQ(p[2], 3.0f);

    // Verify reduction on contiguous
    EXPECT_FLOAT_EQ(col0_contig.min().item(), 1.0f);
    EXPECT_FLOAT_EQ(col0_contig.max().item(), 3.0f);
}

TEST_F(TensorCatReductionBugTest, Contiguous_ColumnSlice_Large) {
    const size_t N = 100000;
    auto cpu = Tensor::zeros({N, 3}, Device::CPU);
    float* p = cpu.ptr<float>();
    for (size_t i = 0; i < N; ++i) {
        p[i * 3 + 0] = static_cast<float>(i);
        p[i * 3 + 1] = 0.0f;
        p[i * 3 + 2] = 0.0f;
    }

    auto t = cpu.to(Device::CUDA);
    auto col0 = t.slice(1, 0, 1).squeeze(1);
    auto col0_contig = col0.contiguous();

    // Verify contiguous data
    auto verify_cpu = col0_contig.to(Device::CPU);
    const float* verify_p = verify_cpu.ptr<float>();
    for (size_t i = 0; i < std::min(N, size_t(100)); ++i) {
        EXPECT_FLOAT_EQ(verify_p[i], static_cast<float>(i)) << "Mismatch at index " << i;
    }

    verifyMinMax(col0_contig, "Contiguous_ColumnSlice_Large");
    EXPECT_FLOAT_EQ(col0_contig.min().item(), 0.0f);
    EXPECT_FLOAT_EQ(col0_contig.max().item(), static_cast<float>(N - 1));
}

TEST_F(TensorCatReductionBugTest, Contiguous_AfterCatSlice) {
    const size_t N = 10000;
    auto cpu1 = Tensor::zeros({N, 3}, Device::CPU);
    auto cpu2 = Tensor::zeros({N, 3}, Device::CPU);
    float* p1 = cpu1.ptr<float>();
    float* p2 = cpu2.ptr<float>();

    for (size_t i = 0; i < N; ++i) {
        p1[i * 3] = 100.0f + static_cast<float>(i % 100);
        p2[i * 3] = -100.0f - static_cast<float>(i % 100);
    }

    auto t1 = cpu1.to(Device::CUDA);
    auto t2 = cpu2.to(Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    auto col0 = cat.slice(1, 0, 1).squeeze(1);
    auto col0_contig = col0.contiguous();

    verifyMinMax(col0_contig, "Contiguous_AfterCatSlice");
    EXPECT_NEAR(col0_contig.min().item(), -199.0f, 0.01f);
    EXPECT_NEAR(col0_contig.max().item(), 199.0f, 0.01f);
}

// ============================================================================
// PART 6: sum reduction (not just min/max)
// ============================================================================

TEST_F(TensorCatReductionBugTest, Sum_Cat_1D) {
    std::vector<float> d1 = {1.0f, 2.0f, 3.0f};    // sum = 6
    std::vector<float> d2 = {10.0f, 20.0f, 30.0f}; // sum = 60

    auto t1 = Tensor::from_vector(d1, {3}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {3}, Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    float sum = cat.sum_scalar();
    EXPECT_FLOAT_EQ(sum, 66.0f);
}

TEST_F(TensorCatReductionBugTest, Sum_Cat_2D_Column) {
    std::vector<float> d1 = {1.0f, 10.0f, 2.0f, 20.0f, 3.0f, 30.0f}; // [3, 2]
    std::vector<float> d2 = {4.0f, 40.0f, 5.0f, 50.0f};              // [2, 2]

    auto t1 = Tensor::from_vector(d1, {3, 2}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {2, 2}, Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    auto col0 = cat.slice(1, 0, 1).squeeze(1); // [1, 2, 3, 4, 5]
    auto col1 = cat.slice(1, 1, 2).squeeze(1); // [10, 20, 30, 40, 50]

    EXPECT_FLOAT_EQ(col0.sum_scalar(), 15.0f);
    EXPECT_FLOAT_EQ(col1.sum_scalar(), 150.0f);
}

TEST_F(TensorCatReductionBugTest, Sum_Slice_Large) {
    const size_t N = 100000;
    auto cpu = Tensor::zeros({N}, Device::CPU);
    float* p = cpu.ptr<float>();
    float expected_first = 0.0f, expected_second = 0.0f;
    for (size_t i = 0; i < N / 2; ++i) {
        p[i] = 1.0f;
        expected_first += 1.0f;
    }
    for (size_t i = N / 2; i < N; ++i) {
        p[i] = 2.0f;
        expected_second += 2.0f;
    }

    auto t = cpu.to(Device::CUDA);
    auto slice1 = t.slice(0, 0, N / 2);
    auto slice2 = t.slice(0, N / 2, N);

    EXPECT_FLOAT_EQ(slice1.sum_scalar(), expected_first);
    EXPECT_FLOAT_EQ(slice2.sum_scalar(), expected_second);
}

// ============================================================================
// PART 7: mean reduction
// ============================================================================

TEST_F(TensorCatReductionBugTest, Mean_Cat_1D) {
    std::vector<float> d1 = {2.0f, 4.0f, 6.0f};    // mean = 4
    std::vector<float> d2 = {10.0f, 20.0f, 30.0f}; // mean = 20, combined mean = 12

    auto t1 = Tensor::from_vector(d1, {3}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {3}, Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    float mean = cat.mean_scalar();
    EXPECT_FLOAT_EQ(mean, 12.0f); // (2+4+6+10+20+30)/6 = 72/6 = 12
}

TEST_F(TensorCatReductionBugTest, Mean_Slice_Column) {
    std::vector<float> data = {
        1.0f, 10.0f,
        3.0f, 30.0f,
        5.0f, 50.0f};
    auto t = Tensor::from_vector(data, {3, 2}, Device::CUDA);

    auto col0 = t.slice(1, 0, 1).squeeze(1);
    auto col1 = t.slice(1, 1, 2).squeeze(1);

    EXPECT_FLOAT_EQ(col0.mean_scalar(), 3.0f);  // (1+3+5)/3 = 3
    EXPECT_FLOAT_EQ(col1.mean_scalar(), 30.0f); // (10+30+50)/3 = 30
}

// ============================================================================
// PART 8: torch comparison (ground truth)
// ============================================================================

TEST_F(TensorCatReductionBugTest, VsTorch_Cat_1D) {
    std::vector<float> d1 = {100.0f, 200.0f, 300.0f};
    std::vector<float> d2 = {1.0f, 2.0f, 3.0f};

    auto our_t1 = Tensor::from_vector(d1, {3}, Device::CUDA);
    auto our_t2 = Tensor::from_vector(d2, {3}, Device::CUDA);
    auto our_cat = Tensor::cat({our_t1, our_t2}, 0);

    auto torch_t1 = torch::tensor(d1, torch::kCUDA);
    auto torch_t2 = torch::tensor(d2, torch::kCUDA);
    auto torch_cat = torch::cat({torch_t1, torch_t2}, 0);

    verifyVsTorch(our_cat, torch_cat, "VsTorch_Cat_1D");
}

TEST_F(TensorCatReductionBugTest, VsTorch_Cat_2D_SliceColumn) {
    std::vector<float> d1 = {31.0f, 1.0f, 66.0f, 2.0f};  // [2, 2]
    std::vector<float> d2 = {-14.0f, 3.0f, 21.0f, 4.0f}; // [2, 2]

    auto our_t1 = Tensor::from_vector(d1, {2, 2}, Device::CUDA);
    auto our_t2 = Tensor::from_vector(d2, {2, 2}, Device::CUDA);
    auto our_cat = Tensor::cat({our_t1, our_t2}, 0);
    auto our_col0 = our_cat.slice(1, 0, 1).squeeze(1);

    auto torch_t1 = torch::from_blob(d1.data(), {2, 2}, torch::kFloat).clone().to(torch::kCUDA);
    auto torch_t2 = torch::from_blob(d2.data(), {2, 2}, torch::kFloat).clone().to(torch::kCUDA);
    auto torch_cat = torch::cat({torch_t1, torch_t2}, 0);
    auto torch_col0 = torch_cat.slice(1, 0, 1).squeeze(1);

    verifyVsTorch(our_col0, torch_col0, "VsTorch_Cat_2D_SliceColumn");

    // Explicit expected values
    EXPECT_FLOAT_EQ(torch_col0.min().item<float>(), -14.0f);
    EXPECT_FLOAT_EQ(torch_col0.max().item<float>(), 66.0f);
}

TEST_F(TensorCatReductionBugTest, VsTorch_Slice_Range) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 100.0f, 200.0f, 300.0f};

    auto our_t = Tensor::from_vector(data, {6}, Device::CUDA);
    auto our_slice = our_t.slice(0, 3, 6);

    auto torch_t = torch::tensor(data, torch::kCUDA);
    auto torch_slice = torch_t.slice(0, 3, 6);

    verifyVsTorch(our_slice, torch_slice, "VsTorch_Slice_Range");
}

TEST_F(TensorCatReductionBugTest, VsTorch_Large_Cat_Column) {
    const size_t N = 100000;
    std::vector<float> d1(N * 3), d2(N * 3);

    for (size_t i = 0; i < N; ++i) {
        d1[i * 3 + 0] = 100.0f + static_cast<float>(i % 100);
        d1[i * 3 + 1] = 0.0f;
        d1[i * 3 + 2] = 0.0f;
        d2[i * 3 + 0] = -100.0f - static_cast<float>(i % 100);
        d2[i * 3 + 1] = 0.0f;
        d2[i * 3 + 2] = 0.0f;
    }

    auto our_t1 = Tensor::from_vector(d1, {N, 3}, Device::CUDA);
    auto our_t2 = Tensor::from_vector(d2, {N, 3}, Device::CUDA);
    auto our_cat = Tensor::cat({our_t1, our_t2}, 0);
    auto our_col0 = our_cat.slice(1, 0, 1).squeeze(1);

    auto torch_t1 = torch::from_blob(d1.data(), {static_cast<long>(N), 3}, torch::kFloat).clone().to(torch::kCUDA);
    auto torch_t2 = torch::from_blob(d2.data(), {static_cast<long>(N), 3}, torch::kFloat).clone().to(torch::kCUDA);
    auto torch_cat = torch::cat({torch_t1, torch_t2}, 0);
    auto torch_col0 = torch_cat.slice(1, 0, 1).squeeze(1);

    verifyVsTorch(our_col0, torch_col0, "VsTorch_Large_Cat_Column");
}

// ============================================================================
// PART 9: Multiple cat operations
// ============================================================================

TEST_F(TensorCatReductionBugTest, Cat_Multiple_Tensors) {
    std::vector<float> d1 = {100.0f, 200.0f};
    std::vector<float> d2 = {1.0f, 2.0f};
    std::vector<float> d3 = {-50.0f, -60.0f};
    std::vector<float> d4 = {500.0f, 600.0f};

    auto t1 = Tensor::from_vector(d1, {2}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {2}, Device::CUDA);
    auto t3 = Tensor::from_vector(d3, {2}, Device::CUDA);
    auto t4 = Tensor::from_vector(d4, {2}, Device::CUDA);

    auto cat = Tensor::cat({t1, t2, t3, t4}, 0);

    verifyMinMax(cat, "Cat_Multiple_Tensors");
    EXPECT_FLOAT_EQ(cat.min().item(), -60.0f);
    EXPECT_FLOAT_EQ(cat.max().item(), 600.0f);
}

TEST_F(TensorCatReductionBugTest, Cat_Then_Cat) {
    std::vector<float> d1 = {100.0f, 200.0f};
    std::vector<float> d2 = {1.0f, 2.0f};
    std::vector<float> d3 = {-50.0f, -60.0f};

    auto t1 = Tensor::from_vector(d1, {2}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {2}, Device::CUDA);
    auto t3 = Tensor::from_vector(d3, {2}, Device::CUDA);

    auto cat1 = Tensor::cat({t1, t2}, 0);
    auto cat2 = Tensor::cat({cat1, t3}, 0);

    verifyMinMax(cat2, "Cat_Then_Cat");
    EXPECT_FLOAT_EQ(cat2.min().item(), -60.0f);
    EXPECT_FLOAT_EQ(cat2.max().item(), 200.0f);
}

// ============================================================================
// PART 10: 3D tensor operations
// ============================================================================

TEST_F(TensorCatReductionBugTest, Cat_3D_Dim0) {
    std::vector<float> d1(2 * 3 * 4), d2(2 * 3 * 4);
    for (size_t i = 0; i < d1.size(); ++i) {
        d1[i] = 100.0f + static_cast<float>(i);
        d2[i] = -100.0f - static_cast<float>(i);
    }

    auto t1 = Tensor::from_vector(d1, {2, 3, 4}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {2, 3, 4}, Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    ASSERT_EQ(cat.size(0), 4);

    verifyMinMax(cat, "Cat_3D_Dim0");
    EXPECT_FLOAT_EQ(cat.min().item(), -123.0f); // -100 - 23
    EXPECT_FLOAT_EQ(cat.max().item(), 123.0f);  // 100 + 23
}

TEST_F(TensorCatReductionBugTest, Slice_3D_Dim1) {
    std::vector<float> data(4 * 5 * 6);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<float>(i);
    }

    auto t = Tensor::from_vector(data, {4, 5, 6}, Device::CUDA);

    // Slice along dim 1
    auto slice = t.slice(1, 2, 4); // [4, 2, 6]
    ASSERT_EQ(slice.size(0), 4);
    ASSERT_EQ(slice.size(1), 2);
    ASSERT_EQ(slice.size(2), 6);

    verifyMinMax(slice, "Slice_3D_Dim1");
}

// ============================================================================
// PART 11: Edge cases
// ============================================================================

TEST_F(TensorCatReductionBugTest, Cat_SingleElement) {
    std::vector<float> d1 = {100.0f};
    std::vector<float> d2 = {-50.0f};

    auto t1 = Tensor::from_vector(d1, {1}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {1}, Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    verifyMinMax(cat, "Cat_SingleElement");
    EXPECT_FLOAT_EQ(cat.min().item(), -50.0f);
    EXPECT_FLOAT_EQ(cat.max().item(), 100.0f);
}

TEST_F(TensorCatReductionBugTest, Slice_SingleElement) {
    std::vector<float> data = {100.0f, 200.0f, 300.0f};
    auto t = Tensor::from_vector(data, {3}, Device::CUDA);

    auto slice = t.slice(0, 1, 2); // Just [200.0f]
    EXPECT_FLOAT_EQ(slice.min().item(), 200.0f);
    EXPECT_FLOAT_EQ(slice.max().item(), 200.0f);
}

TEST_F(TensorCatReductionBugTest, NegativeValues) {
    std::vector<float> d1 = {-100.0f, -200.0f, -300.0f};
    std::vector<float> d2 = {-1.0f, -2.0f, -3.0f};

    auto t1 = Tensor::from_vector(d1, {3}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {3}, Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    verifyMinMax(cat, "NegativeValues");
    EXPECT_FLOAT_EQ(cat.min().item(), -300.0f);
    EXPECT_FLOAT_EQ(cat.max().item(), -1.0f);
}

TEST_F(TensorCatReductionBugTest, MixedSignValues) {
    std::vector<float> d1 = {-100.0f, 100.0f};
    std::vector<float> d2 = {-200.0f, 200.0f};

    auto t1 = Tensor::from_vector(d1, {2}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {2}, Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    verifyMinMax(cat, "MixedSignValues");
    EXPECT_FLOAT_EQ(cat.min().item(), -200.0f);
    EXPECT_FLOAT_EQ(cat.max().item(), 200.0f);
}

// ============================================================================
// PART 12: Verify is_contiguous() reporting
// ============================================================================

TEST_F(TensorCatReductionBugTest, IsContiguous_Fresh) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t = Tensor::from_vector(data, {2, 3}, Device::CUDA);

    EXPECT_TRUE(t.is_contiguous()) << "Fresh tensor should be contiguous";
}

TEST_F(TensorCatReductionBugTest, IsContiguous_AfterCat) {
    std::vector<float> d1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> d2 = {4.0f, 5.0f, 6.0f};

    auto t1 = Tensor::from_vector(d1, {3}, Device::CUDA);
    auto t2 = Tensor::from_vector(d2, {3}, Device::CUDA);
    auto cat = Tensor::cat({t1, t2}, 0);

    EXPECT_TRUE(cat.is_contiguous()) << "Cat result should be contiguous";
}

TEST_F(TensorCatReductionBugTest, IsContiguous_RowSlice) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t = Tensor::from_vector(data, {2, 3}, Device::CUDA);

    auto row0 = t.slice(0, 0, 1); // [1, 3]
    // Row slice should be contiguous (consecutive memory)
    EXPECT_TRUE(row0.is_contiguous()) << "Row slice should be contiguous";
}

TEST_F(TensorCatReductionBugTest, IsContiguous_ColumnSlice) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto t = Tensor::from_vector(data, {2, 3}, Device::CUDA);

    auto col0 = t.slice(1, 0, 1); // [2, 1] - elements at positions 0 and 3
    EXPECT_FALSE(col0.is_contiguous());
}

// ============================================================================
// PART 13: Stress tests with random data
// ============================================================================

TEST_F(TensorCatReductionBugTest, Random_Cat_Reduction_Small) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    for (int trial = 0; trial < 10; ++trial) {
        const size_t N1 = 100 + (gen() % 100);
        const size_t N2 = 100 + (gen() % 100);

        std::vector<float> d1(N1), d2(N2);
        for (auto& v : d1)
            v = dist(gen);
        for (auto& v : d2)
            v = dist(gen);

        auto t1 = Tensor::from_vector(d1, {N1}, Device::CUDA);
        auto t2 = Tensor::from_vector(d2, {N2}, Device::CUDA);
        auto cat = Tensor::cat({t1, t2}, 0);

        verifyMinMax(cat, "Random_Cat_Reduction_Small trial " + std::to_string(trial));
    }
}

TEST_F(TensorCatReductionBugTest, Random_Cat_Reduction_Large) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    for (int trial = 0; trial < 3; ++trial) {
        const size_t N1 = 100000 + (gen() % 100000);
        const size_t N2 = 100000 + (gen() % 100000);

        auto cpu1 = Tensor::zeros({N1}, Device::CPU);
        auto cpu2 = Tensor::zeros({N2}, Device::CPU);
        float* p1 = cpu1.ptr<float>();
        float* p2 = cpu2.ptr<float>();

        for (size_t i = 0; i < N1; ++i)
            p1[i] = dist(gen);
        for (size_t i = 0; i < N2; ++i)
            p2[i] = dist(gen);

        auto t1 = cpu1.to(Device::CUDA);
        auto t2 = cpu2.to(Device::CUDA);
        auto cat = Tensor::cat({t1, t2}, 0);

        verifyMinMax(cat, "Random_Cat_Reduction_Large trial " + std::to_string(trial));
    }
}

TEST_F(TensorCatReductionBugTest, Random_2D_Column_Reduction) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1000.0f, 1000.0f);

    for (int trial = 0; trial < 5; ++trial) {
        const size_t N = 10000 + (gen() % 10000);
        const size_t C = 3 + (gen() % 5);

        auto cpu = Tensor::zeros({N, C}, Device::CPU);
        float* p = cpu.ptr<float>();

        for (size_t i = 0; i < N * C; ++i) {
            p[i] = dist(gen);
        }

        auto t = cpu.to(Device::CUDA);

        for (size_t c = 0; c < C; ++c) {
            auto col = t.slice(1, c, c + 1).squeeze(1);
            verifyMinMax(col, "Random_2D_Column trial " + std::to_string(trial) + " col " + std::to_string(c));
        }
    }
}

// ============================================================================
// PART 14: Specific size thresholds (find where bug appears)
// ============================================================================

TEST_F(TensorCatReductionBugTest, SizeThresholdsPreserveReductionAcrossKernelBoundaries) {
    for (const size_t size : {1'000u, 10'000u, 100'000u, 500'000u, 1'000'000u}) {
        SCOPED_TRACE(::testing::Message() << "size=" << size);
        auto cpu1 = Tensor::full({size}, 100.0f, Device::CPU);
        auto cpu2 = Tensor::full({size}, -100.0f, Device::CPU);

        const auto cat = Tensor::cat({cpu1.cuda(), cpu2.cuda()}, 0);

        verifyMinMax(cat, "SizeThresholdsPreserveReductionAcrossKernelBoundaries");
    }
}

// ============================================================================
// PART 15: clone() after slice
// ============================================================================

TEST_F(TensorCatReductionBugTest, Clone_After_Slice) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 100.0f, 200.0f, 300.0f};
    auto t = Tensor::from_vector(data, {6}, Device::CUDA);

    auto slice = t.slice(0, 3, 6);
    auto cloned = slice.clone();

    EXPECT_TRUE(cloned.is_contiguous());
    EXPECT_EQ(cloned.storage_offset(), 0);

    verifyMinMax(cloned, "Clone_After_Slice");
    EXPECT_FLOAT_EQ(cloned.min().item(), 100.0f);
    EXPECT_FLOAT_EQ(cloned.max().item(), 300.0f);
}

TEST_F(TensorCatReductionBugTest, Clone_After_ColumnSlice) {
    std::vector<float> data = {
        1.0f, 100.0f,
        2.0f, 200.0f,
        3.0f, 300.0f};
    auto t = Tensor::from_vector(data, {3, 2}, Device::CUDA);

    auto col1 = t.slice(1, 1, 2).squeeze(1);
    auto cloned = col1.clone();

    EXPECT_TRUE(cloned.is_contiguous());
    EXPECT_EQ(cloned.storage_offset(), 0);

    verifyMinMax(cloned, "Clone_After_ColumnSlice");
    EXPECT_FLOAT_EQ(cloned.min().item(), 100.0f);
    EXPECT_FLOAT_EQ(cloned.max().item(), 300.0f);
}

// ============================================================================
// PART 16: to() device transfer after slice
// ============================================================================

TEST_F(TensorCatReductionBugTest, ToDevice_After_Slice) {
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 100.0f, 200.0f, 300.0f};
    auto t = Tensor::from_vector(data, {6}, Device::CUDA);

    auto slice = t.slice(0, 3, 6);

    // Transfer to CPU and back
    auto cpu = slice.to(Device::CPU);
    auto back_to_gpu = cpu.to(Device::CUDA);

    verifyMinMax(back_to_gpu, "ToDevice_After_Slice");
    EXPECT_FLOAT_EQ(back_to_gpu.min().item(), 100.0f);
    EXPECT_FLOAT_EQ(back_to_gpu.max().item(), 300.0f);
}

TEST_F(TensorCatReductionBugTest, CatTrimmedRowsWithZeroInitializedGrowth) {
    const auto storage = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f,
                           4.0f, 5.0f, 6.0f,
                           7.0f, 8.0f, 9.0f,
                           10.0f, 11.0f, 12.0f,
                           13.0f, 14.0f, 15.0f},
        {5, 3}, Device::CUDA);
    const auto trimmed = storage.slice(0, 0, 3).contiguous();
    const auto result = Tensor::cat({trimmed, Tensor::zeros({2, 3}, Device::CUDA)}, 0);

    EXPECT_EQ(result.shape(), TensorShape({5, 3}));
    EXPECT_EQ(result.cpu().to_vector(),
              (std::vector<float>{1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f,
                                  7.0f, 8.0f, 9.0f,
                                  0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f}));
}

TEST_F(TensorCatReductionBugTest, CatShCoefficientsAlongMiddleDimension) {
    const auto sh0 = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f},
        {2, 1, 3}, Device::CUDA);
    const auto shn = Tensor::zeros({2, 2, 3}, Device::CUDA);
    const auto result = Tensor::cat({sh0, shn}, 1);

    EXPECT_EQ(result.shape(), TensorShape({2, 3, 3}));
    EXPECT_EQ(result.cpu().to_vector(),
              (std::vector<float>{1.0f, 2.0f, 3.0f,
                                  0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f,
                                  4.0f, 5.0f, 6.0f,
                                  0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f}));
}

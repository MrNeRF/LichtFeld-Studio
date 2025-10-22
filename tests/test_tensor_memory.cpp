/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <torch/torch.h>

using namespace lfs::core;

// ============= Helper Functions =============

namespace {

    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         float rtol = 1e-4f, float atol = 1e-5f, const std::string& msg = "") {
        auto ref_cpu = reference.to(torch::kCPU).contiguous().flatten();
        auto custom_cpu = custom.cpu();

        ASSERT_EQ(custom_cpu.ndim(), reference.dim()) << msg << ": Rank mismatch";

        for (size_t i = 0; i < custom_cpu.ndim(); ++i) {
            ASSERT_EQ(custom_cpu.size(i), static_cast<size_t>(reference.size(i)))
                << msg << ": Shape mismatch at dim " << i;
        }

        auto custom_vec = custom_cpu.to_vector();
        auto ref_accessor = ref_cpu.accessor<float, 1>();

        for (size_t i = 0; i < custom_vec.size(); ++i) {
            float ref_val = ref_accessor[i];
            float custom_val = custom_vec[i];

            if (std::isnan(ref_val)) {
                EXPECT_TRUE(std::isnan(custom_val)) << msg << ": Expected NaN at index " << i;
            } else if (std::isinf(ref_val)) {
                EXPECT_TRUE(std::isinf(custom_val)) << msg << ": Expected Inf at index " << i;
            } else {
                float diff = std::abs(custom_val - ref_val);
                float threshold = atol + rtol * std::abs(ref_val);
                EXPECT_LE(diff, threshold)
                    << msg << ": Mismatch at index " << i
                    << " (custom=" << custom_val << ", ref=" << ref_val << ")";
            }
        }
    }

} // anonymous namespace

class TensorMemoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA is not available for testing";

        // Record initial CUDA memory
        cudaMemGetInfo(&initial_free_mem, &total_mem);

        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }

    void TearDown() override {
        // Force PyTorch to empty its CUDA cache
        c10::cuda::CUDACachingAllocator::emptyCache();

        // Force synchronization
        cudaDeviceSynchronize();

        // Check for memory leaks
        size_t current_free_mem, current_total;
        cudaMemGetInfo(&current_free_mem, &current_total);

        // PyTorch uses a caching allocator - need much larger tolerance
        // Also, CUDA driver may cache memory for performance
        size_t tolerance = 150 * 1024 * 1024; // 150 MB tolerance
        EXPECT_NEAR(static_cast<long long>(current_free_mem),
                    static_cast<long long>(initial_free_mem),
                    static_cast<long long>(tolerance))
            << "Possible memory leak detected. Difference: "
            << (static_cast<long long>(initial_free_mem) - static_cast<long long>(current_free_mem)) / (1024 * 1024)
            << " MB";
    }

    size_t initial_free_mem;
    size_t total_mem;
};

// ============= Memory Ownership Tests =============

TEST_F(TensorMemoryTest, MemoryOwnership) {
    // Test owning memory - compare with PyTorch behavior
    {
        auto custom_t = Tensor::zeros({100, 100}, Device::CUDA);
        auto torch_t = torch::zeros({100, 100}, torch::TensorOptions().device(torch::kCUDA));

        EXPECT_TRUE(custom_t.owns_memory());
        EXPECT_NE(custom_t.raw_ptr(), nullptr);
        EXPECT_NE(torch_t.data_ptr(), nullptr);

        compare_tensors(custom_t, torch_t, 1e-6f, 1e-7f, "OwningMemory");
    }
    // Memory should be freed after scope

    // Test non-owning view
    float* cuda_data;
    cudaMalloc(&cuda_data, 100 * sizeof(float));
    cudaMemset(cuda_data, 0, 100 * sizeof(float));

    {
        auto custom_t = Tensor::from_blob(cuda_data, {10, 10}, Device::CUDA, DataType::Float32);
        auto torch_t = torch::from_blob(cuda_data, {10, 10},
                                        torch::TensorOptions().device(torch::kCUDA));

        EXPECT_FALSE(custom_t.owns_memory());
        EXPECT_EQ(custom_t.raw_ptr(), cuda_data);
        EXPECT_EQ(torch_t.data_ptr(), cuda_data);

        compare_tensors(custom_t, torch_t, 1e-6f, 1e-7f, "NonOwningBlob");
    }
    // Memory should NOT be freed after scope
    cudaFree(cuda_data);
}

TEST_F(TensorMemoryTest, MoveSemantics) {
    // Test move constructor
    void* original_ptr = nullptr;
    {
        auto custom_t1 = Tensor::ones({50, 50}, Device::CUDA);
        EXPECT_TRUE(custom_t1.owns_memory());
        original_ptr = custom_t1.raw_ptr();

        auto custom_t2 = std::move(custom_t1);
        EXPECT_EQ(custom_t2.raw_ptr(), original_ptr);
        EXPECT_TRUE(custom_t2.owns_memory());
        EXPECT_EQ(custom_t1.raw_ptr(), nullptr);
        EXPECT_FALSE(custom_t1.owns_memory());

        // Verify data is correct
        auto torch_t = torch::ones({50, 50}, torch::TensorOptions().device(torch::kCUDA));
        compare_tensors(custom_t2, torch_t, 1e-6f, 1e-7f, "MoveConstructor");
    }

    // Test move assignment
    {
        auto custom_t1 = Tensor::zeros({30, 30}, Device::CUDA);
        auto custom_t2 = Tensor::ones({20, 20}, Device::CUDA);

        void* ptr1 = custom_t1.raw_ptr();
        void* ptr2 = custom_t2.raw_ptr();

        custom_t2 = std::move(custom_t1);

        EXPECT_EQ(custom_t2.raw_ptr(), ptr1);
        EXPECT_EQ(custom_t1.raw_ptr(), nullptr);

        // Verify data is zeros
        auto torch_zeros = torch::zeros({30, 30}, torch::TensorOptions().device(torch::kCUDA));
        compare_tensors(custom_t2, torch_zeros, 1e-6f, 1e-7f, "MoveAssignment");
    }
}

// ============= View and Slice Tests =============

TEST_F(TensorMemoryTest, ViewDoesNotOwnMemory) {
    auto custom_original = Tensor::ones({4, 5, 6}, Device::CUDA);
    auto torch_original = torch::ones({4, 5, 6}, torch::TensorOptions().device(torch::kCUDA));

    EXPECT_TRUE(custom_original.owns_memory());

    auto custom_view = custom_original.view({20, 6});
    auto torch_view = torch_original.view({20, 6});

    EXPECT_FALSE(custom_view.owns_memory());
    EXPECT_EQ(custom_view.raw_ptr(), custom_original.raw_ptr());
    EXPECT_EQ(torch_view.data_ptr(), torch_original.data_ptr());

    compare_tensors(custom_view, torch_view, 1e-6f, 1e-7f, "ViewDoesNotOwn");

    // Modifying view should affect original
    custom_view.fill_(2.0f);
    torch_view.fill_(2.0f);

    compare_tensors(custom_original, torch_original, 1e-6f, 1e-7f, "ViewModifiesOriginal");

    auto custom_values = custom_original.to_vector();
    for (float val : custom_values) {
        EXPECT_FLOAT_EQ(val, 2.0f);
    }
}

TEST_F(TensorMemoryTest, SliceDoesNotOwnMemory) {
    auto custom_original = Tensor::full({10, 10}, 3.0f, Device::CUDA);
    auto torch_original = torch::full({10, 10}, 3.0f,
                                      torch::TensorOptions().device(torch::kCUDA));

    EXPECT_TRUE(custom_original.owns_memory());

    auto custom_slice = custom_original.slice(0, 2, 5);
    auto torch_slice = torch_original.slice(0, 2, 5);

    EXPECT_FALSE(custom_slice.owns_memory());

    // Both slices should point to part of original memory
    EXPECT_GE(custom_slice.raw_ptr(), custom_original.raw_ptr());
    EXPECT_LT(custom_slice.raw_ptr(),
              static_cast<char*>(custom_original.raw_ptr()) + custom_original.bytes());

    compare_tensors(custom_slice, torch_slice, 1e-6f, 1e-7f, "SliceDoesNotOwn");
}

TEST_F(TensorMemoryTest, CloneOwnsMemory) {
    auto custom_original = Tensor::ones({3, 3}, Device::CUDA);
    auto torch_original = torch::ones({3, 3}, torch::TensorOptions().device(torch::kCUDA));

    EXPECT_TRUE(custom_original.owns_memory());

    auto custom_cloned = custom_original.clone();
    auto torch_cloned = torch_original.clone();

    EXPECT_TRUE(custom_cloned.owns_memory());
    EXPECT_NE(custom_cloned.raw_ptr(), custom_original.raw_ptr());
    EXPECT_NE(torch_cloned.data_ptr(), torch_original.data_ptr());

    compare_tensors(custom_cloned, torch_cloned, 1e-6f, 1e-7f, "CloneOwnsMemory");

    // Modifying clone should not affect original
    custom_cloned.fill_(5.0f);
    torch_cloned.fill_(5.0f);

    compare_tensors(custom_cloned, torch_cloned, 1e-6f, 1e-7f, "CloneModified");

    // Verify custom original is still 1.0
    auto custom_values = custom_original.to_vector();
    for (float val : custom_values) {
        EXPECT_FLOAT_EQ(val, 1.0f);
    }

    // Verify torch original is still 1.0 (properly flatten for indexing)
    auto torch_flat = torch_original.to(torch::kCPU).flatten();
    for (int64_t i = 0; i < torch_flat.numel(); ++i) {
        EXPECT_FLOAT_EQ(torch_flat[i].item<float>(), 1.0f);
    }
}

// ============= Device Transfer Tests =============

TEST_F(TensorMemoryTest, DeviceTransferOwnsMemory) {
    auto custom_cuda = Tensor::full({5, 5}, 2.5f, Device::CUDA);
    auto torch_cuda = torch::full({5, 5}, 2.5f,
                                  torch::TensorOptions().device(torch::kCUDA));

    EXPECT_TRUE(custom_cuda.owns_memory());

    auto custom_cpu = custom_cuda.to(Device::CPU);
    auto torch_cpu = torch_cuda.to(torch::kCPU);

    EXPECT_TRUE(custom_cpu.owns_memory());
    EXPECT_NE(custom_cpu.raw_ptr(), custom_cuda.raw_ptr());
    EXPECT_NE(torch_cpu.data_ptr(), torch_cuda.data_ptr());

    compare_tensors(custom_cpu, torch_cpu, 1e-6f, 1e-7f, "CPUTransfer");

    auto custom_cuda2 = custom_cpu.to(Device::CUDA);
    auto torch_cuda2 = torch_cpu.to(torch::kCUDA);

    EXPECT_TRUE(custom_cuda2.owns_memory());
    EXPECT_NE(custom_cuda2.raw_ptr(), custom_cuda.raw_ptr());
    EXPECT_NE(custom_cuda2.raw_ptr(), custom_cpu.raw_ptr());

    compare_tensors(custom_cuda2, torch_cuda2, 1e-6f, 1e-7f, "CUDATransfer");
}

TEST_F(TensorMemoryTest, DeviceTransferRoundtrip) {
    auto custom_original = Tensor::randn({10, 10}, Device::CUDA);
    auto custom_data = custom_original.to_vector(); // Save original data

    auto torch_original = torch::from_blob(custom_data.data(), {10, 10},
                                           torch::TensorOptions().dtype(torch::kFloat32))
                              .clone()
                              .to(torch::kCUDA);

    // Roundtrip: CUDA -> CPU -> CUDA
    auto custom_result = custom_original.cpu().cuda();
    auto torch_result = torch_original.cpu().cuda();

    compare_tensors(custom_result, torch_result, 1e-5f, 1e-6f, "DeviceRoundtrip");

    // Verify roundtrip preserves original values
    auto result_data = custom_result.to_vector();
    EXPECT_EQ(result_data.size(), custom_data.size());
    for (size_t i = 0; i < result_data.size(); ++i) {
        EXPECT_NEAR(result_data[i], custom_data[i], 1e-5f)
            << "Roundtrip should preserve values at index " << i;
    }
}

// ============= Large Allocation Tests =============

TEST_F(TensorMemoryTest, LargeTensorAllocation) {
    // Test large tensor allocation
    const size_t large_size = 1024 * 1024; // 1M elements = 4MB for float32

    {
        auto custom_t = Tensor::zeros({large_size}, Device::CUDA);
        auto torch_t = torch::zeros({static_cast<long>(large_size)},
                                    torch::TensorOptions().device(torch::kCUDA));

        EXPECT_TRUE(custom_t.is_valid());
        EXPECT_EQ(custom_t.numel(), large_size);
        EXPECT_EQ(custom_t.bytes(), large_size * sizeof(float));
        EXPECT_EQ(torch_t.numel(), large_size);

        // Sample comparison (don't copy all 1M elements)
        auto custom_sample = custom_t.slice(0, 0, 100);
        auto torch_sample = torch_t.slice(0, 0, 100);
        compare_tensors(custom_sample, torch_sample, 1e-6f, 1e-7f, "LargeTensorSample");
    }
    // Should be freed
}

TEST_F(TensorMemoryTest, VeryLargeTensor) {
    // Test with 100MB tensor
    const size_t size = 25 * 1024 * 1024; // 25M floats = 100MB

    size_t free_mem, total;
    cudaMemGetInfo(&free_mem, &total);

    if (free_mem < size * sizeof(float) * 2) {
        GTEST_SKIP() << "Not enough memory for very large tensor test";
    }

    {
        auto custom_t = Tensor::ones({size}, Device::CUDA);
        auto torch_t = torch::ones({static_cast<long>(size)},
                                   torch::TensorOptions().device(torch::kCUDA));

        EXPECT_TRUE(custom_t.is_valid());
        EXPECT_EQ(custom_t.numel(), size);

        // Verify a sample
        auto custom_sample = custom_t.slice(0, 1000, 1100);
        auto torch_sample = torch_t.slice(0, 1000, 1100);
        compare_tensors(custom_sample, torch_sample, 1e-6f, 1e-7f, "VeryLargeSample");
    }
}

// ============= Multiple Views Tests =============

TEST_F(TensorMemoryTest, MultipleViewsOfSameMemory) {
    auto custom_original = Tensor::ones({24}, Device::CUDA);
    auto torch_original = torch::ones({24}, torch::TensorOptions().device(torch::kCUDA));

    auto custom_view1 = custom_original.view({2, 12});
    auto custom_view2 = custom_original.view({3, 8});
    auto custom_view3 = custom_original.view({4, 6});

    auto torch_view1 = torch_original.view({2, 12});
    auto torch_view2 = torch_original.view({3, 8});
    auto torch_view3 = torch_original.view({4, 6});

    // All views should share the same memory
    EXPECT_EQ(custom_view1.raw_ptr(), custom_original.raw_ptr());
    EXPECT_EQ(custom_view2.raw_ptr(), custom_original.raw_ptr());
    EXPECT_EQ(custom_view3.raw_ptr(), custom_original.raw_ptr());

    EXPECT_EQ(torch_view1.data_ptr(), torch_original.data_ptr());
    EXPECT_EQ(torch_view2.data_ptr(), torch_original.data_ptr());
    EXPECT_EQ(torch_view3.data_ptr(), torch_original.data_ptr());

    // None of the views should own memory
    EXPECT_FALSE(custom_view1.owns_memory());
    EXPECT_FALSE(custom_view2.owns_memory());
    EXPECT_FALSE(custom_view3.owns_memory());

    // Modifying through one view affects all
    custom_view1.fill_(3.0f);
    torch_view1.fill_(3.0f);

    compare_tensors(custom_original, torch_original, 1e-6f, 1e-7f, "MultiViewOriginal");
    compare_tensors(custom_view2, torch_view2, 1e-6f, 1e-7f, "MultiView2");
    compare_tensors(custom_view3, torch_view3, 1e-6f, 1e-7f, "MultiView3");
}

TEST_F(TensorMemoryTest, NestedViews) {
    auto custom_t = Tensor::ones({120}, Device::CUDA);
    auto torch_t = torch::ones({120}, torch::TensorOptions().device(torch::kCUDA));

    // Create nested views
    auto custom_v1 = custom_t.view({10, 12});
    auto custom_v2 = custom_v1.view({2, 60});
    auto custom_v3 = custom_v2.view({120});

    auto torch_v1 = torch_t.view({10, 12});
    auto torch_v2 = torch_v1.view({2, 60});
    auto torch_v3 = torch_v2.view({120});

    // All should point to same memory
    EXPECT_EQ(custom_v1.raw_ptr(), custom_t.raw_ptr());
    EXPECT_EQ(custom_v2.raw_ptr(), custom_t.raw_ptr());
    EXPECT_EQ(custom_v3.raw_ptr(), custom_t.raw_ptr());

    compare_tensors(custom_v3, torch_v3, 1e-6f, 1e-7f, "NestedViews");
}

// ============= Copy Operations Tests =============

TEST_F(TensorMemoryTest, CopyFromPreservesOwnership) {
    auto custom_t1 = Tensor::ones({3, 3}, Device::CUDA);
    auto custom_t2 = Tensor::zeros({3, 3}, Device::CUDA);

    auto torch_t1 = torch::ones({3, 3}, torch::TensorOptions().device(torch::kCUDA));
    auto torch_t2 = torch::zeros({3, 3}, torch::TensorOptions().device(torch::kCUDA));

    EXPECT_TRUE(custom_t1.owns_memory());
    EXPECT_TRUE(custom_t2.owns_memory());

    void* ptr1 = custom_t1.raw_ptr();
    void* ptr2 = custom_t2.raw_ptr();

    custom_t2.copy_from(custom_t1);
    torch_t2.copy_(torch_t1);

    // Both should still own their memory
    EXPECT_TRUE(custom_t1.owns_memory());
    EXPECT_TRUE(custom_t2.owns_memory());

    // Pointers should not have changed
    EXPECT_EQ(custom_t1.raw_ptr(), ptr1);
    EXPECT_EQ(custom_t2.raw_ptr(), ptr2);

    compare_tensors(custom_t1, torch_t1, 1e-6f, 1e-7f, "CopyFromSource");
    compare_tensors(custom_t2, torch_t2, 1e-6f, 1e-7f, "CopyFromDest");
}

TEST_F(TensorMemoryTest, CopyBetweenDevices) {
    auto custom_cuda = Tensor::full({5, 5}, 7.0f, Device::CUDA);
    auto custom_cpu = Tensor::zeros({5, 5}, Device::CPU);

    auto torch_cuda = torch::full({5, 5}, 7.0f,
                                  torch::TensorOptions().device(torch::kCUDA));
    auto torch_cpu = torch::zeros({5, 5}, torch::TensorOptions().device(torch::kCPU));

    custom_cpu.copy_from(custom_cuda);
    torch_cpu.copy_(torch_cuda);

    compare_tensors(custom_cpu, torch_cpu, 1e-6f, 1e-7f, "CopyBetweenDevices");
}

// ============= Invalid Tensor Tests =============

TEST_F(TensorMemoryTest, InvalidTensorOperations) {
    Tensor custom_invalid;

    EXPECT_FALSE(custom_invalid.is_valid());
    EXPECT_EQ(custom_invalid.raw_ptr(), nullptr);
    EXPECT_FALSE(custom_invalid.owns_memory());

    // Operations on invalid tensor should return invalid tensors
    auto result1 = custom_invalid.clone();
    EXPECT_FALSE(result1.is_valid());

    auto result2 = custom_invalid.to(Device::CPU);
    EXPECT_FALSE(result2.is_valid());

    auto result3 = custom_invalid.view({2, 2});
    EXPECT_FALSE(result3.is_valid());
}

TEST_F(TensorMemoryTest, EmptyTensor) {
    auto custom_empty = Tensor::empty({0}, Device::CUDA);
    auto torch_empty = torch::empty({0}, torch::TensorOptions().device(torch::kCUDA));

    EXPECT_TRUE(custom_empty.is_valid());
    EXPECT_EQ(custom_empty.numel(), 0);
    EXPECT_EQ(torch_empty.numel(), 0);

    // Both should handle empty tensors gracefully
    auto custom_clone = custom_empty.clone();
    auto torch_clone = torch_empty.clone();

    EXPECT_TRUE(custom_clone.is_valid());
    EXPECT_EQ(custom_clone.numel(), 0);
    EXPECT_EQ(torch_clone.numel(), 0);
}

// ============= Memory Alignment Tests (Implementation-Specific) =============

TEST_F(TensorMemoryTest, MemoryAlignmentAndPadding) {
    // Test that memory is properly aligned
    auto custom_t = Tensor::empty({17}, Device::CUDA); // Odd size

    // CUDA memory should be aligned to at least 256 bytes
    uintptr_t addr = reinterpret_cast<uintptr_t>(custom_t.raw_ptr());
    EXPECT_EQ(addr % 256, 0) << "CUDA memory not properly aligned";

    // PyTorch also aligns memory
    auto torch_t = torch::empty({17}, torch::TensorOptions().device(torch::kCUDA));
    uintptr_t torch_addr = reinterpret_cast<uintptr_t>(torch_t.data_ptr());
    EXPECT_EQ(torch_addr % 256, 0) << "PyTorch CUDA memory not properly aligned";
}

// ============= Stress Tests =============

TEST_F(TensorMemoryTest, StressTestManyAllocations) {
    // Stress test with many small allocations
    std::vector<Tensor> custom_tensors;
    std::vector<torch::Tensor> torch_tensors;

    for (int i = 0; i < 100; ++i) {
        custom_tensors.emplace_back(Tensor::zeros({10, 10}, Device::CUDA));
        torch_tensors.push_back(torch::zeros({10, 10},
                                             torch::TensorOptions().device(torch::kCUDA)));
    }

    // All should be valid
    for (size_t i = 0; i < custom_tensors.size(); ++i) {
        EXPECT_TRUE(custom_tensors[i].is_valid());
        EXPECT_TRUE(custom_tensors[i].owns_memory());
        EXPECT_TRUE(torch_tensors[i].defined());
    }

    // Spot check a few
    compare_tensors(custom_tensors[0], torch_tensors[0], 1e-6f, 1e-7f, "StressTest0");
    compare_tensors(custom_tensors[50], torch_tensors[50], 1e-6f, 1e-7f, "StressTest50");
    compare_tensors(custom_tensors[99], torch_tensors[99], 1e-6f, 1e-7f, "StressTest99");

    // Clear them
    custom_tensors.clear();
    torch_tensors.clear();

    // Memory should be freed (checked in TearDown)
}

TEST_F(TensorMemoryTest, StressTestRapidAllocDealloc) {
    // Rapid allocation and deallocation
    for (int i = 0; i < 1000; ++i) {
        auto custom_t = Tensor::randn({100}, Device::CUDA);
        auto torch_t = torch::randn({100}, torch::TensorOptions().device(torch::kCUDA));

        EXPECT_TRUE(custom_t.is_valid());
        EXPECT_TRUE(torch_t.defined());
        // Both destroyed at end of iteration
    }
    // Memory should be stable (checked in TearDown)
}

TEST_F(TensorMemoryTest, MixedSizeAllocations) {
    std::vector<Tensor> tensors;
    std::vector<size_t> sizes = {10, 100, 1000, 10000, 100000};

    for (size_t size : sizes) {
        auto custom_t = Tensor::ones({size}, Device::CUDA);
        auto torch_t = torch::ones({static_cast<long>(size)},
                                   torch::TensorOptions().device(torch::kCUDA));

        EXPECT_EQ(custom_t.numel(), size);
        EXPECT_EQ(torch_t.numel(), size);

        tensors.emplace_back(std::move(custom_t));
    }

    // All should still be valid
    for (size_t i = 0; i < tensors.size(); ++i) {
        EXPECT_TRUE(tensors[i].is_valid());
        EXPECT_EQ(tensors[i].numel(), sizes[i]);
    }
}

// ============= Lifetime Management =============

TEST_F(TensorMemoryTest, ViewLifetimeSafety) {
    // Ensure views remain valid even after original goes out of scope
    // This tests reference counting or similar mechanisms

    Tensor custom_view;
    torch::Tensor torch_view;

    {
        auto custom_original = Tensor::ones({100}, Device::CUDA);
        auto torch_original = torch::ones({100}, torch::TensorOptions().device(torch::kCUDA));

        custom_view = custom_original.view({10, 10});
        torch_view = torch_original.view({10, 10});

        EXPECT_FALSE(custom_view.owns_memory());
        // Original goes out of scope here, but view should keep memory alive
    }

    // View should still be valid
    EXPECT_TRUE(custom_view.is_valid());
    EXPECT_TRUE(torch_view.defined());

    compare_tensors(custom_view, torch_view, 1e-6f, 1e-7f, "ViewAfterOriginalDestroyed");
}

TEST_F(TensorMemoryTest, CloneIndependence) {
    auto custom_original = Tensor::randn({50}, Device::CUDA);
    auto custom_data = custom_original.to_vector();

    auto torch_original = torch::from_blob(custom_data.data(), {50},
                                           torch::TensorOptions().dtype(torch::kFloat32))
                              .clone()
                              .to(torch::kCUDA);

    auto custom_clone = custom_original.clone();
    auto torch_clone = torch_original.clone();

    // Destroy original
    custom_original = Tensor();
    torch_original = torch::Tensor();

    // Clone should still be valid and have correct data
    EXPECT_TRUE(custom_clone.is_valid());
    EXPECT_TRUE(torch_clone.defined());

    // Verify data matches what we saved
    auto result_data = custom_clone.to_vector();
    EXPECT_EQ(result_data.size(), custom_data.size());
    for (size_t i = 0; i < result_data.size(); ++i) {
        EXPECT_NEAR(result_data[i], custom_data[i], 1e-5f);
    }
}
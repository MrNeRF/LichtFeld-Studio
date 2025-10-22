/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core_new/tensor.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <torch/torch.h>

class BugVerificationTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        lfs::core::Tensor::manual_seed(42);
    }
};

// ============================================================================
// BUG 1: eq() for Int32 tensors
// ============================================================================

TEST_F(BugVerificationTest, Bug1_Int32_Equality) {
    std::cout << "\n=== BUG 1: Int32 Equality ===" << std::endl;

    // Create Int32 tensor
    std::vector<int> data = {0, 1, 2, 1, 0, 2};
    auto gs_labels = lfs::core::Tensor::from_vector(data, {6}, lfs::core::Device::CUDA);

    std::cout << "Input: [0, 1, 2, 1, 0, 2]" << std::endl;
    std::cout << "Testing: labels == 1" << std::endl;
    std::cout << "Expected: [false, true, false, true, false, false]" << std::endl;

    // Test equality
    auto mask = gs_labels.eq(1);

    std::cout << "Result dtype: " << lfs::core::dtype_name(mask.dtype()) << std::endl;

    auto mask_cpu = mask.cpu();

    // Convert to readable format
    std::vector<bool> result;
    if (mask_cpu.dtype() == lfs::core::DataType::Bool) {
        auto vec = mask_cpu.to_vector_bool();
        result.assign(vec.begin(), vec.end());
    } else if (mask_cpu.dtype() == lfs::core::DataType::Float32) {
        auto vec = mask_cpu.to_vector();
        for (float v : vec) {
            result.push_back(v > 0.5f);
        }
    } else if (mask_cpu.dtype() == lfs::core::DataType::Int32) {
        auto vec = mask_cpu.to_vector_int();
        for (int v : vec) {
            result.push_back(v != 0);
        }
    }

    std::cout << "Actual:   [";
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << (result[i] ? "true" : "false");
        if (i < result.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Verify
    std::vector<bool> expected = {false, true, false, true, false, false};
    bool all_match = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (result[i] != expected[i]) {
            std::cout << "✗ MISMATCH at index " << i << ": got " << result[i]
                      << " expected " << expected[i] << std::endl;
            all_match = false;
        }
    }

    if (all_match) {
        std::cout << "✓ PASS: eq() works correctly" << std::endl;
    } else {
        std::cout << "✗ FAIL: eq() is broken for Int32 tensors!" << std::endl;
    }

    EXPECT_TRUE(all_match) << "BUG CONFIRMED: eq() broken for Int32";
}

TEST_F(BugVerificationTest, Bug1_Float32_Equality_AsBaseline) {
    std::cout << "\n=== Float32 Equality (baseline) ===" << std::endl;

    // Test with Float32 to see if it works there
    std::vector<float> data = {0.0f, 1.0f, 2.0f, 1.0f, 0.0f, 2.0f};
    auto gs_labels = lfs::core::Tensor::from_vector(data, {6}, lfs::core::Device::CUDA);

    std::cout << "Input: [0.0, 1.0, 2.0, 1.0, 0.0, 2.0]" << std::endl;
    std::cout << "Testing: labels == 1.0" << std::endl;

    auto mask = gs_labels.eq(1.0f);
    auto mask_cpu = mask.cpu();

    std::vector<bool> result;
    if (mask_cpu.dtype() == lfs::core::DataType::Bool) {
        auto vec = mask_cpu.to_vector_bool();
        result.assign(vec.begin(), vec.end());
    } else {
        auto vec = mask_cpu.to_vector();
        for (float v : vec) {
            result.push_back(v > 0.5f);
        }
    }

    std::cout << "Result:   [";
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << (result[i] ? "true" : "false");
        if (i < result.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::vector<bool> expected = {false, true, false, true, false, false};
    bool all_match = true;
    for (size_t i = 0; i < expected.size(); ++i) {
        if (result[i] != expected[i]) {
            all_match = false;
            break;
        }
    }

    if (all_match) {
        std::cout << "✓ Float32 equality works" << std::endl;
    } else {
        std::cout << "✗ Float32 equality also broken!" << std::endl;
    }
}

// ============================================================================
// BUG 2: clone() doesn't create independent copy
// ============================================================================

TEST_F(BugVerificationTest, Bug2_Clone_Independence) {
    std::cout << "\n=== BUG 2: Clone Independence ===" << std::endl;

    // Create original tensor
    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto original = lfs::core::Tensor::from_vector(data, {5}, lfs::core::Device::CUDA);

    std::cout << "Original values: ";
    auto orig_cpu = original.cpu().to_vector();
    for (float v : orig_cpu)
        std::cout << v << " ";
    std::cout << std::endl;

    // Clone it
    std::cout << "Creating clone..." << std::endl;
    auto cloned = original.clone();

    std::cout << "Clone values: ";
    auto clone_cpu = cloned.cpu().to_vector();
    for (float v : clone_cpu)
        std::cout << v << " ";
    std::cout << std::endl;

    // Check if they share memory
    std::cout << "Original raw_ptr: " << original.raw_ptr() << std::endl;
    std::cout << "Clone raw_ptr:    " << cloned.raw_ptr() << std::endl;

    if (original.raw_ptr() == cloned.raw_ptr()) {
        std::cout << "✗ ERROR: Clone shares same memory pointer!" << std::endl;
    }

    // Modify clone
    std::cout << "\nZeroing clone..." << std::endl;
    cloned.zero_();

    std::cout << "Clone after zero: ";
    clone_cpu = cloned.cpu().to_vector();
    for (float v : clone_cpu)
        std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "Original after clone.zero_(): ";
    orig_cpu = original.cpu().to_vector();
    for (float v : orig_cpu)
        std::cout << v << " ";
    std::cout << std::endl;

    // Check if original was affected
    bool original_unchanged = true;
    for (size_t i = 0; i < orig_cpu.size(); ++i) {
        if (std::abs(orig_cpu[i] - data[i]) > 1e-5f) {
            std::cout << "✗ Original modified at index " << i << ": "
                      << orig_cpu[i] << " (was " << data[i] << ")" << std::endl;
            original_unchanged = false;
        }
    }

    if (original_unchanged) {
        std::cout << "✓ PASS: Clone is independent" << std::endl;
    } else {
        std::cout << "✗ FAIL: Clone is not independent - BUG CONFIRMED!" << std::endl;
    }

    EXPECT_TRUE(original_unchanged) << "BUG CONFIRMED: clone() not independent";
}

TEST_F(BugVerificationTest, Bug2_Clone_MemoryOwnership) {
    std::cout << "\n=== Clone Memory Ownership Check ===" << std::endl;

    auto original = lfs::core::Tensor::from_vector({10.0f, 20.0f, 30.0f}, {3}, lfs::core::Device::CUDA);

    std::cout << "Original owns_memory: " << original.owns_memory() << std::endl;
    std::cout << "Original is_view: " << original.is_view() << std::endl;

    auto cloned = original.clone();

    std::cout << "Clone owns_memory: " << cloned.owns_memory() << std::endl;
    std::cout << "Clone is_view: " << cloned.is_view() << std::endl;

    if (!cloned.owns_memory()) {
        std::cout << "✗ ERROR: Clone doesn't own its memory!" << std::endl;
    }

    if (cloned.is_view()) {
        std::cout << "✗ ERROR: Clone is marked as a view!" << std::endl;
    }

    EXPECT_TRUE(cloned.owns_memory());
    EXPECT_FALSE(cloned.is_view());
}

// ============================================================================
// Impact on K-means
// ============================================================================

TEST_F(BugVerificationTest, Bug_Impact_KMeans_CentroidUpdate) {
    std::cout << "\n=== K-means Centroid Update Pattern ===" << std::endl;

    // Simulate k-means centroid update
    std::vector<int> labels_data = {0, 1, 0, 2, 1, 0, 2, 1};
    auto labels = lfs::core::Tensor::from_vector(labels_data, {8}, lfs::core::Device::CUDA);

    std::vector<float> data_vals = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto data = lfs::core::Tensor::from_vector(data_vals, {8}, lfs::core::Device::CUDA);

    int cluster_id = 1;

    std::cout << "Finding points in cluster " << cluster_id << std::endl;
    std::cout << "Labels: [";
    for (int l : labels_data)
        std::cout << l << " ";
    std::cout << "]" << std::endl;
    std::cout << "Expected mask: [false, true, false, false, true, false, false, true]" << std::endl;

    // This is what k-means does
    auto mask = labels.eq(cluster_id);

    std::cout << "\nChecking if any points in cluster..." << std::endl;
    bool has_points = false;
    try {
        has_points = mask.any_scalar();
        std::cout << "any_scalar() returned: " << has_points << std::endl;
    } catch (const std::exception& e) {
        std::cout << "✗ any_scalar() failed: " << e.what() << std::endl;
    }

    if (!has_points) {
        std::cout << "✗ CRITICAL: K-means would skip this cluster!" << std::endl;
        FAIL() << "eq() bug causes k-means to skip clusters";
    }

    std::cout << "✓ Cluster has points" << std::endl;

    // Try to select points
    std::cout << "\nSelecting points with masked_select..." << std::endl;
    try {
        auto selected = data.masked_select(mask);
        std::cout << "Selected " << selected.numel() << " points (expected 3)" << std::endl;

        if (selected.numel() != 3) {
            std::cout << "✗ CRITICAL: Wrong number of points selected!" << std::endl;
        }

        auto selected_cpu = selected.cpu().to_vector();
        std::cout << "Selected values: ";
        for (float v : selected_cpu)
            std::cout << v << " ";
        std::cout << std::endl;
        std::cout << "Expected: 2.0 5.0 8.0" << std::endl;

    } catch (const std::exception& e) {
        std::cout << "✗ masked_select failed: " << e.what() << std::endl;
    }
}

TEST_F(BugVerificationTest, Bug_Impact_KMeans_ConvergenceCheck) {
    std::cout << "\n=== K-means Convergence Check Pattern ===" << std::endl;

    // Simulate convergence check
    auto centroids = lfs::core::Tensor::from_vector({1.0f, 2.0f, 3.0f}, {3}, lfs::core::Device::CUDA);

    std::cout << "Saving old centroids with clone()..." << std::endl;
    auto old_centroids = centroids.clone();

    std::cout << "Updating centroids..." << std::endl;
    centroids.zero_();
    centroids.add_(5.0f); // Set to [5, 5, 5]

    std::cout << "Old centroids: ";
    auto old_cpu = old_centroids.cpu().to_vector();
    for (float v : old_cpu)
        std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "Expected: 1.0 2.0 3.0" << std::endl;

    bool old_unchanged = (std::abs(old_cpu[0] - 1.0f) < 1e-5f &&
                          std::abs(old_cpu[1] - 2.0f) < 1e-5f &&
                          std::abs(old_cpu[2] - 3.0f) < 1e-5f);

    if (!old_unchanged) {
        std::cout << "✗ CRITICAL: K-means convergence check broken!" << std::endl;
        std::cout << "  (centroids - old_centroids) would be zero, causing immediate convergence" << std::endl;
        FAIL() << "clone() bug breaks k-means convergence";
    }

    std::cout << "✓ Old centroids preserved correctly" << std::endl;
}

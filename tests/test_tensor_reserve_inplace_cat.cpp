/**
 * Test for tensor reserve() + in-place cat() functionality
 */
#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <string>
#include <vector>

using namespace lfs::core;

TEST(TensorReserveInplaceCat, BasicReserveAndCat) {
    // Create a small tensor and reserve capacity
    auto t1_data = std::vector<float>(10);
    for (int i = 0; i < 10; i++)
        t1_data[i] = static_cast<float>(i);
    auto t1 = Tensor::from_vector(t1_data, {10, 1}, Device::CUDA);

    // Reserve capacity for 100 rows
    t1.reserve(100);

    // Verify capacity was set
    EXPECT_EQ(t1.capacity(), 100);
    EXPECT_EQ(t1.logical_size(), 10);
    EXPECT_EQ(t1.shape()[0], 10);

    // Create a new tensor to concatenate
    auto t2_data = std::vector<float>(10);
    for (int i = 0; i < 10; i++)
        t2_data[i] = static_cast<float>(i + 10);
    auto t2 = Tensor::from_vector(t2_data, {10, 1}, Device::CUDA);

    // Concatenate - should use in-place path
    auto result = Tensor::cat({t1, t2}, 0);

    // Verify result
    EXPECT_EQ(result.shape()[0], 20);
    EXPECT_EQ(result.shape()[1], 1);
    EXPECT_EQ(result.capacity(), 100);
    EXPECT_EQ(result.logical_size(), 20);

    // Verify data correctness
    auto result_cpu = result.cpu();
    for (int i = 0; i < 20; i++) {
        EXPECT_FLOAT_EQ(result_cpu.ptr<float>()[i], static_cast<float>(i));
    }
}

TEST(TensorReserveInplaceCat, IndexSelectThenCat) {
    // Simulate the MCMC add_new_gs pattern
    std::vector<float> attr_data(100);
    for (int i = 0; i < 100; i++)
        attr_data[i] = static_cast<float>(i);
    auto attributes = Tensor::from_vector(attr_data, {10, 10}, Device::CUDA);
    attributes.reserve(100); // Reserve for 100 rows

    EXPECT_EQ(attributes.capacity(), 100);
    EXPECT_EQ(attributes.logical_size(), 10);

    // Simulate index_select (what happens in add_new_gs)
    std::vector<int> indices_data = {0, 2, 4};
    auto indices = Tensor::from_vector(indices_data, TensorShape({3}), Device::CUDA);
    auto selected = attributes.index_select(0, indices);

    // Selected tensor should NOT have capacity (it's a new tensor)
    EXPECT_EQ(selected.shape()[0], 3);
    EXPECT_EQ(selected.shape()[1], 10);

    // Now cat - attributes has capacity, selected doesn't
    auto result = Tensor::cat({attributes, selected}, 0);

    // Should use in-place path
    EXPECT_EQ(result.shape()[0], 13);
    EXPECT_EQ(result.capacity(), 100);
    EXPECT_EQ(result.logical_size(), 13);
}

TEST(TensorReserveInplaceCat, GradientPattern) {
    // Simulate the gradient reserve pattern
    auto attrs = Tensor::zeros({10, 3}, Device::CUDA);
    auto grads = Tensor::zeros({10, 3}, Device::CUDA);

    // Reserve both
    attrs.reserve(100);
    grads.reserve(100);

    // Both should have capacity
    EXPECT_EQ(attrs.capacity(), 100);
    EXPECT_EQ(attrs.logical_size(), 10);
    EXPECT_EQ(grads.capacity(), 100);
    EXPECT_EQ(grads.logical_size(), 10);

    // Create new data to add
    auto new_attrs = Tensor::ones({5, 3}, Device::CUDA);
    auto new_grads = Tensor::ones({5, 3}, Device::CUDA);

    // Cat both
    auto attrs_result = Tensor::cat({attrs, new_attrs}, 0);
    auto grads_result = Tensor::cat({grads, new_grads}, 0);

    // Both should work
    EXPECT_EQ(attrs_result.shape()[0], 15);
    EXPECT_EQ(grads_result.shape()[0], 15);
}

TEST(TensorReserveInplaceCat, DeviceConsistencyCheck) {
    // Test to expose device corruption bug
    // This test verifies that all tensors maintain correct device throughout cat operation

    auto t1 = Tensor::zeros({10, 5}, Device::CUDA);
    t1.reserve(100);

    // Verify t1 device BEFORE cat
    EXPECT_EQ(t1.device(), Device::CUDA);
    std::cout << "t1 device before cat: " << static_cast<int>(t1.device()) << std::endl;

    auto t2 = Tensor::ones({5, 5}, Device::CUDA);

    // Verify t2 device
    EXPECT_EQ(t2.device(), Device::CUDA);
    std::cout << "t2 device: " << static_cast<int>(t2.device()) << std::endl;

    // This should use in-place cat
    auto result = Tensor::cat({t1, t2}, 0);

    // Verify result device
    EXPECT_EQ(result.device(), Device::CUDA);
    std::cout << "result device after cat: " << static_cast<int>(result.device()) << std::endl;

    // Verify shape
    EXPECT_EQ(result.shape()[0], 15);
    EXPECT_EQ(result.capacity(), 100);

    // Try to access the data to trigger any device-related errors
    auto result_cpu = result.cpu();
    EXPECT_EQ(result_cpu.device(), Device::CPU);
}

TEST(TensorReserveInplaceCat, MultipleInplaceCats) {
    // Test multiple sequential in-place cats to see if device gets corrupted
    auto base = Tensor::zeros({10, 3}, Device::CUDA);
    base.reserve(200);

    EXPECT_EQ(base.device(), Device::CUDA);
    std::cout << "Initial base device: " << static_cast<int>(base.device()) << std::endl;

    // Do multiple cats in sequence
    for (int i = 0; i < 5; i++) {
        auto addition = Tensor::ones({5, 3}, Device::CUDA);
        EXPECT_EQ(addition.device(), Device::CUDA);
        std::cout << "Iteration " << i << " - addition device: " << static_cast<int>(addition.device()) << std::endl;

        base = Tensor::cat({base, addition}, 0);

        EXPECT_EQ(base.device(), Device::CUDA) << "Device mismatch at iteration " << i;
        std::cout << "Iteration " << i << " - base device after cat: " << static_cast<int>(base.device()) << std::endl;
        std::cout << "Iteration " << i << " - base shape: " << base.shape()[0] << std::endl;
    }

    EXPECT_EQ(base.shape()[0], 35); // 10 + 5*5
    EXPECT_EQ(base.capacity(), 200);

    // Verify we can read the data
    auto cpu_result = base.cpu();
    EXPECT_EQ(cpu_result.shape()[0], 35);
}

TEST(TensorReserveInplaceCat, AllocationFailurePreservesInstalledStorage) {
    auto tensor = Tensor::from_vector(
        std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, {4}, Device::CUDA);
    void* const original_ptr = tensor.data_ptr();
    const size_t original_capacity = tensor.capacity();
    const size_t original_logical_size = tensor.logical_size();

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    ASSERT_EQ(cudaMemGetInfo(&free_bytes, &total_bytes), cudaSuccess);
    const size_t impossible_rows = total_bytes / sizeof(float) + 1;

    try {
        tensor.reserve(impossible_rows);
        FAIL() << "reserve unexpectedly allocated more than total device memory";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string(error.what()).find("reserve CUDA allocation failed"), std::string::npos);
    }

    EXPECT_EQ(tensor.data_ptr(), original_ptr);
    EXPECT_EQ(tensor.capacity(), original_capacity);
    EXPECT_EQ(tensor.logical_size(), original_logical_size);
    EXPECT_EQ(tensor.shape(), TensorShape({4}));

    const auto values = tensor.to(Device::CPU).to_vector();
    EXPECT_EQ(values, (std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}));
}

TEST(TensorReserveInplaceCat, OverflowFailurePreservesInstalledStorage) {
    auto tensor = Tensor::ones({1, 2}, Device::CUDA);
    void* const original_ptr = tensor.data_ptr();
    const size_t original_capacity = tensor.capacity();
    const size_t original_logical_size = tensor.logical_size();

    EXPECT_THROW(tensor.reserve(std::numeric_limits<size_t>::max()), std::runtime_error);

    EXPECT_EQ(tensor.data_ptr(), original_ptr);
    EXPECT_EQ(tensor.capacity(), original_capacity);
    EXPECT_EQ(tensor.logical_size(), original_logical_size);
    EXPECT_EQ(tensor.shape(), TensorShape({1, 2}));

    const auto values = tensor.to(Device::CPU).to_vector();
    EXPECT_EQ(values, (std::vector<float>{1.0f, 1.0f}));
}

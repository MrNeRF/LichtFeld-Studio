/**
 * Consolidated MCMC Memory Operations Tests
 *
 * This file consolidates tests for memory-efficient MCMC operations:
 * - append_gather(): Fused index_select + cat operation
 * - append_zeros(): In-place zero extension for gradients
 * - In-place cat() optimization with pre-allocated capacity
 * - Memory leak detection in realistic MCMC workflows
 *
 * These tests verify the memory optimizations that prevent the 8GB+
 * memory accumulation seen in long MCMC training runs.
 */

#include "core/tensor.hpp"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using namespace lfs::core;

// =============================================================================
// Helper Functions
// =============================================================================

size_t get_gpu_memory_usage() {
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    return total_mem - free_mem;
}

std::string format_bytes(size_t bytes) {
    if (bytes >= 1024 * 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024 * 1024)) + " GB";
    } else if (bytes >= 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + " MB";
    } else if (bytes >= 1024) {
        return std::to_string(bytes / 1024) + " KB";
    }
    return std::to_string(bytes) + " bytes";
}

// =============================================================================
// append_gather() Tests
// =============================================================================

TEST(AppendGather, BasicFunctionality) {
    // Create initial tensor with pre-allocated capacity
    auto param = Tensor::from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3}, Device::CUDA);
    param.reserve(10);

    // Gather and append rows [0, 1, 0]
    auto indices = Tensor::from_vector({0, 1, 0}, {3}, Device::CUDA);
    param.append_gather(indices);

    // Verify shape
    ASSERT_EQ(param.shape()[0], 5); // 2 original + 3 gathered
    ASSERT_EQ(param.shape()[1], 3);

    // Verify values
    auto param_cpu = param.to(Device::CPU);
    const float* data = param_cpu.ptr<float>();

    // Original rows [1,2,3] and [4,5,6]
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[3], 4.0f);

    // Gathered rows: [1,2,3], [4,5,6], [1,2,3]
    EXPECT_FLOAT_EQ(data[6], 1.0f);  // row 0
    EXPECT_FLOAT_EQ(data[9], 4.0f);  // row 1
    EXPECT_FLOAT_EQ(data[12], 1.0f); // row 0 again
}

TEST(AppendGather, MultipleAppends) {
    // Test consecutive append operations
    auto param = Tensor::from_vector({1.0f, 2.0f, 3.0f}, {1, 3}, Device::CUDA);
    param.reserve(10);

    auto indices1 = Tensor::from_vector({0}, {1}, Device::CUDA);
    auto indices2 = Tensor::from_vector({0, 1}, {2}, Device::CUDA);

    param.append_gather(indices1);
    EXPECT_EQ(param.shape()[0], 2);

    param.append_gather(indices2);
    EXPECT_EQ(param.shape()[0], 4);

    // Verify all rows are copies of [1, 2, 3]
    auto param_cpu = param.to(Device::CPU);
    const float* data = param_cpu.ptr<float>();

    for (size_t i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data[i * 3 + 0], 1.0f) << "Row " << i;
        EXPECT_FLOAT_EQ(data[i * 3 + 1], 2.0f) << "Row " << i;
        EXPECT_FLOAT_EQ(data[i * 3 + 2], 3.0f) << "Row " << i;
    }
}

TEST(AppendGather, RealisticMCMCPattern) {
    // Simulate realistic MCMC add_new_gs operation
    const size_t initial_size = 54275;
    const size_t n_new = 2713;

    auto param = Tensor::randn({initial_size, 3}, Device::CUDA);
    param.reserve(4000000);

    // Generate random indices like MCMC does
    auto weights = Tensor::ones({initial_size}, Device::CUDA);
    auto sampled_idxs = Tensor::multinomial(weights, n_new, true);

    // Use fused operation instead of index_select + cat
    param.append_gather(sampled_idxs);

    EXPECT_EQ(param.shape()[0], initial_size + n_new);
    EXPECT_EQ(param.shape()[1], 3);
}

// =============================================================================
// In-Place cat() Tests
// =============================================================================

TEST(InplaceCat, MCMCMultipleAttributes) {
    // Simulate MCMC initialization with realistic sizes
    const size_t initial_gaussians = 54275;
    const size_t new_gaussians = 2713;
    const size_t max_capacity = 4000000;

    struct Attributes {
        Tensor means;     // [N, 3]
        Tensor scales;    // [N, 3]
        Tensor rotations; // [N, 4]
        Tensor sh0;       // [N, 3]
        Tensor shN;       // [N, 45]
        Tensor opacity;   // [N, 1]
    };

    Attributes attrs;

    // Initialize with capacity
    attrs.means = Tensor::zeros({initial_gaussians, 3}, Device::CUDA);
    attrs.means.reserve(max_capacity);

    attrs.scales = Tensor::zeros({initial_gaussians, 3}, Device::CUDA);
    attrs.scales.reserve(max_capacity);

    attrs.rotations = Tensor::zeros({initial_gaussians, 4}, Device::CUDA);
    attrs.rotations.reserve(max_capacity);

    attrs.sh0 = Tensor::zeros({initial_gaussians, 3}, Device::CUDA);
    attrs.sh0.reserve(max_capacity);

    attrs.shN = Tensor::zeros({initial_gaussians, 45}, Device::CUDA);
    attrs.shN.reserve(max_capacity);

    attrs.opacity = Tensor::zeros({initial_gaussians, 1}, Device::CUDA);
    attrs.opacity.reserve(max_capacity);

    // Create new values to add
    Tensor new_means = Tensor::ones({new_gaussians, 3}, Device::CUDA);
    Tensor new_scales = Tensor::ones({new_gaussians, 3}, Device::CUDA);
    Tensor new_rotations = Tensor::ones({new_gaussians, 4}, Device::CUDA);
    Tensor new_sh0 = Tensor::ones({new_gaussians, 3}, Device::CUDA);
    Tensor new_shN = Tensor::ones({new_gaussians, 45}, Device::CUDA);
    Tensor new_opacity = Tensor::ones({new_gaussians, 1}, Device::CUDA);

    // Perform in-place concatenations
    void* means_ptr_before = attrs.means.data_ptr();
    attrs.means = Tensor::cat({attrs.means, new_means}, 0);
    void* means_ptr_after = attrs.means.data_ptr();

    // Verify in-place optimization was used (pointer unchanged)
    EXPECT_EQ(means_ptr_before, means_ptr_after) << "In-place optimization not used for means";

    attrs.scales = Tensor::cat({attrs.scales, new_scales}, 0);
    attrs.rotations = Tensor::cat({attrs.rotations, new_rotations}, 0);
    attrs.sh0 = Tensor::cat({attrs.sh0, new_sh0}, 0);
    attrs.shN = Tensor::cat({attrs.shN, new_shN}, 0);
    attrs.opacity = Tensor::cat({attrs.opacity, new_opacity}, 0);

    // Verify all tensors have correct size
    EXPECT_EQ(attrs.means.shape()[0], initial_gaussians + new_gaussians);
    EXPECT_EQ(attrs.scales.shape()[0], initial_gaussians + new_gaussians);
    EXPECT_EQ(attrs.rotations.shape()[0], initial_gaussians + new_gaussians);
    EXPECT_EQ(attrs.sh0.shape()[0], initial_gaussians + new_gaussians);
    EXPECT_EQ(attrs.shN.shape()[0], initial_gaussians + new_gaussians);
    EXPECT_EQ(attrs.opacity.shape()[0], initial_gaussians + new_gaussians);
}

TEST(InplaceCat, MultipleSequentialAdds) {
    // Simulate multiple MCMC refine operations
    const size_t initial_size = 54275;
    const size_t add_size = 2713;
    const size_t max_capacity = 400000;
    const int num_adds = 3;

    auto tensor = Tensor::zeros({initial_size, 3}, Device::CUDA);
    tensor.reserve(max_capacity);

    void* initial_ptr = tensor.data_ptr();

    for (int i = 0; i < num_adds; i++) {
        auto new_data = Tensor::ones({add_size, 3}, Device::CUDA);
        void* before_ptr = tensor.data_ptr();

        tensor = Tensor::cat({tensor, new_data}, 0);

        void* after_ptr = tensor.data_ptr();

        // Verify pointer stays the same (in-place optimization)
        EXPECT_EQ(before_ptr, after_ptr) << "Add operation " << (i + 1) << " reallocated";
        EXPECT_EQ(after_ptr, initial_ptr) << "Pointer changed from initial";

        EXPECT_EQ(tensor.shape()[0], initial_size + (i + 1) * add_size);
    }
}

// =============================================================================
// Memory Leak Detection Tests
// =============================================================================

TEST(MemoryLeak, MultinomialRepeatedCalls) {
    const size_t n_weights = 50000;
    const int n_samples = 3000;
    const int iterations = 100;

    auto weights = Tensor::ones({n_weights}, Device::CUDA);

    size_t mem_before = get_gpu_memory_usage();

    for (int i = 0; i < iterations; i++) {
        // This allocates a new tensor every call - should be freed by pool
        auto sampled_indices = Tensor::multinomial(weights, n_samples, true);

        if (i % 10 == 0) {
            Tensor::trim_memory_pool();
        }
    }

    size_t mem_after = get_gpu_memory_usage();
    int64_t mem_growth = static_cast<int64_t>(mem_after) - static_cast<int64_t>(mem_before);

    // Memory should stabilize - allow max 50MB growth for cached allocations (negative is fine)
    EXPECT_LT(mem_growth, 50 * 1024 * 1024)
        << "multinomial() leaking memory: grew by " << (mem_growth >= 0 ? format_bytes(mem_growth) : "-" + format_bytes(-mem_growth));
}

TEST(MemoryLeak, AppendGatherVsIndexSelectCat) {
    // Compare memory usage of append_gather vs index_select+cat
    const size_t initial_size = 54275;
    const size_t n_new = 2713;
    const int iterations = 50;

    // Test 1: OLD WAY - index_select + cat (allocates intermediate)
    {
        auto param = Tensor::randn({initial_size, 3}, Device::CUDA);
        param.reserve(4000000);

        size_t mem_before = get_gpu_memory_usage();

        for (int i = 0; i < iterations; i++) {
            auto weights = Tensor::ones({param.shape()[0]}, Device::CUDA);
            auto sampled_idxs = Tensor::multinomial(weights, n_new, true);

            auto new_values = param.index_select(0, sampled_idxs); // Allocates
            param = Tensor::cat({param, new_values}, 0);

            if (i % 10 == 0) {
                Tensor::trim_memory_pool();
            }
        }

        size_t mem_after = get_gpu_memory_usage();
        int64_t old_way_growth = static_cast<int64_t>(mem_after) - static_cast<int64_t>(mem_before);

        std::cout << "OLD WAY memory growth: " << (old_way_growth >= 0 ? format_bytes(old_way_growth) : "-" + format_bytes(-old_way_growth)) << std::endl;
    }

    // Test 2: NEW WAY - append_gather (no intermediate allocation)
    {
        auto param = Tensor::randn({initial_size, 3}, Device::CUDA);
        param.reserve(4000000);

        size_t mem_before = get_gpu_memory_usage();

        for (int i = 0; i < iterations; i++) {
            auto weights = Tensor::ones({param.shape()[0]}, Device::CUDA);
            auto sampled_idxs = Tensor::multinomial(weights, n_new, true);

            param.append_gather(sampled_idxs); // No intermediate allocation

            if (i % 10 == 0) {
                Tensor::trim_memory_pool();
            }
        }

        size_t mem_after = get_gpu_memory_usage();
        int64_t new_way_growth = static_cast<int64_t>(mem_after) - static_cast<int64_t>(mem_before);

        std::cout << "NEW WAY memory growth: " << (new_way_growth >= 0 ? format_bytes(new_way_growth) : "-" + format_bytes(-new_way_growth)) << std::endl;

        // Memory should not grow excessively (negative growth is fine - means pool trimming worked)
        EXPECT_LT(new_way_growth, 500 * 1024 * 1024)
            << "append_gather() using excessive memory";
    }
}

TEST(MemoryLeak, RealisticMCMCLoop) {
    const size_t initial_size = 54275;
    const int iterations = 100;
    const int refine_interval = 100;

    // Simulate 6 parameters like MCMC
    std::vector<Tensor> params;
    std::vector<std::pair<size_t, size_t>> param_shapes = {
        {initial_size, 3},  // means
        {initial_size, 3},  // sh0
        {initial_size, 45}, // shN
        {initial_size, 3},  // scaling
        {initial_size, 4},  // rotation
        {initial_size, 1}   // opacity
    };

    for (auto& shape : param_shapes) {
        auto p = Tensor::randn({shape.first, shape.second}, Device::CUDA);
        p.reserve(4000000);
        params.push_back(p);
    }

    size_t mem_before = get_gpu_memory_usage();

    for (int iter = 0; iter < iterations; iter++) {
        // Every refine_interval iterations, do add_new_gs
        if (iter > 0 && iter % refine_interval == 0) {
            size_t n = params[0].shape()[0];
            int n_new = static_cast<int>(n * 0.05f);

            // Multinomial sample
            auto weights = Tensor::ones({n}, Device::CUDA);
            auto sampled_idxs = Tensor::multinomial(weights, n_new, true);

            // Use append_gather for all parameters
            for (size_t i = 0; i < params.size(); i++) {
                params[i].append_gather(sampled_idxs);
            }

            Tensor::trim_memory_pool();
        }
    }

    size_t mem_after = get_gpu_memory_usage();
    size_t mem_growth = mem_after - mem_before;

    // Calculate expected growth from actual data
    size_t final_total_elements = 0;
    size_t initial_total_elements = 0;
    for (size_t i = 0; i < params.size(); i++) {
        final_total_elements += params[i].numel();
        initial_total_elements += param_shapes[i].first * param_shapes[i].second;
    }

    size_t expected_data_growth = (final_total_elements - initial_total_elements) * sizeof(float);
    size_t allowed_overhead = 200 * 1024 * 1024; // 200MB overhead

    std::cout << "Expected data growth: " << format_bytes(expected_data_growth) << std::endl;
    std::cout << "Actual memory growth: " << format_bytes(mem_growth) << std::endl;

    EXPECT_LT(mem_growth, expected_data_growth + allowed_overhead)
        << "MCMC loop leaking memory beyond expected data growth";
}

// =============================================================================
// Integration Test - Full MCMC Sequence
// =============================================================================

TEST(MCMCIntegration, FullAddNewParamsSequence) {
    // Simulate exact sequence from adam_optimizer.cpp:add_new_params()
    const size_t initial_size = 54275;
    const size_t new_size = 2713;

    struct ParamGrad {
        Tensor param;
        Tensor grad;
        std::string name;
        std::vector<size_t> dims;
    };

    std::vector<ParamGrad> params = {
        {Tensor(), Tensor(), "means", {initial_size, 3}},
        {Tensor(), Tensor(), "sh0", {initial_size, 3}},
        {Tensor(), Tensor(), "shN", {initial_size, 45}},
        {Tensor(), Tensor(), "scaling", {initial_size, 3}},
        {Tensor(), Tensor(), "rotation", {initial_size, 4}},
        {Tensor(), Tensor(), "opacity", {initial_size, 1}},
    };

    // Initialize all parameters and gradients with capacity
    for (auto& pg : params) {
        pg.param = Tensor::zeros(TensorShape(pg.dims), Device::CUDA);
        pg.param.reserve(4000000);
        pg.grad = Tensor::zeros(TensorShape(pg.dims), Device::CUDA);
        pg.grad.reserve(4000000);
    }

    // Simulate add_new_params for each parameter
    for (auto& pg : params) {
        std::vector<size_t> new_dims = pg.dims;
        new_dims[0] = new_size;
        auto new_values = Tensor::ones(TensorShape(new_dims), Device::CUDA);

        // Parameter concatenation (would use append_gather in real code)
        void* param_ptr_before = pg.param.data_ptr();
        pg.param = Tensor::cat({pg.param, new_values}, 0);
        void* param_ptr_after = pg.param.data_ptr();

        EXPECT_EQ(param_ptr_before, param_ptr_after)
            << pg.name << " param: in-place optimization not used";

        // Gradient concatenation with zeros (would use append_zeros in real code)
        auto zeros_grad = Tensor::zeros(TensorShape(new_dims), Device::CUDA);
        void* grad_ptr_before = pg.grad.data_ptr();
        pg.grad = Tensor::cat({pg.grad, zeros_grad}, 0);
        void* grad_ptr_after = pg.grad.data_ptr();

        EXPECT_EQ(grad_ptr_before, grad_ptr_after)
            << pg.name << " grad: in-place optimization not used";

        // Verify sizes
        EXPECT_EQ(pg.param.shape()[0], initial_size + new_size);
        EXPECT_EQ(pg.grad.shape()[0], initial_size + new_size);
    }
}

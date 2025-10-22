/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <cuda_runtime.h>
#include <expected>
#include <gtest/gtest.h>
#include <memory>
#include <print>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "core_new/tensor.hpp"

namespace {

    constexpr float FLOAT_TOLERANCE = 1e-5f;

    /**
     * @brief Convert torch::Tensor to lfs::core::Tensor
     */
    lfs::core::Tensor torch_to_tensor(const torch::Tensor& torch_tensor) {
        auto cpu_tensor = torch_tensor.cpu().contiguous();
        std::vector<size_t> shape;
        for (int i = 0; i < torch_tensor.dim(); ++i) {
            shape.push_back(torch_tensor.size(i));
        }

        std::vector<float> data(cpu_tensor.data_ptr<float>(),
                                cpu_tensor.data_ptr<float>() + cpu_tensor.numel());
        return lfs::core::Tensor::from_vector(data, lfs::core::TensorShape(shape), lfs::core::Device::CUDA);
    }

    /**
     * @brief Convert lfs::core::Tensor to torch::Tensor
     */
    torch::Tensor tensor_to_torch(const lfs::core::Tensor& gs_tensor) {
        auto cpu_tensor = gs_tensor.cpu();
        std::vector<int64_t> shape;
        for (size_t i = 0; i < cpu_tensor.ndim(); ++i) {
            shape.push_back(cpu_tensor.shape()[i]);
        }

        auto data = cpu_tensor.to_vector();
        auto torch_tensor = torch::from_blob(data.data(), shape, torch::kFloat32).clone();
        return torch_tensor.cuda();
    }

    /**
     * @brief Compare two tensors with tolerance
     */
    bool tensors_equal(const torch::Tensor& t1, const torch::Tensor& t2, float tol = FLOAT_TOLERANCE) {
        if (t1.sizes() != t2.sizes())
            return false;
        auto diff = (t1 - t2).abs().max().item<float>();
        return diff < tol;
    }

    bool tensors_equal(const lfs::core::Tensor& t1, const lfs::core::Tensor& t2, float tol = FLOAT_TOLERANCE) {
        if (t1.shape() != t2.shape())
            return false;
        auto diff = t1.sub(t2).abs().max_scalar();
        return diff < tol;
    }

    bool tensors_equal(const torch::Tensor& torch_t, const lfs::core::Tensor& gs_t, float tol = FLOAT_TOLERANCE) {
        return tensors_equal(torch_t, tensor_to_torch(gs_t), tol);
    }

} // anonymous namespace

// ============================================================================
// Test Fixture
// ============================================================================

class TensorMoveTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(torch::cuda::is_available()) << "CUDA not available";
        torch::manual_seed(42);
        torch::cuda::manual_seed(42);
        lfs::core::Tensor::manual_seed(42);
    }

    void print_test_header(const std::string& test_name) {
        std::println("\n=== {} ===", test_name);
    }
};

// ============================================================================
// Basic Move Constructor Tests
// ============================================================================

TEST_F(TensorMoveTest, BasicMoveConstructor) {
    print_test_header("BasicMoveConstructor");

    // Torch version
    auto torch_orig = torch::randn({100, 3}, torch::kCUDA);
    auto torch_orig_data = torch_orig.data_ptr<float>();
    auto torch_moved = std::move(torch_orig);

    EXPECT_FALSE(torch_orig.defined()) << "Torch: source should be undefined after move";
    EXPECT_TRUE(torch_moved.defined()) << "Torch: destination should be defined";
    EXPECT_EQ(torch_moved.data_ptr<float>(), torch_orig_data) << "Torch: data pointer should match";

    // Our Tensor version
    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_orig_ptr = gs_orig.ptr<float>();
    auto gs_moved = std::move(gs_orig);

    EXPECT_FALSE(gs_orig.is_valid()) << "Tensor: source should be invalid after move";
    EXPECT_TRUE(gs_moved.is_valid()) << "Tensor: destination should be valid";
    EXPECT_EQ(gs_moved.ptr<float>(), gs_orig_ptr) << "Tensor: data pointer should match";

    std::println("✓ BasicMoveConstructor: Both implementations behave correctly");
}

TEST_F(TensorMoveTest, MoveAssignment) {
    print_test_header("MoveAssignment");

    // Torch version
    auto torch_orig = torch::randn({100, 3}, torch::kCUDA);
    auto torch_orig_data = torch_orig.data_ptr<float>();
    torch::Tensor torch_dest;
    torch_dest = std::move(torch_orig);

    EXPECT_FALSE(torch_orig.defined()) << "Torch: source should be undefined after move assignment";
    EXPECT_TRUE(torch_dest.defined()) << "Torch: destination should be defined";
    EXPECT_EQ(torch_dest.data_ptr<float>(), torch_orig_data) << "Torch: data pointer should match";

    // Our Tensor version
    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_orig_ptr = gs_orig.ptr<float>();
    lfs::core::Tensor gs_dest;
    gs_dest = std::move(gs_orig);

    EXPECT_FALSE(gs_orig.is_valid()) << "Tensor: source should be invalid after move assignment";
    EXPECT_TRUE(gs_dest.is_valid()) << "Tensor: destination should be valid";
    EXPECT_EQ(gs_dest.ptr<float>(), gs_orig_ptr) << "Tensor: data pointer should match";

    std::println("✓ MoveAssignment: Both implementations behave correctly");
}

TEST_F(TensorMoveTest, MoveToExistingTensor) {
    print_test_header("MoveToExistingTensor");

    // Torch version
    auto torch_existing = torch::ones({50, 2}, torch::kCUDA);
    auto torch_orig = torch::randn({100, 3}, torch::kCUDA);
    auto torch_orig_data = torch_orig.data_ptr<float>();
    torch_existing = std::move(torch_orig);

    EXPECT_TRUE(torch_existing.defined());
    EXPECT_EQ(torch_existing.sizes(), torch::IntArrayRef({100, 3}));
    EXPECT_EQ(torch_existing.data_ptr<float>(), torch_orig_data);

    // Our Tensor version
    auto gs_existing = lfs::core::Tensor::ones({50, 2}, lfs::core::Device::CUDA);
    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_orig_ptr = gs_orig.ptr<float>();
    gs_existing = std::move(gs_orig);

    EXPECT_TRUE(gs_existing.is_valid());
    EXPECT_EQ(gs_existing.shape()[0], 100);
    EXPECT_EQ(gs_existing.shape()[1], 3);
    EXPECT_EQ(gs_existing.ptr<float>(), gs_orig_ptr);

    std::println("✓ MoveToExistingTensor: Both implementations handle reassignment");
}

// ============================================================================
// Data Preservation Tests
// ============================================================================

TEST_F(TensorMoveTest, DataPreservationAfterMove) {
    print_test_header("DataPreservationAfterMove");

    // Torch version
    auto torch_orig = torch::arange(0, 100, torch::kCUDA).reshape({10, 10});
    auto torch_copy = torch_orig.clone(); // Save a copy for comparison
    auto torch_moved = std::move(torch_orig);

    EXPECT_TRUE(tensors_equal(torch_moved, torch_copy));

    // Our Tensor version
    auto gs_orig = lfs::core::Tensor::arange(0, 100).reshape({10, 10});
    auto gs_copy = gs_orig.clone();
    auto gs_moved = std::move(gs_orig);

    EXPECT_TRUE(tensors_equal(gs_moved, gs_copy));

    std::println("✓ DataPreservationAfterMove: Data preserved correctly");
}

TEST_F(TensorMoveTest, MultipleSequentialMoves) {
    print_test_header("MultipleSequentialMoves");

    // Torch version
    auto torch_orig = torch::randn({100, 3}, torch::kCUDA);
    auto torch_copy = torch_orig.clone();
    auto torch_1 = std::move(torch_orig);
    auto torch_2 = std::move(torch_1);
    auto torch_3 = std::move(torch_2);

    EXPECT_TRUE(tensors_equal(torch_3, torch_copy));
    EXPECT_FALSE(torch_orig.defined());
    EXPECT_FALSE(torch_1.defined());
    EXPECT_FALSE(torch_2.defined());

    // Our Tensor version
    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_copy = gs_orig.clone();
    auto gs_1 = std::move(gs_orig);
    auto gs_2 = std::move(gs_1);
    auto gs_3 = std::move(gs_2);

    EXPECT_TRUE(tensors_equal(gs_3, gs_copy));
    EXPECT_FALSE(gs_orig.is_valid());
    EXPECT_FALSE(gs_1.is_valid());
    EXPECT_FALSE(gs_2.is_valid());

    std::println("✓ MultipleSequentialMoves: Chain of moves works correctly");
}

// ============================================================================
// Move with Different Shapes/Sizes
// ============================================================================

TEST_F(TensorMoveTest, MoveScalarTensor) {
    print_test_header("MoveScalarTensor");

    // Torch version
    auto torch_scalar = torch::tensor(42.0f, torch::kCUDA);
    auto torch_moved = std::move(torch_scalar);

    EXPECT_TRUE(torch_moved.defined());
    EXPECT_EQ(torch_moved.numel(), 1);
    EXPECT_FLOAT_EQ(torch_moved.item<float>(), 42.0f);

    // Our Tensor version
    auto gs_scalar = lfs::core::Tensor::full({1}, 42.0f, lfs::core::Device::CUDA);
    auto gs_moved = std::move(gs_scalar);

    EXPECT_TRUE(gs_moved.is_valid());
    EXPECT_EQ(gs_moved.numel(), 1);
    EXPECT_FLOAT_EQ(gs_moved.item(), 42.0f);

    std::println("✓ MoveScalarTensor: Scalar tensors move correctly");
}

TEST_F(TensorMoveTest, MoveLargeTensor) {
    print_test_header("MoveLargeTensor");

    const size_t large_size = 10000;

    // Torch version
    auto torch_large = torch::randn({large_size, large_size}, torch::kCUDA);
    auto torch_ptr = torch_large.data_ptr<float>();
    auto torch_moved = std::move(torch_large);

    EXPECT_EQ(torch_moved.data_ptr<float>(), torch_ptr);
    EXPECT_EQ(torch_moved.numel(), large_size * large_size);

    // Our Tensor version
    auto gs_large = lfs::core::Tensor::randn({large_size, large_size}, lfs::core::Device::CUDA);
    auto gs_ptr = gs_large.ptr<float>();
    auto gs_moved = std::move(gs_large);

    EXPECT_EQ(gs_moved.ptr<float>(), gs_ptr);
    EXPECT_EQ(gs_moved.numel(), large_size * large_size);

    std::println("✓ MoveLargeTensor: Large tensors move correctly");
}

TEST_F(TensorMoveTest, MoveHighDimensionalTensor) {
    print_test_header("MoveHighDimensionalTensor");

    // Torch version
    auto torch_nd = torch::randn({5, 4, 3, 2}, torch::kCUDA);
    auto torch_shape = torch_nd.sizes();
    auto torch_moved = std::move(torch_nd);

    EXPECT_EQ(torch_moved.sizes(), torch_shape);
    EXPECT_EQ(torch_moved.dim(), 4);

    // Our Tensor version
    auto gs_nd = lfs::core::Tensor::randn({5, 4, 3, 2}, lfs::core::Device::CUDA);
    auto gs_shape = gs_nd.shape();
    auto gs_moved = std::move(gs_nd);

    EXPECT_EQ(gs_moved.shape(), gs_shape);
    EXPECT_EQ(gs_moved.ndim(), 4);

    std::println("✓ MoveHighDimensionalTensor: Multi-dimensional tensors move correctly");
}

// ============================================================================
// Move in Containers
// ============================================================================

TEST_F(TensorMoveTest, MoveIntoVector) {
    print_test_header("MoveIntoVector");

    // Torch version
    std::vector<torch::Tensor> torch_vec;
    auto torch_orig = torch::randn({100, 3}, torch::kCUDA);
    auto torch_ptr = torch_orig.data_ptr<float>();
    torch_vec.push_back(std::move(torch_orig));

    EXPECT_FALSE(torch_orig.defined());
    EXPECT_TRUE(torch_vec[0].defined());
    EXPECT_EQ(torch_vec[0].data_ptr<float>(), torch_ptr);

    // Our Tensor version
    std::vector<lfs::core::Tensor> gs_vec;
    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_ptr = gs_orig.ptr<float>();
    gs_vec.push_back(std::move(gs_orig));

    EXPECT_FALSE(gs_orig.is_valid());
    EXPECT_TRUE(gs_vec[0].is_valid());
    EXPECT_EQ(gs_vec[0].ptr<float>(), gs_ptr);

    std::println("✓ MoveIntoVector: Moving into vector works correctly");
}

TEST_F(TensorMoveTest, MoveFromVector) {
    print_test_header("MoveFromVector");

    // Torch version
    std::vector<torch::Tensor> torch_vec;
    torch_vec.push_back(torch::randn({100, 3}, torch::kCUDA));
    auto torch_ptr = torch_vec[0].data_ptr<float>();
    auto torch_moved = std::move(torch_vec[0]);

    EXPECT_FALSE(torch_vec[0].defined());
    EXPECT_TRUE(torch_moved.defined());
    EXPECT_EQ(torch_moved.data_ptr<float>(), torch_ptr);

    // Our Tensor version
    std::vector<lfs::core::Tensor> gs_vec;
    gs_vec.push_back(lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA));
    auto gs_ptr = gs_vec[0].ptr<float>();
    auto gs_moved = std::move(gs_vec[0]);

    EXPECT_FALSE(gs_vec[0].is_valid());
    EXPECT_TRUE(gs_moved.is_valid());
    EXPECT_EQ(gs_moved.ptr<float>(), gs_ptr);

    std::println("✓ MoveFromVector: Moving from vector works correctly");
}

TEST_F(TensorMoveTest, MoveMultipleIntoVector) {
    print_test_header("MoveMultipleIntoVector");

    const int count = 10;

    // Torch version
    std::vector<torch::Tensor> torch_vec;
    std::vector<float*> torch_ptrs;
    for (int i = 0; i < count; ++i) {
        auto t = torch::randn({100, 3}, torch::kCUDA);
        torch_ptrs.push_back(t.data_ptr<float>());
        torch_vec.push_back(std::move(t));
    }

    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(torch_vec[i].defined());
        EXPECT_EQ(torch_vec[i].data_ptr<float>(), torch_ptrs[i]);
    }

    // Our Tensor version
    std::vector<lfs::core::Tensor> gs_vec;
    std::vector<float*> gs_ptrs;
    for (int i = 0; i < count; ++i) {
        auto t = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
        gs_ptrs.push_back(t.ptr<float>());
        gs_vec.push_back(std::move(t));
    }

    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(gs_vec[i].is_valid());
        EXPECT_EQ(gs_vec[i].ptr<float>(), gs_ptrs[i]);
    }

    std::println("✓ MoveMultipleIntoVector: Multiple moves into vector work correctly");
}

// ============================================================================
// Move with Operations
// ============================================================================

TEST_F(TensorMoveTest, MoveAfterClone) {
    print_test_header("MoveAfterClone");

    // Torch version
    auto torch_orig = torch::randn({100, 3}, torch::kCUDA);
    auto torch_clone = torch_orig.clone();
    auto torch_moved = std::move(torch_clone);

    EXPECT_TRUE(torch_orig.defined());
    EXPECT_FALSE(torch_clone.defined());
    EXPECT_TRUE(torch_moved.defined());
    EXPECT_TRUE(tensors_equal(torch_orig, torch_moved));

    // Our Tensor version
    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_clone = gs_orig.clone();
    auto gs_moved = std::move(gs_clone);

    EXPECT_TRUE(gs_orig.is_valid());
    EXPECT_FALSE(gs_clone.is_valid());
    EXPECT_TRUE(gs_moved.is_valid());
    EXPECT_TRUE(tensors_equal(gs_orig, gs_moved));

    std::println("✓ MoveAfterClone: Moving cloned tensors works correctly");
}

TEST_F(TensorMoveTest, MoveAfterContiguous) {
    print_test_header("MoveAfterContiguous");

    // Torch version
    auto torch_orig = torch::randn({100, 3}, torch::kCUDA);
    auto torch_contig = torch_orig.contiguous();
    auto torch_moved = std::move(torch_contig);

    EXPECT_TRUE(torch_moved.defined());
    EXPECT_TRUE(torch_moved.is_contiguous());

    // Our Tensor version
    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_contig = gs_orig.contiguous();
    auto gs_moved = std::move(gs_contig);

    EXPECT_TRUE(gs_moved.is_valid());
    EXPECT_TRUE(gs_moved.is_contiguous());

    std::println("✓ MoveAfterContiguous: Moving after contiguous() works correctly");
}

TEST_F(TensorMoveTest, MoveAfterReshape) {
    print_test_header("MoveAfterReshape");

    // Torch version
    auto torch_orig = torch::randn({100, 3}, torch::kCUDA);
    auto torch_reshaped = torch_orig.reshape({10, 30});
    auto torch_moved = std::move(torch_reshaped);

    EXPECT_TRUE(torch_moved.defined());
    EXPECT_EQ(torch_moved.sizes(), torch::IntArrayRef({10, 30}));

    // Our Tensor version
    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_reshaped = gs_orig.reshape({10, 30});
    auto gs_moved = std::move(gs_reshaped);

    EXPECT_TRUE(gs_moved.is_valid());
    EXPECT_EQ(gs_moved.shape()[0], 10);
    EXPECT_EQ(gs_moved.shape()[1], 30);

    std::println("✓ MoveAfterReshape: Moving reshaped tensors works correctly");
}

// ============================================================================
// Move in Function Returns
// ============================================================================

torch::Tensor create_torch_tensor() {
    auto t = torch::randn({100, 3}, torch::kCUDA);
    return t; // RVO should apply
}

lfs::core::Tensor create_gs_tensor() {
    auto t = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    return t; // RVO should apply
}

torch::Tensor create_torch_tensor_explicit_move() {
    auto t = torch::randn({100, 3}, torch::kCUDA);
    return std::move(t);
}

lfs::core::Tensor create_gs_tensor_explicit_move() {
    auto t = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    return std::move(t);
}

TEST_F(TensorMoveTest, ReturnByValue_RVO) {
    print_test_header("ReturnByValue_RVO");

    // Torch version
    auto torch_returned = create_torch_tensor();
    EXPECT_TRUE(torch_returned.defined());
    EXPECT_EQ(torch_returned.sizes(), torch::IntArrayRef({100, 3}));

    // Our Tensor version
    auto gs_returned = create_gs_tensor();
    EXPECT_TRUE(gs_returned.is_valid());
    EXPECT_EQ(gs_returned.shape()[0], 100);
    EXPECT_EQ(gs_returned.shape()[1], 3);

    std::println("✓ ReturnByValue_RVO: RVO works correctly");
}

TEST_F(TensorMoveTest, ReturnByValue_ExplicitMove) {
    print_test_header("ReturnByValue_ExplicitMove");

    // Torch version
    auto torch_returned = create_torch_tensor_explicit_move();
    EXPECT_TRUE(torch_returned.defined());
    EXPECT_EQ(torch_returned.sizes(), torch::IntArrayRef({100, 3}));

    // Our Tensor version
    auto gs_returned = create_gs_tensor_explicit_move();
    EXPECT_TRUE(gs_returned.is_valid());
    EXPECT_EQ(gs_returned.shape()[0], 100);
    EXPECT_EQ(gs_returned.shape()[1], 3);

    std::println("✓ ReturnByValue_ExplicitMove: Explicit move in return works correctly");
}

// ============================================================================
// Move with std::expected
// ============================================================================

std::expected<torch::Tensor, std::string> create_torch_expected() {
    auto t = torch::randn({100, 3}, torch::kCUDA);
    return t;
}

std::expected<lfs::core::Tensor, std::string> create_gs_expected() {
    auto t = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    return t;
}

TEST_F(TensorMoveTest, MoveInExpected) {
    print_test_header("MoveInExpected");

    // Torch version
    auto torch_result = create_torch_expected();
    ASSERT_TRUE(torch_result.has_value());
    EXPECT_TRUE(torch_result.value().defined());
    EXPECT_EQ(torch_result.value().sizes(), torch::IntArrayRef({100, 3}));

    // Our Tensor version
    auto gs_result = create_gs_expected();
    ASSERT_TRUE(gs_result.has_value());
    EXPECT_TRUE(gs_result.value().is_valid());
    EXPECT_EQ(gs_result.value().shape()[0], 100);
    EXPECT_EQ(gs_result.value().shape()[1], 3);

    std::println("✓ MoveInExpected: Moving through std::expected works correctly");
}

// ============================================================================
// Move with Cloned Tensors in Constructor
// ============================================================================

struct TorchWrapper {
    torch::Tensor data;

    TorchWrapper(torch::Tensor t) : data(std::move(t)) {}
};

struct TensorWrapper {
    lfs::core::Tensor data;

    TensorWrapper(lfs::core::Tensor t) : data(std::move(t)) {}
};

TEST_F(TensorMoveTest, MoveIntoStructConstructor) {
    print_test_header("MoveIntoStructConstructor");

    // Torch version
    auto torch_orig = torch::randn({100, 3}, torch::kCUDA);
    auto torch_ptr = torch_orig.data_ptr<float>();
    TorchWrapper torch_wrapper(std::move(torch_orig));

    EXPECT_FALSE(torch_orig.defined());
    EXPECT_TRUE(torch_wrapper.data.defined());
    EXPECT_EQ(torch_wrapper.data.data_ptr<float>(), torch_ptr);

    // Our Tensor version
    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_ptr = gs_orig.ptr<float>();
    TensorWrapper gs_wrapper(std::move(gs_orig));

    EXPECT_FALSE(gs_orig.is_valid());
    EXPECT_TRUE(gs_wrapper.data.is_valid());
    EXPECT_EQ(gs_wrapper.data.ptr<float>(), gs_ptr);

    std::println("✓ MoveIntoStructConstructor: Moving into struct constructor works");
}

TEST_F(TensorMoveTest, MoveClonedTensorIntoConstructor) {
    print_test_header("MoveClonedTensorIntoConstructor");

    // Torch version - simulating the SplatData pattern
    auto torch_orig = torch::randn({100, 3}, torch::kCUDA);
    auto torch_copy = torch_orig.clone();
    TorchWrapper torch_wrapper(torch_orig.clone());

    EXPECT_TRUE(torch_wrapper.data.defined());
    EXPECT_TRUE(tensors_equal(torch_wrapper.data, torch_orig));

    // Our Tensor version - simulating the SplatDataNew pattern
    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_copy = gs_orig.clone();
    TensorWrapper gs_wrapper(gs_orig.clone());

    EXPECT_TRUE(gs_wrapper.data.is_valid());
    EXPECT_TRUE(tensors_equal(gs_wrapper.data, gs_orig));

    std::println("✓ MoveClonedTensorIntoConstructor: Clone-then-move pattern works");
}

// ============================================================================
// Move with std::unique_ptr
// ============================================================================

TEST_F(TensorMoveTest, MoveWithUniquePtr) {
    print_test_header("MoveWithUniquePtr");

    // Torch version
    auto torch_ptr = std::make_unique<torch::Tensor>(torch::randn({100, 3}, torch::kCUDA));
    auto torch_data_ptr = torch_ptr->data_ptr<float>();
    auto torch_moved_ptr = std::move(torch_ptr);

    EXPECT_FALSE(torch_ptr);
    EXPECT_TRUE(torch_moved_ptr);
    EXPECT_EQ(torch_moved_ptr->data_ptr<float>(), torch_data_ptr);

    // Our Tensor version
    auto gs_ptr = std::make_unique<lfs::core::Tensor>(lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA));
    auto gs_data_ptr = gs_ptr->ptr<float>();
    auto gs_moved_ptr = std::move(gs_ptr);

    EXPECT_FALSE(gs_ptr);
    EXPECT_TRUE(gs_moved_ptr);
    EXPECT_EQ(gs_moved_ptr->ptr<float>(), gs_data_ptr);

    std::println("✓ MoveWithUniquePtr: Moving with unique_ptr works correctly");
}

// ============================================================================
// NEW TESTS: Edge Cases and Advanced Scenarios
// ============================================================================

TEST_F(TensorMoveTest, MovedFromTensorState) {
    print_test_header("MovedFromTensorState");

    // Test the ACTUAL behavior of moved-from tensor
    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto orig_ptr = gs_orig.ptr<float>();

    auto gs_moved = std::move(gs_orig);

    // After move, destination should have the data
    EXPECT_TRUE(gs_moved.is_valid());
    EXPECT_EQ(gs_moved.numel(), 300);
    EXPECT_EQ(gs_moved.ptr<float>(), orig_ptr);

    // CRITICAL TEST: What is the state of gs_orig after move?
    // According to your implementation, is_valid() checks initialized_ flag
    std::println("  After move - gs_orig.is_valid(): {}", gs_orig.is_valid());
    std::println("  After move - gs_orig.numel(): {}", gs_orig.numel());
    std::println("  After move - gs_orig.ptr<float>(): {}", static_cast<void*>(gs_orig.ptr<float>()));
    std::println("  After move - gs_orig.data_owner_.use_count(): {}",
                 gs_orig.owns_memory() ? "owns" : "doesn't own");

    // The moved-from tensor state depends on implementation
    // If initialized_ is set to false, is_valid() returns false
    // But shape metadata might still be there

    if (!gs_orig.is_valid()) {
        std::println("  ✓ Moved-from tensor is invalid (as expected)");
    } else {
        std::println("  ⚠ Moved-from tensor still reports valid!");
        std::println("    This means is_valid() is NOT checking correctly");
    }

    std::println("✓ MovedFromTensorState: State documented");
}

TEST_F(TensorMoveTest, MoveEmptyTensor) {
    print_test_header("MoveEmptyTensor");

    // Test moving an empty tensor
    auto gs_empty = lfs::core::Tensor::empty({0}, lfs::core::Device::CUDA);
    EXPECT_TRUE(gs_empty.is_valid());
    EXPECT_EQ(gs_empty.numel(), 0);

    auto gs_moved = std::move(gs_empty);
    EXPECT_TRUE(gs_moved.is_valid());
    EXPECT_EQ(gs_moved.numel(), 0);
    EXPECT_FALSE(gs_empty.is_valid());

    std::println("✓ MoveEmptyTensor: Empty tensors move correctly");
}

TEST_F(TensorMoveTest, CRITICAL_MovedFromNumelBug) {
    print_test_header("CRITICAL_MovedFromNumelBug");

    // This test MUST FAIL to expose the bug
    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_moved = std::move(gs_orig);

    std::println("  After move - moved-from tensor:");
    std::println("    is_valid(): {}", gs_orig.is_valid());
    std::println("    numel(): {}", gs_orig.numel());

    // HARD ASSERT: If tensor is invalid, numel MUST be 0
    ASSERT_FALSE(gs_orig.is_valid()) << "Moved-from tensor must be invalid";
    ASSERT_EQ(gs_orig.numel(), 0)
        << "BUG EXPOSED: Moved-from tensor is invalid but reports numel()="
        << gs_orig.numel() << " instead of 0!";

    std::println("✓ CRITICAL_MovedFromNumelBug: Bug test passed");
}

TEST_F(TensorMoveTest, CRITICAL_MovedFromShapeBug) {
    print_test_header("CRITICAL_MovedFromShapeBug");

    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_moved = std::move(gs_orig);

    std::println("  After move - moved-from tensor:");
    std::println("    is_valid(): {}", gs_orig.is_valid());
    std::println("    ndim(): {}", gs_orig.ndim());
    std::println("    shape()[0]: {}", gs_orig.shape()[0]);

    // HARD ASSERT: Invalid tensor should have no dimensions
    ASSERT_FALSE(gs_orig.is_valid());
    ASSERT_EQ(gs_orig.ndim(), 0)
        << "BUG EXPOSED: Moved-from invalid tensor has ndim()="
        << gs_orig.ndim() << " instead of 0!";

    std::println("✓ CRITICAL_MovedFromShapeBug: Bug test passed");
}

TEST_F(TensorMoveTest, CRITICAL_InconsistentState) {
    print_test_header("CRITICAL_InconsistentState");

    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_moved = std::move(gs_orig);

    bool is_valid = gs_orig.is_valid();
    size_t numel = gs_orig.numel();
    size_t ndim = gs_orig.ndim();

    std::println("  Moved-from tensor state:");
    std::println("    is_valid(): {}", is_valid);
    std::println("    numel(): {}", numel);
    std::println("    ndim(): {}", ndim);

    // HARD ASSERTS: All state must be consistent
    if (!is_valid) {
        ASSERT_EQ(numel, 0)
            << "INCONSISTENT: is_valid()=false but numel()=" << numel;
        ASSERT_EQ(ndim, 0)
            << "INCONSISTENT: is_valid()=false but ndim()=" << ndim;
    }

    std::println("✓ CRITICAL_InconsistentState: Consistency verified");
}

TEST_F(TensorMoveTest, MoveSelfAssignment) {
    print_test_header("MoveSelfAssignment");

    // Test self-move-assignment (should be safe)
    auto gs_tensor = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto original_ptr = gs_tensor.ptr<float>();

    gs_tensor = std::move(gs_tensor); // Self-move

    // Should remain valid (implementation-defined but should not crash)
    EXPECT_TRUE(gs_tensor.is_valid());
    EXPECT_EQ(gs_tensor.ptr<float>(), original_ptr);

    std::println("✓ MoveSelfAssignment: Self-move-assignment is safe");
}

TEST_F(TensorMoveTest, MoveWithViews) {
    print_test_header("MoveWithViews");

    // Create a view and move it
    auto gs_orig = lfs::core::Tensor::randn({100, 30}, lfs::core::Device::CUDA);
    auto gs_view = gs_orig.reshape({10, 300});

    EXPECT_TRUE(gs_view.is_view());

    auto gs_moved = std::move(gs_view);

    EXPECT_TRUE(gs_moved.is_valid());
    EXPECT_TRUE(gs_moved.is_view());
    EXPECT_FALSE(gs_view.is_valid());

    std::println("✓ MoveWithViews: Views move correctly preserving view status");
}

TEST_F(TensorMoveTest, MoveChainWithOperations) {
    print_test_header("MoveChainWithOperations");

    // Complex chain: create -> reshape -> move -> transpose -> move
    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_copy = gs_orig.clone();

    auto gs_1 = std::move(gs_orig);
    auto gs_2 = gs_1.reshape({10, 30});
    auto gs_3 = std::move(gs_2);
    auto gs_4 = gs_3.transpose();
    auto gs_final = std::move(gs_4);

    EXPECT_TRUE(gs_final.is_valid());
    EXPECT_EQ(gs_final.shape()[0], 30);
    EXPECT_EQ(gs_final.shape()[1], 10);

    // All intermediate tensors should be invalid
    EXPECT_FALSE(gs_orig.is_valid());
    EXPECT_FALSE(gs_2.is_valid());
    EXPECT_FALSE(gs_4.is_valid());

    std::println("✓ MoveChainWithOperations: Complex move chains work correctly");
}

TEST_F(TensorMoveTest, MoveWithDifferentDtypes) {
    print_test_header("MoveWithDifferentDtypes");

    // Test moving tensors with different dtypes
    auto gs_float = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA, lfs::core::DataType::Float32);
    auto gs_int = lfs::core::Tensor::randint({100, 3}, 0, 100, lfs::core::Device::CUDA, lfs::core::DataType::Int32);
    auto gs_bool = lfs::core::Tensor::full_bool({100, 3}, true, lfs::core::Device::CUDA);

    auto gs_float_moved = std::move(gs_float);
    auto gs_int_moved = std::move(gs_int);
    auto gs_bool_moved = std::move(gs_bool);

    EXPECT_TRUE(gs_float_moved.is_valid());
    EXPECT_EQ(gs_float_moved.dtype(), lfs::core::DataType::Float32);

    EXPECT_TRUE(gs_int_moved.is_valid());
    EXPECT_EQ(gs_int_moved.dtype(), lfs::core::DataType::Int32);

    EXPECT_TRUE(gs_bool_moved.is_valid());
    EXPECT_EQ(gs_bool_moved.dtype(), lfs::core::DataType::Bool);

    std::println("✓ MoveWithDifferentDtypes: Different dtypes move correctly");
}

TEST_F(TensorMoveTest, MoveFromTemporaryInExpression) {
    print_test_header("MoveFromTemporaryInExpression");

    // Test that temporaries in expressions move correctly
    auto result = [] {
        return lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA).add(1.0f);
    }();

    EXPECT_TRUE(result.is_valid());
    EXPECT_EQ(result.numel(), 300);

    std::println("✓ MoveFromTemporaryInExpression: Temporaries move correctly");
}

TEST_F(TensorMoveTest, MoveInLambdaCapture) {
    print_test_header("MoveInLambdaCapture");

    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_ptr = gs_orig.ptr<float>();

    auto lambda = [t = std::move(gs_orig)]() {
        return t.is_valid();
    };

    EXPECT_FALSE(gs_orig.is_valid());
    EXPECT_TRUE(lambda());

    std::println("✓ MoveInLambdaCapture: Lambda capture by move works correctly");
}

TEST_F(TensorMoveTest, MoveWithSlicing) {
    print_test_header("MoveWithSlicing");

    auto gs_orig = lfs::core::Tensor::randn({100, 30}, lfs::core::Device::CUDA);
    auto gs_slice = gs_orig.slice(0, 10, 20);
    auto gs_moved = std::move(gs_slice);

    EXPECT_TRUE(gs_moved.is_valid());
    EXPECT_EQ(gs_moved.shape()[0], 10);
    EXPECT_EQ(gs_moved.shape()[1], 30);
    EXPECT_FALSE(gs_slice.is_valid());

    std::println("✓ MoveWithSlicing: Sliced tensors move correctly");
}

TEST_F(TensorMoveTest, MoveSequenceInVector) {
    print_test_header("MoveSequenceInVector");

    // Create vector, move into it, move out of it
    std::vector<lfs::core::Tensor> vec;

    auto t1 = lfs::core::Tensor::randn({10, 3}, lfs::core::Device::CUDA);
    auto ptr1 = t1.ptr<float>();
    vec.push_back(std::move(t1));

    auto t2 = lfs::core::Tensor::randn({20, 3}, lfs::core::Device::CUDA);
    auto ptr2 = t2.ptr<float>();
    vec.push_back(std::move(t2));

    EXPECT_FALSE(t1.is_valid());
    EXPECT_FALSE(t2.is_valid());
    EXPECT_EQ(vec[0].ptr<float>(), ptr1);
    EXPECT_EQ(vec[1].ptr<float>(), ptr2);

    // Move out
    auto retrieved = std::move(vec[0]);
    EXPECT_TRUE(retrieved.is_valid());
    EXPECT_EQ(retrieved.ptr<float>(), ptr1);
    EXPECT_FALSE(vec[0].is_valid());

    std::println("✓ MoveSequenceInVector: Complex vector operations work correctly");
}

TEST_F(TensorMoveTest, MoveAfterDeviceTransfer) {
    print_test_header("MoveAfterDeviceTransfer");

    // Create on CUDA, move to CPU, then move the CPU tensor
    auto gs_cuda = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_cpu = gs_cuda.cpu();
    auto gs_moved = std::move(gs_cpu);

    EXPECT_TRUE(gs_moved.is_valid());
    EXPECT_EQ(gs_moved.device(), lfs::core::Device::CPU);
    EXPECT_FALSE(gs_cpu.is_valid());

    std::println("✓ MoveAfterDeviceTransfer: Moving after device transfer works");
}

TEST_F(TensorMoveTest, MoveWithBroadcasting) {
    print_test_header("MoveWithBroadcasting");

    auto gs_small = lfs::core::Tensor::randn({1, 3}, lfs::core::Device::CUDA);
    auto gs_broadcast = gs_small.broadcast_to(lfs::core::TensorShape({100, 3}));
    auto gs_moved = std::move(gs_broadcast);

    EXPECT_TRUE(gs_moved.is_valid());
    EXPECT_EQ(gs_moved.shape()[0], 100);
    EXPECT_FALSE(gs_broadcast.is_valid());

    std::println("✓ MoveWithBroadcasting: Broadcasted tensors move correctly");
}

TEST_F(TensorMoveTest, MoveReassignmentLoop) {
    print_test_header("MoveReassignmentLoop");

    // Test repeated reassignment via move
    lfs::core::Tensor current = lfs::core::Tensor::randn({10, 3}, lfs::core::Device::CUDA);

    for (int i = 0; i < 100; ++i) {
        auto temp = lfs::core::Tensor::randn({10, 3}, lfs::core::Device::CUDA);
        current = std::move(temp);
        EXPECT_TRUE(current.is_valid());
        EXPECT_FALSE(temp.is_valid());
    }

    std::println("✓ MoveReassignmentLoop: Repeated reassignment works correctly");
}

TEST_F(TensorMoveTest, CRITICAL_MovedFromBytesSize) {
    print_test_header("CRITICAL_MovedFromBytesSize");

    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    size_t orig_bytes = gs_orig.bytes();

    auto gs_moved = std::move(gs_orig);

    std::println("  Original bytes: {}", orig_bytes);
    std::println("  Moved-to bytes: {}", gs_moved.bytes());
    std::println("  Moved-from bytes: {}", gs_orig.bytes());

    // HARD ASSERT: Invalid tensor must report 0 bytes
    ASSERT_FALSE(gs_orig.is_valid());
    ASSERT_EQ(gs_orig.bytes(), 0)
        << "BUG: Invalid tensor reports bytes()=" << gs_orig.bytes() << " instead of 0!";

    ASSERT_EQ(gs_moved.bytes(), orig_bytes);

    std::println("✓ CRITICAL_MovedFromBytesSize: Bytes size consistent");
}

TEST_F(TensorMoveTest, CRITICAL_MovedFromShapeAccess) {
    print_test_header("CRITICAL_MovedFromShapeAccess");

    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_moved = std::move(gs_orig);

    ASSERT_FALSE(gs_orig.is_valid());

    // HARD ASSERT: Accessing shape dimensions should be safe and return 0
    std::println("  Accessing shape of invalid tensor:");
    std::println("    shape()[0]: {}", gs_orig.shape()[0]);
    std::println("    shape()[1]: {}", gs_orig.shape()[1]);

    // Either shape is uninitialized or returns 0
    // Both are acceptable, but numel() MUST be 0
    ASSERT_EQ(gs_orig.numel(), 0)
        << "BUG: Invalid tensor shape access returns non-zero numel()";

    std::println("✓ CRITICAL_MovedFromShapeAccess: Shape access safe");
}

TEST_F(TensorMoveTest, CRITICAL_ExpectedValueAccess) {
    print_test_header("CRITICAL_ExpectedValueAccess");

    // This reproduces the exact SplatDataNew pattern
    auto create_in_expected = []() -> std::expected<lfs::core::Tensor, std::string> {
        auto t = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
        std::println("    Created tensor - is_valid(): {}", t.is_valid());
        return t; // Moves into expected
    };

    auto result = create_in_expected();

    ASSERT_TRUE(result.has_value());

    std::println("  Accessing value from expected:");
    auto& tensor_ref = result.value();
    std::println("    is_valid(): {}", tensor_ref.is_valid());
    std::println("    numel(): {}", tensor_ref.numel());

    // HARD ASSERT: This is the SplatDataNew bug!
    ASSERT_TRUE(tensor_ref.is_valid())
        << "BUG EXPOSED: Tensor in std::expected is invalid after moves!";
    ASSERT_GT(tensor_ref.numel(), 0)
        << "BUG EXPOSED: Valid tensor in expected has numel()=0!";

    std::println("✓ CRITICAL_ExpectedValueAccess: Expected pattern works");
}

TEST_F(TensorMoveTest, CRITICAL_ConstructorMoveChain) {
    print_test_header("CRITICAL_ConstructorMoveChain");

    struct Wrapper {
        lfs::core::Tensor data;
        Wrapper(lfs::core::Tensor t) : data(std::move(t)) {
            std::println("    In constructor - data.is_valid(): {}", data.is_valid());
            std::println("    In constructor - data.numel(): {}", data.numel());
        }
    };

    auto create_wrapped = []() -> std::expected<Wrapper, std::string> {
        auto t = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
        std::println("  Created tensor - is_valid(): {}", t.is_valid());
        return Wrapper(std::move(t)); // Move into Wrapper, then into expected
    };

    auto result = create_wrapped();
    ASSERT_TRUE(result.has_value());

    auto& wrapper = result.value();
    std::println("  Final state:");
    std::println("    wrapper.data.is_valid(): {}", wrapper.data.is_valid());
    std::println("    wrapper.data.numel(): {}", wrapper.data.numel());

    // HARD ASSERT: This is THE bug from SplatDataNew!
    ASSERT_TRUE(wrapper.data.is_valid())
        << "BUG EXPOSED: This is the SplatDataNew failure! "
        << "Tensor becomes invalid after constructor+expected moves!";
    ASSERT_EQ(wrapper.data.numel(), 300)
        << "BUG EXPOSED: Valid tensor has wrong numel after move chain!";

    std::println("✓ CRITICAL_ConstructorMoveChain: Move chain works");
}

// ============================================================================
// Stress Tests
// ============================================================================

TEST_F(TensorMoveTest, StressTest_ManyMoves) {
    print_test_header("StressTest_ManyMoves");

    const int iterations = 1000;

    // Torch version
    auto torch_orig = torch::randn({100, 3}, torch::kCUDA);
    auto torch_copy = torch_orig.clone();
    torch::Tensor torch_current = std::move(torch_orig);

    for (int i = 0; i < iterations; ++i) {
        torch::Tensor torch_temp = std::move(torch_current);
        torch_current = std::move(torch_temp);
    }

    EXPECT_TRUE(torch_current.defined());
    EXPECT_TRUE(tensors_equal(torch_current, torch_copy));

    // Our Tensor version
    auto gs_orig = lfs::core::Tensor::randn({100, 3}, lfs::core::Device::CUDA);
    auto gs_copy = gs_orig.clone();
    lfs::core::Tensor gs_current = std::move(gs_orig);

    for (int i = 0; i < iterations; ++i) {
        lfs::core::Tensor gs_temp = std::move(gs_current);
        gs_current = std::move(gs_temp);
    }

    EXPECT_TRUE(gs_current.is_valid());
    EXPECT_TRUE(tensors_equal(gs_current, gs_copy));

    std::println("✓ StressTest_ManyMoves: {} moves completed successfully", iterations);
}

TEST_F(TensorMoveTest, StressTest_VectorResize) {
    print_test_header("StressTest_VectorResize");

    // Test that moves work correctly during vector resizing
    std::vector<lfs::core::Tensor> vec;
    std::vector<float*> ptrs;

    const int count = 1000;
    for (int i = 0; i < count; ++i) {
        auto t = lfs::core::Tensor::randn({10, 3}, lfs::core::Device::CUDA);
        ptrs.push_back(t.ptr<float>());
        vec.push_back(std::move(t));
    }

    // Verify all tensors survived the resizing
    EXPECT_EQ(vec.size(), count);
    for (int i = 0; i < count; ++i) {
        EXPECT_TRUE(vec[i].is_valid());
        EXPECT_EQ(vec[i].ptr<float>(), ptrs[i]);
    }

    std::println("✓ StressTest_VectorResize: {} tensors in vector all valid", count);
}

// ============================================================================
// Summary
// ============================================================================

TEST_F(TensorMoveTest, Summary) {
    const std::string sep(80, '=');
    std::println("\n{}", sep);
    std::println("TENSOR MOVE SEMANTICS TEST SUMMARY");
    std::println("{}", sep);

    // Get test results
    auto test_info = ::testing::UnitTest::GetInstance()->current_test_info();
    auto test_case = ::testing::UnitTest::GetInstance()->current_test_case();

    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;

    // Count tests in this test case
    for (int i = 0; i < test_case->total_test_count(); ++i) {
        total_tests++;
        auto* info = test_case->GetTestInfo(i);
        if (info->result()->Passed()) {
            passed_tests++;
        } else if (info->result()->Failed()) {
            failed_tests++;
        }
    }

    std::println("Tests run: {}", total_tests);
    std::println("Tests passed: {}", passed_tests);
    std::println("Tests FAILED: {}", failed_tests);
    std::println("");

    if (failed_tests > 0) {
        std::println("⚠️  CRITICAL: {} test(s) failed!", failed_tests);
        std::println("");
        std::println("DIAGNOSIS:");
        std::println("The Tensor move semantics have bugs that need fixing.");
        std::println("");
        std::println("Most likely issues:");
        std::println("1. initialized_ flag is set to false but shape data remains");
        std::println("2. is_valid() checks initialized_ but not actual data presence");
        std::println("3. Multiple moves through std::expected cause validation to fail");
        std::println("");
        std::println("RECOMMENDED FIXES:");
        std::println("1. Don't set initialized_=false in move constructor");
        std::println("2. Change is_valid() to: shape_.is_initialized() && (data_ || data_owner_)");
        std::println("3. Use std::exchange() to properly null out data_ in move operations");
        std::println("");
        std::println("See the CRITICAL_* tests for specific failure scenarios.");
    } else {
        std::println("✅ All tensor move tests passed!");
        std::println("The custom Tensor implementation has correct move semantics");
        std::println("matching PyTorch's behavior.");
    }

    std::println("{}\n", sep);
}

/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

/**
 * Tests to expose the bug in slice() and mm() operations.
 *
 * The bug was discovered in transforms.cpp where extracting R and T
 * from a world-to-camera matrix using slice() produced incorrect results.
 *
 * The workaround was to use direct GLM matrix extraction instead of tensor operations.
 */

#include "core/logger.hpp"
#include "core/tensor.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <numbers>
#include <torch/torch.h>

using namespace lfs::core;

namespace {

    void compare_tensors(const Tensor& custom, const torch::Tensor& reference,
                         float rtol = 1e-4f, float atol = 1e-5f, const std::string& msg = "") {
        auto ref_cpu = reference.to(torch::kCPU).contiguous().flatten();
        auto custom_cpu = custom.cpu().contiguous();

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

            float diff = std::abs(custom_val - ref_val);
            float threshold = atol + rtol * std::abs(ref_val);
            EXPECT_LE(diff, threshold)
                << msg << ": Mismatch at index " << i
                << " (custom=" << custom_val << ", ref=" << ref_val << ")";
        }
    }

    // Create a Y-axis rotation matrix as in transforms.cpp
    Tensor createYRotationMatrix(float angle_radians) {
        Tensor rotMat = Tensor::eye(4, Device::CPU);
        float cos_angle = std::cos(angle_radians);
        float sin_angle = std::sin(angle_radians);

        rotMat[0][0] = cos_angle;
        rotMat[0][1] = 0.0f;
        rotMat[0][2] = sin_angle;
        rotMat[1][0] = 0.0f;
        rotMat[1][1] = 1.0f;
        rotMat[1][2] = 0.0f;
        rotMat[2][0] = -sin_angle;
        rotMat[2][1] = 0.0f;
        rotMat[2][2] = cos_angle;

        return rotMat;
    }

    torch::Tensor createYRotationMatrixTorch(float angle_radians) {
        float cos_angle = std::cos(angle_radians);
        float sin_angle = std::sin(angle_radians);

        auto rotMat = torch::eye(4, torch::kFloat32);
        rotMat[0][0] = cos_angle;
        rotMat[0][1] = 0.0f;
        rotMat[0][2] = sin_angle;
        rotMat[1][0] = 0.0f;
        rotMat[1][1] = 1.0f;
        rotMat[1][2] = 0.0f;
        rotMat[2][0] = -sin_angle;
        rotMat[2][1] = 0.0f;
        rotMat[2][2] = cos_angle;

        return rotMat;
    }

} // namespace

class SliceMMBugTest : public ::testing::Test {
protected:
    void SetUp() override {
        torch::manual_seed(42);
        Tensor::manual_seed(42);
    }
};

// Test 1: Basic slice then contiguous
TEST_F(SliceMMBugTest, SliceContiguous_3x3_From_4x4) {
    // Create a 4x4 matrix with known values
    // Row-major: [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]]
    std::vector<float> data(16);
    for (int i = 0; i < 16; i++) {
        data[i] = static_cast<float>(i + 1);
    }

    auto custom = Tensor::from_vector(data, {4, 4}, Device::CPU);
    auto torch_t = torch::tensor(data).reshape({4, 4});

    // Extract 3x3 submatrix: [:3, :3]
    auto custom_slice = custom.slice(0, 0, 3).slice(1, 0, 3);
    auto torch_slice = torch_t.slice(0, 0, 3).slice(1, 0, 3);

    LOG_INFO("Custom slice shape: {}x{}", custom_slice.shape()[0], custom_slice.shape()[1]);
    LOG_INFO("Custom slice strides: [{}, {}]", custom_slice.stride(0), custom_slice.stride(1));
    LOG_INFO("Custom slice is_contiguous: {}", custom_slice.is_contiguous());

    // Make contiguous
    auto custom_contiguous = custom_slice.contiguous();
    auto torch_contiguous = torch_slice.contiguous();

    LOG_INFO("Custom contiguous strides: [{}, {}]", custom_contiguous.stride(0), custom_contiguous.stride(1));
    LOG_INFO("Custom contiguous is_contiguous: {}", custom_contiguous.is_contiguous());

    // Verify values
    auto custom_vec = custom_contiguous.to_vector();

    LOG_INFO("Custom values: [{}, {}, {}, {}, {}, {}, {}, {}, {}]",
             custom_vec[0], custom_vec[1], custom_vec[2],
             custom_vec[3], custom_vec[4], custom_vec[5],
             custom_vec[6], custom_vec[7], custom_vec[8]);

    // Expected: [[1,2,3], [5,6,7], [9,10,11]]
    std::vector<float> expected = {1, 2, 3, 5, 6, 7, 9, 10, 11};
    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(custom_vec[i], expected[i])
            << "Mismatch at index " << i << ": got " << custom_vec[i] << ", expected " << expected[i];
    }

    compare_tensors(custom_contiguous, torch_contiguous, 1e-5f, 1e-6f, "Slice3x3");
}

// Test 2: Extract column slice (translation vector)
TEST_F(SliceMMBugTest, SliceColumn_3x1_From_4x4) {
    std::vector<float> data(16);
    for (int i = 0; i < 16; i++) {
        data[i] = static_cast<float>(i + 1);
    }

    auto custom = Tensor::from_vector(data, {4, 4}, Device::CPU);
    auto torch_t = torch::tensor(data).reshape({4, 4});

    // Extract translation column: [:3, 3:4]
    auto custom_slice = custom.slice(0, 0, 3).slice(1, 3, 4);
    auto torch_slice = torch_t.slice(0, 0, 3).slice(1, 3, 4);

    LOG_INFO("Custom T slice shape: {}x{}", custom_slice.shape()[0], custom_slice.shape()[1]);
    LOG_INFO("Custom T slice strides: [{}, {}]", custom_slice.stride(0), custom_slice.stride(1));
    LOG_INFO("Custom T slice storage_offset: {}", custom_slice.storage_offset());

    // Squeeze to 1D
    auto custom_squeezed = custom_slice.squeeze(1);
    auto torch_squeezed = torch_slice.squeeze(1);

    LOG_INFO("Custom squeezed shape: {}", custom_squeezed.shape()[0]);
    LOG_INFO("Custom squeezed strides: [{}]", custom_squeezed.stride(0));

    // Make contiguous
    auto custom_contiguous = custom_squeezed.contiguous();
    auto torch_contiguous = torch_squeezed.contiguous();

    auto custom_vec = custom_contiguous.to_vector();

    LOG_INFO("Custom T values: [{}, {}, {}]", custom_vec[0], custom_vec[1], custom_vec[2]);

    // Expected: column 3 = [4, 8, 12]
    std::vector<float> expected = {4, 8, 12};
    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_FLOAT_EQ(custom_vec[i], expected[i])
            << "Mismatch at index " << i << ": got " << custom_vec[i] << ", expected " << expected[i];
    }

    compare_tensors(custom_contiguous, torch_contiguous, 1e-5f, 1e-6f, "SliceColumn");
}

// Test 3: Matrix multiply then slice - exactly as in transforms.cpp
TEST_F(SliceMMBugTest, MMThenSlice_TransformsPattern) {
    // Create a sample world-to-camera matrix
    std::vector<float> w2c_data = {
        0.866f, 0.0f, -0.5f, 1.0f,     // Row 0
        0.0f, 1.0f, 0.0f, 2.0f,        // Row 1
        0.5f, 0.0f, 0.866f, 3.0f,      // Row 2
        0.0f, 0.0f, 0.0f, 1.0f         // Row 3
    };

    auto custom_w2c = Tensor::from_vector(w2c_data, {4, 4}, Device::CPU);
    auto torch_w2c = torch::tensor(w2c_data).reshape({4, 4});

    // Create rotation matrix (180 degrees around Y)
    auto custom_fix = createYRotationMatrix(M_PI);
    auto torch_fix = createYRotationMatrixTorch(M_PI);

    LOG_INFO("Custom fixMat: ");
    custom_fix.print_formatted("fixMat");
    std::cout << "Torch fixMat: " << torch_fix << std::endl;

    // Perform matrix multiplication: w2c @ fixMat
    auto custom_result = custom_w2c.mm(custom_fix);
    auto torch_result = torch::mm(torch_w2c, torch_fix);

    LOG_INFO("Custom mm result: ");
    custom_result.print_formatted("mm_result");
    std::cout << "Torch mm result: " << torch_result << std::endl;

    // Verify mm result
    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "MM_Result");

    // Now extract R (3x3) and T (3x1) - this is where the bug was!
    auto custom_R = custom_result.slice(0, 0, 3).slice(1, 0, 3);
    auto torch_R = torch_result.slice(0, 0, 3).slice(1, 0, 3);

    LOG_INFO("Custom R slice shape: {}x{}", custom_R.shape()[0], custom_R.shape()[1]);
    LOG_INFO("Custom R is_contiguous: {}", custom_R.is_contiguous());
    LOG_INFO("Custom R strides: [{}, {}]", custom_R.stride(0), custom_R.stride(1));

    auto custom_R_cont = custom_R.contiguous();
    auto torch_R_cont = torch_R.contiguous();

    LOG_INFO("Custom R contiguous: ");
    custom_R_cont.print_formatted("R_contiguous");
    std::cout << "Torch R contiguous: " << torch_R_cont << std::endl;

    compare_tensors(custom_R_cont, torch_R_cont, 1e-4f, 1e-5f, "R_Matrix");

    // Extract T
    auto custom_T = custom_result.slice(0, 0, 3).slice(1, 3, 4).squeeze(1);
    auto torch_T = torch_result.slice(0, 0, 3).slice(1, 3, 4).squeeze(1);

    LOG_INFO("Custom T slice shape: {}", custom_T.shape()[0]);

    auto custom_T_cont = custom_T.contiguous();
    auto torch_T_cont = torch_T.contiguous();

    LOG_INFO("Custom T contiguous: ");
    custom_T_cont.print_formatted("T_contiguous");
    std::cout << "Torch T contiguous: " << torch_T_cont << std::endl;

    compare_tensors(custom_T_cont, torch_T_cont, 1e-4f, 1e-5f, "T_Vector");
}

// Test 4: Chained operations - slice, mm, slice
TEST_F(SliceMMBugTest, SliceMMSlice_Chain) {
    // Create input matrix
    std::vector<float> data(16);
    for (int i = 0; i < 16; i++) {
        data[i] = static_cast<float>(i + 1) * 0.1f;
    }

    auto custom = Tensor::from_vector(data, {4, 4}, Device::CPU);
    auto torch_t = torch::tensor(data).reshape({4, 4});

    // Create a simple 4x4 transformation
    auto custom_transform = createYRotationMatrix(0.5f);
    auto torch_transform = createYRotationMatrixTorch(0.5f);

    // Chain: mm -> slice -> contiguous
    auto custom_result = custom.mm(custom_transform);
    auto torch_result = torch::mm(torch_t, torch_transform);

    // Extract 3x3
    auto custom_3x3 = custom_result.slice(0, 0, 3).slice(1, 0, 3).contiguous();
    auto torch_3x3 = torch_result.slice(0, 0, 3).slice(1, 0, 3).contiguous();

    compare_tensors(custom_3x3, torch_3x3, 1e-4f, 1e-5f, "ChainedSlice3x3");

    // Extract column
    auto custom_col = custom_result.slice(0, 0, 3).slice(1, 3, 4).squeeze(1).contiguous();
    auto torch_col = torch_result.slice(0, 0, 3).slice(1, 3, 4).squeeze(1).contiguous();

    compare_tensors(custom_col, torch_col, 1e-4f, 1e-5f, "ChainedSliceCol");
}

// Test 5: Multiple mm operations followed by slice
TEST_F(SliceMMBugTest, MultipleMMThenSlice) {
    std::vector<float> a_data(16), b_data(16), c_data(16);
    for (int i = 0; i < 16; i++) {
        a_data[i] = static_cast<float>(i + 1) * 0.1f;
        b_data[i] = static_cast<float>(16 - i) * 0.1f;
        c_data[i] = static_cast<float>((i % 4) + 1) * 0.25f;
    }

    auto custom_a = Tensor::from_vector(a_data, {4, 4}, Device::CPU);
    auto custom_b = Tensor::from_vector(b_data, {4, 4}, Device::CPU);
    auto custom_c = Tensor::from_vector(c_data, {4, 4}, Device::CPU);

    auto torch_a = torch::tensor(a_data).reshape({4, 4});
    auto torch_b = torch::tensor(b_data).reshape({4, 4});
    auto torch_c = torch::tensor(c_data).reshape({4, 4});

    // A @ B @ C
    auto custom_result = custom_a.mm(custom_b).mm(custom_c);
    auto torch_result = torch::mm(torch::mm(torch_a, torch_b), torch_c);

    // Extract submatrices
    auto custom_3x3 = custom_result.slice(0, 0, 3).slice(1, 0, 3).contiguous();
    auto torch_3x3 = torch_result.slice(0, 0, 3).slice(1, 0, 3).contiguous();

    compare_tensors(custom_3x3, torch_3x3, 1e-3f, 1e-4f, "MultiMM_Slice3x3");
}

// Test 6: Verify storage_offset is correctly handled
TEST_F(SliceMMBugTest, StorageOffsetCorrectness) {
    std::vector<float> data(16);
    for (int i = 0; i < 16; i++) {
        data[i] = static_cast<float>(i + 1);
    }

    auto tensor = Tensor::from_vector(data, {4, 4}, Device::CPU);

    // Get element [1][2] directly
    float direct_access = tensor[1][2];
    EXPECT_FLOAT_EQ(direct_access, 7.0f); // Row 1, Col 2 = 5 + 2 + 1 = 7

    // Get element [1][2] via slice
    auto row_slice = tensor.slice(0, 1, 2); // Get row 1
    auto elem_slice = row_slice.slice(1, 2, 3); // Get column 2
    float slice_access = elem_slice.contiguous()[0][0];

    LOG_INFO("Direct access [1][2]: {}", direct_access);
    LOG_INFO("Slice access [1][2]: {}", slice_access);
    LOG_INFO("Row slice storage_offset: {}", row_slice.storage_offset());
    LOG_INFO("Elem slice storage_offset: {}", elem_slice.storage_offset());

    EXPECT_FLOAT_EQ(slice_access, direct_access)
        << "Storage offset not handled correctly!";
}

// Test 7: Non-contiguous tensor used in mm
TEST_F(SliceMMBugTest, NonContiguousMM) {
    std::vector<float> data(16);
    for (int i = 0; i < 16; i++) {
        data[i] = static_cast<float>(i + 1);
    }

    auto tensor = Tensor::from_vector(data, {4, 4}, Device::CPU);
    auto identity = Tensor::eye(4, Device::CPU);

    auto torch_t = torch::tensor(data).reshape({4, 4});
    auto torch_eye = torch::eye(4);

    // Create non-contiguous view via slice (taking rows 0-2)
    auto custom_slice = tensor.slice(0, 0, 3);  // 3x4, non-contiguous
    auto torch_slice = torch_t.slice(0, 0, 3);  // 3x4

    LOG_INFO("Custom slice is_contiguous: {}", custom_slice.is_contiguous());
    LOG_INFO("Custom slice strides: [{}, {}]", custom_slice.stride(0), custom_slice.stride(1));

    // Now try to use this non-contiguous tensor in mm
    // Note: identity is 4x4, we need a 4xN matrix for mm to work
    auto custom_eye_slice = identity.slice(0, 0, 4).slice(1, 0, 3); // 4x3

    // 3x4 @ 4x3 should work
    auto custom_result = custom_slice.mm(custom_eye_slice.contiguous());
    auto torch_result = torch::mm(torch_slice, torch_eye.slice(0, 0, 4).slice(1, 0, 3));

    LOG_INFO("Custom mm result shape: {}x{}", custom_result.shape()[0], custom_result.shape()[1]);

    compare_tensors(custom_result, torch_result, 1e-4f, 1e-5f, "NonContiguousMM");
}

// Test 8: Verify slice indices are correct for column extraction
TEST_F(SliceMMBugTest, ColumnExtractionVerification) {
    // Create a matrix where each column has a distinct pattern
    // Col 0: 1, 5, 9, 13
    // Col 1: 2, 6, 10, 14
    // Col 2: 3, 7, 11, 15
    // Col 3: 4, 8, 12, 16
    std::vector<float> data(16);
    for (int i = 0; i < 16; i++) {
        data[i] = static_cast<float>(i + 1);
    }

    auto tensor = Tensor::from_vector(data, {4, 4}, Device::CPU);

    // Extract each column and verify
    for (int col = 0; col < 4; col++) {
        auto col_slice = tensor.slice(1, col, col + 1).squeeze(1).contiguous();
        auto col_vec = col_slice.to_vector();

        LOG_INFO("Column {} values: [{}, {}, {}, {}]", col,
                 col_vec[0], col_vec[1], col_vec[2], col_vec[3]);

        for (int row = 0; row < 4; row++) {
            float expected = row * 4 + col + 1;
            EXPECT_FLOAT_EQ(col_vec[row], expected)
                << "Column " << col << ", Row " << row
                << ": got " << col_vec[row] << ", expected " << expected;
        }
    }
}

// Test 9: Verify row extraction
TEST_F(SliceMMBugTest, RowExtractionVerification) {
    std::vector<float> data(16);
    for (int i = 0; i < 16; i++) {
        data[i] = static_cast<float>(i + 1);
    }

    auto tensor = Tensor::from_vector(data, {4, 4}, Device::CPU);

    // Extract each row and verify
    for (int row = 0; row < 4; row++) {
        auto row_slice = tensor.slice(0, row, row + 1).squeeze(0).contiguous();
        auto row_vec = row_slice.to_vector();

        LOG_INFO("Row {} values: [{}, {}, {}, {}]", row,
                 row_vec[0], row_vec[1], row_vec[2], row_vec[3]);

        for (int col = 0; col < 4; col++) {
            float expected = row * 4 + col + 1;
            EXPECT_FLOAT_EQ(row_vec[col], expected)
                << "Row " << row << ", Col " << col
                << ": got " << row_vec[col] << ", expected " << expected;
        }
    }
}

// Test 10: Exact reproduction of the transforms.cpp pattern
TEST_F(SliceMMBugTest, ExactTransformsPattern) {
    // Simulate a realistic camera transform matrix
    // This is a world-to-camera matrix after conversion from OpenGL coordinates
    std::vector<float> w2c_data = {
        0.9848f, -0.0312f, 0.1710f, -0.5f,
        0.0f, 0.9848f, 0.1736f, 1.2f,
        -0.1736f, -0.1710f, 0.9698f, 3.5f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    auto custom_w2c = Tensor::from_vector(w2c_data, {4, 4}, Device::CPU);
    auto torch_w2c = torch::tensor(w2c_data).reshape({4, 4});

    // Create Y rotation matrix for 180 degrees (as in transforms.cpp)
    auto custom_fixMat = createYRotationMatrix(M_PI);
    auto torch_fixMat = createYRotationMatrixTorch(M_PI);

    // Perform w2c @ fixMat
    auto custom_w2c_fixed = custom_w2c.mm(custom_fixMat);
    auto torch_w2c_fixed = torch::mm(torch_w2c, torch_fixMat);

    // Extract R and T exactly as in transforms.cpp:
    // R = w2c[:3,:3]
    // T = w2c[:3, 3]
    auto custom_R = custom_w2c_fixed.slice(0, 0, 3).slice(1, 0, 3);
    auto custom_T = custom_w2c_fixed.slice(0, 0, 3).slice(1, 3, 4).squeeze(1);

    auto torch_R = torch_w2c_fixed.slice(0, 0, 3).slice(1, 0, 3);
    auto torch_T = torch_w2c_fixed.slice(0, 0, 3).slice(1, 3, 4).squeeze(1);

    // Make contiguous (as done in transforms.cpp)
    auto custom_R_cont = custom_R.contiguous();
    auto custom_T_cont = custom_T.contiguous();

    auto torch_R_cont = torch_R.contiguous();
    auto torch_T_cont = torch_T.contiguous();

    LOG_INFO("=== Exact Transforms Pattern Test ===");
    LOG_INFO("Custom R shape: {}x{}", custom_R_cont.shape()[0], custom_R_cont.shape()[1]);
    LOG_INFO("Custom T shape: {}", custom_T_cont.shape()[0]);

    custom_R_cont.print_formatted("Custom R");
    custom_T_cont.print_formatted("Custom T");

    std::cout << "Torch R: " << torch_R_cont << std::endl;
    std::cout << "Torch T: " << torch_T_cont << std::endl;

    compare_tensors(custom_R_cont, torch_R_cont, 1e-4f, 1e-5f, "ExactPattern_R");
    compare_tensors(custom_T_cont, torch_T_cont, 1e-4f, 1e-5f, "ExactPattern_T");
}

// Test 11: Test slice on CUDA tensors
TEST_F(SliceMMBugTest, SliceOnCUDA) {
    if (!torch::cuda::is_available()) {
        GTEST_SKIP() << "CUDA not available";
    }

    std::vector<float> data(16);
    for (int i = 0; i < 16; i++) {
        data[i] = static_cast<float>(i + 1);
    }

    auto custom = Tensor::from_vector(data, {4, 4}, Device::CUDA);
    auto torch_t = torch::tensor(data, torch::kCUDA).reshape({4, 4});

    // Create rotation matrix on CUDA
    auto custom_fix = createYRotationMatrix(M_PI).cuda();
    auto torch_fix = createYRotationMatrixTorch(M_PI).cuda();

    // mm on CUDA
    auto custom_result = custom.mm(custom_fix);
    auto torch_result = torch::mm(torch_t, torch_fix);

    // Slice on CUDA
    auto custom_R = custom_result.slice(0, 0, 3).slice(1, 0, 3).contiguous();
    auto torch_R = torch_result.slice(0, 0, 3).slice(1, 0, 3).contiguous();

    auto custom_T = custom_result.slice(0, 0, 3).slice(1, 3, 4).squeeze(1).contiguous();
    auto torch_T = torch_result.slice(0, 0, 3).slice(1, 3, 4).squeeze(1).contiguous();

    compare_tensors(custom_R, torch_R, 1e-4f, 1e-5f, "CUDA_Slice_R");
    compare_tensors(custom_T, torch_T, 1e-4f, 1e-5f, "CUDA_Slice_T");
}

// Test 12: Verify that comment in transforms.cpp says transpose but code doesn't do it
TEST_F(SliceMMBugTest, TransposeVsNoTranspose_Comment_Code_Mismatch) {
    // The comment in transforms.cpp says:
    // "R = np.transpose(w2c[:3,:3])"
    // But the code does:
    // lfs::core::Tensor R = w2c.slice(0, 0, 3).slice(1, 0, 3);
    // This is WITHOUT transpose!

    // The GLM extraction does transpose:
    // r_ptr[row * 3 + col] = w2c_glm[col][row];
    // This is equivalent to R = transpose(mat)

    // Let's verify if there's a difference

    std::vector<float> w2c_data = {
        0.9848f, -0.0312f, 0.1710f, -0.5f,
        0.0f, 0.9848f, 0.1736f, 1.2f,
        -0.1736f, -0.1710f, 0.9698f, 3.5f,
        0.0f, 0.0f, 0.0f, 1.0f
    };

    auto tensor_w2c = Tensor::from_vector(w2c_data, {4, 4}, Device::CPU);

    // What the current code does (NO transpose)
    auto R_no_transpose = tensor_w2c.slice(0, 0, 3).slice(1, 0, 3).contiguous();

    // What the comment says should happen (WITH transpose)
    auto R_with_transpose = tensor_w2c.slice(0, 0, 3).slice(1, 0, 3).t().contiguous();

    LOG_INFO("=== R without transpose (current code) ===");
    R_no_transpose.print_formatted("R_no_transpose");

    LOG_INFO("=== R with transpose (what comment says) ===");
    R_with_transpose.print_formatted("R_with_transpose");

    // These are DIFFERENT!
    auto no_t_vec = R_no_transpose.to_vector();
    auto with_t_vec = R_with_transpose.to_vector();

    bool are_different = false;
    for (size_t i = 0; i < no_t_vec.size(); i++) {
        if (std::abs(no_t_vec[i] - with_t_vec[i]) > 1e-5f) {
            are_different = true;
            break;
        }
    }

    // For a non-symmetric rotation matrix, they should be different
    // This highlights a potential code/comment mismatch
    LOG_INFO("R matrices are different: {}", are_different);

    // The tensor library is working correctly - this is just documentation
    // that the comment and code may not match
}

// Test 13: Direct element access on sliced tensor WITHOUT contiguous()
TEST_F(SliceMMBugTest, DirectElementAccess_NonContiguousSlice) {
    std::vector<float> data(16);
    for (int i = 0; i < 16; i++) {
        data[i] = static_cast<float>(i + 1);
    }

    auto tensor = Tensor::from_vector(data, {4, 4}, Device::CPU);

    // Create a non-contiguous slice
    auto slice = tensor.slice(0, 0, 3).slice(1, 0, 3);  // 3x3 from 4x4

    EXPECT_FALSE(slice.is_contiguous()) << "Slice should be non-contiguous";

    // Test direct element access WITHOUT calling contiguous()
    // This tests the operator[] path for non-contiguous tensors
    LOG_INFO("Testing direct element access on non-contiguous slice:");

    // Expected values: upper-left 3x3 of [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
    // Which is [[1,2,3],[5,6,7],[9,10,11]]
    float expected[3][3] = {
        {1, 2, 3},
        {5, 6, 7},
        {9, 10, 11}
    };

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            float actual = slice[i][j];
            LOG_INFO("slice[{}][{}] = {} (expected {})", i, j, actual, expected[i][j]);
            EXPECT_FLOAT_EQ(actual, expected[i][j])
                << "Element [" << i << "][" << j << "] incorrect";
        }
    }
}

// Test 14: Element access on squeeze result WITHOUT contiguous()
TEST_F(SliceMMBugTest, DirectElementAccess_SqueezedSlice) {
    std::vector<float> data(16);
    for (int i = 0; i < 16; i++) {
        data[i] = static_cast<float>(i + 1);
    }

    auto tensor = Tensor::from_vector(data, {4, 4}, Device::CPU);

    // Get column 3 (4th column), rows 0-2
    auto T_slice = tensor.slice(0, 0, 3).slice(1, 3, 4).squeeze(1);

    LOG_INFO("T_slice shape: {}", T_slice.shape()[0]);
    LOG_INFO("T_slice is_contiguous: {}", T_slice.is_contiguous());

    // Expected: column 3 = [4, 8, 12]
    float expected[] = {4, 8, 12};

    for (int i = 0; i < 3; i++) {
        float actual = T_slice[i];
        LOG_INFO("T_slice[{}] = {} (expected {})", i, actual, expected[i]);
        EXPECT_FLOAT_EQ(actual, expected[i])
            << "Element [" << i << "] incorrect: got " << actual << ", expected " << expected[i];
    }
}

// Test 15: Test squeeze on 1D result
TEST_F(SliceMMBugTest, SqueezeNonContiguous1D) {
    std::vector<float> data(16);
    for (int i = 0; i < 16; i++) {
        data[i] = static_cast<float>(i + 1);
    }

    auto tensor = Tensor::from_vector(data, {4, 4}, Device::CPU);
    auto torch_t = torch::tensor(data).reshape({4, 4});

    // Get column 2, rows 1-3: should be [7, 11, 15] (indices 6, 10, 14 in flat array)
    auto custom_col = tensor.slice(0, 1, 4).slice(1, 2, 3);  // 3x1
    auto torch_col = torch_t.slice(0, 1, 4).slice(1, 2, 3);  // 3x1

    LOG_INFO("Custom col shape: {}x{}", custom_col.shape()[0], custom_col.shape()[1]);
    LOG_INFO("Custom col storage_offset: {}", custom_col.storage_offset());
    LOG_INFO("Custom col strides: [{}, {}]", custom_col.stride(0), custom_col.stride(1));

    // Squeeze to 1D
    auto custom_squeezed = custom_col.squeeze(1);
    auto torch_squeezed = torch_col.squeeze(1);

    LOG_INFO("Custom squeezed shape: {}", custom_squeezed.shape()[0]);
    LOG_INFO("Custom squeezed stride: {}", custom_squeezed.stride(0));

    auto custom_cont = custom_squeezed.contiguous();
    auto torch_cont = torch_squeezed.contiguous();

    auto custom_vec = custom_cont.to_vector();
    LOG_INFO("Custom squeezed values: [{}, {}, {}]", custom_vec[0], custom_vec[1], custom_vec[2]);

    // Expected: [7, 11, 15]
    EXPECT_FLOAT_EQ(custom_vec[0], 7.0f);
    EXPECT_FLOAT_EQ(custom_vec[1], 11.0f);
    EXPECT_FLOAT_EQ(custom_vec[2], 15.0f);

    compare_tensors(custom_cont, torch_cont, 1e-5f, 1e-6f, "SqueezeNonContiguous");
}

/* Test: Interleaved buffer construction via slice + copy_
 *
 * Tests the pattern used in point cloud renderer:
 *   interleaved[N, 7].slice(1, 0, 3).copy_(positions[N, 3])
 *   interleaved[N, 7].slice(1, 3, 6).copy_(colors[N, 3])
 *   interleaved[N, 7].slice(1, 6, 7).copy_(indices[N, 1])
 */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>

using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::Tensor;

class InterleavedSliceCopyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Warm up GPU
        auto warmup = Tensor::ones({100, 100}, Device::CUDA);
        warmup.cpu();
    }
};

// Test basic column slice shape
TEST_F(InterleavedSliceCopyTest, ColumnSliceShape) {
    const size_t N = 10;
    Tensor t = Tensor::zeros({N, 7}, Device::CUDA);

    auto slice_0_3 = t.slice(1, 0, 3);
    auto slice_3_6 = t.slice(1, 3, 6);
    auto slice_6_7 = t.slice(1, 6, 7);

    EXPECT_EQ(slice_0_3.size(0), N);
    EXPECT_EQ(slice_0_3.size(1), 3) << "slice(1, 0, 3) should have 3 columns";

    EXPECT_EQ(slice_3_6.size(0), N);
    EXPECT_EQ(slice_3_6.size(1), 3) << "slice(1, 3, 6) should have 3 columns";

    EXPECT_EQ(slice_6_7.size(0), N);
    EXPECT_EQ(slice_6_7.size(1), 1) << "slice(1, 6, 7) should have 1 column";
}

// Compare with LibTorch behavior
TEST_F(InterleavedSliceCopyTest, ColumnSliceShapeVsTorch) {
    const size_t N = 10;

    auto torch_t = torch::zeros({N, 7}, torch::kCUDA);
    auto torch_slice = torch_t.slice(1, 0, 3);

    Tensor lfs_t = Tensor::zeros({N, 7}, Device::CUDA);
    auto lfs_slice = lfs_t.slice(1, 0, 3);

    EXPECT_EQ(lfs_slice.size(0), torch_slice.size(0));
    EXPECT_EQ(lfs_slice.size(1), torch_slice.size(1));
}

// Test copy_ into column slice
TEST_F(InterleavedSliceCopyTest, CopyIntoColumnSlice) {
    const size_t N = 5;

    Tensor dst = Tensor::zeros({N, 7}, Device::CUDA);
    Tensor src = Tensor::ones({N, 3}, Device::CUDA) * 2.0f;

    dst.slice(1, 0, 3).copy_(src);

    auto cpu = dst.cpu();
    const float* data = cpu.ptr<float>();

    for (size_t i = 0; i < N; ++i) {
        // Columns 0, 1, 2 should be 2.0
        EXPECT_FLOAT_EQ(data[i * 7 + 0], 2.0f) << "Row " << i << ", col 0";
        EXPECT_FLOAT_EQ(data[i * 7 + 1], 2.0f) << "Row " << i << ", col 1";
        EXPECT_FLOAT_EQ(data[i * 7 + 2], 2.0f) << "Row " << i << ", col 2";
        // Columns 3-6 should still be 0.0
        EXPECT_FLOAT_EQ(data[i * 7 + 3], 0.0f) << "Row " << i << ", col 3";
        EXPECT_FLOAT_EQ(data[i * 7 + 4], 0.0f) << "Row " << i << ", col 4";
        EXPECT_FLOAT_EQ(data[i * 7 + 5], 0.0f) << "Row " << i << ", col 5";
        EXPECT_FLOAT_EQ(data[i * 7 + 6], 0.0f) << "Row " << i << ", col 6";
    }
}

// Test the full interleaved pattern
TEST_F(InterleavedSliceCopyTest, FullInterleavedPattern) {
    const size_t N = 5;

    // Create source tensors with distinct values
    Tensor positions = Tensor::empty({N, 3}, Device::CUDA);
    Tensor colors = Tensor::empty({N, 3}, Device::CUDA);
    Tensor indices = Tensor::empty({N, 1}, Device::CUDA);

    // Fill with known values on CPU then upload
    {
        auto pos_cpu = Tensor::empty({N, 3}, Device::CPU);
        auto col_cpu = Tensor::empty({N, 3}, Device::CPU);
        auto idx_cpu = Tensor::empty({N, 1}, Device::CPU);

        float* pos_ptr = pos_cpu.ptr<float>();
        float* col_ptr = col_cpu.ptr<float>();
        float* idx_ptr = idx_cpu.ptr<float>();

        for (size_t i = 0; i < N; ++i) {
            pos_ptr[i * 3 + 0] = static_cast<float>(i * 10 + 1); // 1, 11, 21, 31, 41
            pos_ptr[i * 3 + 1] = static_cast<float>(i * 10 + 2); // 2, 12, 22, 32, 42
            pos_ptr[i * 3 + 2] = static_cast<float>(i * 10 + 3); // 3, 13, 23, 33, 43
            col_ptr[i * 3 + 0] = static_cast<float>(i * 10 + 4); // 4, 14, 24, 34, 44
            col_ptr[i * 3 + 1] = static_cast<float>(i * 10 + 5); // 5, 15, 25, 35, 45
            col_ptr[i * 3 + 2] = static_cast<float>(i * 10 + 6); // 6, 16, 26, 36, 46
            idx_ptr[i] = static_cast<float>(i);                  // 0, 1, 2, 3, 4
        }

        positions = pos_cpu.cuda();
        colors = col_cpu.cuda();
        indices = idx_cpu.cuda();
    }

    // Build interleaved buffer using slice + copy_
    Tensor interleaved = Tensor::empty({N, 7}, Device::CUDA);
    interleaved.slice(1, 0, 3).copy_(positions);
    interleaved.slice(1, 3, 6).copy_(colors);
    interleaved.slice(1, 6, 7).copy_(indices);

    // Verify
    auto cpu = interleaved.cpu();
    const float* data = cpu.ptr<float>();

    for (size_t i = 0; i < N; ++i) {
        float expected_pos[3] = {
            static_cast<float>(i * 10 + 1),
            static_cast<float>(i * 10 + 2),
            static_cast<float>(i * 10 + 3)};
        float expected_col[3] = {
            static_cast<float>(i * 10 + 4),
            static_cast<float>(i * 10 + 5),
            static_cast<float>(i * 10 + 6)};
        float expected_idx = static_cast<float>(i);

        EXPECT_FLOAT_EQ(data[i * 7 + 0], expected_pos[0]) << "Row " << i << ", pos.x";
        EXPECT_FLOAT_EQ(data[i * 7 + 1], expected_pos[1]) << "Row " << i << ", pos.y";
        EXPECT_FLOAT_EQ(data[i * 7 + 2], expected_pos[2]) << "Row " << i << ", pos.z";
        EXPECT_FLOAT_EQ(data[i * 7 + 3], expected_col[0]) << "Row " << i << ", col.r";
        EXPECT_FLOAT_EQ(data[i * 7 + 4], expected_col[1]) << "Row " << i << ", col.g";
        EXPECT_FLOAT_EQ(data[i * 7 + 5], expected_col[2]) << "Row " << i << ", col.b";
        EXPECT_FLOAT_EQ(data[i * 7 + 6], expected_idx) << "Row " << i << ", idx";
    }
}

// Compare slice+copy vs Tensor::cat
TEST_F(InterleavedSliceCopyTest, SliceCopyVsCat) {
    const size_t N = 100;

    Tensor positions = Tensor::ones({N, 3}, Device::CUDA) * 1.0f;
    Tensor colors = Tensor::ones({N, 3}, Device::CUDA) * 2.0f;
    Tensor indices = Tensor::ones({N, 1}, Device::CUDA) * 3.0f;

    // Method 1: Tensor::cat
    Tensor cat_result = Tensor::cat({positions, colors, indices}, 1);

    // Method 2: slice + copy_
    Tensor slice_result = Tensor::empty({N, 7}, Device::CUDA);
    slice_result.slice(1, 0, 3).copy_(positions);
    slice_result.slice(1, 3, 6).copy_(colors);
    slice_result.slice(1, 6, 7).copy_(indices);

    // Compare
    auto cat_cpu = cat_result.cpu();
    auto slice_cpu = slice_result.cpu();

    const float* cat_data = cat_cpu.ptr<float>();
    const float* slice_data = slice_cpu.ptr<float>();

    for (size_t i = 0; i < N * 7; ++i) {
        EXPECT_FLOAT_EQ(cat_data[i], slice_data[i]) << "Mismatch at index " << i;
    }
}

// Test with int32 to float conversion (like transform_indices)
TEST_F(InterleavedSliceCopyTest, Int32ToFloatCopy) {
    const size_t N = 5;

    // Create int32 indices
    Tensor indices_int = Tensor::empty({N}, Device::CPU, DataType::Int32);
    int32_t* int_ptr = indices_int.ptr<int32_t>();
    for (size_t i = 0; i < N; ++i) {
        int_ptr[i] = static_cast<int32_t>(i * 10);
    }
    indices_int = indices_int.cuda();

    // Copy into float slice
    Tensor dst = Tensor::zeros({N, 7}, Device::CUDA, DataType::Float32);
    dst.slice(1, 6, 7).copy_(indices_int.unsqueeze(1));

    // Verify
    auto cpu = dst.cpu();
    const float* data = cpu.ptr<float>();

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(data[i * 7 + 6], static_cast<float>(i * 10)) << "Row " << i;
    }
}

// Test stride handling - column slices are non-contiguous
TEST_F(InterleavedSliceCopyTest, NonContiguousSliceStrides) {
    const size_t N = 10;

    Tensor t = Tensor::zeros({N, 7}, Device::CUDA);
    auto slice = t.slice(1, 0, 3);

    // Column slice should have stride of 7 in the row dimension, not 3
    // This is because it's a view into a larger tensor
    std::cout << "Slice shape: [" << slice.size(0) << ", " << slice.size(1) << "]" << std::endl;
    std::cout << "Slice strides: [" << slice.stride(0) << ", " << slice.stride(1) << "]" << std::endl;

    // The slice should still be usable for copy_
    Tensor src = Tensor::ones({N, 3}, Device::CUDA);
    slice.copy_(src);

    auto cpu = t.cpu();
    const float* data = cpu.ptr<float>();

    // Verify first 3 columns are 1.0
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(data[i * 7 + 0], 1.0f);
        EXPECT_FLOAT_EQ(data[i * 7 + 1], 1.0f);
        EXPECT_FLOAT_EQ(data[i * 7 + 2], 1.0f);
    }
}

// Test copy_ from CPU to non-contiguous CUDA slice (the actual bug: #675)
TEST_F(InterleavedSliceCopyTest, CpuToStridedCudaCopy) {
    const size_t N = 100;

    // Source: contiguous CPU tensors (like random point cloud from blender loader)
    Tensor pos_cpu = Tensor::rand({N, 3}, Device::CPU);
    Tensor col_cpu = Tensor::rand({N, 3}, Device::CPU);

    // Destination: CUDA interleaved buffer
    Tensor interleaved = Tensor::zeros({N, 7}, Device::CUDA);
    interleaved.slice(1, 0, 3).copy_(pos_cpu);
    interleaved.slice(1, 3, 6).copy_(col_cpu);

    auto result = interleaved.cpu();
    auto pos_acc = pos_cpu.accessor<float, 2>();
    auto res_acc = result.accessor<float, 2>();

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(res_acc(i, 0), pos_acc(i, 0)) << "Row " << i << " pos.x";
        EXPECT_FLOAT_EQ(res_acc(i, 1), pos_acc(i, 1)) << "Row " << i << " pos.y";
        EXPECT_FLOAT_EQ(res_acc(i, 2), pos_acc(i, 2)) << "Row " << i << " pos.z";
    }

    auto col_acc = col_cpu.accessor<float, 2>();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(res_acc(i, 3), col_acc(i, 0)) << "Row " << i << " col.r";
        EXPECT_FLOAT_EQ(res_acc(i, 4), col_acc(i, 1)) << "Row " << i << " col.g";
        EXPECT_FLOAT_EQ(res_acc(i, 5), col_acc(i, 2)) << "Row " << i << " col.b";
    }
}

// Test copy_ from CPU UInt8 to non-contiguous CUDA float slice (type + device conversion)
TEST_F(InterleavedSliceCopyTest, CpuUint8ToStridedCudaFloatCopy) {
    const size_t N = 50;

    Tensor colors_u8 = Tensor::randint({N, 3}, 0, 256, Device::CPU, DataType::UInt8);
    Tensor colors_f32 = colors_u8.to(DataType::Float32) / 255.0f;

    Tensor interleaved = Tensor::zeros({N, 7}, Device::CUDA);
    interleaved.slice(1, 3, 6).copy_(colors_f32);

    auto result = interleaved.cpu();
    auto col_acc = colors_f32.accessor<float, 2>();
    auto res_acc = result.accessor<float, 2>();

    for (size_t i = 0; i < N; ++i) {
        EXPECT_NEAR(res_acc(i, 3), col_acc(i, 0), 1e-6f) << "Row " << i << " col.r";
        EXPECT_NEAR(res_acc(i, 4), col_acc(i, 1), 1e-6f) << "Row " << i << " col.g";
        EXPECT_NEAR(res_acc(i, 5), col_acc(i, 2), 1e-6f) << "Row " << i << " col.b";
    }
}

// Large scale test
TEST_F(InterleavedSliceCopyTest, LargeScale) {
    const size_t N = 100000;

    Tensor positions = Tensor::ones({N, 3}, Device::CUDA) * 1.0f;
    Tensor colors = Tensor::ones({N, 3}, Device::CUDA) * 2.0f;
    Tensor indices = Tensor::ones({N, 1}, Device::CUDA) * 3.0f;

    Tensor interleaved = Tensor::empty({N, 7}, Device::CUDA);
    interleaved.slice(1, 0, 3).copy_(positions);
    interleaved.slice(1, 3, 6).copy_(colors);
    interleaved.slice(1, 6, 7).copy_(indices);

    // Spot check a few values
    auto cpu = interleaved.cpu();
    const float* data = cpu.ptr<float>();

    // Check first row
    EXPECT_FLOAT_EQ(data[0], 1.0f); // pos.x
    EXPECT_FLOAT_EQ(data[1], 1.0f); // pos.y
    EXPECT_FLOAT_EQ(data[2], 1.0f); // pos.z
    EXPECT_FLOAT_EQ(data[3], 2.0f); // col.r
    EXPECT_FLOAT_EQ(data[4], 2.0f); // col.g
    EXPECT_FLOAT_EQ(data[5], 2.0f); // col.b
    EXPECT_FLOAT_EQ(data[6], 3.0f); // idx

    // Check last row
    size_t last = (N - 1) * 7;
    EXPECT_FLOAT_EQ(data[last + 0], 1.0f);
    EXPECT_FLOAT_EQ(data[last + 3], 2.0f);
    EXPECT_FLOAT_EQ(data[last + 6], 3.0f);

    // Check middle row
    size_t mid = (N / 2) * 7;
    EXPECT_FLOAT_EQ(data[mid + 0], 1.0f);
    EXPECT_FLOAT_EQ(data[mid + 3], 2.0f);
    EXPECT_FLOAT_EQ(data[mid + 6], 3.0f);
}

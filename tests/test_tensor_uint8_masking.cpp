/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include <gtest/gtest.h>
#include <torch/torch.h>

using namespace lfs::core;

namespace {

    void compare_tensors(const Tensor& lfs, const torch::Tensor& ref,
                         const std::string& ctx = "", float tol = 1e-5f) {
        const auto torch_cpu = ref.cpu().contiguous().to(torch::kFloat32);
        const auto lfs_cpu = lfs.to(DataType::Float32).cpu();

        ASSERT_EQ(lfs_cpu.ndim(), static_cast<size_t>(torch_cpu.dim())) << ctx;
        for (size_t i = 0; i < lfs_cpu.ndim(); ++i) {
            ASSERT_EQ(lfs_cpu.size(i), static_cast<size_t>(torch_cpu.size(i))) << ctx << " dim " << i;
        }
        if (lfs_cpu.numel() == 0)
            return;

        const auto* lfs_ptr = lfs_cpu.ptr<float>();
        const auto torch_flat = torch_cpu.flatten();
        const auto acc = torch_flat.accessor<float, 1>();
        for (size_t i = 0; i < lfs_cpu.numel(); ++i) {
            EXPECT_NEAR(lfs_ptr[i], acc[i], tol) << ctx << " at " << i;
        }
    }

    Tensor make_uint8_mask(const std::vector<uint8_t>& values, size_t n, Device dev) {
        assert(values.size() == n);
        auto t = Tensor::empty({n}, Device::CPU, DataType::UInt8);
        auto* p = t.ptr<uint8_t>();
        for (size_t i = 0; i < n; ++i)
            p[i] = values[i];
        return (dev == Device::GPU) ? t.cuda() : t;
    }

    Tensor make_bool_mask(const std::vector<uint8_t>& values, size_t n, Device dev) {
        auto t = Tensor::from_vector(
            std::vector<bool>(values.begin(), values.end()), {n}, dev);
        return t;
    }

    torch::Tensor make_torch_bool_mask(const std::vector<uint8_t>& values) {
        auto torch_mask = torch::zeros(
            {static_cast<int64_t>(values.size())},
            torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
        auto torch_mask_cpu = torch_mask.cpu();
        auto mask_acc = torch_mask_cpu.accessor<bool, 1>();
        for (int64_t i = 0; i < static_cast<int64_t>(values.size()); ++i)
            mask_acc[i] = values[static_cast<size_t>(i)] != 0;
        return torch_mask_cpu.cuda();
    }

    torch::Tensor make_torch_uint8_tensor(const std::vector<uint8_t>& values, const c10::DeviceType device_type) {
        auto torch_values = torch::zeros(
            {static_cast<int64_t>(values.size())},
            torch::TensorOptions().dtype(torch::kUInt8).device(device_type));
        auto torch_values_cpu = torch_values.cpu();
        auto* data = torch_values_cpu.data_ptr<uint8_t>();
        for (size_t i = 0; i < values.size(); ++i)
            data[i] = values[i];
        return (device_type == c10::kCUDA) ? torch_values_cpu.cuda() : torch_values_cpu;
    }

    void compare_masked_fill_vs_torch_cuda(DataType lfs_dtype,
                                           torch::ScalarType torch_dtype,
                                           float fill_value,
                                           const std::string& ctx,
                                           float tol = 1e-5f) {
        constexpr int64_t N = 4096;
        auto lfs_values = Tensor::zeros({static_cast<size_t>(N)}, Device::GPU, lfs_dtype);
        auto torch_values = torch::zeros(
            {N}, torch::TensorOptions().dtype(torch_dtype).device(torch::kCUDA));

        std::vector<uint8_t> mask_vals(N);
        for (int64_t i = 0; i < N; ++i)
            mask_vals[i] = (i % 7 == 0 || i % 11 == 0) ? 1 : 0;

        auto lfs_mask = make_bool_mask(mask_vals, N, Device::GPU);
        auto torch_mask = make_torch_bool_mask(mask_vals);

        lfs_values.masked_fill_(lfs_mask, fill_value);
        torch_values.masked_fill_(torch_mask, fill_value);

        compare_tensors(lfs_values, torch_values, ctx, tol);
    }

} // namespace

class TensorUInt8MaskingTest : public ::testing::Test {
protected:
    void SetUp() override {
        Tensor::manual_seed(42);
        torch::manual_seed(42);
        ASSERT_TRUE(Tensor::zeros({1}, Device::GPU).is_valid());
    }
};

// ============= masked_select =============

TEST_F(TensorUInt8MaskingTest, MaskedSelectUInt8_CPU) {
    constexpr size_t N = 100;
    auto data = Tensor::randn({N}, Device::CPU);

    std::vector<uint8_t> mask_vals(N);
    for (size_t i = 0; i < N; ++i)
        mask_vals[i] = (i % 3 == 0) ? 1 : 0;

    auto uint8_mask = make_uint8_mask(mask_vals, N, Device::CPU);
    auto bool_mask = make_bool_mask(mask_vals, N, Device::CPU);

    auto result_uint8 = data.masked_select(uint8_mask);
    auto result_bool = data.masked_select(bool_mask);

    ASSERT_EQ(result_uint8.numel(), result_bool.numel());
    auto u_cpu = result_uint8.cpu();
    auto b_cpu = result_bool.cpu();
    for (size_t i = 0; i < result_uint8.numel(); ++i) {
        EXPECT_FLOAT_EQ(u_cpu.ptr<float>()[i], b_cpu.ptr<float>()[i]) << "at " << i;
    }
}

TEST_F(TensorUInt8MaskingTest, MaskedSelectUInt8_CUDA) {
    constexpr size_t N = 1000;
    auto data = Tensor::randn({N}, Device::GPU);

    std::vector<uint8_t> mask_vals(N);
    for (size_t i = 0; i < N; ++i)
        mask_vals[i] = (i % 5 == 0) ? 1 : 0;

    auto uint8_mask = make_uint8_mask(mask_vals, N, Device::GPU);
    auto bool_mask = make_bool_mask(mask_vals, N, Device::GPU);

    auto result_uint8 = data.masked_select(uint8_mask);
    auto result_bool = data.masked_select(bool_mask);

    ASSERT_EQ(result_uint8.numel(), result_bool.numel());
    auto u_cpu = result_uint8.cpu();
    auto b_cpu = result_bool.cpu();
    for (size_t i = 0; i < result_uint8.numel(); ++i) {
        EXPECT_FLOAT_EQ(u_cpu.ptr<float>()[i], b_cpu.ptr<float>()[i]) << "at " << i;
    }
}

TEST_F(TensorUInt8MaskingTest, MaskedSelectUInt8_VsTorch_CUDA) {
    constexpr int64_t N = 500;
    auto t_data = torch::randn({N}, torch::kFloat32).cuda();

    auto lfs_data = Tensor::zeros({static_cast<size_t>(N)}, Device::GPU);
    auto t_cpu = t_data.cpu().contiguous();
    cudaMemcpy(lfs_data.data_ptr(), t_cpu.data_ptr<float>(), N * sizeof(float), cudaMemcpyHostToDevice);

    std::vector<uint8_t> mask_vals(N);
    for (int64_t i = 0; i < N; ++i)
        mask_vals[i] = (i % 4 == 0) ? 1 : 0;

    auto t_bool_mask = torch::zeros({N}, torch::kBool).cuda();
    auto t_mask_cpu = t_bool_mask.cpu();
    auto mask_acc = t_mask_cpu.accessor<bool, 1>();
    for (int64_t i = 0; i < N; ++i)
        mask_acc[i] = mask_vals[i] != 0;
    t_bool_mask = t_mask_cpu.cuda();

    auto lfs_uint8_mask = make_uint8_mask(mask_vals, N, Device::GPU);

    auto t_result = torch::masked_select(t_data, t_bool_mask);
    auto lfs_result = lfs_data.masked_select(lfs_uint8_mask);

    compare_tensors(lfs_result, t_result, "masked_select uint8 vs torch");
}

TEST_F(TensorUInt8MaskingTest, EqualUInt8_VsTorch_CPU) {
    constexpr size_t N = 16;
    const std::vector<uint8_t> lhs_vals = {0, 1, 2, 3, 4, 5, 1, 9, 7, 7, 0, 8, 6, 2, 2, 3};
    const std::vector<uint8_t> rhs_vals = {0, 0, 2, 4, 4, 1, 1, 9, 7, 3, 0, 2, 6, 2, 5, 3};

    const auto lfs_lhs = make_uint8_mask(lhs_vals, N, Device::CPU);
    const auto lfs_rhs = make_uint8_mask(rhs_vals, N, Device::CPU);
    const auto torch_lhs = make_torch_uint8_tensor(lhs_vals, c10::kCPU);
    const auto torch_rhs = make_torch_uint8_tensor(rhs_vals, c10::kCPU);

    const auto lfs_eq = lfs_lhs.eq(lfs_rhs);
    const auto torch_eq = torch_lhs.eq(torch_rhs);

    compare_tensors(lfs_eq, torch_eq, "uint8 eq cpu vs torch");
    EXPECT_EQ(lfs_eq.all().item<bool>(), torch_eq.all().item<bool>());
}

TEST_F(TensorUInt8MaskingTest, EqualUInt8_VsTorch_CUDA) {
    constexpr size_t N = 4096;
    std::vector<uint8_t> lhs_vals(N);
    std::vector<uint8_t> rhs_vals(N);
    for (size_t i = 0; i < N; ++i) {
        lhs_vals[i] = static_cast<uint8_t>((i * 7) % 13);
        rhs_vals[i] = static_cast<uint8_t>((i % 5 == 0) ? lhs_vals[i] : ((i * 11) % 13));
    }

    const auto lfs_lhs = make_uint8_mask(lhs_vals, N, Device::GPU);
    const auto lfs_rhs = make_uint8_mask(rhs_vals, N, Device::GPU);
    const auto torch_lhs = make_torch_uint8_tensor(lhs_vals, c10::kCUDA);
    const auto torch_rhs = make_torch_uint8_tensor(rhs_vals, c10::kCUDA);

    const auto lfs_eq = lfs_lhs.eq(lfs_rhs);
    const auto torch_eq = torch_lhs.eq(torch_rhs);

    compare_tensors(lfs_eq, torch_eq, "uint8 eq cuda vs torch");
    EXPECT_EQ(lfs_eq.all().item<bool>(), torch_eq.all().item<bool>());
}

// ============= masked_fill_ =============

TEST_F(TensorUInt8MaskingTest, MaskedFillInPlaceUInt8_CPU) {
    constexpr size_t N = 100;
    auto data_uint8 = Tensor::randn({N}, Device::CPU);
    auto data_bool = data_uint8.clone();

    std::vector<uint8_t> mask_vals(N);
    for (size_t i = 0; i < N; ++i)
        mask_vals[i] = (i % 2 == 0) ? 1 : 0;

    auto uint8_mask = make_uint8_mask(mask_vals, N, Device::CPU);
    auto bool_mask = make_bool_mask(mask_vals, N, Device::CPU);

    data_uint8.masked_fill_(uint8_mask, -999.0f);
    data_bool.masked_fill_(bool_mask, -999.0f);

    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(data_uint8.ptr<float>()[i], data_bool.ptr<float>()[i]) << "at " << i;
    }
}

TEST_F(TensorUInt8MaskingTest, MaskedFillInPlaceUint8Target_CPU) {
    constexpr size_t N = 8;
    auto values = Tensor::zeros({N}, Device::CPU, DataType::UInt8);

    std::vector<uint8_t> mask_vals = {0, 1, 0, 1, 1, 0, 0, 1};
    auto bool_mask = make_bool_mask(mask_vals, N, Device::CPU);

    values.masked_fill_(bool_mask, 3.0f);

    const auto* data = values.ptr<uint8_t>();
    const std::vector<uint8_t> expected = {0, 3, 0, 3, 3, 0, 0, 3};
    for (size_t i = 0; i < N; ++i) {
        EXPECT_EQ(data[i], expected[i]) << "at " << i;
    }
}

TEST_F(TensorUInt8MaskingTest, MaskedFillInPlaceUint8Target_VsTorch_CUDA) {
    compare_masked_fill_vs_torch_cuda(
        DataType::UInt8, torch::kUInt8, 9.0f, "masked_fill_ uint8 target vs torch");
}

TEST_F(TensorUInt8MaskingTest, MaskedFillInPlaceInt32Target_VsTorch_CUDA) {
    compare_masked_fill_vs_torch_cuda(
        DataType::Int32, torch::kInt32, -17.0f, "masked_fill_ int32 target vs torch");
}

TEST_F(TensorUInt8MaskingTest, MaskedFillInPlaceInt64Target_VsTorch_CUDA) {
    compare_masked_fill_vs_torch_cuda(
        DataType::Int64, torch::kInt64, 23.0f, "masked_fill_ int64 target vs torch");
}

TEST_F(TensorUInt8MaskingTest, MaskedFillInPlaceFloat16Target_VsTorch_CUDA) {
    compare_masked_fill_vs_torch_cuda(
        DataType::Float16, torch::kFloat16, 3.25f, "masked_fill_ float16 target vs torch", 1e-3f);
}

TEST_F(TensorUInt8MaskingTest, MaskedFillInPlaceUInt8_CUDA) {
    constexpr size_t N = 1000;
    auto data_uint8 = Tensor::randn({N}, Device::GPU);
    auto data_bool = data_uint8.clone();

    std::vector<uint8_t> mask_vals(N);
    for (size_t i = 0; i < N; ++i)
        mask_vals[i] = (i % 3 == 0) ? 1 : 0;

    auto uint8_mask = make_uint8_mask(mask_vals, N, Device::GPU);
    auto bool_mask = make_bool_mask(mask_vals, N, Device::GPU);

    data_uint8.masked_fill_(uint8_mask, 42.0f);
    data_bool.masked_fill_(bool_mask, 42.0f);

    auto u_cpu = data_uint8.cpu();
    auto b_cpu = data_bool.cpu();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(u_cpu.ptr<float>()[i], b_cpu.ptr<float>()[i]) << "at " << i;
    }
}

TEST_F(TensorUInt8MaskingTest, MaskedFillUInt8_CUDA) {
    constexpr size_t N = 500;
    auto data = Tensor::randn({N}, Device::GPU);

    std::vector<uint8_t> mask_vals(N);
    for (size_t i = 0; i < N; ++i)
        mask_vals[i] = (i < N / 2) ? 1 : 0;

    auto uint8_mask = make_uint8_mask(mask_vals, N, Device::GPU);
    auto result = data.masked_fill(uint8_mask, 0.0f);

    auto r_cpu = result.cpu();
    auto d_cpu = data.cpu();
    for (size_t i = 0; i < N; ++i) {
        if (i < N / 2) {
            EXPECT_FLOAT_EQ(r_cpu.ptr<float>()[i], 0.0f) << "at " << i;
        } else {
            EXPECT_FLOAT_EQ(r_cpu.ptr<float>()[i], d_cpu.ptr<float>()[i]) << "at " << i;
        }
    }
}

// ============= index_select with bool-like UInt8 mask =============

TEST_F(TensorUInt8MaskingTest, IndexSelectBoolLikeUInt8_CPU) {
    constexpr size_t N = 50;
    auto data = Tensor::randn({N, 3}, Device::CPU);

    std::vector<uint8_t> mask_vals(N);
    for (size_t i = 0; i < N; ++i)
        mask_vals[i] = (i % 4 == 0) ? 1 : 0;

    auto uint8_mask = make_uint8_mask(mask_vals, N, Device::CPU);
    auto bool_mask = make_bool_mask(mask_vals, N, Device::CPU);

    auto result_uint8 = data.index_select(0, uint8_mask);
    auto result_bool = data.index_select(0, bool_mask);

    ASSERT_EQ(result_uint8.size(0), result_bool.size(0));
    ASSERT_EQ(result_uint8.size(1), 3);

    for (size_t i = 0; i < result_uint8.numel(); ++i) {
        EXPECT_FLOAT_EQ(result_uint8.ptr<float>()[i], result_bool.ptr<float>()[i]) << "at " << i;
    }
}

TEST_F(TensorUInt8MaskingTest, IndexSelectBoolLikeUInt8_CUDA) {
    constexpr size_t N = 200;
    auto data = Tensor::randn({N, 3}, Device::GPU);

    std::vector<uint8_t> mask_vals(N);
    for (size_t i = 0; i < N; ++i)
        mask_vals[i] = (i % 7 == 0) ? 1 : 0;

    auto uint8_mask = make_uint8_mask(mask_vals, N, Device::GPU);
    auto bool_mask = make_bool_mask(mask_vals, N, Device::GPU);

    auto result_uint8 = data.index_select(0, uint8_mask);
    auto result_bool = data.index_select(0, bool_mask);

    ASSERT_EQ(result_uint8.size(0), result_bool.size(0));
    ASSERT_EQ(result_uint8.size(1), 3);

    auto u_cpu = result_uint8.cpu();
    auto b_cpu = result_bool.cpu();
    for (size_t i = 0; i < result_uint8.numel(); ++i) {
        EXPECT_FLOAT_EQ(u_cpu.ptr<float>()[i], b_cpu.ptr<float>()[i]) << "at " << i;
    }
}

// ============= MaskedTensorProxy (operator[]) =============

TEST_F(TensorUInt8MaskingTest, MaskedProxyScalarAssign_CUDA) {
    constexpr size_t N = 100;
    auto data_uint8 = Tensor::ones({N}, Device::GPU);
    auto data_bool = Tensor::ones({N}, Device::GPU);

    std::vector<uint8_t> mask_vals(N);
    for (size_t i = 0; i < N; ++i)
        mask_vals[i] = (i >= 50) ? 1 : 0;

    auto uint8_mask = make_uint8_mask(mask_vals, N, Device::GPU);
    auto bool_mask = make_bool_mask(mask_vals, N, Device::GPU);

    data_uint8[uint8_mask] = 0.0f;
    data_bool[bool_mask] = 0.0f;

    auto u_cpu = data_uint8.cpu();
    auto b_cpu = data_bool.cpu();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(u_cpu.ptr<float>()[i], b_cpu.ptr<float>()[i]) << "at " << i;
    }
}

TEST_F(TensorUInt8MaskingTest, MaskedProxyTensorAssign_CUDA) {
    constexpr size_t N = 100;
    auto data_uint8 = Tensor::zeros({N}, Device::GPU);
    auto data_bool = Tensor::zeros({N}, Device::GPU);

    std::vector<uint8_t> mask_vals(N);
    size_t true_count = 0;
    for (size_t i = 0; i < N; ++i) {
        mask_vals[i] = (i % 2 == 0) ? 1 : 0;
        if (mask_vals[i])
            true_count++;
    }

    auto uint8_mask = make_uint8_mask(mask_vals, N, Device::GPU);
    auto bool_mask = make_bool_mask(mask_vals, N, Device::GPU);

    auto values = Tensor::ones({true_count}, Device::GPU) * 7.0f;

    data_uint8[uint8_mask] = values;
    data_bool[bool_mask] = values;

    auto u_cpu = data_uint8.cpu();
    auto b_cpu = data_bool.cpu();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(u_cpu.ptr<float>()[i], b_cpu.ptr<float>()[i]) << "at " << i;
    }
}

// ============= count_nonzero =============

TEST_F(TensorUInt8MaskingTest, CountNonzeroUInt8_CPU) {
    constexpr size_t N = 256;
    auto t = Tensor::empty({N}, Device::CPU, DataType::UInt8);
    auto* p = t.ptr<uint8_t>();
    size_t expected = 0;
    for (size_t i = 0; i < N; ++i) {
        p[i] = (i % 3 != 0) ? 1 : 0;
        if (p[i])
            expected++;
    }

    EXPECT_EQ(t.count_nonzero(), expected);
}

TEST_F(TensorUInt8MaskingTest, CountNonzeroUInt8_CUDA) {
    constexpr size_t N = 1000;
    auto t_cpu = Tensor::empty({N}, Device::CPU, DataType::UInt8);
    auto* p = t_cpu.ptr<uint8_t>();
    size_t expected = 0;
    for (size_t i = 0; i < N; ++i) {
        p[i] = (i % 5 != 0) ? 1 : 0;
        if (p[i])
            expected++;
    }

    auto t_cuda = t_cpu.cuda();
    EXPECT_EQ(t_cuda.count_nonzero(), expected);
}

// ============= Edge cases =============

TEST_F(TensorUInt8MaskingTest, MaskedSelectAllTrue_CUDA) {
    constexpr size_t N = 200;
    auto data = Tensor::randn({N}, Device::GPU);

    std::vector<uint8_t> mask_vals(N, 1);
    auto uint8_mask = make_uint8_mask(mask_vals, N, Device::GPU);

    auto result = data.masked_select(uint8_mask);
    ASSERT_EQ(result.numel(), N);

    auto r_cpu = result.cpu();
    auto d_cpu = data.cpu();
    for (size_t i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(r_cpu.ptr<float>()[i], d_cpu.ptr<float>()[i]) << "at " << i;
    }
}

TEST_F(TensorUInt8MaskingTest, MaskedSelectAllFalse_CUDA) {
    constexpr size_t N = 200;
    auto data = Tensor::randn({N}, Device::GPU);

    std::vector<uint8_t> mask_vals(N, 0);
    auto uint8_mask = make_uint8_mask(mask_vals, N, Device::GPU);

    auto result = data.masked_select(uint8_mask);
    ASSERT_EQ(result.numel(), 0);
}

TEST_F(TensorUInt8MaskingTest, ConsistencyBoolVsUInt8_CUDA) {
    constexpr size_t N = 500;
    auto data = Tensor::randn({N, 4}, Device::GPU);

    std::vector<uint8_t> mask_vals(N);
    for (size_t i = 0; i < N; ++i)
        mask_vals[i] = (i * 7 + 3) % 11 < 5 ? 1 : 0;

    auto uint8_mask = make_uint8_mask(mask_vals, N, Device::GPU);
    auto bool_mask = make_bool_mask(mask_vals, N, Device::GPU);

    // masked_select on flattened
    auto flat = data.flatten();
    std::vector<uint8_t> flat_mask(N * 4);
    for (size_t i = 0; i < N * 4; ++i)
        flat_mask[i] = mask_vals[i / 4];
    auto flat_u8_mask = make_uint8_mask(flat_mask, N * 4, Device::GPU);
    auto flat_bool_mask = make_bool_mask(flat_mask, N * 4, Device::GPU);

    auto sel_u8 = flat.masked_select(flat_u8_mask);
    auto sel_bool = flat.masked_select(flat_bool_mask);
    ASSERT_EQ(sel_u8.numel(), sel_bool.numel());

    // index_select row selection
    auto rows_u8 = data.index_select(0, uint8_mask);
    auto rows_bool = data.index_select(0, bool_mask);
    ASSERT_EQ(rows_u8.size(0), rows_bool.size(0));
    ASSERT_EQ(rows_u8.size(1), 4);

    auto u_cpu = rows_u8.cpu();
    auto b_cpu = rows_bool.cpu();
    for (size_t i = 0; i < rows_u8.numel(); ++i) {
        EXPECT_FLOAT_EQ(u_cpu.ptr<float>()[i], b_cpu.ptr<float>()[i]) << "at " << i;
    }

    // count_nonzero
    EXPECT_EQ(uint8_mask.count_nonzero(), bool_mask.count_nonzero());

    // masked_fill_
    auto fill_u8 = data.clone();
    auto fill_bool = data.clone();
    fill_u8.masked_fill_(flat_u8_mask.reshape({static_cast<int>(N), 4}), -1.0f);
    fill_bool.masked_fill_(flat_bool_mask.reshape({static_cast<int>(N), 4}), -1.0f);

    auto fu_cpu = fill_u8.cpu();
    auto fb_cpu = fill_bool.cpu();
    for (size_t i = 0; i < data.numel(); ++i) {
        EXPECT_FLOAT_EQ(fu_cpu.ptr<float>()[i], fb_cpu.ptr<float>()[i]) << "at " << i;
    }
}

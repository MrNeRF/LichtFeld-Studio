/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor.hpp"
#include "core/tensor/backend/cuda/runtime/cuda_memory_guard.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <numeric>
#include <vector>

namespace {

    using namespace lfs::core;

    bool has_cuda_device() {
        int device_count = 0;
        return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    }

    void expect_values(const Tensor& tensor, const std::vector<float>& expected) {
        EXPECT_EQ(tensor.device(), Device::CUDA);
        const std::vector<float> actual = tensor.to_vector();
        ASSERT_EQ(actual.size(), expected.size());
        for (size_t i = 0; i < actual.size(); ++i) {
            EXPECT_FLOAT_EQ(actual[i], expected[i]) << "index " << i;
        }
    }

    Tensor cuda_indices(std::vector<int> values) {
        return Tensor::from_vector(values, {values.size()}, Device::CUDA);
    }

    TEST(TensorBackendFacadeC, IndexAdaptersPreserveValuesAndDuplicateAccumulation) {
        if (!has_cuda_device())
            GTEST_SKIP() << "CUDA device required";
        const Tensor matrix = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4, 5, 6}, {3, 2}, Device::CUDA);

        // Catches gather/index_select/take descriptors dropping sizes, dimensions, or modes.
        expect_values(matrix.gather(0, cuda_indices({2, 0, 1, 1, 0, 2}).reshape({3, 2})),
                      {5, 2, 3, 4, 1, 6});
        expect_values(matrix.index_select(0, cuda_indices({2, 0})), {5, 6, 1, 2});
        expect_values(matrix.take(cuda_indices({5, 0, 3})), {6, 1, 4});

        // Catches scatter/index_copy writing the input instead of the destination descriptor.
        Tensor scattered = Tensor::zeros({4}, Device::CUDA);
        scattered.scatter_(0, cuda_indices({3, 1}),
                           Tensor::from_vector(std::vector<float>{8, 9}, {2}, Device::CUDA));
        expect_values(scattered, {0, 9, 0, 8});

        Tensor copied = Tensor::zeros({4}, Device::CUDA);
        copied.index_copy_(0, cuda_indices({2, 0}),
                           Tensor::from_vector(std::vector<float>{7, 6}, {2}, Device::CUDA));
        expect_values(copied, {6, 0, 7, 0});

        // Catches index_add selecting non-atomic scatter semantics for duplicate indices.
        Tensor accumulated = Tensor::zeros({3}, Device::CUDA);
        accumulated.index_add_(
            0, cuda_indices({1, 1, 2}),
            Tensor::from_vector(std::vector<float>{2, 3, 4}, {3}, Device::CUDA));
        expect_values(accumulated, {0, 5, 4});
    }

    TEST(TensorBackendFacadeC, MaskAdaptersPreserveCountsValuesAndBackend) {
        if (!has_cuda_device())
            GTEST_SKIP() << "CUDA device required";
        const Tensor mask = Tensor::from_vector(
            std::vector<bool>{true, false, true, false}, {4}, Device::CUDA);

        // Catches masked_fill/masked_select losing the selected-count contract.
        Tensor filled = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4}, {4}, Device::CUDA);
        filled.masked_fill_(mask, -1.0f);
        expect_values(filled, {-1, 2, -1, 4});
        expect_values(filled.masked_select(mask), {-1, -1});

        // Catches the masked-scatter facade swapping source and destination storage.
        Tensor destination = Tensor::zeros({4}, Device::CUDA);
        destination[mask] = Tensor::from_vector(
            std::vector<float>{10, 20}, {2}, Device::CUDA);
        expect_values(destination, {10, 0, 20, 0});

        // Catches and_live using Bool values as four-byte elements.
        Tensor live = Tensor::from_vector(
            std::vector<bool>{true, false, false, true}, {4}, Device::CUDA);
        Tensor combined = mask.clone();
        combined.and_live_(live);
        EXPECT_EQ(combined.to_vector_bool(),
                  (std::vector<bool>{true, false, false, false}));

        // Catches where broadcasting descriptors or nonzero result counts being truncated.
        const Tensor selected = Tensor::where(
            mask,
            Tensor::from_vector(std::vector<float>{1, 2, 3, 4}, {4}, Device::CUDA),
            Tensor::full({4}, 9.0f, Device::CUDA));
        expect_values(selected, {1, 9, 3, 9});
        EXPECT_EQ(selected.gt(3.0f).nonzero().to_vector_int64(),
                  (std::vector<int64_t>{1, 3}));
        EXPECT_EQ(mask.nonzero().to_vector_int64(),
                  (std::vector<int64_t>{0, 2}));
    }

    TEST(TensorBackendFacadeC, StridedMovementAdaptersKeepLogicalLayouts) {
        if (!has_cuda_device())
            GTEST_SKIP() << "CUDA device required";
        const Tensor base = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, Device::CUDA);

        // Catches strided_copy/immediate ignoring rank-2 source strides.
        expect_values(base.transpose(0, 1).contiguous(), {1, 4, 2, 5, 3, 6});

        std::vector<float> rank_five_values(32);
        std::iota(rank_five_values.begin(), rank_five_values.end(), 0.0f);
        const Tensor rank_five = Tensor::from_vector(
            rank_five_values, {2, 2, 2, 2, 2}, Device::CUDA);
        EXPECT_EQ(rank_five.transpose(0, 4).contiguous().numel(), 32u);

        // Catches strided_upload treating pageable host strides as dense.
        const Tensor cpu = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, Device::CPU);
        expect_values(cpu.transpose(0, 1).to(Device::CUDA), {1, 4, 2, 5, 3, 6});

        // Catches strided scatter and fill losing destination offsets.
        Tensor scatter_base = Tensor::zeros({3, 3}, Device::CUDA);
        Tensor scatter_view = scatter_base.slice(1, 1, 2);
        scatter_view.copy_from(Tensor::from_vector(
            std::vector<float>{7, 8, 9}, {3, 1}, Device::CUDA));
        scatter_view.fill_(5.0f);
        expect_values(scatter_base, {0, 5, 0, 0, 5, 0, 0, 5, 0});
    }

    TEST(TensorBackendFacadeC, CatPadAndGrowthAdaptersPreserveStorageRules) {
        if (!has_cuda_device())
            GTEST_SKIP() << "CUDA device required";
        const Tensor lhs = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4}, {2, 2}, Device::CUDA);
        const Tensor rhs = Tensor::from_vector(
            std::vector<float>{5, 6}, {2, 1}, Device::CUDA);

        // Catches last- and middle-dimension cat metadata staying in host code.
        expect_values(Tensor::cat({lhs, rhs}, 1), {1, 2, 5, 3, 4, 6});
        const Tensor cube = Tensor::from_vector(
            std::vector<float>{1, 2, 3, 4}, {1, 2, 2}, Device::CUDA);
        expect_values(Tensor::cat({cube, cube}, 1),
                      {1, 2, 3, 4, 1, 2, 3, 4});

        // Catches pad descriptors dropping the source strides or before-padding array.
        MovementArgs pad_args;
        pad_args.args = std::vector<std::pair<int, int>>{{1, 0}, {1, 1}};
        expect_values(lhs.movement(MovementOp::Pad, pad_args),
                      {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0});

        // Catches allocation/copy services breaking reserve ownership and stale-view checks.
        Tensor growing = lhs.clone();
        Tensor stale = growing.slice(0, 0, 1);
        growing.reserve(8);
        EXPECT_THROW((void)stale.ptr<float>(), std::runtime_error);
        growing.append_gather(cuda_indices({1}));
        growing.append_zeros(1);
        EXPECT_EQ(growing.capacity(), 8u);
        expect_values(growing, {1, 2, 3, 4, 3, 4, 0, 0});

        // Catches in-place cat deallocating or replacing the reserved first storage.
        Tensor reserved = lhs.clone();
        reserved.reserve(6);
        const Tensor tail = Tensor::from_vector(
            std::vector<float>{5, 6}, {1, 2}, Device::CUDA);
        const Tensor cat = Tensor::cat({reserved, tail}, 0);
        EXPECT_EQ(cat.capacity(), 6u);
        expect_values(cat, {1, 2, 3, 4, 5, 6});
    }

    TEST(TensorBackendFacadeC, CopyScalarAndScratchServicesPreserveBoundaries) {
        if (!has_cuda_device())
            GTEST_SKIP() << "CUDA device required";
        const Tensor scalar = Tensor::from_vector(
            std::vector<int>{42}, {1}, Device::CUDA);

        // Catches exported scalar reads bypassing synchronization or using a wrong byte size.
        EXPECT_EQ(scalar.item<int>(), 42);
        const Tensor rows = Tensor::from_vector(
            std::vector<int>{7, 8}, {2}, Device::CUDA);
        EXPECT_EQ(rows[1].item_as<int>(), 8);

        // Catches D2H helpers reading physical rather than logical order.
        const Tensor strided = Tensor::from_vector(
                                   std::vector<float>{1, 2, 3, 4, 5, 6},
                                   {2, 3}, Device::CUDA)
                                   .transpose(0, 1);
        EXPECT_EQ(strided.to_vector(), (std::vector<float>{1, 4, 2, 5, 3, 6}));

        // Catches direct and cross-device copy services selecting the wrong direction.
        expect_values(strided.clone(), {1, 4, 2, 5, 3, 6});
        Tensor upload_target = Tensor::zeros({2, 3}, Device::CUDA);
        upload_target.copy_from(Tensor::from_vector(
            std::vector<float>{9, 8, 7, 6, 5, 4}, {2, 3}, Device::CPU));
        expect_values(upload_target, {9, 8, 7, 6, 5, 4});

        // Catches CudaDeviceMemory retaining raw allocation/copy/free calls.
        CudaDeviceMemory<int> scratch(3);
        ASSERT_TRUE(scratch.valid());
        const int source[] = {3, 1, 4};
        int destination[] = {0, 0, 0};
        EXPECT_EQ(scratch.copy_from_host(source, 3), cudaSuccess);
        EXPECT_EQ(scratch.copy_to_host(destination, 3), cudaSuccess);
        EXPECT_EQ(std::vector<int>(destination, destination + 3),
                  (std::vector<int>{3, 1, 4}));
    }

} // namespace

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "cuda_backend_test.hpp"

#include "core/tensor_backend.hpp"
#include "core/tensor_cuda_interop.hpp"
#include "core/tensor_readback.hpp"
#include "cuda_stream_gate.hpp"
#include "rendering/selection_ops.hpp"
#include "selection/selection_group_mask.hpp"

#include <gtest/gtest.h>

#include <optional>

namespace {
    using namespace lfs::core;

    class SelectionGroupMaskBackends : public testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!gpu_backend_available(GetParam()))
                GTEST_SKIP() << "Backend unavailable";
            scope.emplace(GetParam());
        }
        std::optional<GpuBackendScope> scope;
    };

    TEST_P(SelectionGroupMaskBackends, CachedVersionsKeepTheirFlagsAndStorageBackend) {
        Scene scene;
        for (int group = 2; group < 256; ++group)
            ASSERT_EQ(scene.addSelectionGroup("group", glm::vec3(1, 0, 0)), group);
        for (int group : {1, 31, 32, 63, 128, 255})
            scene.setSelectionGroupLocked(group, true);
        Tensor selected = Tensor::full_bool({3}, true, Device::GPU);
        Tensor cache;
        lfs::vis::selection::LockedGroupMask words{};
        bool valid = false;
        const GpuBackendScope opposite(GetParam() == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA);
        const Tensor first = lfs::vis::selection::update_locked_group_mask(scene, selected, cache, words, valid);
        EXPECT_EQ(gpu_backend_of(first), GetParam());
        std::vector<bool> expected(256, false);
        for (int group : {1, 31, 32, 63, 128, 255})
            expected[group] = true;
        EXPECT_EQ(first.to_vector_bool(), expected);
        const Tensor unchanged = lfs::vis::selection::update_locked_group_mask(scene, selected, cache, words, valid);
        EXPECT_EQ(unchanged.to_vector_bool(), expected);
        scene.setSelectionGroupLocked(32, false);
        scene.setSelectionGroupLocked(254, true);
        const Tensor second = lfs::vis::selection::update_locked_group_mask(scene, selected, cache, words, valid);
        EXPECT_EQ(first.to_vector_bool(), expected);
        expected[32] = false;
        expected[254] = true;
        EXPECT_EQ(gpu_backend_of(second), GetParam());
        EXPECT_EQ(second.to_vector_bool(), expected);
    }

    TEST_P(SelectionGroupMaskBackends, LockedByteBoundariesPreserveTheirGroups) {
        const auto selected = Tensor::full_bool({4}, true, Device::GPU);
        const auto existing = Tensor::from_vector(std::vector<int>{255, 31, 32, 0}, {4}, Device::GPU).to(DataType::UInt8);
        std::vector<bool> flags(256, false);
        for (int group : {255, 31, 32})
            flags[group] = true;
        const auto locks = Tensor::from_vector(flags, {256}, Device::GPU);
        auto output = Tensor::empty({4}, Device::GPU, DataType::UInt8);
        lfs::rendering::apply_selection_group_tensor_mask(selected, existing, output, 1, locks, true, nullptr, {});
        EXPECT_EQ(output.to_vector_uint8(), (std::vector<uint8_t>{255, 31, 32, 1}));
    }

    TEST_P(SelectionGroupMaskBackends, EmptySelectionClearsReusedHistogram) {
        const auto mask = Tensor::full({19}, 3.f, Device::GPU, DataType::UInt8);
        Tensor scratch;
        lfs::rendering::count_selection_groups_async(mask, scratch);
        EXPECT_EQ(scratch.to_vector_int()[3], 19);
        for (const Tensor& empty : {Tensor::empty({0}, Device::GPU, DataType::UInt8), Tensor{}}) {
            scratch.fill_(9.f);
            lfs::rendering::count_selection_groups_async(empty, scratch);
            lfs::core::TensorReadback readback;
            readback.enqueue(scratch);
            std::array<int, 256> counts;
            counts.fill(-1);
            readback.wait(std::as_writable_bytes(std::span(counts)));
            EXPECT_EQ(counts, (std::array<int, 256>{}));
        }
    }

    TEST_P(SelectionGroupMaskBackends, RejectsMalformedLockFlags) {
        Tensor selected = Tensor::full_bool({2}, true, Device::GPU);
        Tensor existing;
        Tensor output = Tensor::empty({2}, Device::GPU, DataType::UInt8);
        for (Tensor invalid : {Tensor::zeros({256}, Device::GPU, DataType::Int32),
                               Tensor::zeros({8}, Device::GPU, DataType::Bool),
                               Tensor::zeros({256}, Device::CPU, DataType::Bool)}) {
            EXPECT_THROW(lfs::rendering::apply_selection_group_tensor_mask(
                             selected, existing, output, 1, invalid, true, nullptr, {}),
                         std::invalid_argument);
        }
    }

    INSTANTIATE_TEST_SUITE_P(Backends, SelectionGroupMaskBackends,
                             testing::ValuesIn(kGpuBackends),
                             [](const testing::TestParamInfo<GpuBackend>& info) {
                                 return info.param == GpuBackend::CUDA ? "Cuda" : "Vulkan";
                             });

    class SelectionGroupMaskCuda : public lfs::test::CudaBackendTest {};

    TEST_F(SelectionGroupMaskCuda, PendingSelectionRetainsLockStateAcrossUpdates) {
        cudaStream_t stream = nullptr;
        ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
        {
            Scene scene;
            ASSERT_EQ(scene.addSelectionGroup("Protected", glm::vec3(1, 0, 0)), 2);
            scene.setSelectionGroupLocked(2, true);
            Tensor cache;
            lfs::vis::selection::LockedGroupMask words{};
            bool valid = false;
            Tensor selected = Tensor::full_bool({16}, true, Device::GPU);
            Tensor existing = Tensor::full({16}, 2, Device::GPU, DataType::UInt8);
            Tensor output = Tensor::zeros({16}, Device::GPU, DataType::UInt8);
            auto locked = lfs::vis::selection::update_locked_group_mask(scene, selected, cache, words, valid);
            ASSERT_TRUE(locked.is_valid());
            (void)selected.data_ptr();
            (void)existing.data_ptr();
            (void)output.data_ptr();
            for (bool state : {false, true}) {
                scene.setSelectionGroupLocked(2, state);
                lfs::vis::selection::update_locked_group_mask(scene, selected, cache, words, valid);
            }
            {
                const CUDAStreamGuard stream_scope(stream);
                lfs::rendering::apply_selection_group_tensor_mask(
                    selected, existing, output, 1, locked, true, nullptr, {});
            }
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            lfs::test::CudaStreamGate gate;
            ASSERT_EQ(gate.block(stream), cudaSuccess);
            {
                const CUDAStreamGuard stream_scope(stream);
                lfs::rendering::apply_selection_group_tensor_mask(
                    selected, existing, output, 1, locked, true, nullptr, {});
            }
            EXPECT_TRUE(gate.entered());
            scene.setSelectionGroupLocked(2, false);
            auto unlocked = lfs::vis::selection::update_locked_group_mask(scene, selected, cache, words, valid);
            EXPECT_TRUE(unlocked.is_valid());
            EXPECT_EQ(cudaStreamSynchronize(nullptr), cudaSuccess);
            EXPECT_FALSE(gate.released()) << "Update did not finish while the selection was pending";
            gate.release();
            EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
            EXPECT_EQ(output.cpu().to_vector_uint8(), std::vector<uint8_t>(16, 2));
        }
        release_cuda_stream(stream);
        EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
} // namespace

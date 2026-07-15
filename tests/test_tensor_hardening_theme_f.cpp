/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "tensor_hardening_test_utils.hpp"

#include "core/pinned_memory_allocator.hpp"
#include "core/tensor/internal/lazy_executor.hpp"
#include "core/tensor/internal/lazy_ir.hpp"

#include <limits>
#include <optional>

using namespace lfs::core;
using namespace tensor_hardening;

namespace {

    class LazyStateGuard {
    public:
        LazyStateGuard() {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_size_heuristic_override_for_testing(false);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
        }

        ~LazyStateGuard() {
            internal::clear_lazy_ir_for_testing();
            internal::lazy_executor_clear_registry_for_testing();
            internal::lazy_executor_reset_diagnostics_for_testing();
            internal::lazy_executor_set_size_heuristic_override_for_testing(std::nullopt);
            internal::lazy_executor_set_size_threshold_override_for_testing(std::nullopt);
        }
    };

    class PinnedFallbackGuard {
    public:
        PinnedFallbackGuard() {
            auto& allocator = PinnedMemoryAllocator::instance();
            previous_enabled_ = allocator.is_enabled();
            allocator.set_enabled(true);
            allocator.set_force_fallback_for_testing(true);
        }

        ~PinnedFallbackGuard() {
            auto& allocator = PinnedMemoryAllocator::instance();
            allocator.set_force_fallback_for_testing(false);
            allocator.set_enabled(previous_enabled_);
        }

    private:
        bool previous_enabled_ = false;
    };

} // namespace

TEST_F(CudaTest, F1_DeferredExpressionSnapshotsSourceBeforeMutation_CPUAndCUDA) {
    for (const Device device : {Device::CPU, Device::CUDA}) {
        LazyStateGuard guard;
        auto source = Tensor::ones({1024}, device);
        const auto deferred = source.add(1.0f);
        ASSERT_TRUE(deferred.has_lazy_expr());
        source.fill_(5.0f);

        auto torch_source = torch::ones({1024});
        if (device == Device::CUDA) {
            torch_source = torch_source.cuda();
        }
        const auto torch_result = torch_source.add(1.0f);
        torch_source.fill_(5.0f);
        expect_float_values_match(
            deferred, torch_result,
            device == Device::CPU ? "F1 deferred source CPU" : "F1 deferred source CUDA");
    }
}

TEST_F(CudaTest, F2_CUDARowReferencesDoNotAliasReusableStaging) {
    auto ours = lfs_float_tensor({1.0f, 2.0f}, {1, 2}, Device::CUDA);
    {
        auto row = ours[0];
        float& first = row[0];
        float& second = row[1];
        static_cast<void>(second);
        first = 5.0f;
    }

    auto theirs = torch::tensor({1.0f, 2.0f}, torch::TensorOptions().device(torch::kCUDA))
                      .reshape({1, 2});
    theirs.index_put_({0, 0}, 5.0f);
    expect_float_values_match(ours, theirs, "F2 CUDA row proxy references");
}

TEST(HardeningThemeF_Misc, F3_Rank11WhereDoesNotOverwriteFixedMetadataArrays) {
    const auto torch_result = torch::where(
        torch::ones({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2},
                    torch::TensorOptions().dtype(torch::kBool)),
        torch::ones({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2},
                    torch::TensorOptions()),
        torch::zeros({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2}));
    ASSERT_TRUE(torch_result.eq(1.0f).all().item<bool>());

    EXPECT_THROW(
        static_cast<void>(TensorShape(
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2})),
        std::exception);
}

TEST(HardeningThemeF_Misc, F3_Rank11GatherDoesNotOverwriteFixedMetadataArrays) {
    EXPECT_THROW(
        static_cast<void>(Tensor::ones(
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2}, Device::CPU)),
        std::exception);
}

TEST_F(CudaTest, F4_UnpinnedFallbackStridedUploadStagesSafely) {
    const auto torch_result = torch::tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f})
                                  .reshape({2, 3})
                                  .transpose(0, 1)
                                  .cuda()
                                  .contiguous();
    ASSERT_EQ(torch_result.numel(), 6);
    const std::vector<float> expected = {1, 4, 2, 5, 3, 6};

    PinnedFallbackGuard fallback_guard;
    const auto input = Tensor::from_vector(
        std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}, Device::CPU);
    const auto values = input.transpose(0, 1).cuda().cpu().to_vector();
    EXPECT_EQ(values, expected);
}

TEST_F(CudaTest, F5_NegativeScatterIndicesMatchAcrossCPUCUDAAndTorch) {
    const auto torch_index = torch::tensor({-1}, torch::TensorOptions().dtype(torch::kInt64));
    const auto torch_source = torch::tensor({7.0f});
    auto torch_expected = torch::zeros({3});
    bool torch_threw = false;
    try {
        torch_expected.scatter_(0, torch_index, torch_source);
    } catch (const c10::Error&) {
        torch_threw = true;
    }

    for (const Device device : {Device::CPU, Device::CUDA}) {
        auto ours = Tensor::zeros({3}, device);
        const auto index = lfs_int_tensor({-1}, {1}, device);
        const auto source = lfs_float_tensor({7.0f}, {1}, device);
        bool ours_threw = false;
        try {
            ours.scatter_(0, index, source);
        } catch (const std::exception&) {
            ours_threw = true;
        }
        EXPECT_EQ(ours_threw, torch_threw)
            << (device == Device::CPU ? "F5 CPU exception contract" : "F5 CUDA exception contract");
        if (!torch_threw) {
            auto expected = torch_expected;
            if (device == Device::CUDA) {
                expected = expected.cuda();
            }
            expect_float_values_match(
                ours, expected,
                device == Device::CPU ? "F5 negative scatter CPU" : "F5 negative scatter CUDA");
        }
    }
}

TEST(HardeningThemeF_Misc, F6_IncompatibleEmptyBroadcastRejectsLikeTorch) {
    const auto torch_input = torch::empty({0, 2});
    EXPECT_THROW(static_cast<void>(torch_input.expand({3, 4})), c10::Error);
    const auto ours_input = Tensor::empty({0, 2}, Device::CPU);
    EXPECT_THROW(static_cast<void>(ours_input.broadcast_to(TensorShape({3, 4}))),
                 std::exception);
}

TEST(HardeningThemeF_Misc, F7_Rank9BroadcastDoesNotOverwriteFixedMetadataArrays) {
    const auto torch_result = torch::ones(
                                  {1, 1, 1, 1, 1, 1, 1, 2, 2})
                                  .expand({2, 1, 1, 1, 1, 1, 1, 2, 2});
    ASSERT_TRUE(torch_result.eq(1.0f).all().item<bool>());

    EXPECT_THROW(
        static_cast<void>(TensorShape(
            {2, 1, 1, 1, 1, 1, 1, 2, 2})),
        std::exception);
}

TEST_F(CudaTest, F8_GenericBroadcastUsesInlineMetadataAndWritesResult) {
    const auto ours = Tensor::ones({2, 1, 3, 1}, Device::CUDA)
                          .add(Tensor::ones({1, 4, 1, 5}, Device::CUDA));
    const auto theirs = torch::ones(
                            {2, 1, 3, 1}, torch::TensorOptions().device(torch::kCUDA))
                            .add(torch::ones(
                                {1, 4, 1, 5}, torch::TensorOptions().device(torch::kCUDA)));
    expect_float_values_match(ours, theirs, "F8 generic broadcast inline metadata");
}

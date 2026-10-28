/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor.hpp"
#include "training/components/popspa_controller.hpp"
#include <cstring>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <vector>

using namespace lfs::core;
using namespace lfs::training;
namespace {
    bool has_cuda() {
        int n = 0;
        return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
    }
    void set_scores(POPSpaController& controller, const std::vector<double>& scores) {
        auto host = Tensor::empty({scores.size(), sizeof(double)}, Device::CPU, DataType::UInt8);
        std::memcpy(host.data_ptr(), scores.data(), scores.size() * sizeof(double));
        controller.scores() = host.cuda();
    }
    void expect_roundtrip(POPSpaController& source) {
        std::stringstream bytes;
        ASSERT_TRUE(source.serialize(bytes));
        const auto size = source.checkpoint_size_bytes(source.state_size());
        ASSERT_TRUE(size);
        EXPECT_EQ(size.value(), bytes.str().size());
        POPSpaController restored;
        ASSERT_TRUE(restored.deserialize(bytes));
        EXPECT_EQ(restored.phase(), source.phase());
        EXPECT_EQ(restored.phase_step(), source.phase_step());
        EXPECT_EQ(restored.score_view(), source.score_view());
        EXPECT_EQ(restored.config().camera_fingerprint, source.config().camera_fingerprint);
        EXPECT_EQ(restored.proxy().to_vector(), source.proxy().to_vector());
        EXPECT_EQ(restored.dual().to_vector(), source.dual().to_vector());
        std::stringstream second;
        ASSERT_TRUE(restored.serialize(second));
        EXPECT_EQ(second.str(), bytes.str());
        auto consumed = POPSpaController::consume_checkpoint(second);
        ASSERT_TRUE(consumed);
        EXPECT_EQ(consumed.value(), source.state_size());
    }
} // namespace

TEST(POPSpaSelection, TiesKeepLowestRowsAndRespectFrozenAndInactive) {
    const std::vector<double> scores{7, 7, 100, 0, 7};
    const std::vector<uint8_t> active{1, 1, 0, 1, 1}, frozen{0, 0, 0, 1, 0};
    const auto keep = POPSpaController::select_keep_indices(scores, 3, active, frozen);
    ASSERT_TRUE(keep);
    EXPECT_EQ(keep.value(), (std::vector<int64_t>{0, 1, 3}));
    EXPECT_FALSE(POPSpaController::select_keep_indices(scores, 5, active, frozen));
    EXPECT_FALSE(POPSpaController::select_keep_indices(scores, 0, active, frozen));
}
TEST(POPSpaSelection, RejectsImpossibleFrozenBudgetAndNonfiniteScores) {
    const std::vector<double> scores{1, 2, 3};
    const std::vector<uint8_t> frozen{1, 1, 0}, active{0, 1, 1};
    EXPECT_FALSE(POPSpaController::select_keep_indices(scores, 1, {}, frozen));
    EXPECT_FALSE(POPSpaController::select_keep_indices(scores, 2, active, frozen));
    const std::vector<double> nan{0, std::numeric_limits<double>::quiet_NaN()};
    EXPECT_FALSE(POPSpaController::select_keep_indices(nan, 1));
}
TEST(POPSpaControllerTest, PhasesAndEveryPhaseCheckpointRoundTrip) {
    if (!has_cuda())
        GTEST_SKIP();
    POPSpaController controller;
    POPSpaController::Config config{.target_count = 2, .sparsify_steps = 2, .refine_steps = 1, .projection_interval = 2, .camera_fingerprint = 12345};
    auto original = Tensor::zeros({8, 1}, Device::CUDA);
    ASSERT_TRUE(controller.initialize(config, original, 2));
    EXPECT_EQ(controller.config().first_prune_count, 4u);
    expect_roundtrip(controller);
    EXPECT_FALSE(controller.prune_indices());
    ASSERT_TRUE(controller.finish_score_view());
    expect_roundtrip(controller);
    ASSERT_TRUE(controller.finish_score_view());
    EXPECT_EQ(controller.phase(), POPSpaPhase::FirstPrune);
    expect_roundtrip(controller);
    const auto keep = controller.prune_indices();
    ASSERT_TRUE(keep);
    EXPECT_EQ(keep.value().to_vector_int64(), (std::vector<int64_t>{0, 1, 2, 3}));
    auto sparse = Tensor::zeros({4, 1}, Device::CUDA);
    ASSERT_TRUE(controller.accept_prune(sparse));
    EXPECT_EQ(controller.phase(), POPSpaPhase::Sparsify);
    EXPECT_EQ(controller.proxy().to_vector(), (std::vector<float>{0.5f, 0.5f, 0, 0}));
    EXPECT_FALSE(controller.projection_due());
    expect_roundtrip(controller);
    ASSERT_TRUE(controller.finish_optimization_step());
    EXPECT_TRUE(controller.projection_due());
    ASSERT_TRUE(controller.after_backward(sparse));
    EXPECT_EQ(controller.dual().to_vector(), (std::vector<float>{0, 0, 0.5f, 0.5f}));
    expect_roundtrip(controller);
    ASSERT_TRUE(controller.finish_optimization_step());
    EXPECT_EQ(controller.phase(), POPSpaPhase::SecondScore);
    expect_roundtrip(controller);
    // Final topology is determined by new POP scores, not ADMM support.
    set_scores(controller, {0, 0, 4, 3});
    ASSERT_TRUE(controller.finish_score_view());
    ASSERT_TRUE(controller.finish_score_view());
    EXPECT_EQ(controller.phase(), POPSpaPhase::FinalPrune);
    expect_roundtrip(controller);
    const auto final_keep = controller.prune_indices();
    ASSERT_TRUE(final_keep);
    EXPECT_EQ(final_keep.value().to_vector_int64(), (std::vector<int64_t>{2, 3}));
    auto final = Tensor::zeros({2, 1}, Device::CUDA);
    ASSERT_TRUE(controller.accept_prune(final));
    EXPECT_EQ(controller.phase(), POPSpaPhase::Recover);
    expect_roundtrip(controller);
    ASSERT_TRUE(controller.finish_optimization_step());
    EXPECT_EQ(controller.phase(), POPSpaPhase::Complete);
    expect_roundtrip(controller);
    EXPECT_FALSE(controller.finish_optimization_step());
}
TEST(POPSpaControllerTest, ZeroLengthPhasesSkipOptimizationAndInvalidRestoreIsAtomic) {
    if (!has_cuda())
        GTEST_SKIP();
    POPSpaController controller;
    auto opacity = Tensor::zeros({2}, Device::CUDA);
    ASSERT_TRUE(controller.initialize({.target_count = 1, .first_prune_count = 2, .sparsify_steps = 0, .refine_steps = 0}, opacity, 1));
    ASSERT_TRUE(controller.finish_score_view());
    ASSERT_TRUE(controller.accept_prune(opacity));
    EXPECT_EQ(controller.phase(), POPSpaPhase::SecondScore);
    ASSERT_TRUE(controller.finish_score_view());
    auto final = Tensor::zeros({1}, Device::CUDA);
    ASSERT_TRUE(controller.accept_prune(final));
    EXPECT_EQ(controller.phase(), POPSpaPhase::Complete);
    std::stringstream valid;
    ASSERT_TRUE(controller.serialize(valid));
    auto corrupt = valid.str();
    const uint32_t invalid_phase = 99;
    std::memcpy(corrupt.data() + 8, &invalid_phase, sizeof(invalid_phase));
    std::stringstream invalid(corrupt);
    EXPECT_FALSE(controller.deserialize(invalid));
    std::stringstream after;
    ASSERT_TRUE(controller.serialize(after));
    EXPECT_EQ(after.str(), valid.str());
    std::stringstream truncated(valid.str().substr(0, valid.str().size() - 1));
    EXPECT_FALSE(POPSpaController::consume_checkpoint(truncated));
}
TEST(POPSpaControllerTest, FrozenRowsCountAgainstBudgetAndGPUSelectionIsExact) {
    if (!has_cuda())
        GTEST_SKIP();
    POPSpaController controller;
    auto opacity = Tensor::zeros({5}, Device::CUDA);
    auto active = Tensor::from_vector(std::vector<bool>{true, true, false, true, true}, {5}, Device::CUDA);
    auto frozen = Tensor::from_vector(std::vector<bool>{false, false, false, true, false}, {5}, Device::CUDA);
    ASSERT_TRUE(controller.initialize({.target_count = 2, .first_prune_count = 3}, opacity, 1, active, frozen));
    set_scores(controller, {7, 7, 100, 0, 7});
    ASSERT_TRUE(controller.finish_score_view());
    auto keep = controller.prune_indices();
    ASSERT_TRUE(keep);
    EXPECT_EQ(keep.value().to_vector_int64(), (std::vector<int64_t>{0, 1, 3}));
}
TEST(POPSpaControllerTest, ADMMUsesSumOnlyOnProjectionStepAndResumeHasSameNextUpdate) {
    if (!has_cuda())
        GTEST_SKIP();
    POPSpaController controller;
    auto opacity = Tensor::zeros({4}, Device::CUDA);
    ASSERT_TRUE(controller.initialize({.target_count = 1, .first_prune_count = 4, .sparsify_steps = 3, .rho = 0.2f, .projection_interval = 2, .erank_weight = 0, .thin_scale_weight = 0}, opacity, 1));
    ASSERT_TRUE(controller.finish_score_view());
    ASSERT_TRUE(controller.accept_prune(opacity));
    auto scales = Tensor::zeros({4, 3}, Device::CUDA);
    auto og = Tensor::zeros({4}, Device::CUDA), sg = Tensor::zeros({4, 3}, Device::CUDA);
    auto loss = controller.regularization(opacity, scales, og, sg);
    ASSERT_TRUE(loss);
    EXPECT_FLOAT_EQ(loss.value().item<float>(), 0);
    ASSERT_TRUE(controller.finish_optimization_step());
    std::stringstream checkpoint;
    ASSERT_TRUE(controller.serialize(checkpoint));
    POPSpaController restored;
    ASSERT_TRUE(restored.deserialize(checkpoint));
    auto rg = Tensor::zeros({4}, Device::CUDA), rs = Tensor::zeros({4, 3}, Device::CUDA);
    auto next = controller.regularization(opacity, scales, og, sg);
    auto resumed = restored.regularization(opacity, scales, rg, rs);
    ASSERT_TRUE(next);
    ASSERT_TRUE(resumed);
    EXPECT_NEAR(next.value().item<float>(), 0.075f, 1e-7f); // rho/2 * 3 * 0.5^2
    EXPECT_EQ(og.to_vector(), (std::vector<float>{0, 0.025f, 0.025f, 0.025f}));
    EXPECT_EQ(og.to_vector(), rg.to_vector());
    ASSERT_TRUE(controller.after_backward(opacity));
    ASSERT_TRUE(restored.after_backward(opacity));
    EXPECT_EQ(controller.proxy().to_vector(), restored.proxy().to_vector());
    EXPECT_EQ(controller.dual().to_vector(), restored.dual().to_vector());
}

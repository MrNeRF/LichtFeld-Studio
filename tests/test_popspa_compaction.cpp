/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/strategies/postprocess_compaction.hpp"
#include "training/strategies/strategy_factory.hpp"

#include <gtest/gtest.h>
#include <vector>

namespace {
    using namespace lfs::core;
    using namespace lfs::training;

    SplatData make_model() {
        return SplatData(1,
                         Tensor::from_vector(std::vector<float>{0, 0, 2, 1, 0, 2, 2, 0, 2, 3, 0, 2, 4, 0, 2, 5, 0, 2}, {6, 3}, Device::CUDA),
                         Tensor::full({6, 1, 3}, 0.25f, Device::CUDA),
                         Tensor::full({6, 3, 3}, 0.125f, Device::CUDA),
                         Tensor::full({6, 3}, -2.0f, Device::CUDA),
                         Tensor::from_vector(std::vector<float>{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, {6, 4}, Device::CUDA),
                         Tensor::zeros({6, 1}, Device::CUDA), 1.0f);
    }

    class POPSpaCompactionTest : public ::testing::TestWithParam<const char*> {};

    TEST_P(POPSpaCompactionTest, RetainsExactRowsAndResetsAlignedStrategyState) {
        LiveModelMutationGuard mutation("POPSpa compaction test");
        auto model = make_model();
        model.set_frozen_ranges({{3, 1}});
        auto params = param::OptimizationParameters::mrnf_defaults();
        params.strategy = GetParam();
        params.max_cap = 6;
        auto created = StrategyFactory::instance().create(GetParam(), model);
        ASSERT_TRUE(created.has_value());
        auto strategy = std::move(*created);
        strategy->initialize(params);
        model.set_capacity_ensure([](size_t) { return true; });
        const auto indices = Tensor::from_vector(std::vector<int>{1, 3, 5}, {3}, Device::CUDA).to(DataType::Int64);
        const auto result = compact_gaussians_for_postprocess(*strategy, params, indices);
        ASSERT_TRUE(result.has_value());
        ASSERT_EQ(model.size(), 3);
        EXPECT_TRUE(model.has_capacity_ensure());
        EXPECT_EQ(model.means().to_vector(), (std::vector<float>{1, 0, 2, 3, 0, 2, 5, 0, 2}));
        EXPECT_EQ(model.get_active_sh_degree(), 1);
        EXPECT_EQ(model.get_max_sh_degree(), 1);
        ASSERT_EQ(model.frozen_ranges().size(), 1);
        EXPECT_EQ(model.frozen_ranges()[0].start, 1);
        EXPECT_EQ(model.frozen_ranges()[0].count, 1);
        EXPECT_EQ(model.deleted_count(), 0);
        for (float value : model.shN_canonical().to_vector())
            EXPECT_NEAR(value, 0.125f, 1e-4f);
        auto& optimizer = strategy->get_optimizer();
        EXPECT_EQ(optimizer.frozen_mask().numel(), 3);
        EXPECT_EQ(optimizer.get_step_count(lfs::training::ParamType::Means), 0);
        EXPECT_EQ(optimizer.get_grad(lfs::training::ParamType::Means).size(0), 3);
    }

    TEST_P(POPSpaCompactionTest, RejectsInvalidRowsWithoutChangingModel) {
        LiveModelMutationGuard mutation("POPSpa rejected compaction test");
        auto model = make_model();
        model.set_frozen_ranges({{3, 1}});
        auto params = param::OptimizationParameters::mrnf_defaults();
        params.strategy = GetParam();
        params.max_cap = 6;
        auto created = StrategyFactory::instance().create(GetParam(), model);
        ASSERT_TRUE(created.has_value());
        auto strategy = std::move(*created);
        strategy->initialize(params);
        const auto original = model.means().to_vector();
        for (const auto& ids : {std::vector<int>{1, 1, 3}, std::vector<int>{1, 3, 6}, std::vector<int>{1, 2}}) {
            const auto indices = Tensor::from_vector(ids, {ids.size()}, Device::CUDA).to(DataType::Int64);
            EXPECT_FALSE(compact_gaussians_for_postprocess(*strategy, params, indices).has_value());
            EXPECT_EQ(model.size(), 6);
            EXPECT_EQ(model.means().to_vector(), original);
        }
        const auto valid_indices = Tensor::from_vector(std::vector<int>{1, 3, 5}, {3}, Device::CUDA).to(DataType::Int64);
        bool prepared = false;
        const auto rejected = compact_gaussians_for_postprocess(
            *strategy, params, valid_indices, [&](SplatData& staged_model) -> lfs::Status {
                prepared = true;
                EXPECT_EQ(staged_model.size(), 3);
                return lfs::Status::failure(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::ResourceExhausted,
                    .domain = lfs::ErrorDomain::Training,
                    .user_message = "Injected staged allocation failure.",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            });
        EXPECT_TRUE(prepared);
        EXPECT_FALSE(rejected);
        EXPECT_EQ(model.size(), 6);
        EXPECT_EQ(model.means().to_vector(), original);
        EXPECT_EQ(strategy->get_optimizer().get_grad(lfs::training::ParamType::Means).size(0), 6);
    }

    INSTANTIATE_TEST_SUITE_P(BuiltInStrategies, POPSpaCompactionTest,
                             ::testing::Values("mcmc", "mrnf", "igs+"));
} // namespace

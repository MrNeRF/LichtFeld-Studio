/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/parameters.hpp"
#include "core/splat_data.hpp"
#include "lfs/training/live_model_mutation_guard.hpp"
#include "lfs/training/sh_value_codec.hpp"
#include "lfs/training/sh_value_storage.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/strategies/postprocess_compaction.hpp"
#include "training/strategies/strategy_factory.hpp"

#include <cstring>
#include <gtest/gtest.h>
#include <optional>
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

    // q16 rows are gathered code to code. The result must be exactly what the
    // fp32 staging it replaces gave: decode the kept rows, then quantize them.
    TEST_P(POPSpaCompactionTest, QuantizedRowsMatchFp32Staging) {
        struct QuantOn {
            QuantOn() { sh_value::set_sh_value_quant_enabled_for_testing(true); }
            ~QuantOn() { sh_value::set_sh_value_quant_enabled_for_testing(std::nullopt); }
        } quant_on;
        LiveModelMutationGuard mutation("POPSpa compaction q16 test");
        constexpr size_t n = 1300;
        constexpr size_t rest = 15;
        std::vector<float> means(n * 3), rotations(n * 4, 0.0f), sh_rest(n * rest * 3);
        for (size_t i = 0; i < n; ++i) {
            means[i * 3] = static_cast<float>(i) * 0.01f;
            means[i * 3 + 2] = 2.0f;
            rotations[i * 4] = 1.0f;
        }
        for (size_t i = 0; i < sh_rest.size(); ++i)
            sh_rest[i] = 0.003f * static_cast<float>(static_cast<int>((i * 2654435761u) % 401) - 200);
        auto model = SplatData(3,
                               Tensor::from_vector(means, {n, 3}, Device::CUDA),
                               Tensor::full({n, 1, 3}, 0.25f, Device::CUDA),
                               Tensor::from_vector(sh_rest, {n, rest, 3}, Device::CUDA),
                               Tensor::full({n, 3}, -2.0f, Device::CUDA),
                               Tensor::from_vector(rotations, {n, 4}, Device::CUDA),
                               Tensor::zeros({n, 1}, Device::CUDA), 1.0f);
        ASSERT_TRUE(sh_value::apply_shN_value_quant(model));
        std::vector<int> kept;
        for (size_t i = 0; i < n; ++i) {
            if (i % 3 != 1 && i % 517 != 0)
                kept.push_back(static_cast<int>(i));
        }
        const size_t k = kept.size();
        const auto indices = Tensor::from_vector(kept, {k}, Device::CUDA).to(DataType::Int64);

        auto canonical = Tensor::empty({k, rest, 3}, Device::CUDA);
        sh_value::gather_shN_to_canonical(model, indices, canonical);
        auto reference = SplatData(3, model.means().index_select(0, indices), model.sh0().index_select(0, indices),
                                   std::move(canonical), model.scaling_raw().index_select(0, indices),
                                   model.rotation_raw().index_select(0, indices),
                                   model.opacity_raw().index_select(0, indices), 1.0f);
        ASSERT_TRUE(sh_value::apply_shN_value_quant(reference));

        auto params = param::OptimizationParameters::mrnf_defaults();
        params.strategy = GetParam();
        params.max_cap = static_cast<int>(n);
        auto created = StrategyFactory::instance().create(GetParam(), model);
        ASSERT_TRUE(created.has_value());
        auto strategy = std::move(*created);
        strategy->initialize(params);
        model.set_capacity_ensure([](size_t) { return true; });
        ASSERT_TRUE(model.shN_value_quantized());
        ASSERT_TRUE(compact_gaussians_for_postprocess(*strategy, params, indices).has_value());

        ASSERT_TRUE(model.shN_value_quantized());
        const auto bytes_of = [](const Tensor& t) {
            const auto host = t.cpu();
            const auto* p = static_cast<const unsigned char*>(host.data_ptr());
            return std::vector<unsigned char>(p, p + host.bytes());
        };
        EXPECT_EQ(bytes_of(model.shN()), bytes_of(reference.shN()));
        EXPECT_EQ(bytes_of(model.shN_value_bounds()), bytes_of(reference.shN_value_bounds()));
    }

    INSTANTIATE_TEST_SUITE_P(BuiltInStrategies, POPSpaCompactionTest,
                             ::testing::Values("mcmc", "mrnf", "igs+"));
} // namespace

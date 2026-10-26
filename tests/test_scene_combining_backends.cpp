/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "combined_model_worker_fixture.hpp"
#include "core/scene.hpp"
#include "core/sh_value_quant.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <exception>
#include <optional>
#include <thread>
#include <vector>

namespace {
    using namespace lfs::core;

    class SceneCombiningBackends : public testing::TestWithParam<GpuBackend> {
    protected:
        void SetUp() override {
            if (!gpu_backend_available(GetParam()))
                GTEST_SKIP() << "Backend unavailable";
            scope_.emplace(GetParam());
            sh_value_quant::set_enabled_for_testing(false);
        }
        void TearDown() override { sh_value_quant::set_enabled_for_testing(std::nullopt); }

        std::shared_ptr<SplatData> model(size_t n, int degree, float base, int format) {
            const size_t rest = (degree + 1) * (degree + 1) - 1;
            std::vector<float> colors(n * rest * 3);
            for (size_t i = 0; i < colors.size(); ++i)
                colors[i] = base + static_cast<float>(i % 23) / 32.0f;
            auto result = std::make_shared<SplatData>(
                degree, Tensor::full({n, 3}, base, Device::GPU),
                Tensor::full({n, 1, 3}, base / 2.0f, Device::GPU),
                Tensor::from_vector(colors, {n, rest, 3}, Device::GPU),
                Tensor::full({n, 3}, -2.0f, Device::GPU),
                Tensor::full({n, 4}, 0.5f, Device::GPU),
                Tensor::full({n, 1}, 1.0f, Device::GPU), 1.0f,
                SplatData::ShNLayout::Canonical);
            if (format == 1) {
                result->shN_raw() = result->shN_raw().to(DataType::Float16);
                result->shN_raw().reserve(result->shN_raw().numel());
            } else if (format == 2) {
                sh_value_quant::set_enabled_for_testing(true);
                EXPECT_TRUE(result->apply_shN_value_quant());
                sh_value_quant::set_enabled_for_testing(false);
            }
            return result;
        }

        void check(bool opposite_default, bool include_hidden) {
            for (int format = 0; format < 3; ++format) {
                SCOPED_TRACE(format);
                auto first = model(33, 1, -0.5f, format);
                auto hidden = model(5, 2, 0.25f, format);
                auto second = model(67, 3, 1.0f, format);
                first->deleted() = Tensor::zeros({33}, Device::GPU, DataType::Bool);
                first->deleted().slice(0, 31, 33).fill_(true);
                std::vector<Scene::CombinedModelBuildInput> inputs{
                    {first, true, 0},
                    {hidden, false, 33},
                    {second, true, 38}};
                std::vector<float> expected_colors, expected_means;
                std::vector<int> expected_transforms, expected_selection;
                std::vector<bool> expected_deleted;
                int slot = 0;
                for (const auto& input : inputs) {
                    if (!include_hidden && !input.visible)
                        continue;
                    const auto colors = input.model->shN_canonical().to_vector();
                    const auto means = input.model->means_raw().to_vector();
                    expected_means.insert(expected_means.end(), means.begin(), means.end());
                    const size_t components = input.model->max_sh_coeffs_rest() * 3;
                    for (size_t row = 0; row < input.model->size(); ++row) {
                        expected_colors.insert(expected_colors.end(), colors.begin() + row * components,
                                               colors.begin() + (row + 1) * components);
                        expected_colors.insert(expected_colors.end(), 45 - components, 0.0f);
                        expected_transforms.push_back(slot);
                        expected_selection.push_back(static_cast<int>(input.selection_offset + row));
                        expected_deleted.push_back(slot == 0 && row >= 31);
                    }
                    ++slot;
                }
                const auto worker_backend = opposite_default
                                                ? (GetParam() == GpuBackend::CUDA ? GpuBackend::Vulkan : GpuBackend::CUDA)
                                                : GetParam();
                Scene::CombinedModelBuild result;
                std::exception_ptr failure;
                std::jthread worker([&] {
                    const GpuBackendScope worker_scope(worker_backend);
                    try {
                        result = Scene::buildCombinedModelCache(inputs, 105, {}, 17, include_hidden);
                    } catch (...) {
                        failure = std::current_exception();
                    }
                });
                worker.join();
                ASSERT_FALSE(failure) << ([&] {
                    try {
                        std::rethrow_exception(failure);
                    } catch (const std::exception& error) { return std::string(error.what()); }
                }());
                ASSERT_NE(result.model, nullptr);
                EXPECT_EQ(gpu_backend_of(result.model->means_raw()), GetParam());
                EXPECT_EQ(gpu_backend_of(result.model->shN_raw()), GetParam());
                EXPECT_EQ(result.generation, 17);
                EXPECT_EQ(result.model->get_max_sh_degree(), 3);
                EXPECT_EQ(result.model->get_active_sh_degree(), 3);
                EXPECT_EQ(result.model->means_raw().to_vector(), expected_means);
                EXPECT_EQ(result.model->shN_canonical().to_vector(), expected_colors);
                EXPECT_EQ(result.model->deleted().to_vector_bool(), expected_deleted);
                EXPECT_EQ(result.transform_indices->to_vector_int(), expected_transforms);
                if (include_hidden) {
                    EXPECT_EQ(result.visible_selection_indices, nullptr);
                } else {
                    ASSERT_NE(result.visible_selection_indices, nullptr);
                    EXPECT_EQ(result.visible_selection_indices->to_vector_int(), expected_selection);
                }
            }
        }

        std::optional<GpuBackendScope> scope_;
    };

    TEST_P(SceneCombiningBackends, WorkerPreservesBackendAcrossCallerDefaults) {
        check(true, false);
    }

    TEST_P(SceneCombiningBackends, WorkerKeepsHiddenAndDeletedRowsAligned) {
        check(false, false);
        check(false, true);
    }

    TEST_P(SceneCombiningBackends, StaleBackgroundResultKeepsRetiredInputsAliveUntilJoin) {
        namespace combined = lfs::test::combined_model;
        Scene scene;
        combined::HeldWorker worker;
        const auto first = scene.addSplat("first", std::make_unique<SplatData>(model(33, 1, -0.5f, 2)->clone()));
        const auto second = scene.addSplat("second", std::make_unique<SplatData>(model(67, 3, 1.0f, 2)->clone()));
        auto replacement = std::make_unique<SplatData>(model(17, 2, 0.25f, 1)->clone());
        scene.setCombinedModelAllocator(worker.allocator());
        const SplatData* retired = scene.getNodeById(first)->model.get();
        const SplatData* removed = scene.getNodeById(second)->model.get();
        scene.requestCombinedModelBuild(true);
        ASSERT_TRUE(worker.waitEntered());
        scene.replaceNodeModel("first", std::move(replacement));
        scene.removeNodeById(second);
        EXPECT_EQ(retired->size(), 33);
        const SplatData* current = scene.getNodeById(first)->model.get();
        const auto retained = combined::combinedModelWithin(scene, worker);
        ASSERT_TRUE(retained.prompt);
        EXPECT_EQ(retained.model, current);
        EXPECT_EQ(current->size(), 17);
        EXPECT_TRUE(scene.combinedModelBuildPending());
        // Returning the replacement must not retire inputs still used by the worker.
        EXPECT_EQ(retired->size(), 33);
        EXPECT_EQ(removed->size(), 67);
        EXPECT_EQ(gpu_backend_of(retired->means_raw()), GetParam());
        EXPECT_EQ(gpu_backend_of(removed->means_raw()), GetParam());

        worker.release();
        EXPECT_EQ(combined::settledCombinedModel(scene), current);
        EXPECT_FALSE(scene.combinedModelBuildPending());
        EXPECT_EQ(current->size(), 17);
        EXPECT_EQ(gpu_backend_of(current->means_raw()), GetParam());
    }

    INSTANTIATE_TEST_SUITE_P(Backends, SceneCombiningBackends,
                             testing::ValuesIn(kCompiledGpuBackends),
                             [](const testing::TestParamInfo<GpuBackend>& info) {
                                 return info.param == GpuBackend::CUDA ? "Cuda" : info.param == GpuBackend::Vulkan ? "Vulkan"
                                                                                                                   : "Metal";
                             });
} // namespace

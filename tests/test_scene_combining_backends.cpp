/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/scene.hpp"
#include "core/sh_value_quant.hpp"
#include "core/tensor_backend.hpp"

#include <gtest/gtest.h>

#include <exception>
#include <future>
#include <latch>
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
        std::promise<void> entered;
        auto started = entered.get_future();
        std::latch release(1);
        std::atomic<bool> first_allocation{true};
        Scene scene;
        const auto first = scene.addSplat("first", std::make_unique<SplatData>(model(33, 1, -0.5f, 2)->clone()));
        const auto second = scene.addSplat("second", std::make_unique<SplatData>(model(67, 3, 1.0f, 2)->clone()));
        auto replacement = std::make_unique<SplatData>(model(17, 2, 0.25f, 1)->clone());
        struct ReleaseOnExit {
            std::latch& signal;
            bool released = false;
            void now() {
                if (!released) {
                    released = true;
                    signal.count_down();
                }
            }
            ~ReleaseOnExit() { now(); }
        } guard{release};
        scene.setCombinedModelAllocator([&](TensorShape shape, size_t, DataType dtype, std::string_view) {
            if (first_allocation.exchange(false))
                entered.set_value();
            release.wait();
            return Tensor::empty(std::move(shape), Device::GPU, dtype);
        });
        const SplatData* retired = scene.getNodeById(first)->model.get();
        scene.requestCombinedModelBuild(true);
        ASSERT_EQ(started.wait_for(std::chrono::seconds(5)), std::future_status::ready);
        scene.replaceNodeModel("first", std::move(replacement));
        scene.removeNodeById(second);
        EXPECT_EQ(retired->size(), 33);
        guard.now();
        const SplatData* current = scene.getNodeById(first)->model.get();
        EXPECT_EQ(scene.getCombinedModel(), current);
        EXPECT_EQ(current->size(), 17);
        EXPECT_FALSE(scene.combinedModelBuildPending());
        EXPECT_EQ(gpu_backend_of(current->means_raw()), GetParam());
    }

    INSTANTIATE_TEST_SUITE_P(Backends, SceneCombiningBackends,
                             testing::ValuesIn(kGpuBackends),
                             [](const testing::TestParamInfo<GpuBackend>& info) {
                                 return info.param == GpuBackend::CUDA ? "Cuda" : "Vulkan";
                             });
} // namespace

// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <vector>

#include "core/cuda/sh_layout.cuh"
#include "core/parameters.hpp"
#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "python/python_runtime.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/training_setup.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/scene/selection_state.hpp"

namespace lfs::python {

    namespace {
        std::unique_ptr<core::SplatData> make_test_splat(size_t count, const int sh_degree = 0) {
            std::vector<float> means(count * 3, 0.0f);
            std::vector<float> rotations(count * 4, 0.0f);
            for (size_t i = 0; i < count; ++i) {
                means[i * 3] = static_cast<float>(i);
                rotations[i * 4] = 1.0f;
            }

            return std::make_unique<core::SplatData>(
                sh_degree,
                core::Tensor::from_vector(means, {count, size_t{3}}, core::Device::CPU),
                core::Tensor::zeros({count, size_t{1}, size_t{3}}, core::Device::CPU, core::DataType::Float32),
                core::Tensor::zeros({count, core::sh_rest_coefficients_for_degree(sh_degree), size_t{3}}, core::Device::CPU, core::DataType::Float32),
                core::Tensor::zeros({count, size_t{3}}, core::Device::CPU, core::DataType::Float32),
                core::Tensor::from_vector(rotations, {count, size_t{4}}, core::Device::CPU),
                core::Tensor::zeros({count, size_t{1}}, core::Device::CPU, core::DataType::Float32),
                1.0f);
        }

        std::shared_ptr<core::PointCloud> make_test_point_cloud(size_t count) {
            std::vector<float> means(count * 3, 0.0f);
            std::vector<float> colors(count * 3, 0.5f);
            for (size_t i = 0; i < count; ++i) {
                means[i * 3 + 0] = static_cast<float>(i);
                means[i * 3 + 1] = static_cast<float>(i % 3);
                means[i * 3 + 2] = static_cast<float>(i % 5);
            }

            return std::make_shared<core::PointCloud>(
                core::Tensor::from_vector(means, {count, size_t{3}}, core::Device::CPU),
                core::Tensor::from_vector(colors, {count, size_t{3}}, core::Device::CPU));
        }

        core::Tensor make_external_float_tensor(
            std::vector<std::shared_ptr<std::vector<float>>>& owners,
            core::TensorShape shape,
            const size_t capacity,
            std::string kind = "splat.exportable") {
            const size_t element_count = std::max(shape.elements(), capacity);
            auto owner = std::make_shared<std::vector<float>>(element_count, 0.0f);
            auto* data = owner->data();
            owners.push_back(owner);
            return core::Tensor::from_external_owner(data,
                                                     std::move(shape),
                                                     core::Device::CPU,
                                                     core::DataType::Float32,
                                                     std::move(owner),
                                                     capacity,
                                                     nullptr,
                                                     std::move(kind));
        }

        void expect_sh_degree(const core::SplatData& splat,
                              const int max_sh_degree,
                              const int active_sh_degree,
                              const size_t count) {
            const size_t expected_active_rest = core::sh_rest_coefficients_for_degree(active_sh_degree);
            const size_t expected_layout_rest = core::sh_rest_coefficients_for_degree(max_sh_degree);
            const size_t expected_swizzled_floats =
                core::sh_swizzled_float_count(count, static_cast<std::uint32_t>(expected_layout_rest));

            EXPECT_EQ(splat.get_max_sh_degree(), max_sh_degree);
            EXPECT_EQ(splat.get_active_sh_degree(), active_sh_degree);
            ASSERT_TRUE(splat.shN_raw().is_valid());
            ASSERT_EQ(splat.shN_raw().ndim(), 1);
            EXPECT_EQ(splat.shN_raw().shape()[0], expected_swizzled_floats);

            const auto shN_canonical = splat.shN_canonical();
            ASSERT_EQ(shN_canonical.ndim(), 3);
            EXPECT_EQ(shN_canonical.shape()[0], count);
            EXPECT_EQ(shN_canonical.shape()[1], expected_layout_rest);
            EXPECT_EQ(shN_canonical.shape()[2], size_t{3});
            EXPECT_EQ(splat.get_shs().shape()[1], expected_active_rest + 1);
        }

        void expect_sh_degree(const core::SplatData& splat, const int sh_degree, const size_t count) {
            expect_sh_degree(splat, sh_degree, sh_degree, count);
        }
    } // namespace

    class SceneValidityTest : public ::testing::Test {
    protected:
        void SetUp() override {
            set_application_scene(nullptr);
        }

        void TearDown() override {
            set_scene_generation_callback(nullptr);
            set_application_scene(nullptr);
        }

        core::Scene dummy_scene_;
    };

    TEST_F(SceneValidityTest, GenerationNonNegative) {
        auto gen = get_scene_generation();
        EXPECT_GE(gen, 0u);
    }

    TEST_F(SceneValidityTest, GenerationIncrementsOnSet) {
        auto gen1 = get_scene_generation();
        set_application_scene(&dummy_scene_);
        auto gen2 = get_scene_generation();
        EXPECT_GT(gen2, gen1);
    }

    TEST_F(SceneValidityTest, GenerationIncrementsOnClear) {
        set_application_scene(&dummy_scene_);
        auto gen1 = get_scene_generation();
        set_application_scene(nullptr);
        auto gen2 = get_scene_generation();
        EXPECT_GT(gen2, gen1);
    }

    TEST_F(SceneValidityTest, GetApplicationSceneReturnsCorrectPointer) {
        EXPECT_EQ(get_application_scene(), nullptr);
        set_application_scene(&dummy_scene_);
        EXPECT_EQ(get_application_scene(), &dummy_scene_);
        set_application_scene(nullptr);
        EXPECT_EQ(get_application_scene(), nullptr);
    }

    TEST_F(SceneValidityTest, ConcurrentReadsAreSafe) {
        set_application_scene(&dummy_scene_);
        std::atomic<int> success_count{0};
        std::vector<std::thread> threads;

        for (int i = 0; i < 10; ++i) {
            threads.emplace_back([&]() {
                for (int j = 0; j < 1000; ++j) {
                    auto gen = get_scene_generation();
                    auto* scene = get_application_scene();
                    EXPECT_GE(gen, 0u);
                    EXPECT_EQ(scene, &dummy_scene_);
                }
                success_count++;
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        EXPECT_EQ(success_count.load(), 10);
    }

    TEST_F(SceneValidityTest, GenerationIsMonotonic) {
        std::vector<uint64_t> generations;
        generations.push_back(get_scene_generation());

        for (int i = 0; i < 10; ++i) {
            set_application_scene(&dummy_scene_);
            generations.push_back(get_scene_generation());
            set_application_scene(nullptr);
            generations.push_back(get_scene_generation());
        }

        for (size_t i = 1; i < generations.size(); ++i) {
            EXPECT_GT(generations[i], generations[i - 1]);
        }
    }

    TEST_F(SceneValidityTest, SceneGenerationCallbackCanPublishToAppStore) {
        set_scene_generation_callback([](const uint64_t generation) {
            lfs::vis::app_store().scene_generation.set(generation);
        });

        set_application_scene(&dummy_scene_);
        EXPECT_EQ(lfs::vis::app_store().scene_generation.get(), get_scene_generation());

        bump_scene_generation();
        EXPECT_EQ(lfs::vis::app_store().scene_generation.get(), get_scene_generation());

        set_scene_generation_callback(nullptr);
    }

    TEST_F(SceneValidityTest, MutationFlagsAccumulateUntilConsumed) {
        set_application_scene(&dummy_scene_);

        constexpr uint32_t node_added = 1u << 0;
        constexpr uint32_t transform_changed = 1u << 4;
        constexpr uint32_t combined = node_added | transform_changed;

        set_scene_mutation_flags(node_added);
        set_scene_mutation_flags(transform_changed);

        EXPECT_EQ(get_scene_mutation_flags(), combined);
        EXPECT_EQ(consume_scene_mutation_flags(), combined);
        EXPECT_EQ(get_scene_mutation_flags(), 0u);
        EXPECT_EQ(consume_scene_mutation_flags(), 0u);
    }

    TEST_F(SceneValidityTest, SelectionGenerationPublishesToAppStore) {
        lfs::vis::SelectionState selection;

        selection.selectNode(7);

        EXPECT_EQ(lfs::vis::app_store().selection_generation.get(), selection.generation());
    }

    TEST_F(SceneValidityTest, ClearResetsDatasetMetadata) {
        auto means = core::Tensor::from_vector({0.0f, 0.0f, 0.0f}, {size_t{1}, size_t{3}}, core::Device::CPU);
        auto colors = core::Tensor::from_vector({1.0f, 1.0f, 1.0f}, {size_t{1}, size_t{3}}, core::Device::CPU);

        dummy_scene_.setInitialPointCloud(std::make_shared<core::PointCloud>(std::move(means), std::move(colors)));
        dummy_scene_.setSceneCenter(core::Tensor::from_vector({1.0f, 2.0f, 3.0f}, {size_t{3}}, core::Device::CPU));
        dummy_scene_.setImagesHaveAlpha(true);
        dummy_scene_.setTrainingModelNode("Model");
        const auto dataset_id = dummy_scene_.addDataset("Dataset");
        const auto cameras_group_id = dummy_scene_.addGroup("Cameras", dataset_id);
        const auto train_group_id = dummy_scene_.addCameraGroup("Training (1)", cameras_group_id, 1);
        dummy_scene_.addCamera("cam_0001.png", train_group_id, std::make_shared<core::Camera>());

        ASSERT_TRUE(dummy_scene_.getInitialPointCloud());
        ASSERT_TRUE(dummy_scene_.getSceneCenter().is_valid());
        ASSERT_TRUE(dummy_scene_.imagesHaveAlpha());
        ASSERT_EQ(dummy_scene_.getTrainingModelNodeName(), "Model");
        ASSERT_EQ(dummy_scene_.getAllCameras().size(), 1u);
        ASSERT_GT(dummy_scene_.getNodeCount(), 0u);

        dummy_scene_.clear();

        EXPECT_FALSE(dummy_scene_.getInitialPointCloud());
        EXPECT_FALSE(dummy_scene_.getSceneCenter().is_valid());
        EXPECT_FALSE(dummy_scene_.imagesHaveAlpha());
        EXPECT_TRUE(dummy_scene_.getTrainingModelNodeName().empty());
        EXPECT_TRUE(dummy_scene_.getAllCameras().empty());
        EXPECT_EQ(dummy_scene_.getNodeCount(), 0u);
    }

    TEST_F(SceneValidityTest, TrainingModelActiveCountUsesSyncedTopologyCount) {
        dummy_scene_.addNode("Model", make_test_splat(2));
        dummy_scene_.setTrainingModelNode("Model");

        const auto model_id = dummy_scene_.getNodeIdByName("Model");
        ASSERT_NE(model_id, core::NULL_NODE);

        dummy_scene_.syncTrainingModelTopology(6);

        const auto counts = dummy_scene_.getActiveGaussianCountsByNode();
        const auto count_it = counts.find(model_id);
        ASSERT_NE(count_it, counts.end());
        EXPECT_EQ(count_it->second, 6u);
        EXPECT_EQ(dummy_scene_.getTrainingModelGaussianCount(), 6u);
    }

    TEST_F(SceneValidityTest, SplatDataSetSHDegreeSupportsAllDegrees) {
        constexpr size_t count = 4;

        for (const int target_degree : {0, 1, 2, 3}) {
            auto splat = make_test_splat(count, 3);

            splat->set_sh_degree(target_degree);

            expect_sh_degree(*splat, target_degree, count);
        }
    }

    TEST_F(SceneValidityTest, SplatDataSetSHDegreeCanExpandMissingCoefficients) {
        constexpr size_t count = 4;
        auto splat = make_test_splat(count, 0);

        EXPECT_TRUE(splat->set_sh_degree(2));

        expect_sh_degree(*splat, 2, count);
    }

    TEST_F(SceneValidityTest, InitializeTrainingModelAdjustsExistingTrainingModelSHDegree) {
        constexpr size_t count = 4;
        dummy_scene_.addNode("Model", make_test_splat(count, 3));
        dummy_scene_.setTrainingModelNode("Model");

        core::param::TrainingParameters params;
        params.optimization.sh_degree = 1;

        const auto result = lfs::training::initializeTrainingModel(params, dummy_scene_);

        ASSERT_TRUE(result.has_value()) << result.error();
        const auto* model = dummy_scene_.getTrainingModel();
        ASSERT_NE(model, nullptr);
        expect_sh_degree(*model, 1, 0, count);
    }

    TEST_F(SceneValidityTest, InitializeTrainingModelMigratesExistingModelIntoProvidedAllocator) {
        constexpr size_t count = 4;
        constexpr size_t capacity = 16;
        dummy_scene_.addNode("Model", make_test_splat(count, 1));
        dummy_scene_.setTrainingModelNode("Model");

        struct AllocCall {
            std::string name;
            size_t capacity;
        };
        auto calls = std::make_shared<std::vector<AllocCall>>();
        core::SplatTensorAllocator allocator =
            [calls](core::TensorShape shape,
                    const size_t requested_capacity,
                    const core::DataType dtype,
                    const std::string_view name) {
                EXPECT_EQ(dtype, core::DataType::Float32);
                calls->push_back({std::string{name}, requested_capacity});
                auto tensor = core::Tensor::zeros_direct(std::move(shape), requested_capacity, core::Device::CUDA);
                tensor.set_name(std::string{name});
                return tensor;
            };

        core::param::TrainingParameters params;
        params.optimization.sh_degree = 1;
        params.optimization.max_cap = static_cast<int>(capacity);

        const auto result = lfs::training::initializeTrainingModel(params, dummy_scene_, allocator);

        ASSERT_TRUE(result.has_value()) << result.error();
        const auto* model = dummy_scene_.getTrainingModel();
        ASSERT_NE(model, nullptr);
        EXPECT_EQ(model->means_raw().capacity(), capacity);
        EXPECT_EQ(model->sh0_raw().capacity(), capacity);
        EXPECT_EQ(model->scaling_raw().capacity(), capacity);
        EXPECT_EQ(model->rotation_raw().capacity(), capacity);
        EXPECT_EQ(model->opacity_raw().capacity(), capacity);
        EXPECT_EQ(model->shN_raw().capacity(),
                  core::sh_swizzled_float_count(capacity, core::sh_rest_coefficients_for_degree(1)));

        const auto capacity_for = [&](const std::string_view name) -> size_t {
            const auto it = std::find_if(calls->begin(), calls->end(), [&](const AllocCall& call) {
                return call.name == name;
            });
            return it == calls->end() ? 0 : it->capacity;
        };
        EXPECT_EQ(capacity_for("SplatData.means"), capacity);
        EXPECT_EQ(capacity_for("SplatData.sh0"), capacity);
        EXPECT_EQ(capacity_for("SplatData.scaling"), capacity);
        EXPECT_EQ(capacity_for("SplatData.rotation"), capacity);
        EXPECT_EQ(capacity_for("SplatData.opacity"), capacity);
        EXPECT_EQ(capacity_for("SplatData.shN"),
                  core::sh_swizzled_float_count(capacity, core::sh_rest_coefficients_for_degree(1)));
    }

    TEST_F(SceneValidityTest, InitializeTrainingModelCapsPointCloudBeforeAllocatorAllocation) {
        constexpr size_t source_count = 12;
        constexpr size_t capacity = 5;
        dummy_scene_.addPointCloud("PointCloud", make_test_point_cloud(source_count));

        struct AllocCall {
            std::string name;
            size_t capacity;
        };
        auto calls = std::make_shared<std::vector<AllocCall>>();
        core::SplatTensorAllocator allocator =
            [calls](core::TensorShape shape,
                    const size_t requested_capacity,
                    const core::DataType dtype,
                    const std::string_view name) {
                EXPECT_EQ(dtype, core::DataType::Float32);
                calls->push_back({std::string{name}, requested_capacity});
                auto tensor = core::Tensor::zeros_direct(std::move(shape), requested_capacity, core::Device::CUDA);
                tensor.set_name(std::string{name});
                return tensor;
            };

        core::param::TrainingParameters params;
        params.optimization.random = false;
        params.optimization.sh_degree = 1;
        params.optimization.max_cap = static_cast<int>(capacity);

        const auto result = lfs::training::initializeTrainingModel(params, dummy_scene_, allocator);

        ASSERT_TRUE(result.has_value()) << result.error();
        const auto* model = dummy_scene_.getTrainingModel();
        ASSERT_NE(model, nullptr);
        EXPECT_EQ(model->size(), capacity);
        EXPECT_LE(model->means_raw().capacity(), capacity);
        EXPECT_LE(model->sh0_raw().capacity(), capacity);
        EXPECT_LE(model->scaling_raw().capacity(), capacity);
        EXPECT_LE(model->rotation_raw().capacity(), capacity);
        EXPECT_LE(model->opacity_raw().capacity(), capacity);
        EXPECT_LE(model->shN_raw().capacity(),
                  core::sh_swizzled_float_count(capacity, core::sh_rest_coefficients_for_degree(1)));

        const auto max_capacity_for = [&](const std::string_view name) -> size_t {
            size_t max_capacity = 0;
            for (const auto& call : *calls) {
                if (call.name == name) {
                    max_capacity = std::max(max_capacity, call.capacity);
                }
            }
            return max_capacity;
        };
        EXPECT_EQ(max_capacity_for("SplatData.means"), capacity);
        EXPECT_EQ(max_capacity_for("SplatData.sh0"), capacity);
        EXPECT_EQ(max_capacity_for("SplatData.scaling"), capacity);
        EXPECT_EQ(max_capacity_for("SplatData.rotation"), capacity);
        EXPECT_EQ(max_capacity_for("SplatData.opacity"), capacity);
        EXPECT_EQ(max_capacity_for("SplatData.shN"),
                  core::sh_swizzled_float_count(capacity, core::sh_rest_coefficients_for_degree(1)));
    }

    TEST_F(SceneValidityTest, MigrateTrainingModelAcceptsCudaOnlyExportableStorage) {
        constexpr size_t count = 4;
        constexpr size_t capacity = 16;
        constexpr int sh_degree = 1;
        const auto rest_coeffs = core::sh_rest_coefficients_for_degree(sh_degree);
        std::vector<std::shared_ptr<std::vector<float>>> owners;

        auto model = std::make_unique<core::SplatData>(
            sh_degree,
            make_external_float_tensor(owners, {count, size_t{3}}, capacity),
            make_external_float_tensor(owners, {count, size_t{1}, size_t{3}}, capacity),
            make_external_float_tensor(owners,
                                       {core::sh_swizzled_float_count(count, rest_coeffs)},
                                       core::sh_swizzled_float_count(capacity, rest_coeffs)),
            make_external_float_tensor(owners, {count, size_t{3}}, capacity),
            make_external_float_tensor(owners, {count, size_t{4}}, capacity),
            make_external_float_tensor(owners, {count, size_t{1}}, capacity),
            1.0f,
            core::SplatData::ShNLayout::Swizzled);
        dummy_scene_.addNode("Model", std::move(model));
        dummy_scene_.setTrainingModelNode("Model");

        core::param::TrainingParameters params;
        params.optimization.sh_degree = sh_degree;
        params.optimization.max_cap = static_cast<int>(capacity);

        int allocation_calls = 0;
        core::SplatTensorAllocator allocator =
            [&allocation_calls](core::TensorShape shape,
                                const size_t requested_capacity,
                                const core::DataType,
                                const std::string_view) {
                ++allocation_calls;
                return core::Tensor::zeros_direct(std::move(shape), requested_capacity, core::Device::CUDA);
            };

        auto* training_model = dummy_scene_.getTrainingModel();
        ASSERT_NE(training_model, nullptr);
        const auto result =
            lfs::training::migrateTrainingModelToAllocator(params, *training_model, allocator);

        ASSERT_TRUE(result.has_value()) << result.error();
        EXPECT_EQ(allocation_calls, 0);
    }

    TEST_F(SceneValidityTest, AdamAddNewParamsPreservesExportableStorage) {
        constexpr size_t count = 4;
        constexpr size_t capacity = 16;
        constexpr int sh_degree = 1;
        const auto rest_coeffs = core::sh_rest_coefficients_for_degree(sh_degree);
        std::vector<std::shared_ptr<std::vector<float>>> owners;

        auto model = std::make_unique<core::SplatData>(
            sh_degree,
            make_external_float_tensor(owners, {count, size_t{3}}, capacity),
            make_external_float_tensor(owners, {count, size_t{1}, size_t{3}}, capacity),
            make_external_float_tensor(owners,
                                       {core::sh_swizzled_float_count(count, rest_coeffs)},
                                       core::sh_swizzled_float_count(capacity, rest_coeffs)),
            make_external_float_tensor(owners, {count, size_t{3}}, capacity),
            make_external_float_tensor(owners, {count, size_t{4}}, capacity),
            make_external_float_tensor(owners, {count, size_t{1}}, capacity),
            1.0f,
            core::SplatData::ShNLayout::Swizzled);

        training::AdamConfig config;
        training::AdamOptimizer optimizer(*model, config);

        auto new_means = core::Tensor::from_vector(
            {10.0f, 11.0f, 12.0f, 20.0f, 21.0f, 22.0f},
            {size_t{2}, size_t{3}},
            core::Device::CPU);
        optimizer.add_new_params(training::ParamType::Means, new_means, true);

        EXPECT_EQ(model->means_raw().shape()[0], count + 2);
        EXPECT_EQ(model->means_raw().capacity(), capacity);
        EXPECT_EQ(model->means_raw().external_storage_kind(), "splat.exportable");

        const auto values = model->means_raw().to_vector();
        ASSERT_EQ(values.size(), (count + 2) * 3);
        EXPECT_FLOAT_EQ(values[count * 3 + 0], 10.0f);
        EXPECT_FLOAT_EQ(values[count * 3 + 1], 11.0f);
        EXPECT_FLOAT_EQ(values[count * 3 + 2], 12.0f);
        EXPECT_FLOAT_EQ(values[(count + 1) * 3 + 0], 20.0f);
        EXPECT_FLOAT_EQ(values[(count + 1) * 3 + 1], 21.0f);
        EXPECT_FLOAT_EQ(values[(count + 1) * 3 + 2], 22.0f);
    }

    TEST_F(SceneValidityTest, SceneManagerEmptyStateKeepsApplicationSceneContext) {
        lfs::vis::SceneManager scene_manager;
        EXPECT_EQ(get_application_scene(), &scene_manager.getScene());

        scene_manager.addGroupNode("Bootstrap");
        ASSERT_GT(scene_manager.getScene().getNodeCount(), 0u);

        ASSERT_TRUE(scene_manager.clear());

        EXPECT_EQ(get_application_scene(), &scene_manager.getScene());
        EXPECT_EQ(scene_manager.getContentType(), lfs::vis::SceneManager::ContentType::Empty);
        EXPECT_EQ(scene_manager.getScene().getNodeCount(), 0u);
    }

} // namespace lfs::python

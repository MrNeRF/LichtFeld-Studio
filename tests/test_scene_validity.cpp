// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <future>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "core/cuda/sh_layout.cuh"
#include "core/parameters.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "python/python_runtime.hpp"
#include "training/components/bilateral_grid.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/core/services.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/scene/selection_state.hpp"
#include "visualizer/training/training_manager.hpp"

namespace lfs::python {

    TEST(TrainingStateMachineTest, PublishesFinishReasonBeforeFinishedCallback) {
        vis::TrainingStateMachine state_machine;
        ASSERT_TRUE(state_machine.transitionTo(vis::TrainingState::Paused));
        ASSERT_TRUE(state_machine.transitionTo(vis::TrainingState::Stopping));

        vis::FinishReason callback_reason = vis::FinishReason::None;
        state_machine.setStateChangeCallback([&](vis::TrainingState, vis::TrainingState new_state) {
            if (new_state == vis::TrainingState::Finished) {
                callback_reason = state_machine.getFinishReason();
            }
        });

        ASSERT_TRUE(state_machine.transitionToFinished(vis::FinishReason::UserStopped));
        EXPECT_EQ(callback_reason, vis::FinishReason::UserStopped);
        EXPECT_EQ(state_machine.getFinishReason(), vis::FinishReason::UserStopped);
    }

    TEST(TrainingStateMachineTest, SerializesConcurrentTransitionsThroughCallbacks) {
        vis::TrainingStateMachine state_machine;
        ASSERT_TRUE(state_machine.transitionTo(vis::TrainingState::Ready));
        ASSERT_TRUE(state_machine.transitionTo(vis::TrainingState::Running));

        std::promise<void> pause_callback_entered;
        std::promise<void> release_pause_callback;
        auto release_future = release_pause_callback.get_future().share();
        state_machine.setStateChangeCallback(
            [&](vis::TrainingState, vis::TrainingState new_state) {
                if (new_state == vis::TrainingState::Paused) {
                    pause_callback_entered.set_value();
                    release_future.wait();
                }
            });

        bool pause_succeeded = false;
        std::thread pause_thread([&] {
            pause_succeeded = state_machine.transitionTo(vis::TrainingState::Paused);
        });
        pause_callback_entered.get_future().wait();

        auto stop_future = std::async(std::launch::async, [&] {
            return state_machine.transitionTo(vis::TrainingState::Stopping);
        });
        EXPECT_EQ(stop_future.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);

        release_pause_callback.set_value();
        pause_thread.join();
        EXPECT_TRUE(pause_succeeded);
        EXPECT_TRUE(stop_future.get());
        EXPECT_EQ(state_machine.getState(), vis::TrainingState::Stopping);
    }

    TEST(BilateralGridValidationTest, RejectsInvalidConstructorAndImageContracts) {
        EXPECT_THROW((lfs::training::BilateralGrid(0, 16, 16, 8, 100)), std::invalid_argument);
        EXPECT_THROW((lfs::training::BilateralGrid(1, -1, 16, 8, 100)), std::invalid_argument);
        EXPECT_THROW((lfs::training::BilateralGrid(1, 2, 2, 2, 100,
                                                   {.warmup_steps = -1})),
                     std::invalid_argument);
        EXPECT_THROW((lfs::training::BilateralGrid(1, 50'000, 50'000, 1, 100)), std::length_error);

        lfs::training::BilateralGrid grid(1, 2, 2, 2, 100);
        auto cpu_image = core::Tensor::ones({3, 2, 2}, core::Device::CPU);
        auto wrong_channels = core::Tensor::ones({2, 2, 2}, core::Device::CUDA);
        auto image = core::Tensor::ones({3, 2, 2}, core::Device::CUDA);
        auto wrong_grad = core::Tensor::ones({3, 2, 1}, core::Device::CUDA);

        EXPECT_THROW((void)grid.apply(cpu_image, 0), std::invalid_argument);
        EXPECT_THROW((void)grid.apply(wrong_channels, 0), std::invalid_argument);
        EXPECT_THROW((void)grid.backward(image, wrong_grad, 0), std::invalid_argument);
    }

    TEST(BilateralGridValidationTest, SingletonImageAxesRemainFinite) {
        lfs::training::BilateralGrid grid(1, 2, 2, 2, 100);
        auto image = core::Tensor::ones({3, 1, 1}, core::Device::CUDA);
        auto grad = core::Tensor::ones({3, 1, 1}, core::Device::CUDA);

        const auto output = grid.apply(image, 0).cpu().to_vector();
        const auto grad_input = grid.backward(image, grad, 0).cpu().to_vector();
        ASSERT_EQ(output.size(), 3u);
        ASSERT_EQ(grad_input.size(), 3u);
        for (const float value : output)
            EXPECT_TRUE(std::isfinite(value));
        for (const float value : grad_input)
            EXPECT_TRUE(std::isfinite(value));
    }

    TEST(TrainerConstructionTest, RejectsInvalidSceneBeforeAllocatingCudaResources) {
        core::Scene scene;
        const auto before = core::PinnedMemoryAllocator::instance().get_stats();

        try {
            training::Trainer trainer(scene);
            FAIL() << "Trainer accepted a scene without cameras";
        } catch (const std::runtime_error& error) {
            EXPECT_NE(std::string_view(error.what()).find("Scene has no cameras"),
                      std::string_view::npos);
        }

        const auto after = core::PinnedMemoryAllocator::instance().get_stats();
        EXPECT_EQ(after.allocated_bytes, before.allocated_bytes);
        EXPECT_EQ(after.cached_bytes, before.cached_bytes);
        EXPECT_EQ(after.num_allocs, before.num_allocs);
        EXPECT_EQ(after.num_deallocs, before.num_deallocs);
    }

    TEST(TrainerConstructionTest, CreatesAndReleasesCudaResourcesForValidScene) {
        core::Scene scene;
        const core::NodeId cameras = scene.addGroup("Cameras");
        scene.addCamera("camera.png", cameras, std::make_shared<core::Camera>());
        const auto before = core::PinnedMemoryAllocator::instance().get_stats();

        {
            training::Trainer trainer(scene);
            const auto active = core::PinnedMemoryAllocator::instance().get_stats();
            EXPECT_GT(active.allocated_bytes, before.allocated_bytes);
        }

        const auto after = core::PinnedMemoryAllocator::instance().get_stats();
        EXPECT_EQ(after.allocated_bytes, before.allocated_bytes);
        EXPECT_GT(after.num_deallocs, before.num_deallocs);
    }

    TEST(TrainerConstructionTest, ParameterSnapshotsStayGenerationConsistent) {
        core::Scene scene;
        const core::NodeId cameras = scene.addGroup("Cameras");
        scene.addCamera("camera.png", cameras, std::make_shared<core::Camera>());
        training::Trainer trainer(scene);

        core::param::TrainingParameters initial;
        initial.optimization.iterations = 1;
        initial.dataset.output_name = "generation_1";
        trainer.setParams(initial);

        std::atomic<bool> writer_done{false};
        std::thread writer([&] {
            for (size_t generation = 2; generation <= 2'000; ++generation) {
                auto params = trainer.getParams();
                params.optimization.iterations = generation;
                params.dataset.output_name = "generation_" + std::to_string(generation);
                trainer.setParams(params);
            }
            writer_done.store(true, std::memory_order_release);
        });

        while (!writer_done.load(std::memory_order_acquire)) {
            const auto snapshot = trainer.getParams();
            EXPECT_EQ(snapshot.dataset.output_name,
                      "generation_" + std::to_string(snapshot.optimization.iterations));
        }
        writer.join();
    }

    TEST(TrainerConstructionTest, InitializeRejectsInvalidIntervalsBeforeTraining) {
        core::Scene scene;
        const core::NodeId cameras = scene.addGroup("Cameras");
        scene.addCamera("camera.png", cameras, std::make_shared<core::Camera>());
        training::Trainer trainer(scene);

        core::param::TrainingParameters params;
        params.optimization.refine_every = 0;
        const auto result = trainer.initialize(params);

        ASSERT_FALSE(result);
        EXPECT_NE(result.error().find("refine_every"), std::string::npos);
        EXPECT_FALSE(trainer.isInitialized());
    }

    TEST(TrainerConstructionTest, ManagerClearReleasesTrainerResourcesAndPoolCache) {
        core::Scene scene;
        const core::NodeId cameras = scene.addGroup("Cameras");
        scene.addCamera("camera.png", cameras, std::make_shared<core::Camera>());
        lfs::vis::TrainerManager manager;
        manager.setScene(&scene);
        const auto before = core::PinnedMemoryAllocator::instance().get_stats();

        manager.setTrainer(std::make_unique<training::Trainer>(scene));
        const auto active = core::PinnedMemoryAllocator::instance().get_stats();
        ASSERT_TRUE(manager.hasTrainer());
        EXPECT_GT(active.allocated_bytes, before.allocated_bytes);

        ASSERT_TRUE(manager.clearTrainer());

        const auto after = core::PinnedMemoryAllocator::instance().get_stats();
        EXPECT_FALSE(manager.hasTrainer());
        EXPECT_EQ(manager.splatExportableStorage(), nullptr);
        EXPECT_EQ(after.allocated_bytes, before.allocated_bytes);
        EXPECT_EQ(after.cached_bytes, 0u);
    }

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

        core::MeshData make_test_mesh_data(const float x_offset = 0.0f) {
            return core::MeshData(
                core::Tensor::from_vector(
                    std::vector<float>{x_offset - 1.0f, -1.0f, 0.0f,
                                       x_offset + 1.0f, -1.0f, 0.0f,
                                       x_offset, 1.0f, 0.0f},
                    {size_t{3}, size_t{3}}, core::Device::CPU),
                core::Tensor::from_vector(
                    std::vector<int32_t>{0, 1, 2},
                    {size_t{1}, size_t{3}}, core::Device::CPU));
        }

        std::shared_ptr<core::MeshData> make_test_mesh(const core::Device device = core::Device::CPU) {
            auto mesh = std::make_shared<core::MeshData>(make_test_mesh_data());
            if (device == core::Device::CUDA) {
                mesh->vertices = mesh->vertices.to(device);
                mesh->indices = mesh->indices.to(device);
            }
            return mesh;
        }

        struct TrainingSceneNodes {
            core::NodeId dataset = core::NULL_NODE;
            core::NodeId cameras = core::NULL_NODE;
            core::NodeId train_group = core::NULL_NODE;
            core::NodeId camera = core::NULL_NODE;
            core::NodeId model = core::NULL_NODE;
        };

        struct ScopedServicesClear {
            ScopedServicesClear() { lfs::vis::services().clear(); }
            ~ScopedServicesClear() { lfs::vis::services().clear(); }
        };

        TrainingSceneNodes build_training_scene(lfs::vis::SceneManager& manager) {
            auto& scene = manager.getScene();
            TrainingSceneNodes nodes;
            nodes.dataset = scene.addDataset("Dataset");
            nodes.model = scene.addSplat("Model", make_test_splat(1), nodes.dataset);
            scene.setTrainingModelNode("Model");
            nodes.cameras = scene.addGroup("Cameras", nodes.dataset);
            nodes.train_group = scene.addCameraGroup("Training (1)", nodes.cameras, 1);
            nodes.camera = scene.addCamera("cam_0001.png", nodes.train_group, std::make_shared<core::Camera>());
            return nodes;
        }

        bool transition_trainer_manager_for_test(lfs::vis::TrainerManager& trainer_manager,
                                                 lfs::vis::TrainingState state) {
            auto& state_machine = const_cast<lfs::vis::TrainingStateMachine&>(trainer_manager.getStateMachine());
            return state_machine.transitionTo(state);
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

    TEST_F(SceneValidityTest, SelectionNodeMaskQueriesReturnOwnedSnapshots) {
        const auto first_id = dummy_scene_.addSplat("first", make_test_splat(1));
        const auto second_id = dummy_scene_.addSplat("second", make_test_splat(1));
        lfs::vis::SelectionState selection;
        static_assert(std::is_same_v<decltype(selection.getNodeMask(dummy_scene_)), std::vector<bool>>);

        selection.selectNode(first_id);
        const auto first_snapshot = selection.getNodeMask(dummy_scene_);
        ASSERT_EQ(first_snapshot.size(), 2u);
        EXPECT_TRUE(first_snapshot[0]);
        EXPECT_FALSE(first_snapshot[1]);

        selection.selectNode(second_id);
        const auto second_snapshot = selection.getNodeMask(dummy_scene_);
        EXPECT_EQ(first_snapshot, (std::vector<bool>{true, false}));
        EXPECT_EQ(second_snapshot, (std::vector<bool>{false, true}));
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
        dummy_scene_.addSplat("Model", make_test_splat(2));
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

    TEST_F(SceneValidityTest, TrainingModelAccessDoesNotDependOnSceneVisibility) {
        constexpr size_t count = 4;
        dummy_scene_.addSplat("Model", make_test_splat(count));
        dummy_scene_.setTrainingModelNode("Model");

        dummy_scene_.setNodeVisibility("Model", false);

        EXPECT_NE(dummy_scene_.getTrainingModel(), nullptr);
        EXPECT_FALSE(dummy_scene_.isTrainingModelEffectivelyVisible());
        EXPECT_EQ(dummy_scene_.getTrainingModelGaussianCount(), count);
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
        dummy_scene_.addSplat("Model", make_test_splat(count, 3));
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
        dummy_scene_.addSplat("Model", make_test_splat(count, 1));
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
        dummy_scene_.addSplat("Model", std::move(model));
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

    TEST_F(SceneValidityTest, MigrateTrainingModelCanForceRehomeAllocatorBackedModel) {
        constexpr size_t count = 4;
        constexpr size_t capacity = 16;
        constexpr int sh_degree = 1;
        const auto rest_coeffs = core::sh_rest_coefficients_for_degree(sh_degree);
        std::vector<std::shared_ptr<std::vector<float>>> owners;

        auto model = std::make_unique<core::SplatData>(
            sh_degree,
            make_external_float_tensor(owners, {count, size_t{3}}, capacity, "vulkan_external_buffer"),
            make_external_float_tensor(owners, {count, size_t{1}, size_t{3}}, capacity, "vulkan_external_buffer"),
            make_external_float_tensor(owners,
                                       {core::sh_swizzled_float_count(count, rest_coeffs)},
                                       core::sh_swizzled_float_count(capacity, rest_coeffs),
                                       "vulkan_external_buffer"),
            make_external_float_tensor(owners, {count, size_t{3}}, capacity, "vulkan_external_buffer"),
            make_external_float_tensor(owners, {count, size_t{4}}, capacity, "vulkan_external_buffer"),
            make_external_float_tensor(owners, {count, size_t{1}}, capacity, "vulkan_external_buffer"),
            1.0f,
            core::SplatData::ShNLayout::Swizzled);
        dummy_scene_.addSplat("Model", std::move(model));
        dummy_scene_.setTrainingModelNode("Model");

        core::param::TrainingParameters params;
        params.optimization.sh_degree = sh_degree;
        params.optimization.max_cap = static_cast<int>(capacity);

        int allocation_calls = 0;
        core::SplatTensorAllocator allocator =
            [&allocation_calls](core::TensorShape shape,
                                const size_t requested_capacity,
                                const core::DataType,
                                const std::string_view name) {
                ++allocation_calls;
                auto tensor = core::Tensor::zeros_direct(std::move(shape), requested_capacity, core::Device::CUDA);
                tensor.set_name(std::string{name});
                return tensor;
            };

        auto* training_model = dummy_scene_.getTrainingModel();
        ASSERT_NE(training_model, nullptr);
        const auto result =
            lfs::training::migrateTrainingModelToAllocator(params, *training_model, allocator, true);

        ASSERT_TRUE(result.has_value()) << result.error();
        EXPECT_EQ(allocation_calls, 6);
        EXPECT_EQ(training_model->means_raw().capacity(), capacity);
        EXPECT_EQ(training_model->shN_raw().capacity(),
                  core::sh_swizzled_float_count(capacity, rest_coeffs));
        EXPECT_FALSE(training_model->means_raw().is_external_storage());
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

    TEST_F(SceneValidityTest, SceneManagerClearReleasesMeshRayPickCpuCache) {
        lfs::vis::SceneManager scene_manager;
        auto mesh = make_test_mesh(core::Device::CUDA);
        ASSERT_NE(scene_manager.getScene().addMesh("Triangle", mesh), core::NULL_NODE);
        const auto before_pick = core::PinnedMemoryAllocator::instance().get_stats();

        EXPECT_EQ(scene_manager.pickNodeByRay({0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}),
                  "Triangle");
        const auto after_pick = core::PinnedMemoryAllocator::instance().get_stats();
        ASSERT_GT(after_pick.allocated_bytes, before_pick.allocated_bytes);

        ASSERT_TRUE(scene_manager.clear());

        const auto after_clear = core::PinnedMemoryAllocator::instance().get_stats();
        EXPECT_EQ(after_clear.allocated_bytes, before_pick.allocated_bytes);
    }

    TEST_F(SceneValidityTest, MeshRayPickCacheRejectsReusedObjectAddressAndNodeId) {
        alignas(core::MeshData) std::byte storage[sizeof(core::MeshData)];
        lfs::vis::SceneManager scene_manager;
        const auto construct_mesh = [&](const float x_offset) {
            auto* mesh = std::construct_at(
                reinterpret_cast<core::MeshData*>(storage),
                make_test_mesh_data(x_offset));
            return std::shared_ptr<core::MeshData>(mesh, [](core::MeshData* value) {
                std::destroy_at(value);
            });
        };

        auto original = construct_mesh(0.0f);
        const core::NodeId original_id = scene_manager.getScene().addMesh("Original", original);
        ASSERT_NE(original_id, core::NULL_NODE);
        ASSERT_EQ(scene_manager.pickNodeByRay({0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}),
                  "Original");

        // Bypass the manager cache clear to exercise the identity guard directly.
        scene_manager.getScene().clear();
        original.reset();
        auto replacement = construct_mesh(10.0f);
        const core::NodeId replacement_id =
            scene_manager.getScene().addMesh("Replacement", replacement);
        ASSERT_EQ(replacement_id, original_id);

        EXPECT_TRUE(scene_manager.pickNodeByRay({0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}).empty());
    }

    TEST_F(SceneValidityTest, MoveNodeIntoGroupAppendsAsChild) {
        const auto a = dummy_scene_.addSplat("A", make_test_splat(1));
        const auto g = dummy_scene_.addGroup("G");

        ASSERT_TRUE(dummy_scene_.moveNode(a, g, -1));

        const auto* group = dummy_scene_.getNodeById(g);
        ASSERT_NE(group, nullptr);
        ASSERT_EQ(group->children.size(), 1u);
        EXPECT_EQ(group->children[0], a);
        EXPECT_EQ(dummy_scene_.getNodeById(a)->parent_id, g);
    }

    TEST_F(SceneValidityTest, MoveNodeOutToRoot) {
        const auto g = dummy_scene_.addGroup("G");
        const auto a = dummy_scene_.addSplat("A", make_test_splat(1), g);
        ASSERT_EQ(dummy_scene_.getNodeById(a)->parent_id, g);

        ASSERT_TRUE(dummy_scene_.moveNode(a, core::NULL_NODE, -1));

        EXPECT_EQ(dummy_scene_.getNodeById(a)->parent_id, core::NULL_NODE);
        EXPECT_TRUE(dummy_scene_.getNodeById(g)->children.empty());
        const auto roots = dummy_scene_.getRootNodes();
        EXPECT_NE(std::find(roots.begin(), roots.end(), a), roots.end());
    }

    TEST_F(SceneValidityTest, MoveNodeReordersWithinGroup) {
        const auto g = dummy_scene_.addGroup("G");
        const auto a = dummy_scene_.addSplat("A", make_test_splat(1), g);
        const auto b = dummy_scene_.addSplat("B", make_test_splat(1), g);
        const auto c = dummy_scene_.addSplat("C", make_test_splat(1), g);

        ASSERT_TRUE(dummy_scene_.moveNode(a, g, 3));

        const auto& children = dummy_scene_.getNodeById(g)->children;
        ASSERT_EQ(children.size(), 3u);
        EXPECT_EQ(children[0], b);
        EXPECT_EQ(children[1], c);
        EXPECT_EQ(children[2], a);
    }

    TEST_F(SceneValidityTest, MoveNodeReordersWithinRoot) {
        const auto a = dummy_scene_.addSplat("A", make_test_splat(1));
        const auto b = dummy_scene_.addSplat("B", make_test_splat(1));
        const auto c = dummy_scene_.addSplat("C", make_test_splat(1));

        ASSERT_TRUE(dummy_scene_.moveNode(c, core::NULL_NODE, 0));

        const auto roots = dummy_scene_.getRootNodes();
        ASSERT_EQ(roots.size(), 3u);
        EXPECT_EQ(roots[0], c);
        EXPECT_EQ(roots[1], a);
        EXPECT_EQ(roots[2], b);
    }

    TEST_F(SceneValidityTest, MoveNodeRejectsCycleIntoOwnDescendant) {
        const auto parent = dummy_scene_.addGroup("Parent");
        const auto child = dummy_scene_.addGroup("Child", parent);

        EXPECT_FALSE(dummy_scene_.moveNode(parent, child, -1));
        EXPECT_EQ(dummy_scene_.getNodeById(parent)->parent_id, core::NULL_NODE);
        EXPECT_EQ(dummy_scene_.getNodeById(child)->parent_id, parent);
    }

    TEST_F(SceneValidityTest, MoveNodeNoOpReturnsFalse) {
        const auto g = dummy_scene_.addGroup("G");
        const auto a = dummy_scene_.addSplat("A", make_test_splat(1), g);
        const auto b = dummy_scene_.addSplat("B", make_test_splat(1), g);

        EXPECT_FALSE(dummy_scene_.moveNode(a, g, 0));
        EXPECT_FALSE(dummy_scene_.moveNode(a, g, 1));

        const auto& children = dummy_scene_.getNodeById(g)->children;
        ASSERT_EQ(children.size(), 2u);
        EXPECT_EQ(children[0], a);
        EXPECT_EQ(children[1], b);
    }

    TEST_F(SceneValidityTest, GroupTransformPropagatesToChildWorldTransform) {
        const auto group = dummy_scene_.addGroup("Group");
        const auto splat = dummy_scene_.addSplat("Child", make_test_splat(1), group);

        const glm::mat4 before = dummy_scene_.getWorldTransform(splat);

        glm::mat4 t(1.0f);
        t[3] = glm::vec4(5.0f, 0.0f, 0.0f, 1.0f);
        dummy_scene_.setNodeTransform("Group", t);

        const glm::mat4 after = dummy_scene_.getWorldTransform(splat);

        EXPECT_NE(before[3].x, after[3].x);
        EXPECT_FLOAT_EQ(after[3].x - before[3].x, 5.0f);
    }

    TEST_F(SceneValidityTest, MoveNodeOutOfTransformedGroupPreservesWorldTransform) {
        const auto group = dummy_scene_.addGroup("Group");
        const auto splat = dummy_scene_.addSplat("Child", make_test_splat(1), group);

        glm::mat4 group_t(1.0f);
        group_t = glm::rotate(group_t, 0.7f, glm::vec3(0.0f, 0.0f, 1.0f));
        group_t[3] = glm::vec4(3.0f, 1.0f, -2.0f, 1.0f);
        dummy_scene_.setNodeTransform("Group", group_t);

        const glm::mat4 world_before = dummy_scene_.getWorldTransform(splat);

        ASSERT_TRUE(dummy_scene_.moveNode(splat, core::NULL_NODE, -1));
        ASSERT_EQ(dummy_scene_.getNodeById(splat)->parent_id, core::NULL_NODE);

        const glm::mat4 world_after = dummy_scene_.getWorldTransform(splat);
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row)
                EXPECT_NEAR(world_before[col][row], world_after[col][row], 1e-4f);
    }

    TEST_F(SceneValidityTest, ReparentIntoTransformedGroupPreservesWorldTransform) {
        const auto group = dummy_scene_.addGroup("Group");
        const auto splat = dummy_scene_.addSplat("Child", make_test_splat(1));

        glm::mat4 group_t(1.0f);
        group_t = glm::rotate(group_t, -0.4f, glm::vec3(0.0f, 1.0f, 0.0f));
        group_t[3] = glm::vec4(-1.0f, 2.0f, 4.0f, 1.0f);
        dummy_scene_.setNodeTransform("Group", group_t);

        glm::mat4 splat_t(1.0f);
        splat_t[3] = glm::vec4(5.0f, 0.0f, 0.0f, 1.0f);
        dummy_scene_.setNodeTransform("Child", splat_t);

        const glm::mat4 world_before = dummy_scene_.getWorldTransform(splat);

        ASSERT_TRUE(dummy_scene_.reparent(splat, group));

        const glm::mat4 world_after = dummy_scene_.getWorldTransform(splat);
        for (int col = 0; col < 4; ++col)
            for (int row = 0; row < 4; ++row)
                EXPECT_NEAR(world_before[col][row], world_after[col][row], 1e-4f);
    }

    TEST_F(SceneValidityTest, SceneManagerMoveNodeReparentsIntoGroup) {
        lfs::vis::SceneManager sm;
        auto& scene = sm.getScene();
        const auto group = scene.addGroup("Group");
        const auto splat = scene.addSplat("Splat", make_test_splat(1));
        ASSERT_EQ(scene.getNodeById(splat)->parent_id, core::NULL_NODE);

        EXPECT_TRUE(sm.moveNode(splat, group, -1));

        ASSERT_NE(scene.getNodeById(splat), nullptr);
        EXPECT_EQ(scene.getNodeById(splat)->parent_id, group);
        const auto& children = scene.getNodeById(group)->children;
        ASSERT_EQ(children.size(), 1u);
        EXPECT_EQ(children[0], splat);
    }

    TEST_F(SceneValidityTest, SceneManagerMoveNodeMovesOutToRoot) {
        lfs::vis::SceneManager sm;
        auto& scene = sm.getScene();
        const auto group = scene.addGroup("Group");
        const auto splat = scene.addSplat("Splat", make_test_splat(1), group);
        ASSERT_EQ(scene.getNodeById(splat)->parent_id, group);

        EXPECT_TRUE(sm.moveNode(splat, core::NULL_NODE, -1));

        EXPECT_EQ(scene.getNodeById(splat)->parent_id, core::NULL_NODE);
        EXPECT_TRUE(scene.getNodeById(group)->children.empty());
    }

    TEST_F(SceneValidityTest, SceneManagerBlocksActiveCameraSubtreeAndAllowsInactiveCameraRemoval) {
        const ScopedServicesClear services_scope;
        lfs::vis::SceneManager sm;
        lfs::vis::TrainerManager trainer_manager;
        lfs::vis::services().set(&sm);
        lfs::vis::services().set(&trainer_manager);

        auto& scene = sm.getScene();
        const auto nodes = build_training_scene(sm);
        sm.selectNodesById({nodes.camera});
        ASSERT_EQ(scene.getActiveCameraCount(), 1u);
        ASSERT_EQ(scene.getTrainingModelNodeName(), "Model");

        trainer_manager.setScene(&scene);
        ASSERT_TRUE(transition_trainer_manager_for_test(trainer_manager, lfs::vis::TrainingState::Paused));
        ASSERT_TRUE(transition_trainer_manager_for_test(trainer_manager, lfs::vis::TrainingState::Running));
        ASSERT_TRUE(trainer_manager.isRunning());
        ASSERT_FALSE(trainer_manager.canPerform(lfs::vis::TrainingAction::DeleteTrainingNode));

        sm.removePLY("Training (1)");

        EXPECT_NE(scene.getNodeById(nodes.train_group), nullptr);
        EXPECT_NE(scene.getNodeById(nodes.camera), nullptr);
        EXPECT_EQ(scene.getActiveCameraCount(), 1u);
        EXPECT_EQ(scene.getTrainingModelNodeName(), "Model");

        const auto selected = sm.getSelectedNodeNames();
        ASSERT_EQ(selected.size(), 1u);
        EXPECT_EQ(selected[0], "cam_0001.png");

        ASSERT_TRUE(transition_trainer_manager_for_test(trainer_manager, lfs::vis::TrainingState::Stopping));
        ASSERT_EQ(trainer_manager.getState(), lfs::vis::TrainingState::Stopping);
        ASSERT_FALSE(trainer_manager.canPerform(lfs::vis::TrainingAction::DeleteTrainingNode));

        sm.removePLY("Training (1)");

        EXPECT_NE(scene.getNodeById(nodes.train_group), nullptr);
        EXPECT_NE(scene.getNodeById(nodes.camera), nullptr);
        EXPECT_EQ(scene.getActiveCameraCount(), 1u);
        EXPECT_EQ(scene.getTrainingModelNodeName(), "Model");
        ASSERT_EQ(sm.getSelectedNodeNames().size(), 1u);

        scene.setCameraTrainingEnabled(nodes.camera, false);
        ASSERT_EQ(scene.getActiveCameraCount(), 0u);
        ASSERT_EQ(scene.getAllCameras().size(), 1u);

        sm.removePLY("cam_0001.png");

        EXPECT_EQ(scene.getNodeById(nodes.camera), nullptr);
        EXPECT_TRUE(sm.getSelectedNodeNames().empty());
        EXPECT_EQ(scene.getAllCameras().size(), 0u);
        EXPECT_EQ(scene.getActiveCameraCount(), 0u);
        EXPECT_EQ(scene.getTrainingModelNodeName(), "Model");
    }

    TEST_F(SceneValidityTest, SceneManagerRemovesAllowedActiveCameraSubtreeWithoutClearingTrainer) {
        const ScopedServicesClear services_scope;
        lfs::vis::SceneManager sm;
        lfs::vis::TrainerManager trainer_manager;
        lfs::vis::services().set(&sm);
        lfs::vis::services().set(&trainer_manager);

        auto& scene = sm.getScene();
        const auto nodes = build_training_scene(sm);
        sm.selectNodesById({nodes.camera});

        trainer_manager.setScene(&scene);
        ASSERT_TRUE(transition_trainer_manager_for_test(trainer_manager, lfs::vis::TrainingState::Ready));
        ASSERT_TRUE(trainer_manager.canPerform(lfs::vis::TrainingAction::DeleteTrainingNode));

        sm.removePLY("Training (1)");

        EXPECT_EQ(scene.getNodeById(nodes.train_group), nullptr);
        EXPECT_EQ(scene.getNodeById(nodes.camera), nullptr);
        EXPECT_EQ(scene.getAllCameras().size(), 0u);
        EXPECT_EQ(scene.getActiveCameraCount(), 0u);
        EXPECT_TRUE(sm.getSelectedNodeNames().empty());
        EXPECT_NE(scene.getNodeById(nodes.model), nullptr);
        EXPECT_EQ(scene.getTrainingModelNodeName(), "Model");
        EXPECT_EQ(trainer_manager.getState(), lfs::vis::TrainingState::Ready);
        EXPECT_TRUE(trainer_manager.canPerform(lfs::vis::TrainingAction::DeleteTrainingNode));
    }

    TEST_F(SceneValidityTest, SceneManagerRemovesAllowedTrainingModelAndClearsTrainerState) {
        const ScopedServicesClear services_scope;
        lfs::vis::SceneManager sm;
        lfs::vis::TrainerManager trainer_manager;
        lfs::vis::services().set(&sm);
        lfs::vis::services().set(&trainer_manager);

        auto& scene = sm.getScene();
        const auto nodes = build_training_scene(sm);

        trainer_manager.setScene(&scene);
        ASSERT_TRUE(transition_trainer_manager_for_test(trainer_manager, lfs::vis::TrainingState::Ready));
        ASSERT_TRUE(trainer_manager.canPerform(lfs::vis::TrainingAction::DeleteTrainingNode));

        sm.removePLY("Model");

        EXPECT_EQ(scene.getNodeById(nodes.model), nullptr);
        EXPECT_EQ(scene.getTrainingModelNodeName(), "");
        EXPECT_EQ(trainer_manager.getState(), lfs::vis::TrainingState::Idle);
    }

} // namespace lfs::python

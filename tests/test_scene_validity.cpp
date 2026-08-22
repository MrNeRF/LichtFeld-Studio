// SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <future>
#include <glm/gtc/matrix_transform.hpp>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "core/cuda/sh_layout.cuh"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/parameters.hpp"
#include "core/pinned_memory_allocator.hpp"
#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/tensor.hpp"
#include "io/exporter.hpp"
#include "ppisp_fixture.hpp"
#include "python/python_runtime.hpp"
#include "training/components/bilateral_grid.hpp"
#include "training/components/ppisp.hpp"
#include "training/components/ppisp_file.hpp"
#include "training/control/command_api.hpp"
#include "training/optimizer/adam_optimizer.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include "visualizer/app_store.hpp"
#include "visualizer/core/services.hpp"
#include "visualizer/operation/undo_history.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/scene/selection_state.hpp"
#include "visualizer/sequencer/sequencer_controller.hpp"
#include "visualizer/training/training_manager.hpp"

namespace lfs::python {

    namespace {
        std::shared_ptr<core::Camera> make_test_camera() {
            return std::make_shared<core::Camera>(
                core::Tensor::eye(3, core::Device::CPU),
                core::Tensor::zeros({3}, core::Device::CPU),
                100.0f, 100.0f, 32.0f, 32.0f,
                core::Tensor(), core::Tensor(), core::CameraModelType::PINHOLE,
                "camera.png", std::filesystem::path{}, std::filesystem::path{},
                64, 64, 0);
        }

        class ScopedTestDirectory {
        public:
            explicit ScopedTestDirectory(const std::string_view name)
                : path_(std::filesystem::temp_directory_path() / std::string(name)) {
                std::error_code ec;
                std::filesystem::remove_all(path_, ec);
                std::filesystem::create_directories(path_);
            }

            ~ScopedTestDirectory() {
                std::error_code ec;
                std::filesystem::remove_all(path_, ec);
            }

            [[nodiscard]] const std::filesystem::path& path() const { return path_; }

        private:
            std::filesystem::path path_;
        };

        bool transition_trainer_manager_for_test(lfs::vis::TrainerManager& trainer_manager,
                                                 lfs::vis::TrainingState state);
    } // namespace

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
        scene.addCamera("camera.png", cameras, make_test_camera());
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
        scene.addCamera("camera.png", cameras, make_test_camera());
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

        const auto final_snapshot = trainer.getParams();
        EXPECT_EQ(final_snapshot.dataset.output_name,
                  "generation_" + std::to_string(final_snapshot.optimization.iterations));
    }

    TEST(TrainerConstructionTest, InitializeRejectsInvalidIntervalsBeforeTraining) {
        core::Scene scene;
        const core::NodeId cameras = scene.addGroup("Cameras");
        scene.addCamera("camera.png", cameras, make_test_camera());
        training::Trainer trainer(scene);

        core::param::TrainingParameters params;
        params.optimization.refine_every = 0;
        const auto result = trainer.initialize(params);

        ASSERT_FALSE(result);
        EXPECT_NE(result.error().find("refine_every"), std::string::npos);
        EXPECT_FALSE(trainer.isInitialized());
    }

    TEST(TrainerConstructionTest, InitializeDoesNotRejectGutWithShRest) {
        core::Scene scene;
        const core::NodeId cameras = scene.addGroup("Cameras");
        scene.addCamera("camera.png", cameras, make_test_camera());
        training::Trainer trainer(scene);

        core::param::TrainingParameters params;
        params.optimization.gut = true;
        params.optimization.sh_degree = 1;
        const auto result = trainer.initialize(params);
        if (!result) {
            EXPECT_EQ(result.error().find("unfused"), std::string::npos);
            EXPECT_EQ(result.error().find("joint-codec SH-rest"), std::string::npos);
            EXPECT_EQ(result.error().find("sh_degree=0"), std::string::npos);
        }
    }

    TEST(TrainerConstructionTest, ExportableDensifyBarrierDistinguishesAbsentAndFailed) {
        core::Scene scene;
        const core::NodeId cameras = scene.addGroup("Cameras");
        scene.addCamera("camera.png", cameras, make_test_camera());
        training::Trainer trainer(scene);
        using Begin = training::Trainer::ExportableDensifyBarrierBegin;

        EXPECT_EQ(trainer.beginExportableDensifyBarrier(), Begin::NotInstalled);

        int begin_calls = 0;
        int end_calls = 0;
        trainer.setExportableDensifyBarrier(
            [&] {
                ++begin_calls;
                return false;
            },
            [&] {
                ++end_calls;
                return true;
            });
        EXPECT_EQ(trainer.beginExportableDensifyBarrier(), Begin::Failed);
        EXPECT_TRUE(trainer.endExportableDensifyBarrier());
        EXPECT_EQ(begin_calls, 1);
        EXPECT_EQ(end_calls, 0);

        trainer.setExportableDensifyBarrier(
            [&] {
                ++begin_calls;
                return true;
            },
            [&] {
                ++end_calls;
                return true;
            });
        EXPECT_EQ(trainer.beginExportableDensifyBarrier(), Begin::Acquired);
        EXPECT_EQ(trainer.beginExportableDensifyBarrier(), Begin::Acquired);
        EXPECT_TRUE(trainer.endExportableDensifyBarrier());
        EXPECT_EQ(end_calls, 0);
        EXPECT_TRUE(trainer.endExportableDensifyBarrier());
        EXPECT_EQ(begin_calls, 2);
        EXPECT_EQ(end_calls, 1);
    }

    TEST(TrainerConstructionTest, ExportableDensifyBarrierEndFailureStopsTrainer) {
        core::Scene scene;
        const core::NodeId cameras = scene.addGroup("Cameras");
        scene.addCamera("camera.png", cameras, make_test_camera());
        training::Trainer trainer(scene);
        using Begin = training::Trainer::ExportableDensifyBarrierBegin;

        int end_calls = 0;
        trainer.setExportableDensifyBarrier(
            [] { return true; },
            [&] {
                ++end_calls;
                return false;
            });

        EXPECT_EQ(trainer.beginExportableDensifyBarrier(), Begin::Acquired);
        EXPECT_FALSE(trainer.endExportableDensifyBarrier());
        EXPECT_EQ(end_calls, 1);
        EXPECT_TRUE(trainer.has_stopped());
    }

    TEST(TrainerConstructionTest, ManagerClearReleasesTrainerResourcesAndPoolCache) {
        core::Scene scene;
        const core::NodeId cameras = scene.addGroup("Cameras");
        scene.addCamera("camera.png", cameras, make_test_camera());
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

    TEST(TrainerConstructionTest, ClearTrainerSuppressesStoppedNotificationForPausedTrainer) {
        core::Scene scene;
        const core::NodeId cameras = scene.addGroup("Cameras");
        scene.addCamera("camera.png", cameras, make_test_camera());
        lfs::vis::TrainerManager manager;
        manager.setScene(&scene);
        manager.setTrainer(std::make_unique<training::Trainer>(scene));

        ASSERT_TRUE(transition_trainer_manager_for_test(manager, lfs::vis::TrainingState::Running));
        ASSERT_TRUE(transition_trainer_manager_for_test(manager, lfs::vis::TrainingState::Paused));

        int completions = 0;
        bool user_stopped = false;
        bool suppress_notification = false;
        const auto id = lfs::core::events::state::TrainingCompleted::when([&](const auto& e) {
            ++completions;
            user_stopped = e.user_stopped;
            suppress_notification = e.suppress_notification;
        });

        ASSERT_TRUE(manager.clearTrainer());
        EXPECT_EQ(completions, 1);
        EXPECT_TRUE(user_stopped);
        EXPECT_TRUE(suppress_notification);
        EXPECT_EQ(manager.getState(), lfs::vis::TrainingState::Idle);

        lfs::event::EventBridge::instance().unsubscribe(
            typeid(lfs::core::events::state::TrainingCompleted), id);
    }

    TEST(TrainerConstructionTest, ClearTrainerResetsPausedCommandCenterSnapshot) {
        core::Scene scene;
        const core::NodeId cameras = scene.addGroup("Cameras");
        scene.addCamera("camera.png", cameras, make_test_camera());
        lfs::vis::TrainerManager manager;
        manager.setScene(&scene);
        manager.setTrainer(std::make_unique<training::Trainer>(scene));

        ASSERT_TRUE(transition_trainer_manager_for_test(manager, lfs::vis::TrainingState::Running));
        ASSERT_TRUE(transition_trainer_manager_for_test(manager, lfs::vis::TrainingState::Paused));

        auto& command_center = lfs::training::CommandCenter::instance();
        command_center.reset_snapshot();
        const lfs::training::HookContext context{.iteration = 8200};
        command_center.update_snapshot(
            context, 0, true, false, false, lfs::training::TrainingPhase::Idle);
        {
            const auto paused = command_center.snapshot();
            ASSERT_TRUE(paused.is_paused);
            EXPECT_FALSE(paused.is_running);
            EXPECT_EQ(paused.iteration, 8200);
            EXPECT_EQ(paused.trainer, nullptr);
        }

        ASSERT_TRUE(manager.clearTrainer());

        const auto snapshot = command_center.snapshot();
        EXPECT_FALSE(snapshot.is_paused);
        EXPECT_FALSE(snapshot.is_running);
        EXPECT_FALSE(snapshot.stop_requested);
        EXPECT_EQ(snapshot.iteration, 0);
        EXPECT_EQ(snapshot.max_iterations, 0);
        EXPECT_EQ(snapshot.trainer, nullptr);
        EXPECT_EQ(snapshot.phase, lfs::training::TrainingPhase::Idle);
        EXPECT_EQ(manager.getState(), lfs::vis::TrainingState::Idle);
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

        core::Tensor make_test_selection_mask(const std::vector<uint8_t>& values) {
            auto tensor = core::Tensor::empty(
                {values.size()}, core::Device::CPU, core::DataType::UInt8);
            std::copy(values.begin(), values.end(), tensor.ptr<uint8_t>());
            return tensor;
        }

        std::vector<uint8_t> selection_mask_values(const core::Scene& scene) {
            const auto mask = scene.getSelectionMask();
            return mask && mask->is_valid() ? mask->cpu().to_vector_uint8()
                                            : std::vector<uint8_t>{};
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
            nodes.camera = scene.addCamera("cam_0001.png", nodes.train_group, make_test_camera());
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
            lfs::event::EventBridge::instance().clear_all();
            core::event::bus().clear_all();
            lfs::vis::services().clear();
            lfs::vis::op::undoHistory().clear();
            set_application_scene(nullptr);
        }

        void TearDown() override {
            set_scene_generation_callback(nullptr);
            set_application_scene(nullptr);
            lfs::vis::op::undoHistory().clear();
            lfs::vis::services().clear();
            core::event::bus().clear_all();
            lfs::event::EventBridge::instance().clear_all();
        }

        core::Scene dummy_scene_;
    };

    TEST_F(SceneValidityTest, AddedNodesMintUniqueNonNilUuids) {
        std::unordered_set<core::Uuid> uuids;
        for (int i = 0; i < 128; ++i) {
            const auto id = dummy_scene_.addGroup("Node_" + std::to_string(i));
            ASSERT_NE(id, core::NULL_NODE);
            const auto uuid = dummy_scene_.getNodeUuid(id);
            EXPECT_FALSE(uuid.is_nil());
            EXPECT_TRUE(uuids.insert(uuid).second);
            EXPECT_EQ(dummy_scene_.getNodeIdByUuid(uuid), id);
            EXPECT_EQ(dummy_scene_.getNodeByUuid(uuid), dummy_scene_.getNodeById(id));
        }
    }

    TEST_F(SceneValidityTest, RenamePreservesNodeUuid) {
        const auto id = dummy_scene_.addGroup("Before");
        ASSERT_NE(id, core::NULL_NODE);
        const auto uuid = dummy_scene_.getNodeUuid(id);

        ASSERT_TRUE(dummy_scene_.renameNode(id, "After"));

        EXPECT_EQ(dummy_scene_.getNodeUuid(id), uuid);
        ASSERT_NE(dummy_scene_.getNodeByUuid(uuid), nullptr);
        EXPECT_EQ(dummy_scene_.getNodeByUuid(uuid)->name, "After");
    }

    TEST_F(SceneValidityTest, SequencerUuidResolveWinsOverRenamedReusedDisplayName) {
        const auto original_id = dummy_scene_.addGroup("SequenceFrame");
        ASSERT_NE(original_id, core::NULL_NODE);
        const core::Uuid original_uuid = dummy_scene_.getNodeUuid(original_id);

        ASSERT_TRUE(dummy_scene_.renameNode(original_id, "RenamedFrame"));
        const auto reused_name_id = dummy_scene_.addGroup("SequenceFrame");
        ASSERT_NE(reused_name_id, core::NULL_NODE);

        const auto resolved = lfs::vis::resolvePlySequenceNode(
            dummy_scene_, original_uuid, "SequenceFrame");
        ASSERT_TRUE(resolved.has_value());
        EXPECT_EQ(*resolved, original_id);
        EXPECT_NE(*resolved, reused_name_id);

        core::Uuid missing_uuid;
        do {
            missing_uuid = core::generate_uuid_v4();
        } while (dummy_scene_.getNodeByUuid(missing_uuid));
        const auto missing = lfs::vis::resolvePlySequenceNode(
            dummy_scene_, missing_uuid, "SequenceFrame");
        EXPECT_FALSE(missing.has_value());
        EXPECT_EQ(missing.error().code, lfs::vis::PlySequenceResolveErrorCode::UUID_NOT_FOUND);
        EXPECT_NE(missing.error().message.find(missing_uuid.to_string()), std::string::npos);
    }

    TEST_F(SceneValidityTest, UuidKeyedPlyPathAndTrainingBindingSurviveRename) {
        lfs::vis::SceneManager scene_manager;
        auto& scene = scene_manager.getScene();
        const auto model_id = scene.addSplat("Model", make_test_splat(2));
        ASSERT_NE(model_id, core::NULL_NODE);
        const core::Uuid model_uuid = scene.getNodeUuid(model_id);
        const std::filesystem::path source_path = "uuid-keyed-model.ply";

        scene_manager.setPlyPath(model_uuid, source_path);
        scene.setTrainingModelNode(model_id);
        ASSERT_TRUE(scene_manager.renamePLY("Model", "Renamed Model"));

        EXPECT_EQ(scene.getTrainingModelNodeUuid(), model_uuid);
        EXPECT_EQ(scene.getTrainingModelNodeId(), model_id);
        EXPECT_EQ(scene.getTrainingModelNodeName(), "Renamed Model");
        EXPECT_EQ(scene.getTrainingModel(), scene.getNodeByUuid(model_uuid)->model.get());
        ASSERT_TRUE(scene_manager.getPlyPath(model_uuid).has_value());
        EXPECT_EQ(*scene_manager.getPlyPath(model_uuid), source_path);
        EXPECT_EQ(scene_manager.getPlyPath("Renamed Model"), source_path);
    }

    TEST_F(SceneValidityTest, PlyStateEventsCarryUuidWithoutChangingExistingFields) {
        lfs::vis::SceneManager scene_manager;
        std::optional<core::events::state::PLYAdded> added_event;
        std::optional<core::events::state::PLYRemoved> removed_event;
        auto added_handler = core::events::state::PLYAdded::when(
            [&](const auto& event) { added_event = event; });
        auto removed_handler = core::events::state::PLYRemoved::when(
            [&](const auto& event) { removed_event = event; });

        const std::string group_name = scene_manager.addGroupNode("Event Group");
        ASSERT_FALSE(group_name.empty());
        const auto* group = scene_manager.getScene().getNode(group_name);
        ASSERT_NE(group, nullptr);
        const core::Uuid group_uuid = group->uuid;

        ASSERT_TRUE(added_event.has_value());
        EXPECT_EQ(added_event->name, group_name);
        EXPECT_EQ(added_event->uuid, group_uuid);
        EXPECT_TRUE(added_event->is_group);
        EXPECT_EQ(added_event->node_type, static_cast<int>(core::NodeType::GROUP));

        scene_manager.removePLY(group_name);
        ASSERT_TRUE(removed_event.has_value());
        EXPECT_EQ(removed_event->name, group_name);
        EXPECT_EQ(removed_event->uuid, group_uuid);
        EXPECT_FALSE(removed_event->children_kept);

        (void)added_handler;
        (void)removed_handler;
    }

    TEST_F(SceneValidityTest, UuidIndexInvalidatesOnDeleteAndClear) {
        const auto deleted_id = dummy_scene_.addGroup("Deleted");
        const auto cleared_id = dummy_scene_.addGroup("Cleared");
        ASSERT_NE(deleted_id, core::NULL_NODE);
        ASSERT_NE(cleared_id, core::NULL_NODE);
        const auto deleted_uuid = dummy_scene_.getNodeUuid(deleted_id);
        const auto cleared_uuid = dummy_scene_.getNodeUuid(cleared_id);

        dummy_scene_.removeNodeById(deleted_id);
        EXPECT_EQ(dummy_scene_.getNodeIdByUuid(deleted_uuid), core::NULL_NODE);
        EXPECT_EQ(dummy_scene_.getNodeByUuid(deleted_uuid), nullptr);

        dummy_scene_.clear();
        EXPECT_EQ(dummy_scene_.getNodeIdByUuid(cleared_uuid), core::NULL_NODE);
        EXPECT_EQ(dummy_scene_.getNodeByUuid(cleared_uuid), nullptr);
    }

    TEST_F(SceneValidityTest, RestoreNodeWithUuidRejectsInvalidIdentityAndPreservesProvidedUuid) {
        core::Scene::RestoreNodeDesc nil_desc;
        nil_desc.type = core::NodeType::GROUP;
        nil_desc.name = "Nil";
        EXPECT_EQ(dummy_scene_.restoreNodeWithUuid(std::move(nil_desc)), core::NULL_NODE);

        const auto original_id = dummy_scene_.addGroup("Original");
        ASSERT_NE(original_id, core::NULL_NODE);
        const auto original_uuid = dummy_scene_.getNodeUuid(original_id);

        core::Scene::RestoreNodeDesc duplicate_desc;
        duplicate_desc.uuid = original_uuid;
        duplicate_desc.type = core::NodeType::GROUP;
        duplicate_desc.name = "Duplicate";
        EXPECT_EQ(dummy_scene_.restoreNodeWithUuid(std::move(duplicate_desc)), core::NULL_NODE);
        EXPECT_EQ(dummy_scene_.getNodeIdByUuid(original_uuid), original_id);
        EXPECT_EQ(dummy_scene_.getNode("Duplicate"), nullptr);

        dummy_scene_.removeNodeById(original_id);
        ASSERT_EQ(dummy_scene_.getNodeIdByUuid(original_uuid), core::NULL_NODE);

        core::Scene::RestoreNodeDesc restore_desc;
        restore_desc.uuid = original_uuid;
        restore_desc.type = core::NodeType::GROUP;
        restore_desc.name = "Restored";
        const auto restored_id = dummy_scene_.restoreNodeWithUuid(std::move(restore_desc));
        ASSERT_NE(restored_id, core::NULL_NODE);
        EXPECT_EQ(dummy_scene_.getNodeIdByUuid(original_uuid), restored_id);
        ASSERT_NE(dummy_scene_.getNodeByUuid(original_uuid), nullptr);
        EXPECT_EQ(dummy_scene_.getNodeByUuid(original_uuid)->uuid, original_uuid);
        EXPECT_EQ(dummy_scene_.getNodeByUuid(original_uuid)->name, "Restored");
    }

    TEST_F(SceneValidityTest, ApplyPerNodeSelectionSlicesWarnsAndFitsLengthMismatch) {
        const auto id = dummy_scene_.addSplat("Model", make_test_splat(3));
        ASSERT_NE(id, core::NULL_NODE);

        core::Scene::PerNodeSelectionSlices slices;
        slices.emplace(dummy_scene_.getNodeUuid(id),
                       make_test_selection_mask({4, 7}));

        EXPECT_NO_THROW(dummy_scene_.applyPerNodeSelectionSlices(slices));
        EXPECT_EQ(selection_mask_values(dummy_scene_),
                  (std::vector<uint8_t>{4, 7, 0}));
    }

    TEST_F(SceneValidityTest, CapturePerNodeSelectionSlicesRejectsOrderedConsolidationDisagreement) {
        const auto first_id = dummy_scene_.addSplat("First", make_test_splat(2));
        const auto second_id = dummy_scene_.addSplat("Second", make_test_splat(2));
        ASSERT_NE(first_id, core::NULL_NODE);
        ASSERT_NE(second_id, core::NULL_NODE);
        dummy_scene_.setSelectionMask(std::make_shared<core::Tensor>(
            make_test_selection_mask({1, 0, 0, 2})));

        ASSERT_EQ(dummy_scene_.consolidateNodeModels(), 2u);
        ASSERT_TRUE(dummy_scene_.isConsolidated());
        EXPECT_EQ(dummy_scene_.capturePerNodeSelectionSlices().size(), 2u);

        auto* first = dummy_scene_.getNodeById(first_id);
        auto* second = dummy_scene_.getNodeById(second_id);
        ASSERT_NE(first, nullptr);
        ASSERT_NE(second, nullptr);
        first->gaussian_count.store(3, std::memory_order_release);
        second->gaussian_count.store(1, std::memory_order_release);

        EXPECT_THROW(
            {
                const auto ignored = dummy_scene_.capturePerNodeSelectionSlices();
                (void)ignored;
            },
            core::SelectionTopologyError);

        first->gaussian_count.store(2, std::memory_order_release);
        second->gaussian_count.store(2, std::memory_order_release);
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

    TEST_F(SceneValidityTest, TransactionBatchesSetNodeTransformIntoSingleSceneChanged) {
        ASSERT_NE(dummy_scene_.addGroup("Left"), core::NULL_NODE);
        ASSERT_NE(dummy_scene_.addGroup("Right"), core::NULL_NODE);
        ASSERT_NE(dummy_scene_.addGroup("Front"), core::NULL_NODE);

        int changed = 0;
        std::uint32_t flags = 0;
        auto handler = core::events::state::SceneChanged::when(
            [&](const auto& event) {
                ++changed;
                flags = event.mutation_flags;
            });

        {
            core::Scene::Transaction txn(dummy_scene_);
            dummy_scene_.setNodeTransform(
                "Left",
                glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 0.0f, 0.0f)));
            dummy_scene_.setNodeTransform(
                "Right",
                glm::translate(glm::mat4(1.0f), glm::vec3(-1.0f, 0.0f, 0.0f)));
            dummy_scene_.setNodeTransform(
                "Front",
                glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 1.0f)));
            EXPECT_EQ(changed, 0);
        }

        EXPECT_EQ(changed, 1);
        EXPECT_EQ(
            flags,
            static_cast<std::uint32_t>(
                core::Scene::MutationType::TRANSFORM_CHANGED));

        changed = 0;
        dummy_scene_.setNodeTransform(
            "Left",
            glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f)));
        dummy_scene_.setNodeTransform(
            "Right",
            glm::translate(glm::mat4(1.0f), glm::vec3(-2.0f, 0.0f, 0.0f)));
        EXPECT_EQ(changed, 2);

        (void)handler;
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
        const auto dataset_id = dummy_scene_.addDataset("Dataset");
        const auto model_id = dummy_scene_.addSplat("Model", make_test_splat(1), dataset_id);
        ASSERT_NE(model_id, core::NULL_NODE);
        dummy_scene_.setTrainingModelNode(model_id);
        const auto cameras_group_id = dummy_scene_.addGroup("Cameras", dataset_id);
        const auto train_group_id = dummy_scene_.addCameraGroup("Training (1)", cameras_group_id, 1);
        dummy_scene_.addCamera("cam_0001.png", train_group_id, make_test_camera());

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
        // zeros_direct is still "external" (cuda.direct) — assert we left the
        // Vulkan-external interop kind, not that storage is non-external.
        EXPECT_NE(training_model->means_raw().external_storage_kind(),
                  "vulkan_external_buffer");
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

    TEST_F(SceneValidityTest, ColmapSparsePathTracksSuccessfulLoadAndClearsOnReset) {
        const ScopedTestDirectory sparse_dir("lfs_scene_manager_colmap_path");
        {
            std::ofstream cameras(sparse_dir.path() / "cameras.txt");
            cameras << "1 PINHOLE 64 64 50 50 32 32\n";
            std::ofstream images(sparse_dir.path() / "images.txt");
            images << "1 1 0 0 0 0 0 0 1 frame.png\n\n";
        }

        lfs::vis::SceneManager scene_manager;
        scene_manager.loadColmapCamerasOnly(sparse_dir.path());

        EXPECT_EQ(scene_manager.getColmapSparsePath(), sparse_dir.path());
        ASSERT_TRUE(scene_manager.clear());
        EXPECT_TRUE(scene_manager.getColmapSparsePath().empty());
    }

    TEST_F(SceneValidityTest, PPISPPathTracksSuccessfulLoadAndClearsOnReset) {
        const ScopedTestDirectory temp_dir("lfs_scene_manager_ppisp_path");
        const auto splat_path = temp_dir.path() / "appearance.ply";
        const auto splat = make_test_splat(1);
        const auto saved_splat = io::save_ply(*splat, {
                                                          .output_path = splat_path,
                                                          .binary = true,
                                                          .async = false,
                                                      });
        ASSERT_TRUE(saved_splat.has_value());

        const auto path = training::get_ppisp_companion_path(splat_path);

        training::PPISP source(1);
        source.register_frame(0, 0);
        source.finalize();
        const auto saved =
            test::write_ppisp_fixture(
                path, source);
        ASSERT_TRUE(saved.has_value()) << saved.error();

        lfs::vis::SceneManager scene_manager;
        scene_manager.loadSplatFile(splat_path);

        EXPECT_EQ(scene_manager.getPPISPPath(), path);
        ASSERT_TRUE(scene_manager.clear());
        EXPECT_TRUE(scene_manager.getPPISPPath().empty());
    }

    TEST_F(SceneValidityTest, StandaloneSplatLoadDisablesInheritedPointCloudMode) {
        const ScopedTestDirectory temp_dir("lfs_scene_manager_splat_render_mode");
        const auto splat_path = temp_dir.path() / "standalone.ply";
        const auto saved = io::save_ply(*make_test_splat(1), {
                                                                 .output_path = splat_path,
                                                                 .binary = true,
                                                                 .async = false,
                                                             });
        ASSERT_TRUE(saved.has_value());

        std::vector<core::events::ui::PointCloudModeChanged> mode_events;
        auto mode_handler = core::events::ui::PointCloudModeChanged::when(
            [&](const auto& event) { mode_events.push_back(event); });
        core::events::ui::PointCloudModeChanged{
            .enabled = true,
            .voxel_size = 0.25f}
            .emit();

        lfs::vis::SceneManager scene_manager;
        scene_manager.loadSplatFile(splat_path);

        ASSERT_EQ(mode_events.size(), 2u);
        EXPECT_TRUE(mode_events[0].enabled);
        EXPECT_FLOAT_EQ(mode_events[0].voxel_size, 0.25f);
        EXPECT_FALSE(mode_events[1].enabled);
        EXPECT_FLOAT_EQ(mode_events[1].voxel_size, 0.01f);
        const auto* node = scene_manager.getScene().getNode("standalone");
        ASSERT_NE(node, nullptr);
        EXPECT_EQ(node->type, core::NodeType::SPLAT);

        (void)mode_handler;
    }

    TEST_F(SceneValidityTest, StandaloneMeshLoadDisablesInheritedPointCloudMode) {
        const ScopedTestDirectory temp_dir("lfs_scene_manager_mesh_render_mode");
        const auto mesh_path = temp_dir.path() / "standalone-mesh.ply";
        {
            std::ofstream mesh_file(mesh_path);
            mesh_file << "ply\n"
                         "format ascii 1.0\n"
                         "element vertex 3\n"
                         "property float x\n"
                         "property float y\n"
                         "property float z\n"
                         "element face 1\n"
                         "property list uchar int vertex_indices\n"
                         "end_header\n"
                         "0 0 0\n"
                         "1 0 0\n"
                         "0 1 0\n"
                         "3 0 1 2\n";
            ASSERT_TRUE(mesh_file.good());
        }

        std::vector<core::events::ui::PointCloudModeChanged> mode_events;
        auto mode_handler = core::events::ui::PointCloudModeChanged::when(
            [&](const auto& event) { mode_events.push_back(event); });
        core::events::ui::PointCloudModeChanged{
            .enabled = true,
            .voxel_size = 0.25f}
            .emit();

        lfs::vis::SceneManager scene_manager;
        scene_manager.loadSplatFile(mesh_path);

        ASSERT_EQ(mode_events.size(), 2u);
        EXPECT_TRUE(mode_events[0].enabled);
        EXPECT_FLOAT_EQ(mode_events[0].voxel_size, 0.25f);
        EXPECT_FALSE(mode_events[1].enabled);
        EXPECT_FLOAT_EQ(mode_events[1].voxel_size, 0.01f);
        const auto* node = scene_manager.getScene().getNode("standalone-mesh");
        ASSERT_NE(node, nullptr);
        EXPECT_EQ(node->type, core::NodeType::MESH);

        (void)mode_handler;
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

    // Catches the ray-pick CPU cache accepting stale geometry when a new node reuses a mesh address.
    TEST_F(SceneValidityTest, MeshRayPickCacheRejectsReusedObjectAddressForDifferentNodeId) {
        alignas(core::MeshData) std::byte storage[sizeof(core::MeshData)];
        bool original_is_live = true;
        lfs::vis::SceneManager scene_manager;
        const auto construct_mesh = [&](const float x_offset) {
            return std::construct_at(
                reinterpret_cast<core::MeshData*>(storage),
                make_test_mesh_data(x_offset));
        };

        auto original = std::shared_ptr<core::MeshData>(
            construct_mesh(0.0f),
            [&original_is_live](core::MeshData* value) {
                if (original_is_live) {
                    std::destroy_at(value);
                }
            });
        const auto* const original_address = original.get();
        const uint32_t original_generation = original->generation();
        const core::NodeId original_id = scene_manager.getScene().addMesh("Original", original);
        ASSERT_NE(original_id, core::NULL_NODE);
        ASSERT_EQ(scene_manager.pickNodeByRay({0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, -1.0f}),
                  "Original");

        scene_manager.getScene().clear();
        std::destroy_at(original.get());
        original_is_live = false;
        auto replacement = std::shared_ptr<core::MeshData>(
            construct_mesh(10.0f),
            [](core::MeshData* value) {
                std::destroy_at(value);
            });
        ASSERT_EQ(replacement.get(), original_address);
        ASSERT_EQ(replacement->generation(), original_generation);
        const core::NodeId replacement_id =
            scene_manager.getScene().addMesh("Replacement", replacement);
        ASSERT_NE(replacement_id, core::NULL_NODE);
        ASSERT_NE(replacement_id, original_id);

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

    TEST_F(SceneValidityTest, CameraGroupTrainingCountsTrackEnabledDirectChildren) {
        lfs::vis::SceneManager sm;
        auto& scene = sm.getScene();
        const core::NodeId cameras = scene.addGroup("Cameras");
        const core::NodeId training = scene.addCameraGroup("Training", cameras, 3);
        const core::NodeId validation = scene.addCameraGroup("Validation", cameras, 1);
        const core::NodeId non_camera_child = scene.addGroup("Nested", training);
        const core::NodeId train_a = scene.addCamera("train_a.png", training, make_test_camera());
        const core::NodeId train_b = scene.addCamera("train_b.png", training, make_test_camera());
        const core::NodeId train_c = scene.addCamera("train_c.png", training, make_test_camera());
        const core::NodeId val_a = scene.addCamera("val_a.png", validation, make_test_camera());
        ASSERT_NE(non_camera_child, core::NULL_NODE);
        ASSERT_NE(train_a, core::NULL_NODE);
        ASSERT_NE(train_b, core::NULL_NODE);
        ASSERT_NE(train_c, core::NULL_NODE);
        ASSERT_NE(val_a, core::NULL_NODE);

        auto counts = scene.getCameraTrainingCounts(training);
        EXPECT_EQ(counts.enabled, 3u);
        EXPECT_EQ(counts.total, 3u);
        EXPECT_EQ(scene.getActiveCameraCount(), 4u);

        scene.setCameraTrainingEnabled(train_b, false);

        counts = scene.getCameraTrainingCounts(training);
        EXPECT_EQ(counts.enabled, 2u);
        EXPECT_EQ(counts.total, 3u);
        EXPECT_EQ(scene.getActiveCameraCount(), 3u);

        const auto validation_counts = scene.getCameraTrainingCounts(validation);
        EXPECT_EQ(validation_counts.enabled, 1u);
        EXPECT_EQ(validation_counts.total, 1u);

        const auto invalid_counts = scene.getCameraTrainingCounts(core::NULL_NODE);
        EXPECT_EQ(invalid_counts.enabled, 0u);
        EXPECT_EQ(invalid_counts.total, 0u);

        const auto non_camera_group_counts = scene.getCameraTrainingCounts(cameras);
        EXPECT_EQ(non_camera_group_counts.enabled, 0u);
        EXPECT_EQ(non_camera_group_counts.total, 0u);
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

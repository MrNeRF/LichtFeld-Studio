/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/camera.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/events.hpp"
#include "core/mesh_data.hpp"
#include "core/parameters.hpp"
#include "core/point_cloud.hpp"
#include "core/scene.hpp"
#include "core/splat_data.hpp"
#include "core/tensor.hpp"
#include "core/uuid.hpp"
#include "training/trainer.hpp"
#include "training/training_setup.hpp"
#include "visualizer/core/services.hpp"
#include "visualizer/scene/scene_manager.hpp"
#include "visualizer/training/training_manager.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <expected>
#include <filesystem>
#include <functional>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

    std::shared_ptr<lfs::core::Camera> make_test_camera() {
        return std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({3}, lfs::core::Device::CPU),
            100.0f, 100.0f, 32.0f, 32.0f,
            lfs::core::Tensor(), lfs::core::Tensor(), lfs::core::CameraModelType::PINHOLE,
            "camera.png", std::filesystem::path{}, std::filesystem::path{},
            64, 64, 0);
    }

    std::unique_ptr<lfs::core::SplatData> make_test_splat(const size_t count) {
        std::vector<float> means(count * 3, 0.0f);
        std::vector<float> rotations(count * 4, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            means[i * 3] = static_cast<float>(i);
            rotations[i * 4] = 1.0f;
        }
        return std::make_unique<lfs::core::SplatData>(
            0,
            lfs::core::Tensor::from_vector(means, {count, size_t{3}}, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({count, size_t{1}, size_t{3}}, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({count, size_t{0}, size_t{3}}, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({count, size_t{3}}, lfs::core::Device::CPU),
            lfs::core::Tensor::from_vector(rotations, {count, size_t{4}}, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros({count, size_t{1}}, lfs::core::Device::CPU),
            1.0f);
    }

    std::shared_ptr<lfs::core::PointCloud> make_test_point_cloud(const size_t count) {
        std::vector<float> means(count * 3, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            means[i * 3 + 0] = static_cast<float>(i);
            means[i * 3 + 1] = static_cast<float>(i % 3);
            means[i * 3 + 2] = static_cast<float>(i % 5);
        }
        return std::make_shared<lfs::core::PointCloud>(
            lfs::core::Tensor::from_vector(means, {count, size_t{3}}, lfs::core::Device::CPU),
            lfs::core::Tensor::full({count, size_t{3}}, 128.0f, lfs::core::Device::CPU, lfs::core::DataType::UInt8));
    }

    std::shared_ptr<lfs::core::MeshData> make_test_mesh() {
        return std::make_shared<lfs::core::MeshData>(
            lfs::core::Tensor::from_vector(
                std::vector<float>{-1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f},
                {size_t{3}, size_t{3}}, lfs::core::Device::CPU),
            lfs::core::Tensor::from_vector(
                std::vector<int32_t>{0, 1, 2},
                {size_t{1}, size_t{3}}, lfs::core::Device::CPU));
    }

    [[nodiscard]] bool populate_init_scene(lfs::core::Scene& scene) {
        const auto dataset = scene.addDataset("Dataset");
        const auto cameras = scene.addCameraGroup("Training (1)", dataset, 1);
        if (scene.addCamera("cam_0001.png", cameras, make_test_camera()) == lfs::core::NULL_NODE) {
            return false;
        }
        const auto pc = scene.addPointCloud("PointCloud", make_test_point_cloud(8), dataset);
        if (pc == lfs::core::NULL_NODE) {
            return false;
        }
        const auto cropbox = scene.addCropBox("PointCloud_cropbox", pc);
        if (cropbox == lfs::core::NULL_NODE) {
            return false;
        }
        lfs::core::CropBoxData cropbox_data;
        cropbox_data.enabled = true;
        cropbox_data.min = {-100.0f, -100.0f, -100.0f};
        cropbox_data.max = {100.0f, 100.0f, 100.0f};
        scene.setCropBoxData(cropbox, cropbox_data);
        return scene.addEllipsoid("PointCloud_ellipsoid", pc) != lfs::core::NULL_NODE &&
               scene.addMesh("Mesh", make_test_mesh(), dataset) != lfs::core::NULL_NODE;
    }

    // Every nodes_ walker used by SceneManager::buildRenderState and trainer
    // initialization. Must run on the scene-owner thread.
    void scan_scene_graph_readers(lfs::core::Scene& scene) {
        volatile std::uintptr_t sink = 0;
        const auto cameras = scene.getVisibleCameraSceneTransforms();
        sink ^= cameras.size();
        const auto visible_cameras = scene.getVisibleCameras();
        sink ^= visible_cameras.size();
        const auto meshes = scene.getVisibleMeshes();
        sink ^= meshes.size();
        const auto cropboxes = scene.getRenderableCropBoxes();
        sink ^= cropboxes.size();
        const auto ellipsoids = scene.getRenderableEllipsoids();
        sink ^= ellipsoids.size();
        const auto transforms = scene.getVisibleNodeTransforms();
        sink ^= transforms.size();
        const auto mask = scene.getNodeVisibilityMask();
        sink ^= mask.size();
        const auto nodes = scene.getNodes();
        sink ^= nodes.size();
        for (const auto* node : nodes) {
            if (node) {
                sink ^= static_cast<std::uintptr_t>(node->id);
            }
        }
        const auto active = scene.getActiveCameras();
        sink ^= active.size();
        const auto visible = scene.getVisibleNodes();
        sink ^= visible.size();
        sink ^= scene.getTotalGaussianCount();
        sink ^= scene.getAllCameras().size();
        sink ^= scene.isTrainingModelEffectivelyVisible() ? 1u : 0u;
        sink ^= scene.getTrainingModel() != nullptr ? 1u : 0u;
        static_cast<void>(sink);
    }

    struct OwnerWorkQueue {
        struct Item {
            std::function<void()> run;
            std::function<void()> cancel;
        };

        bool post(std::function<void()> run, std::function<void()> cancel) {
            {
                std::lock_guard lock(mutex);
                if (!accepting) {
                    return false;
                }
                queue.push_back(Item{std::move(run), std::move(cancel)});
            }
            cv.notify_one();
            return true;
        }

        bool pump_one() {
            Item item;
            {
                std::lock_guard lock(mutex);
                if (queue.empty()) {
                    return false;
                }
                item = std::move(queue.front());
                queue.pop_front();
            }
            if (item.run) {
                item.run();
            }
            return true;
        }

        [[nodiscard]] bool has_queued() {
            std::lock_guard lock(mutex);
            return !queue.empty();
        }

        void close() {
            std::vector<Item> leftover;
            {
                std::lock_guard lock(mutex);
                accepting = false;
                leftover.reserve(queue.size());
                while (!queue.empty()) {
                    leftover.push_back(std::move(queue.front()));
                    queue.pop_front();
                }
            }
            cv.notify_all();
            for (auto& item : leftover) {
                if (item.cancel) {
                    item.cancel();
                }
            }
        }

        std::mutex mutex;
        std::condition_variable cv;
        std::deque<Item> queue;
        bool accepting = true;
    };

    [[nodiscard]] bool wait_until(const std::function<bool()>& predicate,
                                  const std::chrono::milliseconds timeout = std::chrono::seconds(5)) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!predicate() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return predicate();
    }

    [[nodiscard]] bool scene_has_type(const lfs::core::Scene& scene, const lfs::core::NodeType type) {
        for (const auto* node : scene.getNodes()) {
            if (node && node->type == type) {
                return true;
            }
        }
        return false;
    }

} // namespace

class TrainingSceneInitConcurrencyTest : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::vis::services().clear();
    }

    void TearDown() override {
        lfs::vis::services().clear();
        lfs::event::EventBridge::instance().clear_all();
    }

    void set_scene_owner_poster(
        lfs::vis::TrainerManager& manager,
        std::function<bool(std::function<void()>, std::function<void()>)> poster) {
        manager.test_scene_owner_poster_ = std::move(poster);
    }

    void bind_queue_poster(lfs::vis::TrainerManager& manager, OwnerWorkQueue& queue) {
        set_scene_owner_poster(manager, [&queue](std::function<void()> run, std::function<void()> cancel) {
            return queue.post(std::move(run), std::move(cancel));
        });
    }

    void dispatch_on_owner(lfs::vis::TrainerManager& manager,
                           std::function<void()> run,
                           std::function<void()> cancel) {
        manager.runOnSceneOwnerThread(std::move(run), std::move(cancel));
    }
};

TEST_F(TrainingSceneInitConcurrencyTest, DelayedOwnerInstallMutatesGraphWhileReadersScan) {
    lfs::vis::SceneManager scene_manager;
    scene_manager.changeContentType(lfs::vis::SceneManager::ContentType::Dataset);
    auto& scene = scene_manager.getScene();
    ASSERT_TRUE(populate_init_scene(scene));
    ASSERT_TRUE(scene_has_type(scene, lfs::core::NodeType::POINTCLOUD));

    OwnerWorkQueue owner_queue;
    lfs::vis::TrainerManager manager;
    bind_queue_poster(manager, owner_queue);

    std::atomic<bool> allow_pump{false};
    std::atomic<bool> stop{false};
    std::atomic<int> owner_runs{0};
    std::atomic<int> scans{0};
    std::promise<std::thread::id> owner_id_promise;
    auto owner_id_future = owner_id_promise.get_future();

    std::thread owner([&] {
        owner_id_promise.set_value(std::this_thread::get_id());
        while (!stop.load(std::memory_order_acquire) || owner_queue.has_queued()) {
            if (allow_pump.load(std::memory_order_acquire)) {
                while (owner_queue.pump_one()) {
                    owner_runs.fetch_add(1, std::memory_order_relaxed);
                }
            }
            scan_scene_graph_readers(scene);
            static_cast<void>(scene_manager.buildRenderState());
            scans.fetch_add(1, std::memory_order_relaxed);
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    });
    const auto owner_thread = owner_id_future.get();

    lfs::training::TrainingModelGraphInstall install;
    install.model = make_test_splat(4);
    for (const auto* node : scene.getNodes()) {
        if (node && node->type == lfs::core::NodeType::POINTCLOUD) {
            install.point_cloud_node_id = node->id;
            install.parent_id = node->parent_id;
            install.node_transform = node->transform();
            const auto cropbox_id = scene.getCropBoxForSplat(node->id);
            if (const auto* cropbox = scene.getNodeById(cropbox_id); cropbox && cropbox->cropbox) {
                install.has_preserved_cropbox = true;
                install.preserved_cropbox_data = *cropbox->cropbox;
                install.preserved_cropbox_transform = cropbox->transform();
            }
            break;
        }
    }

    std::expected<void, std::string> result;
    std::atomic<bool> worker_finished{false};
    std::thread worker([&] {
        dispatch_on_owner(
            manager,
            [&] {
                EXPECT_EQ(std::this_thread::get_id(), owner_thread);
                result = lfs::training::installTrainingModel(scene, std::move(install));
            },
            [&] { result = std::unexpected("cancelled"); });
        worker_finished.store(true, std::memory_order_release);
    });

    ASSERT_TRUE(wait_until([&] { return owner_queue.has_queued(); }));
    EXPECT_FALSE(worker_finished.load(std::memory_order_acquire));
    EXPECT_TRUE(scene_has_type(scene, lfs::core::NodeType::POINTCLOUD));
    EXPECT_EQ(scene.getTrainingModel(), nullptr);

    allow_pump.store(true, std::memory_order_release);
    worker.join();
    stop.store(true, std::memory_order_release);
    owner.join();
    owner_queue.close();

    ASSERT_TRUE(result) << result.error();
    EXPECT_GE(owner_runs.load(), 1);
    EXPECT_GE(scans.load(), 1);
    ASSERT_NE(scene.getTrainingModel(), nullptr);
    EXPECT_EQ(scene.getTrainingModel()->size(), 4u);
    EXPECT_FALSE(scene_has_type(scene, lfs::core::NodeType::POINTCLOUD));
}

TEST_F(TrainingSceneInitConcurrencyTest, CancelledQueuedOwnerWorkDoesNotMutateGraph) {
    lfs::core::Scene scene;
    ASSERT_TRUE(populate_init_scene(scene));
    const auto node_count = scene.getNodeCount();

    OwnerWorkQueue owner_queue;
    lfs::vis::TrainerManager manager;
    bind_queue_poster(manager, owner_queue);

    lfs::training::TrainingModelGraphInstall install;
    install.model = make_test_splat(2);
    for (const auto* node : scene.getNodes()) {
        if (node && node->type == lfs::core::NodeType::POINTCLOUD) {
            install.point_cloud_node_id = node->id;
            install.parent_id = node->parent_id;
            break;
        }
    }

    std::expected<void, std::string> result;
    std::thread worker([&] {
        dispatch_on_owner(
            manager,
            [&] { result = lfs::training::installTrainingModel(scene, std::move(install)); },
            [&] { result = std::unexpected("cancelled"); });
    });

    ASSERT_TRUE(wait_until([&] { return owner_queue.has_queued(); }));
    owner_queue.close();
    worker.join();

    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), "cancelled");
    EXPECT_EQ(scene.getNodeCount(), node_count);
    EXPECT_EQ(scene.getTrainingModel(), nullptr);
    EXPECT_TRUE(scene_has_type(scene, lfs::core::NodeType::POINTCLOUD));
    EXPECT_TRUE(scene_has_type(scene, lfs::core::NodeType::CROPBOX));
}

TEST_F(TrainingSceneInitConcurrencyTest, PrepareCropboxFailureDoesNotMutateGraph) {
    lfs::core::Scene scene;
    ASSERT_TRUE(populate_init_scene(scene));
    lfs::core::NodeId cropbox_id = lfs::core::NULL_NODE;
    for (const auto* node : scene.getNodes()) {
        if (node && node->type == lfs::core::NodeType::CROPBOX) {
            cropbox_id = node->id;
            break;
        }
    }
    ASSERT_NE(cropbox_id, lfs::core::NULL_NODE);
    lfs::core::CropBoxData empty_box;
    empty_box.enabled = true;
    empty_box.min = {50.0f, 50.0f, 50.0f};
    empty_box.max = {60.0f, 60.0f, 60.0f};
    scene.setCropBoxData(cropbox_id, empty_box);

    const auto node_count = scene.getNodeCount();
    lfs::core::param::TrainingParameters params;
    params.optimization.sh_degree = 0;
    params.optimization.max_cap = 16;
    params.optimization.random = false;

    const auto prepared = lfs::training::prepareTrainingModel(params, scene);
    ASSERT_FALSE(prepared);
    EXPECT_NE(prepared.error().find("CropBox"), std::string::npos);
    EXPECT_EQ(scene.getNodeCount(), node_count);
    EXPECT_EQ(scene.getTrainingModel(), nullptr);
}

TEST_F(TrainingSceneInitConcurrencyTest, StopTrainingReturnsWhileWorkerWaitsForOwner) {
    struct EventScope {
        EventScope() { lfs::event::EventBridge::instance().clear_all(); }
        ~EventScope() { lfs::event::EventBridge::instance().clear_all(); }
    } event_scope;

    lfs::core::Scene scene;
    ASSERT_TRUE(populate_init_scene(scene));
    const auto node_count = scene.getNodeCount();

    OwnerWorkQueue owner_queue;
    lfs::vis::TrainerManager manager;
    manager.setScene(&scene);
    bind_queue_poster(manager, owner_queue);

    auto trainer = std::make_unique<lfs::training::Trainer>(scene);
    auto params = trainer->getParams();
    params.optimization.enable_eval = false;
    params.no_download = true;
    params.init_path = (std::filesystem::temp_directory_path() /
                        "lichtfeld-missing-training-init.ply")
                           .string();
    trainer->setParams(params);
    manager.setTrainer(std::move(trainer));

    ASSERT_TRUE(manager.startTraining());
    EXPECT_EQ(manager.getState(), lfs::vis::TrainingState::Starting);
    ASSERT_TRUE(wait_until([&] { return owner_queue.has_queued(); }));

    const auto started = std::chrono::steady_clock::now();
    manager.stopTraining();
    const auto elapsed = std::chrono::steady_clock::now() - started;
    EXPECT_LT(elapsed, std::chrono::milliseconds(200));
    EXPECT_NE(manager.getState(), lfs::vis::TrainingState::Starting);
    EXPECT_EQ(scene.getNodeCount(), node_count);

    owner_queue.close();
    ASSERT_TRUE(wait_until([&] {
        return manager.getState() == lfs::vis::TrainingState::Finished;
    }));
    EXPECT_TRUE(manager.getLastError().empty());
    EXPECT_EQ(scene.getNodeCount(), node_count);
    EXPECT_TRUE(scene_has_type(scene, lfs::core::NodeType::POINTCLOUD));
}

TEST_F(TrainingSceneInitConcurrencyTest, TrainerManagerFailureAndRepeatedStart) {
    struct EventScope {
        EventScope() { lfs::event::EventBridge::instance().clear_all(); }
        ~EventScope() { lfs::event::EventBridge::instance().clear_all(); }
    } event_scope;

    lfs::core::Scene scene;
    ASSERT_TRUE(populate_init_scene(scene));
    const auto node_count = scene.getNodeCount();

    OwnerWorkQueue owner_queue;
    std::atomic<bool> stop_owner{false};
    std::thread owner([&] {
        while (!stop_owner.load(std::memory_order_acquire) || owner_queue.has_queued()) {
            while (owner_queue.pump_one()) {
            }
            scan_scene_graph_readers(scene);
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });

    auto make_trainer = [&] {
        auto trainer = std::make_unique<lfs::training::Trainer>(scene);
        auto params = trainer->getParams();
        params.optimization.enable_eval = false;
        params.no_download = true;
        params.init_path = (std::filesystem::temp_directory_path() /
                            "lichtfeld-missing-training-init.ply")
                               .string();
        trainer->setParams(params);
        return trainer;
    };

    lfs::vis::TrainerManager manager;
    manager.setScene(&scene);
    bind_queue_poster(manager, owner_queue);
    manager.setTrainer(make_trainer());

    const auto wait_finished = [&] {
        return wait_until([&] {
            return manager.getState() == lfs::vis::TrainingState::Finished;
        });
    };

    ASSERT_TRUE(manager.startTraining());
    EXPECT_EQ(manager.getState(), lfs::vis::TrainingState::Starting);
    ASSERT_TRUE(wait_finished());
    EXPECT_EQ(manager.getStateMachine().getFinishReason(), lfs::vis::FinishReason::Error);
    EXPECT_EQ(scene.getNodeCount(), node_count);

    ASSERT_TRUE(manager.clearTrainer());
    manager.setTrainer(make_trainer());
    bind_queue_poster(manager, owner_queue);
    ASSERT_TRUE(manager.startTraining());
    ASSERT_TRUE(wait_finished());
    EXPECT_EQ(manager.getStateMachine().getFinishReason(), lfs::vis::FinishReason::Error);
    EXPECT_EQ(scene.getNodeCount(), node_count);
    EXPECT_TRUE(scene_has_type(scene, lfs::core::NodeType::POINTCLOUD));

    stop_owner.store(true, std::memory_order_release);
    owner.join();
    owner_queue.close();
}

TEST_F(TrainingSceneInitConcurrencyTest, StartTrainingWaitsForOwnerBeforeReplacingPointCloud) {
    lfs::core::Scene scene;
    ASSERT_TRUE(populate_init_scene(scene));
    OwnerWorkQueue queue;
    lfs::vis::TrainerManager manager;
    manager.setScene(&scene);
    bind_queue_poster(manager, queue);
    struct CancelPending {
        OwnerWorkQueue& queue;
        ~CancelPending() { queue.close(); }
    } cancel_pending{queue};

    auto trainer = std::make_unique<lfs::training::Trainer>(scene);
    auto params = trainer->getParams();
    params.optimization.enable_eval = false;
    params.optimization.iterations = 1;
    params.optimization.sh_degree = 0;
    params.optimization.max_cap = 16;
    params.optimization.random = false;
    params.no_download = true;
    params.dataset.output_path = std::filesystem::temp_directory_path() /
                                 ("lfs-owner-install-" + lfs::core::generate_uuid_v4().to_string());
    params.dataset.data_path = params.dataset.output_path;
    std::filesystem::create_directories(params.dataset.output_path);
    trainer->setParams(params);
    manager.setTrainer(std::move(trainer));

    ASSERT_TRUE(manager.startTraining());
    ASSERT_TRUE(wait_until([&] { return queue.has_queued(); }));
    ASSERT_TRUE(queue.pump_one()); // Owner captures the original graph.
    ASSERT_TRUE(wait_until([&] { return queue.has_queued(); }));
    EXPECT_TRUE(scene_has_type(scene, lfs::core::NodeType::POINTCLOUD));
    EXPECT_EQ(scene.getTrainingModel(), nullptr);
    ASSERT_TRUE(queue.pump_one()); // Actual initialization publishes its prepared model.
    ASSERT_NE(scene.getTrainingModel(), nullptr);
    EXPECT_EQ(scene.getTrainingModel()->size(), 8u);
    EXPECT_FALSE(scene_has_type(scene, lfs::core::NodeType::POINTCLOUD));
    manager.stopTraining();
    ASSERT_TRUE(wait_until([&] {
        while (queue.pump_one()) {}
        return !manager.isCompletionPending();
    }));
    std::filesystem::remove_all(params.dataset.output_path);
}

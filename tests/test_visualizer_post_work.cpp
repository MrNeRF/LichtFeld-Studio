/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include <SDL3/SDL.h>

#include "core/checkpoint_format.hpp"
#include "core/event_bridge/event_bridge.hpp"
#include "core/event_bus.hpp"
#include "core/events.hpp"
#include "core/guarded_task.hpp"
#include "core/main_loop.hpp"
#include "core/scene.hpp"
#include "core/services.hpp"
#include "input/input_controller.hpp"
#include "io/project_chapters.hpp"
#include "io/project_container.hpp"
#include "io/project_document.hpp"
#include "io/project_path.hpp"
#include "io/project_recovery.hpp"
#include "licht_test_support.hpp"
#include "operation/undo_history.hpp"
#include "python/python_runtime.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "tools/unified_tool_registry.hpp"
#include "training/checkpoint.hpp"
#include "training/components/ppisp_file.hpp"
#include "training/strategies/mcmc.hpp"
#include "training/trainer.hpp"
#include "training/training_state.hpp"
#include "visualizer/core/data_loading_service.hpp"
#include "visualizer/include/visualizer/visualizer.hpp"
#include "visualizer/post_work_utils.hpp"
#include "visualizer/visualizer_impl.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <mutex>
#include <optional>
#include <ranges>
#include <sstream>
#include <utility>
#include <vector>

namespace {

    class NoopUndoEntry final
        : public lfs::vis::op::UndoEntry {
    public:
        void undo() override {}
        void redo() override {}
        [[nodiscard]] std::string name()
            const override {
            return "test.noop";
        }
    };

    lfs::Error posted_work_cancelled_error() {
        return lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::Cancelled,
            .domain = lfs::ErrorDomain::Core,
            .operation_id = lfs::OperationId::generate(),
            .detail = "Viewer is shutting down",
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
    }

    lfs::core::TaskContext posted_work_context() {
        return {
            .name = "test.posted-work",
            .domain = lfs::ErrorDomain::Core,
            .operation_id = lfs::OperationId::generate(),
            .site = LFS_SOURCE_SITE_CURRENT(),
        };
    }

    class PostedWorkTestVisualizer final : public lfs::vis::Visualizer {
    public:
        void run() override {}
        void setParameters(const lfs::core::param::TrainingParameters&) override {}
        std::expected<void, std::string> loadPLY(const std::filesystem::path&) override { return {}; }
        std::expected<void, std::string> addSplatFile(const std::filesystem::path&) override { return {}; }
        std::expected<void, std::string> loadDataset(const std::filesystem::path&) override { return {}; }
        std::expected<void, std::string> loadCheckpointForTraining(const std::filesystem::path&) override { return {}; }
        void consolidateModels() override {}
        std::expected<void, std::string> clearScene() override { return {}; }
        lfs::core::Scene& getScene() override { return scene_; }
        lfs::vis::SceneManager* getSceneManager() override { return nullptr; }
        lfs::vis::RenderingManager* getRenderingManager() override { return nullptr; }

        bool postWork(WorkItem work) override {
            if (!accepts_work_) {
                return false;
            }
            {
                std::lock_guard lock(mutex_);
                work_.push_back(std::move(work));
            }
            cv_.notify_one();
            return true;
        }

        [[nodiscard]] bool acceptsPostedWork() const override { return accepts_work_; }
        void setShutdownRequestedCallback(std::function<void()>) override {}
        std::expected<void, std::string> startTraining() override { return {}; }
        lfs::Result<void> projectSave(bool) override { return {}; }
        lfs::Result<void> projectSaveAs(
            const std::filesystem::path&, bool) override {
            return {};
        }
        lfs::Result<lfs::vis::ProjectOpenOutcome> projectOpen(
            const std::filesystem::path&,
            lfs::vis::ProjectSwitchDisposition) override {
            return lfs::vis::ProjectOpenOutcome::Opened;
        }
        lfs::Result<void> projectCompact() override {
            return {};
        }
        lfs::Result<bool> projectIsDirty() override {
            return false;
        }
        lfs::Result<bool> projectHasPath() override {
            return false;
        }
        lfs::Result<lfs::vis::ProjectInfo>
        projectGetInfo() override {
            return lfs::vis::ProjectInfo{};
        }

        void rejectPostedWork() { accepts_work_ = false; }

        [[nodiscard]] bool waitForWork(const std::chrono::milliseconds timeout) {
            std::unique_lock lock(mutex_);
            return cv_.wait_for(lock, timeout, [this] { return !work_.empty(); });
        }

        void cancelNext() {
            WorkItem work;
            {
                std::lock_guard lock(mutex_);
                work = std::move(work_.front());
                work_.pop_front();
            }
            work.cancel();
        }

    private:
        lfs::core::Scene scene_;
        bool accepts_work_ = true;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::deque<WorkItem> work_;
    };

} // namespace

TEST(VisualizerPostWorkTest, QueuedWorkWakesEventLoop) {
    ASSERT_TRUE(SDL_Init(SDL_INIT_EVENTS));
    SDL_FlushEvents(SDL_EVENT_USER, SDL_EVENT_USER);

    lfs::vis::ViewerOptions options;
    options.show_startup_overlay = false;

    bool ran = false;
    {
        auto viewer = lfs::vis::Visualizer::create(options);
        SDL_FlushEvents(SDL_EVENT_USER, SDL_EVENT_USER);

        EXPECT_FALSE(SDL_HasEvents(SDL_EVENT_USER, SDL_EVENT_USER));
        EXPECT_TRUE(viewer->postWork({
            .run = [&ran]() { ran = true; },
            .cancel = nullptr,
        }));

        EXPECT_FALSE(ran);
        EXPECT_TRUE(SDL_HasEvents(SDL_EVENT_USER, SDL_EVENT_USER));
    }
}

TEST(VisualizerPostedWorkTest, GuardedFastPathSettlesAgainstRealViewer) {
    lfs::vis::ViewerOptions options;
    options.show_startup_overlay = false;
    auto viewer = lfs::vis::Visualizer::create(options);

    auto result = lfs::vis::post_guarded_and_wait<int>(
        *viewer, posted_work_context(), [] { return lfs::Result<int>(17); },
        posted_work_cancelled_error());

    ASSERT_TRUE(result);
    EXPECT_EQ(*result, 17);
}

TEST(VisualizerPostedWorkTest, GuardedQueueRejectionReturnsCancellationWithoutWaiting) {
    PostedWorkTestVisualizer viewer;
    viewer.rejectPostedWork();

    auto future = std::async(std::launch::async, [&viewer] {
        return lfs::vis::post_guarded_and_wait<void>(
            viewer, posted_work_context(), [] { return lfs::Result<void>{}; },
            posted_work_cancelled_error());
    });

    ASSERT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = future.get();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code(), lfs::ErrorCode::Cancelled);
}

TEST(VisualizerPostedWorkTest, GuardedShutdownCancellationMakesWaitingFutureReady) {
    PostedWorkTestVisualizer viewer;
    auto future = std::async(std::launch::async, [&viewer] {
        return lfs::vis::post_guarded_and_wait<void>(
            viewer, posted_work_context(), [] { return lfs::Result<void>{}; },
            posted_work_cancelled_error());
    });

    ASSERT_TRUE(viewer.waitForWork(std::chrono::seconds(1)));
    viewer.cancelNext();
    ASSERT_EQ(future.wait_for(std::chrono::seconds(1)), std::future_status::ready);
    auto result = future.get();
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().code(), lfs::ErrorCode::Cancelled);
}

class VisualizerImplResetTest : public ::testing::Test {
protected:
    void SetUp() override {
        lfs::event::EventBridge::instance().clear_all();
        lfs::core::event::bus().clear_all();
        lfs::vis::services().clear();
        lfs::vis::op::undoHistory().clear();
    }

    void TearDown() override {
        lfs::vis::op::undoHistory().clear();
        lfs::vis::services().clear();
        lfs::core::event::bus().clear_all();
        lfs::event::EventBridge::instance().clear_all();
    }

    [[nodiscard]] lfs::vis::ViewerOptions projectOptions() const {
        lfs::vis::ViewerOptions options;
        options.show_startup_overlay = false;
        options.project_lifecycle_settings_path =
            temporary_.path / "lifecycle.json";
        return options;
    }

    template <typename Predicate>
    [[nodiscard]] bool pumpUntil(
        std::mutex& queue_mutex, std::vector<lfs::vis::Visualizer::WorkItem>& queue,
        Predicate&& condition,
        const std::chrono::milliseconds timeout = std::chrono::seconds(10)) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!condition() && std::chrono::steady_clock::now() < deadline) {
            lfs::test::licht::drain_work_queue(queue_mutex, queue);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return condition();
    }

    template <typename Predicate>
    [[nodiscard]] bool waitUntil(
        Predicate&& condition,
        const std::chrono::milliseconds timeout = std::chrono::seconds(10)) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        while (!condition() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return condition();
    }

    void installModalOverlay(
        std::unique_ptr<lfs::vis::gui::RmlModalOverlay>& overlay,
        lfs::vis::gui::RmlUIManager& manager) {
        overlay =
            std::make_unique<lfs::vis::gui::RmlModalOverlay>(
                &manager);
    }

    [[nodiscard]] lfs::core::ModalRequest
    takeModalRequest(std::mutex& queue_mutex,
                     std::deque<lfs::core::ModalRequest>& queue) {
        std::lock_guard lock(queue_mutex);
        if (queue.size() != 1) {
            ADD_FAILURE() << "expected exactly one modal request, got "
                          << queue.size();
            return {};
        }
        auto request = std::move(queue.front());
        queue.pop_front();
        return request;
    }

    lfs::test::licht::TemporaryDirectory temporary_{"lfs-visualizer-project"};
};

namespace {

    bool cuda_device_available() {
        int count = 0;
        return cudaGetDeviceCount(&count) ==
                   cudaSuccess &&
               count > 0;
    }

    std::shared_ptr<lfs::core::Camera>
    make_project_request_test_camera() {
        const auto empty_distortion =
            lfs::core::Tensor::zeros(
                {0}, lfs::core::Device::CPU,
                lfs::core::DataType::Float32);
        return std::make_shared<lfs::core::Camera>(
            lfs::core::Tensor::eye(
                3, lfs::core::Device::CPU),
            lfs::core::Tensor::zeros(
                {3}, lfs::core::Device::CPU),
            100.0F, 100.0F, 32.0F, 32.0F,
            empty_distortion, empty_distortion,
            lfs::core::CameraModelType::PINHOLE,
            "camera.png", std::filesystem::path{},
            std::filesystem::path{}, 64, 64, 0);
    }

    void write_minimal_transforms_dataset(const std::filesystem::path& dataset_path) {
        std::filesystem::create_directories(dataset_path);
        const auto png = lfs::test::licht::one_pixel_png();
        lfs::test::licht::write_file_bytes(dataset_path / "frame_0001.png", png);
        const auto transforms = lfs::test::licht::byte_vector(R"({
  "fl_x": 1.0,
  "fl_y": 1.0,
  "cx": 0.5,
  "cy": 0.5,
  "w": 1,
  "h": 1,
  "frames": [
    {
      "file_path": "frame_0001.png",
      "transform_matrix": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
}
)");
        lfs::test::licht::write_file_bytes(
            dataset_path / "transforms.json", transforms);
    }

    void write_dataset_project_without_checkpoint(
        const std::filesystem::path& project_path,
        const std::filesystem::path& dataset_path,
        const bool bind_dataset_reference = true) {
        auto document =
            lfs::test::licht::make_empty_document(
                lfs::core::generate_uuid_v4(), 1);
        lfs::core::Scene source;
        const auto dataset =
            source.addDataset("Dataset");
        const auto cameras = source.addCameraGroup(
            "Training", dataset, 1);
        source.addCamera(
            "frame_0001.png", cameras,
            make_project_request_test_camera());
        document->edit_scene_graph() =
            lfs::test::licht::require_result(
                lfs::io::project::capture_scene_graph(
                    source, {}));

        auto parameters =
            lfs::test::licht::require_result(
                document->parameters().snapshot());
        parameters.dataset.data_path = dataset_path;
        parameters.mrnf_session.iterations = 1234;
        parameters.mrnf_current.iterations = 1234;
        lfs::test::licht::require_status(
            document->edit_parameters().set_snapshot(
                parameters));

        if (bind_dataset_reference) {
            const auto reference =
                lfs::test::licht::require_result(
                    lfs::io::project::upsert_path_reference(
                        document->edit_references(),
                        project_path.parent_path(),
                        dataset_path, "dataset", "dataset"));
            lfs::test::licht::require_status(
                document->edit_project()
                    .set_dataset_reference(reference));
        }

        auto options =
            lfs::test::licht::
                deterministic_document_save_options(
                    0x76000008, 1, 2);
        options.commit.snapshot_uuid = {};
        (void)lfs::test::licht::require_result(
            document->save(project_path, options));
    }

    void write_non_dataset_project_with_stale_dataset_params(
        const std::filesystem::path& project_path,
        const std::filesystem::path& stale_dataset_path) {
        auto document =
            lfs::test::licht::make_empty_document(
                lfs::core::generate_uuid_v4(), 1);
        lfs::core::Scene source;
        source.addGroup("PLY-only marker");
        document->edit_scene_graph() =
            lfs::test::licht::require_result(
                lfs::io::project::capture_scene_graph(
                    source, {}));

        auto parameters =
            lfs::test::licht::require_result(
                document->parameters().snapshot());
        parameters.dataset.data_path =
            stale_dataset_path;
        lfs::test::licht::require_status(
            document->edit_parameters().set_snapshot(
                parameters));

        auto options =
            lfs::test::licht::
                deterministic_document_save_options(
                    0x76000009, 1, 2);
        options.commit.snapshot_uuid = {};
        (void)lfs::test::licht::require_result(
            document->save(project_path, options));
    }

    lfs::io::project::LazyChunkValue
    make_training_autosave_checkpoint_payload(
        const lfs::core::Uuid& checkpoint_uuid,
        const std::filesystem::path& dataset_path = {}) {
        auto model = lfs::test::licht::make_splat(2);
        lfs::training::MCMC strategy(*model);
        lfs::core::param::TrainingParameters parameters;
        parameters.optimization =
            lfs::core::param::OptimizationParameters::
                mcmc_defaults();
        parameters.optimization.sh_degree = 0;
        parameters.optimization.max_cap = 2;
        parameters.dataset.data_path = dataset_path;
        strategy.initialize(parameters.optimization);
        std::ostringstream stream(
            std::ios::binary | std::ios::out);
        (void)lfs::test::licht::require_result(
            lfs::training::serialize_checkpoint(
                stream, 11, strategy, parameters,
                nullptr, nullptr, nullptr, nullptr));
        const auto encoded = stream.str();
        std::vector<std::byte> bytes(encoded.size());
        std::memcpy(
            bytes.data(), encoded.data(), encoded.size());
        return lfs::test::licht::require_result(
            lfs::io::project::LazyChunkValue::from_owned(
                std::move(bytes), checkpoint_uuid));
    }

    void write_project_with_specified_checkpoint(
        const std::filesystem::path& path,
        const lfs::core::Uuid& training_uuid,
        const lfs::core::Uuid& checkpoint_uuid) {
        auto document = lfs::test::licht::make_empty_document(
            lfs::core::generate_uuid_v4(), 1);
        const auto root_uuid = lfs::core::generate_uuid_v4();
        lfs::test::licht::require_status(
            document->edit_scene_graph().upsert_node(
                lfs::io::project::SceneNodeRecord{
                    .uuid = root_uuid,
                    .type = "group",
                    .name = "Root",
                    .child_order = 0,
                }));
        lfs::test::licht::require_status(
            document->edit_scene_graph().upsert_node(
                lfs::io::project::SceneNodeRecord{
                    .uuid = training_uuid,
                    .type = "splat",
                    .name = "Training",
                    .parent_uuid = root_uuid,
                    .child_order = 0,
                    .payload =
                        lfs::io::project::PayloadBinding{
                            .fourcc = "CKPT",
                            .instance_uuid =
                                checkpoint_uuid,
                            .source_kind = "training",
                        },
                }));
        lfs::test::licht::require_status(
            document->edit_scene_graph()
                .set_training_model_uuid(training_uuid));
        lfs::test::licht::require_status(
            document->set_checkpoint(
                checkpoint_uuid,
                make_training_autosave_checkpoint_payload(
                    checkpoint_uuid)));
        auto options =
            lfs::test::licht::
                deterministic_document_save_options(
                    0x76000010, 1, 2);
        options.commit.snapshot_uuid = checkpoint_uuid;
        (void)lfs::test::licht::require_result(
            document->save(path, options));
    }

    void write_resumable_project_with_checkpoint(
        const std::filesystem::path& path,
        const lfs::core::Uuid& training_uuid,
        const lfs::core::Uuid& checkpoint_uuid,
        const std::filesystem::path& dataset_path) {
        auto document = lfs::test::licht::make_empty_document(
            lfs::core::generate_uuid_v4(), 1);
        lfs::core::Scene source;
        const auto root = source.addGroup("Root");
        const auto cameras = source.addCameraGroup(
            "Training cameras", root, 1);
        source.addCamera(
            "frame_0001.png", cameras,
            make_project_request_test_camera());
        const auto training = source.restoreNodeWithUuid(
            lfs::core::Scene::RestoreNodeDesc{
                .uuid = training_uuid,
                .type = lfs::core::NodeType::SPLAT,
                .name = "Training",
                .parent = root,
                .gaussian_count = 2,
                .model = lfs::test::licht::make_splat(2),
            });
        if (training == lfs::core::NULL_NODE) {
            throw std::runtime_error(
                "failed to create training fixture node");
        }
        source.setTrainingModelNode(training);
        auto scene_chapter = lfs::test::licht::require_result(
            lfs::io::project::capture_scene_graph(
                source,
                lfs::io::project::ScenePayloadBindings{
                    {training_uuid,
                     lfs::io::project::PayloadBinding{
                         .fourcc = "CKPT",
                         .instance_uuid = checkpoint_uuid,
                         .source_kind = "training",
                     }},
                }));
        document->edit_scene_graph() =
            std::move(scene_chapter);
        lfs::test::licht::require_status(
            document->set_checkpoint(
                checkpoint_uuid,
                make_training_autosave_checkpoint_payload(
                    checkpoint_uuid, dataset_path)));
        auto options =
            lfs::test::licht::
                deterministic_document_save_options(
                    0x76000010, 1, 2);
        options.commit.snapshot_uuid = checkpoint_uuid;
        (void)lfs::test::licht::require_result(
            document->save(path, options));
    }

    void write_empty_project(
        const std::filesystem::path& path,
        const std::optional<float>
            focal_length_mm = std::nullopt) {
        auto document = lfs::test::licht::make_empty_document(
            lfs::core::generate_uuid_v4(), 1);
        if (focal_length_mm) {
            lfs::test::licht::require_status(document->edit_view().dom().set_json(
                "render_settings.focal_length_mm", *focal_length_mm));
        }
        auto options = lfs::test::licht::deterministic_document_save_options(
            0x76000000, 1, 2);
        options.commit.snapshot_uuid = {};
        (void)lfs::test::licht::require_result(document->save(path, options));
    }

    void write_recoverable_project(
        const std::filesystem::path& path,
        const std::string& marker =
            "autosaved",
        lfs::core::Uuid* checkpoint_uuid_out =
            nullptr) {
        write_empty_project(path);
        auto document = lfs::test::licht::require_result_ptr(
            lfs::io::project::ProjectDocument::open(path));
        const auto payload_uuid =
            lfs::core::generate_uuid_v4();
        const lfs::training::PPISPFileHeader header{
            .num_cameras = 1,
            .num_frames = 1,
        };
        std::vector<std::byte> payload_bytes(
            sizeof(header));
        std::memcpy(payload_bytes.data(), &header,
                    sizeof(header));
        auto payload =
            lfs::io::project::LazyChunkValue::
                from_owned(
                    std::move(payload_bytes),
                    payload_uuid);
        lfs::test::licht::require_status(document->set_ppisp(
            payload_uuid, lfs::test::licht::require_result(std::move(payload))));
        auto options = lfs::test::licht::deterministic_document_save_options(
            0x76000000, 10, 3);
        options.commit.snapshot_uuid = {};
        (void)lfs::test::licht::require_result(document->save(path, options));
        const auto base = lfs::test::licht::require_result(
            lfs::io::project::ProjectReader::open(path));
        lfs::test::licht::require_status(
            document->edit_view().dom().set("recovery_marker", marker));
        (void)lfs::test::licht::require_result(document->save_autosave(
            lfs::io::project::
                autosave_sidecar_path(path),
            {
                .file_uuid =
                    lfs::core::generate_uuid_v4(),
                .base_explicit_commit_uuid =
                    base.commit().commit_uuid,
                .autosave_sequence = 1,
                .snapshot_uuid =
                    lfs::core::generate_uuid_v4(),
                .index_compression =
                    lfs::io::project::
                        IndexCompression::
                            StoredForDeterministicTests,
                .disk_reserve_bytes = 0,
            }));
        if (checkpoint_uuid_out) {
            *checkpoint_uuid_out =
                payload_uuid;
        }
    }

    void write_invalid_phase_a_project(
        const std::filesystem::path& path,
        const bool corrupt_session) {
        write_empty_project(path);
        auto reader =
            lfs::io::project::ProjectReader::open(
                path);
        ASSERT_TRUE(reader)
            << lfs::format_for_developer(
                   reader.error());
        auto writer =
            lfs::io::project::ProjectWriter::append(
                path,
                {
                    .compatibility = {},
                    .index_compression =
                        lfs::io::project::
                            IndexCompression::
                                StoredForDeterministicTests,
                    .disk_reserve_bytes = 0,
                    .boundary_observer = {},
                });
        ASSERT_TRUE(writer)
            << lfs::format_for_developer(
                   writer.error());
        ASSERT_TRUE(writer->plan_commit(
            {
                .kind =
                    lfs::io::project::
                        CommitKind::Explicit,
                .commit_uuid =
                    lfs::core::
                        generate_uuid_v4(),
                .snapshot_uuid = {},
                .wallclock_unix_ns = 3,
            }));

        const std::string invalid_json =
            corrupt_session
                ? R"({"version":1,"render_settings":{"focal_length_mm":"invalid"}})"
                : R"({"version":1,"active_strategy":"not-a-strategy","presets":{},"dataset":{}})";
        const auto invalid_bytes =
            std::as_bytes(
                std::span(invalid_json));
        ASSERT_TRUE(writer->preflight(
            invalid_bytes.size()));
        const std::string target =
            corrupt_session ? "VIEW" : "PRMS";
        bool replaced = false;
        for (const auto& chunk :
             reader->chunks()) {
            if (!chunk.is_live()) {
                continue;
            }
            if (chunk.key.fourcc.to_string() ==
                target) {
                replaced = true;
                const auto written =
                    writer->write_chunk(
                        chunk.key,
                        invalid_bytes,
                        {
                            .chunk_version =
                                chunk.chunk_version,
                            .compression =
                                lfs::io::project::
                                    Compression::
                                        Stored,
                        });
                ASSERT_TRUE(written)
                    << lfs::format_for_developer(
                           written.error());
                continue;
            }
            auto proof =
                reader->make_clean_proof(
                    chunk, 1);
            ASSERT_TRUE(proof)
                << lfs::format_for_developer(
                       proof.error());
            const auto reused =
                writer->reuse_if_clean(
                    *proof, 1);
            ASSERT_TRUE(reused)
                << lfs::format_for_developer(
                       reused.error());
        }
        ASSERT_TRUE(replaced);
        const auto committed =
            writer->commit();
        ASSERT_TRUE(committed)
            << lfs::format_for_developer(
                   committed.error());
    }

} // namespace

namespace lfs::vis {

    TEST_F(VisualizerImplResetTest, DestructorClearsSharedEventBridgeHandlers) {
        ViewerOptions options;
        options.show_startup_overlay = false;

        {
            VisualizerImpl viewer(options);
            EXPECT_GT(lfs::event::EventBridge::instance().handler_count(
                          typeid(lfs::core::events::cmd::ResetTraining)),
                      0u);
        }

        EXPECT_EQ(lfs::event::EventBridge::instance().handler_count(
                      typeid(lfs::core::events::cmd::ResetTraining)),
                  0u);
    }

    TEST_F(VisualizerImplResetTest,
           SuccessfulProjectOpenClearsUndoHistory) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "empty.licht";
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_NE(
                viewer.getScene().addGroup("Current"),
                lfs::core::NULL_NODE);
            op::undoHistory().push(
                std::make_unique<NoopUndoEntry>());
            ASSERT_EQ(
                op::undoHistory().undoCount(), 1u);

            const auto opened =
                viewer.projectOpen(
                    project_path,
                    ProjectSwitchDisposition::
                        DiscardChanges);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(
                       opened.error());
            EXPECT_EQ(
                op::undoHistory().undoCount(), 0u);
            EXPECT_EQ(
                viewer.getScene().getNodeCount(), 0u);
        }
    }

    TEST_F(VisualizerImplResetTest,
           RecoveryDeclineKeepsSidecarSuppressesRepeatAndExplicitSaveDeletesIt) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "decline.licht";
        const auto other_path =
            temporary / "other.licht";
        const auto sidecar =
            lfs::io::project::
                autosave_sidecar_path(project_path);
        write_recoverable_project(project_path);
        write_empty_project(other_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            auto* const gui = viewer.getGuiManager();
            ASSERT_NE(gui, nullptr);
            installModalOverlay(gui->rml_modal_overlay_, gui->rmlui_manager_);

            auto pending = viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges);
            ASSERT_TRUE(pending);
            EXPECT_EQ(*pending,
                      ProjectOpenOutcome::
                          RecoveryPromptPending);
            auto still_pending = viewer.projectOpen(
                other_path,
                ProjectSwitchDisposition::DiscardChanges);
            ASSERT_TRUE(still_pending);
            EXPECT_EQ(*still_pending,
                      ProjectOpenOutcome::Opened);
            auto request = takeModalRequest(
                gui->rml_modal_overlay_->queue_mutex_, gui->rml_modal_overlay_->queue_);
            ASSERT_EQ(
                request.title,
                "Recover Autosaved Project?");
            ASSERT_TRUE(request.on_cancel);
            request.on_cancel();
            EXPECT_TRUE(
                std::filesystem::is_regular_file(
                    sidecar));
            auto after_stale_cancel =
                viewer.projectGetInfo();
            ASSERT_TRUE(after_stale_cancel);
            EXPECT_EQ(after_stale_cancel->path,
                      other_path);
            EXPECT_FALSE(viewer.project_lifecycle_
                             ->recovery_prompt_pending_);

            pending = viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges);
            ASSERT_TRUE(pending);
            EXPECT_EQ(*pending,
                      ProjectOpenOutcome::
                          RecoveryPromptPending);
            request = takeModalRequest(
                gui->rml_modal_overlay_->queue_mutex_, gui->rml_modal_overlay_->queue_);
            ASSERT_TRUE(request.on_cancel);
            request.on_cancel();
            EXPECT_FALSE(viewer.project_lifecycle_
                             ->recovery_prompt_pending_);

            pending = viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges);
            ASSERT_TRUE(pending);
            EXPECT_EQ(*pending,
                      ProjectOpenOutcome::
                          RecoveryPromptPending);
            request = takeModalRequest(
                gui->rml_modal_overlay_->queue_mutex_, gui->rml_modal_overlay_->queue_);
            ASSERT_TRUE(request.on_result);
            request.on_result(
                lfs::core::ModalResult{
                    .button_label =
                        "Open Saved"});
            EXPECT_TRUE(
                std::filesystem::is_regular_file(
                    sidecar));

            auto reopened = viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges);
            ASSERT_TRUE(reopened);
            EXPECT_EQ(*reopened,
                      ProjectOpenOutcome::Opened);
            {
                auto& overlay =
                    *viewer.getGuiManager()->rml_modal_overlay_;
                std::lock_guard lock(
                    overlay.queue_mutex_);
                EXPECT_TRUE(
                    overlay.queue_.empty());
            }
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Explicit after decline"),
                lfs::core::NULL_NODE);
            ASSERT_TRUE(
                viewer.projectSave(false));
            EXPECT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_,
                [&] { return !viewer.jobs().anyRunning(JobType::ProjectWrite); },
                std::chrono::seconds(10)));
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_FALSE(
                std::filesystem::exists(sidecar));
            auto master =
                lfs::io::project::ProjectReader::open(
                    project_path);
            ASSERT_TRUE(master);
            EXPECT_EQ(master->commit().kind,
                      lfs::io::project::
                          CommitKind::Explicit);
        }
    }

    TEST_F(VisualizerImplResetTest,
           NewProjectClearsRecoveryPromptPendingSoNextOpenProceeds) {
        // Latch left true after newProject made every later open()
        // return RecoveryPromptPending with no modal.
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "latched.licht";
        const auto empty_path =
            temporary / "empty.licht";
        write_recoverable_project(project_path);
        write_empty_project(empty_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            auto* const gui = viewer.getGuiManager();
            ASSERT_NE(gui, nullptr);
            installModalOverlay(gui->rml_modal_overlay_, gui->rmlui_manager_);

            auto pending = viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges);
            ASSERT_TRUE(pending);
            EXPECT_EQ(*pending,
                      ProjectOpenOutcome::
                          RecoveryPromptPending);
            auto created =
                viewer.project_lifecycle_->newProject(
                    ProjectSwitchDisposition::
                        DiscardChanges);
            ASSERT_TRUE(created)
                << lfs::format_for_developer(
                       created.error());
            EXPECT_FALSE(viewer.project_lifecycle_
                             ->recovery_prompt_pending_);
            auto opened = viewer.projectOpen(
                empty_path,
                ProjectSwitchDisposition::
                    DiscardChanges);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(
                       opened.error());
            EXPECT_EQ(*opened, ProjectOpenOutcome::Opened);
        }
    }

    TEST(MainLoopSignalTest, InstallInterruptHandlersRestoresStolenTermAndInt) {
        // Would fail if pycolmap/glog could steal SIGTERM/SIGINT and
        // installInterruptHandlers did not put MainLoop's handler back.
        auto stolen = +[](int) {};
        auto previous_term = std::signal(SIGTERM, stolen);
        MainLoop::installInterruptHandlers();
        auto installed_term = std::signal(SIGTERM, stolen);
        EXPECT_EQ(installed_term, MainLoop::interruptHandlerForTest());
        std::signal(SIGTERM, previous_term);

        auto previous_int = std::signal(SIGINT, stolen);
        MainLoop::installInterruptHandlers();
        auto installed_int = std::signal(SIGINT, stolen);
        EXPECT_EQ(installed_int, MainLoop::interruptHandlerForTest());
        std::signal(SIGINT, previous_int);
    }

    TEST_F(VisualizerImplResetTest,
           RecoveredPublishUsesRecoveredCommitKind) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "recover.licht";
        const auto sidecar =
            lfs::io::project::
                autosave_sidecar_path(project_path);
        write_recoverable_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            auto* const gui = viewer.getGuiManager();
            ASSERT_NE(gui, nullptr);
            installModalOverlay(gui->rml_modal_overlay_, gui->rmlui_manager_);

            auto pending = viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges);
            ASSERT_TRUE(pending);
            EXPECT_EQ(*pending,
                      ProjectOpenOutcome::
                          RecoveryPromptPending);
            auto request = takeModalRequest(
                gui->rml_modal_overlay_->queue_mutex_, gui->rml_modal_overlay_->queue_);
            ASSERT_TRUE(request.on_result);
            request.on_result(
                lfs::core::ModalResult{
                    .button_label = "Recover"});
            auto info = viewer.projectGetInfo();
            ASSERT_TRUE(info);
            EXPECT_TRUE(info->recovery_session);
            ASSERT_TRUE(
                std::filesystem::remove(sidecar));
            ASSERT_TRUE(
                std::filesystem::create_directory(
                    sidecar));
            {
                std::ofstream blocker(
                    sidecar / "cleanup-blocker");
                ASSERT_TRUE(blocker);
                blocker << "force post-publish cleanup warning";
            }
            ASSERT_TRUE(
                viewer.projectSave(false));

            EXPECT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_,
                [&] { return !viewer.jobs().anyRunning(JobType::ProjectWrite); },
                std::chrono::seconds(10)));
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            auto master =
                lfs::io::project::ProjectReader::open(
                    project_path);
            ASSERT_TRUE(master)
                << lfs::format_for_developer(
                       master.error());
            EXPECT_EQ(master->commit().kind,
                      lfs::io::project::
                          CommitKind::Recovered);
            auto saved_info = viewer.projectGetInfo();
            ASSERT_TRUE(saved_info);
            EXPECT_TRUE(
                saved_info->project_write_error.empty());
            EXPECT_TRUE(
                std::filesystem::is_directory(
                    sidecar));
        }
    }

    TEST_F(VisualizerImplResetTest,
           RecoveredProjectSwitchDeletesTempOnlyAfterReplacement) {
        const auto& temporary = temporary_.path;
        const auto recovered_path =
            temporary / "recovered.licht";
        const auto replacement_path =
            temporary / "replacement.licht";
        lfs::core::Uuid checkpoint_uuid;
        write_recoverable_project(
            recovered_path, "switch",
            &checkpoint_uuid);
        write_empty_project(replacement_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            auto* const gui = viewer.getGuiManager();
            ASSERT_NE(gui, nullptr);
            installModalOverlay(gui->rml_modal_overlay_, gui->rmlui_manager_);
            auto offered = viewer.projectOpen(
                recovered_path,
                ProjectSwitchDisposition::DiscardChanges);
            ASSERT_TRUE(offered);
            ASSERT_EQ(*offered,
                      ProjectOpenOutcome::
                          RecoveryPromptPending);
            auto request = takeModalRequest(
                gui->rml_modal_overlay_->queue_mutex_, gui->rml_modal_overlay_->queue_);
            request.on_result({.button_label = "Recover"});
            ASSERT_TRUE(viewer.project_lifecycle_
                            ->recovery_session_path_);
            const auto recovery_temp =
                *viewer.project_lifecycle_
                     ->recovery_session_path_;
            ASSERT_TRUE(std::filesystem::is_regular_file(
                recovery_temp));
            ASSERT_TRUE(viewer.project_lifecycle_
                            ->recovery_session_);
            EXPECT_TRUE(viewer.project_lifecycle_
                            ->recovery_session_
                            ->document_attached());

            const auto* checkpoint =
                viewer.project_lifecycle_
                    ->document_->find_ppisp(
                        checkpoint_uuid);
            ASSERT_NE(checkpoint, nullptr);
            std::array<std::byte, 4> marker{};
            auto read = checkpoint->read_at(0, marker);
            ASSERT_TRUE(read)
                << lfs::format_for_developer(
                       read.error());
            EXPECT_EQ(marker[0], std::byte{0x49});

            auto switched = viewer.projectOpen(
                replacement_path,
                ProjectSwitchDisposition::DiscardChanges);
            ASSERT_TRUE(switched)
                << lfs::format_for_developer(
                       switched.error());
            EXPECT_EQ(*switched,
                      ProjectOpenOutcome::Opened);
            ASSERT_TRUE(viewer.project_lifecycle_
                            ->document_->source_path());
            EXPECT_EQ(viewer.project_lifecycle_
                          ->document_->source_path()
                          ->lexically_normal(),
                      replacement_path.lexically_normal());
            EXPECT_FALSE(std::filesystem::exists(
                recovery_temp));
            EXPECT_FALSE(viewer.project_lifecycle_
                             ->recovery_session_);
        }
    }

    TEST_F(VisualizerImplResetTest,
           FailedNewProjectKeepsRecoveredSessionTemp) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "recovered.licht";
        lfs::core::Uuid checkpoint_uuid;
        write_recoverable_project(
            project_path, "failed-new",
            &checkpoint_uuid);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            auto* const gui = viewer.getGuiManager();
            ASSERT_NE(gui, nullptr);
            installModalOverlay(gui->rml_modal_overlay_, gui->rmlui_manager_);
            auto offered = viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::DiscardChanges);
            ASSERT_TRUE(offered)
                << lfs::format_for_developer(
                       offered.error());
            ASSERT_EQ(*offered,
                      ProjectOpenOutcome::
                          RecoveryPromptPending);
            auto request = takeModalRequest(
                gui->rml_modal_overlay_->queue_mutex_, gui->rml_modal_overlay_->queue_);
            request.on_result({.button_label = "Recover"});
            const auto recovery_temp =
                *viewer.project_lifecycle_
                     ->recovery_session_path_;
            auto data_loader =
                std::move(viewer.data_loader_);
            auto failed = viewer.project_lifecycle_
                              ->newProject(
                                  ProjectSwitchDisposition::
                                      DiscardChanges);
            viewer.data_loader_ =
                std::move(data_loader);
            ASSERT_FALSE(failed);
            EXPECT_TRUE(std::filesystem::is_regular_file(
                recovery_temp));
            ASSERT_TRUE(viewer.project_lifecycle_
                            ->recovery_session_);
            EXPECT_TRUE(viewer.project_lifecycle_
                            ->recovery_session_
                            ->document_attached());
            const auto* checkpoint =
                viewer.project_lifecycle_
                    ->document_->find_ppisp(
                        checkpoint_uuid);
            ASSERT_NE(checkpoint, nullptr);
            std::array<std::byte, 4> marker{};
            ASSERT_TRUE(checkpoint->read_at(0, marker));
            EXPECT_EQ(marker[0], std::byte{0x49});
        }
    }

    TEST_F(VisualizerImplResetTest,
           RecoveredCloseDeletesTempAfterDocumentTeardown) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "recovered.licht";
        write_recoverable_project(
            project_path, "close");
        std::filesystem::path recovery_temp;
        {
            auto options = projectOptions();
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            auto* const gui = viewer.getGuiManager();
            ASSERT_NE(gui, nullptr);
            installModalOverlay(gui->rml_modal_overlay_, gui->rmlui_manager_);
            auto offered = viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::DiscardChanges);
            ASSERT_TRUE(offered)
                << lfs::format_for_developer(
                       offered.error());
            ASSERT_EQ(*offered,
                      ProjectOpenOutcome::
                          RecoveryPromptPending);
            auto request = takeModalRequest(
                gui->rml_modal_overlay_->queue_mutex_, gui->rml_modal_overlay_->queue_);
            request.on_result({.button_label = "Recover"});
            recovery_temp =
                *viewer.project_lifecycle_
                     ->recovery_session_path_;
            ASSERT_TRUE(std::filesystem::is_regular_file(
                recovery_temp));
        }
        EXPECT_FALSE(std::filesystem::exists(
            recovery_temp));
    }

    TEST_F(VisualizerImplResetTest,
           AutosaveStartsAfterFirstSaveAsWithoutReopen) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "first-save.licht";
        const auto sidecar =
            lfs::io::project::
                autosave_sidecar_path(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            auto untitled = viewer.projectGetInfo();
            ASSERT_TRUE(untitled);
            ASSERT_EQ(untitled->hydration_state,
                      "empty");
            ASSERT_FALSE(untitled->path.has_value());
            ASSERT_TRUE(viewer.project_lifecycle_
                            ->startAutosave());
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_EQ(viewer.project_lifecycle_
                          ->autosave_sequence_,
                      0u);
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "First explicit save"),
                lfs::core::NULL_NODE);
            auto saved = viewer.projectSaveAs(
                project_path, false);
            ASSERT_TRUE(saved)
                << lfs::format_for_developer(
                       saved.error());
            ASSERT_TRUE(pumpUntil(viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                return !viewer.jobs().anyRunning(JobType::ProjectWrite);
            }));
            ASSERT_FALSE(
                viewer.jobs().anyRunning(
                    JobType::ProjectWrite));
            ASSERT_TRUE(
                std::filesystem::is_regular_file(
                    project_path));
            auto info = viewer.projectGetInfo();
            ASSERT_TRUE(info);
            EXPECT_EQ(info->hydration_state,
                      "empty");

            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Dirty after first save"),
                lfs::core::NULL_NODE);
            ASSERT_TRUE(viewer.project_lifecycle_
                            ->startAutosave());
            EXPECT_TRUE(pumpUntil(viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                return !viewer.jobs().anyRunning(JobType::ProjectWrite);
            }));
            EXPECT_FALSE(
                viewer.jobs().anyRunning(
                    JobType::ProjectWrite));
            EXPECT_TRUE(
                std::filesystem::is_regular_file(
                    sidecar));
        }
    }

    TEST_F(VisualizerImplResetTest,
           AutosaveSkipsWhileManualProjectWriteJobIsRunning) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "exclusive.licht";
        const auto sidecar =
            lfs::io::project::
                autosave_sidecar_path(project_path);
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_,
                [&] {
                    const auto info = viewer.projectGetInfo();
                    EXPECT_TRUE(info);
                    return info && info->hydration_state == "complete";
                }));
            auto hydrated =
                viewer.projectGetInfo();
            ASSERT_TRUE(hydrated);
            ASSERT_EQ(hydrated->hydration_state,
                      "complete");
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Dirty for autosave"),
                lfs::core::NULL_NODE);

            auto manual = viewer.jobs().init(
                JobType::ProjectWrite,
                "Manual save held open");
            ASSERT_TRUE(manual);
            ASSERT_TRUE(viewer.project_lifecycle_
                            ->startAutosave());
            EXPECT_FALSE(
                std::filesystem::exists(sidecar));
            EXPECT_TRUE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_FALSE(viewer.project_lifecycle_
                             ->project_write_job_
                             .has_value());

            std::jthread worker([&] {
                viewer.jobs().work(*manual);
                viewer.jobs().finishWork(
                    *manual, false);
            });
            worker.join();
            auto completion =
                viewer.jobs().update(*manual);
            ASSERT_TRUE(completion);
            ASSERT_EQ(
                completion->status,
                JobStatus::CompletionPending);
            viewer.jobs().completed(*manual);
            viewer.jobs().free(*manual);

            ASSERT_TRUE(viewer.project_lifecycle_
                            ->startAutosave());
            EXPECT_TRUE(pumpUntil(viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                return !viewer.jobs().anyRunning(JobType::ProjectWrite);
            }));
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_TRUE(
                std::filesystem::is_regular_file(
                    sidecar));
        }
    }

    TEST_F(VisualizerImplResetTest,
           ProjectWriteSettlementCompletesBeforeNextDocumentWrite) {
        const auto& temporary = temporary_.path;
        const auto first_path =
            temporary / "first.licht";
        const auto second_path =
            temporary / "second.licht";
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr, viewer.getViewport());
            ASSERT_NE(viewer.getScene().addGroup(
                          "Settlement ordering"),
                      lfs::core::NULL_NODE);
            std::vector<std::string> sequence;
            ASSERT_TRUE(viewer.projectSaveAs(
                first_path, false));
            ASSERT_TRUE(viewer.project_lifecycle_
                            ->project_write_job_);
            const auto first_job =
                *viewer.project_lifecycle_
                     ->project_write_job_;
            ASSERT_TRUE(waitUntil(
                [&] {
                    const auto snapshot = viewer.jobs().peek(first_job);
                    return snapshot &&
                           snapshot->status == JobStatus::CompletionPending;
                }));
            auto pending =
                viewer.jobs().peek(first_job);
            ASSERT_TRUE(pending);
            ASSERT_EQ(pending->status,
                      JobStatus::CompletionPending);
            sequence.emplace_back(
                "settlement_queued");

            auto blocked = viewer.projectSaveAs(
                second_path, false);
            ASSERT_FALSE(blocked);
            sequence.emplace_back(
                "new_write_blocked");

            lfs::test::licht::drain_work_queue(
                viewer.work_queue_mutex_, viewer.work_queue_);
            ASSERT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            sequence.emplace_back("settled");

            ASSERT_TRUE(viewer.projectSaveAs(
                second_path, false));
            sequence.emplace_back(
                "new_write_started");
            EXPECT_TRUE(pumpUntil(viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                return !viewer.jobs().anyRunning(JobType::ProjectWrite);
            }));
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_EQ(sequence,
                      (std::vector<std::string>{
                          "settlement_queued",
                          "new_write_blocked",
                          "settled",
                          "new_write_started"}));
        }
    }

    TEST_F(VisualizerImplResetTest,
           TrainingSnapshotCleanupTerminalizesProjectWrite) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "cleanup.licht";
        write_empty_project(project_path);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::DiscardChanges));
            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Request cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model =
                scene.addGroup("Request model");
            scene.setTrainingModelNode(model);
            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto* const trainer = viewer.getTrainer();
            ASSERT_NE(trainer, nullptr);
            auto base =
                lfs::io::project::ProjectReader::open(
                    project_path);
            ASSERT_TRUE(base);
            const auto request_id =
                trainer->request_project_autosave(
                    project_path,
                    base->commit().commit_uuid, 1);
            ASSERT_TRUE(viewer.project_lifecycle_
                            ->startTrainingWrite(
                                project::ProjectLifecycle::
                                    ProjectWritePurpose::
                                        TrainingAutosave,
                                request_id,
                                project_path,
                                viewer.project_lifecycle_
                                    ->document_->dirty_epoch(),
                                1));
            viewer.project_lifecycle_
                ->project_write_autosave_sequence_ = 1;

            trainer->cleanup();
            EXPECT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_,
                [&] { return !viewer.jobs().anyRunning(JobType::ProjectWrite); },
                std::chrono::seconds(2)));
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_EQ(viewer.project_lifecycle_
                          ->autosave_sequence_,
                      0u);
            const auto metrics =
                trainer->get_project_snapshot_metrics();
            EXPECT_GE(metrics.last_failed_request_id,
                      request_id);

            scene.setTrainingModelNode(
                lfs::core::NULL_NODE);
            ASSERT_NE(scene.addGroup(
                          "Explicit after cleanup"),
                      lfs::core::NULL_NODE);
            auto explicit_save =
                viewer.projectSave(false);
            ASSERT_TRUE(explicit_save)
                << lfs::format_for_developer(
                       explicit_save.error());
            EXPECT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_,
                [&] { return !viewer.jobs().anyRunning(JobType::ProjectWrite); }));
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            auto info = viewer.projectGetInfo();
            ASSERT_TRUE(info);
            EXPECT_TRUE(info->project_write_error.empty());
        }
    }

    TEST_F(VisualizerImplResetTest,
           TrainingSnapshotPrepareFailureTerminalizesProjectWrite) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "prepare.licht";
        write_empty_project(project_path);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::DiscardChanges));
            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Prepare cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model =
                scene.addGroup("Prepare model");
            scene.setTrainingModelNode(model);
            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto* const trainer = viewer.getTrainer();
            auto base =
                lfs::io::project::ProjectReader::open(
                    project_path);
            ASSERT_TRUE(base);
            const auto request_id =
                trainer->request_project_autosave(
                    project_path,
                    base->commit().commit_uuid, 1);
            ASSERT_TRUE(viewer.project_lifecycle_
                            ->startTrainingWrite(
                                project::ProjectLifecycle::
                                    ProjectWritePurpose::
                                        TrainingAutosave,
                                request_id,
                                project_path, 0, 0));

            std::filesystem::path dequeued_path;
            std::vector<std::byte> preview;
            lfs::training::Trainer::
                ProjectSnapshotWriteKind write_kind;
            lfs::core::Uuid base_uuid;
            std::uint64_t sequence = 0;
            {
                std::lock_guard lock(
                    trainer->project_snapshot_mutex_);
                dequeued_path = std::move(
                    *trainer->requested_project_path_);
                trainer->requested_project_path_.reset();
                preview = std::move(
                    trainer->requested_project_preview_png_);
                trainer->requested_project_request_id_.reset();
                write_kind = std::exchange(
                    trainer->requested_project_write_kind_,
                    lfs::training::Trainer::
                        ProjectSnapshotWriteKind::Explicit);
                base_uuid = std::exchange(
                    trainer->requested_project_base_commit_uuid_,
                    lfs::core::Uuid{});
                sequence = std::exchange(
                    trainer->requested_project_autosave_sequence_,
                    0);
            }
            trainer->prepare_project_snapshot_at_safe_point(
                0, dequeued_path, std::move(preview),
                request_id, write_kind, base_uuid,
                sequence);

            EXPECT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_,
                [&] { return !viewer.jobs().anyRunning(JobType::ProjectWrite); },
                std::chrono::seconds(2)));
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_GE(trainer->get_project_snapshot_metrics()
                          .last_failed_request_id,
                      request_id);

            scene.setTrainingModelNode(
                lfs::core::NULL_NODE);
            ASSERT_NE(scene.addGroup(
                          "Explicit after prepare failure"),
                      lfs::core::NULL_NODE);
            auto explicit_save =
                viewer.projectSave(false);
            ASSERT_TRUE(explicit_save)
                << lfs::format_for_developer(
                       explicit_save.error());
            EXPECT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_,
                [&] { return !viewer.jobs().anyRunning(JobType::ProjectWrite); }));
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            auto info = viewer.projectGetInfo();
            ASSERT_TRUE(info);
            EXPECT_TRUE(
                info->project_write_error.empty());
        }
    }

    TEST_F(VisualizerImplResetTest,
           TrainingSnapshotSupersedeTerminalizesOldAndCompletesNew) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "supersede.licht";
        write_empty_project(project_path);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::DiscardChanges));
            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Supersede cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model =
                scene.addGroup("Supersede model");
            scene.setTrainingModelNode(model);
            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto* const trainer = viewer.getTrainer();
            auto base =
                lfs::io::project::ProjectReader::open(
                    project_path);
            ASSERT_TRUE(base);
            const auto first_id =
                trainer->request_project_autosave(
                    project_path,
                    base->commit().commit_uuid, 1);
            ASSERT_TRUE(viewer.project_lifecycle_
                            ->startTrainingWrite(
                                project::ProjectLifecycle::
                                    ProjectWritePurpose::
                                        TrainingAutosave,
                                first_id,
                                project_path, 0, 0));
            const auto second_id =
                trainer->request_project_autosave(
                    project_path,
                    base->commit().commit_uuid, 2);
            ASSERT_GT(second_id, first_id);

            EXPECT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_,
                [&] { return !viewer.jobs().anyRunning(JobType::ProjectWrite); },
                std::chrono::seconds(2)));
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_GE(trainer->get_project_snapshot_metrics()
                          .last_failed_request_id,
                      first_id);

            ASSERT_TRUE(viewer.project_lifecycle_
                            ->startTrainingWrite(
                                project::ProjectLifecycle::
                                    ProjectWritePurpose::
                                        TrainingAutosave,
                                second_id,
                                project_path, 0, 0));
            viewer.project_lifecycle_
                ->project_write_autosave_sequence_ = 2;
            {
                std::lock_guard lock(
                    trainer->project_snapshot_mutex_);
                trainer->last_completed_project_request_id_ =
                    std::max(
                        trainer->last_completed_project_request_id_,
                        second_id);
                trainer->last_project_writer_error_.clear();
                trainer->requested_project_path_.reset();
                trainer->requested_project_request_id_.reset();
                trainer->prestaged_project_chapters_.reset();
                trainer->prestaged_project_request_id_ = 0;
            }
            EXPECT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_,
                [&] { return !viewer.jobs().anyRunning(JobType::ProjectWrite); },
                std::chrono::seconds(2)));
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_GE(trainer->get_project_snapshot_metrics()
                          .last_completed_request_id,
                      second_id);
            EXPECT_EQ(viewer.project_lifecycle_
                          ->autosave_sequence_,
                      2u);
        }
    }

    TEST_F(VisualizerImplResetTest,
           TrainingSnapshotCancelTerminalizesBeforeSettlement) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "cancel.licht";
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::DiscardChanges));
            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Canceled snapshot cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model =
                scene.addGroup("Canceled snapshot model");
            scene.setTrainingModelNode(model);
            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto* const trainer = viewer.getTrainer();
            auto base =
                lfs::io::project::ProjectReader::open(
                    project_path);
            ASSERT_TRUE(base)
                << lfs::format_for_developer(
                       base.error());
            const auto request_id =
                trainer->request_project_autosave(
                    project_path,
                    base->commit().commit_uuid, 1);
            ASSERT_TRUE(viewer.project_lifecycle_
                            ->startTrainingWrite(
                                project::ProjectLifecycle::
                                    ProjectWritePurpose::
                                        TrainingAutosave,
                                request_id,
                                project_path, 0, 0));
            ASSERT_TRUE(viewer.project_lifecycle_
                            ->project_write_job_);
            const auto handle =
                *viewer.project_lifecycle_
                     ->project_write_job_;
            viewer.jobs().requestCancel(handle);

            EXPECT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_,
                [&] { return !viewer.project_lifecycle_->project_write_job_; },
                std::chrono::seconds(2)));

            EXPECT_FALSE(viewer.project_lifecycle_
                             ->project_write_job_);
            EXPECT_FALSE(viewer.jobs().peek(handle));
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            const auto metrics =
                trainer->get_project_snapshot_metrics();
            EXPECT_GE(metrics.last_failed_request_id,
                      request_id);
            EXPECT_NE(metrics.last_writer_error.find(
                          "Cancelled"),
                      std::string::npos);
        }
    }

    TEST_F(VisualizerImplResetTest,
           ImportWorkerFailureSettlesFailed) {
        ViewerOptions options;
        options.show_startup_overlay = false;
        VisualizerImpl viewer(options);
        auto* const gui = viewer.getGuiManager();
        ASSERT_NE(gui, nullptr);
        auto& tasks = gui->asyncTasks();
        const auto handle = viewer.jobs().init(
            JobType::Import, "Loading test dataset");
        ASSERT_TRUE(handle);
        tasks.import_state_.job = *handle;
        {
            const std::lock_guard lock(
                tasks.import_state_.mutex);
            tasks.import_state_.success = false;
        }

        std::jthread worker([&] {
            viewer.jobs().work(*handle);
            viewer.jobs().report(
                *handle, std::nullopt, "Failed",
                "injected import failure");
            viewer.jobs().finishWork(
                *handle, false,
                "injected import failure");
        });
        worker.join();
        tasks.import_state_.load_complete.store(
            true, std::memory_order_release);
        tasks.checkAsyncImportCompletion();

        const auto failed =
            viewer.jobs().update(*handle);
        ASSERT_TRUE(failed);
        EXPECT_EQ(failed->status,
                  JobStatus::Failed);
        EXPECT_FALSE(failed->running());
        EXPECT_FALSE(viewer.jobs().anyRunning(
            JobType::Import));
        EXPECT_EQ(failed->error,
                  "injected import failure");
        viewer.jobs().free(*handle);
        tasks.import_state_.job = {};
    }

    TEST_F(VisualizerImplResetTest,
           FailedOpenOverOpenPreservesCurrentSceneAndUndoHistory) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "first.licht";
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.projectOpen(project_path));
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Current after first open"),
                lfs::core::NULL_NODE);
            op::undoHistory().push(
                std::make_unique<NoopUndoEntry>());

            const auto failed = viewer.projectOpen(
                temporary / "missing.licht",
                ProjectSwitchDisposition::
                    DiscardChanges);
            ASSERT_FALSE(failed);
            EXPECT_NE(
                viewer.getScene().getNode(
                    "Current after first open"),
                nullptr);
            EXPECT_EQ(
                op::undoHistory().undoCount(), 1u);
        }
    }

    TEST_F(VisualizerImplResetTest,
           PhaseAParameterAndSessionFailuresPreserveCurrentProject) {
        const auto& temporary = temporary_.path;
        const auto bad_parameters =
            temporary / "bad-parameters.licht";
        const auto bad_session =
            temporary / "bad-session.licht";
        write_invalid_phase_a_project(
            bad_parameters, false);
        write_invalid_phase_a_project(
            bad_session, true);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Current survives Phase A"),
                lfs::core::NULL_NODE);
            op::undoHistory().push(
                std::make_unique<NoopUndoEntry>());

            for (const auto& candidate :
                 {bad_parameters, bad_session}) {
                const auto failed =
                    viewer.projectOpen(
                        candidate,
                        ProjectSwitchDisposition::
                            DiscardChanges);
                ASSERT_FALSE(failed);
                EXPECT_NE(
                    viewer.getScene().getNode(
                        "Current survives Phase A"),
                    nullptr);
                EXPECT_EQ(
                    op::undoHistory().undoCount(),
                    1u);
            }
        }
    }

    TEST_F(VisualizerImplResetTest,
           DirtyProjectSwitchRequiresExplicitDiscardAuthorization) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "candidate.licht";
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Unsaved current project"),
                lfs::core::NULL_NODE);

            bool drag_drop_prompted = false;
            std::filesystem::path prompted_path;
            lfs::core::events::cmd::
                ShowProjectSwitchConfirmation::
                    when([&](const auto& event) {
                        drag_drop_prompted = true;
                        prompted_path =
                            event.path;
                    });
            lfs::core::events::cmd::
                ProjectOpen{
                    .path = project_path}
                    .emit();
            EXPECT_TRUE(drag_drop_prompted);
            EXPECT_EQ(
                prompted_path, project_path);
            EXPECT_NE(
                viewer.getScene().getNode(
                    "Unsaved current project"),
                nullptr);

            const auto blocked =
                viewer.projectOpen(project_path);
            ASSERT_FALSE(blocked);
            EXPECT_EQ(
                blocked.error().code(),
                lfs::ErrorCode::
                    FailedPrecondition);
            EXPECT_EQ(
                blocked.error().user_message(),
                "The current project has unsaved changes.");
            EXPECT_NE(
                viewer.getScene().getNode(
                    "Unsaved current project"),
                nullptr);

            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            EXPECT_EQ(
                viewer.getScene().getNodeCount(),
                0u);
        }
    }

    TEST_F(VisualizerImplResetTest,
           NewProjectDirtyGateRunsBelowEveryCommandEntry) {
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Unsaved current project"),
                lfs::core::NULL_NODE);

            lfs::core::events::cmd::
                NewProject{}
                    .emit();
            EXPECT_NE(
                viewer.getScene().getNode(
                    "Unsaved current project"),
                nullptr);

            lfs::core::events::cmd::
                NewProject{
                    .discard_changes = true}
                    .emit();
            EXPECT_EQ(
                viewer.getScene().getNodeCount(),
                0u);
        }
    }

    TEST_F(VisualizerImplResetTest,
           FailedAutosaveSettlementAppliesBackoffBeforeRetry) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "backoff.licht";
        const auto backup_path =
            temporary / "backoff.backup";
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Dirty for failed autosave"),
                lfs::core::NULL_NODE);

            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            lifecycle->settings_
                .autosave_dirty_epoch_threshold = 1;
            lifecycle->last_autosaved_dirty_epoch_ =
                0;
            lifecycle->last_autosaved_scene_serial_ =
                0;
            lifecycle->last_autosave_at_ =
                std::chrono::steady_clock::now() -
                std::chrono::hours(1);

            std::filesystem::rename(
                project_path, backup_path);
            lifecycle->updateMaintenance();
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_FALSE(
                lifecycle->last_project_write_error_
                    .empty());
            EXPECT_EQ(
                lifecycle
                    ->autosave_failure_backoff_seconds_,
                60u);
            EXPECT_GE(
                lifecycle->autosave_retry_not_before_,
                std::chrono::steady_clock::now());

            lifecycle->last_project_write_error_
                .clear();
            lifecycle->updateMaintenance();
            EXPECT_TRUE(
                lifecycle->last_project_write_error_
                    .empty());
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_FALSE(
                lifecycle->project_write_job_
                    .has_value());
            EXPECT_EQ(
                lifecycle
                    ->autosave_failure_backoff_seconds_,
                60u);

            std::filesystem::rename(
                backup_path, project_path);
        }
    }

    TEST_F(VisualizerImplResetTest,
           PendingCloseSuppressesBackgroundAutosave) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "close-suppress.licht";
        const auto sidecar =
            lfs::io::project::
                autosave_sidecar_path(project_path);
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Dirty before close"),
                lfs::core::NULL_NODE);

            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ASSERT_TRUE(
                lifecycle->setAutoSaveOnClose(
                    false));
            lifecycle->settings_
                .autosave_dirty_epoch_threshold = 1;
            lifecycle->last_autosaved_dirty_epoch_ =
                0;
            lifecycle->last_autosaved_scene_serial_ =
                0;
            lifecycle->last_autosave_at_ =
                std::chrono::steady_clock::now() -
                std::chrono::hours(1);

            EXPECT_EQ(
                lifecycle->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::NeedsPrompt);
            EXPECT_TRUE(
                lifecycle->application_close_pending_);
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));

            lifecycle->updateMaintenance();
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_FALSE(
                lifecycle->project_write_job_
                    .has_value());
            EXPECT_FALSE(
                std::filesystem::exists(sidecar));

            lifecycle->resetCloseSaveAttempt();
            EXPECT_FALSE(
                lifecycle->application_close_pending_);
        }
    }

    TEST_F(VisualizerImplResetTest,
           FileExitWithDefaultSettingsNeedsPrompt) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "session.licht";
        constexpr float restored_focal_length =
            73.0f;
        write_empty_project(
            project_path, restored_focal_length);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            const auto opened =
                viewer.projectOpen(project_path);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(
                       opened.error());
            ASSERT_TRUE(
                viewer
                    .isProjectSessionRestorePending());
            const auto before =
                viewer.projectGetInfo();
            ASSERT_TRUE(before);
            EXPECT_FALSE(before->dirty);
            EXPECT_FALSE(before->auto_save_on_close);

            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Unsaved before File Exit"),
                lfs::core::NULL_NODE);
            auto dirty = viewer.projectGetInfo();
            ASSERT_TRUE(dirty);
            ASSERT_TRUE(dirty->dirty);

            const auto mtime_before =
                std::filesystem::last_write_time(
                    project_path);

            lfs::core::events::cmd::
                RequestExit{}
                    .emit();
            ASSERT_TRUE(
                viewer.getWindowManager()
                    ->shouldClose());
            ASSERT_FALSE(
                viewer.getGuiManager()
                    ->isForceExit());

            EXPECT_FALSE(viewer.allowclose());
            EXPECT_FALSE(
                viewer.getWindowManager()
                    ->shouldClose());
            EXPECT_EQ(
                viewer.project_lifecycle_
                    ->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::NeedsPrompt);
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_EQ(
                std::filesystem::last_write_time(
                    project_path),
                mtime_before);
        }
    }

    TEST_F(VisualizerImplResetTest,
           CloseSavePendingActionSkipsPreviewRegen) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "close-save-exit.licht";
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            const auto opened =
                viewer.projectOpen(project_path);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(
                       opened.error());

            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Unsaved before CloseSave exit"),
                lfs::core::NULL_NODE);
            auto dirty = viewer.projectGetInfo();
            ASSERT_TRUE(dirty);
            ASSERT_TRUE(dirty->dirty);

            // Catches CloseSave exit failing to complete a save in a
            // frame-less session (capture carries prior THMB forward).
            // Explicit save with regenerate_preview=true must still succeed
            // when capture is unavailable under the partial-hydration contract.
            const auto with_preview =
                viewer.projectSave(true);
            ASSERT_TRUE(with_preview)
                << lfs::format_for_developer(
                       with_preview.error());
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info && !info->dirty &&
                           !viewer.jobs().anyRunning(
                               JobType::ProjectWrite);
                }));

            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Unsaved again for CloseSave"),
                lfs::core::NULL_NODE);
            dirty = viewer.projectGetInfo();
            ASSERT_TRUE(dirty);
            ASSERT_TRUE(dirty->dirty);

            viewer.pending_training_action_ =
                VisualizerImpl::PendingTrainingAction::
                    CloseSave;
            viewer.performPendingTrainingAction();

            ASSERT_TRUE(
                viewer.getWindowManager()
                    ->shouldClose());
            EXPECT_FALSE(viewer.allowclose());
            EXPECT_FALSE(
                viewer.getWindowManager()
                    ->shouldClose());

            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info && !info->dirty &&
                           !viewer.jobs().anyRunning(
                               JobType::ProjectWrite);
                }));

            const auto after =
                viewer.projectGetInfo();
            ASSERT_TRUE(after);
            EXPECT_FALSE(after->dirty);
            EXPECT_TRUE(
                viewer.getWindowManager()
                    ->shouldClose());
            EXPECT_TRUE(viewer.allowclose());
        }
    }

    TEST_F(VisualizerImplResetTest,
           SaveAsAndExitContinuesAfterProjectWriteCompletes) {
        const auto project_path =
            temporary_.path /
            "save-as-exit.licht";

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Unsaved before Save As exit"),
                lfs::core::NULL_NODE);

            const auto saved =
                viewer.projectSaveAs(
                    project_path, false);
            ASSERT_TRUE(saved)
                << lfs::format_for_developer(
                       saved.error());
            viewer.requestApplicationClose();

            ASSERT_TRUE(
                viewer.getWindowManager()
                    ->shouldClose());
            EXPECT_FALSE(viewer.allowclose());
            EXPECT_FALSE(
                viewer.getWindowManager()
                    ->shouldClose());

            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    return viewer
                        .getWindowManager()
                        ->shouldClose();
                }));
            const auto after =
                viewer.projectGetInfo();
            ASSERT_TRUE(after)
                << lfs::format_for_developer(
                       after.error());
            EXPECT_FALSE(after->dirty);
            ASSERT_TRUE(after->path);
            EXPECT_EQ(
                after->path->lexically_normal(),
                project_path.lexically_normal());
            EXPECT_TRUE(viewer.allowclose());
        }
    }

    TEST_F(VisualizerImplResetTest,
           SaveAsAndExitClearsSelectionDirtyBaseline) {
        const auto source_path = temporary_.path / "selection-source.licht";
        const auto destination = temporary_.path / "selection-save-as.licht";
        write_empty_project(source_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(nullptr, viewer.getViewport());
            ASSERT_TRUE(viewer.projectOpen(source_path));
            ASSERT_NE(viewer.getScene().addGroup("Selectable"),
                      lfs::core::NULL_NODE);
            ASSERT_TRUE(viewer.projectSave(false));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                    return !viewer.jobs().anyRunning(JobType::ProjectWrite);
                }));
            ASSERT_FALSE(viewer.project_lifecycle_->hasDirtyProject());

            viewer.getSceneManager()->selectNode("Selectable");
            viewer.project_lifecycle_->markSceneMutation(
                static_cast<std::uint32_t>(
                    lfs::core::Scene::MutationType::SELECTION_CHANGED));
            ASSERT_TRUE(viewer.project_lifecycle_->hasDirtyProject());

            ASSERT_TRUE(viewer.projectSaveAs(destination, false));
            viewer.requestApplicationClose();
            ASSERT_TRUE(viewer.getWindowManager()->shouldClose());
            EXPECT_FALSE(viewer.allowclose());
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                    return viewer.getWindowManager()->shouldClose();
                }));

            const auto after = viewer.projectGetInfo();
            ASSERT_TRUE(after);
            EXPECT_FALSE(after->dirty);
            EXPECT_EQ(after->path->lexically_normal(),
                      destination.lexically_normal());
            EXPECT_TRUE(viewer.allowclose());
        }
    }

    TEST_F(VisualizerImplResetTest,
           SaveAsAndExitClearsParameterDirtyBaseline) {
        const auto source_path = temporary_.path / "parameter-source.licht";
        const auto destination = temporary_.path / "parameter-save-as.licht";
        write_empty_project(source_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(nullptr, viewer.getViewport());
            ASSERT_TRUE(viewer.projectOpen(source_path));
            ASSERT_TRUE(viewer.projectSave(false));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                    return !viewer.jobs().anyRunning(JobType::ProjectWrite);
                }));
            ASSERT_FALSE(viewer.project_lifecycle_->hasDirtyProject());

            viewer.getParameterManager()->modifyActiveParams(
                [](auto& params) { params.iterations += 1; });
            ASSERT_TRUE(viewer.project_lifecycle_->hasDirtyProject());

            ASSERT_TRUE(viewer.projectSaveAs(destination, false));
            viewer.requestApplicationClose();
            ASSERT_TRUE(viewer.getWindowManager()->shouldClose());
            EXPECT_FALSE(viewer.allowclose());
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                    return viewer.getWindowManager()->shouldClose();
                }));

            const auto after = viewer.projectGetInfo();
            ASSERT_TRUE(after);
            EXPECT_FALSE(after->dirty);
            EXPECT_FALSE(viewer.getParameterManager()->isDirty());
            EXPECT_EQ(after->path->lexically_normal(),
                      destination.lexically_normal());
            EXPECT_TRUE(viewer.allowclose());
        }
    }

    TEST_F(VisualizerImplResetTest,
           DialogSaveAsReplacesExistingFirstSave) {
        const auto destination = temporary_.path / "dialog-existing.licht";
        write_empty_project(destination);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(nullptr, viewer.getViewport());
            ASSERT_NE(viewer.getScene().addGroup("Dialog save"),
                      lfs::core::NULL_NODE);

            ASSERT_TRUE(viewer.projectSaveAsFromDialog(destination, false));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                    return !viewer.jobs().anyRunning(JobType::ProjectWrite);
                }));
            const auto info = viewer.projectGetInfo();
            ASSERT_TRUE(info);
            ASSERT_TRUE(info->path);
            EXPECT_EQ(info->path->lexically_normal(),
                      destination.lexically_normal());
            EXPECT_FALSE(info->dirty);
            viewer.requestApplicationClose();
            EXPECT_TRUE(viewer.getWindowManager()->shouldClose());
            EXPECT_TRUE(viewer.allowclose());
        }
    }

    TEST_F(VisualizerImplResetTest,
           McpExplicitSaveAsReplacesExistingFirstSave) {
        const auto destination = temporary_.path / "mcp-existing.licht";
        write_empty_project(destination);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(nullptr, viewer.getViewport());
            ASSERT_NE(viewer.getScene().addGroup("MCP save"),
                      lfs::core::NULL_NODE);

            ASSERT_TRUE(viewer.projectSaveAsExplicit(destination, false));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                    return !viewer.jobs().anyRunning(JobType::ProjectWrite);
                }));
            const auto info = viewer.projectGetInfo();
            ASSERT_TRUE(info);
            EXPECT_FALSE(info->project_write_error_code.has_value());
            EXPECT_FALSE(info->dirty);
            ASSERT_TRUE(info->path);
            EXPECT_EQ(info->path->lexically_normal(),
                      destination.lexically_normal());
        }
    }

    TEST_F(VisualizerImplResetTest,
           McpImplicitSaveAsReportsTypedFailure) {
        const auto destination = temporary_.path / "mcp-implicit.licht";
        write_empty_project(destination);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(nullptr, viewer.getViewport());
            ASSERT_NE(viewer.getScene().addGroup("MCP implicit save"),
                      lfs::core::NULL_NODE);

            ASSERT_TRUE(viewer.projectSaveAs(destination, false));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                    return !viewer.jobs().anyRunning(JobType::ProjectWrite);
                }));
            const auto info = viewer.projectGetInfo();
            ASSERT_TRUE(info);
            EXPECT_EQ(info->project_write_error_code,
                      std::optional<lfs::ErrorCode>{
                          lfs::ErrorCode::AlreadyExists});
            EXPECT_FALSE(info->project_write_error.empty());
            EXPECT_TRUE(info->dirty);
        }
    }

    TEST_F(VisualizerImplResetTest,
           FileExitRoutesThroughCloseSaveWhenAutoSaveOnCloseEnabled) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "session.licht";
        constexpr float restored_focal_length =
            73.0f;
        write_empty_project(
            project_path, restored_focal_length);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            const auto opened =
                viewer.projectOpen(project_path);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(
                       opened.error());
            ASSERT_TRUE(
                viewer
                    .isProjectSessionRestorePending());
            ASSERT_TRUE(
                viewer.project_lifecycle_
                    ->setAutoSaveOnClose(true));
            const auto before =
                viewer.projectGetInfo();
            ASSERT_TRUE(before);
            EXPECT_FALSE(before->dirty);
            EXPECT_TRUE(before->auto_save_on_close);

            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Unsaved before File Exit"),
                lfs::core::NULL_NODE);
            auto dirty = viewer.projectGetInfo();
            ASSERT_TRUE(dirty);
            ASSERT_TRUE(dirty->dirty);

            lfs::core::events::cmd::
                RequestExit{}
                    .emit();
            ASSERT_TRUE(
                viewer.getWindowManager()
                    ->shouldClose());
            ASSERT_FALSE(
                viewer.getGuiManager()
                    ->isForceExit());

            EXPECT_FALSE(viewer.allowclose());
            EXPECT_FALSE(
                viewer.getWindowManager()
                    ->shouldClose());
            EXPECT_EQ(
                viewer.project_lifecycle_
                    ->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::Saving);

            ASSERT_TRUE(pumpUntil(viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                return viewer.getWindowManager()->shouldClose();
            }));
            ASSERT_TRUE(
                viewer.getWindowManager()
                    ->shouldClose());
            ASSERT_EQ(
                viewer.project_lifecycle_
                    ->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::Succeeded);

            const auto after =
                viewer.projectGetInfo();
            ASSERT_TRUE(after);
            EXPECT_GT(
                after->generation,
                before->generation);
            EXPECT_FALSE(after->dirty);
            EXPECT_TRUE(viewer.allowclose());

            auto reopened =
                lfs::io::project::
                    ProjectDocument::open(
                        project_path);
            ASSERT_TRUE(reopened)
                << lfs::format_for_developer(
                       reopened.error());
            const auto focal_length =
                reopened->view().dom().get_json(
                    "render_settings.focal_length_mm");
            ASSERT_TRUE(focal_length);
            ASSERT_TRUE(
                focal_length->is_number());
            EXPECT_FLOAT_EQ(
                focal_length->get<float>(),
                restored_focal_length);
        }
    }

    TEST_F(VisualizerImplResetTest,
           CancelExitAndNextWindowAttemptRecoverFromFailedCloseSave) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "session.licht";
        const auto backup_path =
            temporary / "session.backup";

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            ASSERT_NE(
                viewer.getScene().addGroup("Saved"),
                lfs::core::NULL_NODE);
            const auto initial_save =
                viewer.projectSaveAs(
                    project_path, false);
            ASSERT_TRUE(initial_save)
                << lfs::format_for_developer(
                       initial_save.error());
            ASSERT_TRUE(pumpUntil(viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                return !viewer.jobs().anyRunning(JobType::ProjectWrite);
            }));
            ASSERT_FALSE(
                viewer.jobs().anyRunning(
                    JobType::ProjectWrite));
            ASSERT_TRUE(
                std::filesystem::is_regular_file(
                    project_path));
            ASSERT_TRUE(
                viewer.project_lifecycle_
                    ->setAutoSaveOnClose(true));
            std::filesystem::rename(
                project_path, backup_path);
            std::filesystem::create_directory(
                project_path);
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Unsaved before failed close"),
                lfs::core::NULL_NODE);

            lfs::core::events::cmd::
                RequestExit{}
                    .emit();
            ASSERT_FALSE(viewer.allowclose());

            EXPECT_TRUE(pumpUntil(viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                return viewer.getWindowManager()->shouldClose();
            }));
            ASSERT_TRUE(
                viewer.getWindowManager()
                    ->shouldClose());
            ASSERT_EQ(
                viewer.project_lifecycle_
                    ->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::Failed);

            std::filesystem::remove_all(
                project_path);
            std::filesystem::rename(
                backup_path, project_path);
            lfs::core::events::cmd::
                CancelExit{}
                    .emit();
            EXPECT_FALSE(
                viewer.getWindowManager()
                    ->shouldClose());

            viewer.getWindowManager()
                ->requestClose();
            EXPECT_FALSE(viewer.allowclose());
            EXPECT_EQ(
                viewer.project_lifecycle_
                    ->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::Saving);

            EXPECT_TRUE(pumpUntil(viewer.work_queue_mutex_, viewer.work_queue_, [&] {
                return viewer.getWindowManager()->shouldClose();
            }));
            ASSERT_TRUE(
                viewer.getWindowManager()
                    ->shouldClose());
            EXPECT_EQ(
                viewer.project_lifecycle_
                    ->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::Succeeded);
            EXPECT_TRUE(viewer.allowclose());
        }
    }

    // Catches background maintenance grabbing the master writer lock while a
    // stopping trainer still owes its terminal append (lost training generation).
    TEST_F(VisualizerImplResetTest,
           StoppingTrainerBlocksIdleCompactionAndAutosave) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "stopping-window.licht";
        const auto sidecar =
            lfs::io::project::
                autosave_sidecar_path(project_path);
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));

            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            ASSERT_TRUE(viewer.projectSave(false));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    return !viewer.jobs().anyRunning(
                        JobType::ProjectWrite);
                }));
            ASSERT_FALSE(
                lifecycle->hasDirtyProject());
            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto& state_machine =
                const_cast<TrainingStateMachine&>(
                    viewer.getTrainerManager()
                        ->getStateMachine());
            if (state_machine.getState() ==
                TrainingState::Idle) {
                ASSERT_TRUE(state_machine.transitionTo(
                    TrainingState::Ready));
            }
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Running));
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Stopping));
            ASSERT_FALSE(viewer.getTrainerManager()
                             ->isTrainingActive());

            const auto prime_maintenance =
                [&] {
                    lifecycle->next_storage_check_at_ =
                        std::chrono::steady_clock::
                            now() +
                        std::chrono::hours(1);
                    lifecycle->settings_
                        .compaction_idle_seconds = 1;
                    lifecycle->last_mutation_at_ =
                        std::chrono::steady_clock::
                            now() -
                        std::chrono::hours(1);
                    lifecycle->settings_
                        .autosave_dirty_epoch_threshold =
                        1;
                    lifecycle
                        ->last_autosaved_dirty_epoch_ =
                        0;
                    lifecycle
                        ->last_autosaved_scene_serial_ =
                        0;
                    lifecycle->last_autosave_at_ =
                        std::chrono::steady_clock::
                            now() -
                        std::chrono::hours(1);
                };

            // Idle compaction would take the master
            // writer lock the terminal append needs.
            lifecycle->compaction_suggested_ = true;
            lifecycle->scene_dirty_.store(
                false, std::memory_order_release);
            lifecycle->payload_dirty_.store(
                false, std::memory_order_release);
            prime_maintenance();
            lifecycle->updateMaintenance();
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_FALSE(
                lifecycle->project_write_job_
                    .has_value());

            // Hard dirt blocks compaction, so this leg
            // proves the autosave path stays parked too.
            lifecycle->compaction_suggested_ = false;
            ASSERT_NE(
                scene.addGroup("Hard dirt"),
                lfs::core::NULL_NODE);
            ASSERT_TRUE(
                lifecycle->hasDirtyProject());
            prime_maintenance();
            lifecycle->updateMaintenance();
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_FALSE(
                lifecycle->project_write_job_
                    .has_value());
            EXPECT_FALSE(
                std::filesystem::exists(sidecar));

            viewer.getTrainerManager()
                ->clearTrainer();
        }
    }

    TEST_F(VisualizerImplResetTest,
           BaselineIdleCheckpointTrainerClosesWithoutTrainingPrompt) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "baseline-idle.licht";
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            const auto opened =
                viewer.projectOpen(
                    project_path,
                    ProjectSwitchDisposition::
                        DiscardChanges);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(
                       opened.error());
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));

            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ASSERT_NE(lifecycle->document_, nullptr);
            (void)lifecycle->document_
                ->edit_gui_layout();
            (void)lifecycle->document_
                ->edit_view();

            // Soft-only dirt must not force a prompt;
            // clear any hard dirt from scene edits.
            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            ASSERT_TRUE(viewer.projectSave(false));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    return !viewer.jobs().anyRunning(
                        JobType::ProjectWrite);
                }));
            (void)lifecycle->document_
                ->edit_gui_layout();
            (void)lifecycle->document_
                ->edit_view();

            auto* const trainer_manager =
                viewer.getTrainerManager();
            ASSERT_NE(trainer_manager, nullptr);
            trainer_manager->setTrainerFromCheckpoint(
                std::make_unique<
                    lfs::training::Trainer>(scene),
                0);
            ASSERT_TRUE(
                trainer_manager->isTrainingActive());
            ASSERT_TRUE(
                trainer_manager->isPaused());
            ASSERT_TRUE(
                trainer_manager
                    ->isPausedAtCheckpointBaseline());
            EXPECT_EQ(
                trainer_manager
                    ->getCurrentIteration(),
                0);
            ASSERT_TRUE(
                trainer_manager
                    ->checkpointBaselineIteration()
                    .has_value());
            EXPECT_EQ(
                *trainer_manager
                     ->checkpointBaselineIteration(),
                0);

            const auto info =
                viewer.projectGetInfo();
            ASSERT_TRUE(info)
                << lfs::format_for_developer(
                       info.error());
            EXPECT_FALSE(info->dirty);
            EXPECT_TRUE(info->session_dirty);
            EXPECT_NE(
                std::find(
                    info->dirty_chapters.begin(),
                    info->dirty_chapters.end(),
                    "GUIL"),
                info->dirty_chapters.end());
            EXPECT_NE(
                std::find(
                    info->dirty_chapters.begin(),
                    info->dirty_chapters.end(),
                    "VIEW"),
                info->dirty_chapters.end());
            EXPECT_FALSE(
                lifecycle->hasDirtyProject());
            EXPECT_EQ(
                lifecycle->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::NotDirty);
            lifecycle->resetCloseSaveAttempt();

            viewer.getWindowManager()
                ->requestClose();
            EXPECT_TRUE(viewer.allowclose());
        }
    }

    TEST_F(VisualizerImplResetTest,
           ProgressedPausedTrainerStillBlocksCleanClose) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "progressed-paused.licht";
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            const auto opened =
                viewer.projectOpen(
                    project_path,
                    ProjectSwitchDisposition::
                        DiscardChanges);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(
                       opened.error());
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));

            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ASSERT_TRUE(
                lifecycle->setAutoSaveOnClose(
                    false));

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());

            auto* const trainer_manager =
                viewer.getTrainerManager();
            ASSERT_NE(trainer_manager, nullptr);
            // Baseline N with trainer still at 0
            // models unsaved progress off the
            // installed checkpoint iteration.
            trainer_manager->setTrainerFromCheckpoint(
                std::make_unique<
                    lfs::training::Trainer>(scene),
                42);
            ASSERT_TRUE(
                trainer_manager->isTrainingActive());
            ASSERT_TRUE(
                trainer_manager->isPaused());
            ASSERT_FALSE(
                trainer_manager
                    ->isPausedAtCheckpointBaseline());
            EXPECT_EQ(
                trainer_manager
                    ->getCurrentIteration(),
                0);
            ASSERT_TRUE(
                trainer_manager
                    ->checkpointBaselineIteration()
                    .has_value());
            EXPECT_EQ(
                *trainer_manager
                     ->checkpointBaselineIteration(),
                42);

            EXPECT_TRUE(
                lifecycle->hasDirtyProject());
            const auto info =
                viewer.projectGetInfo();
            ASSERT_TRUE(info)
                << lfs::format_for_developer(
                       info.error());
            EXPECT_TRUE(info->dirty);

            viewer.getWindowManager()
                ->requestClose();
            EXPECT_FALSE(viewer.allowclose());
            EXPECT_FALSE(
                viewer.getWindowManager()
                    ->shouldClose());
            // Soft-only session dirt never reaches
            // the dirty prompt while training
            // progress still owns the exit gate.
            lifecycle->resetCloseSaveAttempt();
        }
    }

    TEST_F(VisualizerImplResetTest,
           SessionSoftDirtyDoesNotPromptOrArmAutosave) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "session-soft.licht";
        const auto sidecar =
            lfs::io::project::
                autosave_sidecar_path(project_path);
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            const auto opened =
                viewer.projectOpen(
                    project_path,
                    ProjectSwitchDisposition::
                        DiscardChanges);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(
                       opened.error());
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));

            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ASSERT_NE(lifecycle->document_, nullptr);
            (void)lifecycle->document_
                ->edit_gui_layout();
            (void)lifecycle->document_
                ->edit_view();

            const auto info =
                viewer.projectGetInfo();
            ASSERT_TRUE(info)
                << lfs::format_for_developer(
                       info.error());
            EXPECT_FALSE(info->dirty);
            EXPECT_TRUE(info->session_dirty);
            EXPECT_NE(
                std::find(
                    info->dirty_chapters.begin(),
                    info->dirty_chapters.end(),
                    "GUIL"),
                info->dirty_chapters.end());
            EXPECT_NE(
                std::find(
                    info->dirty_chapters.begin(),
                    info->dirty_chapters.end(),
                    "VIEW"),
                info->dirty_chapters.end());
            EXPECT_FALSE(
                lifecycle->hasDirtyProject());
            EXPECT_EQ(
                lifecycle->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::NotDirty);
            lifecycle->resetCloseSaveAttempt();

            ASSERT_TRUE(
                lifecycle->setAutoSaveOnClose(
                    false));
            lifecycle->settings_
                .autosave_dirty_epoch_threshold = 1;
            lifecycle->last_autosaved_dirty_epoch_ =
                0;
            lifecycle
                ->last_autosaved_scene_serial_ = 0;
            lifecycle->last_autosave_at_ =
                std::chrono::steady_clock::now() -
                std::chrono::hours(1);
            lifecycle->updateMaintenance();
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_FALSE(
                lifecycle->project_write_job_
                    .has_value());
            EXPECT_FALSE(
                std::filesystem::exists(sidecar));
        }
    }

    TEST_F(VisualizerImplResetTest,
           SceneEditStillPromptsAndArmsAutosave) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "scene-hard.licht";
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            const auto opened =
                viewer.projectOpen(
                    project_path,
                    ProjectSwitchDisposition::
                        DiscardChanges);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(
                       opened.error());
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));

            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Hard dirt scene edit"),
                lfs::core::NULL_NODE);

            const auto info =
                viewer.projectGetInfo();
            ASSERT_TRUE(info)
                << lfs::format_for_developer(
                       info.error());
            EXPECT_TRUE(info->dirty);

            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ASSERT_TRUE(
                lifecycle->setAutoSaveOnClose(
                    false));
            EXPECT_TRUE(
                lifecycle->hasDirtyProject());
            EXPECT_EQ(
                lifecycle->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::NeedsPrompt);
            lifecycle->resetCloseSaveAttempt();

            lifecycle->settings_
                .autosave_dirty_epoch_threshold = 1;
            lifecycle->last_autosaved_dirty_epoch_ =
                0;
            lifecycle
                ->last_autosaved_scene_serial_ = 0;
            lifecycle->last_autosave_at_ =
                std::chrono::steady_clock::now() -
                std::chrono::hours(1);
            lifecycle->updateMaintenance();
            EXPECT_TRUE(
                lifecycle->project_write_job_
                    .has_value() ||
                viewer.jobs().anyRunning(
                    JobType::ProjectWrite));
        }
    }

    TEST_F(VisualizerImplResetTest,
           ParametersUnchangedRoundTripStaysClean) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "prms-clean.licht";
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            const auto opened =
                viewer.projectOpen(
                    project_path,
                    ProjectSwitchDisposition::
                        DiscardChanges);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(
                       opened.error());
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));

            const auto info =
                viewer.projectGetInfo();
            ASSERT_TRUE(info)
                << lfs::format_for_developer(
                       info.error());
            EXPECT_FALSE(info->dirty);
            EXPECT_EQ(
                std::find(
                    info->dirty_chapters.begin(),
                    info->dirty_chapters.end(),
                    "PRMS"),
                info->dirty_chapters.end());
            EXPECT_FALSE(
                viewer.project_lifecycle_
                    ->hasDirtyProject());
        }
    }

    TEST_F(VisualizerImplResetTest,
           ParametersValueChangeIsHardDirty) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "prms-dirty.licht";
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            const auto opened =
                viewer.projectOpen(
                    project_path,
                    ProjectSwitchDisposition::
                        DiscardChanges);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(
                       opened.error());
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));

            viewer.getParameterManager()
                ->modifyActiveParams(
                    [](lfs::core::param::
                           OptimizationParameters&
                               params) {
                        params.iterations += 1;
                    });

            const auto info =
                viewer.projectGetInfo();
            ASSERT_TRUE(info)
                << lfs::format_for_developer(
                       info.error());
            EXPECT_TRUE(info->dirty);
            EXPECT_EQ(
                std::find(
                    info->dirty_chapters.begin(),
                    info->dirty_chapters.end(),
                    "PRMS"),
                info->dirty_chapters.end());
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            auto synchronized =
                lifecycle
                    ->synchronizeDocumentFromViewer();
            ASSERT_TRUE(synchronized)
                << lfs::format_for_developer(
                       synchronized.error());
            const auto synced =
                viewer.projectGetInfo();
            ASSERT_TRUE(synced)
                << lfs::format_for_developer(
                       synced.error());
            EXPECT_NE(
                std::find(
                    synced->dirty_chapters.begin(),
                    synced->dirty_chapters.end(),
                    "PRMS"),
                synced->dirty_chapters.end());
            EXPECT_TRUE(
                viewer.project_lifecycle_
                    ->hasDirtyProject());
            EXPECT_EQ(
                viewer.project_lifecycle_
                    ->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::NeedsPrompt);
            viewer.project_lifecycle_
                ->resetCloseSaveAttempt();
        }
    }

    TEST_F(VisualizerImplResetTest,
           ProjectGetInfoSucceedsDuringUnboundTrainingWithoutCkpt) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "unbound_train.licht";
        write_empty_project(project_path);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model =
                scene.addSplatPlaceholder(
                    "Unbound training model");
            ASSERT_NE(model, lfs::core::NULL_NODE);
            scene.setTrainingModelNode(model);

            auto query = viewer.projectGetInfo();
            ASSERT_TRUE(query)
                << lfs::format_for_developer(
                       query.error());
            EXPECT_TRUE(query->dirty);
            auto blocked = viewer.projectSave(false);
            ASSERT_FALSE(blocked);
            EXPECT_EQ(
                blocked.error().code(),
                lfs::ErrorCode::FailedPrecondition);

            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            ASSERT_NE(viewer.getTrainer(), nullptr);
            auto& state_machine =
                const_cast<TrainingStateMachine&>(
                    viewer.getTrainerManager()
                        ->getStateMachine());
            // setTrainer lands in Ready; Running is
            // active for isTrainingActive().
            if (state_machine.getState() ==
                TrainingState::Idle) {
                ASSERT_TRUE(state_machine.transitionTo(
                    TrainingState::Ready));
            }
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Running));
            ASSERT_TRUE(viewer.getTrainerManager()
                            ->isTrainingActive());

            auto info = viewer.projectGetInfo();
            ASSERT_TRUE(info)
                << lfs::format_for_developer(
                       info.error());
            EXPECT_TRUE(info->dirty);
            ASSERT_TRUE(info->path.has_value());
            EXPECT_EQ(*info->path, project_path);
            EXPECT_FALSE(info->project_uuid.empty());
        }
    }

    TEST_F(VisualizerImplResetTest,
           TrainingAutosaveIsLightOnlyAndRecoversSpecifiedCkpt) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "train-autosave-light.licht";
        const auto sidecar =
            lfs::io::project::autosave_sidecar_path(
                project_path);
        const auto recovered =
            temporary / "train-autosave-light-recovered.licht";
        const auto training_uuid =
            lfs::core::generate_uuid_v4();
        const auto checkpoint_uuid =
            lfs::core::generate_uuid_v4();
        write_project_with_specified_checkpoint(
            project_path, training_uuid, checkpoint_uuid);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto& state_machine =
                const_cast<TrainingStateMachine&>(
                    viewer.getTrainerManager()
                        ->getStateMachine());
            if (state_machine.getState() ==
                TrainingState::Idle) {
                ASSERT_TRUE(state_machine.transitionTo(
                    TrainingState::Ready));
            }
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Running));
            ASSERT_TRUE(viewer.getTrainerManager()
                            ->isTrainingActive());

            auto* const trainer = viewer.getTrainer();
            ASSERT_NE(trainer, nullptr);
            const auto before =
                trainer->get_project_snapshot_metrics();

            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Light dirty during training"),
                lfs::core::NULL_NODE);
            auto started =
                viewer.project_lifecycle_
                    ->startAutosave();
            ASSERT_TRUE(started)
                << lfs::format_for_developer(
                       started.error());
            EXPECT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    return !viewer.jobs().anyRunning(
                        JobType::ProjectWrite);
                }));
            EXPECT_FALSE(viewer.jobs().anyRunning(
                JobType::ProjectWrite));
            EXPECT_TRUE(std::filesystem::is_regular_file(
                sidecar));

            const auto after =
                trainer->get_project_snapshot_metrics();
            EXPECT_EQ(
                after.capture.completed_snapshots,
                before.capture.completed_snapshots);
            EXPECT_FALSE(after.request_pending);
            EXPECT_FALSE(after.last_completed_was_autosave);

            auto overlay =
                lfs::io::project::ProjectReader::open(
                    sidecar);
            ASSERT_TRUE(overlay)
                << lfs::format_for_developer(
                       overlay.error());
            const auto* ckpt_row = overlay->find(
                lfs::io::project::FOURCC_CKPT,
                checkpoint_uuid);
            ASSERT_NE(ckpt_row, nullptr);
            EXPECT_EQ(
                ckpt_row->row_kind,
                lfs::io::project::RowKind::
                    SidecarBaseReference);
            for (const auto& row : overlay->chunks()) {
                if (row.key.fourcc ==
                    lfs::io::project::FOURCC_CKPT) {
                    EXPECT_NE(
                        row.row_kind,
                        lfs::io::project::RowKind::Live)
                        << row.key_string();
                }
            }

            auto offered =
                lfs::io::project::
                    inspect_autosave_recovery(
                        project_path);
            ASSERT_TRUE(offered)
                << lfs::format_for_developer(
                       offered.error());
            EXPECT_EQ(
                offered->disposition,
                lfs::io::project::RecoveryDisposition::
                    Offer);

            auto materialized =
                lfs::io::project::
                    materialize_recovered_project(
                        project_path, sidecar,
                        recovered);
            ASSERT_TRUE(materialized)
                << lfs::format_for_developer(
                       materialized.error());
            auto restored =
                lfs::test::licht::require_result_ptr(
                    lfs::io::project::ProjectDocument::
                        open(recovered));
            auto nodes =
                restored->scene_graph().nodes();
            ASSERT_TRUE(nodes)
                << lfs::format_for_developer(
                       nodes.error());
            EXPECT_TRUE(std::ranges::any_of(
                *nodes, [](const auto& node) {
                    return node.name ==
                           "Light dirty during training";
                }));
            const auto* recovered_ckpt =
                restored->find_checkpoint(
                    checkpoint_uuid);
            ASSERT_NE(recovered_ckpt, nullptr);
            std::optional<int> iteration;
            auto visited = recovered_ckpt->visit_stream(
                [&](std::istream& stream,
                    const std::uint64_t bytes)
                    -> lfs::Result<void> {
                    auto header =
                        lfs::core::load_checkpoint_header(
                            stream, bytes);
                    if (header) {
                        iteration = header->iteration;
                    }
                    return {};
                });
            ASSERT_TRUE(visited)
                << lfs::format_for_developer(
                       visited.error());
            ASSERT_TRUE(iteration.has_value());
            EXPECT_EQ(*iteration, 11);
        }
    }

    TEST_F(VisualizerImplResetTest,
           TrainingAutosaveWithoutSpecifiedCkptStillWritesLightChapters) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "train-autosave-omit.licht";
        const auto sidecar =
            lfs::io::project::autosave_sidecar_path(
                project_path);
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model =
                scene.addSplatPlaceholder(
                    "Unbound training model");
            ASSERT_NE(model, lfs::core::NULL_NODE);
            scene.setTrainingModelNode(model);
            ASSERT_NE(
                scene.addGroup(
                    "Light edit before first ckpt"),
                lfs::core::NULL_NODE);

            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto& state_machine =
                const_cast<TrainingStateMachine&>(
                    viewer.getTrainerManager()
                        ->getStateMachine());
            if (state_machine.getState() ==
                TrainingState::Idle) {
                ASSERT_TRUE(state_machine.transitionTo(
                    TrainingState::Ready));
            }
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Running));
            ASSERT_TRUE(viewer.getTrainerManager()
                            ->isTrainingActive());

            ASSERT_TRUE(viewer.project_lifecycle_
                            ->startAutosave());
            EXPECT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    return !viewer.jobs().anyRunning(
                        JobType::ProjectWrite);
                }));
            EXPECT_TRUE(std::filesystem::is_regular_file(
                sidecar));

            auto overlay =
                lfs::io::project::ProjectReader::open(
                    sidecar);
            ASSERT_TRUE(overlay)
                << lfs::format_for_developer(
                       overlay.error());
            for (const auto& row : overlay->chunks()) {
                EXPECT_NE(
                    row.key.fourcc,
                    lfs::io::project::FOURCC_CKPT)
                    << row.key_string();
            }

            const auto recovered =
                temporary /
                "train-autosave-omit-recovered.licht";
            ASSERT_TRUE(
                lfs::io::project::
                    materialize_recovered_project(
                        project_path, sidecar,
                        recovered));
            auto restored =
                lfs::test::licht::require_result_ptr(
                    lfs::io::project::ProjectDocument::
                        open(recovered));
            auto training =
                restored->scene_graph()
                    .training_model_uuid();
            ASSERT_TRUE(training)
                << lfs::format_for_developer(
                       training.error());
            EXPECT_FALSE(training->has_value());
            auto nodes =
                restored->scene_graph().nodes();
            ASSERT_TRUE(nodes)
                << lfs::format_for_developer(
                       nodes.error());
            EXPECT_TRUE(std::ranges::any_of(
                *nodes, [](const auto& node) {
                    return node.name ==
                           "Light edit before first ckpt";
                }));
            EXPECT_TRUE(restored->checkpoint_uuids()
                            .empty());
        }
    }

    TEST_F(VisualizerImplResetTest, ResetTrainingPreservesExplicitInitPath) {
        ViewerOptions options;
        options.show_startup_overlay = false;

        const auto dataset_path = std::filesystem::temp_directory_path() / "lfs_reset_preserves_init_dataset";
        std::filesystem::create_directories(dataset_path);

        VisualizerImpl viewer(options);
        viewer.getSceneManager()->changeContentType(SceneManager::ContentType::Dataset);
        viewer.getSceneManager()->setDatasetPath(dataset_path);

        lfs::core::param::TrainingParameters params;
        params.init_path = "seed_points.ply";
        viewer.getDataLoader()->setParameters(params);

        lfs::core::events::cmd::ResetTraining{}.emit();

        ASSERT_TRUE(viewer.getDataLoader()->getParameters().init_path.has_value());
        EXPECT_EQ(*viewer.getDataLoader()->getParameters().init_path, "seed_points.ply");

        std::error_code ec;
        std::filesystem::remove_all(dataset_path, ec);
    }

    TEST_F(VisualizerImplResetTest, ResetTrainingPreservesViewportCameraAfterSuccessfulReload) {
        ViewerOptions options;
        options.show_startup_overlay = false;

        const auto dataset_path = std::filesystem::temp_directory_path() / "lfs_reset_preserves_camera_dataset";
        std::error_code ec;
        std::filesystem::remove_all(dataset_path, ec);
        write_minimal_transforms_dataset(dataset_path);

        VisualizerImpl viewer(options);
        InputController controller(nullptr, viewer.getViewport());

        viewer.getSceneManager()->changeContentType(SceneManager::ContentType::Dataset);
        viewer.getSceneManager()->setDatasetPath(dataset_path);

        lfs::core::param::TrainingParameters params;
        params.dataset.data_path = dataset_path;
        viewer.getDataLoader()->setParameters(params);

        const glm::vec3 preserved_eye(2.0f, 3.0f, 4.0f);
        const glm::vec3 preserved_target(-1.0f, 0.5f, 1.5f);
        viewer.getViewport().camera.R = lfs::rendering::makeVisualizerLookAtRotation(preserved_eye, preserved_target);
        viewer.getViewport().camera.t = preserved_eye;
        viewer.getViewport().camera.pivot = preserved_target;

        viewer.getViewport().camera.home_t = glm::vec3(100.0f, 200.0f, 300.0f);
        viewer.getViewport().camera.home_pivot = glm::vec3(10.0f, 20.0f, 30.0f);
        viewer.getViewport().camera.home_R = glm::mat3(1.0f);

        const auto preserved_camera = viewer.getViewport().camera;

        lfs::core::events::cmd::ResetTraining{}.emit();

        ASSERT_NE(viewer.getTrainer(), nullptr);
        EXPECT_EQ(viewer.getSceneManager()->getScene().getAllCameras().size(), 1u);

        EXPECT_EQ(viewer.getViewport().camera.t, preserved_camera.t);
        EXPECT_EQ(viewer.getViewport().camera.pivot, preserved_camera.pivot);
        EXPECT_EQ(viewer.getViewport().camera.home_t, preserved_camera.home_t);
        EXPECT_EQ(viewer.getViewport().camera.home_pivot, preserved_camera.home_pivot);
        EXPECT_EQ(viewer.getViewport().camera.R[0], preserved_camera.R[0]);
        EXPECT_EQ(viewer.getViewport().camera.R[1], preserved_camera.R[1]);
        EXPECT_EQ(viewer.getViewport().camera.R[2], preserved_camera.R[2]);
        EXPECT_EQ(viewer.getViewport().camera.home_R[0], preserved_camera.home_R[0]);
        EXPECT_EQ(viewer.getViewport().camera.home_R[1], preserved_camera.home_R[1]);
        EXPECT_EQ(viewer.getViewport().camera.home_R[2], preserved_camera.home_R[2]);

        std::filesystem::remove_all(dataset_path, ec);
    }

    TEST_F(VisualizerImplResetTest,
           CloseSaveRoutesTrainingSnapshotToLiveDocument) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "close-train.licht";
        write_empty_project(project_path);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ASSERT_TRUE(
                lifecycle->setAutoSaveOnClose(true));

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model =
                scene.addGroup("Train model");
            scene.setTrainingModelNode(model);
            viewer.getTrainerManager()
                ->setTrainerFromCheckpoint(
                    std::make_unique<
                        lfs::training::Trainer>(
                        scene),
                    42);
            ASSERT_TRUE(
                viewer.getTrainerManager()
                    ->isTrainingActive());
            ASSERT_FALSE(
                viewer.getTrainerManager()
                    ->isPausedAtCheckpointBaseline());
            EXPECT_TRUE(lifecycle->hasDirtyProject());

            EXPECT_EQ(
                lifecycle->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::Saving);
            EXPECT_EQ(
                lifecycle->project_write_purpose_,
                project::ProjectLifecycle::
                    ProjectWritePurpose::
                        TrainingCloseSave);
            EXPECT_EQ(
                lifecycle->project_write_destination_
                    .lexically_normal(),
                project_path.lexically_normal());
            auto* const trainer = viewer.getTrainer();
            ASSERT_NE(trainer, nullptr);
            {
                std::lock_guard lock(
                    trainer->project_snapshot_mutex_);
                ASSERT_TRUE(
                    trainer->requested_project_path_);
                EXPECT_EQ(
                    trainer->requested_project_path_
                        ->lexically_normal(),
                    project_path.lexically_normal());
                EXPECT_NE(
                    trainer->requested_project_path_
                        ->filename(),
                    "project.licht");
                const auto request_id =
                    trainer
                        ->requested_project_request_id_
                        .value_or(0);
                ASSERT_NE(request_id, 0u);
                trainer->last_completed_project_request_id_ =
                    std::max(
                        trainer
                            ->last_completed_project_request_id_,
                        request_id);
                trainer->last_project_snapshot_path_ =
                    project_path;
                trainer->last_project_writer_error_
                    .clear();
                trainer->requested_project_path_
                    .reset();
                trainer->requested_project_request_id_
                    .reset();
                trainer->prestaged_project_chapters_
                    .reset();
                trainer->prestaged_project_request_id_ =
                    0;
            }
            EXPECT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    return !lifecycle
                                ->project_write_job_;
                }));
            EXPECT_EQ(
                lifecycle->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::Succeeded);
        }
    }

    TEST_F(VisualizerImplResetTest,
           TrainerOwnedSaveTargetsLiveDocumentPath) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "trainer-target.licht";
        write_empty_project(project_path);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));
            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model =
                scene.addGroup("Train model");
            scene.setTrainingModelNode(model);
            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto* const trainer = viewer.getTrainer();
            ASSERT_NE(trainer, nullptr);
            EXPECT_EQ(
                trainer->live_or_default_project_path()
                    .lexically_normal(),
                project_path.lexically_normal());
            const auto request_id =
                trainer->request_project_save();
            ASSERT_NE(request_id, 0u);
            {
                std::lock_guard lock(
                    trainer->project_snapshot_mutex_);
                ASSERT_TRUE(
                    trainer->requested_project_path_);
                EXPECT_EQ(
                    trainer->requested_project_path_
                        ->lexically_normal(),
                    project_path.lexically_normal());
                EXPECT_EQ(
                    trainer->requested_project_path_
                        ->filename(),
                    project_path.filename());
            }
        }
    }

    TEST_F(VisualizerImplResetTest,
           SaveAsRoutesThroughFinishedTrainer) {
        // A Finished-only saveAs gate (active/pending only) falls through
        // to synchronizeDocumentFromViewer and fails with
        // "needs a safe-point project snapshot".
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto destination =
            temporary / "finished-saveas.licht";
        auto splat = lfs::test::licht::make_splat(2);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ASSERT_FALSE(lifecycle->hasSourcePath());

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model =
                scene.addGroup("Train model");
            scene.setTrainingModelNode(model);
            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto* const trainer = viewer.getTrainer();
            ASSERT_NE(trainer, nullptr);
            trainer->strategy_ =
                std::make_unique<lfs::training::MCMC>(
                    *splat);
            trainer->is_paused_.store(true);

            auto& state_machine =
                const_cast<TrainingStateMachine&>(
                    viewer.getTrainerManager()
                        ->getStateMachine());
            if (state_machine.getState() ==
                TrainingState::Idle) {
                ASSERT_TRUE(state_machine.transitionTo(
                    TrainingState::Ready));
            }
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Running));
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Paused));
            ASSERT_TRUE(
                state_machine.transitionToFinished(
                    FinishReason::UserStopped));
            ASSERT_TRUE(viewer.getTrainerManager()
                            ->isFinished());
            ASSERT_TRUE(
                trainer->can_flush_project_snapshot());

            auto saved = lifecycle->saveAs(
                destination, false, true);
            ASSERT_TRUE(saved)
                << lfs::format_for_developer(
                       saved.error());
            EXPECT_EQ(
                lifecycle->project_write_purpose_,
                project::ProjectLifecycle::
                    ProjectWritePurpose::
                        TrainingExplicitSave);
            EXPECT_EQ(
                lifecycle->project_write_destination_
                    .lexically_normal(),
                destination.lexically_normal());
            {
                std::lock_guard lock(
                    trainer->project_snapshot_mutex_);
                ASSERT_TRUE(
                    trainer->requested_project_path_);
                EXPECT_EQ(
                    trainer->requested_project_path_
                        ->lexically_normal(),
                    destination.lexically_normal());
            }
        }
    }

    TEST_F(VisualizerImplResetTest,
           SaveWhilePausedTrainingRoutesThroughLiveTrainer) {
        // Treating completion_pending_ as "publishing the final
        // snapshot" rejects Save Project while paused instead of
        // starting the live trainer snapshot write.
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "paused-save.licht";
        write_empty_project(project_path);
        auto splat = lfs::test::licht::make_splat(2);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model =
                scene.addGroup("Train model");
            scene.setTrainingModelNode(model);
            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto* const trainer = viewer.getTrainer();
            ASSERT_NE(trainer, nullptr);
            trainer->strategy_ =
                std::make_unique<lfs::training::MCMC>(
                    *splat);
            trainer->is_paused_.store(true);

            auto* const trainer_manager =
                viewer.getTrainerManager();
            auto& state_machine =
                const_cast<TrainingStateMachine&>(
                    trainer_manager->getStateMachine());
            if (state_machine.getState() ==
                TrainingState::Idle) {
                ASSERT_TRUE(state_machine.transitionTo(
                    TrainingState::Ready));
            }
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Running));
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Paused));
            trainer_manager->completion_pending_.store(
                true, std::memory_order_release);
            trainer_manager->training_joined_ = false;

            auto saved = lifecycle->save(false);
            ASSERT_TRUE(saved)
                << lfs::format_for_developer(
                       saved.error());
            EXPECT_EQ(
                lifecycle->project_write_purpose_,
                project::ProjectLifecycle::
                    ProjectWritePurpose::
                        TrainingExplicitSave);
            {
                std::lock_guard lock(
                    trainer->project_snapshot_mutex_);
                ASSERT_TRUE(
                    trainer->requested_project_path_);
                EXPECT_EQ(
                    trainer->requested_project_path_
                        ->lexically_normal(),
                    project_path.lexically_normal());
            }

            trainer_manager->completion_pending_.store(
                false, std::memory_order_release);
            trainer_manager->training_joined_ = true;
        }
    }

    TEST_F(VisualizerImplResetTest,
           SaveWhileStoppingStillBlocksUntilSnapshotPublished) {
        // The stop-window save gate must still refuse Save Project
        // while the trainer is Stopping and publishing its final
        // snapshot.
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "stopping-save.licht";
        write_empty_project(project_path);
        auto splat = lfs::test::licht::make_splat(2);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model =
                scene.addGroup("Train model");
            scene.setTrainingModelNode(model);
            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto* const trainer = viewer.getTrainer();
            ASSERT_NE(trainer, nullptr);
            trainer->strategy_ =
                std::make_unique<lfs::training::MCMC>(
                    *splat);
            trainer->is_paused_.store(false);

            auto* const trainer_manager =
                viewer.getTrainerManager();
            auto& state_machine =
                const_cast<TrainingStateMachine&>(
                    trainer_manager->getStateMachine());
            if (state_machine.getState() ==
                TrainingState::Idle) {
                ASSERT_TRUE(state_machine.transitionTo(
                    TrainingState::Ready));
            }
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Running));
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Stopping));
            trainer_manager->completion_pending_.store(
                true, std::memory_order_release);
            trainer_manager->training_joined_ = false;

            auto saved = lifecycle->save(false);
            ASSERT_FALSE(saved);
            EXPECT_EQ(
                saved.error().user_message(),
                "Wait for training completion before saving.");

            trainer_manager->completion_pending_.store(
                false, std::memory_order_release);
            trainer_manager->training_joined_ = true;
        }
    }

    TEST_F(VisualizerImplResetTest,
           SaveAsWhilePausedTrainingRoutesThroughLiveTrainer) {
        // The same completion_pending_ inner gate on saveAs rejects
        // Save As during paused training instead of routing through
        // the live trainer snapshot write.
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto destination =
            temporary / "paused-saveas.licht";
        auto splat = lfs::test::licht::make_splat(2);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ASSERT_FALSE(lifecycle->hasSourcePath());

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model =
                scene.addGroup("Train model");
            scene.setTrainingModelNode(model);
            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto* const trainer = viewer.getTrainer();
            ASSERT_NE(trainer, nullptr);
            trainer->strategy_ =
                std::make_unique<lfs::training::MCMC>(
                    *splat);
            trainer->is_paused_.store(true);

            auto* const trainer_manager =
                viewer.getTrainerManager();
            auto& state_machine =
                const_cast<TrainingStateMachine&>(
                    trainer_manager->getStateMachine());
            if (state_machine.getState() ==
                TrainingState::Idle) {
                ASSERT_TRUE(state_machine.transitionTo(
                    TrainingState::Ready));
            }
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Running));
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Paused));
            trainer_manager->completion_pending_.store(
                true, std::memory_order_release);
            trainer_manager->training_joined_ = false;

            auto saved = lifecycle->saveAs(
                destination, false, true);
            ASSERT_TRUE(saved)
                << lfs::format_for_developer(
                       saved.error());
            EXPECT_EQ(
                lifecycle->project_write_purpose_,
                project::ProjectLifecycle::
                    ProjectWritePurpose::
                        TrainingExplicitSave);
            {
                std::lock_guard lock(
                    trainer->project_snapshot_mutex_);
                ASSERT_TRUE(
                    trainer->requested_project_path_);
                EXPECT_EQ(
                    trainer->requested_project_path_
                        ->lexically_normal(),
                    destination.lexically_normal());
            }

            trainer_manager->completion_pending_.store(
                false, std::memory_order_release);
            trainer_manager->training_joined_ = true;
        }
    }

    TEST_F(VisualizerImplResetTest,
           SaveAsRoutesThroughFailedTerminalSnapshotAftermath) {
        // Returning adoptCompletedTrainingSnapshot's failure from saveAs
        // after a failed terminal write dead-ends with
        // "The latest training project generation failed."
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto destination =
            temporary / "aftermath-saveas.licht";
        const auto failed_path =
            temporary / "failed-terminal.licht";
        auto splat = lfs::test::licht::make_splat(2);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ASSERT_FALSE(lifecycle->hasSourcePath());

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model =
                scene.addGroup("Train model");
            scene.setTrainingModelNode(model);
            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto* const trainer = viewer.getTrainer();
            ASSERT_NE(trainer, nullptr);
            trainer->strategy_ =
                std::make_unique<lfs::training::MCMC>(
                    *splat);
            trainer->is_paused_.store(true);

            auto& state_machine =
                const_cast<TrainingStateMachine&>(
                    viewer.getTrainerManager()
                        ->getStateMachine());
            if (state_machine.getState() ==
                TrainingState::Idle) {
                ASSERT_TRUE(state_machine.transitionTo(
                    TrainingState::Ready));
            }
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Running));
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Paused));
            ASSERT_TRUE(
                state_machine.transitionToFinished(
                    FinishReason::UserStopped));
            ASSERT_TRUE(viewer.getTrainerManager()
                            ->isFinished());

            trainer->last_project_writer_error_ =
                "P3 first-save assembly refuses implicit replacement";
            trainer->last_project_snapshot_path_ =
                failed_path;
            ASSERT_NE(
                trainer->project_snapshot_service_,
                nullptr);
            trainer->project_snapshot_service_
                ->testing_advance_completed_snapshots(
                    1);
            lifecycle
                ->adopted_training_snapshot_count_ =
                0;

            auto saved = lifecycle->saveAs(
                destination, false, true);
            ASSERT_TRUE(saved)
                << lfs::format_for_developer(
                       saved.error());
            EXPECT_EQ(
                lifecycle->project_write_purpose_,
                project::ProjectLifecycle::
                    ProjectWritePurpose::
                        TrainingExplicitSave);
            EXPECT_EQ(
                lifecycle->project_write_destination_
                    .lexically_normal(),
                destination.lexically_normal());
            {
                std::lock_guard lock(
                    trainer->project_snapshot_mutex_);
                ASSERT_TRUE(
                    trainer->requested_project_path_);
                EXPECT_EQ(
                    trainer->requested_project_path_
                        ->lexically_normal(),
                    destination.lexically_normal());
            }
        }
    }

    TEST_F(VisualizerImplResetTest,
           InfoSurvivesFailedTerminalSnapshotAftermath) {
        // Catches info() returning synchronizeDocumentFromViewer
        // FailedPrecondition ("The training model needs a
        // safe-point project snapshot") after a failed
        // terminal generation. A GROUP training node skips
        // the geometry CKPT check; a real unbound SPLAT
        // node is the shape that killed MCP save_as
        // preflight even after the adoption tolerance.
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto failed_path =
            temporary / "failed-terminal.licht";
        auto splat = lfs::test::licht::make_splat(2);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ASSERT_NE(lifecycle->document_, nullptr);
            const auto expected_uuid =
                lifecycle->document_->project_uuid()
                    .to_string();

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model = scene.addSplat(
                "Train model",
                lfs::test::licht::make_splat(2));
            ASSERT_NE(model, lfs::core::NULL_NODE);
            scene.setTrainingModelNode(model);
            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto* const trainer = viewer.getTrainer();
            ASSERT_NE(trainer, nullptr);
            trainer->strategy_ =
                std::make_unique<lfs::training::MCMC>(
                    *splat);
            trainer->is_paused_.store(true);

            auto& state_machine =
                const_cast<TrainingStateMachine&>(
                    viewer.getTrainerManager()
                        ->getStateMachine());
            if (state_machine.getState() ==
                TrainingState::Idle) {
                ASSERT_TRUE(state_machine.transitionTo(
                    TrainingState::Ready));
            }
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Running));
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Paused));
            ASSERT_TRUE(
                state_machine.transitionToFinished(
                    FinishReason::UserStopped));
            ASSERT_TRUE(viewer.getTrainerManager()
                            ->isFinished());
            ASSERT_TRUE(
                trainer->can_flush_project_snapshot());
            ASSERT_TRUE(
                lifecycle
                    ->canFlushFinishedTrainerSnapshot());

            trainer->last_project_writer_error_ =
                "P3 first-save assembly refuses implicit replacement";
            trainer->last_project_snapshot_path_ =
                failed_path;
            ASSERT_NE(
                trainer->project_snapshot_service_,
                nullptr);
            trainer->project_snapshot_service_
                ->testing_advance_completed_snapshots(
                    1);
            lifecycle
                ->adopted_training_snapshot_count_ =
                0;

            auto info = viewer.projectGetInfo();
            ASSERT_TRUE(info)
                << lfs::format_for_developer(
                       info.error());
            EXPECT_EQ(info->project_uuid, expected_uuid);
            EXPECT_EQ(
                info->path,
                lifecycle->document_->source_path());
        }
    }

    TEST_F(VisualizerImplResetTest,
           AdoptCompletedTrainingSnapshotSkipsOpenWhenCountersEqual) {
        // Equal counters mean nothing new to adopt.
        // last_path is a nonexistent file: success
        // proves ProjectDocument::open was not called.
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto missing_path =
            temporary / "never-written.licht";
        ASSERT_FALSE(std::filesystem::exists(
            missing_path));
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            viewer.getTrainerManager()->setTrainer(
                std::make_unique<
                    lfs::training::Trainer>(scene));
            auto* const trainer = viewer.getTrainer();
            ASSERT_NE(trainer, nullptr);
            trainer->last_project_snapshot_path_ =
                missing_path;
            ASSERT_EQ(
                trainer->get_project_snapshot_metrics()
                    .capture.completed_snapshots,
                lifecycle
                    ->adopted_training_snapshot_count_);

            auto adopted =
                lifecycle
                    ->adoptCompletedTrainingSnapshot();
            ASSERT_TRUE(adopted)
                << lfs::format_for_developer(
                       adopted.error());
        }
    }

    TEST_F(VisualizerImplResetTest,
           CancelExitDuringCloseSaveDoesNotClose) {
        // CancelExit while CloseSaveState::Saving must
        // drop the close latch without aborting the
        // writer. Settlement then must not close.
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "cancel-during-save.licht";
        write_empty_project(project_path);

        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Dirty before cancel-during-save"),
                lfs::core::NULL_NODE);
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ASSERT_TRUE(
                lifecycle->setAutoSaveOnClose(true));

            lfs::core::events::cmd::RequestExit{}.emit();
            ASSERT_FALSE(viewer.allowclose());
            ASSERT_EQ(
                lifecycle->close_save_state_.load(
                    std::memory_order_acquire),
                project::ProjectLifecycle::
                    CloseSaveState::Saving);

            lfs::core::events::cmd::CancelExit{}.emit();
            EXPECT_FALSE(
                lifecycle->application_close_pending_);
            EXPECT_FALSE(
                lifecycle->suppress_training_adoption_);
            EXPECT_EQ(
                lifecycle->close_save_state_.load(
                    std::memory_order_acquire),
                project::ProjectLifecycle::
                    CloseSaveState::Saving);
            EXPECT_FALSE(
                viewer.getWindowManager()->shouldClose());

            EXPECT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    return !viewer.jobs().anyRunning(
                        JobType::ProjectWrite);
                }));
            lifecycle->settleProjectWrite();
            EXPECT_FALSE(
                lifecycle->application_close_pending_);
            EXPECT_EQ(
                lifecycle->close_save_state_.load(
                    std::memory_order_acquire),
                project::ProjectLifecycle::
                    CloseSaveState::Idle);
            EXPECT_FALSE(
                viewer.getWindowManager()->shouldClose());

            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Dirt after canceled close-save"),
                lfs::core::NULL_NODE);
            viewer.getWindowManager()->requestClose();
            EXPECT_FALSE(viewer.allowclose());
            EXPECT_NE(
                lifecycle->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::Succeeded);
        }
    }

    TEST_F(VisualizerImplResetTest,
           ProjectGetInfoSucceedsDuringCloseSave) {
        // info() used to hard-fail while
        // CloseSaveState::Saving even though a normal
        // project_write_job_ already served the cache.
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ProjectInfo cached;
            cached.project_uuid = "close-save-cache";
            cached.generation = 3;
            lifecycle->cached_project_info_ = cached;
            lifecycle->close_save_state_.store(
                project::ProjectLifecycle::
                    CloseSaveState::Saving,
                std::memory_order_release);

            auto info = viewer.projectGetInfo();
            ASSERT_TRUE(info)
                << lfs::format_for_developer(
                       info.error());
            EXPECT_TRUE(info->project_write_running);
            EXPECT_EQ(info->project_uuid, "close-save-cache");
            EXPECT_EQ(info->generation, 3u);
        }
    }

    TEST_F(VisualizerImplResetTest,
           ProjectGetInfoSucceedsWithUnboundPausedTrainer) {
        // Standalone checkpoint load leaves a Paused
        // trainer at baseline. info() used to require
        // Finished+flushable before skipping the unbound
        // CKPT sync error.
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "unbound-paused.licht";
        write_empty_project(project_path);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            const auto model =
                scene.addSplatPlaceholder(
                    "Unbound training model");
            ASSERT_NE(model, lfs::core::NULL_NODE);
            scene.setTrainingModelNode(model);

            auto* const trainer_manager =
                viewer.getTrainerManager();
            trainer_manager->setTrainerFromCheckpoint(
                std::make_unique<lfs::training::Trainer>(
                    scene),
                0);
            ASSERT_TRUE(
                trainer_manager->isTrainingActive());
            ASSERT_TRUE(trainer_manager->isPaused());
            ASSERT_TRUE(
                trainer_manager
                    ->isPausedAtCheckpointBaseline());

            auto info = viewer.projectGetInfo();
            ASSERT_TRUE(info)
                << lfs::format_for_developer(
                       info.error());
            ASSERT_TRUE(info->path.has_value());
            EXPECT_EQ(*info->path, project_path);
        }
    }

    TEST_F(VisualizerImplResetTest,
           FailedSaveAsAndExitResetsCloseLatches) {
        // A failed SaveAs-and-exit used to leave
        // application_close_pending_ set, so later
        // adoption was a silent no-op.
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "failed-saveas-exit.licht";
        write_empty_project(project_path);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Dirty for failed save-as-exit"),
                lfs::core::NULL_NODE);
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ASSERT_TRUE(
                lifecycle->setAutoSaveOnClose(true));
            ASSERT_EQ(
                lifecycle->beginOrPollCloseSave(),
                project::ProjectLifecycle::
                    CloseSaveStatus::Saving);
            EXPECT_TRUE(
                lifecycle->application_close_pending_);

            viewer.completeSaveAsAndExit(
                temporary / "second-exit.licht");
            EXPECT_FALSE(
                lifecycle->application_close_pending_);
            EXPECT_FALSE(
                lifecycle->suppress_training_adoption_);
            auto adopted =
                lifecycle
                    ->adoptCompletedTrainingSnapshot();
            ASSERT_TRUE(adopted)
                << lfs::format_for_developer(
                       adopted.error());
        }
    }

    TEST_F(VisualizerImplResetTest,
           ForceExitWhileSavingDoesNotWaitForSettlement) {
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            lifecycle->close_save_state_.store(
                project::ProjectLifecycle::
                    CloseSaveState::Saving,
                std::memory_order_release);
            viewer.getGuiManager()->setForceExit(true);
            viewer.getWindowManager()->requestClose();
            EXPECT_TRUE(viewer.allowclose());
        }
    }

    TEST_F(VisualizerImplResetTest,
           ForceExitWhileStoppingArmsWatcher) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            auto* const trainer_manager =
                viewer.getTrainerManager();
            trainer_manager->setTrainer(
                std::make_unique<lfs::training::Trainer>(
                    scene));
            auto& state_machine =
                const_cast<TrainingStateMachine&>(
                    trainer_manager->getStateMachine());
            if (state_machine.getState() ==
                TrainingState::Idle) {
                ASSERT_TRUE(state_machine.transitionTo(
                    TrainingState::Ready));
            }
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Running));
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Stopping));
            trainer_manager->completion_pending_.store(
                true, std::memory_order_release);
            trainer_manager->training_joined_ = false;
            viewer.force_exit_completion_timeout_ =
                std::chrono::milliseconds(80);
            viewer.getGuiManager()->setForceExit(true);
            viewer.getWindowManager()->requestClose();
            EXPECT_FALSE(viewer.allowclose());
            EXPECT_TRUE(viewer.force_exit_watcher_armed_);
            EXPECT_FALSE(
                viewer.getWindowManager()->shouldClose());

            EXPECT_TRUE(waitUntil(
                [&] {
                    return viewer.force_exit_wait_expired_
                        .load(std::memory_order_acquire);
                },
                std::chrono::seconds(2)));
            viewer.getWindowManager()->requestClose();
            EXPECT_TRUE(viewer.allowclose());

            trainer_manager->completion_pending_.store(
                false, std::memory_order_release);
            trainer_manager->training_joined_ = true;
        }
    }

    TEST_F(VisualizerImplResetTest,
           OpenAndNewProjectClearSuppressAdoption) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "clear-suppress.licht";
        write_empty_project(project_path);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            lifecycle->setSuppressTrainingAdoption(true);
            lifecycle->markApplicationClosePending();
            ASSERT_TRUE(
                lifecycle->newProject(
                    ProjectSwitchDisposition::
                        DiscardChanges));
            EXPECT_FALSE(
                lifecycle->application_close_pending_);
            EXPECT_FALSE(
                lifecycle->suppress_training_adoption_);

            lifecycle->setSuppressTrainingAdoption(true);
            lifecycle->markApplicationClosePending();
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            EXPECT_FALSE(
                lifecycle->application_close_pending_);
            EXPECT_FALSE(
                lifecycle->suppress_training_adoption_);
        }
    }

    TEST_F(VisualizerImplResetTest,
           ExitConfirmationPendingOwnedByGuiManager) {
        // Without a live overlay enqueue, C++ must not
        // leave the pending bit stuck true.
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            auto* const gui = viewer.getGuiManager();
            ASSERT_NE(gui, nullptr);
            gui->requestExitConfirmation(false);
            EXPECT_FALSE(gui->isExitConfirmationPending());
            EXPECT_FALSE(
                lfs::python::is_exit_popup_open());
        }
    }

    TEST_F(VisualizerImplResetTest,
           StopSaveAndExitBindsUntitledDestinationBeforeStop) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto chosen =
            temporary / "chosen-stop-save.licht";
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            auto* const trainer_manager =
                viewer.getTrainerManager();
            trainer_manager->setTrainer(
                std::make_unique<lfs::training::Trainer>(
                    scene));
            auto* const trainer = viewer.getTrainer();
            ASSERT_NE(trainer, nullptr);
            auto& state_machine =
                const_cast<TrainingStateMachine&>(
                    trainer_manager->getStateMachine());
            if (state_machine.getState() ==
                TrainingState::Idle) {
                ASSERT_TRUE(state_machine.transitionTo(
                    TrainingState::Ready));
            }
            ASSERT_TRUE(state_machine.transitionTo(
                TrainingState::Running));
            ASSERT_TRUE(
                trainer_manager->isTrainingActive());
            ASSERT_FALSE(*viewer.projectHasPath());

            viewer.armStopSaveAndExit(chosen);
            EXPECT_EQ(
                viewer.pending_training_action_,
                VisualizerImpl::PendingTrainingAction::
                    CloseSave);
            ASSERT_TRUE(viewer.pending_close_save_path_);
            EXPECT_EQ(
                *viewer.pending_close_save_path_, chosen);
            EXPECT_EQ(
                trainer->live_or_default_project_path(),
                chosen);
        }
    }

    TEST_F(VisualizerImplResetTest,
           InfoDoesNotMutateDocumentUntilExplicitSync) {
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "info-no-mutate.licht";
        write_empty_project(project_path);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));

            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ASSERT_NE(lifecycle->document_, nullptr);
            ASSERT_NE(
                viewer.getScene().addGroup(
                    "Unsynced scene dirt"),
                lfs::core::NULL_NODE);

            const auto before =
                lifecycle->document_->scene_graph()
                    .to_bytes();
            for (int i = 0; i < 3; ++i) {
                const auto info =
                    viewer.projectGetInfo();
                ASSERT_TRUE(info)
                    << lfs::format_for_developer(
                           info.error());
                EXPECT_TRUE(info->dirty);
                EXPECT_EQ(
                    lifecycle->document_->scene_graph()
                        .to_bytes(),
                    before);
            }

            auto synchronized =
                lifecycle
                    ->synchronizeDocumentFromViewer();
            ASSERT_TRUE(synchronized)
                << lfs::format_for_developer(
                       synchronized.error());
            EXPECT_NE(
                lifecycle->document_->scene_graph()
                    .to_bytes(),
                before);
            const auto after = viewer.projectGetInfo();
            ASSERT_TRUE(after)
                << lfs::format_for_developer(
                       after.error());
            EXPECT_TRUE(after->dirty);
        }
    }

    TEST_F(VisualizerImplResetTest,
           BoundCheckpointIterationCacheSkipsHeaderWhenWarm) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto project_path =
            temporary / "ckpt-iter-cache.licht";
        write_empty_project(project_path);
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(
                viewer.getParameterManager()
                    ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            ASSERT_TRUE(viewer.projectOpen(
                project_path,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));

            auto& scene = viewer.getScene();
            const auto cameras =
                scene.addGroup("Train cameras");
            scene.addCamera(
                "camera.png", cameras,
                make_project_request_test_camera());
            auto* const trainer_manager =
                viewer.getTrainerManager();
            trainer_manager->setTrainerFromCheckpoint(
                std::make_unique<lfs::training::Trainer>(
                    scene),
                0);
            ASSERT_EQ(
                trainer_manager->getCurrentIteration(),
                0);

            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            ASSERT_NE(lifecycle->document_, nullptr);
            EXPECT_TRUE(
                lifecycle
                    ->isTrainingCheckpointStale());

            const auto checkpoint_uuid =
                lfs::core::generate_uuid_v4();
            auto payload =
                lfs::io::project::LazyChunkValue::
                    from_owned(
                        std::vector<std::byte>(
                            16, std::byte{0}),
                        checkpoint_uuid);
            ASSERT_TRUE(payload)
                << lfs::format_for_developer(
                       payload.error());
            ASSERT_TRUE(
                lifecycle->document_->set_checkpoint(
                    checkpoint_uuid,
                    std::move(*payload)));
            EXPECT_TRUE(
                lifecycle
                    ->isTrainingCheckpointStale());

            lifecycle
                ->cached_bound_checkpoint_iteration_ =
                0;
            EXPECT_FALSE(
                lifecycle
                    ->isTrainingCheckpointStale());
            EXPECT_FALSE(
                lifecycle
                    ->isTrainingCheckpointStale());

            lifecycle
                ->cached_bound_checkpoint_iteration_
                .reset();
            EXPECT_TRUE(
                lifecycle
                    ->isTrainingCheckpointStale());
        }
    }

    TEST_F(VisualizerImplResetTest,
           SelectedGaussiansAndSelectionToolSurviveSaveAndReopen) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto destination =
            temporary_.path / "selection-tool-reopen.licht";
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr, viewer.getViewport());

            auto& scene = viewer.getScene();
            const auto splat = scene.addSplat(
                "Selected splat",
                lfs::test::licht::make_splat(4));
            ASSERT_NE(splat, lfs::core::NULL_NODE);
            auto mask = lfs::core::Tensor::from_vector(
                            std::vector<int>{1, 0, 1, 0},
                            lfs::core::TensorShape{4},
                            lfs::core::Device::CPU)
                            .to(lfs::core::DataType::UInt8);
            scene.setSelectionMask(
                std::make_shared<lfs::core::Tensor>(
                    std::move(mask)));
            viewer.getSceneManager()->selectNode(splat);
            viewer.getEditorContext().update(
                viewer.getSceneManager(),
                viewer.getTrainerManager());
            lfs::core::events::tools::SetToolbarTool{
                .tool_mode = static_cast<int>(
                    ToolType::Selection)}
                .emit();
            lfs::core::events::tools::SetSelectionSubMode{
                .selection_mode = static_cast<int>(
                    SelectionSubMode::Rectangle)}
                .emit();
            ASSERT_EQ(
                viewer.getEditorContext().getActiveTool(),
                ToolType::Selection);

            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);
            auto saved = lifecycle->saveAs(
                destination, false, true);
            ASSERT_TRUE(saved)
                << lfs::format_for_developer(
                       saved.error());
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_, [&] {
                    return !viewer.jobs().anyRunning(
                        JobType::ProjectWrite);
                }));
            ASSERT_TRUE(std::filesystem::is_regular_file(
                destination));
        }
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            ASSERT_TRUE(viewer.getWindowManager()->init());
            auto opened = viewer.projectOpen(
                destination,
                ProjectSwitchDisposition::DiscardChanges);
            ASSERT_TRUE(opened)
                << lfs::format_for_developer(
                       opened.error());
            viewer.noteGuiSessionRestoreOwnerReady(1);
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_, [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete" &&
                           viewer.getEditorContext()
                                   .getActiveTool() ==
                               ToolType::Selection;
                }));

            EXPECT_EQ(
                viewer.getSceneManager()
                    ->getSelectedNodeName(),
                "Selected splat");
            const auto restored_mask =
                viewer.getScene().getSelectionMask();
            ASSERT_NE(restored_mask, nullptr);
            const auto cpu = restored_mask->cpu()
                                 .to(lfs::core::DataType::UInt8)
                                 .contiguous();
            const auto* begin =
                cpu.ptr<std::uint8_t>();
            EXPECT_EQ(
                (std::vector<std::uint8_t>{
                    begin, begin + cpu.numel()}),
                (std::vector<std::uint8_t>{
                    1, 0, 1, 0}));
            EXPECT_EQ(
                UnifiedToolRegistry::instance()
                    .getActiveTool(),
                "builtin.select");
            EXPECT_EQ(
                viewer.getGuiManager()->gizmo().getSelectionSubMode(),
                SelectionSubMode::Rectangle);
        }
    }

    TEST_F(VisualizerImplResetTest,
           DatasetProjectWithoutCheckpointReloadsTrainer) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto project_path =
            temporary_.path /
            "dataset-without-checkpoint.licht";
        const auto dataset_path =
            temporary_.path /
            "dataset-without-checkpoint-source";
        write_minimal_transforms_dataset(dataset_path);
        write_dataset_project_without_checkpoint(
            project_path, dataset_path);

        auto options = projectOptions();
        VisualizerImpl viewer(options);
        ASSERT_TRUE(viewer.getParameterManager()
                        ->ensureLoaded());
        ASSERT_TRUE(viewer.getWindowManager()->init());
        auto opened = viewer.projectOpen(
            project_path,
            ProjectSwitchDisposition::DiscardChanges);
        ASSERT_TRUE(opened)
            << lfs::format_for_developer(
                   opened.error());
        viewer.noteGuiSessionRestoreOwnerReady(1);
        ASSERT_TRUE(pumpUntil(
            viewer.work_queue_mutex_,
            viewer.work_queue_, [&] {
                viewer.getGuiManager()
                    ->asyncTasks()
                    .pollImportCompletion();
                const auto info = viewer.projectGetInfo();
                return info &&
                       info->hydration_state ==
                           "complete" &&
                       viewer.getTrainerManager()
                           ->hasTrainer() &&
                       !viewer.jobs().anyRunning(
                           JobType::Import);
            }));

        EXPECT_TRUE(
            viewer.getSceneManager()->hasDataset());
        EXPECT_EQ(
            viewer.getSceneManager()->getDatasetPath(),
            dataset_path);
        ASSERT_NE(viewer.getTrainer(), nullptr);
        EXPECT_EQ(
            viewer.getTrainer()->getParams().optimization.iterations,
            1234);
    }

    TEST_F(VisualizerImplResetTest,
           DatasetProjectWithoutReferenceIsNotRecoveredFromContainingDirectory) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto dataset_path =
            temporary_.path /
            "dataset-without-reference";
        write_minimal_transforms_dataset(dataset_path);
        const auto project_path =
            dataset_path / "project.licht";
        write_dataset_project_without_checkpoint(
            project_path, dataset_path, false);

        auto options = projectOptions();
        VisualizerImpl viewer(options);
        ASSERT_TRUE(viewer.getParameterManager()
                        ->ensureLoaded());
        ASSERT_TRUE(viewer.getWindowManager()->init());
        auto opened = viewer.projectOpen(
            project_path,
            ProjectSwitchDisposition::DiscardChanges);
        ASSERT_TRUE(opened)
            << lfs::format_for_developer(
                   opened.error());
        viewer.noteGuiSessionRestoreOwnerReady(1);
        ASSERT_TRUE(pumpUntil(
            viewer.work_queue_mutex_,
            viewer.work_queue_, [&] {
                const auto info = viewer.projectGetInfo();
                return info &&
                       info->hydration_state ==
                           "complete" &&
                       !viewer.jobs().anyRunning(
                           JobType::ProjectOpen) &&
                       !viewer.jobs().anyRunning(
                           JobType::Import);
            }));

        EXPECT_FALSE(
            viewer.getTrainerManager()->hasTrainer());
        EXPECT_TRUE(
            viewer.getSceneManager()->hasDataset());
        EXPECT_TRUE(
            viewer.getSceneManager()
                ->getDatasetPath()
                .empty());
    }

    TEST_F(VisualizerImplResetTest,
           NonDatasetProjectInsideDatasetRootIsNotReimported) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto dataset_path =
            temporary_.path / "dataset-containing-ply-project";
        write_minimal_transforms_dataset(dataset_path);
        const auto project_path =
            dataset_path / "ply-only.licht";
        write_non_dataset_project_with_stale_dataset_params(
            project_path, dataset_path);

        auto options = projectOptions();
        VisualizerImpl viewer(options);
        ASSERT_TRUE(viewer.getParameterManager()
                        ->ensureLoaded());
        ASSERT_TRUE(viewer.getWindowManager()->init());
        auto opened = viewer.projectOpen(
            project_path,
            ProjectSwitchDisposition::DiscardChanges);
        ASSERT_TRUE(opened)
            << lfs::format_for_developer(
                   opened.error());
        viewer.noteGuiSessionRestoreOwnerReady(1);
        ASSERT_TRUE(pumpUntil(
            viewer.work_queue_mutex_,
            viewer.work_queue_, [&] {
                const auto info = viewer.projectGetInfo();
                return info &&
                       info->hydration_state ==
                           "complete" &&
                       !viewer.jobs().anyRunning(
                           JobType::ProjectOpen) &&
                       !viewer.jobs().anyRunning(
                           JobType::Import);
            }));

        EXPECT_FALSE(
            viewer.getTrainerManager()->hasTrainer());
        EXPECT_FALSE(
            viewer.getSceneManager()->hasDataset());
        EXPECT_NE(
            viewer.getScene().getNode(
                "PLY-only marker"),
            nullptr);
    }

    TEST_F(VisualizerImplResetTest,
           OpeningAnotherProjectCancelsPendingDatasetRestoreImport) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto dataset_path =
            temporary_.path /
            "dataset-pending-restore-import";
        write_minimal_transforms_dataset(dataset_path);
        const auto project_a =
            temporary_.path /
            "dataset-pending-restore-a.licht";
        write_dataset_project_without_checkpoint(
            project_a, dataset_path);
        const auto project_b =
            temporary_.path /
            "ply-only-pending-restore-b.licht";
        write_non_dataset_project_with_stale_dataset_params(
            project_b, dataset_path);

        auto options = projectOptions();
        VisualizerImpl viewer(options);
        ASSERT_TRUE(viewer.getParameterManager()
                        ->ensureLoaded());
        ASSERT_TRUE(viewer.getWindowManager()->init());
        auto opened_a = viewer.projectOpen(
            project_a,
            ProjectSwitchDisposition::DiscardChanges);
        ASSERT_TRUE(opened_a)
            << lfs::format_for_developer(
                   opened_a.error());
        viewer.noteGuiSessionRestoreOwnerReady(1);
        ASSERT_TRUE(pumpUntil(
            viewer.work_queue_mutex_,
            viewer.work_queue_, [&] {
                const auto info = viewer.projectGetInfo();
                return info &&
                       info->hydration_state ==
                           "complete" &&
                       viewer.jobs().anyRunning(
                           JobType::Import);
            }));

        auto opened_b = viewer.projectOpen(
            project_b,
            ProjectSwitchDisposition::DiscardChanges);
        ASSERT_TRUE(opened_b)
            << lfs::format_for_developer(
                   opened_b.error());
        viewer.noteGuiSessionRestoreOwnerReady(1);
        ASSERT_TRUE(pumpUntil(
            viewer.work_queue_mutex_,
            viewer.work_queue_, [&] {
                viewer.getGuiManager()
                    ->asyncTasks()
                    .pollImportCompletion();
                const auto info = viewer.projectGetInfo();
                return info &&
                       info->hydration_state ==
                           "complete" &&
                       !viewer.jobs().anyRunning(
                           JobType::Import);
            }));

        EXPECT_NE(
            viewer.getScene().getNode(
                "PLY-only marker"),
            nullptr);
        EXPECT_FALSE(
            viewer.getSceneManager()->hasDataset());
        EXPECT_TRUE(
            viewer.getSceneManager()
                ->getDatasetPath()
                .empty());
        EXPECT_FALSE(
            viewer.getTrainerManager()->hasTrainer());
    }

    TEST_F(VisualizerImplResetTest,
           NonDatasetSaveDoesNotBindStaleDatasetPath) {
        const auto source_path =
            temporary_.path / "non-dataset-source.licht";
        const auto destination =
            temporary_.path / "non-dataset-saved.licht";
        const auto stale_dataset_path =
            temporary_.path / "stale-dataset";
        write_minimal_transforms_dataset(
            stale_dataset_path);
        write_empty_project(source_path);

        auto options = projectOptions();
        VisualizerImpl viewer(options);
        ASSERT_TRUE(viewer.getParameterManager()
                        ->ensureLoaded());
        auto opened = viewer.projectOpen(
            source_path,
            ProjectSwitchDisposition::DiscardChanges);
        ASSERT_TRUE(opened)
            << lfs::format_for_developer(
                   opened.error());
        ASSERT_TRUE(pumpUntil(
            viewer.work_queue_mutex_,
            viewer.work_queue_, [&] {
                const auto info = viewer.projectGetInfo();
                return info &&
                       info->hydration_state ==
                           "complete";
            }));

        viewer.getParameterManager()
            ->getDatasetConfig()
            .data_path = stale_dataset_path;
        ASSERT_FALSE(
            viewer.getSceneManager()->hasDataset());
        ASSERT_TRUE(
            viewer.projectSaveAs(destination, false));
        ASSERT_TRUE(pumpUntil(
            viewer.work_queue_mutex_,
            viewer.work_queue_, [&] {
                return !viewer.jobs().anyRunning(
                    JobType::ProjectWrite);
            }));

        auto saved =
            lfs::test::licht::require_result_ptr(
                lfs::io::project::ProjectDocument::open(
                    destination));
        const auto dataset_reference =
            saved->project().dataset_reference();
        ASSERT_TRUE(dataset_reference)
            << lfs::format_for_developer(
                   dataset_reference.error());
        EXPECT_FALSE(dataset_reference->has_value());
    }

    TEST_F(VisualizerImplResetTest,
           TrainingCheckpointReopenRestoresPausedResumableState) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto project_path =
            temporary_.path / "training-resume-reopen.licht";
        const auto dataset_path =
            temporary_.path / "training-resume-dataset";
        write_minimal_transforms_dataset(dataset_path);
        write_resumable_project_with_checkpoint(
            project_path,
            lfs::core::generate_uuid_v4(),
            lfs::core::generate_uuid_v4(),
            dataset_path);

        auto options = projectOptions();
        VisualizerImpl viewer(options);
        ASSERT_TRUE(viewer.getParameterManager()
                        ->ensureLoaded());
        ASSERT_TRUE(viewer.getWindowManager()->init());
        auto opened = viewer.projectOpen(
            project_path,
            ProjectSwitchDisposition::DiscardChanges);
        ASSERT_TRUE(opened)
            << lfs::format_for_developer(
                   opened.error());
        viewer.noteGuiSessionRestoreOwnerReady(1);
        ASSERT_TRUE(pumpUntil(
            viewer.work_queue_mutex_,
            viewer.work_queue_, [&] {
                const auto info = viewer.projectGetInfo();
                return info &&
                       info->hydration_state ==
                           "complete" &&
                       viewer.getTrainerManager()
                           ->hasTrainer();
            }));

        auto* const manager =
            viewer.getTrainerManager();
        ASSERT_NE(manager, nullptr);
        EXPECT_TRUE(manager->isPaused());
        EXPECT_TRUE(manager->canResume());
        EXPECT_EQ(manager->checkpointBaselineIteration(),
                  std::optional<int>{11});
        EXPECT_EQ(manager->getCurrentIteration(), 11);
    }

    TEST_F(VisualizerImplResetTest,
           EditModeSaveDropsFormerTrainingCheckpoint) {
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto project_path =
            temporary_.path / "training-edit-mode-save.licht";
        const auto dataset_path =
            temporary_.path / "training-edit-mode-dataset";
        write_minimal_transforms_dataset(dataset_path);
        write_resumable_project_with_checkpoint(
            project_path,
            lfs::core::generate_uuid_v4(),
            lfs::core::generate_uuid_v4(),
            dataset_path);

        auto options = projectOptions();
        VisualizerImpl viewer(options);
        ASSERT_TRUE(viewer.getParameterManager()
                        ->ensureLoaded());
        ASSERT_TRUE(viewer.getWindowManager()->init());
        viewer.input_controller_ =
            std::make_unique<InputController>(
                nullptr, viewer.getViewport());
        auto opened = viewer.projectOpen(
            project_path,
            ProjectSwitchDisposition::DiscardChanges);
        ASSERT_TRUE(opened)
            << lfs::format_for_developer(
                   opened.error());
        viewer.noteGuiSessionRestoreOwnerReady(1);
        ASSERT_TRUE(pumpUntil(
            viewer.work_queue_mutex_,
            viewer.work_queue_, [&] {
                const auto info = viewer.projectGetInfo();
                return info &&
                       info->hydration_state ==
                           "complete" &&
                       viewer.getTrainerManager()
                           ->hasTrainer();
            }));

        // The trainer writes the final generation outside ProjectDocument.
        // Keep the viewer pinned to the previous commit while advancing the
        // same master, then expose that completed generation through the
        // trainer metrics. Edit Mode must adopt it before clearing Trainer.
        {
            auto advanced =
                lfs::test::licht::require_result_ptr(
                    lfs::io::project::ProjectDocument::open(
                        project_path));
            auto advanced_options =
                lfs::test::licht::
                    deterministic_document_save_options(
                        0x76000020, 3, 4);
            advanced_options.commit.snapshot_uuid = {};
            (void)lfs::test::licht::require_result(
                advanced->save(
                    project_path,
                    advanced_options));
        }
        auto* const trainer = viewer.getTrainer();
        ASSERT_NE(trainer, nullptr);
        trainer->last_project_snapshot_path_ =
            project_path;
        trainer->last_project_writer_error_.clear();
        ASSERT_NE(trainer->project_snapshot_service_,
                  nullptr);
        trainer->project_snapshot_service_
            ->testing_advance_completed_snapshots(1);

        // A restored checkpoint is project-backed in this synthetic fixture.
        // The real training workflow is still in Dataset mode when the user
        // chooses Edit Mode, so reproduce that precondition explicitly.
        viewer.getSceneManager()->changeContentType(
            SceneManager::ContentType::Dataset);
        lfs::core::events::cmd::SwitchToEditMode{}.emit();
        EXPECT_FALSE(
            viewer.getTrainerManager()->hasTrainer());
        EXPECT_TRUE(
            viewer.getScene()
                .getTrainingModelNodeUuid()
                .is_nil());

        auto* const lifecycle =
            viewer.project_lifecycle_.get();
        ASSERT_NE(lifecycle, nullptr);
        auto saved = lifecycle->save(false);
        ASSERT_TRUE(saved)
            << lfs::format_for_developer(
                   saved.error());
        ASSERT_TRUE(pumpUntil(
            viewer.work_queue_mutex_,
            viewer.work_queue_, [&] {
                return !viewer.jobs().anyRunning(
                    JobType::ProjectWrite);
            }));

        auto saved_document =
            lfs::test::licht::require_result_ptr(
                lfs::io::project::ProjectDocument::open(
                    project_path));
        EXPECT_TRUE(saved_document->checkpoint_uuids()
                        .empty());
        const auto training_uuid =
            saved_document->scene_graph()
                .training_model_uuid();
        ASSERT_TRUE(training_uuid)
            << lfs::format_for_developer(
                   training_uuid.error());
        EXPECT_FALSE(training_uuid->has_value());
        auto nodes =
            saved_document->scene_graph().nodes();
        ASSERT_TRUE(nodes)
            << lfs::format_for_developer(
                   nodes.error());
        EXPECT_TRUE(std::ranges::any_of(
            *nodes, [](const auto& node) {
                return node.name == "Trained Model" &&
                       node.type == "splat";
            }));
    }

    TEST_F(VisualizerImplResetTest,
           ReopenedTwoSplatProjectBuildsExternalCombinedModel) {
        // Reopening a two-visible-splat .licht left the combined
        // model on plain CUDA tensors (no Vulkan-external allocator),
        // so VkSplat refused to render until one node was hidden.
        if (!cuda_device_available()) {
            GTEST_SKIP() << "CUDA device unavailable";
        }
        const auto& temporary = temporary_.path;
        const auto destination =
            temporary / "two-splat-reopen.licht";
        auto options = projectOptions();
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            viewer.input_controller_ =
                std::make_unique<InputController>(
                    nullptr,
                    viewer.getViewport());
            auto* const lifecycle =
                viewer.project_lifecycle_.get();
            ASSERT_NE(lifecycle, nullptr);

            auto& scene = viewer.getScene();
            const auto splat_a = scene.addSplat(
                "Splat A",
                lfs::test::licht::make_splat(2));
            const auto splat_b = scene.addSplat(
                "Splat B",
                lfs::test::licht::make_splat(2));
            ASSERT_NE(splat_a, lfs::core::NULL_NODE);
            ASSERT_NE(splat_b, lfs::core::NULL_NODE);

            auto saved = lifecycle->saveAs(
                destination, false, true);
            ASSERT_TRUE(saved)
                << lfs::format_for_developer(
                       saved.error());
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    return !viewer.jobs().anyRunning(
                        JobType::ProjectWrite);
                }));
            ASSERT_TRUE(
                std::filesystem::is_regular_file(
                    destination));
        }
        {
            VisualizerImpl viewer(options);
            ASSERT_TRUE(viewer.getParameterManager()
                            ->ensureLoaded());
            ASSERT_TRUE(
                viewer.getWindowManager()->init());
            ASSERT_TRUE(viewer.projectOpen(
                destination,
                ProjectSwitchDisposition::
                    DiscardChanges));
            ASSERT_TRUE(pumpUntil(
                viewer.work_queue_mutex_,
                viewer.work_queue_,
                [&] {
                    const auto info =
                        viewer.projectGetInfo();
                    return info &&
                           info->hydration_state ==
                               "complete";
                }));

            auto& scene = viewer.getScene();
            const auto* splat_a =
                scene.getNode("Splat A");
            const auto* splat_b =
                scene.getNode("Splat B");
            ASSERT_NE(splat_a, nullptr);
            ASSERT_NE(splat_b, nullptr);
            ASSERT_NE(splat_a->model, nullptr);
            ASSERT_NE(splat_b->model, nullptr);

            const auto* combined =
                viewer.getScene().getCombinedModel();
            ASSERT_NE(combined, nullptr);
            EXPECT_TRUE(
                combined->has_tensor_allocator());
        }
    }

} // namespace lfs::vis

namespace {

    TEST(TrainerProjectWriterTest,
         StaleDefaultRecoveryOnlyForAdoptIdentity) {
        using lfs::training::StaleTrainerDefaultRecovery;
        EXPECT_EQ(
            lfs::training::stale_trainer_default_recovery(
                false, true, std::nullopt),
            StaleTrainerDefaultRecovery::FailLoudly);
        EXPECT_EQ(
            lfs::training::stale_trainer_default_recovery(
                true, true, std::nullopt),
            StaleTrainerDefaultRecovery::
                RelocateAndFirstSave);
        EXPECT_EQ(
            lfs::training::stale_trainer_default_recovery(
                true,
                false,
                lfs::ErrorCode::Unsupported),
            StaleTrainerDefaultRecovery::
                RelocateAndFirstSave);
        EXPECT_EQ(
            lfs::training::stale_trainer_default_recovery(
                true,
                false,
                lfs::ErrorCode::AlreadyExists),
            StaleTrainerDefaultRecovery::FailLoudly);
        EXPECT_EQ(
            lfs::training::stale_trainer_default_recovery(
                false,
                false,
                lfs::ErrorCode::Unsupported),
            StaleTrainerDefaultRecovery::FailLoudly);
    }

    TEST(TrainerProjectWriterTest,
         RelocatesUnreadableTrainerDefaultAside) {
        lfs::test::licht::TemporaryDirectory temporary;
        const auto destination =
            temporary.path / "project.licht";
        {
            std::ofstream out(
                destination, std::ios::binary);
            ASSERT_TRUE(out);
            out << "not-a-licht-container";
        }
        ASSERT_TRUE(std::filesystem::exists(destination));
        const auto cause = lfs::make_error(lfs::ErrorInit{
            .code = lfs::ErrorCode::DataLoss,
            .domain = lfs::ErrorDomain::IO,
            .detail = "invalid magic",
            .detection = LFS_SOURCE_SITE_CURRENT(),
        });
        auto relocated = lfs::training::
            relocate_unreadable_trainer_default_project(
                destination, cause);
        ASSERT_TRUE(relocated)
            << lfs::format_for_developer(
                   relocated.error());
        EXPECT_FALSE(std::filesystem::exists(destination));
        EXPECT_TRUE(std::filesystem::exists(*relocated));
        EXPECT_EQ(
            relocated->parent_path(),
            destination.parent_path());
        const auto name = relocated->filename().string();
        const auto prefix =
            destination.filename().string() + ".corrupt-";
        ASSERT_GE(name.size(), prefix.size());
        EXPECT_EQ(name.substr(0, prefix.size()), prefix);
        auto failed = lfs::training::
            relocate_unreadable_trainer_default_project(
                destination, cause);
        ASSERT_FALSE(failed);
    }

} // namespace

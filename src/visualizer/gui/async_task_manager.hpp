/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/events.hpp"
#include "core/export.hpp"
#include "core/job_registry.hpp"
#include "core/mesh2splat.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "core/splat_simplify.hpp"
#include "io/loader.hpp"
#include "io/video/video_export_options.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <glm/glm.hpp>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <thread>
#include <vector>

namespace lfs::core {
    class SplatData;
    struct MeshData;
} // namespace lfs::core

namespace lfs::vis {
    class VisualizerImpl;
    class VisualizerImplResetTest_ImportWorkerFailureSettlesFailed_Test;

    namespace gui {

        struct VideoExportEnvironmentState;
        struct VideoExportMeshRendererState;

        class LFS_VIS_API AsyncTaskManager {
        public:
            explicit AsyncTaskManager(VisualizerImpl* viewer);
            ~AsyncTaskManager();

            void shutdown();

            void setupEvents();
            void pollImportCompletion();
            [[nodiscard]] bool hasPendingMainThreadCompletions() const;

            // Export
            void performExport(lfs::core::ExportFormat format, const std::filesystem::path& path,
                               const std::vector<std::string>& node_names, int sh_degree,
                               bool rad_flip_y = false,
                               bool rad_streamable = true,
                               int spz_version = 4,
                               bool include_provenance = true);
            [[nodiscard]] bool isExporting() const {
                return jobs_.anyRunning(JobType::Export);
            }
            [[nodiscard]] float getExportProgress() const {
                return jobProgress(export_state_.job);
            }
            [[nodiscard]] std::string getExportStage() const {
                return jobStage(export_state_.job);
            }
            [[nodiscard]] std::string getExportOutcome() const {
                return jobOutcome(export_state_.job);
            }
            [[nodiscard]] std::string getExportError() const {
                return jobError(export_state_.job);
            }
            [[nodiscard]] std::filesystem::path getExportPath() const {
                std::lock_guard lock(export_state_.mutex);
                return export_state_.path;
            }
            [[nodiscard]] lfs::core::ExportFormat getExportFormat() const {
                std::lock_guard lock(export_state_.mutex);
                return export_state_.format;
            }
            void cancelExport();

            // Import
            [[nodiscard]] bool isImporting() const {
                return jobs_.anyRunning(JobType::Import);
            }
            [[nodiscard]] bool isImportCompletionShowing() const { return import_state_.show_completion.load(); }
            [[nodiscard]] float getImportProgress() const {
                return jobProgress(import_state_.job);
            }
            [[nodiscard]] std::string getImportStage() const {
                return jobStage(import_state_.job);
            }
            [[nodiscard]] std::string getImportOutcome() const {
                return jobOutcome(import_state_.job);
            }
            [[nodiscard]] std::string getImportDatasetType() const {
                std::lock_guard lock(import_state_.mutex);
                return import_state_.dataset_type;
            }
            [[nodiscard]] std::string getImportPath() const {
                std::lock_guard lock(import_state_.mutex);
                return lfs::core::path_to_utf8(import_state_.path.filename());
            }
            [[nodiscard]] bool getImportSuccess() const {
                std::lock_guard lock(import_state_.mutex);
                return import_state_.success;
            }
            [[nodiscard]] std::string getImportError() const {
                return jobError(import_state_.job);
            }
            [[nodiscard]] size_t getImportNumImages() const {
                std::lock_guard lock(import_state_.mutex);
                return import_state_.num_images;
            }
            void setImportNumImages(size_t num_images);
            [[nodiscard]] size_t getImportNumPoints() const {
                std::lock_guard lock(import_state_.mutex);
                return import_state_.num_points;
            }
            [[nodiscard]] float getImportSecondsSinceCompletion() const {
                if (!import_state_.show_completion.load())
                    return 0.0f;
                std::lock_guard lock(import_state_.mutex);
                auto elapsed = std::chrono::steady_clock::now() - import_state_.completion_time;
                return std::chrono::duration<float>(elapsed).count();
            }
            void dismissImport();
            void cancelImport();

            // Video export
            [[nodiscard]] bool isExportingVideo() const {
                return jobs_.anyRunning(JobType::VideoExport);
            }
            [[nodiscard]] float getVideoExportProgress() const {
                return jobProgress(video_export_state_.job);
            }
            [[nodiscard]] int getVideoExportCurrentFrame() const { return video_export_state_.current_frame.load(); }
            [[nodiscard]] int getVideoExportTotalFrames() const { return video_export_state_.total_frames.load(); }
            [[nodiscard]] std::string getVideoExportStage() const {
                return jobStage(video_export_state_.job);
            }
            [[nodiscard]] std::string getVideoExportOutcome() const {
                return jobOutcome(video_export_state_.job);
            }
            [[nodiscard]] std::string getVideoExportError() const {
                return jobError(video_export_state_.job);
            }
            [[nodiscard]] std::filesystem::path getVideoExportPath() const {
                std::lock_guard lock(video_export_state_.mutex);
                return video_export_state_.path;
            }
            void cancelVideoExport();

            // Mesh to Splat conversion
            void startMesh2Splat(std::shared_ptr<lfs::core::MeshData> mesh,
                                 const std::string& source_name,
                                 const lfs::core::Mesh2SplatOptions& options);
            void pollMesh2SplatCompletion();
            [[nodiscard]] bool isMesh2SplatActive() const {
                return jobs_.anyRunning(JobType::Mesh2Splat);
            }
            [[nodiscard]] float getMesh2SplatProgress() const {
                return jobProgress(mesh2splat_state_.job);
            }
            [[nodiscard]] std::string getMesh2SplatStage() const {
                return jobStage(mesh2splat_state_.job);
            }
            [[nodiscard]] std::string getMesh2SplatOutcome() const {
                return jobOutcome(mesh2splat_state_.job);
            }
            [[nodiscard]] std::string getMesh2SplatError() const {
                return jobError(mesh2splat_state_.job);
            }
            [[nodiscard]] std::string getMesh2SplatSourceName() const {
                std::lock_guard lock(mesh2splat_state_.mutex);
                return mesh2splat_state_.source_name;
            }

            // Splat simplification
            void startSplatSimplify(const std::string& source_name,
                                    const lfs::core::SplatSimplifyOptions& options);
            void pollSplatSimplifyCompletion();
            [[nodiscard]] bool isSplatSimplifyActive() const {
                return jobs_.anyRunning(JobType::SplatSimplify);
            }
            [[nodiscard]] float getSplatSimplifyProgress() const {
                return jobProgress(splat_simplify_state_.job);
            }
            [[nodiscard]] std::string getSplatSimplifyStage() const {
                return jobStage(splat_simplify_state_.job);
            }
            [[nodiscard]] std::string getSplatSimplifyError() const {
                return jobError(splat_simplify_state_.job);
            }
            void cancelSplatSimplify();

        private:
            friend class lfs::vis::VisualizerImplResetTest_ImportWorkerFailureSettlesFailed_Test;
            struct ExportSplatSource {
                const lfs::core::SplatData* data = nullptr;
                glm::mat4 transform{1.0f};
            };

            void startAsyncExport(lfs::core::ExportFormat format, const std::filesystem::path& path,
                                  std::vector<ExportSplatSource> splats,
                                  int sh_degree,
                                  bool borrow_single_identity,
                                  std::shared_mutex* model_mutex,
                                  bool rad_flip_y,
                                  bool rad_streamable,
                                  int spz_version,
                                  lfs::core::ProvenanceStamp provenance);
            void startColmapExport(const std::filesystem::path& path);
            void startAsyncImport(const std::filesystem::path& path,
                                  const lfs::core::param::TrainingParameters& params);
            void checkAsyncImportCompletion();
            void applyLoadedDataToScene();
            void applyAutoCropToLoadedScene();
            void startVideoExport(const std::filesystem::path& path,
                                  const io::video::VideoExportOptions& options);
            void resetVideoExportEnvironmentState();
            void resetVideoExportMeshRendererState();
            void cancelImportCompletionDismiss();
            void scheduleImportCompletionDismiss();
            void publishExportFailureState(lfs::core::ExportFormat format,
                                           const std::filesystem::path& path,
                                           std::string error);
            void publishExportState();
            void publishImportOverlayState();
            void publishVideoExportOverlayState();
            void publishMesh2SplatState();
            void publishSplatSimplifyState();
            [[nodiscard]] float
            jobProgress(JobHandle handle) const;
            [[nodiscard]] std::string
            jobStage(JobHandle handle) const;
            [[nodiscard]] std::string
            jobError(JobHandle handle) const;
            [[nodiscard]] std::string
            jobOutcome(JobHandle handle) const;
            [[nodiscard]] bool
            beginJob(JobHandle& handle, JobType type,
                     std::string stage);
            void settlePendingJobs();

            VisualizerImpl* viewer_;
            JobRegistry& jobs_;

            struct ExportState {
                JobHandle job;
                lfs::core::ExportFormat format{lfs::core::ExportFormat::PLY};
                std::filesystem::path path;
                mutable std::mutex mutex;
                std::optional<std::jthread> thread;
            };
            ExportState export_state_;

            struct VideoExportState {
                JobHandle job;
                std::atomic<int> current_frame{0};
                std::atomic<int> total_frames{0};
                std::filesystem::path path;
                mutable std::mutex mutex;
                std::optional<std::jthread> thread;
            };
            VideoExportState video_export_state_;
            std::unique_ptr<VideoExportEnvironmentState> video_export_environment_state_;
            std::unique_ptr<VideoExportMeshRendererState> video_export_mesh_renderer_state_;

            struct ImportState {
                JobHandle job;
                std::atomic<bool> show_completion{false};
                std::atomic<bool> load_complete{false};
                mutable std::mutex mutex;
                std::filesystem::path path;
                std::string dataset_type;
                size_t num_images{0};
                size_t num_points{0};
                bool success{false};
                bool is_mesh{false};
                std::atomic<bool> apply_auto_crop{false};
                std::atomic<std::uint64_t> completion_generation{0};
                std::chrono::steady_clock::time_point completion_time;
                std::optional<lfs::io::LoadResult> load_result;
                lfs::core::param::TrainingParameters params;
                std::optional<std::jthread> thread;
                std::optional<std::jthread> completion_dismiss_thread;
            };
            ImportState import_state_;

            struct Mesh2SplatState {
                JobHandle job;
                std::atomic<bool> pending{false};
                mutable std::mutex mutex;
                std::string source_name;
                std::shared_ptr<lfs::core::MeshData> pending_mesh;
                lfs::core::Mesh2SplatOptions pending_options;
                std::unique_ptr<lfs::core::SplatData> result;
            };
            Mesh2SplatState mesh2splat_state_;

            void executeMesh2SplatOnGraphicsThread();
            void applyMesh2SplatResult();

            struct SplatSimplifyState {
                JobHandle job;
                std::atomic<bool> completed{false};
                std::atomic<bool> apply_pending{false};
                mutable std::mutex mutex;
                std::string source_name;
                std::string output_name;
                std::unique_ptr<lfs::core::SplatData> result;
                std::optional<std::jthread> thread;
            };
            SplatSimplifyState splat_simplify_state_;
        };

    } // namespace gui
} // namespace lfs::vis

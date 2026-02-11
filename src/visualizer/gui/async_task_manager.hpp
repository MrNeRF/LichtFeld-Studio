/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/events.hpp"
#include "core/mesh2splat.hpp"
#include "core/parameters.hpp"
#include "core/path_utils.hpp"
#include "io/loader.hpp"
#include "io/video/video_export_options.hpp"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace lfs::core {
    class SplatData;
    struct MeshData;
} // namespace lfs::core

namespace lfs::vis {
    class VisualizerImpl;

    namespace gui {

        class AsyncTaskManager {
        public:
            explicit AsyncTaskManager(VisualizerImpl* viewer);
            ~AsyncTaskManager();

            void shutdown();

            void setupEvents();
            void pollImportCompletion();

            // Export
            void performExport(lfs::core::ExportFormat format, const std::filesystem::path& path,
                               const std::vector<std::string>& node_names, int sh_degree);
            [[nodiscard]] bool isExporting() const { return export_state_.active.load(); }
            [[nodiscard]] float getExportProgress() const { return export_state_.progress.load(); }
            [[nodiscard]] std::string getExportStage() const {
                std::lock_guard lock(export_state_.mutex);
                return export_state_.stage;
            }
            [[nodiscard]] lfs::core::ExportFormat getExportFormat() const {
                std::lock_guard lock(export_state_.mutex);
                return export_state_.format;
            }
            void cancelExport();

            // Import
            [[nodiscard]] bool isImporting() const { return import_state_.active.load(); }
            [[nodiscard]] bool isImportCompletionShowing() const { return import_state_.show_completion.load(); }
            [[nodiscard]] float getImportProgress() const { return import_state_.progress.load(); }
            [[nodiscard]] std::string getImportStage() const {
                std::lock_guard lock(import_state_.mutex);
                return import_state_.stage;
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
                std::lock_guard lock(import_state_.mutex);
                return import_state_.error;
            }
            [[nodiscard]] size_t getImportNumImages() const {
                std::lock_guard lock(import_state_.mutex);
                return import_state_.num_images;
            }
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
            void dismissImport() { import_state_.show_completion.store(false); }

            // Video export
            [[nodiscard]] bool isExportingVideo() const { return video_export_state_.active.load(); }
            [[nodiscard]] float getVideoExportProgress() const { return video_export_state_.progress.load(); }
            [[nodiscard]] int getVideoExportCurrentFrame() const { return video_export_state_.current_frame.load(); }
            [[nodiscard]] int getVideoExportTotalFrames() const { return video_export_state_.total_frames.load(); }
            [[nodiscard]] std::string getVideoExportStage() const {
                std::lock_guard lock(video_export_state_.mutex);
                return video_export_state_.stage;
            }
            void cancelVideoExport();

            // Mesh to Splat conversion
            void startMesh2Splat(std::shared_ptr<lfs::core::MeshData> mesh,
                                 const std::string& source_name,
                                 const lfs::core::Mesh2SplatOptions& options);
            void pollMesh2SplatCompletion();
            [[nodiscard]] bool isMesh2SplatActive() const { return mesh2splat_state_.active.load(); }
            [[nodiscard]] float getMesh2SplatProgress() const { return mesh2splat_state_.progress.load(); }
            [[nodiscard]] std::string getMesh2SplatStage() const {
                std::lock_guard lock(mesh2splat_state_.mutex);
                return mesh2splat_state_.stage;
            }
            [[nodiscard]] std::string getMesh2SplatError() const {
                std::lock_guard lock(mesh2splat_state_.mutex);
                return mesh2splat_state_.error;
            }

        private:
            void startAsyncExport(lfs::core::ExportFormat format, const std::filesystem::path& path,
                                  std::unique_ptr<lfs::core::SplatData> data);
            void startAsyncImport(const std::filesystem::path& path,
                                  const lfs::core::param::TrainingParameters& params);
            void checkAsyncImportCompletion();
            void applyLoadedDataToScene();
            void startVideoExport(const std::filesystem::path& path,
                                  const io::video::VideoExportOptions& options);

            VisualizerImpl* viewer_;

            struct ExportState {
                std::atomic<bool> active{false};
                std::atomic<bool> cancel_requested{false};
                std::atomic<float> progress{0.0f};
                lfs::core::ExportFormat format{lfs::core::ExportFormat::PLY};
                std::string stage;
                std::string error;
                mutable std::mutex mutex;
                std::optional<std::jthread> thread;
            };
            ExportState export_state_;

            struct VideoExportState {
                std::atomic<bool> active{false};
                std::atomic<bool> cancel_requested{false};
                std::atomic<float> progress{0.0f};
                std::atomic<int> current_frame{0};
                std::atomic<int> total_frames{0};
                std::string stage;
                std::string error;
                mutable std::mutex mutex;
                std::optional<std::jthread> thread;
            };
            VideoExportState video_export_state_;

            struct ImportState {
                std::atomic<bool> active{false};
                std::atomic<bool> show_completion{false};
                std::atomic<bool> load_complete{false};
                std::atomic<float> progress{0.0f};
                mutable std::mutex mutex;
                std::filesystem::path path;
                std::string stage;
                std::string dataset_type;
                std::string error;
                size_t num_images{0};
                size_t num_points{0};
                bool success{false};
                bool is_mesh{false};
                std::chrono::steady_clock::time_point completion_time;
                std::optional<lfs::io::LoadResult> load_result;
                lfs::core::param::TrainingParameters params;
                std::optional<std::jthread> thread;
            };
            ImportState import_state_;

            struct Mesh2SplatState {
                std::atomic<bool> active{false};
                std::atomic<bool> pending{false};
                std::atomic<float> progress{0.0f};
                mutable std::mutex mutex;
                std::string stage;
                std::string error;
                std::string source_name;
                std::shared_ptr<lfs::core::MeshData> pending_mesh;
                lfs::core::Mesh2SplatOptions pending_options;
                std::unique_ptr<lfs::core::SplatData> result;
            };
            Mesh2SplatState mesh2splat_state_;

            void executeMesh2SplatOnGlThread();
            void applyMesh2SplatResult();
        };

    } // namespace gui
} // namespace lfs::vis

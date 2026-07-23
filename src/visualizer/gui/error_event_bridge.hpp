/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/error_bus.hpp"
#include "core/events.hpp"
#include "core/export.hpp"

#include <optional>

// Native events -> ErrorBus bridge (Phase 8, packet P1). Subscribes to today's
// string failure events and translates each into an lfs::ErrorNotification
// published on ErrorBus::instance(), so the failures surface natively even when
// Python is absent. Registered from GuiManager::setupEventHandlers() (before
// Python), replacing the Python notification_bridge failure handlers.
//
// The per-event translate functions are exposed so they can be unit-tested
// without a live GuiManager. Each returns std::nullopt when the event must NOT
// surface as an error (user-initiated training stop, a successful completion,
// or the disk-space case already owned by the native GuiManager handler).
namespace lfs::vis::gui {

    // Stable operation labels attached to each translated error's context frame.
    // The consumer keys its localized title off these where the (domain, code)
    // pair alone is ambiguous (e.g. IO covers dataset/export/video/save).
    namespace error_op {
        inline constexpr const char* kTrain = "train";
        inline constexpr const char* kLoadDataset = "load_dataset";
        inline constexpr const char* kLoadConfig = "load_config";
        inline constexpr const char* kExport = "export";
        inline constexpr const char* kExportVideo = "export_video";
        inline constexpr const char* kMesh2Splat = "mesh2splat";
        inline constexpr const char* kFileDrop = "open_dropped_files";
        inline constexpr const char* kCudaCheck = "cuda_check";
        inline constexpr const char* kSave = "save";
        inline constexpr const char* kRenderFrame = "render_frame";
    } // namespace error_op

    LFS_VIS_API std::optional<lfs::ErrorNotification>
    translateTrainingCompleted(const core::events::state::TrainingCompleted& e);
    LFS_VIS_API std::optional<lfs::ErrorNotification>
    translateDatasetLoadCompleted(const core::events::state::DatasetLoadCompleted& e);
    LFS_VIS_API std::optional<lfs::ErrorNotification>
    translateConfigLoadFailed(const core::events::state::ConfigLoadFailed& e);
    LFS_VIS_API std::optional<lfs::ErrorNotification>
    translateExportFailed(const core::events::state::ExportFailed& e);
    LFS_VIS_API std::optional<lfs::ErrorNotification>
    translateVideoExportFailed(const core::events::state::VideoExportFailed& e);
    LFS_VIS_API std::optional<lfs::ErrorNotification>
    translateMesh2SplatFailed(const core::events::state::Mesh2SplatFailed& e);
    LFS_VIS_API std::optional<lfs::ErrorNotification>
    translateFileDropFailed(const core::events::state::FileDropFailed& e);
    LFS_VIS_API std::optional<lfs::ErrorNotification>
    translateCudaUnavailable(const core::events::state::CudaUnavailable& e);
    LFS_VIS_API std::optional<lfs::ErrorNotification>
    translateCudaVersionUnsupported(const core::events::state::CudaVersionUnsupported& e);
    LFS_VIS_API std::optional<lfs::ErrorNotification>
    translateDiskSpaceSaveFailed(const core::events::state::DiskSpaceSaveFailed& e);

    // Reveals the durable default log file in the OS file manager when the user
    // presses Open Log. Shared by the event bridge and the frame-state modals so
    // the reveal logic lives in exactly one place. Runs on the UI thread.
    LFS_VIS_API lfs::ErrorAction openLogAction();

    // Subscribes every translate function above to EventBridge, publishing each
    // non-null result to ErrorBus::instance(). Independent of Python init.
    LFS_VIS_API void registerErrorEventBridge();

} // namespace lfs::vis::gui

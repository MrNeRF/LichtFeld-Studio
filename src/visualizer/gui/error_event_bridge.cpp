/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/error_event_bridge.hpp"

#include "core/error.hpp"
#include "core/error_bus.hpp"
#include "core/error_codes.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/source_site.hpp"
#include "gui/string_keys.hpp"

#include <algorithm>
#include <filesystem>
#include <format>
#include <utility>
#include <vector>

namespace lfs::vis::gui {

    using namespace lfs::core::events;

    namespace {

        lfs::ErrorAction dismissAction() {
            return lfs::ErrorAction{.kind = lfs::ErrorActionKind::Dismiss, .label = {}, .on_invoke = {}};
        }

        // Operation-terminal failures offer Dismiss plus Open Log (§4).
        std::vector<lfs::ErrorAction> operationFailureActions() {
            return {dismissAction(), openLogAction()};
        }

        lfs::ErrorNotification makeNotification(const lfs::ErrorCode code, const lfs::ErrorDomain domain,
                                                const lfs::Severity severity, std::string message,
                                                const char* operation, const lfs::core::SourceSite site,
                                                const lfs::ErrorSurface surface = lfs::ErrorSurface::Modal,
                                                std::vector<lfs::ErrorAction> actions =
                                                    operationFailureActions()) {
            lfs::Error base = lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = domain,
                .severity = severity,
                .retryability = lfs::Retryability::NotRetryable,
                .operation_id = lfs::OperationId{},
                .user_message = message,
                .detail = std::move(message),
                .detection = site,
                .fields = lfs::SmallFields{},
                .native = std::nullopt,
            });

            // Aggregate-init: lfs::Error has no public default ctor, so the error
            // member must be provided rather than default-constructed.
            return lfs::ErrorNotification{
                .error = std::move(base).with_context(operation, site),
                .surface = surface,
                .actions = std::move(actions),
                .operation_id = lfs::OperationId::generate(),
            };
        }

    } // namespace

    // Runs on the UI thread when the user presses Open Log; the default file sink
    // is always installed when creatable, so the path exists in normal runs.
    lfs::ErrorAction openLogAction() {
        return lfs::ErrorAction{
            .kind = lfs::ErrorActionKind::OpenLog,
            .label = {},
            .on_invoke = [](lfs::OperationId) {
                const auto path =
                    lfs::core::utf8_to_path(lfs::core::Logger::default_log_file_path());
                if (!lfs::core::reveal_in_file_manager(path)) {
                    LOG_WARN("Open Log: could not reveal '{}'",
                             lfs::core::Logger::default_log_file_path());
                }
            },
        };
    }

    std::optional<lfs::ErrorNotification>
    translateTrainingCompleted(const state::TrainingCompleted& e) {
        if (e.user_stopped) {
            return std::nullopt; // a user-initiated Stop is not a failure
        }
        if (e.success) {
            return std::nullopt; // success rides the Python success modal, not the bus
        }
        const lfs::ErrorCode code =
            e.resource_exhausted ? lfs::ErrorCode::ResourceExhausted : lfs::ErrorCode::Internal;
        std::string message = e.error.value_or("Unknown error occurred during training.");
        return makeNotification(code, lfs::ErrorDomain::Training, lfs::Severity::Error,
                                std::move(message), error_op::kTrain, LFS_SOURCE_SITE_CURRENT());
    }

    std::optional<lfs::ErrorNotification>
    translateDatasetLoadCompleted(const state::DatasetLoadCompleted& e) {
        if (e.success || !e.error.has_value()) {
            return std::nullopt;
        }
        return makeNotification(lfs::ErrorCode::DataLoss, lfs::ErrorDomain::IO, lfs::Severity::Error,
                                *e.error, error_op::kLoadDataset, LFS_SOURCE_SITE_CURRENT());
    }

    std::optional<lfs::ErrorNotification>
    translateConfigLoadFailed(const state::ConfigLoadFailed& e) {
        std::string message = std::format("Could not load '{}':\n\n{}",
                                          lfs::core::path_to_utf8(e.path.filename()), e.error);
        return makeNotification(lfs::ErrorCode::InvalidArgument, lfs::ErrorDomain::App,
                                lfs::Severity::Error, std::move(message), error_op::kLoadConfig,
                                LFS_SOURCE_SITE_CURRENT());
    }

    std::optional<lfs::ErrorNotification>
    translateExportFailed(const state::ExportFailed& e) {
        if (e.cancelled) {
            return makeNotification(lfs::ErrorCode::Cancelled, lfs::ErrorDomain::IO, lfs::Severity::Info,
                                    LOC(lichtfeld::Strings::StatusBar::EXPORT_CANCELLED),
                                    error_op::kExport, LFS_SOURCE_SITE_CURRENT(),
                                    lfs::ErrorSurface::StatusOnly, /*actions=*/{});
        }
        return makeNotification(lfs::ErrorCode::Internal, lfs::ErrorDomain::IO, lfs::Severity::Error,
                                std::format("Failed to export:\n\n{}", e.error), error_op::kExport,
                                LFS_SOURCE_SITE_CURRENT());
    }

    std::optional<lfs::ErrorNotification>
    translateVideoExportFailed(const state::VideoExportFailed& e) {
        return makeNotification(lfs::ErrorCode::Internal, lfs::ErrorDomain::IO, lfs::Severity::Error,
                                std::format("Failed to export video:\n\n{}", e.error),
                                error_op::kExportVideo, LFS_SOURCE_SITE_CURRENT());
    }

    std::optional<lfs::ErrorNotification>
    translateMesh2SplatFailed(const state::Mesh2SplatFailed& e) {
        return makeNotification(lfs::ErrorCode::Internal, lfs::ErrorDomain::Rendering,
                                lfs::Severity::Error, std::format("Conversion failed:\n\n{}", e.error),
                                error_op::kMesh2Splat, LFS_SOURCE_SITE_CURRENT());
    }

    std::optional<lfs::ErrorNotification>
    translateFileDropFailed(const state::FileDropFailed& e) {
        namespace Notif = lichtfeld::Strings::Notification;
        constexpr std::size_t MAX_DISPLAY = 5;

        const std::size_t count = e.files.size();
        const std::size_t display_count = std::min(count, MAX_DISPLAY);

        std::string file_list;
        file_list.reserve(display_count * 64);
        for (std::size_t i = 0; i < display_count; ++i) {
            const std::filesystem::path p(e.files[i]);
            const bool is_dir = std::filesystem::is_directory(p);
            file_list += std::format("  - {} ({})\n", lfs::core::path_to_utf8(p.filename()),
                                     is_dir ? LOC(Notif::DIRECTORY) : LOC(Notif::FILE));
        }
        if (count > MAX_DISPLAY) {
            file_list += std::format("  {} {}\n", LOC(Notif::AND_MORE), count - MAX_DISPLAY);
        }

        const bool single_dir = count == 1 && std::filesystem::is_directory(e.files[0]);
        const char* item_type = count == 1 ? (single_dir ? LOC(Notif::DIRECTORY) : LOC(Notif::FILE))
                                           : LOC(Notif::ITEMS);

        std::string message = std::format("{} {}:\n\n{}\n{}", LOC(Notif::DROPPED_NOT_RECOGNIZED),
                                          item_type, file_list, e.error);
        return makeNotification(lfs::ErrorCode::InvalidArgument, lfs::ErrorDomain::App,
                                lfs::Severity::Error, std::move(message), error_op::kFileDrop,
                                LFS_SOURCE_SITE_CURRENT());
    }

    std::optional<lfs::ErrorNotification>
    translateCudaUnavailable(const state::CudaUnavailable& e) {
        return makeNotification(lfs::ErrorCode::Unavailable, lfs::ErrorDomain::CUDA,
                                lfs::Severity::Error, e.message, error_op::kCudaCheck,
                                LFS_SOURCE_SITE_CURRENT(), lfs::ErrorSurface::Modal, {dismissAction()});
    }

    std::optional<lfs::ErrorNotification>
    translateCudaVersionUnsupported(const state::CudaVersionUnsupported& e) {
        std::string message = std::format(
            "Your CUDA driver version ({}.{}) is not supported.\n\n"
            "LichtFeld Studio requires CUDA {}.{} or later\n"
            "(NVIDIA driver 570+).\n\n"
            "Please update your NVIDIA driver for full functionality.",
            e.major, e.minor, e.min_major, e.min_minor);
        return makeNotification(lfs::ErrorCode::FailedPrecondition, lfs::ErrorDomain::CUDA,
                                lfs::Severity::Warning, std::move(message), error_op::kCudaCheck,
                                LFS_SOURCE_SITE_CURRENT(), lfs::ErrorSurface::Toast, /*actions=*/{});
    }

    std::optional<lfs::ErrorNotification>
    translateDiskSpaceSaveFailed(const state::DiskSpaceSaveFailed& e) {
        if (e.is_disk_space_error) {
            return std::nullopt; // the native GuiManager rich disk-space modal owns this case
        }
        std::string message =
            e.is_checkpoint
                ? std::format("Failed to save checkpoint at iteration {}:\n\n{}", e.iteration, e.error)
                : std::format("Failed to export:\n\n{}", e.error);
        return makeNotification(lfs::ErrorCode::Internal, lfs::ErrorDomain::IO, lfs::Severity::Error,
                                std::move(message), error_op::kSave, LFS_SOURCE_SITE_CURRENT());
    }

    void registerErrorEventBridge() {
        const auto publish = [](std::optional<lfs::ErrorNotification> notification) {
            if (notification.has_value()) {
                lfs::ErrorBus::instance().publish(std::move(*notification));
            }
        };

        state::TrainingCompleted::when(
            [publish](const auto& e) { publish(translateTrainingCompleted(e)); });
        state::DatasetLoadCompleted::when(
            [publish](const auto& e) { publish(translateDatasetLoadCompleted(e)); });
        state::ConfigLoadFailed::when(
            [publish](const auto& e) { publish(translateConfigLoadFailed(e)); });
        state::ExportFailed::when(
            [publish](const auto& e) { publish(translateExportFailed(e)); });
        state::VideoExportFailed::when(
            [publish](const auto& e) { publish(translateVideoExportFailed(e)); });
        state::Mesh2SplatFailed::when(
            [publish](const auto& e) { publish(translateMesh2SplatFailed(e)); });
        state::FileDropFailed::when(
            [publish](const auto& e) { publish(translateFileDropFailed(e)); });
        state::CudaUnavailable::when(
            [publish](const auto& e) { publish(translateCudaUnavailable(e)); });
        state::CudaVersionUnsupported::when(
            [publish](const auto& e) { publish(translateCudaVersionUnsupported(e)); });
        state::DiskSpaceSaveFailed::when(
            [publish](const auto& e) { publish(translateDiskSpaceSaveFailed(e)); });
    }

} // namespace lfs::vis::gui

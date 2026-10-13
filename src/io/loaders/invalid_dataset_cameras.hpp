/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/error_bus.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

namespace lfs::io {

    inline constexpr size_t kMaxInvalidDatasetCameraExamples = 5;

    [[nodiscard]] inline std::string join_invalid_dataset_camera_examples(
        const std::vector<std::string>& names) {
        const size_t shown = std::min(names.size(), kMaxInvalidDatasetCameraExamples);
        std::string result;
        for (size_t i = 0; i < shown; ++i) {
            if (i != 0)
                result += ", ";
            result += names[i];
        }
        if (names.size() > shown)
            result += ", ...";
        return result;
    }

    [[nodiscard]] inline std::string format_invalid_dataset_cameras_warning(
        const std::vector<std::string>& names) {
        if (names.empty())
            return {};
        const auto& localization = lfs::event::LocalizationManager::getInstance();
        constexpr auto key = "io.dataset_camera_invalid_focal";
        if (localization.hasKey(key)) {
            return lfs::event::formatLocalized(
                key, names.size(), join_invalid_dataset_camera_examples(names));
        }
        return std::format("Skipped {} dataset camera(s) with invalid focal lengths ({}).",
                           names.size(), join_invalid_dataset_camera_examples(names));
    }

    [[nodiscard]] inline std::string format_no_valid_dataset_cameras_error() {
        const auto& localization = lfs::event::LocalizationManager::getInstance();
        constexpr auto key = "io.dataset_camera_no_valid_focal";
        return localization.hasKey(key)
                   ? lfs::event::formatLocalized(key)
                   : "No dataset cameras have valid focal lengths.";
    }

    inline void notify_invalid_dataset_cameras(const std::vector<std::string>& names) {
        if (names.empty())
            return;
        const std::string message = format_invalid_dataset_cameras_warning(names);
        LOG_WARN("{}", message);
        lfs::ErrorBus::instance().publish(lfs::ErrorNotification{
            .error = lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::InvalidArgument,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Warning,
                .retryability = lfs::Retryability::NotRetryable,
                .user_message = message,
                .detail = message,
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = lfs::SmallFields{}
                              .add("invalid_camera_count", static_cast<std::int64_t>(names.size()))
                              .add("examples", join_invalid_dataset_camera_examples(names)),
            }),
            .surface = lfs::ErrorSurface::Toast,
            .actions = {},
            .operation_id = lfs::OperationId::generate(),
        });
    }

} // namespace lfs::io

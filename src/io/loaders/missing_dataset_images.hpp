/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/error.hpp"
#include "core/error_bus.hpp"
#include "core/logger.hpp"
#include "io/error.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>
#include <vector>

namespace lfs::io {

    inline constexpr size_t kMaxMissingDatasetImageExamples = 5;

    [[nodiscard]] inline std::string join_missing_dataset_image_examples(
        const std::vector<std::string>& names,
        const size_t max_examples = kMaxMissingDatasetImageExamples) {
        const size_t shown = std::min(max_examples, names.size());
        std::string joined;
        for (size_t i = 0; i < shown; ++i) {
            if (i != 0) {
                joined += ", ";
            }
            joined += names[i];
        }
        if (names.size() > max_examples) {
            joined += ", ...";
        }
        return joined;
    }

    [[nodiscard]] inline std::string format_missing_dataset_images_warning(
        const std::vector<std::string>& names) {
        const std::string examples = join_missing_dataset_image_examples(names);
        if (names.size() == 1) {
            return std::format(
                "1 dataset image file is missing ({}). That camera stays in the scene with an "
                "empty image preview and is excluded from training.",
                examples);
        }
        return std::format(
            "{} dataset image files are missing (examples: {}). Those cameras stay in the scene "
            "with empty image previews and are excluded from training.",
            names.size(),
            examples);
    }

    [[nodiscard]] inline std::string format_all_dataset_images_missing_message(
        const std::vector<std::string>& names,
        const std::string_view location) {
        const std::string examples = join_missing_dataset_image_examples(names);
        return std::format(
            "All {} dataset image files are missing under '{}'. Training cannot start with an "
            "empty image set (examples: {}).",
            names.size(),
            location,
            examples);
    }

    inline void log_missing_dataset_images(const std::vector<std::string>& names) {
        if (names.empty()) {
            return;
        }
        LOG_WARN("{}", format_missing_dataset_images_warning(names));
        for (const auto& name : names) {
            LOG_WARN("  missing dataset image: {}", name);
        }
    }

    inline void notify_missing_dataset_images(const std::vector<std::string>& names) {
        if (names.empty()) {
            return;
        }
        const std::string summary = format_missing_dataset_images_warning(names);
        log_missing_dataset_images(names);
        lfs::ErrorBus::instance().publish(lfs::ErrorNotification{
            .error = lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::NotFound,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Warning,
                .retryability = lfs::Retryability::NotRetryable,
                .user_message = summary,
                .detail = summary,
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = lfs::SmallFields{}
                              .add("missing_count", static_cast<std::int64_t>(names.size()))
                              .add("examples", join_missing_dataset_image_examples(names)),
            }),
            .surface = lfs::ErrorSurface::Toast,
            .actions = {},
            .operation_id = lfs::OperationId::generate(),
        });
    }

    [[nodiscard]] inline Diagnostic missing_dataset_images_diagnostic(
        const std::vector<std::string>& names) {
        return Diagnostic{
            .code = lfs::ErrorCode::NotFound,
            .message = format_missing_dataset_images_warning(names),
            .fields = lfs::SmallFields{}
                          .add("missing_count", static_cast<std::int64_t>(names.size()))
                          .add("examples", join_missing_dataset_image_examples(names)),
        };
    }

} // namespace lfs::io

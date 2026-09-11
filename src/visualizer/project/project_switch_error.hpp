/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#pragma once

#include "core/error.hpp"

#include <string_view>
#include <variant>

namespace lfs::vis::project {

    // Machine-readable identities for the two project-switch preflight refusals.
    // User-facing copy may be reworded or localized; control flow matches these
    // field tags, not `user_message()`.
    inline constexpr std::string_view kProjectSwitchTrainingActiveField =
        "project.switch.training_active";
    inline constexpr std::string_view kProjectSwitchDirtyField =
        "project.switch.dirty";

    namespace detail {

        [[nodiscard]] inline std::string_view
        detectionField(const lfs::Error& error) noexcept {
            for (const auto& frame : error.frames()) {
                for (const auto& entry : frame.fields.entries()) {
                    if (entry.key != "field") {
                        continue;
                    }
                    if (const auto* value =
                            std::get_if<std::string>(&entry.value)) {
                        return *value;
                    }
                }
            }
            return {};
        }

        [[nodiscard]] inline bool isProjectSwitchError(
            const lfs::Error& error,
            const std::string_view field) noexcept {
            return error.code() == lfs::ErrorCode::FailedPrecondition &&
                   detectionField(error) == field;
        }

    } // namespace detail

    [[nodiscard]] inline bool
    isDirtyProjectSwitchError(const lfs::Error& error) noexcept {
        return detail::isProjectSwitchError(error, kProjectSwitchDirtyField);
    }

    [[nodiscard]] inline bool
    isTrainingProjectSwitchError(const lfs::Error& error) noexcept {
        return detail::isProjectSwitchError(
            error, kProjectSwitchTrainingActiveField);
    }

} // namespace lfs::vis::project

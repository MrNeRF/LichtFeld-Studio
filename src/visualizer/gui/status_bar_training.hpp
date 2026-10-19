/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "gui/string_keys.hpp"
#include "theme/theme.hpp"
#include "training/training_state.hpp"

namespace lfs::vis::gui {

    struct FinishedTrainingStatus {
        const char* label_key;
        ThemeColor color;
    };

    inline FinishedTrainingStatus finishedTrainingStatus(const FinishReason reason,
                                                         const ThemePalette& palette) {
        switch (reason) {
        case FinishReason::UserStopped:
            return {lichtfeld::Strings::Status::STOPPED, palette.text_dim};
        case FinishReason::Error:
            return {lichtfeld::Strings::Status::ERROR_STATE, palette.error};
        default:
            return {lichtfeld::Strings::Status::COMPLETE, palette.success};
        }
    }

} // namespace lfs::vis::gui

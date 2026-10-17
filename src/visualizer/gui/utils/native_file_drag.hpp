/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <filesystem>
#include <optional>
#include <span>
#include <string>

struct SDL_Window;

namespace lfs::vis::gui {
    [[nodiscard]] std::optional<bool> pointerOutsideApplication(SDL_Window* source);

    enum class FileDragStatus { Copied,
                                Cancelled,
                                Unsupported,
                                Failed };
    struct FileDragResult {
        FileDragStatus status = FileDragStatus::Failed;
        std::string error;
    };

    // Call on the UI thread, during a held left-button gesture, after export has
    // committed every file. Runs a native nested event loop. Callers must guard
    // against reentrancy and retain the files after return for deferred readers.
    // Copy only: accepting the OS drop does not prove the receiver imported it.
    [[nodiscard]] FileDragResult dragReadyFiles(std::span<const std::filesystem::path> files, SDL_Window* window = nullptr);
} // namespace lfs::vis::gui

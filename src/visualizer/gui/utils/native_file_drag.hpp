/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

struct SDL_Window;

namespace lfs::vis::gui {
    enum class FileDragTarget { Unknown,
                                Blender };
    using FileProvider = std::function<std::vector<std::filesystem::path>(FileDragTarget)>;
    // A receiver can request CF_HDROP during DragEnter. Nested application
    // frames must leave native input queued for OLE until GetData returns.
    [[nodiscard]] bool preserveNativeDragInput();
    struct FileDragResult;
    struct NativeDragPointer {
        float x = 0, y = 0;
        bool held = false;
        bool outside = false;
    };
    class NativeDragContext {
        struct Impl;
        std::unique_ptr<Impl> impl_;

    public:
        NativeDragContext();
        ~NativeDragContext();
        NativeDragContext(const NativeDragContext&) = delete;
        NativeDragContext& operator=(const NativeDragContext&) = delete;
        void pump();
        [[nodiscard]] std::optional<NativeDragPointer> pointer(SDL_Window* window);
        [[nodiscard]] FileDragResult drag(FileProvider files, SDL_Window* window);
        [[nodiscard]] bool dropPerformed() const;
    };
    [[nodiscard]] std::optional<bool> pointerOutsideApplication(SDL_Window* source);
    [[nodiscard]] bool mayShowTransferProgress(const NativeDragContext* context);

    enum class FileDragStatus { Copied,
                                Cancelled,
                                Unsupported,
                                Failed };
    struct FileDragResult {
        FileDragStatus status = FileDragStatus::Failed;
        std::string error;
    };

    // UI-thread native drag. The provider resolves once per target variant on a data request,
    // which a receiver may issue before or after release. It must return only
    // committed regular files and must not capture UI/model pointers: OLE can
    // retain the data object. Exceptions become a failed native transfer.
    [[nodiscard]] FileDragResult dragFiles(FileProvider provider, SDL_Window* window, NativeDragContext* context = nullptr);

    // Call on the UI thread, during a held left-button gesture, after export has
    // committed every file. Runs a native nested event loop. Callers must guard
    // against reentrancy and retain the files after return for deferred readers.
    // Copy only: accepting the OS drop does not prove the receiver imported it.
    [[nodiscard]] FileDragResult dragReadyFiles(std::span<const std::filesystem::path> files, SDL_Window* window = nullptr, NativeDragContext* context = nullptr);
} // namespace lfs::vis::gui

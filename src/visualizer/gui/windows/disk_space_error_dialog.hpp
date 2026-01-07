/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <filesystem>
#include <functional>
#include <string>

namespace lfs::vis::gui {

    /**
     * Dialog for handling disk space errors during checkpoint/export operations.
     * Provides options to: Retry, Change Location, or Cancel
     */
    class DiskSpaceErrorDialog {
    public:
        using RetryCallback = std::function<void()>;
        using ChangeLocationCallback = std::function<void(const std::filesystem::path&)>;
        using CancelCallback = std::function<void()>;

        struct ErrorInfo {
            std::filesystem::path path;
            std::string error_message;
            size_t required_bytes;
            size_t available_bytes;
            int iteration; // For checkpoint saves
            bool is_checkpoint;
        };

        void show(const ErrorInfo& info,
                  RetryCallback on_retry,
                  ChangeLocationCallback on_change_location,
                  CancelCallback on_cancel = nullptr);

        void render();
        [[nodiscard]] bool isOpen() const { return open_; }

    private:
        bool open_ = false;
        bool pending_open_ = false;
        ErrorInfo info_;
        RetryCallback on_retry_;
        ChangeLocationCallback on_change_location_;
        CancelCallback on_cancel_;
    };

} // namespace lfs::vis::gui

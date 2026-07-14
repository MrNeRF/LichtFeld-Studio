/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <exception>
#include <functional>

namespace lfs::vis {

    class MainLoop {
    public:
        using InitCallback = std::function<bool()>;
        using UpdateCallback = std::function<void()>;
        using RenderCallback = std::function<void()>;
        using ShutdownCallback = std::function<void()>;
        using ShouldCloseCallback = std::function<bool()>;
        // Handles an exception escaping update/render. It must contain the error
        // and return; it must never rethrow, so the frame loop never aborts.
        using FrameErrorCallback = std::function<void(std::exception_ptr)>;
        // Runs after every fully successful frame, for recovery bookkeeping.
        using FrameCompletedCallback = std::function<void()>;

        void setInitCallback(InitCallback cb) { init_callback_ = cb; }
        void setUpdateCallback(UpdateCallback cb) { update_callback_ = cb; }
        void setRenderCallback(RenderCallback cb) { render_callback_ = cb; }
        void setShutdownCallback(ShutdownCallback cb) { shutdown_callback_ = cb; }
        void setShouldCloseCallback(ShouldCloseCallback cb) { should_close_callback_ = cb; }
        void setFrameErrorCallback(FrameErrorCallback cb) { frame_error_callback_ = cb; }
        void setFrameCompletedCallback(FrameCompletedCallback cb) { frame_completed_callback_ = cb; }

        void run();

    private:
        InitCallback init_callback_;
        UpdateCallback update_callback_;
        RenderCallback render_callback_;
        ShutdownCallback shutdown_callback_;
        ShouldCloseCallback should_close_callback_;
        FrameErrorCallback frame_error_callback_;
        FrameCompletedCallback frame_completed_callback_;
    };

} // namespace lfs::vis
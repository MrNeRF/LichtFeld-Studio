/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/export.hpp"

#include <exception>
#include <functional>

namespace lfs::vis {

    class LFS_VIS_API MainLoop {
    public:
        using InitCallback = std::function<bool()>;
        using UpdateCallback = std::function<void()>;
        using RenderCallback = std::function<void()>;
        using ShutdownCallback = std::function<void()>;
        using ShouldCloseCallback = std::function<bool()>;
        using InterruptCallback = std::function<void()>;
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
        void setInterruptCallback(InterruptCallback cb) { interrupt_callback_ = cb; }
        void setFrameErrorCallback(FrameErrorCallback cb) { frame_error_callback_ = cb; }
        void setFrameCompletedCallback(FrameCompletedCallback cb) { frame_completed_callback_ = cb; }
        // Invoked from a helper thread when the signal self-pipe/eventfd
        // becomes readable. Must be safe to call from a non-viewer thread
        // (wakeEventLoop / SDL_PushEvent). Never invoked from the handler.
        void setWakeCallback(std::function<void()> cb) { wake_callback_ = std::move(cb); }

        void run();

        static void installInterruptHandlers();
        static void (*interruptHandlerForTest())(int);

    private:
        InitCallback init_callback_;
        UpdateCallback update_callback_;
        RenderCallback render_callback_;
        ShutdownCallback shutdown_callback_;
        ShouldCloseCallback should_close_callback_;
        InterruptCallback interrupt_callback_;
        FrameErrorCallback frame_error_callback_;
        FrameCompletedCallback frame_completed_callback_;
        std::function<void()> wake_callback_;
    };

} // namespace lfs::vis

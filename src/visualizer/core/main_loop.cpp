/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "main_loop.hpp"
#include "core/logger.hpp"

#include <atomic>
#include <csignal>

namespace lfs::vis {

    namespace {
        std::atomic<bool> g_interrupt_requested{false};

        void signal_handler(int signal) {
            if (signal == SIGINT || signal == SIGTERM) {
                g_interrupt_requested.store(true, std::memory_order_release);
            }
        }
    } // namespace

    void MainLoop::run() {
        LOG_INFO("Main loop starting");

        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        if (init_callback_) {
            if (!init_callback_()) {
                LOG_ERROR("Initialization failed");
                return;
            }
        }

        LOG_DEBUG("Entering main render loop");

        while (true) {
            if (g_interrupt_requested.load(std::memory_order_acquire)) {
                LOG_INFO("Interrupt signal received, shutting down");
                break;
            }

            if (should_close_callback_ && should_close_callback_()) {
                LOG_DEBUG("Should close callback requested exit");
                break;
            }

            try {
                if (update_callback_) {
                    update_callback_();
                }

                if (should_close_callback_ && should_close_callback_()) {
                    LOG_DEBUG("Should close callback requested exit after update");
                    break;
                }

                if (render_callback_) {
                    render_callback_();
                }

                if (frame_completed_callback_) {
                    frame_completed_callback_();
                }
            } catch (...) {
                if (!frame_error_callback_) {
                    throw;
                }
                frame_error_callback_(std::current_exception());
            }
        }

        LOG_DEBUG("Exiting main render loop");

        if (shutdown_callback_) {
            shutdown_callback_();
        }

        LOG_INFO("Main loop ended");
    }

} // namespace lfs::vis

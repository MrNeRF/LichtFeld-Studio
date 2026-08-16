/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "main_loop.hpp"
#include "core/logger.hpp"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <thread>

#ifdef _WIN32
#include <io.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/eventfd.h>
#endif
#endif

namespace lfs::vis {

    namespace {
        std::atomic<bool> g_interrupt_requested{false};
        std::atomic<int> g_interrupt_count{0};
        std::atomic<int> g_interrupt_write_fd{-1};
        std::atomic<int> g_interrupt_read_fd{-1};

        void signal_handler(int signal) {
            if (signal != SIGINT && signal != SIGTERM) {
                return;
            }
            const int seen =
                g_interrupt_count.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (seen >= 2) {
                _exit(1);
            }
            g_interrupt_requested.store(true, std::memory_order_release);
            const int write_fd =
                g_interrupt_write_fd.load(std::memory_order_acquire);
            if (write_fd < 0) {
                return;
            }
#ifdef __linux__
            const std::uint64_t one = 1;
            (void)!::write(write_fd, &one, sizeof(one));
#elif !defined(_WIN32)
            const char byte = 1;
            (void)!::write(write_fd, &byte, 1);
#endif
        }

        void close_fd(const int fd) {
            if (fd >= 0) {
#ifndef _WIN32
                (void)::close(fd);
#endif
            }
        }

        void close_interrupt_wake_fds() {
            const int write_fd =
                g_interrupt_write_fd.exchange(-1, std::memory_order_acq_rel);
            const int read_fd =
                g_interrupt_read_fd.exchange(-1, std::memory_order_acq_rel);
            close_fd(write_fd);
            if (read_fd != write_fd) {
                close_fd(read_fd);
            }
        }

        bool create_interrupt_wake_fds() {
#ifdef __linux__
            const int fd = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
            if (fd < 0) {
                return false;
            }
            g_interrupt_write_fd.store(fd, std::memory_order_release);
            g_interrupt_read_fd.store(fd, std::memory_order_release);
            return true;
#elif !defined(_WIN32)
            int fds[2] = {-1, -1};
            if (pipe(fds) != 0) {
                return false;
            }
            (void)fcntl(fds[0], F_SETFD, FD_CLOEXEC);
            (void)fcntl(fds[1], F_SETFD, FD_CLOEXEC);
            (void)fcntl(fds[0], F_SETFL, O_NONBLOCK);
            (void)fcntl(fds[1], F_SETFL, O_NONBLOCK);
            g_interrupt_read_fd.store(fds[0], std::memory_order_release);
            g_interrupt_write_fd.store(fds[1], std::memory_order_release);
            return true;
#else
            return false;
#endif
        }

        void drain_interrupt_wake_fd(const int read_fd) {
            if (read_fd < 0) {
                return;
            }
#ifdef __linux__
            std::uint64_t value = 0;
            while (::read(read_fd, &value, sizeof(value)) > 0) {
            }
#elif !defined(_WIN32)
            char buffer[32];
            while (::read(read_fd, buffer, sizeof(buffer)) > 0) {
            }
#else
            (void)read_fd;
#endif
        }
    } // namespace

    void MainLoop::run() {
        LOG_INFO("Main loop starting");

        g_interrupt_count.store(0, std::memory_order_relaxed);
        g_interrupt_requested.store(false, std::memory_order_relaxed);
        close_interrupt_wake_fds();
        const bool have_wake = create_interrupt_wake_fds();
        const int read_fd =
            g_interrupt_read_fd.load(std::memory_order_acquire);
#ifdef _WIN32
        (void)have_wake;
        (void)read_fd;
#endif

        installInterruptHandlers();

#ifndef _WIN32
        std::jthread interrupt_wake_thread;
        if (have_wake && read_fd >= 0 && wake_callback_) {
            interrupt_wake_thread = std::jthread(
                [this, read_fd](const std::stop_token stop) {
                    while (!stop.stop_requested()) {
                        pollfd poll_fd{
                            .fd = read_fd,
                            .events = POLLIN,
                            .revents = 0,
                        };
                        const int ready = poll(&poll_fd, 1, 100);
                        if (ready < 0) {
                            if (errno == EINTR) {
                                continue;
                            }
                            break;
                        }
                        if (ready == 0 ||
                            (poll_fd.revents & POLLIN) == 0) {
                            continue;
                        }
                        drain_interrupt_wake_fd(read_fd);
                        if (wake_callback_) {
                            wake_callback_();
                        }
                    }
                });
        }
#endif

        if (init_callback_) {
            if (!init_callback_()) {
                LOG_ERROR("Initialization failed");
                return;
            }
        }

        LOG_DEBUG("Entering main render loop");

        while (true) {
            installInterruptHandlers();
            if (g_interrupt_requested.exchange(false, std::memory_order_acq_rel)) {
                LOG_INFO("Interrupt signal received, shutting down");
                if (!interrupt_callback_) {
                    break;
                }
                interrupt_callback_();
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

#ifndef _WIN32
        interrupt_wake_thread.request_stop();
        interrupt_wake_thread = {};
#endif
        std::signal(SIGINT, SIG_DFL);
        std::signal(SIGTERM, SIG_DFL);
        close_interrupt_wake_fds();

        LOG_INFO("Main loop ended");
    }

    void MainLoop::installInterruptHandlers() {
        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);
    }

    void (*MainLoop::interruptHandlerForTest())(int) {
        return &signal_handler;
    }

} // namespace lfs::vis

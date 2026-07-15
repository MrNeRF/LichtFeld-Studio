/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <atomic>
#include <chrono>
#include <csignal>
#include <stdexcept>
#include <stop_token>
#include <thread>

namespace lfs::app {

    class HeadlessRunCoordinator {
    public:
        HeadlessRunCoordinator() {
            pending_signal_ = 0;
            previous_sigint_ = std::signal(SIGINT, &HeadlessRunCoordinator::signal_handler);
            previous_sigterm_ = std::signal(SIGTERM, &HeadlessRunCoordinator::signal_handler);
            if (previous_sigint_ == SIG_ERR || previous_sigterm_ == SIG_ERR) {
                if (previous_sigint_ != SIG_ERR) {
                    std::signal(SIGINT, previous_sigint_);
                }
                if (previous_sigterm_ != SIG_ERR) {
                    std::signal(SIGTERM, previous_sigterm_);
                }
                throw std::runtime_error("Failed to install headless signal handlers");
            }
            monitor_ = std::jthread([this](const std::stop_token token) { monitor_signals(token); });
        }

        ~HeadlessRunCoordinator() {
            monitor_.request_stop();
            if (monitor_.joinable()) {
                monitor_.join();
            }
            if (previous_sigint_ != SIG_ERR) {
                std::signal(SIGINT, previous_sigint_);
            }
            if (previous_sigterm_ != SIG_ERR) {
                std::signal(SIGTERM, previous_sigterm_);
            }
        }

        HeadlessRunCoordinator(const HeadlessRunCoordinator&) = delete;
        HeadlessRunCoordinator& operator=(const HeadlessRunCoordinator&) = delete;

        [[nodiscard]] std::stop_token stop_token() const noexcept {
            return stop_source_.get_token();
        }

        [[nodiscard]] int received_signal() const noexcept {
            return received_signal_.load(std::memory_order_acquire);
        }

        [[nodiscard]] bool interrupted() const noexcept {
            return received_signal() != 0;
        }

        [[nodiscard]] int interrupted_exit_code() const noexcept {
            const int signal = received_signal();
            return signal == 0 ? 0 : 128 + signal;
        }

    private:
        using SignalHandler = void (*)(int);

        static void signal_handler(const int signal) noexcept {
            if (signal == SIGINT || signal == SIGTERM) {
                pending_signal_ = signal;
            }
        }

        void monitor_signals(const std::stop_token token) {
            while (!token.stop_requested()) {
                const int signal = pending_signal_;
                if (signal != 0) {
                    received_signal_.store(signal, std::memory_order_release);
                    stop_source_.request_stop();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        }

        inline static volatile std::sig_atomic_t pending_signal_ = 0;

        SignalHandler previous_sigint_ = SIG_DFL;
        SignalHandler previous_sigterm_ = SIG_DFL;
        std::stop_source stop_source_;
        std::atomic<int> received_signal_{0};
        std::jthread monitor_;
    };

} // namespace lfs::app

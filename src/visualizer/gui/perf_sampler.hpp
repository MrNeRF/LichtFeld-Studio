/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "core/host_metrics.hpp"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

namespace lfs::vis::gui {

    struct PerfSample {
        lfs::core::host_metrics::Sample host;
        float gpu_utilization_percent = -1.f;
        bool gpu_utilization_valid = false;
    };

    class PerfSampler {
    public:
        PerfSampler() = default;
        ~PerfSampler();

        PerfSampler(const PerfSampler&) = delete;
        PerfSampler& operator=(const PerfSampler&) = delete;

        void start();
        void stop();
        [[nodiscard]] std::shared_ptr<const PerfSample> latest() const;

    private:
        void run(std::stop_token stop_token);

        mutable std::mutex mutex_;
        std::mutex wait_mutex_;
        std::condition_variable_any cv_;
        std::shared_ptr<const PerfSample> latest_;
        std::jthread thread_;
    };

} // namespace lfs::vis::gui

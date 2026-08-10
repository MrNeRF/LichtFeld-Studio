/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "gui/perf_sampler.hpp"

#include "gui/gpu_memory_query.hpp"

namespace lfs::vis::gui {

    PerfSampler::~PerfSampler() {
        stop();
    }

    void PerfSampler::start() {
        std::scoped_lock lock(mutex_);
        if (thread_.joinable())
            return;
        thread_ = std::jthread([this](std::stop_token token) { run(token); });
    }

    void PerfSampler::stop() {
        std::jthread thread;
        {
            std::scoped_lock lock(mutex_);
            thread = std::move(thread_);
        }
        if (thread.joinable())
            thread.request_stop();
    }

    std::shared_ptr<const PerfSample> PerfSampler::latest() const {
        std::scoped_lock lock(mutex_);
        return latest_;
    }

    void PerfSampler::run(std::stop_token stop_token) {
        while (!stop_token.stop_requested()) {
            auto snapshot = std::make_shared<PerfSample>();
            snapshot->host = lfs::core::host_metrics::sample();
            snapshot->gpu_utilization_percent = queryGpuUtilization();
            snapshot->gpu_utilization_valid = snapshot->gpu_utilization_percent >= 0.f;
            {
                std::scoped_lock lock(mutex_);
                latest_ = std::move(snapshot);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }

} // namespace lfs::vis::gui

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/gpu_device_info.hpp>
#include <core/tensor_backend.hpp>
#include <core/tensor_completion.hpp>
#include "../../src/core/tensor/backend/hip/os_metrics.hpp"
#include "../../src/core/tensor/backend/hip/hip_backend_ops.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>
#ifdef LFS_HIP_METRICS_GUI
#include "gui/gpu_memory_query.hpp"
#endif
using namespace lfs::core;
int main(int argc, char** argv) try {
    if (argc < 2) throw std::runtime_error("Arguments: HIP runtime module [require-native]");
    if (!set_default_gpu_backend(GpuBackend::HIP)) throw std::runtime_error("Cannot select HIP");
    const bool required = argc > 2;
    {
        const auto identity = internal::hip_device_identity(argv[1], selected_hip_device());
        GpuDeviceInfo absent;
        LfsHipDeviceIdentity invalid{};
        internal::hip_process_budget(invalid, absent);
        if (absent.supports_process_memory_budget) throw std::runtime_error("Zero LUID matched an unrelated GPU");
        invalid = identity;
        for (auto& byte : invalid.luid) byte = 0xff;
        internal::hip_process_budget(invalid, absent);
        if (absent.supports_process_memory_budget) throw std::runtime_error("Unknown LUID matched an unrelated GPU");
        const auto before = gpu_backend_device_info(GpuBackend::HIP);
        if (!before) throw std::runtime_error("No HIP device information");
        if (required && (!before->supports_process_memory_budget || !before->process_memory_budget_bytes))
            throw std::runtime_error("Required native process budget unavailable");
        std::printf("Selected HIP GPU: %s, process used=%zu budget=%zu\n", before->name.c_str(), before->process_memory_used_bytes, before->process_memory_budget_bytes);
        std::fflush(stdout);
        float peak = -1.f;
        std::atomic<bool> load_failed{false};
        std::jthread load([&](std::stop_token stop) {
            try {
                auto input = Tensor::full({4 * 1024 * 1024}, .5f, Device::GPU);
                while (!stop.stop_requested()) {
                    auto output = input.sin();
                    TensorCompletion done; done.include(output); done.wait();
                }
            } catch (...) { load_failed = true; }
        });
        for (int i = 0; i < 7; ++i) {
            const float sample = hip_gpu_utilization();
            if (sample > 100.f || sample < -1.f) throw std::runtime_error("Invalid utilization range");
            peak = std::max(peak, sample);
            std::printf("HIP utilization sample=%g\n", sample); std::fflush(stdout);
            std::this_thread::sleep_for(std::chrono::milliseconds(550));
        }
        load.request_stop(); load.join();
        if (load_failed) throw std::runtime_error("GPU load failed");
        if (required && peak <= 0.f) throw std::runtime_error("Required native utilization did not observe GPU load");
        const auto after = gpu_backend_device_info(GpuBackend::HIP);
        if (!after || after->name != before->name || after->uuid != before->uuid) throw std::runtime_error("Metric query changed device identity");
#ifdef LFS_HIP_METRICS_GUI
        const auto ui = lfs::vis::gui::queryGpuMemory(GpuBackend::HIP);
        if (after->supports_process_memory_budget &&
            (!ui.native_process_budget || ui.native_metrics_unavailable || !ui.process_valid ||
             !ui.uses_process_budget || !ui.process_budget || ui.device_name != after->name ||
             ui.process_budget_used != ui.process_used ||
             std::string_view(lfs::vis::gui::gpuDeviceMemoryTooltipKey(ui)) != "ui.vram_hip_budget_tooltip"))
            throw std::runtime_error("GUI misrepresents the selected HIP process budget");
        if (required && lfs::vis::gui::queryGpuUtilization() < 0.f)
            throw std::runtime_error("GUI did not expose native HIP utilization");
        std::puts("PASS GUI: HIP utilization, process budget, valid source and budget tooltip");
#endif
        std::printf("HIP peak=%g, process used=%zu budget=%zu\n", peak, after->process_memory_used_bytes, after->process_memory_budget_bytes);
    }
    if (!shutdown_gpu_backend(GpuBackend::HIP)) throw std::runtime_error("Metric test leaked HIP storage");
    std::puts("PASS HIP OS metrics and exact physical adapter matching");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "HIP METRICS FAILED: %s\n", error.what()); return 1;
}

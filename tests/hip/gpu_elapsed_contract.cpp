/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include <core/gpu_elapsed.hpp>
#include <core/tensor.hpp>
#include <core/tensor_backend.hpp>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>

using namespace lfs::core;
static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

int main() try {
    {
        GpuElapsed empty(GpuBackend::HIP, 0);
        require(!empty.ready() && !gpu_backend_live(GpuBackend::HIP), "Empty timing pool initialized HIP");
    }
    require(bool(set_default_gpu_backend(GpuBackend::HIP)), "Cannot select HIP");
    {
        GpuElapsed timer(GpuBackend::HIP, 4), other(GpuBackend::HIP, 2);
        require(timer.ready() && other.ready(), "HIP timing extension is not ready");
        require(!timer.milliseconds(0, 1) && !timer.wait_event(0), "Unrecorded HIP events were accepted");
        require(!timer.mark(4, nullptr) && !timer.wait_event(4) && !timer.milliseconds(0, 4), "Out-of-range HIP event accepted");
        auto* foreign = reinterpret_cast<void*>(uintptr_t(1));
        require(!timer.mark(0, foreign) && !timer.wait_queue(foreign), "Foreign native queue accepted");
        bool rejected = false;
        try { (void)timer.mark(0, TensorExecutionTarget::default_queue(GpuBackend::CUDA)); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected && !gpu_backend_live(GpuBackend::CUDA), "Foreign backend queue was accepted or initialized");

        auto data = Tensor::empty({4 * 1024 * 1024}, Device::GPU);
        const auto start = std::chrono::steady_clock::now();
        require(timer.mark(0, nullptr) && other.mark(0, nullptr), "HIP start mark failed");
        for (int i = 0; i < 32; ++i) data.fill_(float(i));
        require(timer.mark(1, nullptr) && other.mark(1, nullptr), "HIP end mark failed");
        require(timer.wait_event(1) && other.wait_event(1), "HIP event wait failed");
        const auto elapsed = timer.milliseconds(0, 1), second = other.milliseconds(0, 1);
        const double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        require(elapsed && second && std::isfinite(*elapsed) && *elapsed > 0.f && *elapsed <= wall + 20., "HIP elapsed time is invalid or has incorrect units");
        require(!timer.milliseconds(1, 0), "Reversed HIP timestamp sequence accepted");
        require(!timer.milliseconds(0, 2), "Unwritten end timestamp accepted");
        require(timer.mark(0, nullptr) && timer.wait_event(0), "HIP event re-record failed");
        require(!timer.milliseconds(0, 1), "Re-recorded start exposed stale elapsed time");
        require(timer.mark(3, nullptr) && timer.mark(2, nullptr) && timer.wait_queue(nullptr), "HIP queue completion failed");
        require(timer.milliseconds(3, 2).has_value() && !timer.milliseconds(2, 3), "HIP timestamps incorrectly ordered by index instead of record order");
        // Deliberately unequal adjacent spans expose shifted/batched timestamps
        // that a single start/end test cannot detect.
        require(timer.wait_queue(nullptr), "Phase warmup completion failed");
        require(timer.mark(0,nullptr), "Phase start failed");
        for(int i=0;i<32;++i) data.fill_(31.f);
        require(timer.mark(1,nullptr), "Heavy phase mark failed");
        data.fill_(31.f);
        require(timer.mark(2,nullptr), "Light phase mark failed");
        for(int i=0;i<16;++i) data.fill_(31.f);
        require(timer.mark(3,nullptr) && timer.wait_event(3), "Final phase wait failed");
        const auto heavy=timer.milliseconds(0,1), light=timer.milliseconds(1,2), medium=timer.milliseconds(2,3);
        require(heavy && light && medium, "Missing adjacent phase spans");
        std::printf("HIP adjacent phases heavy=%.6f light=%.6f medium=%.6f ms\n",*heavy,*light,*medium);
        require(*heavy > 8.f * *light && *medium > 4.f * *light && *heavy > 1.25f * *medium,
                "HIP phase timestamps do not track deliberately unequal queued workloads");
        const auto values = data.cpu().to_vector();
        require(values.front() == 31.f && values.back() == 31.f, "Timing altered queued GPU work");
        data = {};
        require(!shutdown_gpu_backend(GpuBackend::HIP), "HIP context destroyed with live timing pools");
        std::printf("HIP timing observed %.6f ms (second pool %.6f ms), CPU bound %.6f ms\n", *elapsed, *second, wall);
    }
    require(bool(shutdown_gpu_backend(GpuBackend::HIP)), "HIP timing pool retained context after destruction");
    std::puts("PASS HIP GPU elapsed: native timestamps, queue admission, unset/bounds checks, re-record ordering, isolated pools and lifetime");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL HIP GPU elapsed: %s\n", error.what());
    return 1;
}

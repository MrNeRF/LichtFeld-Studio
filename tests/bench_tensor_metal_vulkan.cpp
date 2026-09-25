/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal vs Vulkan (MoltenVK) on the same Mac: identical ops through the public
// tensor API on each backend, median of synchronized runs. Built on demand:
//   cmake --build <dir> --target bench_tensor_metal_vulkan

#include "core/gpu_device_runtime.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_completion.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace lfs::core;

namespace {

    constexpr int kWarmupRuns = 5;
    constexpr int kTimedRuns = 31;

    void settle(const Tensor& tensor) {
        const Tensor* const tensors[] = {&tensor};
        TensorCompletion(tensors).wait();
    }

    double run_ms(const GpuBackend backend, const std::function<Tensor()>& run) {
        GpuBackendScope scope(backend);
        gpu_device_barrier(backend);
        const auto start = std::chrono::steady_clock::now();
        settle(run());
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    double median(std::vector<double> samples) {
        std::ranges::sort(samples);
        return samples[samples.size() / 2];
    }

    // Runs alternate between the backends, and which goes first, so clock and
    // thermal drift hit both equally.
    std::pair<double, double> median_ms(const std::function<Tensor()>& vulkan, const std::function<Tensor()>& metal) {
        std::vector<double> vulkan_samples, metal_samples;
        for (int i = 0; i < kWarmupRuns + kTimedRuns; ++i) {
            double vulkan_ms = 0.0, metal_ms = 0.0;
            if (i % 2 == 0) {
                vulkan_ms = run_ms(GpuBackend::Vulkan, vulkan);
                metal_ms = run_ms(GpuBackend::Metal, metal);
            } else {
                metal_ms = run_ms(GpuBackend::Metal, metal);
                vulkan_ms = run_ms(GpuBackend::Vulkan, vulkan);
            }
            if (i >= kWarmupRuns) {
                vulkan_samples.push_back(vulkan_ms);
                metal_samples.push_back(metal_ms);
            }
        }
        return {median(vulkan_samples), median(metal_samples)};
    }

    struct Inputs {
        Tensor host, a, b;
    };

    Inputs make_inputs(const GpuBackend backend, const size_t count) {
        GpuBackendScope scope(backend);
        Inputs inputs{.host = Tensor::rand({count}, Device::CPU) + 0.5f};
        inputs.a = inputs.host.to(Device::GPU);
        inputs.b = (inputs.host * 0.5f).to(Device::GPU);
        settle(inputs.b);
        return inputs;
    }

    void report(const char* const name, const size_t count, const double vulkan, const double metal) {
        std::printf("%-14s %10zu %12.3f %12.3f %9.2fx\n", name, count, vulkan, metal, vulkan / metal);
    }

} // namespace

int main() {
    if (!gpu_backend_available(GpuBackend::Metal) || !gpu_backend_available(GpuBackend::Vulkan)) {
        std::fprintf(stderr, "Metal and Vulkan must both be available\n");
        return 1;
    }
    {
        // Bring both GPUs' clocks up before measuring.
        const Tensor warm = Tensor::rand({size_t{1} << 24}, Device::CPU);
        for (const auto backend : {GpuBackend::Vulkan, GpuBackend::Metal}) {
            GpuBackendScope scope(backend);
            Tensor x = warm.to(Device::GPU);
            for (int i = 0; i < 200; ++i)
                x = x * 1.0001f;
            settle(x);
        }
    }
    std::printf("%-14s %10s %12s %12s %10s\n", "op", "elements", "vulkan ms", "metal ms", "speedup");
    for (const size_t count : {size_t{4096}, size_t{1} << 20, size_t{1} << 24}) {
        const Inputs vulkan = make_inputs(GpuBackend::Vulkan, count);
        const Inputs metal = make_inputs(GpuBackend::Metal, count);
        const auto compare = [&](const char* const name, const std::function<Tensor(const Inputs&)>& op) {
            const auto [vulkan_ms, metal_ms] = median_ms([&] { return op(vulkan); }, [&] { return op(metal); });
            report(name, count, vulkan_ms, metal_ms);
        };
        compare("add", [](const Inputs& in) { return in.a + in.b; });
        compare("mul_scalar", [](const Inputs& in) { return in.a * 1.5f; });
        compare("exp", [](const Inputs& in) { return in.a.exp(); });
        compare("chain4", [](const Inputs& in) { return ((in.a + in.b) * 2.0f - in.b).exp(); });
        compare("to_int32", [](const Inputs& in) { return in.a.to(DataType::Int32); });
        compare("transpose", [&](const Inputs& in) {
            return in.a.reshape({static_cast<int>(count / 64), 64}).transpose(0, 1).contiguous();
        });
        compare("sum", [](const Inputs& in) { return Tensor::full({1}, in.a.sum_scalar(), Device::CPU); });
        compare("upload", [](const Inputs& in) { return in.host.to(Device::GPU); });
        compare("download", [](const Inputs& in) { return in.a.cpu(); });
        if (count == 4096) {
            // Int32 tensor ops dispatch eagerly, so this measures per-dispatch cost;
            // Float32 scalar chains would mostly measure the lazy planner.
            compare("100_dispatches", [](const Inputs& in) {
                const Tensor step = in.b.to(DataType::Int32);
                Tensor x = in.a.to(DataType::Int32);
                for (int i = 0; i < 100; ++i)
                    x = x + step;
                return x;
            });
        }
    }
    return 0;
}

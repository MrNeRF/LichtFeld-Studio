/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

// Metal vs Vulkan (MoltenVK) on the same Mac: identical ops through the public
// tensor API on each backend, median of synchronized runs. Built on demand:
//   cmake --build <dir> --target bench_tensor_metal_vulkan

#include "core/gpu_device_runtime.hpp"
#include "core/nn/ops.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_completion.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
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

    // Layers at the sizes of the SAM2 and MoGe encoders, in both precisions.
    struct Layers {
        Tensor tokens, weight, bias, heads, image, kernel3, kernel1, strided, up, channels;
    };

    Layers make_layers(const GpuBackend backend, const DataType dtype) {
        GpuBackendScope scope(backend);
        std::mt19937 generator(7);
        const auto random = [&](const TensorShape& shape) {
            std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);
            std::vector<float> values(shape.elements());
            for (float& value : values)
                value = distribution(generator);
            return Tensor::from_vector(values, shape, Device::CPU).to(Device::GPU).to(dtype);
        };
        Layers layers{.tokens = random({4096, 1024}),
                      .weight = random({1024, 1024}),
                      .bias = random({1024}),
                      .heads = random({1, 16, 4096, 64}),
                      .image = random({1, 256, 64, 64}),
                      .kernel3 = random({256, 256, 3, 3}),
                      .kernel1 = random({256, 256, 1, 1}),
                      .strided = random({256, 64, 3, 3}),
                      .up = random({256, 128, 2, 2}),
                      .channels = random({256})};
        settle(layers.channels);
        return layers;
    }

    void run_layers() {
        namespace nn = lfs::core::nn;
        std::printf("\n%-24s %12s %12s %10s\n", "layer", "vulkan ms", "metal ms", "speedup");
        for (const auto dtype : {DataType::Float32, DataType::Float16}) {
            const Layers vulkan = make_layers(GpuBackend::Vulkan, dtype);
            const Layers metal = make_layers(GpuBackend::Metal, dtype);
            const auto compare = [&](const char* const name, const std::function<Tensor(const Layers&)>& op) {
                const auto [vulkan_ms, metal_ms] = median_ms([&] { return op(vulkan); }, [&] { return op(metal); });
                std::printf("%-20s %s %12.3f %12.3f %9.2fx\n", name, dtype == DataType::Float16 ? "f16" : "f32",
                            vulkan_ms, metal_ms, vulkan_ms / metal_ms);
            };
            compare("linear_gelu", [](const Layers& l) {
                return nn::linear(l.tokens, l.weight, &l.bias, nn::Activation::GeluErf);
            });
            compare("attention", [](const Layers& l) { return nn::attention(l.heads, l.heads, l.heads); });
            compare("layer_norm", [](const Layers& l) { return nn::layer_norm(l.tokens, l.bias, l.bias); });
            compare("softmax", [](const Layers& l) { return nn::softmax(l.tokens); });
            compare("gelu", [](const Layers& l) { return nn::gelu(l.tokens); });
            compare("conv3x3", [](const Layers& l) {
                return nn::conv2d(l.image, l.kernel3, &l.channels, {.pad_h = 1, .pad_w = 1});
            });
            compare("conv1x1", [](const Layers& l) { return nn::conv2d(l.image, l.kernel1, &l.channels, {}); });
            compare("conv3x3_s2_g4", [](const Layers& l) {
                return nn::conv2d(l.image, l.strided, &l.channels,
                                  {.stride_h = 2, .stride_w = 2, .pad_h = 1, .pad_w = 1, .groups = 4});
            });
            compare("conv_transpose", [](const Layers& l) {
                return nn::conv_transpose2d(l.image, l.up, nullptr, {.stride_h = 2, .stride_w = 2});
            });
            compare("resize_bilinear", [](const Layers& l) {
                return nn::resize2d(l.image, 128, 128, nn::ResizeMode::Bilinear, nn::CoordTransform::HalfPixel);
            });
        }
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
        // Row and column operands broadcast, as in bias and normalization layers.
        const int rows = static_cast<int>(count / 64);
        compare("add_row", [&](const Inputs& in) { return in.a.reshape({rows, 64}) + in.b.slice(0, 0, 64); });
        compare("div_column", [&](const Inputs& in) {
            return in.a.reshape({rows, 64}) / in.b.slice(0, 0, rows).reshape({rows, 1});
        });
        compare("mul_scalar", [](const Inputs& in) { return in.a * 1.5f; });
        compare("exp", [](const Inputs& in) { return in.a.exp(); });
        compare("chain4", [](const Inputs& in) { return ((in.a + in.b) * 2.0f - in.b).exp(); });
        compare("to_int32", [](const Inputs& in) { return in.a.to(DataType::Int32); });
        compare("transpose", [&](const Inputs& in) { return in.a.reshape({rows, 64}).transpose(0, 1).contiguous(); });
        compare("sum", [](const Inputs& in) { return Tensor::full({1}, in.a.sum_scalar(), Device::CPU); });
        // Square matrices of count elements: 64, 1024 and 4096 on a side.
        const int side = static_cast<int>(std::lround(std::sqrt(static_cast<double>(count))));
        compare("matmul", [&](const Inputs& in) {
            return in.a.reshape({side, side}).mm(in.b.reshape({side, side}));
        });
        // A 64 -> 64 fully connected layer with bias over count / 64 rows.
        compare("linear64", [&](const Inputs& in) {
            return in.a.reshape({rows, 64})
                .linear(in.b.slice(0, 0, 64 * 64).reshape({64, 64}), in.b.slice(0, 0, 64));
        });
        compare("sum_rows", [&](const Inputs& in) { return in.a.reshape({rows, 64}).sum(1); });
        compare("max_cols", [&](const Inputs& in) { return in.a.reshape({rows, 64}).max(0); });
        compare("cumsum", [](const Inputs& in) { return in.a.cumsum(0); });
        compare("sort", [](const Inputs& in) { return in.a.sort().first; });
        compare("argsort", [](const Inputs& in) { return in.a.sort(-1, true).second; });
        // Rows of 64 picked at random, as when gathering splat attributes.
        const Tensor host_rows = (Tensor::rand({static_cast<size_t>(rows)}, Device::CPU) * static_cast<float>(rows - 1))
                                     .to(DataType::Int64);
        compare("index_select", [&](const Inputs& in) {
            return in.a.reshape({rows, 64}).index_select(0, host_rows.to(Device::GPU));
        });
        compare("index_add", [&](const Inputs& in) {
            Tensor target = Tensor::zeros({static_cast<size_t>(rows), 64}, Device::GPU);
            return target.index_add_(0, host_rows.to(Device::GPU), in.b.reshape({rows, 64}));
        });
        compare("nonzero", [](const Inputs& in) { return (in.a > 1.2f).nonzero(); });
        compare("masked_select", [](const Inputs& in) { return in.a.masked_select(in.a > 1.2f); });
        compare("where", [](const Inputs& in) { return in.a.where(in.a > 1.0f, in.b); });
        compare("cat", [](const Inputs& in) { return Tensor::cat({in.a, in.b}, 0); });
        compare("randn", [count](const Inputs&) { return Tensor::randn({count}, Device::GPU); });
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
    run_layers();
    return 0;
}

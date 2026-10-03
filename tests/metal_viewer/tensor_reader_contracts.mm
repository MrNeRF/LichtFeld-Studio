/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "device_requirements.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_metal_reader.hpp"
#include <array>
#include <cstdio>
#include <stdexcept>
using namespace lfs::core;
static void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
static void run(GpuBackend backend) {
    std::printf("Testing native tensor access on %s storage\n", gpu_backend_name(backend));
    const GpuBackendScope scope(backend);
    MetalTensorReader reader;
    constexpr size_t count = 4097;
    for (int iteration = 0; iteration < 8; ++iteration) {
        Tensor source = Tensor::full({count + 2}, float(iteration + 1), Device::GPU);
        Tensor view = source.slice(0, 1, count + 1);
        std::array<const Tensor*, 2> inputs{&view, nullptr};
        auto snapshot = [reader.device() newBufferWithLength:count * sizeof(float) options:MTLResourceStorageModeShared];
        auto command = reader.submit(inputs, [&](id<MTLCommandBuffer> command, std::span<const MetalTensorView> views) {
            require(views.size() == 2 && !views[1].buffer, "Optional tensor view mismatch");
            require(views[0].bytes == count * sizeof(float), "Tensor slice byte count mismatch");
            auto encoder = [command blitCommandEncoder];
            [encoder copyFromBuffer:views[0].buffer
                       sourceOffset:views[0].offset
                           toBuffer:snapshot
                  destinationOffset:0
                               size:views[0].bytes];
            [encoder endEncoding];
        });
        // This writes the SAME source storage immediately after submit. The
        // consumer must see the original producer values, never this mutation.
        source.add_(10.f);
        const auto cpu = source.to(Device::CPU);
        require(cpu.ptr<float>()[1] == float(iteration + 11), "Tensor mutation did not finish");
        view = Tensor{};
        source = Tensor{};
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Native tensor consumer failed");
        const auto* values = static_cast<const float*>(snapshot.contents);
        for (size_t i = 0; i < count; ++i)
            require(values[i] == float(iteration + 1), "Producer/consumer ordering or slice offset mismatch");
    }
    for (int iteration = 0; iteration < 8; ++iteration) {
        auto source = Tensor::full({count + 2}, float(iteration + 1), Device::GPU);
        auto output = Tensor::full({count + 2}, -9.f, Device::GPU);
        auto input_slice = source.slice(0, 1, count + 1);
        auto output_slice = output.slice(0, 1, count + 1);
        const std::array<const Tensor*, 1> inputs{&input_slice};
        const std::array<Tensor*, 1> outputs{&output_slice};
        const auto command = reader.submitWrites(inputs, outputs, [&](id<MTLCommandBuffer> command, std::span<const MetalTensorView> input, std::span<const MetalTensorView> destination) {
            require(input.size() == 1 && destination.size() == 1, "Mutable native view count mismatch");
            auto blit = [command blitCommandEncoder];
            [blit copyFromBuffer:input[0].buffer sourceOffset:input[0].offset toBuffer:destination[0].buffer destinationOffset:destination[0].offset size:input[0].bytes];
            [blit endEncoding];
        });
        // No command completion wait: dependent tensor reads and source writes
        // must both be ordered after the native writer, including slice offsets.
        source.add_(20.f);
        const auto result = output.add(10.f).cpu();
        require(result.ptr<float>()[0] == 1.f && result.ptr<float>()[count + 1] == 1.f, "Native write overwrote slice borders");
        for (size_t n = 1; n <= count; ++n)
            require(result.ptr<float>()[n] == float(iteration + 11), "Native write-to-tensor read ordering differs");
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Native tensor write failed");
    }
    {
        auto output = Tensor::full({count}, -5.f, Device::GPU);
        const std::array<Tensor*, 1> outputs{&output};
        bool discarded = false;
        try {
            (void)reader.submitWrites({}, outputs, [](auto, auto, auto) { throw std::runtime_error("Write encode failure"); });
        } catch (const std::runtime_error&) { discarded = true; }
        require(discarded && output.cpu().ptr<float>()[0] == -5.f, "Unsubmitted native write changed or poisoned output");
    }
    // A dedicated allocation exercises its full allocation offset, not only
    // the ordinary small-buffer slab path used by selection masks.
    {
        constexpr size_t large_count = (17 * 1024 * 1024) / sizeof(float);
        auto large = Tensor::full({large_count}, 3.f, Device::GPU);
        const std::array<const Tensor*, 1> inputs{&large};
        auto sample = [reader.device() newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];
        const auto command = reader.submit(inputs, [&](id<MTLCommandBuffer> command, auto views) {
            auto blit = [command blitCommandEncoder];
            [blit copyFromBuffer:views[0].buffer sourceOffset:views[0].offset + views[0].bytes - sizeof(float) toBuffer:sample destinationOffset:0 size:sizeof(float)];
            [blit endEncoding];
        });
        large = {};
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted && *static_cast<const float*>(sample.contents) == 3.f, "Large native allocation/owner contract differs");
    }
    if (gpu_backend_available(GpuBackend::Vulkan)) {
        const auto other = backend == GpuBackend::Metal ? GpuBackend::Vulkan : GpuBackend::Metal;
        auto source = Tensor::full({count}, 7.f, Device::GPU);
        Tensor output;
        {
            const GpuBackendScope other_scope(other);
            output = Tensor::zeros({count}, Device::GPU);
        }
        const std::array<const Tensor*, 1> inputs{&source};
        const std::array<Tensor*, 1> outputs{&output};
        const auto command = reader.submitWrites(inputs, outputs, [](id<MTLCommandBuffer> command, auto input, auto output) {
            auto blit = [command blitCommandEncoder];
            [blit copyFromBuffer:input[0].buffer sourceOffset:input[0].offset toBuffer:output[0].buffer destinationOffset:output[0].offset size:input[0].bytes];
            [blit endEncoding];
        });
        source.add_(4.f);
        require(output.add(2.f).cpu().ptr<float>()[count - 1] == 9.f, "Mixed native producer/consumer ordering differs");
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Mixed native command failed");
    }
    Tensor cpu = Tensor::ones({3}, Device::CPU);
    std::array<const Tensor*, 1> inputs{&cpu};
    bool rejected = false;
    try {
        (void)reader.submit(inputs, [](auto, auto) {});
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "CPU tensor was accepted as a native Metal buffer");
    Tensor gpu = Tensor::ones({2, 3}, Device::GPU);
    Tensor strided = gpu.transpose(0, 1);
    inputs[0] = &strided;
    rejected = false;
    try {
        (void)reader.submit(inputs, [](auto, auto) {});
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Noncontiguous tensor was accepted as a flat buffer");
    inputs[0] = &gpu;
    bool failed = false;
    try {
        (void)reader.submit(inputs, [](auto, auto) { throw std::runtime_error("Encoding failure"); });
    } catch (const std::runtime_error&) { failed = true; }
    require(failed, "Encoding error was swallowed");
    auto command = reader.submit(inputs, [](auto, auto) {});
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted, "Encoding failure poisoned the next submission");
    std::puts("Metal tensor access contracts passed: resident read/write views, offsets, GPU dependencies, mutations and encoding failure recovery.");
}
int main() {
    @autoreleasepool {
        if (!gpu_backend_available(GpuBackend::Metal)) {
            return lfs::metal_test::unavailableMetal4();
        }
        try {
            run(GpuBackend::Metal);
            if (gpu_backend_available(GpuBackend::Vulkan))
                run(GpuBackend::Vulkan);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
}

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/splat_data.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_metal_reader.hpp"
#include "core/tensor_rad.hpp"
#include "splat_preprocessor.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using namespace lfs::core;
using namespace lfs::rendering::metal;
static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
static void run() {
    const GpuBackendScope scope(GpuBackend::Metal);
    MetalTensorReader reader;
    SplatPreprocessor projector(reader.device());
    constexpr uint32_t n = 131, page_size = 64, capacity = 192;
    std::vector<float> xyz(n * 3), rgb(n * 3), scales(n * 3, -3), rotation(n * 4), opacity(n, 1), rest(n * 45);
    for (uint32_t s = 0; s < n; ++s) {
        xyz[s * 3 + 2] = 3;
        rotation[s * 4] = 1;
        for (uint32_t c = 0; c < 3; ++c)
            rgb[s * 3 + c] = float(int((s + c) % 7) - 3) / 8.f;
        for (uint32_t c = 0; c < 45; ++c)
            rest[(size_t(s) * 15 + c / 3) * 3 + c % 3] = float(int((s + c) % 13) - 6) * .01f;
    }
    for (int degree : {0, 3}) {
        SplatData model(degree, Tensor::from_vector(xyz, {n, 3}, Device::GPU), Tensor::from_vector(rgb, {n, 1, 3}, Device::GPU),
                        degree ? Tensor::from_vector(rest, {n, 15, 3}, Device::GPU) : Tensor{}, Tensor::from_vector(scales, {n, 3}, Device::GPU),
                        Tensor::from_vector(rotation, {n, 4}, Device::GPU), Tensor::from_vector(opacity, {n, 1}, Device::GPU), 1.f);
        if (degree)
            (void)model.apply_shN_value_quant();
        require(!degree || model.shN_value_quantized(), "Integration fixture must use production Q16 source storage");
        RadPagePool pool;
        pool.page_splats = page_size;
        pool.sh_slots = degree ? 12 : 0;
        const std::array<size_t, 7> sizes = {capacity * 12, capacity * 8, size_t(capacity) * pool.sh_slots * 4, capacity * 8, capacity * 8, capacity * 2, 3 * 64};
        for (size_t r = 0; r < sizes.size(); ++r)
            if (sizes[r])
                pool.regions[r] = Tensor::empty({sizes[r]}, Device::GPU, DataType::UInt8);
        for (uint32_t page = 0; page < 3; ++page) {
            RadPageSources source{model.means_raw(), model.sh0_raw(), degree ? model.shN_raw() : Tensor{}, model.rotation_raw(), model.scaling_raw(), model.opacity_raw(),
                                  degree ? model.shN_value_bounds() : Tensor{}, page * page_size, std::min(page_size, n - page * page_size), degree ? 15u : 0u, bool(degree)};
            rad_page_quantize(source, pool, page);
        }
        std::array<const Tensor*, 7> tensors{};
        std::array<id<MTLBuffer>, 7> snapshot{};
        for (size_t r = 0; r < sizes.size(); ++r)
            if (sizes[r]) {
                tensors[r] = &pool.regions[r];
                snapshot[r] = [reader.device() newBufferWithLength:sizes[r] options:MTLResourceStorageModeShared];
            }
        auto output = [reader.device() newBufferWithLength:capacity * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
        const Projection view{matrix_identity_float4x4, matrix_identity_float4x4, {0, 0, 0, 0}, {40, 40, 16, 16}, {.01f, 100, 1, .3f}, {32, 32, 0, 0}};
        // Submit immediately after the actual tensor uploader. The consumer
        // must use GPU producer ordering without any CPU completion readback.
        auto command = reader.submit(tensors, [&](id<MTLCommandBuffer> command, std::span<const MetalTensorView> views) {
            auto blit = [command blitCommandEncoder];
            for (size_t r = 0; r < sizes.size(); ++r)
                if (sizes[r])
                    [blit copyFromBuffer:views[r].buffer sourceOffset:views[r].offset toBuffer:snapshot[r] destinationOffset:0 size:sizes[r]];
            [blit endEncoding];
            const auto slice = [&](size_t r) { return BufferSlice{views[r].buffer, views[r].offset}; };
            SplatBuffers inputs{slice(0), slice(4), slice(3), slice(5), slice(1), slice(2), slice(6), {}, capacity, degree ? 15u : 0u, ShStorage::RadSigned8, false, page_size};
            projector.encode(command, inputs, view, degree, PrimitiveMode::Gaussian, {output});
        });
        // Mutate the exact pool after submission. The snapshot and native
        // projection must keep the prior generation until their reads finish.
        pool.regions[1].zero_();
        const auto changed = pool.regions[1].cpu();
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Uploader→native RAD consumer ordering failed");
        const auto result = static_cast<const ProjectedSplat*>(output.contents);
        const auto dc = static_cast<const _Float16*>(snapshot[1].contents);
        const auto codes = static_cast<const int8_t*>(snapshot[2].contents);
        const auto frames = static_cast<const float*>(snapshot[6].contents);
        for (uint32_t s = 0; s < capacity; ++s) {
            require((result[s].bounds.z > result[s].bounds.x) == (s < n), "Real RAD uploader tail reached native projection");
            if (s >= n)
                continue;
            for (uint32_t c = 0; c < 3; ++c) {
                double expected = .5 + .2820947917738781 * double(dc[s * 4 + c]);
                if (degree) {
                    for (auto [k, basis] : std::array<std::pair<uint32_t, double>, 3>{{{1, std::sqrt(3. / (4 * M_PI))}, {5, std::sqrt(5. / (4 * M_PI))}, {11, std::sqrt(7. / (4 * M_PI))}}}) {
                        const uint32_t component = k * 3 + c, band = component < 9 ? 0 : component < 24 ? 1
                                                                                                        : 2;
                        const size_t index = (size_t(s / 32) * pool.sh_slots * 32 + (component / 4) * 32 + s % 32) * 4 + component % 4;
                        expected += basis * double(codes[index]) / 127. * frames[(s / page_size) * 16 + band];
                    }
                }
                require(std::abs(result[s].color[c] - std::max(0., expected)) < 2e-6, "Real uploader RAD packing and native decoder disagree");
            }
        }
        require(std::memcmp(changed.ptr<uint8_t>(), snapshot[1].contents, sizes[1]) != 0, "Pool mutation fixture did not modify the original allocation");
    }
    std::puts("Metal RAD tensor contracts passed: real Q16/SH0 quantizer, padded tails, native zero-copy binding and producer/consumer mutation ordering.");
}
int main() {
    @autoreleasepool {
        if (!gpu_backend_available(GpuBackend::Metal))
            return 77;
        try {
            run();
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
}

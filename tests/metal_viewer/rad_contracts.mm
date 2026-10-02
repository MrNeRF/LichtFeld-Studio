/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_preprocessor.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
using namespace lfs::rendering::metal;
static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
static void run(id<MTLDevice> device) {
    SplatPreprocessor projector(device);
    auto queue = [device newCommandQueue];
    constexpr uint32_t count = 131, page = 64;
    const auto upload = [&](const void* data, size_t size) { return BufferSlice{[device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared]}; };
    std::vector<float> xyz(count * 3);
    std::vector<_Float16> rgb(count * 4), scales(count * 4), rotation(count * 4), alpha(count);
    for (uint32_t n = 0; n < count; ++n) {
        xyz[n * 3 + 2] = 3;
        alpha[n] = 1;
        rotation[n * 4] = 1;
        for (uint32_t c = 0; c < 3; ++c) {
            scales[n * 4 + c] = -3;
            rgb[n * 4 + c] = _Float16(float(int((n + c) % 7) - 3) / 8.f);
        }
        // Padding must not become the next splat's first component.
        rgb[n * 4 + 3] = scales[n * 4 + 3] = NAN;
    }
    std::array<simd_float4, 12> frames{};
    for (uint32_t p = 0; p < 3; ++p)
        frames[p * 4] = {.25f * float(p + 1), .5f * float(p + 1), .75f * float(p + 1), NAN};
    SplatBuffers input;
    input.count = count;
    input.storage = ShStorage::RadSigned8;
    input.rad_page_splats = page;
    input.means = upload(xyz.data(), xyz.size() * 4);
    input.sh0 = upload(rgb.data(), rgb.size() * 2);
    input.log_scales = upload(scales.data(), scales.size() * 2);
    input.rotations = upload(rotation.data(), rotation.size() * 2);
    input.opacity_logits = upload(alpha.data(), alpha.size() * 2);
    input.sh_bounds = upload(frames.data(), sizeof(frames));
    const Projection view{matrix_identity_float4x4, matrix_identity_float4x4, {0, 0, 0, 0}, {40, 40, 16, 16}, {.01f, 100, 1, .3f}, {32, 32, 0, 0}};
    auto output = [device newBufferWithLength:count * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
    auto geometry = [device newBufferWithLength:count * sizeof(GutSplat) options:MTLResourceStorageModeShared];
    const auto code = [](uint32_t n, uint32_t c) { return int8_t(int((n + c) % 3) * 127 - 127); };
    size_t checks = 0;
    for (uint32_t rest : {3u, 8u, 15u}) {
        input.layout_rest = rest;
        const uint32_t slots = (rest * 3 + 3) / 4, padded = (count + 31) / 32 * 32;
        std::vector<int8_t> codes(size_t(padded) * slots * 4, 0);
        for (uint32_t n = 0; n < count; ++n)
            for (uint32_t c = 0; c < rest * 3; ++c)
                codes[(size_t(n / 32) * slots * 32 + (c / 4) * 32 + n % 32) * 4 + c % 4] = code(n, c);
        input.sh_rest = upload(codes.data(), codes.size());
        for (uint32_t degree = 0; degree <= 3; ++degree) {
            if ((degree + 1) * (degree + 1) - 1 > rest)
                continue;
            for (auto mode : {PrimitiveMode::Gaussian, PrimitiveMode::Points, PrimitiveMode::Discs, PrimitiveMode::Gut}) {
                auto command = [queue commandBuffer];
                projector.encode(command, input, view, degree, mode, {output}, {}, {}, mode == PrimitiveMode::Gut ? BufferSlice{geometry} : BufferSlice{});
                [command commit];
                [command waitUntilCompleted];
                require(command.status == MTLCommandBufferStatusCompleted, "RAD projection command failed");
                const auto result = static_cast<const ProjectedSplat*>(output.contents);
                for (uint32_t n = 0; n < count; ++n) {
                    require(result[n].bounds.z > result[n].bounds.x, "Padded RAD attributes culled valid input");
                    for (uint32_t c = 0; c < 3; ++c) {
                        const auto coefficient = [&](uint32_t k) { const uint32_t component=k*3+c,band=component<9?0:component<24?1:2;
                            return double(code(n,component))/127.*frames[(n/page)*4][band]; };
                        double expected = .5 + .2820947917738781 * double(rgb[n * 4 + c]);
                        if (degree >= 1)
                            expected += std::sqrt(3. / (4 * M_PI)) * coefficient(1);
                        if (degree >= 2)
                            expected += std::sqrt(5. / (4 * M_PI)) * coefficient(5);
                        if (degree >= 3)
                            expected += std::sqrt(7. / (4 * M_PI)) * coefficient(11);
                        require(std::abs(result[n].color[c] - std::max(0., expected)) < 2e-6, "RAD signed SH component or page/band scale differs from analytic SH");
                        ++checks;
                    }
                    require(std::abs(result[n].conic_opacity.w - 1.f / (1 + std::exp(-1.f))) < 1e-6, "RAD half opacity differs");
                    if (mode == PrimitiveMode::Gut) {
                        const auto g = static_cast<const GutSplat*>(geometry.contents);
                        require(std::abs(g[n].inverse0.x - std::exp(3.f)) < 1e-4 && std::abs(g[n].inverse1.y - std::exp(3.f)) < 1e-4, "RAD half scale or quaternion stride differs");
                    }
                }
            }
        }
    }
    const std::array<uint32_t, 7> indices = {130, 127, 64, 63, 32, 31, 0};
    LodSelection cut{upload(indices.data(), sizeof(indices)), {}, {}, {}, 7, count, true, false};
    const std::array<uint32_t, 7> logical_ids = {1000, 900, 800, 700, 2, 1, 0};
    cut.logical_indices = upload(logical_ids.data(), sizeof(logical_ids));
    cut.logical_count = 1001;
    auto command = [queue commandBuffer];
    projector.encode(command, input, view, 3, PrimitiveMode::Gaussian, {output}, {}, {}, {}, cut);
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted, "Sparse RAD cut projection failed");
    const auto selected = static_cast<const ProjectedSplat*>(output.contents);
    for (size_t n = 0; n < indices.size(); ++n) {
        const uint32_t source = indices[n];
        double expected = .5 + .2820947917738781 * double(rgb[source * 4]);
        for (auto [k, basis] : std::array<std::pair<uint32_t, double>, 3>{{{1, std::sqrt(3. / (4 * M_PI))}, {5, std::sqrt(5. / (4 * M_PI))}, {11, std::sqrt(7. / (4 * M_PI))}}}) {
            const uint32_t c = k * 3, band = c < 9 ? 0 : c < 24 ? 1
                                                                : 2;
            expected += basis * double(code(source, c)) / 127. * frames[(source / page) * 4][band];
        }
        require(std::abs(selected[n].color.x - std::max(0., expected)) < 2e-6, "Sparse RAD cut used compact-slot rather than physical-page SH scale");
    }
    // The logical scene can be larger than its physical pool. A small
    // resident deletion prefix must not be read using physical slot IDs.
    const std::array<uint8_t, 3> deleted = {1, 0, 0};
    input.deleted = upload(deleted.data(), sizeof(deleted));
    input.deleted_count = 3;
    command = [queue commandBuffer];
    projector.encode(command, input, view, 3, PrimitiveMode::Gaussian, {output}, {}, {}, {}, cut);
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted, "Logical RAD deletion dispatch failed");
    for (size_t n = 0; n < logical_ids.size(); ++n)
        require((selected[n].bounds.z > selected[n].bounds.x) == (logical_ids[n] != 0), "RAD deletion used a physical ID or read beyond the logical mask prefix");
    cut.logical_count = 1000;
    command = [queue commandBuffer];
    projector.encode(command, input, view, 3, PrimitiveMode::Gaussian, {output}, {}, {}, {}, cut);
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted && selected[0].bounds.z == 0, "Out-of-range logical RAD node was not culled");
    for (uint32_t invalid : {0u, 63u}) {
        input.rad_page_splats = invalid;
        bool rejected = false;
        try {
            projector.encode([queue commandBuffer], input, view, 3, PrimitiveMode::Gaussian, { output });
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "Invalid RAD page stride was accepted");
    }
    std::printf("Metal RAD contracts passed: %zu analytic SH checks, padded half geometry, page boundaries and sparse physical cuts.\n", checks);
}
int main() {
    @autoreleasepool {
        auto device = MTLCreateSystemDefaultDevice();
        if (!device)
            return LFS_METAL_TEST_REQUIRE_DEVICE ? 1 : 77;
        try {
            run(device);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
}

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "tile_rasterizer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>
using namespace lfs::rendering::metal;
static void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
struct Readback {
    id<MTLBuffer> color, depth, pick;
    size_t color_stride, depth_stride, pick_stride;
};
static Readback readback(id<MTLDevice> device, id<MTLCommandBuffer> command, RasterFrame& frame) {
    const auto w = frame.color().width, h = frame.color().height;
    Readback r;
    r.color_stride = (w * 8 + 255) / 256 * 256;
    r.depth_stride = (w * 16 + 255) / 256 * 256;
    r.pick_stride = (w * 4 + 255) / 256 * 256;
    r.color = [device newBufferWithLength:r.color_stride * h options:MTLResourceStorageModeShared];
    r.depth = [device newBufferWithLength:r.depth_stride * h options:MTLResourceStorageModeShared];
    r.pick = [device newBufferWithLength:r.pick_stride * h options:MTLResourceStorageModeShared];
    auto e = [command blitCommandEncoder];
    const auto copy = [&](id<MTLTexture> texture, id<MTLBuffer> buffer, size_t stride) {
        [e copyFromTexture:texture
                         sourceSlice:0
                         sourceLevel:0
                        sourceOrigin:MTLOriginMake(0, 0, 0)
                          sourceSize:MTLSizeMake(w, h, 1)
                            toBuffer:buffer
                   destinationOffset:0
              destinationBytesPerRow:stride
            destinationBytesPerImage:stride * h];
    };
    copy(frame.color(), r.color, r.color_stride);
    copy(frame.depth(), r.depth, r.depth_stride);
    copy(frame.pick(), r.pick, r.pick_stride);
    [e endEncoding];
    return r;
}
static void wait(id<MTLCommandBuffer> command) {
    [command commit];
    [command waitUntilCompleted];
    require(command.status == MTLCommandBufferStatusCompleted, command.error.localizedDescription.UTF8String ?: "GPU command failed");
}
static void compare(const Readback& r, const std::vector<ProjectedSplat>& splats, uint32_t w, uint32_t h,
                    simd_float4 bg, RasterMode mode) {
    std::vector<uint32_t> sorted(splats.size());
    for (uint32_t i = 0; i < sorted.size(); ++i)
        sorted[i] = i;
    std::stable_sort(sorted.begin(), sorted.end(), [&](auto a, auto b) { return splats[a].color.w < splats[b].color.w; });
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            double rgb[3] = {}, trans = 1, z = 0, near = 0, median = 1e10;
            uint32_t picked = UINT32_MAX;
            for (auto id : sorted) {
                const auto& s = splats[id];
                // Tile membership is relevant at the support boundary.
                if (s.bounds.z <= s.bounds.x || s.bounds.w <= s.bounds.y ||
                    x / 16 < s.bounds.x / 16 || x / 16 >= (s.bounds.z + 15) / 16 ||
                    y / 16 < s.bounds.y / 16 || y / 16 >= (s.bounds.w + 15) / 16)
                    continue;
                const double dx = x - s.mean_depth.x, dy = y - s.mean_depth.y;
                const auto c = s.conic_opacity;
                const double q = c.x * dx * dx + 2 * c.y * dx * dy + c.z * dy * dy;
                double a = mode == RasterMode::Points ? (dx * dx + dy * dy <= s.mean_depth.w * s.mean_depth.w ? c.w : 0) : mode == RasterMode::Discs ? (q <= 9 ? c.w : 0)
                                                                                                                                                     : c.w * std::exp(-.5 * q);
                a = std::min(a, double(.999f));
                if (a < .5 / 255)
                    continue;
                if (picked == UINT32_MAX) {
                    picked = id;
                    near = s.mean_depth.z;
                }
                for (int c = 0; c < 3; ++c)
                    rgb[c] += s.color[c] * a * trans;
                z += s.mean_depth.z * a * trans;
                const double next = trans * (1 - a);
                if (trans > .5 && next <= .5)
                    median = s.mean_depth.z;
                trans = next;
                if (trans < 1e-4)
                    break;
            }
            const auto* rgba = reinterpret_cast<const _Float16*>(static_cast<const char*>(r.color.contents) + y * r.color_stride) + x * 4;
            const auto* depth = reinterpret_cast<const float*>(static_cast<const char*>(r.depth.contents) + y * r.depth_stride) + x * 4;
            const auto* pick = reinterpret_cast<const uint32_t*>(static_cast<const char*>(r.pick.contents) + y * r.pick_stride) + x;
            for (int c = 0; c < 3; ++c)
                require(std::abs(float(rgba[c]) - (rgb[c] + bg[c] * bg.w * trans)) < .002, "Compositing differs from stable CPU reference");
            require(std::abs(float(rgba[3]) - (1 - trans + bg.w * trans)) < .001, "Alpha mismatch");
            require(std::abs(depth[0] - z) < .001, "Weighted depth mismatch");
            require(std::abs(depth[1] - (1 - trans)) < 2e-5, "Depth alpha mismatch");
            require(std::abs(depth[2] - near) < 1e-5, "First contributor depth mismatch");
            require(std::abs(depth[3] - median) < 1e-5, "Median depth mismatch");
            require(*pick == picked, "Pick source ID mismatch");
        }
}
static void run(id<MTLDevice> device) {
    TileRasterizer raster(device);
    auto queue = [device newCommandQueue];
    constexpr uint32_t width = 37, height = 29;
    const simd_float4 bg{.15f, .1f, .2f, .4f};
    RasterFrame frame(device, width, height, 4097, 4097 * 6);
    std::mt19937 random(0x1939);
    for (uint32_t count : {0u, 1u, 255u, 256u, 257u, 2047u, 2048u, 2049u, 4097u}) {
        std::vector<ProjectedSplat> splats(count);
        uint64_t expected_instances = 0;
        for (uint32_t i = 0; i < count; ++i) {
            const float x = float(random() % width) + .5f, y = float(random() % height) + .5f;
            auto& s = splats[i];
            // Many equal-depth keys; source order must be stable across blocks.
            s.mean_depth = {x, y, 1 + float(random() % 11), 3};
            s.conic_opacity = {.5f, 0, .5f, .05f + float(random() % 800) / 1000};
            s.color = {float(random() % 100) / 100, float(random() % 100) / 100, float(random() % 100) / 100, 1};
            // Deliberately disagree with linear Z; radial ties remain stable.
            s.color.w = 1 + float((i * 7) % 11);
            s.bounds = {uint32_t(std::max(0.f, x - 6)), uint32_t(std::max(0.f, y - 6)),
                        uint32_t(std::min(float(width), x + 7)), uint32_t(std::min(float(height), y + 7))};
            if (i % 19 == 3)
                s.bounds = {};
            else
                expected_instances += ((s.bounds.z + 15) / 16 - s.bounds.x / 16) * ((s.bounds.w + 15) / 16 - s.bounds.y / 16);
        }
        auto input = count ? [device newBufferWithBytes:splats.data()
                                                 length:count * sizeof(ProjectedSplat)
                                                options:MTLResourceStorageModeShared]
                           : nil;
        for (auto mode : {RasterMode::Gaussian, RasterMode::Points, RasterMode::Discs}) {
            auto command = [queue commandBuffer];
            raster.encode(command, {input}, count, mode, bg, frame);
            require(frame.busy(), "Frame was not reserved");
            bool rejected = false;
            try {
                raster.encode(command, {input}, count, mode, bg, frame);
            } catch (const std::logic_error&) { rejected = true; }
            require(rejected, "In-flight scratch was reused");
            auto read = readback(device, command, frame);
            wait(command);
            require(!frame.busy(), "Completed frame remains busy");
            require(frame.status().error == RasterError::None, "Unexpected overflow");
            require(frame.status().required_instances == expected_instances, "Tile count mismatch");
            compare(read, splats, width, height, bg, mode);
        }
    }
    // Exact 50% crossing is inclusive, matching desktop median depth. This
    // analytic case catches a wrong strict comparison independently of Vulkan.
    const std::vector<ProjectedSplat> threshold_splats = {
        {{18, 14, 3, 3}, {1, 0, 1, .5f}, {1, 0, 0, 9}, {16, 12, 21, 17}},
        {{18, 14, 6, 3}, {1, 0, 1, .9f}, {0, 1, 0, 36}, {16, 12, 21, 17}}};
    auto threshold_input = [device newBufferWithBytes:threshold_splats.data() length:threshold_splats.size() * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
    auto threshold_command = [queue commandBuffer];
    raster.encode(threshold_command, {threshold_input}, 2, RasterMode::Gaussian, bg, frame);
    const auto threshold_read = readback(device, threshold_command, frame);
    wait(threshold_command);
    compare(threshold_read, threshold_splats, width, height, bg, RasterMode::Gaussian);
    // Overflow cannot publish only part of a scene. Reusing that reservation for
    // an empty scene must clear stale ranges and recover a successful status.
    RasterFrame small(device, width, height, 1, 1);
    const ProjectedSplat large{{18, 14, 1, 30}, {.01f, 0, .01f, .9f}, {1, 0, 0, 1}, {0, 0, width, height}};
    auto input = [device newBufferWithBytes:&large length:sizeof(large) options:MTLResourceStorageModeShared];
    auto command = [queue commandBuffer];
    raster.encode(command, {input}, 1, RasterMode::Gaussian, bg, small);
    auto read = readback(device, command, small);
    wait(command);
    require(small.status().error == RasterError::InstanceCapacityExceeded, "Overflow not reported");
    require(small.status().required_instances == 6, "Overflow requirement incorrect");
    compare(read, {}, width, height, bg, RasterMode::Gaussian);
    command = [queue commandBuffer];
    raster.encode(command, {}, 0, RasterMode::Gaussian, bg, small);
    wait(command);
    require(small.status().error == RasterError::None, "Empty scene did not recover");
    std::puts("Metal tile raster contracts passed: stable depth, RGB/alpha/depth/pick, modes, scan/block boundaries, overflow and frame reuse.");
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

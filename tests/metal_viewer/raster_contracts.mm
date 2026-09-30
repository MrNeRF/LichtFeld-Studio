/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "tile_rasterizer.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
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
    // Independent 3D ray reference deliberately disagrees with the projected
    // conic/center depth: the raster must evaluate the normalized 3D Gaussian.
    RasterFrame gut_frame(device, width, height, 1, 6);
    ProjectedSplat gut_projected{{float(width) / 2 - .5f, float(height) / 2 - .5f, 99, 100},
                                 {1000, 0, 1000, .1f},
                                 {1, .25f, .125f, 9},
                                 {0, 0, width, height}};
    GutSplat gut_geometry{{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}, {0, 0, 3, .8f}};
    auto gut_input = [device newBufferWithBytes:&gut_projected length:sizeof(gut_projected) options:MTLResourceStorageModeShared];
    auto geometry = [device newBufferWithBytes:&gut_geometry length:sizeof(gut_geometry) options:MTLResourceStorageModeShared];
    Projection camera{};
    camera.intrinsics = {32, 32, float(width) / 2, float(height) / 2};
    camera.clip_scale = {.01f, 100, 1, .3f};
    camera.extent = {width, height, 0, 0};
    command = [queue commandBuffer];
    bool rejected = false;
    try {
        raster.encode(command, {gut_input}, 1, RasterMode::Gut, bg, gut_frame, {}, {}, camera);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && !gut_frame.busy(), "Missing 3DGUT geometry consumed reservation");
    raster.encode(command, {gut_input}, 1, RasterMode::Gut, bg, gut_frame, {}, {geometry}, camera);
    const auto gut_read = readback(device, command, gut_frame);
    wait(command);
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x) {
            const double u = (x + .5 - double(width) / 2) / 32, v = (y + .5 - double(height) / 2) / 32;
            const double alpha = .8 * std::exp(-18 * (1 - 1 / (1 + u * u + v * v)));
            const bool contributes = alpha >= .5 / 255;
            const auto color = reinterpret_cast<const _Float16*>(static_cast<const char*>(gut_read.color.contents) + y * gut_read.color_stride) + x * 4;
            const auto depth = reinterpret_cast<const float*>(static_cast<const char*>(gut_read.depth.contents) + y * gut_read.depth_stride) + x * 4;
            const auto id = reinterpret_cast<const uint32_t*>(static_cast<const char*>(gut_read.pick.contents) + y * gut_read.pick_stride)[x];
            const double a = contributes ? alpha : 0;
            require(std::abs(float(color[0]) - (a + bg.x * bg.w * (1 - a))) < .001, "3DGUT independent ray color differs");
            require(std::abs(depth[1] - a) < 1e-5, "3DGUT independent ray alpha differs");
            require(id == (contributes ? 0u : 0xffffffffu), "3DGUT independent ray picking differs");
            if (contributes) {
                const double z = 3 / (1 + u * u + v * v);
                require(std::abs(depth[0] - z * a) < 1e-4 && std::abs(depth[2] - z) < 1e-4, "3DGUT independent ray depth differs");
                require(a > .5 ? std::abs(depth[3] - z) < 1e-4 : depth[3] >= 1e9, "3DGUT independent median depth differs");
            }
        }
    // Independent spherical ray reference at a width that is not tile-aligned.
    // The wrapped span visits each column exactly once, including both edges.
    camera.extent.z = uint32_t(CameraModel::Equirectangular);
    gut_projected.bounds = {32, 0, 80, height};
    std::memcpy(gut_input.contents, &gut_projected, sizeof(gut_projected));
    for (bool subregion : {false, true}) {
        camera.panorama = subregion ? simd_float4{float(width * 2), float(height * 2), float(width), float(height) / 2} : simd_float4{float(width), float(height), 0, 0};
        command = [queue commandBuffer];
        raster.encode(command, {gut_input}, 1, RasterMode::Gut, bg, gut_frame, {}, {geometry}, camera);
        const auto spherical = readback(device, command, gut_frame);
        wait(command);
        require(gut_frame.status().error == RasterError::None && gut_frame.status().required_instances == 6,
                "Panorama duplicated wrapped tile instances");
        for (uint32_t y = 0; y < height; ++y)
            for (uint32_t x = 0; x < width; ++x) {
                const double azimuth = 2 * M_PI * ((x + .5 + camera.panorama.z) / camera.panorama.x - .5);
                const double elevation = M_PI * ((y + .5 + camera.panorama.w) / camera.panorama.y - .5);
                const double ray_z = std::cos(azimuth) * std::cos(elevation);
                const double alpha = .8 * std::exp(-18 * (1 - ray_z * ray_z));
                const bool contributes = alpha >= .5 / 255;
                const double a = contributes ? alpha : 0;
                const auto color = reinterpret_cast<const _Float16*>(static_cast<const char*>(spherical.color.contents) + y * spherical.color_stride) + x * 4;
                const auto depth = reinterpret_cast<const float*>(static_cast<const char*>(spherical.depth.contents) + y * spherical.depth_stride) + x * 4;
                const auto id = reinterpret_cast<const uint32_t*>(static_cast<const char*>(spherical.pick.contents) + y * spherical.pick_stride)[x];
                require(std::abs(float(color[0]) - (a + bg.x * bg.w * (1 - a))) < .001, "Independent spherical ray color differs");
                require(std::abs(depth[1] - a) < 1e-5, "Independent spherical ray alpha differs");
                require(id == (contributes ? 0u : 0xffffffffu), "Independent spherical ray picking differs");
                if (contributes) {
                    const double z = 3 * ray_z * ray_z;
                    if (ray_z > 0 && z > camera.clip_scale.x)
                        require(std::abs(depth[2] - z) < 1e-4, "Independent spherical ray depth differs");
                    else
                        require(depth[2] >= 1e9, "Panorama lost the desktop invalid-depth sentinel");
                }
            }
    }
    // Run projection and binning together: a rear splat straddles both edges
    // of a 37-pixel camera, whose padded tile grid has a different period.
    SplatPreprocessor projector(device);
    const std::array<float, 3> seam_mean{0, 0, -3}, seam_scale{-.69314718056f, -.69314718056f, -.69314718056f}, seam_dc{0, 0, 0};
    const simd_float4 seam_rotation{1, 0, 0, 0};
    const float seam_logit = 4;
    const auto upload = [&](const void* data, size_t bytes) {
        return BufferSlice{[device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared], 0};
    };
    SplatBuffers seam_source;
    seam_source.count = 1;
    seam_source.means = upload(&seam_mean, sizeof(seam_mean));
    seam_source.log_scales = upload(&seam_scale, sizeof(seam_scale));
    seam_source.sh0 = upload(&seam_dc, sizeof(seam_dc));
    seam_source.rotations = upload(&seam_rotation, sizeof(seam_rotation));
    seam_source.opacity_logits = upload(&seam_logit, sizeof(seam_logit));
    camera.model_to_world = camera.world_to_camera = matrix_identity_float4x4;
    camera.panorama = {float(width), float(height), 0, 0};
    command = [queue commandBuffer];
    projector.encode(command, seam_source, camera, 0, PrimitiveMode::Gut, {gut_input}, {}, {}, {geometry});
    raster.encode(command, {gut_input}, 1, RasterMode::Gut, {0, 0, 0, 0}, gut_frame, {}, {geometry}, camera);
    const auto seam_read = readback(device, command, gut_frame);
    wait(command);
    require(gut_frame.status().error == RasterError::None, "Odd-width seam overflowed its reservation");
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x) {
            const double azimuth = 2 * M_PI * ((x + .5) / width - .5);
            const double elevation = M_PI * ((y + .5) / height - .5);
            const double ray_z = std::cos(azimuth) * std::cos(elevation);
            // The desktop GUT evaluator measures distance to a line. Its
            // 16-pixel projected support prevents unrelated tiles contributing.
            double alpha = (x < 16 || x >= 32) ? 1 / (1 + std::exp(-4.)) * std::exp(-18 * (1 - ray_z * ray_z)) : 0;
            if (alpha < .5 / 255)
                alpha = 0;
            const auto color = reinterpret_cast<const _Float16*>(static_cast<const char*>(seam_read.color.contents) + y * seam_read.color_stride) + x * 4;
            if (std::abs(float(color[3]) - alpha) >= .001)
                std::fprintf(stderr, "Seam pixel %u,%u: alpha %.9g expected %.9g, bounds %u,%u,%u,%u\n", x, y, float(color[3]), alpha,
                             static_cast<const ProjectedSplat*>(gut_input.contents)->bounds.x, static_cast<const ProjectedSplat*>(gut_input.contents)->bounds.y,
                             static_cast<const ProjectedSplat*>(gut_input.contents)->bounds.z, static_cast<const ProjectedSplat*>(gut_input.contents)->bounds.w);
            require(std::abs(float(color[3]) - alpha) < .001, "Odd-width panorama seam lost or duplicated a contribution");
        }
    // Invalid requests fail before consuming the reusable frame.
    camera.panorama.x = 0;
    command = [queue commandBuffer];
    rejected = false;
    try {
        raster.encode(command, {gut_input}, 1, RasterMode::Gut, bg, gut_frame, {}, {geometry}, camera);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && !gut_frame.busy(), "Invalid panorama consumed its frame reservation");
    // A behind-camera Gaussian can contribute line-distance alpha in a
    // conservative GUT tile. Only valid forward depths belong in the average.
    const std::array<ProjectedSplat, 2> mixed_projection{{{{18, 14, 3, 100}, {1, 0, 1, .8f}, {1, 0, 0, 9}, {0, 0, width, height}},
                                                          {{18, 14, 3, 100}, {1, 0, 1, .8f}, {0, 1, 0, 9}, {0, 0, width, height}}}};
    const std::array<GutSplat, 2> mixed_geometry{{{{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}, {0, 0, -3, .8f}},
                                                  {{2, 0, 0, 0}, {0, 2, 0, 0}, {0, 0, 2, 0}, {0, 0, 3, .8f}}}};
    auto mixed_input = [device newBufferWithBytes:mixed_projection.data() length:sizeof(mixed_projection) options:MTLResourceStorageModeShared];
    auto mixed_gut = [device newBufferWithBytes:mixed_geometry.data() length:sizeof(mixed_geometry) options:MTLResourceStorageModeShared];
    RasterFrame expected_frame(device, width, height, 2, 12);
    camera.extent.z = uint32_t(CameraModel::Perspective);
    camera.rasterization = {1, 1, 100, 0};
    for (float far : {100.f, 2.5f}) {
        camera.rasterization.z = far;
        command = [queue commandBuffer];
        raster.encode(command, {mixed_input}, 2, RasterMode::Gut, {0, 0, 0, 0}, expected_frame, {}, {mixed_gut}, camera);
        const auto expected = readback(device, command, expected_frame);
        wait(command);
        const auto center_depth = reinterpret_cast<const float*>(static_cast<const char*>(expected.depth.contents) + 14 * expected.depth_stride) + 18 * 4;
        require(std::abs(center_depth[1] - .96f) < 1e-5f, "Invalid depth incorrectly removed visible GUT opacity");
        if (far > 3) {
            require(std::abs(center_depth[0] - .48f) < 1e-5f && std::abs(center_depth[2] - .16f) < 1e-5f,
                    "Invalid GUT contributor contaminated expected-depth weights");
            require(std::abs(center_depth[0] / center_depth[2] - 3.f) < 1e-5f,
                    "Expected GUT depth normalized against visible rather than valid opacity");
        } else
            require(center_depth[0] == 0 && center_depth[2] == 0, "Expected-depth capture ignored its far cutoff");
    }
    camera.rasterization.y = 1;
    camera.rasterization.z = NAN;
    rejected = false;
    try {
        raster.encode([queue commandBuffer], { mixed_input }, 2, RasterMode::Gut, bg, expected_frame, {}, {mixed_gut}, camera);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected && !expected_frame.busy(), "Invalid expected-depth parameters consumed the reservation");
    // The standard portal normalizes its Gaussian tail to zero at q=8.
    // Verify analytic coverage independently of the Vulkan reference and projector.
    const ProjectedSplat portal_splat{{18, 14, 3, 100}, {1, 0, 1, .8f}, {1, 0, 0, 9}, {0, 0, width, height}};
    auto portal_input = [device newBufferWithBytes:&portal_splat length:sizeof(portal_splat) options:MTLResourceStorageModeShared];
    camera.rasterization = {1, 0, 0, 1};
    command = [queue commandBuffer];
    raster.encode(command, {portal_input}, 1, RasterMode::Gaussian, {0, 0, 0, 0}, expected_frame, {}, {}, camera);
    const auto portal_read = readback(device, command, expected_frame);
    wait(command);
    for (uint32_t x = 18; x <= 22; ++x) {
        const double q = double(x - 18) * (x - 18), edge = std::exp(-4.);
        double alpha = .8 * std::max(0., (std::exp(-.5 * q) - edge) / (1 - edge));
        if (alpha < 1. / 255)
            alpha = 0;
        const auto color = reinterpret_cast<const _Float16*>(static_cast<const char*>(portal_read.color.contents) + 14 * portal_read.color_stride) + x * 4;
        require(std::abs(float(color[3]) - alpha) < .001, "Portal Gaussian tail differs from its normalized alpha contract");
    }
    // Compacted draw slot zero must retain logical primitive seven for picking.
    const uint32_t logical_id = 7;
    LodSelection lod;
    lod.enabled = true;
    lod.count = 1;
    lod.source_count = 8;
    lod.indices = {[device newBufferWithBytes:&logical_id length:sizeof(logical_id) options:MTLResourceStorageModeShared]};
    camera.rasterization = {1, 0, 0, 0};
    command = [queue commandBuffer];
    raster.encode(command, {portal_input}, 1, RasterMode::Gaussian, {0, 0, 0, 0}, expected_frame, {}, {}, camera, lod);
    const auto lod_read = readback(device, command, expected_frame);
    wait(command);
    const auto picked = reinterpret_cast<const uint32_t*>(static_cast<const char*>(lod_read.pick.contents) + 14 * lod_read.pick_stride) + 18;
    require(*picked == logical_id, "Resident LOD picking published a compact draw slot instead of its logical primitive");
    // A logical scene need not fit in the physical page pool. The selection
    // prefix can also be smaller than the scene; unknown IDs remain unselected.
    lod.source_count = 1;
    lod.logical_count = 8;
    std::array<simd_float4, 207> bounded_params{};
    bounded_params[24].x = 1;
    std::array<simd_float4, 258> bounded_colors{};
    bounded_colors[2] = {0, 1, 0, 1};
    const std::array<uint8_t, 2> bounded_selection = {2, 2};
    const uint32_t bounded_flags = 0;
    OverlayBuffers bounded_overlay;
    bounded_overlay.parameters = {[device newBufferWithBytes:bounded_params.data() length:sizeof(bounded_params) options:MTLResourceStorageModeShared]};
    bounded_overlay.colors = {[device newBufferWithBytes:bounded_colors.data() length:sizeof(bounded_colors) options:MTLResourceStorageModeShared]};
    bounded_overlay.flags = {[device newBufferWithBytes:&bounded_flags length:sizeof(bounded_flags) options:MTLResourceStorageModeShared]};
    bounded_overlay.selection = {[device newBufferWithBytes:bounded_selection.data() length:sizeof(bounded_selection) options:MTLResourceStorageModeShared]};
    bounded_overlay.parameter_count = 207;
    bounded_overlay.selection_count = 2;
    command = [queue commandBuffer];
    raster.encode(command, {portal_input}, 1, RasterMode::Gaussian, {0, 0, 0, 0}, expected_frame, bounded_overlay, {}, camera, lod);
    const auto bounded_read = readback(device, command, expected_frame);
    wait(command);
    const auto bounded_pick = reinterpret_cast<const uint32_t*>(static_cast<const char*>(bounded_read.pick.contents) + 14 * bounded_read.pick_stride) + 18;
    const auto bounded_color = reinterpret_cast<const _Float16*>(static_cast<const char*>(bounded_read.color.contents) + 14 * bounded_read.color_stride) + 18 * 4;
    require(*bounded_pick == 7 && std::abs(float(bounded_color[0]) - .8f) < .001 && bounded_color[1] == 0, "Logical pool ID was truncated or read outside its selection prefix");
    bounded_overlay.selection_count = 8;
    bool bounded_rejected = false;
    try {
        raster.encode([queue commandBuffer], { portal_input }, 1, RasterMode::Gaussian, bg, expected_frame, bounded_overlay, {}, camera, lod);
    } catch (const std::invalid_argument&) { bounded_rejected = true; }
    require(bounded_rejected, "Logical selection extent exceeded its resident buffer");
    // Spark high-opacity nodes encode a density kernel, not a probability.
    // Independent double-precision oracle checks the nonlinear saturation.
    ProjectedSplat density_splat = portal_splat;
    density_splat.conic_opacity.w = 2.f;
    auto density_input = [device newBufferWithBytes:&density_splat length:sizeof(density_splat) options:MTLResourceStorageModeShared];
    camera.display.z = 1;
    command = [queue commandBuffer];
    raster.encode(command, {density_input}, 1, RasterMode::Gaussian, {0, 0, 0, 0}, expected_frame, {}, {}, camera);
    const auto density_read = readback(device, command, expected_frame);
    wait(command);
    const double density = std::exp(3. / std::exp(1.));
    for (uint32_t x = 18; x <= 22; ++x) {
        const double q = double(x - 18) * (x - 18), power = .5 * q;
        double alpha = power > .5 * std::pow(std::sqrt(8.) + .7, 2) ? 0 : std::min(.999, 1 - std::pow(1 - std::exp(-power), density));
        if (alpha < .5 / 255)
            alpha = 0;
        const auto color = reinterpret_cast<const _Float16*>(static_cast<const char*>(density_read.color.contents) + 14 * density_read.color_stride) + x * 4;
        require(std::abs(float(color[3]) - alpha) < .001, "Spark density was clamped/treated as a probability");
    }
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

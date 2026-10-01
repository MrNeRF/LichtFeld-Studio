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
                    simd_float4 bg, RasterMode mode, bool expected_depth = false, float far = 100.f) {
    std::vector<uint32_t> sorted(splats.size());
    for (uint32_t i = 0; i < sorted.size(); ++i)
        sorted[i] = i;
    std::stable_sort(sorted.begin(), sorted.end(), [&](auto a, auto b) { return splats[a].color.w < splats[b].color.w; });
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            double rgb[3] = {}, trans = 1, z = 0, valid_weight = 0, near = 0, median = 1e10;
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
                if (!expected_depth || s.mean_depth.z <= far) {
                    z += s.mean_depth.z * a * trans;
                    valid_weight += a * trans;
                }
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
            require(std::abs(depth[2] - (expected_depth ? valid_weight : near)) < 1e-5, "First contributor/valid-weight depth mismatch");
            require(std::abs(depth[3] - median) < 1e-5, "Median depth mismatch");
            require(*pick == picked, "Pick source ID mismatch");
        }
}
// Long lists with sparse contributors put the first pick, median and
// saturation in different depth chunks. Compare every output channel to the
// independent double-precision source-order oracle, including partial tiles.
static void compare_depth_chunks(id<MTLDevice> device) {
    constexpr uint32_t n = 32769, w = 19, h = 17;
    std::vector<ProjectedSplat> splats(n);
    auto input = [device newBufferWithLength:n * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
    auto queue = [device newCommandQueue];
    TileRasterizer raster(device);
    RasterFrame frame(device, w, h, n, n * 4);
    const simd_float4 bg{.1f, .2f, .3f, 1};
    for (uint32_t scenario = 0; scenario < 4; ++scenario) {
        const uint32_t first = scenario == 0 ? 0 : 1088;
        const uint32_t stride = scenario == 0 ? 1 : scenario == 1 ? 97
                                                                  : 31;
        const float opacity = scenario == 1 ? .02f : .25f;
        for (uint32_t i = 0; i < n; ++i) {
            const bool visible = i >= first && (i - first) % stride == 0;
            splats[i] = {{visible ? 9.f : 1000.f, 8, .5f + float(i) * .0001f, 20},
                         {.02f, .004f, .03f, opacity},
                         {float(i % 7) / 7, float(i % 11) / 11, float(i % 13) / 13, float(i / 4 + 1)},
                         {0, 0, w, h}};
        }
        std::memcpy(input.contents, splats.data(), splats.size() * sizeof(ProjectedSplat));
        // Reusing scratch must not expose partials from a previous dense frame.
        for (uint32_t count : {n, 0u, 1u, n, n}) {
            auto command = [queue commandBuffer];
            Projection projection{};
            const bool expected_depth = scenario == 3;
            projection.rasterization = {1, expected_depth ? 1.f : 0.f, 1.5f, 0};
            raster.encode(command, {input}, count, RasterMode::Gaussian, bg, frame, {}, {}, projection);
            const auto actual = readback(device, command, frame);
            wait(command);
            require(frame.status().error == RasterError::None && frame.status().required_instances == uint64_t(count) * 4,
                    "Depth chunk fixture has unexpected tile membership");
            compare(actual, std::vector<ProjectedSplat>(splats.begin(), splats.begin() + count), w, h, bg, RasterMode::Gaussian, expected_depth, 1.5f);
        }
    }
    RasterFrame overflow(device, w, h, n, n * 2);
    auto command = [queue commandBuffer];
    raster.encode(command, {input}, n, RasterMode::Gaussian, bg, overflow);
    const auto empty = readback(device, command, overflow);
    wait(command);
    require(overflow.status().error == RasterError::InstanceCapacityExceeded,
            "Depth chunk admission ignored instance overflow");
    compare(empty, {}, w, h, bg, RasterMode::Gaussian);
}

// A fresh reservation uses the established two-SIMD GUT kernel. Reused
// dense reservations can choose SIMD32 without changing a single output bit.
static void compare_dense_gut_subtiles(id<MTLDevice> device) {
    constexpr uint32_t n = 8193, w = 19, h = 17;
    std::vector<ProjectedSplat> splats(n);
    std::vector<GutSplat> geometry(n);
    for (uint32_t i = 0; i < n; ++i) {
        splats[i] = {{9, 8, 3, 20}, {1, 0, 1, .05f},
                     {float(i % 7) / 7, float(i % 11) / 11, float(i % 13) / 13, float(i / 3 + 1)}, {0, 0, w, h}};
        geometry[i] = {{2.5f, 0, 0, 2}, {0, 2.5f, 0, 0}, {0, 0, 2.5f, 0},
                       {float(int(i % 5) - 2) * .15f, float(int(i % 7) - 3) * .15f, 3 + .02f * (i % 8), .05f}};
    }
    auto input = [device newBufferWithBytes:splats.data() length:splats.size() * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
    auto gut = [device newBufferWithBytes:geometry.data() length:geometry.size() * sizeof(GutSplat) options:MTLResourceStorageModeShared];
    auto queue = [device newCommandQueue];
    TileRasterizer raster(device);
    RasterFrame adaptive(device, w, h, n, n * 4);
    Projection camera{matrix_identity_float4x4, matrix_identity_float4x4, {0, 0, 0, 0}, {17, 17, w * .5f, h * .5f}, {.01f, 100, 1, .3f}, {w, h, uint32_t(CameraModel::Perspective), 0}};
    uint32_t previous_count = 0;
    for (auto model : {CameraModel::Perspective, CameraModel::Orthographic, CameraModel::Equirectangular})
        for (bool spark : {false, true}) {
            camera.extent.z = uint32_t(model);
            camera.panorama = {float(w), float(h), 0, 0};
            camera.display.z = spark ? 1 : 0;
            for (uint32_t i = 0; i < n; ++i) {
                splats[i].bounds.z = model == CameraModel::Equirectangular ? 32 : w;
                geometry[i].mean_opacity.w = spark ? 1.2f : .05f;
            }
            std::memcpy(input.contents, splats.data(), splats.size() * sizeof(ProjectedSplat));
            std::memcpy(gut.contents, geometry.data(), geometry.size() * sizeof(GutSplat));
            // Dense -> dense -> sparse -> empty -> dense -> dense exercises
            // both choices, source-count invalidation and old-output recovery.
            for (uint32_t count : {n, n, 1u, 0u, n, n}) {
                RasterFrame reference(device, w, h, n, n * 4);
                const simd_float4 bg = spark ? simd_float4{.1f, .2f, .3f, 1} : simd_float4{0, 0, 0, 0};
                auto command = [queue commandBuffer];
                raster.encode(command, {input}, count, RasterMode::Gut, bg, reference, {}, {gut}, camera);
                const auto expected = readback(device, command, reference);
                wait(command);
                command = [queue commandBuffer];
                raster.encode(command, {input}, count, RasterMode::Gut, bg, adaptive, {}, {gut}, camera);
                const auto actual = readback(device, command, adaptive);
                wait(command);
                require(adaptive.status().error == RasterError::None && adaptive.status().required_instances == uint64_t(count) * 4,
                        "Dense GUT fixture did not exercise the intended list density");
                require(reference.status().blend_threads == 64, "Fresh GUT reference did not use the original grouping");
                require(adaptive.status().blend_threads == (count == n && previous_count == n ? 32u : 64u),
                        "Dense/sparse GUT dispatch did not follow the completed density policy");
                previous_count = count;
                for (uint32_t y = 0; y < h; ++y) {
                    const auto equal = [&](id<MTLBuffer> a, size_t a_stride, id<MTLBuffer> b, size_t b_stride, size_t bytes) {
                        return std::memcmp(static_cast<const char*>(a.contents) + y * a_stride,
                                           static_cast<const char*>(b.contents) + y * b_stride, bytes) == 0;
                    };
                    require(equal(actual.color, actual.color_stride, expected.color, expected.color_stride, w * 8), "Dense GUT subtiles changed RGB/alpha");
                    require(equal(actual.depth, actual.depth_stride, expected.depth, expected.depth_stride, w * 16), "Dense GUT subtiles changed a depth channel");
                    require(equal(actual.pick, actual.pick_stride, expected.pick, expected.pick_stride, w * 4), "Dense GUT subtiles changed stable source IDs");
                }
            }
        }
}
// Compare the same 3D ray pipeline with support-sphere culling enabled and
// explicitly disabled. Color, all depth channels and source IDs must be exact,
// including anisotropy, affine shear, partial subtiles and ill-conditioned input.
static void compare_gut_culling_for_count(id<MTLDevice> device, uint32_t n) {
    constexpr uint32_t w = 129, h = 97;
    std::mt19937 random(1939);
    std::uniform_real_distribution<float> unit(0.f, 1.f);
    std::vector<float> means(n * 3), scales(n * 3), rotations(n * 4), sh0(n * 3), opacity(n);
    for (uint32_t i = 0; i < n; ++i) {
        const float angle = unit(random) * 6.28f;
        rotations[4 * i] = std::cos(angle * .5f);
        const float sine = std::sin(angle * .5f) / std::sqrt(3.f);
        for (uint32_t c = 0; c < 3; ++c) {
            means[3 * i + c] = c == 2 ? .06f + 5.f * unit(random) : (unit(random) - .5f) * 2.f;
            scales[3 * i + c] = std::log(.008f + .18f * unit(random));
            rotations[4 * i + c + 1] = sine;
            sh0[3 * i + c] = (unit(random) - .5f) * 2;
        }
        opacity[i] = unit(random) * 5 - 2;
    }
    const auto buffer = [&](const void* bytes, size_t length) {
        return [device newBufferWithBytes:bytes length:length options:MTLResourceStorageModeShared];
    };
    SplatBuffers input{{buffer(means.data(), means.size() * 4)}, {buffer(scales.data(), scales.size() * 4)}, {buffer(rotations.data(), rotations.size() * 4)}, {buffer(opacity.data(), opacity.size() * 4)}, {buffer(sh0.data(), sh0.size() * 4)}, {}, {}, {}, n, 0, ShStorage::CanonicalFloat32};
    auto projected = [device newBufferWithLength:n * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
    auto geometry = [device newBufferWithLength:n * sizeof(GutSplat) options:MTLResourceStorageModeShared];
    auto queue = [device newCommandQueue];
    SplatPreprocessor projector(device);
    TileRasterizer raster(device);
    RasterFrame frame(device, w, h, n, n * 63);
    for (auto camera : {CameraModel::Perspective, CameraModel::Orthographic})
        for (bool spark : {false, true})
            for (bool ill_conditioned : {false, true}) {
                Projection p{matrix_identity_float4x4, matrix_identity_float4x4, {0, 0, 0, 0}, {64, 57, w * .5f, h * .5f}, {.01f, 100, 1, .3f}, {w, h, uint32_t(camera), 0}};
                p.model_to_world.columns[0].x = 1.4f;
                p.model_to_world.columns[1].x = .35f;
                p.model_to_world.columns[1].y = ill_conditioned ? 1e-7f : .7f;
                p.model_to_world.columns[2].y = .12f;
                p.model_to_world.columns[2].z = 1.1f;
                p.display.z = spark ? 1 : 0;
                // Spark uses encoded density rather than opacity logits.
                if (spark) {
                    auto encoded = std::vector<float>(n);
                    for (uint32_t i = 0; i < n; ++i)
                        encoded[i] = .3f + float(i % 8) * .2f;
                    std::memcpy(input.opacity_logits.buffer.contents, encoded.data(), n * 4);
                } else
                    std::memcpy(input.opacity_logits.buffer.contents, opacity.data(), n * 4);
                auto command = [queue commandBuffer];
                projector.encode(command, input, p, 0, PrimitiveMode::Gut, {projected}, {}, {}, {geometry});
                wait(command);
                auto splats = static_cast<ProjectedSplat*>(projected.contents);
                auto guts = static_cast<GutSplat*>(geometry.contents);
                size_t bounds = 0;
                for (uint32_t i = 0; i < n; ++i)
                    if (splats[i].bounds.z > splats[i].bounds.x && splats[i].bounds.w > splats[i].bounds.y) {
                        require(std::isfinite(guts[i].inverse0.w), "Invalid GUT support sphere");
                        bounds += guts[i].inverse0.w > 0;
                        // For the dense case, enlarge valid conservative bins
                        // to exercise SIMD32 frustum rejection independently of
                        // the projected rectangle (never shrink alpha support).
                        if (n >= 4096)
                            splats[i].bounds = {0, 0, w, h};
                    }
                require(ill_conditioned ? bounds == 0 : bounds > 0, "GUT support-sphere fallback was not exercised");
                command = [queue commandBuffer];
                raster.encode(command, {projected}, n, RasterMode::Gut, {.1f, .2f, .3f, 1}, frame, {}, {geometry}, p);
                const auto culled = readback(device, command, frame);
                wait(command);
                require(frame.status().error == RasterError::None, "GUT culling fixture overflow");
                if (n >= 4096 && !ill_conditioned)
                    require(frame.status().required_instances > uint64_t(63) * 512, "Dense GUT culling fixture missed the SIMD32 policy");
                const std::vector<GutSplat> saved_geometry(guts, guts + n);
                for (uint32_t i = 0; i < n; ++i)
                    guts[i].inverse0.w = 0;
                command = [queue commandBuffer];
                raster.encode(command, {projected}, n, RasterMode::Gut, {.1f, .2f, .3f, 1}, frame, {}, {geometry}, p);
                const auto full = readback(device, command, frame);
                wait(command);
                std::memcpy(guts, saved_geometry.data(), n * sizeof(GutSplat));
                command = [queue commandBuffer];
                raster.encode(command, {projected}, n, RasterMode::Gut, {.1f, .2f, .3f, 1}, frame, {}, {geometry}, p);
                const auto reused_culled = readback(device, command, frame);
                wait(command);
                if (n >= 4096 && !ill_conditioned)
                    require(frame.status().blend_threads == 32, "Dense GUT frustum test did not execute SIMD32");
                if (n < 4096)
                    require(frame.status().blend_threads == 64, "Sparse GUT frustum test changed its grouping");
                for (uint32_t y = 0; y < h; ++y) {
                    const auto same = [&](id<MTLBuffer> a, id<MTLBuffer> b, size_t stride, size_t pixel_bytes) {
                        return std::memcmp(static_cast<const char*>(a.contents) + y * stride,
                                           static_cast<const char*>(b.contents) + y * stride, w * pixel_bytes) == 0;
                    };
                    require(same(culled.color, full.color, full.color_stride, 8), "GUT culling changed color/alpha");
                    require(same(culled.depth, full.depth, full.depth_stride, 16), "GUT culling changed depth");
                    require(same(culled.pick, full.pick, full.pick_stride, 4), "GUT culling changed source IDs");
                    require(same(reused_culled.color, full.color, full.color_stride, 8), "Adaptive GUT culling changed color/alpha");
                    require(same(reused_culled.depth, full.depth, full.depth_stride, 16), "Adaptive GUT culling changed depth");
                    require(same(reused_culled.pick, full.pick, full.pick_stride, 4), "Adaptive GUT culling changed source IDs");
                }
            }
    std::puts("GUT subtile culling preserves exact color, depth and IDs across 8 affine/density/camera cases.");
}
static void compare_gut_culling(id<MTLDevice> device) {
    compare_gut_culling_for_count(device, 259);
    compare_gut_culling_for_count(device, 4097);
}
// Independent double-precision oracle: an opaque foreground leaves a small
// background contribution spread over thousands of weak depth-ordered splats.
// Keeping the entire RGB accumulator in half loses that contribution even
// though each opacity is well above the viewer's visibility threshold.
static void compare_weak_transparent_layers(id<MTLDevice> device) {
    constexpr uint32_t n = 2145;
    std::vector<ProjectedSplat> splats(n);
    for (uint32_t i = 0; i < n; ++i) {
        splats[i] = {{0, 0, 5, 1}, {1, 0, 1, i ? 1.f / 256.f : 15.f / 16.f}, {i ? .8f : .4f, i ? .5f : .3f, i ? .25f : .2f, float(i + 1)}, {0, 0, 1, 1}};
    }
    for (uint32_t i = 1; i <= 96; ++i) {
        splats[i].mean_depth.x = 3;
        splats[i].mean_depth.y = 3;
    }
    auto input = [device newBufferWithBytes:splats.data() length:n * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
    auto queue = [device newCommandQueue];
    TileRasterizer raster(device);
    RasterFrame frame(device, 1, 1, n, n);
    auto command = [queue commandBuffer];
    raster.encode(command, {input}, n, RasterMode::Gaussian, {}, frame, {}, {}, {}, {}, false, true);
    const auto output = readback(device, command, frame);
    wait(command);
    require(frame.status().error == RasterError::None, "Weak transparent layer fixture overflow");
    double rgb[3] = {}, transmittance = 1;
    for (const auto& splat : splats) {
        const double a = double(splat.conic_opacity.w) * std::exp(-.5 * (double(splat.mean_depth.x) * splat.mean_depth.x + double(splat.mean_depth.y) * splat.mean_depth.y));
        if (a < .5 / 255)
            continue;
        for (uint32_t c = 0; c < 3; ++c)
            rgb[c] += double(splat.color[c]) * a * transmittance;
        transmittance *= 1. - a;
        if (transmittance < 1e-4)
            break;
    }
    const auto rgba = static_cast<const _Float16*>(output.color.contents);
    const auto depth = static_cast<const float*>(output.depth.contents);
    for (uint32_t c = 0; c < 3; ++c)
        require(std::abs(double(rgba[c]) - rgb[c]) < .003, "Weak transparent contributions were lost to half accumulator rounding");
    require(std::abs(depth[1] - (1 - transmittance)) < .0001, "Weak transparent layer coverage differs from the independent oracle");
    require(*static_cast<const uint32_t*>(output.pick.contents) == 0, "Weak layer compositing changed the nearest source ID");
    require(depth[3] == 5, "Weak layer compositing changed median depth");
    std::puts("2145 depth-ordered transparent splats preserve weak background radiance against an independent double oracle.");
}
static void compare_unaligned_macro_crop(id<MTLDevice> device) {
    constexpr uint32_t w = 129, h = 97, cw = 33, ch = 27, ox = 59, oy = 29;
    std::array<ProjectedSplat, 2> source{{{{63.4f, 31.4f, 5, 100}, {.16f, .015f, .12f, .8f}, {.55f, .2f, .8f, 25}, {0, 0, w, h}},
                                          {{72.2f, 41.1f, 6, 100}, {.12f, -.02f, .2f, .65f}, {.2f, .8f, .4f, 36}, {0, 0, w, h}}}};
    auto queue = [device newCommandQueue];
    TileRasterizer raster(device);
    RasterFrame full(device, w, h, 2, 126), crop(device, cw, ch, 2, 12);
    auto input = [device newBufferWithBytes:source.data() length:sizeof(source) options:MTLResourceStorageModeShared];
    auto command = [queue commandBuffer];
    raster.encode(command, {input}, 2, RasterMode::Gaussian, {}, full, {}, {}, {}, {}, false, true);
    const auto a = readback(device, command, full);
    wait(command);
    for (auto& splat : source) {
        splat.mean_depth.x -= ox;
        splat.mean_depth.y -= oy;
        splat.bounds = {0, 0, cw, ch};
    }
    std::memcpy(input.contents, source.data(), sizeof(source));
    OverlayBuffers overlay;
    overlay.render_origin = {float(ox), float(oy), 0, 0};
    command = [queue commandBuffer];
    raster.encode(command, {input}, 2, RasterMode::Gaussian, {}, crop, overlay, {}, {}, {}, false, true);
    const auto b = readback(device, command, crop);
    wait(command);
    require(full.status().error == RasterError::None && crop.status().error == RasterError::None, "Unaligned crop overflow");
    for (uint32_t y = 0; y < ch; ++y) {
        const auto same = [&](id<MTLBuffer> left, id<MTLBuffer> right, size_t left_stride, size_t right_stride, size_t bytes) {
            return std::memcmp(static_cast<const char*>(left.contents) + (y + oy) * left_stride + ox * bytes,
                               static_cast<const char*>(right.contents) + y * right_stride, cw * bytes) == 0;
        };
        require(same(a.color, b.color, a.color_stride, b.color_stride, 8), "Macro cache changed unaligned crop color/alpha");
        require(same(a.depth, b.depth, a.depth_stride, b.depth_stride, 16), "Macro cache changed unaligned crop depth");
        require(same(a.pick, b.pick, a.pick_stride, b.pick_stride, 4), "Macro cache changed unaligned crop IDs");
    }
}
// These pixels straddle the FP16 body admission threshold. Ring alpha is
// discontinuous, so accidentally accepting a rounded-down body contribution
// turns a negligible splat into an opaque edge.
static void compare_half_ring_threshold(id<MTLDevice> device) {
    const std::array<ProjectedSplat, 2> source = {{
        {{34.547904968f, 66.692443848f, 4.f, 3.f},
         {2.882635355f, .016116982f, 2.940344095f, .790994585f},
         {.4f, .2f, .1f, 16.f}, {0, 0, 128, 96}},
        {{82.307426453f, 71.455528259f, 4.f, 3.f},
         {2.921408415f, -.013935118f, 2.862745047f, .772662878f},
         {.4f, .2f, .1f, 16.f}, {0, 0, 128, 96}},
    }};
    const std::array<simd_uint2, 2> pixels = {{{33, 68}, {81, 73}}};
    std::array<simd_float4, 207> parameters{};
    parameters[21].w = .02f;
    parameters[22].x = 1.f;
    std::array<simd_float4, 128> colors{};
    const uint32_t flags = 0;
    OverlayBuffers overlay;
    overlay.parameters = {[device newBufferWithBytes:parameters.data() length:sizeof(parameters) options:MTLResourceStorageModeShared]};
    overlay.flags = {[device newBufferWithBytes:&flags length:sizeof(flags) options:MTLResourceStorageModeShared]};
    overlay.colors = {[device newBufferWithBytes:colors.data() length:sizeof(colors) options:MTLResourceStorageModeShared]};
    overlay.parameter_count = parameters.size();
    TileRasterizer raster(device);
    RasterFrame frame(device, 128, 96, 1, 48);
    auto queue = [device newCommandQueue];
    for (size_t n = 0; n < source.size(); ++n) {
        auto input = [device newBufferWithBytes:&source[n] length:sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
        auto command = [queue commandBuffer];
        raster.encode(command, {input}, 1, RasterMode::Gaussian, {}, frame, overlay, {}, {}, {}, false, true);
        const auto result = readback(device, command, frame);
        wait(command);
        require(frame.status().error == RasterError::None, "Half ring threshold fixture overflow");
        const auto pixel = pixels[n];
        const auto rgba = reinterpret_cast<const _Float16*>(static_cast<const char*>(result.color.contents) + pixel.y * result.color_stride) + pixel.x * 4;
        const auto pick = reinterpret_cast<const uint32_t*>(static_cast<const char*>(result.pick.contents) + pixel.y * result.pick_stride) + pixel.x;
        if (n == 0) {
            require(float(rgba[3]) == 0.f && *pick == 0xffffffff, "Rounded-down half alpha was promoted to an opaque ring");
        } else {
            require(std::abs(float(rgba[3]) - .8f) < .001f && *pick == 0, "Half body omitted the full-conic ring overlay");
        }
    }
}
static void run(id<MTLDevice> device) {
    TileRasterizer raster(device);
    auto queue = [device newCommandQueue];
    constexpr uint32_t width = 37, height = 29;
    const simd_float4 bg{.15f, .1f, .2f, .4f};
    // Reserve three scan levels, then alternate large, small, empty and fully
    // culled live domains without reallocating. The independent CPU oracle
    // catches stale histogram data and equal-depth instability across blocks.
    constexpr uint32_t capacity = 131073;
    RasterFrame frame(device, width, height, capacity, capacity * 6);
    std::mt19937 random(0x1939);
    uint32_t sequence = 0;
    for (uint32_t count : {capacity, capacity, capacity, capacity, 2049u, 1u, 2u, 0u, 255u, 256u, 257u, 2047u, 2048u, 4097u, capacity, 257u, 2u, 0u}) {
        // Repeat the large source extent to exercise depth-before-duplication,
        // then a fully culled frame and recovery without changing reservations.
        // The CPU oracle checks radial ties and original IDs in both sort paths.
        const bool fully_culled = sequence++ == 2;
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
            if (fully_culled || count == 2 || i % 19 == 3)
                s.bounds = {};
            else
                expected_instances += ((s.bounds.z + 15) / 16 - s.bounds.x / 16) * ((s.bounds.w + 15) / 16 - s.bounds.y / 16);
        }
        auto input = count ? [device newBufferWithBytes:splats.data()
                                                 length:count * sizeof(ProjectedSplat)
                                                options:MTLResourceStorageModeShared]
                           : nil;
        for (auto mode : {RasterMode::Gaussian, RasterMode::Points, RasterMode::Discs}) {
            if (count > 4097 && mode != RasterMode::Gaussian)
                continue;
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
    // Legacy desktop color saturation is independent of depth collection.
    // At this exact center, alpha=.95 then .999 leaves T<1e-4. The final
    // color is omitted only when compatibility is explicitly requested.
    const std::vector<ProjectedSplat> opaque_splats = {
        {{18, 14, 3, 3}, {1, 0, 1, .95f}, {.7f, .1f, .1f, 9}, {16, 12, 21, 17}},
        {{18, 14, 6, 3}, {1, 0, 1, .999f}, {.8f, .8f, .8f, 36}, {16, 12, 21, 17}}};
    auto opaque_input = [device newBufferWithBytes:opaque_splats.data() length:opaque_splats.size() * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
    Projection opaque_camera{};
    opaque_camera.rasterization = {1, 1, 100, 0};
    for (bool omit : {false, true}) {
        auto opaque_command = [queue commandBuffer];
        raster.encode(opaque_command, {opaque_input}, 2, RasterMode::Gaussian, bg, frame,
                      {}, {}, opaque_camera, {}, omit);
        const auto opaque_read = readback(device, opaque_command, frame);
        wait(opaque_command);
        const auto color = reinterpret_cast<const _Float16*>(static_cast<const char*>(opaque_read.color.contents) + 14 * opaque_read.color_stride) + 18 * 4;
        const auto depth = reinterpret_cast<const float*>(static_cast<const char*>(opaque_read.depth.contents) + 14 * opaque_read.depth_stride) + 18 * 4;
        const float trans = omit ? 1 - .95f : (1 - .95f) * (1 - .999f);
        for (int c = 0; c < 3; ++c) {
            const float expected = opaque_splats[0].color[c] * .95f +
                                   (omit ? 0 : .8f * (1 - .95f) * .999f) + bg[c] * bg.w * trans;
            require(std::abs(float(color[c]) - expected) < .001, "Saturation color compatibility differs");
        }
        require(std::abs(depth[0] - (3 * .95f + 6 * (1 - .95f) * .999f)) < 1e-5, "Saturation dropped expected-depth numerator");
        require(std::abs(depth[2] - (.95f + (1 - .95f) * .999f)) < 1e-6, "Saturation dropped expected-depth weight");
        require(std::abs(depth[1] - (1 - trans)) < 1e-6 && depth[3] == 3, "Saturation alpha or median differs");
    }
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
    // Input and intersection capacities are independent. Thousands of culled
    // sources can share one admitted instance; source presorting must never use
    // intersection scratch for an input extent larger than that reservation.
    constexpr uint32_t sparse_count = 4097;
    RasterFrame sparse(device, width, height, sparse_count, 1);
    std::vector<ProjectedSplat> sparse_sources(sparse_count, large);
    for (auto& splat : sparse_sources)
        splat.bounds = {};
    sparse_sources[0] = {{20, 4, 1, 3}, {1, 0, 1, .9f}, {1, 0, 0, 1}, {16, 0, 25, 9}};
    auto sparse_input = [device newBufferWithBytes:sparse_sources.data() length:sparse_sources.size() * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
    for (uint32_t phase = 0; phase < 4; ++phase) {
        if (phase == 2)
            sparse_sources[0].bounds = large.bounds;
        else if (phase == 3)
            sparse_sources[0].bounds = {};
        std::memcpy(sparse_input.contents, sparse_sources.data(), sparse_sources.size() * sizeof(ProjectedSplat));
        auto sparse_command = [queue commandBuffer];
        raster.encode(sparse_command, {sparse_input}, sparse_count, RasterMode::Gaussian, bg, sparse);
        const auto sparse_read = readback(device, sparse_command, sparse);
        wait(sparse_command);
        const auto status = sparse.status();
        require(status.required_instances == (phase == 2 ? 6u : phase == 3 ? 0u : 1u), "Sparse source/instance extent changed the exact count");
        require(status.error == (phase == 2 ? RasterError::InstanceCapacityExceeded : RasterError::None), "Sparse overflow/recovery status differs");
        compare(sparse_read, phase == 2 ? std::vector<ProjectedSplat>{} : sparse_sources, width, height, bg, RasterMode::Gaussian);
    }
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

// Narrow binning must preserve the actual pixels, exact depth and logical IDs.
// Exercise rotated anisotropy, mip compensation, partial tiles, clipping and
// poorly conditioned covariance against the original circular projection.
static void compare_tight_projection(id<MTLDevice> device) {
    constexpr uint32_t n = 257, w = 129, h = 97;
    SplatPreprocessor project(device);
    TileRasterizer raster(device);
    auto queue = [device newCommandQueue];
    std::vector<float> xyz(n * 3), logs(n * 3), rotations(n * 4, 0), opacity(n), dc(n * 3);
    for (uint32_t i = 0; i < n; ++i) {
        xyz[i * 3] = float(int(i % 17) - 8) * .09f;
        xyz[i * 3 + 1] = float(int(i % 13) - 6) * .08f;
        xyz[i * 3 + 2] = 2.f + float(i % 7) * .1f;
        logs[i * 3] = i % 11 == 0 ? 0.f : -1.5f;
        logs[i * 3 + 1] = i % 11 == 0 ? -12.f : -5.f;
        logs[i * 3 + 2] = -6.f;
        const float angle = float(i % 9) * .19f;
        rotations[i * 4] = std::cos(angle);
        rotations[i * 4 + 3] = std::sin(angle);
        opacity[i] = -5.f + float(i % 11);
        for (uint32_t c = 0; c < 3; ++c)
            dc[i * 3 + c] = float(int((i + c * 3) % 13) - 6) * .1f;
    }
    const auto upload = [&](const std::vector<float>& data) {
        return BufferSlice{[device newBufferWithBytes:data.data() length:data.size() * sizeof(float) options:MTLResourceStorageModeShared]};
    };
    SplatBuffers inputs;
    inputs.count = n;
    inputs.storage = ShStorage::CanonicalFloat32;
    inputs.means = upload(xyz);
    inputs.log_scales = upload(logs);
    inputs.rotations = upload(rotations);
    inputs.opacity_logits = upload(opacity);
    inputs.sh0 = upload(dc);
    const simd_float4 bg{.1f, .2f, .3f, 1};
    id<MTLBuffer> projected[2];
    for (auto& buffer : projected)
        buffer = [device newBufferWithLength:n * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
    RasterFrame frames[2] = {{device, w, h, n, n * 63}, {device, w, h, n, n * 63}};
    size_t reduced = 0;
    for (uint32_t scenario = 0; scenario < 8; ++scenario) {
        Projection camera{matrix_identity_float4x4, matrix_identity_float4x4, {},
                          {100, 100, 64.25f, 48.75f}, {.01f, 1000, 1, scenario % 2 ? 0.f : .3f},
                          {w, h, scenario / 4, scenario % 2}};
        if (scenario % 4 >= 2)
            camera.model_to_world.columns[3].x = .7f;
        Readback images[2];
        for (uint32_t variant = 0; variant < 2; ++variant) {
            auto command = [queue commandBuffer];
            project.encode(command, inputs, camera, 0, PrimitiveMode::Gaussian, {projected[variant]}, {}, {}, {}, {}, nullptr, variant != 0);
            raster.encode(command, {projected[variant]}, n, RasterMode::Gaussian, bg, frames[variant]);
            images[variant] = readback(device, command, frames[variant]);
            wait(command);
            require(frames[variant].status().error == RasterError::None, "Tight-bounds fixture overflowed");
        }
        const auto before = frames[0].status().required_instances;
        const auto after = frames[1].status().required_instances;
        require(after <= before, "Ellipse bounds expanded tile membership");
        reduced += before - after;
        for (uint32_t y = 0; y < h; ++y) {
            const auto identical = [&](id<MTLBuffer> a, id<MTLBuffer> b, size_t stride, size_t bytes) {
                return std::memcmp(static_cast<const char*>(a.contents) + y * stride,
                                   static_cast<const char*>(b.contents) + y * stride, bytes) == 0;
            };
            require(identical(images[0].color, images[1].color, images[0].color_stride, w * 8), "Ellipse bounds changed rendered RGBA");
            require(identical(images[0].depth, images[1].depth, images[0].depth_stride, w * 16), "Ellipse bounds changed exact depth");
            require(identical(images[0].pick, images[1].pick, images[0].pick_stride, w * 4), "Ellipse bounds changed source picking IDs");
        }
    }
    require(reduced > n, "Anisotropic bounds did not remove redundant tile instances");
    std::printf("Tight Gaussian bounds preserved all pixel/depth/pick bits; removed %zu tile instances.\n", reduced);
}


// The tile radix grows from one byte to two at 257 columns. Compare a fresh
// full-key frame with reused depth-before-duplication frames and an empty cut;
// keys must stay stable across radix parity changes and partial edge tiles.
static void compare_tile_key_widths(id<MTLDevice> device) {
    constexpr uint32_t n = 4097;
    TileRasterizer raster(device);
    auto queue = [device newCommandQueue];
    const simd_float4 bg{.1f, .2f, .3f, 1};
    for (uint32_t width : {4096u, 4097u}) {
        std::vector<ProjectedSplat> splats(n);
        for (uint32_t i = 0; i < n; ++i) {
            const uint32_t x = i % width;
            splats[i] = {{float(x), 0, float(i % 7 + 1), 35}, {.02f, 0, 1, .2f},
                         {float(i % 11) / 11, float(i % 13) / 13, float(i % 17) / 17, float(i % 5 + 1)},
                         {x > 35 ? x - 35 : 0, 0, std::min(width, x + 36), 1}};
        }
        auto input = [device newBufferWithBytes:splats.data() length:n * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
        RasterFrame frame(device, width, 1, n, n * 6);
        Readback reference{};
        uint64_t instances = 0;
        for (uint32_t count : {n, n, 0u, n, n}) {
            auto command = [queue commandBuffer];
            raster.encode(command, {input}, count, RasterMode::Gaussian, bg, frame);
            const auto actual = readback(device, command, frame);
            wait(command);
            require(frame.status().error == RasterError::None, "Tile-key boundary fixture overflowed");
            if (!count) {
                compare(actual, {}, width, 1, bg, RasterMode::Gaussian);
                continue;
            }
            if (!reference.color) {
                reference = actual;
                instances = frame.status().required_instances;
                require(instances > n * 5 / 4, "Tile-key fixture does not exercise source sorting");
                continue;
            }
            require(frame.status().required_instances == instances, "Tile-key width changed membership");
            require(std::memcmp(reference.color.contents, actual.color.contents, width * 8) == 0,
                    "Compact tile sorting changed RGBA across digit widths");
            require(std::memcmp(reference.depth.contents, actual.depth.contents, width * 16) == 0,
                    "Compact tile sorting changed exact depth across digit widths");
            require(std::memcmp(reference.pick.contents, actual.pick.contents, width * 4) == 0,
                    "Compact tile sorting changed stable source IDs across digit widths");
        }
    }
}


// GUT depth is the closest point on a 3D ray. Deliberately disagree with the
// projected center Z so the separate 2D Portal median cannot silently win.
static void compare_portal_gut_median(id<MTLDevice> device) {
    const ProjectedSplat splat{{0, 0, 17, 2}, {1, 0, 1, .75f}, {.2f, .3f, .4f, 9}, {0, 0, 1, 1}};
    const GutSplat geometry{{1, 0, 0, 1}, {0, 1, 0, 0}, {0, 0, 1, 2}, {0, 0, 3, .75f}};
    auto input = [device newBufferWithBytes:&splat length:sizeof(splat) options:MTLResourceStorageModeShared];
    auto gut = [device newBufferWithBytes:&geometry length:sizeof(geometry) options:MTLResourceStorageModeShared];
    auto queue = [device newCommandQueue];
    TileRasterizer raster(device);
    RasterFrame frame(device, 1, 1, 1, 1);
    Projection camera{matrix_identity_float4x4, matrix_identity_float4x4, {},
                      {1, 1, .5f, .5f}, {.01f, 100, 1, .3f}, {1, 1, 0, 0}};
    camera.rasterization.w = 1;
    for (auto model : {CameraModel::Perspective, CameraModel::Orthographic})
        for (bool exact : {false, true}) {
            camera.extent.z = uint32_t(model);
            auto command = [queue commandBuffer];
            raster.encode(command, {input}, 1, RasterMode::Gut, {0, 0, 0, 1}, frame, {}, {gut}, camera, {}, false, false, nullptr, exact);
            const auto actual = readback(device, command, frame);
            wait(command);
            const auto depth = static_cast<const float*>(actual.depth.contents);
            require(std::abs(depth[3] - 3.f) < 1e-6f, "Portal GUT median used the projected 2D center instead of the 3D ray depth");
            require(std::abs(depth[2] - 3.f) < 1e-6f && std::abs(depth[0] - 2.25f) < 1e-6f,
                    "Portal GUT nearest/weighted ray depth changed");
        }
}

int main() {
    @autoreleasepool {
        auto device = MTLCreateSystemDefaultDevice();
        if (!device)
            return LFS_METAL_TEST_REQUIRE_DEVICE ? 1 : 77;
        try {
            run(device);
            compare_tight_projection(device);
            compare_tile_key_widths(device);
            compare_portal_gut_median(device);
            compare_gut_culling(device);
            compare_dense_gut_subtiles(device);
            compare_depth_chunks(device);
            compare_unaligned_macro_crop(device);
            compare_weak_transparent_layers(device);
            compare_half_ring_threshold(device);
            return 0;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
}

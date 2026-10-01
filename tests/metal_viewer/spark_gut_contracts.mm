/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "tile_rasterizer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
using namespace lfs::rendering::metal;
static void require(bool value, const char* message) {
    if (!value)
        throw std::runtime_error(message);
}
static void run(id<MTLDevice> device) {
    SplatPreprocessor project(device);
    TileRasterizer raster(device);
    auto queue = [device newCommandQueue];
    const auto buffer = [&](const void* data, size_t bytes) { return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]; };
    const float means[]{0, 0, 3}, scales[]{std::log(.18f), std::log(.18f), std::log(.18f)}, rotation[]{1, 0, 0, 0};
    const float sh0[]{(.8f - .5f) / .28209479177387814f, (.4f - .5f) / .28209479177387814f, (.2f - .5f) / .28209479177387814f};
    uint32_t source = 0, logical = 127;
    auto xyz = buffer(means, sizeof(means)), scale = buffer(scales, sizeof(scales)), quat = buffer(rotation, sizeof(rotation)), rgb = buffer(sh0, sizeof(sh0));
    auto indices = buffer(&source, 4), ids = buffer(&logical, 4);
    auto projected = [device newBufferWithLength:sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
    auto gut = [device newBufferWithLength:sizeof(GutSplat) options:MTLResourceStorageModePrivate];
    size_t compared = 0;
    for (auto camera : {CameraModel::Perspective, CameraModel::Orthographic, CameraModel::Equirectangular}) {
        const uint32_t w = camera == CameraModel::Equirectangular ? 193 : 97, h = 97;
        const size_t stride = (size_t(w) * 8 + 255) / 256 * 256;
        auto read = [device newBufferWithLength:stride * h options:MTLResourceStorageModeShared];
        RasterFrame frame(device, w, h, 1, 256);
        Projection projection{matrix_identity_float4x4, matrix_identity_float4x4, {0, 0, 0, 0}, {64, 64, .5f * w, .5f * h}, {.01f, 100, 1, .3f}, {w, h, uint32_t(camera), 0}};
        projection.display.z = 1;
        projection.panorama = {float(w), float(h), 0, 0};
        for (bool mip : {false, true})
            for (float output_scale : {1.f, 2.f})
                for (float opacity : {.3f, 1.f, 1.5f, 3.f, 5.f})
                    for (float weight : {1.f, .6f, .15f}) {
                        projection.extent.w = uint32_t(mip);
                        projection.rasterization.x = output_scale;
                        projection.clip_scale.w = mip ? .1f : .3f;
                        const float encoded = opacity > 1 ? (opacity + 3) / 4 : opacity;
                        auto alpha = buffer(&encoded, 4), weights = buffer(&weight, 4);
                        SplatBuffers input{{xyz}, {scale}, {quat}, {alpha}, {rgb}, {}, {}, {}, 1, 0, ShStorage::CanonicalFloat32};
                        LodSelection cut;
                        cut.enabled = true;
                        cut.count = cut.source_count = 1;
                        cut.logical_count = 128;
                        cut.indices = {indices};
                        cut.logical_indices = {ids};
                        cut.weights = {weights};
                        auto command = [queue commandBuffer];
                        project.encode(command, input, projection, 0, PrimitiveMode::Gut, {projected}, {}, {}, {gut}, cut);
                        raster.encode(command, {projected}, 1, RasterMode::Gut, {}, frame, {}, {gut}, projection, cut);
                        auto blit = [command blitCommandEncoder];
                        [blit copyFromTexture:frame.color()
                                         sourceSlice:0
                                         sourceLevel:0
                                        sourceOrigin:MTLOriginMake(0, 0, 0)
                                          sourceSize:MTLSizeMake(w, h, 1)
                                            toBuffer:read
                                   destinationOffset:0
                              destinationBytesPerRow:stride
                            destinationBytesPerImage:stride * h];
                        [blit endEncoding];
                        [command commit];
                        [command waitUntilCompleted];
                        require(command.status == MTLCommandBufferStatusCompleted, "Spark GUT GPU command failed");
                        require(frame.status().error == RasterError::None, "Spark GUT capacity overflow");
                        const double effective = double(opacity) * weight;
                        const double power = effective > 1 ? .5 * std::pow(std::sqrt(8.) + .7 * (std::min(effective, 5.) - 1), 2) : std::max(4., std::log(std::max(effective, .5 / 255) * 510));
                        const double density = effective > 1 ? std::exp((effective * effective - 1) / std::exp(1.)) : 0;
                        for (uint32_t y = 0; y < h; ++y)
                            for (uint32_t x = 0; x < w; ++x) {
                                double distance2;
                                bool forward = true;
                                if (camera == CameraModel::Orthographic) {
                                    const double px = (double(x) + .5 - .5 * w) / 64, py = (double(y) + .5 - .5 * h) / 64;
                                    distance2 = (px * px + py * py) / (.18 * .18);
                                } else {
                                    double z;
                                    if (camera == CameraModel::Equirectangular) {
                                        const double longitude = ((double(x) + .5) / w - .5) * 2 * M_PI;
                                        const double latitude = ((double(y) + .5) / h - .5) * M_PI;
                                        z = std::cos(latitude) * std::cos(longitude);
                                        forward = z > 0;
                                    } else {
                                        const double px = (double(x) + .5 - .5 * w) / 64, py = (double(y) + .5 - .5 * h) / 64;
                                        z = 1 / std::sqrt(1 + px * px + py * py);
                                    }
                                    distance2 = 9 * (1 - z * z) / (.18 * .18);
                                }
                                const double value = std::exp(-.5 * distance2);
                                double expected = !forward || .5 * distance2 > power ? 0 : density ? -std::expm1(density * std::log1p(-value))
                                                                                                   : effective * value;
                                expected = std::min(expected, double(.999f));
                                if (expected < .5 / 255)
                                    expected = 0;
                                const auto actual = reinterpret_cast<const _Float16*>(static_cast<const char*>(read.contents) + y * stride) + x * 4;
                                if (std::abs(float(actual[3]) - expected) > .0022) {
                                    std::fprintf(stderr, "Spark GUT camera=%u opacity=%g weight=%g pixel=%u,%u q=%g actual=%g expected=%g\n",
                                                 uint32_t(camera), opacity, weight, x, y, distance2, float(actual[3]), expected);
                                    throw std::runtime_error("Spark GUT support/density differs from independent 3D ray alpha");
                                }
                                for (size_t c = 0; c < 3; ++c)
                                    require(std::abs(float(actual[c]) - expected * (c == 0 ? .8 : c == 1 ? .4
                                                                                                         : .2)) < .0022,
                                            "Spark GUT color differs");
                                ++compared;
                            }
                    }
    }
    std::printf("Spark GUT independent ray/density/LOD-weight contracts passed: %zu pixels\n", compared);
}
int main() {
    @autoreleasepool {
        auto device = MTLCreateSystemDefaultDevice();
        if (!device)
            return LFS_METAL_TEST_REQUIRE_DEVICE ? 1 : 77;
        try {
            run(device);
            return 0;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "%s\n", error.what());
            return 1;
        }
    }
}

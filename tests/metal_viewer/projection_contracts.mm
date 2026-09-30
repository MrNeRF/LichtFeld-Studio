/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "splat_preprocessor.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

using namespace lfs::rendering::metal;
static void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
static id<MTLBuffer> buffer(id<MTLDevice> device, const void* data, size_t size) {
    auto result = [device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
    require(result != nil, "Test allocation failed");
    return result;
}
template <class F>
static void reject(F&& f) {
    bool caught = false;
    try {
        f();
    } catch (const std::invalid_argument&) { caught = true; }
    require(caught, "Invalid input was not rejected");
}
static Projection frame() {
    return {matrix_identity_float4x4, matrix_identity_float4x4, {0, 0, 0, 0}, {200, 200, 128, 128}, {.01f, 1000, 1, .3f}, {256, 256, 0, 0}};
}

static void run(id<MTLDevice> device) {
    SplatPreprocessor pipeline(device);
    auto queue = [device newCommandQueue];
    size_t comparisons = 0;
    // Warp boundary, quantization block boundary, and partially filled last block.
    for (uint32_t n : {1u, 31u, 32u, 33u, 255u, 256u, 257u, 511u}) {
        std::vector<float> xyz(n * 3, 0), scales(n * 3, -3), rotations(n * 4, 0), opacity(n, 5), dc(n * 3, 0);
        for (uint32_t i = 0; i < n; ++i) {
            xyz[i * 3 + 2] = 3;
            rotations[i * 4] = 1;
        }
        SplatBuffers input;
        input.count = n;
        input.means = {buffer(device, xyz.data(), xyz.size() * 4)};
        input.log_scales = {buffer(device, scales.data(), scales.size() * 4)};
        input.rotations = {buffer(device, rotations.data(), rotations.size() * 4)};
        input.opacity_logits = {buffer(device, opacity.data(), opacity.size() * 4)};
        input.sh0 = {buffer(device, dc.data(), dc.size() * 4)};
        auto output = [device newBufferWithLength:n * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
        std::vector<float> bounds(((n + 255) / 256) * 2);
        for (size_t b = 0; b < bounds.size() / 2; ++b) {
            bounds[b * 2] = -.25f - float(b) / 8;
            bounds[b * 2 + 1] = .25f + float(b) / 4;
        }
        input.sh_bounds = {buffer(device, bounds.data(), bounds.size() * 4)};
        for (uint32_t rest : {3u, 8u, 15u}) {
            input.layout_rest = rest;
            const size_t padded = (n + 31) / 32 * 32, slots = (rest * 3 + 3) / 4;
            for (ShStorage storage : {ShStorage::CanonicalFloat32, ShStorage::SwizzledFloat32,
                                      ShStorage::SwizzledFloat16, ShStorage::Q16}) {
                input.storage = storage;
                std::vector<float> canonical(n * rest * 3), swizzled(padded * slots * 4, 0);
                std::vector<uint16_t> halves(swizzled.size(), 0), codes(padded * rest * 3, 0);
                for (uint32_t i = 0; i < n; ++i)
                    for (uint32_t c = 0; c < rest * 3; ++c) {
                        const uint16_t q = (i + c) % 3 == 0 ? 0 : (i + c) % 3 == 1 ? 65535
                                                                                   : 32768;
                        codes[size_t(i / 32) * (rest * 3 * 32) + c * 32 + i % 32] = q;
                        const float value = storage == ShStorage::Q16 ? std::fma(bounds[(i / 256) * 2 + 1] - bounds[(i / 256) * 2], float(q) * (1.f / 65535), bounds[(i / 256) * 2]) : float(int((i + c) % 9) - 4) / 32;
                        canonical[size_t(i) * rest * 3 + c] = value;
                        const size_t index = (size_t(i / 32) * slots * 32 + (c / 4) * 32 + i % 32) * 4 + c % 4;
                        swizzled[index] = value;
                        const _Float16 half = value;
                        std::memcpy(&halves[index], &half, 2);
                    }
                switch (storage) {
                case ShStorage::CanonicalFloat32: input.sh_rest = {buffer(device, canonical.data(), canonical.size() * 4)}; break;
                case ShStorage::SwizzledFloat32: input.sh_rest = {buffer(device, swizzled.data(), swizzled.size() * 4)}; break;
                case ShStorage::SwizzledFloat16: input.sh_rest = {buffer(device, halves.data(), halves.size() * 2)}; break;
                case ShStorage::Q16: input.sh_rest = {buffer(device, codes.data(), codes.size() * 2)}; break;
                case ShStorage::RadSigned8: throw std::logic_error("RAD has separate page-frame contracts");
                }
                for (uint32_t degree = 0; degree <= 3; ++degree) {
                    if ((degree + 1) * (degree + 1) - 1 > rest)
                        continue;
                    auto command = [queue commandBuffer];
                    pipeline.encode(command, input, frame(), degree, PrimitiveMode::Gaussian, {output});
                    [command commit];
                    [command waitUntilCompleted];
                    require(command.status == MTLCommandBufferStatusCompleted, "GPU projection failed");
                    auto result = static_cast<const ProjectedSplat*>(output.contents);
                    for (uint32_t i = 0; i < n; ++i) {
                        require(result[i].bounds.z > result[i].bounds.x, "Valid Gaussian was culled");
                        require(std::abs(result[i].mean_depth.x - 127.5f) < 1e-5, "Projection center mismatch");
                        for (uint32_t c = 0; c < 3; ++c) {
                            // Analytic SH on +Z: only m=0 survives, no copied shader basis.
                            const auto value = [&](uint32_t k) { return canonical[(size_t(i) * rest + k) * 3 + c]; };
                            double expected = .5;
                            if (degree >= 1)
                                expected += std::sqrt(3.0 / (4 * M_PI)) * value(1);
                            if (degree >= 2)
                                expected += std::sqrt(5.0 / (4 * M_PI)) * value(5);
                            if (degree >= 3)
                                expected += std::sqrt(7.0 / (4 * M_PI)) * value(11);
                            require(std::abs(result[i].color[c] - std::max(0.0, expected)) < 2e-6, "SH codec/color mismatch");
                            ++comparisons;
                        }
                    }
                }
            }
        }
        // SH0/point specialization must not require or read higher SH, scales or rotations.
        input.sh_rest = {};
        input.sh_bounds = {};
        input.log_scales = {};
        input.rotations = {};
        auto command = [queue commandBuffer];
        pipeline.encode(command, input, frame(), 0, PrimitiveMode::Points, {output});
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Point specialization read absent buffers");
        auto result = static_cast<const ProjectedSplat*>(output.contents);
        require(result[0].mean_depth.w == 2, "Point radius mismatch");
        reject([&] { pipeline.encode([queue commandBuffer], input, frame(), 1, PrimitiveMode::Points, { output }); });
        auto invalid = frame();
        invalid.clip_scale.y = invalid.clip_scale.x;
        reject([&] { pipeline.encode([queue commandBuffer], input, invalid, 0, PrimitiveMode::Points, { output }); });
        reject([&] { pipeline.encode([queue commandBuffer], input, frame(), 0, PrimitiveMode::Points, { output, 16 }); });
        auto clipped = frame();
        clipped.clip_scale.x = 4;
        command = [queue commandBuffer];
        pipeline.encode(command, input, clipped, 0, PrimitiveMode::Points, {output});
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Clipping dispatch failed");
        for (uint32_t i = 0; i < n; ++i)
            require(result[i].bounds.z == 0, "Culled output retained an earlier frame");
        // Object transform applies without changing source positions or selecting points.
        auto moved = frame();
        moved.model_to_world.columns[3].x = .3f;
        command = [queue commandBuffer];
        pipeline.encode(command, input, moved, 0, PrimitiveMode::Points, {output});
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Object transform dispatch failed");
        require(std::abs(result[0].mean_depth.x - 147.5f) < 1e-4, "Object transform/pixel offset was not applied");
        require(std::abs(result[0].color.w - 9.09f) < 1e-4, "Radial sort key was replaced by camera Z");
        require(static_cast<const float*>(input.means.buffer.contents)[0] == 0, "Object transform mutated source geometry");
        auto ortho = moved;
        ortho.extent.z = 1;
        command = [queue commandBuffer];
        pipeline.encode(command, input, ortho, 0, PrimitiveMode::Points, {output});
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Orthographic dispatch failed");
        require(std::abs(result[0].mean_depth.x - 187.5f) < 1e-4, "Orthographic projection used perspective division");
        std::array<SceneObject, 2> objects{{{matrix_identity_float4x4, {0, 0, 0, 0}, {1, 0, 0, 0}},
                                            {matrix_identity_float4x4, {0, 0, 0, 0}, {0, 0, 0, 0}}}};
        objects[0].model_to_world.columns[3].x = .6f;
        std::vector<uint32_t> object_indices(n);
        for (uint32_t i = 0; i < n; ++i)
            object_indices[i] = i % 3;
        SceneBuffers scene{{buffer(device, object_indices.data(), n * 4)},
                           {buffer(device, objects.data(), sizeof(objects))},
                           2};
        command = [queue commandBuffer];
        pipeline.encode(command, input, frame(), 0, PrimitiveMode::Points, {output}, scene);
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Scene object dispatch failed");
        for (uint32_t i = 0; i < n; ++i) {
            if (i % 3 == 0)
                require(std::abs(result[i].mean_depth.x - 167.5f) < 1e-4, "Scene transform not applied");
            else
                require(result[i].bounds.z == 0, "Invisible/invalid scene node was not culled");
        }
        // Large, near-camera splats must have the same bounded covariance as Studio.
        input.log_scales = {buffer(device, xyz.data(), xyz.size() * 4)};
        input.rotations = {buffer(device, rotations.data(), rotations.size() * 4)};
        command = [queue commandBuffer];
        pipeline.encode(command, input, frame(), 0, PrimitiveMode::Gaussian, {output});
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Extent clamp dispatch failed");
        require(result[0].mean_depth.w <= 512.501f, "Projected radius exceeds Studio's covariance cap");
        require(result[0].mean_depth.w > 0, "Extent-clamped splat was dropped");
        std::vector<uint8_t> deleted(n, 1);
        input.deleted = {buffer(device, deleted.data(), deleted.size())};
        command = [queue commandBuffer];
        pipeline.encode(command, input, frame(), 0, PrimitiveMode::Points, {output});
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Deleted-mask dispatch failed");
        for (uint32_t i = 0; i < n; ++i)
            require(result[i].bounds.z == 0, "Deleted splat contributed");
        auto empty = input;
        empty.count = 0;
        command = [queue commandBuffer];
        pipeline.encode(command, empty, frame(), 0, PrimitiveMode::Points, {});
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Empty scene dispatch failed");
    }
    // Double-precision spherical UT oracle: check actual projected centers,
    // including rear hemisphere, longitude seam and near-pole directions.
    auto panorama = frame();
    panorama.extent = {128, 96, uint32_t(CameraModel::Equirectangular), 0};
    panorama.panorama = {128, 96, 0, 0};
    const std::array<float, 3> logs{std::log(.05f), std::log(.07f), std::log(.08f)}, dc{};
    const std::array<float, 4> quaternion{1, 0, 0, 0};
    const float logit = 2;
    SplatBuffers spherical;
    spherical.count = 1;
    spherical.log_scales = {buffer(device, logs.data(), sizeof(logs))};
    spherical.rotations = {buffer(device, quaternion.data(), sizeof(quaternion))};
    spherical.opacity_logits = {buffer(device, &logit, sizeof(logit))};
    spherical.sh0 = {buffer(device, dc.data(), sizeof(dc))};
    auto projected = [device newBufferWithLength:sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
    auto geometry = [device newBufferWithLength:sizeof(GutSplat) options:MTLResourceStorageModeShared];
    const float lambda = .1f * .1f * 3.f - 3.f;
    const float denominator = 3.f + lambda;
    const double scale = std::sqrt(denominator), central = lambda / denominator, weight = 1.f / (2.f * denominator);
    for (const std::array<float, 3> mean : {std::array<float, 3>{1, .5f, 3}, {-1, -.5f, -3}, {.01f, .2f, -3}, {-.01f, .2f, -3}, {3, .1f, .2f}, {.1f, 3, .1f}}) {
        spherical.means = {buffer(device, mean.data(), sizeof(mean))};
        auto command = [queue commandBuffer];
        pipeline.encode(command, spherical, panorama, 0, PrimitiveMode::Gut, {projected}, {}, {}, {geometry});
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Spherical projection failed");
        std::array<std::array<double, 2>, 7> image{};
        for (size_t n = 0; n < image.size(); ++n) {
            std::array<double, 3> point{mean[0], mean[1], mean[2]};
            if (n) {
                const size_t axis = (n - 1) % 3;
                point[axis] += (n < 4 ? 1 : -1) * scale * std::exp(double(logs[axis]));
            }
            const double norm = std::sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
            image[n] = {(std::atan2(point[0], point[2]) / (2 * M_PI) + .5) * 128,
                        (std::asin(point[1] / norm) / M_PI + .5) * 96};
            if (n)
                image[n][0] -= 128 * std::round((image[n][0] - image[0][0]) / 128);
        }
        std::array<double, 2> expected{};
        for (size_t n = 0; n < image.size(); ++n)
            for (size_t c = 0; c < 2; ++c)
                expected[c] += (n ? weight : central) * image[n][c];
        expected[0] -= 128 * std::floor(expected[0] / 128);
        const auto result = static_cast<const ProjectedSplat*>(projected.contents);
        require(result->bounds.z > result->bounds.x, "Spherical projection rejected valid rear/polar geometry");
        for (size_t c = 0; c < 2; ++c)
            require(std::abs(result->mean_depth[c] - (expected[c] - .5)) < .005,
                    "Spherical UT center differs from the double-precision oracle");
        require(std::abs(result->mean_depth.z - std::sqrt(mean[0] * mean[0] + mean[1] * mean[1] + mean[2] * mean[2])) < 1e-5,
                "Spherical binning uses view Z instead of radial depth");
    }
    auto invalid_panorama = panorama;
    invalid_panorama.panorama.x = 0;
    reject([&] { pipeline.encode([queue commandBuffer], spherical, invalid_panorama, 0, PrimitiveMode::Gut, { projected }, {}, {}, {geometry}); });
    reject([&] { pipeline.encode([queue commandBuffer], spherical, panorama, 0, PrimitiveMode::Gaussian, { projected }); });
    const std::array<float, 3> nonfinite{NAN, 0, 3};
    spherical.means = {buffer(device, nonfinite.data(), sizeof(nonfinite))};
    auto invalid_command = [queue commandBuffer];
    pipeline.encode(invalid_command, spherical, panorama, 0, PrimitiveMode::Gut, {projected}, {}, {}, {geometry});
    [invalid_command commit];
    [invalid_command waitUntilCompleted];
    require(invalid_command.status == MTLCommandBufferStatusCompleted && static_cast<const ProjectedSplat*>(projected.contents)->bounds.z == 0,
            "Nonfinite spherical input retained an earlier projection");
    const std::array<float, 3> resident_mean{0, 0, 3};
    spherical.means = {buffer(device, resident_mean.data(), sizeof(resident_mean))};
    const std::array<uint32_t, 2> cut{0, 0xffffffffu}, logical{0, 0}, levels{1, 0};
    const std::array<float, 2> weights{.25f, 1};
    LodSelection lod{{buffer(device, cut.data(), sizeof(cut))}, {buffer(device, logical.data(), sizeof(logical))}, {buffer(device, levels.data(), sizeof(levels))}, {buffer(device, weights.data(), sizeof(weights))}, 2, 1, true, true};
    auto cut_output = [device newBufferWithLength:2 * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
    auto cut_camera = frame();
    auto cut_command = [queue commandBuffer];
    pipeline.encode(cut_command, spherical, cut_camera, 0, PrimitiveMode::Gaussian, {cut_output}, {}, {}, {}, lod);
    [cut_command commit];
    [cut_command waitUntilCompleted];
    const auto cut_result = static_cast<const ProjectedSplat*>(cut_output.contents);
    require(cut_command.status == MTLCommandBufferStatusCompleted && cut_result[0].bounds.z > 0 && cut_result[1].bounds.z == 0,
            "Resident LOD indirection retained an invalid physical index");
    require(std::abs(cut_result[0].conic_opacity.w - .25f / (1 + std::exp(-2.f))) < 1e-5f,
            "Resident LOD weights did not scale activated source opacity");
    require(cut_result[0].color.x == 0 && std::abs(cut_result[0].color.y - .5f) < 1e-5f && cut_result[0].color.z == 0,
            "LOD level palette did not use the selected-cut level");
    cut_camera.display.z = 1;
    cut_command = [queue commandBuffer];
    pipeline.encode(cut_command, spherical, cut_camera, 0, PrimitiveMode::Gaussian, {cut_output}, {}, {}, {}, lod);
    [cut_command commit];
    [cut_command waitUntilCompleted];
    const double rx = std::pow(double(cut_camera.intrinsics.x) * std::exp(double(logs[0])) / 3., 2);
    const double ry = std::pow(double(cut_camera.intrinsics.y) * std::exp(double(logs[1])) / 3., 2);
    const double expected_density_opacity = 1.25 * std::sqrt(rx * ry / ((rx + .3) * (ry + .3)));
    require(cut_command.status == MTLCommandBufferStatusCompleted && std::abs(cut_result[0].conic_opacity.w - expected_density_opacity) < 1e-5,
            "Spark compact 1..2 activation, transition weight or non-mip compensation differs");
    lod.indices.offset = 1;
    reject([&] { pipeline.encode([queue commandBuffer], spherical, cut_camera, 0, PrimitiveMode::Gaussian, { cut_output }, {}, {}, {}, lod); });
    std::printf("Metal projection contracts passed: %zu SH component comparisons; 4 storage formats; SH0-3; boundary, point and clipping checks.\n", comparisons);
}
int main() {
    @autoreleasepool {
        auto device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::puts("No Metal device");
            return LFS_METAL_TEST_REQUIRE_DEVICE ? 1 : 77;
        }
        try {
            run(device);
            return 0;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL: %s\n", error.what());
            return 1;
        }
    }
}

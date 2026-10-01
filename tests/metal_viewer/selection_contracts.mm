/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "selection_query.hpp"
#include <algorithm>
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
    SelectionQuery query(device);
    auto queue = [device newCommandQueue];
    const auto buffer = [&](const void* data, size_t bytes) { return [device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared]; };
    constexpr uint32_t count = 259, width = 97, height = 73;
    std::vector<float> xyz(count * 3), scale(count * 3, std::log(.1f)), quaternion(count * 4), opacity(count, -100.f);
    std::vector<uint8_t> deleted(count), visibility{1, 0};
    std::vector<int32_t> indices(count);
    for (uint32_t n = 0; n < count; ++n) {
        xyz[n * 3] = float(int(n % 17) - 8) * .06f;
        xyz[n * 3 + 1] = float(int(n / 17) - 7) * .06f;
        xyz[n * 3 + 2] = 3;
        quaternion[n * 4] = 1;
        indices[n] = n % 5 == 0 ? 1 : 0;
        deleted[n] = n % 11 == 0;
    }
    xyz[(count - 1) * 3 + 2] = -1;
    std::array<simd_float4x4, 2> matrices{matrix_identity_float4x4, matrix_identity_float4x4};
    matrices[0].columns[3].x = .125f;
    matrices[1].columns[3].x = -.125f;
    auto means = buffer(xyz.data(), xyz.size() * 4), scales = buffer(scale.data(), scale.size() * 4), rotations = buffer(quaternion.data(), quaternion.size() * 4), alpha = buffer(opacity.data(), opacity.size() * 4);
    auto delete_mask = buffer(deleted.data(), count), objects = buffer(matrices.data(), sizeof(matrices)), object_indices = buffer(indices.data(), count * 4), node_mask = buffer(visibility.data(), visibility.size());
    auto output = [device newBufferWithLength:count options:MTLResourceStorageModeShared];
    auto coverage = [device newBufferWithLength:width * height options:MTLResourceStorageModePrivate];
    auto pick = [device newBufferWithLength:8 options:MTLResourceStorageModeShared];
    // Fractional boundaries keep these geometry oracles away from the known
    // FP32 UT cancellation at exactly symmetric integer pixel coordinates.
    const std::array<simd_float4, 2> primitives{simd_float4{47.75f, 35.75f, 75.25f, 55.25f}, simd_float4{20.25f, 20.25f, 60.25f, 40.25f}};
    const std::array<simd_float2, 5> polygon{simd_float2{25, 20}, simd_float2{75, 20}, simd_float2{55, 40}, simd_float2{75, 60}, simd_float2{25, 55}};
    auto shape = buffer(primitives.data(), sizeof(primitives)), vertices = buffer(polygon.data(), sizeof(polygon));
    SelectionBuffers buffers{{means}, {scales}, {rotations}, {alpha}, {delete_mask}, {objects}, {object_indices}, {node_mask}, {shape}, {vertices}, {coverage}, {output}, {pick}};
    size_t compared = 0;
    for (bool gut : {false, true})
        for (auto camera : {CameraModel::Perspective, CameraModel::Orthographic, CameraModel::Equirectangular}) {
            if (!gut && camera == CameraModel::Equirectangular)
                continue;
            for (auto selected_shape : {SelectionShape::Brush, SelectionShape::Rectangle, SelectionShape::Polygon}) {
                SelectionParameters p;
                p.intrinsics = {48, 48, 48.5f, 36.5f};
                p.image = {width, height, uint32_t(camera), uint32_t(gut)};
                p.source = {count, uint32_t(selected_shape), uint32_t(primitives.size()), uint32_t(polygon.size())};
                p.scene = {2, 1, 2, count};
                p.aabb = {0, 0, width, height};
                auto command = [queue commandBuffer];
                query.encode(command, buffers, p);
                [command commit];
                [command waitUntilCompleted];
                require(command.status == MTLCommandBufferStatusCompleted, command.error.localizedDescription.UTF8String ?: "Selection GPU command failed");
                const auto actual = static_cast<const uint8_t*>(output.contents);
                const auto project = [&](double x, double y, double z) -> std::array<double, 2> {
                    if (camera == CameraModel::Equirectangular) {
                        const double length = std::sqrt(x * x + y * y + z * z);
                        return {(std::atan2(x, z) / (2 * M_PI) + .5) * width, (std::asin(y / length) / M_PI + .5) * height};
                    }
                    return {48 * x / (camera == CameraModel::Orthographic ? 1 : z) + 48.5, 48 * y / (camera == CameraModel::Orthographic ? 1 : z) + 36.5};
                };
                for (uint32_t n = 0; n < count; ++n) {
                    bool expected = !deleted[n] && visibility[indices[n]] && xyz[n * 3 + 2] > 0;
                    const double x = double(xyz[n * 3]) + (indices[n] ? -.125 : .125), y = xyz[n * 3 + 1], z = xyz[n * 3 + 2];
                    auto center = project(x, y, z);
                    if (gut && expected) {
                        const float lambda = .1f * .1f * 3.f - 3.f, den = 3.f + lambda;
                        const double w0 = lambda / den, wn = 1.f / (2.f * den), sigma = std::sqrt(double(den)) * std::exp(double(scale[n * 3]));
                        auto origin = center;
                        center = {w0 * origin[0], w0 * origin[1]};
                        for (int axis = 0; axis < 3; ++axis)
                            for (double sign : {-1., 1.}) {
                                auto point = project(x + (axis == 0 ? sign * sigma : 0), y + (axis == 1 ? sign * sigma : 0), z + (axis == 2 ? sign * sigma : 0));
                                if (camera == CameraModel::Equirectangular)
                                    point[0] -= width * std::round((point[0] - origin[0]) / width);
                                expected &= point[0] >= -.1 * width && point[0] < 1.1 * width && point[1] >= -.1 * height && point[1] < 1.1 * height;
                                center[0] += wn * point[0];
                                center[1] += wn * point[1];
                            }
                        if (camera == CameraModel::Equirectangular)
                            center[0] -= width * std::floor(center[0] / width);
                    }
                    center[0] -= .5;
                    center[1] -= .5;
                    bool inside = false;
                    if (selected_shape == SelectionShape::Polygon) {
                        const double px = std::floor(center[0]) + .5, py = std::floor(center[1]) + .5;
                        size_t previous = polygon.size() - 1;
                        for (size_t v = 0; v < polygon.size(); ++v) {
                            const auto a = polygon[v], b = polygon[previous];
                            if ((a.y > py) != (b.y > py)) {
                                const double edge = (double(b.x) - a.x) * (py - a.y) / (double(b.y) - a.y) + a.x;
                                if (px < edge)
                                    inside = !inside;
                            }
                            previous = v;
                        }
                    } else
                        for (const auto primitive : primitives) {
                            const double dx = center[0] - primitive.x, dy = center[1] - primitive.y;
                            inside |= selected_shape == SelectionShape::Brush ? dx * dx + dy * dy <= primitive.z : center[0] >= primitive.x && center[0] <= primitive.z && center[1] >= primitive.y && center[1] <= primitive.w;
                        }
                    if (bool(actual[n]) != (expected && inside)) {
                        std::fprintf(stderr, "Selection gut=%d camera=%u shape=%u source=%u center=%g,%g actual=%u expected=%d\n", gut, uint32_t(camera), uint32_t(selected_shape), n, center[0], center[1], actual[n], expected && inside);
                        throw std::runtime_error("Native selection differs from independent center/polygon contract");
                    }
                    ++compared;
                }
            }
        }
    // Ring selection picks the nearest hit, ties choose the lowest source ID.
    const float tie_means[]{0, 0, 3, 0, 0, 3, 0, 0, 4}, tie_scales[]{-2, -2, -2, -2, -2, -2, -2, -2, -2}, tie_quat[]{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, tie_opacity[]{2, 2, 2};
    auto tie_xyz = buffer(tie_means, sizeof(tie_means)), tie_scale = buffer(tie_scales, sizeof(tie_scales)), tie_rotation = buffer(tie_quat, sizeof(tie_quat)), tie_alpha = buffer(tie_opacity, sizeof(tie_opacity));
    const double covariance = std::pow(48 * std::exp(-2.) / 3, 2) + .3, opac = 1 / (1 + std::exp(-2.));
    const float distance = float(std::sqrt(2 * covariance * std::log(opac * 510)));
    const simd_float4 ring_primitive{48 + distance, 36, 0, 0};
    auto ring_shape = buffer(&ring_primitive, sizeof(ring_primitive));
    SelectionBuffers ring_buffers{{tie_xyz}, {tie_scale}, {tie_rotation}, {tie_alpha}, {}, {}, {}, {}, {ring_shape}, {}, {}, {output}, {pick}};
    SelectionParameters ring;
    ring.intrinsics = {48, 48, 48.5f, 36.5f};
    ring.image = {width, height, 0, 0};
    ring.source = {3, uint32_t(SelectionShape::Ring), 1, 0};
    for (uint32_t removed : {0xffffffffu, 0u}) {
        const uint8_t values[]{uint8_t(removed == 0), 0, 0};
        auto deletion = buffer(values, 3);
        ring_buffers.deleted = {deletion};
        ring.scene.w = 3;
        auto command = [queue commandBuffer];
        query.encode(command, ring_buffers, ring);
        [command commit];
        [command waitUntilCompleted];
        require(command.status == MTLCommandBufferStatusCompleted, "Native ring selection failed");
        const auto result = static_cast<const uint32_t*>(pick.contents);
        require(result[1] == (removed == 0 ? 1 : 0), "Native ring depth/source tie break differs");
        const auto mask = static_cast<const uint8_t*>(output.contents);
        require(mask[result[1]] == 1 && mask[2] == 0 && unsigned(mask[0]) + unsigned(mask[1]) == 1, "Native ring output contains multiple hits");
    }
    // More than 2048 vertices exercises both cached and global polygon edges.
    // This is a screen-space coverage oracle independent of Gaussian projection.
    std::vector<simd_float2> large_polygon(4099);
    for (size_t n = 0; n < large_polygon.size(); ++n) {
        const double angle = 2 * M_PI * double(n) / large_polygon.size();
        large_polygon[n] = {float(48.25 + 27.1 * std::cos(angle)), float(36.75 + 19.3 * std::sin(angle))};
    }
    auto large_vertices = buffer(large_polygon.data(), large_polygon.size() * sizeof(simd_float2));
    auto polygon_buffers = buffers;
    polygon_buffers.polygon_vertices = {large_vertices};
    auto large_coverage = [device newBufferWithLength:width * height options:MTLResourceStorageModeShared];
    polygon_buffers.polygon_mask = {large_coverage};
    SelectionParameters large;
    large.intrinsics = {48, 48, 48.5f, 36.5f};
    large.image = {width, height, 0, 0};
    large.source = {count, uint32_t(SelectionShape::Polygon), 0, uint32_t(large_polygon.size())};
    large.scene = {2, 1, 2, count};
    large.aabb = {0, 0, width, height};
    auto large_command = [queue commandBuffer];
    query.encode(large_command, polygon_buffers, large);
    [large_command commit];
    [large_command waitUntilCompleted];
    require(large_command.status == MTLCommandBufferStatusCompleted, "Large native polygon query failed");
    for (uint32_t y = 0; y < height; ++y)
        for (uint32_t x = 0; x < width; ++x) {
            const double px = x + .5, py = y + .5;
            bool expected = false;
            size_t previous = large_polygon.size() - 1;
            for (size_t n = 0; n < large_polygon.size(); ++n) {
                const auto a = large_polygon[n], b = large_polygon[previous];
                if ((a.y > py) != (b.y > py) && px < (double(b.x) - a.x) * (py - a.y) / (double(b.y) - a.y) + a.x)
                    expected = !expected;
                previous = n;
            }
            require(bool(static_cast<const uint8_t*>(large_coverage.contents)[y * width + x]) == expected, "Large polygon cached/global edge coverage differs");
            ++compared;
        }
    bool rejected = false;
    auto bad = ring_buffers;
    bad.output = {pick, 7};
    auto command = [queue commandBuffer];
    try {
        query.encode(command, bad, ring);
    } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "Native query accepted a short output buffer");
    std::printf("Native selection contracts passed: %zu source checks, polygon coverage, ring depth/ties and malformed buffers\n", compared);
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

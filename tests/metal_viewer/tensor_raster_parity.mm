/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// The single-source Slang rasterizer passes, run through the tensor library on
// Metal, against the native Metal kernels they replace, on identical inputs.
#include "core/gpu_kernel_module.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "splat_preprocessor.hpp"
#include "splat_project.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <format>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace lfs::rendering::metal;
using lfs::core::DataType;
using lfs::core::Device;
using lfs::core::GpuBackend;
using lfs::core::Tensor;
using M = lfs::core::GpuKernelModule;

namespace {
    void require(bool condition, const std::string& message) {
        if (!condition)
            throw std::runtime_error(message);
    }
    template <class T>
    id<MTLBuffer> native_buffer(id<MTLDevice> device, const std::vector<T>& values) {
        auto result = [device newBufferWithBytes:values.data() length:std::max<size_t>(16, values.size() * sizeof(T))
                                         options:MTLResourceStorageModeShared];
        require(result != nil, "Native allocation failed");
        return result;
    }
    template <class T>
    Tensor tensor(const std::vector<T>& values) {
        std::vector<T> padded = values;
        if (padded.size() * sizeof(T) < 16)
            padded.resize((16 + sizeof(T) - 1) / sizeof(T));
        return Tensor::from_blob(padded.data(), {padded.size() * sizeof(T)}, Device::CPU, DataType::UInt8).to(Device::GPU);
    }
    template <class T>
    std::vector<T> download(const Tensor& source, size_t count) {
        auto host = source.to(Device::CPU);
        std::vector<T> result(count);
        std::memcpy(result.data(), host.data_ptr(), count * sizeof(T));
        return result;
    }

    struct Scene {
        uint32_t count = 0;
        std::vector<float> means, scales, rotations, opacity, sh0, rest;
        std::vector<uint16_t> q16;
        std::vector<float> q16_bounds;
    };
    // Random splats around a camera at the origin looking down +Z.
    Scene make_scene(uint32_t count, uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> unit(-1.f, 1.f), depth(0.5f, 30.f), log_scale(-5.f, -0.5f), logit(-4.f, 6.f);
        Scene s;
        s.count = count;
        for (uint32_t i = 0; i < count; ++i) {
            const float z = depth(rng);
            s.means.insert(s.means.end(), {unit(rng) * z * .8f, unit(rng) * z * .6f, i % 53 == 0 ? -z : z});
            s.scales.insert(s.scales.end(), {log_scale(rng), log_scale(rng), log_scale(rng)});
            s.rotations.insert(s.rotations.end(), {unit(rng), unit(rng), unit(rng), unit(rng)});
            s.opacity.push_back(logit(rng));
            s.sh0.insert(s.sh0.end(), {unit(rng), unit(rng), unit(rng)});
            for (int c = 0; c < 45; ++c)
                s.rest.push_back(unit(rng) * .3f);
        }
        // Q16: 32-row cell swizzle with per-256 bounds (core/sh_value_codec.cuh).
        const size_t padded = (count + 31) / 32 * 32;
        s.q16.assign(padded * 45, 0);
        for (size_t b = 0; b < (count + 255) / 256; ++b)
            s.q16_bounds.insert(s.q16_bounds.end(), {-.4f - float(b % 3) * .1f, .4f + float(b % 5) * .05f});
        for (uint32_t i = 0; i < count; ++i)
            for (uint32_t c = 0; c < 45; ++c)
                s.q16[size_t(i / 32) * (45 * 32) + c * 32 + i % 32] = uint16_t(rng() & 0xffff);
        return s;
    }

    struct Config {
        const char* name;
        CameraModel camera;
        PrimitiveMode mode;
        ShStorage storage;
        uint32_t degree;
        bool mip;
        bool tight;
    };

    Projection make_projection(const Config& config) {
        Projection projection{};
        projection.model_to_world = matrix_identity_float4x4;
        projection.world_to_camera = matrix_identity_float4x4;
        // A slight camera rotation and translation exercise every matrix path.
        const float a = .1f;
        projection.world_to_camera.columns[0] = simd_make_float4(std::cos(a), 0, -std::sin(a), 0);
        projection.world_to_camera.columns[2] = simd_make_float4(std::sin(a), 0, std::cos(a), 0);
        projection.world_to_camera.columns[3] = simd_make_float4(.2f, -.1f, .3f, 1);
        projection.camera_local = simd_make_float4(-.2f, .1f, -.3f, 0);
        projection.intrinsics = simd_make_float4(900, 900, 640, 360);
        if (config.camera == CameraModel::Orthographic)
            projection.intrinsics = simd_make_float4(60, 60, 640, 360);
        projection.clip_scale = simd_make_float4(kViewerNearClip, 1000, 1, .3f);
        projection.extent = simd_make_uint4(1280, 720, uint32_t(config.camera), config.mip ? 1u : 0u);
        projection.rasterization = simd_make_float4(1, 0, 0, 0);
        projection.display = simd_make_float4(0, 1, 0, 0);
        if (config.camera == CameraModel::Equirectangular)
            projection.panorama = simd_make_float4(1280, 720, 0, 0);
        return projection;
    }

    struct Diff {
        size_t compared = 0, bounds_mismatch = 0, culled_mismatch = 0;
        uint32_t max_ulp = 0;
    };
    uint32_t ulp(float a, float b) {
        if (a == b || (std::isnan(a) && std::isnan(b)))
            return 0;
        const int32_t ia = std::bit_cast<int32_t>(a), ib = std::bit_cast<int32_t>(b);
        const int64_t oa = ia < 0 ? int64_t(INT32_MIN) - ia : ia, ob = ib < 0 ? int64_t(INT32_MIN) - ib : ib;
        return uint32_t(std::min<int64_t>(std::llabs(oa - ob), UINT32_MAX));
    }

    Diff compare(const std::vector<ProjectedSplat>& native, const std::vector<ProjectedSplat>& slang) {
        Diff diff;
        for (size_t i = 0; i < native.size(); ++i) {
            const auto& a = native[i];
            const auto& b = slang[i];
            const bool a_culled = a.bounds.x >= a.bounds.z || a.bounds.y >= a.bounds.w;
            const bool b_culled = b.bounds.x >= b.bounds.z || b.bounds.y >= b.bounds.w;
            if (a_culled != b_culled) {
                ++diff.culled_mismatch;
                continue;
            }
            if (a_culled)
                continue;
            ++diff.compared;
            if (!simd_equal(a.bounds, b.bounds))
                ++diff.bounds_mismatch;
            for (const auto& [x, y] : {std::pair{a.mean_depth, b.mean_depth}, std::pair{a.conic_opacity, b.conic_opacity}, std::pair{a.color, b.color}})
                for (int c = 0; c < 4; ++c)
                    diff.max_ulp = std::max(diff.max_ulp, ulp(x[c], y[c]));
        }
        return diff;
    }

    int run(id<MTLDevice> device) {
        const lfs::core::GpuBackendScope scope(GpuBackend::Metal);
        auto module = M::load(splat_project_entries(), GpuBackend::Metal);
        require(bool(module), "Slang projection did not load");
        SplatPreprocessor native(device);
        auto queue = [device newCommandQueue];
        const uint32_t count = 6000;
        const auto scene = make_scene(count, 7);
        const std::array configs{
            Config{"perspective-sh0", CameraModel::Perspective, PrimitiveMode::Gaussian, ShStorage::CanonicalFloat32, 0, false, false},
            Config{"perspective-sh3", CameraModel::Perspective, PrimitiveMode::Gaussian, ShStorage::CanonicalFloat32, 3, false, false},
            Config{"perspective-q16-mip-tight", CameraModel::Perspective, PrimitiveMode::Gaussian, ShStorage::Q16, 3, true, true},
            Config{"orthographic", CameraModel::Orthographic, PrimitiveMode::Gaussian, ShStorage::CanonicalFloat32, 2, false, false},
            Config{"equirectangular", CameraModel::Equirectangular, PrimitiveMode::Gaussian, ShStorage::CanonicalFloat32, 1, false, false},
            Config{"gut", CameraModel::Perspective, PrimitiveMode::Gut, ShStorage::CanonicalFloat32, 3, false, false},
            Config{"discs", CameraModel::Perspective, PrimitiveMode::Discs, ShStorage::CanonicalFloat32, 0, false, false},
            Config{"points", CameraModel::Perspective, PrimitiveMode::Points, ShStorage::CanonicalFloat32, 0, false, false},
        };
        int failures = 0;
        for (const auto& config : configs) {
            const bool q16 = config.storage == ShStorage::Q16;
            const auto projection = make_projection(config);
            // Native kernels.
            SplatBuffers in;
            in.count = count;
            in.layout_rest = config.degree ? 15 : 0;
            in.storage = config.storage;
            in.means = {native_buffer(device, scene.means)};
            in.log_scales = {native_buffer(device, scene.scales)};
            in.rotations = {native_buffer(device, scene.rotations)};
            in.opacity_logits = {native_buffer(device, scene.opacity)};
            in.sh0 = {native_buffer(device, scene.sh0)};
            in.sh_rest = {q16 ? native_buffer(device, scene.q16) : native_buffer(device, scene.rest)};
            in.sh_bounds = {native_buffer(device, scene.q16_bounds)};
            auto output = [device newBufferWithLength:count * sizeof(ProjectedSplat) options:MTLResourceStorageModeShared];
            auto gut = [device newBufferWithLength:count * sizeof(GutSplat) options:MTLResourceStorageModeShared];
            native.prepare(config.storage, config.degree, config.mode, config.tight);
            auto command = [queue commandBuffer];
            native.encode(command, in, projection, config.degree, config.mode, {output}, {}, {},
                          config.mode == PrimitiveMode::Gut ? BufferSlice{gut} : BufferSlice{}, {}, nullptr, config.tight);
            [command commit];
            [command waitUntilCompleted];
            require(command.status == MTLCommandBufferStatusCompleted, "Native projection failed");
            std::vector<ProjectedSplat> expected(count);
            std::memcpy(expected.data(), output.contents, count * sizeof(ProjectedSplat));

            // Slang through the tensor library.
            const std::array<uint32_t, 12> layout{count, in.layout_rest, 0, 0, 0, 0, 0, count, 0, 0, count, count};
            const auto means = tensor(scene.means), scales = tensor(scene.scales), rotations = tensor(scene.rotations);
            const auto opacity = tensor(scene.opacity), sh0 = tensor(scene.sh0);
            const auto rest = q16 ? tensor(scene.q16) : tensor(scene.rest), bounds = tensor(scene.q16_bounds);
            const auto frame = tensor(std::vector<Projection>{projection});
            const auto layout_tensor = tensor(std::vector<uint32_t>(layout.begin(), layout.end()));
            auto projected = Tensor::zeros({count * sizeof(ProjectedSplat)}, Device::GPU, DataType::UInt8);
            auto gut_projected = Tensor::zeros({count * sizeof(GutSplat)}, Device::GPU, DataType::UInt8);
            struct Parameters {
                uint64_t pointers[22] = {};
                uint32_t sh_storage, sh_degree, primitive_mode, tight_bounds;
            } parameters{.sh_storage = uint32_t(config.storage), .sh_degree = config.degree,
                         .primitive_mode = uint32_t(config.mode), .tight_bounds = config.tight ? 1u : 0u};
            const std::array bindings{
                M::Binding{0, &means}, M::Binding{8, &scales}, M::Binding{16, &rotations}, M::Binding{24, &opacity},
                M::Binding{32, &sh0}, M::Binding{40, &rest}, M::Binding{48, &bounds}, M::Binding{56, nullptr},
                M::Binding{64, &projected, M::Access::ReadWrite}, M::Binding{72, &frame}, M::Binding{80, &layout_tensor},
                M::Binding{88, nullptr}, M::Binding{96, nullptr}, M::Binding{104, nullptr}, M::Binding{112, nullptr},
                M::Binding{120, nullptr}, M::Binding{128, &gut_projected, M::Access::ReadWrite}, M::Binding{136, nullptr},
                M::Binding{144, nullptr}, M::Binding{152, nullptr}, M::Binding{160, nullptr}, M::Binding{168, nullptr}};
            auto dispatched = (*module)->dispatch({.function = "project_splats",
                                                   .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                                                   .groups = {M::groups_for(count, 256), 1, 1},
                                                   .group = {256, 1, 1}});
            require(bool(dispatched), std::format("{}: dispatch failed: {}", config.name, dispatched ? "" : dispatched.error().detail()));
            const auto actual = download<ProjectedSplat>(projected, count);
            const auto diff = compare(expected, actual);
            const bool ok = diff.culled_mismatch == 0 && diff.bounds_mismatch == 0 && diff.max_ulp <= 4;
            std::printf("%-28s compared=%zu culled_mismatch=%zu bounds_mismatch=%zu max_ulp=%u %s\n", config.name,
                        diff.compared, diff.culled_mismatch, diff.bounds_mismatch, diff.max_ulp, ok ? "ok" : "FAIL");
            failures += ok ? 0 : 1;
        }
        return failures ? 1 : 0;
    }
} // namespace

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device || !lfs::core::gpu_backend_available(GpuBackend::Metal)) {
            std::puts("Metal tensor backend unavailable");
            return LFS_METAL_TEST_REQUIRE_DEVICE ? 1 : 77;
        }
        try {
            return run(device);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "tensor raster parity: %s\n", error.what());
            return 1;
        }
    }
}

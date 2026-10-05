/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
// The single-source Slang rasterizer passes, run through the tensor library on
// Metal, against the native Metal kernels they replace, on identical inputs.
#include "core/gpu_kernel_module.hpp"
#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "splat_preprocessor.hpp"
#include "splat_project.hpp"
#include "gpu_profile.hpp"
#include "splat_rasterizer.hpp"
#include "splat_tile_binner.hpp"
#include "tile_rasterizer.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <memory>
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
        // Worst relative error per field (mean_depth, conic_opacity, color) and component.
        std::array<std::array<double, 4>, 3> relative{};
        std::array<std::array<std::pair<float, float>, 4>, 3> worst{};
        size_t worst_index = 0;
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
            const std::array fields{std::pair{a.mean_depth, b.mean_depth}, std::pair{a.conic_opacity, b.conic_opacity}, std::pair{a.color, b.color}};
            for (size_t f = 0; f < fields.size(); ++f)
                for (int c = 0; c < 4; ++c) {
                    const float x = fields[f].first[c], y = fields[f].second[c];
                    diff.max_ulp = std::max(diff.max_ulp, ulp(x, y));
                    // The conic's off-diagonal is judged against its diagonal scale.
                    const double magnitude = f == 1 && c == 1 ? std::sqrt(std::fabs(double(fields[f].first[0]) * fields[f].first[2]))
                                                              : std::max(std::fabs(double(x)), std::fabs(double(y)));
                    const double scale = std::max(magnitude, 1e-6);
                    const double relative = std::fabs(double(x) - double(y)) / scale;
                    if (relative > diff.relative[f][c]) {
                        diff.relative[f][c] = relative;
                        diff.worst[f][c] = {x, y};
                        if (f == 1 && c == 0)
                            diff.worst_index = i;
                    }
                }
        }
        return diff;
    }

    // Reads a private texture through a shared staging buffer.
    std::vector<uint8_t> read_texture(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLTexture> texture, size_t pixel_bytes) {
        const size_t row = texture.width * pixel_bytes, bytes = row * texture.height;
        auto staging = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        auto command = [queue commandBuffer];
        auto blit = [command blitCommandEncoder];
        [blit copyFromTexture:texture sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                   sourceSize:MTLSizeMake(texture.width, texture.height, 1) toBuffer:staging destinationOffset:0
          destinationBytesPerRow:row destinationBytesPerImage:bytes];
        [blit endEncoding];
        [command commit];
        [command waitUntilCompleted];
        std::vector<uint8_t> result(bytes);
        std::memcpy(result.data(), staging.contents, bytes);
        return result;
    }

    // Full frames: native preprocessor + TileRasterizer against the Slang
    // projection + SplatRasterizer, on the same scene and camera.
    // Replays a dumped viewer frame (count, Projection, raster parameters,
    // ProjectedSplat records) through both rasterizers: timing and image.
    int run_replay(id<MTLDevice> device, const char* path) {
        const lfs::core::GpuBackendScope scope(GpuBackend::Metal);
        FILE* file = std::fopen(path, "rb");
        require(file != nullptr, "cannot open replay");
        uint32_t count = 0;
        Projection projection{};
        lfs::rendering::SplatRasterParameters raster{};
        require(std::fread(&count, 4, 1, file) == 1 && std::fread(&projection, sizeof(projection), 1, file) == 1 &&
                    std::fread(&raster, sizeof(raster), 1, file) == 1, "short replay header");
        std::vector<uint8_t> bytes(size_t(count) * 64);
        require(std::fread(bytes.data(), bytes.size(), 1, file) == 1, "short replay body");
        std::fclose(file);
        const uint32_t width = raster.width, height = raster.height, capacity = raster.capacity;
        auto queue = [device newCommandQueue];
        auto projected_native = [device newBufferWithBytes:bytes.data() length:bytes.size() options:MTLResourceStorageModeShared];
        TileRasterizer native(device);
        std::vector<std::unique_ptr<RasterFrame>> ring;
        for (int i = 0; i < 3; ++i)
            ring.push_back(std::make_unique<RasterFrame>(device, width, height, count, capacity));
        const simd_float4 background = simd_make_float4(raster.background[0], raster.background[1], raster.background[2], raster.background[3]);
        constexpr int kFrames = 40;
        std::array<id<MTLCommandBuffer>, 3> inflight{};
        double gpu_ms = 0;
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < kFrames; ++i) {
            if (inflight[i % 3]) {
                [inflight[i % 3] waitUntilCompleted];
                if (i >= 6)
                    gpu_ms += (inflight[i % 3].GPUEndTime - inflight[i % 3].GPUStartTime) * 1e3;
            }
            auto command = [queue commandBuffer];
            native.encode(command, {projected_native}, count, RasterMode::Gaussian, background, *ring[i % 3], {}, {}, projection);
            [command commit];
            inflight[i % 3] = command;
        }
        for (auto& command : inflight)
            [command waitUntilCompleted];
        const double native_wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / kFrames;
        const auto native_status = ring[(kFrames - 1) % 3]->status();
        const auto native_color = read_texture(device, queue, ring[(kFrames - 1) % 3]->color(), 8);

        lfs::rendering::SplatRasterizer slang(GpuBackend::Metal);
        const auto projected = tensor(bytes);
        require(bool(slang.reserve(count, width, height, capacity)), "Slang reserve failed");
        auto slang_start = std::chrono::steady_clock::now();
        for (int i = 0; i < kFrames; ++i) {
            if (i == 3) {
                (void)download<uint64_t>(slang.status(), 1);
                slang_start = std::chrono::steady_clock::now();
            }
            auto r = slang.rasterize(projected, nullptr, count, lfs::rendering::SplatRasterMode::Gaussian, raster);
            require(bool(r), "Slang rasterize failed");
        }
        (void)download<uint64_t>(slang.status(), 1);
        const double slang_wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - slang_start).count() / (kFrames - 3);
        const auto slang_color = download<uint16_t>(slang.color(), size_t(width) * height * 4);
        const auto* nc = reinterpret_cast<const _Float16*>(native_color.data());
        const auto* sc = reinterpret_cast<const _Float16*>(slang_color.data());
        double max_color = 0;
        size_t over_one = 0;
        for (size_t i = 0; i < size_t(width) * height; ++i) {
            double pixel = 0;
            for (int c = 0; c < 4; ++c)
                pixel = std::max(pixel, std::fabs(double(nc[i * 4 + c]) - double(sc[i * 4 + c])));
            max_color = std::max(max_color, pixel);
            over_one += pixel > 1.0 / 255;
        }
        std::printf("replay %ux%u count=%u instances=%llu: native wall=%.2f ms gpu=%.2f ms | slang wall=%.2f ms | color max=%.2f/255 >1/255=%.4f%%\n",
                    width, height, count, (unsigned long long)native_status.required_instances, native_wall, gpu_ms / (kFrames - 6),
                    slang_wall, max_color * 255, 100.0 * over_one / (size_t(width) * height));
        return 0;
    }

    int run_frames(id<MTLDevice> device) {
        const lfs::core::GpuBackendScope scope(GpuBackend::Metal);
        auto module = M::load(splat_project_entries(), GpuBackend::Metal);
        require(bool(module), "Slang projection did not load");
        SplatPreprocessor preprocessor(device);
        TileRasterizer native(device);
        lfs::rendering::SplatRasterizer slang(GpuBackend::Metal);
        auto queue = [device newCommandQueue];
        const uint32_t count = 60000, width = 1280, height = 720, capacity = 16'000'000;
        const auto scene = make_scene(count, 21);
        int failures = 0;
        for (const auto& config : {Config{"frame-perspective-sh3", CameraModel::Perspective, PrimitiveMode::Gaussian, ShStorage::CanonicalFloat32, 3, false, false},
                                   Config{"frame-mip", CameraModel::Perspective, PrimitiveMode::Gaussian, ShStorage::CanonicalFloat32, 3, true, false}}) {
            const auto projection = make_projection(config);
            const simd_float4 background = simd_make_float4(.1f, .2f, .3f, 1);
            // Native.
            SplatBuffers in;
            in.count = count;
            in.layout_rest = 15;
            in.storage = config.storage;
            in.means = {native_buffer(device, scene.means)};
            in.log_scales = {native_buffer(device, scene.scales)};
            in.rotations = {native_buffer(device, scene.rotations)};
            in.opacity_logits = {native_buffer(device, scene.opacity)};
            in.sh0 = {native_buffer(device, scene.sh0)};
            in.sh_rest = {native_buffer(device, scene.rest)};
            in.sh_bounds = {native_buffer(device, scene.q16_bounds)};
            auto projected_native = [device newBufferWithLength:count * sizeof(ProjectedSplat) options:MTLResourceStorageModePrivate];
            RasterFrame frame(device, width, height, count, capacity);
            auto command = [queue commandBuffer];
            preprocessor.encode(command, in, projection, config.degree, config.mode, {projected_native});
            native.encode(command, {projected_native}, count, RasterMode::Gaussian, background, frame, {}, {}, projection);
            [command commit];
            [command waitUntilCompleted];
            require(command.status == MTLCommandBufferStatusCompleted, "Native frame failed");
            const auto native_status = frame.status();
            const auto native_color = read_texture(device, queue, frame.color(), 8);
            const auto native_depth = read_texture(device, queue, frame.depth(), 16);

            // Slang.
            const std::array<uint32_t, 12> layout{count, 15, 0, 0, 0, 0, 0, count, 0, 0, count, count};
            const auto means = tensor(scene.means), scales = tensor(scene.scales), rotations = tensor(scene.rotations);
            const auto opacity = tensor(scene.opacity), sh0 = tensor(scene.sh0), rest = tensor(scene.rest), bounds = tensor(scene.q16_bounds);
            const auto frame_tensor = tensor(std::vector<Projection>{projection});
            const auto layout_tensor = tensor(std::vector<uint32_t>(layout.begin(), layout.end()));
            auto projected = Tensor::zeros({count * sizeof(ProjectedSplat)}, Device::GPU, DataType::UInt8);
            struct Parameters {
                uint64_t pointers[22] = {};
                uint32_t sh_storage, sh_degree, primitive_mode, tight_bounds;
            } parameters{.sh_storage = uint32_t(config.storage), .sh_degree = config.degree, .primitive_mode = 0, .tight_bounds = 0};
            const std::array bindings{
                M::Binding{0, &means}, M::Binding{8, &scales}, M::Binding{16, &rotations}, M::Binding{24, &opacity},
                M::Binding{32, &sh0}, M::Binding{40, &rest}, M::Binding{48, &bounds}, M::Binding{56, nullptr},
                M::Binding{64, &projected, M::Access::ReadWrite}, M::Binding{72, &frame_tensor}, M::Binding{80, &layout_tensor},
                M::Binding{88, nullptr}, M::Binding{96, nullptr}, M::Binding{104, nullptr}, M::Binding{112, nullptr},
                M::Binding{120, nullptr}, M::Binding{128, nullptr}, M::Binding{136, nullptr},
                M::Binding{144, nullptr}, M::Binding{152, nullptr}, M::Binding{160, nullptr}, M::Binding{168, nullptr}};
            auto projected_ok = (*module)->dispatch({.function = "project_splats", .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                                                     .groups = {M::groups_for(count, 256), 1, 1}, .group = {256, 1, 1}});
            require(bool(projected_ok), "Slang projection failed");
            lfs::rendering::SplatRasterParameters raster{};
            raster.count = count;
            raster.width = width;
            raster.height = height;
            raster.columns = (width + 15) / 16;
            raster.tiles = raster.columns * ((height + 15) / 16);
            raster.capacity = capacity;
            raster.flags = 128; // one subgroup per 8x4 pixels, as native picks for Gaussians
            raster.background = {background.x, background.y, background.z, background.w};
            raster.intrinsics = {projection.intrinsics.x, projection.intrinsics.y, projection.intrinsics.z, projection.intrinsics.w};
            raster.clip = {projection.clip_scale.x, projection.clip_scale.y, projection.clip_scale.z, projection.clip_scale.w};
            raster.camera = {projection.extent.x, projection.extent.y, projection.extent.z, projection.extent.w};
            raster.panorama = {projection.panorama.x, projection.panorama.y, projection.panorama.z, projection.panorama.w};
            require(bool(slang.reserve(count, width, height, capacity)), "Slang reserve failed");
            auto rasterized = slang.rasterize(projected, nullptr, count, lfs::rendering::SplatRasterMode::Gaussian, raster);
            require(bool(rasterized), std::format("Slang rasterize failed: {}", rasterized ? "" : rasterized.error().detail()));
            const auto slang_color = download<uint16_t>(slang.color(), size_t(width) * height * 4);
            const auto slang_depth = download<float>(slang.depth(), size_t(width) * height * 4);
            // Timing: frames back to back, host waits once at the end.
            constexpr int kFrames = 30;
            const auto clock = [] { return std::chrono::steady_clock::now(); };
            const auto native_start = clock();
            std::vector<std::unique_ptr<RasterFrame>> ring;
            for (int i = 0; i < 3; ++i)
                ring.push_back(std::make_unique<RasterFrame>(device, width, height, count, capacity));
            id<MTLCommandBuffer> last;
            std::array<id<MTLCommandBuffer>, 3> inflight{};
            for (int i = 0; i < kFrames; ++i) {
                if (inflight[i % 3])
                    [inflight[i % 3] waitUntilCompleted];
                last = [queue commandBuffer];
                preprocessor.encode(last, in, projection, config.degree, config.mode, {projected_native});
                native.encode(last, {projected_native}, count, RasterMode::Gaussian, background, *ring[i % 3], {}, {}, projection);
                [last commit];
                inflight[i % 3] = last;
            }
            [last waitUntilCompleted];
            const double native_ms = std::chrono::duration<double, std::milli>(clock() - native_start).count() / kFrames;
            const auto slang_start = clock();
            for (int i = 0; i < kFrames; ++i) {
                require(bool((*module)->dispatch({.function = "project_splats", .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                                                  .groups = {M::groups_for(count, 256), 1, 1}, .group = {256, 1, 1}})),
                        "Slang projection failed");
                require(bool(slang.rasterize(projected, nullptr, count, lfs::rendering::SplatRasterMode::Gaussian, raster)), "Slang rasterize failed");
            }
            (void)download<uint64_t>(slang.status(), 1);
            const double slang_ms = std::chrono::duration<double, std::milli>(clock() - slang_start).count() / kFrames;
            std::printf("%-28s frame native=%.2f ms slang=%.2f ms\n", config.name, native_ms, slang_ms);
            // Stage split. Native: GPU timestamps of one frame. Slang: wall
            // clock of the cumulative prefixes, each run back to back.
            if (std::getenv("LFS_RASTER_STAGES")) {
                GpuProfile profile(device);
                auto command = [queue commandBuffer];
                preprocessor.encode(command, in, projection, config.degree, config.mode, {projected_native}, {}, {}, {}, {}, &profile);
                native.encode(command, {projected_native}, count, RasterMode::Gaussian, background, *ring[0], {}, {}, projection, {}, false, false, &profile);
                [command commit];
                [command waitUntilCompleted];
                const auto stages = profile.resolve();
                std::printf("  native GPU: projection=%.2f instances=%.2f sort=%.2f blend=%.2f ms\n", stages[0], stages[1], stages[2], stages[3]);
                lfs::rendering::SplatTileBinner binner(GpuBackend::Metal);
                require(bool(binner.reserve(count, raster.tiles, capacity)), "binner reserve failed");
                const auto raster_tensor = tensor(std::vector<lfs::rendering::SplatRasterParameters>{raster});
                const auto time = [&](int depth) {
                    const auto start = clock();
                    for (int i = 0; i < kFrames; ++i) {
                        require(bool((*module)->dispatch({.function = "project_splats", .arguments = {std::as_bytes(std::span(&parameters, 1)), bindings},
                                                          .groups = {M::groups_for(count, 256), 1, 1}, .group = {256, 1, 1}})),
                                "projection failed");
                        if (depth >= 1)
                            require(bool(binner.bin(projected, raster_tensor, count, raster.tiles)), "bin failed");
                        if (depth >= 2)
                            require(bool(slang.rasterize(projected, nullptr, count, lfs::rendering::SplatRasterMode::Gaussian, raster)), "rasterize failed");
                    }
                    (void)download<uint64_t>(depth >= 1 ? binner.status() : projected, 1);
                    return std::chrono::duration<double, std::milli>(clock() - start).count() / kFrames;
                };
                time(0);
                const double project = time(0), binned = time(1), full = time(2);
                std::printf("  slang wall: projection=%.2f bin=%.2f rasterize(bin+blend)=%.2f ms\n", project, binned - project, full - binned);
            }
            // Compare the first frame and a steady-state (source-sorted) one.
            const auto compare = [&](const char* label, const std::vector<uint16_t>& slang_color, const std::vector<float>& slang_depth) {
                const auto status = download<uint64_t>(slang.status(), 3);
                const auto* nc = reinterpret_cast<const _Float16*>(native_color.data());
                const auto* sc = reinterpret_cast<const _Float16*>(slang_color.data());
                const auto* nd = reinterpret_cast<const float*>(native_depth.data());
                double max_color = 0, sum_color = 0;
                size_t over_one = 0, median_diff = 0, covered = 0;
                for (size_t i = 0; i < size_t(width) * height; ++i) {
                    double pixel_max = 0;
                    for (int c = 0; c < 4; ++c) {
                        const double d = std::fabs(double(nc[i * 4 + c]) - double(sc[i * 4 + c]));
                        pixel_max = std::max(pixel_max, d);
                        sum_color += d;
                    }
                    max_color = std::max(max_color, pixel_max);
                    covered += double(nc[i * 4 + 3]) > .5;
                    over_one += pixel_max > 1.0 / 255;
                    const float a = nd[i * 4 + 3], b = slang_depth[i * 4 + 3];
                    median_diff += std::fabs(a - b) > 1e-3f * std::max(1.f, std::fabs(a));
                }
                const size_t pixels = size_t(width) * height;
                const auto status_words = download<uint32_t>(slang.status(), 6);
                const bool ok = status_words[2] == 0 && covered * 10 > pixels && status[0] == native_status.required_instances &&
                                max_color * 255 <= 2 && over_one * 1000 <= pixels && median_diff * 1000 <= pixels;
                std::printf("%-28s %-6s instances native=%llu slang=%llu color max=%.2f/255 mean=%.4f/255 >1/255=%.4f%% median_depth_diff=%.4f%% covered=%.1f%% %s\n",
                            config.name, label, (unsigned long long)native_status.required_instances, (unsigned long long)status[0], max_color * 255,
                            sum_color / (pixels * 4) * 255, 100.0 * over_one / pixels, 100.0 * median_diff / pixels, 100.0 * covered / pixels, ok ? "ok" : "FAIL");
                return ok;
            };
            const bool first_ok = compare("first", slang_color, slang_depth);
            const bool steady_ok = compare("steady", download<uint16_t>(slang.color(), size_t(width) * height * 4),
                                           download<float>(slang.depth(), size_t(width) * height * 4));
            const bool ok = first_ok && steady_ok;
            failures += ok ? 0 : 1;
        }
        return failures;
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
            // Different compilers round differently; ill-conditioned covariances
            // near the extent cap amplify that, and pixel bounds may flip at a
            // floor/ceil boundary. Image parity is checked end to end elsewhere.
            double worst_relative = 0;
            for (const auto& field : diff.relative)
                for (const double value : field)
                    worst_relative = std::max(worst_relative, value);
            const bool ok = diff.culled_mismatch * 1000 <= diff.compared && diff.bounds_mismatch * 1000 <= diff.compared &&
                            worst_relative <= 1e-3;
            std::printf("%-28s compared=%zu culled_mismatch=%zu bounds_mismatch=%zu worst_relative=%.3g %s\n", config.name,
                        diff.compared, diff.culled_mismatch, diff.bounds_mismatch, worst_relative, ok ? "ok" : "FAIL");
            const char* names[] = {"mean_depth", "conic_opacity", "color"};
            for (size_t f = 0; f < 3; ++f)
                for (int c = 0; c < 4; ++c)
                    if (diff.relative[f][c] > 1e-6)
                        std::printf("    %s[%d] relative=%.3g native=%.9g slang=%.9g\n", names[f], c, diff.relative[f][c],
                                    diff.worst[f][c].first, diff.worst[f][c].second);
            if (std::getenv("LFS_PARITY_DUMP") && config.mode == PrimitiveMode::Gaussian && config.camera == CameraModel::Perspective) {
                const size_t w = diff.worst_index;
                std::printf("    worst %zu mean=%.9g %.9g %.9g scale=%.9g %.9g %.9g rot=%.9g %.9g %.9g %.9g logit=%.9g\n", w,
                            scene.means[w * 3], scene.means[w * 3 + 1], scene.means[w * 3 + 2], scene.scales[w * 3], scene.scales[w * 3 + 1],
                            scene.scales[w * 3 + 2], scene.rotations[w * 4], scene.rotations[w * 4 + 1], scene.rotations[w * 4 + 2],
                            scene.rotations[w * 4 + 3], scene.opacity[w]);
                std::printf("    native conic %.9g %.9g %.9g slang %.9g %.9g %.9g\n", expected[w].conic_opacity[0], expected[w].conic_opacity[1],
                            expected[w].conic_opacity[2], actual[w].conic_opacity[0], actual[w].conic_opacity[1], actual[w].conic_opacity[2]);
            }
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
            if (const char* replay = std::getenv("LFS_REPLAY"))
                return run_replay(device, replay);
            return (run(device) | run_frames(device)) ? 1 : 0;
        } catch (const std::exception& error) {
            std::fprintf(stderr, "tensor raster parity: %s\n", error.what());
            return 1;
        }
    }
}

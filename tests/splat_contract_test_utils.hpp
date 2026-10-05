/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor.hpp"
#include "core/tensor_backend.hpp"
#include "splat_projector.hpp"
#include "splat_rasterizer.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace lfs::test::splat {
    using core::DataType;
    using core::Device;
    using core::Tensor;

    struct ProjectedSplat {
        std::array<float, 4> mean_depth{}, conic_opacity{}, color{};
        std::array<uint32_t, 4> bounds{};
    };
    struct GutSplat {
        std::array<float, 4> inverse0{}, inverse1{}, inverse2{}, mean_opacity{};
    };
    struct RasterStatus {
        uint64_t required_instances = 0;
        uint32_t error = 0, blend_threads = 0, maximum_tile_instances = 0, padding = 0;
    };
    struct SceneObject {
        std::array<float, 16> model_to_world{};
        std::array<float, 4> camera_local{};
        std::array<uint32_t, 4> flags{};
    };
    static_assert(sizeof(ProjectedSplat) == 64);
    static_assert(sizeof(GutSplat) == 64);
    static_assert(sizeof(RasterStatus) == 24);
    static_assert(sizeof(SceneObject) == 96);

    inline std::array<float, 16> identity() {
        std::array<float, 16> m{};
        m[0] = m[5] = m[10] = m[15] = 1;
        return m;
    }

    inline rendering::SplatProjection projection(uint32_t width = 256, uint32_t height = 256) {
        rendering::SplatProjection p;
        p.model_to_world = p.world_to_camera = identity();
        p.intrinsics = {200, 200, 128, 128};
        p.clip_scale = {.01f, 1000, 1, .3f};
        p.extent = {width, height, 0, 0};
        p.rasterization = {1, 0, 0, 0};
        p.display = {0, 1, 0, 0};
        return p;
    }

    inline rendering::SplatRasterParameters raster_parameters(uint32_t count, uint32_t width, uint32_t height,
                                                               rendering::SplatRasterMode mode, uint32_t capacity,
                                                               uint32_t flags = 0, std::array<float, 4> background = {}) {
        rendering::SplatRasterParameters r;
        r.count = count;
        r.width = width;
        r.height = height;
        r.columns = (width + 15) / 16;
        r.tiles = r.columns * ((height + 15) / 16);
        r.capacity = capacity;
        r.mode = uint32_t(mode);
        r.flags = flags;
        r.background = background;
        r.intrinsics = {32, 32, width * .5f, height * .5f};
        r.clip = {.01f, 100, 1, .3f};
        r.camera = {width, height, 0, 0};
        return r;
    }

    inline Tensor upload_bytes(const void* data, size_t bytes) {
        if (!bytes)
            return {};
        return Tensor::from_blob(const_cast<void*>(data), {bytes}, Device::CPU, DataType::UInt8).to(Device::GPU);
    }
    template <class T>
    Tensor upload(std::span<const T> values) {
        return upload_bytes(values.data(), values.size_bytes());
    }
    template <class T, size_t N>
    Tensor upload(const std::array<T, N>& values) {
        return upload(std::span<const T>(values));
    }
    template <class T>
    Tensor upload(const std::vector<T>& values) {
        return upload(std::span<const T>(values));
    }
    template <class T>
    Tensor upload_one(const T& value) {
        return upload_bytes(&value, sizeof(value));
    }
    template <class T>
    std::vector<T> download(const Tensor& tensor, size_t count) {
        const auto host = tensor.to(Device::CPU);
        std::vector<T> result(count);
        std::memcpy(result.data(), host.data_ptr(), count * sizeof(T));
        return result;
    }
    template <class T>
    T download_one(const Tensor& tensor) {
        return download<T>(tensor, 1)[0];
    }

    inline float half_to_float(uint16_t bits) {
        const uint32_t sign = uint32_t(bits >> 15) << 31;
        const uint32_t exponent = (bits >> 10) & 31;
        const uint32_t mantissa = bits & 1023;
        uint32_t value = sign;
        if (exponent == 31)
            value |= 0x7f800000u | (mantissa << 13);
        else if (exponent)
            value |= ((exponent + 112) << 23) | (mantissa << 13);
        else if (mantissa) {
            int e = -1;
            uint32_t m = mantissa;
            do {
                ++e;
                m <<= 1;
            } while (!(m & 1024));
            value |= ((112 - e) << 23) | ((m & 1023) << 13);
        }
        return std::bit_cast<float>(value);
    }

    inline uint16_t float_to_half(float value) {
        _Float16 h = value;
        uint16_t bits;
        std::memcpy(&bits, &h, sizeof(bits));
        return bits;
    }

    struct RasterReadback {
        std::vector<uint16_t> color;
        std::vector<float> depth;
        std::vector<uint32_t> pick;
        RasterStatus status;
    };
    inline RasterReadback readback(const rendering::SplatRasterizer& rasterizer, uint32_t width, uint32_t height) {
        const size_t pixels = size_t(width) * height;
        return {download<uint16_t>(rasterizer.color(), pixels * 4), download<float>(rasterizer.depth(), pixels * 4),
                download<uint32_t>(rasterizer.pick(), pixels), download_one<RasterStatus>(rasterizer.status())};
    }

    inline bool backend_unavailable_or_cuda(core::GpuBackend backend) {
        return !core::gpu_backend_available(backend) || backend == core::GpuBackend::CUDA;
    }
} // namespace lfs::test::splat

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "geometry_camera.hpp"
#include "core/assert.hpp"
#include "core/crash_handler.hpp"
#include "core/gpu_kernel_module.hpp"
#include "core/tensor_backend.hpp"
#include "geometry_camera_program.hpp"
#include <algorithm>
#include <array>
#include <cstring>
namespace lfs::training {
    std::vector<gpu_ops::AnchorSample> collect_camera_anchor_samples(gpu_ops::In points, gpu_ops::In view, gpu_ops::In prior, const gpu_ops::AnchorParams& params) {
        using namespace core;
        constexpr size_t capacity = 262144;
        const size_t count = points.shape()[0];
        if (!count)
            return {};
        LFS_ASSERT_MSG(count <= UINT32_MAX / 3, "Anchor point count exceeds 32-bit indexing");
        const auto backend = gpu_backend_of(points).value_or(default_gpu_backend());
        const GpuBackendScope scope(backend);
        const size_t stride = std::max(size_t(1), count / capacity), samples = (count + stride - 1) / stride;
        auto loaded = GpuKernelModule::load(geometry_camera_program_entries(), backend);
        if (!loaded)
            throw Exception(std::move(loaded).error());
        auto camera = Tensor::empty({64}, Device::GPU);
        auto radial = Tensor::from_vector(std::vector<float>(params.projection.radial, params.projection.radial + 6), {6}, Device::GPU);
        auto tangent = Tensor::from_vector(std::vector<float>(params.projection.tangential, params.projection.tangential + 2), {2}, Device::GPU);
        auto prism = Tensor::from_vector(std::vector<float>(params.projection.thin_prism, params.projection.thin_prism + 4), {4}, Device::GPU);
        auto pairs = Tensor::empty({capacity, 2}, Device::GPU);
        auto found = Tensor::zeros({1}, Device::GPU, DataType::Int32);
        struct Parameters {
            uint64_t camera = 0, view = 0, radial = 0, tangent = 0, prism = 0, points = 0, prior = 0, pairs = 0, count = 0;
            float fx, fy, cx, cy, near_plane;
            uint32_t width, height, model, samples, stride, capacity;
            std::array<float, 3> lo, hi;
            float padding = 0;
        } p{.fx = params.intrinsics.fx, .fy = params.intrinsics.fy, .cx = params.intrinsics.cx, .cy = params.intrinsics.cy, .near_plane = params.near_plane, .width = uint32_t(prior.shape()[1]), .height = uint32_t(prior.shape()[0]), .model = uint32_t(params.projection.model), .samples = uint32_t(samples), .stride = uint32_t(stride), .capacity = capacity, .lo = params.aabb_lo, .hi = params.aabb_hi};
        static_assert(sizeof(Parameters) == 144 && offsetof(Parameters, padding) == 140);
        if (p.model == 3) {
            p.fx = float(p.width);
            p.fy = float(p.height);
            p.cx = 0;
            p.cy = 0;
        }
        const std::array bindings{GpuKernelModule::Binding{0, &camera, GpuKernelModule::Access::ReadWrite},
                                  GpuKernelModule::Binding{8, &view}, GpuKernelModule::Binding{16, &radial}, GpuKernelModule::Binding{24, &tangent}, GpuKernelModule::Binding{32, &prism},
                                  GpuKernelModule::Binding{40, &points}, GpuKernelModule::Binding{48, &prior}, GpuKernelModule::Binding{56, &pairs, GpuKernelModule::Access::ReadWrite}, GpuKernelModule::Binding{64, &found, GpuKernelModule::Access::ReadWrite}};
        for (const auto entry : {"setup", "collect"}) {
            auto result = (*loaded)->dispatch({.function = entry, .arguments = {std::as_bytes(std::span(&p, 1)), bindings}, .groups = {entry == std::string_view("setup") ? 1u : uint32_t((samples + 255) / 256), 1, 1}, .group = {256, 1, 1}});
            if (!result)
                throw Exception(std::move(result).error());
        }
        const int n = std::min(found.item<int>(), int(capacity));
        if (n < kernels::kMinAnchorSamples)
            return {};
        auto host = pairs.slice(0, 0, size_t(n)).cpu().contiguous();
        std::vector<gpu_ops::AnchorSample> result(size_t(n), gpu_ops::AnchorSample{});
        std::memcpy(result.data(), host.data_ptr(), result.size() * sizeof(gpu_ops::AnchorSample));
        return result;
    }
} // namespace lfs::training

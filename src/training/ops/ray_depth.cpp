/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "ray_depth.hpp"
#include "core/gpu_kernel_module.hpp"
#include "core/tensor_backend.hpp"
#include "ray_depth_program.hpp"
#include <array>

namespace lfs::training {
    namespace {
        void dispatch(const RayDepthStep& step, std::string_view function) {
            using namespace core;
            if (!step.state.is_valid())
                return;
            const auto backend = *gpu_backend_of(step.state);
            GpuBackendScope scope(backend);
            auto program = GpuKernelModule::load(ray_depth_program_entries(), backend);
            if (!program)
                throw Exception(std::move(program).error());
            struct Parameters {
                uint64_t addresses[10]{};
                uint32_t count, frozen_count, crop_count, far_count;
                float lr, mean_lr, far_limit, frozen_scale, crop_scale, median_extent;
            } p{.count = uint32_t(step.state.size(0)), .frozen_count = uint32_t(step.frozen.numel()), .crop_count = uint32_t(step.crop.numel()), .far_count = uint32_t(step.far_mask.numel()), .lr = step.lr, .mean_lr = step.mean_lr, .far_limit = step.far_limit, .frozen_scale = step.frozen_scale, .crop_scale = step.crop_scale, .median_extent = step.median_extent};
            using Binding = GpuKernelModule::Binding;
            constexpr auto rw = GpuKernelModule::Access::ReadWrite;
            const std::array bindings{Binding{0, &step.state, rw}, Binding{8, &step.means, rw}, Binding{16, &step.scales, rw},
                                      Binding{24, &step.rotations}, Binding{32, &step.gradient}, Binding{40, &step.camera},
                                      Binding{48, &step.proposals, rw}, Binding{56, p.frozen_count ? &step.frozen : nullptr},
                                      Binding{64, p.crop_count ? &step.crop : nullptr}, Binding{72, p.far_count ? &step.far_mask : nullptr}};
            auto result = (*program)->dispatch({.function = function, .arguments = {std::as_bytes(std::span(&p, 1)), bindings}, .groups = {(p.count + 255) / 256, 1, 1}, .group = {256, 1, 1}});
            if (!result)
                throw Exception(std::move(result).error());
        }
    } // namespace
    void ray_depth_propose(const RayDepthStep& step) { dispatch(step, "propose"); }
    void ray_depth_commit(const RayDepthStep& step) { dispatch(step, "commit"); }
} // namespace lfs::training

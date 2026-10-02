/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/gpu_kernel_module.hpp"
#include "core/tensor.hpp"

#include <cstdint>
#include <format>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>

// Launches the trainer's Metal kernels (kernels/metal/*.metal) on the tensor
// timeline. Parameter blocks carry tensor addresses from address().
namespace lfs::training::metal {

    inline constexpr uint32_t kGroupWidth = 256;

    // MSL vector layouts inside parameter blocks: float3 and float4 are both
    // 16 bytes with 16-byte alignment.
    struct alignas(16) Float4 {
        float x = 0.f, y = 0.f, z = 0.f, w = 0.f;
    };
    struct alignas(16) Float3 {
        float x = 0.f, y = 0.f, z = 0.f;
        float unused = 0.f;
    };
    struct alignas(8) Float2 {
        float x = 0.f, y = 0.f;
    };

    // A count passed to a kernel's 32-bit index space.
    inline uint32_t count32(const size_t count, const std::string_view what) {
        if (count > std::numeric_limits<uint32_t>::max())
            throw std::invalid_argument(std::format("{} count {} exceeds the 32-bit kernel index range", what, count));
        return static_cast<uint32_t>(count);
    }

    core::GpuKernelModule& kernels();

    // Device address of a tensor, or 0 when it is absent.
    inline uint64_t address(const core::Tensor& tensor) { return kernels().address(tensor); }

    template <class Params>
    void launch(const std::string_view function, const Params& params,
                const std::initializer_list<const core::Tensor*> uses, const uint32_t groups,
                const uint32_t width = kGroupWidth,
                const std::initializer_list<std::pair<uint32_t, uint32_t>> constants = {}) {
        kernels().launch({.function = function,
                          .params = std::as_bytes(std::span(&params, 1)),
                          .uses = std::span(uses.begin(), uses.size()),
                          .groups = {groups, 1, 1},
                          .group = {width, 1, 1},
                          .constants = std::span(constants.begin(), constants.size())});
    }

    // Threadgroups of width x height threads over a 2D grid of groups.
    template <class Params>
    void launch_2d(const std::string_view function, const Params& params,
                   const std::initializer_list<const core::Tensor*> uses, const uint32_t groups_x,
                   const uint32_t groups_y, const uint32_t width, const uint32_t height,
                   const std::initializer_list<std::pair<uint32_t, uint32_t>> constants = {}) {
        kernels().launch({.function = function,
                          .params = std::as_bytes(std::span(&params, 1)),
                          .uses = std::span(uses.begin(), uses.size()),
                          .groups = {groups_x, groups_y, 1},
                          .group = {width, height, 1},
                          .constants = std::span(constants.begin(), constants.size())});
    }

    // One thread per item.
    template <class Params>
    void launch_items(const std::string_view function, const Params& params,
                      const std::initializer_list<const core::Tensor*> uses, const size_t items,
                      const std::initializer_list<std::pair<uint32_t, uint32_t>> constants = {}) {
        launch(function, params, uses, core::GpuKernelModule::groups_for(items, kGroupWidth), kGroupWidth, constants);
    }

} // namespace lfs::training::metal

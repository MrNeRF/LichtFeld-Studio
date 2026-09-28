/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/gpu_kernel_module.hpp"
#include "core/tensor.hpp"

#include <cstdint>
#include <initializer_list>
#include <span>
#include <string_view>
#include <utility>

// Launches the trainer's Metal kernels (kernels/metal/*.metal) on the tensor
// timeline. Parameter blocks carry tensor addresses from address().
namespace lfs::training::metal {

    inline constexpr uint32_t kGroupWidth = 256;

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

    // One thread per item.
    template <class Params>
    void launch_items(const std::string_view function, const Params& params,
                      const std::initializer_list<const core::Tensor*> uses, const size_t items,
                      const std::initializer_list<std::pair<uint32_t, uint32_t>> constants = {}) {
        launch(function, params, uses, core::GpuKernelModule::groups_for(items, kGroupWidth), kGroupWidth, constants);
    }

} // namespace lfs::training::metal

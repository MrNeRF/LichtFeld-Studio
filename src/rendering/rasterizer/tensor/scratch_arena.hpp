/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <string_view>

namespace lfs::rendering {
    // Byte buffers of one lifetime carved from a single allocation. The tensor
    // allocator rounds requests below 64 MiB up to a power of two, so separate
    // buffers can waste nearly half their size; one arena rounds once.
    // Views are 256-byte aligned UInt8 tensors; kernels bind them by address.
    template <size_t N>
    std::array<core::Tensor, N> carve_arena(const std::array<size_t, N>& bytes,
                                            const std::string_view name) {
        constexpr size_t kAlignment = 256;
        std::array<size_t, N> offsets{};
        size_t total = 0;
        for (size_t i = 0; i < N; ++i) {
            offsets[i] = total;
            total += (std::max<size_t>(bytes[i], 1) + kAlignment - 1) / kAlignment * kAlignment;
        }
        auto arena = core::Tensor::empty({total}, core::Device::GPU, core::DataType::UInt8);
        arena.set_name(std::string(name));
        std::array<core::Tensor, N> views;
        for (size_t i = 0; i < N; ++i)
            views[i] = arena.slice(0, offsets[i], offsets[i] + std::max<size_t>(bytes[i], 1));
        return views;
    }
} // namespace lfs::rendering

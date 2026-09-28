/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>

// SH storage helpers the Metal Adam, Sh and Morton families share.
namespace lfs::training::metal {

    // Device address of a tensor through its live exportable region, as the
    // CUDA families resolve it: exportable storage can move on growth.
    uint64_t live_address(const core::Tensor& tensor);

    // Copies `rows` rows of `width` bytes between device addresses. `uses`
    // lists the tensors behind the addresses that have one.
    void copy_rows(uint64_t source, uint64_t destination, size_t width, size_t rows, size_t source_pitch,
                   size_t destination_pitch, std::initializer_list<const core::Tensor*> uses);

} // namespace lfs::training::metal

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include <cstdint>

namespace lfs::core {
    class Tensor;

    enum class LabelUpdateMode : uint32_t { Add,
                                            Remove,
                                            Replace };

    struct LabelUpdate {
        uint8_t label = 1;
        LabelUpdateMode mode = LabelUpdateMode::Add;
        const Tensor* existing = nullptr;
        const Tensor* locked = nullptr;
        const Tensor* indices = nullptr;
        const Tensor* categories = nullptr;
        const Tensor* allowed = nullptr;
    };

    // Update UInt8 [M] labels from Bool/UInt8 [N] selection flags. Without
    // indices, N == M. Optional Int32 [N] indices map rows to output labels;
    // out-of-range entries are ignored. Untouched labels retain existing [M]
    // values, or zero when existing is absent. Locked Bool/UInt8 [256] labels
    // cannot be overwritten, except for zero and the label being edited.
    // Add assigns selected rows, Remove clears selected rows of this label,
    // Replace also clears unselected rows of this label. With repeated indices,
    // selected rows take precedence over unselected rows during replacement.
    // Optional Int32 [N] categories and Bool/UInt8 [K] allowed flags restrict
    // eligible rows; an invalid category is ineligible. Supply both or neither.
    // Tensors share a device/backend. Offset and strided views and aliased
    // inputs are supported. Counts must fit int32. GPU work is asynchronous.
    LFS_CORE_API void update_labels(Tensor& output, const Tensor& selected, const LabelUpdate& update);
} // namespace lfs::core

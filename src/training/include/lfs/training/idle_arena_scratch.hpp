/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "lfs/training/ops/registry.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace lfs::training {

    // Borrows the rasterizer arena's committed memory between training frames,
    // when no rasterizer or viewer frame owns it. Takes up to `wanted` bytes
    // without growing the arena, or nothing when a frame is active or fewer
    // than `minimum` bytes are committed. All use must be enqueued on `stream`
    // and finish before destruction, whose end_frame orders the next frame
    // after that work.
    class IdleArenaScratch {
    public:
        IdleArenaScratch(size_t wanted, size_t minimum, core::TensorExecutionTarget target)
            : ops_(training_session_ops()), target_(target), borrow_(ops_.borrow_idle_arena(wanted, minimum, target)) {}
        ~IdleArenaScratch() { ops_.release_idle_arena(borrow_, target_); }
        IdleArenaScratch(const IdleArenaScratch&) = delete;
        IdleArenaScratch& operator=(const IdleArenaScratch&) = delete;
        [[nodiscard]] char* data() const { return borrow_.data; }
        [[nodiscard]] size_t capacity() const { return borrow_.bytes; }
        [[nodiscard]] void* zeroed(size_t bytes) const { return ops_.zero_idle_arena(borrow_, bytes, target_); }

    private:
        const ops::SessionOps& ops_;
        core::TensorExecutionTarget target_;
        ops::IdleArenaBorrow borrow_;
    };
} // namespace lfs::training

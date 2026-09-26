/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"

#include <cstddef>
#include <memory>
#include <optional>

namespace lfs::core {

    class LFS_CORE_API GpuElapsed {
    public:
        GpuElapsed(GpuBackend backend, std::size_t event_count);
        ~GpuElapsed();
        GpuElapsed(const GpuElapsed&) = delete;
        GpuElapsed& operator=(const GpuElapsed&) = delete;
        GpuElapsed(GpuElapsed&&) = delete;
        GpuElapsed& operator=(GpuElapsed&&) = delete;

        [[nodiscard]] bool ready() const noexcept;
        // CUDA execution target: nullptr selects the default CUDA stream. A
        // Vulkan timer is unsupported and rejects a non-null native queue.
        [[nodiscard]] bool mark(std::size_t index, void* execution_target);
        [[nodiscard]] bool wait_event(std::size_t index);
        [[nodiscard]] bool wait_queue(void* execution_target);
        [[nodiscard]] std::optional<float> milliseconds(std::size_t begin,
                                                        std::size_t end) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::core

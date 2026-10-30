/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/detail/backend_runtime.hpp"
#include "core/tensor/internal/private_access.hpp"
#include "core/tensor_completion.hpp"
#include <exception>

namespace lfs::core::internal {
    class CompletionDrain {
        std::exception_ptr failure_;

    public:
        void settle(auto&& operation) noexcept {
            try {
                operation();
            } catch (...) {
                // LFS-CENSUS-OK(empty-catch): Drain all obligations before rethrowing the first failure.
                if (!failure_)
                    failure_ = std::current_exception();
            }
        }
        void rethrow() const {
            if (failure_)
                std::rethrow_exception(failure_);
        }
    };

    struct NativeCompletionRecord {
        ExecContext target{};
        uint64_t value = 0;
        VulkanTimelinePoint timeline;
        void* external_device = nullptr;
        bool settle_on_release = true;
        bool record = true;
        bool use_recorded_timeline = false;
    };
} // namespace lfs::core::internal

namespace lfs::core {
    struct TensorCompletionAccess {
        static TensorCompletion record(GpuBackend backend, internal::NativeCompletionRecord record);
    };
} // namespace lfs::core

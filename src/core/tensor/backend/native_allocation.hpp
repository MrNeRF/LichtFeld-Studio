/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/detail/descriptors.hpp"
#include <memory>
namespace lfs::core::internal {
    enum class TensorAllocationKind { Pooled,
                                      Exact,
                                      Direct };
    struct TensorAllocationRequest {
        size_t bytes = 0;
        size_t alignment = 0;
        TensorAllocationKind kind = TensorAllocationKind::Pooled;
        bool allow_arena = false;
        const char* label = "tensor.storage";
        const char* operation = "tensor.allocate";
    };
    // Allocation-only descriptor: the hot StorageRef layout does not grow.
    struct NativeTensorAllocation {
        StorageRef storage;
        void* target = nullptr;
        std::shared_ptr<void> owner;
        const char* external_kind = "";
    };
    template <class Backend>
    NativeTensorAllocation allocate_native_tensor(Backend& backend, const TensorAllocationRequest& request, void* target = nullptr) {
        const auto kind = request.kind == TensorAllocationKind::Direct  ? AllocationClass::Direct
                          : request.kind == TensorAllocationKind::Exact ? AllocationClass::Exact
                                                                        : AllocationClass::Pooled;
        auto storage = backend.allocate(request.bytes, request.alignment,
                                        ExecContext{static_cast<cudaStream_t>(target), kind, request.label, request.operation});
        return {.storage = storage, .target = target, .external_kind = allocation_label(storage)};
    }
} // namespace lfs::core::internal

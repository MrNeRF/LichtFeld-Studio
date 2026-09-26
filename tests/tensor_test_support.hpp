/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

// Test observations of the exported Vulkan timeline. Storage metadata is part of
// the header-only tensor body; the two queries are exported from lfs_core.
#include "core/tensor.hpp"

#include <string>
#include <vector>

namespace lfs::core::internal {
    std::vector<std::string> vulkan_validation_messages_for_testing();
    uint64_t vulkan_completed_timeline_for_testing();
} // namespace lfs::core::internal

namespace lfs::test {
    inline auto vulkan_validation_messages() {
        return core::internal::vulkan_validation_messages_for_testing();
    }
    inline uint64_t vulkan_pending_value(const core::Tensor& tensor) {
        return core::internal::storage_ref(tensor).meta->pending_value.load(std::memory_order_acquire);
    }
    inline uint64_t vulkan_completed_value() {
        return core::internal::vulkan_completed_timeline_for_testing();
    }
} // namespace lfs::test

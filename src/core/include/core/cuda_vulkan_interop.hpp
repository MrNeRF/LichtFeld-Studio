/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include <memory>

namespace lfs::core::cuda {
    class LFS_CORE_API ExternalMemoryImportScope {
    public:
        ExternalMemoryImportScope() noexcept;
        ~ExternalMemoryImportScope() noexcept;
        ExternalMemoryImportScope(const ExternalMemoryImportScope&) = delete;
        ExternalMemoryImportScope& operator=(const ExternalMemoryImportScope&) = delete;
        [[nodiscard]] static bool active() noexcept;
    };

    // The alias pointer is the CUDA semaphore consumed by the trainer.
    LFS_CORE_API std::shared_ptr<void> import_vulkan_semaphore(void* device, void* semaphore);
} // namespace lfs::core::cuda

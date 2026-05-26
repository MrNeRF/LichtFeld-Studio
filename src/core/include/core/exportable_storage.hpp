/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <string>

namespace lfs::core {

#ifdef _WIN32
    using ExportNativeHandle = void*;
#else
    using ExportNativeHandle = int;
#endif

    struct ExportHandle {
        ExportNativeHandle native = ExportNativeHandle{};
        std::size_t size = 0;
        [[nodiscard]] bool valid() const noexcept {
#ifdef _WIN32
            return native != nullptr;
#else
            return native >= 0;
#endif
        }
    };

    // Owns a CUDA VMM allocation backed by a single mapped chunk that is
    // exportable to Vulkan via VK_KHR_external_memory_{fd,win32}.
    // Destruction (via shared_ptr deleter) runs:
    //   recordDeallocation -> cuMemUnmap -> cuMemRelease -> cuMemAddressFree
    //   -> close(fd) / CloseHandle
    struct ExportableBlock {
        void* device_ptr = nullptr;
        std::size_t size = 0;
        ExportHandle handle{};
    };

    [[nodiscard]] std::expected<std::shared_ptr<ExportableBlock>, std::string>
    allocateExportableDeviceBlock(std::size_t size, int device = 0);

} // namespace lfs::core

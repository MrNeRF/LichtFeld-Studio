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
    //
    // A large virtual address range is reserved up front (reserve_bytes) while
    // only `size` physical memory is committed. growExportableDeviceBlock()
    // commits more physical under the SAME virtual address, so device_ptr never
    // changes across growth — consumers holding the base pointer (e.g. the
    // training arena) stay valid and never see a use-after-free. Growth swaps
    // the exported OS handle, so the Vulkan side must re-import after a grow.
    //
    // Destruction (via shared_ptr deleter) runs:
    //   recordDeallocation -> cuMemUnmap -> cuMemRelease -> cuMemAddressFree
    //   -> close(fd) / CloseHandle
    struct ExportableBlock {
        void* device_ptr = nullptr;
        std::size_t size = 0;
        ExportHandle handle{};
        std::shared_ptr<void> state; // opaque OwnedAllocation, mutated by growExportableDeviceBlock
    };

    // Allocates a block committing `size` physical bytes, reserving a virtual
    // range of max(reserve_bytes, size) for in-place growth (reserve is free).
    [[nodiscard]] std::expected<std::shared_ptr<ExportableBlock>, std::string>
    allocateExportableDeviceBlock(std::size_t size, int device = 0, bool track_splat_bytes = true,
                                  std::size_t reserve_bytes = 0);

    // Grows committed physical to >= new_size under the block's stable virtual
    // address. Returns true if the block grew (handle/size changed, callers must
    // re-import into Vulkan), false if it already satisfied new_size. device_ptr
    // is unchanged on success. Must not run while the GPU is using the block.
    [[nodiscard]] std::expected<bool, std::string>
    growExportableDeviceBlock(const std::shared_ptr<ExportableBlock>& block, std::size_t new_size);

} // namespace lfs::core

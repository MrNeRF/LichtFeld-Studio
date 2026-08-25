/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include <cstddef>
#include <expected>
#include <memory>
#include <string>
#include <vector>

namespace lfs::core {

#ifdef _WIN32
    using ExportNativeHandle = void*;
#else
    using ExportNativeHandle = int;
#endif

    // CUDA-side OPAQUE_FD / WIN32 shareable handle for one committed physical
    // chunk. The exporter owns `native` for the lifetime of the chunk; Vulkan
    // dups (POSIX) / imports (Win32) and never closes this handle. Chunks are
    // released only at block teardown, after every Vulkan import of the block
    // has been destroyed.
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
        [[nodiscard]] bool operator==(const ExportHandle& other) const noexcept {
            return native == other.native && size == other.size;
        }
    };

    struct ExportableChunk {
        std::size_t offset = 0;
        std::size_t bytes = 0;
        ExportHandle handle{};
    };

    // One CUDA VMM virtual reservation with physical memory committed as
    // shareable chunks. `device_ptr` is stable for the lifetime of the block.
    // Growth commits additional slices; existing chunks are never unmapped,
    // copied, or re-exported. `chunks` is kept sorted by offset, so a
    // per-region tail commit inserts in the middle of the vector. Importers
    // must bind by offset (see unboundExportableChunkIndices), never by a
    // prefix count of that vector.
    //
    // Destruction (via shared_ptr deleter) runs:
    //   recordDeallocation -> unmap each chunk -> cuMemRelease each chunk
    //   -> close each fd/HANDLE -> cuMemAddressFree
    struct ExportableBlock {
        void* device_ptr = nullptr;
        std::size_t reserved_bytes = 0;
        std::size_t committed_bytes = 0;
        std::vector<ExportableChunk> chunks;
        std::shared_ptr<void> state; // opaque OwnedAllocation

        [[nodiscard]] std::size_t committedPrefixBytes() const noexcept;
    };

    [[nodiscard]] inline std::size_t ExportableBlock::committedPrefixBytes() const noexcept {
        std::size_t prefix = 0;
        for (const auto& chunk : chunks) {
            if (chunk.offset > prefix) {
                break;
            }
            if (chunk.offset + chunk.bytes > prefix) {
                prefix = chunk.offset + chunk.bytes;
            }
        }
        return prefix;
    }

    // Indices of `chunks` whose offset is not in `bound_offsets`. Grow of a
    // packed multi-region block inserts new slices between existing region
    // tails, so `chunks[bound_count:]` is not the unbound set.
    [[nodiscard]] inline std::vector<std::size_t>
    unboundExportableChunkIndices(const std::vector<ExportableChunk>& chunks,
                                  const std::vector<std::size_t>& bound_offsets) {
        std::vector<std::size_t> unbound;
        unbound.reserve(chunks.size());
        for (std::size_t i = 0; i < chunks.size(); ++i) {
            bool bound = false;
            for (const std::size_t offset : bound_offsets) {
                if (offset == chunks[i].offset) {
                    bound = true;
                    break;
                }
            }
            if (!bound) {
                unbound.push_back(i);
            }
        }
        return unbound;
    }

    // CUDA VMM allocation granularity for `device` (2 MiB on current NVIDIA).
    [[nodiscard]] std::size_t exportable_allocation_granularity(int device = 0);

    // Allocates a block reserving max(reserve_bytes, size) virtual bytes and
    // committing [0, size) as shareable chunks. `size` must be non-zero.
    [[nodiscard]] std::expected<std::shared_ptr<ExportableBlock>, std::string>
    allocateExportableDeviceBlock(std::size_t size, int device = 0, bool track_splat_bytes = true,
                                  std::size_t reserve_bytes = 0);

    // Commits the granularity-aligned cover of [offset, offset+bytes) minus
    // ranges that are already committed. Holes are allowed. Existing chunks are
    // untouched.
    [[nodiscard]] std::expected<void, std::string>
    commitExportableDeviceRange(const std::shared_ptr<ExportableBlock>& block, std::size_t offset,
                                std::size_t bytes);

    // Commits [committedPrefixBytes, new_size) as new chunk(s). Returns true if
    // any new physical was committed, false if the prefix already covered
    // new_size. device_ptr is unchanged. Importers bind newly appended chunks.
    [[nodiscard]] std::expected<bool, std::string>
    growExportableDeviceBlock(const std::shared_ptr<ExportableBlock>& block, std::size_t new_size);

} // namespace lfs::core

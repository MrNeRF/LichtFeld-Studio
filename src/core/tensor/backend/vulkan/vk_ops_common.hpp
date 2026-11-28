/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "../../internal/tensor_impl.hpp"
#include "core/assert.hpp"
#include "core/detail/fused_pointwise.hpp"
#include "core/source_site.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>

namespace lfs::core::internal::vk {

    constexpr uint32_t kLocalSize = 256;

    // Device address of a kernel binding, guarded. Without VK_EXT_shader_64bit_indexing a kernel that indexes its buffer
    // from this address (an access chain) is only promised maxStorageBufferRange bytes past it, and NVIDIA's driver
    // computes the offsets in 32 bits. So on a guarded device (vulkan_address_guard_limit in vk_context.hpp) a binding
    // whose elements reach further throws and names the launcher, instead of the kernel reading and writing the wrong
    // addresses. The reach is from the ref's byte_end; a ref without one is let through (and logged in report mode
    // when its storage is past the limit), because a guard that refuses a safe call is worse than one that misses a
    // hand-made ref.
    inline uint64_t address(const StorageRef storage, const SourceSite site = LFS_SOURCE_SITE_CURRENT()) {
        LFS_ASSERT_MSG(storage.backend == GpuBackend::Vulkan && storage.meta != nullptr,
                       "Vulkan operation received non-Vulkan storage");
        if (const uint64_t limit = vulkan_address_guard_limit(); limit != 0) [[unlikely]] {
            const uint64_t size = storage.meta->gpu_descriptor.byte_size;
            const uint64_t room = size > storage.byte_offset ? size - storage.byte_offset : 0;
            if (room > limit) {
                if (storage.byte_end == 0)
                    vulkan_binding_past_guard(room, limit, false, site);
                else if (const uint64_t reach = storage.byte_end > storage.byte_offset
                                                    ? std::min<uint64_t>(storage.byte_end - storage.byte_offset, room)
                                                    : 0;
                         reach > limit)
                    vulkan_binding_past_guard(reach, limit, true, site);
            }
        }
        return storage.meta->gpu_descriptor.base_address + storage.byte_offset;
    }

    // Device address of a binding whose kernel forms every address past a small offset in 64-bit integer math (the
    // address form), so it may reach any distance from its address on every driver. Also for addresses that are not
    // kernel bindings.
    inline uint64_t wide_address(const StorageRef storage) {
        LFS_ASSERT_MSG(storage.backend == GpuBackend::Vulkan && storage.meta != nullptr,
                       "Vulkan operation received non-Vulkan storage");
        return storage.meta->gpu_descriptor.base_address + storage.byte_offset;
    }

    // The `bytes` of a ref from `offset` bytes past its address, for a launcher that binds each dispatch at its own
    // piece of a buffer: address(storage_piece(ref, offset, bytes)) is address(ref) + offset, and the guard measures
    // what that dispatch's kernel reaches from there instead of the whole buffer.
    inline StorageRef storage_piece(StorageRef storage, const size_t offset, const size_t bytes) {
        storage.byte_offset += offset;
        const size_t end = storage.byte_offset + bytes;
        storage.byte_end = storage.byte_end == 0 ? end : std::min(storage.byte_end, end);
        return storage;
    }

    inline uint32_t checked_u32(const size_t value, const char* const description) {
        LFS_ASSERT_MSG(value <= std::numeric_limits<uint32_t>::max(), description);
        return static_cast<uint32_t>(value);
    }

    // Scans radix.slang's digit-major block histograms ([lines, entries] uint32) into inclusive per-line
    // offsets; the scatter phase takes each block's own counts back off.
    void scan_radix_histogram(StorageRef histogram, size_t lines, size_t entries);

    // Kernels iterate grid-stride over NumWorkgroups, so a count above the
    // device's group limit (65535 on lavapipe) is covered by fewer groups.
    inline uint32_t dispatch_groups(const VulkanContext& context, const size_t work) {
        if (work == 0) {
            return 0;
        }
        const uint64_t groups = (work + kLocalSize - 1) / kLocalSize;
        return static_cast<uint32_t>(
            std::min<uint64_t>(groups, context.caps().max_workgroup_count[0]));
    }

    // Owns a device allocation for the duration of an adapter call; release is
    // timeline-tracked by VulkanMemory, so an exception between allocate and the
    // end of recording no longer leaks the block.
    class ScopedAllocation {
    public:
        ScopedAllocation(VulkanContext& context, const size_t bytes,
                         const bool host_visible = false)
            : context_(&context),
              storage_(host_visible ? context.memory().allocate_readback(bytes)
                                    : context.memory().allocate(bytes, 16, {})) {}
        ScopedAllocation(ScopedAllocation&& other) noexcept
            : context_(std::exchange(other.context_, nullptr)),
              storage_(other.storage_) {}
        ScopedAllocation& operator=(ScopedAllocation&& other) noexcept {
            if (this != &other) {
                release();
                context_ = std::exchange(other.context_, nullptr);
                storage_ = other.storage_;
            }
            return *this;
        }
        ScopedAllocation(const ScopedAllocation&) = delete;
        ScopedAllocation& operator=(const ScopedAllocation&) = delete;
        ~ScopedAllocation() { release(); }

        [[nodiscard]] StorageRef storage() const { return storage_; }

    private:
        void release() noexcept {
            if (context_ != nullptr) {
                context_->memory().deallocate(storage_);
                context_ = nullptr;
            }
        }

        VulkanContext* context_;
        StorageRef storage_;
    };

    struct ChainOp {
        uint32_t kind;
        float scalar;
        uint64_t rhs_address;
    };
    static_assert(sizeof(ChainOp) == 16);

    using ChainTable = std::array<ChainOp, tensor_ops::FUSED_POINTWISE_MAX_OPS>;

    // Uploads the chain descriptors to a device table the shader indexes by
    // element; the caller lists the table and every rhs operand as reads.
    inline ScopedAllocation upload_chain(VulkanContext& context,
                                         const tensor_ops::FusedPointwiseOpChain& chain) {
        LFS_ASSERT_MSG(chain.num_ops > 0 && chain.num_ops <= tensor_ops::FUSED_POINTWISE_MAX_OPS,
                       "Vulkan fused chain requires 1 to 16 operations");
        ChainTable descriptors{};
        for (int i = 0; i < chain.num_ops; ++i) {
            descriptors[i] = ChainOp{
                .kind = chain.ops[i].kind,
                .scalar = chain.ops[i].scalar,
                .rhs_address = static_cast<uint64_t>(
                    reinterpret_cast<uintptr_t>(chain.ops[i].rhs)),
            };
        }
        ScopedAllocation table(context, sizeof(descriptors));
        context.memory().copy_host_to_device(CopyRequest{
            .src = raw_storage_ref(descriptors.data()),
            .dst = table.storage(),
            .bytes = sizeof(descriptors),
            .synchronous = false,
        });
        return table;
    }

} // namespace lfs::core::internal::vk

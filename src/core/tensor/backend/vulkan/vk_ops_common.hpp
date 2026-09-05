/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once

#include "../../internal/tensor_impl.hpp"
#include "../../internal/tensor_ops.hpp"
#include "core/assert.hpp"
#include "vk_context.hpp"
#include "vk_memory.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <utility>

namespace lfs::core::internal::vk {

    constexpr uint32_t kLocalSize = 256;

    inline uint64_t address(const StorageRef storage) {
        LFS_ASSERT_MSG(storage.backend == GpuBackend::Vulkan && storage.meta != nullptr,
                       "Vulkan operation received non-Vulkan storage");
        return storage.meta->gpu_descriptor.base_address + storage.byte_offset;
    }

    inline uint32_t checked_u32(const size_t value, const char* const description) {
        LFS_ASSERT_MSG(value <= std::numeric_limits<uint32_t>::max(), description);
        return static_cast<uint32_t>(value);
    }

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

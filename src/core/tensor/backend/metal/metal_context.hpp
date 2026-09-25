/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "../../internal/tensor_impl.hpp"
#include "../gpu_backend_ops.hpp"

#include <cstdint>
#include <optional>

#ifdef __OBJC__
#import <Metal/Metal.h>

#include <array>
#include <atomic>
#include <initializer_list>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#endif

namespace lfs::core {
    struct GpuDeviceInfo;
}

namespace lfs::core::internal {

    bool metal_backend_available();
    bool metal_backend_live();
    void shutdown_metal_backend();
    std::optional<GpuDeviceInfo> metal_device_info();

    // Completion for TensorCompletion: work is submitted in batches numbered by
    // serial, and StorageMeta::pending_value holds the batch that last used it.
    uint64_t metal_flush();
    uint64_t metal_completed_serial();
    void metal_wait(uint64_t serial);

} // namespace lfs::core::internal

#ifdef __OBJC__
namespace lfs::core::internal::metal {

    // Scalar reduction ids, defined for kernels.metal as LFS_REDUCE_*.
    inline constexpr uint32_t kReduceSum = 0, kReduceMean = 1, kReduceMax = 2, kReduceMin = 3;

    struct Located {
        id<MTLBuffer> buffer;
        size_t offset;
    };

    class Context {
    public:
        Context();
        ~Context();

        Context(const Context&) = delete;
        Context& operator=(const Context&) = delete;

        id<MTLDevice> device() const { return device_; }

        // Pipelines are specialized by (function constant index, value) pairs.
        id<MTLComputePipelineState> pipeline(
            const char* function,
            std::initializer_list<std::pair<uint32_t, uint32_t>> constants = {});

        // Encodes one dispatch into the open batch and stamps every storage it
        // uses with the batch serial, which host access and completion wait for.
        template <class Encode>
        void encode(std::span<const StorageRef> uses, Encode&& encode) {
            std::lock_guard lock(encode_mutex_);
            encode(open_encoder_locked());
            for (const StorageRef& use : uses) {
                if (use.meta != nullptr)
                    const_cast<StorageMeta*>(use.meta)->pending_value.store(open_serial_, std::memory_order_release);
            }
            if (++open_dispatches_ >= kBatchDispatches)
                commit_locked();
        }

        uint64_t flush();
        void wait(uint64_t serial);
        uint64_t completed() const;
        void wait_idle();

        StorageRef allocate(size_t bytes);
        void release(const StorageRef& storage) noexcept;
        void trim();
        size_t cached_bytes();
        MemoryInfo stats();
        bool owns(const void* pointer);
        Located locate(const void* pointer);
        Located locate(const StorageRef& storage);
        // CPU view of shared storage; callers wait for last_use() before touching it.
        std::byte* host(const StorageRef& storage);
        uint64_t pending(const StorageRef& storage) const;
        // The newest batch that may still use this memory, including batches of
        // an earlier owner of a reused block.
        uint64_t last_use(const StorageRef& storage);

    private:
        static constexpr uint32_t kBatchDispatches = 64;

        // A block is reused as soon as it is released: batches run in order on
        // one queue with tracked hazards, so only CPU access and returning the
        // buffer to the system wait for guard, the last batch of its previous owner.
        struct Block {
            id<MTLBuffer> buffer;
            uint64_t address = 0;
            size_t capacity = 0;
            uint64_t guard = 0;
            std::unique_ptr<StorageMeta> meta;
        };

        struct PipelineKey {
            std::string_view function;
            std::array<uint32_t, 8> values{};
            uint32_t defined = 0;
            bool operator==(const PipelineKey&) const = default;
        };

        struct PipelineKeyHash {
            size_t operator()(const PipelineKey& key) const noexcept {
                size_t hash = std::hash<std::string_view>{}(key.function) ^ key.defined;
                for (const uint32_t value : key.values)
                    hash = hash * 1099511628211ull ^ value;
                return hash;
            }
        };

        id<MTLComputeCommandEncoder> open_encoder_locked();
        void commit_locked();
        void trim_locked();

        id<MTLDevice> device_;
        id<MTLCommandQueue> queue_;
        id<MTLLibrary> library_;
        uint64_t context_id_ = 0;

        std::mutex pipeline_mutex_;
        std::unordered_map<PipelineKey, id<MTLComputePipelineState>, PipelineKeyHash> pipelines_;

        std::mutex encode_mutex_;
        id<MTLCommandBuffer> command_buffer_;
        id<MTLComputeCommandEncoder> encoder_;
        uint64_t open_serial_ = 0;
        uint32_t open_dispatches_ = 0;
        std::atomic<uint64_t> submitted_{0};
        // The open batch's serial while one is open, else the last submitted.
        std::atomic<uint64_t> newest_serial_{0};

        // Every batch signals its serial on event_; a failed batch records failure_.
        id<MTLSharedEvent> event_;
        mutable std::mutex failure_mutex_;
        std::string failure_;

        std::mutex memory_mutex_;
        std::map<uint64_t, Block> live_;
        std::unordered_map<size_t, std::vector<Block>> free_;
    };

    std::shared_ptr<Context> acquire_context();
    std::shared_ptr<Context> live_context();

} // namespace lfs::core::internal::metal
#endif

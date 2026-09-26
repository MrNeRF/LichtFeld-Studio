/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor/internal/private_access.hpp"

#include "../../internal/expression_runtime.hpp"
#include "../../internal/tensor_impl.hpp"
#include "../gpu_backend_ops.hpp"

#include <cstdint>
#include <optional>

#ifdef __OBJC__
#import <Metal/Metal.h>

#include <array>
#include <atomic>
#include <cstddef>
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

    ExpressionCacheStats metal_expression_cache_stats();

} // namespace lfs::core::internal

#ifdef __OBJC__
namespace lfs::core::internal::metal {

    inline constexpr NSUInteger kThreadgroupWidth = 256;
    // Element code of float2 (sum, compensation) reduction partials, next to
    // the DataType codes; kernels.metal knows it as LFS_DT_Pair.
    inline constexpr uint32_t kPairDType = 255;

    // The backend is built on Metal 4, so everything below needs macOS 26;
    // metal_backend_available() gates every path into it.
    API_AVAILABLE_BEGIN(macos(26.0))

    struct Located {
        id<MTLBuffer> buffer;
        uint64_t address; // GPU address of the buffer's first byte
        size_t offset;
    };

    // One compute dispatch. Buffers bind by GPU address at [[buffer(0)]] onward
    // and the parameter block, copied into the batch, at the next index.
    struct Dispatch {
        id<MTLComputePipelineState> pipeline;
        std::initializer_list<uint64_t> buffers;
        std::span<const std::byte> params;
        // Threads to run, or threadgroups when group_size is set.
        MTLSize grid;
        MTLSize group_size{};
    };

    template <class Params>
    std::span<const std::byte> param_bytes(const Params& params) {
        return std::as_bytes(std::span(&params, 1));
    }

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
        // Pipelines of fused expressions, compiled from generated MSL.
        id<MTLComputePipelineState> expression_pipeline(const ExpressionProgram& program,
                                                        const ExpressionSignature& signature);
        ExpressionCache& expressions() { return expressions_; }

        // Encodes one dispatch into the open batch, ordered after every earlier
        // dispatch, and stamps every storage it uses with the batch serial,
        // which host access and completion wait for.
        void dispatch(std::span<const StorageRef> uses, const Dispatch& dispatch);

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
        // where_select binds four buffers and its parameters.
        static constexpr NSUInteger kArgumentSlots = 5;
        // Kernels record the first out-of-range index of their batch in the
        // fault record bound after the argument slots; the first wait after
        // the batch completed raises it as a BoundsViolation.
        static constexpr NSUInteger kFaultSlot = kArgumentSlots;
        static constexpr size_t kParamsAlignment = 256;
        // Fused expressions pass up to ExpressionLayout::max_words argument words.
        static constexpr size_t kMaxParamsBytes = ExpressionLayout::max_words * sizeof(uint32_t);
        // Batches in flight before recording waits for the oldest.
        static constexpr size_t kMaxFrames = 64;

        // A block is reused as soon as it is released: batches run in order, so
        // only CPU access and returning the buffer to the system wait for guard,
        // the last batch of its previous owner.
        struct Block {
            id<MTLBuffer> buffer;
            uint64_t address = 0;
            size_t capacity = 0;
            uint64_t guard = 0;
            std::unique_ptr<StorageMeta> meta;
        };

        // A batch records into a frame, its command memory and the parameter
        // blocks of its dispatches, which are reused once the batch completes.
        // Command buffers do not retain pipelines, and the expression cache may
        // evict one while its batch runs, so the frame holds them.
        struct Frame {
            id<MTL4CommandAllocator> allocator;
            id<MTLBuffer> params;
            std::vector<id<MTLComputePipelineState>> pipelines;
            uint64_t serial = 0;
        };

        // Written by commit feedback, which may arrive after the context is gone.
        struct Failure {
            std::mutex mutex;
            std::string message;
        };

        struct PipelineKey {
            std::string_view function;
            std::array<uint32_t, 32> values{};
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

        id<MTL4ComputeCommandEncoder> open_encoder_locked();
        void prepare_locked();
        size_t acquire_frame_locked();
        void commit_locked();
        void wait_signaled(uint64_t serial);
        void check_failures() const;
        void check_fault();
        void consume_fault_locked(size_t slot);
        void evict_locked(size_t limit);

        id<MTLDevice> device_;
        id<MTLLibrary> library_;
        uint64_t context_id_ = 0;

        std::mutex pipeline_mutex_;
        std::unordered_map<PipelineKey, id<MTLComputePipelineState>, PipelineKeyHash> pipelines_;
        ExpressionCache expressions_;

        // Batches are recorded into one command buffer, reused after each commit.
        std::mutex encode_mutex_;
        id<MTL4CommandQueue> queue_;
        id<MTL4CommandBuffer> command_buffer_;
        id<MTL4ComputeCommandEncoder> encoder_;
        id<MTL4ArgumentTable> arguments_;
        MTL4CommitOptions* commit_options_;
        std::vector<Frame> frames_;
        size_t frame_ = 0;
        size_t params_used_ = 0;
        uint64_t open_serial_ = 0;
        uint32_t open_dispatches_ = 0;
        std::atomic<uint64_t> submitted_{0};
        // The open batch's serial while one is open, else the last submitted.
        std::atomic<uint64_t> newest_serial_{0};

        // The queue signals every batch's serial on event_ after the batch.
        id<MTLSharedEvent> event_;
        std::shared_ptr<Failure> failure_;

        // One record per batch in flight, so a record is read only after its
        // batch completed and before a later batch reuses it.
        std::mutex fault_mutex_;
        id<MTLBuffer> fault_;
        std::array<uint64_t, kMaxFrames> fault_serials_{};
        std::array<uint32_t, 4> fault_record_{};

        // Every buffer the context creates stays resident for the queue.
        std::mutex memory_mutex_;
        id<MTLResidencySet> residency_;
        std::map<uint64_t, Block> live_;
        // Released blocks by capacity, kept for reuse up to cache_limit_ bytes.
        std::map<size_t, std::vector<Block>> free_;
        size_t cached_bytes_ = 0;
        size_t cache_limit_ = 0;
    };

    std::shared_ptr<Context> acquire_context();
    std::shared_ptr<Context> live_context();

    API_AVAILABLE_END

} // namespace lfs::core::internal::metal
#endif

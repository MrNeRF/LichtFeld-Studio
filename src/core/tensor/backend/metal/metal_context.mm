/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "metal_context.hpp"

#include "core/assert.hpp"
#include "core/gpu_device_info.hpp"

#include <algorithm>
#include <bit>
#include <format>
#include <string>

namespace lfs::core::internal::metal {

    extern const char* const kKernelSource;

    namespace {
        std::string kernel_source() {
            std::string source;
#define LFS_POINTWISE_OP(Id, FunctorType, Name) \
    source += std::format("#define LFS_OP_{} {}\n", #Id, static_cast<unsigned>(PointwiseOp::Id));
#include "core/detail/pointwise_ops.def"
#undef LFS_POINTWISE_OP
            for (const auto& [name, dtype] : {std::pair{"Float32", DataType::Float32},
                                              {"Float16", DataType::Float16},
                                              {"Int32", DataType::Int32},
                                              {"Int64", DataType::Int64},
                                              {"UInt8", DataType::UInt8},
                                              {"Bool", DataType::Bool},
                                              {"UInt32", DataType::UInt32}}) {
                source += std::format("#define LFS_DT_{} {}\n", name, static_cast<unsigned>(dtype));
            }
            source += std::format("#define LFS_REDUCE_SUM {}\n#define LFS_REDUCE_MEAN {}\n"
                                  "#define LFS_REDUCE_MAX {}\n#define LFS_REDUCE_MIN {}\n",
                                  kReduceSum, kReduceMean, kReduceMax, kReduceMin);
            return source + kKernelSource;
        }

        // Power-of-two size classes keep reuse simple; large blocks round to 2 MiB.
        size_t size_class(const size_t bytes) {
            constexpr size_t kLargeBlock = size_t{64} << 20;
            constexpr size_t kLargeGranule = size_t{2} << 20;
            if (bytes <= 256)
                return 256;
            if (bytes <= kLargeBlock)
                return std::bit_ceil(bytes);
            return (bytes + kLargeGranule - 1) / kLargeGranule * kLargeGranule;
        }

        std::atomic<uint64_t> next_context_id{1};
        std::mutex context_mutex;
        std::shared_ptr<Context> context_instance;
    } // namespace

    Context::Context() {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_ || ![device_ supportsFamily:MTLGPUFamilyMetal3])
            throw TensorError("No Metal 3 device is available");
        queue_ = [device_ newCommandQueue];
        event_ = [device_ newSharedEvent];
        context_id_ = next_context_id.fetch_add(1);

        // Match the Vulkan kernels, which are compiled precise: no fast math,
        // precise transcendental functions, and no contraction (in the source).
        MTLCompileOptions* const options = [MTLCompileOptions new];
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        NSError* error = nil;
        library_ = [device_ newLibraryWithSource:@(kernel_source().c_str()) options:options error:&error];
        if (!library_)
            throw TensorError(std::format("Metal tensor kernels failed to compile: {}",
                                          error.localizedDescription.UTF8String));
    }

    Context::~Context() {
        try {
            wait_idle();
        } catch (...) {
            // LFS-CENSUS-OK(empty-catch): A failed batch must not keep the device alive.
        }
    }

    id<MTLComputePipelineState> Context::pipeline(
        const char* const function,
        const std::initializer_list<std::pair<uint32_t, uint32_t>> constants) {
        PipelineKey key{.function = function};
        for (const auto& [index, value] : constants) {
            LFS_ASSERT_MSG(index < key.values.size(), "Metal function constant index out of range");
            key.values[index] = value;
            key.defined |= 1u << index;
        }
        std::lock_guard lock(pipeline_mutex_);
        if (const auto found = pipelines_.find(key); found != pipelines_.end())
            return found->second;
        MTLFunctionConstantValues* const values = [MTLFunctionConstantValues new];
        for (const auto& [index, value] : constants)
            [values setConstantValue:&value type:MTLDataTypeUInt atIndex:index];
        NSError* error = nil;
        id<MTLFunction> const kernel = [library_ newFunctionWithName:@(function) constantValues:values error:&error];
        id<MTLComputePipelineState> const state =
            kernel ? [device_ newComputePipelineStateWithFunction:kernel error:&error] : nil;
        if (!state)
            throw TensorError(std::format("Metal pipeline '{}' failed: {}", function,
                                          error ? error.localizedDescription.UTF8String : "unknown error"));
        pipelines_.emplace(key, state);
        return state;
    }

    id<MTLComputeCommandEncoder> Context::open_encoder_locked() {
        if (!command_buffer_) {
            // Buffers outlive every batch that uses them (see trim_locked), so
            // batches need not retain their resources.
            command_buffer_ = [queue_ commandBufferWithUnretainedReferences];
            encoder_ = [command_buffer_ computeCommandEncoder];
            open_serial_ = submitted_.load(std::memory_order_relaxed) + 1;
            newest_serial_.store(open_serial_, std::memory_order_release);
        }
        return encoder_;
    }

    void Context::commit_locked() {
        if (!command_buffer_)
            return;
        [encoder_ endEncoding];
        const uint64_t serial = open_serial_;
        [command_buffer_ encodeSignalEvent:event_ value:serial];
        [command_buffer_ addCompletedHandler:^(id<MTLCommandBuffer> buffer) {
          if (buffer.status != MTLCommandBufferStatusError)
              return;
          std::lock_guard lock(failure_mutex_);
          if (failure_.empty())
              failure_ = buffer.error ? buffer.error.localizedDescription.UTF8String : "unknown error";
        }];
        [command_buffer_ commit];
        submitted_.store(serial, std::memory_order_release);
        command_buffer_ = nil;
        encoder_ = nil;
        open_dispatches_ = 0;
    }

    uint64_t Context::flush() {
        std::lock_guard lock(encode_mutex_);
        commit_locked();
        return submitted_.load(std::memory_order_acquire);
    }

    void Context::wait(const uint64_t serial) {
        if (serial == 0)
            return;
        {
            std::lock_guard lock(encode_mutex_);
            if (command_buffer_ && serial >= open_serial_)
                commit_locked();
        }
        // A failed batch may never signal, so the wait checks for failures in slices.
        while (![event_ waitUntilSignaledValue:serial timeoutMS:100]) {
            std::lock_guard lock(failure_mutex_);
            if (!failure_.empty())
                break;
        }
        std::lock_guard lock(failure_mutex_);
        if (!failure_.empty())
            throw TensorError(std::format("Metal tensor work failed: {}", failure_));
    }

    uint64_t Context::completed() const {
        return event_.signaledValue;
    }

    void Context::wait_idle() {
        wait(flush());
    }

    StorageRef Context::allocate(const size_t bytes) {
        const size_t capacity = size_class(bytes);
        std::lock_guard lock(memory_mutex_);
        Block block;
        if (auto found = free_.find(capacity); found != free_.end() && !found->second.empty()) {
            block = std::move(found->second.back());
            found->second.pop_back();
        } else {
            id<MTLBuffer> buffer = [device_ newBufferWithLength:capacity options:MTLResourceStorageModeShared];
            if (!buffer) {
                trim_locked();
                buffer = [device_ newBufferWithLength:capacity options:MTLResourceStorageModeShared];
            }
            if (!buffer)
                throw TensorError(std::format("Metal tensor allocation of {} bytes failed", capacity));
            block.buffer = buffer;
            block.address = buffer.gpuAddress;
            block.capacity = capacity;
            block.meta = std::make_unique<StorageMeta>();
            block.meta->backend = GpuBackend::Metal;
            block.meta->gpu_descriptor = GpuStorageDescriptor{
                .native_buffer = reinterpret_cast<uint64_t>((__bridge void*)buffer),
                .native_allocation = 0,
                .native_context = context_id_,
                .base_address = block.address,
                .byte_size = capacity,
                .accounting_kind = StorageAccountingKind::MetalOwned,
            };
        }
        block.meta->pending_value.store(block.guard, std::memory_order_relaxed);
        const StorageRef storage{
            .backend = GpuBackend::Metal,
            .data = reinterpret_cast<void*>(block.address),
            .byte_offset = 0,
            .dtype = DataType::UInt8,
            .meta = block.meta.get(),
        };
        live_.emplace(block.address, std::move(block));
        return storage;
    }

    void Context::release(const StorageRef& storage) noexcept {
        std::lock_guard lock(memory_mutex_);
        const auto found = live_.find(reinterpret_cast<uint64_t>(storage.data));
        if (found == live_.end())
            return;
        Block block = std::move(found->second);
        live_.erase(found);
        block.guard = newest_serial_.load(std::memory_order_acquire);
        free_[block.capacity].push_back(std::move(block));
    }

    // Batches do not retain their buffers, so only blocks whose last batch
    // completed go back to the system.
    void Context::trim_locked() {
        const uint64_t done = completed();
        for (auto& [capacity, blocks] : free_)
            std::erase_if(blocks, [&](const Block& block) { return block.guard <= done; });
    }

    void Context::trim() {
        std::lock_guard lock(memory_mutex_);
        trim_locked();
    }

    size_t Context::cached_bytes() {
        std::lock_guard lock(memory_mutex_);
        size_t bytes = 0;
        for (const auto& [capacity, blocks] : free_)
            bytes += capacity * blocks.size();
        return bytes;
    }

    MemoryInfo Context::stats() {
        MemoryInfo result;
        result.total_bytes = static_cast<size_t>(device_.recommendedMaxWorkingSetSize);
        result.allocated_bytes = static_cast<size_t>(device_.currentAllocatedSize);
        result.free_bytes = result.total_bytes > result.allocated_bytes ? result.total_bytes - result.allocated_bytes : 0;
        result.device_id = 0;
        return result;
    }

    bool Context::owns(const void* const pointer) {
        const auto address = reinterpret_cast<uint64_t>(pointer);
        std::lock_guard lock(memory_mutex_);
        auto found = live_.upper_bound(address);
        if (found == live_.begin())
            return false;
        --found;
        return address < found->first + found->second.capacity;
    }

    Located Context::locate(const void* const pointer) {
        const auto address = reinterpret_cast<uint64_t>(pointer);
        std::lock_guard lock(memory_mutex_);
        auto found = live_.upper_bound(address);
        LFS_ASSERT_MSG(found != live_.begin(), "Metal operation received storage it does not own");
        --found;
        LFS_ASSERT_MSG(address < found->first + found->second.capacity,
                       "Metal operation received storage it does not own");
        return {found->second.buffer, static_cast<size_t>(address - found->first)};
    }

    Located Context::locate(const StorageRef& storage) {
        LFS_ASSERT_MSG(storage.backend == GpuBackend::Metal, "Metal operation received non-Metal storage");
        if (storage.meta != nullptr && storage.meta->gpu_descriptor.native_buffer != 0) {
            const auto& descriptor = storage.meta->gpu_descriptor;
            return {(__bridge id<MTLBuffer>)reinterpret_cast<void*>(descriptor.native_buffer),
                    static_cast<size_t>(reinterpret_cast<uint64_t>(storage.data) - descriptor.base_address) +
                        storage.byte_offset};
        }
        Located located = locate(storage.data);
        located.offset += storage.byte_offset;
        return located;
    }

    std::byte* Context::host(const StorageRef& storage) {
        const Located located = locate(storage);
        return static_cast<std::byte*>(located.buffer.contents) + located.offset;
    }

    uint64_t Context::pending(const StorageRef& storage) const {
        return storage.meta != nullptr ? storage.meta->pending_value.load(std::memory_order_acquire)
                                       : newest_serial_.load(std::memory_order_acquire);
    }

    uint64_t Context::last_use(const StorageRef& storage) {
        const auto address = reinterpret_cast<uint64_t>(storage.data);
        std::lock_guard lock(memory_mutex_);
        auto found = live_.upper_bound(address);
        const uint64_t guard = found != live_.begin() ? std::prev(found)->second.guard : 0;
        return std::max(pending(storage), guard);
    }

    std::shared_ptr<Context> acquire_context() {
        std::lock_guard lock(context_mutex);
        if (!context_instance)
            context_instance = std::make_shared<Context>();
        return context_instance;
    }

    std::shared_ptr<Context> live_context() {
        std::lock_guard lock(context_mutex);
        return context_instance;
    }

} // namespace lfs::core::internal::metal

namespace lfs::core::internal {

    bool metal_backend_available() {
        static const bool available = [] {
            id<MTLDevice> const device = MTLCreateSystemDefaultDevice();
            return device != nil && [device supportsFamily:MTLGPUFamilyMetal3];
        }();
        return available;
    }

    bool metal_backend_live() {
        return metal::live_context() != nullptr;
    }

    void shutdown_metal_backend() {
        std::shared_ptr<metal::Context> context;
        {
            std::lock_guard lock(metal::context_mutex);
            context = std::move(metal::context_instance);
        }
        if (context)
            context->wait_idle();
    }

    std::optional<GpuDeviceInfo> metal_device_info() {
        const auto context = metal::live_context();
        if (!context)
            return std::nullopt;
        return GpuDeviceInfo{
            .name = context->device().name.UTF8String,
            .total_memory_bytes = static_cast<size_t>(context->device().recommendedMaxWorkingSetSize),
        };
    }

    uint64_t metal_flush() {
        const auto context = metal::live_context();
        return context ? context->flush() : 0;
    }

    uint64_t metal_completed_serial() {
        const auto context = metal::live_context();
        return context ? context->completed() : 0;
    }

    void metal_wait(const uint64_t serial) {
        if (const auto context = metal::live_context())
            context->wait(serial);
    }

} // namespace lfs::core::internal

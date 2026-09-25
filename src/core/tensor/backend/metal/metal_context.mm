/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "metal_context.hpp"

#include "core/assert.hpp"
#include "core/gpu_device_info.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>
#include <format>
#include <string>

namespace lfs::core::internal::metal {

    extern const char* const kKernelSource;

    API_AVAILABLE_BEGIN(macos(26.0))

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
            source += std::format("#define LFS_DT_Pair {}\n", kPairDType);
            for (const auto& [name, op] : {std::pair{"SUM", ReduceOp::Sum},
                                           {"MEAN", ReduceOp::Mean},
                                           {"MAX", ReduceOp::Max},
                                           {"MIN", ReduceOp::Min},
                                           {"PROD", ReduceOp::Prod},
                                           {"ANY", ReduceOp::Any},
                                           {"ALL", ReduceOp::All}}) {
                source += std::format("#define LFS_REDUCE_{} {}\n", name, static_cast<unsigned>(op));
            }
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
        if (!device_ || ![device_ supportsFamily:MTLGPUFamilyMetal4])
            throw TensorError("No Metal 4 device is available");
        NSError* error = nil;
        MTL4ArgumentTableDescriptor* const arguments = [MTL4ArgumentTableDescriptor new];
        arguments.maxBufferBindCount = kArgumentSlots;
        queue_ = [device_ newMTL4CommandQueue];
        command_buffer_ = [device_ newCommandBuffer];
        arguments_ = [device_ newArgumentTableWithDescriptor:arguments error:&error];
        residency_ = [device_ newResidencySetWithDescriptor:[MTLResidencySetDescriptor new] error:&error];
        event_ = [device_ newSharedEvent];
        if (!queue_ || !command_buffer_ || !arguments_ || !residency_ || !event_)
            throw TensorError(std::format("Metal 4 queue setup failed: {}",
                                          error ? error.localizedDescription.UTF8String : "unknown error"));
        [queue_ addResidencySet:residency_];
        context_id_ = next_context_id.fetch_add(1);

        const std::shared_ptr<Failure> failure = failure_ = std::make_shared<Failure>();
        commit_options_ = [MTL4CommitOptions new];
        [commit_options_ addFeedbackHandler:^(id<MTL4CommitFeedback> feedback) {
            if (feedback.error == nil)
                return;
            std::lock_guard lock(failure->mutex);
            if (failure->message.empty())
                failure->message = feedback.error.localizedDescription.UTF8String;
        }];

        // Match the Vulkan kernels, which are compiled precise: no fast math,
        // precise transcendental functions, and no contraction (in the source).
        // MSL 4 brings the tensor types the matrix-unit GEMM is built on.
        MTLCompileOptions* const options = [MTLCompileOptions new];
        options.languageVersion = MTLLanguageVersion4_0;
        options.mathMode = MTLMathModeSafe;
        options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
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
        if (encoder_) {
            [encoder_ endEncoding];
            [command_buffer_ endCommandBuffer];
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

    void Context::dispatch(const std::span<const StorageRef> uses, const Dispatch& dispatch) {
        LFS_ASSERT_MSG(dispatch.buffers.size() + (dispatch.params.empty() ? 0 : 1) <= kArgumentSlots &&
                           dispatch.params.size() <= kMaxParamsBytes,
                       "Metal dispatch exceeds its argument slots");
        std::lock_guard lock(encode_mutex_);
        id<MTL4ComputeCommandEncoder> const encoder = open_encoder_locked();
        if (open_dispatches_ > 0)
            [encoder barrierAfterEncoderStages:MTLStageDispatch
                           beforeEncoderStages:MTLStageDispatch
                             visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder setComputePipelineState:dispatch.pipeline];
        NSUInteger slot = 0;
        for (const uint64_t address : dispatch.buffers)
            [arguments_ setAddress:address atIndex:slot++];
        if (!dispatch.params.empty()) {
            id<MTLBuffer> const params = frames_[frame_].params;
            std::memcpy(static_cast<std::byte*>(params.contents) + params_used_, dispatch.params.data(),
                        dispatch.params.size());
            [arguments_ setAddress:params.gpuAddress + params_used_ atIndex:slot];
            params_used_ += (dispatch.params.size() + kParamsAlignment - 1) / kParamsAlignment * kParamsAlignment;
        }
        if (dispatch.group_size.width == 0) {
            const NSUInteger width = std::min(dispatch.pipeline.maxTotalThreadsPerThreadgroup, kThreadgroupWidth);
            [encoder dispatchThreads:dispatch.grid threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        } else {
            [encoder dispatchThreadgroups:dispatch.grid threadsPerThreadgroup:dispatch.group_size];
        }
        for (const StorageRef& use : uses) {
            if (use.meta != nullptr)
                const_cast<StorageMeta*>(use.meta)->pending_value.store(open_serial_, std::memory_order_release);
        }
        // An idle GPU starts at once; while it is busy, dispatches batch up.
        if (++open_dispatches_ >= kBatchDispatches || completed() >= submitted_.load(std::memory_order_relaxed))
            commit_locked();
    }

    id<MTL4ComputeCommandEncoder> Context::open_encoder_locked() {
        if (!encoder_)
            prepare_locked();
        if (open_dispatches_ == 0) {
            open_serial_ = submitted_.load(std::memory_order_relaxed) + 1;
            newest_serial_.store(open_serial_, std::memory_order_release);
            frames_[frame_].serial = open_serial_;
        }
        return encoder_;
    }

    void Context::prepare_locked() {
        frame_ = acquire_frame_locked();
        [frames_[frame_].allocator reset];
        params_used_ = 0;
        [command_buffer_ beginCommandBufferWithAllocator:frames_[frame_].allocator];
        encoder_ = [command_buffer_ computeCommandEncoder];
        // Batches run in submission order.
        [encoder_ barrierAfterQueueStages:MTLStageDispatch
                             beforeStages:MTLStageDispatch
                        visibilityOptions:MTL4VisibilityOptionDevice];
        [encoder_ setArgumentTable:arguments_];
    }

    size_t Context::acquire_frame_locked() {
        const uint64_t done = completed();
        for (size_t index = 0; index < frames_.size(); ++index) {
            if (frames_[index].serial <= done)
                return index;
        }
        if (frames_.size() == kMaxFrames) {
            const auto oldest = std::ranges::min_element(frames_, {}, &Frame::serial);
            wait_signaled(oldest->serial);
            return static_cast<size_t>(oldest - frames_.begin());
        }
        Frame frame{
            .allocator = [device_ newCommandAllocator],
            .params = [device_ newBufferWithLength:kBatchDispatches * kMaxParamsBytes
                                           options:MTLResourceStorageModeShared],
        };
        if (!frame.allocator || !frame.params)
            throw TensorError("Metal command memory allocation failed");
        {
            std::lock_guard lock(memory_mutex_);
            [residency_ addAllocation:frame.params];
            [residency_ commit];
        }
        frames_.push_back(frame);
        return frames_.size() - 1;
    }

    void Context::commit_locked() {
        if (open_dispatches_ == 0)
            return;
        [encoder_ endEncoding];
        encoder_ = nil;
        [command_buffer_ endCommandBuffer];
        const uint64_t serial = open_serial_;
        id<MTL4CommandBuffer> const buffers[] = {command_buffer_};
        [queue_ commit:buffers count:1 options:commit_options_];
        [queue_ signalEvent:event_ value:serial];
        submitted_.store(serial, std::memory_order_release);
        open_dispatches_ = 0;
        // The GPU is already running; setting up the next batch here keeps it
        // off the next dispatch's path.
        prepare_locked();
    }

    void Context::check_failures() const {
        std::lock_guard lock(failure_->mutex);
        if (!failure_->message.empty())
            throw TensorError(std::format("Metal tensor work failed: {}", failure_->message));
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
            if (open_dispatches_ > 0 && serial >= open_serial_)
                commit_locked();
        }
        wait_signaled(serial);
    }

    void Context::wait_signaled(const uint64_t serial) {
        // A GPU round trip takes ~150 us and a blocking wait adds wake-up latency,
        // so spin briefly first.
        const auto spin_until = std::chrono::steady_clock::now() + std::chrono::microseconds(300);
        while (completed() < serial && std::chrono::steady_clock::now() < spin_until) {
        }
        while (completed() < serial && ![event_ waitUntilSignaledValue:serial timeoutMS:100])
            check_failures();
        check_failures();
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
            [residency_ addAllocation:buffer];
            [residency_ commit];
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
        bool removed = false;
        for (auto& [capacity, blocks] : free_) {
            std::erase_if(blocks, [&](const Block& block) {
                if (block.guard > done)
                    return false;
                [residency_ removeAllocation:block.buffer];
                removed = true;
                return true;
            });
        }
        if (removed)
            [residency_ commit];
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
        return {found->second.buffer, found->first, static_cast<size_t>(address - found->first)};
    }

    Located Context::locate(const StorageRef& storage) {
        LFS_ASSERT_MSG(storage.backend == GpuBackend::Metal, "Metal operation received non-Metal storage");
        if (storage.meta != nullptr && storage.meta->gpu_descriptor.native_buffer != 0) {
            const auto& descriptor = storage.meta->gpu_descriptor;
            return {(__bridge id<MTLBuffer>)reinterpret_cast<void*>(descriptor.native_buffer),
                    descriptor.base_address,
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

    API_AVAILABLE_END

} // namespace lfs::core::internal::metal

namespace lfs::core::internal {

    bool metal_backend_available() {
        static const bool available = [] {
            if (@available(macOS 26.0, *)) {
                id<MTLDevice> const device = MTLCreateSystemDefaultDevice();
                return device != nil && [device supportsFamily:MTLGPUFamilyMetal4];
            }
            return false;
        }();
        return available;
    }

    bool metal_backend_live() {
        if (@available(macOS 26.0, *))
            return metal::live_context() != nullptr;
        return false;
    }

    void shutdown_metal_backend() {
        if (@available(macOS 26.0, *)) {
            std::shared_ptr<metal::Context> context;
            {
                std::lock_guard lock(metal::context_mutex);
                context = std::move(metal::context_instance);
            }
            if (context)
                context->wait_idle();
        }
    }

    std::optional<GpuDeviceInfo> metal_device_info() {
        if (@available(macOS 26.0, *)) {
            if (const auto context = metal::live_context()) {
                return GpuDeviceInfo{
                    .name = context->device().name.UTF8String,
                    .total_memory_bytes = static_cast<size_t>(context->device().recommendedMaxWorkingSetSize),
                };
            }
        }
        return std::nullopt;
    }

    uint64_t metal_flush() {
        if (@available(macOS 26.0, *)) {
            if (const auto context = metal::live_context())
                return context->flush();
        }
        return 0;
    }

    uint64_t metal_completed_serial() {
        if (@available(macOS 26.0, *)) {
            if (const auto context = metal::live_context())
                return context->completed();
        }
        return 0;
    }

    void metal_wait(const uint64_t serial) {
        if (@available(macOS 26.0, *)) {
            if (const auto context = metal::live_context())
                context->wait(serial);
        }
    }

} // namespace lfs::core::internal

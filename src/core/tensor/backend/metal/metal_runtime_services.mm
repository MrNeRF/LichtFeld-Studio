/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/expression_pairing.hpp"

#include "../native_allocation.hpp"
#include "../prepared_readback.hpp"

#include "../../internal/expression_runtime.hpp"
#include "../tensor_vulkan_interop.hpp"
#include "metal_context.hpp"

#include "../../internal/tensor_impl.hpp"
#include "../tensor_completion.hpp"
#include "metal_backend_ops.hpp"
#include "metal_queue.hpp"
#include <stdexcept>
#include <vector>

namespace lfs::core::internal {
namespace {
    class MetalUpload final : public BackendUploadBuffer {
        MetalBackendOps& ops_;
        uint64_t serial_ = 0;
        bool in_batch_ = false;
        TensorCompletion completion_;

    public:
        explicit MetalUpload(MetalBackendOps& ops)
            : ops_(ops)
        {
        }
        BackendUploadSubmission enqueue(Tensor& destination, const Tensor& source, BackendQueueTarget target, bool in_batch) override
        {
            if (!metal_queue::valid(target.current ? 0 : reinterpret_cast<uint64_t>(target.handle)))
                throw std::invalid_argument("Execution target is not a Metal tensor queue");
            destination.set_stream(nullptr);
            ops_.copy_host_to_device({ .src = raw_storage_ref(const_cast<void*>(source.data_ptr()), source.dtype()),
                .dst = storage_ref(destination),
                .bytes = source.bytes(),
                .synchronous = false });
            const auto* meta = storage_ref(destination).meta;
            serial_ = meta ? meta->pending_value.load(std::memory_order_acquire) : 0;
            in_batch_ = in_batch;
            const bool complete = in_batch ? metal_queue::completed(serial_) : metal_queue::ready(serial_);
            completion_ = complete ? TensorCompletion {} : TensorCompletionAccess::record(GpuBackend::Metal, { .value = serial_, .settle_on_release = false });
            return { complete, true };
        }
        BackendUploadSubmission enqueue_bytes(Tensor& destination, std::span<const std::byte> source, BackendQueueTarget target, bool in_batch) override
        {
            const auto snapshot = Tensor::from_blob(const_cast<std::byte*>(source.data()), { destination.numel() }, Device::CPU, destination.dtype());
            return enqueue(destination, snapshot, target, in_batch);
        }
        void wait() override { completion_.wait(); }
        bool poll() override { return in_batch_ ? metal_queue::completed(serial_) : completion_.ready(); }

        void settle_enqueue_failure() override { ops_.synchronize_stream({}); }
    };
    class MetalTimestamps final : public BackendTimestamps {
        std::unique_ptr<metal_queue::Timestamps> timestamps_;
        struct Mark {
            uint64_t timeline = 0;
            bool written = false;
        };
        std::vector<Mark> marks_;

    public:
        explicit MetalTimestamps(size_t count)
            : timestamps_(metal_queue::timestamps(count))
            , marks_(count)
        {
        }
        bool available() const { return timestamps_ != nullptr; }
        bool mark(size_t index, void* target) override
        {
            if (!metal_queue::valid(reinterpret_cast<uint64_t>(target)))
                throw std::invalid_argument("GpuElapsed target is not a Metal tensor queue");
            if (index >= marks_.size())
                return false;
            marks_[index] = { timestamps_->write(index), true };
            return true;
        }
        bool wait_event(size_t index) override
        {
            if (index >= marks_.size() || !marks_[index].written)
                return false;
            metal_queue::wait(marks_[index].timeline);
            return true;
        }
        bool wait_queue(void* target) override
        {
            if (!metal_queue::valid(reinterpret_cast<uint64_t>(target)))
                throw std::invalid_argument("GpuElapsed target is not a Metal tensor queue");
            metal_queue::wait(metal_queue::submit());
            return true;
        }
        std::optional<float> milliseconds(size_t begin, size_t end) const override
        {
            return timestamps_->milliseconds(begin, end);
        }
    };
}
std::unique_ptr<BackendTimestamps> MetalBackendOps::create_timestamps(size_t count)
{
    auto result = std::make_unique<MetalTimestamps>(count);
    return result->available() ? std::move(result) : nullptr;
}
std::unique_ptr<BackendUploadBuffer> MetalBackendOps::create_upload_buffer()
{
    return std::make_unique<MetalUpload>(*this);
}
namespace {
    class MetalFence final : public BackendFence {
        uint64_t value_ = 0;
        static void validate(ExecContext target)
        {
            if (!metal_queue::valid(reinterpret_cast<uint64_t>(target.cuda_stream)))
                throw std::invalid_argument("Execution target is not a Metal tensor queue");
        }

    public:
        explicit MetalFence(std::optional<void*> imported)
        {
            if (imported)
                value_ = reinterpret_cast<uint64_t>(*imported);
        }
        void record(ExecContext target) override
        {
            validate(target);
            value_ = metal_queue::submit();
        }
        void wait_on(ExecContext target) const override { validate(target); }
        void wait() const override { metal_queue::wait(value_); }
        bool ready() const override { return metal_queue::ready(value_); }
    };
}
std::unique_ptr<BackendFence> MetalBackendOps::create_fence(std::optional<void*> imported)
{
    if (!imported && !metal_backend_available())
        return {};
    return std::make_unique<MetalFence>(imported);
}
void MetalBackendOps::where_scalar(Tensor& output, const Tensor& mask, float value, const Tensor& source)
{
    metal_where_into(output, mask, value, source);
}
ExpressionCacheStats MetalBackendOps::expression_cache_stats() { return metal_expression_cache_stats(); }
namespace {
    class MetalReadbackSlot final : public BackendReadbackSlot {
        MetalBackendOps& ops_;
        Tensor host_;

    public:
        MetalReadbackSlot(MetalBackendOps& ops, size_t bytes)
            : ops_(ops)
        {
            const GpuBackendScope scope(GpuBackend::Metal);
            host_ = Tensor::zeros({ bytes }, Device::GPU, DataType::UInt8);
        }
        void* data() const override { return metal_queue::host(storage_ref(host_)); }
        bool begin_append(const Tensor& source, size_t offset, size_t bytes, size_t destination, ExecContext) override
        {
            auto input = storage_ref(source), output = storage_ref(host_);
            input.byte_offset += offset;
            output.byte_offset += destination;
            return metal_queue::host_copy_if_idle(output, input, bytes);
        }
        void append(StorageRef source, size_t bytes, size_t destination, ExecContext) override
        {
            auto output = storage_ref(host_);
            output.byte_offset += destination;
            ops_.copy_device_to_device({ .src = source, .dst = output, .bytes = bytes });
        }
        bool prefers_recycled_host_staging() const noexcept override { return true; }
    };
}
std::unique_ptr<BackendReadbackSlot> MetalBackendOps::create_readback_slot(size_t bytes)
{
    return std::make_unique<MetalReadbackSlot>(*this, bytes);
}
namespace {
    class MetalCompletion final : public BackendCompletion {
        uint64_t value_ = 0;

    public:
        void capture(std::span<const Tensor* const> tensors) override
        {
            for (const auto* tensor : tensors) {
                const auto storage = storage_ref(*tensor);
                if (storage.meta)
                    value_ = std::max(value_, storage.meta->pending_value.load(std::memory_order_acquire));
            }
            if (value_)
                metal_flush();
        }
        void record(ExecContext, uint64_t value) override { value_ = value; }

        bool ready() override { return !value_ || metal_completed_serial() >= value_; }
        void wait() override
        {
            if (const auto value = std::exchange(value_, 0))
                metal_wait(value);
        }
    };
}
std::unique_ptr<BackendCompletion> MetalBackendOps::create_completion(BackendCompletionTarget target)
{
    if (target.external_device || target.external_point.semaphore)
        return {};
    return std::make_unique<MetalCompletion>();
}
StorageRef MetalBackendOps::borrow_native_range(void*, size_t) { throw std::runtime_error("Raw device tensor views are unsupported on this backend"); }
void MetalBackendOps::name_queue(ExecContext, const char*) { }
std::unique_ptr<PreparedReadback> MetalBackendOps::prepare_readback(const Tensor& source, const Tensor& destination)
{
    return std::make_unique<BufferedPreparedReadback>(source, destination,
        source.bytes() ? create_readback_buffer() : nullptr, [](const Tensor&, TensorWorkQueue&) {});
}
void MetalBackendOps::validate_checked_index_execution(ExecContext context) { (void)stream_is_capturing(context); }
NativeTensorAllocation MetalBackendOps::allocate_tensor(const TensorAllocationRequest& request) { return allocate_native_tensor(*this, request); }
uint64_t MetalBackendOps::expression_device_address(StorageRef storage) const { return descriptor_expression_address(storage); }
void MetalBackendOps::optimize_expression_pairing(ExpressionSignature&, const ExpressionPairingRequest&) { }
}

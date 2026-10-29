/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../prepared_readback.hpp"

#include "../../internal/expression_runtime.hpp"
#include "../tensor_vulkan_interop.hpp"

#include "../gpu_backend_ops.hpp"
#include "../upload_staging.hpp"
#include "core/cuda_error.hpp"
#include "core/device_fault.hpp"
#include "core/tensor_backend.hpp"
#include "runtime/cuda_event_pool.hpp"
#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <exception>
#include <stdexcept>
#include <vector>

namespace lfs::core::internal {
    namespace {
        void check(cudaError_t status) {
            if (status != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(status));
        }
        class CudaUpload final : public BackendUploadBuffer {
            CudaBackendOps& ops_;
            UploadStaging staging_;
            cudaEvent_t completion_ = nullptr;
            cudaStream_t stream_ = nullptr;

        public:
            explicit CudaUpload(CudaBackendOps& ops) : ops_(ops) {}
            ~CudaUpload() override {
                if (completion_)
                    (void)cudaEventDestroy(completion_);
            }
            BackendUploadSubmission enqueue(Tensor& destination, const Tensor& source, BackendQueueTarget target, bool) override {
                const auto stream = target.current ? getCurrentCUDAStream() : static_cast<cudaStream_t>(target.handle);
                stream_ = stream;
                destination.set_stream(stream);
                ops_.copy_host_to_device({.src = raw_storage_ref(const_cast<void*>(source.data_ptr()), source.dtype()),
                                          .dst = storage_ref(destination),
                                          .bytes = source.bytes(),
                                          .synchronous = false,
                                          .context = ExecContext{stream}});
                if (!completion_)
                    check(cudaEventCreateWithFlags(&completion_, cudaEventDisableTiming));
                check(cudaEventRecord(completion_, stream));
                return {};
            }
            BackendUploadSubmission enqueue_bytes(Tensor& destination, std::span<const std::byte> source, BackendQueueTarget target, bool in_batch) override {
                const auto snapshot = staging_.snapshot(source, destination.dtype());
                return enqueue(destination, snapshot, target, in_batch);
            }
            void wait() override { check(cudaEventSynchronize(completion_)); }
            bool poll() override {
                const auto status = cudaEventQuery(completion_);
                if (status == cudaErrorNotReady)
                    return false;
                if (status != cudaSuccess) {
                    (void)cudaGetLastError();
                    throw std::runtime_error("Tensor upload completion query failed");
                }
                return true;
            }

            void settle_enqueue_failure() override { check(cudaStreamSynchronize(stream_)); }
        };
        class CudaTimestamps final : public BackendTimestamps {
            std::vector<cudaEvent_t> events_;

        public:
            explicit CudaTimestamps(size_t count) : events_(count, nullptr) {
                for (auto& event : events_) {
                    if (cudaEventCreate(&event) != cudaSuccess) {
                        for (const auto created : events_)
                            if (created)
                                (void)cudaEventDestroy(created);
                        events_.clear();
                        (void)cudaGetLastError();
                        return;
                    }
                }
            }
            ~CudaTimestamps() override {
                for (const auto event : events_)
                    if (event)
                        (void)cudaEventDestroy(event);
            }
            bool available() const { return !events_.empty(); }
            bool mark(size_t index, void* target) override {
                if (index >= events_.size())
                    return false;
                if (cudaEventRecord(events_[index], static_cast<cudaStream_t>(target)) == cudaSuccess)
                    return true;
                (void)cudaGetLastError();
                return false;
            }
            bool wait_event(size_t index) override {
                if (index >= events_.size())
                    return false;
                if (cudaEventSynchronize(events_[index]) == cudaSuccess)
                    return true;
                (void)cudaGetLastError();
                return false;
            }
            bool wait_queue(void* target) override {
                if (cudaStreamSynchronize(static_cast<cudaStream_t>(target)) == cudaSuccess)
                    return true;
                (void)cudaGetLastError();
                return false;
            }
            std::optional<float> milliseconds(size_t begin, size_t end) const override {
                if (begin >= events_.size() || end >= events_.size())
                    return std::nullopt;
                float elapsed = 0.f;
                if (cudaEventElapsedTime(&elapsed, events_[begin], events_[end]) == cudaSuccess)
                    return elapsed;
                (void)cudaGetLastError();
                return std::nullopt;
            }
        };
    } // namespace
    std::unique_ptr<BackendTimestamps> CudaBackendOps::create_timestamps(size_t count) {
        auto result = std::make_unique<CudaTimestamps>(count);
        return result->available() ? std::move(result) : nullptr;
    }
    std::unique_ptr<BackendUploadBuffer> CudaBackendOps::create_upload_buffer() {
        return std::make_unique<CudaUpload>(*this);
    }
    namespace {
        class CudaFence final : public BackendFence {
            cudaEvent_t event_ = nullptr;

        public:
            explicit CudaFence(std::optional<void*> imported) {
                if (imported)
                    event_ = static_cast<cudaEvent_t>(*imported);
                else
                    check(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));
            }
            ~CudaFence() override {
                if (event_)
                    (void)cudaEventDestroy(event_);
            }
            void record(ExecContext target) override { check(cudaEventRecord(event_, target.cuda_stream)); }
            void wait_on(ExecContext target) const override { check(cudaStreamWaitEvent(target.cuda_stream, event_, 0)); }
            void wait() const override { check(cudaEventSynchronize(event_)); }
            bool ready() const override {
                const auto status = cudaEventQuery(event_);
                if (status == cudaErrorNotReady)
                    return false;
                check(status);
                return true;
            }
        };
    } // namespace
    std::unique_ptr<BackendFence> CudaBackendOps::create_fence(std::optional<void*> imported) {
        if (imported && !*imported)
            return {};
        return std::make_unique<CudaFence>(imported);
    }
    void CudaBackendOps::where_scalar(Tensor& output, const Tensor& mask, float value, const Tensor& source) {
        cuda_where_into(output, mask, value, source);
    }
    ExpressionCacheStats CudaBackendOps::expression_cache_stats() { return cuda_expression_cache().stats(); }
    namespace {
        class CudaReadbackSlot final : public BackendReadbackSlot {
            void* host_ = nullptr;

        public:
            explicit CudaReadbackSlot(size_t bytes) {
                check(cudaHostAlloc(&host_, bytes, cudaHostAllocPortable));
                std::memset(host_, 0, bytes);
            }
            ~CudaReadbackSlot() override {
                if (host_)
                    (void)cudaFreeHost(host_);
            }
            void* data() const override { return host_; }
            bool begin_append(const Tensor& source, size_t, size_t, size_t, ExecContext execution) override {
                prepare_inputs_for_stream({&source}, execution.cuda_stream);
                return false;
            }
            void append(StorageRef source, size_t bytes, size_t destination, ExecContext execution) override {
                const auto* input = static_cast<const std::byte*>(source.data) + source.byte_offset;
                check(cudaMemcpyAsync(static_cast<std::byte*>(host_) + destination, input, bytes,
                                      cudaMemcpyDeviceToHost, execution.cuda_stream));
            }
            bool prefers_recycled_host_staging() const noexcept override { return false; }

        };
    } // namespace
    std::unique_ptr<BackendReadbackSlot> CudaBackendOps::create_readback_slot(size_t bytes) {
        return std::make_unique<CudaReadbackSlot>(bytes);
    }
    namespace {
        class CudaCompletion final : public BackendCompletion {
            std::vector<std::pair<cudaStream_t, cudaEvent_t>> events_;

        public:
            void capture(std::span<const Tensor* const> tensors) override {
                std::vector<cudaStream_t> streams;
                for (const auto* tensor : tensors)
                    if (std::find(streams.begin(), streams.end(), tensor->stream()) == streams.end())
                        streams.push_back(tensor->stream());
                for (auto stream : streams)
                    record({stream}, 0);
            }
            void record(ExecContext execution, uint64_t) override {
                const auto stream = execution.cuda_stream;
                auto found = std::find_if(events_.begin(), events_.end(), [stream](const auto& event) { return event.first == stream; });
                if (found == events_.end()) {
                    events_.emplace_back(stream, nullptr);
                    found = std::prev(events_.end());
                    found->second = CudaEventPool::instance().acquire();
                    if (!found->second) {
                        const auto status = cudaStreamSynchronize(stream);
                        events_.pop_back();
                        ensure_cuda_success(status, "cudaStreamSynchronize(completion cleanup)", {}, LFS_SOURCE_SITE_CURRENT());
                        throw TensorError("Cannot acquire tensor completion event");
                    }
                }
                ensure_cuda_success(cudaEventRecord(found->second, stream), "cudaEventRecord(completion)", {}, LFS_SOURCE_SITE_CURRENT());
            }

            bool ready() override {
                for (const auto& [stream, event] : events_) {
                    const auto status = cudaEventQuery(event);
                    if (status == cudaErrorNotReady)
                        return false;
                    ensure_cuda_success(status, "cudaEventQuery(completion)", {}, LFS_SOURCE_SITE_CURRENT());
                }
                return true;
            }
            void wait() override {
                std::exception_ptr failure;
                for (const auto& [stream, event] : events_) {
                    try {
                        ensure_cuda_success(cudaEventSynchronize(event), "cudaEventSynchronize(completion)", {}, LFS_SOURCE_SITE_CURRENT());
                        device_fault_registry_consume_or_throw(LFS_SOURCE_SITE_CURRENT(), false);
                    } catch (...) {
                        // LFS-CENSUS-OK(empty-catch): Release every captured event before rethrowing the first failure.
                        if (!failure)
                            failure = std::current_exception();
                    }
                    CudaEventPool::instance().release(event);
                }
                events_.clear();
                if (failure)
                    std::rethrow_exception(failure);
            }
        };
    } // namespace
    std::unique_ptr<BackendCompletion> CudaBackendOps::create_completion(BackendCompletionTarget target) {
        if (target.external_device || target.external_point.semaphore)
            return {};
        return std::make_unique<CudaCompletion>();
    }
    StorageRef CudaBackendOps::borrow_native_range(void* pointer, size_t) { return {GpuBackend::CUDA, pointer, 0, DataType::Float32, nullptr, allocation_flags(StorageAccountingKind::CudaExternal, false, false, false)}; }
    namespace {
        class CudaPreparedReadback final : public PreparedReadback {
            Tensor source_, destination_;
            void* destination_pointer_;
            size_t bytes_;
            TensorFence producer_{GpuBackend::CUDA}, completion_{GpuBackend::CUDA};

        public:
            CudaPreparedReadback(const Tensor& source, const Tensor& destination)
                : source_(source), destination_(destination), destination_pointer_(destination_.data_ptr()), bytes_(source.bytes()) {
                if (bytes_) {
                    cudaPointerAttributes attributes{};
                    LFS_CUDA_CHECK(cudaPointerGetAttributes(&attributes, destination_pointer_));
                    if (attributes.type != cudaMemoryTypeHost)
                        throw std::invalid_argument("Direct readback destination must be pinned CPU storage");
                }
            }
            void enqueue(TensorWorkQueue& queue) override {
                producer_.record(source_.stream());
                queue.wait_for(producer_);
                if (bytes_)
                    LFS_CUDA_CHECK(cudaMemcpyAsync(destination_pointer_, std::as_const(source_).data_ptr(), bytes_, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(queue.native_handle())));
                queue.record(completion_);
            }
            bool poll() override {
                if (!completion_.ready())
                    return false;
                const auto status = cudaStreamQuery(source_.stream());
                if (status == cudaSuccess)
                    device_fault_slot_consume_or_throw(source_.stream(), "tensor.readback", LFS_SOURCE_SITE_CURRENT());
                else if (status != cudaErrorNotReady)
                    LFS_CUDA_CHECK(status);
                return true;
            }
            void wait() override { completion_.wait(); }
        };
    } // namespace
    std::unique_ptr<PreparedReadback> CudaBackendOps::prepare_readback(const Tensor& source, const Tensor& destination) {
        return std::make_unique<CudaPreparedReadback>(source, destination);
    }
} // namespace lfs::core::internal

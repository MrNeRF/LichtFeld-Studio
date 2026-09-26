/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_readback.hpp"
#include "backend/readback_buffer.hpp"
#include "core/device_fault.hpp"
#include "core/logger.hpp"
#include "core/tensor_upload.hpp"
#include "internal/tensor_impl.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
#if LFS_HAS_CUDA
#include "core/cuda_error_typed.hpp"
#include <cuda_runtime.h>
#endif

namespace lfs::core {
    struct TensorReadback::Impl {
        // Destroy staging before releasing the tensor that its queued copy reads.
        Tensor source;
        Tensor destination;
        void* destination_pointer = nullptr;
        std::unique_ptr<TensorFence> producer;
        std::unique_ptr<TensorFence> completion;
        bool direct = false;
        std::unique_ptr<internal::ReadbackBuffer> buffer;
        std::optional<GpuBackend> backend;
        std::vector<std::byte> host;
        size_t bytes = 0;
        bool pending = false;
    };

    TensorReadback::TensorReadback() = default;
    TensorReadback::~TensorReadback() {
        if (impl_ && impl_->direct && impl_->pending) {
            try {
                wait();
            } catch (const std::exception& error) {
                LOG_ERROR("Direct readback storage retained after incomplete transfer: {}", error.what());
                (void)impl_.release();
            }
        }
    }
    TensorReadback::TensorReadback(TensorReadback&&) noexcept = default;
    TensorReadback& TensorReadback::operator=(TensorReadback&& other) noexcept {
        if (this != &other) {
            TensorReadback previous;
            previous.impl_ = std::move(impl_);
            impl_ = std::move(other.impl_);
        }
        return *this;
    }

    void TensorReadback::enqueue(const Tensor& source) {
        if (!source.is_valid())
            throw std::invalid_argument("TensorReadback requires a valid tensor");
        enqueue_range(source, 0, source.bytes());
    }

    void TensorReadback::enqueue_range(const Tensor& source,
                                       const size_t byte_offset,
                                       const size_t byte_count) {
        if (!source.is_valid())
            throw std::invalid_argument("TensorReadback requires a valid tensor");
        enqueue_range_on(source, byte_offset, byte_count, source.stream());
    }

    void TensorReadback::enqueue(const Tensor& source, TensorWorkQueue& queue) {
        if (gpu_backend_of(source) != queue.backend())
            throw std::invalid_argument("TensorReadback queue backend mismatch");
        TensorWorkQueue::Scope scope(queue);
        enqueue_range_on(source, 0, source.bytes(), queue.native_handle());
    }

    void TensorReadback::enqueue_range_on(const Tensor& source, size_t byte_offset, size_t byte_count, void* target) {
        if (!source.is_valid()) {
            throw std::invalid_argument("TensorReadback requires a valid tensor");
        }
        if (pending()) {
            throw std::logic_error("TensorReadback already has a pending download");
        }
        if (impl_ && impl_->direct)
            throw std::logic_error("Prepared readback requires enqueue(queue)");
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        auto& state = *impl_;
        state.source = source.contiguous();
        pin_operands({&state.source});
        if (byte_offset > state.source.bytes() ||
            byte_count > state.source.bytes() - byte_offset)
            throw std::out_of_range("TensorReadback range exceeds tensor storage");
        state.bytes = byte_count;
        const auto backend = gpu_backend_of(state.source);
        if (backend != state.backend) {
            state.buffer.reset();
            state.backend = backend;
        }
        if (state.bytes != 0) {
            if (backend) {
                if (!state.buffer) {
                    state.buffer = internal::backend_ops_for(state.source).create_readback_buffer();
                }
                const auto stream = prepare_inputs_for_stream({&state.source}, static_cast<cudaStream_t>(target));
                try {
                    auto storage = internal::storage_ref(state.source);
                    storage.byte_offset += byte_offset;
                    state.buffer->enqueue(storage, state.bytes,
                                          internal::ExecContext{stream});
                } catch (...) {
                    // Drain a partially queued transfer before releasing its input.
                    state.buffer.reset();
                    state.source = {};
                    state.bytes = 0;
                    throw;
                }
            } else {
                state.host.resize(state.bytes);
                std::memcpy(state.host.data(),
                            static_cast<const std::byte*>(std::as_const(state.source).data_ptr()) + byte_offset,
                            state.bytes);
            }
        }
        state.pending = true;
    }

    void TensorReadback::prepare(const Tensor& source, const Tensor& destination) {
        if (pending())
            throw std::logic_error("Cannot prepare a pending readback");
        if (gpu_backend_of(source) != GpuBackend::CUDA)
            throw std::runtime_error("Direct tensor readback requires CUDA");
        if (!source.is_valid() || !source.is_contiguous() ||
            !destination.is_valid() || destination.device() != Device::CPU ||
            !destination.is_contiguous() || destination.bytes() != source.bytes())
            throw std::invalid_argument("Direct readback requires matching contiguous GPU and CPU tensors");
        auto next = std::make_unique<Impl>();
        next->source = source;
        next->destination = destination;
        next->destination_pointer = next->destination.data_ptr();
        next->bytes = source.bytes();
#if LFS_HAS_CUDA
        if (next->bytes) {
            cudaPointerAttributes attributes{};
            LFS_CUDA_CHECK(cudaPointerGetAttributes(&attributes, next->destination_pointer));
            if (attributes.type != cudaMemoryTypeHost)
                throw std::invalid_argument("Direct readback destination must be pinned CPU storage");
        }
#endif
        next->producer = std::make_unique<TensorFence>(GpuBackend::CUDA);
        next->completion = std::make_unique<TensorFence>(GpuBackend::CUDA);
        next->direct = true;
        impl_ = std::move(next);
    }

    void TensorReadback::enqueue(TensorWorkQueue& queue) {
        if (!impl_ || !impl_->direct)
            throw std::logic_error("Readback must be prepared before enqueue(queue)");
        auto& s = *impl_;
        if (s.pending)
            throw std::logic_error("Readback already has a pending download");
        if (queue.backend() != GpuBackend::CUDA)
            throw std::invalid_argument("Direct readback queue backend mismatch");
#if LFS_HAS_CUDA
        s.producer->record(s.source.stream());
        queue.wait_for(*s.producer);
        try {
            if (s.bytes)
                LFS_CUDA_CHECK(cudaMemcpyAsync(s.destination_pointer, std::as_const(s.source).data_ptr(),
                                               s.bytes, cudaMemcpyDeviceToHost,
                                               static_cast<cudaStream_t>(queue.native_handle())));
            queue.record(*s.completion);
            s.pending = true;
        } catch (...) {
            // Retain both tensors until any partially submitted copy has settled.
            try {
                queue.wait();
            } catch (...) {
                LOG_ERROR("Direct readback storage retained after failed queue drain");
                (void)impl_.release();
            }
            throw;
        }
#else
        throw std::runtime_error("CUDA tensor readback is unavailable in this build");
#endif
    }

    bool TensorReadback::poll() {
        if (!impl_ || !impl_->direct || !impl_->pending)
            throw std::logic_error("No pending direct readback");
        if (!impl_->completion->ready())
            return false;
#if LFS_HAS_CUDA
        // Only this retained producer can have supplied the downloaded bytes.
        // Scanning the global registry allocates a vector on each publication.
        const auto status = cudaStreamQuery(impl_->source.stream());
        if (status == cudaSuccess)
            device_fault_slot_consume_or_throw(impl_->source.stream(), "tensor.readback", LFS_SOURCE_SITE_CURRENT());
        else if (status != cudaErrorNotReady)
            LFS_CUDA_CHECK(status);
#endif
        impl_->pending = false;
        return true;
    }

    void TensorReadback::wait() {
        if (!impl_ || !impl_->direct || !impl_->pending)
            throw std::logic_error("No pending direct readback");
        impl_->completion->wait();
        (void)poll();
    }

    bool TensorReadback::pending() const noexcept {
        return impl_ && impl_->pending;
    }

    bool TensorReadback::poll(const std::span<std::byte> destination) {
        if (impl_ && impl_->direct)
            throw std::logic_error("Prepared readback requires poll() or wait()");
        if (!pending()) {
            throw std::logic_error("TensorReadback has no pending download");
        }
        auto& state = *impl_;
        if (destination.size() != state.bytes) {
            throw std::invalid_argument("TensorReadback destination size mismatch");
        }
        if (state.bytes != 0) {
            if (state.backend) {
                if (!state.buffer->poll(destination.data()))
                    return false;
            } else {
                std::memcpy(destination.data(), state.host.data(), state.bytes);
            }
        }
        state.pending = false;
        state.source = {};
        return true;
    }

    void TensorReadback::wait(const std::span<std::byte> destination) {
        if (impl_ && impl_->direct)
            throw std::logic_error("Prepared readback requires poll() or wait()");
        if (!pending()) {
            throw std::logic_error("TensorReadback has no pending download");
        }
        if (destination.size() != impl_->bytes) {
            throw std::invalid_argument("TensorReadback destination size mismatch");
        }
        if (impl_->bytes != 0 && impl_->backend)
            impl_->buffer->wait();
        if (!poll(destination))
            throw std::runtime_error("TensorReadback did not complete after waiting");
    }
} // namespace lfs::core

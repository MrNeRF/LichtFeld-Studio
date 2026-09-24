/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/tensor_readback.hpp"
#include "backend/readback_buffer.hpp"
#include "internal/tensor_impl.hpp"

#include <cstring>
#include <stdexcept>
#include <vector>

namespace lfs::core {
    struct TensorReadback::Impl {
        // Destroy staging before releasing the tensor that its queued copy reads.
        Tensor source;
        std::unique_ptr<internal::ReadbackBuffer> buffer;
        std::optional<GpuBackend> backend;
        std::vector<std::byte> host;
        size_t bytes = 0;
        bool pending = false;
    };

    TensorReadback::TensorReadback() = default;
    TensorReadback::~TensorReadback() = default;
    TensorReadback::TensorReadback(TensorReadback&&) noexcept = default;
    TensorReadback& TensorReadback::operator=(TensorReadback&&) noexcept = default;

    void TensorReadback::enqueue(const Tensor& source) {
        if (!source.is_valid())
            throw std::invalid_argument("TensorReadback requires a valid tensor");
        enqueue_range(source, 0, source.bytes());
    }

    void TensorReadback::enqueue_range(const Tensor& source,
                                       const size_t byte_offset,
                                       const size_t byte_count) {
        if (!source.is_valid()) {
            throw std::invalid_argument("TensorReadback requires a valid tensor");
        }
        if (pending()) {
            throw std::logic_error("TensorReadback already has a pending download");
        }
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
                const auto stream = prepare_inputs_for_stream({&state.source}, state.source.stream());
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
                            static_cast<const std::byte*>(state.source.data_ptr()) + byte_offset,
                            state.bytes);
            }
        }
        state.pending = true;
    }

    bool TensorReadback::pending() const noexcept {
        return impl_ && impl_->pending;
    }

    bool TensorReadback::poll(const std::span<std::byte> destination) {
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

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/tensor.hpp"
#include "core/tensor_upload.hpp"
#include "readback_buffer.hpp"
namespace lfs::core::internal {
    class PreparedReadback {
    public:
        virtual ~PreparedReadback() = default;
        virtual void enqueue(TensorWorkQueue&) = 0;
        virtual bool poll() = 0;
        virtual void wait() = 0;
    };
    // Used by native backends that download through reusable staging. The
    // producer hook supplies their own ordering, without SDK code in the facade.
    class BufferedPreparedReadback final : public PreparedReadback {
        Tensor source_, destination_;
        void* destination_pointer_;
        size_t bytes_;
        void (*prepare_)(const Tensor&, TensorWorkQueue&);
        TensorFence completion_;
        std::unique_ptr<ReadbackBuffer> buffer_;

    public:
        BufferedPreparedReadback(const Tensor& source, const Tensor& destination,
                                 std::unique_ptr<ReadbackBuffer> buffer,
                                 void (*prepare)(const Tensor&, TensorWorkQueue&))
            : source_(source), destination_(destination), destination_pointer_(destination_.data_ptr()), bytes_(source.bytes()), prepare_(prepare), completion_(*gpu_backend_of(source)), buffer_(std::move(buffer)) {}
        void enqueue(TensorWorkQueue& queue) override {
            prepare_(source_, queue);
            const TensorWorkQueue::Scope scope(queue);
            if (bytes_)
                buffer_->enqueue(storage_ref(source_), bytes_, {});
            queue.record(completion_);
        }
        bool poll() override {
            if (!completion_.ready())
                return false;
            return !bytes_ || buffer_->poll(destination_pointer_);
        }
        void wait() override { completion_.wait(); }
    };
} // namespace lfs::core::internal

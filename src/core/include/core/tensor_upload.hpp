/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "core/export.hpp"
#include "core/gpu_backend_fwd.hpp"
#include "core/tensor_completion.hpp"
#include <cstdint>
#include <functional>
#include <memory>
namespace lfs::core {
    class Tensor;
    // Reusable asynchronous H2D slot. Owns both tensors until the queued copy
    // completes. Destruction waits; poll never blocks. Does not alter copy_from.
    class LFS_CORE_API TensorUpload {
    public:
        TensorUpload();
        ~TensorUpload();
        TensorUpload(TensorUpload&&) noexcept;
        TensorUpload& operator=(TensorUpload&&) noexcept;
        void enqueue(Tensor destination, const Tensor& source);
        [[nodiscard]] bool pending() const noexcept;
        [[nodiscard]] bool poll();
        void wait();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
    // Device and consumer timeline outlive the queue.
    // CUDA timeline imports are private to tensorlib. execute batches every
    // command in the callback and returns one completion ticket.
    class LFS_CORE_API TensorWorkQueue {
    public:
        TensorWorkQueue(GpuBackend backend, void* vulkan_device, void* consumer_timeline);
        ~TensorWorkQueue();
        TensorWorkQueue(const TensorWorkQueue&) = delete;
        TensorWorkQueue& operator=(const TensorWorkQueue&) = delete;
        TensorCompletion execute(const std::function<void()>& commands, uint64_t consumer_value = 0);
        [[nodiscard]] void* timeline() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
} // namespace lfs::core

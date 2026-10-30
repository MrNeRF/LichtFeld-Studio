/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#pragma once
#include "../backend_work_queue.hpp"
#include <cuda_runtime.h>
namespace lfs::core::internal {
    class CudaQueueTimeline {
    public:
        virtual ~CudaQueueTimeline() = default;
        virtual void replace(void*, VulkanTimelinePoint) = 0;
        virtual void wait(uint64_t) = 0;
        virtual void signal(uint64_t) = 0;
        virtual VulkanTimelinePoint point(uint64_t) const = 0;
        virtual void* timeline() const = 0;
        virtual bool retained() const = 0;
    };
    void validate_cuda_queue_create(const BackendQueueCreateInfo&);
    std::unique_ptr<CudaQueueTimeline> make_cuda_queue_timeline(const BackendQueueCreateInfo&, cudaStream_t);
} // namespace lfs::core::internal

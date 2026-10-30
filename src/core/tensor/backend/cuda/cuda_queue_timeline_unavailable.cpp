/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "cuda_queue_timeline.hpp"
namespace lfs::core::internal {
    namespace {
        class NoCudaQueueTimeline final : public CudaQueueTimeline {
        public:
            void replace(void*, VulkanTimelinePoint) override { throw std::runtime_error("CUDA/Vulkan timeline interop is unavailable in this build"); }
            void wait(uint64_t) override {}
            void signal(uint64_t) override {}
            VulkanTimelinePoint point(uint64_t value) const override { return {nullptr, value, {}}; }
            void* timeline() const override { return nullptr; }
            bool retained() const override { return false; }
        };
    } // namespace
    void validate_cuda_queue_create([[maybe_unused]] const BackendQueueCreateInfo& info) {
        if (info.kind == BackendQueueCreateInfo::Kind::Consumer)
            throw std::runtime_error("CUDA/Vulkan interoperable work queues are unavailable in this build");
    }
    std::unique_ptr<CudaQueueTimeline> make_cuda_queue_timeline(const BackendQueueCreateInfo& info, cudaStream_t) {
        if (info.kind == BackendQueueCreateInfo::Kind::Consumer)
            throw std::runtime_error("CUDA/Vulkan interoperable work queues are unavailable in this build");
        return std::make_unique<NoCudaQueueTimeline>();
    }
} // namespace lfs::core::internal

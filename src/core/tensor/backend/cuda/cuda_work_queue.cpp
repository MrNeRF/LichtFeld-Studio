/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../backend_work_queue.hpp"
#include "../tensor_completion.hpp"
#include "core/tensor_backend.hpp"
#include "cuda_queue_timeline.hpp"
#include "runtime/memory_pool.hpp"
namespace lfs::core::internal {
    namespace {
        void check_queue(cudaError_t status) {
            if (status != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(status));
        }
        struct CudaQueueStream {
            cudaStream_t stream = nullptr;
            bool owns = true;
            ~CudaQueueStream() {
                if (stream && owns) {
                    (void)cudaStreamSynchronize(stream);
                    CudaMemoryPool::instance().release_stream(stream);
                    (void)cudaStreamDestroy(stream);
                }
            }
        };
        class CudaWorkQueue final : public BackendWorkQueue {
            CudaQueueStream stream_;
            std::unique_ptr<CudaQueueTimeline> timeline_;
            uint64_t counter_ = 0;

        public:
            explicit CudaWorkQueue(const BackendQueueCreateInfo& info) {
                if (info.kind == BackendQueueCreateInfo::Kind::Borrowed) {
                    stream_.stream = static_cast<cudaStream_t>(info.target);
                    stream_.owns = false;
                } else {
                    activate_selected_cuda_device();
                    const auto flags = info.kind == BackendQueueCreateInfo::Kind::Owned && info.legacy_ordered ? cudaStreamDefault : cudaStreamNonBlocking;
                    check_queue(cudaStreamCreateWithFlags(&stream_.stream, flags));
                }
                timeline_ = make_cuda_queue_timeline(info, stream_.stream);
            }
            ~CudaWorkQueue() override {
                // Imports must outlive commands on the owned stream, including
                // commands enqueued through Scope without execute().
                if (stream_.stream && stream_.owns)
                    (void)cudaStreamSynchronize(stream_.stream);
            }
            void* native_handle() const override { return stream_.stream; }
            BackendTargetBinding bind(bool) override {
                const auto previous = reinterpret_cast<uintptr_t>(getCurrentCUDAStream());
                setCurrentCUDAStream(stream_.stream);
                return {previous, 0};
            }
            void unbind(BackendTargetBinding token) noexcept override { setCurrentCUDAStream(reinterpret_cast<cudaStream_t>(token[0])); }
            bool ready() override {
                const auto status = cudaStreamQuery(stream_.stream);
                if (status == cudaErrorNotReady)
                    return false;
                check_queue(status);
                return true;
            }
            void wait() override { check_queue(cudaStreamSynchronize(stream_.stream)); }
            void enqueue_host_callback(void (*callback)(void*), void* user) override { check_queue(cudaLaunchHostFunc(stream_.stream, callback, user)); }
            void set_consumer_timeline(void* device, VulkanTimelinePoint point) override { timeline_->replace(device, std::move(point)); }
            void wait_timeline(uint64_t value) override { timeline_->wait(value); }
            void* timeline() const override { return timeline_->timeline(); }
            void begin(uint64_t value) { timeline_->wait(value); }
            TensorCompletion finish() {
                ++counter_;
                timeline_->signal(counter_);
                return TensorCompletionAccess::cuda(stream_.stream, timeline_->point(counter_));
            }
            void settle_failed() { (void)cudaStreamSynchronize(stream_.stream); }
            TensorCompletion execute(const std::function<void()>& commands, uint64_t value) override { return execute_commands(*this, GpuBackend::CUDA, commands, value); }
            void validate_readback_ring() const override {}
            bool close() override {
                if (timeline_->retained() || stream_.owns)
                    wait();
                last_.wait();
                return true;
            }
        };
    } // namespace
    std::unique_ptr<BackendWorkQueue> make_cuda_work_queue(const BackendQueueCreateInfo& info) {
        validate_cuda_queue_create(info);
        return std::make_unique<CudaWorkQueue>(info);
    }
} // namespace lfs::core::internal

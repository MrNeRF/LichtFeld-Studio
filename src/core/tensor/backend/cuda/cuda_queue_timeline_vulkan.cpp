/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../vulkan/vk_cuda_bridge.hpp"
#include "core/tensor_backend.hpp"
#include "cuda_queue_timeline.hpp"
#include <utility>
#if !defined(_WIN32)
#include <unistd.h>
#endif
namespace lfs::core::internal {
    namespace {
        void check(cudaError_t status) {
            if (status != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(status));
        }
        struct ImportedQueueTimelines {
            VkDevice device = VK_NULL_HANDLE;
            VkSemaphore ready = VK_NULL_HANDLE;
            cudaExternalSemaphore_t ready_cuda = nullptr, consumer_cuda = nullptr;
            std::vector<std::pair<cudaExternalSemaphore_t, VulkanTimelinePoint>> retired;
            ~ImportedQueueTimelines() {
                if (ready_cuda)
                    (void)cudaDestroyExternalSemaphore(ready_cuda);
                if (consumer_cuda)
                    (void)cudaDestroyExternalSemaphore(consumer_cuda);
                for (const auto& entry : retired)
                    (void)cudaDestroyExternalSemaphore(entry.first);
                if (ready)
                    vkDestroySemaphore(device, ready, nullptr);
            }
            cudaExternalSemaphore_t import_semaphore(VkSemaphore semaphore) {
                cudaExternalSemaphoreHandleDesc desc{};
#ifdef _WIN32
                const auto get = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR"));
                VkSemaphoreGetWin32HandleInfoKHR info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
                info.semaphore = semaphore;
                info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
                HANDLE handle = nullptr;
                if (!get || get(device, &info, &handle) != VK_SUCCESS)
                    throw std::runtime_error("Cannot export tensor queue semaphore");
                desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
                desc.handle.win32.handle = handle;
                cudaExternalSemaphore_t result = nullptr;
                const auto status = cudaImportExternalSemaphore(&result, &desc);
                CloseHandle(handle);
                check(status);
#else
                const auto get = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR"));
                VkSemaphoreGetFdInfoKHR info{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
                info.semaphore = semaphore;
                info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
                int fd = -1;
                if (!get || get(device, &info, &fd) != VK_SUCCESS)
                    throw std::runtime_error("Cannot export tensor queue semaphore");
                desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
                desc.handle.fd = fd;
                cudaExternalSemaphore_t result = nullptr;
                const auto status = cudaImportExternalSemaphore(&result, &desc);
                if (status != cudaSuccess) {
                    close(fd);
                    check(status);
                }
#endif
                return result;
            }
        };
        class CudaVulkanQueueTimeline final : public CudaQueueTimeline {
            ImportedQueueTimelines native_;
            QueueConsumerTimeline consumer_;
            cudaStream_t stream_;
            std::optional<bool> direct_;
            bool host_ = false;

        public:
            CudaVulkanQueueTimeline(const BackendQueueCreateInfo& info, cudaStream_t stream)
                : consumer_(info.consumer_device, info.consumer), stream_(stream), direct_(info.external_interop) {
                native_.device = static_cast<VkDevice>(info.consumer_device);
                if (info.kind != BackendQueueCreateInfo::Kind::Consumer)
                    return;
                host_ = !direct_.value_or(cuda_vulkan_direct_interop());
                if (host_ || !native_.device)
                    return;
                VkExportSemaphoreCreateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
                export_info.handleTypes = kVulkanExportSemaphoreHandleType;
                VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
                type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
                type.pNext = &export_info;
                VkSemaphoreCreateInfo create{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
                create.pNext = &type;
                if (vkCreateSemaphore(native_.device, &create, nullptr, &native_.ready) != VK_SUCCESS)
                    throw std::runtime_error("Cannot create tensor queue semaphore");
                native_.ready_cuda = native_.import_semaphore(native_.ready);
                if (info.consumer)
                    native_.consumer_cuda = native_.import_semaphore(static_cast<VkSemaphore>(info.consumer));
            }
            void replace(void* device, VulkanTimelinePoint point) override {
                if (consumer_.matches(device, point))
                    return;
                if (!direct_.value_or(cuda_vulkan_direct_interop())) {
                    host_ = true;
                    consumer_.replace(device, std::move(point));
                    return;
                }
                if (native_.ready && native_.device != device)
                    throw std::invalid_argument("Cannot change the device of an exported tensor queue");
                // Reserve both retirement lists before acquiring a native import.
                consumer_.reserve_replacement();
                if (native_.consumer_cuda)
                    native_.retired.reserve(native_.retired.size() + 1);
                const auto previous = native_.device;
                native_.device = static_cast<VkDevice>(device);
                cudaExternalSemaphore_t replacement = nullptr;
                try {
                    if (point.semaphore)
                        replacement = native_.import_semaphore(static_cast<VkSemaphore>(point.semaphore));
                } catch (...) {
                    native_.device = previous;
                    throw;
                }
                if (native_.consumer_cuda)
                    native_.retired.emplace_back(native_.consumer_cuda, consumer_.point());
                native_.consumer_cuda = replacement;
                consumer_.replace(device, std::move(point));
            }
            void wait(uint64_t value) override {
                if (host_) {
                    consumer_.wait(value);
                    return;
                }
                if (value && native_.consumer_cuda) {
                    cudaExternalSemaphoreWaitParams params{};
                    params.params.fence.value = value;
                    check(cudaWaitExternalSemaphoresAsync(&native_.consumer_cuda, &params, 1, stream_));
                }
            }
            void signal(uint64_t value) override {
                if (native_.ready_cuda) {
                    cudaExternalSemaphoreSignalParams params{};
                    params.params.fence.value = value;
                    check(cudaSignalExternalSemaphoresAsync(&native_.ready_cuda, &params, 1, stream_));
                }
            }
            VulkanTimelinePoint point(uint64_t value) const override { return {native_.ready, value, {}}; }
            void* timeline() const override { return host_ ? nullptr : native_.ready; }
            bool retained() const override { return native_.ready_cuda || native_.consumer_cuda || !native_.retired.empty() || consumer_.retained(); }
        };
    } // namespace
    void validate_cuda_queue_create([[maybe_unused]] const BackendQueueCreateInfo& info) {}
    std::unique_ptr<CudaQueueTimeline> make_cuda_queue_timeline(const BackendQueueCreateInfo& info, cudaStream_t stream) { return std::make_unique<CudaVulkanQueueTimeline>(info, stream); }
} // namespace lfs::core::internal

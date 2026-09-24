/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "core/tensor_upload.hpp"
#include "../internal/tensor_impl.hpp"
#include "core/cuda_error_typed.hpp"
#include "core/logger.hpp"
#include "core/tensor_backend.hpp"
#include "cuda/runtime/memory_pool.hpp"
#include "tensor_completion.hpp"
#include "vulkan/vk_context.hpp"
#include "vulkan/vk_cuda_bridge.hpp"
#include "vulkan/vk_recorder.hpp"
#include <algorithm>
#include <stdexcept>
#ifndef _WIN32
#include <unistd.h>
#endif
namespace lfs::core {
    namespace {
        void check(cudaError_t e) {
            if (e != cudaSuccess)
                throw std::runtime_error(cudaGetErrorString(e));
        }
    } // namespace
    struct TensorUpload::Impl {
        Tensor source, destination;
        TensorCompletion completion;
        std::shared_ptr<internal::VulkanContext> vk;
        bool pending = false;
    };
    TensorUpload::TensorUpload() = default;
    TensorUpload::~TensorUpload() {
        try {
            wait();
        } catch (const std::exception& e) {
            LOG_ERROR("Tensor upload storage retained after incomplete transfer: {}", e.what());
            (void)impl_.release();
        }
    }
    TensorUpload::TensorUpload(TensorUpload&&) noexcept = default;
    TensorUpload& TensorUpload::operator=(TensorUpload&& other) noexcept {
        if (this != &other) {
            TensorUpload previous;
            previous.impl_ = std::move(impl_);
            impl_ = std::move(other.impl_);
        }
        return *this;
    }
    bool TensorUpload::pending() const noexcept { return impl_ && impl_->pending; }
    void TensorUpload::enqueue(Tensor destination, const Tensor& source) {
        if (pending())
            throw std::logic_error("TensorUpload already pending");
        if (!gpu_backend_of(destination) || source.device() != Device::CPU || !source.is_contiguous() ||
            !destination.is_contiguous() || destination.dtype() != source.dtype() || destination.bytes() != source.bytes())
            throw std::invalid_argument("TensorUpload requires matching contiguous CPU and GPU tensors");
        if (!impl_)
            impl_ = std::make_unique<Impl>();
        auto& s = *impl_;
        s.source = source;
        s.destination = std::move(destination);
        s.completion = {};
        s.vk.reset();
        pin_operands({&s.source, &s.destination});
        const auto stream = getCurrentCUDAStream();
        s.destination.set_stream(stream);
        s.pending = true;
        try {
            internal::backend_ops_for(s.destination).copy_host_to_device({.src = internal::raw_storage_ref(const_cast<void*>(s.source.data_ptr()), s.source.dtype()), .dst = internal::storage_ref(s.destination), .bytes = s.source.bytes(), .synchronous = false, .context = internal::ExecContext{stream}});
            if (gpu_backend_of(s.destination) == GpuBackend::CUDA)
                s.completion = TensorCompletionAccess::cuda(stream);
            else {
                s.vk = internal::acquire_vulkan_context();
                s.completion = TensorCompletionAccess::vulkan(
                    s.vk->recorders().pending_value(internal::storage_ref(s.destination)));
            }
        } catch (...) {
            internal::backend_ops_for(s.destination).synchronize_stream(internal::ExecContext{stream});
            s.pending = false;
            throw;
        }
    }
    bool TensorUpload::poll() {
        if (!pending())
            return true;
        auto& s = *impl_;
        if (s.vk)
            s.vk->recorders().flush_storage(internal::storage_ref(s.destination));
        if (!s.completion.ready())
            return false;
        s.pending = false;
        s.source = {};
        s.destination = {};
        return true;
    }
    void TensorUpload::wait() {
        if (!pending())
            return;
        auto& s = *impl_;
        if (s.vk)
            s.vk->recorders().flush_storage(internal::storage_ref(s.destination));
        s.completion.wait();
        (void)poll();
    }
    struct TensorWorkQueue::Impl {
        GpuBackend backend;
        cudaStream_t stream = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VkSemaphore ready = VK_NULL_HANDLE, consumer = VK_NULL_HANDLE;
        cudaExternalSemaphore_t ready_cuda = nullptr, consumer_cuda = nullptr;
        uint64_t counter = 0;
        TensorCompletion last;
        ~Impl() {
            if (stream) {
                (void)cudaStreamSynchronize(stream);
                CudaMemoryPool::instance().release_stream(stream);
                (void)cudaStreamDestroy(stream);
            }
            if (ready_cuda)
                (void)cudaDestroyExternalSemaphore(ready_cuda);
            if (consumer_cuda)
                (void)cudaDestroyExternalSemaphore(consumer_cuda);
            if (ready && backend == GpuBackend::CUDA)
                vkDestroySemaphore(device, ready, nullptr);
        }
        cudaExternalSemaphore_t import(VkSemaphore semaphore) {
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
    TensorWorkQueue::TensorWorkQueue(GpuBackend backend, void* device, void* consumer) : impl_(std::make_unique<Impl>()) {
        auto& s = *impl_;
        s.backend = backend;
        s.device = static_cast<VkDevice>(device);
        s.consumer = static_cast<VkSemaphore>(consumer);
        if (backend == GpuBackend::Vulkan) {
            s.ready = internal::acquire_vulkan_context()->timeline();
            return;
        }
        check(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        if (!s.device)
            return;
        VkExportSemaphoreCreateInfo export_info{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
        export_info.handleTypes = internal::kVulkanExportSemaphoreHandleType;
        VkSemaphoreTypeCreateInfo type{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        type.pNext = &export_info;
        VkSemaphoreCreateInfo info{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        info.pNext = &type;
        if (vkCreateSemaphore(s.device, &info, nullptr, &s.ready) != VK_SUCCESS)
            throw std::runtime_error("Cannot create tensor queue semaphore");
        s.ready_cuda = s.import(s.ready);
        if (s.consumer)
            s.consumer_cuda = s.import(s.consumer);
    }
    TensorWorkQueue::~TensorWorkQueue() {
        try {
            impl_->last.wait();
        } catch (const std::exception& e) {
            LOG_ERROR("Tensor queue retained after incomplete work: {}", e.what());
            (void)impl_.release();
        }
    }
    void* TensorWorkQueue::timeline() const { return impl_->ready; }
    TensorCompletion TensorWorkQueue::execute(const std::function<void()>& commands, uint64_t value) {
        auto& s = *impl_;
        const GpuBackendScope backend(s.backend);
        const CUDAStreamGuard stream(s.stream);
        uint64_t consumer_completion = 0;
        if (value && s.consumer) {
            if (s.backend == GpuBackend::CUDA) {
                cudaExternalSemaphoreWaitParams params{};
                params.params.fence.value = value;
                check(cudaWaitExternalSemaphoresAsync(&s.consumer_cuda, &params, 1, s.stream));
            } else {
                const auto ctx = internal::acquire_vulkan_context();
                consumer_completion = ctx->recorders().wait_external({}, s.consumer, value, {});
            }
        }
        TensorCompletion result;
        try {
            commands();
            if (s.backend == GpuBackend::CUDA) {
                ++s.counter;
                if (s.ready_cuda) {
                    cudaExternalSemaphoreSignalParams params{};
                    params.params.fence.value = s.counter;
                    check(cudaSignalExternalSemaphoresAsync(&s.ready_cuda, &params, 1, s.stream));
                }
                result = TensorCompletionAccess::cuda(s.stream, {s.ready, s.counter, {}});
            } else {
                result = TensorCompletionAccess::vulkan(
                    std::max(consumer_completion, internal::acquire_vulkan_context()->recorders().flush_current()));
            }
        } catch (...) {
            if (s.backend == GpuBackend::CUDA)
                (void)cudaStreamSynchronize(s.stream);
            else
                internal::acquire_vulkan_context()->recorders().wait_all();
            throw;
        }
        s.last = result;
        return result;
    }
} // namespace lfs::core

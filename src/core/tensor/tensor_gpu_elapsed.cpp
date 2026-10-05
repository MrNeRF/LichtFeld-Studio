/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/gpu_elapsed.hpp"

#include "backend/metal/metal_queue.hpp"
#ifdef LFS_TENSOR_VULKAN
#include "backend/vulkan/vk_context.hpp"
#include "backend/vulkan/vk_recorder.hpp"
#include <vulkan/vulkan.h>
#endif
#include "core/cuda_types.hpp"

#include <stdexcept>
#include <vector>

#if LFS_HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace lfs::core {

    struct GpuElapsed::Impl {
        struct Mark {
            uint64_t queue = 0;
            uint64_t timeline = 0;
            bool written = false;
        };
        GpuBackend backend;
        bool ready = false;
#if LFS_HAS_CUDA
        std::vector<cudaEvent_t> events;
#endif
        std::unique_ptr<internal::metal_queue::Timestamps> metal;
#ifdef LFS_TENSOR_VULKAN
        std::shared_ptr<internal::VulkanContext> vulkan;
        VkQueryPool queries = VK_NULL_HANDLE;
#endif
        std::vector<Mark> marks;
        float timestamp_period = 0.f;
        ~Impl() {
#ifdef LFS_TENSOR_VULKAN
            if (queries != VK_NULL_HANDLE && vulkan)
                vkDestroyQueryPool(vulkan->device(), queries, nullptr);
#endif
        }
    };

    GpuElapsed::GpuElapsed(const GpuBackend backend, const std::size_t event_count)
        : impl_(std::make_unique<Impl>()) {
        impl_->backend = backend;
        if (event_count == 0)
            return;
        if (backend == GpuBackend::Metal) {
            impl_->metal = internal::metal_queue::timestamps(event_count);
            impl_->marks.resize(event_count);
            impl_->ready = impl_->metal != nullptr;
            return;
        }
#ifdef LFS_TENSOR_VULKAN
        if (backend == GpuBackend::Vulkan) {
            auto context = internal::acquire_vulkan_context();
            uint32_t families = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(context->physical_device(), &families, nullptr);
            std::vector<VkQueueFamilyProperties> properties(families);
            if (families != 0)
                vkGetPhysicalDeviceQueueFamilyProperties(context->physical_device(), &families, properties.data());
            const auto family = context->queue_family();
            if (family >= families || properties[family].timestampValidBits == 0 ||
                context->caps().timestamp_period <= 0.f)
                return;
            VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            info.queryType = VK_QUERY_TYPE_TIMESTAMP;
            info.queryCount = static_cast<uint32_t>(event_count);
            if (vkCreateQueryPool(context->device(), &info, nullptr, &impl_->queries) != VK_SUCCESS)
                return;
            impl_->vulkan = std::move(context);
            impl_->marks.resize(event_count);
            impl_->timestamp_period = impl_->vulkan->caps().timestamp_period;
            impl_->ready = true;
            return;
        }
#endif
#if LFS_HAS_CUDA
        if (backend != GpuBackend::CUDA)
            return;
        impl_->events.resize(event_count, nullptr);
        for (auto& event : impl_->events) {
            if (cudaEventCreate(&event) != cudaSuccess) {
                for (auto& created : impl_->events) {
                    if (created != nullptr)
                        (void)cudaEventDestroy(created);
                    created = nullptr;
                }
                impl_->events.clear();
                (void)cudaGetLastError();
                return;
            }
        }
        impl_->ready = true;
#else
        (void)event_count;
#endif
    }

    GpuElapsed::~GpuElapsed() {
#if LFS_HAS_CUDA
        if (impl_) {
            for (const auto event : impl_->events) {
                if (event != nullptr)
                    (void)cudaEventDestroy(event);
            }
        }
#endif
    }

    bool GpuElapsed::ready() const noexcept {
        return impl_ && impl_->ready;
    }

    bool GpuElapsed::mark(std::size_t index, TensorExecutionTarget target) {
        if (target.backend() != impl_->backend)
            throw std::invalid_argument("GpuElapsed queue backend mismatch");
        return mark(index, target.native_handle());
    }

    bool GpuElapsed::wait_queue(TensorExecutionTarget target) {
        if (target.backend() != impl_->backend)
            throw std::invalid_argument("GpuElapsed queue backend mismatch");
        return wait_queue(target.native_handle());
    }

    bool GpuElapsed::mark(const std::size_t index, void* const execution_target) {
        if (impl_->backend == GpuBackend::Metal) {
            const auto id = reinterpret_cast<uint64_t>(execution_target);
            if (!internal::metal_queue::valid(id))
                throw std::invalid_argument("GpuElapsed target is not a Metal tensor queue");
            if (!ready() || index >= impl_->marks.size())
                return false;
            impl_->marks[index] = {id, impl_->metal->write(index), true};
            return true;
        }
#ifdef LFS_TENSOR_VULKAN
        if (impl_->backend == GpuBackend::Vulkan) {
            if (!ready() || index >= impl_->marks.size())
                return false;
            const auto id = execution_target ? reinterpret_cast<uint64_t>(execution_target)
                                             : impl_->vulkan->recorders().current_queue();
            const uint64_t timeline = impl_->vulkan->recorders().write_timestamp(
                id, impl_->queries, static_cast<uint32_t>(index));
            impl_->marks[index] = {id, timeline, true};
            return true;
        }
#endif
#if LFS_HAS_CUDA
        if (!ready() || index >= impl_->events.size() ||
            (impl_->backend != GpuBackend::CUDA && execution_target != nullptr))
            return false;
        const auto stream = reinterpret_cast<cudaStream_t>(execution_target);
        if (cudaEventRecord(impl_->events[index], stream) == cudaSuccess)
            return true;
        (void)cudaGetLastError();
#else
        (void)index;
        (void)execution_target;
#endif
        return false;
    }

    bool GpuElapsed::wait_queue(void* const execution_target) {
        if (impl_->backend == GpuBackend::Metal) {
            if (!internal::metal_queue::valid(reinterpret_cast<uint64_t>(execution_target)))
                throw std::invalid_argument("GpuElapsed target is not a Metal tensor queue");
            if (!ready())
                return false;
            internal::metal_queue::wait(internal::metal_queue::submit());
            return true;
        }
#ifdef LFS_TENSOR_VULKAN
        if (impl_->backend == GpuBackend::Vulkan) {
            if (!ready())
                return false;
            const auto id = execution_target ? reinterpret_cast<uint64_t>(execution_target)
                                             : impl_->vulkan->recorders().current_queue();
            impl_->vulkan->recorders().queue_wait(id);
            // A finished submission does not make its timestamps readable yet:
            // MoltenVK resolves them after the timeline signal. Wait for them
            // so milliseconds() holds right after this returns.
            for (std::size_t index = 0; index < impl_->marks.size(); ++index) {
                if (!impl_->marks[index].written || impl_->marks[index].queue != id)
                    continue;
                uint64_t ticks = 0;
                if (vkGetQueryPoolResults(impl_->vulkan->device(), impl_->queries, static_cast<uint32_t>(index), 1,
                                          sizeof(ticks), &ticks, sizeof(ticks),
                                          VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS)
                    return false;
            }
            return true;
        }
#endif
#if LFS_HAS_CUDA
        if (!ready() ||
            (impl_->backend != GpuBackend::CUDA && execution_target != nullptr))
            return false;
        if (cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(execution_target)) == cudaSuccess)
            return true;
        (void)cudaGetLastError();
#else
        (void)execution_target;
#endif
        return false;
    }

    bool GpuElapsed::wait_event(const std::size_t index) {
        if (impl_->backend == GpuBackend::Metal) {
            if (!ready() || index >= impl_->marks.size() || !impl_->marks[index].written)
                return false;
            internal::metal_queue::wait(impl_->marks[index].timeline);
            return true;
        }
#ifdef LFS_TENSOR_VULKAN
        if (impl_->backend == GpuBackend::Vulkan) {
            if (!ready() || index >= impl_->marks.size() || !impl_->marks[index].written)
                return false;
            const auto& mark = impl_->marks[index];
            impl_->vulkan->recorders().flush_queue(mark.queue);
            impl_->vulkan->wait(mark.timeline);
            uint64_t ticks = 0;
            const VkResult result = vkGetQueryPoolResults(
                impl_->vulkan->device(), impl_->queries, static_cast<uint32_t>(index), 1,
                sizeof(ticks), &ticks, sizeof(ticks),
                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
            return result == VK_SUCCESS;
        }
#endif
#if LFS_HAS_CUDA
        if (!ready() || index >= impl_->events.size())
            return false;
        if (cudaEventSynchronize(impl_->events[index]) == cudaSuccess)
            return true;
        (void)cudaGetLastError();
#else
        (void)index;
#endif
        return false;
    }

    std::optional<float> GpuElapsed::milliseconds(const std::size_t begin,
                                                  const std::size_t end) const {
        if (impl_->backend == GpuBackend::Metal)
            return ready() ? impl_->metal->milliseconds(begin, end) : std::nullopt;
#ifdef LFS_TENSOR_VULKAN
        if (impl_->backend == GpuBackend::Vulkan) {
            if (!ready() || begin >= impl_->marks.size() || end >= impl_->marks.size() ||
                !impl_->marks[begin].written || !impl_->marks[end].written)
                return std::nullopt;
            const auto read = [&](const std::size_t index, uint64_t& ticks) {
                const VkResult result = vkGetQueryPoolResults(
                    impl_->vulkan->device(), impl_->queries, static_cast<uint32_t>(index), 1,
                    sizeof(ticks), &ticks, sizeof(ticks), VK_QUERY_RESULT_64_BIT);
                return result == VK_SUCCESS;
            };
            uint64_t first = 0;
            uint64_t second = 0;
            if (!read(begin, first) || !read(end, second) || second < first)
                return std::nullopt;
            const double nanoseconds = static_cast<double>(second - first) * impl_->timestamp_period;
            return static_cast<float>(nanoseconds / 1.0e6);
        }
#endif
#if LFS_HAS_CUDA
        if (!ready() || begin >= impl_->events.size() || end >= impl_->events.size())
            return std::nullopt;
        float elapsed = 0.0f;
        if (cudaEventElapsedTime(&elapsed, impl_->events[begin], impl_->events[end]) == cudaSuccess)
            return elapsed;
        (void)cudaGetLastError();
#else
        (void)begin;
        (void)end;
#endif
        return std::nullopt;
    }

} // namespace lfs::core

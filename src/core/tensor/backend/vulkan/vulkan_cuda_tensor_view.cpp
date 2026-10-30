/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "../../internal/tensor_impl.hpp"
#include "core/cuda_error.hpp"
#include "core/gpu_device_info.hpp"
#include "core/tensor_backend_vulkan.hpp"
#include "vk_context.hpp"
#include "vk_cuda_bridge.hpp"
#include "vk_memory.hpp"
#include "vk_recorder.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <cuda_runtime.h>
#include <format>
namespace lfs::core {
    namespace {
        lfs::Error cuda_view_error(const lfs::ErrorCode code,
                                   const lfs::ErrorDomain domain,
                                   std::string message) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = domain,
                .user_message = std::move(message),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        std::string uuid_hex(const std::array<uint8_t, 16>& uuid) {
            std::string text;
            text.resize(32);
            static constexpr char kHex[] = "0123456789abcdef";
            for (size_t i = 0; i < uuid.size(); ++i) {
                text[i * 2] = kHex[uuid[i] >> 4];
                text[i * 2 + 1] = kHex[uuid[i] & 0x0f];
            }
            return text;
        }

        struct CudaVulkanViewOwner {
            std::shared_ptr<void> vulkan_keep_alive;
            std::shared_ptr<internal::VulkanCudaMemoryImport> import;
            void* registered = nullptr;

            ~CudaVulkanViewOwner() {
                if (registered != nullptr) {
                    unregister_cuda_address_range(registered);
                }
            }
        };
    } // namespace

    bool vulkan_backend_exports_memory() {

        try {
            const auto context = internal::try_live_vulkan_context();
            if (!context || context->dead() || !context->cuda_imports()) {
                return false;
            }
            if (!context->memory().exports_memory() || !context->timeline_exportable()) {
                return false;
            }
            const auto cuda = gpu_backend_device_info(GpuBackend::CUDA, selected_cuda_device());
            VkPhysicalDeviceIDProperties identity{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            properties.pNext = &identity;
            vkGetPhysicalDeviceProperties2(context->physical_device(), &properties);
            if (!cuda || !std::equal(cuda->uuid.begin(), cuda->uuid.end(), identity.deviceUUID))
                return false;
            context->cuda_imports()->import_timeline(context->timeline());
            return context->cuda_imports()->timeline_imported();
        } catch (...) { // LFS-CENSUS-OK(empty-catch): export probe fails closed when CUDA import is absent
            return false;
        }
    }
    lfs::Result<Tensor> cuda_view_of_vulkan_tensor(const Tensor& tensor,
                                                   const cudaStream_t stream) {

        if (!tensor.is_valid()) {
            return cuda_view_error(lfs::ErrorCode::InvalidArgument, lfs::ErrorDomain::Tensor,
                                   "CUDA view of a Vulkan tensor requires a valid tensor");
        }
        const auto vulkan = tensor_vulkan_buffer(tensor);
        if (!vulkan) {
            return cuda_view_error(
                lfs::ErrorCode::InvalidArgument, lfs::ErrorDomain::Tensor,
                gpu_backend_of(tensor) != GpuBackend::Vulkan
                    ? "CUDA view requires a Vulkan-backend tensor"
                    : "CUDA view could not read the Vulkan tensor buffer");
        }
        if (!tensor.is_contiguous()) {
            return cuda_view_error(
                lfs::ErrorCode::InvalidArgument, lfs::ErrorDomain::Tensor,
                "CUDA view of a Vulkan tensor requires contiguous storage");
        }
        if (!vulkan_backend_exports_memory()) {
            return cuda_view_error(
                lfs::ErrorCode::Unsupported, lfs::ErrorDomain::Vulkan,
                "Vulkan tensor backend is not exporting device memory for CUDA");
        }
        const auto context = internal::try_live_vulkan_context();
        if (!context || context->dead() || !context->cuda_imports()) {
            return cuda_view_error(lfs::ErrorCode::FailedPrecondition, lfs::ErrorDomain::Vulkan,
                                   "Vulkan tensor backend is not live");
        }
        const std::array<uint8_t, 16>& vulkan_uuid = context->caps().device_uuid;
        int cuda_device = 0;
        cudaError_t status = cudaGetDevice(&cuda_device);
        if (status != cudaSuccess) {
            (void)cudaGetLastError();
            return cuda_view_error(lfs::ErrorCode::Internal, lfs::ErrorDomain::CUDA,
                                   std::format("cudaGetDevice failed: {}",
                                               cudaGetErrorString(status)));
        }
        cudaDeviceProp properties{};
        status = cudaGetDeviceProperties(&properties, cuda_device);
        if (status != cudaSuccess) {
            (void)cudaGetLastError();
            return cuda_view_error(lfs::ErrorCode::Internal, lfs::ErrorDomain::CUDA,
                                   std::format("cudaGetDeviceProperties failed: {}",
                                               cudaGetErrorString(status)));
        }
        std::array<uint8_t, 16> cuda_uuid{};
        std::memcpy(cuda_uuid.data(), properties.uuid.bytes, cuda_uuid.size());
        if (cuda_uuid != vulkan_uuid) {
            return cuda_view_error(
                lfs::ErrorCode::InvalidArgument, lfs::ErrorDomain::CUDA,
                std::format(
                    "CUDA device {} (UUID {}) does not match the Vulkan tensor device (UUID {})",
                    cuda_device, uuid_hex(cuda_uuid), uuid_hex(vulkan_uuid)));
        }

        const internal::StorageRef storage = internal::storage_ref(tensor);
        const auto block = context->memory().cuda_block_info(storage);
        if (!block) {
            return cuda_view_error(lfs::ErrorCode::Internal, lfs::ErrorDomain::Vulkan,
                                   "CUDA view could not resolve the Vulkan memory block");
        }
        if (block->host_visible) {
            return cuda_view_error(
                lfs::ErrorCode::Unsupported, lfs::ErrorDomain::Vulkan,
                "CUDA view of host-visible Vulkan readback storage is not supported");
        }
        if (!block->exportable) {
            return cuda_view_error(
                lfs::ErrorCode::Unsupported, lfs::ErrorDomain::Vulkan,
                "CUDA view needs pool-backed Vulkan storage; direct (large) allocations are not exportable");
        }
        auto imported = context->cuda_imports()->import_memory(
            block->memory, block->block_size, block->dedicated);
        if (!imported) {
            return imported.error();
        }
        const auto wait = context->cuda_imports()->wait_timeline(vulkan->pending_timeline_value,
                                                                 stream);
        if (!wait) {
            return wait.error();
        }

        auto owner = std::make_shared<CudaVulkanViewOwner>();
        owner->vulkan_keep_alive = vulkan->keep_alive;
        owner->import = *imported;
        void* const data = static_cast<std::byte*>((*imported)->mapped) +
                           block->allocation_offset + storage.byte_offset;
        register_cuda_address_range(data, tensor.bytes(), "vulkan.cuda_view");
        owner->registered = data;
        const GpuBackendScope backend(GpuBackend::CUDA);
        return Tensor::from_external_owner(data, tensor.shape(), Device::GPU, tensor.dtype(),
                                           std::move(owner), 0, stream, "vulkan.cuda_view");
    }
} // namespace lfs::core

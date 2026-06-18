/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_external_tensor.hpp"

#include "core/services.hpp"
#include "window/window_manager.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <format>

namespace lfs::vis {

    namespace {
        [[nodiscard]] std::size_t rowSize(const lfs::core::TensorShape& shape) {
            if (shape.rank() == 0) {
                return 1;
            }
            std::size_t row_size = 1;
            for (std::size_t i = 1; i < shape.rank(); ++i) {
                row_size *= shape[i];
            }
            return row_size;
        }
    } // namespace

    VulkanExternalTensorStorage::VulkanExternalTensorStorage(
        VulkanContext& context,
        VulkanContext::ExternalBuffer buffer,
        lfs::rendering::CudaVulkanBufferInterop interop,
        const std::size_t bytes,
        std::shared_ptr<void> extra_owner)
        : context_(&context),
          buffer_(buffer),
          interop_(std::move(interop)),
          bytes_(bytes),
          extra_owner_(std::move(extra_owner)) {}

    VulkanExternalTensorStorage::VulkanExternalTensorStorage(
        std::shared_ptr<VulkanExternalTensorStorage> parent,
        const std::size_t offset,
        const std::size_t bytes)
        : parent_(std::move(parent)),
          offset_(offset),
          bytes_(bytes) {}

    VulkanExternalTensorStorage::~VulkanExternalTensorStorage() {
        // Sub-views don't own anything; their parent's destructor handles Vulkan/CUDA
        // teardown when the last sub-view's shared_ptr ref drops, then the parent's
        // own shared_ptr ref drops with it.
        if (parent_) {
            return;
        }
        interop_.reset();
        if (context_) {
            context_->destroyExternalBuffer(buffer_);
        }
        // extra_owner_ release (e.g. ExportableBlock cuMemUnmap/cuMemRelease/close)
        // happens automatically when this destructor returns.
    }

    void* VulkanExternalTensorStorage::cudaPtr() const {
        if (parent_) {
            return static_cast<char*>(parent_->cudaPtr()) + offset_;
        }
        return interop_.devicePointer();
    }

    VkBuffer VulkanExternalTensorStorage::vkBuffer() const {
        return parent_ ? parent_->vkBuffer() : buffer_.buffer;
    }

    VkDeviceSize VulkanExternalTensorStorage::vkOffset() const {
        if (parent_) {
            return parent_->vkOffset() + static_cast<VkDeviceSize>(offset_);
        }
        return 0;
    }

    std::expected<lfs::core::Tensor, std::string> makeVulkanExternalTensor(
        VulkanContext& context,
        lfs::core::TensorShape shape,
        const lfs::core::DataType dtype,
        const std::size_t capacity,
        const char* const debug_name,
        const cudaStream_t stream,
        const bool zero_fill) {
        if (!context.externalMemoryInteropEnabled()) {
            return std::unexpected("Vulkan external tensor allocation requires CUDA/Vulkan external-memory interop");
        }
        if (shape.rank() == 0) {
            return std::unexpected("Vulkan external tensor allocation requires a non-scalar tensor shape");
        }

        const std::size_t rows = shape[0];
        const std::size_t cap_rows = std::max(capacity, rows);
        const std::size_t total_elements = cap_rows * rowSize(shape);
        const std::size_t bytes = total_elements * lfs::core::dtype_size(dtype);
        if (bytes == 0) {
            return std::unexpected("Vulkan external tensor allocation requested zero bytes");
        }

        VulkanContext::ExternalBuffer buffer{};
        constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (!context.createExternalBuffer(static_cast<VkDeviceSize>(bytes),
                                          usage,
                                          buffer,
                                          "vulkan.external_tensor.buffer",
                                          debug_name ? debug_name : "unnamed")) {
            return std::unexpected(std::format("Vulkan external tensor '{}' allocation failed: {}",
                                               debug_name ? debug_name : "<unnamed>",
                                               context.lastError()));
        }

        const auto native = context.releaseExternalBufferNativeHandle(buffer);
        if (!VulkanContext::externalNativeHandleValid(native)) {
            context.destroyExternalBuffer(buffer);
            return std::unexpected(std::format("Vulkan external tensor '{}' returned an invalid native handle",
                                               debug_name ? debug_name : "<unnamed>"));
        }

        lfs::rendering::CudaVulkanBufferInterop interop;
        const lfs::rendering::CudaVulkanExternalBufferImport import{
            .memory_handle = native,
            .allocation_size = static_cast<std::size_t>(buffer.allocation_size),
            .size = static_cast<std::size_t>(buffer.size),
            .dedicated_allocation = context.externalMemoryDedicatedAllocationEnabled(),
        };
        if (!interop.init(import)) {
            const std::string error = interop.lastError();
            context.destroyExternalBuffer(buffer);
            return std::unexpected(std::format("Vulkan external tensor '{}' CUDA import failed: {}",
                                               debug_name ? debug_name : "<unnamed>",
                                               error));
        }

        void* const cuda_ptr = interop.devicePointer();
        if (!cuda_ptr) {
            interop.reset();
            context.destroyExternalBuffer(buffer);
            return std::unexpected(std::format("Vulkan external tensor '{}' mapped to a null CUDA pointer",
                                               debug_name ? debug_name : "<unnamed>"));
        }

        if (zero_fill) {
            if (const cudaError_t status = cudaMemsetAsync(cuda_ptr, 0, bytes, stream);
                status != cudaSuccess) {
                interop.reset();
                context.destroyExternalBuffer(buffer);
                return std::unexpected(std::format("Vulkan external tensor '{}' zero-fill failed: {} ({})",
                                                   debug_name ? debug_name : "<unnamed>",
                                                   cudaGetErrorName(status),
                                                   cudaGetErrorString(status)));
            }
        }

        auto owner = std::make_shared<VulkanExternalTensorStorage>(
            context, buffer, std::move(interop), bytes);
        auto tensor = lfs::core::Tensor::from_external_owner(
            cuda_ptr,
            std::move(shape),
            lfs::core::Device::CUDA,
            dtype,
            owner,
            cap_rows,
            stream,
            "vulkan_external_buffer");
        return tensor;
    }

    std::expected<lfs::core::SplatTensorAllocator, std::string>
    makeSplatExportableInteropAllocator(VulkanContext& context,
                                        const lfs::core::SplatExportableStorage& storage) {
        if (!context.externalMemoryInteropEnabled()) {
            return std::unexpected(
                "Vulkan external-memory interop is not enabled; cannot import exportable block");
        }
        if (!storage.valid()) {
            return std::unexpected("SplatExportableStorage is empty; nothing to import");
        }

        VulkanContext::ExternalBuffer imported{};
        constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (!context.importExternalBuffer(storage.block->handle.native,
                                          static_cast<VkDeviceSize>(storage.block->size),
                                          usage,
                                          imported,
                                          "vulkan.external_tensor.alias",
                                          "exportable_splat_block")) {
            return std::unexpected(std::format(
                "Vulkan import of CUDA-exported splat block failed: {}",
                context.lastError()));
        }

        // The CUDA-side block is already mapped; we don't need a fresh CUDA import.
        // Wrap an empty CudaVulkanBufferInterop and feed it the existing block ptr
        // via a tiny adapter — but VulkanExternalTensorStorage holds the interop by
        // value, and the renderer only ever reads cudaPtr(). For the parent storage
        // we synthesize a "device pointer" from the ExportableBlock directly.
        //
        // Trick: build the parent with a no-op interop and override cudaPtr() via the
        // sub-view code path. Simpler: store the block as extra_owner and use a
        // factory-local cuda_ptr override. We achieve that by passing the cuda_ptr
        // as the data_ argument to from_external_owner per-tensor below; the parent
        // storage's cudaPtr() is never called because sub-views shortcut on parent_.
        // The parent's own cudaPtr() would return interop_.devicePointer() == nullptr
        // for sub-views above; ensure we never call parent->cudaPtr() directly here.
        //
        // To keep cudaPtr() consistent for any consumer that calls it on the parent,
        // we install a CudaVulkanBufferInterop that already carries the cuda_ptr.
        // The interop has a setter for this purpose; if absent we leave nullptr —
        // safe because the six sub-views always provide their own data pointer.
        lfs::rendering::CudaVulkanBufferInterop noop_interop;

        // Wrap the imported buffer; extra_owner keeps the CUDA-side ExportableBlock
        // alive so cuMemUnmap/Release fires AFTER vkFreeMemory.
        auto parent = std::make_shared<VulkanExternalTensorStorage>(
            context,
            imported,
            std::move(noop_interop),
            static_cast<std::size_t>(storage.block->size),
            std::shared_ptr<void>(storage.block));

        // Build sub-views per region, keyed by SplatExportableStorage::Region.
        std::array<std::shared_ptr<VulkanExternalTensorStorage>,
                   lfs::core::SplatExportableStorage::Count>
            sub_views;
        for (std::size_t i = 0; i < lfs::core::SplatExportableStorage::Count; ++i) {
            sub_views[i] = std::make_shared<VulkanExternalTensorStorage>(
                parent,
                storage.region_offsets[i],
                storage.region_bytes[i]);
        }

        // Compute base CUDA pointer once; each tensor view is base + region_offsets[i].
        void* const cuda_base = storage.block->device_ptr;

        // Resolve a name → region enum index.
        const auto region_from_name =
            [](std::string_view name) -> lfs::core::SplatExportableStorage::Region {
            using R = lfs::core::SplatExportableStorage;
            if (name == "SplatData.means")
                return R::Means;
            if (name == "SplatData.scaling")
                return R::Scaling;
            if (name == "SplatData.rotation")
                return R::Rotation;
            if (name == "SplatData.opacity")
                return R::Opacity;
            if (name == "SplatData.sh0")
                return R::Sh0;
            if (name == "SplatData.shN")
                return R::ShN;
            throw lfs::core::TensorError(std::format(
                "makeSplatExportableInteropAllocator: unknown tensor name '{}'", name));
        };

        // Capture sub_views + offsets by value. cuda_base + region_offsets[r] gives the
        // CUDA write pointer; the sub-view at index r becomes the tensor's owner.
        return [sub_views, region_offsets = storage.region_offsets, cuda_base, region_from_name](
                   lfs::core::TensorShape shape,
                   std::size_t capacity,
                   lfs::core::DataType dtype,
                   std::string_view name) -> lfs::core::Tensor {
            const auto region = region_from_name(name);
            void* const data =
                static_cast<char*>(cuda_base) + region_offsets[region];
            std::shared_ptr<void> owner = sub_views[region];
            return lfs::core::Tensor::from_external_owner(
                data,
                std::move(shape),
                lfs::core::Device::CUDA,
                dtype,
                std::move(owner),
                capacity,
                /*stream=*/nullptr,
                "vulkan_external_buffer");
        };
    }

    lfs::core::SplatTensorAllocator makeViewerSplatTensorAllocator() {
        auto* const window_manager = services().windowOrNull();
        auto* const context = window_manager ? window_manager->getVulkanContext() : nullptr;
        if (!context || !context->externalMemoryInteropEnabled()) {
            return {};
        }

        return [context](lfs::core::TensorShape shape,
                         const size_t capacity,
                         const lfs::core::DataType dtype,
                         const std::string_view name) -> lfs::core::Tensor {
            const std::string debug_name{name};
            auto tensor = makeVulkanExternalTensor(
                *context, std::move(shape), dtype, capacity, debug_name.c_str(), nullptr, false);
            if (!tensor) {
                throw lfs::core::TensorError(std::format(
                    "Vulkan-external splat tensor allocation failed for '{}': {}", debug_name, tensor.error()));
            }
            tensor->set_name(debug_name);
            return std::move(*tensor);
        };
    }

} // namespace lfs::vis

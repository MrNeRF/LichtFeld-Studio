/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_external_tensor.hpp"

#include "core/cuda_error.hpp"
#include "core/services.hpp"
#include "window/window_manager.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <format>
#include <limits>
#include <optional>

namespace lfs::vis {

    namespace {
        [[nodiscard]] std::size_t rowSize(const lfs::core::TensorShape& shape) {
            if (shape.rank() == 0) {
                return 1;
            }
            std::size_t row_size = 1;
            for (std::size_t i = 1; i < shape.rank(); ++i) {
                if (shape[i] != 0 && row_size > std::numeric_limits<std::size_t>::max() / shape[i]) {
                    return 0;
                }
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
        std::string debug_label,
        std::shared_ptr<void> extra_owner)
        : context_(&context),
          buffer_(buffer),
          interop_(std::move(interop)),
          bytes_(bytes),
          extra_owner_(std::move(extra_owner)) {
        registered_cuda_base_ = interop_.devicePointer();
        if (registered_cuda_base_ != nullptr) {
            lfs::core::register_cuda_address_range(
                registered_cuda_base_, bytes_, std::move(debug_label));
        }
    }

    VulkanExternalTensorStorage::VulkanExternalTensorStorage(
        std::shared_ptr<VulkanExternalTensorStorage> parent,
        const std::size_t offset,
        const std::size_t bytes)
        : parent_(std::move(parent)),
          offset_(offset),
          bytes_(bytes) {}

    VulkanExternalTensorStorage::VulkanExternalTensorStorage(
        std::shared_ptr<VulkanExternalTensorStorage> parent,
        std::shared_ptr<lfs::core::SplatExportableStorage::Control> control,
        const lfs::core::SplatExportableStorage::Region region)
        : parent_(std::move(parent)),
          live_control_(std::move(control)),
          live_region_(region) {
        if (live_control_ &&
            static_cast<std::size_t>(live_region_) < lfs::core::SplatExportableStorage::Count) {
            offset_ = live_control_->region_offsets[live_region_];
            bytes_ = live_control_->region_bytes[live_region_];
        }
    }

    VulkanExternalTensorStorage::~VulkanExternalTensorStorage() {
        // Sub-views don't own anything; their parent's destructor handles Vulkan/CUDA
        // teardown when the last sub-view's shared_ptr ref drops, then the parent's
        // own shared_ptr ref drops with it.
        if (parent_) {
            return;
        }
        if (registered_cuda_base_ != nullptr) {
            lfs::core::unregister_cuda_address_range(registered_cuda_base_);
        }
        interop_.reset();
        if (context_) {
            context_->destroyExternalBuffer(buffer_);
        }
        // extra_owner_ release (e.g. ExportableBlock cuMemUnmap/cuMemRelease/close)
        // happens automatically when this destructor returns.
    }

    VkBuffer VulkanExternalTensorStorage::vkBuffer() const {
        return parent_ ? parent_->vkBuffer() : buffer_.buffer;
    }

    VkDeviceSize VulkanExternalTensorStorage::vkBufferSize() const {
        return parent_ ? parent_->vkBufferSize() : buffer_.size;
    }

    VkDeviceSize VulkanExternalTensorStorage::vkOffset() const {
        if (parent_) {
            std::size_t rel = offset_;
            if (live_control_ &&
                static_cast<std::size_t>(live_region_) < lfs::core::SplatExportableStorage::Count) {
                rel = live_control_->region_offsets[live_region_];
            }
            return parent_->vkOffset() + static_cast<VkDeviceSize>(rel);
        }
        return 0;
    }

    std::size_t VulkanExternalTensorStorage::bytes() const {
        if (live_control_ &&
            static_cast<std::size_t>(live_region_) < lfs::core::SplatExportableStorage::Count) {
            return live_control_->region_bytes[live_region_];
        }
        return bytes_;
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
        const std::size_t row_elements = rowSize(shape);
        const std::size_t element_bytes = lfs::core::dtype_size(dtype);
        if (row_elements == 0 || element_bytes == 0 || cap_rows == 0 ||
            cap_rows > std::numeric_limits<std::size_t>::max() / row_elements ||
            cap_rows * row_elements > std::numeric_limits<std::size_t>::max() / element_bytes) {
            return std::unexpected(std::format(
                "Vulkan external tensor byte sizing must be non-zero and overflow-free (name='{}', rows={}, capacity_rows={}, row_elements={}, element_bytes={}, rank={})",
                debug_name ? debug_name : "<unnamed>",
                rows,
                cap_rows,
                row_elements,
                element_bytes,
                shape.rank()));
        }
        const std::size_t total_elements = cap_rows * row_elements;
        const std::size_t bytes = total_elements * element_bytes;
        std::optional<lfs::rendering::CudaVulkanUploadStream> owned_zero_fill_stream;
        cudaStream_t operation_stream = stream;
        if (zero_fill && operation_stream == nullptr) {
            owned_zero_fill_stream.emplace();
            if (!owned_zero_fill_stream->init()) {
                return std::unexpected(std::format(
                    "Vulkan external tensor could not create the required non-default zero-fill stream (name='{}', requested_stream={:#x}, bytes={}, error={})",
                    debug_name ? debug_name : "<unnamed>",
                    reinterpret_cast<std::uintptr_t>(stream),
                    bytes,
                    owned_zero_fill_stream->lastError()));
            }
            operation_stream = owned_zero_fill_stream->stream();
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
        context.setDebugObjectNamef(VK_OBJECT_TYPE_BUFFER,
                                    buffer.buffer,
                                    "interop.tensor.{}[{}]",
                                    debug_name ? debug_name : "unnamed",
                                    bytes);
        context.setDebugObjectNamef(VK_OBJECT_TYPE_DEVICE_MEMORY,
                                    buffer.memory,
                                    "interop.tensor.{}[{}].memory",
                                    debug_name ? debug_name : "unnamed",
                                    buffer.allocation_size);
        if (buffer.size != bytes || buffer.allocation_size < buffer.size) {
            const std::string error = std::format(
                "Vulkan external tensor allocation size disagrees with the CUDA-visible payload (name='{}', requested_bytes={}, vulkan_visible_size={}, vulkan_allocation_size={})",
                debug_name ? debug_name : "<unnamed>",
                bytes,
                buffer.size,
                buffer.allocation_size);
            context.destroyExternalBuffer(buffer);
            return std::unexpected(error);
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
            if (const cudaError_t status = cudaMemsetAsync(cuda_ptr, 0, bytes, operation_stream);
                status != cudaSuccess) {
                interop.reset();
                context.destroyExternalBuffer(buffer);
                return std::unexpected(std::format("Vulkan external tensor '{}' zero-fill failed: {} ({})",
                                                   debug_name ? debug_name : "<unnamed>",
                                                   cudaGetErrorName(status),
                                                   cudaGetErrorString(status)));
            }
            if (owned_zero_fill_stream && !owned_zero_fill_stream->synchronize()) {
                interop.reset();
                context.destroyExternalBuffer(buffer);
                return std::unexpected(std::format(
                    "Vulkan external tensor '{}' zero-fill stream synchronization failed: {}",
                    debug_name ? debug_name : "<unnamed>",
                    owned_zero_fill_stream->lastError()));
            }
        }

        auto owner = std::make_shared<VulkanExternalTensorStorage>(
            context, buffer, std::move(interop), bytes,
            debug_name ? debug_name : "<unnamed>");
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
        if (storage.block->device_ptr == nullptr || storage.block->size == 0) {
            return std::unexpected(std::format(
                "SplatExportableStorage block must expose non-null CUDA storage (device_pointer={:#x}, block_bytes={})",
                reinterpret_cast<std::uintptr_t>(storage.block->device_ptr),
                storage.block->size));
        }
        for (std::size_t i = 0; i < lfs::core::SplatExportableStorage::Count; ++i) {
            const std::size_t offset = storage.region_offsets[i];
            const std::size_t bytes = storage.region_bytes[i];
            // Degree-0 layouts leave ShN/ShNBounds empty; nothing binds an empty region.
            if (bytes == 0)
                continue;
            if (offset > storage.block->size || bytes > storage.block->size - offset) {
                return std::unexpected(std::format(
                    "SplatExportableStorage region must fit inside the exported Vulkan/CUDA block (region={}, offset={}, bytes={}, block_bytes={})",
                    i,
                    offset,
                    bytes,
                    storage.block->size));
            }
        }

        VulkanContext::ExternalBuffer imported{};
        constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (!context.importExternalBuffer(storage.block->handle.native,
                                          static_cast<VkDeviceSize>(storage.block->size),
                                          static_cast<VkDeviceSize>(storage.block->handle.size),
                                          usage,
                                          imported,
                                          "vulkan.external_tensor.alias",
                                          "exportable_splat_block")) {
            return std::unexpected(std::format(
                "Vulkan import of CUDA-exported splat block failed: {}",
                context.lastError()));
        }

        // The CUDA-side block is already mapped, so the Vulkan ownership object
        // needs no second CUDA import. Tensor::from_external_owner below receives
        // the existing CUDA region pointer; this parent owns only the imported
        // Vulkan buffer and anchors the ExportableBlock lifetime.
        lfs::rendering::CudaVulkanBufferInterop noop_interop;

        // Wrap the imported buffer; extra_owner keeps the CUDA-side ExportableBlock
        // alive so cuMemUnmap/Release fires AFTER vkFreeMemory.
        auto parent = std::make_shared<VulkanExternalTensorStorage>(
            context,
            imported,
            std::move(noop_interop),
            static_cast<std::size_t>(storage.block->size),
            "exportable_splat_block",
            std::shared_ptr<void>(storage.block));

        // Live control block: offsets/bytes update on grow(). Capturing them by
        // value was stale — any Tensor allocated from a pre-grow interop
        // allocator (or a holder that survived rebind) would bake stale region
        // bases. Sub-views still pin the Vulkan import at construction; physical
        // grow always re-imports (new allocator) before further use.
        auto ctrl = storage.control();
        if (!ctrl) {
            return std::unexpected(
                "SplatExportableStorage control block missing; refuse by-value "
                "interop snapshot allocator");
        }

        // Live-control sub-views: offset/bytes re-resolve on every vkOffset()/
        // bytes() so a capacity grow cannot leave the viewer binding a pre-grow
        // region while CUDA readers already use live control.
        // Physical grow still rebuilds the interop parent (new Vk import);
        // these sub-views cover same-block generation bumps and rebindSplatData.
        std::array<std::shared_ptr<VulkanExternalTensorStorage>,
                   lfs::core::SplatExportableStorage::Count>
            sub_views;
        for (std::size_t i = 0; i < lfs::core::SplatExportableStorage::Count; ++i) {
            sub_views[i] = std::make_shared<VulkanExternalTensorStorage>(
                parent,
                ctrl,
                static_cast<lfs::core::SplatExportableStorage::Region>(i));
        }

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
            if (name == "SplatData.shN_value_bounds")
                return R::ShNBounds;
            throw lfs::core::TensorError(std::format(
                "makeSplatExportableInteropAllocator: unknown tensor name '{}'", name));
        };

        // Capture control for live offsets + sub_views for Vulkan ownership.
        // clamp requested capacity to the committed exportable layout.
        // shape/capacity bytes must fit region_bytes (fail loud).
        return [sub_views, ctrl, region_from_name](
                   lfs::core::TensorShape shape,
                   std::size_t capacity,
                   lfs::core::DataType dtype,
                   std::string_view name) -> lfs::core::Tensor {
            using R = lfs::core::SplatExportableStorage;
            const auto region = region_from_name(name);
            if (!ctrl || !ctrl->block || !ctrl->block->device_ptr) {
                throw lfs::core::TensorError(
                    "makeSplatExportableInteropAllocator: control block invalid");
            }
            // Live pointer from control (not a by-value offset snapshot).
            void* const data = ctrl->region_ptr(region);
            const std::size_t region_bytes = ctrl->region_bytes[region];
            std::shared_ptr<void> owner = sub_views[region];
            std::size_t clamped = capacity;
            if (region == R::ShN) {
                dtype = lfs::core::DataType::Float16;
                const std::size_t max_cells = region_bytes / sizeof(std::uint16_t);
                if (max_cells > 0) {
                    clamped = std::min(capacity, max_cells);
                }
            } else if (region == R::ShNBounds) {
                dtype = lfs::core::DataType::Float32;
                const std::size_t max_floats = region_bytes / sizeof(float);
                if (max_floats > 0) {
                    clamped = std::min(capacity, max_floats);
                }
            } else if (ctrl->capacity > 0) {
                clamped = std::min(capacity, ctrl->capacity);
            }

            const auto dtype_bytes = [](lfs::core::DataType dt) -> std::size_t {
                switch (dt) {
                case lfs::core::DataType::Float32:
                    return 4;
                case lfs::core::DataType::Float16:
                    return 2;
                case lfs::core::DataType::Int32:
                case lfs::core::DataType::UInt8:
                    return dt == lfs::core::DataType::UInt8 ? 1 : 4;
                case lfs::core::DataType::Int64:
                    return 8;
                case lfs::core::DataType::Bool:
                    return 1;
                default:
                    return 0;
                }
            };
            const std::size_t elem_b = dtype_bytes(dtype);
            if (elem_b == 0) {
                throw lfs::core::TensorError(std::format(
                    "makeSplatExportableInteropAllocator: invalid dtype for '{}'", name));
            }
            std::size_t row_elems = 1;
            if (shape.rank() > 1) {
                for (std::size_t i = 1; i < shape.rank(); ++i) {
                    row_elems *= shape[i];
                }
            }
            const std::size_t shape_rows = shape.rank() == 0 ? 0 : shape[0];
            const std::size_t shape_bytes = shape_rows * row_elems * elem_b;
            if (shape_bytes > region_bytes) {
                throw lfs::core::TensorError(std::format(
                    "makeSplatExportableInteropAllocator: shape for '{}' needs {} bytes "
                    "but region only holds {}",
                    name,
                    shape_bytes,
                    region_bytes));
            }
            const std::size_t rows = shape.rank() == 0 ? 0 : (clamped == 0 ? shape[0] : clamped);
            const std::size_t alloc_bytes = rows * row_elems * elem_b;
            if (alloc_bytes > region_bytes) {
                throw lfs::core::TensorError(std::format(
                    "makeSplatExportableInteropAllocator: capacity for '{}' needs {} bytes "
                    "but region only holds {}",
                    name,
                    alloc_bytes,
                    region_bytes));
            }

            auto t = lfs::core::Tensor::from_external_owner(
                data,
                std::move(shape),
                lfs::core::Device::CUDA,
                dtype,
                std::move(owner),
                clamped,
                /*stream=*/nullptr,
                "vulkan_external_buffer");
            lfs::core::stamp_exportable_provenance(t, ctrl, region);
            return t;
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

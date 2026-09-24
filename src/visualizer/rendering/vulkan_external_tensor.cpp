/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_external_tensor.hpp"

#include "core/exportable_storage.hpp"
#include "core/services.hpp"
#include "core/shareable_allocation_limit.hpp"
#include "core/tensor_backend.hpp"
#include "window/window_manager.hpp"

#include <algorithm>
#include <array>
#include <format>
#include <limits>

namespace lfs::vis {

#if LFS_BUILD_TRAINER
    namespace {
        [[nodiscard]] lfs::Error interop_error(lfs::ErrorCode code, std::string message) {
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::Vulkan,
                .user_message = std::move(message),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

    } // namespace

    VulkanExternalTensorStorage::VulkanExternalTensorStorage(
        VulkanContext& context, std::shared_ptr<lfs::core::ExportableBlock> block)
        : context_(&context) {
        void* const pointer = block->device_ptr;
        block_tensor_ = lfs::core::Tensor::from_external_owner(pointer, {1},
                                                               lfs::core::Device::GPU, lfs::core::DataType::UInt8, std::move(block), 1,
                                                               nullptr, "vulkan_external_buffer");
        buffer_ = context.tensorInterop().buffer(block_tensor_).value();
    }

    VulkanExternalTensorStorage::~VulkanExternalTensorStorage() = default;

    bool VulkanExternalTensorStorage::bindNewExportableChunks(const lfs::core::ExportableBlock& block) {
        if (!context_) {
            return false;
        }
        (void)block;
        buffer_ = context_->tensorInterop().buffer(block_tensor_).value();
        return true;
    }

    std::expected<lfs::core::Tensor, std::string> makeVulkanExternalTensor(
        VulkanContext& context,
        lfs::core::TensorShape shape,
        const lfs::core::DataType dtype,
        const std::size_t capacity,
        const char* const debug_name) {
        try {
            auto tensor = context.tensorInterop().empty(std::move(shape), dtype,
                                                        lfs::core::GpuBackend::CUDA, capacity);
            if (debug_name)
                tensor.set_name(debug_name);
            return tensor;
        } catch (const std::exception& error) {
            return std::unexpected(error.what());
        }
    }

    lfs::Result<lfs::core::SplatTensorAllocator>
    makeSplatExportableInteropAllocator(VulkanContext& context,
                                        const lfs::core::SplatExportableStorage& storage,
                                        std::shared_ptr<VulkanExternalTensorStorage>* parent_keep) {
        if (!context.externalMemoryInteropEnabled()) {
            return lfs::Result<lfs::core::SplatTensorAllocator>(interop_error(
                lfs::ErrorCode::FailedPrecondition,
                "Vulkan external-memory interop is not enabled; cannot import exportable block"));
        }
        if (!storage.valid()) {
            return lfs::Result<lfs::core::SplatTensorAllocator>(interop_error(
                lfs::ErrorCode::FailedPrecondition,
                "SplatExportableStorage is empty; nothing to import"));
        }
        if (storage.block->device_ptr == nullptr || storage.block->reserved_bytes == 0) {
            return lfs::Result<lfs::core::SplatTensorAllocator>(interop_error(
                lfs::ErrorCode::FailedPrecondition,
                std::format(
                    "SplatExportableStorage block must expose non-null CUDA storage (device_pointer={:#x}, reserved_bytes={})",
                    reinterpret_cast<std::uintptr_t>(storage.block->device_ptr),
                    storage.block->reserved_bytes)));
        }
        for (std::size_t i = 0; i < lfs::core::SplatExportableStorage::Count; ++i) {
            const std::size_t offset = storage.region_offsets[i];
            const std::size_t bytes = storage.region_bytes[i];
            // Degree-0 layouts leave ShN/ShNBounds empty; nothing binds an empty region.
            if (bytes == 0)
                continue;
            if (offset > storage.block->reserved_bytes ||
                bytes > storage.block->reserved_bytes - offset) {
                return lfs::Result<lfs::core::SplatTensorAllocator>(interop_error(
                    lfs::ErrorCode::FailedPrecondition,
                    std::format(
                        "SplatExportableStorage region must fit inside the reserved Vulkan/CUDA block (region={}, offset={}, bytes={}, reserved_bytes={})",
                        i,
                        offset,
                        bytes,
                        storage.block->reserved_bytes)));
            }
        }

        std::shared_ptr<VulkanExternalTensorStorage> parent;
        if (parent_keep && *parent_keep) {
            parent = *parent_keep;
            if (!parent->bindNewExportableChunks(*storage.block)) {
                return lfs::Result<lfs::core::SplatTensorAllocator>(interop_error(
                    lfs::ErrorCode::Internal,
                    std::format("Vulkan bind of new exportable chunks failed: {}",
                                context.lastError())));
            }
        } else {
            parent = std::make_shared<VulkanExternalTensorStorage>(context, storage.block);
            if (parent_keep) {
                *parent_keep = parent;
            }
        }

        // Live control block: offsets are constant; bytes/generation update on
        // grow(). The parent pins the stable VkBuffer; bindNewChunks appends.
        auto ctrl = storage.control();
        if (!ctrl) {
            return lfs::Result<lfs::core::SplatTensorAllocator>(interop_error(
                lfs::ErrorCode::FailedPrecondition,
                "SplatExportableStorage control block missing; refuse by-value "
                "interop snapshot allocator"));
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

        // Shape and capacity must fit the committed region; the parent pins its import.
        return [parent, ctrl, region_from_name](
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
            std::shared_ptr<void> owner = parent;
            std::size_t clamped = capacity;
            if (region == R::ShN) {
                if (lfs::core::sh_value_quant::enabled()) {
                    dtype = lfs::core::DataType::Float16;
                    const std::size_t max_cells = region_bytes / sizeof(std::uint16_t);
                    if (max_cells > 0) {
                        clamped = std::min(capacity, max_cells);
                    }
                } else {
                    dtype = lfs::core::DataType::Float32;
                    const std::size_t max_floats = region_bytes / sizeof(float);
                    if (max_floats > 0) {
                        clamped = std::min(capacity, max_floats);
                    }
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
                lfs::core::Device::GPU,
                dtype,
                std::move(owner),
                clamped,
                /*stream=*/nullptr,
                "vulkan_external_buffer");
            lfs::core::stamp_exportable_provenance(t, ctrl, region);
            return t;
        };
    }

#endif

    lfs::core::SplatTensorAllocator makeViewerSplatTensorAllocator(const bool preserve_float_shN) {
        auto* window = services().windowOrNull();
        auto* context = window ? window->getVulkanContext() : nullptr;
        return context ? context->tensorInterop().splat_allocator(preserve_float_shN)
                       : lfs::core::SplatTensorAllocator{};
    }

} // namespace lfs::vis

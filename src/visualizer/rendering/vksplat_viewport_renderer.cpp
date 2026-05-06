/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vksplat_viewport_renderer.hpp"

#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "viewport/vksplat_compose.comp.spv.h"
#include "vksplat_input_packer.hpp"
#include "window/vulkan_result.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <glm/glm.hpp>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::vis {
    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        constexpr std::uint32_t kVkSplatProjectionModeShift = 8u;
        constexpr std::uint32_t kVkSplatProjectionModeGut = 1u;

        [[nodiscard]] std::string vkError(const char* const operation, const VkResult result) {
            return std::format("{} failed: {}", operation, vkResultToString(result));
        }

        [[nodiscard]] std::map<std::string, std::string> makeVkSplatSpirvPaths() {
            const std::filesystem::path root{LFS_VKSPLAT_SPV_DIR};
            return {
                {"projection_forward", (root / "generated/projection_forward.spv").string()},
                {"projection_forward_gut", (root / "generated/projection_forward_gut.spv").string()},
                {"selection_mask", (root / "generated/selection_mask.spv").string()},
                {"generate_keys", (root / "generated/generate_keys.spv").string()},
                {"compute_tile_ranges", (root / "generated/compute_tile_ranges.spv").string()},
                {"setup_dispatch_indirect", (root / "generated/setup_dispatch_indirect.spv").string()},
                {"rasterize_forward", (root / "generated/rasterize_forward.spv").string()},
                {"rasterize_forward_gut", (root / "generated/rasterize_forward_gut.spv").string()},
                {"cumsum_single_pass", (root / "generated/cumsum_single_pass.spv").string()},
                {"cumsum_block_scan", (root / "generated/cumsum_block_scan.spv").string()},
                {"cumsum_scan_block_sums", (root / "generated/cumsum_scan_block_sums.spv").string()},
                {"cumsum_add_block_offsets", (root / "generated/cumsum_add_block_offsets.spv").string()},
                {"radix_sort/upsweep", (root / "radix_sort/upsweep.spv").string()},
                {"radix_sort/spine", (root / "radix_sort/spine.spv").string()},
                {"radix_sort/downsweep", (root / "radix_sort/downsweep.spv").string()},
            };
        }

        [[nodiscard]] std::array<float, 16> rowMajorMat4(const glm::mat4& matrix) {
            std::array<float, 16> row_major{};
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    row_major[static_cast<std::size_t>(row * 4 + col)] = matrix[col][row];
                }
            }
            return row_major;
        }

        [[nodiscard]] std::size_t alignUp(const std::size_t value, const std::size_t alignment) {
            return ((value + alignment - 1) / alignment) * alignment;
        }

        [[nodiscard]] constexpr std::size_t outputSlotIndex(const VksplatViewportRenderer::OutputSlot slot) {
            return static_cast<std::size_t>(slot);
        }

        struct ScopedStagingBuffer {
            VmaAllocator allocator = VK_NULL_HANDLE;
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VmaAllocationInfo allocation_info{};

            ~ScopedStagingBuffer() {
                if (allocator != VK_NULL_HANDLE && buffer != VK_NULL_HANDLE) {
                    vmaDestroyBuffer(allocator, buffer, allocation);
                }
            }
        };

        struct ScopedCommandPool {
            VkDevice device = VK_NULL_HANDLE;
            VkCommandPool pool = VK_NULL_HANDLE;

            ~ScopedCommandPool() {
                if (device != VK_NULL_HANDLE && pool != VK_NULL_HANDLE) {
                    vkDestroyCommandPool(device, pool, nullptr);
                }
            }
        };

        struct ScopedFence {
            VkDevice device = VK_NULL_HANDLE;
            VkFence fence = VK_NULL_HANDLE;

            ~ScopedFence() {
                if (device != VK_NULL_HANDLE && fence != VK_NULL_HANDLE) {
                    vkDestroyFence(device, fence, nullptr);
                }
            }
        };

        template <std::size_t RegionCount>
        [[nodiscard]] std::size_t layoutRegions(const std::array<std::size_t, RegionCount>& region_bytes,
                                                std::array<std::size_t, RegionCount>& region_offset,
                                                const std::size_t alignment) {
            std::size_t cursor = 0;
            for (std::size_t i = 0; i < RegionCount; ++i) {
                region_offset[i] = cursor;
                cursor += alignUp(region_bytes[i], alignment);
            }
            return cursor;
        }

        [[nodiscard]] _VulkanBuffer makeRegionView(const VulkanContext::ExternalBuffer& buffer,
                                                   const std::size_t offset,
                                                   const std::size_t bytes) {
            _VulkanBuffer view{};
            view.buffer = buffer.buffer;
            view.allocation = VK_NULL_HANDLE;
            view.allocSize = static_cast<std::size_t>(buffer.allocation_size);
            view.offset = offset;
            view.size = bytes;
            return view;
        }

        [[nodiscard]] std::expected<void, std::string> ensureCudaInteropBuffer(
            VulkanContext& context,
            VulkanContext::ExternalBuffer& buffer,
            lfs::rendering::CudaVulkanBufferInterop& interop,
            const std::size_t required_bytes,
            const char* const debug_name,
            const std::string_view label) {
            if (required_bytes == 0) {
                return std::unexpected(std::format("VkSplat {} slot '{}' requested zero-byte allocation",
                                                   label,
                                                   debug_name));
            }
            if (buffer.buffer != VK_NULL_HANDLE && buffer.allocation_size >= required_bytes) {
                return {};
            }

            interop.reset();
            context.destroyExternalBuffer(buffer);

            const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            if (!context.createExternalBuffer(static_cast<VkDeviceSize>(required_bytes), usage, buffer)) {
                return std::unexpected(std::format("VkSplat external {} buffer '{}' allocation failed: {}",
                                                   label,
                                                   debug_name,
                                                   context.lastError()));
            }
            const auto native = context.releaseExternalBufferNativeHandle(buffer);
            if (!VulkanContext::externalNativeHandleValid(native)) {
                context.destroyExternalBuffer(buffer);
                return std::unexpected(std::format("VkSplat external {} buffer '{}' returned invalid native handle",
                                                   label,
                                                   debug_name));
            }
            lfs::rendering::CudaVulkanExternalBufferImport import{
                .memory_handle = native,
                .allocation_size = static_cast<std::size_t>(buffer.allocation_size),
                .size = static_cast<std::size_t>(buffer.size),
                .dedicated_allocation = context.externalMemoryDedicatedAllocationEnabled(),
            };
            if (!interop.init(import)) {
                const std::string err = interop.lastError();
                context.destroyExternalBuffer(buffer);
                return std::unexpected(std::format("VkSplat external {} buffer '{}' CUDA import failed: {}",
                                                   label,
                                                   debug_name,
                                                   err));
            }
            return {};
        }

        enum InputRegion : std::size_t {
            InputXyzWs = 0,
            InputRotations = 1,
            InputScalesOpacs = 2,
            InputShCoeffs = 3,
        };

        enum OverlayRegion : std::size_t {
            OverlaySelectionMask = 0,
            OverlayPreviewMask = 1,
            OverlaySelectionColors = 2,
            OverlayTransformIndices = 3,
            OverlayNodeMask = 4,
            OverlayParams = 5,
            OverlayModelTransforms = 6,
        };

        enum SelectionQueryRegion : std::size_t {
            SelectionQueryOutput = 0,
            SelectionQueryTransformIndices = 1,
            SelectionQueryNodeMask = 2,
            SelectionQueryPrimitives = 3,
            SelectionQueryModelTransforms = 4,
        };

        [[nodiscard]] bool hasOverlayTensor(const Tensor* const tensor, const std::size_t num_splats) {
            return tensor && tensor->is_valid() && tensor->numel() >= num_splats && num_splats > 0;
        }

        [[nodiscard]] bool hasOverlayTensor(const std::shared_ptr<Tensor>& tensor, const std::size_t num_splats) {
            return tensor && hasOverlayTensor(tensor.get(), num_splats);
        }

        [[nodiscard]] std::expected<Tensor, std::string> prepareOverlayMaskTensor(
            Tensor tensor,
            const std::size_t num_splats,
            const std::string_view label) {
            if (!tensor.is_valid() || tensor.numel() < num_splats) {
                return std::unexpected(std::format(
                    "VkSplat selection overlay expected {} mask with at least {} entries",
                    label,
                    num_splats));
            }
            try {
                if (tensor.dtype() != DataType::UInt8 && tensor.dtype() != DataType::Bool) {
                    tensor = tensor.to(DataType::UInt8);
                }
                if (tensor.device() != Device::CUDA) {
                    tensor = tensor.to(Device::CUDA);
                }
                if (!tensor.is_contiguous()) {
                    tensor = tensor.contiguous();
                }
            } catch (const std::exception& e) {
                return std::unexpected(std::format("VkSplat failed to prepare {} mask: {}", label, e.what()));
            }
            if (tensor.bytes() < num_splats) {
                return std::unexpected(std::format(
                    "VkSplat {} mask has {} bytes, expected at least {}",
                    label,
                    tensor.bytes(),
                    num_splats));
            }
            return tensor;
        }

        [[nodiscard]] std::expected<Tensor, std::string> makeSelectionColorTableTensor(
            const lfs::rendering::GaussianOverlayState& overlay) {
            try {
                Tensor cpu = Tensor::empty(
                    {lfs::rendering::kSelectionColorTableCount, std::size_t{4}},
                    Device::CPU,
                    DataType::Float32);
                float* const dst = cpu.ptr<float>();
                if (!dst) {
                    return std::unexpected("VkSplat selection color table allocation returned a null pointer");
                }
                for (std::size_t i = 0; i < lfs::rendering::kSelectionColorTableCount; ++i) {
                    const glm::vec4 color = overlay.selection_colors[i];
                    dst[i * 4 + 0] = color.r;
                    dst[i * 4 + 1] = color.g;
                    dst[i * 4 + 2] = color.b;
                    dst[i * 4 + 3] = color.a;
                }
                return cpu.to(Device::CUDA).contiguous();
            } catch (const std::exception& e) {
                return std::unexpected(std::format("VkSplat failed to stage selection color table: {}", e.what()));
            }
        }

        enum OverlayParamIndex : std::size_t {
            CropFlags = 0,
            CropMin = 1,
            CropMax = 2,
            CropTransform = 3,
            EllipsoidFlags = 7,
            EllipsoidRadii = 8,
            EllipsoidTransform = 9,
            ViewFlags = 13,
            ViewMin = 14,
            ViewMax = 15,
            ViewTransform = 16,
            EmphasisFlags = 20,
            CursorFlags = 21,
            MarkerFlags = 22,
            SelectionCursor = 23,
            SelectionFlags = 24,
            ParamCount = 25,
        };

        [[nodiscard]] bool hasTransformIndices(const std::shared_ptr<Tensor>& tensor,
                                               const std::size_t num_splats) {
            return tensor && tensor->is_valid() && tensor->numel() >= num_splats;
        }

        [[nodiscard]] std::expected<Tensor, std::string> prepareTransformIndicesTensor(
            const std::shared_ptr<Tensor>& tensor,
            const std::size_t num_splats) {
            try {
                if (!hasTransformIndices(tensor, num_splats)) {
                    return Tensor::zeros({std::size_t{1}}, Device::CUDA, DataType::Int32);
                }
                Tensor prepared = *tensor;
                if (prepared.dtype() != DataType::Int32) {
                    prepared = prepared.to(DataType::Int32);
                }
                if (prepared.device() != Device::CUDA) {
                    prepared = prepared.to(Device::CUDA);
                }
                if (!prepared.is_contiguous()) {
                    prepared = prepared.contiguous();
                }
                return prepared;
            } catch (const std::exception& e) {
                return std::unexpected(std::format("VkSplat failed to stage transform indices: {}", e.what()));
            }
        }

        [[nodiscard]] std::expected<Tensor, std::string> makeNodeMaskTensor(const std::vector<bool>& mask) {
            try {
                const std::size_t count = std::max<std::size_t>(mask.size(), 1);
                Tensor cpu = Tensor::zeros({count}, Device::CPU, DataType::UInt8);
                auto* const dst = cpu.ptr<std::uint8_t>();
                if (!dst) {
                    return std::unexpected("VkSplat node mask allocation returned a null pointer");
                }
                for (std::size_t i = 0; i < mask.size(); ++i) {
                    dst[i] = mask[i] ? 1u : 0u;
                }
                return cpu.to(Device::CUDA).contiguous();
            } catch (const std::exception& e) {
                return std::unexpected(std::format("VkSplat failed to stage node mask: {}", e.what()));
            }
        }

        [[nodiscard]] std::expected<Tensor, std::string> makeSelectionPrimitiveTensor(
            const std::vector<glm::vec4>& primitives) {
            try {
                if (primitives.empty()) {
                    return std::unexpected("VkSplat selection query requires at least one primitive");
                }
                Tensor cpu = Tensor::empty({primitives.size(), std::size_t{4}},
                                           Device::CPU,
                                           DataType::Float32);
                float* const dst = cpu.ptr<float>();
                if (!dst) {
                    return std::unexpected("VkSplat selection primitive allocation returned a null pointer");
                }
                for (std::size_t i = 0; i < primitives.size(); ++i) {
                    dst[i * 4 + 0] = primitives[i].x;
                    dst[i * 4 + 1] = primitives[i].y;
                    dst[i * 4 + 2] = primitives[i].z;
                    dst[i * 4 + 3] = primitives[i].w;
                }
                return cpu.to(Device::CUDA).contiguous();
            } catch (const std::exception& e) {
                return std::unexpected(std::format("VkSplat failed to stage selection primitives: {}", e.what()));
            }
        }

        [[nodiscard]] std::size_t modelTransformCount(const std::vector<glm::mat4>* const transforms) {
            return transforms && !transforms->empty() ? transforms->size() : std::size_t{1};
        }

        [[nodiscard]] std::expected<Tensor, std::string> makeModelTransformsTensor(
            const std::vector<glm::mat4>* const transforms) {
            try {
                const std::size_t count = modelTransformCount(transforms);
                Tensor cpu = Tensor::empty({count * std::size_t{4}, std::size_t{4}},
                                           Device::CPU,
                                           DataType::Float32);
                float* const dst = cpu.ptr<float>();
                if (!dst) {
                    return std::unexpected("VkSplat model-transform allocation returned a null pointer");
                }
                for (std::size_t i = 0; i < count; ++i) {
                    const glm::mat4 transform =
                        transforms && i < transforms->size() ? (*transforms)[i] : glm::mat4(1.0f);
                    const auto rows = rowMajorMat4(transform);
                    std::memcpy(dst + i * 16, rows.data(), rows.size() * sizeof(float));
                }
                return cpu.to(Device::CUDA).contiguous();
            } catch (const std::exception& e) {
                return std::unexpected(std::format("VkSplat failed to stage model transforms: {}", e.what()));
            }
        }

        void writeVec4(float* dst, const std::size_t index, const glm::vec4& value) {
            dst[index * 4 + 0] = value.x;
            dst[index * 4 + 1] = value.y;
            dst[index * 4 + 2] = value.z;
            dst[index * 4 + 3] = value.w;
        }

        void writeMat4Rows(float* dst, const std::size_t index, const glm::mat4& matrix) {
            for (int row = 0; row < 4; ++row) {
                writeVec4(dst,
                          index + static_cast<std::size_t>(row),
                          glm::vec4(matrix[0][row],
                                    matrix[1][row],
                                    matrix[2][row],
                                    matrix[3][row]));
            }
        }

        [[nodiscard]] std::expected<Tensor, std::string> makeOverlayParamsTensor(
            const lfs::rendering::ViewportRenderRequest& request,
            const bool selection_enabled,
            const bool preview_enabled,
            const bool transform_indices_enabled,
            const std::size_t node_mask_count) {
            try {
                Tensor cpu = Tensor::zeros({static_cast<std::size_t>(ParamCount), std::size_t{4}},
                                           Device::CPU,
                                           DataType::Float32);
                float* const dst = cpu.ptr<float>();
                if (!dst) {
                    return std::unexpected("VkSplat overlay parameter allocation returned a null pointer");
                }

                if (request.filters.crop_region) {
                    const auto& crop = *request.filters.crop_region;
                    writeVec4(dst,
                              CropFlags,
                              glm::vec4(1.0f,
                                        crop.inverse ? 1.0f : 0.0f,
                                        crop.desaturate ? 1.0f : 0.0f,
                                        static_cast<float>(crop.parent_node_index)));
                    writeVec4(dst, CropMin, glm::vec4(crop.bounds.min, 0.0f));
                    writeVec4(dst, CropMax, glm::vec4(crop.bounds.max, 0.0f));
                    writeMat4Rows(dst, CropTransform, crop.bounds.transform);
                }

                if (request.filters.ellipsoid_region) {
                    const auto& ellipsoid = *request.filters.ellipsoid_region;
                    writeVec4(dst,
                              EllipsoidFlags,
                              glm::vec4(1.0f,
                                        ellipsoid.inverse ? 1.0f : 0.0f,
                                        ellipsoid.desaturate ? 1.0f : 0.0f,
                                        static_cast<float>(ellipsoid.parent_node_index)));
                    writeVec4(dst, EllipsoidRadii, glm::vec4(ellipsoid.bounds.radii, 0.0f));
                    writeMat4Rows(dst, EllipsoidTransform, ellipsoid.bounds.transform);
                }

                if (request.filters.view_volume) {
                    writeVec4(dst,
                              ViewFlags,
                              glm::vec4(1.0f,
                                        request.filters.cull_outside_view_volume ? 1.0f : 0.0f,
                                        0.0f,
                                        0.0f));
                    writeVec4(dst, ViewMin, glm::vec4(request.filters.view_volume->min, 0.0f));
                    writeVec4(dst, ViewMax, glm::vec4(request.filters.view_volume->max, 0.0f));
                    writeMat4Rows(dst, ViewTransform, request.filters.view_volume->transform);
                }

                writeVec4(dst,
                          EmphasisFlags,
                          glm::vec4(request.overlay.emphasis.dim_non_emphasized ? 1.0f : 0.0f,
                                    transform_indices_enabled ? 1.0f : 0.0f,
                                    static_cast<float>(node_mask_count),
                                    request.overlay.emphasis.flash_intensity));
                writeVec4(dst,
                          CursorFlags,
                          glm::vec4(request.overlay.cursor.enabled ? 1.0f : 0.0f,
                                    request.overlay.cursor.saturation_preview ? 1.0f : 0.0f,
                                    request.overlay.cursor.saturation_amount,
                                    request.overlay.markers.ring_width));
                writeVec4(dst,
                          MarkerFlags,
                          glm::vec4(request.overlay.markers.show_rings ? 1.0f : 0.0f,
                                    request.overlay.markers.show_center_markers ? 1.0f : 0.0f,
                                    0.0f,
                                    0.0f));
                const bool cursor_selection_enabled =
                    !request.overlay.cursor.saturation_preview &&
                    request.overlay.cursor.enabled &&
                    request.overlay.cursor.radius > 0.0f;
                writeVec4(dst,
                          SelectionCursor,
                          glm::vec4(request.overlay.cursor.cursor.x,
                                    request.overlay.cursor.cursor.y,
                                    std::max(request.overlay.cursor.radius, 0.0f),
                                    cursor_selection_enabled ? 1.0f : 0.0f));
                writeVec4(dst,
                          SelectionFlags,
                          glm::vec4(selection_enabled ? 1.0f : 0.0f,
                                    preview_enabled ? 1.0f : 0.0f,
                                    request.overlay.emphasis.transient_mask.additive ? 1.0f : 0.0f,
                                    request.overlay.emphasis.focused_gaussian_id >= 0
                                        ? static_cast<float>(request.overlay.emphasis.focused_gaussian_id)
                                        : -1.0f));

                return cpu.to(Device::CUDA).contiguous();
            } catch (const std::exception& e) {
                return std::unexpected(std::format("VkSplat failed to stage overlay parameters: {}", e.what()));
            }
        }

        [[nodiscard]] VksplatViewportRenderer::ModelInputSnapshot makeModelInputSnapshot(
            const lfs::core::SplatData& splat_data) {
            const auto tensor_ptr = [](const Tensor& tensor) -> const void* {
                return tensor.is_valid() ? tensor.data_ptr() : nullptr;
            };
            const auto tensor_bytes = [](const Tensor& tensor) -> std::size_t {
                return tensor.is_valid() ? tensor.bytes() : 0;
            };

            const Tensor& means = splat_data.means_raw();
            const Tensor& scaling = splat_data.scaling_raw();
            const Tensor& rotation = splat_data.rotation_raw();
            const Tensor& opacity = splat_data.opacity_raw();
            const Tensor& sh0 = splat_data.sh0_raw();
            const Tensor& shn = splat_data.shN_raw();
            return VksplatViewportRenderer::ModelInputSnapshot{
                .model = &splat_data,
                .count = static_cast<std::size_t>(splat_data.size()),
                .max_sh_degree = splat_data.get_max_sh_degree(),
                .means = tensor_ptr(means),
                .scaling = tensor_ptr(scaling),
                .rotation = tensor_ptr(rotation),
                .opacity = tensor_ptr(opacity),
                .sh0 = tensor_ptr(sh0),
                .shn = tensor_ptr(shn),
                .means_bytes = tensor_bytes(means),
                .scaling_bytes = tensor_bytes(scaling),
                .rotation_bytes = tensor_bytes(rotation),
                .opacity_bytes = tensor_bytes(opacity),
                .sh0_bytes = tensor_bytes(sh0),
                .shn_bytes = tensor_bytes(shn),
            };
        }

        void populateVksplatCameraUniforms(
            VulkanGSRendererUniforms& uniforms,
            const lfs::rendering::FrameView& frame_view,
            const lfs::rendering::GaussianSceneState& scene,
            const int active_sh_degree,
            const std::size_t num_splats) {
            (void)scene;
            uniforms = {};
            uniforms.image_width = static_cast<std::uint32_t>(frame_view.size.x);
            uniforms.image_height = static_cast<std::uint32_t>(frame_view.size.y);
            uniforms.grid_width = _CEIL_DIV(uniforms.image_width, TILE_WIDTH);
            uniforms.grid_height = _CEIL_DIV(uniforms.image_height, TILE_HEIGHT);
            uniforms.num_splats = static_cast<std::uint32_t>(num_splats);
            uniforms.active_sh = static_cast<std::uint32_t>(active_sh_degree);
            uniforms.camera_model = 0;

            if (frame_view.intrinsics_override) {
                const auto& intrinsics = *frame_view.intrinsics_override;
                uniforms.fx = intrinsics.focal_x;
                uniforms.fy = intrinsics.focal_y;
                uniforms.cx = intrinsics.center_x;
                uniforms.cy = intrinsics.center_y;
            } else {
                const auto [fx, fy] = lfs::rendering::computePixelFocalLengths(
                    frame_view.size, frame_view.focal_length_mm);
                uniforms.fx = fx;
                uniforms.fy = fy;
                uniforms.cx = static_cast<float>(frame_view.size.x) * 0.5f;
                uniforms.cy = static_cast<float>(frame_view.size.y) * 0.5f;
            }

            const glm::mat3 camera_to_world =
                lfs::rendering::dataCameraToWorldFromVisualizerRotation(frame_view.rotation);
            const glm::mat3 world_to_camera = glm::transpose(camera_to_world);
            const glm::vec3 translation = -world_to_camera * frame_view.translation;

            std::array<float, 16> row_major_view{};
            row_major_view[15] = 1.0f;
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    row_major_view[static_cast<std::size_t>(row * 4 + col)] = world_to_camera[col][row];
                }
            }
            row_major_view[3] = translation.x;
            row_major_view[7] = translation.y;
            row_major_view[11] = translation.z;
            for (int row = 0; row < 4; ++row) {
                for (int col = 0; col < 4; ++col) {
                    uniforms.world_view_transform[4 * row + col] =
                        row_major_view[static_cast<std::size_t>(4 * col + row)];
                }
            }
        }

        struct ComposePushConstants {
            std::uint32_t width = 0;
            std::uint32_t height = 0;
            std::uint32_t transparent_background = 0;
            std::uint32_t pad1 = 0;
            glm::vec4 background{0.0f, 0.0f, 0.0f, 1.0f};
        };

    } // namespace

    struct VksplatViewportRenderer::ComposePipeline {
        VkShaderModule shader_module = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        void destroy(VkDevice device) {
            if (device == VK_NULL_HANDLE) {
                return;
            }
            if (pipeline != VK_NULL_HANDLE) {
                vkDestroyPipeline(device, pipeline, nullptr);
            }
            if (pipeline_layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            }
            if (descriptor_set_layout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
            }
            if (shader_module != VK_NULL_HANDLE) {
                vkDestroyShaderModule(device, shader_module, nullptr);
            }
            *this = {};
        }
    };

    VksplatViewportRenderer::VksplatViewportRenderer() = default;

    VksplatViewportRenderer::~VksplatViewportRenderer() {
        reset();
    }

    void VksplatViewportRenderer::reset() {
        if (context_ && context_->device() != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(context_->device());
        }
        // Detach our managed VkBuffers from buffers_ before the renderer's
        // cleanupBuffers runs so it does not vkDestroyBuffer them out from
        // under us.
        if (initialized_) {
            detachManagedBuffers();
            renderer_.cleanupBuffers(buffers_);
            renderer_.cleanup();
        }
        for (auto& slot : cuda_inputs_) {
            slot.interop.reset();
            if (context_) {
                context_->destroyExternalBuffer(slot.buffer);
            }
            slot = {};
        }
        for (auto& slot : cuda_overlays_) {
            slot.interop.reset();
            if (context_) {
                context_->destroyExternalBuffer(slot.buffer);
            }
            slot = {};
        }
        cuda_selection_query_.interop.reset();
        if (context_) {
            context_->destroyExternalBuffer(cuda_selection_query_.buffer);
        }
        cuda_selection_query_ = {};
        for (auto& snap : ring_uploaded_) {
            snap = {};
        }
        for (auto& timeline : upload_timelines_) {
            timeline.cuda_semaphore.reset();
            if (context_) {
                context_->destroyExternalSemaphore(timeline.vk_semaphore);
            }
            timeline.vk_semaphore = {};
            timeline.value = 0;
        }
        for (auto& timeline : overlay_upload_timelines_) {
            timeline.cuda_semaphore.reset();
            if (context_) {
                context_->destroyExternalSemaphore(timeline.vk_semaphore);
            }
            timeline.vk_semaphore = {};
            timeline.value = 0;
        }
        if (context_) {
            for (auto& slot : output_slots_) {
                if (slot.image.image != VK_NULL_HANDLE) {
                    context_->imageBarriers().forgetImage(slot.image.image);
                }
                if (slot.depth_image.image != VK_NULL_HANDLE) {
                    context_->imageBarriers().forgetImage(slot.depth_image.image);
                }
                context_->destroyExternalImage(slot.image);
                context_->destroyExternalImage(slot.depth_image);
                slot = {};
            }
            if (compose_) {
                compose_->destroy(context_->device());
            }
        }
        compose_.reset();
        buffers_ = {};
        initialized_ = false;
        context_ = nullptr;
    }

    void VksplatViewportRenderer::detachManagedBuffers() {
        const auto detach = [](_VulkanBuffer& dev) {
            dev.buffer = VK_NULL_HANDLE;
            dev.allocation = VK_NULL_HANDLE;
            dev.allocSize = 0;
            dev.size = 0;
            dev.offset = 0;
        };
        detach(buffers_.xyz_ws.deviceBuffer);
        detach(buffers_.rotations.deviceBuffer);
        detach(buffers_.scales_opacs.deviceBuffer);
        detach(buffers_.sh_coeffs.deviceBuffer);
    }

    void VksplatViewportRenderer::plugRingInputs(const std::size_t ring_slot, const std::size_t num_splats) {
        assert(ring_slot < cuda_inputs_.size());
        auto& slot = cuda_inputs_[ring_slot];
        // All four region views share one VkBuffer / one device allocation; only
        // (offset, size) differs per binding. allocation is left null because the
        // CudaInputSlot owns it.
        const auto plug = [&](_VulkanBuffer& dev, std::size_t region) {
            dev = makeRegionView(slot.buffer, slot.region_offset[region], slot.region_bytes[region]);
        };
        plug(buffers_.xyz_ws.deviceBuffer, InputXyzWs);
        plug(buffers_.rotations.deviceBuffer, InputRotations);
        plug(buffers_.scales_opacs.deviceBuffer, InputScalesOpacs);
        plug(buffers_.sh_coeffs.deviceBuffer, InputShCoeffs);

        // Resize host-shadow vectors so the rasterizer's bookkeeping (which
        // calls byteLength()) keeps matching the device-side payload. The host
        // vectors are not read by the rasterizer; only their size() matters
        // when the renderer cross-checks element counts.
        buffers_.xyz_ws.resize(slot.region_bytes[InputXyzWs] / sizeof(float));
        buffers_.rotations.resize(slot.region_bytes[InputRotations] / sizeof(float));
        buffers_.scales_opacs.resize(slot.region_bytes[InputScalesOpacs] / sizeof(float));
        buffers_.sh_coeffs.resize(slot.region_bytes[InputShCoeffs] / sizeof(float));

        buffers_.num_splats = num_splats;
        buffers_.num_indices = 0;
        buffers_.is_unsorted_1 = true;
    }

    std::expected<VksplatViewportRenderer::OverlayBindingViews, std::string>
    VksplatViewportRenderer::uploadSelectionOverlay(
        VulkanContext& context,
        const lfs::rendering::ViewportRenderRequest& request,
        const std::size_t num_splats,
        const std::size_t ring_slot) {
        if (num_splats == 0) {
            return std::unexpected("VkSplat selection overlay cannot bind an empty model");
        }
        if (!context.externalMemoryInteropEnabled()) {
            return std::unexpected("VkSplat selection overlay requires CUDA/Vulkan external-memory interop");
        }
        assert(ring_slot < cuda_overlays_.size());
        auto& slot = cuda_overlays_[ring_slot];

        const bool selection_enabled =
            !request.overlay.cursor.saturation_preview &&
            hasOverlayTensor(request.overlay.emphasis.mask, num_splats);
        const bool preview_enabled =
            !request.overlay.cursor.saturation_preview &&
            hasOverlayTensor(request.overlay.emphasis.transient_mask.mask, num_splats);
        const bool transform_indices_enabled = hasTransformIndices(request.scene.transform_indices, num_splats);

        const std::size_t mask_region_bytes = alignUp(std::max<std::size_t>(num_splats, 1), 4);
        const std::size_t color_region_bytes =
            lfs::rendering::kSelectionColorTableCount * 4 * sizeof(float);
        const std::size_t transform_region_bytes =
            std::max<std::size_t>(transform_indices_enabled ? num_splats * sizeof(std::int32_t)
                                                            : sizeof(std::int32_t),
                                  sizeof(std::int32_t));
        const std::size_t node_mask_region_bytes =
            alignUp(std::max<std::size_t>(request.overlay.emphasis.emphasized_node_mask.size(), 1), 4);
        const std::size_t overlay_params_region_bytes =
            static_cast<std::size_t>(ParamCount) * 4 * sizeof(float);
        const std::size_t model_transforms_region_bytes =
            modelTransformCount(request.scene.model_transforms) * 16 * sizeof(float);
        std::array<std::size_t, kOverlayRegionCount> region_bytes{};
        region_bytes[OverlaySelectionMask] = mask_region_bytes;
        region_bytes[OverlayPreviewMask] = mask_region_bytes;
        region_bytes[OverlaySelectionColors] = color_region_bytes;
        region_bytes[OverlayTransformIndices] = transform_region_bytes;
        region_bytes[OverlayNodeMask] = node_mask_region_bytes;
        region_bytes[OverlayParams] = overlay_params_region_bytes;
        region_bytes[OverlayModelTransforms] = model_transforms_region_bytes;
        std::array<std::size_t, kOverlayRegionCount> region_offset{};
        const std::size_t total_bytes = layoutRegions(region_bytes, region_offset, kRegionAlignment);

        if (auto ok = ensureCudaInteropBuffer(context,
                                              slot.buffer,
                                              slot.interop,
                                              total_bytes,
                                              "vksplat_selection_overlay",
                                              "selection overlay");
            !ok) {
            return std::unexpected(ok.error());
        }
        slot.region_offset = region_offset;
        slot.region_bytes = region_bytes;

        if (selection_enabled) {
            auto prepared = prepareOverlayMaskTensor(*request.overlay.emphasis.mask, num_splats, "selection");
            if (!prepared) {
                return std::unexpected(prepared.error());
            }
            slot.selection_source = std::move(*prepared);
        } else {
            slot.selection_source = {};
        }
        if (preview_enabled) {
            auto prepared = prepareOverlayMaskTensor(*request.overlay.emphasis.transient_mask.mask,
                                                     num_splats,
                                                     "preview selection");
            if (!prepared) {
                return std::unexpected(prepared.error());
            }
            slot.preview_source = std::move(*prepared);
        } else {
            slot.preview_source = {};
        }
        auto color_table = makeSelectionColorTableTensor(request.overlay);
        if (!color_table) {
            return std::unexpected(color_table.error());
        }
        slot.color_table_source = std::move(*color_table);
        auto transform_indices = prepareTransformIndicesTensor(request.scene.transform_indices, num_splats);
        if (!transform_indices) {
            return std::unexpected(transform_indices.error());
        }
        slot.transform_indices_source = std::move(*transform_indices);
        auto node_mask = makeNodeMaskTensor(request.overlay.emphasis.emphasized_node_mask);
        if (!node_mask) {
            return std::unexpected(node_mask.error());
        }
        slot.node_mask_source = std::move(*node_mask);
        auto overlay_params = makeOverlayParamsTensor(request,
                                                      selection_enabled,
                                                      preview_enabled,
                                                      transform_indices_enabled,
                                                      request.overlay.emphasis.emphasized_node_mask.size());
        if (!overlay_params) {
            return std::unexpected(overlay_params.error());
        }
        slot.overlay_params_source = std::move(*overlay_params);
        auto model_transforms = makeModelTransformsTensor(request.scene.model_transforms);
        if (!model_transforms) {
            return std::unexpected(model_transforms.error());
        }
        slot.model_transforms_source = std::move(*model_transforms);

        cudaStream_t stream = slot.color_table_source.stream();
        if (selection_enabled) {
            stream = slot.selection_source.stream();
        } else if (preview_enabled) {
            stream = slot.preview_source.stream();
        }

        if (selection_enabled &&
            !slot.interop.copyFromTensor(slot.selection_source,
                                         num_splats,
                                         slot.region_offset[OverlaySelectionMask],
                                         stream)) {
            return std::unexpected(std::format("VkSplat selection mask upload failed: {}",
                                               slot.interop.lastError()));
        }
        if (preview_enabled &&
            !slot.interop.copyFromTensor(slot.preview_source,
                                         num_splats,
                                         slot.region_offset[OverlayPreviewMask],
                                         stream)) {
            return std::unexpected(std::format("VkSplat preview selection mask upload failed: {}",
                                               slot.interop.lastError()));
        }
        if (!slot.interop.copyFromTensor(slot.color_table_source,
                                         slot.region_bytes[OverlaySelectionColors],
                                         slot.region_offset[OverlaySelectionColors],
                                         stream)) {
            return std::unexpected(std::format("VkSplat selection color upload failed: {}",
                                               slot.interop.lastError()));
        }
        if (!slot.interop.copyFromTensor(slot.transform_indices_source,
                                         slot.region_bytes[OverlayTransformIndices],
                                         slot.region_offset[OverlayTransformIndices],
                                         stream)) {
            return std::unexpected(std::format("VkSplat transform-index upload failed: {}",
                                               slot.interop.lastError()));
        }
        if (!slot.interop.copyFromTensor(slot.node_mask_source,
                                         slot.node_mask_source.bytes(),
                                         slot.region_offset[OverlayNodeMask],
                                         stream)) {
            return std::unexpected(std::format("VkSplat node-mask upload failed: {}",
                                               slot.interop.lastError()));
        }
        if (!slot.interop.copyFromTensor(slot.overlay_params_source,
                                         slot.region_bytes[OverlayParams],
                                         slot.region_offset[OverlayParams],
                                         stream)) {
            return std::unexpected(std::format("VkSplat overlay parameter upload failed: {}",
                                               slot.interop.lastError()));
        }
        if (!slot.interop.copyFromTensor(slot.model_transforms_source,
                                         slot.region_bytes[OverlayModelTransforms],
                                         slot.region_offset[OverlayModelTransforms],
                                         stream)) {
            return std::unexpected(std::format("VkSplat model-transform upload failed: {}",
                                               slot.interop.lastError()));
        }

        auto& timeline = overlay_upload_timelines_[ring_slot];
        const std::uint64_t signal_value = ++timeline.value;
        if (!timeline.cuda_semaphore.cudaSignal(signal_value, stream)) {
            return std::unexpected(std::format("VkSplat selection overlay upload signal failed: {}",
                                               timeline.cuda_semaphore.lastError()));
        }
        context.addFrameTimelineWait(timeline.vk_semaphore.semaphore,
                                     signal_value,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        const auto view = [&](const std::size_t region) {
            return makeRegionView(slot.buffer, slot.region_offset[region], slot.region_bytes[region]);
        };
        return OverlayBindingViews{
            .selection_mask = view(OverlaySelectionMask),
            .preview_mask = view(OverlayPreviewMask),
            .selection_colors = view(OverlaySelectionColors),
            .transform_indices = view(OverlayTransformIndices),
            .node_mask = view(OverlayNodeMask),
            .overlay_params = view(OverlayParams),
            .model_transforms = view(OverlayModelTransforms),
        };
    }

    std::expected<void, std::string> VksplatViewportRenderer::ensureInitialized(VulkanContext& context) {
        if (context_ != nullptr && context_ != &context) {
            reset();
        }
        context_ = &context;
        if (initialized_) {
            return {};
        }
        try {
            // Submit the splat dispatch chain on the dedicated async-compute queue
            // when the device exposes one (NVIDIA family 2, AMD family 1, etc.). The
            // existing per-frame timeline-semaphore wait that gates the swapchain pass
            // on the rasterizer's output already provides cross-queue ordering, so
            // graphics-queue work (RmlUi, viewport overlays) can overlap the splat
            // compute pass with no additional synchronization.
            const bool use_async_compute = context.hasDedicatedComputeQueue();
            renderer_.initializeExternal(makeVkSplatSpirvPaths(),
                                         context.instance(),
                                         context.physicalDevice(),
                                         context.device(),
                                         use_async_compute ? context.computeQueue()
                                                           : context.graphicsQueue(),
                                         use_async_compute ? context.computeQueueFamily()
                                                           : context.graphicsQueueFamily(),
                                         context.allocator());
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat initialization failed: {}", e.what()));
        }

        // Per-ring-slot upload timeline: a Vulkan-exportable timeline semaphore
        // imported into CUDA so we can signal CUDA-side after the upload's
        // cudaMemcpyAsync and have Vulkan compute wait on it — replacing the
        // per-frame cudaStreamSynchronize that previously blocked the CPU.
        for (auto& timeline : upload_timelines_) {
            if (!context.createExternalTimelineSemaphore(0, timeline.vk_semaphore)) {
                return std::unexpected(std::format(
                    "VkSplat upload timeline semaphore creation failed: {}",
                    context.lastError()));
            }
            const auto handle = context.releaseExternalSemaphoreNativeHandle(timeline.vk_semaphore);
            if (!VulkanContext::externalNativeHandleValid(handle)) {
                context.destroyExternalSemaphore(timeline.vk_semaphore);
                return std::unexpected("VkSplat upload timeline semaphore export failed");
            }
            lfs::rendering::CudaVulkanExternalSemaphoreImport import{};
            import.semaphore_handle = handle;
            import.initial_value = timeline.vk_semaphore.initial_value;
            if (!timeline.cuda_semaphore.init(import)) {
                std::string err = timeline.cuda_semaphore.lastError();
                context.destroyExternalSemaphore(timeline.vk_semaphore);
                return std::unexpected(std::format(
                    "VkSplat upload timeline semaphore CUDA import failed: {}", err));
            }
            timeline.value = 0;
        }
        for (auto& timeline : overlay_upload_timelines_) {
            if (!context.createExternalTimelineSemaphore(0, timeline.vk_semaphore)) {
                return std::unexpected(std::format(
                    "VkSplat selection overlay timeline semaphore creation failed: {}",
                    context.lastError()));
            }
            const auto handle = context.releaseExternalSemaphoreNativeHandle(timeline.vk_semaphore);
            if (!VulkanContext::externalNativeHandleValid(handle)) {
                context.destroyExternalSemaphore(timeline.vk_semaphore);
                return std::unexpected("VkSplat selection overlay timeline semaphore export failed");
            }
            lfs::rendering::CudaVulkanExternalSemaphoreImport import{};
            import.semaphore_handle = handle;
            import.initial_value = timeline.vk_semaphore.initial_value;
            if (!timeline.cuda_semaphore.init(import)) {
                std::string err = timeline.cuda_semaphore.lastError();
                context.destroyExternalSemaphore(timeline.vk_semaphore);
                return std::unexpected(std::format(
                    "VkSplat selection overlay timeline semaphore CUDA import failed: {}", err));
            }
            timeline.value = 0;
        }

        initialized_ = true;
        return {};
    }

    bool VksplatViewportRenderer::inputsResident(const lfs::core::SplatData& splat_data,
                                                 const std::size_t ring_slot) const {
        if (ring_slot >= ring_uploaded_.size())
            return false;
        const auto current = makeModelInputSnapshot(splat_data);
        return ring_uploaded_[ring_slot].valid() && ring_uploaded_[ring_slot] == current;
    }

    std::expected<void, std::string> VksplatViewportRenderer::uploadInputs(
        VulkanContext& context,
        const lfs::core::SplatData& splat_data,
        const std::size_t ring_slot,
        const bool synchronize_upload) {
        const std::size_t n = static_cast<std::size_t>(splat_data.size());
        if (n == 0) {
            return std::unexpected("VkSplat cannot render an empty model");
        }

        if (!context.externalMemoryInteropEnabled()) {
            return std::unexpected("VkSplat input upload requires CUDA/Vulkan external-memory interop");
        }
        assert(ring_slot < cuda_inputs_.size());
        auto& slot = cuda_inputs_[ring_slot];

        auto packed = vksplat::packDeviceInputs(splat_data);
        if (!packed) {
            return std::unexpected(packed.error());
        }

        std::array<std::size_t, kInputRegionCount> region_bytes{};
        region_bytes[InputXyzWs] = static_cast<std::size_t>(packed->xyz_ws.bytes());
        region_bytes[InputRotations] = static_cast<std::size_t>(packed->rotations.bytes());
        region_bytes[InputScalesOpacs] = static_cast<std::size_t>(packed->scales_opacs.bytes());
        region_bytes[InputShCoeffs] = static_cast<std::size_t>(packed->sh_coeffs.bytes());

        // Lay out the four regions back-to-back, padding each to kRegionAlignment
        // so the resulting offsets are valid for VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
        // bindings on every conformant device. Driver-required alignment is at
        // most 256 bytes (often less) — overshooting here costs ≤ 1 KiB per ring.
        std::array<std::size_t, kInputRegionCount> region_offset{};
        const std::size_t total_bytes = layoutRegions(region_bytes, region_offset, kRegionAlignment);

        if (auto ok = ensureCudaInteropBuffer(context,
                                              slot.buffer,
                                              slot.interop,
                                              total_bytes,
                                              "vksplat_inputs",
                                              "input");
            !ok) {
            return std::unexpected(ok.error());
        }

        slot.region_offset = region_offset;
        slot.region_bytes = region_bytes;

        // Single CUDA-imported VkBuffer — four offset-targeted memcpys. Half the
        // setup cost (1 import / 1 mapped pointer) of the prior 4-buffer scheme;
        // the 4× cudaMemcpyAsync calls overlap on the upload stream as before.
        const cudaStream_t stream = packed->xyz_ws.stream();
        const auto copy_region = [&](std::size_t i, const lfs::core::Tensor& t) {
            return slot.interop.copyFromTensor(t, region_bytes[i], region_offset[i], stream);
        };
        if (!copy_region(InputXyzWs, packed->xyz_ws) ||
            !copy_region(InputRotations, packed->rotations) ||
            !copy_region(InputScalesOpacs, packed->scales_opacs) ||
            !copy_region(InputShCoeffs, packed->sh_coeffs)) {
            return std::unexpected(std::format("VkSplat CUDA buffer copy failed: {}",
                                               slot.interop.lastError()));
        }
        if (synchronize_upload) {
            if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
                return std::unexpected(std::format("VkSplat CUDA input upload sync failed: {} ({})",
                                                   cudaGetErrorName(status),
                                                   cudaGetErrorString(status)));
            }
        }

        // Cross-API handoff: signal CUDA-side after the memcpys complete, queue
        // a Vulkan-side wait so the next vksplat compute submit waits on it
        // before reading the buffers. No CPU stall.
        auto& timeline = upload_timelines_[ring_slot];
        const std::uint64_t signal_value = ++timeline.value;
        if (!timeline.cuda_semaphore.cudaSignal(signal_value, stream)) {
            return std::unexpected(std::format("VkSplat CUDA upload signal failed: {}",
                                               timeline.cuda_semaphore.lastError()));
        }
        context.addFrameTimelineWait(timeline.vk_semaphore.semaphore,
                                     signal_value,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);

        ring_uploaded_[ring_slot] = makeModelInputSnapshot(splat_data);
        return {};
    }

    std::expected<void, std::string> VksplatViewportRenderer::ensureOutputImages(
        VulkanContext& context,
        const glm::ivec2 size,
        const OutputSlot output_slot) {
        auto& slot = output_slots_[outputSlotIndex(output_slot)];
        if (slot.image.image != VK_NULL_HANDLE && slot.depth_image.image != VK_NULL_HANDLE &&
            slot.size == size) {
            return {};
        }
        if (slot.image.image != VK_NULL_HANDLE) {
            context.imageBarriers().forgetImage(slot.image.image);
        }
        if (slot.depth_image.image != VK_NULL_HANDLE) {
            context.imageBarriers().forgetImage(slot.depth_image.image);
        }
        context.destroyExternalImage(slot.image);
        context.destroyExternalImage(slot.depth_image);
        slot.size = {0, 0};
        slot.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        slot.depth_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        const VkExtent2D extent{
            .width = static_cast<std::uint32_t>(size.x),
            .height = static_cast<std::uint32_t>(size.y),
        };
        if (!context.createExternalImage(extent, VK_FORMAT_R8G8B8A8_UNORM, slot.image)) {
            return std::unexpected(context.lastError());
        }
        if (!context.createExternalImage(extent, VK_FORMAT_R32_SFLOAT, slot.depth_image)) {
            const std::string error = context.lastError();
            context.destroyExternalImage(slot.image);
            return std::unexpected(error);
        }
        context.imageBarriers().registerImage(slot.image.image,
                                              VK_IMAGE_ASPECT_COLOR_BIT,
                                              VK_IMAGE_LAYOUT_UNDEFINED,
                                              /*external=*/true);
        context.imageBarriers().registerImage(slot.depth_image.image,
                                              VK_IMAGE_ASPECT_COLOR_BIT,
                                              VK_IMAGE_LAYOUT_UNDEFINED,
                                              /*external=*/true);
        slot.size = size;
        ++slot.generation;
        return {};
    }

    std::expected<void, std::string> VksplatViewportRenderer::ensureComposePipeline(VulkanContext& context) {
        if (compose_ && compose_->pipeline != VK_NULL_HANDLE) {
            return {};
        }
        compose_ = std::make_unique<ComposePipeline>();
        VkDevice device = context.device();
        VkResult result = VK_SUCCESS;

        VkShaderModuleCreateInfo shader_info{};
        shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_info.codeSize = sizeof(viewport_shaders::kVkSplatComposeCompSpv);
        shader_info.pCode = viewport_shaders::kVkSplatComposeCompSpv;
        result = vkCreateShaderModule(device, &shader_info, nullptr, &compose_->shader_module);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreateShaderModule(VkSplat compose)", result));
        }

        std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
        bindings[0].binding = 0;
        bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[0].descriptorCount = 1;
        bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[1].binding = 1;
        bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[1].descriptorCount = 1;
        bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[2].binding = 2;
        bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[2].descriptorCount = 1;
        bindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings[3].binding = 3;
        bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[3].descriptorCount = 1;
        bindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

        VkDescriptorSetLayoutCreateInfo layout_info{};
        layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layout_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
        layout_info.pBindings = bindings.data();
        result = vkCreateDescriptorSetLayout(device, &layout_info, nullptr, &compose_->descriptor_set_layout);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreateDescriptorSetLayout(VkSplat compose)", result));
        }

        VkPushConstantRange push_range{};
        push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        push_range.size = sizeof(ComposePushConstants);
        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &compose_->descriptor_set_layout;
        pipeline_layout_info.pushConstantRangeCount = 1;
        pipeline_layout_info.pPushConstantRanges = &push_range;
        result = vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &compose_->pipeline_layout);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreatePipelineLayout(VkSplat compose)", result));
        }

        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = compose_->shader_module;
        stage.pName = "main";
        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = stage;
        pipeline_info.layout = compose_->pipeline_layout;
        result = vkCreateComputePipelines(device, context.pipelineCache(), 1, &pipeline_info, nullptr, &compose_->pipeline);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreateComputePipelines(VkSplat compose)", result));
        }
        return {};
    }

    std::expected<void, std::string> VksplatViewportRenderer::composePixelState(
        VulkanContext& context,
        VkCommandBuffer cmd,
        const VulkanGSRendererUniforms& uniforms,
        const glm::vec3& background,
        const OutputSlot output_slot,
        const bool transparent_background) {
        if (auto ok = ensureComposePipeline(context); !ok) {
            return ok;
        }
        auto& output = output_slots_[outputSlotIndex(output_slot)];

        const bool has_pixel_state = buffers_.num_indices > 0 &&
                                     buffers_.pixel_state.deviceBuffer.buffer != VK_NULL_HANDLE &&
                                     buffers_.pixel_state.deviceBuffer.size > 0 &&
                                     buffers_.pixel_depth.deviceBuffer.buffer != VK_NULL_HANDLE &&
                                     buffers_.pixel_depth.deviceBuffer.size > 0;
        if (!has_pixel_state) {
            context.imageBarriers().transitionImage(cmd,
                                                    output.image.image,
                                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            context.imageBarriers().transitionImage(cmd,
                                                    output.depth_image.image,
                                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                                    VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearColorValue clear{{background.r, background.g, background.b, 1.0f}};
            VkClearColorValue depth_clear{{1.0e10f, 0.0f, 0.0f, 0.0f}};
            VkImageSubresourceRange range{};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.levelCount = 1;
            range.layerCount = 1;
            vkCmdClearColorImage(cmd,
                                 output.image.image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &clear,
                                 1,
                                 &range);
            vkCmdClearColorImage(cmd,
                                 output.depth_image.image,
                                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                 &depth_clear,
                                 1,
                                 &range);
            context.imageBarriers().transitionImage(cmd,
                                                    output.image.image,
                                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            context.imageBarriers().transitionImage(cmd,
                                                    output.depth_image.image,
                                                    VK_IMAGE_ASPECT_COLOR_BIT,
                                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            output.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            output.depth_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            ++output.generation;
            return {};
        }

        VkDescriptorBufferInfo pixel_info{};
        pixel_info.buffer = buffers_.pixel_state.deviceBuffer.buffer;
        pixel_info.range = buffers_.pixel_state.deviceBuffer.size;
        VkDescriptorBufferInfo depth_info{};
        depth_info.buffer = buffers_.pixel_depth.deviceBuffer.buffer;
        depth_info.range = buffers_.pixel_depth.deviceBuffer.size;
        VkDescriptorImageInfo image_info{};
        image_info.imageView = output.image.view;
        image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkDescriptorImageInfo depth_image_info{};
        depth_image_info.imageView = output.depth_image.view;
        depth_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        std::array<VkWriteDescriptorSet, 4> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &pixel_info;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].descriptorCount = 1;
        writes[1].pBufferInfo = &depth_info;
        writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[2].dstBinding = 2;
        writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[2].descriptorCount = 1;
        writes[2].pImageInfo = &image_info;
        writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[3].dstBinding = 3;
        writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        writes[3].descriptorCount = 1;
        writes[3].pImageInfo = &depth_image_info;

        std::array<VkBufferMemoryBarrier2, 2> pixel_barriers{};
        pixel_barriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
        pixel_barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        pixel_barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        pixel_barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        pixel_barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        pixel_barriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pixel_barriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pixel_barriers[0].buffer = buffers_.pixel_state.deviceBuffer.buffer;
        pixel_barriers[0].size = buffers_.pixel_state.deviceBuffer.size;
        pixel_barriers[1] = pixel_barriers[0];
        pixel_barriers[1].buffer = buffers_.pixel_depth.deviceBuffer.buffer;
        pixel_barriers[1].size = buffers_.pixel_depth.deviceBuffer.size;
        VkDependencyInfo pixel_dep{};
        pixel_dep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        pixel_dep.bufferMemoryBarrierCount = static_cast<std::uint32_t>(pixel_barriers.size());
        pixel_dep.pBufferMemoryBarriers = pixel_barriers.data();
        vkCmdPipelineBarrier2(cmd, &pixel_dep);
        context.imageBarriers().transitionImage(cmd,
                                                output.image.image,
                                                VK_IMAGE_ASPECT_COLOR_BIT,
                                                VK_IMAGE_LAYOUT_GENERAL);
        context.imageBarriers().transitionImage(cmd,
                                                output.depth_image.image,
                                                VK_IMAGE_ASPECT_COLOR_BIT,
                                                VK_IMAGE_LAYOUT_GENERAL);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compose_->pipeline);
        context.vkCmdPushDescriptorSet()(cmd,
                                         VK_PIPELINE_BIND_POINT_COMPUTE,
                                         compose_->pipeline_layout,
                                         0,
                                         static_cast<std::uint32_t>(writes.size()),
                                         writes.data());
        ComposePushConstants push{
            .width = uniforms.image_width,
            .height = uniforms.image_height,
            .transparent_background = transparent_background ? 1u : 0u,
            .background = glm::vec4(background, 1.0f),
        };
        vkCmdPushConstants(cmd,
                           compose_->pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(push),
                           &push);
        vkCmdDispatch(cmd,
                      _CEIL_DIV(uniforms.image_width, 16),
                      _CEIL_DIV(uniforms.image_height, 16),
                      1);
        context.imageBarriers().transitionImage(cmd,
                                                output.image.image,
                                                VK_IMAGE_ASPECT_COLOR_BIT,
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        context.imageBarriers().transitionImage(cmd,
                                                output.depth_image.image,
                                                VK_IMAGE_ASPECT_COLOR_BIT,
                                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        output.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        output.depth_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        ++output.generation;
        return {};
    }

    std::expected<std::shared_ptr<lfs::core::Tensor>, std::string>
    VksplatViewportRenderer::readOutputImage(VulkanContext& context, const OutputSlot output_slot) const {
        if (!context_) {
            return std::unexpected("VkSplat output readback requested before renderer initialization");
        }
        if (&context != context_) {
            return std::unexpected("VkSplat output readback received a different Vulkan context");
        }

        const auto& output = output_slots_[outputSlotIndex(output_slot)];
        if (output.image.image == VK_NULL_HANDLE ||
            output.size.x <= 0 ||
            output.size.y <= 0) {
            return std::unexpected("VkSplat output readback requested for an empty output slot");
        }
        if (output.image.format != VK_FORMAT_R8G8B8A8_UNORM) {
            return std::unexpected("VkSplat output readback only supports RGBA8 output images");
        }

        if (!context.waitForSubmittedFrames()) {
            return std::unexpected(context.lastError());
        }

        const VkDevice device = context.device();
        const VkDeviceSize byte_count =
            static_cast<VkDeviceSize>(output.size.x) *
            static_cast<VkDeviceSize>(output.size.y) *
            static_cast<VkDeviceSize>(4);
        if (byte_count == 0) {
            return std::unexpected("VkSplat output readback has zero bytes");
        }

        ScopedStagingBuffer staging{.allocator = context.allocator()};
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = byte_count;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        alloc_info.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
            VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VkResult result = vmaCreateBuffer(
            staging.allocator,
            &buffer_info,
            &alloc_info,
            &staging.buffer,
            &staging.allocation,
            &staging.allocation_info);
        if (result != VK_SUCCESS || staging.buffer == VK_NULL_HANDLE) {
            return std::unexpected(vkError("vmaCreateBuffer(VkSplat readback)", result));
        }
        if (staging.allocation_info.pMappedData == nullptr) {
            return std::unexpected("VkSplat readback staging buffer is not host-mapped");
        }

        ScopedCommandPool command_pool{.device = device};
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        pool_info.queueFamilyIndex = context.graphicsQueueFamily();
        result = vkCreateCommandPool(device, &pool_info, nullptr, &command_pool.pool);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreateCommandPool(VkSplat readback)", result));
        }

        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo command_info{};
        command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_info.commandPool = command_pool.pool;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        result = vkAllocateCommandBuffers(device, &command_info, &command_buffer);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkAllocateCommandBuffers(VkSplat readback)", result));
        }

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vkBeginCommandBuffer(command_buffer, &begin_info);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkBeginCommandBuffer(VkSplat readback)", result));
        }

        const VkImageLayout restore_layout =
            output.layout != VK_IMAGE_LAYOUT_UNDEFINED
                ? output.layout
                : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        context.imageBarriers().transitionImage(
            command_buffer,
            output.image.image,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        VkBufferImageCopy copy_region{};
        copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.imageSubresource.layerCount = 1;
        copy_region.imageExtent = {
            static_cast<std::uint32_t>(output.size.x),
            static_cast<std::uint32_t>(output.size.y),
            1};
        vkCmdCopyImageToBuffer(command_buffer,
                               output.image.image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging.buffer,
                               1,
                               &copy_region);

        context.imageBarriers().transitionImage(
            command_buffer,
            output.image.image,
            VK_IMAGE_ASPECT_COLOR_BIT,
            restore_layout);

        result = vkEndCommandBuffer(command_buffer);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkEndCommandBuffer(VkSplat readback)", result));
        }

        ScopedFence fence{.device = device};
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        result = vkCreateFence(device, &fence_info, nullptr, &fence.fence);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreateFence(VkSplat readback)", result));
        }

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        result = vkQueueSubmit(context.graphicsQueue(), 1, &submit_info, fence.fence);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkQueueSubmit(VkSplat readback)", result));
        }
        result = vkWaitForFences(device, 1, &fence.fence, VK_TRUE, UINT64_MAX);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkWaitForFences(VkSplat readback)", result));
        }

        result = vmaInvalidateAllocation(staging.allocator, staging.allocation, 0, byte_count);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vmaInvalidateAllocation(VkSplat readback)", result));
        }

        const auto* const rgba =
            static_cast<const std::uint8_t*>(staging.allocation_info.pMappedData);
        const int width = output.size.x;
        const int height = output.size.y;
        const std::size_t pixel_count =
            static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
        std::vector<float> chw(pixel_count * 3u);
        for (std::size_t i = 0; i < pixel_count; ++i) {
            const std::size_t src = i * 4u;
            chw[i] = static_cast<float>(rgba[src]) / 255.0f;
            chw[pixel_count + i] = static_cast<float>(rgba[src + 1u]) / 255.0f;
            chw[pixel_count * 2u + i] = static_cast<float>(rgba[src + 2u]) / 255.0f;
        }

        auto tensor = lfs::core::Tensor::from_vector(
            chw,
            {3u, static_cast<std::size_t>(height), static_cast<std::size_t>(width)},
            lfs::core::Device::CPU);
        return std::make_shared<lfs::core::Tensor>(std::move(tensor));
    }

    std::expected<lfs::core::Tensor, std::string> VksplatViewportRenderer::buildSelectionMask(
        VulkanContext& context,
        const lfs::core::SplatData& splat_data,
        const SelectionMaskRequest& request,
        const bool force_input_upload) {
        const glm::ivec2 size = request.frame_view.size;
        if (size.x <= 0 || size.y <= 0) {
            return std::unexpected("VkSplat selection query received an invalid viewport size");
        }
        if (request.frame_view.orthographic) {
            return std::unexpected("VkSplat selection query supports pinhole cameras, not orthographic cameras");
        }
        if (request.equirectangular) {
            return std::unexpected("VkSplat selection query supports pinhole cameras, not equirectangular cameras");
        }
        if (request.primitives.empty()) {
            return std::unexpected("VkSplat selection query requires at least one primitive");
        }
        const std::size_t num_splats = static_cast<std::size_t>(splat_data.size());
        if (num_splats == 0) {
            return std::unexpected("VkSplat selection query cannot run on an empty model");
        }
        if (num_splats > std::numeric_limits<std::uint32_t>::max() ||
            request.primitives.size() > std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected("VkSplat selection query exceeds shader dispatch limits");
        }
        if (!context.externalMemoryInteropEnabled()) {
            return std::unexpected("VkSplat selection query requires CUDA/Vulkan external-memory interop");
        }

        if (auto ok = ensureInitialized(context); !ok) {
            return std::unexpected(ok.error());
        }

        const std::size_t ring_slot = context.currentFrameSlot() % kInputRingSize;
        assert(context.framesInFlight() == kInputRingSize);

        if (force_input_upload || !inputsResident(splat_data, ring_slot)) {
            if (auto ok = uploadInputs(context, splat_data, ring_slot, true); !ok) {
                return std::unexpected(ok.error());
            }
            renderer_.resetNumIndicesEstimate();
        }
        plugRingInputs(ring_slot, num_splats);

        auto& slot = cuda_selection_query_;
        const bool transform_indices_enabled = hasTransformIndices(request.scene.transform_indices, num_splats);
        const bool node_visibility_enabled = !request.scene.node_visibility_mask.empty();

        const std::size_t output_region_bytes = alignUp(std::max<std::size_t>(num_splats, 1), 4);
        const std::size_t transform_region_bytes =
            std::max<std::size_t>(transform_indices_enabled ? num_splats * sizeof(std::int32_t)
                                                            : sizeof(std::int32_t),
                                  sizeof(std::int32_t));
        const std::size_t node_mask_region_bytes =
            alignUp(std::max<std::size_t>(request.scene.node_visibility_mask.size(), 1), 4);
        const std::size_t primitive_region_bytes = request.primitives.size() * 4 * sizeof(float);
        const std::size_t model_transforms_region_bytes =
            modelTransformCount(request.scene.model_transforms) * 16 * sizeof(float);
        std::array<std::size_t, kSelectionQueryRegionCount> region_bytes{};
        region_bytes[SelectionQueryOutput] = output_region_bytes;
        region_bytes[SelectionQueryTransformIndices] = transform_region_bytes;
        region_bytes[SelectionQueryNodeMask] = node_mask_region_bytes;
        region_bytes[SelectionQueryPrimitives] = primitive_region_bytes;
        region_bytes[SelectionQueryModelTransforms] = model_transforms_region_bytes;
        std::array<std::size_t, kSelectionQueryRegionCount> region_offset{};
        const std::size_t total_bytes = layoutRegions(region_bytes, region_offset, kRegionAlignment);

        if (auto ok = ensureCudaInteropBuffer(context,
                                              slot.buffer,
                                              slot.interop,
                                              total_bytes,
                                              "vksplat_selection_query",
                                              "selection query");
            !ok) {
            return std::unexpected(ok.error());
        }
        slot.region_offset = region_offset;
        slot.region_bytes = region_bytes;

        auto transform_indices = prepareTransformIndicesTensor(request.scene.transform_indices, num_splats);
        if (!transform_indices) {
            return std::unexpected(transform_indices.error());
        }
        slot.transform_indices_source = std::move(*transform_indices);
        auto node_mask = makeNodeMaskTensor(request.scene.node_visibility_mask);
        if (!node_mask) {
            return std::unexpected(node_mask.error());
        }
        slot.node_mask_source = std::move(*node_mask);
        auto primitive_source = makeSelectionPrimitiveTensor(request.primitives);
        if (!primitive_source) {
            return std::unexpected(primitive_source.error());
        }
        slot.primitive_source = std::move(*primitive_source);
        auto model_transforms = makeModelTransformsTensor(request.scene.model_transforms);
        if (!model_transforms) {
            return std::unexpected(model_transforms.error());
        }
        slot.model_transforms_source = std::move(*model_transforms);

        const cudaStream_t stream = slot.primitive_source.stream();
        if (!slot.interop.copyFromTensor(slot.transform_indices_source,
                                         slot.region_bytes[SelectionQueryTransformIndices],
                                         slot.region_offset[SelectionQueryTransformIndices],
                                         stream)) {
            return std::unexpected(std::format("VkSplat selection transform-index upload failed: {}",
                                               slot.interop.lastError()));
        }
        if (!slot.interop.copyFromTensor(slot.node_mask_source,
                                         slot.node_mask_source.bytes(),
                                         slot.region_offset[SelectionQueryNodeMask],
                                         stream)) {
            return std::unexpected(std::format("VkSplat selection node-mask upload failed: {}",
                                               slot.interop.lastError()));
        }
        if (!slot.interop.copyFromTensor(slot.primitive_source,
                                         slot.region_bytes[SelectionQueryPrimitives],
                                         slot.region_offset[SelectionQueryPrimitives],
                                         stream)) {
            return std::unexpected(std::format("VkSplat selection primitive upload failed: {}",
                                               slot.interop.lastError()));
        }
        if (!slot.interop.copyFromTensor(slot.model_transforms_source,
                                         slot.region_bytes[SelectionQueryModelTransforms],
                                         slot.region_offset[SelectionQueryModelTransforms],
                                         stream)) {
            return std::unexpected(std::format("VkSplat selection model-transform upload failed: {}",
                                               slot.interop.lastError()));
        }
        auto* const output_ptr =
            static_cast<std::uint8_t*>(slot.interop.devicePointer()) + slot.region_offset[SelectionQueryOutput];
        if (const cudaError_t status = cudaMemsetAsync(output_ptr, 0, output_region_bytes, stream);
            status != cudaSuccess) {
            return std::unexpected(std::format("VkSplat selection output clear failed: {} ({})",
                                               cudaGetErrorName(status),
                                               cudaGetErrorString(status)));
        }
        if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
            return std::unexpected(std::format("VkSplat selection upload sync failed: {} ({})",
                                               cudaGetErrorName(status),
                                               cudaGetErrorString(status)));
        }

        const auto view = [&](const std::size_t region) {
            return makeRegionView(slot.buffer, slot.region_offset[region], slot.region_bytes[region]);
        };

        VulkanGSRendererUniforms camera_uniforms{};
        populateVksplatCameraUniforms(camera_uniforms,
                                      request.frame_view,
                                      request.scene,
                                      0,
                                      num_splats);
        VulkanGSSelectionMaskUniforms selection_uniforms{};
        selection_uniforms.num_splats = static_cast<std::uint32_t>(num_splats);
        selection_uniforms.primitive_count = static_cast<std::uint32_t>(request.primitives.size());
        selection_uniforms.mode = static_cast<std::uint32_t>(request.shape);
        selection_uniforms.transform_indices_enabled = transform_indices_enabled ? 1u : 0u;
        selection_uniforms.node_visibility_enabled = node_visibility_enabled ? 1u : 0u;
        selection_uniforms.node_visibility_count =
            static_cast<std::uint32_t>(request.scene.node_visibility_mask.size());
        selection_uniforms.num_model_transforms =
            static_cast<std::uint32_t>(modelTransformCount(request.scene.model_transforms));
        selection_uniforms.image_height = camera_uniforms.image_height;
        selection_uniforms.image_width = camera_uniforms.image_width;
        selection_uniforms.camera_model = request.gut
                                              ? (kVkSplatProjectionModeGut << kVkSplatProjectionModeShift)
                                              : 0u;
        selection_uniforms.fx = camera_uniforms.fx;
        selection_uniforms.fy = camera_uniforms.fy;
        selection_uniforms.cx = camera_uniforms.cx;
        selection_uniforms.cy = camera_uniforms.cy;
        for (std::size_t i = 0; i < 4; ++i) {
            selection_uniforms.dist_coeffs[i] = camera_uniforms.dist_coeffs[i];
        }
        for (std::size_t i = 0; i < 16; ++i) {
            selection_uniforms.world_view_transform[i] = camera_uniforms.world_view_transform[i];
        }

        try {
            auto batch = DeviceGuard(&renderer_);
            renderer_.executeSelectionMask(selection_uniforms,
                                           buffers_,
                                           view(SelectionQueryTransformIndices),
                                           view(SelectionQueryNodeMask),
                                           view(SelectionQueryPrimitives),
                                           view(SelectionQueryModelTransforms),
                                           view(SelectionQueryOutput));
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat selection query failed: {}", e.what()));
        }

        slot.output_tensor = Tensor::empty({num_splats}, Device::CUDA, DataType::Bool);
        if (!slot.interop.copyToTensor(slot.output_tensor,
                                       num_splats,
                                       slot.region_offset[SelectionQueryOutput],
                                       slot.output_tensor.stream())) {
            return std::unexpected(std::format("VkSplat selection output download failed: {}",
                                               slot.interop.lastError()));
        }
        if (const cudaError_t status = cudaStreamSynchronize(slot.output_tensor.stream());
            status != cudaSuccess) {
            return std::unexpected(std::format("VkSplat selection output sync failed: {} ({})",
                                               cudaGetErrorName(status),
                                               cudaGetErrorString(status)));
        }

        return slot.output_tensor;
    }

    std::expected<VksplatViewportRenderer::RenderResult, std::string> VksplatViewportRenderer::render(
        VulkanContext& context,
        const lfs::core::SplatData& splat_data,
        const lfs::rendering::ViewportRenderRequest& request,
        const bool force_input_upload,
        const OutputSlot output_slot) {
        const glm::ivec2 size = request.frame_view.size;
        if (size.x <= 0 || size.y <= 0) {
            return std::unexpected("VkSplat received an invalid viewport size");
        }
        if (request.frame_view.orthographic) {
            return std::unexpected("VkSplat forward path supports pinhole cameras, not orthographic cameras");
        }
        if (request.equirectangular) {
            return std::unexpected("VkSplat forward path supports pinhole cameras, not equirectangular cameras");
        }
        if (!context.externalMemoryInteropEnabled()) {
            return std::unexpected("VkSplat forward path requires CUDA/Vulkan external-memory interop");
        }

        const int active_sh_degree = std::clamp(request.sh_degree, 0, std::min(3, splat_data.get_max_sh_degree()));
        if (auto ok = ensureInitialized(context); !ok) {
            return std::unexpected(ok.error());
        }

        const std::size_t ring_slot = context.currentFrameSlot() % kInputRingSize;
        assert(context.framesInFlight() == kInputRingSize);

        if (force_input_upload || !inputsResident(splat_data, ring_slot)) {
            if (auto ok = uploadInputs(context, splat_data, ring_slot); !ok) {
                return std::unexpected(ok.error());
            }
            // Drop the deferred-readback high-water-mark whenever the model identity
            // changes — a fresh model can have a wildly different num_indices range,
            // and stale estimates risk under-sizing the sort buffers (or wasting VRAM
            // if oversized). The next frame re-seeds via heuristic and grows from there.
            renderer_.resetNumIndicesEstimate();
        }
        plugRingInputs(ring_slot, static_cast<std::size_t>(splat_data.size()));
        auto overlay_bindings = uploadSelectionOverlay(
            context, request, static_cast<std::size_t>(splat_data.size()), ring_slot);
        if (!overlay_bindings) {
            return std::unexpected(overlay_bindings.error());
        }
        if (auto ok = ensureOutputImages(context, size, output_slot); !ok) {
            return std::unexpected(ok.error());
        }

        VulkanGSRendererUniforms uniforms{};
        uniforms.image_width = static_cast<std::uint32_t>(size.x);
        uniforms.image_height = static_cast<std::uint32_t>(size.y);
        uniforms.grid_width = _CEIL_DIV(uniforms.image_width, TILE_WIDTH);
        uniforms.grid_height = _CEIL_DIV(uniforms.image_height, TILE_HEIGHT);
        uniforms.num_splats = static_cast<std::uint32_t>(buffers_.num_splats);
        uniforms.active_sh = static_cast<std::uint32_t>(active_sh_degree);
        uniforms.step = static_cast<std::uint32_t>(modelTransformCount(request.scene.model_transforms));
        uniforms.camera_model = request.gut
                                    ? (kVkSplatProjectionModeGut << kVkSplatProjectionModeShift)
                                    : 0u;

        if (request.frame_view.intrinsics_override) {
            const auto& intrinsics = *request.frame_view.intrinsics_override;
            uniforms.fx = intrinsics.focal_x;
            uniforms.fy = intrinsics.focal_y;
            uniforms.cx = intrinsics.center_x;
            uniforms.cy = intrinsics.center_y;
        } else {
            const auto [fx, fy] = lfs::rendering::computePixelFocalLengths(
                size, request.frame_view.focal_length_mm);
            uniforms.fx = fx;
            uniforms.fy = fy;
            uniforms.cx = static_cast<float>(size.x) * 0.5f;
            uniforms.cy = static_cast<float>(size.y) * 0.5f;
        }

        const glm::mat3 camera_to_world =
            lfs::rendering::dataCameraToWorldFromVisualizerRotation(request.frame_view.rotation);
        const glm::mat3 world_to_camera = glm::transpose(camera_to_world);
        const glm::vec3 translation = -world_to_camera * request.frame_view.translation;

        std::array<float, 16> row_major_view{};
        row_major_view[15] = 1.0f;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                row_major_view[static_cast<std::size_t>(row * 4 + col)] = world_to_camera[col][row];
            }
        }
        row_major_view[3] = translation.x;
        row_major_view[7] = translation.y;
        row_major_view[11] = translation.z;
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                uniforms.world_view_transform[4 * row + col] =
                    row_major_view[static_cast<std::size_t>(4 * col + row)];
            }
        }

        std::expected<void, std::string> compose_status;
        try {
            auto batch = DeviceGuard(&renderer_);
            renderer_.executeProjectionForward(uniforms,
                                               buffers_,
                                               overlay_bindings->transform_indices,
                                               overlay_bindings->node_mask,
                                               overlay_bindings->overlay_params,
                                               overlay_bindings->model_transforms,
                                               0,
                                               request.gut);
            renderer_.executeCalculateIndexBufferOffset(buffers_);
            if (buffers_.num_indices > 0) {
                renderer_.executeGenerateKeys(uniforms, buffers_);
                renderer_.executeSort(uniforms, buffers_, -1);
                renderer_.executeComputeTileRanges(uniforms, buffers_);
                renderer_.executeRasterizeForward(uniforms,
                                                  buffers_,
                                                  overlay_bindings->selection_mask,
                                                  overlay_bindings->preview_mask,
                                                  overlay_bindings->selection_colors,
                                                  buffers_.overlay_flags.deviceBuffer,
                                                  overlay_bindings->overlay_params,
                                                  overlay_bindings->transform_indices,
                                                  overlay_bindings->model_transforms,
                                                  request.gut);
            }
            // Record compose into the rasterizer's batch so the entire frame
            // submits and waits exactly once instead of fence-blocking twice.
            compose_status = composePixelState(
                context,
                renderer_.activeCommandBuffer(),
                uniforms,
                request.frame_view.background_color,
                output_slot,
                request.transparent_background);
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat forward pass failed: {}", e.what()));
        }
        if (!compose_status) {
            return std::unexpected(compose_status.error());
        }

        const auto& output = output_slots_[outputSlotIndex(output_slot)];
        return RenderResult{
            .image = output.image.image,
            .image_view = output.image.view,
            .image_layout = output.layout,
            .generation = output.generation,
            .depth_image = output.depth_image.image,
            .depth_image_view = output.depth_image.view,
            .depth_image_layout = output.depth_layout,
            .depth_generation = output.generation,
            .size = size,
            .flip_y = false,
        };
    }

} // namespace lfs::vis

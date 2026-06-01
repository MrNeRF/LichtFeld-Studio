/* SPDX-FileCopyrightText: 2025 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vksplat_viewport_renderer.hpp"

#include "core/cuda/memory_arena.hpp"
#include "core/cuda/sh_layout.cuh"
#include "core/executable_path.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "core/tensor.hpp"
#include "core/tensor/internal/cuda_stream_context.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "rendering/coordinate_conventions.hpp"
#include "viewport/vksplat_compose.comp.spv.h"
#include "vksplat_input_packer.hpp"
#include "vulkan_external_tensor.hpp"
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
#include <optional>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace lfs::vis {
    namespace {
        using lfs::core::DataType;
        using lfs::core::Device;
        using lfs::core::Tensor;

        constexpr std::uint32_t kVkSplatCameraModelPinhole = 0u;
        constexpr std::uint32_t kVkSplatCameraModelOrthographic = 1u;
        constexpr std::uint32_t kVkSplatCameraModelEquirectangular = 3u;
        constexpr std::uint32_t kVkSplatProjectionModeShift = 8u;
        constexpr std::uint32_t kVkSplatProjectionModeGut = 1u;

        template <typename Fn>
        class ScopeExit final {
        public:
            explicit ScopeExit(Fn fn) : fn_(std::move(fn)) {}
            ScopeExit(const ScopeExit&) = delete;
            ScopeExit& operator=(const ScopeExit&) = delete;

            ScopeExit(ScopeExit&& other) noexcept
                : fn_(std::move(other.fn_)),
                  active_(std::exchange(other.active_, false)) {}

            ~ScopeExit() {
                if (active_) {
                    fn_();
                }
            }

        private:
            Fn fn_;
            bool active_ = true;
        };

        class RasterizerArenaRenderGuard final {
        public:
            RasterizerArenaRenderGuard() {
                arena_ = &lfs::core::GlobalArenaManager::instance().get_arena();
                arena_->set_rendering_active(true);
                render_pending_ = true;
                try {
                    auto frame_id = arena_->try_begin_frame(true);
                    if (!frame_id) {
                        throw std::runtime_error("rasterizer arena is busy");
                    }
                    frame_id_ = *frame_id;
                    frame_active_ = true;
                    arena_->set_rendering_active(false);
                    render_pending_ = false;
                } catch (...) {
                    if (render_pending_) {
                        arena_->set_rendering_active(false);
                    }
                    arena_ = nullptr;
                    throw;
                }
            }

            RasterizerArenaRenderGuard(const RasterizerArenaRenderGuard&) = delete;
            RasterizerArenaRenderGuard& operator=(const RasterizerArenaRenderGuard&) = delete;
            RasterizerArenaRenderGuard(RasterizerArenaRenderGuard&&) = delete;
            RasterizerArenaRenderGuard& operator=(RasterizerArenaRenderGuard&&) = delete;

            ~RasterizerArenaRenderGuard() {
                if (!arena_) {
                    return;
                }
                if (frame_active_) {
                    arena_->end_frame(frame_id_, true);
                }
            }

        private:
            lfs::core::RasterizerMemoryArena* arena_ = nullptr;
            std::uint64_t frame_id_ = 0;
            bool frame_active_ = false;
            bool render_pending_ = false;
        };

        [[nodiscard]] std::string vkError(const char* const operation, const VkResult result) {
            return std::format("{} failed: {}", operation, vkResultToString(result));
        }

        [[nodiscard]] std::uint32_t vksplatBaseCameraModel(
            const lfs::rendering::FrameView& frame_view,
            const bool equirectangular) {
            if (equirectangular) {
                return kVkSplatCameraModelEquirectangular;
            }
            return frame_view.orthographic ? kVkSplatCameraModelOrthographic
                                           : kVkSplatCameraModelPinhole;
        }

        [[nodiscard]] const char* outputSlotDiagnosticName(const VksplatViewportRenderer::OutputSlot slot) {
            switch (slot) {
            case VksplatViewportRenderer::OutputSlot::Main:
                return "main";
            case VksplatViewportRenderer::OutputSlot::SplitLeft:
                return "split_left";
            case VksplatViewportRenderer::OutputSlot::SplitRight:
                return "split_right";
            case VksplatViewportRenderer::OutputSlot::Preview:
                return "preview";
            }
            return "unknown";
        }

        [[nodiscard]] std::uint32_t packedVksplatCameraModel(
            const lfs::rendering::FrameView& frame_view,
            const bool equirectangular,
            const bool gut) {
            return vksplatBaseCameraModel(frame_view, equirectangular) |
                   (gut ? (kVkSplatProjectionModeGut << kVkSplatProjectionModeShift) : 0u);
        }

        [[nodiscard]] std::uint32_t shLayoutSlotsForDegree(const int layout_sh_degree) {
            if (layout_sh_degree <= 0) {
                return 0;
            }
            return lfs::core::sh_float4_slots_for_rest(
                lfs::core::sh_rest_coefficients_for_degree(layout_sh_degree));
        }

        [[nodiscard]] std::uint32_t renderShNLayoutSlots(
            const int active_sh_degree,
            const int input_layout_sh_degree) {
            if (active_sh_degree <= 0) {
                return 0;
            }
            return shLayoutSlotsForDegree(input_layout_sh_degree);
        }

        [[nodiscard]] int effectiveRenderShDegree(
            const lfs::core::SplatData& splat_data,
            const int requested_sh_degree) {
            const int max_model_degree = std::min(3, splat_data.get_max_sh_degree());
            const int active_model_degree = std::clamp(
                splat_data.get_active_sh_degree(),
                0,
                max_model_degree);
            return std::clamp(requested_sh_degree, 0, active_model_degree);
        }

        [[nodiscard]] std::filesystem::path resolveVkSplatSpirvRoot() {
            constexpr std::string_view probe_file = "generated/projection_forward.spv";
            std::vector<std::filesystem::path> search_paths;

            search_paths.push_back(lfs::core::getResourceBaseDir() / "shaders" / "vulkan_rasterizer");

#ifdef LFS_VULKAN_RASTERIZER_DEV_SPV_DIR
            search_paths.push_back(lfs::core::utf8_to_path(LFS_VULKAN_RASTERIZER_DEV_SPV_DIR));
#endif

#ifdef PROJECT_ROOT_PATH
            search_paths.push_back(lfs::core::utf8_to_path(PROJECT_ROOT_PATH) /
                                   "src/rendering/rasterizer/vulkan/shader");
#endif

            for (const auto& path : search_paths) {
                if (std::filesystem::exists(path / probe_file)) {
                    return path;
                }
            }

            std::string error = "Cannot find VkSplat SPIR-V shaders. Searched in:";
            for (const auto& path : search_paths) {
                error += "\n  - " + lfs::core::path_to_utf8(path);
            }
            error += "\nExecutable directory: " + lfs::core::path_to_utf8(lfs::core::getExecutableDir());
            throw std::runtime_error(error);
        }

        [[nodiscard]] std::map<std::string, std::string> makeVkSplatSpirvPaths() {
            const std::filesystem::path root = resolveVkSplatSpirvRoot();
            return {
                {"projection_forward", (root / "generated/projection_forward.spv").string()},
                {"projection_forward_3dgut", (root / "generated/projection_forward_3dgut.spv").string()},
                {"selection_mask", (root / "generated/selection_mask.spv").string()},
                {"selection_polygon_rasterize",
                 (root / "generated/selection_polygon_rasterize.spv").string()},
                {"generate_keys", (root / "generated/generate_keys.spv").string()},
                {"compute_tile_ranges", (root / "generated/compute_tile_ranges.spv").string()},
                {"rasterize_forward", (root / "generated/rasterize_forward.spv").string()},
                {"rasterize_forward_3dgut", (root / "generated/rasterize_forward_3dgut.spv").string()},
                {"rasterize_forward_plain", (root / "generated/rasterize_forward_plain.spv").string()},
                {"rasterize_forward_3dgut_plain",
                 (root / "generated/rasterize_forward_3dgut_plain.spv").string()},
                {"cumsum_single_pass", (root / "generated/cumsum_single_pass.spv").string()},
                {"cumsum_block_scan", (root / "generated/cumsum_block_scan.spv").string()},
                {"cumsum_scan_block_sums", (root / "generated/cumsum_scan_block_sums.spv").string()},
                {"cumsum_add_block_offsets", (root / "generated/cumsum_add_block_offsets.spv").string()},
                {"radix_sort/upsweep", (root / "radix_sort/upsweep.spv").string()},
                {"radix_sort/spine", (root / "radix_sort/spine.spv").string()},
                {"radix_sort/downsweep", (root / "radix_sort/downsweep.spv").string()},
                {"radix_sort/upsweep_indirect", (root / "radix_sort/upsweep_indirect.spv").string()},
                {"radix_sort/spine_indirect", (root / "radix_sort/spine_indirect.spv").string()},
                {"radix_sort/downsweep_indirect", (root / "radix_sort/downsweep_indirect.spv").string()},
                {"seed_primitive_indices", (root / "generated/seed_primitive_indices.spv").string()},
                {"apply_depth_ordering", (root / "generated/apply_depth_ordering.spv").string()},
                {"visible_flags", (root / "generated/visible_flags.spv").string()},
                {"prepare_visible_sort", (root / "generated/prepare_visible_sort.spv").string()},
                {"prepare_tile_sort", (root / "generated/prepare_tile_sort.spv").string()},
                {"compact_visible_primitives", (root / "generated/compact_visible_primitives.spv").string()},
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

        [[nodiscard]] bool hasDeviceBuffer(const _VulkanBuffer& buffer) {
            return buffer.buffer != VK_NULL_HANDLE && buffer.size > 0;
        }

        template <typename T>
        [[nodiscard]] bool hasDeviceBuffer(const Buffer<T>& buffer) {
            return hasDeviceBuffer(buffer.deviceBuffer);
        }

        template <typename T>
        void releaseHostStorage(Buffer<T>& buffer) {
            auto& host = static_cast<std::vector<T>&>(buffer);
            if (!host.empty() || host.capacity() != 0) {
                std::vector<T>{}.swap(host);
            }
        }

        void releaseInputHostStorage(VulkanGSPipelineBuffers& buffers) {
            releaseHostStorage(buffers.xyz_ws);
            releaseHostStorage(buffers.sh0);
            releaseHostStorage(buffers.shN);
            releaseHostStorage(buffers.rotations);
            releaseHostStorage(buffers.scaling_raw);
            releaseHostStorage(buffers.opacity_raw);
            releaseHostStorage(buffers.scales_opacs);
            releaseHostStorage(buffers.sh_coeffs);
        }

        [[nodiscard]] double gib(const std::size_t bytes) {
            return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
        }

        template <typename T>
        [[nodiscard]] std::size_t viewBytes(const Buffer<T>& buffer) {
            return buffer.deviceBuffer.size;
        }

        struct ScopedStagingBuffer {
            VmaAllocator allocator = VK_NULL_HANDLE;
            VkBuffer buffer = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VmaAllocationInfo allocation_info{};
            std::string vram_scope;
            std::string vram_label;

            ~ScopedStagingBuffer() {
                if (allocator != VK_NULL_HANDLE && buffer != VK_NULL_HANDLE) {
                    if (!vram_scope.empty() && !vram_label.empty()) {
                        lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(vram_scope, vram_label, 0);
                    }
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

        [[nodiscard]] std::size_t growRegionCapacity(const std::size_t current,
                                                     const std::size_t required,
                                                     const std::size_t minimum) {
            if (current >= required) {
                return current;
            }
            const std::size_t target = std::max({required, minimum, std::size_t{4}});
            const std::size_t slack =
                target >= (1u << 20)
                    ? std::max(target / 4, static_cast<std::size_t>(8u << 20))
                    : std::max(target / 2, static_cast<std::size_t>(16u << 10));
            if (target > std::numeric_limits<std::size_t>::max() - slack) {
                return alignUp(target, 4);
            }
            return alignUp(target + slack, 4);
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

        [[nodiscard]] _VulkanBuffer makeBorrowedBufferView(const VkBuffer buffer,
                                                           const std::size_t allocation_size,
                                                           const std::size_t bytes,
                                                           const VkDeviceSize offset = 0) {
            _VulkanBuffer view{};
            view.buffer = buffer;
            view.allocation = VK_NULL_HANDLE;
            view.allocSize = allocation_size;
            view.offset = offset;
            view.size = bytes;
            return view;
        }

        [[nodiscard]] _VulkanBuffer makeResizableRegionView(const VulkanContext::ExternalBuffer& buffer,
                                                            const std::size_t offset,
                                                            const std::size_t capacity_bytes) {
            _VulkanBuffer view{};
            view.buffer = buffer.buffer;
            view.allocation = VK_NULL_HANDLE;
            view.allocSize = capacity_bytes;
            view.offset = offset;
            view.size = 0;
            return view;
        }

        [[nodiscard]] std::expected<void, std::string> ensureCudaInteropBuffer(
            VulkanContext& context,
            VulkanContext::ExternalBuffer& buffer,
            lfs::rendering::CudaVulkanBufferInterop& interop,
            const std::size_t required_bytes,
            const std::string_view diagnostic_scope,
            const std::string_view diagnostic_label,
            const std::string_view error_label) {
            if (required_bytes == 0) {
                return std::unexpected(std::format("VkSplat {} slot '{}' requested zero-byte allocation",
                                                   error_label,
                                                   diagnostic_label));
            }
            const bool label_matches =
                buffer.diagnostic_scope == diagnostic_scope &&
                buffer.diagnostic_label.starts_with(std::string(diagnostic_label));
            if (buffer.buffer != VK_NULL_HANDLE && buffer.size >= required_bytes && label_matches) {
                return {};
            }

            interop.reset();
            context.destroyExternalBuffer(buffer);

            const VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            if (!context.createExternalBuffer(static_cast<VkDeviceSize>(required_bytes),
                                              usage,
                                              buffer,
                                              diagnostic_scope,
                                              diagnostic_label)) {
                return std::unexpected(std::format("VkSplat external {} buffer '{}' allocation failed: {}",
                                                   error_label,
                                                   diagnostic_label,
                                                   context.lastError()));
            }
            const auto native = context.releaseExternalBufferNativeHandle(buffer);
            if (!VulkanContext::externalNativeHandleValid(native)) {
                context.destroyExternalBuffer(buffer);
                return std::unexpected(std::format("VkSplat external {} buffer '{}' returned invalid native handle",
                                                   error_label,
                                                   diagnostic_label));
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
                                                   error_label,
                                                   diagnostic_label,
                                                   err));
            }
            return {};
        }

        constexpr std::size_t kSharedScratchPageBytes = std::size_t{2} << 20;
        constexpr std::size_t kSharedScratchMinBytes = std::size_t{384} << 20;

        enum InputRegion : std::size_t {
            InputXyzWs = 0,
            InputSh0 = 1,
            InputShN = 2,
            InputRotations = 3,
            InputScalingRaw = 4,
            InputOpacityRaw = 5,
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
            SelectionQueryPolygonVertices = 5,
            SelectionQueryPolygonMask = 6,
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

        void stageSelectionColorTableCpu(std::vector<float>& dst,
                                         const lfs::rendering::GaussianOverlayState& overlay) {
            dst.resize(lfs::rendering::kSelectionColorTableCount * 4u);
            for (std::size_t i = 0; i < lfs::rendering::kSelectionColorTableCount; ++i) {
                const glm::vec4 color = overlay.selection_colors[i];
                dst[i * 4u + 0u] = color.r;
                dst[i * 4u + 1u] = color.g;
                dst[i * 4u + 2u] = color.b;
                dst[i * 4u + 3u] = color.a;
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
            VisibilityFlags = 25,
            ParamCount = 26,
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

        void stageNodeMaskCpu(std::vector<std::uint8_t>& dst,
                              const std::vector<bool>& mask,
                              const std::size_t byte_count) {
            dst.assign(byte_count, 0u);
            const std::size_t count = std::min(mask.size(), byte_count);
            for (std::size_t i = 0; i < count; ++i) {
                dst[i] = mask[i] ? 1u : 0u;
            }
        }

        void stageSelectionPrimitivesCpu(std::vector<float>& dst,
                                         const std::vector<glm::vec4>& primitives) {
            const std::size_t count = std::max<std::size_t>(primitives.size(), 1u);
            dst.resize(count * 4u);
            std::fill(dst.begin(), dst.end(), 0.0f);
            for (std::size_t i = 0; i < primitives.size(); ++i) {
                dst[i * 4u + 0u] = primitives[i].x;
                dst[i * 4u + 1u] = primitives[i].y;
                dst[i * 4u + 2u] = primitives[i].z;
                dst[i * 4u + 3u] = primitives[i].w;
            }
        }

        void stageSelectionPolygonVerticesCpu(std::vector<float>& dst,
                                              const std::vector<glm::vec2>& vertices) {
            const std::size_t count = std::max<std::size_t>(vertices.size(), 1u);
            dst.resize(count * 2u);
            std::fill(dst.begin(), dst.end(), 0.0f);
            for (std::size_t i = 0; i < vertices.size(); ++i) {
                dst[i * 2u + 0u] = vertices[i].x;
                dst[i * 2u + 1u] = vertices[i].y;
            }
        }

        [[nodiscard]] std::expected<void, std::string> copyHostBytesToInteropRegion(
            const lfs::rendering::CudaVulkanBufferInterop& interop,
            const void* src,
            const std::size_t src_byte_count,
            const std::size_t byte_count,
            const std::size_t dst_offset,
            const cudaStream_t stream,
            const std::string_view label) {
            if (byte_count == 0 || src_byte_count < byte_count || src == nullptr) {
                return std::unexpected(std::format(
                    "VkSplat {} upload requested {} bytes from {} bytes",
                    label,
                    byte_count,
                    src_byte_count));
            }
            if (dst_offset > interop.size() || byte_count > interop.size() - dst_offset) {
                return std::unexpected(std::format(
                    "VkSplat {} upload range [{}, {}+{}) exceeds mapped query buffer {}",
                    label,
                    dst_offset,
                    dst_offset,
                    byte_count,
                    interop.size()));
            }
            auto* const base = static_cast<std::uint8_t*>(interop.devicePointer());
            if (base == nullptr) {
                return std::unexpected(std::format(
                    "VkSplat {} upload requires a mapped CUDA/Vulkan buffer",
                    label));
            }
            const cudaError_t status = cudaMemcpyAsync(base + dst_offset,
                                                       src,
                                                       byte_count,
                                                       cudaMemcpyHostToDevice,
                                                       stream);
            if (status != cudaSuccess) {
                return std::unexpected(std::format("VkSplat {} H2D upload failed: {} ({})",
                                                   label,
                                                   cudaGetErrorName(status),
                                                   cudaGetErrorString(status)));
            }
            return {};
        }

        [[nodiscard]] std::expected<void, std::string> copyHostFloatsToInteropRegion(
            const lfs::rendering::CudaVulkanBufferInterop& interop,
            const std::vector<float>& src,
            const std::size_t byte_count,
            const std::size_t dst_offset,
            const cudaStream_t stream,
            const std::string_view label) {
            return copyHostBytesToInteropRegion(interop,
                                                src.data(),
                                                src.size() * sizeof(float),
                                                byte_count,
                                                dst_offset,
                                                stream,
                                                label);
        }

        [[nodiscard]] std::expected<void, std::string> copyHostBytesToInteropRegion(
            const lfs::rendering::CudaVulkanBufferInterop& interop,
            const std::vector<std::uint8_t>& src,
            const std::size_t byte_count,
            const std::size_t dst_offset,
            const cudaStream_t stream,
            const std::string_view label) {
            return copyHostBytesToInteropRegion(interop,
                                                src.data(),
                                                src.size(),
                                                byte_count,
                                                dst_offset,
                                                stream,
                                                label);
        }

        struct PolygonAabb {
            std::uint32_t x0 = 0;
            std::uint32_t y0 = 0;
            std::uint32_t w = 0;
            std::uint32_t h = 0;
        };

        [[nodiscard]] PolygonAabb computePolygonAabbClipped(
            const std::vector<glm::vec2>& vertices,
            const glm::ivec2 viewport_size) {
            if (vertices.empty() || viewport_size.x <= 0 || viewport_size.y <= 0) {
                return {};
            }
            float min_x = vertices[0].x;
            float min_y = vertices[0].y;
            float max_x = vertices[0].x;
            float max_y = vertices[0].y;
            for (std::size_t i = 1; i < vertices.size(); ++i) {
                min_x = std::min(min_x, vertices[i].x);
                min_y = std::min(min_y, vertices[i].y);
                max_x = std::max(max_x, vertices[i].x);
                max_y = std::max(max_y, vertices[i].y);
            }
            const int clipped_x0 = std::max(0, static_cast<int>(std::floor(min_x)));
            const int clipped_y0 = std::max(0, static_cast<int>(std::floor(min_y)));
            const int clipped_x1 = std::min(viewport_size.x, static_cast<int>(std::ceil(max_x)));
            const int clipped_y1 = std::min(viewport_size.y, static_cast<int>(std::ceil(max_y)));
            if (clipped_x1 <= clipped_x0 || clipped_y1 <= clipped_y0) {
                return {};
            }
            PolygonAabb aabb;
            aabb.x0 = static_cast<std::uint32_t>(clipped_x0);
            aabb.y0 = static_cast<std::uint32_t>(clipped_y0);
            aabb.w = static_cast<std::uint32_t>(clipped_x1 - clipped_x0);
            aabb.h = static_cast<std::uint32_t>(clipped_y1 - clipped_y0);
            return aabb;
        }

        [[nodiscard]] std::size_t modelTransformCount(const std::vector<glm::mat4>* const transforms) {
            return transforms && !transforms->empty() ? transforms->size() : std::size_t{1};
        }

        // CPU-only build of the model-transform upload payload. H2D is paid
        // only when the bytes differ from the cached copy.
        [[nodiscard]] std::expected<std::vector<float>, std::string> buildModelTransformsCpuFloats(
            const std::vector<glm::mat4>* const transforms) {
            try {
                const std::size_t count = modelTransformCount(transforms);
                std::vector<float> cpu(count * 16u, 0.0f);
                for (std::size_t i = 0; i < count; ++i) {
                    const glm::mat4 transform =
                        transforms && i < transforms->size() ? (*transforms)[i] : glm::mat4(1.0f);
                    const auto rows = rowMajorMat4(transform);
                    std::memcpy(cpu.data() + i * 16u, rows.data(), rows.size() * sizeof(float));
                }
                return cpu;
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

        // Builds the overlay parameter table on CPU only. The H2D transfer is
        // performed at the call site, conditionally on an output-bytes diff.
        [[nodiscard]] std::expected<std::vector<float>, std::string> buildOverlayParamsCpuFloats(
            const lfs::rendering::ViewportRenderRequest& request,
            const bool selection_enabled,
            const bool preview_enabled,
            const bool transform_indices_enabled,
            const std::size_t node_mask_count,
            const bool node_visibility_cull) {
            try {
                std::vector<float> cpu(static_cast<std::size_t>(ParamCount) * 4u, 0.0f);
                float* const dst = cpu.data();

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
                          VisibilityFlags,
                          glm::vec4(node_visibility_cull ? 1.0f : 0.0f, 0.0f, 0.0f, 0.0f));
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

                return cpu;
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
            const Tensor* deleted_ptr_src =
                splat_data.has_deleted_mask() ? &splat_data.deleted() : nullptr;
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
                .deleted = deleted_ptr_src ? tensor_ptr(*deleted_ptr_src) : nullptr,
                .means_bytes = tensor_bytes(means),
                .scaling_bytes = tensor_bytes(scaling),
                .rotation_bytes = tensor_bytes(rotation),
                .opacity_bytes = tensor_bytes(opacity),
                .sh0_bytes = tensor_bytes(sh0),
                .shn_bytes = tensor_bytes(shn),
                .deleted_bytes = deleted_ptr_src ? tensor_bytes(*deleted_ptr_src) : 0,
            };
        }

        [[nodiscard]] std::shared_ptr<VulkanExternalTensorStorage> vulkanExternalStorage(
            const Tensor& tensor) {
            if (!tensor.is_valid() || !tensor.is_external_storage() ||
                tensor.external_storage_kind() != "vulkan_external_buffer") {
                return nullptr;
            }
            auto owner = tensor.external_storage_owner();
            if (!owner) {
                return nullptr;
            }
            return std::static_pointer_cast<VulkanExternalTensorStorage>(std::move(owner));
        }

        [[nodiscard]] std::expected<void, std::string> waitForInputTensorStream(
            const cudaStream_t stream,
            const Tensor& tensor,
            const std::string_view label) {
            try {
                lfs::core::waitForCUDAStream(stream, tensor.stream());
                return {};
            } catch (const std::exception& e) {
                return std::unexpected(std::format(
                    "VkSplat failed to order {} stream before Vulkan read: {}",
                    label,
                    e.what()));
            }
        }

        [[nodiscard]] std::expected<void, std::string> waitForSplatInputStreams(
            const cudaStream_t stream,
            const lfs::core::SplatData& splat_data) {
            if (auto ok = waitForInputTensorStream(stream, splat_data.means_raw(), "means"); !ok) {
                return std::unexpected(ok.error());
            }
            if (auto ok = waitForInputTensorStream(stream, splat_data.sh0_raw(), "sh0"); !ok) {
                return std::unexpected(ok.error());
            }
            if (splat_data.shN_raw().is_valid() && splat_data.shN_raw().numel() > 0) {
                if (auto ok = waitForInputTensorStream(stream, splat_data.shN_raw(), "shN"); !ok) {
                    return std::unexpected(ok.error());
                }
            }
            if (auto ok = waitForInputTensorStream(stream, splat_data.rotation_raw(), "rotation"); !ok) {
                return std::unexpected(ok.error());
            }
            if (auto ok = waitForInputTensorStream(stream, splat_data.scaling_raw(), "scaling"); !ok) {
                return std::unexpected(ok.error());
            }
            if (auto ok = waitForInputTensorStream(stream, splat_data.opacity_raw(), "opacity"); !ok) {
                return std::unexpected(ok.error());
            }
            return {};
        }

        void populateVksplatCameraUniforms(
            VulkanGSRendererUniforms& uniforms,
            const lfs::rendering::FrameView& frame_view,
            const lfs::rendering::GaussianSceneState& scene,
            const int active_sh_degree,
            const std::uint32_t shN_layout_slots,
            const std::size_t num_splats,
            const bool equirectangular,
            const bool gut,
            const bool mip_filter) {
            (void)scene;
            uniforms = {};
            uniforms.image_width = static_cast<std::uint32_t>(frame_view.size.x);
            uniforms.image_height = static_cast<std::uint32_t>(frame_view.size.y);
            uniforms.grid_width = _CEIL_DIV(uniforms.image_width, TILE_WIDTH);
            uniforms.grid_height = _CEIL_DIV(uniforms.image_height, TILE_HEIGHT);
            uniforms.num_splats = static_cast<std::uint32_t>(num_splats);
            uniforms.active_sh = static_cast<std::uint32_t>(active_sh_degree);
            uniforms.shN_layout_slots = shN_layout_slots;
            uniforms.camera_model = packedVksplatCameraModel(frame_view, equirectangular, gut);
            uniforms.mip_filter = mip_filter ? 1u : 0u;

            if (frame_view.orthographic) {
                const float ortho_scale =
                    std::isfinite(frame_view.ortho_scale) && frame_view.ortho_scale > 1.0e-5f
                        ? frame_view.ortho_scale
                        : lfs::rendering::DEFAULT_ORTHO_SCALE;
                uniforms.fx = ortho_scale;
                uniforms.fy = ortho_scale;
                uniforms.cx = static_cast<float>(frame_view.size.x) * 0.5f;
                uniforms.cy = static_cast<float>(frame_view.size.y) * 0.5f;
            } else if (frame_view.intrinsics_override) {
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
            std::uint32_t depth_view = 0;
            glm::vec4 background{0.0f, 0.0f, 0.0f, 1.0f};
            float depth_min = 0.0f;
            float depth_max = 1.0f;
            float pad1 = 0.0f;
            float pad2 = 0.0f;
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
        std::lock_guard<std::mutex> readback_lock(readback_mutex_);
        if (context_ && context_->device() != VK_NULL_HANDLE) {
            const VkDevice device = context_->device();
            vkDeviceWaitIdle(device);
            if (readback_fence_ != VK_NULL_HANDLE) {
                vkDestroyFence(device, readback_fence_, nullptr);
                readback_fence_ = VK_NULL_HANDLE;
            }
            if (readback_pool_ != VK_NULL_HANDLE) {
                vkDestroyCommandPool(device, readback_pool_, nullptr);
                readback_pool_ = VK_NULL_HANDLE;
                readback_cmd_ = VK_NULL_HANDLE;
            }
        }
        releaseSharedScratchArena();
        drainRetiredScratchBuffers(true);
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
        for (auto& slot : cuda_opacity_copies_) {
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
        selection_query_timeline_.cuda_semaphore.reset();
        if (context_) {
            context_->destroyExternalSemaphore(selection_query_timeline_.vk_semaphore);
        }
        selection_query_timeline_.vk_semaphore = {};
        selection_query_timeline_.value = 0;
        if (context_) {
            for (auto& logical_slot : output_slots_) {
                for (auto& slot : logical_slot) {
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
            }
            if (compose_) {
                compose_->destroy(context_->device());
            }
            if (render_complete_timeline_ != VK_NULL_HANDLE) {
                vkDestroySemaphore(context_->device(), render_complete_timeline_, nullptr);
            }
        }
        render_complete_timeline_ = VK_NULL_HANDLE;
        render_complete_value_ = 0;
        latest_output_ring_slot_ = {};
        output_generations_ = {};
        ring_completion_values_ = {};
        next_ring_slot_ = 0;
        current_input_sh_degree_ = -1;
        compose_.reset();
        buffers_ = {};
        if (overlay_upload_stream_ != nullptr) {
            cudaStreamDestroy(overlay_upload_stream_);
            overlay_upload_stream_ = nullptr;
        }
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
        detach(buffers_.sh0.deviceBuffer);
        detach(buffers_.shN.deviceBuffer);
        detach(buffers_.scaling_raw.deviceBuffer);
        detach(buffers_.opacity_raw.deviceBuffer);
    }

    void VksplatViewportRenderer::plugRingInputs(
        const std::size_t ring_slot,
        const std::size_t num_splats,
        const bool reset_cached_raster_state) {
        assert(ring_slot < cuda_inputs_.size());
        auto& slot = cuda_inputs_[ring_slot];
        // All raw input region views share one VkBuffer / one device allocation; only
        // (offset, size) differs per binding. allocation is left null because the
        // CudaInputSlot owns it.
        const auto plug = [&](_VulkanBuffer& dev, std::size_t region) {
            dev = makeRegionView(slot.buffer, slot.region_offset[region], slot.region_bytes[region]);
        };
        plug(buffers_.xyz_ws.deviceBuffer, InputXyzWs);
        plug(buffers_.sh0.deviceBuffer, InputSh0);
        plug(buffers_.shN.deviceBuffer, InputShN);
        plug(buffers_.rotations.deviceBuffer, InputRotations);
        plug(buffers_.scaling_raw.deviceBuffer, InputScalingRaw);
        plug(buffers_.opacity_raw.deviceBuffer, InputOpacityRaw);
        buffers_.scales_opacs.deviceBuffer = {};
        buffers_.sh_coeffs.deviceBuffer = {};

        releaseInputHostStorage(buffers_);

        buffers_.num_splats = num_splats;
        if (reset_cached_raster_state) {
            buffers_.num_indices = 0;
            buffers_.is_unsorted_1 = true;
        }
    }

    void VksplatViewportRenderer::aliasSortScratchToInputSlot(const std::size_t ring_slot) {
        assert(ring_slot < cuda_inputs_.size());
        auto& slot = cuda_inputs_[ring_slot];
        if (slot.buffer.buffer == VK_NULL_HANDLE || slot.buffer.allocation_size == 0) {
            return;
        }

        // Regular 2D splat rasterization no longer reads the packed input regions
        // after projection has written projected splat state. Reuse the same
        // imported allocation for the four equally-sized sort arrays in that path;
        // callers must not use this alias for 3DGUT, whose raster pass reloads raw
        // means/rotations/scales/opacities analytically per pixel.
        const std::size_t array_capacity =
            (static_cast<std::size_t>(slot.buffer.allocation_size) / 4u) & ~std::size_t{3u};
        if (array_capacity == 0) {
            return;
        }

        const auto attach = [&](auto& buffer, const std::size_t index) {
            renderer_.destroyBuffer(buffer.deviceBuffer);
            buffer.deviceBuffer = makeResizableRegionView(slot.buffer, index * array_capacity, array_capacity);
        };

        attach(buffers_.sorting_keys_1, 0);
        attach(buffers_.sorting_keys_2, 1);
        attach(buffers_.sorting_gauss_idx_1, 2);
        attach(buffers_.sorting_gauss_idx_2, 3);

        // Sort compute will overwrite the input regions with keys/indices, so the
        // snapshot no longer reflects what's resident in slot.buffer. Invalidate
        // it so the next prepareInputs() falls into the upload path (cheap, ~20us)
        // — but keep the buffer + CUDA import alive so ensureCudaInteropBuffer
        // early-outs instead of doing a full destroy+create+import (~3.7ms).
        ring_uploaded_[ring_slot] = {};
    }

    void VksplatViewportRenderer::releaseInputSlot(VulkanContext& context, const std::size_t ring_slot) {
        assert(ring_slot < cuda_inputs_.size());
        auto& slot = cuda_inputs_[ring_slot];
        const VkBuffer released_buffer = slot.buffer.buffer;

        if (released_buffer != VK_NULL_HANDLE) {
            const auto detach_view = [released_buffer](_VulkanBuffer& dev) {
                if (dev.buffer == released_buffer && dev.allocation == VK_NULL_HANDLE) {
                    dev = {};
                }
            };
            detach_view(buffers_.xyz_ws.deviceBuffer);
            detach_view(buffers_.sh0.deviceBuffer);
            detach_view(buffers_.shN.deviceBuffer);
            detach_view(buffers_.rotations.deviceBuffer);
            detach_view(buffers_.scaling_raw.deviceBuffer);
            detach_view(buffers_.opacity_raw.deviceBuffer);
            detach_view(buffers_.scales_opacs.deviceBuffer);
            detach_view(buffers_.sh_coeffs.deviceBuffer);
            detach_view(buffers_.sorting_keys_1.deviceBuffer);
            detach_view(buffers_.sorting_keys_2.deviceBuffer);
            detach_view(buffers_.sorting_gauss_idx_1.deviceBuffer);
            detach_view(buffers_.sorting_gauss_idx_2.deviceBuffer);
        }

        slot.interop.reset();
        context.destroyExternalBuffer(slot.buffer);
        slot.region_offset = {};
        slot.region_bytes = {};
        ring_uploaded_[ring_slot] = {};
    }

    void VksplatViewportRenderer::releaseOpacityCopySlot(VulkanContext& context, const std::size_t ring_slot) {
        assert(ring_slot < cuda_opacity_copies_.size());
        auto& slot = cuda_opacity_copies_[ring_slot];
        const VkBuffer released_buffer = slot.buffer.buffer;

        if (released_buffer != VK_NULL_HANDLE &&
            buffers_.opacity_raw.deviceBuffer.buffer == released_buffer &&
            buffers_.opacity_raw.deviceBuffer.allocation == VK_NULL_HANDLE) {
            buffers_.opacity_raw.deviceBuffer = {};
        }

        slot.interop.reset();
        context.destroyExternalBuffer(slot.buffer);
        slot = {};
    }

    std::size_t VksplatViewportRenderer::estimateSharedScratchBytes(
        const std::size_t num_splats,
        const std::size_t sort_capacity,
        const std::size_t num_pixels,
        const std::size_t num_tiles) const {
        std::size_t cursor = 0;
        const auto add = [&](const std::size_t bytes) {
            cursor = alignUp(cursor, kRegionAlignment);
            cursor += alignUp(std::max<std::size_t>(bytes, 4), kRegionAlignment);
        };
        const auto add_count = [&](const std::size_t count, const std::size_t elem_size) {
            add(count * elem_size);
        };

        add_count(num_splats, sizeof(std::uint32_t));                                                            // primitive_depth_keys
        add_count(num_splats, sizeof(std::int32_t));                                                             // tiles_touched
        add_count(num_splats, sizeof(std::int64_t));                                                             // rect_tile_space
        add_count(num_splats, sizeof(std::int32_t));                                                             // radii
        add_count(2 * num_splats, sizeof(float));                                                                // xy_vs
        add_count(num_splats, sizeof(float));                                                                    // depths
        add_count(4 * num_splats, sizeof(float));                                                                // inv_cov_vs_opacity
        add_count(3 * num_splats, sizeof(float));                                                                // rgb
        add_count(num_splats, sizeof(std::int32_t));                                                             // overlay_flags
        add_count(num_splats, sizeof(std::int32_t));                                                             // primitive_sort_indices
        add_count(num_splats, sizeof(std::int32_t));                                                             // tiles_touched_depth_ordered
        add_count(num_splats, sizeof(std::int32_t));                                                             // visible_flags
        add_count(num_splats, sizeof(std::int32_t));                                                             // visible_prefix
        add_count(1, sizeof(std::uint32_t));                                                                     // visible_count
        add_count(3, sizeof(std::uint32_t));                                                                     // visible_sort_dispatch_args
        add_count(num_splats, sizeof(std::int32_t));                                                             // index_buffer_offset
        add_count(sort_capacity, sizeof(sortingKey_t));                                                          // sorting_keys_1
        add_count(sort_capacity, sizeof(sortingKey_t));                                                          // sorting_keys_2
        add_count(sort_capacity, sizeof(std::int32_t));                                                          // sorting_gauss_idx_1
        add_count(sort_capacity, sizeof(std::int32_t));                                                          // sorting_gauss_idx_2
        add_count(1, sizeof(std::uint32_t));                                                                     // tile_sort_count
        add_count(3, sizeof(std::uint32_t));                                                                     // tile_sort_dispatch_args
        add_count(num_tiles + 1, sizeof(std::int32_t));                                                          // tile_ranges
        add_count(4 * num_pixels, sizeof(float));                                                                // pixel_state
        add_count(num_pixels, sizeof(float));                                                                    // pixel_depth
        add_count(num_pixels, sizeof(std::int32_t));                                                             // n_contributors
        add_count(_CEIL_DIV(num_splats, std::size_t{1024}), sizeof(std::int32_t));                               // _cumsum_blockSums
        add_count(_CEIL_DIV(_CEIL_DIV(num_splats, std::size_t{1024}), std::size_t{1024}), sizeof(std::int32_t)); // _cumsum_blockSums2
        add_count(8 * 256, sizeof(std::int32_t));                                                                // _sorting_histogram
        add_count(_CEIL_DIV(sort_capacity, std::size_t{512 * 8}) * 256, sizeof(std::int32_t));                   // _sorting_histogram_cumsum
        return alignUp(cursor, kRegionAlignment);
    }

    std::expected<void, std::string> VksplatViewportRenderer::ensureSharedScratchArena(
        VulkanContext& context,
        const std::size_t required_bytes) {
        if (required_bytes == 0) {
            return std::unexpected("VkSplat shared scratch requested zero bytes");
        }
        if (!context.externalMemoryInteropEnabled()) {
            return std::unexpected("VkSplat shared scratch requires CUDA/Vulkan external-memory interop");
        }

        const std::size_t target_bytes = alignUp(
            std::max(required_bytes + required_bytes / 8, kSharedScratchMinBytes),
            kSharedScratchPageBytes);
        int device = 0;
        if (const cudaError_t err = cudaGetDevice(&device); err != cudaSuccess) {
            return std::unexpected(std::format("VkSplat shared scratch cudaGetDevice failed: {} ({})",
                                               cudaGetErrorName(err),
                                               cudaGetErrorString(err)));
        }

        constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;

        // The per-region rows are published by bindSharedScratchBuffers (the single
        // source of truth for the breakdown); here we only refresh the capacity gauge.
        const auto publish_capacity = [this]() {
            lfs::diagnostics::VramProfiler::instance().setGauge(
                "vram.audit.shared_scratch.capacity", static_cast<double>(shared_scratch_.bytes));
        };

        // Grow callback handed to the arena: when training's scratch demand
        // outgrows the committed capacity, the arena invokes this (on the training
        // thread) to grow the exportable block in place under its stable base
        // address — contents preserved — and returns the new committed size. The
        // render re-imports the larger range into Vulkan on its next frame.
        const auto make_grow_fn = [](std::shared_ptr<lfs::core::ExportableBlock> block) {
            return [block = std::move(block)](std::size_t need) -> std::size_t {
                const std::size_t want =
                    need > (std::numeric_limits<std::size_t>::max() / 2) ? need : need + need / 2;
                auto grew = lfs::core::growExportableDeviceBlock(block, want);
                if (!grew) {
                    return std::size_t{0};
                }
                return block->size;
            };
        };

        const auto try_install_existing = [&]() -> bool {
            if (!shared_scratch_.block) {
                return false;
            }
            lfs::core::RasterizerMemoryArena::ExternalBacking backing{
                .device_ptr = shared_scratch_.block->device_ptr,
                .size = shared_scratch_.block->size,
                .device = device,
                .owner = std::shared_ptr<void>(shared_scratch_.block),
                .label = "vksplat.shared_scratch",
                .grow = make_grow_fn(shared_scratch_.block),
            };
            return lfs::core::GlobalArenaManager::instance().try_install_external_backing(std::move(backing));
        };

        // NOTE: if the training thread grew the block in place (new export handle),
        // the Vulkan re-import is NOT done here — doing it on the render thread
        // without holding the arena frame would race training's grow. It is done
        // in reimportSharedScratchIfGrown() once the render owns the arena frame
        // (which excludes training), so the block is stable when we read it.

        // Fast path: an installed block already large enough for this frame.
        if (shared_scratch_.block && shared_scratch_.bytes >= required_bytes &&
            shared_scratch_.imported_buffer.buffer != VK_NULL_HANDLE) {
            if (shared_scratch_.installed_in_training_arena || try_install_existing()) {
                shared_scratch_.installed_in_training_arena = true;
                publish_capacity();
                return {};
            }
            return std::unexpected("VkSplat shared scratch training rasterizer arena is busy");
        }

        // Grow path: a block exists but is too small. Grow the committed physical
        // IN PLACE under the stable virtual address so training's arena base
        // pointer never changes (no use-after-free), then re-import the larger
        // range into Vulkan. The arena drains all frames + the device before the
        // commit callback runs, so the unmap/recommit is race-free.
        if (shared_scratch_.block && shared_scratch_.installed_in_training_arena &&
            shared_scratch_.imported_buffer.buffer != VK_NULL_HANDLE) {
            void* const device_ptr = shared_scratch_.block->device_ptr;
            std::string commit_error;
            const auto commit = [&](std::size_t new_size) -> bool {
                auto grew = lfs::core::growExportableDeviceBlock(shared_scratch_.block, new_size);
                if (!grew) {
                    commit_error = grew.error();
                    return false;
                }
                if (*grew) {
                    VulkanContext::ExternalBuffer reimported{};
                    if (!context.importExternalBuffer(shared_scratch_.block->handle.native,
                                                      static_cast<VkDeviceSize>(shared_scratch_.block->size),
                                                      usage,
                                                      reimported,
                                                      "shared.scratch",
                                                      "cuda_vulkan_arena")) {
                        commit_error = context.lastError();
                        return false;
                    }
                    detachSharedScratchBuffers();
                    retireSharedScratchBuffer(std::move(shared_scratch_.imported_buffer));
                    shared_scratch_.imported_buffer = reimported;
                    shared_scratch_.bytes = shared_scratch_.block->size;
                    ++shared_scratch_.generation;
                }
                return true;
            };
            if (!lfs::core::GlobalArenaManager::instance().grow_external_backing(device_ptr, target_bytes, commit)) {
                return std::unexpected(commit_error.empty()
                                           ? std::string("VkSplat shared scratch training rasterizer arena is busy")
                                           : std::format("VkSplat shared scratch grow failed: {}", commit_error));
            }
            shared_scratch_.installed_in_training_arena = true;
            publish_capacity();
            LOG_INFO("VkSplat shared scratch arena grew to {} MiB (stable address)",
                     shared_scratch_.bytes >> 20);
            return {};
        }

        // First allocation: reserve a large virtual range (free) so the block can
        // grow in place up to whatever the densifying model needs, but commit only
        // the current target. The reservation is bounded by total device memory.
        std::size_t reserve_bytes = target_bytes;
        {
            std::size_t free_mem = 0, total_mem = 0;
            if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess && total_mem > reserve_bytes) {
                reserve_bytes = total_mem;
            }
        }

        auto block_result = lfs::core::allocateExportableDeviceBlock(target_bytes, device, false, reserve_bytes);
        if (!block_result) {
            return std::unexpected(std::format("VkSplat shared scratch CUDA allocation failed: {}",
                                               block_result.error()));
        }

        VulkanContext::ExternalBuffer imported{};
        if (!context.importExternalBuffer((*block_result)->handle.native,
                                          static_cast<VkDeviceSize>((*block_result)->size),
                                          usage,
                                          imported,
                                          "shared.scratch",
                                          "cuda_vulkan_arena")) {
            return std::unexpected(std::format("VkSplat shared scratch Vulkan import failed: {}",
                                               context.lastError()));
        }

        lfs::core::RasterizerMemoryArena::ExternalBacking backing{
            .device_ptr = (*block_result)->device_ptr,
            .size = (*block_result)->size,
            .device = device,
            .owner = std::shared_ptr<void>(*block_result),
            .label = "vksplat.shared_scratch",
            .grow = make_grow_fn(*block_result),
        };
        if (!lfs::core::GlobalArenaManager::instance().try_install_external_backing(std::move(backing))) {
            context.destroyExternalBuffer(imported);
            return std::unexpected("VkSplat shared scratch training rasterizer arena is busy");
        }

        releaseSharedScratchImportOnly();
        shared_scratch_.block = std::move(*block_result);
        shared_scratch_.imported_buffer = imported;
        shared_scratch_.bytes = shared_scratch_.block->size;
        shared_scratch_.installed_in_training_arena = true;
        ++shared_scratch_.generation;

        publish_capacity();

        LOG_INFO("VkSplat shared scratch arena: {} MiB committed, {} MiB reserved (grows in place)",
                 shared_scratch_.bytes >> 20,
                 reserve_bytes >> 20);
        return {};
    }

    std::expected<void, std::string>
    VksplatViewportRenderer::reimportSharedScratchIfGrown(VulkanContext& context) {
        if (!shared_scratch_.block || shared_scratch_.imported_buffer.buffer == VK_NULL_HANDLE) {
            return {};
        }
        if (shared_scratch_.bytes == shared_scratch_.block->size) {
            return {}; // not grown since last import
        }
        // Precondition: caller owns the arena render-frame, so training cannot grow
        // the block concurrently — block->size and block->handle are stable here.
        constexpr VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                             VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                             VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
        VulkanContext::ExternalBuffer reimported{};
        if (!context.importExternalBuffer(shared_scratch_.block->handle.native,
                                          static_cast<VkDeviceSize>(shared_scratch_.block->size),
                                          usage,
                                          reimported,
                                          "shared.scratch",
                                          "cuda_vulkan_arena")) {
            return std::unexpected(std::format("VkSplat shared scratch re-import after grow failed: {}",
                                               context.lastError()));
        }
        detachSharedScratchBuffers();
        retireSharedScratchBuffer(std::move(shared_scratch_.imported_buffer));
        shared_scratch_.imported_buffer = reimported;
        shared_scratch_.bytes = shared_scratch_.block->size;
        ++shared_scratch_.generation;
        lfs::diagnostics::VramProfiler::instance().setGauge(
            "vram.audit.shared_scratch.capacity", static_cast<double>(shared_scratch_.bytes));
        LOG_INFO("VkSplat shared scratch re-imported after in-place grow: {} MiB", shared_scratch_.bytes >> 20);
        return {};
    }

    void VksplatViewportRenderer::bindSharedScratchBuffers(
        const std::size_t num_splats,
        const std::size_t sort_capacity,
        const std::size_t num_pixels,
        const std::size_t num_tiles) {
        if (shared_scratch_.imported_buffer.buffer == VK_NULL_HANDLE) {
            return;
        }

        std::size_t cursor = 0;
        const auto bind_bytes = [&](auto& typed_buffer, const std::size_t bytes) {
            auto& dev = typed_buffer.deviceBuffer;
            const char* const label = dev.label;
            renderer_.destroyBuffer(dev);
            cursor = alignUp(cursor, kRegionAlignment);
            const std::size_t capacity = alignUp(std::max<std::size_t>(bytes, 4), kRegionAlignment);
            dev = makeResizableRegionView(shared_scratch_.imported_buffer, cursor, capacity);
            dev.label = label;
            cursor += capacity;
        };
        const auto bind_count = [&](auto& typed_buffer, const std::size_t count) {
            using Value = typename std::remove_reference_t<decltype(typed_buffer)>::value_type;
            bind_bytes(typed_buffer, count * sizeof(Value));
        };

        bind_count(buffers_.primitive_depth_keys, num_splats);
        bind_count(buffers_.tiles_touched, num_splats);
        bind_count(buffers_.rect_tile_space, num_splats);
        bind_count(buffers_.radii, num_splats);
        bind_count(buffers_.xy_vs, 2 * num_splats);
        bind_count(buffers_.depths, num_splats);
        bind_count(buffers_.inv_cov_vs_opacity, 4 * num_splats);
        bind_count(buffers_.rgb, 3 * num_splats);
        bind_count(buffers_.overlay_flags, num_splats);
        bind_count(buffers_.primitive_sort_indices, num_splats);
        bind_count(buffers_.tiles_touched_depth_ordered, num_splats);
        bind_count(buffers_.visible_flags, num_splats);
        bind_count(buffers_.visible_prefix, num_splats);
        bind_count(buffers_.visible_count, 1);
        bind_count(buffers_.visible_sort_dispatch_args, 3);
        bind_count(buffers_.index_buffer_offset, num_splats);
        const std::size_t per_splat_end = cursor;
        bind_count(buffers_.sorting_keys_1, sort_capacity);
        bind_count(buffers_.sorting_keys_2, sort_capacity);
        bind_count(buffers_.sorting_gauss_idx_1, sort_capacity);
        bind_count(buffers_.sorting_gauss_idx_2, sort_capacity);
        const std::size_t sort_end = cursor;
        bind_count(buffers_.tile_sort_count, 1);
        bind_count(buffers_.tile_sort_dispatch_args, 3);
        bind_count(buffers_.tile_ranges, num_tiles + 1);
        const std::size_t tiles_end = cursor;
        bind_count(buffers_.pixel_state, 4 * num_pixels);
        bind_count(buffers_.pixel_depth, num_pixels);
        bind_count(buffers_.n_contributors, num_pixels);
        const std::size_t pixel_end = cursor;
        bind_count(buffers_._cumsum_blockSums, _CEIL_DIV(num_splats, std::size_t{1024}));
        bind_count(buffers_._cumsum_blockSums2, _CEIL_DIV(_CEIL_DIV(num_splats, std::size_t{1024}), std::size_t{1024}));
        bind_count(buffers_._sorting_histogram, 8 * 256);
        bind_count(buffers_._sorting_histogram_cumsum,
                   _CEIL_DIV(sort_capacity, std::size_t{512 * 8}) * 256);

        // Attribute the committed arena exactly: each region's span comes straight from
        // the bind cursor, and reserve_unbound is the committed-but-unbound remainder.
        // Their sum equals shared_scratch_.bytes (the real committed VMM footprint), so
        // the HUD breaks the single "shared" row into its true components with no estimate.
        auto& profiler = lfs::diagnostics::VramProfiler::instance();
        const auto record_region = [&](const char* region, const std::size_t bytes) {
            profiler.recordCurrentBytes("shared.scratch", region, bytes);
        };
        record_region("per_splat", per_splat_end);
        record_region("sort_buffers", sort_end - per_splat_end);
        record_region("tiles", tiles_end - sort_end);
        record_region("pixel", pixel_end - tiles_end);
        record_region("scan", cursor - pixel_end);
        record_region("reserve_unbound",
                      shared_scratch_.bytes > cursor ? shared_scratch_.bytes - cursor : 0);
        profiler.setGauge("vram.audit.shared_scratch.vksplat_view_bytes", static_cast<double>(cursor));
    }

    void VksplatViewportRenderer::releasePrivateScratchBuffers() {
        std::size_t released_bytes = 0;
        const auto release = [&](auto& typed_buffer) {
            auto& dev = typed_buffer.deviceBuffer;
            if (dev.buffer == VK_NULL_HANDLE || dev.allocation == VK_NULL_HANDLE) {
                return;
            }
            released_bytes += dev.allocSize;
            renderer_.destroyBuffer(dev);
            typed_buffer.clear();
            typed_buffer.shrink_to_fit();
        };

#define RELEASE_PRIVATE_SCRATCH(name) release(buffers_.name)
        RELEASE_PRIVATE_SCRATCH(tiles_touched);
        RELEASE_PRIVATE_SCRATCH(rect_tile_space);
        RELEASE_PRIVATE_SCRATCH(radii);
        RELEASE_PRIVATE_SCRATCH(xy_vs);
        RELEASE_PRIVATE_SCRATCH(depths);
        RELEASE_PRIVATE_SCRATCH(inv_cov_vs_opacity);
        RELEASE_PRIVATE_SCRATCH(rgb);
        RELEASE_PRIVATE_SCRATCH(overlay_flags);
        RELEASE_PRIVATE_SCRATCH(primitive_depth_keys);
        RELEASE_PRIVATE_SCRATCH(primitive_sort_indices);
        RELEASE_PRIVATE_SCRATCH(tiles_touched_depth_ordered);
        RELEASE_PRIVATE_SCRATCH(visible_flags);
        RELEASE_PRIVATE_SCRATCH(visible_prefix);
        RELEASE_PRIVATE_SCRATCH(visible_count);
        RELEASE_PRIVATE_SCRATCH(visible_sort_dispatch_args);
        RELEASE_PRIVATE_SCRATCH(index_buffer_offset);
        RELEASE_PRIVATE_SCRATCH(sorting_keys_1);
        RELEASE_PRIVATE_SCRATCH(sorting_keys_2);
        RELEASE_PRIVATE_SCRATCH(sorting_gauss_idx_1);
        RELEASE_PRIVATE_SCRATCH(sorting_gauss_idx_2);
        RELEASE_PRIVATE_SCRATCH(tile_sort_count);
        RELEASE_PRIVATE_SCRATCH(tile_sort_dispatch_args);
        RELEASE_PRIVATE_SCRATCH(tile_ranges);
        RELEASE_PRIVATE_SCRATCH(pixel_state);
        RELEASE_PRIVATE_SCRATCH(pixel_depth);
        RELEASE_PRIVATE_SCRATCH(n_contributors);
        RELEASE_PRIVATE_SCRATCH(_cumsum_blockSums);
        RELEASE_PRIVATE_SCRATCH(_cumsum_blockSums2);
        RELEASE_PRIVATE_SCRATCH(_sorting_histogram);
        RELEASE_PRIVATE_SCRATCH(_sorting_histogram_cumsum);
#undef RELEASE_PRIVATE_SCRATCH

        if (released_bytes != 0) {
            buffers_.num_indices = 0;
            buffers_.is_unsorted_1 = true;
            LOG_PERF("vksplat.memory.release_private_scratch bytes={}MiB reason=live_training_shared_scratch",
                     released_bytes >> 20);
        }
    }

    void VksplatViewportRenderer::detachSharedScratchBuffers() {
        const VkBuffer shared_buffer = shared_scratch_.imported_buffer.buffer;
        if (shared_buffer == VK_NULL_HANDLE) {
            return;
        }
        const auto detach = [shared_buffer](_VulkanBuffer& dev) {
            if (dev.buffer != shared_buffer || dev.allocation != VK_NULL_HANDLE) {
                return;
            }
            const char* const label = dev.label;
            dev = {};
            dev.label = label;
        };

#define DETACH_SHARED(name) detach(buffers_.name.deviceBuffer)
        DETACH_SHARED(tiles_touched);
        DETACH_SHARED(rect_tile_space);
        DETACH_SHARED(radii);
        DETACH_SHARED(xy_vs);
        DETACH_SHARED(depths);
        DETACH_SHARED(inv_cov_vs_opacity);
        DETACH_SHARED(rgb);
        DETACH_SHARED(overlay_flags);
        DETACH_SHARED(primitive_depth_keys);
        DETACH_SHARED(primitive_sort_indices);
        DETACH_SHARED(tiles_touched_depth_ordered);
        DETACH_SHARED(visible_flags);
        DETACH_SHARED(visible_prefix);
        DETACH_SHARED(visible_count);
        DETACH_SHARED(visible_sort_dispatch_args);
        DETACH_SHARED(index_buffer_offset);
        DETACH_SHARED(sorting_keys_1);
        DETACH_SHARED(sorting_keys_2);
        DETACH_SHARED(sorting_gauss_idx_1);
        DETACH_SHARED(sorting_gauss_idx_2);
        DETACH_SHARED(tile_sort_count);
        DETACH_SHARED(tile_sort_dispatch_args);
        DETACH_SHARED(tile_ranges);
        DETACH_SHARED(pixel_state);
        DETACH_SHARED(pixel_depth);
        DETACH_SHARED(n_contributors);
        DETACH_SHARED(_cumsum_blockSums);
        DETACH_SHARED(_cumsum_blockSums2);
        DETACH_SHARED(_sorting_histogram);
        DETACH_SHARED(_sorting_histogram_cumsum);
#undef DETACH_SHARED

        buffers_.num_indices = 0;
        buffers_.is_unsorted_1 = true;
    }

    void VksplatViewportRenderer::releaseSharedScratchImportOnly() {
        detachSharedScratchBuffers();
        if (context_ != nullptr && shared_scratch_.imported_buffer.buffer != VK_NULL_HANDLE) {
            context_->destroyExternalBuffer(shared_scratch_.imported_buffer);
        }
        if (shared_scratch_.bytes != 0) {
            lfs::diagnostics::VramProfiler::instance().clearScope("shared.scratch");
            lfs::diagnostics::VramProfiler::instance().setGauge("vram.audit.shared_scratch.capacity", 0.0);
            lfs::diagnostics::VramProfiler::instance().setGauge("vram.audit.shared_scratch.vksplat_view_bytes", 0.0);
        }
        shared_scratch_ = {};
    }

    void VksplatViewportRenderer::releaseSharedScratchArena() {
        if (shared_scratch_.installed_in_training_arena && shared_scratch_.block) {
            lfs::core::GlobalArenaManager::instance().clear_external_backing(shared_scratch_.block->device_ptr);
        }
        releaseSharedScratchImportOnly();
    }

    void VksplatViewportRenderer::retireSharedScratchBuffer(VulkanContext::ExternalBuffer&& old) {
        if (old.buffer == VK_NULL_HANDLE || context_ == nullptr) {
            if (context_ != nullptr) {
                context_->destroyExternalBuffer(old);
            }
            old = {};
            return;
        }
        // render_complete_value_ is the value the most recently submitted batch
        // signals; that batch is the last one that could reference this buffer
        // (the current frame rebinds to the new import before recording). When no
        // frame has been submitted, the buffer was never seen by the GPU.
        if (render_complete_timeline_ == VK_NULL_HANDLE || render_complete_value_ == 0) {
            context_->destroyExternalBuffer(old);
            old = {};
            return;
        }
        retired_scratch_buffers_.emplace_back(render_complete_value_, std::move(old));
        old = {};
    }

    void VksplatViewportRenderer::drainRetiredScratchBuffers(bool force) {
        if (context_ == nullptr || retired_scratch_buffers_.empty()) {
            return;
        }
        auto retired = [&](std::uint64_t value) {
            if (force || value == 0 || render_complete_timeline_ == VK_NULL_HANDLE) {
                return true;
            }
            try {
                return renderer_.timelineValueComplete(render_complete_timeline_, value);
            } catch (const std::exception&) {
                return false;
            }
        };
        auto it = retired_scratch_buffers_.begin();
        while (it != retired_scratch_buffers_.end()) {
            if (retired(it->first)) {
                context_->destroyExternalBuffer(it->second);
                it = retired_scratch_buffers_.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::expected<VksplatViewportRenderer::OverlayBindingViews, std::string>
    VksplatViewportRenderer::uploadOverlayBindings(
        VulkanContext& context,
        const lfs::rendering::ViewportRenderRequest& request,
        const std::size_t num_splats,
        const std::size_t ring_slot) {
        if (num_splats == 0) {
            return std::unexpected("VkSplat overlay bindings cannot bind an empty model");
        }
        if (!context.externalMemoryInteropEnabled()) {
            return std::unexpected("VkSplat overlay bindings require CUDA/Vulkan external-memory interop");
        }
        assert(ring_slot < cuda_overlays_.size());
        // NOTE: Tried running this whole function on a dedicated non-blocking
        // CUDA stream (CUDAStreamGuard + overlay_upload_stream_) to escape the
        // legacy NULL-stream implicit-sync tax that was costing ~6 ms/frame.
        // It correctly delivered the perf win but introduced selection
        // flicker — even with explicit event-based cross-stream sync from
        // foreign source tensors (selection_source/preview_source/
        // transform_indices_source) onto our stream, some race remained that
        // wasn't tracked down. Reverted to running on the current stream
        // (typically NULL) so legacy implicit-FIFO ordering protects us.
        // The overlay_upload_stream_ member is still created in
        // ensureInitialized (cheap, ~µs) and reserved for a future retry once
        // we can compute-sanitizer racecheck the failing frames.
        auto& slot = cuda_overlays_[ring_slot];

        const bool selection_enabled =
            !request.overlay.cursor.saturation_preview &&
            hasOverlayTensor(request.overlay.emphasis.mask, num_splats);
        const bool preview_enabled =
            !request.overlay.cursor.saturation_preview &&
            hasOverlayTensor(request.overlay.emphasis.transient_mask.mask, num_splats);
        const bool transform_indices_enabled = hasTransformIndices(request.scene.transform_indices, num_splats);

        // Compare/split view restricts which scene nodes a panel may draw via
        // request.scene.node_visibility_mask. The forward path reuses the per-node
        // node_mask buffer (indexed by transform_indices, same as emphasis) to
        // hard-cull hidden nodes, mirroring the selection-query path. Visibility
        // culling and emphasis dimming are mutually exclusive in practice (compare
        // mode clears emphasis), so the restricting mask owns the shared buffer.
        const auto& node_visibility_mask = request.scene.node_visibility_mask;
        const bool node_visibility_restricts =
            transform_indices_enabled &&
            std::any_of(node_visibility_mask.begin(), node_visibility_mask.end(),
                        [](const bool visible) { return !visible; });
        const std::vector<bool>& forward_node_mask_source =
            node_visibility_restricts ? node_visibility_mask
                                      : request.overlay.emphasis.emphasized_node_mask;

        // Whether the forward rasterizer must run the overlay/selection path.
        // When nothing draws an overlay, the host dispatches the *_plain shader
        // variant, which strips that work from the per-pixel inner loop — the
        // dominant cost when zoomed out. Conservative: any uncertainty keeps the
        // full path (correctness over speed).
        const auto& emphasis = request.overlay.emphasis;
        const bool overlays_active =
            selection_enabled ||
            preview_enabled ||
            node_visibility_restricts ||
            !emphasis.emphasized_node_mask.empty() ||
            request.filters.crop_region.has_value() ||
            request.filters.ellipsoid_region.has_value() ||
            request.filters.view_volume.has_value() ||
            emphasis.dim_non_emphasized ||
            emphasis.flash_intensity > 0.0f ||
            emphasis.focused_gaussian_id >= 0 ||
            request.overlay.cursor.enabled ||
            request.overlay.markers.show_rings ||
            request.overlay.markers.show_center_markers;

        // Only reserve the per-gaussian mask regions when their feature is active.
        // The compose shader reads selection_mask/preview_mask solely under the
        // matching selection flag (alphablend_shader.slang), and the C++ upload is
        // likewise gated below — so when nothing is selected these collapse to a few
        // bytes instead of num_splats × ring-slots (~30 MiB at 5M splats).
        const std::size_t selection_mask_region_bytes =
            alignUp(selection_enabled ? num_splats : std::size_t{1}, 4);
        const std::size_t preview_mask_region_bytes =
            alignUp(preview_enabled ? num_splats : std::size_t{1}, 4);
        const std::size_t color_region_bytes =
            lfs::rendering::kSelectionColorTableCount * 4 * sizeof(float);
        const std::size_t transform_region_bytes =
            std::max<std::size_t>(transform_indices_enabled ? num_splats * sizeof(std::int32_t)
                                                            : sizeof(std::int32_t),
                                  sizeof(std::int32_t));
        const std::size_t node_mask_region_bytes =
            alignUp(std::max<std::size_t>(forward_node_mask_source.size(), 1), 4);
        const std::size_t overlay_params_region_bytes =
            static_cast<std::size_t>(ParamCount) * 4 * sizeof(float);
        const std::size_t model_transforms_region_bytes =
            modelTransformCount(request.scene.model_transforms) * 16 * sizeof(float);
        std::array<std::size_t, kOverlayRegionCount> region_bytes{};
        region_bytes[OverlaySelectionMask] = selection_mask_region_bytes;
        region_bytes[OverlayPreviewMask] = preview_mask_region_bytes;
        region_bytes[OverlaySelectionColors] = color_region_bytes;
        region_bytes[OverlayTransformIndices] = transform_region_bytes;
        region_bytes[OverlayNodeMask] = node_mask_region_bytes;
        region_bytes[OverlayParams] = overlay_params_region_bytes;
        region_bytes[OverlayModelTransforms] = model_transforms_region_bytes;
        std::array<std::size_t, kOverlayRegionCount> region_offset{};
        const std::size_t total_bytes = layoutRegions(region_bytes, region_offset, kRegionAlignment);
        const bool overlay_buffer_reallocated =
            slot.buffer.buffer == VK_NULL_HANDLE || slot.buffer.size < total_bytes;
        const auto previous_region_offset = slot.region_offset;
        const auto previous_region_bytes = slot.region_bytes;

        {
            LOG_TIMER("uploadOverlayBindings.ensure_buffer");
            if (auto ok = ensureCudaInteropBuffer(context,
                                                  slot.buffer,
                                                  slot.interop,
                                                  total_bytes,
                                                  "vulkan.vksplat.overlay_bindings",
                                                  std::format("ring{}.overlay_bindings", ring_slot),
                                                  "overlay bindings");
                !ok) {
                return std::unexpected(ok.error());
            }
        }
        slot.region_offset = region_offset;
        slot.region_bytes = region_bytes;
        const auto region_storage_changed = [&](const std::size_t region) {
            return overlay_buffer_reallocated ||
                   previous_region_offset[region] != region_offset[region] ||
                   previous_region_bytes[region] != region_bytes[region];
        };
        if (region_storage_changed(OverlaySelectionColors)) {
            slot.color_table_uploaded = false;
        }
        if (region_storage_changed(OverlayNodeMask)) {
            slot.node_mask_uploaded = false;
        }
        if (region_storage_changed(OverlayParams)) {
            slot.overlay_params_uploaded = false;
        }
        if (region_storage_changed(OverlayModelTransforms)) {
            slot.model_transforms_uploaded = false;
        }

        {
            LOG_TIMER("uploadOverlayBindings.prepare_sources");
            if (selection_enabled) {
                LOG_TIMER("uploadOverlayBindings.prepare_sources.selection_mask");
                auto prepared = prepareOverlayMaskTensor(*request.overlay.emphasis.mask, num_splats, "selection");
                if (!prepared) {
                    return std::unexpected(prepared.error());
                }
                slot.selection_source = std::move(*prepared);
            } else {
                slot.selection_source = {};
            }
            if (preview_enabled) {
                LOG_TIMER("uploadOverlayBindings.prepare_sources.preview_mask");
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
            // Palette is constant across most lasso-drag frames; rebuilding the
            // 1 KB CUDA tensor cost ~5 ms (CPU alloc + H2D + sync). Skip on hit.
            // GPU-side region is refreshed only when the bytes or target storage
            // changed.
            const bool color_table_cache_hit =
                !slot.color_table_upload_cpu.empty() &&
                slot.cached_color_palette == request.overlay.selection_colors;
            if (!color_table_cache_hit) {
                LOG_TIMER("uploadOverlayBindings.prepare_sources.color_table");
                stageSelectionColorTableCpu(slot.color_table_upload_cpu, request.overlay);
                slot.cached_color_palette = request.overlay.selection_colors;
                slot.color_table_uploaded = false;
            }
            {
                LOG_TIMER("uploadOverlayBindings.prepare_sources.transform_indices");
                auto transform_indices = prepareTransformIndicesTensor(request.scene.transform_indices, num_splats);
                if (!transform_indices) {
                    return std::unexpected(transform_indices.error());
                }
                slot.transform_indices_source = std::move(*transform_indices);
            }
            // Same caching pattern as color_table: the emphasized-node mask only
            // changes when the user selects a different scene-tree node. During
            // a lasso drag it is constant, but rebuilding the staging tensor +
            // H2D copy was costing ~6.5 ms/frame.
            const bool node_mask_cache_hit =
                !slot.node_mask_upload_cpu.empty() &&
                slot.cached_emphasized_node_mask == forward_node_mask_source;
            if (!node_mask_cache_hit) {
                LOG_TIMER("uploadOverlayBindings.prepare_sources.node_mask");
                stageNodeMaskCpu(slot.node_mask_upload_cpu,
                                 forward_node_mask_source,
                                 slot.region_bytes[OverlayNodeMask]);
                slot.cached_emphasized_node_mask = forward_node_mask_source;
                slot.node_mask_uploaded = false;
            }
            {
                // Output-bytes fingerprint cache. The CPU build is sub-µs; the
                // former ~6 ms cost was entirely the .to(Device::CUDA) sync.
                LOG_TIMER("uploadOverlayBindings.prepare_sources.overlay_params");
                auto overlay_params_cpu = buildOverlayParamsCpuFloats(
                    request,
                    selection_enabled,
                    preview_enabled,
                    transform_indices_enabled,
                    forward_node_mask_source.size(),
                    node_visibility_restricts);
                if (!overlay_params_cpu) {
                    return std::unexpected(overlay_params_cpu.error());
                }
                const bool overlay_params_cache_hit =
                    slot.cached_overlay_params_cpu == *overlay_params_cpu;
                if (!overlay_params_cache_hit) {
                    slot.cached_overlay_params_cpu = std::move(*overlay_params_cpu);
                    slot.overlay_params_upload_cpu = slot.cached_overlay_params_cpu;
                    slot.overlay_params_uploaded = false;
                }
            }
            {
                // Same output-bytes fingerprint pattern as overlay_params.
                LOG_TIMER("uploadOverlayBindings.prepare_sources.model_transforms");
                auto model_transforms_cpu =
                    buildModelTransformsCpuFloats(request.scene.model_transforms);
                if (!model_transforms_cpu) {
                    return std::unexpected(model_transforms_cpu.error());
                }
                const bool model_transforms_cache_hit =
                    slot.cached_model_transforms_cpu == *model_transforms_cpu;
                if (!model_transforms_cache_hit) {
                    slot.cached_model_transforms_cpu = std::move(*model_transforms_cpu);
                    slot.model_transforms_upload_cpu = slot.cached_model_transforms_cpu;
                    slot.model_transforms_uploaded = false;
                }
            }
        }

        // Restore the original per-source stream pick. With the upload running
        // on the current stream (NULL by default), legacy implicit-FIFO
        // ordering already chains us correctly behind whichever stream wrote
        // the foreign sources.
        cudaStream_t stream = nullptr;
        if (selection_enabled) {
            stream = slot.selection_source.stream();
        } else if (preview_enabled) {
            stream = slot.preview_source.stream();
        }

        {
            LOG_TIMER("uploadOverlayBindings.copy_to_interop");
            if (selection_enabled) {
                LOG_TIMER("uploadOverlayBindings.copy_to_interop.selection_mask");
                if (!slot.interop.copyFromTensor(slot.selection_source,
                                                 num_splats,
                                                 slot.region_offset[OverlaySelectionMask],
                                                 stream)) {
                    return std::unexpected(std::format("VkSplat selection mask upload failed: {}",
                                                       slot.interop.lastError()));
                }
            }
            if (preview_enabled) {
                LOG_TIMER("uploadOverlayBindings.copy_to_interop.preview_mask");
                if (!slot.interop.copyFromTensor(slot.preview_source,
                                                 num_splats,
                                                 slot.region_offset[OverlayPreviewMask],
                                                 stream)) {
                    return std::unexpected(std::format("VkSplat preview selection mask upload failed: {}",
                                                       slot.interop.lastError()));
                }
            }
            {
                LOG_TIMER("uploadOverlayBindings.copy_to_interop.color_table");
                if (!slot.color_table_uploaded) {
                    if (auto ok = copyHostFloatsToInteropRegion(slot.interop,
                                                                slot.color_table_upload_cpu,
                                                                slot.region_bytes[OverlaySelectionColors],
                                                                slot.region_offset[OverlaySelectionColors],
                                                                stream,
                                                                "selection color table");
                        !ok) {
                        return std::unexpected(ok.error());
                    }
                    slot.color_table_uploaded = true;
                }
            }
            {
                LOG_TIMER("uploadOverlayBindings.copy_to_interop.transform_indices");
                if (!slot.interop.copyFromTensor(slot.transform_indices_source,
                                                 slot.region_bytes[OverlayTransformIndices],
                                                 slot.region_offset[OverlayTransformIndices],
                                                 stream)) {
                    return std::unexpected(std::format("VkSplat transform-index upload failed: {}",
                                                       slot.interop.lastError()));
                }
            }
            {
                LOG_TIMER("uploadOverlayBindings.copy_to_interop.node_mask");
                if (!slot.node_mask_uploaded) {
                    if (auto ok = copyHostBytesToInteropRegion(slot.interop,
                                                               slot.node_mask_upload_cpu,
                                                               slot.region_bytes[OverlayNodeMask],
                                                               slot.region_offset[OverlayNodeMask],
                                                               stream,
                                                               "node mask");
                        !ok) {
                        return std::unexpected(ok.error());
                    }
                    slot.node_mask_uploaded = true;
                }
            }
            {
                LOG_TIMER("uploadOverlayBindings.copy_to_interop.overlay_params");
                if (!slot.overlay_params_uploaded) {
                    if (auto ok = copyHostFloatsToInteropRegion(slot.interop,
                                                                slot.overlay_params_upload_cpu,
                                                                slot.region_bytes[OverlayParams],
                                                                slot.region_offset[OverlayParams],
                                                                stream,
                                                                "overlay parameter");
                        !ok) {
                        return std::unexpected(ok.error());
                    }
                    slot.overlay_params_uploaded = true;
                }
            }
            {
                LOG_TIMER("uploadOverlayBindings.copy_to_interop.model_transforms");
                if (!slot.model_transforms_uploaded) {
                    if (auto ok = copyHostFloatsToInteropRegion(slot.interop,
                                                                slot.model_transforms_upload_cpu,
                                                                slot.region_bytes[OverlayModelTransforms],
                                                                slot.region_offset[OverlayModelTransforms],
                                                                stream,
                                                                "model transform");
                        !ok) {
                        return std::unexpected(ok.error());
                    }
                    slot.model_transforms_uploaded = true;
                }
            }
        }

        {
            LOG_TIMER("uploadOverlayBindings.signal_timeline");
            auto& timeline = overlay_upload_timelines_[ring_slot];
            const std::uint64_t signal_value = ++timeline.value;
            if (!timeline.cuda_semaphore.cudaSignal(signal_value, stream)) {
                return std::unexpected(std::format("VkSplat overlay binding upload signal failed: {}",
                                                   timeline.cuda_semaphore.lastError()));
            }
            renderer_.addTimelineWait(timeline.vk_semaphore.semaphore,
                                      signal_value,
                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        }

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
            .overlays_active = overlays_active,
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
            // Dedicated CUDA stream for overlay H2D uploads. Non-blocking flag
            // is essential — cudaStreamCreate (no flags) still implicitly
            // serializes with the legacy default (NULL) stream, which would
            // make our "isolated" stream sync with every other CUDA op in the
            // process. cudaStreamNonBlocking opts out of that.
            if (overlay_upload_stream_ == nullptr) {
                const cudaError_t err = cudaStreamCreateWithFlags(
                    &overlay_upload_stream_, cudaStreamNonBlocking);
                if (err != cudaSuccess) {
                    return std::unexpected(std::format(
                        "VkSplat overlay upload stream creation failed: {}",
                        cudaGetErrorString(err)));
                }
            }
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
                                         context.allocator(),
                                         context.pipelineCache());
            renderer_.assignBufferLabels(buffers_);
            renderer_.setCpuTimerCallback([](const std::string_view name, const double ms) {
                LOG_PERF("{} took {:.2f}ms", name, ms);
            });
            renderer_.addTimerCallback([](const std::vector<std::pair<size_t, double>>& updates) {
                for (size_t stage = 0; stage < updates.size() && stage < PerfTimer::stage_count(); ++stage) {
                    const auto [count, seconds] = updates[stage];
                    if (count == 0)
                        continue;
                    LOG_PERF("vksplat.gpu.{} took {:.3f}ms count={}",
                             PerfTimer::stage_name(stage),
                             seconds * 1000.0,
                             count);
                }
            });

            VkSemaphoreTypeCreateInfo timeline_info{};
            timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
            timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
            timeline_info.initialValue = 0;
            VkSemaphoreCreateInfo semaphore_info{};
            semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
            semaphore_info.pNext = &timeline_info;
            const VkResult semaphore_result =
                vkCreateSemaphore(context.device(), &semaphore_info, nullptr, &render_complete_timeline_);
            if (semaphore_result != VK_SUCCESS) {
                return std::unexpected(std::format(
                    "VkSplat render completion timeline creation failed: {}",
                    vkError("vkCreateSemaphore", semaphore_result)));
            }
            context.setDebugObjectName(VK_OBJECT_TYPE_SEMAPHORE,
                                       render_complete_timeline_,
                                       "VkSplat render completion timeline");
            render_complete_value_ = 0;
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat initialization failed: {}", e.what()));
        }

        // Per-ring-slot upload timeline: a Vulkan-exportable timeline semaphore
        // imported into CUDA so we can signal CUDA-side after the upload's
        // cudaMemcpyAsync and have Vulkan compute wait on it, replacing the
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
                    "VkSplat overlay upload timeline semaphore creation failed: {}",
                    context.lastError()));
            }
            const auto handle = context.releaseExternalSemaphoreNativeHandle(timeline.vk_semaphore);
            if (!VulkanContext::externalNativeHandleValid(handle)) {
                context.destroyExternalSemaphore(timeline.vk_semaphore);
                return std::unexpected("VkSplat overlay upload timeline semaphore export failed");
            }
            lfs::rendering::CudaVulkanExternalSemaphoreImport import{};
            import.semaphore_handle = handle;
            import.initial_value = timeline.vk_semaphore.initial_value;
            if (!timeline.cuda_semaphore.init(import)) {
                std::string err = timeline.cuda_semaphore.lastError();
                context.destroyExternalSemaphore(timeline.vk_semaphore);
                return std::unexpected(std::format(
                    "VkSplat overlay upload timeline semaphore CUDA import failed: {}", err));
            }
            timeline.value = 0;
        }
        {
            auto& timeline = selection_query_timeline_;
            if (!context.createExternalTimelineSemaphore(0, timeline.vk_semaphore)) {
                return std::unexpected(std::format(
                    "VkSplat selection query timeline semaphore creation failed: {}",
                    context.lastError()));
            }
            const auto handle = context.releaseExternalSemaphoreNativeHandle(timeline.vk_semaphore);
            if (!VulkanContext::externalNativeHandleValid(handle)) {
                context.destroyExternalSemaphore(timeline.vk_semaphore);
                timeline.vk_semaphore = {};
                return std::unexpected("VkSplat selection query timeline semaphore export failed");
            }
            lfs::rendering::CudaVulkanExternalSemaphoreImport import{};
            import.semaphore_handle = handle;
            import.initial_value = timeline.vk_semaphore.initial_value;
            if (!timeline.cuda_semaphore.init(import)) {
                std::string err = timeline.cuda_semaphore.lastError();
                context.destroyExternalSemaphore(timeline.vk_semaphore);
                timeline.vk_semaphore = {};
                return std::unexpected(std::format(
                    "VkSplat selection query timeline semaphore CUDA import failed: {}", err));
            }
            timeline.value = 0;
        }

        initialized_ = true;
        return {};
    }

    std::expected<void, std::string> VksplatViewportRenderer::waitForRingSlot(
        const std::size_t ring_slot,
        const std::string_view reason) {
        if (ring_slot >= ring_completion_values_.size() ||
            render_complete_timeline_ == VK_NULL_HANDLE) {
            return {};
        }
        const std::uint64_t value = ring_completion_values_[ring_slot];
        if (value == 0) {
            return {};
        }
        try {
            if (renderer_.timelineValueComplete(render_complete_timeline_, value)) {
                ring_completion_values_[ring_slot] = 0;
                return {};
            }
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat {} ring-slot status failed: {}",
                                               reason,
                                               e.what()));
        }

        LOG_TIMER("vksplat.ring_slot.wait_reuse");
        VkSemaphoreWaitInfo wait_info{};
        wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait_info.semaphoreCount = 1;
        wait_info.pSemaphores = &render_complete_timeline_;
        wait_info.pValues = &value;
        const VkResult result = vkWaitSemaphores(context_->device(), &wait_info, UINT64_MAX);
        if (result != VK_SUCCESS) {
            return std::unexpected(std::format("VkSplat {} ring-slot wait failed: {}",
                                               reason,
                                               vkError("vkWaitSemaphores", result)));
        }
        ring_completion_values_[ring_slot] = 0;
        return {};
    }

    std::size_t VksplatViewportRenderer::acquireRingSlot() {
        const std::size_t slot = next_ring_slot_;
        next_ring_slot_ = (next_ring_slot_ + 1) % kFrameRingSize;
        return slot;
    }

    std::size_t VksplatViewportRenderer::latestOutputRingSlot(const OutputSlot output_slot) const {
        return latest_output_ring_slot_[outputSlotIndex(output_slot)];
    }

    bool VksplatViewportRenderer::inputsResident(const lfs::core::SplatData& splat_data,
                                                 const std::size_t ring_slot) const {
        if (ring_slot >= ring_uploaded_.size())
            return false;
        const auto current = makeModelInputSnapshot(splat_data);
        return ring_uploaded_[ring_slot].valid() && ring_uploaded_[ring_slot] == current;
    }

    std::expected<VksplatViewportRenderer::InputBindingResult, std::string> VksplatViewportRenderer::prepareInputs(
        VulkanContext& context,
        const lfs::core::SplatData& splat_data,
        const std::size_t ring_slot,
        const bool force_upload,
        const int upload_sh_degree,
        const bool synchronize_upload) {
        const std::size_t n = static_cast<std::size_t>(splat_data.size());
        if (n == 0) {
            return std::unexpected("VkSplat cannot render an empty model");
        }

        if (!context.externalMemoryInteropEnabled()) {
            return std::unexpected("VkSplat input binding requires CUDA/Vulkan external-memory interop");
        }
        assert(ring_slot < cuda_inputs_.size());
        auto& slot = cuda_inputs_[ring_slot];

        const int effective_upload_sh_degree =
            upload_sh_degree < 0
                ? splat_data.get_max_sh_degree()
                : std::clamp(upload_sh_degree, 0, splat_data.get_max_sh_degree());
        auto upload_layout = vksplat::rawDeviceInputLayout(splat_data, effective_upload_sh_degree);
        if (!upload_layout) {
            return std::unexpected(upload_layout.error());
        }
        auto external_layout = vksplat::rawDeviceInputLayout(splat_data, splat_data.get_max_sh_degree());
        if (!external_layout) {
            return std::unexpected(external_layout.error());
        }
        const bool input_snapshot_changed = !inputsResident(splat_data, ring_slot);
        const bool input_upload_requested = force_upload || input_snapshot_changed;

        std::shared_ptr<VulkanExternalTensorStorage> means_storage, sh0_storage, shN_storage,
            rotations_storage, scaling_storage, opacity_storage;
        {
            LOG_TIMER("prepareInputs.storage_lookup");
            means_storage = vulkanExternalStorage(splat_data.means_raw());
            sh0_storage = vulkanExternalStorage(splat_data.sh0_raw());
            shN_storage = vulkanExternalStorage(splat_data.shN_raw());
            rotations_storage = vulkanExternalStorage(splat_data.rotation_raw());
            scaling_storage = vulkanExternalStorage(splat_data.scaling_raw());
            opacity_storage = vulkanExternalStorage(splat_data.opacity_raw());
        }
        // Soft deletes only need opacity rewritten; all geometry/color tensors can
        // still be borrowed from Vulkan-external model storage. Keep that path
        // narrow so a delete mask costs N floats instead of a full raw-model copy.
        const bool has_deleted_mask = splat_data.has_deleted_mask();
        const bool base_inputs_external =
            means_storage && sh0_storage && rotations_storage && scaling_storage;
        const bool can_bind_external =
            base_inputs_external &&
            (upload_layout->omits_shN || shN_storage) &&
            (opacity_storage || has_deleted_mask);
        const auto& layout = can_bind_external && shN_storage ? external_layout : upload_layout;

        std::vector<std::string> input_copy_reasons;
        const auto note_missing_storage =
            [&](const std::shared_ptr<VulkanExternalTensorStorage>& storage,
                const char* const name) {
                if (!storage) {
                    input_copy_reasons.emplace_back(std::format("missing_{}", name));
                }
            };
        note_missing_storage(means_storage, "means");
        note_missing_storage(sh0_storage, "sh0");
        note_missing_storage(rotations_storage, "rotation");
        note_missing_storage(scaling_storage, "scaling");
        if (!has_deleted_mask) {
            note_missing_storage(opacity_storage, "opacity");
        }
        if (!upload_layout->omits_shN) {
            note_missing_storage(shN_storage, "shN");
        }
        if (!can_bind_external && has_deleted_mask) {
            input_copy_reasons.emplace_back("soft_deleted_mask");
        }
        std::string input_copy_reason = "unknown";
        if (!input_copy_reasons.empty()) {
            input_copy_reason.clear();
            for (const auto& reason : input_copy_reasons) {
                if (!input_copy_reason.empty()) {
                    input_copy_reason += "+";
                }
                input_copy_reason += reason;
            }
        }

        const auto update_input_metadata = [&](const bool reset_cached_raster_state) {
            releaseInputHostStorage(buffers_);
            buffers_.num_splats = n;
            if (reset_cached_raster_state) {
                buffers_.num_indices = 0;
                buffers_.is_unsorted_1 = true;
            }
        };

        if (can_bind_external) {
            if (slot.buffer.buffer != VK_NULL_HANDLE) {
                LOG_PERF("vksplat.memory.release_input_copy ring={} bytes={} reason=zero_copy_external_tensors",
                         ring_slot,
                         static_cast<std::size_t>(slot.buffer.allocation_size));
                releaseInputSlot(context, ring_slot);
            }
            const auto require_capacity =
                [](const std::shared_ptr<VulkanExternalTensorStorage>& storage,
                   const std::size_t bytes,
                   const char* const label) -> std::expected<void, std::string> {
                if (!storage) {
                    return std::unexpected(std::format(
                        "VkSplat Vulkan-external {} storage is missing",
                        label));
                }
                if (storage->vkBuffer() == VK_NULL_HANDLE || storage->bytes() < bytes) {
                    return std::unexpected(std::format(
                        "VkSplat Vulkan-external {} storage is too small: have {} bytes, need {}",
                        label,
                        storage->bytes(),
                        bytes));
                }
                return {};
            };
            {
                LOG_TIMER("prepareInputs.require_capacity");
                if (auto ok = require_capacity(means_storage, layout->xyz_bytes, "means"); !ok) {
                    return std::unexpected(ok.error());
                }
                if (auto ok = require_capacity(sh0_storage, layout->sh0_bytes, "sh0"); !ok) {
                    return std::unexpected(ok.error());
                }
                if (!layout->omits_shN) {
                    if (auto ok = require_capacity(shN_storage, layout->shN_bytes, "shN"); !ok) {
                        return std::unexpected(ok.error());
                    }
                }
                if (auto ok = require_capacity(rotations_storage, layout->rotations_bytes, "rotation"); !ok) {
                    return std::unexpected(ok.error());
                }
                if (auto ok = require_capacity(scaling_storage, layout->scaling_bytes, "scaling"); !ok) {
                    return std::unexpected(ok.error());
                }
                if (!has_deleted_mask) {
                    if (auto ok = require_capacity(opacity_storage, layout->opacity_bytes, "opacity"); !ok) {
                        return std::unexpected(ok.error());
                    }
                }
            }

            auto& opacity_slot = cuda_opacity_copies_[ring_slot];
            bool opacity_copy_upload_needed = false;
            if (has_deleted_mask) {
                const VkBuffer previous_opacity_buffer = opacity_slot.buffer.buffer;
                const std::size_t previous_opacity_bytes = opacity_slot.bytes;
                const bool opacity_slot_had_buffer = previous_opacity_buffer != VK_NULL_HANDLE;
                {
                    LOG_TIMER("prepareInputs.opacity_copy.ensure_buffer");
                    if (auto ok = ensureCudaInteropBuffer(context,
                                                          opacity_slot.buffer,
                                                          opacity_slot.interop,
                                                          layout->opacity_bytes,
                                                          "vulkan.vksplat.opacity_copy",
                                                          std::format("ring{}.soft_deleted_opacity", ring_slot),
                                                          "deleted opacity");
                        !ok) {
                        return std::unexpected(ok.error());
                    }
                }
                opacity_copy_upload_needed =
                    input_upload_requested ||
                    !opacity_slot_had_buffer ||
                    opacity_slot.buffer.buffer != previous_opacity_buffer ||
                    previous_opacity_bytes != layout->opacity_bytes;
                opacity_slot.bytes = layout->opacity_bytes;
            } else if (opacity_slot.buffer.buffer != VK_NULL_HANDLE) {
                LOG_PERF("vksplat.memory.release_opacity_copy ring={} bytes={} reason=no_deleted_mask",
                         ring_slot,
                         static_cast<std::size_t>(opacity_slot.buffer.allocation_size));
                releaseOpacityCopySlot(context, ring_slot);
            }

            if (has_deleted_mask && opacity_slot.interop.devicePointer() == nullptr) {
                return std::unexpected("VkSplat deleted-opacity buffer is not mapped");
            }

            if (has_deleted_mask) {
                buffers_.opacity_raw.deviceBuffer = makeRegionView(opacity_slot.buffer, 0, layout->opacity_bytes);
            } else {
                buffers_.opacity_raw.deviceBuffer = makeBorrowedBufferView(
                    opacity_storage->vkBuffer(), opacity_storage->bytes(), layout->opacity_bytes, opacity_storage->vkOffset());
            }

            if (has_deleted_mask) {
                LOG_PERF("vksplat.memory.opacity_copy ring={} bytes={} upload_needed={}",
                         ring_slot,
                         layout->opacity_bytes,
                         opacity_copy_upload_needed);
            }

            {
                LOG_TIMER("prepareInputs.borrow_views");
                buffers_.xyz_ws.deviceBuffer = makeBorrowedBufferView(
                    means_storage->vkBuffer(), means_storage->bytes(), layout->xyz_bytes, means_storage->vkOffset());
                buffers_.sh0.deviceBuffer = makeBorrowedBufferView(
                    sh0_storage->vkBuffer(), sh0_storage->bytes(), layout->sh0_bytes, sh0_storage->vkOffset());
                buffers_.shN.deviceBuffer = layout->omits_shN
                                                ? makeBorrowedBufferView(
                                                      rotations_storage->vkBuffer(),
                                                      rotations_storage->bytes(),
                                                      layout->shN_bytes,
                                                      rotations_storage->vkOffset())
                                                : makeBorrowedBufferView(
                                                      shN_storage->vkBuffer(),
                                                      shN_storage->bytes(),
                                                      layout->shN_bytes,
                                                      shN_storage->vkOffset());
                buffers_.rotations.deviceBuffer = makeBorrowedBufferView(
                    rotations_storage->vkBuffer(), rotations_storage->bytes(), layout->rotations_bytes, rotations_storage->vkOffset());
                buffers_.scaling_raw.deviceBuffer = makeBorrowedBufferView(
                    scaling_storage->vkBuffer(), scaling_storage->bytes(), layout->scaling_bytes, scaling_storage->vkOffset());
                buffers_.scales_opacs.deviceBuffer = {};
                buffers_.sh_coeffs.deviceBuffer = {};
                update_input_metadata(input_snapshot_changed);
            }

            const cudaStream_t stream = splat_data.means_raw().stream();
            {
                LOG_TIMER("prepareInputs.wait_streams");
                if (auto ok = waitForSplatInputStreams(stream, splat_data); !ok) {
                    return std::unexpected(ok.error());
                }
            }
            if (has_deleted_mask && opacity_copy_upload_needed) {
                LOG_TIMER("prepareInputs.opacity_copy.copyRawOpacity");
                if (auto ok = vksplat::copyRawOpacityToBuffer(
                        splat_data,
                        opacity_slot.interop.devicePointer(),
                        stream);
                    !ok) {
                    return std::unexpected(ok.error());
                }
            }
            if (synchronize_upload) {
                LOG_TIMER("prepareInputs.stream_sync");
                if (const cudaError_t status = cudaStreamSynchronize(stream); status != cudaSuccess) {
                    return std::unexpected(std::format("VkSplat CUDA input stream sync failed: {} ({})",
                                                       cudaGetErrorName(status),
                                                       cudaGetErrorString(status)));
                }
            }

            {
                LOG_TIMER("prepareInputs.cuda_signal");
                auto& timeline = upload_timelines_[ring_slot];
                const std::uint64_t signal_value = ++timeline.value;
                if (!timeline.cuda_semaphore.cudaSignal(signal_value, stream)) {
                    return std::unexpected(std::format("VkSplat CUDA input-ready signal failed: {}",
                                                       timeline.cuda_semaphore.lastError()));
                }
                renderer_.addTimelineWait(timeline.vk_semaphore.semaphore,
                                          signal_value,
                                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            }

            {
                LOG_TIMER("prepareInputs.snapshot");
                ring_uploaded_[ring_slot] = makeModelInputSnapshot(splat_data);
            }
            current_input_sh_degree_ = shN_storage ? splat_data.get_max_sh_degree()
                                                   : effective_upload_sh_degree;
            return InputBindingResult{.uses_temporary_upload_slot = false};
        }

        if (slot.buffer.buffer != VK_NULL_HANDLE) {
            LOG_PERF("vksplat.memory.release_input_copy ring={} bytes={} reason=missing_external_storage",
                     ring_slot,
                     static_cast<std::size_t>(slot.buffer.allocation_size));
            releaseInputSlot(context, ring_slot);
        }
        if (cuda_opacity_copies_[ring_slot].buffer.buffer != VK_NULL_HANDLE) {
            LOG_PERF("vksplat.memory.release_opacity_copy ring={} bytes={} reason=missing_external_storage",
                     ring_slot,
                     static_cast<std::size_t>(cuda_opacity_copies_[ring_slot].buffer.allocation_size));
            releaseOpacityCopySlot(context, ring_slot);
        }
        ring_uploaded_[ring_slot] = {};
        return std::unexpected(std::format(
            "VkSplat refusing full input-copy fallback; model tensors must use Vulkan-external storage ({})",
            input_copy_reason));
    }

    void VksplatViewportRenderer::logVramBreakdownIfChanged(const std::string_view reason) {
        const std::size_t owned_total = buffers_.getTotalOwnedAllocSize();
        const std::size_t pipeline_current = renderer_.getCurrentAllocSize();
        const std::size_t pipeline_peak = renderer_.getPeakAllocSize();
        const std::size_t input_view_bytes =
            viewBytes(buffers_.xyz_ws) +
            viewBytes(buffers_.sh0) +
            viewBytes(buffers_.shN) +
            viewBytes(buffers_.rotations) +
            viewBytes(buffers_.scaling_raw) +
            viewBytes(buffers_.opacity_raw);
        const std::size_t sort_buffer_bytes =
            buffers_.sorting_keys_1.deviceBuffer.allocSize +
            buffers_.sorting_keys_2.deviceBuffer.allocSize +
            buffers_.sorting_gauss_idx_1.deviceBuffer.allocSize +
            buffers_.sorting_gauss_idx_2.deviceBuffer.allocSize;

        std::size_t fallback_input_bytes = 0;
        for (const auto& slot : cuda_inputs_) {
            fallback_input_bytes += static_cast<std::size_t>(slot.buffer.allocation_size);
        }
        std::size_t opacity_copy_bytes = 0;
        for (const auto& slot : cuda_opacity_copies_) {
            opacity_copy_bytes += static_cast<std::size_t>(slot.buffer.allocation_size);
        }
        std::size_t overlay_bytes = 0;
        for (const auto& slot : cuda_overlays_) {
            overlay_bytes += static_cast<std::size_t>(slot.buffer.allocation_size);
        }
        overlay_bytes += static_cast<std::size_t>(cuda_selection_query_.buffer.allocation_size);

        std::size_t output_image_bytes = 0;
        for (const auto& output_slots : output_slots_) {
            for (const auto& slot : output_slots) {
                output_image_bytes += static_cast<std::size_t>(slot.image.allocation_size);
                output_image_bytes += static_cast<std::size_t>(slot.depth_image.allocation_size);
            }
        }
        const std::size_t shared_scratch_bytes = shared_scratch_.bytes;

        const auto mix = [](const std::size_t seed, const std::size_t value) {
            return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6u) + (seed >> 2u));
        };
        std::size_t signature = 0;
        signature = mix(signature, owned_total);
        signature = mix(signature, pipeline_current);
        signature = mix(signature, pipeline_peak);
        signature = mix(signature, input_view_bytes);
        signature = mix(signature, fallback_input_bytes);
        signature = mix(signature, opacity_copy_bytes);
        signature = mix(signature, overlay_bytes);
        signature = mix(signature, output_image_bytes);
        signature = mix(signature, sort_buffer_bytes);
        signature = mix(signature, shared_scratch_bytes);
        if (signature == last_vram_report_signature_) {
            return;
        }
        last_vram_report_signature_ = signature;

        std::vector<std::pair<std::string, std::size_t>> entries;
        for (const auto& [name, bytes] : buffers_.getOwnedVramBreakdown()) {
            if (bytes != 0) {
                entries.emplace_back(name, bytes);
            }
        }
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
            return a.second > b.second;
        });

        std::string top;
        const std::size_t top_count = std::min<std::size_t>(entries.size(), 10);
        for (std::size_t i = 0; i < top_count; ++i) {
            if (!top.empty()) {
                top += ", ";
            }
            top += std::format("{}={:.2f}GiB", entries[i].first, gib(entries[i].second));
        }

        LOG_PERF("vksplat.memory reason={} renderer_owned={:.2f}GiB pipeline_current={:.2f}GiB pipeline_peak={:.2f}GiB input_views={:.2f}GiB fallback_inputs={:.2f}GiB opacity_copies={:.2f}GiB overlays={:.2f}GiB outputs={:.2f}GiB sort_buffers={:.2f}GiB shared_scratch={:.2f}GiB sort_capacity={} top=[{}]",
                 reason,
                 gib(owned_total),
                 gib(pipeline_current),
                 gib(pipeline_peak),
                 gib(input_view_bytes),
                 gib(fallback_input_bytes),
                 gib(opacity_copy_bytes),
                 gib(overlay_bytes),
                 gib(output_image_bytes),
                 gib(sort_buffer_bytes),
                 gib(shared_scratch_bytes),
                 buffers_.num_indices,
                 top);
    }

    std::expected<void, std::string> VksplatViewportRenderer::ensureOutputImages(
        VulkanContext& context,
        const glm::ivec2 size,
        const OutputSlot output_slot,
        const std::size_t ring_slot) {
        auto& slot = output_slots_[outputSlotIndex(output_slot)][ring_slot];
        if (slot.image.image != VK_NULL_HANDLE && slot.depth_image.image != VK_NULL_HANDLE &&
            slot.size == size) {
            return {};
        }
        const bool replacing_existing_output =
            slot.image.image != VK_NULL_HANDLE ||
            slot.depth_image.image != VK_NULL_HANDLE;
        // The previous GUI submit may still be sampling this slot. Drain submitted
        // frames before destroying images during viewport-size changes.
        if (replacing_existing_output && !context.waitForSubmittedFrames()) {
            return std::unexpected(std::format("VkSplat output resize wait failed: {}",
                                               context.lastError()));
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
        if (!context.createExternalImage(extent,
                                         VK_FORMAT_R8G8B8A8_UNORM,
                                         slot.image,
                                         "vulkan.vksplat.output_image",
                                         std::format("{}.color.ring{}", outputSlotDiagnosticName(output_slot), ring_slot))) {
            return std::unexpected(context.lastError());
        }
        if (!context.createExternalImage(extent,
                                         VK_FORMAT_R32_SFLOAT,
                                         slot.depth_image,
                                         "vulkan.vksplat.output_image",
                                         std::format("{}.depth.ring{}", outputSlotDiagnosticName(output_slot), ring_slot))) {
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
        const std::size_t output_ring_slot,
        const bool transparent_background,
        const bool depth_view,
        const float depth_min,
        const float depth_max) {
        if (auto ok = ensureComposePipeline(context); !ok) {
            return ok;
        }
        const std::size_t output_index = outputSlotIndex(output_slot);
        auto& output = output_slots_[output_index][output_ring_slot];

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
            output.generation = ++output_generations_[output_index];
            latest_output_ring_slot_[output_index] = output_ring_slot;
            return {};
        }

        VkDescriptorBufferInfo pixel_info{};
        pixel_info.buffer = buffers_.pixel_state.deviceBuffer.buffer;
        pixel_info.offset = buffers_.pixel_state.deviceBuffer.offset;
        pixel_info.range = buffers_.pixel_state.deviceBuffer.size;
        VkDescriptorBufferInfo depth_info{};
        depth_info.buffer = buffers_.pixel_depth.deviceBuffer.buffer;
        depth_info.offset = buffers_.pixel_depth.deviceBuffer.offset;
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
        pixel_barriers[0].offset = buffers_.pixel_state.deviceBuffer.offset;
        pixel_barriers[0].size = buffers_.pixel_state.deviceBuffer.size;
        pixel_barriers[1] = pixel_barriers[0];
        pixel_barriers[1].buffer = buffers_.pixel_depth.deviceBuffer.buffer;
        pixel_barriers[1].offset = buffers_.pixel_depth.deviceBuffer.offset;
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
            .depth_view = depth_view ? 1u : 0u,
            .background = glm::vec4(background, 1.0f),
            .depth_min = depth_min,
            .depth_max = depth_max,
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
        output.generation = ++output_generations_[output_index];
        latest_output_ring_slot_[output_index] = output_ring_slot;
        return {};
    }

    std::expected<void, std::string> VksplatViewportRenderer::ensureReadbackContext() const {
        if (!context_ || context_->device() == VK_NULL_HANDLE) {
            return std::unexpected("VkSplat readback context requested before device initialization");
        }
        if (readback_pool_ != VK_NULL_HANDLE) {
            return {};
        }
        const VkDevice device = context_->device();
        VkCommandPoolCreateInfo pool_info{};
        pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pool_info.queueFamilyIndex = context_->graphicsQueueFamily();
        if (const VkResult r = vkCreateCommandPool(device, &pool_info, nullptr, &readback_pool_);
            r != VK_SUCCESS) {
            return std::unexpected(vkError("vkCreateCommandPool(VkSplat readback)", r));
        }
        VkCommandBufferAllocateInfo command_info{};
        command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_info.commandPool = readback_pool_;
        command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_info.commandBufferCount = 1;
        if (const VkResult r = vkAllocateCommandBuffers(device, &command_info, &readback_cmd_);
            r != VK_SUCCESS) {
            vkDestroyCommandPool(device, readback_pool_, nullptr);
            readback_pool_ = VK_NULL_HANDLE;
            return std::unexpected(vkError("vkAllocateCommandBuffers(VkSplat readback)", r));
        }
        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        if (const VkResult r = vkCreateFence(device, &fence_info, nullptr, &readback_fence_);
            r != VK_SUCCESS) {
            vkDestroyCommandPool(device, readback_pool_, nullptr);
            readback_pool_ = VK_NULL_HANDLE;
            readback_cmd_ = VK_NULL_HANDLE;
            return std::unexpected(vkError("vkCreateFence(VkSplat readback)", r));
        }
        return {};
    }

    std::expected<std::shared_ptr<lfs::core::Tensor>, std::string>
    VksplatViewportRenderer::readOutputImage(VulkanContext& context, const OutputSlot output_slot) const {
        std::lock_guard<std::mutex> readback_lock(readback_mutex_);
        if (!context_) {
            return std::unexpected("VkSplat output readback requested before renderer initialization");
        }
        if (&context != context_) {
            return std::unexpected("VkSplat output readback received a different Vulkan context");
        }
        try {
            const_cast<VulkanGSRenderer&>(renderer_).waitForPendingBatch();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat output readback pending-batch wait failed: {}", e.what()));
        }

        const auto& output = output_slots_[outputSlotIndex(output_slot)][latestOutputRingSlot(output_slot)];
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

        ScopedStagingBuffer staging{};
        staging.allocator = context.allocator();
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
        staging.vram_scope = "vulkan.vksplat.readback_buffer";
        staging.vram_label = std::format("rgba:{}x{}", output.size.x, output.size.y);
        lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
            staging.vram_scope,
            staging.vram_label,
            static_cast<std::size_t>(staging.allocation_info.size));
        if (staging.allocation_info.pMappedData == nullptr) {
            return std::unexpected("VkSplat readback staging buffer is not host-mapped");
        }

        if (const auto ready = ensureReadbackContext(); !ready) {
            return std::unexpected(ready.error());
        }
        const VkCommandBuffer command_buffer = readback_cmd_;
        result = vkResetCommandBuffer(command_buffer, 0);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkResetCommandBuffer(VkSplat readback)", result));
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

        result = vkResetFences(device, 1, &readback_fence_);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkResetFences(VkSplat readback)", result));
        }

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        result = vkQueueSubmit(context.graphicsQueue(), 1, &submit_info, readback_fence_);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkQueueSubmit(VkSplat readback)", result));
        }
        result = vkWaitForFences(device, 1, &readback_fence_, VK_TRUE, UINT64_MAX);
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
        std::vector<float> hwc(pixel_count * 3u);
        for (std::size_t i = 0; i < pixel_count; ++i) {
            const std::size_t src = i * 4u;
            const std::size_t dst = i * 3u;
            hwc[dst] = static_cast<float>(rgba[src]) / 255.0f;
            hwc[dst + 1u] = static_cast<float>(rgba[src + 1u]) / 255.0f;
            hwc[dst + 2u] = static_cast<float>(rgba[src + 2u]) / 255.0f;
        }

        auto tensor = lfs::core::Tensor::from_vector(
            hwc,
            {static_cast<std::size_t>(height), static_cast<std::size_t>(width), std::size_t{3}},
            lfs::core::Device::CPU);
        return std::make_shared<lfs::core::Tensor>(std::move(tensor));
    }

    std::expected<float, std::string> VksplatViewportRenderer::sampleDepthAtPixel(
        VulkanContext& context,
        const int x,
        const int y,
        const OutputSlot output_slot) const {
        std::lock_guard<std::mutex> readback_lock(readback_mutex_);
        if (!context_) {
            return std::unexpected("VkSplat depth sample requested before renderer initialization");
        }
        if (&context != context_) {
            return std::unexpected("VkSplat depth sample received a different Vulkan context");
        }
        try {
            const_cast<VulkanGSRenderer&>(renderer_).waitForPendingBatch();
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat depth sample pending-batch wait failed: {}", e.what()));
        }

        const auto& output = output_slots_[outputSlotIndex(output_slot)][latestOutputRingSlot(output_slot)];
        if (output.depth_image.image == VK_NULL_HANDLE ||
            output.size.x <= 0 ||
            output.size.y <= 0) {
            return std::unexpected("VkSplat depth sample requested for an empty output slot");
        }
        if (output.depth_image.format != VK_FORMAT_R32_SFLOAT) {
            return std::unexpected("VkSplat depth sample only supports R32F depth images");
        }
        if (x < 0 || y < 0 || x >= output.size.x || y >= output.size.y) {
            return -1.0f;
        }

        if (!context.waitForSubmittedFrames()) {
            return std::unexpected(context.lastError());
        }

        const VkDevice device = context.device();
        constexpr VkDeviceSize byte_count = sizeof(float);
        ScopedStagingBuffer staging{};
        staging.allocator = context.allocator();
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
            return std::unexpected(vkError("vmaCreateBuffer(VkSplat depth sample)", result));
        }
        staging.vram_scope = "vulkan.vksplat.readback_buffer";
        staging.vram_label = "depth_sample";
        lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
            staging.vram_scope,
            staging.vram_label,
            static_cast<std::size_t>(staging.allocation_info.size));
        if (staging.allocation_info.pMappedData == nullptr) {
            return std::unexpected("VkSplat depth sample staging buffer is not host-mapped");
        }

        if (const auto ready = ensureReadbackContext(); !ready) {
            return std::unexpected(ready.error());
        }
        const VkCommandBuffer command_buffer = readback_cmd_;
        result = vkResetCommandBuffer(command_buffer, 0);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkResetCommandBuffer(VkSplat depth sample)", result));
        }

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        result = vkBeginCommandBuffer(command_buffer, &begin_info);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkBeginCommandBuffer(VkSplat depth sample)", result));
        }

        const VkImageLayout restore_layout =
            output.depth_layout != VK_IMAGE_LAYOUT_UNDEFINED
                ? output.depth_layout
                : VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        context.imageBarriers().transitionImage(
            command_buffer,
            output.depth_image.image,
            VK_IMAGE_ASPECT_COLOR_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

        VkBufferImageCopy copy_region{};
        copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy_region.imageSubresource.layerCount = 1;
        copy_region.imageOffset = {x, y, 0};
        copy_region.imageExtent = {1, 1, 1};
        vkCmdCopyImageToBuffer(command_buffer,
                               output.depth_image.image,
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               staging.buffer,
                               1,
                               &copy_region);

        context.imageBarriers().transitionImage(
            command_buffer,
            output.depth_image.image,
            VK_IMAGE_ASPECT_COLOR_BIT,
            restore_layout);

        result = vkEndCommandBuffer(command_buffer);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkEndCommandBuffer(VkSplat depth sample)", result));
        }

        result = vkResetFences(device, 1, &readback_fence_);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkResetFences(VkSplat depth sample)", result));
        }

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;
        result = vkQueueSubmit(context.graphicsQueue(), 1, &submit_info, readback_fence_);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkQueueSubmit(VkSplat depth sample)", result));
        }
        result = vkWaitForFences(device, 1, &readback_fence_, VK_TRUE, UINT64_MAX);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vkWaitForFences(VkSplat depth sample)", result));
        }

        result = vmaInvalidateAllocation(staging.allocator, staging.allocation, 0, byte_count);
        if (result != VK_SUCCESS) {
            return std::unexpected(vkError("vmaInvalidateAllocation(VkSplat depth sample)", result));
        }

        float depth = -1.0f;
        std::memcpy(&depth, staging.allocation_info.pMappedData, sizeof(depth));
        if (!std::isfinite(depth) || depth <= 0.0f || depth >= 1.0e9f) {
            return -1.0f;
        }
        return depth;
    }

    std::expected<lfs::core::Tensor, std::string> VksplatViewportRenderer::buildSelectionMask(
        VulkanContext& context,
        const lfs::core::SplatData& splat_data,
        const SelectionMaskRequest& request,
        const bool force_input_upload) {
        LOG_TIMER("VksplatViewportRenderer::buildSelectionMask");
        const glm::ivec2 size = request.frame_view.size;
        if (size.x <= 0 || size.y <= 0) {
            return std::unexpected("VkSplat selection query received an invalid viewport size");
        }
        if (request.equirectangular && !request.gut) {
            return std::unexpected("VkSplat equirectangular selection requires the 3DGUT backend");
        }
        const bool polygon_mode = (request.shape == SelectionMaskShape::Polygon);
        if (polygon_mode) {
            if (request.polygon_vertices.size() < 3) {
                return std::unexpected("VkSplat polygon selection requires at least 3 vertices");
            }
        } else {
            if (request.primitives.empty()) {
                return std::unexpected("VkSplat selection query requires at least one primitive");
            }
        }
        const std::size_t num_splats = static_cast<std::size_t>(splat_data.size());
        if (num_splats == 0) {
            return std::unexpected("VkSplat selection query cannot run on an empty model");
        }
        if (num_splats > std::numeric_limits<std::uint32_t>::max() ||
            request.primitives.size() > std::numeric_limits<std::uint32_t>::max() ||
            request.polygon_vertices.size() > std::numeric_limits<std::uint32_t>::max()) {
            return std::unexpected("VkSplat selection query exceeds shader dispatch limits");
        }
        const PolygonAabb polygon_aabb = polygon_mode
                                             ? computePolygonAabbClipped(request.polygon_vertices, size)
                                             : PolygonAabb{};
        if (polygon_mode && (polygon_aabb.w == 0 || polygon_aabb.h == 0)) {
            auto empty_output = Tensor::empty({num_splats}, Device::CUDA, DataType::Bool);
            if (const cudaError_t status = cudaMemsetAsync(empty_output.ptr<bool>(),
                                                           0,
                                                           num_splats * sizeof(bool),
                                                           empty_output.stream());
                status != cudaSuccess) {
                return std::unexpected(std::format("VkSplat polygon empty-output clear failed: {} ({})",
                                                   cudaGetErrorName(status),
                                                   cudaGetErrorString(status)));
            }
            if (const cudaError_t status = cudaStreamSynchronize(empty_output.stream());
                status != cudaSuccess) {
                return std::unexpected(std::format("VkSplat polygon empty-output sync failed: {} ({})",
                                                   cudaGetErrorName(status),
                                                   cudaGetErrorString(status)));
            }
            return empty_output;
        }
        if (!context.externalMemoryInteropEnabled()) {
            return std::unexpected("VkSplat selection query requires CUDA/Vulkan external-memory interop");
        }

        {
            LOG_TIMER("VksplatViewportRenderer::buildSelectionMask.ensureInitialized");
            if (auto ok = ensureInitialized(context); !ok) {
                return std::unexpected(ok.error());
            }
        }

        std::size_t ring_slot = 0;
        {
            LOG_TIMER("VksplatViewportRenderer::buildSelectionMask.wait_ring_slot");
            ring_slot = acquireRingSlot();
            if (auto ok = waitForRingSlot(ring_slot, "selection query"); !ok) {
                return std::unexpected(ok.error());
            }
        }

        auto input_binding = [&] {
            LOG_TIMER("VksplatViewportRenderer::buildSelectionMask.prepareInputs");
            return prepareInputs(context, splat_data, ring_slot, force_input_upload,
                                 0,
                                 request.synchronize_input_upload);
        }();
        if (!input_binding) {
            return std::unexpected(input_binding.error());
        }
        // Intentionally do NOT release ring_slot here. The slot's CUDA-imported
        // Vulkan buffer + uploaded splat data stay resident across selection
        // calls, so the next prepareInputs() hits the inputsResident() fast
        // path (plug_only) instead of re-running destroy+create+import+upload
        // (~3.7 ms). Stale data is detected via the makeModelInputSnapshot
        // comparison and re-uploaded automatically.
        (void)input_binding;

        auto& slot = cuda_selection_query_;
        const bool transform_indices_enabled = hasTransformIndices(request.scene.transform_indices, num_splats);
        const bool node_visibility_enabled = !request.scene.node_visibility_mask.empty();

        const std::size_t output_tensor_region_bytes = alignUp(std::max<std::size_t>(num_splats, 1), 4);
        const std::size_t unused_query_output_region_bytes = sizeof(std::uint32_t);
        const std::size_t transform_region_bytes =
            std::max<std::size_t>(transform_indices_enabled ? num_splats * sizeof(std::int32_t)
                                                            : sizeof(std::int32_t),
                                  sizeof(std::int32_t));
        const std::size_t node_mask_region_bytes =
            alignUp(std::max<std::size_t>(request.scene.node_visibility_mask.size(), 1), 4);
        const std::size_t primitive_count = std::max<std::size_t>(request.primitives.size(), 1u);
        const std::size_t primitive_region_bytes = primitive_count * 4 * sizeof(float);
        const std::size_t model_transforms_region_bytes =
            modelTransformCount(request.scene.model_transforms) * 16 * sizeof(float);
        const std::size_t polygon_vertex_count =
            std::max<std::size_t>(polygon_mode ? request.polygon_vertices.size() : 0u, 1u);
        const std::size_t polygon_vertices_region_bytes = polygon_vertex_count * 2 * sizeof(float);
        const std::size_t polygon_mask_pixels =
            polygon_mode ? static_cast<std::size_t>(polygon_aabb.w) * static_cast<std::size_t>(polygon_aabb.h)
                         : std::size_t{0};
        const std::size_t polygon_mask_region_bytes =
            alignUp(std::max<std::size_t>(polygon_mask_pixels, 1u), 4u);
        std::array<std::size_t, kSelectionQueryRegionCount> region_bytes{};
        region_bytes[SelectionQueryOutput] = unused_query_output_region_bytes;
        region_bytes[SelectionQueryTransformIndices] = transform_region_bytes;
        region_bytes[SelectionQueryNodeMask] = node_mask_region_bytes;
        region_bytes[SelectionQueryPrimitives] = primitive_region_bytes;
        region_bytes[SelectionQueryModelTransforms] = model_transforms_region_bytes;
        region_bytes[SelectionQueryPolygonVertices] = polygon_vertices_region_bytes;
        region_bytes[SelectionQueryPolygonMask] = polygon_mask_region_bytes;
        auto region_capacity_bytes = slot.region_capacity_bytes;
        const auto grow_fixed = [&](const std::size_t region) {
            region_capacity_bytes[region] =
                growRegionCapacity(region_capacity_bytes[region], region_bytes[region], region_bytes[region]);
        };
        const auto grow_dynamic = [&](const std::size_t region, const std::size_t minimum) {
            region_capacity_bytes[region] =
                growRegionCapacity(region_capacity_bytes[region], region_bytes[region], minimum);
        };
        grow_fixed(SelectionQueryOutput);
        grow_fixed(SelectionQueryTransformIndices);
        grow_fixed(SelectionQueryNodeMask);
        grow_dynamic(SelectionQueryPrimitives, 4u * 256u * sizeof(float));
        grow_fixed(SelectionQueryModelTransforms);
        grow_dynamic(SelectionQueryPolygonVertices, 8192u * 2u * sizeof(float));
        const std::size_t viewport_polygon_mask_bytes =
            alignUp(static_cast<std::size_t>(size.x) * static_cast<std::size_t>(size.y), 4u);
        grow_dynamic(SelectionQueryPolygonMask, viewport_polygon_mask_bytes);
        std::array<std::size_t, kSelectionQueryRegionCount> region_offset{};
        const std::size_t total_bytes = layoutRegions(region_capacity_bytes, region_offset, kRegionAlignment);
        const bool query_buffer_reallocated =
            slot.buffer.buffer == VK_NULL_HANDLE || slot.buffer.size < total_bytes;
        const auto previous_region_offset = slot.region_offset;
        const auto previous_region_bytes = slot.region_bytes;

        {
            LOG_TIMER("VksplatViewportRenderer::buildSelectionMask.ensure_query_buffer");
            if (query_buffer_reallocated) {
                LOG_PERF("VksplatViewportRenderer::buildSelectionMask.query_buffer_reallocate "
                         "required={} previous={}",
                         total_bytes,
                         static_cast<std::size_t>(slot.buffer.size));
            }
            if (auto ok = ensureCudaInteropBuffer(context,
                                                  slot.buffer,
                                                  slot.interop,
                                                  total_bytes,
                                                  "vulkan.vksplat.selection_query",
                                                  "selection_query",
                                                  "selection query");
                !ok) {
                return std::unexpected(ok.error());
            }
        }
        slot.region_offset = region_offset;
        slot.region_bytes = region_bytes;
        slot.region_capacity_bytes = region_capacity_bytes;
        // Upload caches are only valid for the exact region they populated.
        // Brush/rectangle primitive counts can move later regions without reallocating.
        const auto region_storage_changed = [&](const std::size_t region) {
            return query_buffer_reallocated ||
                   previous_region_offset[region] != region_offset[region] ||
                   previous_region_bytes[region] != region_bytes[region];
        };
        if (region_storage_changed(SelectionQueryTransformIndices)) {
            slot.transform_indices_uploaded = false;
        }
        if (region_storage_changed(SelectionQueryNodeMask)) {
            slot.node_mask_uploaded = false;
        }
        if (region_storage_changed(SelectionQueryModelTransforms)) {
            slot.model_transforms_uploaded = false;
        }
        if (region_storage_changed(SelectionQueryPolygonVertices)) {
            slot.polygon_vertices_uploaded = false;
        }

        {
            LOG_TIMER("VksplatViewportRenderer::buildSelectionMask.ensure_output_tensor");
            auto output_storage = vulkanExternalStorage(slot.output_tensor);
            if (!slot.output_tensor.is_valid() ||
                slot.output_tensor.dtype() != DataType::Bool ||
                slot.output_tensor.device() != Device::CUDA ||
                slot.output_tensor.numel() != num_splats ||
                !output_storage ||
                output_storage->bytes() < output_tensor_region_bytes) {
                auto output_tensor = makeVulkanExternalTensor(
                    context,
                    {num_splats},
                    DataType::Bool,
                    output_tensor_region_bytes,
                    "vksplat_selection_query_output");
                if (!output_tensor) {
                    return std::unexpected(output_tensor.error());
                }
                slot.output_tensor = std::move(*output_tensor);
            }
        }
        const auto output_storage = vulkanExternalStorage(slot.output_tensor);
        if (!output_storage) {
            return std::unexpected("VkSplat selection output tensor is not Vulkan external storage");
        }
        const auto output_view = makeBorrowedBufferView(output_storage->vkBuffer(),
                                                        output_storage->bytes(),
                                                        output_tensor_region_bytes,
                                                        output_storage->vkOffset());

        {
            LOG_TIMER("VksplatViewportRenderer::buildSelectionMask.staging");
            auto transform_indices = prepareTransformIndicesTensor(request.scene.transform_indices, num_splats);
            if (!transform_indices) {
                return std::unexpected(transform_indices.error());
            }
            const bool transform_indices_cache_hit =
                slot.transform_indices_source.is_valid() &&
                slot.cached_transform_indices_ptr == transform_indices->data_ptr() &&
                slot.cached_transform_indices_bytes == transform_indices->bytes();
            slot.transform_indices_source = std::move(*transform_indices);
            if (!transform_indices_cache_hit) {
                slot.cached_transform_indices_ptr = slot.transform_indices_source.data_ptr();
                slot.cached_transform_indices_bytes = slot.transform_indices_source.bytes();
                slot.transform_indices_uploaded = false;
            }

            const bool node_mask_cache_hit =
                !slot.node_mask_upload_cpu.empty() &&
                slot.cached_node_visibility_mask == request.scene.node_visibility_mask;
            if (!node_mask_cache_hit) {
                stageNodeMaskCpu(slot.node_mask_upload_cpu,
                                 request.scene.node_visibility_mask,
                                 slot.region_bytes[SelectionQueryNodeMask]);
                slot.cached_node_visibility_mask = request.scene.node_visibility_mask;
                slot.node_mask_uploaded = false;
            }

            stageSelectionPrimitivesCpu(slot.primitive_upload_cpu, request.primitives);

            auto model_transforms_cpu =
                buildModelTransformsCpuFloats(request.scene.model_transforms);
            if (!model_transforms_cpu) {
                return std::unexpected(model_transforms_cpu.error());
            }
            const bool model_transforms_cache_hit =
                slot.cached_model_transforms_cpu == *model_transforms_cpu;
            if (!model_transforms_cache_hit) {
                slot.cached_model_transforms_cpu = std::move(*model_transforms_cpu);
                slot.model_transforms_upload_cpu = slot.cached_model_transforms_cpu;
                slot.model_transforms_uploaded = false;
            }

            if (polygon_mode) {
                const bool polygon_vertices_cache_hit =
                    slot.cached_polygon_vertices.size() == request.polygon_vertices.size() &&
                    std::equal(slot.cached_polygon_vertices.begin(),
                               slot.cached_polygon_vertices.end(),
                               request.polygon_vertices.begin(),
                               [](const glm::vec2& a, const glm::vec2& b) {
                                   return a.x == b.x && a.y == b.y;
                               });
                if (!polygon_vertices_cache_hit) {
                    stageSelectionPolygonVerticesCpu(slot.polygon_vertices_upload_cpu, request.polygon_vertices);
                    slot.cached_polygon_vertices = request.polygon_vertices;
                    slot.polygon_vertices_uploaded = false;
                }
            }
        }

        const cudaStream_t selection_query_stream = nullptr;
        {
            LOG_TIMER("VksplatViewportRenderer::buildSelectionMask.upload");
            if (!slot.transform_indices_uploaded) {
                if (!slot.interop.copyFromTensor(slot.transform_indices_source,
                                                 slot.region_bytes[SelectionQueryTransformIndices],
                                                 slot.region_offset[SelectionQueryTransformIndices],
                                                 selection_query_stream)) {
                    return std::unexpected(std::format("VkSplat selection transform-index upload failed: {}",
                                                       slot.interop.lastError()));
                }
                slot.transform_indices_uploaded = true;
            }
            if (!slot.node_mask_uploaded) {
                if (auto ok = copyHostBytesToInteropRegion(slot.interop,
                                                           slot.node_mask_upload_cpu,
                                                           slot.region_bytes[SelectionQueryNodeMask],
                                                           slot.region_offset[SelectionQueryNodeMask],
                                                           selection_query_stream,
                                                           "selection node mask");
                    !ok) {
                    return std::unexpected(ok.error());
                }
                slot.node_mask_uploaded = true;
            }
            if (!slot.primitive_upload_cpu.empty()) {
                if (auto ok = copyHostFloatsToInteropRegion(slot.interop,
                                                            slot.primitive_upload_cpu,
                                                            slot.region_bytes[SelectionQueryPrimitives],
                                                            slot.region_offset[SelectionQueryPrimitives],
                                                            selection_query_stream,
                                                            "selection primitive");
                    !ok) {
                    return std::unexpected(ok.error());
                }
            }
            if (!slot.model_transforms_uploaded) {
                if (auto ok = copyHostFloatsToInteropRegion(slot.interop,
                                                            slot.model_transforms_upload_cpu,
                                                            slot.region_bytes[SelectionQueryModelTransforms],
                                                            slot.region_offset[SelectionQueryModelTransforms],
                                                            selection_query_stream,
                                                            "selection model transform");
                    !ok) {
                    return std::unexpected(ok.error());
                }
                slot.model_transforms_uploaded = true;
            }
            if (polygon_mode && !slot.polygon_vertices_uploaded) {
                if (!slot.polygon_vertices_upload_cpu.empty()) {
                    if (auto ok = copyHostFloatsToInteropRegion(slot.interop,
                                                                slot.polygon_vertices_upload_cpu,
                                                                slot.region_bytes[SelectionQueryPolygonVertices],
                                                                slot.region_offset[SelectionQueryPolygonVertices],
                                                                selection_query_stream,
                                                                "polygon vertex");
                        !ok) {
                        return std::unexpected(ok.error());
                    }
                }
                slot.polygon_vertices_uploaded = true;
            }
            if (polygon_mode) {
                auto* const polygon_mask_ptr =
                    static_cast<std::uint8_t*>(slot.interop.devicePointer()) +
                    slot.region_offset[SelectionQueryPolygonMask];
                if (const cudaError_t status =
                        cudaMemsetAsync(polygon_mask_ptr, 0, polygon_mask_region_bytes, selection_query_stream);
                    status != cudaSuccess) {
                    return std::unexpected(std::format("VkSplat polygon mask clear failed: {} ({})",
                                                       cudaGetErrorName(status),
                                                       cudaGetErrorString(status)));
                }
            }
        }

        const auto view = [&](const std::size_t region) {
            return makeRegionView(slot.buffer, slot.region_offset[region], slot.region_capacity_bytes[region]);
        };

        VulkanGSRendererUniforms camera_uniforms{};
        populateVksplatCameraUniforms(camera_uniforms,
                                      request.frame_view,
                                      request.scene,
                                      0,
                                      0,
                                      num_splats,
                                      request.equirectangular,
                                      request.gut,
                                      false);
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
        selection_uniforms.camera_model = camera_uniforms.camera_model;
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
        selection_uniforms.aabb_x0 = polygon_aabb.x0;
        selection_uniforms.aabb_y0 = polygon_aabb.y0;
        selection_uniforms.aabb_w = polygon_aabb.w;
        selection_uniforms.aabb_h = polygon_aabb.h;

        if (renderer_.isCommandBatchInProgress()) {
            return std::unexpected("VkSplat selection query cannot run inside an active command batch");
        }

        std::uint64_t selection_query_complete_value = 0;
        {
            LOG_TIMER("VksplatViewportRenderer::buildSelectionMask.upload_ready_signal");
            auto& timeline = selection_query_timeline_;
            const std::uint64_t upload_ready_value = ++timeline.value;
            if (!timeline.cuda_semaphore.cudaSignal(upload_ready_value, selection_query_stream)) {
                return std::unexpected(std::format("VkSplat selection query upload-ready signal failed: {}",
                                                   timeline.cuda_semaphore.lastError()));
            }
            renderer_.addTimelineWait(timeline.vk_semaphore.semaphore,
                                      upload_ready_value,
                                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            selection_query_complete_value = ++timeline.value;
        }

        {
            LOG_TIMER("VksplatViewportRenderer::buildSelectionMask.dispatch");
            try {
                auto batch = DeviceGuard(&renderer_,
                                         false,
                                         selection_query_timeline_.vk_semaphore.semaphore,
                                         selection_query_complete_value);
                if (polygon_mode) {
                    LOG_TIMER("VksplatViewportRenderer::buildSelectionMask.dispatch.polygon_rasterize");
                    VulkanGSSelectionPolygonRasterizeUniforms rasterize_uniforms{};
                    rasterize_uniforms.vertex_count =
                        static_cast<std::uint32_t>(request.polygon_vertices.size());
                    rasterize_uniforms.aabb_x0 = polygon_aabb.x0;
                    rasterize_uniforms.aabb_y0 = polygon_aabb.y0;
                    rasterize_uniforms.aabb_w = polygon_aabb.w;
                    rasterize_uniforms.aabb_h = polygon_aabb.h;
                    renderer_.executeSelectionPolygonRasterize(rasterize_uniforms,
                                                               view(SelectionQueryPolygonVertices),
                                                               view(SelectionQueryPolygonMask));
                }
                {
                    LOG_TIMER("VksplatViewportRenderer::buildSelectionMask.dispatch.selection_mask");
                    renderer_.executeSelectionMask(selection_uniforms,
                                                   buffers_,
                                                   view(SelectionQueryTransformIndices),
                                                   view(SelectionQueryNodeMask),
                                                   view(SelectionQueryPrimitives),
                                                   view(SelectionQueryModelTransforms),
                                                   output_view,
                                                   view(SelectionQueryPolygonMask));
                }
            } catch (const std::exception& e) {
                return std::unexpected(std::format("VkSplat selection query failed: {}", e.what()));
            }
        }

        {
            LOG_TIMER("VksplatViewportRenderer::buildSelectionMask.dispatch.cuda_wait");
            if (!selection_query_timeline_.cuda_semaphore.cudaWait(selection_query_complete_value,
                                                                   selection_query_stream)) {
                return std::unexpected(std::format("VkSplat selection query completion wait failed: {}",
                                                   selection_query_timeline_.cuda_semaphore.lastError()));
            }
        }

        {
            LOG_TIMER("VksplatViewportRenderer::buildSelectionMask.output_tensor");
            if (!slot.output_tensor.is_valid() || slot.output_tensor.numel() != num_splats) {
                return std::unexpected("VkSplat selection output tensor became invalid after dispatch");
            }
        }

        return slot.output_tensor;
    }

    std::expected<VksplatViewportRenderer::RenderResult, std::string>
    VksplatViewportRenderer::rerenderSelectionOverlay(
        VulkanContext& context,
        const lfs::core::SplatData& splat_data,
        const lfs::rendering::ViewportRenderRequest& request,
        const OutputSlot output_slot,
        const bool synchronize_input_read) {
        const glm::ivec2 size = request.frame_view.size;
        if (size.x <= 0 || size.y <= 0) {
            return std::unexpected("VkSplat selection overlay received an invalid viewport size");
        }
        if (request.equirectangular && !request.gut) {
            return std::unexpected("VkSplat equirectangular rendering requires the 3DGUT backend");
        }
        if (!context.externalMemoryInteropEnabled()) {
            return std::unexpected("VkSplat selection overlay requires CUDA/Vulkan external-memory interop");
        }
        if (!initialized_ || context_ != &context) {
            return std::unexpected("VkSplat selection overlay requested without reusable render state");
        }

        const std::size_t ring_slot = acquireRingSlot();
        if (auto ok = waitForRingSlot(ring_slot, "selection overlay"); !ok) {
            return std::unexpected(ok.error());
        }
        {
            LOG_TIMER("vksplat.selection_overlay.ensureOutputImages");
            if (auto ok = ensureOutputImages(context, size, output_slot, ring_slot); !ok) {
                return std::unexpected(ok.error());
            }
        }

        const auto& output = output_slots_[outputSlotIndex(output_slot)][ring_slot];
        if (output.image.image == VK_NULL_HANDLE ||
            output.image.view == VK_NULL_HANDLE ||
            output.depth_image.image == VK_NULL_HANDLE ||
            output.depth_image.view == VK_NULL_HANDLE ||
            output.size != size) {
            return std::unexpected("VkSplat selection overlay requested for an empty or mismatched output slot");
        }

        const auto num_splats = static_cast<std::size_t>(splat_data.size());
        if (num_splats == 0 || buffers_.num_splats != num_splats) {
            return std::unexpected("VkSplat selection overlay cached model does not match the current model");
        }
        if (buffers_.num_indices == 0 ||
            !hasDeviceBuffer(buffers_.sorted_gauss_idx()) ||
            !hasDeviceBuffer(buffers_.tile_ranges) ||
            !hasDeviceBuffer(buffers_.xy_vs) ||
            !hasDeviceBuffer(buffers_.inv_cov_vs_opacity) ||
            !hasDeviceBuffer(buffers_.rgb) ||
            !hasDeviceBuffer(buffers_.depths) ||
            !hasDeviceBuffer(buffers_.overlay_flags)) {
            return std::unexpected("VkSplat selection overlay cached raster state is unavailable");
        }
        if (request.gut &&
            (!hasDeviceBuffer(buffers_.xyz_ws) ||
             !hasDeviceBuffer(buffers_.rotations) ||
             !hasDeviceBuffer(buffers_.scaling_raw) ||
             !hasDeviceBuffer(buffers_.opacity_raw))) {
            return std::unexpected("VkSplat 3DGUT selection overlay cached model inputs are unavailable");
        }

        auto overlay_bindings = [&] {
            LOG_TIMER("vksplat.selection_overlay.uploadOverlayBindings");
            return uploadOverlayBindings(context, request, num_splats, ring_slot);
        }();
        if (!overlay_bindings) {
            return std::unexpected(overlay_bindings.error());
        }

        VulkanGSRendererUniforms uniforms{};
        {
            LOG_TIMER("vksplat.selection_overlay.populateUniforms");
            const int active_sh_degree = effectiveRenderShDegree(splat_data, request.sh_degree);
            const int resident_sh_degree =
                current_input_sh_degree_ >= 0
                    ? std::min(active_sh_degree, current_input_sh_degree_)
                    : active_sh_degree;
            populateVksplatCameraUniforms(uniforms,
                                          request.frame_view,
                                          request.scene,
                                          resident_sh_degree,
                                          renderShNLayoutSlots(resident_sh_degree, current_input_sh_degree_),
                                          buffers_.num_splats,
                                          request.equirectangular,
                                          request.gut,
                                          request.mip_filter);
            uniforms.step = static_cast<std::uint32_t>(modelTransformCount(request.scene.model_transforms));
            uniforms.sort_capacity = static_cast<uint32_t>(
                std::min<std::size_t>(buffers_.num_indices,
                                      static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())));
        }

        std::expected<void, std::string> compose_status;
        const std::uint64_t completion_value = ++render_complete_value_;
        try {
            LOG_TIMER("vksplat.selection_overlay.batch_total");
            auto batch = DeviceGuard(&renderer_,
                                     synchronize_input_read,
                                     render_complete_timeline_,
                                     completion_value);
            {
                LOG_TIMER("vksplat.selection_overlay.record");
                {
                    LOG_TIMER("vksplat.selection_overlay.record.executeRasterizeForward");
                    renderer_.executeRasterizeForward(uniforms,
                                                      buffers_,
                                                      overlay_bindings->selection_mask,
                                                      overlay_bindings->preview_mask,
                                                      overlay_bindings->selection_colors,
                                                      buffers_.overlay_flags.deviceBuffer,
                                                      overlay_bindings->overlay_params,
                                                      overlay_bindings->transform_indices,
                                                      overlay_bindings->model_transforms,
                                                      request.gut,
                                                      overlay_bindings->overlays_active);
                }
                {
                    LOG_TIMER("vksplat.selection_overlay.record.composePixelState");
                    compose_status = composePixelState(
                        context,
                        renderer_.activeCommandBuffer(),
                        uniforms,
                        request.frame_view.background_color,
                        output_slot,
                        ring_slot,
                        request.transparent_background,
                        request.depth_view,
                        request.depth_view_min,
                        request.depth_view_max);
                }
            }
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat selection overlay pass failed: {}", e.what()));
        }
        if (!compose_status) {
            return std::unexpected(compose_status.error());
        }

        ring_completion_values_[ring_slot] = completion_value;
        const auto& updated_output = output_slots_[outputSlotIndex(output_slot)][ring_slot];
        return RenderResult{
            .image = updated_output.image.image,
            .image_view = updated_output.image.view,
            .image_layout = updated_output.layout,
            .generation = updated_output.generation,
            .depth_image = updated_output.depth_image.image,
            .depth_image_view = updated_output.depth_image.view,
            .depth_image_layout = updated_output.depth_layout,
            .depth_generation = updated_output.generation,
            .size = size,
            .flip_y = false,
            .completion_semaphore = render_complete_timeline_,
            .completion_value = completion_value,
        };
    }

    std::expected<VksplatViewportRenderer::RenderResult, std::string> VksplatViewportRenderer::render(
        VulkanContext& context,
        const lfs::core::SplatData& splat_data,
        const lfs::rendering::ViewportRenderRequest& request,
        const bool force_input_upload,
        const OutputSlot output_slot,
        const bool synchronize_input_upload) {
        const glm::ivec2 size = request.frame_view.size;
        if (size.x <= 0 || size.y <= 0) {
            return std::unexpected("VkSplat received an invalid viewport size");
        }
        if (request.equirectangular && !request.gut) {
            return std::unexpected("VkSplat equirectangular rendering requires the 3DGUT backend");
        }
        if (!context.externalMemoryInteropEnabled()) {
            return std::unexpected("VkSplat forward path requires CUDA/Vulkan external-memory interop");
        }

        const int active_sh_degree = effectiveRenderShDegree(splat_data, request.sh_degree);
        if (auto ok = ensureInitialized(context); !ok) {
            return std::unexpected(ok.error());
        }

        drainRetiredScratchBuffers(false);

        const std::size_t ring_slot = acquireRingSlot();
        if (auto ok = waitForRingSlot(ring_slot, "render"); !ok) {
            return std::unexpected(ok.error());
        }
        if (const auto visibility_stats = renderer_.pollDeferredPrimitiveVisibilityStats()) {
            const double ratio = visibility_stats->num_splats == 0
                                     ? 0.0
                                     : static_cast<double>(visibility_stats->visible_count) /
                                           static_cast<double>(visibility_stats->num_splats);
            LOG_PERF("vksplat.render.visible_primitives count={} total={} ratio={:.4f}",
                     visibility_stats->visible_count,
                     visibility_stats->num_splats,
                     ratio);
        }

        auto input_binding = prepareInputs(context,
                                           splat_data,
                                           ring_slot,
                                           force_input_upload,
                                           active_sh_degree,
                                           synchronize_input_upload);
        if (!input_binding) {
            return std::unexpected(input_binding.error());
        }
        // Keep the ring slot's buffer + CUDA import alive across frames so
        // prepareInputs() can hit the resident fast path. Input-slot sort
        // aliasing is disabled in the async path because overwriting the upload
        // buffer would couple render throughput to selection-overlay correctness.
        (void)input_binding;

        VulkanGSRendererUniforms uniforms{};
        {
            LOG_TIMER("vksplat.render.populateUniforms");
            populateVksplatCameraUniforms(uniforms,
                                          request.frame_view,
                                          request.scene,
                                          active_sh_degree,
                                          renderShNLayoutSlots(active_sh_degree, current_input_sh_degree_),
                                          buffers_.num_splats,
                                          request.equirectangular,
                                          request.gut,
                                          request.mip_filter);
            uniforms.step = static_cast<std::uint32_t>(modelTransformCount(request.scene.model_transforms));
        }

        const std::size_t target_sort_capacity = std::max(buffers_.num_indices, buffers_.num_splats);
        // Reserve from the exact measured tile-instance count. num_indices is the
        // exact total from the previous frame's executeCalculateIndexBufferOffset
        // and num_indices_high_water is its running max, so after the first frame
        // the sort buffers are sized to exactly what the tile sort needs. The 4x
        // estimate only seeds the very first frame, before any tile-instance count
        // exists; the shared block grows in place if it turns out to be too small.
        const std::size_t first_frame_estimate =
            buffers_.num_indices_high_water == 0
                ? (buffers_.num_splats > (std::numeric_limits<std::size_t>::max() / 4u)
                       ? buffers_.num_splats
                       : buffers_.num_splats * 4u)
                : 0u;
        // Size the sort buffers to the exact measured peak: num_indices_high_water is
        // the running max tile-instance count, so this is exactly what every frame seen
        // so far needed. A genuine new peak (viewport maximize or a densification burst
        // pushing the count beyond all history) renders one frame clamped to the prior
        // peak, then the next frame grows the block in place (ensureSharedScratchArena).
        // The former unconditional *2 cost ~100 MiB of permanently-committed sort scratch
        // at 5M splats; a rare one-frame regrow during transients is the better trade.
        const std::size_t shared_sort_capacity =
            std::max({buffers_.num_indices, buffers_.num_splats,
                      buffers_.num_indices_high_water, first_frame_estimate});
        const std::size_t num_pixels =
            static_cast<std::size_t>(uniforms.image_width) * static_cast<std::size_t>(uniforms.image_height);
        const std::size_t num_tiles =
            static_cast<std::size_t>(uniforms.grid_width) * static_cast<std::size_t>(uniforms.grid_height);
        bool shared_scratch_bound = false;
        std::optional<RasterizerArenaRenderGuard> shared_arena_guard;

        static const bool kDisableSharedScratch = (std::getenv("LFS_NO_SHARED_SCRATCH") != nullptr);
        if (synchronize_input_upload && !kDisableSharedScratch) {
            // A busy training arena makes this frame fall back to the cached viewport.
            // Do not resize output images until this render is guaranteed to proceed.
            releasePrivateScratchBuffers();
            const std::size_t required_shared_scratch =
                estimateSharedScratchBytes(buffers_.num_splats, shared_sort_capacity, num_pixels, num_tiles);
            if (auto ok = ensureSharedScratchArena(context, required_shared_scratch); ok) {
                try {
                    shared_arena_guard.emplace();
                    // Now that the render owns the arena frame (training is
                    // excluded), it is safe to re-import the block if training grew
                    // it in place since the last frame — the handle/size are stable.
                    if (auto rok = reimportSharedScratchIfGrown(context); !rok) {
                        shared_arena_guard.reset();
                        return std::unexpected(rok.error());
                    }
                    bindSharedScratchBuffers(buffers_.num_splats, shared_sort_capacity, num_pixels, num_tiles);
                    shared_scratch_bound = true;
                    LOG_PERF("vksplat.memory.shared_scratch required={}MiB capacity={}MiB sort_capacity={} splats={}",
                             required_shared_scratch >> 20,
                             shared_scratch_.bytes >> 20,
                             shared_sort_capacity,
                             buffers_.num_splats);
                } catch (const std::exception& e) {
                    shared_arena_guard.reset();
                    detachSharedScratchBuffers();
                    return std::unexpected(std::format(
                        "VkSplat shared scratch activation failed: {}", e.what()));
                }
            } else {
                return std::unexpected(ok.error());
            }
        }

        auto shared_scratch_cleanup = ScopeExit([&]() {
            if (shared_scratch_bound) {
                detachSharedScratchBuffers();
            }
        });

        if (!synchronize_input_upload &&
            renderer_.shrinkSortBuffersForCapacity(buffers_, target_sort_capacity)) {
            LOG_PERF("vksplat.memory.shrink_sort_buffers target_capacity={} splats={}",
                     target_sort_capacity,
                     buffers_.num_splats);
        }

        auto overlay_bindings = [&] {
            LOG_TIMER("vksplat.render.uploadOverlayBindings");
            return uploadOverlayBindings(
                context, request, static_cast<std::size_t>(splat_data.size()), ring_slot);
        }();
        if (!overlay_bindings) {
            return std::unexpected(overlay_bindings.error());
        }
        {
            LOG_TIMER("vksplat.render.ensureOutputImages");
            if (auto ok = ensureOutputImages(context, size, output_slot, ring_slot); !ok) {
                return std::unexpected(ok.error());
            }
        }

        if (input_binding->uses_temporary_upload_slot && !request.gut) {
            const VkBuffer input_buffer = cuda_inputs_[ring_slot].buffer.buffer;
            const auto detach_alias = [input_buffer](auto& buffer) {
                auto& device_buffer = buffer.deviceBuffer;
                if (device_buffer.buffer == input_buffer && device_buffer.allocation == VK_NULL_HANDLE) {
                    device_buffer = {};
                }
            };
            detach_alias(buffers_.sorting_keys_1);
            detach_alias(buffers_.sorting_keys_2);
            detach_alias(buffers_.sorting_gauss_idx_1);
            detach_alias(buffers_.sorting_gauss_idx_2);
        }

        std::expected<void, std::string> compose_status;
        const std::uint64_t completion_value = ++render_complete_value_;
        try {
            // Timer/guard ordering trick: the LOG_TIMER for batch_total is
            // declared FIRST so it destructs LAST. The DeviceGuard `batch`
            // destructs first at try-block exit, triggering endCommandBatch().
            // When rendering a live training model, keep the caller's shared
            // render lock held until Vulkan has finished reading the zero-copy
            // tensors. Otherwise CUDA training can mutate scales/opacities for
            // the next iteration while this frame is still in flight.
            LOG_TIMER("vksplat.render.batch_total");
            auto batch = DeviceGuard(&renderer_,
                                     synchronize_input_upload,
                                     render_complete_timeline_,
                                     completion_value);
            {
                LOG_TIMER("vksplat.render.record");
                {
                    LOG_TIMER("vksplat.render.record.executeProjectionForward");
                    renderer_.executeProjectionForward(uniforms,
                                                       buffers_,
                                                       overlay_bindings->transform_indices,
                                                       overlay_bindings->node_mask,
                                                       overlay_bindings->overlay_params,
                                                       overlay_bindings->model_transforms,
                                                       0,
                                                       request.gut);
                }
                // Two-stage sort (Splatshop, matches gsplat_fwd reference):
                //   1. Depth-sort N primitives by radial distance (full 32-bit key).
                //   2. Reorder tiles_touched into depth-rank order so the cumsum
                //      offsets match a depth-ordered emission walk.
                //   3. Stable-sort tile instances by tile id only (no depth bits
                //      packed in), which preserves depth order within each tile.
                {
                    LOG_TIMER("vksplat.render.record.executeSortPrimitivesByDepth");
                    renderer_.executeSortPrimitivesByDepth(uniforms, buffers_);
                }
                {
                    LOG_TIMER("vksplat.render.record.executeApplyDepthOrdering");
                    renderer_.executeApplyDepthOrdering(uniforms, buffers_);
                }
                {
                    LOG_TIMER("vksplat.render.record.executeCalculateIndexBufferOffset");
                    renderer_.executeCalculateIndexBufferOffset(uniforms, buffers_);
                }
                if (shared_scratch_bound && buffers_.num_indices > shared_sort_capacity) {
                    return std::unexpected(std::format(
                        "VkSplat shared scratch sort capacity insufficient: have {}, need {}",
                        shared_sort_capacity,
                        buffers_.num_indices));
                }
                if (buffers_.num_splats > 0) {
                    const double instances_per_splat =
                        static_cast<double>(buffers_.num_indices) /
                        static_cast<double>(buffers_.num_splats);
                    const std::uint32_t grid_width = uniforms.grid_width;
                    const std::uint32_t grid_height = uniforms.grid_height;
                    LOG_PERF("vksplat.render.tile_instances count={} splats={} instances_per_splat={:.3f} grid={}x{}",
                             buffers_.num_indices,
                             buffers_.num_splats,
                             instances_per_splat,
                             grid_width,
                             grid_height);
                }
                uniforms.sort_capacity = static_cast<uint32_t>(
                    std::min<std::size_t>(buffers_.num_indices,
                                          static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())));
                if (buffers_.num_indices > 0) {
                    {
                        LOG_TIMER("vksplat.render.record.executeGenerateKeys");
                        renderer_.executeGenerateKeys(uniforms, buffers_);
                    }
                    // Stage-2 sort bits: ceil(log2(grid_w*grid_h + 1)) rounded up to
                    // the next byte. Real tile ids fit in this range; the sentinel
                    // (grid_w*grid_h) sorts to the end.
                    int tile_bits = 0;
                    uint32_t tile_max = uniforms.grid_width * uniforms.grid_height;
                    while (tile_max) {
                        tile_max >>= 1;
                        ++tile_bits;
                    }
                    {
                        LOG_TIMER("vksplat.render.record.executeSort");
                        renderer_.executeSortTileInstances(uniforms, buffers_, tile_bits);
                    }
                    {
                        LOG_TIMER("vksplat.render.record.executeComputeTileRanges");
                        renderer_.executeComputeTileRanges(uniforms, buffers_);
                    }
                    {
                        LOG_TIMER("vksplat.render.record.executeRasterizeForward");
                        renderer_.executeRasterizeForward(uniforms,
                                                          buffers_,
                                                          overlay_bindings->selection_mask,
                                                          overlay_bindings->preview_mask,
                                                          overlay_bindings->selection_colors,
                                                          buffers_.overlay_flags.deviceBuffer,
                                                          overlay_bindings->overlay_params,
                                                          overlay_bindings->transform_indices,
                                                          overlay_bindings->model_transforms,
                                                          request.gut,
                                                          overlay_bindings->overlays_active);
                    }
                }
                {
                    LOG_TIMER("vksplat.render.record.composePixelState");
                    // Record compose into the rasterizer's batch so the entire frame
                    // submits and waits exactly once instead of fence-blocking twice.
                    compose_status = composePixelState(
                        context,
                        renderer_.activeCommandBuffer(),
                        uniforms,
                        request.frame_view.background_color,
                        output_slot,
                        ring_slot,
                        request.transparent_background,
                        request.depth_view,
                        request.depth_view_min,
                        request.depth_view_max);
                }
            }
            // record/composePixelState timer scope ends here.
            // On try-block exit: `batch` destructs (endCommandBatch fence wait),
            // then batch_total timer logs.
        } catch (const std::exception& e) {
            return std::unexpected(std::format("VkSplat forward pass failed: {}", e.what()));
        }
        logVramBreakdownIfChanged("render");
        if (!compose_status) {
            return std::unexpected(compose_status.error());
        }

        renderer_.tagDeferredVisibleCountReadback(render_complete_timeline_, completion_value);
        ring_completion_values_[ring_slot] = completion_value;

        const auto& output = output_slots_[outputSlotIndex(output_slot)][ring_slot];
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
            .completion_semaphore = render_complete_timeline_,
            .completion_value = completion_value,
        };
    }

} // namespace lfs::vis

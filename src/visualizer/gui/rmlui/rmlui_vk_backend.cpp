/* Adapted from RmlUi 6.2 Backends/RmlUi_Renderer_VK.cpp.
 *
 * SPDX-License-Identifier: MIT */

#include "gui/rmlui/rmlui_vk_backend.hpp"
#include "core/image_io.hpp"
#include "core/logger.hpp"
#include "core/path_utils.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "gui/rmlui/vulkan/rmlui_shaders_spv.hpp"
#include "internal/resource_paths.hpp"
#include "python/python_runtime.hpp"
#include <RmlUi/Core/Core.h>
#include <RmlUi/Core/FileInterface.h>
#include <RmlUi/Core/Log.h>
#include <RmlUi/Core/Math.h>
#include <RmlUi/Core/Profiling.h>
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <format>
#include <limits>
#include <optional>
#include <semaphore>
#include <stb_image.h>
#include <string.h>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>

// AlignUp(314, 256) = 512
template <typename T>
static T AlignUp(T val, T alignment) {
    return (val + alignment - (T)1) & ~(alignment - (T)1);
}

static bool SupportsHostImageCopyDestinationLayout(VkPhysicalDevice physical_device,
                                                   VkImageLayout layout) {
    VkPhysicalDeviceHostImageCopyPropertiesEXT host_copy_properties{};
    host_copy_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_IMAGE_COPY_PROPERTIES_EXT;

    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &host_copy_properties;
    vkGetPhysicalDeviceProperties2(physical_device, &properties);

    std::vector<VkImageLayout> destination_layouts(host_copy_properties.copyDstLayoutCount);
    host_copy_properties.pCopyDstLayouts = destination_layouts.data();
    vkGetPhysicalDeviceProperties2(physical_device, &properties);
    destination_layouts.resize(host_copy_properties.copyDstLayoutCount);

    return std::find(destination_layouts.begin(), destination_layouts.end(), layout) !=
           destination_layouts.end();
}

#ifdef RMLUI_VK_DEBUG
static Rml::String FormatByteSize(VkDeviceSize size) noexcept {
    constexpr VkDeviceSize K = VkDeviceSize(1024);
    if (size < K)
        return Rml::CreateString("%zu B", size);
    else if (size < K * K)
        return Rml::CreateString("%g KB", double(size) / double(K));
    return Rml::CreateString("%g MB", double(size) / double(K * K));
}

static void InsertDebugUtilsLabel(VkDevice device, VkCommandBuffer command_buffer, const VkDebugUtilsLabelEXT& label) noexcept {
    auto* const fn = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(device, "vkCmdInsertDebugUtilsLabelEXT"));
    if (fn)
        fn(command_buffer, &label);
}

#endif

namespace {

    struct PreviewTextureRequest {
        std::filesystem::path path;
        int max_size = 0;
    };

    bool IsPreviewTextureSource(std::string_view source) {
        return source.rfind("preview://", 0) == 0;
    }

    bool IsLfsVkTextureSource(std::string_view source) {
        return source.rfind("lfs-vk://", 0) == 0;
    }

    struct LfsVkTextureRequest {
        VkImageView image_view = VK_NULL_HANDLE;
        VkSampler sampler = VK_NULL_HANDLE;
        int width = 0;
        int height = 0;
    };

    std::optional<std::uintptr_t> ParseHexHandle(std::string_view value) {
        if (value.empty())
            return std::nullopt;
        std::uintptr_t parsed = 0;
        const auto* first = value.data();
        const auto* last = value.data() + value.size();
        const auto [ptr, ec] = std::from_chars(first, last, parsed, 16);
        if (ec != std::errc() || ptr != last)
            return std::nullopt;
        return parsed;
    }

    int ParseIntParam(std::string_view value) {
        int parsed = 0;
        const auto* first = value.data();
        const auto* last = value.data() + value.size();
        const auto [ptr, ec] = std::from_chars(first, last, parsed);
        if (ec != std::errc() || ptr != last)
            return 0;
        return std::max(parsed, 0);
    }

    std::optional<LfsVkTextureRequest> ParseLfsVkTextureRequest(std::string_view source) {
        if (!IsLfsVkTextureSource(source))
            return std::nullopt;
        source.remove_prefix(std::string_view("lfs-vk://").size());
        if (!source.empty() && source.front() == '?')
            source.remove_prefix(1);

        LfsVkTextureRequest request;
        while (!source.empty()) {
            const size_t sep = source.find('&');
            const std::string_view part = sep == std::string_view::npos ? source : source.substr(0, sep);
            if (const size_t eq = part.find('='); eq != std::string_view::npos) {
                const std::string_view key = part.substr(0, eq);
                const std::string_view value = part.substr(eq + 1);
                if (key == "v") {
                    if (const auto h = ParseHexHandle(value))
                        request.image_view = reinterpret_cast<VkImageView>(*h);
                } else if (key == "s") {
                    if (const auto h = ParseHexHandle(value))
                        request.sampler = reinterpret_cast<VkSampler>(*h);
                } else if (key == "w") {
                    request.width = ParseIntParam(value);
                } else if (key == "h") {
                    request.height = ParseIntParam(value);
                }
            }
            if (sep == std::string_view::npos)
                break;
            source.remove_prefix(sep + 1);
        }
        if (request.image_view == VK_NULL_HANDLE || request.sampler == VK_NULL_HANDLE ||
            request.width <= 0 || request.height <= 0)
            return std::nullopt;
        return request;
    }

    int HexValue(char c) {
        if (c >= '0' && c <= '9')
            return c - '0';
        if (c >= 'a' && c <= 'f')
            return 10 + c - 'a';
        if (c >= 'A' && c <= 'F')
            return 10 + c - 'A';
        return -1;
    }

    std::string PercentDecode(std::string_view value) {
        std::string decoded;
        decoded.reserve(value.size());
        for (size_t i = 0; i < value.size(); ++i) {
            if (value[i] == '%' && i + 2 < value.size()) {
                const int hi = HexValue(value[i + 1]);
                const int lo = HexValue(value[i + 2]);
                if (hi >= 0 && lo >= 0) {
                    decoded.push_back(static_cast<char>((hi << 4) | lo));
                    i += 2;
                    continue;
                }
            }
            decoded.push_back(value[i] == '+' ? ' ' : value[i]);
        }
        return decoded;
    }

    std::optional<PreviewTextureRequest> ParsePreviewTextureRequest(std::string_view source) {
        if (!IsPreviewTextureSource(source))
            return std::nullopt;

        source.remove_prefix(std::string_view("preview://").size());
        PreviewTextureRequest request;
        int thumb_size = 0;
        int preview_size = 0;
        int dataset_size = 0;

        while (!source.empty()) {
            const size_t sep = source.find('&');
            const std::string_view part = sep == std::string_view::npos ? source : source.substr(0, sep);
            if (const size_t eq = part.find('='); eq != std::string_view::npos) {
                const std::string_view key = part.substr(0, eq);
                const std::string_view value = part.substr(eq + 1);
                if (key == "path") {
                    request.path = lfs::core::utf8_to_path(PercentDecode(value));
                } else if (key == "thumb") {
                    thumb_size = ParseIntParam(value);
                } else if (key == "pmw") {
                    preview_size = ParseIntParam(value);
                } else if (key == "mw") {
                    dataset_size = ParseIntParam(value);
                }
            }

            if (sep == std::string_view::npos)
                break;
            source.remove_prefix(sep + 1);
        }

        if (request.path.empty())
            return std::nullopt;

        request.max_size = thumb_size > 0 ? thumb_size : (preview_size > 0 ? preview_size : dataset_size);
        return request;
    }

    void RecordRmlUiVram(std::string_view scope, std::string_view label, VkDeviceSize bytes) {
        lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
            scope,
            label,
            static_cast<std::size_t>(bytes));
    }

    std::string TextureVramLabel(std::string_view prefix,
                                 std::string_view name,
                                 int width,
                                 int height,
                                 const void* ptr) {
        if (name.empty()) {
            name = "unnamed";
        }
        return std::format("{}:{}:{}x{}@0x{:x}",
                           prefix,
                           name,
                           width,
                           height,
                           reinterpret_cast<std::uintptr_t>(ptr));
    }

} // namespace

RenderInterface_VK::RenderInterface_VK() : m_is_transform_enabled{false},
                                           m_is_apply_to_regular_geometry_stencil{false},
                                           m_is_clip_mask_enabled{false},
                                           m_is_transformed_scissor_enabled{false},
                                           m_is_use_scissor_specified{false},
                                           m_is_use_stencil_pipeline{false},
                                           m_width{},
                                           m_height{},
                                           m_queue_index_graphics{},
                                           m_p_instance{},
                                           m_p_device{},
                                           m_p_physical_device{},
                                           m_p_pipeline_cache{},
                                           m_p_allocator{},
                                           m_p_current_command_buffer{},
                                           m_p_descriptor_set_layout_vertex_transform{},
                                           m_p_descriptor_set_layout_texture{},
                                           m_p_pipeline_layout{},
                                           m_p_pipeline_with_textures{},
                                           m_p_pipeline_without_textures{},
                                           m_p_pipeline_stencil_for_region_where_geometry_will_be_drawn{},
                                           m_p_pipeline_stencil_for_regular_geometry_that_applied_to_region_with_textures{},
                                           m_p_pipeline_stencil_for_regular_geometry_that_applied_to_region_without_textures{},
                                           m_p_descriptor_set{},
                                           m_p_sampler_linear{},
                                           m_p_sampler_nearest{},
                                           m_scissor{},
                                           m_scissor_original{},
                                           m_viewport{},
                                           m_p_queue_graphics{},
                                           m_swapchain_format{},
                                           m_depth_stencil_format{},
                                           m_pending_for_deletion_textures_by_frames{},
                                           m_live_textures{},
                                           m_render_layers{},
                                           m_external_swapchain_image{},
                                           m_external_swapchain_image_view{},
                                           m_external_depth_stencil_image_view{},
                                           m_external_swapchain_layout{VK_IMAGE_LAYOUT_UNDEFINED},
                                           m_active_render_target{active_render_target_t::None},
                                           m_active_layer{},
                                           m_render_layer_stack_size{} {
    m_context_transform = Rml::Matrix4f::Identity();
    m_rml_transform = Rml::Matrix4f::Identity();
}

RenderInterface_VK::~RenderInterface_VK() {}

std::string RenderInterface_VK::MakeExternalTextureSource(VkImageView image_view, VkSampler sampler,
                                                          int width, int height) {
    if (image_view == VK_NULL_HANDLE || sampler == VK_NULL_HANDLE || width <= 0 || height <= 0)
        return {};
    char buf[160];
    std::snprintf(buf, sizeof(buf),
                  "lfs-vk://?v=%llx&s=%llx&w=%d&h=%d",
                  static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(image_view)),
                  static_cast<unsigned long long>(reinterpret_cast<std::uintptr_t>(sampler)),
                  width, height);
    return std::string(buf);
}

void RenderInterface_VK::SetTextureDebugName(const Rml::TextureHandle texture_handle,
                                             const std::string_view debug_name) const {
    const auto* texture = reinterpret_cast<const texture_data_t*>(texture_handle);
    if (!texture || debug_name.empty())
        return;

    const std::string image_name = std::format("rmlui.cache[{}].image", debug_name);
    (void)m_debug_name_writer.set(VK_OBJECT_TYPE_IMAGE,
                                  (uint64_t)texture->m_p_vk_image,
                                  image_name.c_str());
    const std::string view_name = std::format("rmlui.cache[{}].view", debug_name);
    (void)m_debug_name_writer.set(VK_OBJECT_TYPE_IMAGE_VIEW,
                                  (uint64_t)texture->m_p_vk_image_view,
                                  view_name.c_str());
}

Rml::CompiledGeometryHandle RenderInterface_VK::CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices) {
    RMLUI_ZoneScopedN("Vulkan - CompileGeometry");

    VkDescriptorSet p_current_descriptor_set = nullptr;
    p_current_descriptor_set = m_p_descriptor_set;

    RMLUI_VK_ASSERTMSG(p_current_descriptor_set,
                       "you can't have here an invalid pointer of VkDescriptorSet. Two reason might be. 1. - you didn't allocate it "
                       "at all or 2. - Somehing is wrong with allocation and somehow it was corrupted by something.");

    auto* p_geometry_handle = new geometry_handle_t{};

    uint32_t* pCopyDataToBuffer = nullptr;
    const void* pData = reinterpret_cast<const void*>(vertices.data());

    bool status = m_memory_pool.Alloc_VertexBuffer((uint32_t)vertices.size(), sizeof(Rml::Vertex), reinterpret_cast<void**>(&pCopyDataToBuffer),
                                                   &p_geometry_handle->m_p_vertex, &p_geometry_handle->m_p_vertex_allocation);
    RMLUI_VK_ASSERTMSG(status, "failed to AllocVertexBuffer");

    memcpy(pCopyDataToBuffer, pData, sizeof(Rml::Vertex) * vertices.size());

    status = m_memory_pool.Alloc_IndexBuffer((uint32_t)indices.size(), sizeof(int), reinterpret_cast<void**>(&pCopyDataToBuffer),
                                             &p_geometry_handle->m_p_index, &p_geometry_handle->m_p_index_allocation);
    RMLUI_VK_ASSERTMSG(status, "failed to AllocIndexBuffer");

    memcpy(pCopyDataToBuffer, indices.data(), sizeof(int) * indices.size());

    p_geometry_handle->m_num_indices = (int)indices.size();

    return Rml::CompiledGeometryHandle(p_geometry_handle);
}

void RenderInterface_VK::RenderGeometry(Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation, Rml::TextureHandle texture) {
    RMLUI_ZoneScopedN("Vulkan - RenderCompiledGeometry");

    if (m_p_current_command_buffer == nullptr)
        return;

    RMLUI_VK_ASSERTMSG(m_p_current_command_buffer, "must be valid otherwise you can't render now!!! (can't be)");

    texture_data_t* p_texture = reinterpret_cast<texture_data_t*>(texture);
    if (p_texture && p_texture->m_is_async_preview)
        m_current_context_used_preview_texture = true;

    VkDescriptorImageInfo info_descriptor_image = {};
    if (p_texture && p_texture->m_p_vk_descriptor_set == nullptr) {
        VkDescriptorSet p_texture_set = nullptr;
        m_manager_descriptors.Alloc_Descriptor(m_p_device, &m_p_descriptor_set_layout_texture, &p_texture_set);

        info_descriptor_image.imageView = p_texture->m_p_vk_image_view;
        info_descriptor_image.sampler = p_texture->m_p_vk_sampler;
        info_descriptor_image.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkWriteDescriptorSet info_write = {};

        info_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        info_write.dstSet = p_texture_set;
        info_write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        info_write.dstBinding = 2;
        info_write.pImageInfo = &info_descriptor_image;
        info_write.descriptorCount = 1;

        vkUpdateDescriptorSets(m_p_device, 1, &info_write, 0, nullptr);
        p_texture->m_p_vk_descriptor_set = p_texture_set;
    }

    geometry_handle_t* p_casted_compiled_geometry = reinterpret_cast<geometry_handle_t*>(geometry);

    m_user_data_for_vertex_shader.m_translate = translation;

    VkDescriptorSet p_current_descriptor_set = nullptr;
    p_current_descriptor_set = m_p_descriptor_set;

    RMLUI_VK_ASSERTMSG(p_current_descriptor_set,
                       "you can't have here an invalid pointer of VkDescriptorSet. Two reason might be. 1. - you didn't allocate it "
                       "at all or 2. - Somehing is wrong with allocation and somehow it was corrupted by something.");

    shader_vertex_user_data_t* p_data = nullptr;
    VkDescriptorBufferInfo shader_buffer = {};
    VmaVirtualAllocation shader_allocation = {};
    // Dynamic uniform offsets are consumed when the recorded command buffer executes, so keep
    // per-draw transform data alive until the owning frame slot's fence has completed.
    bool status = m_memory_pool.Alloc_GeneralBuffer(sizeof(m_user_data_for_vertex_shader), reinterpret_cast<void**>(&p_data),
                                                    &shader_buffer, &shader_allocation);
    RMLUI_VK_ASSERTMSG(status, "failed to allocate VkDescriptorBufferInfo for uniform data to shaders");
    m_transient_shader_allocations_by_frame[ActiveResourceSlot()].push_back(shader_allocation);

    if (p_data) {
        p_data->m_transform = m_user_data_for_vertex_shader.m_transform;
        p_data->m_translate = m_user_data_for_vertex_shader.m_translate;
    } else {
        RMLUI_VK_ASSERTMSG(p_data, "you can't reach this zone, it means something bad");
    }

    const uint32_t pDescriptorOffsets = static_cast<uint32_t>(shader_buffer.offset);

    VkDescriptorSet p_texture_descriptor_set = nullptr;

    if (p_texture) {
        p_texture_descriptor_set = p_texture->m_p_vk_descriptor_set;
    }

    VkDescriptorSet p_sets[] = {p_current_descriptor_set, p_texture_descriptor_set};
    int real_size_of_sets = 2;

    if (p_texture == nullptr)
        real_size_of_sets = 1;

    vkCmdBindDescriptorSets(m_p_current_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_p_pipeline_layout, 0, real_size_of_sets, p_sets, 1,
                            &pDescriptorOffsets);

    if (m_is_use_stencil_pipeline) {
        vkCmdBindPipeline(m_p_current_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_p_pipeline_stencil_for_region_where_geometry_will_be_drawn);
    } else {
        if (p_texture) {
            if (m_is_apply_to_regular_geometry_stencil) {
                vkCmdBindPipeline(m_p_current_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  m_p_pipeline_stencil_for_regular_geometry_that_applied_to_region_with_textures);
            } else {
                vkCmdBindPipeline(m_p_current_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_p_pipeline_with_textures);
            }
        } else {
            if (m_is_apply_to_regular_geometry_stencil) {
                vkCmdBindPipeline(m_p_current_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                  m_p_pipeline_stencil_for_regular_geometry_that_applied_to_region_without_textures);
            } else {
                vkCmdBindPipeline(m_p_current_command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, m_p_pipeline_without_textures);
            }
        }
    }

    vkCmdBindVertexBuffers(m_p_current_command_buffer, 0, 1, &p_casted_compiled_geometry->m_p_vertex.buffer,
                           &p_casted_compiled_geometry->m_p_vertex.offset);

    vkCmdBindIndexBuffer(m_p_current_command_buffer, p_casted_compiled_geometry->m_p_index.buffer, p_casted_compiled_geometry->m_p_index.offset,
                         VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(m_p_current_command_buffer, p_casted_compiled_geometry->m_num_indices, 1, 0, 0, 0);
}

void RenderInterface_VK::ReleaseGeometry(Rml::CompiledGeometryHandle geometry) {
    RMLUI_ZoneScopedN("Vulkan - ReleaseCompiledGeometry");

    geometry_handle_t* p_casted_geometry = reinterpret_cast<geometry_handle_t*>(geometry);

    m_pending_for_deletion_geometries_by_frame[ActiveResourceSlot()].push_back(p_casted_geometry);
}

void RenderInterface_VK::EnableScissorRegion(bool enable) {
    if (m_p_current_command_buffer == nullptr)
        return;

    m_is_use_scissor_specified = enable;

    if (m_is_use_scissor_specified == false) {
        m_is_transformed_scissor_enabled = false;
        m_is_apply_to_regular_geometry_stencil = m_is_clip_mask_enabled;
        m_scissor = ContextClipScissor();
        vkCmdSetScissor(m_p_current_command_buffer, 0, 1, &m_scissor);
    }
}

void RenderInterface_VK::SetScissorRegion(Rml::Rectanglei region) {
    if (m_is_use_scissor_specified) {
        if (m_is_transform_enabled) {
            Rml::Vertex vertices[4];

            vertices[0].position = Rml::Vector2f(region.TopLeft());
            vertices[1].position = Rml::Vector2f(region.TopRight());
            vertices[2].position = Rml::Vector2f(region.BottomRight());
            vertices[3].position = Rml::Vector2f(region.BottomLeft());

            int indices[6] = {0, 2, 1, 0, 3, 2};

            m_is_use_stencil_pipeline = true;
            m_scissor = ContextClipScissor();
            vkCmdSetScissor(m_p_current_command_buffer, 0, 1, &m_scissor);

#ifdef RMLUI_VK_DEBUG
            VkDebugUtilsLabelEXT info{};
            info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
            info.color[0] = 1.0f;
            info.color[1] = 1.0f;
            info.color[2] = 0.0f;
            info.color[3] = 1.0f;
            info.pLabelName = "SetScissorRegion (generated region)";

            InsertDebugUtilsLabel(m_p_device, m_p_current_command_buffer, info);
#endif

            VkClearDepthStencilValue info_clear_color{};

            info_clear_color.depth = 1.0f;
            info_clear_color.stencil = 0;

            VkClearAttachment clear_attachment = {};
            clear_attachment.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
            clear_attachment.clearValue.depthStencil = info_clear_color;
            clear_attachment.colorAttachment = 1;

            VkClearRect clear_rect = {};
            clear_rect.layerCount = 1;
            clear_rect.rect.extent.width = m_width;
            clear_rect.rect.extent.height = m_height;

            vkCmdClearAttachments(m_p_current_command_buffer, 1, &clear_attachment, 1, &clear_rect);

            if (Rml::CompiledGeometryHandle handle = CompileGeometry({vertices, 4}, {indices, 6})) {
                RenderGeometry(handle, {}, {});
                ReleaseGeometry(handle);
            }

            m_is_use_stencil_pipeline = false;

            m_is_transformed_scissor_enabled = true;
            m_is_apply_to_regular_geometry_stencil = true;
            m_scissor = ContextClipScissor();
            vkCmdSetScissor(m_p_current_command_buffer, 0, 1, &m_scissor);
        } else {
            m_is_transformed_scissor_enabled = false;
            m_is_apply_to_regular_geometry_stencil = m_is_clip_mask_enabled;
            // Enclose the translated rect; fractional panel offsets otherwise clip text edges.
            const float left_f = static_cast<float>(region.Left()) + m_context_offset.x;
            const float top_f = static_cast<float>(region.Top()) + m_context_offset.y;
            const float right_f = left_f + static_cast<float>(region.Width());
            const float bottom_f = top_f + static_cast<float>(region.Height());
            const int left = Rml::Math::Clamp(static_cast<int>(std::floor(left_f)), 0, m_width);
            const int top = Rml::Math::Clamp(static_cast<int>(std::floor(top_f)), 0, m_height);
            const int right = Rml::Math::Clamp(static_cast<int>(std::ceil(right_f)), 0, m_width);
            const int bottom = Rml::Math::Clamp(static_cast<int>(std::ceil(bottom_f)), 0, m_height);
            m_scissor.offset.x = left;
            m_scissor.offset.y = top;
            m_scissor.extent.width = static_cast<uint32_t>(std::max(0, right - left));
            m_scissor.extent.height = static_cast<uint32_t>(std::max(0, bottom - top));
            m_scissor = IntersectContextClip(m_scissor);

#ifdef RMLUI_VK_DEBUG
            VkDebugUtilsLabelEXT info{};
            info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
            info.color[0] = 1.0f;
            info.color[1] = 0.0f;
            info.color[2] = 0.0f;
            info.color[3] = 1.0f;
            info.pLabelName = "SetScissorRegion (offset)";

            InsertDebugUtilsLabel(m_p_device, m_p_current_command_buffer, info);
#endif

            vkCmdSetScissor(m_p_current_command_buffer, 0, 1, &m_scissor);
        }
    }
}

void RenderInterface_VK::EnableClipMask(bool enable) {
    m_is_clip_mask_enabled = enable;
    m_is_apply_to_regular_geometry_stencil = m_is_clip_mask_enabled || m_is_transformed_scissor_enabled;
}

void RenderInterface_VK::RenderToClipMask(Rml::ClipMaskOperation operation, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation) {
    if (m_p_current_command_buffer == nullptr || !geometry)
        return;

    VkClearDepthStencilValue clear_depth_stencil{};
    clear_depth_stencil.depth = 1.0f;
    clear_depth_stencil.stencil = operation == Rml::ClipMaskOperation::SetInverse ? 1 : 0;

    VkClearAttachment clear_attachment = {};
    clear_attachment.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    clear_attachment.clearValue.depthStencil = clear_depth_stencil;
    clear_attachment.colorAttachment = 1;

    VkClearRect clear_rect = {};
    clear_rect.layerCount = 1;
    clear_rect.rect.extent.width = m_width;
    clear_rect.rect.extent.height = m_height;

    // Match RmlUi's legacy compatibility behavior for intersect until the Vulkan renderer has
    // stencil increment/decrement pipelines for true nested clip-mask intersections.
    vkCmdClearAttachments(m_p_current_command_buffer, 1, &clear_attachment, 1, &clear_rect);

    m_is_use_stencil_pipeline = true;
    if (operation == Rml::ClipMaskOperation::SetInverse)
        vkCmdSetStencilReference(m_p_current_command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
    else
        vkCmdSetStencilReference(m_p_current_command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, 1);

    RenderGeometry(geometry, translation, {});

    vkCmdSetStencilReference(m_p_current_command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, 1);
    m_is_use_stencil_pipeline = false;
    m_is_clip_mask_enabled = true;
    m_is_apply_to_regular_geometry_stencil = true;
}

Rml::LayerHandle RenderInterface_VK::PushLayer() {
    if (m_p_current_command_buffer == nullptr || m_width <= 0 || m_height <= 0)
        return {};

    const Rml::LayerHandle layer_handle = static_cast<Rml::LayerHandle>(m_render_layer_stack_size + 1);
    EnsureRenderLayer(layer_handle);

    EndActiveRendering();
    BeginLayerRendering(layer_handle, true);

    m_render_layer_stack_size += 1;
    m_active_layer = layer_handle;
    return layer_handle;
}

void RenderInterface_VK::CompositeLayers(Rml::LayerHandle source, Rml::LayerHandle destination, Rml::BlendMode blend_mode,
                                         Rml::Span<const Rml::CompiledFilterHandle> filters) {
    if (m_p_current_command_buffer == nullptr)
        return;

    static bool warned_unsupported_filters = false;
    if (!filters.empty() && !warned_unsupported_filters) {
        Rml::Log::Message(Rml::Log::LT_WARNING, "[Vulkan] RmlUi layer filters are not implemented yet; compositing unfiltered layer.");
        warned_unsupported_filters = true;
    }

    render_layer_t* source_layer = GetRenderLayer(source);
    if (!source_layer) {
        if (CopySwapchainToLayer(destination)) {
            const Rml::LayerHandle top_layer =
                m_render_layer_stack_size > 0 ? static_cast<Rml::LayerHandle>(m_render_layer_stack_size) : Rml::LayerHandle{};
            if (top_layer != 0 && destination != top_layer) {
                EndActiveRendering();
                BeginLayerRendering(top_layer, false);
            }
            return;
        }

        static bool warned_base_layer = false;
        if (!warned_base_layer) {
            Rml::Log::Message(Rml::Log::LT_WARNING, "[Vulkan] Cannot composite from the swapchain base layer yet.");
            warned_base_layer = true;
        }
        return;
    }

    EndActiveRendering();
    TransitionImageLayout(source_layer->m_color.m_p_vk_image, VK_IMAGE_ASPECT_COLOR_BIT, source_layer->m_color_layout,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    source_layer->m_color_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    if (destination == 0)
        BeginSwapchainRendering(VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_LOAD);
    else
        BeginLayerRendering(destination, false);

    RenderFullscreenTexture(source_layer->m_color, blend_mode);

    const Rml::LayerHandle top_layer = m_render_layer_stack_size > 0 ? static_cast<Rml::LayerHandle>(m_render_layer_stack_size) : Rml::LayerHandle{};
    if (top_layer != 0 && destination != top_layer) {
        EndActiveRendering();
        BeginLayerRendering(top_layer, false);
    }
}

void RenderInterface_VK::PopLayer() {
    if (m_render_layer_stack_size <= 0)
        return;

    EndActiveRendering();
    m_render_layer_stack_size -= 1;
    m_active_layer = {};

    if (m_render_layer_stack_size > 0) {
        const Rml::LayerHandle layer_handle = static_cast<Rml::LayerHandle>(m_render_layer_stack_size);
        BeginLayerRendering(layer_handle, false);
        m_active_layer = layer_handle;
    } else {
        BeginSwapchainRendering(VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_LOAD);
    }
}

Rml::TextureHandle RenderInterface_VK::SaveLayerAsTexture() {
    if (m_p_current_command_buffer == nullptr || m_render_layer_stack_size <= 0)
        return {};

    render_layer_t* source_layer = GetRenderLayer(static_cast<Rml::LayerHandle>(m_render_layer_stack_size));
    if (!source_layer)
        return {};
    if (source_layer->width <= 0 || source_layer->height <= 0)
        return {};

    VkRect2D bounds = m_is_use_scissor_specified ? m_scissor : ContextClipScissor();
    bounds = IntersectContextClip(bounds);
    bounds.offset.x = Rml::Math::Clamp(bounds.offset.x, 0, source_layer->width);
    bounds.offset.y = Rml::Math::Clamp(bounds.offset.y, 0, source_layer->height);
    if (bounds.offset.x + static_cast<int>(bounds.extent.width) > source_layer->width)
        bounds.extent.width = static_cast<uint32_t>(source_layer->width - bounds.offset.x);
    if (bounds.offset.y + static_cast<int>(bounds.extent.height) > source_layer->height)
        bounds.extent.height = static_cast<uint32_t>(source_layer->height - bounds.offset.y);
    if (bounds.extent.width == 0 || bounds.extent.height == 0)
        return {};

    EndActiveRendering();

    auto* texture = new texture_data_t{};
    const VkFormat format = m_swapchain_format.format;
    const VkExtent3D extent{bounds.extent.width, bounds.extent.height, 1};

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = format;
    image_info.extent = extent;
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VmaAllocationInfo allocation_stats{};
    VkResult status = vmaCreateImage(m_p_allocator,
                                     &image_info,
                                     &allocation_info,
                                     &texture->m_p_vk_image,
                                     &texture->m_p_vma_allocation,
                                     &allocation_stats);
    RMLUI_VK_ASSERTMSG(status == VK_SUCCESS, "failed to create saved RmlUi layer texture");
    if (status != VK_SUCCESS) {
        delete texture;
        return {};
    }
    (void)m_debug_name_writer.set(VK_OBJECT_TYPE_IMAGE,
                                  (uint64_t)texture->m_p_vk_image,
                                  "rmlui.saved-layer.image");
    texture->m_vram_scope = "vulkan.rmlui.saved_layer_texture";
    texture->m_vram_label = TextureVramLabel("saved_layer",
                                             "clip",
                                             static_cast<int>(bounds.extent.width),
                                             static_cast<int>(bounds.extent.height),
                                             texture);
    texture->m_vram_allocation_size = allocation_stats.size;
    RecordRmlUiVram(texture->m_vram_scope, texture->m_vram_label, texture->m_vram_allocation_size);

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = texture->m_p_vk_image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;
    status = vkCreateImageView(m_p_device, &view_info, nullptr, &texture->m_p_vk_image_view);
    RMLUI_VK_ASSERTMSG(status == VK_SUCCESS, "failed to create saved RmlUi layer texture view");
    if (status != VK_SUCCESS) {
        if (!texture->m_vram_scope.empty() && !texture->m_vram_label.empty())
            RecordRmlUiVram(texture->m_vram_scope, texture->m_vram_label, 0);
        vmaDestroyImage(m_p_allocator, texture->m_p_vk_image, texture->m_p_vma_allocation);
        delete texture;
        return {};
    }
    (void)m_debug_name_writer.set(VK_OBJECT_TYPE_IMAGE_VIEW,
                                  (uint64_t)texture->m_p_vk_image_view,
                                  "rmlui.saved-layer.view");
    texture->m_p_vk_sampler = m_p_sampler_linear;

    TransitionImageLayout(source_layer->m_color.m_p_vk_image, VK_IMAGE_ASPECT_COLOR_BIT, source_layer->m_color_layout,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    source_layer->m_color_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;

    TransitionImageLayout(texture->m_p_vk_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkImageCopy copy_region{};
    copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.srcSubresource.mipLevel = 0;
    copy_region.srcSubresource.baseArrayLayer = 0;
    copy_region.srcSubresource.layerCount = 1;
    copy_region.srcOffset = {bounds.offset.x, bounds.offset.y, 0};
    copy_region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.dstSubresource.mipLevel = 0;
    copy_region.dstSubresource.baseArrayLayer = 0;
    copy_region.dstSubresource.layerCount = 1;
    copy_region.dstOffset = {0, 0, 0};
    copy_region.extent = extent;

    vkCmdCopyImage(m_p_current_command_buffer, source_layer->m_color.m_p_vk_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   texture->m_p_vk_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

    TransitionImageLayout(texture->m_p_vk_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    BeginLayerRendering(static_cast<Rml::LayerHandle>(m_render_layer_stack_size), false);

    RegisterLiveTexture(texture);
    return reinterpret_cast<Rml::TextureHandle>(texture);
}

// Set to byte packing, or the compiler will expand our struct, which means it won't read correctly from file
#pragma pack(1)
struct TGAHeader {
    char idLength;
    char colourMapType;
    char dataType;
    short int colourMapOrigin;
    short int colourMapLength;
    char colourMapDepth;
    short int xOrigin;
    short int yOrigin;
    short int width;
    short int height;
    char bitsPerPixel;
    char imageDescriptor;
};
// Restore packing
#pragma pack()

Rml::TextureHandle RenderInterface_VK::LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source) {
    if (IsPreviewTextureSource(source))
        return LoadAsyncPreviewTexture(texture_dimensions, source);

    if (auto vk_request = ParseLfsVkTextureRequest(source)) {
        auto* texture = new texture_data_t{};
        texture->m_p_vk_image = VK_NULL_HANDLE;
        texture->m_p_vk_image_view = vk_request->image_view;
        texture->m_p_vk_sampler = vk_request->sampler;
        texture->m_p_vk_descriptor_set = VK_NULL_HANDLE;
        texture->m_p_vma_allocation = VK_NULL_HANDLE;
        texture_dimensions.x = vk_request->width;
        texture_dimensions.y = vk_request->height;
        RegisterLiveTexture(texture);
        return reinterpret_cast<Rml::TextureHandle>(texture);
    }

    auto load_with_stbi = [&](const std::string& path) -> Rml::TextureHandle {
        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned char* data = stbi_load(path.c_str(), &width, &height, &channels, 4);
        if (!data)
            return 0;

        texture_dimensions.x = width;
        texture_dimensions.y = height;

        // Probe the file's native channel count. Single-channel images are masks:
        // stbi expands them to (gray, gray, gray, 255) which renders as fully-opaque
        // gray over the underlying image — useless as an overlay. The mask branch
        // below instead drives the alpha from the gray value so an image-color CSS
        // tint paints only the foreground.
        int probe_w = 0, probe_h = 0, probe_channels = 0;
        const bool is_single_channel =
            stbi_info(path.c_str(), &probe_w, &probe_h, &probe_channels) &&
            probe_channels == 1;

        const int pixel_count = width * height;
        if (is_single_channel) {
            // Mask: the gray value becomes the alpha. RGB is set to the same value so
            // the texture is premultiplied (matching the non-mask branch below and the
            // backend's premultiplied-alpha blend), which keeps alpha==0 pixels fully
            // transparent instead of adding their color over the image.
            for (int i = 0; i < pixel_count; ++i) {
                unsigned char* p = data + i * 4;
                const unsigned char gray = p[0];
                p[0] = gray;
                p[1] = gray;
                p[2] = gray;
                p[3] = gray;
            }
        } else {
            for (int i = 0; i < pixel_count; ++i) {
                unsigned char* p = data + i * 4;
                const unsigned int alpha = p[3];
                p[0] = static_cast<unsigned char>((p[0] * alpha + 127) / 255);
                p[1] = static_cast<unsigned char>((p[1] * alpha + 127) / 255);
                p[2] = static_cast<unsigned char>((p[2] * alpha + 127) / 255);
            }
        }

        const size_t image_size = static_cast<size_t>(width) * static_cast<size_t>(height) * 4;
        const Rml::TextureHandle handle = CreateTexture({data, image_size}, texture_dimensions, path);
        stbi_image_free(data);
        return handle;
    };

    if (const Rml::TextureHandle handle = load_with_stbi(source); handle)
        return handle;

    auto load_asset_fallback = [&](std::string asset_name) -> Rml::TextureHandle {
        while (asset_name.rfind("../", 0) == 0)
            asset_name.erase(0, 3);
        while (asset_name.rfind("./", 0) == 0)
            asset_name.erase(0, 2);
        if (asset_name.empty())
            return 0;

        try {
            const auto path = lfs::vis::getAssetPath(asset_name);
            if (std::filesystem::exists(path))
                return load_with_stbi(path.string());
        } catch (...) {
        }
        return 0;
    };

    if (const Rml::TextureHandle handle = load_asset_fallback(source); handle)
        return handle;

    const std::string_view source_view(source);
    constexpr std::string_view rmlui_icon_segment = "rmlui/icon/";
    if (const auto pos = source_view.find(rmlui_icon_segment); pos != std::string_view::npos) {
        if (const Rml::TextureHandle handle =
                load_asset_fallback("icon/" + std::string(source_view.substr(pos + rmlui_icon_segment.size())));
            handle)
            return handle;
    }
    constexpr std::string_view icon_segment = "/icon/";
    if (const auto pos = source_view.find(icon_segment); pos != std::string_view::npos) {
        if (const Rml::TextureHandle handle =
                load_asset_fallback("icon/" + std::string(source_view.substr(pos + icon_segment.size())));
            handle)
            return handle;
    }

#ifndef _WIN32
    if (!source.empty() && source[0] != '/' && source.find("://") == Rml::String::npos) {
        const std::string absolute_source = "/" + source;
        if (std::filesystem::exists(absolute_source)) {
            if (const Rml::TextureHandle handle = load_with_stbi(absolute_source); handle)
                return handle;
        }
    }
#endif

    Rml::FileInterface* file_interface = Rml::GetFileInterface();
    Rml::FileHandle file_handle = file_interface->Open(source);
    if (!file_handle) {
        return false;
    }

    file_interface->Seek(file_handle, 0, SEEK_END);
    size_t buffer_size = file_interface->Tell(file_handle);
    file_interface->Seek(file_handle, 0, SEEK_SET);

    if (buffer_size <= sizeof(TGAHeader)) {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Texture file size is smaller than TGAHeader, file is not a valid TGA image.");
        file_interface->Close(file_handle);
        return false;
    }

    using Rml::byte;
    Rml::UniquePtr<byte[]> buffer(new byte[buffer_size]);
    file_interface->Read(buffer.get(), buffer_size, file_handle);
    file_interface->Close(file_handle);

    TGAHeader header;
    memcpy(&header, buffer.get(), sizeof(TGAHeader));

    int color_mode = header.bitsPerPixel / 8;
    const size_t image_size = header.width * header.height * 4; // We always make 32bit textures

    if (header.dataType != 2) {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Only 24/32bit uncompressed TGAs are supported.");
        return false;
    }

    // Ensure we have at least 3 colors
    if (color_mode < 3) {
        Rml::Log::Message(Rml::Log::LT_ERROR, "Only 24 and 32bit textures are supported.");
        return false;
    }

    const byte* image_src = buffer.get() + sizeof(TGAHeader);
    Rml::UniquePtr<byte[]> image_dest_buffer(new byte[image_size]);
    byte* image_dest = image_dest_buffer.get();
    const bool top_to_bottom_order = ((header.imageDescriptor & 32) != 0);

    // Targa is BGR, swap to RGB, flip Y axis as necessary, and convert to premultiplied alpha.
    for (long y = 0; y < header.height; y++) {
        long read_index = y * header.width * color_mode;
        long write_index = top_to_bottom_order ? (y * header.width * 4) : (header.height - y - 1) * header.width * 4;
        for (long x = 0; x < header.width; x++) {
            image_dest[write_index] = image_src[read_index + 2];
            image_dest[write_index + 1] = image_src[read_index + 1];
            image_dest[write_index + 2] = image_src[read_index];
            if (color_mode == 4) {
                const byte alpha = image_src[read_index + 3];
                for (size_t j = 0; j < 3; j++)
                    image_dest[write_index + j] = byte((image_dest[write_index + j] * alpha) / 255);
                image_dest[write_index + 3] = alpha;
            } else
                image_dest[write_index + 3] = 255;

            write_index += 4;
            read_index += color_mode;
        }
    }

    texture_dimensions.x = header.width;
    texture_dimensions.y = header.height;

    return CreateTexture({image_dest, image_size}, texture_dimensions, source);
}

Rml::TextureHandle RenderInterface_VK::LoadAsyncPreviewTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source) {
    const auto request = ParsePreviewTextureRequest(source);
    if (!request)
        return 0;

    static constexpr Rml::byte transparent_pixel[4] = {0, 0, 0, 0};
    texture_dimensions = {1, 1};
    const Rml::TextureHandle handle = CreateTexture({transparent_pixel, sizeof(transparent_pixel)}, texture_dimensions, source);
    auto* texture = reinterpret_cast<texture_data_t*>(handle);
    if (!texture)
        return 0;
    texture->m_is_async_preview = true;

    auto state = std::make_shared<async_preview_state_t>();
    state->texture = texture;
    state->source = source;
    m_async_preview_textures.push_back(state);

    try {
        std::thread([state, path = request->path, max_size = request->max_size]() mutable {
            auto result = DecodePreviewTexture(std::move(path), max_size);
            {
                std::lock_guard lock(state->mutex);
                state->result = std::move(result);
            }
            state->ready.store(true, std::memory_order_release);
            lfs::python::request_redraw();
        }).detach();
    } catch (const std::exception& e) {
        LOG_WARN("Failed to start async preview texture worker for '{}': {}", lfs::core::path_to_utf8(request->path), e.what());
        DropAsyncPreviewTexture(texture);
    }

    return handle;
}

RenderInterface_VK::async_preview_result_t RenderInterface_VK::DecodePreviewTexture(std::filesystem::path path, const int max_size) {
    static std::counting_semaphore<4> decode_slots(4);
    struct DecodeSlotGuard {
        std::counting_semaphore<4>& slots;
        explicit DecodeSlotGuard(std::counting_semaphore<4>& s) : slots(s) { slots.acquire(); }
        ~DecodeSlotGuard() { slots.release(); }
    };

    DecodeSlotGuard guard(decode_slots);

    async_preview_result_t result;
    unsigned char* data = nullptr;
    try {
        auto [pixels, width, height, channels] = lfs::core::load_image(path, -1, max_size);
        data = pixels;
        if (pixels && width > 0 && height > 0 && channels > 0) {
            const std::size_t pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
            result.pixels.resize(pixel_count * 4u);
            for (std::size_t i = 0; i < pixel_count; ++i) {
                const unsigned char* src = pixels + i * static_cast<std::size_t>(channels);
                const unsigned char r = src[0];
                const unsigned char g = channels > 1 ? src[1] : r;
                const unsigned char b = channels > 2 ? src[2] : r;
                const unsigned char a = channels > 3 ? src[3] : 255;
                Rml::byte* dst = result.pixels.data() + i * 4u;
                dst[0] = static_cast<Rml::byte>((static_cast<unsigned int>(r) * a + 127u) / 255u);
                dst[1] = static_cast<Rml::byte>((static_cast<unsigned int>(g) * a + 127u) / 255u);
                dst[2] = static_cast<Rml::byte>((static_cast<unsigned int>(b) * a + 127u) / 255u);
                dst[3] = static_cast<Rml::byte>(a);
            }
            result.width = width;
            result.height = height;
        }
    } catch (const std::exception& e) {
        LOG_WARN("Failed to decode preview texture '{}': {}", lfs::core::path_to_utf8(path), e.what());
    }
    if (data)
        lfs::core::free_image(data);
    return result;
}

void RenderInterface_VK::QueueTextureForDeferredDeletion(texture_data_t* texture) {
    if (texture) {
        UnregisterLiveTexture(texture);
        m_pending_for_deletion_textures_by_frames[ActiveResourceSlot()].push_back(texture);
    }
}

void RenderInterface_VK::DropAsyncPreviewTexture(texture_data_t* texture) {
    if (!texture)
        return;
    m_async_preview_textures.erase(std::remove_if(m_async_preview_textures.begin(), m_async_preview_textures.end(),
                                                  [texture](const std::shared_ptr<async_preview_state_t>& state) {
                                                      if (state && state->texture == texture) {
                                                          state->texture = nullptr;
                                                          return true;
                                                      }
                                                      return false;
                                                  }),
                                   m_async_preview_textures.end());
}

void RenderInterface_VK::ProcessAsyncPreviewUploads() {
    if (m_async_preview_textures.empty() || !m_p_device || !m_p_allocator)
        return;

    constexpr int kUploadBudgetPerFrame = 2;
    int uploads_remaining = kUploadBudgetPerFrame;
    bool uploaded_any = false;
    bool needs_followup_redraw = false;

    for (auto it = m_async_preview_textures.begin(); it != m_async_preview_textures.end();) {
        const std::shared_ptr<async_preview_state_t>& state = *it;
        if (!state || !state->texture) {
            it = m_async_preview_textures.erase(it);
            continue;
        }
        if (!state->ready.load(std::memory_order_acquire)) {
            ++it;
            continue;
        }
        if (uploads_remaining <= 0) {
            needs_followup_redraw = true;
            break;
        }

        async_preview_result_t result;
        {
            std::lock_guard lock(state->mutex);
            result = std::move(state->result);
        }

        if (!result.pixels.empty() && result.width > 0 && result.height > 0) {
            Rml::Vector2i dimensions{result.width, result.height};
            const Rml::TextureHandle replacement_handle = CreateTexture({result.pixels.data(), result.pixels.size()}, dimensions, state->source);
            auto* replacement = reinterpret_cast<texture_data_t*>(replacement_handle);
            if (replacement) {
                replacement->m_is_async_preview = true;
                QueueTextureForDeferredDeletion(new texture_data_t(*state->texture));
                *state->texture = *replacement;
                UnregisterLiveTexture(replacement);
                delete replacement;
                --uploads_remaining;
                uploaded_any = true;
            }
        }

        it = m_async_preview_textures.erase(it);
    }

    if (uploaded_any)
        m_preview_texture_generation.fetch_add(1, std::memory_order_acq_rel);
    if (needs_followup_redraw)
        lfs::python::request_redraw();
}

Rml::TextureHandle RenderInterface_VK::GenerateTexture(Rml::Span<const Rml::byte> source_data, Rml::Vector2i source_dimensions) {
    Rml::String source_name = "generated-texture";
    return CreateTexture(source_data, source_dimensions, source_name, m_p_sampler_nearest);
}

Rml::TextureHandle RenderInterface_VK::CreateTexture(Rml::Span<const Rml::byte> source, Rml::Vector2i dimensions, const Rml::String& name,
                                                     VkSampler sampler) {
    RMLUI_ZoneScopedN("Vulkan - GenerateTexture");

    const int width = dimensions.x;
    const int height = dimensions.y;
    const VkSampler texture_sampler = sampler != VK_NULL_HANDLE ? sampler : m_p_sampler_linear;
    const bool size_overflows =
        width > 0 && height > 0 &&
        static_cast<std::size_t>(width) >
            std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(height) / 4u;
    const std::size_t expected_size =
        width > 0 && height > 0 && !size_overflows
            ? static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4u
            : 0u;
    if (m_p_device == VK_NULL_HANDLE || m_p_allocator == VK_NULL_HANDLE ||
        texture_sampler == VK_NULL_HANDLE || source.data() == nullptr || source.empty() ||
        width <= 0 || height <= 0 || size_overflows || source.size() != expected_size) {
        Rml::Log::Message(Rml::Log::LT_ERROR,
                          "[Vulkan] Refusing invalid RmlUi texture '%s' (%dx%d, %zu bytes).",
                          name.c_str(), width, height, source.size());
        return {};
    }

    const VkDeviceSize image_size = source.size();
    const VkFormat format = VkFormat::VK_FORMAT_R8G8B8A8_UNORM;

    const bool use_host_image_copy =
        m_pfn_copy_memory_to_image != nullptr && m_pfn_transition_image_layout != nullptr;

    VkExtent3D extent_image = {};
    extent_image.width = static_cast<uint32_t>(width);
    extent_image.height = static_cast<uint32_t>(height);
    extent_image.depth = 1;

    auto p_texture = std::make_unique<texture_data_t>();

    VkImageCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.pNext = nullptr;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = extent_image;
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = VK_SAMPLE_COUNT_1_BIT;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = VK_IMAGE_USAGE_SAMPLED_BIT |
                 (use_host_image_copy ? VkImageUsageFlags(VK_IMAGE_USAGE_HOST_TRANSFER_BIT_EXT)
                                      : VkImageUsageFlags(VK_IMAGE_USAGE_TRANSFER_DST_BIT));

    VmaAllocationCreateInfo info_allocation = {};
    info_allocation.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VkImage p_image = VK_NULL_HANDLE;
    VmaAllocation p_allocation = VK_NULL_HANDLE;

    VmaAllocationInfo info_stats = {};
    VkResult status = vmaCreateImage(m_p_allocator, &info, &info_allocation, &p_image, &p_allocation, &info_stats);
    if (status != VK_SUCCESS || p_image == VK_NULL_HANDLE || p_allocation == VK_NULL_HANDLE) {
        Rml::Log::Message(Rml::Log::LT_ERROR,
                          "[Vulkan] Failed to allocate RmlUi texture '%s' (%d).",
                          name.c_str(), static_cast<int>(status));
        return {};
    }

#ifdef RMLUI_VK_DEBUG
    Rml::Log::Message(Rml::Log::LT_DEBUG, "Created texture '%s' [%dx%d, %s]", name.c_str(), dimensions.x, dimensions.y,
                      FormatByteSize(info_stats.size).c_str());
#endif

    p_texture->m_p_vk_image = p_image;
    p_texture->m_p_vma_allocation = p_allocation;
    const std::string image_debug_name = std::format("rmlui.texture.image[{}]", name);
    (void)m_debug_name_writer.set(VK_OBJECT_TYPE_IMAGE,
                                  (uint64_t)p_texture->m_p_vk_image,
                                  image_debug_name.c_str());
    p_texture->m_vram_scope = "vulkan.rmlui.texture";
    p_texture->m_vram_label = TextureVramLabel("texture", name, width, height, p_texture.get());
    p_texture->m_vram_allocation_size = info_stats.size;
    RecordRmlUiVram(p_texture->m_vram_scope, p_texture->m_vram_label, p_texture->m_vram_allocation_size);

    const auto fail_texture = [&](const char* operation, const VkResult result) -> Rml::TextureHandle {
        Rml::Log::Message(Rml::Log::LT_ERROR,
                          "[Vulkan] Failed to %s for RmlUi texture '%s' (%d).",
                          operation, name.c_str(), static_cast<int>(result));
        if (p_texture->m_p_vk_image_view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_p_device, p_texture->m_p_vk_image_view, nullptr);
            p_texture->m_p_vk_image_view = VK_NULL_HANDLE;
        }
        if (!p_texture->m_vram_scope.empty() && !p_texture->m_vram_label.empty()) {
            RecordRmlUiVram(p_texture->m_vram_scope, p_texture->m_vram_label, 0);
        }
        if (p_texture->m_p_vk_image != VK_NULL_HANDLE &&
            p_texture->m_p_vma_allocation != VK_NULL_HANDLE) {
            vmaDestroyImage(m_p_allocator,
                            p_texture->m_p_vk_image,
                            p_texture->m_p_vma_allocation);
            p_texture->m_p_vk_image = VK_NULL_HANDLE;
            p_texture->m_p_vma_allocation = VK_NULL_HANDLE;
        }
        return {};
    };

#ifdef RMLUI_VK_DEBUG
    vmaSetAllocationName(m_p_allocator, p_allocation, name.c_str());
#endif

    if (use_host_image_copy) {
        VkHostImageLayoutTransitionInfoEXT to_sampled{};
        to_sampled.sType = VK_STRUCTURE_TYPE_HOST_IMAGE_LAYOUT_TRANSITION_INFO_EXT;
        to_sampled.image = p_image;
        to_sampled.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_sampled.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_sampled.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_sampled.subresourceRange.baseMipLevel = 0;
        to_sampled.subresourceRange.levelCount = 1;
        to_sampled.subresourceRange.baseArrayLayer = 0;
        to_sampled.subresourceRange.layerCount = 1;
        status = m_pfn_transition_image_layout(m_p_device, 1, &to_sampled);
        if (status != VK_SUCCESS)
            return fail_texture("transition host image for shader reads", status);

        VkMemoryToImageCopyEXT region{};
        region.sType = VK_STRUCTURE_TYPE_MEMORY_TO_IMAGE_COPY_EXT;
        region.pHostPointer = source.data();
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = extent_image;

        VkCopyMemoryToImageInfoEXT copy_info{};
        copy_info.sType = VK_STRUCTURE_TYPE_COPY_MEMORY_TO_IMAGE_INFO_EXT;
        copy_info.dstImage = p_image;
        copy_info.dstImageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        copy_info.regionCount = 1;
        copy_info.pRegions = &region;
        status = m_pfn_copy_memory_to_image(m_p_device, &copy_info);
        if (status != VK_SUCCESS)
            return fail_texture("copy host memory to image", status);
    } else {
        buffer_data_t cpu_buffer = CreateResource_StagingBuffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        if (cpu_buffer.m_p_vk_buffer == VK_NULL_HANDLE ||
            cpu_buffer.m_p_vma_allocation == VK_NULL_HANDLE) {
            return fail_texture("allocate texture staging buffer", VK_ERROR_OUT_OF_HOST_MEMORY);
        }
        void* data = nullptr;
        status = vmaMapMemory(m_p_allocator, cpu_buffer.m_p_vma_allocation, &data);
        if (status != VK_SUCCESS || data == nullptr) {
            DestroyResource_StagingBuffer(cpu_buffer);
            return fail_texture("map texture staging buffer", status);
        }
        memcpy(data, source.data(), static_cast<size_t>(image_size));
        status = vmaFlushAllocation(m_p_allocator,
                                    cpu_buffer.m_p_vma_allocation,
                                    0,
                                    image_size);
        vmaUnmapMemory(m_p_allocator, cpu_buffer.m_p_vma_allocation);
        if (status != VK_SUCCESS) {
            DestroyResource_StagingBuffer(cpu_buffer);
            return fail_texture("flush texture staging buffer", status);
        }

        const bool uploaded = m_upload_manager.UploadToGPU([p_image, extent_image, cpu_buffer](VkCommandBuffer p_cmd) {
            lfs::vis::VulkanImageBarrierTracker upload_barriers;
            upload_barriers.registerImage(p_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
            upload_barriers.transitionImage(p_cmd, p_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

            VkBufferImageCopy region = {};
            region.bufferOffset = 0;
            region.bufferRowLength = 0;
            region.bufferImageHeight = 0;

            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.mipLevel = 0;
            region.imageSubresource.baseArrayLayer = 0;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = extent_image;

            vkCmdCopyBufferToImage(p_cmd, cpu_buffer.m_p_vk_buffer, p_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

            upload_barriers.transitionImage(p_cmd, p_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });

        DestroyResource_StagingBuffer(cpu_buffer);
        if (!uploaded)
            return fail_texture("submit texture staging upload", VK_ERROR_UNKNOWN);
    }

    VkImageViewCreateInfo info_image_view = {};
    info_image_view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info_image_view.pNext = nullptr;
    info_image_view.image = p_texture->m_p_vk_image;
    info_image_view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info_image_view.format = format;
    info_image_view.subresourceRange.baseMipLevel = 0;
    info_image_view.subresourceRange.levelCount = 1;
    info_image_view.subresourceRange.baseArrayLayer = 0;
    info_image_view.subresourceRange.layerCount = 1;
    info_image_view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

    VkImageView p_image_view = VK_NULL_HANDLE;
    status = vkCreateImageView(m_p_device, &info_image_view, nullptr, &p_image_view);
    if (status != VK_SUCCESS || p_image_view == VK_NULL_HANDLE)
        return fail_texture("create image view", status);

    p_texture->m_p_vk_image_view = p_image_view;
    const std::string view_debug_name = std::format("rmlui.texture.view[{}]", name);
    (void)m_debug_name_writer.set(VK_OBJECT_TYPE_IMAGE_VIEW,
                                  (uint64_t)p_texture->m_p_vk_image_view,
                                  view_debug_name.c_str());
    p_texture->m_p_vk_sampler = texture_sampler;

    RegisterLiveTexture(p_texture.get());
    return reinterpret_cast<Rml::TextureHandle>(p_texture.release());
}

void RenderInterface_VK::ReleaseTexture(Rml::TextureHandle texture_handle) {
    texture_data_t* p_texture = reinterpret_cast<texture_data_t*>(texture_handle);

    if (p_texture) {
        DropAsyncPreviewTexture(p_texture);
        QueueTextureForDeferredDeletion(p_texture);
    }
}

void RenderInterface_VK::SetTransform(const Rml::Matrix4f* transform) {
    m_is_transform_enabled = (transform != nullptr);
    m_rml_transform = transform ? *transform : Rml::Matrix4f::Identity();
    ApplyTransformState();
}

void RenderInterface_VK::SetContextOffset(float offset_x, float offset_y) {
    offset_x = std::round(offset_x);
    offset_y = std::round(offset_y);
    m_context_offset = Rml::Vector2f(offset_x, offset_y);
    m_context_transform = Rml::Matrix4f::Translate(offset_x, offset_y, 0.0f);
    ApplyTransformState();
}

void RenderInterface_VK::SetContextClipRect(float x1, float y1, float x2, float y2) {
    if (x2 <= x1 || y2 <= y1) {
        m_context_clip_enabled = true;
        m_context_clip_scissor = {};
        if (m_p_current_command_buffer)
            vkCmdSetScissor(m_p_current_command_buffer, 0, 1, &m_context_clip_scissor);
        return;
    }

    const int left = Rml::Math::Clamp(static_cast<int>(std::floor(x1)), 0, m_width);
    const int top = Rml::Math::Clamp(static_cast<int>(std::floor(y1)), 0, m_height);
    const int right = Rml::Math::Clamp(static_cast<int>(std::ceil(x2)), 0, m_width);
    const int bottom = Rml::Math::Clamp(static_cast<int>(std::ceil(y2)), 0, m_height);

    m_context_clip_enabled = true;
    m_context_clip_scissor.offset.x = left;
    m_context_clip_scissor.offset.y = top;
    m_context_clip_scissor.extent.width = static_cast<uint32_t>(std::max(0, right - left));
    m_context_clip_scissor.extent.height = static_cast<uint32_t>(std::max(0, bottom - top));
    if (m_p_current_command_buffer)
        vkCmdSetScissor(m_p_current_command_buffer, 0, 1, &m_context_clip_scissor);
}

void RenderInterface_VK::ApplyTransformState() {
    m_user_data_for_vertex_shader.m_transform = m_projection * m_context_transform * m_rml_transform;
}

void RenderInterface_VK::RenderTextureQuad(Rml::TextureHandle texture, const float x, const float y,
                                           const float w, const float h) {
    if (m_p_current_command_buffer == nullptr || texture == 0 || w <= 0.0f || h <= 0.0f)
        return;

    if (!m_texture_quad_geometry ||
        m_texture_quad_x != x || m_texture_quad_y != y ||
        m_texture_quad_w != w || m_texture_quad_h != h) {
        if (m_texture_quad_geometry) {
            ReleaseGeometry(m_texture_quad_geometry);
            m_texture_quad_geometry = {};
        }

        Rml::Vertex vertices[4];
        vertices[0].position = {x, y};
        vertices[0].tex_coord = {0.0f, 0.0f};
        vertices[1].position = {x + w, y};
        vertices[1].tex_coord = {1.0f, 0.0f};
        vertices[2].position = {x + w, y + h};
        vertices[2].tex_coord = {1.0f, 1.0f};
        vertices[3].position = {x, y + h};
        vertices[3].tex_coord = {0.0f, 1.0f};
        for (Rml::Vertex& vertex : vertices)
            vertex.colour = Rml::ColourbPremultiplied(255, 255, 255, 255);

        static constexpr int indices[6] = {0, 1, 2, 0, 2, 3};
        m_texture_quad_geometry = CompileGeometry({vertices, 4}, {indices, 6});
        m_texture_quad_x = x;
        m_texture_quad_y = y;
        m_texture_quad_w = w;
        m_texture_quad_h = h;
    }

    const bool transform_enabled = m_is_transform_enabled;
    const shader_vertex_user_data_t user_data = m_user_data_for_vertex_shader;
    const Rml::Matrix4f rml_transform = m_rml_transform;
    const Rml::Matrix4f context_transform = m_context_transform;
    const Rml::Vector2f context_offset = m_context_offset;
    SetTransform(nullptr);
    if (m_texture_quad_geometry) {
        RenderGeometry(m_texture_quad_geometry, {}, texture);
    }
    m_is_transform_enabled = transform_enabled;
    m_user_data_for_vertex_shader = user_data;
    m_rml_transform = rml_transform;
    m_context_transform = context_transform;
    m_context_offset = context_offset;
}

VkRect2D RenderInterface_VK::ContextClipScissor() const noexcept {
    return m_context_clip_enabled ? m_context_clip_scissor : m_scissor_original;
}

VkRect2D RenderInterface_VK::IntersectContextClip(VkRect2D scissor) const noexcept {
    if (!m_context_clip_enabled)
        return scissor;

    const int left = std::max(scissor.offset.x, m_context_clip_scissor.offset.x);
    const int top = std::max(scissor.offset.y, m_context_clip_scissor.offset.y);
    const int right = std::min(scissor.offset.x + static_cast<int>(scissor.extent.width),
                               m_context_clip_scissor.offset.x + static_cast<int>(m_context_clip_scissor.extent.width));
    const int bottom = std::min(scissor.offset.y + static_cast<int>(scissor.extent.height),
                                m_context_clip_scissor.offset.y + static_cast<int>(m_context_clip_scissor.extent.height));

    scissor.offset.x = left;
    scissor.offset.y = top;
    scissor.extent.width = static_cast<uint32_t>(std::max(0, right - left));
    scissor.extent.height = static_cast<uint32_t>(std::max(0, bottom - top));
    return scissor;
}

bool RenderInterface_VK::InitializeExternal(const ExternalContext& context) {
    RMLUI_ZoneScopedN("Vulkan - InitializeExternal");

    if (!context.instance || !context.physical_device || !context.device || !context.graphics_queue ||
        context.color_format == VK_FORMAT_UNDEFINED || context.depth_stencil_format == VK_FORMAT_UNDEFINED ||
        context.extent.width == 0 || context.extent.height == 0) {
        Rml::Log::Message(Rml::Log::LT_ERROR, "[Vulkan] Invalid external context for RmlUi renderer.");
        return false;
    }

    m_external_context = true;
    m_p_instance = context.instance;
    m_p_physical_device = context.physical_device;
    m_p_device = context.device;
    m_debug_name_writer.initialize(m_p_device);
    m_p_pipeline_cache = context.pipeline_cache;

    // On Vulkan 1.4 drivers these core-promoted entry points resolve even when
    // the hostImageCopy feature was not enabled on the device; calling them then
    // is UB. Only look them up when the owning context enabled the feature.
    if (context.host_image_copy &&
        SupportsHostImageCopyDestinationLayout(m_p_physical_device,
                                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)) {
        m_pfn_copy_memory_to_image = reinterpret_cast<PFN_vkCopyMemoryToImageEXT>(
            vkGetDeviceProcAddr(m_p_device, "vkCopyMemoryToImageEXT"));
        m_pfn_transition_image_layout = reinterpret_cast<PFN_vkTransitionImageLayoutEXT>(
            vkGetDeviceProcAddr(m_p_device, "vkTransitionImageLayoutEXT"));
    } else if (context.host_image_copy) {
        Rml::Log::Message(Rml::Log::LT_INFO,
                          "[Vulkan] Host image copies do not support the RmlUi shader-read layout; using staging uploads.");
    }
    m_p_queue_graphics = context.graphics_queue;
    m_queue_index_graphics = context.graphics_queue_family;
    m_swapchain_format.format = context.color_format;
    m_depth_stencil_format = context.depth_stencil_format;
    m_width = static_cast<int>(context.extent.width);
    m_height = static_cast<int>(context.extent.height);

    VkPhysicalDeviceProperties physical_device_properties = {};
    vkGetPhysicalDeviceProperties(m_p_physical_device, &physical_device_properties);

    Initialize_Allocator();
    if (m_p_allocator == VK_NULL_HANDLE) {
        Rml::Log::Message(Rml::Log::LT_ERROR,
                          "[Vulkan] Failed to initialize the external RmlUi VMA allocator.");
        return false;
    }
    Initialize_Resources(physical_device_properties);
    UpdateViewportState(context.extent);
    Create_Pipelines();
    return true;
}

void RenderInterface_VK::ShutdownExternal() {
    RMLUI_ZoneScopedN("Vulkan - ShutdownExternal");

    if (!m_external_context)
        return;

    if (m_p_device)
        vkDeviceWaitIdle(m_p_device);

    m_async_preview_textures.clear();
    DestroyRenderLayers();
    Destroy_Pipelines();
    Destroy_Resources();
    Destroy_Allocator();

    m_debug_name_writer.reset();

    m_p_instance = VK_NULL_HANDLE;
    m_p_physical_device = VK_NULL_HANDLE;
    m_p_device = VK_NULL_HANDLE;
    m_p_queue_graphics = VK_NULL_HANDLE;
    m_external_swapchain_image = VK_NULL_HANDLE;
    m_external_swapchain_image_view = VK_NULL_HANDLE;
    m_external_depth_stencil_image_view = VK_NULL_HANDLE;
    m_external_swapchain_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_external_context = false;
}

void RenderInterface_VK::BeginExternalFrame(const VkCommandBuffer command_buffer,
                                            const VkExtent2D extent,
                                            const VkImage swapchain_image,
                                            const VkImageView swapchain_image_view,
                                            const VkImageView depth_stencil_image_view,
                                            const std::size_t frame_slot) {
    if (!m_external_context || command_buffer == VK_NULL_HANDLE || swapchain_image_view == VK_NULL_HANDLE ||
        depth_stencil_image_view == VK_NULL_HANDLE)
        return;

    if (extent.width != static_cast<uint32_t>(m_width) || extent.height != static_cast<uint32_t>(m_height)) {
        m_width = static_cast<int>(extent.width);
        m_height = static_cast<int>(extent.height);
        UpdateViewportState(extent);
    }

    m_resource_slot = static_cast<uint32_t>(frame_slot % kSwapchainBackBufferCount);
    m_reclaim_resource_slot = m_resource_slot;
    FreeTransientShaderAllocations(m_reclaim_resource_slot);
    Update_PendingForDeletion_Textures_By_Frame(m_reclaim_resource_slot);
    Update_PendingForDeletion_Geometries(m_reclaim_resource_slot);
    ProcessAsyncPreviewUploads();

    m_p_current_command_buffer = command_buffer;
    m_external_swapchain_image = swapchain_image;
    m_external_swapchain_image_view = swapchain_image_view;
    m_external_depth_stencil_image_view = depth_stencil_image_view;
    m_external_swapchain_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    vkCmdSetViewport(m_p_current_command_buffer, 0, 1, &m_viewport);
    vkCmdSetScissor(m_p_current_command_buffer, 0, 1, &m_scissor_original);
    vkCmdSetStencilReference(m_p_current_command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, 1);
    m_active_render_target = active_render_target_t::Swapchain;
    m_active_layer = {};
    m_render_layer_stack_size = 0;
    m_is_clip_mask_enabled = false;
    m_is_transformed_scissor_enabled = false;
    m_is_use_scissor_specified = false;
    m_is_use_stencil_pipeline = false;
    m_is_apply_to_regular_geometry_stencil = false;
    SetContextOffset(0.0f, 0.0f);
    SetTransform(nullptr);
    m_context_clip_enabled = false;
}

void RenderInterface_VK::EndExternalFrame() {
    if (!m_external_context)
        return;
    if (m_active_render_target != active_render_target_t::Swapchain) {
        EndActiveRendering();
        BeginSwapchainRendering(VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_LOAD);
    }
    m_external_swapchain_image = VK_NULL_HANDLE;
    m_external_swapchain_image_view = VK_NULL_HANDLE;
    m_external_depth_stencil_image_view = VK_NULL_HANDLE;
    m_external_swapchain_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_active_render_target = active_render_target_t::None;
    m_p_current_command_buffer = nullptr;
}

void RenderInterface_VK::ResetContextRenderState() {
    if (m_p_current_command_buffer == nullptr)
        return;

    m_is_clip_mask_enabled = false;
    m_is_transformed_scissor_enabled = false;
    m_is_use_scissor_specified = false;
    m_is_use_stencil_pipeline = false;
    m_is_apply_to_regular_geometry_stencil = false;
    SetContextOffset(0.0f, 0.0f);
    SetTransform(nullptr);
    m_context_clip_enabled = false;
    vkCmdSetViewport(m_p_current_command_buffer, 0, 1, &m_viewport);
    vkCmdSetScissor(m_p_current_command_buffer, 0, 1, &m_scissor_original);
    vkCmdSetStencilReference(m_p_current_command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, 1);
}

void RenderInterface_VK::Initialize_Resources(const VkPhysicalDeviceProperties& physical_device_properties) noexcept {
    const VkDeviceSize min_buffer_alignment = physical_device_properties.limits.minUniformBufferOffsetAlignment;
    m_memory_pool.Initialize(kVideoMemoryForAllocation, min_buffer_alignment, m_p_allocator, m_p_device);

    m_upload_manager.Initialize(m_p_device, m_p_queue_graphics, m_queue_index_graphics);
    m_manager_descriptors.Initialize(m_p_device, 100, 100, 10, 10);

    CreateShaders();
    CreateDescriptorSetLayout();
    CreatePipelineLayout();
    CreateSamplers();
    CreateDescriptorSets();
}

void RenderInterface_VK::Initialize_Allocator() noexcept {
    m_p_allocator = VK_NULL_HANDLE;
    if (m_p_device == VK_NULL_HANDLE || m_p_physical_device == VK_NULL_HANDLE ||
        m_p_instance == VK_NULL_HANDLE) {
        return;
    }

    VmaVulkanFunctions vulkanFunctions = {};
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo info = {};

    info.vulkanApiVersion = RMLUI_VK_API_VERSION;
    info.device = m_p_device;
    info.instance = m_p_instance;
    info.physicalDevice = m_p_physical_device;
    info.pVulkanFunctions = &vulkanFunctions;

    if (vmaCreateAllocator(&info, &m_p_allocator) != VK_SUCCESS)
        m_p_allocator = VK_NULL_HANDLE;
}

void RenderInterface_VK::Destroy_Resources() noexcept {
    m_upload_manager.Shutdown();

    if (m_p_descriptor_set) {
        m_manager_descriptors.Free_Descriptors(m_p_device, &m_p_descriptor_set);
    }

    vkDestroyDescriptorSetLayout(m_p_device, m_p_descriptor_set_layout_vertex_transform, nullptr);
    vkDestroyDescriptorSetLayout(m_p_device, m_p_descriptor_set_layout_texture, nullptr);

    vkDestroyPipelineLayout(m_p_device, m_p_pipeline_layout, nullptr);

    for (const auto& p_module : m_shaders) {
        vkDestroyShaderModule(m_p_device, p_module, nullptr);
    }

    DestroySamplers();
    Destroy_Textures();
    Destroy_LiveTextures();
    Destroy_Geometries();

    m_manager_descriptors.Shutdown(m_p_device);
}

void RenderInterface_VK::Destroy_Allocator() noexcept {
    if (m_p_allocator != VK_NULL_HANDLE)
        vmaDestroyAllocator(m_p_allocator);
    m_p_allocator = VK_NULL_HANDLE;
}

void RenderInterface_VK::CreateShaders() noexcept {
    RMLUI_VK_ASSERTMSG(m_p_device, "[Vulkan] you must initialize VkDevice before calling this method");

    struct shader_data_t {
        const uint32_t* m_data;
        size_t m_data_size;
        VkShaderStageFlagBits m_shader_type;
    };

    const Rml::Vector<shader_data_t> shaders = {
        {reinterpret_cast<const uint32_t*>(shader_vert), sizeof(shader_vert), VK_SHADER_STAGE_VERTEX_BIT},
        {reinterpret_cast<const uint32_t*>(shader_frag_color), sizeof(shader_frag_color), VK_SHADER_STAGE_FRAGMENT_BIT},
        {reinterpret_cast<const uint32_t*>(shader_frag_texture), sizeof(shader_frag_texture), VK_SHADER_STAGE_FRAGMENT_BIT},
    };

    for (const shader_data_t& shader_data : shaders) {
        VkShaderModuleCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        info.pCode = shader_data.m_data;
        info.codeSize = shader_data.m_data_size;

        VkShaderModule p_module = nullptr;
        VkResult status = vkCreateShaderModule(m_p_device, &info, nullptr, &p_module);

        RMLUI_VK_ASSERTMSG(status == VK_SUCCESS, "[Vulkan] failed to vkCreateShaderModule");

        m_shaders.push_back(p_module);
    }
}

void RenderInterface_VK::CreateDescriptorSetLayout() noexcept {
    RMLUI_VK_ASSERTMSG(m_p_device, "[Vulkan] you must initialize VkDevice before calling this method");
    RMLUI_VK_ASSERTMSG(!m_p_descriptor_set_layout_vertex_transform && !m_p_descriptor_set_layout_texture, "[Vulkan] Already initialized");

    {
        VkDescriptorSetLayoutBinding binding_for_vertex_transform = {};
        binding_for_vertex_transform.binding = 1;
        binding_for_vertex_transform.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
        binding_for_vertex_transform.descriptorCount = 1;
        binding_for_vertex_transform.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.pBindings = &binding_for_vertex_transform;
        info.bindingCount = 1;

        VkResult status = vkCreateDescriptorSetLayout(m_p_device, &info, nullptr, &m_p_descriptor_set_layout_vertex_transform);
        RMLUI_VK_ASSERTMSG(status == VK_SUCCESS, "[Vulkan] failed to vkCreateDescriptorSetLayout");
    }

    {
        VkDescriptorSetLayoutBinding binding_for_fragment_texture = {};
        binding_for_fragment_texture.binding = 2;
        binding_for_fragment_texture.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding_for_fragment_texture.descriptorCount = 1;
        binding_for_fragment_texture.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        info.pBindings = &binding_for_fragment_texture;
        info.bindingCount = 1;

        VkResult status = vkCreateDescriptorSetLayout(m_p_device, &info, nullptr, &m_p_descriptor_set_layout_texture);
        RMLUI_VK_ASSERTMSG(status == VK_SUCCESS, "[Vulkan] failed to vkCreateDescriptorSetLayout");
    }
}

void RenderInterface_VK::CreatePipelineLayout() noexcept {
    RMLUI_VK_ASSERTMSG(m_p_descriptor_set_layout_vertex_transform, "[Vulkan] You must initialize VkDescriptorSetLayout before calling this method");
    RMLUI_VK_ASSERTMSG(m_p_descriptor_set_layout_texture,
                       "[Vulkan] you must initialize VkDescriptorSetLayout for textures before calling this method!");
    RMLUI_VK_ASSERTMSG(m_p_device, "[Vulkan] you must initialize VkDevice before calling this method");

    VkDescriptorSetLayout p_layouts[] = {m_p_descriptor_set_layout_vertex_transform, m_p_descriptor_set_layout_texture};

    VkPipelineLayoutCreateInfo info = {};

    info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info.pNext = nullptr;
    info.pSetLayouts = p_layouts;
    info.setLayoutCount = 2;

    auto status = vkCreatePipelineLayout(m_p_device, &info, nullptr, &m_p_pipeline_layout);

    RMLUI_VK_ASSERTMSG(status == VK_SUCCESS, "[Vulkan] failed to vkCreatePipelineLayout");
}

void RenderInterface_VK::CreateDescriptorSets() noexcept {
    RMLUI_VK_ASSERTMSG(m_p_device, "[Vulkan] you have to initialize your VkDevice before calling this method");
    RMLUI_VK_ASSERTMSG(m_p_descriptor_set_layout_vertex_transform,
                       "[Vulkan] you have to initialize your VkDescriptorSetLayout before calling this method");

    m_manager_descriptors.Alloc_Descriptor(m_p_device, &m_p_descriptor_set_layout_vertex_transform, &m_p_descriptor_set);
    m_memory_pool.SetDescriptorSet(1, sizeof(shader_vertex_user_data_t), VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, m_p_descriptor_set);
}

void RenderInterface_VK::CreateSamplers() noexcept {
    auto create_sampler = [this](VkFilter filter, VkSampler* sampler) {
        VkSamplerCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        info.pNext = nullptr;
        info.magFilter = filter;
        info.minFilter = filter;
        info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

        vkCreateSampler(m_p_device, &info, nullptr, sampler);
    };

    create_sampler(VK_FILTER_LINEAR, &m_p_sampler_linear);
    create_sampler(VK_FILTER_NEAREST, &m_p_sampler_nearest);
}

void RenderInterface_VK::Create_Pipelines() noexcept {
    RMLUI_VK_ASSERTMSG(m_p_pipeline_layout, "must be initialized");
    RMLUI_VK_ASSERTMSG(m_swapchain_format.format != VK_FORMAT_UNDEFINED, "must have a color format");
    RMLUI_VK_ASSERTMSG(m_depth_stencil_format != VK_FORMAT_UNDEFINED, "must have a depth/stencil format");

    VkPipelineInputAssemblyStateCreateInfo info_assembly_state = {};
    info_assembly_state.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    info_assembly_state.pNext = nullptr;
    info_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    info_assembly_state.primitiveRestartEnable = VK_FALSE;
    info_assembly_state.flags = 0;

    VkPipelineRasterizationStateCreateInfo info_raster_state = {};
    info_raster_state.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    info_raster_state.pNext = nullptr;
    info_raster_state.polygonMode = VK_POLYGON_MODE_FILL;
    info_raster_state.cullMode = VK_CULL_MODE_NONE;
    info_raster_state.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    info_raster_state.rasterizerDiscardEnable = VK_FALSE;
    info_raster_state.depthBiasEnable = VK_FALSE;
    info_raster_state.lineWidth = 1.0f;

    VkPipelineColorBlendAttachmentState info_color_blend_att = {};
    info_color_blend_att.colorWriteMask = 0xf;
    info_color_blend_att.blendEnable = VK_TRUE;
    info_color_blend_att.srcColorBlendFactor = VkBlendFactor::VK_BLEND_FACTOR_ONE;
    info_color_blend_att.dstColorBlendFactor = VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    info_color_blend_att.colorBlendOp = VkBlendOp::VK_BLEND_OP_ADD;
    info_color_blend_att.srcAlphaBlendFactor = VkBlendFactor::VK_BLEND_FACTOR_ONE;
    info_color_blend_att.dstAlphaBlendFactor = VkBlendFactor::VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    info_color_blend_att.alphaBlendOp = VkBlendOp::VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo info_color_blend_state = {};
    info_color_blend_state.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    info_color_blend_state.pNext = nullptr;
    info_color_blend_state.attachmentCount = 1;
    info_color_blend_state.pAttachments = &info_color_blend_att;

    VkPipelineDepthStencilStateCreateInfo info_depth = {};
    info_depth.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    info_depth.pNext = nullptr;
    info_depth.depthTestEnable = VK_FALSE;
    info_depth.depthWriteEnable = VK_TRUE;
    info_depth.depthBoundsTestEnable = VK_FALSE;
    info_depth.maxDepthBounds = 1.0f;

    info_depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;

    info_depth.stencilTestEnable = VK_TRUE;
    info_depth.back.compareOp = VK_COMPARE_OP_ALWAYS;
    info_depth.back.failOp = VK_STENCIL_OP_KEEP;
    info_depth.back.depthFailOp = VK_STENCIL_OP_KEEP;
    info_depth.back.passOp = VK_STENCIL_OP_KEEP;
    info_depth.back.compareMask = 1;
    info_depth.back.writeMask = 1;
    info_depth.back.reference = 1;
    info_depth.front = info_depth.back;

    VkPipelineViewportStateCreateInfo info_viewport = {};
    info_viewport.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    info_viewport.pNext = nullptr;
    info_viewport.viewportCount = 1;
    info_viewport.scissorCount = 1;
    info_viewport.flags = 0;

    VkPipelineMultisampleStateCreateInfo info_multisample = {};
    info_multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    info_multisample.pNext = nullptr;
    info_multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    info_multisample.flags = 0;

    Rml::Array<VkDynamicState, 3> dynamicStateEnables = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    };

    VkPipelineDynamicStateCreateInfo info_dynamic_state = {};
    info_dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    info_dynamic_state.pNext = nullptr;
    info_dynamic_state.pDynamicStates = dynamicStateEnables.data();
    info_dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamicStateEnables.size());
    info_dynamic_state.flags = 0;

    Rml::Array<VkPipelineShaderStageCreateInfo, 2> shaders_that_will_be_used_in_pipeline;

    VkPipelineShaderStageCreateInfo info_shader = {};
    info_shader.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info_shader.pNext = nullptr;
    info_shader.pName = "main";
    info_shader.stage = VK_SHADER_STAGE_VERTEX_BIT;
    info_shader.module = m_shaders[static_cast<int>(shader_id_t::Vertex)];

    shaders_that_will_be_used_in_pipeline[0] = info_shader;

    info_shader.module = m_shaders[static_cast<int>(shader_id_t::Fragment_WithTextures)];
    info_shader.stage = VK_SHADER_STAGE_FRAGMENT_BIT;

    shaders_that_will_be_used_in_pipeline[1] = info_shader;

    VkPipelineVertexInputStateCreateInfo info_vertex = {};
    info_vertex.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    info_vertex.pNext = nullptr;
    info_vertex.flags = 0;

    Rml::Array<VkVertexInputAttributeDescription, 3> info_shader_vertex_attributes;
    // describe info about our vertex and what is used in vertex shader as "layout(location = X) in"

    VkVertexInputBindingDescription info_vertex_input_binding = {};
    info_vertex_input_binding.binding = 0;
    info_vertex_input_binding.stride = sizeof(Rml::Vertex);
    info_vertex_input_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    info_shader_vertex_attributes[0].binding = 0;
    info_shader_vertex_attributes[0].location = 0;
    info_shader_vertex_attributes[0].format = VK_FORMAT_R32G32_SFLOAT;
    info_shader_vertex_attributes[0].offset = offsetof(Rml::Vertex, position);

    info_shader_vertex_attributes[1].binding = 0;
    info_shader_vertex_attributes[1].location = 1;
    info_shader_vertex_attributes[1].format = VK_FORMAT_R8G8B8A8_UNORM;
    info_shader_vertex_attributes[1].offset = offsetof(Rml::Vertex, colour);

    info_shader_vertex_attributes[2].binding = 0;
    info_shader_vertex_attributes[2].location = 2;
    info_shader_vertex_attributes[2].format = VK_FORMAT_R32G32_SFLOAT;
    info_shader_vertex_attributes[2].offset = offsetof(Rml::Vertex, tex_coord);

    info_vertex.pVertexAttributeDescriptions = info_shader_vertex_attributes.data();
    info_vertex.vertexAttributeDescriptionCount = static_cast<uint32_t>(info_shader_vertex_attributes.size());
    info_vertex.pVertexBindingDescriptions = &info_vertex_input_binding;
    info_vertex.vertexBindingDescriptionCount = 1;

    VkPipelineRenderingCreateInfo rendering_info{};
    rendering_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachmentFormats = &m_swapchain_format.format;
    rendering_info.depthAttachmentFormat = m_depth_stencil_format;
    rendering_info.stencilAttachmentFormat = (DepthStencilAspectMask() & VK_IMAGE_ASPECT_STENCIL_BIT) != 0 ? m_depth_stencil_format : VK_FORMAT_UNDEFINED;

    VkGraphicsPipelineCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info.pNext = &rendering_info;
    info.pInputAssemblyState = &info_assembly_state;
    info.pRasterizationState = &info_raster_state;
    info.pColorBlendState = &info_color_blend_state;
    info.pMultisampleState = &info_multisample;
    info.pViewportState = &info_viewport;
    info.pDepthStencilState = &info_depth;
    info.pDynamicState = &info_dynamic_state;
    info.stageCount = static_cast<uint32_t>(shaders_that_will_be_used_in_pipeline.size());
    info.pStages = shaders_that_will_be_used_in_pipeline.data();
    info.pVertexInputState = &info_vertex;
    info.layout = m_p_pipeline_layout;
    info.renderPass = VK_NULL_HANDLE;
    info.subpass = 0;

    auto status = vkCreateGraphicsPipelines(m_p_device, m_p_pipeline_cache, 1, &info, nullptr, &m_p_pipeline_with_textures);
    RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vkCreateGraphicsPipelines");

    info_depth.back.passOp = VK_STENCIL_OP_KEEP;
    info_depth.back.failOp = VK_STENCIL_OP_KEEP;
    info_depth.back.depthFailOp = VK_STENCIL_OP_KEEP;
    info_depth.back.compareOp = VK_COMPARE_OP_EQUAL;
    info_depth.back.compareMask = 1;
    info_depth.back.writeMask = 1;
    info_depth.back.reference = 1;
    info_depth.front = info_depth.back;

    status = vkCreateGraphicsPipelines(m_p_device, m_p_pipeline_cache, 1, &info, nullptr,
                                       &m_p_pipeline_stencil_for_regular_geometry_that_applied_to_region_with_textures);
    RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vkCreateGraphicsPipelines");

    info_shader.module = m_shaders[static_cast<int>(shader_id_t::Fragment_WithoutTextures)];
    shaders_that_will_be_used_in_pipeline[1] = info_shader;
    info_depth.back.compareOp = VK_COMPARE_OP_ALWAYS;
    info_depth.back.failOp = VK_STENCIL_OP_KEEP;
    info_depth.back.depthFailOp = VK_STENCIL_OP_KEEP;
    info_depth.back.passOp = VK_STENCIL_OP_KEEP;
    info_depth.back.compareMask = 1;
    info_depth.back.writeMask = 1;
    info_depth.back.reference = 1;
    info_depth.front = info_depth.back;

    status = vkCreateGraphicsPipelines(m_p_device, m_p_pipeline_cache, 1, &info, nullptr, &m_p_pipeline_without_textures);
    RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vkCreateGraphicsPipelines");

    info_depth.back.passOp = VK_STENCIL_OP_KEEP;
    info_depth.back.failOp = VK_STENCIL_OP_KEEP;
    info_depth.back.depthFailOp = VK_STENCIL_OP_KEEP;
    info_depth.back.compareOp = VK_COMPARE_OP_EQUAL;
    info_depth.back.compareMask = 1;
    info_depth.back.writeMask = 1;
    info_depth.back.reference = 1;
    info_depth.front = info_depth.back;

    status = vkCreateGraphicsPipelines(m_p_device, m_p_pipeline_cache, 1, &info, nullptr,
                                       &m_p_pipeline_stencil_for_regular_geometry_that_applied_to_region_without_textures);
    RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vkCreateGraphicsPipelines");

    info_color_blend_att.colorWriteMask = 0x0;
    info_depth.back.passOp = VK_STENCIL_OP_REPLACE;
    info_depth.back.failOp = VK_STENCIL_OP_KEEP;
    info_depth.back.depthFailOp = VK_STENCIL_OP_KEEP;
    info_depth.back.compareOp = VK_COMPARE_OP_ALWAYS;
    info_depth.back.compareMask = 1;
    info_depth.back.writeMask = 1;
    info_depth.back.reference = 1;
    info_depth.front = info_depth.back;

    status = vkCreateGraphicsPipelines(m_p_device, m_p_pipeline_cache, 1, &info, nullptr, &m_p_pipeline_stencil_for_region_where_geometry_will_be_drawn);
    RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vkCreateGraphicsPipelines");

#ifdef RMLUI_VK_DEBUG
    (void)m_debug_name_writer.set(
        VK_OBJECT_TYPE_PIPELINE,
        (uint64_t)m_p_pipeline_stencil_for_region_where_geometry_will_be_drawn,
        "pipeline_stencil for region where geometry will be drawn");
    (void)m_debug_name_writer.set(
        VK_OBJECT_TYPE_PIPELINE,
        (uint64_t)m_p_pipeline_stencil_for_regular_geometry_that_applied_to_region_without_textures,
        "pipeline_stencil_for_regular_geometry_that_applied_to_region_without_textures");
    (void)m_debug_name_writer.set(VK_OBJECT_TYPE_PIPELINE,
                                  (uint64_t)m_p_pipeline_without_textures,
                                  "pipeline_without_textures");
    (void)m_debug_name_writer.set(
        VK_OBJECT_TYPE_PIPELINE,
        (uint64_t)m_p_pipeline_stencil_for_regular_geometry_that_applied_to_region_with_textures,
        "pipeline_stencil_for_regular_geometry_that_applied_to_region_with_textures");
    (void)m_debug_name_writer.set(VK_OBJECT_TYPE_PIPELINE,
                                  (uint64_t)m_p_pipeline_with_textures,
                                  "pipeline_with_textures");
#endif
}

void RenderInterface_VK::UpdateViewportState(const VkExtent2D& real_render_image_size) noexcept {
    m_viewport.height = static_cast<float>(real_render_image_size.height);
    m_viewport.width = static_cast<float>(real_render_image_size.width);
    m_viewport.minDepth = 0.0f;
    m_viewport.maxDepth = 1.0f;
    m_viewport.x = 0.0f;
    m_viewport.y = 0.0f;

    m_scissor.extent.width = real_render_image_size.width;
    m_scissor.extent.height = real_render_image_size.height;
    m_scissor.offset.x = 0;
    m_scissor.offset.y = 0;

    m_scissor_original = m_scissor;

    m_projection = Rml::Matrix4f::ProjectOrtho(0.0f, static_cast<float>(m_width), static_cast<float>(m_height), 0.0f, -10000, 10000);

    // https://matthewwellings.com/blog/the-new-vulkan-coordinate-system/
    Rml::Matrix4f correction_matrix;
    correction_matrix.SetColumns(Rml::Vector4f(1.0f, 0.0f, 0.0f, 0.0f), Rml::Vector4f(0.0f, -1.0f, 0.0f, 0.0f), Rml::Vector4f(0.0f, 0.0f, 0.5f, 0.0f),
                                 Rml::Vector4f(0.0f, 0.0f, 0.5f, 1.0f));

    m_projection = correction_matrix * m_projection;

    SetTransform(nullptr);
}

RenderInterface_VK::buffer_data_t RenderInterface_VK::CreateResource_StagingBuffer(VkDeviceSize size, VkBufferUsageFlags flags) noexcept {
    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.pNext = nullptr;
    info.size = size;
    info.usage = flags;

    VmaAllocationCreateInfo info_allocation = {};
    info_allocation.usage = VMA_MEMORY_USAGE_CPU_ONLY;

    VkBuffer p_buffer = nullptr;
    VmaAllocation p_allocation = nullptr;
    VmaAllocationInfo info_stats = {};

    VkResult status = vmaCreateBuffer(m_p_allocator, &info, &info_allocation, &p_buffer, &p_allocation, &info_stats);
    RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vmaCreateBuffer");

#ifdef RMLUI_VK_DEBUG
    Rml::Log::Message(Rml::Log::LT_DEBUG, "Allocated buffer [%s]", FormatByteSize(info_stats.size).c_str());
#endif

    buffer_data_t result = {};
    result.m_p_vk_buffer = p_buffer;
    result.m_p_vma_allocation = p_allocation;

    return result;
}

void RenderInterface_VK::DestroyResource_StagingBuffer(const buffer_data_t& data) noexcept {
    if (m_p_allocator) {
        if (data.m_p_vk_buffer && data.m_p_vma_allocation) {
            vmaDestroyBuffer(m_p_allocator, data.m_p_vk_buffer, data.m_p_vma_allocation);
        }
    }
}

void RenderInterface_VK::Destroy_Textures() noexcept {
    for (auto& textures : m_pending_for_deletion_textures_by_frames) {
        for (texture_data_t* p_data : textures) {
            UnregisterLiveTexture(p_data);
            Destroy_Texture(*p_data);
            delete p_data;
        }

        textures.clear();
    }
}

void RenderInterface_VK::Destroy_LiveTextures() noexcept {
    auto live_textures = std::move(m_live_textures);
    m_live_textures.clear();

    for (texture_data_t* texture : live_textures) {
        if (!texture)
            continue;
        Destroy_Texture(*texture);
        delete texture;
    }
}

void RenderInterface_VK::RegisterLiveTexture(texture_data_t* texture) {
    if (texture)
        m_live_textures.insert(texture);
}

void RenderInterface_VK::UnregisterLiveTexture(texture_data_t* texture) {
    if (texture)
        m_live_textures.erase(texture);
}

uint32_t RenderInterface_VK::ActiveResourceSlot() const noexcept {
    return m_resource_slot % kSwapchainBackBufferCount;
}

void RenderInterface_VK::FreeTransientShaderAllocations(const uint32_t resource_slot) noexcept {
    auto& allocations = m_transient_shader_allocations_by_frame[resource_slot % kSwapchainBackBufferCount];
    for (VmaVirtualAllocation allocation : allocations)
        m_memory_pool.Free_Allocation(allocation);
    allocations.clear();
}

void RenderInterface_VK::FreeAllTransientShaderAllocations() noexcept {
    for (uint32_t slot = 0; slot < kSwapchainBackBufferCount; ++slot)
        FreeTransientShaderAllocations(slot);
}

void RenderInterface_VK::Destroy_Geometries() noexcept {
    if (m_texture_quad_geometry) {
        ReleaseGeometry(m_texture_quad_geometry);
        m_texture_quad_geometry = {};
    }
    FreeAllTransientShaderAllocations();
    for (uint32_t slot = 0; slot < kSwapchainBackBufferCount; ++slot)
        Update_PendingForDeletion_Geometries(slot);
    m_memory_pool.Shutdown();
}

void RenderInterface_VK::Destroy_Texture(const texture_data_t& texture) noexcept {
    RMLUI_VK_ASSERTMSG(m_p_allocator, "you must have initialized VmaAllocator");
    RMLUI_VK_ASSERTMSG(m_p_device, "you must have initialized VkDevice");

    if (VkDescriptorSet p_set = texture.m_p_vk_descriptor_set; p_set)
        m_manager_descriptors.Free_Descriptors(m_p_device, &p_set);

    if (texture.m_p_vma_allocation) {
        if (!texture.m_vram_scope.empty() && !texture.m_vram_label.empty())
            RecordRmlUiVram(texture.m_vram_scope, texture.m_vram_label, 0);
        m_image_barriers.forgetImage(texture.m_p_vk_image);
        if (texture.m_p_vk_image_view)
            vkDestroyImageView(m_p_device, texture.m_p_vk_image_view, nullptr);
        vmaDestroyImage(m_p_allocator, texture.m_p_vk_image, texture.m_p_vma_allocation);
    }
}

void RenderInterface_VK::Destroy_Pipelines() noexcept {
    RMLUI_VK_ASSERTMSG(m_p_device, "must exist here");

    vkDestroyPipeline(m_p_device, m_p_pipeline_with_textures, nullptr);
    vkDestroyPipeline(m_p_device, m_p_pipeline_without_textures, nullptr);
    vkDestroyPipeline(m_p_device, m_p_pipeline_stencil_for_region_where_geometry_will_be_drawn, nullptr);
    vkDestroyPipeline(m_p_device, m_p_pipeline_stencil_for_regular_geometry_that_applied_to_region_with_textures, nullptr);
    vkDestroyPipeline(m_p_device, m_p_pipeline_stencil_for_regular_geometry_that_applied_to_region_without_textures, nullptr);
}

void RenderInterface_VK::DestroySamplers() noexcept {
    RMLUI_VK_ASSERTMSG(m_p_device, "must exist here");
    if (m_p_sampler_linear != VK_NULL_HANDLE)
        vkDestroySampler(m_p_device, m_p_sampler_linear, nullptr);
    if (m_p_sampler_nearest != VK_NULL_HANDLE)
        vkDestroySampler(m_p_device, m_p_sampler_nearest, nullptr);
    m_p_sampler_linear = VK_NULL_HANDLE;
    m_p_sampler_nearest = VK_NULL_HANDLE;
}

VkImageAspectFlags RenderInterface_VK::DepthStencilAspectMask() const noexcept {
    switch (m_depth_stencil_format) {
    case VK_FORMAT_D16_UNORM_S8_UINT:
    case VK_FORMAT_D24_UNORM_S8_UINT:
    case VK_FORMAT_D32_SFLOAT_S8_UINT: return VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
    default: return VK_IMAGE_ASPECT_DEPTH_BIT;
    }
}

void RenderInterface_VK::EnsureRenderLayer(Rml::LayerHandle layer_handle) {
    if (layer_handle == 0)
        return;
    if (m_width <= 0 || m_height <= 0)
        return;

    const size_t index = static_cast<size_t>(layer_handle - 1);
    if (index >= m_render_layers.size())
        m_render_layers.resize(index + 1);

    render_layer_t& layer = m_render_layers[index];
    const bool has_color = layer.m_color.m_p_vk_image_view != VK_NULL_HANDLE;
    const bool has_depth_stencil = layer.m_depth_stencil.m_p_vk_image_view != VK_NULL_HANDLE;
    if ((has_color || has_depth_stencil) &&
        (!has_color || !has_depth_stencil || layer.width != m_width || layer.height != m_height))
        DestroyRenderLayer(layer);

    if (layer.m_color.m_p_vk_image_view && layer.m_depth_stencil.m_p_vk_image_view)
        return;

    VkExtent3D extent{};
    extent.width = static_cast<uint32_t>(m_width);
    extent.height = static_cast<uint32_t>(m_height);
    extent.depth = 1;

    VkImageCreateInfo color_info{};
    color_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    color_info.imageType = VK_IMAGE_TYPE_2D;
    color_info.format = m_swapchain_format.format;
    color_info.extent = extent;
    color_info.mipLevels = 1;
    color_info.arrayLayers = 1;
    color_info.samples = VK_SAMPLE_COUNT_1_BIT;
    color_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    color_info.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    color_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocation_info{};
    allocation_info.usage = VMA_MEMORY_USAGE_GPU_ONLY;

    VmaAllocationInfo color_allocation_stats{};
    VkResult status = vmaCreateImage(m_p_allocator, &color_info, &allocation_info, &layer.m_color.m_p_vk_image,
                                     &layer.m_color.m_p_vma_allocation, &color_allocation_stats);
    RMLUI_VK_ASSERTMSG(status == VK_SUCCESS, "failed to create RmlUi Vulkan layer color image");
    const std::string color_image_name = std::format("rmlui.layer[{}].color.image", index);
    (void)m_debug_name_writer.set(VK_OBJECT_TYPE_IMAGE,
                                  (uint64_t)layer.m_color.m_p_vk_image,
                                  color_image_name.c_str());
    layer.m_color.m_vram_scope = "vulkan.rmlui.render_layer";
    layer.m_color.m_vram_label = TextureVramLabel("layer_color", "rmlui", m_width, m_height, &layer.m_color);
    layer.m_color.m_vram_allocation_size = color_allocation_stats.size;
    RecordRmlUiVram(layer.m_color.m_vram_scope,
                    layer.m_color.m_vram_label,
                    layer.m_color.m_vram_allocation_size);

    VkImageViewCreateInfo color_view_info{};
    color_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    color_view_info.image = layer.m_color.m_p_vk_image;
    color_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    color_view_info.format = m_swapchain_format.format;
    color_view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    color_view_info.subresourceRange.baseMipLevel = 0;
    color_view_info.subresourceRange.levelCount = 1;
    color_view_info.subresourceRange.baseArrayLayer = 0;
    color_view_info.subresourceRange.layerCount = 1;
    status = vkCreateImageView(m_p_device, &color_view_info, nullptr, &layer.m_color.m_p_vk_image_view);
    RMLUI_VK_ASSERTMSG(status == VK_SUCCESS, "failed to create RmlUi Vulkan layer color image view");
    const std::string color_view_name = std::format("rmlui.layer[{}].color.view", index);
    (void)m_debug_name_writer.set(VK_OBJECT_TYPE_IMAGE_VIEW,
                                  (uint64_t)layer.m_color.m_p_vk_image_view,
                                  color_view_name.c_str());
    layer.m_color.m_p_vk_sampler = m_p_sampler_linear;
    layer.m_color_layout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImageCreateInfo depth_info{};
    depth_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    depth_info.imageType = VK_IMAGE_TYPE_2D;
    depth_info.format = m_depth_stencil_format;
    depth_info.extent = extent;
    depth_info.mipLevels = 1;
    depth_info.arrayLayers = 1;
    depth_info.samples = VK_SAMPLE_COUNT_1_BIT;
    depth_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    depth_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    depth_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationInfo depth_allocation_stats{};
    status = vmaCreateImage(m_p_allocator, &depth_info, &allocation_info, &layer.m_depth_stencil.m_p_vk_image,
                            &layer.m_depth_stencil.m_p_vma_allocation, &depth_allocation_stats);
    RMLUI_VK_ASSERTMSG(status == VK_SUCCESS, "failed to create RmlUi Vulkan layer depth/stencil image");
    const std::string depth_image_name = std::format("rmlui.layer[{}].depth-stencil.image", index);
    (void)m_debug_name_writer.set(VK_OBJECT_TYPE_IMAGE,
                                  (uint64_t)layer.m_depth_stencil.m_p_vk_image,
                                  depth_image_name.c_str());
    layer.m_depth_stencil.m_vram_scope = "vulkan.rmlui.render_layer";
    layer.m_depth_stencil.m_vram_label = TextureVramLabel("layer_depth", "rmlui", m_width, m_height, &layer.m_depth_stencil);
    layer.m_depth_stencil.m_vram_allocation_size = depth_allocation_stats.size;
    RecordRmlUiVram(layer.m_depth_stencil.m_vram_scope,
                    layer.m_depth_stencil.m_vram_label,
                    layer.m_depth_stencil.m_vram_allocation_size);

    VkImageViewCreateInfo depth_view_info{};
    depth_view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    depth_view_info.image = layer.m_depth_stencil.m_p_vk_image;
    depth_view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    depth_view_info.format = m_depth_stencil_format;
    depth_view_info.subresourceRange.aspectMask = DepthStencilAspectMask();
    depth_view_info.subresourceRange.baseMipLevel = 0;
    depth_view_info.subresourceRange.levelCount = 1;
    depth_view_info.subresourceRange.baseArrayLayer = 0;
    depth_view_info.subresourceRange.layerCount = 1;
    status = vkCreateImageView(m_p_device, &depth_view_info, nullptr, &layer.m_depth_stencil.m_p_vk_image_view);
    RMLUI_VK_ASSERTMSG(status == VK_SUCCESS, "failed to create RmlUi Vulkan layer depth/stencil image view");
    const std::string depth_view_name = std::format("rmlui.layer[{}].depth-stencil.view", index);
    (void)m_debug_name_writer.set(VK_OBJECT_TYPE_IMAGE_VIEW,
                                  (uint64_t)layer.m_depth_stencil.m_p_vk_image_view,
                                  depth_view_name.c_str());
    layer.m_depth_stencil_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    layer.width = m_width;
    layer.height = m_height;
}

RenderInterface_VK::render_layer_t* RenderInterface_VK::GetRenderLayer(Rml::LayerHandle layer_handle) {
    if (layer_handle == 0)
        return nullptr;
    const size_t index = static_cast<size_t>(layer_handle - 1);
    return index < m_render_layers.size() ? &m_render_layers[index] : nullptr;
}

const RenderInterface_VK::render_layer_t* RenderInterface_VK::GetRenderLayer(Rml::LayerHandle layer_handle) const {
    if (layer_handle == 0)
        return nullptr;
    const size_t index = static_cast<size_t>(layer_handle - 1);
    return index < m_render_layers.size() ? &m_render_layers[index] : nullptr;
}

void RenderInterface_VK::TransitionImageLayout(VkImage image, VkImageAspectFlags aspect_mask, VkImageLayout old_layout, VkImageLayout new_layout) {
    if (!m_p_current_command_buffer || !image || old_layout == new_layout)
        return;

    m_image_barriers.registerImage(image, aspect_mask, old_layout);
    m_image_barriers.transitionImage(m_p_current_command_buffer, image, aspect_mask, new_layout);
}

void RenderInterface_VK::ResetDynamicRenderState() {
    if (!m_p_current_command_buffer)
        return;
    vkCmdSetViewport(m_p_current_command_buffer, 0, 1, &m_viewport);
    VkRect2D scissor = (m_is_use_scissor_specified && !m_is_transformed_scissor_enabled) ? m_scissor : ContextClipScissor();
    scissor = IntersectContextClip(scissor);
    vkCmdSetScissor(m_p_current_command_buffer, 0, 1, &scissor);
    vkCmdSetStencilReference(m_p_current_command_buffer, VK_STENCIL_FACE_FRONT_AND_BACK, 1);
}

void RenderInterface_VK::BeginLayerRendering(Rml::LayerHandle layer_handle, bool clear) {
    EnsureRenderLayer(layer_handle);

    render_layer_t* layer = GetRenderLayer(layer_handle);
    if (!layer || !layer->m_color.m_p_vk_image_view || !layer->m_depth_stencil.m_p_vk_image_view)
        return;
    if (layer->width <= 0 || layer->height <= 0)
        return;

    TransitionImageLayout(layer->m_color.m_p_vk_image, VK_IMAGE_ASPECT_COLOR_BIT, layer->m_color_layout,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    layer->m_color_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    TransitionImageLayout(layer->m_depth_stencil.m_p_vk_image, DepthStencilAspectMask(), layer->m_depth_stencil_layout,
                          VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    layer->m_depth_stencil_layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkClearValue color_clear{};
    color_clear.color = {{0.0f, 0.0f, 0.0f, 0.0f}};
    VkClearValue depth_clear{};
    depth_clear.depthStencil = {1.0f, 0};

    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView = layer->m_color.m_p_vk_image_view;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.clearValue = color_clear;

    VkRenderingAttachmentInfo depth_attachment{};
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.imageView = layer->m_depth_stencil.m_p_vk_image_view;
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp = clear ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.clearValue = depth_clear;

    VkRenderingInfo rendering_info{};
    rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering_info.renderArea.offset = {0, 0};
    rendering_info.renderArea.extent = {static_cast<uint32_t>(layer->width),
                                        static_cast<uint32_t>(layer->height)};
    rendering_info.layerCount = 1;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachments = &color_attachment;
    rendering_info.pDepthAttachment = &depth_attachment;
    rendering_info.pStencilAttachment = (DepthStencilAspectMask() & VK_IMAGE_ASPECT_STENCIL_BIT) != 0 ? &depth_attachment : nullptr;
    vkCmdBeginRendering(m_p_current_command_buffer, &rendering_info);

    m_active_render_target = active_render_target_t::Layer;
    m_active_layer = layer_handle;
    ResetDynamicRenderState();
}

void RenderInterface_VK::BeginSwapchainRendering(VkAttachmentLoadOp color_load_op, VkAttachmentLoadOp depth_load_op) {
    if (!m_p_current_command_buffer)
        return;

    VkImageView color_view = VK_NULL_HANDLE;
    VkImageView depth_view = VK_NULL_HANDLE;
    if (m_external_context) {
        color_view = m_external_swapchain_image_view;
        depth_view = m_external_depth_stencil_image_view;
        if (m_external_swapchain_image != VK_NULL_HANDLE && m_external_swapchain_layout != VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) {
            TransitionImageLayout(m_external_swapchain_image,
                                  VK_IMAGE_ASPECT_COLOR_BIT,
                                  m_external_swapchain_layout,
                                  VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            m_external_swapchain_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        }
    }

    if (!color_view || !depth_view)
        return;

    VkClearValue color_clear{};
    color_clear.color = {{0.0f, 0.0f, 0.0f, 1.0f}};
    VkClearValue depth_clear{};
    depth_clear.depthStencil = {1.0f, 0};

    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    color_attachment.imageView = color_view;
    color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    color_attachment.loadOp = color_load_op;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.clearValue = color_clear;

    VkRenderingAttachmentInfo depth_attachment{};
    depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depth_attachment.imageView = depth_view;
    depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depth_attachment.loadOp = depth_load_op;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth_attachment.clearValue = depth_clear;

    VkRenderingInfo rendering_info{};
    rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering_info.renderArea.offset = {0, 0};
    rendering_info.renderArea.extent = {static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height)};
    rendering_info.layerCount = 1;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachments = &color_attachment;
    rendering_info.pDepthAttachment = &depth_attachment;
    rendering_info.pStencilAttachment = (DepthStencilAspectMask() & VK_IMAGE_ASPECT_STENCIL_BIT) != 0 ? &depth_attachment : nullptr;
    vkCmdBeginRendering(m_p_current_command_buffer, &rendering_info);

    m_active_render_target = active_render_target_t::Swapchain;
    m_active_layer = {};
    ResetDynamicRenderState();
}

void RenderInterface_VK::EndActiveRendering() {
    if (!m_p_current_command_buffer || m_active_render_target == active_render_target_t::None)
        return;
    vkCmdEndRendering(m_p_current_command_buffer);
    m_active_render_target = active_render_target_t::None;
    m_active_layer = {};
}

bool RenderInterface_VK::CopySwapchainToLayer(Rml::LayerHandle destination) {
    if (!m_external_context || !m_external_swapchain_image || destination == 0)
        return false;

    EnsureRenderLayer(destination);

    render_layer_t* destination_layer = GetRenderLayer(destination);
    if (!destination_layer || !destination_layer->m_color.m_p_vk_image)
        return false;
    if (destination_layer->width <= 0 || destination_layer->height <= 0)
        return false;

    EndActiveRendering();

    TransitionImageLayout(m_external_swapchain_image, VK_IMAGE_ASPECT_COLOR_BIT, m_external_swapchain_layout,
                          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    m_external_swapchain_layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    TransitionImageLayout(destination_layer->m_color.m_p_vk_image, VK_IMAGE_ASPECT_COLOR_BIT, destination_layer->m_color_layout,
                          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    destination_layer->m_color_layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

    VkImageCopy copy_region{};
    copy_region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.srcSubresource.mipLevel = 0;
    copy_region.srcSubresource.baseArrayLayer = 0;
    copy_region.srcSubresource.layerCount = 1;
    copy_region.dstSubresource = copy_region.srcSubresource;
    copy_region.extent = {static_cast<uint32_t>(destination_layer->width),
                          static_cast<uint32_t>(destination_layer->height),
                          1};

    vkCmdCopyImage(m_p_current_command_buffer, m_external_swapchain_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   destination_layer->m_color.m_p_vk_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy_region);

    TransitionImageLayout(m_external_swapchain_image, VK_IMAGE_ASPECT_COLOR_BIT, m_external_swapchain_layout,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    m_external_swapchain_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    TransitionImageLayout(destination_layer->m_color.m_p_vk_image, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    destination_layer->m_color_layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    BeginLayerRendering(destination, false);
    return true;
}

void RenderInterface_VK::RenderFullscreenTexture(texture_data_t& texture, Rml::BlendMode blend_mode) {
    (void)blend_mode;

    Rml::Vertex vertices[4];
    vertices[0].position = {0.0f, 0.0f};
    vertices[0].tex_coord = {0.0f, 0.0f};
    vertices[1].position = {static_cast<float>(m_width), 0.0f};
    vertices[1].tex_coord = {1.0f, 0.0f};
    vertices[2].position = {static_cast<float>(m_width), static_cast<float>(m_height)};
    vertices[2].tex_coord = {1.0f, 1.0f};
    vertices[3].position = {0.0f, static_cast<float>(m_height)};
    vertices[3].tex_coord = {0.0f, 1.0f};
    for (Rml::Vertex& vertex : vertices)
        vertex.colour = Rml::ColourbPremultiplied(255, 255, 255, 255);

    int indices[6] = {0, 1, 2, 0, 2, 3};

    const bool transform_enabled = m_is_transform_enabled;
    const Rml::Matrix4f transform = m_user_data_for_vertex_shader.m_transform;
    SetTransform(nullptr);
    if (Rml::CompiledGeometryHandle handle = CompileGeometry({vertices, 4}, {indices, 6})) {
        RenderGeometry(handle, {}, reinterpret_cast<Rml::TextureHandle>(&texture));
        ReleaseGeometry(handle);
    }
    m_is_transform_enabled = transform_enabled;
    m_user_data_for_vertex_shader.m_transform = transform;
}

void RenderInterface_VK::DestroyRenderLayer(render_layer_t& layer) noexcept {
    if (layer.m_color.m_p_vma_allocation)
        QueueTextureForDeferredDeletion(new texture_data_t(layer.m_color));
    if (layer.m_depth_stencil.m_p_vma_allocation)
        QueueTextureForDeferredDeletion(new texture_data_t(layer.m_depth_stencil));
    layer = {};
}

void RenderInterface_VK::DestroyRenderLayers() noexcept {
    for (render_layer_t& layer : m_render_layers)
        DestroyRenderLayer(layer);
    m_render_layers.clear();
    m_render_layer_stack_size = 0;
    m_active_layer = {};
}

void RenderInterface_VK::Update_PendingForDeletion_Textures_By_Frame(const uint32_t resource_slot) noexcept {
    auto& textures_for_previous_frame = m_pending_for_deletion_textures_by_frames[resource_slot % kSwapchainBackBufferCount];

    for (texture_data_t* p_data : textures_for_previous_frame) {
        Destroy_Texture(*p_data);
        delete p_data;
    }

    textures_for_previous_frame.clear();
}

void RenderInterface_VK::Update_PendingForDeletion_Geometries(const uint32_t resource_slot) noexcept {
    auto& geometries = m_pending_for_deletion_geometries_by_frame[resource_slot % kSwapchainBackBufferCount];
    for (geometry_handle_t* p_geometry_handle : geometries) {
        m_memory_pool.Free_GeometryHandle(p_geometry_handle);
        delete p_geometry_handle;
    }

    geometries.clear();
}

RenderInterface_VK::MemoryPool::MemoryPool() : m_memory_total_size{},
                                               m_device_min_uniform_alignment{},
                                               m_p_data{},
                                               m_p_buffer{},
                                               m_p_buffer_alloc{},
                                               m_p_device{},
                                               m_p_vk_allocator{},
                                               m_p_block{} {}

RenderInterface_VK::MemoryPool::~MemoryPool() {}

void RenderInterface_VK::MemoryPool::Initialize(VkDeviceSize byte_size, VkDeviceSize device_min_uniform_alignment, VmaAllocator p_allocator,
                                                VkDevice p_device) noexcept {
    RMLUI_VK_ASSERTMSG(byte_size > 0, "size must be valid");
    RMLUI_VK_ASSERTMSG(device_min_uniform_alignment > 0, "uniform alignment must be valid");
    RMLUI_VK_ASSERTMSG(p_device, "you must pass a valid VkDevice");
    RMLUI_VK_ASSERTMSG(p_allocator, "you must pass a valid VmaAllocator");

    m_p_device = p_device;
    m_p_vk_allocator = p_allocator;
    m_device_min_uniform_alignment = device_min_uniform_alignment;

#ifdef RMLUI_VK_DEBUG
    Rml::Log::Message(Rml::Log::LT_DEBUG, "[Vulkan][Debug] the alignment for uniform buffer is: %zu", m_device_min_uniform_alignment);
#endif

    m_memory_total_size = AlignUp<VkDeviceSize>(static_cast<VkDeviceSize>(byte_size), m_device_min_uniform_alignment);

    VkBufferCreateInfo info = {};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    info.size = m_memory_total_size;

    VmaAllocationCreateInfo info_alloc = {};

    auto p_commentary = "our pool buffer that manages all memory in vulkan (dynamic)";

    info_alloc.usage = VMA_MEMORY_USAGE_CPU_TO_GPU;
    info_alloc.flags = VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT;
    info_alloc.pUserData = const_cast<char*>(p_commentary);

    VmaAllocationInfo info_stats = {};

    auto status = vmaCreateBuffer(m_p_vk_allocator, &info, &info_alloc, &m_p_buffer, &m_p_buffer_alloc, &info_stats);

    RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vmaCreateBuffer");

    VmaVirtualBlockCreateInfo info_virtual_block = {};
    info_virtual_block.size = m_memory_total_size;

    status = vmaCreateVirtualBlock(&info_virtual_block, &m_p_block);

    RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vmaCreateVirtualBlock");

#ifdef RMLUI_VK_DEBUG
    Rml::Log::Message(Rml::Log::LT_DEBUG, "[Vulkan][Debug] Allocated memory pool [%s]", FormatByteSize(info_stats.size).c_str());
#endif

    status = vmaMapMemory(m_p_vk_allocator, m_p_buffer_alloc, (void**)&m_p_data);

    RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vmaMapMemory");
}

void RenderInterface_VK::MemoryPool::Shutdown() noexcept {
    RMLUI_VK_ASSERTMSG(m_p_vk_allocator, "you must have a valid VmaAllocator");
    RMLUI_VK_ASSERTMSG(m_p_buffer, "you must allocate VkBuffer for deleting");
    RMLUI_VK_ASSERTMSG(m_p_buffer_alloc, "you must allocate VmaAllocation for deleting");

#ifdef RMLUI_VK_DEBUG
    Rml::Log::Message(Rml::Log::LT_DEBUG, "[Vulkan][Debug] Destroyed memory pool [%s]", FormatByteSize(m_memory_total_size).c_str());
#endif

    vmaUnmapMemory(m_p_vk_allocator, m_p_buffer_alloc);
    vmaDestroyVirtualBlock(m_p_block);
    vmaDestroyBuffer(m_p_vk_allocator, m_p_buffer, m_p_buffer_alloc);
}

bool RenderInterface_VK::MemoryPool::Alloc_GeneralBuffer(VkDeviceSize size, void** p_data, VkDescriptorBufferInfo* p_out,
                                                         VmaVirtualAllocation* p_alloc) noexcept {
    RMLUI_VK_ASSERTMSG(p_out, "you must pass a valid pointer");
    RMLUI_VK_ASSERTMSG(m_p_buffer, "you must have a valid VkBuffer");

    RMLUI_VK_ASSERTMSG(*p_alloc == nullptr,
                       "you can't pass a VALID object, because it is for initialization. So it means you passed the already allocated "
                       "VmaVirtualAllocation and it means you did something wrong, like you wanted to allocate into the same object...");

    size = AlignUp<VkDeviceSize>(static_cast<VkDeviceSize>(size), m_device_min_uniform_alignment);

    VkDeviceSize offset_memory{};

    VmaVirtualAllocationCreateInfo info = {};
    info.size = size;
    info.alignment = m_device_min_uniform_alignment;

    auto status = vmaVirtualAllocate(m_p_block, &info, p_alloc, &offset_memory);

    RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vmaVirtualAllocate");

    *p_data = (void*)(m_p_data + offset_memory);

    p_out->buffer = m_p_buffer;
    p_out->offset = offset_memory;
    p_out->range = size;

    return true;
}

bool RenderInterface_VK::MemoryPool::Alloc_VertexBuffer(uint32_t number_of_elements, uint32_t stride_in_bytes, void** p_data,
                                                        VkDescriptorBufferInfo* p_out, VmaVirtualAllocation* p_alloc) noexcept {
    return Alloc_GeneralBuffer(number_of_elements * stride_in_bytes, p_data, p_out, p_alloc);
}

bool RenderInterface_VK::MemoryPool::Alloc_IndexBuffer(uint32_t number_of_elements, uint32_t stride_in_bytes, void** p_data,
                                                       VkDescriptorBufferInfo* p_out, VmaVirtualAllocation* p_alloc) noexcept {
    return Alloc_GeneralBuffer(number_of_elements * stride_in_bytes, p_data, p_out, p_alloc);
}

void RenderInterface_VK::MemoryPool::SetDescriptorSet(uint32_t binding_index, uint32_t size, VkDescriptorType descriptor_type,
                                                      VkDescriptorSet p_set) noexcept {
    RMLUI_VK_ASSERTMSG(m_p_device, "you must have a valid VkDevice here");
    RMLUI_VK_ASSERTMSG(p_set, "you must have a valid VkDescriptorSet here");
    RMLUI_VK_ASSERTMSG(m_p_buffer, "you must have a valid VkBuffer here");

    VkDescriptorBufferInfo info = {};

    info.buffer = m_p_buffer;
    info.offset = 0;
    info.range = size;

    VkWriteDescriptorSet info_write = {};

    info_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    info_write.pNext = nullptr;
    info_write.dstSet = p_set;
    info_write.descriptorCount = 1;
    info_write.descriptorType = descriptor_type;
    info_write.dstArrayElement = 0;
    info_write.dstBinding = binding_index;
    info_write.pBufferInfo = &info;

    vkUpdateDescriptorSets(m_p_device, 1, &info_write, 0, nullptr);
}

void RenderInterface_VK::MemoryPool::SetDescriptorSet(uint32_t binding_index, VkDescriptorBufferInfo* p_info, VkDescriptorType descriptor_type,
                                                      VkDescriptorSet p_set) noexcept {
    RMLUI_VK_ASSERTMSG(m_p_device, "you must have a valid VkDevice here");
    RMLUI_VK_ASSERTMSG(p_set, "you must have a valid VkDescriptorSet here");
    RMLUI_VK_ASSERTMSG(m_p_buffer, "you must have a valid VkBuffer here");
    RMLUI_VK_ASSERTMSG(p_info, "must be valid pointer");

    VkWriteDescriptorSet info_write = {};

    info_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    info_write.pNext = nullptr;
    info_write.dstSet = p_set;
    info_write.descriptorCount = 1;
    info_write.descriptorType = descriptor_type;
    info_write.dstArrayElement = 0;
    info_write.dstBinding = binding_index;
    info_write.pBufferInfo = p_info;

    vkUpdateDescriptorSets(m_p_device, 1, &info_write, 0, nullptr);
}

void RenderInterface_VK::MemoryPool::SetDescriptorSet(uint32_t binding_index, VkSampler p_sampler, VkImageLayout layout, VkImageView p_view,
                                                      VkDescriptorType descriptor_type, VkDescriptorSet p_set) noexcept {
    RMLUI_VK_ASSERTMSG(m_p_device, "you must have a valid VkDevice here");
    RMLUI_VK_ASSERTMSG(p_set, "you must have a valid VkDescriptorSet here");
    RMLUI_VK_ASSERTMSG(m_p_buffer, "you must have a valid VkBuffer here");
    RMLUI_VK_ASSERTMSG(p_view, "you must have a valid VkImageView");
    RMLUI_VK_ASSERTMSG(p_sampler, "you must have a valid VkSampler here");

    VkDescriptorImageInfo info = {};

    info.imageLayout = layout;
    info.imageView = p_view;
    info.sampler = p_sampler;

    VkWriteDescriptorSet info_write = {};

    info_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    info_write.pNext = nullptr;
    info_write.dstSet = p_set;
    info_write.descriptorCount = 1;
    info_write.descriptorType = descriptor_type;
    info_write.dstArrayElement = 0;
    info_write.dstBinding = binding_index;
    info_write.pImageInfo = &info;

    vkUpdateDescriptorSets(m_p_device, 1, &info_write, 0, nullptr);
}

void RenderInterface_VK::MemoryPool::Free_Allocation(VmaVirtualAllocation allocation) noexcept {
    if (allocation)
        vmaVirtualFree(m_p_block, allocation);
}

void RenderInterface_VK::MemoryPool::Free_GeometryHandle(geometry_handle_t* p_valid_geometry_handle) noexcept {
    RMLUI_VK_ASSERTMSG(p_valid_geometry_handle,
                       "you must pass a VALID pointer to geometry_handle_t, otherwise something is wrong and debug your code");
    RMLUI_VK_ASSERTMSG(p_valid_geometry_handle->m_p_vertex_allocation, "you must have a VALID pointer of VmaAllocation for vertex buffer");
    RMLUI_VK_ASSERTMSG(p_valid_geometry_handle->m_p_index_allocation, "you must have a VALID pointer of VmaAllocation for index buffer");

    // TODO: The following assertion is disabled for now. The shader allocation pointer is only set once the geometry
    // handle is rendered with. However, currently the Vulkan renderer does not handle all draw calls from RmlUi, so
    // this pointer may never be set if the geometry was only used in a unsupported draw calls. This can then trigger
    // the following assertion. The free call below gracefully handles zero pointers so this should be safe regardless.
    // RMLUI_VK_ASSERTMSG(p_valid_geometry_handle->m_p_shader_allocation,
    //		"you must have a VALID pointer of VmaAllocation for shader operations (like uniforms and etc)");

    RMLUI_VK_ASSERTMSG(m_p_block, "you have to allocate the virtual block before do this operation...");

    Free_Allocation(p_valid_geometry_handle->m_p_vertex_allocation);
    Free_Allocation(p_valid_geometry_handle->m_p_index_allocation);
    Free_Allocation(p_valid_geometry_handle->m_p_shader_allocation);

    p_valid_geometry_handle->m_p_vertex_allocation = nullptr;
    p_valid_geometry_handle->m_p_shader_allocation = nullptr;
    p_valid_geometry_handle->m_p_index_allocation = nullptr;
    p_valid_geometry_handle->m_num_indices = 0;
}

void RenderInterface_VK::MemoryPool::Free_GeometryHandle_ShaderDataOnly(geometry_handle_t* p_valid_geometry_handle) noexcept {
    RMLUI_VK_ASSERTMSG(p_valid_geometry_handle,
                       "you must pass a VALID pointer to geometry_handle_t, otherwise something is wrong and debug your code");
    RMLUI_VK_ASSERTMSG(p_valid_geometry_handle->m_p_vertex_allocation, "you must have a VALID pointer of VmaAllocation for vertex buffer");
    RMLUI_VK_ASSERTMSG(p_valid_geometry_handle->m_p_index_allocation, "you must have a VALID pointer of VmaAllocation for index buffer");
    RMLUI_VK_ASSERTMSG(p_valid_geometry_handle->m_p_shader_allocation,
                       "you must have a VALID pointer of VmaAllocation for shader operations (like uniforms and etc)");
    RMLUI_VK_ASSERTMSG(m_p_block, "you have to allocate the virtual block before do this operation...");

    Free_Allocation(p_valid_geometry_handle->m_p_shader_allocation);
    p_valid_geometry_handle->m_p_shader_allocation = nullptr;
}

#include <vk_mem_alloc.h>

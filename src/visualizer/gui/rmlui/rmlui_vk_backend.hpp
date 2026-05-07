/* Adapted from RmlUi 6.2 Backends/RmlUi_Renderer_VK.h.
 *
 * SPDX-License-Identifier: MIT */

#pragma once

#include <RmlUi/Core/RenderInterface.h>

#include "window/vulkan_image_barrier_tracker.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define VK_USE_PLATFORM_WIN32_KHR
#include <windows.h>
#endif

#ifndef VMA_STATIC_VULKAN_FUNCTIONS
#define VMA_STATIC_VULKAN_FUNCTIONS 1
#endif
#ifndef VMA_DYNAMIC_VULKAN_FUNCTIONS
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#endif

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <atomic>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef RMLUI_DEBUG
#define RMLUI_VK_ASSERTMSG(statement, msg) RMLUI_ASSERTMSG(statement, msg)

// Uncomment the following line to enable additional Vulkan debugging.
// #define RMLUI_VK_DEBUG
#else
#define RMLUI_VK_ASSERTMSG(statement, msg) static_cast<void>(statement)
#endif

// Your specified API version. Ideally, this will be dynamic in the future.
#define RMLUI_VK_API_VERSION VK_API_VERSION_1_0

class RenderInterface_VK : public Rml::RenderInterface {
public:
    static constexpr uint32_t kSwapchainBackBufferCount = 3;
    static constexpr VkDeviceSize kVideoMemoryForAllocation = 4 * 1024 * 1024; // [bytes]

    RenderInterface_VK();
    ~RenderInterface_VK();

    using CreateSurfaceCallback = bool (*)(VkInstance instance, VkSurfaceKHR* out_surface);

    struct ExternalContext {
        VkInstance instance = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        VkDevice device = VK_NULL_HANDLE;
        VkPipelineCache pipeline_cache = VK_NULL_HANDLE;
        VkQueue graphics_queue = VK_NULL_HANDLE;
        uint32_t graphics_queue_family = 0;
        VkFormat color_format = VK_FORMAT_UNDEFINED;
        VkFormat depth_stencil_format = VK_FORMAT_UNDEFINED;
        VkExtent2D extent{};
    };

    bool Initialize(Rml::Vector<const char*> required_extensions, CreateSurfaceCallback create_surface_callback);
    void Shutdown();

    bool InitializeExternal(const ExternalContext& context);
    void ShutdownExternal();
    void BeginExternalFrame(VkCommandBuffer command_buffer,
                            VkExtent2D extent,
                            VkImage swapchain_image,
                            VkImageView swapchain_image_view,
                            VkImageView depth_stencil_image_view,
                            std::size_t frame_slot);
    void EndExternalFrame();
    void ResetContextRenderState();
    void SetContextOffset(float offset_x, float offset_y);
    void SetContextClipRect(float x1, float y1, float x2, float y2);

    void BeginFrame();
    void EndFrame();

    void SetViewport(int width, int height);
    bool IsSwapchainValid();
    void RecreateSwapchain();

    // Build a Rml::Image src URL referencing an externally-owned VkImageView/VkSampler.
    // The view+sampler must remain alive while any element references this URL. The
    // returned URL form is "lfs-vk://?v=<view_hex>&s=<sampler_hex>&w=W&h=H".
    static std::string MakeExternalTextureSource(VkImageView image_view, VkSampler sampler, int width, int height);

    // -- Inherited from Rml::RenderInterface --

    /// Called by RmlUi when it wants to compile geometry it believes will be static for the forseeable future.
    Rml::CompiledGeometryHandle CompileGeometry(Rml::Span<const Rml::Vertex> vertices, Rml::Span<const int> indices) override;
    /// Called by RmlUi when it wants to render application-compiled geometry.
    void RenderGeometry(Rml::CompiledGeometryHandle handle, Rml::Vector2f translation, Rml::TextureHandle texture) override;
    /// Called by RmlUi when it wants to release application-compiled geometry.
    void ReleaseGeometry(Rml::CompiledGeometryHandle geometry) override;

    /// Called by RmlUi when a texture is required by the library.
    Rml::TextureHandle LoadTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source) override;
    /// Called by RmlUi when a texture is required to be built from an internally-generated sequence of pixels.
    Rml::TextureHandle GenerateTexture(Rml::Span<const Rml::byte> source_data, Rml::Vector2i source_dimensions) override;
    /// Called by RmlUi when a loaded texture is no longer required.
    void ReleaseTexture(Rml::TextureHandle texture_handle) override;

    /// Called by RmlUi when it wants to enable or disable scissoring to clip content.
    void EnableScissorRegion(bool enable) override;
    /// Called by RmlUi when it wants to change the scissor region.
    void SetScissorRegion(Rml::Rectanglei region) override;

    /// Called by RmlUi when it wants to enable or disable the clip mask.
    void EnableClipMask(bool enable) override;
    /// Called by RmlUi when it wants to set or modify the contents of the clip mask.
    void RenderToClipMask(Rml::ClipMaskOperation operation, Rml::CompiledGeometryHandle geometry, Rml::Vector2f translation) override;

    /// Called by RmlUi when it wants to render to an offscreen layer.
    Rml::LayerHandle PushLayer() override;
    /// Called by RmlUi when it wants to composite one layer onto another.
    void CompositeLayers(Rml::LayerHandle source, Rml::LayerHandle destination, Rml::BlendMode blend_mode,
                         Rml::Span<const Rml::CompiledFilterHandle> filters) override;
    /// Called by RmlUi when it wants to remove the current layer.
    void PopLayer() override;
    /// Called by RmlUi when it wants to capture the current layer as a texture.
    Rml::TextureHandle SaveLayerAsTexture() override;

    /// Called by RmlUi when it wants to set the current transform matrix to a new matrix.
    void SetTransform(const Rml::Matrix4f* transform) override;

private:
    enum class shader_type_t : int { Vertex,
                                     Fragment,
                                     Unknown = -1 };
    enum class shader_id_t : int { Vertex,
                                   Fragment_WithoutTextures,
                                   Fragment_WithTextures };

    struct shader_vertex_user_data_t {
        // Member objects are order-sensitive to match shader.
        Rml::Matrix4f m_transform;
        Rml::Vector2f m_translate;
    };

    struct texture_data_t {
        VkImage m_p_vk_image;
        VkImageView m_p_vk_image_view;
        VkSampler m_p_vk_sampler;
        VkDescriptorSet m_p_vk_descriptor_set;
        VmaAllocation m_p_vma_allocation;
    };

    struct async_preview_result_t {
        std::vector<Rml::byte> pixels;
        int width = 0;
        int height = 0;
    };

    struct async_preview_state_t {
        texture_data_t* texture = nullptr;
        Rml::String source;
        std::mutex mutex;
        async_preview_result_t result;
        std::atomic<bool> ready = false;
    };

    struct geometry_handle_t {
        int m_num_indices;

        VkDescriptorBufferInfo m_p_vertex;
        VkDescriptorBufferInfo m_p_index;
        VkDescriptorBufferInfo m_p_shader;

        // @ this is for freeing our logical blocks for VMA
        // see https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/virtual_allocator.html
        VmaVirtualAllocation m_p_vertex_allocation;
        VmaVirtualAllocation m_p_index_allocation;
        VmaVirtualAllocation m_p_shader_allocation;
    };

    struct buffer_data_t {
        VkBuffer m_p_vk_buffer;
        VmaAllocation m_p_vma_allocation;
    };

    struct render_layer_t {
        texture_data_t m_color{};
        texture_data_t m_depth_stencil{};
        VkImageLayout m_color_layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkImageLayout m_depth_stencil_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    };

    enum class active_render_target_t { None,
                                        Swapchain,
                                        Layer };

    class UploadResourceManager {
    public:
        UploadResourceManager() : m_p_device{},
                                  m_p_fence{},
                                  m_p_command_buffer{},
                                  m_p_command_pool{},
                                  m_p_graphics_queue{} {}
        ~UploadResourceManager() {}

        void Initialize(VkDevice p_device, VkQueue p_queue, uint32_t queue_family_index) {
            RMLUI_VK_ASSERTMSG(p_queue, "you have to pass a valid VkQueue");
            RMLUI_VK_ASSERTMSG(p_device, "you have to pass a valid VkDevice for creation resources");

            m_p_device = p_device;
            m_p_graphics_queue = p_queue;

            Create_All(queue_family_index);
        }

        void Shutdown() {
            vkDestroyFence(m_p_device, m_p_fence, nullptr);
            vkDestroyCommandPool(m_p_device, m_p_command_pool, nullptr);
        }

        template <typename Func>
        void UploadToGPU(Func&& p_user_commands) noexcept {
            RMLUI_VK_ASSERTMSG(m_p_command_buffer, "you didn't initialize VkCommandBuffer");

            VkCommandBufferBeginInfo info_command = {};

            info_command.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            info_command.pNext = nullptr;
            info_command.pInheritanceInfo = nullptr;
            info_command.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

            VkResult status = vkBeginCommandBuffer(m_p_command_buffer, &info_command);

            RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vkBeginCommandBuffer");

            p_user_commands(m_p_command_buffer);

            status = vkEndCommandBuffer(m_p_command_buffer);

            RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "faield to vkEndCommandBuffer");

            Submit();
            Wait();
        }

    private:
        void Create_Fence() noexcept {
            VkFenceCreateInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            info.pNext = nullptr;
            info.flags = 0;

            VkResult status = vkCreateFence(m_p_device, &info, nullptr, &m_p_fence);

            RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vkCreateFence");
        }

        void Create_CommandBuffer() noexcept {
            RMLUI_VK_ASSERTMSG(m_p_command_pool, "you have to initialize VkCommandPool before calling this method!");

            VkCommandBufferAllocateInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            info.pNext = nullptr;
            info.commandPool = m_p_command_pool;
            info.commandBufferCount = 1;
            info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

            VkResult status = vkAllocateCommandBuffers(m_p_device, &info, &m_p_command_buffer);

            RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vkAllocateCommandBuffers");
        }

        void Create_CommandPool(uint32_t queue_family_index) noexcept {
            VkCommandPoolCreateInfo info = {};

            info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            info.pNext = nullptr;
            info.queueFamilyIndex = queue_family_index;
            info.flags = 0;

            VkResult status = vkCreateCommandPool(m_p_device, &info, nullptr, &m_p_command_pool);

            RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vkCreateCommandPool");
        }

        void Create_All(uint32_t queue_family_index) noexcept {
            Create_Fence();
            Create_CommandPool(queue_family_index);
            Create_CommandBuffer();
        }

        void Wait() noexcept {
            RMLUI_VK_ASSERTMSG(m_p_fence, "you must initialize your VkFence");

            vkWaitForFences(m_p_device, 1, &m_p_fence, VK_TRUE, UINT64_MAX);
            vkResetFences(m_p_device, 1, &m_p_fence);
            vkResetCommandPool(m_p_device, m_p_command_pool, 0);
        }

        void Submit() noexcept {
            VkSubmitInfo info = {};

            info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            info.pNext = nullptr;
            info.waitSemaphoreCount = 0;
            info.signalSemaphoreCount = 0;
            info.pSignalSemaphores = nullptr;
            info.pWaitSemaphores = nullptr;
            info.pWaitDstStageMask = nullptr;
            info.pCommandBuffers = &m_p_command_buffer;
            info.commandBufferCount = 1;

            auto status = vkQueueSubmit(m_p_graphics_queue, 1, &info, m_p_fence);

            RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vkQueueSubmit");
        }

    private:
        VkDevice m_p_device;
        VkFence m_p_fence;
        VkCommandBuffer m_p_command_buffer;
        VkCommandPool m_p_command_pool;
        VkQueue m_p_graphics_queue;
    };

    // @ main manager for "allocating" vertex, index, uniform stuff
    class MemoryPool {
    public:
        MemoryPool();
        ~MemoryPool();

        void Initialize(VkDeviceSize byte_size, VkDeviceSize device_min_uniform_alignment, VmaAllocator p_allocator, VkDevice p_device) noexcept;
        void Shutdown() noexcept;

        bool Alloc_GeneralBuffer(VkDeviceSize size, void** p_data, VkDescriptorBufferInfo* p_out, VmaVirtualAllocation* p_alloc) noexcept;
        bool Alloc_VertexBuffer(uint32_t number_of_elements, uint32_t stride_in_bytes, void** p_data, VkDescriptorBufferInfo* p_out,
                                VmaVirtualAllocation* p_alloc) noexcept;
        bool Alloc_IndexBuffer(uint32_t number_of_elements, uint32_t stride_in_bytes, void** p_data, VkDescriptorBufferInfo* p_out,
                               VmaVirtualAllocation* p_alloc) noexcept;

        void SetDescriptorSet(uint32_t binding_index, uint32_t size, VkDescriptorType descriptor_type, VkDescriptorSet p_set) noexcept;
        void SetDescriptorSet(uint32_t binding_index, VkDescriptorBufferInfo* p_info, VkDescriptorType descriptor_type,
                              VkDescriptorSet p_set) noexcept;
        void SetDescriptorSet(uint32_t binding_index, VkSampler p_sampler, VkImageLayout layout, VkImageView p_view, VkDescriptorType descriptor_type,
                              VkDescriptorSet p_set) noexcept;

        void Free_Allocation(VmaVirtualAllocation allocation) noexcept;
        void Free_GeometryHandle(geometry_handle_t* p_valid_geometry_handle) noexcept;
        void Free_GeometryHandle_ShaderDataOnly(geometry_handle_t* p_valid_geometry_handle) noexcept;

    private:
        VkDeviceSize m_memory_total_size;
        VkDeviceSize m_device_min_uniform_alignment;
        char* m_p_data;
        VkBuffer m_p_buffer;
        VmaAllocation m_p_buffer_alloc;
        VkDevice m_p_device;
        VmaAllocator m_p_vk_allocator;
        VmaVirtualBlock m_p_block;
    };

    // If we need additional command buffers, we can add them to this list and retrieve them from the ring.
    enum class CommandBufferName { Primary,
                                   Count };

    // The command buffer ring stores a unique set of named command buffers for each bufferd frame.
    // Explanation of how to use Vulkan efficiently: https://vkguide.dev/docs/chapter-4/double_buffering/
    class CommandBufferRing {
    public:
        static constexpr uint32_t kNumFramesToBuffer = kSwapchainBackBufferCount;
        static constexpr uint32_t kNumCommandBuffersPerFrame = static_cast<uint32_t>(CommandBufferName::Count);

        CommandBufferRing();

        void Initialize(VkDevice p_device, uint32_t queue_index_graphics) noexcept;
        void Shutdown();

        void OnBeginFrame();
        VkCommandBuffer GetCommandBufferForActiveFrame(CommandBufferName named_command_buffer);

    private:
        struct CommandBuffersPerFrame {
            Rml::Array<VkCommandPool, kNumCommandBuffersPerFrame> m_command_pools;
            Rml::Array<VkCommandBuffer, kNumCommandBuffersPerFrame> m_command_buffers;
        };

        VkDevice m_p_device;
        uint32_t m_frame_index;
        CommandBuffersPerFrame* m_p_current_frame;
        Rml::Array<CommandBuffersPerFrame, kNumFramesToBuffer> m_frames;
    };

    class DescriptorPoolManager {
    public:
        DescriptorPoolManager() : m_allocated_descriptor_count{},
                                  m_p_descriptor_pool{} {}
        ~DescriptorPoolManager() {
            RMLUI_VK_ASSERTMSG(m_allocated_descriptor_count <= 0, "something is wrong. You didn't free some VkDescriptorSet");
        }

        void Initialize(VkDevice p_device, uint32_t count_uniform_buffer, uint32_t count_image_sampler, uint32_t count_sampler,
                        uint32_t count_storage_buffer) noexcept {
            RMLUI_VK_ASSERTMSG(p_device, "you can't pass an invalid VkDevice here");

            Rml::Array<VkDescriptorPoolSize, 5> sizes;
            sizes[0] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, count_uniform_buffer};
            sizes[1] = {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, count_uniform_buffer};
            sizes[2] = {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, count_image_sampler};
            sizes[3] = {VK_DESCRIPTOR_TYPE_SAMPLER, count_sampler};
            sizes[4] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count_storage_buffer};

            VkDescriptorPoolCreateInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            info.pNext = nullptr;
            info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
            info.maxSets = 1000;
            info.poolSizeCount = static_cast<uint32_t>(sizes.size());
            info.pPoolSizes = sizes.data();

            auto status = vkCreateDescriptorPool(p_device, &info, nullptr, &m_p_descriptor_pool);
            RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vkCreateDescriptorPool");
        }

        void Shutdown(VkDevice p_device) {
            RMLUI_VK_ASSERTMSG(p_device, "you can't pass an invalid VkDevice here");

            vkDestroyDescriptorPool(p_device, m_p_descriptor_pool, nullptr);
        }

        uint32_t Get_AllocatedDescriptorCount() const noexcept { return m_allocated_descriptor_count; }

        bool Alloc_Descriptor(VkDevice p_device, VkDescriptorSetLayout* p_layouts, VkDescriptorSet* p_sets,
                              uint32_t descriptor_count_for_creation = 1) noexcept {
            RMLUI_VK_ASSERTMSG(p_layouts, "you have to pass a valid and initialized VkDescriptorSetLayout (probably you must create it)");
            RMLUI_VK_ASSERTMSG(p_device, "you must pass a valid VkDevice here");

            VkDescriptorSetAllocateInfo info = {};
            info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            info.pNext = nullptr;
            info.descriptorPool = m_p_descriptor_pool;
            info.descriptorSetCount = descriptor_count_for_creation;
            info.pSetLayouts = p_layouts;

            auto status = vkAllocateDescriptorSets(p_device, &info, p_sets);
            RMLUI_VK_ASSERTMSG(status == VkResult::VK_SUCCESS, "failed to vkAllocateDescriptorSets");

            m_allocated_descriptor_count += descriptor_count_for_creation;

            return status == VkResult::VK_SUCCESS;
        }

        void Free_Descriptors(VkDevice p_device, VkDescriptorSet* p_sets, uint32_t descriptor_count = 1) noexcept {
            RMLUI_VK_ASSERTMSG(p_device, "you must pass a valid VkDevice here");

            if (p_sets) {
                m_allocated_descriptor_count -= descriptor_count;
                vkFreeDescriptorSets(p_device, m_p_descriptor_pool, descriptor_count, p_sets);
            }
        }

    private:
        int m_allocated_descriptor_count;
        VkDescriptorPool m_p_descriptor_pool;
    };

    struct PhysicalDeviceWrapper {
        VkPhysicalDevice m_p_physical_device;
        VkPhysicalDeviceProperties m_physical_device_properties;
    };

    using PhysicalDeviceWrapperList = Rml::Vector<PhysicalDeviceWrapper>;
    using LayerPropertiesList = Rml::Vector<VkLayerProperties>;
    using ExtensionPropertiesList = Rml::Vector<VkExtensionProperties>;

private:
    Rml::TextureHandle CreateTexture(Rml::Span<const Rml::byte> source, Rml::Vector2i dimensions,
                                     const Rml::String& name, VkSampler sampler = VK_NULL_HANDLE);
    Rml::TextureHandle LoadAsyncPreviewTexture(Rml::Vector2i& texture_dimensions, const Rml::String& source);
    void ProcessAsyncPreviewUploads();
    void DropAsyncPreviewTexture(texture_data_t* texture);
    void QueueTextureForDeferredDeletion(texture_data_t* texture);
    static async_preview_result_t DecodePreviewTexture(std::filesystem::path path, int max_size);

    void Initialize_Instance(Rml::Vector<const char*> required_extensions) noexcept;
    void Initialize_Device() noexcept;
    void Initialize_PhysicalDevice(VkPhysicalDeviceProperties& out_physical_device_properties) noexcept;
    void Initialize_Swapchain(VkExtent2D window_extent) noexcept;
    void Initialize_Surface(CreateSurfaceCallback create_surface_callback) noexcept;
    void Initialize_QueueIndecies() noexcept;
    void Initialize_Queues() noexcept;
    void Initialize_SyncPrimitives() noexcept;
    void Initialize_Resources(const VkPhysicalDeviceProperties& physical_device_properties) noexcept;
    void Initialize_Allocator() noexcept;

    void Destroy_Instance() noexcept;
    void Destroy_Device() noexcept;
    void Destroy_Swapchain() noexcept;
    void Destroy_Surface() noexcept;
    void Destroy_SyncPrimitives() noexcept;
    void Destroy_Resources() noexcept;
    void Destroy_Allocator() noexcept;

    void QueryInstanceLayers(LayerPropertiesList& result) noexcept;
    void QueryInstanceExtensions(ExtensionPropertiesList& result, const LayerPropertiesList& instance_layer_properties) noexcept;
    bool AddLayerToInstance(Rml::Vector<const char*>& result, const LayerPropertiesList& instance_layer_properties,
                            const char* p_instance_layer_name) noexcept;
    bool AddExtensionToInstance(Rml::Vector<const char*>& result, const ExtensionPropertiesList& instance_extension_properties,
                                const char* p_instance_extension_name) noexcept;
    void CreatePropertiesFor_Instance(Rml::Vector<const char*>& instance_layer_names, Rml::Vector<const char*>& instance_extension_names) noexcept;

    bool IsLayerPresent(const LayerPropertiesList& properties, const char* p_layer_name) noexcept;
    bool IsExtensionPresent(const ExtensionPropertiesList& properties, const char* p_extension_name) noexcept;

    bool AddExtensionToDevice(Rml::Vector<const char*>& result, const ExtensionPropertiesList& device_extension_properties,
                              const char* p_device_extension_name) noexcept;
    void CreatePropertiesFor_Device(ExtensionPropertiesList& result) noexcept;

    void CreateReportDebugCallback() noexcept;
    void Destroy_ReportDebugCallback() noexcept;

    uint32_t GetUserAPIVersion() const noexcept;
    uint32_t GetRequiredVersionAndValidateMachine() noexcept;

    void CollectPhysicalDevices(PhysicalDeviceWrapperList& out_physical_devices) noexcept;
    const PhysicalDeviceWrapper* ChoosePhysicalDevice(const PhysicalDeviceWrapperList& physical_devices, VkPhysicalDeviceType device_type) noexcept;

    VkSurfaceFormatKHR ChooseSwapchainFormat() noexcept;
    VkSurfaceTransformFlagBitsKHR CreatePretransformSwapchain() noexcept;
    VkCompositeAlphaFlagBitsKHR ChooseSwapchainCompositeAlpha() noexcept;
    int Choose_SwapchainImageCount(uint32_t user_swapchain_count_for_creation = kSwapchainBackBufferCount, bool if_failed_choose_min = true) noexcept;
    VkPresentModeKHR GetPresentMode(VkPresentModeKHR type = VkPresentModeKHR::VK_PRESENT_MODE_FIFO_KHR) noexcept;
    VkSurfaceCapabilitiesKHR GetSurfaceCapabilities() noexcept;

    VkExtent2D GetValidSurfaceExtent() noexcept;

    void CreateShaders() noexcept;
    void CreateDescriptorSetLayout() noexcept;
    void CreatePipelineLayout() noexcept;
    void CreateDescriptorSets() noexcept;
    void CreateSamplers() noexcept;
    void Create_Pipelines() noexcept;

    // This method is called in Views, so don't call it manually
    void CreateSwapchainImages() noexcept;
    void CreateSwapchainImageViews() noexcept;

    void Create_DepthStencilImage() noexcept;
    void Create_DepthStencilImageViews() noexcept;

    void UpdateViewportState(const VkExtent2D& real_render_image_size) noexcept;
    void CreateResourcesDependentOnSize(const VkExtent2D& real_render_image_size) noexcept;

    buffer_data_t CreateResource_StagingBuffer(VkDeviceSize size, VkBufferUsageFlags flags) noexcept;
    void DestroyResource_StagingBuffer(const buffer_data_t& data) noexcept;

    void Destroy_Textures() noexcept;
    void Destroy_Geometries() noexcept;

    void Destroy_Texture(const texture_data_t& p_texture) noexcept;

    void DestroyResourcesDependentOnSize() noexcept;
    void DestroySwapchainImageViews() noexcept;
    void DestroyRenderLayers() noexcept;
    void DestroyRenderLayer(render_layer_t& layer) noexcept;
    void Destroy_Pipelines() noexcept;
    void DestroyDescriptorSets() noexcept;
    void DestroyPipelineLayout() noexcept;
    void DestroySamplers() noexcept;
    void FreeTransientShaderAllocations(uint32_t resource_slot) noexcept;
    void FreeAllTransientShaderAllocations() noexcept;

    void EnsureRenderLayer(Rml::LayerHandle layer_handle);
    render_layer_t* GetRenderLayer(Rml::LayerHandle layer_handle);
    const render_layer_t* GetRenderLayer(Rml::LayerHandle layer_handle) const;
    void BeginLayerRendering(Rml::LayerHandle layer_handle, bool clear);
    void BeginSwapchainRendering(VkAttachmentLoadOp color_load_op, VkAttachmentLoadOp depth_load_op);
    void EndActiveRendering();
    bool CopySwapchainToLayer(Rml::LayerHandle destination);
    void TransitionImageLayout(VkImage image, VkImageAspectFlags aspect_mask, VkImageLayout old_layout, VkImageLayout new_layout);
    VkImageAspectFlags DepthStencilAspectMask() const noexcept;
    void ApplyTransformState();
    VkRect2D ContextClipScissor() const noexcept;
    VkRect2D IntersectContextClip(VkRect2D scissor) const noexcept;
    void ResetDynamicRenderState();
    void RenderFullscreenTexture(texture_data_t& texture, Rml::BlendMode blend_mode);

    void Wait() noexcept;

    void Update_PendingForDeletion_Textures_By_Frame(uint32_t resource_slot) noexcept;
    void Update_PendingForDeletion_Geometries(uint32_t resource_slot) noexcept;
    uint32_t ActiveResourceSlot() const noexcept;
    void WaitForSubmittedFrames() noexcept;

    void Submit() noexcept;
    void Present() noexcept;

    VkFormat Get_SupportedDepthFormat();

private:
    bool m_is_transform_enabled;
    bool m_is_apply_to_regular_geometry_stencil;
    bool m_is_clip_mask_enabled;
    bool m_is_transformed_scissor_enabled;
    bool m_is_use_scissor_specified;
    bool m_is_use_stencil_pipeline;
    bool m_external_context = false;

    int m_width;
    int m_height;

    uint32_t m_queue_index_present;
    uint32_t m_queue_index_graphics;
    uint32_t m_queue_index_compute;
    uint32_t m_semaphore_index;
    uint32_t m_semaphore_index_previous;
    uint32_t m_resource_slot = 0;
    uint32_t m_reclaim_resource_slot = 0;
    uint32_t m_image_index;

    VkInstance m_p_instance;
    VkDevice m_p_device;
    VkPhysicalDevice m_p_physical_device;
    VkSurfaceKHR m_p_surface;
    VkSwapchainKHR m_p_swapchain;
    VkPipelineCache m_p_pipeline_cache;
    VmaAllocator m_p_allocator;
    // VK_EXT_host_image_copy: when both pointers load (extension + feature
    // available), texture upload skips the staging-buffer + cmd-buffer +
    // fence-wait path and copies host memory directly into the image.
    PFN_vkCopyMemoryToImageEXT m_pfn_copy_memory_to_image = nullptr;
    PFN_vkTransitionImageLayoutEXT m_pfn_transition_image_layout = nullptr;
    // @ obtained from command list see PrepareRenderBuffer method
    VkCommandBuffer m_p_current_command_buffer;

    VkDescriptorSetLayout m_p_descriptor_set_layout_vertex_transform;
    VkDescriptorSetLayout m_p_descriptor_set_layout_texture;
    VkPipelineLayout m_p_pipeline_layout;
    VkPipeline m_p_pipeline_with_textures;
    VkPipeline m_p_pipeline_without_textures;
    VkPipeline m_p_pipeline_stencil_for_region_where_geometry_will_be_drawn;
    VkPipeline m_p_pipeline_stencil_for_regular_geometry_that_applied_to_region_with_textures;
    VkPipeline m_p_pipeline_stencil_for_regular_geometry_that_applied_to_region_without_textures;
    VkDescriptorSet m_p_descriptor_set;
    VkSampler m_p_sampler_linear;
    VkSampler m_p_sampler_nearest;
    VkRect2D m_scissor;

    // @ means it captures the window size full width and full height, offset equals both x and y to 0
    VkRect2D m_scissor_original;
    VkViewport m_viewport;

    VkQueue m_p_queue_present;
    VkQueue m_p_queue_graphics;
    VkQueue m_p_queue_compute;

#ifdef RMLUI_VK_DEBUG
    VkDebugUtilsMessengerEXT m_debug_messenger;
#endif

    VkSurfaceFormatKHR m_swapchain_format;
    VkFormat m_depth_stencil_format = VK_FORMAT_UNDEFINED;
    shader_vertex_user_data_t m_user_data_for_vertex_shader;
    Rml::Matrix4f m_context_transform;
    Rml::Matrix4f m_rml_transform;
    Rml::Vector2f m_context_offset;
    VkRect2D m_context_clip_scissor{};
    bool m_context_clip_enabled = false;
    texture_data_t m_texture_depthstencil;

    Rml::Matrix4f m_projection;
    Rml::Vector<VkFence> m_executed_fences;
    Rml::Vector<VkSemaphore> m_semaphores_image_available;
    Rml::Vector<VkSemaphore> m_semaphores_finished_render;
    Rml::Vector<VkImage> m_swapchain_images;
    Rml::Vector<VkImageView> m_swapchain_image_views;
    Rml::Vector<VkImageLayout> m_swapchain_image_layouts;
    Rml::Vector<VkShaderModule> m_shaders;
    Rml::Array<Rml::Vector<texture_data_t*>, kSwapchainBackBufferCount> m_pending_for_deletion_textures_by_frames;
    std::vector<std::shared_ptr<async_preview_state_t>> m_async_preview_textures;
    Rml::Vector<render_layer_t> m_render_layers;
    Rml::Array<Rml::Vector<VmaVirtualAllocation>, kSwapchainBackBufferCount> m_transient_shader_allocations_by_frame;
    VkImage m_external_swapchain_image = VK_NULL_HANDLE;
    VkImageView m_external_swapchain_image_view = VK_NULL_HANDLE;
    VkImageView m_external_depth_stencil_image_view = VK_NULL_HANDLE;
    VkImageLayout m_external_swapchain_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout m_depth_stencil_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    lfs::vis::VulkanImageBarrierTracker m_image_barriers;
    active_render_target_t m_active_render_target = active_render_target_t::None;
    Rml::LayerHandle m_active_layer = 0;
    int m_render_layer_stack_size = 0;

    // vma handles that thing, so there's no need for frame splitting
    Rml::Array<Rml::Vector<geometry_handle_t*>, kSwapchainBackBufferCount> m_pending_for_deletion_geometries_by_frame;

    CommandBufferRing m_command_buffer_ring;
    MemoryPool m_memory_pool;
    UploadResourceManager m_upload_manager;
    DescriptorPoolManager m_manager_descriptors;
};

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vulkan_scene_image_uploader.hpp"

#include "core/logger.hpp"
#include "core/tensor.hpp"
#include "diagnostics/vram_profiler.hpp"
#include "vulkan_viewport_pass.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_image_barrier_tracker.hpp"
#include "window/vulkan_result.hpp"

#include <format>
#include <string>

namespace lfs::vis {
    struct VulkanSceneImageUploader::Impl {
        VulkanContext* context = nullptr;
        VkDevice device = VK_NULL_HANDLE;
        VmaAllocator allocator = VK_NULL_HANDLE;
        VkSampler scene_sampler = VK_NULL_HANDLE;

        VkImage scene_image = VK_NULL_HANDLE;
        VmaAllocation scene_image_allocation = VK_NULL_HANDLE;
        VkImageView scene_image_view = VK_NULL_HANDLE;
        VulkanImageBarrierTracker scene_image_barriers;
        glm::ivec2 scene_image_size{0, 0};
        std::string scene_image_vram_label;
        const lfs::core::Tensor* uploaded_scene_tensor = nullptr;
        bool scene_image_external = false;
        std::uint64_t scene_image_external_generation = 0;
        std::uint64_t owned_image_generation = 0;

        [[nodiscard]] bool init(VulkanContext& vulkan_context, const VkSampler sampler) {
            if (device != VK_NULL_HANDLE) {
                return true;
            }
            context = &vulkan_context;
            device = vulkan_context.device();
            allocator = vulkan_context.allocator();
            scene_sampler = sampler;
            if (device == VK_NULL_HANDLE || allocator == VK_NULL_HANDLE || scene_sampler == VK_NULL_HANDLE) {
                return logVkFailure(std::format(
                    "Vulkan scene image uploader requires a live device, allocator, and sampler (device={:#x}, allocator={:#x}, scene_sampler={:#x}) ({}:{})",
                    vkHandleValue(device),
                    reinterpret_cast<std::uintptr_t>(allocator),
                    vkHandleValue(scene_sampler),
                    __FILE__,
                    __LINE__));
            }
            return true;
        }

        void shutdown() {
            destroySceneImage();
            scene_sampler = VK_NULL_HANDLE;
            allocator = VK_NULL_HANDLE;
            device = VK_NULL_HANDLE;
            context = nullptr;
        }

        void clearSceneImageBinding() {
            const std::uint64_t forget_gen =
                scene_image_external ? scene_image_external_generation : owned_image_generation;
            scene_image_barriers.forgetImage(scene_image, forget_gen);
            scene_image = VK_NULL_HANDLE;
            scene_image_allocation = VK_NULL_HANDLE;
            scene_image_view = VK_NULL_HANDLE;
            scene_image_size = {0, 0};
            scene_image_vram_label.clear();
            uploaded_scene_tensor = nullptr;
            scene_image_external = false;
            scene_image_external_generation = 0;
        }

        void updateSceneDescriptor(const VkDescriptorSet scene_descriptor_set,
                                   const VkImageView image_view,
                                   const VkImageLayout image_layout) const {
            if (scene_descriptor_set == VK_NULL_HANDLE) {
                return;
            }
            VkDescriptorImageInfo descriptor_info{};
            descriptor_info.sampler = scene_sampler;
            descriptor_info.imageView = image_view;
            descriptor_info.imageLayout = image_layout;
            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = scene_descriptor_set;
            write.dstBinding = 0;
            write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            write.descriptorCount = 1;
            write.pImageInfo = &descriptor_info;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }

        void destroySceneImage() {
            if (scene_image_external) {
                clearSceneImageBinding();
                return;
            }
            if (scene_image_view != VK_NULL_HANDLE) {
                vkDestroyImageView(device, scene_image_view, nullptr);
            }
            if (scene_image != VK_NULL_HANDLE) {
                if (!scene_image_vram_label.empty()) {
                    lfs::diagnostics::VramProfiler::instance().recordCurrentBytes(
                        "vulkan.scene_image.image",
                        scene_image_vram_label,
                        0);
                }
                vmaDestroyImage(allocator, scene_image, scene_image_allocation);
            }
            clearSceneImageBinding();
        }

        [[nodiscard]] bool bindExternalSceneImage(const VulkanViewportPassParams& params,
                                                  const VkDescriptorSet scene_descriptor_set) {
            if (params.external_scene_image == VK_NULL_HANDLE ||
                params.external_scene_image_view == VK_NULL_HANDLE ||
                params.scene_image_size.x <= 0 ||
                params.scene_image_size.y <= 0) {
                return false;
            }
            if (scene_image_external &&
                scene_image == params.external_scene_image &&
                scene_image_view == params.external_scene_image_view &&
                scene_image_size == params.scene_image_size &&
                scene_image_barriers.imageLayout(scene_image, params.external_scene_image_generation, VK_IMAGE_LAYOUT_UNDEFINED) == params.external_scene_image_layout &&
                scene_image_external_generation == params.external_scene_image_generation) {
                updateSceneDescriptor(scene_descriptor_set, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                return true;
            }

            destroySceneImage();
            scene_image = params.external_scene_image;
            scene_image_view = params.external_scene_image_view;
            scene_image_size = params.scene_image_size;
            uploaded_scene_tensor = params.scene_image.get();
            scene_image_external = true;
            scene_image_external_generation = params.external_scene_image_generation;
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE,
                                         scene_image,
                                         "viewport.scene.external[{}].image",
                                         scene_image_external_generation);
            context->setDebugObjectNamef(VK_OBJECT_TYPE_IMAGE_VIEW,
                                         scene_image_view,
                                         "viewport.scene.external[{}].view",
                                         scene_image_external_generation);
            scene_image_barriers.registerImage(scene_image, scene_image_external_generation, VK_IMAGE_ASPECT_COLOR_BIT, params.external_scene_image_layout);
            updateSceneDescriptor(scene_descriptor_set, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            return true;
        }

        void upload(const VulkanViewportPassParams& params, const VkDescriptorSet scene_descriptor_set) {
            if (params.preserve_scene_image_binding) {
                return;
            }
            const bool has_external_image =
                params.external_scene_image != VK_NULL_HANDLE &&
                params.external_scene_image_view != VK_NULL_HANDLE;
            if ((!params.scene_image && !has_external_image) ||
                params.scene_image_size.x <= 0 || params.scene_image_size.y <= 0) {
                // Release the binding — scene_image_view points into externally-owned
                // interop slots that the caller has already vkDestroyImage'd. Leaving
                // it set keeps hasImage()==true and the viewport pass would sample from
                // a freed image, faulting the device.
                destroySceneImage();
                uploaded_scene_tensor = nullptr;
                return;
            }
            if (has_external_image) {
                if (!bindExternalSceneImage(params, scene_descriptor_set)) {
                    LOG_ERROR("Failed to bind external Vulkan viewport scene image");
                }
                return;
            }
            if (scene_image_external) {
                destroySceneImage();
            }
            if (uploaded_scene_tensor == params.scene_image.get() && scene_image_size == params.scene_image_size &&
                scene_image_view != VK_NULL_HANDLE) {
                updateSceneDescriptor(scene_descriptor_set, scene_image_view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
                return;
            }
            // No CPU staging fallback: external CUDA/Vulkan interop is mandatory.
            // If we reach here with no external image, the upstream path failed
            // to provide one, which is a hard error.
            static bool logged_missing_external = false;
            if (!logged_missing_external) {
                LOG_ERROR("Vulkan viewport scene image upload requires an external CUDA/Vulkan image; none supplied");
                logged_missing_external = true;
            }
            uploaded_scene_tensor = nullptr;
        }

        [[nodiscard]] bool hasImage() const {
            return scene_image_view != VK_NULL_HANDLE;
        }

        [[nodiscard]] bool bindPresentationView(const VkDescriptorSet scene_descriptor_set,
                                                const VkImageView image_view,
                                                const VkImageLayout image_layout) const {
            if (device == VK_NULL_HANDLE || scene_descriptor_set == VK_NULL_HANDLE ||
                image_view == VK_NULL_HANDLE || image_layout == VK_IMAGE_LAYOUT_UNDEFINED) {
                return false;
            }
            updateSceneDescriptor(scene_descriptor_set, image_view, image_layout);
            return true;
        }
    };

    VulkanSceneImageUploader::VulkanSceneImageUploader()
        : impl_(std::make_unique<Impl>()) {}

    VulkanSceneImageUploader::~VulkanSceneImageUploader() {
        if (impl_) {
            impl_->shutdown();
        }
    }

    VulkanSceneImageUploader::VulkanSceneImageUploader(VulkanSceneImageUploader&&) noexcept = default;

    VulkanSceneImageUploader& VulkanSceneImageUploader::operator=(VulkanSceneImageUploader&&) noexcept = default;

    bool VulkanSceneImageUploader::init(VulkanContext& context, const VkSampler scene_sampler) {
        return impl_->init(context, scene_sampler);
    }

    void VulkanSceneImageUploader::shutdown() {
        impl_->shutdown();
    }

    void VulkanSceneImageUploader::upload(const VulkanViewportPassParams& params,
                                          const VkDescriptorSet scene_descriptor_set) {
        impl_->upload(params, scene_descriptor_set);
    }

    bool VulkanSceneImageUploader::bindPresentationView(
        const VkDescriptorSet scene_descriptor_set,
        const VkImageView image_view,
        const VkImageLayout image_layout) {
        return impl_ && impl_->bindPresentationView(scene_descriptor_set, image_view, image_layout);
    }

    bool VulkanSceneImageUploader::hasImage() const {
        return impl_->hasImage();
    }
} // namespace lfs::vis

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "openxr_service.hpp"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <vector>

#ifdef LFS_ENABLE_OPENXR
#define XR_USE_GRAPHICS_API_VULKAN
#include "core/logger.hpp"
#include "preferences.hpp"
#include "rendering/vulkan_wait.hpp"
#include "vr_math.hpp"
#include "window/vulkan_context.hpp"
#include <glm/gtc/quaternion.hpp>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#endif

namespace lfs::vis::vr {
    struct OpenXrService::Impl {
        bool requested = false;
        bool recenter_pending = true;
        std::string status = "VR disabled. Enable VR support and restart with the headset connected.";
        EyeImage mirror;
        glm::vec3 keyboard_movement{0};
        float keyboard_turn = 0;
        bool keyboard_recenter = false;
#ifdef LFS_ENABLE_OPENXR
        XrInstance instance = XR_NULL_HANDLE;
        XrSystemId system = XR_NULL_SYSTEM_ID;
        XrSession session = XR_NULL_HANDLE;
        XrSpace space = XR_NULL_HANDLE;
        XrSpace head_space = XR_NULL_HANDLE;
        XrActionSet action_set = XR_NULL_HANDLE;
        XrAction move = XR_NULL_HANDLE, turn = XR_NULL_HANDLE;
        XrAction up = XR_NULL_HANDLE, down = XR_NULL_HANDLE, center = XR_NULL_HANDLE;
        XrSessionState state = XR_SESSION_STATE_UNKNOWN;
        bool running = false;
        bool frame_begun = false;
        bool submitted = false;
        VulkanContext* context = nullptr;
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer command = VK_NULL_HANDLE;
        VkFence fence = VK_NULL_HANDLE;
        VulkanImageBarrierTracker barriers;
        PFN_xrCreateVulkanInstanceKHR create_instance = nullptr;
        PFN_xrCreateVulkanDeviceKHR create_device = nullptr;
        PFN_xrGetVulkanGraphicsDevice2KHR get_device = nullptr;
        PFN_xrGetVulkanGraphicsRequirements2KHR get_requirements = nullptr;
        struct Eye {
            XrSwapchain swapchain = XR_NULL_HANDLE;
            glm::ivec2 size{0};
            std::vector<XrSwapchainImageVulkan2KHR> images;
            uint32_t index = 0;
            bool acquired = false;
            bool waited = false;
        };
        std::array<Eye, 2> eyes;
        NavigationRig rig;
        glm::quat initial_orientation{1, 0, 0, 0};
        glm::vec3 initial_position{0};
        SnapTurnHysteresis snap;
        XrTime last_time = 0;
        bool center_down = false;
        std::optional<XrEventDataReferenceSpaceChangePending> space_change;
        template <size_t N>
        static void copyName(char (&destination)[N], const std::string_view name) {
            const auto count = std::min(name.size(), N - 1);
            std::copy_n(name.data(), count, destination);
            destination[count] = '\0';
        }

        void check(const XrResult result, const char* operation) {
            if (XR_FAILED(result)) {
                char name[XR_MAX_RESULT_STRING_SIZE]{};
                if (instance)
                    xrResultToString(instance, result, name);
                throw std::runtime_error(std::string(operation) + ": " +
                                         (name[0] ? name : std::to_string(result)));
            }
        }
        void checkVk(const VkResult result, const char* operation) {
            if (result != VK_SUCCESS) {
                auto error = lfs::rendering::vulkan_result_to_error(result, operation, LFS_SOURCE_SITE_CURRENT());
                throw lfs::Exception(std::move(error));
            }
        }
        XrResult queueCall(const std::function<XrResult()>& operation) {
            return static_cast<XrResult>(context->withGraphicsQueue([&] { return int32_t(operation()); }));
        }
        template <typename Fn>
        void load(Fn& function, const char* name) {
            check(xrGetInstanceProcAddr(instance, name, reinterpret_cast<PFN_xrVoidFunction*>(&function)), name);
        }
        XrPath path(const char* value) {
            XrPath result{};
            check(xrStringToPath(instance, value, &result), "xrStringToPath");
            return result;
        }
        XrAction action(const char* name, const char* label, const XrActionType type) {
            XrActionCreateInfo info{XR_TYPE_ACTION_CREATE_INFO};
            copyName(info.actionName, name);
            copyName(info.localizedActionName, label);
            info.actionType = type;
            XrAction result{};
            check(xrCreateAction(action_set, &info, &result), "xrCreateAction");
            return result;
        }
        void createActions() {
            XrActionSetCreateInfo info{XR_TYPE_ACTION_SET_CREATE_INFO};
            copyName(info.actionSetName, "navigation");
            copyName(info.localizedActionSetName, "View-only navigation");
            check(xrCreateActionSet(instance, &info, &action_set), "xrCreateActionSet");
            move = action("move", "Move", XR_ACTION_TYPE_VECTOR2F_INPUT);
            turn = action("turn", "Snap turn and vertical movement", XR_ACTION_TYPE_VECTOR2F_INPUT);
            up = action("up", "Move up", XR_ACTION_TYPE_BOOLEAN_INPUT);
            down = action("down", "Move down", XR_ACTION_TYPE_BOOLEAN_INPUT);
            center = action("recenter", "Recenter", XR_ACTION_TYPE_BOOLEAN_INPUT);

            const auto suggest = [&](const char* profile, const char* axis, const char* center_component,
                                     const char* up_component, const char* down_component) {
                const std::string left = "/user/hand/left/input/";
                const std::string right = "/user/hand/right/input/";
                const std::array<XrActionSuggestedBinding, 5> bindings{{
                    {move, path((left + axis).c_str())},
                    {turn, path((right + axis).c_str())},
                    {center, path((left + center_component).c_str())},
                    {up, path((right + up_component).c_str())},
                    {down, path((left + down_component).c_str())},
                }};
                XrInteractionProfileSuggestedBinding suggestion{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
                suggestion.interactionProfile = path(profile);
                suggestion.countSuggestedBindings = static_cast<uint32_t>(bindings.size());
                suggestion.suggestedBindings = bindings.data();
                const auto result = xrSuggestInteractionProfileBindings(instance, &suggestion);
                // A runtime need not recognize every profile. Other bindings stay usable.
                if (result != XR_ERROR_PATH_UNSUPPORTED)
                    check(result, "xrSuggestInteractionProfileBindings");
            };
            suggest("/interaction_profiles/oculus/touch_controller", "thumbstick", "thumbstick/click", "a/click", "x/click");
            suggest("/interaction_profiles/valve/index_controller", "thumbstick", "thumbstick/click", "a/click", "a/click");
            suggest("/interaction_profiles/htc/vive_controller", "trackpad", "menu/click", "trigger/click", "trigger/click");
            XrSessionActionSetsAttachInfo attach{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
            attach.countActionSets = 1;
            attach.actionSets = &action_set;
            check(xrAttachSessionActionSets(session, &attach), "xrAttachSessionActionSets");
        }
        glm::vec2 vectorAction(const XrAction value) {
            XrActionStateGetInfo info{XR_TYPE_ACTION_STATE_GET_INFO};
            info.action = value;
            XrActionStateVector2f result{XR_TYPE_ACTION_STATE_VECTOR2F};
            check(xrGetActionStateVector2f(session, &info, &result), "xrGetActionStateVector2f");
            return result.isActive ? glm::vec2(result.currentState.x, result.currentState.y) : glm::vec2(0);
        }
        bool booleanAction(const XrAction value) {
            XrActionStateGetInfo info{XR_TYPE_ACTION_STATE_GET_INFO};
            info.action = value;
            XrActionStateBoolean result{XR_TYPE_ACTION_STATE_BOOLEAN};
            check(xrGetActionStateBoolean(session, &info, &result), "xrGetActionStateBoolean");
            return result.isActive && result.currentState;
        }
        static glm::quat orientation(const XrPosef& pose) {
            return {pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z};
        }
        static glm::vec3 position(const XrPosef& pose) {
            return {pose.position.x, pose.position.y, pose.position.z};
        }
        void navigation(const XrPosef& head, const XrTime time) {
            if (space_change && time >= space_change->changeTime) {
                if (space_change->poseValid) {
                    const auto& pose = space_change->poseInPreviousSpace;
                    rig.position += rig.orientation * (position(pose) * rig.scene_units_per_meter);
                    rig.orientation = glm::normalize(rig.orientation * orientation(pose));
                } else {
                    recenter_pending = true;
                }
                space_change.reset();
            }
            const auto head_rotation = orientation(head);
            const auto head_position = position(head);
            const float dt = last_time ? std::clamp(float(double(time - last_time) * 1e-9), 0.0f, 0.05f) : 0.0f;
            last_time = time;
            if (state == XR_SESSION_STATE_FOCUSED) {
                XrActiveActionSet active{action_set, XR_NULL_PATH};
                XrActionsSyncInfo sync{XR_TYPE_ACTIONS_SYNC_INFO};
                sync.countActiveActionSets = 1;
                sync.activeActionSets = &active;
                check(xrSyncActions(session, &sync), "xrSyncActions");
                const bool pressed = booleanAction(center) || keyboard_recenter;
                recenter_pending |= pressed && !center_down;
                center_down = pressed;
            } else {
                snap.reset();
                center_down = false;
            }
            if (recenter_pending) {
                // Align horizontal tracking heading to the desktop entry heading;
                // tracked pitch/roll remain physical head motion.
                rig.orientation = initial_orientation * glm::inverse(yawOnlyOrientation(head_rotation));
                rig.position = initial_position - rig.orientation * (head_position * rig.scene_units_per_meter);
                recenter_pending = false;
                snap.reset();
            }
            if (state != XR_SESSION_STATE_FOCUSED)
                return;
            const auto movement = glm::clamp(deadzone(vectorAction(move), 0.2f) +
                                                 glm::vec2(keyboard_movement.x, -keyboard_movement.z),
                                             glm::vec2(-1), glm::vec2(1));
            auto rotation = vectorAction(turn);
            rotation.x = std::clamp(rotation.x + keyboard_turn, -1.0f, 1.0f);
            const int direction = snap.update(rotation.x);
            if (direction)
                rig.snapTurn(-float(direction) * glm::radians(30.0f), head_position);
            const auto yaw = yawOnlyOrientation(rig.orientation * head_rotation);
            const float vertical = float(booleanAction(up)) - float(booleanAction(down)) +
                                   deadzone(rotation.y, 0.3f) + keyboard_movement.y;
            rig.position += (yaw * glm::vec3(movement.x, 0, -movement.y) + glm::vec3(0, vertical, 0)) *
                            UserPreferences::instance().vrMoveSpeed() * rig.scene_units_per_meter * dt;
        }
        void poll() {
            XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
            XrResult result;
            while ((result = xrPollEvent(instance, &event)) == XR_SUCCESS) {
                if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
                    const auto& changed = reinterpret_cast<const XrEventDataSessionStateChanged&>(event);
                    if (changed.session == session) {
                        state = changed.state;
                        if (state == XR_SESSION_STATE_READY && !running) {
                            XrSessionBeginInfo begin{XR_TYPE_SESSION_BEGIN_INFO};
                            begin.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
                            check(xrBeginSession(session, &begin), "xrBeginSession");
                            running = true;
                        } else if (state == XR_SESSION_STATE_STOPPING) {
                            if (running)
                                check(xrEndSession(session), "xrEndSession");
                            running = false;
                            requested = false;
                        } else if (state == XR_SESSION_STATE_EXITING || state == XR_SESSION_STATE_LOSS_PENDING) {
                            requested = false;
                        }
                    }
                } else if (event.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
                    requested = false;
                    status = "OpenXR runtime was lost. Restart LichtFeld to reconnect.";
                } else if (event.type == XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING) {
                    const auto& change = reinterpret_cast<const XrEventDataReferenceSpaceChangePending&>(event);
                    if (change.session == session && change.referenceSpaceType == XR_REFERENCE_SPACE_TYPE_LOCAL)
                        space_change = change;
                } else if (event.type == XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED) {
                    status = "VR controller profile changed; navigation bindings refreshed by OpenXR.";
                }
                event = {XR_TYPE_EVENT_DATA_BUFFER};
            }
            if (result != XR_EVENT_UNAVAILABLE)
                check(result, "xrPollEvent");
        }
        void waitTransfer() {
            if (!submitted)
                return;
            const auto result = lfs::rendering::wait_fence_bounded(context->device(), fence, {});
            if (!result)
                throw lfs::Exception(result.error());
            if (*result != lfs::rendering::WaitOutcome::Ready)
                throw lfs::Exception(lfs::make_error(lfs::ErrorInit{
                    .code = lfs::ErrorCode::DeadlineExceeded,
                    .domain = lfs::ErrorDomain::Vulkan,
                    .user_message = "VR image transfer did not complete; rendering stopped.",
                    .detection = LFS_SOURCE_SITE_CURRENT(),
                }));
            submitted = false;
        }
        void transfer(const std::array<EyeImage, 2>& sources) {
            waitTransfer();
            checkVk(vkResetCommandPool(context->device(), pool, 0), "vkResetCommandPool(VR)");
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            checkVk(vkBeginCommandBuffer(command, &begin), "vkBeginCommandBuffer(VR)");
            barriers.clearSwapchainOnly();
            std::array<VkSemaphore, 2> waits{};
            std::array<uint64_t, 2> values{};
            std::array<VkPipelineStageFlags, 2> stages{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
            uint32_t wait_count = 0;
            using Scope = VulkanImageBarrierTracker::AccessScope;
            constexpr Scope memory{VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
            constexpr Scope read{VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT};
            constexpr Scope write{VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT};
            for (size_t i = 0; i < eyes.size(); ++i) {
                auto& eye = eyes[i];
                const auto& source = sources[i];
                if (!source.image || source.size != eye.size || source.flip_y)
                    throw std::runtime_error("VR renderer returned an invalid eye image extent or orientation.");
                if (source.ready && source.ready_value) {
                    if (wait_count && waits[0] == source.ready)
                        values[0] = std::max(values[0], source.ready_value);
                    else {
                        waits[wait_count] = source.ready;
                        values[wait_count++] = source.ready_value;
                    }
                }
                const auto destination = eye.images.at(eye.index).image;
                // Runtime-owned images may discard previous contents on acquire.
                // A private tracker keeps the source's restored layout unchanged
                // in the renderer's own tracker.
                barriers.registerImage(source.image, 0, VK_IMAGE_ASPECT_COLOR_BIT, source.layout);
                barriers.registerImage(destination, 0, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_UNDEFINED);
                barriers.transitionImage(command, source.image, 0, VK_IMAGE_ASPECT_COLOR_BIT,
                                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, memory, read);
                barriers.transitionImage(command, destination, 0, VK_IMAGE_ASPECT_COLOR_BIT,
                                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
                VkImageCopy copy{};
                copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                copy.dstSubresource = copy.srcSubresource;
                copy.extent = {uint32_t(eye.size.x), uint32_t(eye.size.y), 1};
                // Splat output already contains display-encoded color. Compatible
                // RGBA8 UNORM -> SRGB copies preserve bits; blitting would apply
                // another gamma conversion.
                vkCmdCopyImage(command, source.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               destination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
                barriers.transitionImage(command, source.image, 0, VK_IMAGE_ASPECT_COLOR_BIT,
                                         source.layout, read, memory);
                barriers.transitionImage(command, destination, 0, VK_IMAGE_ASPECT_COLOR_BIT,
                                         VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, write, memory);
            }
            checkVk(vkEndCommandBuffer(command), "vkEndCommandBuffer(VR)");
            VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            timeline.waitSemaphoreValueCount = wait_count;
            timeline.pWaitSemaphoreValues = values.data();
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.pNext = &timeline;
            submit.waitSemaphoreCount = wait_count;
            submit.pWaitSemaphores = waits.data();
            submit.pWaitDstStageMask = stages.data();
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &command;
            checkVk(vkResetFences(context->device(), 1, &fence), "vkResetFences(VR)");
            checkVk(lfs::rendering::vk_queue_submit_synced(context->graphicsQueue(), 1, &submit, fence), "vkQueueSubmit(VR)");
            submitted = true;
            // One bounded transfer fence (not device-idle) also prevents the eye
            // output ring from being recycled while OpenXR is still reading it.
            waitTransfer();
        }
#endif
    };

    OpenXrService::OpenXrService() : impl_(std::make_unique<Impl>()) {}
    OpenXrService::~OpenXrService() {
        stop();
#ifdef LFS_ENABLE_OPENXR
        if (impl_->instance)
            xrDestroyInstance(impl_->instance);
#endif
    }
    bool OpenXrService::compiled() {
#ifdef LFS_ENABLE_OPENXR
        return true;
#else
        return false;
#endif
    }
    bool OpenXrService::prepare() {
#ifdef LFS_ENABLE_OPENXR
        auto& p = *impl_;
        try {
            uint32_t count = 0;
            p.check(xrEnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr), "OpenXR runtime discovery");
            std::vector<XrExtensionProperties> extensions(count, {XR_TYPE_EXTENSION_PROPERTIES});
            p.check(xrEnumerateInstanceExtensionProperties(nullptr, count, &count, extensions.data()), "OpenXR extension discovery");
            if (!std::ranges::any_of(extensions, [](const auto& extension) {
                    return std::strcmp(extension.extensionName, XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME) == 0;
                }))
                throw std::runtime_error("The active OpenXR runtime does not support Vulkan enable2.");
            const char* enabled[] = {XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME};
            XrInstanceCreateInfo create{XR_TYPE_INSTANCE_CREATE_INFO};
            Impl::copyName(create.applicationInfo.applicationName, "LichtFeld Studio");
            create.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
            create.enabledExtensionCount = 1;
            create.enabledExtensionNames = enabled;
            p.check(xrCreateInstance(&create, &p.instance), "xrCreateInstance");
            XrSystemGetInfo system{XR_TYPE_SYSTEM_GET_INFO};
            system.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
            p.check(xrGetSystem(p.instance, &system, &p.system), "Headset discovery (connect headset before startup)");
            p.load(p.create_instance, "xrCreateVulkanInstanceKHR");
            p.load(p.create_device, "xrCreateVulkanDeviceKHR");
            p.load(p.get_device, "xrGetVulkanGraphicsDevice2KHR");
            p.load(p.get_requirements, "xrGetVulkanGraphicsRequirements2KHR");
            XrGraphicsRequirementsVulkan2KHR requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
            p.check(p.get_requirements(p.instance, p.system, &requirements), "xrGetVulkanGraphicsRequirements2KHR");
            const auto version = XR_MAKE_VERSION(1, 3, 0);
            if (version < requirements.minApiVersionSupported)
                throw std::runtime_error("The OpenXR runtime requires Vulkan newer than LichtFeld's Vulkan 1.3.");
            if (version > requirements.maxApiVersionSupported)
                LOG_WARN("OpenXR runtime's tested Vulkan version is below 1.3; attempting compatible Vulkan 1.3.");
            XrInstanceProperties properties{XR_TYPE_INSTANCE_PROPERTIES};
            p.check(xrGetInstanceProperties(p.instance, &properties), "xrGetInstanceProperties");
            p.status = std::string("OpenXR ready: ") + properties.runtimeName;
            LOG_INFO("{}", p.status);
            return true;
        } catch (const std::exception& error) {
            p.status = error.what();
            if (p.instance)
                xrDestroyInstance(p.instance);
            p.instance = XR_NULL_HANDLE;
            p.system = XR_NULL_SYSTEM_ID;
            return false;
        }
#else
        impl_->status = "This build has no OpenXR support. Configure with LFS_ENABLE_OPENXR=ON.";
        return false;
#endif
    }
    bool OpenXrService::prepared() const {
#ifdef LFS_ENABLE_OPENXR
        return impl_->instance && impl_->system;
#else
        return false;
#endif
    }
    VkResult OpenXrService::createVulkanInstance([[maybe_unused]] const VkInstanceCreateInfo& info, [[maybe_unused]] VkInstance& instance) {
#ifdef LFS_ENABLE_OPENXR
        XrVulkanInstanceCreateInfoKHR create{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
        create.systemId = impl_->system;
        create.pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
        create.vulkanCreateInfo = &info;
        VkResult result = VK_ERROR_INITIALIZATION_FAILED;
        const auto xr_result = impl_->create_instance(impl_->instance, &create, &instance, &result);
        return XR_SUCCEEDED(xr_result) ? result : VK_ERROR_INITIALIZATION_FAILED;
#else
        return VK_ERROR_FEATURE_NOT_PRESENT;
#endif
    }
    VkPhysicalDevice OpenXrService::graphicsDevice([[maybe_unused]] VkInstance instance) {
#ifdef LFS_ENABLE_OPENXR
        XrVulkanGraphicsDeviceGetInfoKHR get{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
        get.systemId = impl_->system;
        get.vulkanInstance = instance;
        VkPhysicalDevice physical{};
        if (XR_FAILED(impl_->get_device(impl_->instance, &get, &physical))) {
            impl_->status = "Failed to query the OpenXR Vulkan GPU.";
            return VK_NULL_HANDLE;
        }
        return physical;
#else
        return VK_NULL_HANDLE;
#endif
    }
    VkResult OpenXrService::createVulkanDevice([[maybe_unused]] VkPhysicalDevice physical, [[maybe_unused]] const VkDeviceCreateInfo& info, [[maybe_unused]] VkDevice& device) {
#ifdef LFS_ENABLE_OPENXR
        XrVulkanDeviceCreateInfoKHR create{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
        create.systemId = impl_->system;
        create.pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
        create.vulkanPhysicalDevice = physical;
        create.vulkanCreateInfo = &info;
        VkResult result = VK_ERROR_INITIALIZATION_FAILED;
        const auto xr_result = impl_->create_device(impl_->instance, &create, &device, &result);
        return XR_SUCCEEDED(xr_result) ? result : VK_ERROR_INITIALIZATION_FAILED;
#else
        return VK_ERROR_FEATURE_NOT_PRESENT;
#endif
    }
    std::expected<void, std::string> OpenXrService::start([[maybe_unused]] VulkanContext& context, [[maybe_unused]] const glm::mat3& rotation, [[maybe_unused]] const glm::vec3& position) {
#ifdef LFS_ENABLE_OPENXR
        auto& p = *impl_;
        if (!prepared())
            return std::unexpected(p.status);
        if (active())
            return {};
        try {
            p.context = &context;
            XrGraphicsBindingVulkan2KHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
            binding.instance = context.instance();
            binding.physicalDevice = context.physicalDevice();
            binding.device = context.device();
            binding.queueFamilyIndex = context.graphicsQueueFamily();
            binding.queueIndex = 0;
            XrSessionCreateInfo create{XR_TYPE_SESSION_CREATE_INFO};
            create.next = &binding;
            create.systemId = p.system;
            p.check(xrCreateSession(p.instance, &create, &p.session), "xrCreateSession");
            XrReferenceSpaceCreateInfo space{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
            space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
            space.poseInReferenceSpace.orientation.w = 1;
            p.check(xrCreateReferenceSpace(p.session, &space, &p.space), "xrCreateReferenceSpace(local)");
            space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
            p.check(xrCreateReferenceSpace(p.session, &space, &p.head_space), "xrCreateReferenceSpace(view)");
            p.createActions();
            uint32_t count = 0;
            p.check(xrEnumerateEnvironmentBlendModes(p.instance, p.system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                                     0, &count, nullptr),
                    "xrEnumerateEnvironmentBlendModes");
            std::vector<XrEnvironmentBlendMode> blends(count);
            p.check(xrEnumerateEnvironmentBlendModes(p.instance, p.system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                                     count, &count, blends.data()),
                    "xrEnumerateEnvironmentBlendModes");
            if (std::ranges::find(blends, XR_ENVIRONMENT_BLEND_MODE_OPAQUE) == blends.end())
                throw std::runtime_error("View-only VR requires an opaque headset display mode.");
            p.check(xrEnumerateSwapchainFormats(p.session, 0, &count, nullptr), "xrEnumerateSwapchainFormats");
            std::vector<int64_t> formats(count);
            p.check(xrEnumerateSwapchainFormats(p.session, count, &count, formats.data()), "xrEnumerateSwapchainFormats");
            if (std::ranges::find(formats, int64_t(VK_FORMAT_R8G8B8A8_SRGB)) == formats.end())
                throw std::runtime_error("OpenXR runtime must expose RGBA8 sRGB swapchains for display-correct GPU image copies.");
            std::array<XrViewConfigurationView, 2> views{{{XR_TYPE_VIEW_CONFIGURATION_VIEW}, {XR_TYPE_VIEW_CONFIGURATION_VIEW}}};
            p.check(xrEnumerateViewConfigurationViews(p.instance, p.system, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO,
                                                      2, &count, views.data()),
                    "xrEnumerateViewConfigurationViews");
            if (count != 2)
                throw std::runtime_error("The headset does not expose a two-eye stereo configuration.");
            const float scale = UserPreferences::instance().vrResolutionScale();
            for (size_t i = 0; i < 2; ++i) {
                auto& eye = p.eyes[i];
                eye.size = {std::clamp(int(views[i].recommendedImageRectWidth * scale), 1, int(views[i].maxImageRectWidth)),
                            std::clamp(int(views[i].recommendedImageRectHeight * scale), 1, int(views[i].maxImageRectHeight))};
                XrSwapchainCreateInfo swapchain{XR_TYPE_SWAPCHAIN_CREATE_INFO};
                swapchain.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
                swapchain.format = VK_FORMAT_R8G8B8A8_SRGB;
                swapchain.sampleCount = 1;
                swapchain.width = uint32_t(eye.size.x);
                swapchain.height = uint32_t(eye.size.y);
                swapchain.faceCount = swapchain.arraySize = swapchain.mipCount = 1;
                p.check(xrCreateSwapchain(p.session, &swapchain, &eye.swapchain), "xrCreateSwapchain");
                p.check(xrEnumerateSwapchainImages(eye.swapchain, 0, &count, nullptr), "xrEnumerateSwapchainImages");
                eye.images.assign(count, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR});
                p.check(xrEnumerateSwapchainImages(eye.swapchain, count, &count,
                                                   reinterpret_cast<XrSwapchainImageBaseHeader*>(eye.images.data())),
                        "xrEnumerateSwapchainImages");
            }
            VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            pool.queueFamilyIndex = context.graphicsQueueFamily();
            p.checkVk(vkCreateCommandPool(context.device(), &pool, nullptr, &p.pool), "vkCreateCommandPool(VR)");
            VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocate.commandPool = p.pool;
            allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate.commandBufferCount = 1;
            p.checkVk(vkAllocateCommandBuffers(context.device(), &allocate, &p.command), "vkAllocateCommandBuffers(VR)");
            VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            p.checkVk(vkCreateFence(context.device(), &fence, nullptr, &p.fence), "vkCreateFence(VR)");
            p.initial_orientation = yawOnlyOrientation(glm::quat_cast(rotation));
            p.initial_position = position;
            p.rig = {};
            p.rig.scene_units_per_meter = UserPreferences::instance().vrSceneScale();
            p.recenter_pending = true;
            p.last_time = 0;
            p.requested = true;
            p.status = "VR session created; waiting for the headset.";
            return {};
        } catch (const std::exception& error) {
            context.noteFailure(error);
            const std::string message = error.what();
            stop();
            p.status = message;
            return std::unexpected(message);
        }
#else
        return std::unexpected(impl_->status);
#endif
    }
    void OpenXrService::stop([[maybe_unused]] const bool graphics_retired) {
        impl_->requested = false;
        impl_->mirror = {};
#ifdef LFS_ENABLE_OPENXR
        auto& p = *impl_;
        // Teardown is allowed to idle; normal frames wait only their transfer fence.
        if (p.context && p.submitted && !graphics_retired) {
            if (p.context->rendererTerminalState() != RendererTerminalState::Running)
                return;
            try {
                p.waitTransfer();
            } catch (const std::exception& error) {
                p.context->noteFailure(error);
                LOG_ERROR("VR teardown: {}", error.what());
                // Keep commands AND runtime-owned images alive until shutdown's
                // device drain. The desktop renderer is quarantined by noteFailure.
                return;
            }
        }
        if (p.context && p.fence)
            vkDestroyFence(p.context->device(), p.fence, nullptr);
        if (p.context && p.pool)
            vkDestroyCommandPool(p.context->device(), p.pool, nullptr);
        p.fence = VK_NULL_HANDLE;
        p.pool = VK_NULL_HANDLE;
        p.command = VK_NULL_HANDLE;
        for (auto& eye : p.eyes) {
            if (eye.swapchain)
                xrDestroySwapchain(eye.swapchain);
            eye = {};
        }
        if (p.head_space)
            xrDestroySpace(p.head_space);
        if (p.space)
            xrDestroySpace(p.space);
        if (p.session)
            xrDestroySession(p.session);
        if (p.action_set)
            xrDestroyActionSet(p.action_set);
        p.session = XR_NULL_HANDLE;
        p.space = p.head_space = XR_NULL_HANDLE;
        p.action_set = XR_NULL_HANDLE;
        p.move = p.turn = p.up = p.down = p.center = XR_NULL_HANDLE;
        p.state = XR_SESSION_STATE_UNKNOWN;
        p.running = p.frame_begun = p.submitted = false;
        p.context = nullptr;
        p.space_change.reset();
        p.barriers.clearSwapchainOnly();
#endif
    }
    void OpenXrService::reportUnavailable(std::string reason) { impl_->status = std::move(reason); }

    bool OpenXrService::active() const {
#ifdef LFS_ENABLE_OPENXR
        return impl_->session != XR_NULL_HANDLE;
#else
        return false;
#endif
    }
    bool OpenXrService::running() const {
#ifdef LFS_ENABLE_OPENXR
        return impl_->running;
#else
        return false;
#endif
    }
    bool OpenXrService::requested() const { return impl_->requested; }
    void OpenXrService::request(const bool enabled) { impl_->requested = enabled; }
    void OpenXrService::recenter() { impl_->recenter_pending = true; }
    void OpenXrService::setKeyboardNavigation(const glm::vec3 movement, const float turn, const bool recenter) {
        impl_->keyboard_movement = movement;
        impl_->keyboard_turn = turn;
        impl_->keyboard_recenter = recenter;
    }
    const std::string& OpenXrService::status() const { return impl_->status; }
    EyeImage OpenXrService::mirror() const { return impl_->mirror; }

    std::expected<void, std::string> OpenXrService::frame([[maybe_unused]] const RenderEyes& render) {
#ifdef LFS_ENABLE_OPENXR
        auto& p = *impl_;
        if (!active())
            return {};
        XrTime display_time = 0;
        try {
            p.poll();
            if (!p.running || !p.requested)
                return {};
            XrFrameWaitInfo wait{XR_TYPE_FRAME_WAIT_INFO};
            XrFrameState state{XR_TYPE_FRAME_STATE};
            p.check(xrWaitFrame(p.session, &wait, &state), "xrWaitFrame");
            display_time = state.predictedDisplayTime;
            XrFrameBeginInfo begin{XR_TYPE_FRAME_BEGIN_INFO};
            p.check(p.queueCall([&] { return xrBeginFrame(p.session, &begin); }), "xrBeginFrame");
            p.frame_begun = true;
            std::array<XrView, 2> views{{{XR_TYPE_VIEW}, {XR_TYPE_VIEW}}};
            XrViewLocateInfo locate{XR_TYPE_VIEW_LOCATE_INFO};
            locate.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
            locate.displayTime = display_time;
            locate.space = p.space;
            XrViewState view_state{XR_TYPE_VIEW_STATE};
            uint32_t count = 0;
            p.check(xrLocateViews(p.session, &locate, &view_state, 2, &count, views.data()), "xrLocateViews");
            constexpr auto valid = XR_VIEW_STATE_POSITION_VALID_BIT | XR_VIEW_STATE_ORIENTATION_VALID_BIT;
            XrSpaceLocation head{XR_TYPE_SPACE_LOCATION};
            p.check(xrLocateSpace(p.head_space, p.space, display_time, &head), "xrLocateSpace(head)");
            constexpr auto head_valid = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
            const bool draw = state.shouldRender && count == 2 && (view_state.viewStateFlags & valid) == valid &&
                              (head.locationFlags & head_valid) == head_valid;
            std::array<XrCompositionLayerProjectionView, 2> projection_views{{{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}, {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}}};
            if (draw) {
                p.navigation(head.pose, display_time);
                std::array<lfs::rendering::FrameView, 2> frame_views;
                for (size_t i = 0; i < 2; ++i) {
                    auto& eye = p.eyes[i];
                    const auto pose = p.rig.cameraPose(Impl::orientation(views[i].pose), Impl::position(views[i].pose));
                    frame_views[i].rotation = pose.rotation;
                    frame_views[i].translation = pose.translation;
                    frame_views[i].size = eye.size;
                    const auto& fov = views[i].fov;
                    frame_views[i].intrinsics_override = eyeIntrinsics({fov.angleLeft, fov.angleRight, fov.angleUp, fov.angleDown}, eye.size);
                    XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
                    p.check(xrAcquireSwapchainImage(eye.swapchain, &acquire, &eye.index), "xrAcquireSwapchainImage");
                    eye.acquired = true;
                    XrSwapchainImageWaitInfo wait_image{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
                    wait_image.timeout = 1'000'000'000;
                    const auto waited = xrWaitSwapchainImage(eye.swapchain, &wait_image);
                    if (waited == XR_TIMEOUT_EXPIRED)
                        throw std::runtime_error("Timed out waiting for an OpenXR swapchain image.");
                    p.check(waited, "xrWaitSwapchainImage");
                    eye.waited = true;
                    projection_views[i].pose = views[i].pose;
                    projection_views[i].fov = views[i].fov;
                    projection_views[i].subImage.swapchain = eye.swapchain;
                    projection_views[i].subImage.imageRect.extent = {eye.size.x, eye.size.y};
                }
                auto images = render(frame_views);
                if (!images)
                    throw std::runtime_error(images.error());
                p.transfer(*images);
                p.mirror = (*images)[0];
                for (auto& eye : p.eyes) {
                    XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
                    p.check(p.queueCall([&] { return xrReleaseSwapchainImage(eye.swapchain, &release); }), "xrReleaseSwapchainImage");
                    eye.acquired = eye.waited = false;
                }
                p.status = "VR active: left stick moves, right stick turns/climbs, left stick click recenters.";
            } else {
                p.last_time = 0;
                p.snap.reset();
                p.mirror = {};
            }
            XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
            layer.space = p.space;
            layer.viewCount = 2;
            layer.views = projection_views.data();
            const XrCompositionLayerBaseHeader* layers[] = {reinterpret_cast<const XrCompositionLayerBaseHeader*>(&layer)};
            XrFrameEndInfo end{XR_TYPE_FRAME_END_INFO};
            end.displayTime = display_time;
            end.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
            end.layerCount = draw ? 1 : 0;
            end.layers = draw ? layers : nullptr;
            p.check(p.queueCall([&] { return xrEndFrame(p.session, &end); }), "xrEndFrame");
            p.frame_begun = false;
            return {};
        } catch (const std::exception& error) {
            p.context->noteFailure(error);
            // Release only images whose acquire/wait completed, and only if no
            // GPU work still references them. A timed-out transfer is retained
            // intact until VulkanContext's device-drain boundary.
            if (!p.submitted) {
                for (auto& eye : p.eyes) {
                    if (eye.acquired && eye.waited) {
                        XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
                        (void)p.queueCall([&] { return xrReleaseSwapchainImage(eye.swapchain, &release); });
                        eye.acquired = eye.waited = false;
                    }
                }
            }
            if (p.frame_begun && !p.submitted) {
                XrFrameEndInfo end{XR_TYPE_FRAME_END_INFO};
                end.displayTime = display_time;
                end.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
                (void)p.queueCall([&] { return xrEndFrame(p.session, &end); });
                p.frame_begun = false;
            }
            const std::string message = error.what();
            stop();
            p.status = message;
            return std::unexpected(message);
        }
#else
        return std::unexpected(impl_->status);
#endif
    }
} // namespace lfs::vis::vr

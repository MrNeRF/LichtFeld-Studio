/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#define XR_USE_GRAPHICS_API_VULKAN
#include "openxr_service.hpp"
#include "core/event_bridge/localization_manager.hpp"
#include "core/logger.hpp"
#include "core/vulkan_queue_sync.hpp"
#include "preferences.hpp"
#include "rendering/vulkan_scene_output.hpp"
#include "rendering/vulkan_wait.hpp"
#include "vr_math.hpp"
#include "window/vulkan_context.hpp"
#include "window/vulkan_graphics_context.hpp"
#include <algorithm>
#include <cstring>
#include <format>
#include <optional>
#include <string_view>
#include <vector>

#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

namespace lfs::vis::vr {

    namespace {
        constexpr XrViewConfigurationType kStereo = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
        const float kSnapTurnRadians = glm::radians(30.0f);
        constexpr std::array<const char*, 9> kSessionStates = {
            "unknown", "idle", "ready", "synchronized", "visible", "focused", "stopping", "loss pending", "exiting"};
        [[nodiscard]] lfs::Error vrError(std::string message) {
            return lfs::make_error(lfs::ErrorInit{
                .code = lfs::ErrorCode::Unavailable,
                .domain = lfs::ErrorDomain::Rendering,
                .user_message = std::move(message),
                .detection = LFS_SOURCE_SITE_CURRENT(),
            });
        }

        template <std::size_t N>
        void copyName(char (&destination)[N], const std::string_view name) {
            const auto count = std::min(name.size(), N - 1);
            std::copy_n(name.data(), count, destination);
            destination[count] = '\0';
        }

        [[nodiscard]] glm::quat orientation(const XrPosef& pose) {
            return {pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z};
        }

        [[nodiscard]] glm::vec3 position(const XrPosef& pose) {
            return {pose.position.x, pose.position.y, pose.position.z};
        }

        [[nodiscard]] VkImageMemoryBarrier2 imageBarrier(const VkImage image,
                                                         const VkImageLayout from, const VkImageLayout to,
                                                         const VkPipelineStageFlags2 src_stage, const VkAccessFlags2 src_access,
                                                         const VkPipelineStageFlags2 dst_stage, const VkAccessFlags2 dst_access) {
            VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
            barrier.srcStageMask = src_stage;
            barrier.srcAccessMask = src_access;
            barrier.dstStageMask = dst_stage;
            barrier.dstAccessMask = dst_access;
            barrier.oldLayout = from;
            barrier.newLayout = to;
            barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image;
            barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            return barrier;
        }
    } // namespace

    struct OpenXrService::Impl {
        // Created once by prepare().
        XrInstance instance = XR_NULL_HANDLE;
        XrSystemId system = XR_NULL_SYSTEM_ID;
        struct VulkanFunctions {
            PFN_xrCreateVulkanInstanceKHR create_instance = nullptr;
            PFN_xrCreateVulkanDeviceKHR create_device = nullptr;
            PFN_xrGetVulkanGraphicsDevice2KHR graphics_device = nullptr;
            PFN_xrGetVulkanGraphicsRequirements2KHR graphics_requirements = nullptr;
        } vk;
        VulkanDeviceHooks hooks;
        // The last device the runtime created; eyes must render on it.
        VkDevice runtime_device = VK_NULL_HANDLE;
        bool lost = false;
        std::string runtime_name;
        // A localization key and its argument; the UI localizes them when shown.
        const char* status_key = "vr.status_off";
        std::string status_detail;
        void setStatus(const char* key, std::string detail = {}) {
            status_key = key;
            status_detail = std::move(detail);
        }
        float move_speed = VrPreferences{}.move_speed;

        // Alive between start() and stop().
        struct Eye {
            XrSwapchain swapchain = XR_NULL_HANDLE;
            glm::ivec2 size{0};
            std::vector<XrSwapchainImageVulkan2KHR> images;
            std::uint32_t index = 0;
            bool acquired = false; // acquired and waited, so it must be released
        };
        struct Session {
            VulkanContext* context = nullptr;
            XrSession handle = XR_NULL_HANDLE;
            XrSpace space = XR_NULL_HANDLE;
            XrSpace head_space = XR_NULL_HANDLE;
            XrActionSet action_set = XR_NULL_HANDLE;
            XrAction move = XR_NULL_HANDLE, turn = XR_NULL_HANDLE;
            XrAction up = XR_NULL_HANDLE, down = XR_NULL_HANDLE, center = XR_NULL_HANDLE;
            XrSessionState state = XR_SESSION_STATE_UNKNOWN;
            bool running = false;
            bool ended = false;
            bool frame_begun = false;
            bool transfer_pending = false;
            XrTime display_time = 0;
            std::array<Eye, 2> eyes;
            VkCommandPool pool = VK_NULL_HANDLE;
            VkCommandBuffer command = VK_NULL_HANDLE;
            VkFence fence = VK_NULL_HANDLE;
            EyeFrame mirror;

            NavigationRig rig;
            glm::quat start_orientation{1.0f, 0.0f, 0.0f, 0.0f};
            glm::vec3 start_position{0.0f};
            bool recenter_pending = true;
            bool center_down = false;
            int pending_turns = 0;
            SnapTurnHysteresis snap;
            XrTime last_time = 0;
            std::optional<XrEventDataReferenceSpaceChangePending> space_change;
        } s;

        [[nodiscard]] lfs::Status check(const XrResult result, const std::string_view what) const {
            if (XR_SUCCEEDED(result))
                return {};
            char name[XR_MAX_RESULT_STRING_SIZE]{};
            if (instance)
                xrResultToString(instance, result, name);
            return lfs::Status::failure(vrError(name[0] ? std::format("{} ({}).", what, name) : std::format("{}.", what)));
        }

        [[nodiscard]] static lfs::Status checkVk(const VkResult result, const std::string_view operation) {
            if (result == VK_SUCCESS)
                return {};
            return lfs::Status::failure(
                lfs::rendering::vulkan_result_to_error(result, operation, LFS_SOURCE_SITE_CURRENT()));
        }

        // The runtime may submit to the graphics queue inside these calls.
        template <class Call>
        [[nodiscard]] XrResult onGraphicsQueue(Call&& call) {
            return s.context->withGraphicsQueue(std::forward<Call>(call));
        }

        lfs::Status prepare() {
            std::uint32_t count = 0;
            if (auto st = check(xrEnumerateInstanceExtensionProperties(nullptr, 0, &count, nullptr),
                                "No OpenXR runtime found");
                !st)
                return st;
            std::vector<XrExtensionProperties> extensions(count, {XR_TYPE_EXTENSION_PROPERTIES});
            if (auto st = check(xrEnumerateInstanceExtensionProperties(nullptr, count, &count, extensions.data()),
                                "Could not query your OpenXR runtime");
                !st)
                return st;
            if (std::ranges::none_of(extensions, [](const XrExtensionProperties& extension) {
                    return std::strcmp(extension.extensionName, XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME) == 0;
                }))
                return lfs::Status::failure(vrError("Your OpenXR runtime does not support Vulkan rendering."));

            const char* const enabled[] = {XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME};
            XrInstanceCreateInfo create{XR_TYPE_INSTANCE_CREATE_INFO};
            copyName(create.applicationInfo.applicationName, "LichtFeld Studio");
            create.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 0);
            create.enabledExtensionCount = 1;
            create.enabledExtensionNames = enabled;
            if (auto st = check(xrCreateInstance(&create, &instance), "Could not start your OpenXR runtime"); !st)
                return st;
            XrSystemGetInfo system_info{XR_TYPE_SYSTEM_GET_INFO};
            system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
            if (auto st = check(xrGetSystem(instance, &system_info, &system),
                                "No headset found; connect it before starting LichtFeld Studio");
                !st)
                return st;

            const auto load = [this](auto& function, const char* name) {
                return check(xrGetInstanceProcAddr(instance, name, reinterpret_cast<PFN_xrVoidFunction*>(&function)), name);
            };
            for (auto st : {load(vk.create_instance, "xrCreateVulkanInstanceKHR"),
                            load(vk.create_device, "xrCreateVulkanDeviceKHR"),
                            load(vk.graphics_device, "xrGetVulkanGraphicsDevice2KHR"),
                            load(vk.graphics_requirements, "xrGetVulkanGraphicsRequirements2KHR")}) {
                if (!st)
                    return st;
            }
            XrGraphicsRequirementsVulkan2KHR requirements{XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN2_KHR};
            if (auto st = check(vk.graphics_requirements(instance, system, &requirements),
                                "Could not read the headset's Vulkan requirements");
                !st)
                return st;
            constexpr XrVersion vulkan = XR_MAKE_VERSION(1, 3, 0);
            if (vulkan < requirements.minApiVersionSupported)
                return lfs::Status::failure(vrError("Your OpenXR runtime requires a Vulkan version newer than 1.3."));
            if (vulkan > requirements.maxApiVersionSupported)
                LOG_WARN("The OpenXR runtime was tested with a Vulkan version below 1.3; using Vulkan 1.3");

            XrInstanceProperties properties{XR_TYPE_INSTANCE_PROPERTIES};
            if (auto st = check(xrGetInstanceProperties(instance, &properties), "xrGetInstanceProperties"); !st)
                return st;
            runtime_name = properties.runtimeName;
            return {};
        }

        // The runtime creates the window's Vulkan instance and device.
        [[nodiscard]] VulkanDeviceHooks deviceHooks() {
            VulkanDeviceHooks hooks;
            hooks.create_instance = [this](const VkInstanceCreateInfo& info, VkInstance& vulkan_instance) {
                runtime_device = VK_NULL_HANDLE;
                XrVulkanInstanceCreateInfoKHR create{XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
                create.systemId = system;
                create.pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
                create.vulkanCreateInfo = &info;
                VkResult result = VK_ERROR_INITIALIZATION_FAILED;
                const XrResult created = vk.create_instance(instance, &create, &vulkan_instance, &result);
                return XR_SUCCEEDED(created) ? result : VK_ERROR_INITIALIZATION_FAILED;
            };
            hooks.required_physical_device = [this](const VkInstance vulkan_instance) {
                XrVulkanGraphicsDeviceGetInfoKHR get{XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
                get.systemId = system;
                get.vulkanInstance = vulkan_instance;
                VkPhysicalDevice physical = VK_NULL_HANDLE;
                return XR_SUCCEEDED(vk.graphics_device(instance, &get, &physical)) ? physical : VK_NULL_HANDLE;
            };
            hooks.create_device = [this](const VkPhysicalDevice physical, const VkDeviceCreateInfo& info, VkDevice& device) {
                XrVulkanDeviceCreateInfoKHR create{XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
                create.systemId = system;
                create.pfnGetInstanceProcAddr = vkGetInstanceProcAddr;
                create.vulkanPhysicalDevice = physical;
                create.vulkanCreateInfo = &info;
                VkResult result = VK_ERROR_INITIALIZATION_FAILED;
                const XrResult created = vk.create_device(instance, &create, &device, &result);
                if (!XR_SUCCEEDED(created))
                    return VK_ERROR_INITIALIZATION_FAILED;
                if (result == VK_SUCCESS)
                    runtime_device = device;
                return result;
            };
            return hooks;
        }

        [[nodiscard]] lfs::Result<XrPath> path(const std::string& value) const {
            XrPath result = XR_NULL_PATH;
            if (auto st = check(xrStringToPath(instance, value.c_str(), &result), "xrStringToPath"); !st)
                return st.error();
            return result;
        }

        lfs::Status createActions() {
            XrActionSetCreateInfo set_info{XR_TYPE_ACTION_SET_CREATE_INFO};
            copyName(set_info.actionSetName, "navigation");
            copyName(set_info.localizedActionSetName, "Navigation");
            if (auto st = check(xrCreateActionSet(instance, &set_info, &s.action_set), "xrCreateActionSet"); !st)
                return st;
            struct ActionSpec {
                XrAction* action;
                const char* name;
                const char* label;
                XrActionType type;
            };
            const ActionSpec actions[] = {
                {&s.move, "move", "Move", XR_ACTION_TYPE_VECTOR2F_INPUT},
                {&s.turn, "turn", "Snap turn and vertical movement", XR_ACTION_TYPE_VECTOR2F_INPUT},
                {&s.up, "up", "Move up", XR_ACTION_TYPE_BOOLEAN_INPUT},
                {&s.down, "down", "Move down", XR_ACTION_TYPE_BOOLEAN_INPUT},
                {&s.center, "recenter", "Recenter", XR_ACTION_TYPE_BOOLEAN_INPUT},
            };
            for (const auto& spec : actions) {
                XrActionCreateInfo info{XR_TYPE_ACTION_CREATE_INFO};
                copyName(info.actionName, spec.name);
                copyName(info.localizedActionName, spec.label);
                info.actionType = spec.type;
                if (auto st = check(xrCreateAction(s.action_set, &info, spec.action), "xrCreateAction"); !st)
                    return st;
            }

            // Left hand moves and recenters; right hand turns and climbs.
            struct ProfileSpec {
                const char* profile;
                const char* axis;
                const char* recenter;
                const char* up;
                const char* down;
            };
            constexpr ProfileSpec profiles[] = {
                {"/interaction_profiles/oculus/touch_controller", "thumbstick", "thumbstick/click", "a/click", "x/click"},
                {"/interaction_profiles/valve/index_controller", "thumbstick", "thumbstick/click", "a/click", "a/click"},
                {"/interaction_profiles/htc/vive_controller", "trackpad", "menu/click", "trigger/click", "trigger/click"},
            };
            const std::string left = "/user/hand/left/input/";
            const std::string right = "/user/hand/right/input/";
            for (const auto& profile : profiles) {
                const std::array<std::pair<XrAction, std::string>, 5> components{{
                    {s.move, left + profile.axis},
                    {s.turn, right + profile.axis},
                    {s.center, left + profile.recenter},
                    {s.up, right + profile.up},
                    {s.down, left + profile.down},
                }};
                std::array<XrActionSuggestedBinding, std::tuple_size_v<decltype(components)>> bindings{};
                for (std::size_t i = 0; i < bindings.size(); ++i) {
                    auto binding = path(components[i].second);
                    if (!binding)
                        return lfs::Status::failure(binding.error());
                    bindings[i] = {components[i].first, *binding};
                }
                auto interaction_profile = path(profile.profile);
                if (!interaction_profile)
                    return lfs::Status::failure(interaction_profile.error());
                XrInteractionProfileSuggestedBinding suggestion{XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING};
                suggestion.interactionProfile = *interaction_profile;
                suggestion.countSuggestedBindings = static_cast<std::uint32_t>(bindings.size());
                suggestion.suggestedBindings = bindings.data();
                // Runtimes need not know every profile; the others stay usable.
                if (const auto result = xrSuggestInteractionProfileBindings(instance, &suggestion);
                    result != XR_ERROR_PATH_UNSUPPORTED) {
                    if (auto st = check(result, "xrSuggestInteractionProfileBindings"); !st)
                        return st;
                }
            }
            XrSessionActionSetsAttachInfo attach{XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO};
            attach.countActionSets = 1;
            attach.actionSets = &s.action_set;
            return check(xrAttachSessionActionSets(s.handle, &attach), "xrAttachSessionActionSets");
        }

        lfs::Status startSession(const glm::mat3& rotation, const glm::vec3& position) {
            auto& context = *s.context;
            XrGraphicsBindingVulkan2KHR binding{XR_TYPE_GRAPHICS_BINDING_VULKAN2_KHR};
            binding.instance = context.instance();
            binding.physicalDevice = context.physicalDevice();
            binding.device = context.device();
            binding.queueFamilyIndex = context.graphicsQueueFamily();
            binding.queueIndex = 0;
            XrSessionCreateInfo create{XR_TYPE_SESSION_CREATE_INFO};
            create.next = &binding;
            create.systemId = system;
            if (auto st = check(xrCreateSession(instance, &create, &s.handle), "Could not start the VR session"); !st)
                return st;
            XrReferenceSpaceCreateInfo space{XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
            space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
            space.poseInReferenceSpace.orientation.w = 1.0f;
            if (auto st = check(xrCreateReferenceSpace(s.handle, &space, &s.space), "xrCreateReferenceSpace"); !st)
                return st;
            space.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
            if (auto st = check(xrCreateReferenceSpace(s.handle, &space, &s.head_space), "xrCreateReferenceSpace"); !st)
                return st;
            if (auto st = createActions(); !st)
                return st;

            std::uint32_t count = 0;
            if (auto st = check(xrEnumerateEnvironmentBlendModes(instance, system, kStereo, 0, &count, nullptr),
                                "xrEnumerateEnvironmentBlendModes");
                !st)
                return st;
            std::vector<XrEnvironmentBlendMode> blends(count);
            if (auto st = check(xrEnumerateEnvironmentBlendModes(instance, system, kStereo, count, &count, blends.data()),
                                "xrEnumerateEnvironmentBlendModes");
                !st)
                return st;
            if (std::ranges::find(blends, XR_ENVIRONMENT_BLEND_MODE_OPAQUE) == blends.end())
                return lfs::Status::failure(vrError("The headset has no opaque display mode."));
            if (auto st = check(xrEnumerateSwapchainFormats(s.handle, 0, &count, nullptr), "xrEnumerateSwapchainFormats"); !st)
                return st;
            std::vector<std::int64_t> formats(count);
            if (auto st = check(xrEnumerateSwapchainFormats(s.handle, count, &count, formats.data()),
                                "xrEnumerateSwapchainFormats");
                !st)
                return st;
            // Eyes hold display-encoded RGBA8; copying their bits into an sRGB
            // swapchain avoids a second gamma conversion.
            if (std::ranges::find(formats, std::int64_t{VK_FORMAT_R8G8B8A8_SRGB}) == formats.end())
                return lfs::Status::failure(vrError("Your OpenXR runtime offers no RGBA8 sRGB swapchain."));
            std::array<XrViewConfigurationView, 2> views{{{XR_TYPE_VIEW_CONFIGURATION_VIEW}, {XR_TYPE_VIEW_CONFIGURATION_VIEW}}};
            if (auto st = check(xrEnumerateViewConfigurationViews(instance, system, kStereo, 2, &count, views.data()),
                                "xrEnumerateViewConfigurationViews");
                !st)
                return st;
            if (count != 2)
                return lfs::Status::failure(vrError("The headset has no two-eye stereo view."));

            const auto preferences = UserPreferences::instance().vrPreferences();
            for (std::size_t i = 0; i < s.eyes.size(); ++i) {
                auto& eye = s.eyes[i];
                const auto scaled = [&](const std::uint32_t recommended, const std::uint32_t max) {
                    return std::clamp(static_cast<int>(static_cast<float>(recommended) * preferences.resolution_scale),
                                      1, static_cast<int>(max));
                };
                eye.size = {scaled(views[i].recommendedImageRectWidth, views[i].maxImageRectWidth),
                            scaled(views[i].recommendedImageRectHeight, views[i].maxImageRectHeight)};
                XrSwapchainCreateInfo swapchain{XR_TYPE_SWAPCHAIN_CREATE_INFO};
                swapchain.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
                swapchain.format = VK_FORMAT_R8G8B8A8_SRGB;
                swapchain.sampleCount = 1;
                swapchain.width = static_cast<std::uint32_t>(eye.size.x);
                swapchain.height = static_cast<std::uint32_t>(eye.size.y);
                swapchain.faceCount = swapchain.arraySize = swapchain.mipCount = 1;
                if (auto st = check(xrCreateSwapchain(s.handle, &swapchain, &eye.swapchain), "xrCreateSwapchain"); !st)
                    return st;
                if (auto st = check(xrEnumerateSwapchainImages(eye.swapchain, 0, &count, nullptr), "xrEnumerateSwapchainImages"); !st)
                    return st;
                eye.images.assign(count, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN2_KHR});
                if (auto st = check(xrEnumerateSwapchainImages(eye.swapchain, count, &count,
                                                               reinterpret_cast<XrSwapchainImageBaseHeader*>(eye.images.data())),
                                    "xrEnumerateSwapchainImages");
                    !st)
                    return st;
            }

            VkCommandPoolCreateInfo pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
            pool.queueFamilyIndex = context.graphicsQueueFamily();
            if (auto st = checkVk(vkCreateCommandPool(context.device(), &pool, nullptr, &s.pool), "vkCreateCommandPool(VR)"); !st)
                return st;
            VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            allocate.commandPool = s.pool;
            allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            allocate.commandBufferCount = 1;
            if (auto st = checkVk(vkAllocateCommandBuffers(context.device(), &allocate, &s.command), "vkAllocateCommandBuffers(VR)"); !st)
                return st;
            VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            if (auto st = checkVk(vkCreateFence(context.device(), &fence, nullptr, &s.fence), "vkCreateFence(VR)"); !st)
                return st;

            s.start_orientation = glm::quat_cast(rotation);
            s.start_position = position;
            s.rig.scene_units_per_meter = preferences.scene_units_per_meter;
            move_speed = preferences.move_speed;
            return {};
        }

        lfs::Status poll() {
            XrEventDataBuffer event{XR_TYPE_EVENT_DATA_BUFFER};
            XrResult result;
            while ((result = xrPollEvent(instance, &event)) == XR_SUCCESS) {
                if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
                    const auto& changed = reinterpret_cast<const XrEventDataSessionStateChanged&>(event);
                    if (changed.session == s.handle) {
                        s.state = changed.state;
                        LOG_INFO("VR session {}", static_cast<std::size_t>(s.state) < kSessionStates.size()
                                                      ? kSessionStates[s.state]
                                                      : kSessionStates[0]);
                        if (s.state == XR_SESSION_STATE_READY && !s.running) {
                            XrSessionBeginInfo begin{XR_TYPE_SESSION_BEGIN_INFO};
                            begin.primaryViewConfigurationType = kStereo;
                            if (auto st = check(xrBeginSession(s.handle, &begin), "xrBeginSession"); !st)
                                return st;
                            s.running = true;
                        } else if (s.state == XR_SESSION_STATE_STOPPING) {
                            s.ended = true;
                            if (s.running) {
                                s.running = false;
                                if (auto st = check(xrEndSession(s.handle), "xrEndSession"); !st)
                                    return st;
                            }
                        } else if (s.state == XR_SESSION_STATE_EXITING || s.state == XR_SESSION_STATE_LOSS_PENDING) {
                            s.ended = true;
                        }
                    }
                } else if (event.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
                    lost = true;
                    s.ended = true;
                    setStatus("vr.status_runtime_lost");
                } else if (event.type == XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING) {
                    const auto& change = reinterpret_cast<const XrEventDataReferenceSpaceChangePending&>(event);
                    if (change.session == s.handle && change.referenceSpaceType == XR_REFERENCE_SPACE_TYPE_LOCAL)
                        s.space_change = change;
                }
                event = {XR_TYPE_EVENT_DATA_BUFFER};
            }
            return result == XR_EVENT_UNAVAILABLE ? lfs::Status{} : check(result, "xrPollEvent");
        }

        struct ControllerInput {
            glm::vec2 move{0.0f};
            glm::vec2 turn{0.0f};
            bool up = false;
            bool down = false;
            bool center = false;
        };

        [[nodiscard]] lfs::Result<ControllerInput> readControllers() const {
            const XrActiveActionSet active{s.action_set, XR_NULL_PATH};
            XrActionsSyncInfo sync{XR_TYPE_ACTIONS_SYNC_INFO};
            sync.countActiveActionSets = 1;
            sync.activeActionSets = &active;
            if (auto st = check(xrSyncActions(s.handle, &sync), "xrSyncActions"); !st)
                return st.error();
            ControllerInput input;
            for (const auto& [action, value] : {std::pair{s.up, &input.up}, std::pair{s.down, &input.down},
                                                std::pair{s.center, &input.center}}) {
                XrActionStateGetInfo info{XR_TYPE_ACTION_STATE_GET_INFO};
                info.action = action;
                XrActionStateBoolean state{XR_TYPE_ACTION_STATE_BOOLEAN};
                if (auto st = check(xrGetActionStateBoolean(s.handle, &info, &state), "xrGetActionStateBoolean"); !st)
                    return st.error();
                *value = state.isActive && state.currentState;
            }
            for (const auto& [action, value] : {std::pair{s.move, &input.move}, std::pair{s.turn, &input.turn}}) {
                XrActionStateGetInfo info{XR_TYPE_ACTION_STATE_GET_INFO};
                info.action = action;
                XrActionStateVector2f state{XR_TYPE_ACTION_STATE_VECTOR2F};
                if (auto st = check(xrGetActionStateVector2f(s.handle, &info, &state), "xrGetActionStateVector2f"); !st)
                    return st.error();
                *value = state.isActive ? glm::vec2(state.currentState.x, state.currentState.y) : glm::vec2(0.0f);
            }
            return input;
        }

        // move: keyboard axes (+x right, +y up, +z back).
        lfs::Status navigate(const XrPosef& head, const XrTime time, const glm::vec3 move) {
            if (s.space_change && time >= s.space_change->changeTime) {
                if (s.space_change->poseValid) {
                    const auto& pose = s.space_change->poseInPreviousSpace;
                    s.rig.position += s.rig.orientation * (vr::position(pose) * s.rig.scene_units_per_meter);
                    s.rig.orientation = glm::normalize(s.rig.orientation * vr::orientation(pose));
                } else {
                    s.recenter_pending = true;
                }
                s.space_change.reset();
            }
            const auto head_orientation = vr::orientation(head);
            const auto head_position = vr::position(head);
            const float dt = s.last_time ? std::clamp(static_cast<float>(static_cast<double>(time - s.last_time) * 1e-9), 0.0f, 0.05f)
                                         : 0.0f;
            s.last_time = time;

            const bool focused = s.state == XR_SESSION_STATE_FOCUSED;
            ControllerInput input;
            if (focused) {
                auto read = readControllers();
                if (!read)
                    return lfs::Status::failure(read.error());
                input = *read;
                s.recenter_pending |= input.center && !s.center_down;
                s.center_down = input.center;
            } else {
                s.snap.reset();
                s.center_down = false;
                s.pending_turns = 0;
            }
            if (s.recenter_pending) {
                s.rig.recenter(head_orientation, head_position, s.start_orientation, s.start_position);
                s.recenter_pending = false;
                s.snap.reset();
            }
            if (!focused)
                return {};

            const int turns = s.pending_turns + s.snap.update(input.turn.x);
            s.pending_turns = 0;
            if (turns != 0)
                s.rig.snapTurn(-static_cast<float>(turns) * kSnapTurnRadians, head_position);
            const glm::quat heading = yawOnlyOrientation(s.rig.orientation * head_orientation);
            const glm::vec2 planar =
                glm::clamp(deadzone(input.move, 0.2f) + glm::vec2(move.x, -move.z), glm::vec2(-1.0f), glm::vec2(1.0f));
            const float vertical = static_cast<float>(input.up) - static_cast<float>(input.down) +
                                   deadzone(input.turn.y, 0.3f) + move.y;
            s.rig.position += (heading * glm::vec3(planar.x, 0.0f, -planar.y) + glm::vec3(0.0f, vertical, 0.0f)) *
                              move_speed * s.rig.scene_units_per_meter * dt;
            return {};
        }

        lfs::Status waitTransfer() {
            const auto waited = lfs::rendering::wait_fence_bounded(s.context->device(), s.fence, {});
            if (!waited)
                return lfs::Status::failure(waited.error());
            if (*waited != lfs::rendering::WaitOutcome::Ready)
                return lfs::Status::failure(vrError("The headset image copy did not finish."));
            s.transfer_pending = false;
            return {};
        }

        // Copies the rendered eyes into the acquired swapchain images.
        lfs::Status transfer(const EyeFrames& sources) {
            const VkDevice device = s.context->device();
            if (auto st = checkVk(vkResetCommandPool(device, s.pool, 0), "vkResetCommandPool(VR)"); !st)
                return st;
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (auto st = checkVk(vkBeginCommandBuffer(s.command, &begin), "vkBeginCommandBuffer(VR)"); !st)
                return st;

            constexpr VkPipelineStageFlags2 all = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            constexpr VkAccessFlags2 memory = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
            constexpr VkPipelineStageFlags2 copy_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            std::array<VkImageMemoryBarrier2, 4> before{};
            std::array<VkImageMemoryBarrier2, 4> after{};
            std::array<VkSemaphore, 2> waits{};
            std::array<std::uint64_t, 2> wait_values{};
            std::uint32_t wait_count = 0;
            std::array<VkImage, 2> images{};
            for (std::size_t i = 0; i < sources.size(); ++i) {
                const auto& source = sources[i];
                const auto& eye = s.eyes[i];
                images[i] = vulkanSceneImage(source.image);
                if (!images[i] || source.size != eye.size || source.flip_y)
                    return lfs::Status::failure(vrError("The VR renderer returned an eye image of the wrong size."));
                if (const VkSemaphore completion = vulkanSceneTimeline(source.completion_timeline);
                    completion && source.completion_value) {
                    if (wait_count > 0 && waits[0] == completion) {
                        wait_values[0] = std::max(wait_values[0], source.completion_value);
                    } else {
                        waits[wait_count] = completion;
                        wait_values[wait_count++] = source.completion_value;
                    }
                }
                // Runtime images may discard their contents on acquire.
                const VkImage destination = eye.images[eye.index].image;
                const VkImageLayout layout = vulkanSceneImageLayout(source.image_layout);
                before[2 * i] = imageBarrier(images[i], layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                             all, memory, copy_stage, VK_ACCESS_2_TRANSFER_READ_BIT);
                before[2 * i + 1] = imageBarrier(destination, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                 all, 0, copy_stage, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                after[2 * i] = imageBarrier(images[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout,
                                            copy_stage, VK_ACCESS_2_TRANSFER_READ_BIT, all, memory);
                after[2 * i + 1] = imageBarrier(destination, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                copy_stage, VK_ACCESS_2_TRANSFER_WRITE_BIT, all, memory);
            }
            const auto barriers = [&](const std::array<VkImageMemoryBarrier2, 4>& images) {
                VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                dependency.imageMemoryBarrierCount = static_cast<std::uint32_t>(images.size());
                dependency.pImageMemoryBarriers = images.data();
                vkCmdPipelineBarrier2(s.command, &dependency);
            };
            barriers(before);
            for (std::size_t i = 0; i < sources.size(); ++i) {
                VkImageCopy copy{};
                copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                copy.dstSubresource = copy.srcSubresource;
                copy.extent = {static_cast<std::uint32_t>(s.eyes[i].size.x), static_cast<std::uint32_t>(s.eyes[i].size.y), 1};
                vkCmdCopyImage(s.command, images[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               s.eyes[i].images[s.eyes[i].index].image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
            }
            barriers(after);
            if (auto st = checkVk(vkEndCommandBuffer(s.command), "vkEndCommandBuffer(VR)"); !st)
                return st;

            const std::array<VkPipelineStageFlags, 2> wait_stages{VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
            VkTimelineSemaphoreSubmitInfo timeline{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            timeline.waitSemaphoreValueCount = wait_count;
            timeline.pWaitSemaphoreValues = wait_values.data();
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            submit.pNext = &timeline;
            submit.waitSemaphoreCount = wait_count;
            submit.pWaitSemaphores = waits.data();
            submit.pWaitDstStageMask = wait_stages.data();
            submit.commandBufferCount = 1;
            submit.pCommandBuffers = &s.command;
            if (auto st = checkVk(vkResetFences(device, 1, &s.fence), "vkResetFences(VR)"); !st)
                return st;
            if (auto st = checkVk(lfs::rendering::vk_queue_submit_synced(s.context->graphicsQueue(), 1, &submit, s.fence),
                                  "vkQueueSubmit(VR)");
                !st)
                return st;
            s.transfer_pending = true;
            // Waiting here also keeps the renderer from reusing the eye images early.
            return waitTransfer();
        }

        // Returns whether the eyes were drawn this frame.
        lfs::Result<bool> drawEyes(const glm::vec3 move, const RenderEyes& render,
                                   std::array<XrCompositionLayerProjectionView, 2>& layer_views) {
            std::array<XrView, 2> views{{{XR_TYPE_VIEW}, {XR_TYPE_VIEW}}};
            XrViewLocateInfo locate{XR_TYPE_VIEW_LOCATE_INFO};
            locate.viewConfigurationType = kStereo;
            locate.displayTime = s.display_time;
            locate.space = s.space;
            XrViewState view_state{XR_TYPE_VIEW_STATE};
            std::uint32_t count = 0;
            if (auto st = check(xrLocateViews(s.handle, &locate, &view_state, 2, &count, views.data()), "xrLocateViews"); !st)
                return st.error();
            XrSpaceLocation head{XR_TYPE_SPACE_LOCATION};
            if (auto st = check(xrLocateSpace(s.head_space, s.space, s.display_time, &head), "xrLocateSpace"); !st)
                return st.error();
            constexpr XrViewStateFlags views_valid = XR_VIEW_STATE_POSITION_VALID_BIT | XR_VIEW_STATE_ORIENTATION_VALID_BIT;
            constexpr XrSpaceLocationFlags head_valid =
                XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
            if (count != 2 || (view_state.viewStateFlags & views_valid) != views_valid ||
                (head.locationFlags & head_valid) != head_valid)
                return false;
            if (auto st = navigate(head.pose, s.display_time, move); !st)
                return st.error();

            std::array<lfs::rendering::FrameView, 2> frame_views;
            for (std::size_t i = 0; i < views.size(); ++i) {
                const auto& fov = views[i].fov;
                const auto intrinsics = eyeIntrinsics({fov.angleLeft, fov.angleRight, fov.angleUp, fov.angleDown}, s.eyes[i].size);
                if (!intrinsics)
                    return false;
                const auto pose = s.rig.cameraPose(vr::orientation(views[i].pose), vr::position(views[i].pose));
                frame_views[i].rotation = pose.rotation;
                frame_views[i].translation = pose.translation;
                frame_views[i].size = s.eyes[i].size;
                frame_views[i].intrinsics_override = intrinsics;
            }
            for (std::size_t i = 0; i < views.size(); ++i) {
                auto& eye = s.eyes[i];
                XrSwapchainImageAcquireInfo acquire{XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
                if (auto st = check(xrAcquireSwapchainImage(eye.swapchain, &acquire, &eye.index), "xrAcquireSwapchainImage"); !st)
                    return st.error();
                XrSwapchainImageWaitInfo wait{XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
                wait.timeout = 1'000'000'000; // 1 s
                const XrResult waited = xrWaitSwapchainImage(eye.swapchain, &wait);
                if (waited == XR_TIMEOUT_EXPIRED)
                    return vrError("Timed out waiting for a headset image.");
                if (auto st = check(waited, "xrWaitSwapchainImage"); !st)
                    return st.error();
                eye.acquired = true;
                layer_views[i].pose = views[i].pose;
                layer_views[i].fov = views[i].fov;
                layer_views[i].subImage.swapchain = eye.swapchain;
                layer_views[i].subImage.imageRect.extent = {eye.size.x, eye.size.y};
            }
            auto eyes = render(frame_views);
            if (!eyes)
                return eyes.error();
            if (auto st = transfer(*eyes); !st)
                return st.error();
            for (auto& eye : s.eyes) {
                XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
                const XrResult released = onGraphicsQueue([&] { return xrReleaseSwapchainImage(eye.swapchain, &release); });
                if (auto st = check(released, "xrReleaseSwapchainImage"); !st)
                    return st.error();
                eye.acquired = false;
            }
            s.mirror = (*eyes)[0];
            return true;
        }

        lfs::Status submitFrame(const glm::vec3 move, const RenderEyes& render) {
            XrFrameWaitInfo wait{XR_TYPE_FRAME_WAIT_INFO};
            XrFrameState frame_state{XR_TYPE_FRAME_STATE};
            if (auto st = check(xrWaitFrame(s.handle, &wait, &frame_state), "xrWaitFrame"); !st)
                return st;
            s.display_time = frame_state.predictedDisplayTime;
            XrFrameBeginInfo begin{XR_TYPE_FRAME_BEGIN_INFO};
            if (auto st = check(onGraphicsQueue([&] { return xrBeginFrame(s.handle, &begin); }), "xrBeginFrame"); !st)
                return st;
            s.frame_begun = true;

            std::array<XrCompositionLayerProjectionView, 2> layer_views{
                {{XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}, {XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW}}};
            bool drawn = false;
            if (frame_state.shouldRender) {
                auto eyes = drawEyes(move, render, layer_views);
                if (!eyes)
                    return lfs::Status::failure(eyes.error());
                drawn = *eyes;
            }
            if (!drawn) {
                s.last_time = 0;
                s.snap.reset();
                s.mirror = {};
            }
            XrCompositionLayerProjection layer{XR_TYPE_COMPOSITION_LAYER_PROJECTION};
            layer.space = s.space;
            layer.viewCount = static_cast<std::uint32_t>(layer_views.size());
            layer.views = layer_views.data();
            const XrCompositionLayerBaseHeader* const layers[] = {
                reinterpret_cast<const XrCompositionLayerBaseHeader*>(&layer)};
            XrFrameEndInfo end{XR_TYPE_FRAME_END_INFO};
            end.displayTime = s.display_time;
            end.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
            end.layerCount = drawn ? 1 : 0;
            end.layers = drawn ? layers : nullptr;
            s.frame_begun = false;
            return check(onGraphicsQueue([&] { return xrEndFrame(s.handle, &end); }), "xrEndFrame");
        }

        // After a failed frame: hand back what the runtime lent us, unless the
        // GPU may still read it (stop() then drains the device first).
        void abandonFrame() {
            if (s.transfer_pending)
                return;
            for (auto& eye : s.eyes) {
                if (!eye.acquired)
                    continue;
                XrSwapchainImageReleaseInfo release{XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
                static_cast<void>(onGraphicsQueue([&] { return xrReleaseSwapchainImage(eye.swapchain, &release); }));
                eye.acquired = false;
            }
            if (s.frame_begun) {
                XrFrameEndInfo end{XR_TYPE_FRAME_END_INFO};
                end.displayTime = s.display_time;
                end.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
                static_cast<void>(onGraphicsQueue([&] { return xrEndFrame(s.handle, &end); }));
                s.frame_begun = false;
            }
        }
    };

    OpenXrService::OpenXrService() : impl_(std::make_unique<Impl>()) {}

    OpenXrService::~OpenXrService() {
        stop();
        if (impl_->instance)
            xrDestroyInstance(impl_->instance);
    }

    bool OpenXrService::prepare() {
        auto& p = *impl_;
        if (auto prepared = p.prepare(); !prepared) {
            p.setStatus("vr.status_unavailable", std::string(prepared.error().user_message()));
            if (p.instance)
                xrDestroyInstance(p.instance);
            p.instance = XR_NULL_HANDLE;
            p.system = XR_NULL_SYSTEM_ID;
            return false;
        }
        p.hooks = p.deviceHooks();
        p.setStatus("vr.status_ready", p.runtime_name);
        LOG_INFO("OpenXR runtime ready: {}", p.runtime_name);
        return true;
    }

    const VulkanDeviceHooks* OpenXrService::deviceHooks() const { return available() ? &impl_->hooks : nullptr; }

    void OpenXrService::setUnavailable(std::string reason) {
        auto& p = *impl_;
        p.setStatus("vr.status_unavailable", std::move(reason));
        p.lost = true;
    }

    bool OpenXrService::available() const { return impl_->instance && impl_->system && !impl_->lost; }
    bool OpenXrService::active() const { return impl_->s.handle != XR_NULL_HANDLE; }
    bool OpenXrService::running() const { return impl_->s.running; }
    const char* OpenXrService::statusKey() const { return impl_->status_key; }
    const std::string& OpenXrService::statusDetail() const { return impl_->status_detail; }
    const EyeFrame& OpenXrService::mirror() const { return impl_->s.mirror; }
    void OpenXrService::recenter() {
        if (active())
            impl_->s.recenter_pending = true;
    }
    void OpenXrService::snapTurn(const int direction) {
        if (active())
            impl_->s.pending_turns += direction;
    }
    void OpenXrService::setMoveSpeed(const float meters_per_second) { impl_->move_speed = meters_per_second; }

    lfs::Status OpenXrService::start(GraphicsContext& graphics, const glm::mat3& rotation, const glm::vec3& position) {
        auto& p = *impl_;
        if (!available())
            return lfs::Status::failure(vrError(LOCF(p.status_key, p.status_detail)));
        if (active())
            return {};
        // The eyes render on the window device; the runtime must have created it.
        VulkanContext* const context = vulkanContextOrNull(&graphics);
        if (!context || !context->device() || context->device() != p.runtime_device)
            return lfs::Status::failure(vrError("The window's Vulkan device was not created through the OpenXR runtime."));
        p.s.context = context;
        auto started = p.startSession(rotation, position);
        if (!started) {
            stop();
            p.setStatus("vr.enter_failed", std::string(started.error().user_message()));
            return started;
        }
        p.setStatus("vr.status_waiting");
        return {};
    }

    void OpenXrService::stop() {
        auto& p = *impl_;
        auto& s = p.s;
        if (!s.context)
            return;
        if (s.transfer_pending)
            static_cast<void>(s.context->deviceWaitIdle());
        const VkDevice device = s.context->device();
        if (s.fence)
            vkDestroyFence(device, s.fence, nullptr);
        if (s.pool)
            vkDestroyCommandPool(device, s.pool, nullptr);
        // A session owns its spaces and swapchains; an action set owns its actions.
        if (s.handle)
            xrDestroySession(s.handle);
        if (s.action_set)
            xrDestroyActionSet(s.action_set);
        s = {};
        if (available())
            p.setStatus("vr.status_ready", p.runtime_name);
    }

    lfs::Result<bool> OpenXrService::frame(const glm::vec3 move, const RenderEyes& render) {
        auto& p = *impl_;
        if (!active())
            return false;
        auto submitted = p.poll();
        if (submitted && p.s.running && !p.s.ended)
            submitted = p.submitFrame(move, render);
        if (!submitted) {
            p.abandonFrame();
            stop();
            p.setStatus("vr.stopped", std::string(submitted.error().user_message()));
            return submitted.error();
        }
        if (!p.s.ended && p.s.mirror.rendered)
            p.setStatus("vr.status_viewing");
        return !p.s.ended;
    }

} // namespace lfs::vis::vr

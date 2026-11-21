/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "vr_tracked_hands.hpp"
#include "core/logger.hpp"
#include "internal/resource_paths.hpp"
#include "rendering/vulkan_wait.hpp"
#include "vr_hand_model.hpp"
#include "vr_hand_renderer.hpp"
#include <algorithm>
#include <exception>
#include <format>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <vk_mem_alloc.h>

namespace lfs::vis::vr {

    namespace {
        // Linear colors of the light of a hand, and of a hand that holds the scene.
        const glm::vec3 kHandColor{0.05f, 0.17f, 0.78f};
        const glm::vec3 kHandGripColor{1.00f, 0.62f, 0.22f};
        // Meters: hands are drawn from 2 cm in front of the eye out to 20 m.
        constexpr float kHandNear = 0.02f;
        constexpr float kHandFar = 20.0f;
        static_assert(kHandJointCount == XR_HAND_JOINT_COUNT_EXT);

        [[nodiscard]] lfs::Status checkVk(const VkResult result, const std::string_view operation) {
            if (result == VK_SUCCESS)
                return {};
            return lfs::Status::failure(
                lfs::rendering::vulkan_result_to_error(result, operation, LFS_SOURCE_SITE_CURRENT()));
        }

        void transition(const VkCommandBuffer command, const VkImage image,
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
            VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            dependency.imageMemoryBarrierCount = 1;
            dependency.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(command, &dependency);
        }

        void copy(const VkCommandBuffer command, const VkImage from, const VkImage to, const glm::ivec2 size) {
            VkImageCopy region{};
            region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.dstSubresource = region.srcSubresource;
            region.extent = {static_cast<std::uint32_t>(size.x), static_cast<std::uint32_t>(size.y), 1};
            vkCmdCopyImage(command, from, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, to, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        }
    } // namespace

    struct TrackedHands::Impl {
        // Found once by findTracking(): hand joints from the runtime, when it
        // tracks hands.
        XrInstance instance = XR_NULL_HANDLE;
        PFN_xrCreateHandTrackerEXT create_tracker = nullptr;
        PFN_xrLocateHandJointsEXT locate_joints = nullptr;
        bool data_sources = false; // XR_EXT_hand_tracking_data_source is enabled
        // Left, right: the bundled hand models, read once.
        std::array<std::optional<HandModel>, 2> models;
        bool models_read = false;

        // Alive between start() and stop().
        VulkanContext* context = nullptr;
        std::array<XrHandTrackerEXT, 2> trackers{XR_NULL_HANDLE, XR_NULL_HANDLE};
        std::array<glm::ivec2, 2> eye_sizes{glm::ivec2(0), glm::ivec2(0)};
        // One view per eye image: the hands are drawn through it.
        std::array<std::vector<VkImageView>, 2> eye_views;
        HandRenderer renderer;
        std::array<std::vector<HandVertex>, 2> meshes;
        std::vector<HandVertex> scratch;
        // Left, right: the hand as the runtime tracks it this frame.
        std::array<std::optional<Hand>, 2> located;
        // The hands as they are drawn.
        std::array<HandPoseSmoother, 2> drawn;
        XrTime time = 0;
        // The left eye with the hands on it, for the desktop mirror.
        struct MirrorCopy {
            VkImage image = VK_NULL_HANDLE;
            VmaAllocation allocation = VK_NULL_HANDLE;
            VkImageView target_view = VK_NULL_HANDLE;  // sRGB: the hands are drawn through it
            VkImageView display_view = VK_NULL_HANDLE; // the eye images' own encoding: the desktop samples it
            glm::ivec2 size{0};
            bool drawn = false; // holds this frame's left eye
            bool shown = false; // the desktop may still be reading it
        } mirror;

        [[nodiscard]] bool tracking() const { return create_tracker && locate_joints; }

        // What a failed OpenXR call is called in the log.
        [[nodiscard]] std::string failure(const XrResult result, const std::string_view what) const {
            char name[XR_MAX_RESULT_STRING_SIZE]{};
            if (instance)
                xrResultToString(instance, result, name);
            return name[0] ? std::format("{} ({}).", what, name) : std::format("{}.", what);
        }

        void readModels() {
            if (std::exchange(models_read, true))
                return;
            constexpr std::array<const char*, 2> names = {"vr/hand_left.glb", "vr/hand_right.glb"};
            for (std::size_t i = 0; i < names.size(); ++i) {
                models[i].reset();
                try {
                    auto mesh = lfs::io::load_skinned_mesh(getAssetPath(names[i]));
                    if (!mesh) {
                        LOG_WARN("VR hands: could not read {}: {}", names[i], mesh.error().format());
                        continue;
                    }
                    models[i] = handModelFromSkinnedMesh(std::move(*mesh));
                    if (!models[i])
                        LOG_WARN("VR hands: {} has no bone for some of the hand joints", names[i]);
                } catch (const std::exception& error) {
                    // Not bundled: this hand is not drawn.
                    LOG_WARN("VR hands: could not read {}: {}", names[i], error.what());
                }
            }
        }

        void destroyMirror() {
            const VkDevice device = context->device();
            for (const VkImageView view : {mirror.target_view, mirror.display_view}) {
                if (view)
                    vkDestroyImageView(device, view, nullptr);
            }
            if (mirror.image)
                vmaDestroyImage(context->allocator(), mirror.image, mirror.allocation);
            mirror = {};
        }

        // Where the runtime tracks the joints of the left (0) or right (1) hand
        // for this frame; nothing while it does not track them all.
        [[nodiscard]] std::optional<Hand> locate(const std::size_t hand, const XrSpace space, const XrTime at) const {
            if (!trackers[hand])
                return std::nullopt;
            std::array<XrHandJointLocationEXT, XR_HAND_JOINT_COUNT_EXT> joints{};
            XrHandJointLocationsEXT locations{XR_TYPE_HAND_JOINT_LOCATIONS_EXT};
            locations.jointCount = static_cast<std::uint32_t>(joints.size());
            locations.jointLocations = joints.data();
            XrHandTrackingDataSourceStateEXT source{XR_TYPE_HAND_TRACKING_DATA_SOURCE_STATE_EXT};
            if (data_sources)
                locations.next = &source;
            XrHandJointsLocateInfoEXT info{XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT};
            info.baseSpace = space;
            info.time = at;
            if (XR_FAILED(locate_joints(trackers[hand], &info, &locations)) || !locations.isActive)
                return std::nullopt;
            Hand tracked;
            for (std::size_t joint = 0; joint < tracked.joints.size(); ++joint) {
                if (!(joints[joint].locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT))
                    return std::nullopt;
                const XrVector3f& position = joints[joint].pose.position;
                tracked.joints[joint] = {position.x, position.y, position.z};
            }
            if (!handJointsValid(tracked.joints))
                return std::nullopt;
            // A runtime that does not say where the joints come from leaves
            // the hand to grab as a controller does.
            tracked.bare = data_sources && source.isActive &&
                           source.dataSource == XR_HAND_TRACKING_DATA_SOURCE_UNOBSTRUCTED_EXT;
            return tracked;
        }

        void draw(const VkCommandBuffer command, const VkImageView target, const glm::ivec2 size, const EyePose& pose) {
            renderer.record(command, {target, size,
                                      eyeViewProjection(pose.fov, pose.orientation, pose.position, kHandNear, kHandFar),
                                      pose.position});
        }
    };

    TrackedHands::TrackedHands() : impl_(std::make_unique<Impl>()) {}

    TrackedHands::~TrackedHands() { stop(); }

    void TrackedHands::findTracking(const XrInstance instance, const XrSystemId system, const bool data_sources) {
        auto& p = *impl_;
        p.instance = instance;
        p.create_tracker = nullptr;
        p.locate_joints = nullptr;
        p.data_sources = false;
        if (!instance)
            return;
        XrSystemHandTrackingPropertiesEXT hand_properties{XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT};
        XrSystemProperties system_properties{XR_TYPE_SYSTEM_PROPERTIES};
        system_properties.next = &hand_properties;
        const auto optional = [instance](auto& function, const char* const name) {
            return XR_SUCCEEDED(xrGetInstanceProcAddr(instance, name, reinterpret_cast<PFN_xrVoidFunction*>(&function)));
        };
        if (XR_SUCCEEDED(xrGetSystemProperties(instance, system, &system_properties)) &&
            hand_properties.supportsHandTracking &&
            optional(p.create_tracker, "xrCreateHandTrackerEXT") &&
            optional(p.locate_joints, "xrLocateHandJointsEXT")) {
            p.data_sources = data_sources;
        } else {
            p.create_tracker = nullptr;
            p.locate_joints = nullptr;
        }
    }

    void TrackedHands::start(VulkanContext& context, const XrSession session, const std::array<Eye, 2>& eyes) {
        auto& p = *impl_;
        p.context = &context;
        for (std::size_t i = 0; i < eyes.size(); ++i)
            p.eye_sizes[i] = eyes[i].size;
        if (!p.tracking())
            return;
        p.readModels();
        if (!p.models[0] && !p.models[1]) {
            LOG_WARN("VR hands: the hand models are missing from the assets");
            return;
        }
        for (std::size_t i = 0; i < p.trackers.size(); ++i) {
            XrHandTrackerCreateInfoEXT create{XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT};
            create.hand = i == 0 ? XR_HAND_LEFT_EXT : XR_HAND_RIGHT_EXT;
            create.handJointSet = XR_HAND_JOINT_SET_DEFAULT_EXT;
            // A hand around a controller counts too: the runtime then poses
            // its fingers from the buttons and sticks they touch.
            std::array<XrHandTrackingDataSourceEXT, 2> sources{XR_HAND_TRACKING_DATA_SOURCE_UNOBSTRUCTED_EXT,
                                                               XR_HAND_TRACKING_DATA_SOURCE_CONTROLLER_EXT};
            XrHandTrackingDataSourceInfoEXT source_info{XR_TYPE_HAND_TRACKING_DATA_SOURCE_INFO_EXT};
            source_info.requestedDataSourceCount = static_cast<std::uint32_t>(sources.size());
            source_info.requestedDataSources = sources.data();
            if (p.data_sources)
                create.next = &source_info;
            if (const XrResult result = p.create_tracker(session, &create, &p.trackers[i]); XR_FAILED(result)) {
                LOG_WARN("VR hands: {}", p.failure(result, "xrCreateHandTrackerEXT"));
                p.trackers[i] = XR_NULL_HANDLE;
            }
        }
        if (!p.trackers[0] && !p.trackers[1])
            return;

        for (std::size_t i = 0; i < eyes.size(); ++i) {
            for (const VkImage image : eyes[i].images) {
                VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
                view.image = image;
                view.viewType = VK_IMAGE_VIEW_TYPE_2D;
                view.format = VK_FORMAT_R8G8B8A8_SRGB;
                view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                VkImageView handle = VK_NULL_HANDLE;
                if (auto st = checkVk(vkCreateImageView(context.device(), &view, nullptr, &handle), "vkCreateImageView(VR eye)"); !st) {
                    LOG_WARN("VR hands: {}", st.error().user_message());
                    return;
                }
                p.eye_views[i].push_back(handle);
            }
        }
        if (auto st = p.renderer.init(context, VK_FORMAT_R8G8B8A8_SRGB, glm::max(p.eye_sizes[0], p.eye_sizes[1])); !st) {
            LOG_WARN("VR hands: {}", st.error().user_message());
            return;
        }
        LOG_INFO("VR hands: drawn from the runtime's hand tracking");
        if (!p.data_sources)
            LOG_INFO("VR hands: the runtime does not tell a bare hand from one around a controller; both grab with the grip");

        // The eye images hold display-encoded bits in a plain RGBA8 image.
        auto& mirror = p.mirror;
        VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = VK_FORMAT_R8G8B8A8_UNORM;
        image.extent = {static_cast<std::uint32_t>(p.eye_sizes[0].x), static_cast<std::uint32_t>(p.eye_sizes[0].y), 1};
        image.mipLevels = image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        image.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo allocation{};
        allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        auto created = checkVk(vmaCreateImage(context.allocator(), &image, &allocation, &mirror.image, &mirror.allocation, nullptr),
                               "vmaCreateImage(VR mirror)");
        for (const auto& [format, view] : {std::pair{VK_FORMAT_R8G8B8A8_SRGB, &mirror.target_view},
                                           std::pair{VK_FORMAT_R8G8B8A8_UNORM, &mirror.display_view}}) {
            if (!created)
                break;
            VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            info.image = mirror.image;
            info.viewType = VK_IMAGE_VIEW_TYPE_2D;
            info.format = format;
            info.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            created = checkVk(vkCreateImageView(context.device(), &info, nullptr, view), "vkCreateImageView(VR mirror)");
        }
        if (created) {
            mirror.size = p.eye_sizes[0];
        } else {
            // Without it the hands still show in the headset.
            LOG_WARN("VR hands are not mirrored on the desktop: {}", created.error().user_message());
            p.destroyMirror();
        }
    }

    void TrackedHands::stop() {
        auto& p = *impl_;
        p.renderer.shutdown();
        if (!p.context)
            return;
        if (p.mirror.shown)
            static_cast<void>(p.context->waitForSubmittedFrames());
        p.destroyMirror();
        const VkDevice device = p.context->device();
        for (auto& views : p.eye_views) {
            for (const VkImageView view : views)
                vkDestroyImageView(device, view, nullptr);
            views.clear();
        }
        p.trackers = {XR_NULL_HANDLE, XR_NULL_HANDLE};
        p.eye_sizes = {glm::ivec2(0), glm::ivec2(0)};
        p.located = {};
        p.drawn = {};
        p.time = 0;
        p.context = nullptr;
    }

    void TrackedHands::locate(const XrSpace space, const XrTime time) {
        auto& p = *impl_;
        for (std::size_t i = 0; i < p.located.size(); ++i)
            p.located[i] = p.locate(i, space, time);
    }

    const std::optional<TrackedHands::Hand>& TrackedHands::hand(const std::size_t index) const {
        return impl_->located[index];
    }

    void TrackedHands::update(const XrTime time, const std::array<bool, 2>& gripping, const bool show) {
        auto& p = *impl_;
        if (!p.renderer.ready())
            return;
        const float dt = p.time ? std::clamp(static_cast<float>(static_cast<double>(time - p.time) * 1e-9), 0.0f, 0.05f)
                                : 0.0f;
        p.time = time;
        std::array<glm::vec3, 2> colors{};
        for (std::size_t i = 0; i < p.meshes.size(); ++i) {
            p.meshes[i].clear();
            colors[i] = gripping[i] ? kHandGripColor : kHandColor;
            const auto& tracked = p.located[i];
            if (!show || !p.models[i] || !tracked) {
                // A hand that comes back starts where it is, not where it was.
                p.drawn[i].reset();
                continue;
            }
            // Only where the joints are is used: see handPoseFromJoints().
            const HandPose pose = p.drawn[i].update(handPoseFromJoints(tracked->joints), dt);
            appendSkinnedHand(p.meshes[i], *p.models[i], pose, p.scratch);
        }
        p.renderer.setHands({p.meshes[0], p.meshes[1]}, colors);
    }

    bool TrackedHands::recordMirroredEye(const VkCommandBuffer command, const VkImage source, const VkImage destination,
                                         const EyePose& pose) {
        auto& p = *impl_;
        auto& mirror = p.mirror;
        mirror.drawn = p.renderer.hasHands() && mirror.image && mirror.size == p.eye_sizes[0];
        if (!mirror.drawn)
            return false;
        constexpr VkPipelineStageFlags2 all = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        constexpr VkAccessFlags2 memory = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
        constexpr VkPipelineStageFlags2 copy_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        constexpr VkPipelineStageFlags2 draw_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        constexpr VkAccessFlags2 draw_access = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        transition(command, mirror.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                   all, 0, copy_stage, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        copy(command, source, mirror.image, mirror.size);
        transition(command, mirror.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                   copy_stage, VK_ACCESS_2_TRANSFER_WRITE_BIT, draw_stage, draw_access);
        p.draw(command, mirror.target_view, mirror.size, pose);
        transition(command, mirror.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                   draw_stage, draw_access, copy_stage, VK_ACCESS_2_TRANSFER_READ_BIT);
        copy(command, mirror.image, destination, mirror.size);
        transition(command, mirror.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                   copy_stage, VK_ACCESS_2_TRANSFER_READ_BIT, all, memory);
        return true;
    }

    void TrackedHands::record(const VkCommandBuffer command, const std::size_t eye, const std::uint32_t image, const EyePose& pose) {
        auto& p = *impl_;
        if (!p.renderer.hasHands() || image >= p.eye_views[eye].size())
            return;
        p.draw(command, p.eye_views[eye][image], p.eye_sizes[eye], pose);
    }

    std::optional<TrackedHands::Mirror> TrackedHands::mirror() {
        auto& mirror = impl_->mirror;
        if (!mirror.drawn)
            return std::nullopt;
        mirror.shown = true;
        return Mirror{mirror.image, mirror.display_view, mirror.size};
    }

} // namespace lfs::vis::vr

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "vr_hands.hpp"
#include "vr_math.hpp"
#include "window/vulkan_context.hpp"
#include <array>
#include <cstddef>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <memory>
#include <optional>
#include <span>

#include <openxr/openxr.h>

namespace lfs::vis::vr {

    // The hands of a VR session: tracked by the runtime (XR_EXT_hand_tracking),
    // skinned from the bundled models and drawn over the eye images. The left
    // eye with the hands on it also goes to an image of their own, for the
    // desktop mirror: people watching the window see the hands as well.
    //
    // The hands are an extra: when anything here fails, a warning is logged
    // and the session goes on without them.
    class TrackedHands {
    public:
        // An eye of the session: the images of its swapchain and their size.
        struct Eye {
            std::span<const VkImage> images;
            glm::ivec2 size{0};
        };
        // Where an eye is, how it is turned and what it sees, for a frame.
        struct EyePose {
            EyeFov fov{};
            glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f};
            glm::vec3 position{0.0f}; // in the tracking space
        };
        // A hand as the runtime tracks it: where its joints are, and whether
        // it is bare: tracked as it is, not posed by the runtime around the
        // controller it holds.
        struct Hand {
            HandJoints joints{};
            bool bare = false;
        };
        // The left eye with the hands on it, in SHADER_READ_ONLY_OPTIMAL.
        struct Mirror {
            VkImage image = VK_NULL_HANDLE;
            VkImageView view = VK_NULL_HANDLE; // the eye images' own encoding
            glm::ivec2 size{0};
        };

        TrackedHands();
        ~TrackedHands();
        TrackedHands(const TrackedHands&) = delete;
        TrackedHands& operator=(const TrackedHands&) = delete;

        // Looks for the hand tracking of the runtime, once its instance exists
        // with XR_EXT_hand_tracking enabled. data_sources tells whether
        // XR_EXT_hand_tracking_data_source is enabled too. A null instance
        // forgets the tracking: the runtime does not offer it.
        void findTracking(XrInstance instance, XrSystemId system, bool data_sources);

        // Starts the hands of a session whose eye images exist.
        void start(VulkanContext& context, XrSession session, const std::array<Eye, 2>& eyes);
        // Ends them. The GPU must be done with the last frame's commands; the
        // session, which owns the hand trackers, is destroyed afterwards.
        void stop();

        // Locates the hands in `space` for the frame shown at `time`.
        void locate(XrSpace space, XrTime time);
        // The left (0) or right (1) hand as located for the frame; nothing
        // while the runtime does not track all of its joints. No hand is bare
        // to a runtime that does not say where the joints come from.
        [[nodiscard]] const std::optional<Hand>& hand(std::size_t index) const;

        // Builds the hands of the frame shown at `time` from where locate()
        // found them. A gripping hand takes the color of a hand that holds the
        // scene. Without `show`, no hand is drawn.
        void update(XrTime time, const std::array<bool, 2>& gripping, bool show);

        // Records the left eye passing through the mirror image: `source`, in
        // TRANSFER_SRC_OPTIMAL, is copied to the mirror image, the hands are
        // drawn on it and it is copied to `destination`, an image of the left
        // eye in TRANSFER_DST_OPTIMAL. Both have the size of that eye. Returns
        // false with nothing recorded when there is no hand to draw or no
        // mirror image: the eye is then copied as the right one is.
        [[nodiscard]] bool recordMirroredEye(VkCommandBuffer command, VkImage source, VkImage destination,
                                             const EyePose& pose);
        // Records the hands straight onto image `image` of the swapchain of
        // eye `eye` (0 left, 1 right), in COLOR_ATTACHMENT_OPTIMAL.
        void record(VkCommandBuffer command, std::size_t eye, std::uint32_t image, const EyePose& pose);

        // The mirror image, when recordMirroredEye() recorded this frame's left
        // eye onto it and those commands have run. The desktop may read it
        // from then on, until stop().
        [[nodiscard]] std::optional<Mirror> mirror();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };

} // namespace lfs::vis::vr

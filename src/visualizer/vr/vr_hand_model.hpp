/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "io/skinned_mesh.hpp"
#include "vr_hands.hpp"
#include "vr_math.hpp"
#include <array>
#include <cstdint>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace lfs::vis::vr {

    // A joint of a hand: where it is and how it is turned.
    struct HandJointPose {
        glm::vec3 position{0.0f};
        glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f};
    };
    using HandPose = std::array<HandJointPose, kHandJointCount>;

    namespace hand_detail {
        [[nodiscard]] inline glm::vec3 direction(const glm::vec3 vector, const glm::vec3 fallback) {
            const float length = glm::length(vector);
            return length > 1e-6f ? vector / length : fallback;
        }

        // The frame of a bone: -Z along it, X as near `side` as a right angle
        // to the bone allows.
        [[nodiscard]] inline glm::quat boneFrame(const glm::vec3 along, const glm::vec3 side) {
            const glm::vec3 z = -along;
            glm::vec3 x = side - z * glm::dot(side, z);
            if (glm::dot(x, x) < 1e-8f) {
                glm::vec3 unused;
                basis(z, x, unused);
            }
            x = glm::normalize(x);
            return glm::normalize(glm::quat_cast(glm::mat3(x, glm::cross(z, x), z)));
        }
    } // namespace hand_detail

    // The pose of a hand worked out from where its joints are, and from that
    // alone. Runtimes report an orientation for each joint too, but not one to
    // build on: with camera hand tracking, SteamVR's wrist orientation jumps
    // by tens of degrees from one frame to the next while thumb and index
    // touch, and the joints themselves stay put. Each bone points at the next
    // joint; what a bone cannot tell, how it is rolled about itself, comes
    // from the two directions a hand keeps whatever its fingers do: across
    // the knuckles for the fingers and the wrist, out of the palm for the
    // thumb, which lies across the hand.
    [[nodiscard]] inline HandPose handPoseFromJoints(const HandJoints& joints) {
        using namespace hand_detail;
        HandPose pose;
        for (std::size_t joint = 0; joint < joints.size(); ++joint)
            pose[joint].position = joints[joint];
        const glm::vec3 wrist = joints[kWrist];
        const glm::vec3 index = joints[kFingers[0] + 1];
        const glm::vec3 middle = joints[kFingers[1] + 1];
        const glm::vec3 little = joints[kFingers[3] + 1];
        const glm::vec3 across = direction(index - little, glm::vec3(1.0f, 0.0f, 0.0f));
        const glm::vec3 out = direction(glm::cross(index - wrist, little - wrist), glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::vec3 ahead = direction(middle - wrist, glm::vec3(0.0f, 0.0f, -1.0f));
        pose[kWrist].orientation = boneFrame(ahead, across);
        pose[0].orientation = pose[kWrist].orientation;
        const auto chain = [&](const std::size_t first, const std::size_t count, const glm::vec3 side) {
            glm::vec3 along = ahead;
            for (std::size_t joint = first; joint < first + count; ++joint) {
                // A fingertip has no next joint: it goes on as its bone came.
                if (joint + 1 < first + count)
                    along = direction(joints[joint + 1] - joints[joint], along);
                pose[joint].orientation = boneFrame(along, side);
            }
        };
        chain(kThumb, 4, out);
        for (const std::size_t finger : kFingers)
            chain(finger, 5, across);
        return pose;
    }

    // The hand as it holds the scene, for a runtime that gives a bare hand no
    // grip pose: at the palm, and turned as OpenXR turns the grip of a
    // controller. -Z runs along the knuckles from the little finger to the
    // index, +X goes through the palm, into that of a right hand and out of
    // that of a left one. Two hands held alike then agree on their axes, as
    // two controllers do, which a grip with both hands turns the scene by;
    // the frame of the wrist joint would not do, mirrored as it is from one
    // hand to the other.
    [[nodiscard]] inline GrabHand grabHandFromJoints(const HandJoints& joints) {
        using namespace hand_detail;
        const glm::vec3 wrist = joints[kWrist];
        const glm::vec3 index = joints[kFingers[0] + 1];
        const glm::vec3 little = joints[kFingers[3] + 1];
        const glm::vec3 z = direction(little - index, glm::vec3(0.0f, 0.0f, 1.0f));
        const glm::vec3 through = glm::cross(little - wrist, index - wrist);
        glm::vec3 x = through - z * glm::dot(through, z);
        if (glm::dot(x, x) < 1e-12f) {
            glm::vec3 unused;
            basis(z, x, unused);
        }
        x = glm::normalize(x);
        return {joints[0], glm::normalize(glm::quat_cast(glm::mat3(x, glm::cross(z, x), z)))};
    }

    // How far a hand is closed into a fist: 0 with the fingers straight, 1
    // with the four fingertips folded back onto the palm. Each finger tells
    // how near its tip has come to its knuckle, as a share of its length, so
    // the size of the hand does not matter, and the fist is as closed as its
    // fingers are on average. The thumb does not count: it lies over the
    // fingers or beside them. Thumb and index brought together, which
    // runtimes take for a grip, leave the other fingers open: far from a fist.
    [[nodiscard]] inline float fistClosure(const HandJoints& joints) {
        using namespace hand_detail;
        // Tip to knuckle over the length of the finger: straight, and folded.
        constexpr float kStraight = 0.95f;
        constexpr float kFolded = 0.40f;
        float closure = 0.0f;
        for (const std::size_t finger : kFingers) {
            const std::size_t knuckle = finger + 1;
            float length = 0.0f;
            for (std::size_t joint = knuckle; joint < knuckle + 3; ++joint)
                length += glm::length(joints[joint + 1] - joints[joint]);
            if (!(length > 1e-4f))
                return 0.0f;
            const float reach = glm::length(joints[knuckle + 3] - joints[knuckle]) / length;
            closure += glm::clamp((kStraight - reach) / (kStraight - kFolded), 0.0f, 1.0f);
        }
        return closure / static_cast<float>(kFingers.size());
    }

    // Steadies a tracked hand before it is drawn: camera hand tracking shakes
    // when it loses sight of a finger. The wrist is steadied where it is;
    // every other joint is steadied in the wrist's frame, where it moves only
    // when the fingers do, so the hand keeps its shape.
    class HandPoseSmoother {
    public:
        // Cutoffs at rest, in Hz: fingers move faster than the wrist does.
        static constexpr float kWristCutoff = 2.0f;
        static constexpr float kFingerCutoff = 4.0f;

        HandPoseSmoother() { fingers_.fill(HandSmoother(kFingerCutoff)); }

        // dt in seconds; the first pose after a reset passes through.
        [[nodiscard]] HandPose update(const HandPose& raw, const float dt) {
            const HandJointPose& tracked = raw[hand_detail::kWrist];
            const GrabHand wrist_raw{tracked.position, glm::normalize(tracked.orientation)};
            const GrabHand wrist = wrist_.update(wrist_raw, dt);
            const glm::quat to_wrist = glm::inverse(wrist_raw.orientation);
            HandPose steady;
            for (std::size_t joint = 0; joint < raw.size(); ++joint) {
                GrabHand local{to_wrist * (raw[joint].position - wrist_raw.position),
                               to_wrist * glm::normalize(raw[joint].orientation)};
                if (joint != hand_detail::kWrist)
                    local = fingers_[joint].update(local, dt);
                steady[joint] = {wrist.position + wrist.orientation * local.position,
                                 glm::normalize(wrist.orientation * local.orientation)};
            }
            return steady;
        }

        void reset() {
            wrist_.reset();
            for (HandSmoother& finger : fingers_)
                finger.reset();
        }

    private:
        HandSmoother wrist_{kWristCutoff};
        std::array<HandSmoother, kHandJointCount> fingers_;
    };

    // A hand mesh with a skeleton, in the layout of the WebXR hand models: one
    // bone per hand joint, named after it. OpenXR and WebXR define the same
    // joints, so each bone goes where its joint is tracked.
    struct HandModel {
        using Vertex = lfs::io::SkinnedMesh::Vertex;
        std::vector<Vertex> vertices;
        std::vector<std::uint32_t> indices; // triangles
        // Per bone: the joint that moves it, the matrix that takes a vertex of
        // the rest pose into the bone's frame, and how the bone is turned in
        // the frame handPoseFromJoints() gives its joint.
        std::vector<std::size_t> joints;
        std::vector<glm::mat4> inverse_bind;
        std::vector<glm::quat> turn;
    };

    namespace hand_detail {
        // The WebXR names of the joints, in the order of XR_EXT_hand_tracking
        // from the wrist on; the palm has no bone.
        inline constexpr std::array<std::string_view, kHandJointCount - 1> kBoneNames = {
            "wrist",
            "thumb-metacarpal",
            "thumb-phalanx-proximal",
            "thumb-phalanx-distal",
            "thumb-tip",
            "index-finger-metacarpal",
            "index-finger-phalanx-proximal",
            "index-finger-phalanx-intermediate",
            "index-finger-phalanx-distal",
            "index-finger-tip",
            "middle-finger-metacarpal",
            "middle-finger-phalanx-proximal",
            "middle-finger-phalanx-intermediate",
            "middle-finger-phalanx-distal",
            "middle-finger-tip",
            "ring-finger-metacarpal",
            "ring-finger-phalanx-proximal",
            "ring-finger-phalanx-intermediate",
            "ring-finger-phalanx-distal",
            "ring-finger-tip",
            "pinky-finger-metacarpal",
            "pinky-finger-phalanx-proximal",
            "pinky-finger-phalanx-intermediate",
            "pinky-finger-phalanx-distal",
            "pinky-finger-tip",
        };
    } // namespace hand_detail

    // Makes a hand model of a skinned mesh in that layout, as
    // lfs::io::load_skinned_mesh() reads the bundled ones. Nothing is returned
    // for a mesh that lacks a bone for any of the hand joints.
    [[nodiscard]] inline std::optional<HandModel> handModelFromSkinnedMesh(lfs::io::SkinnedMesh mesh) {
        using namespace hand_detail;
        if (mesh.indices.empty() || mesh.indices.size() % 3 != 0)
            return std::nullopt;

        HandModel model;
        // Where each joint is, and how its bone is turned, in the rest pose.
        HandJoints rest_joints{};
        std::array<bool, kHandJointCount> found{};
        std::vector<glm::quat> rest_turn;
        for (const lfs::io::SkinnedMesh::Bone& bone : mesh.bones) {
            const glm::mat4 rest = glm::inverse(bone.inverse_bind);
            glm::mat3 turned(rest);
            for (int column = 0; column < 3; ++column)
                turned[column] = direction(turned[column], glm::vec3(0.0f));
            // A bone that is no joint, a root for instance, follows the wrist.
            std::size_t joint = kWrist;
            for (std::size_t i = 0; i < kBoneNames.size(); ++i) {
                if (bone.name == kBoneNames[i]) {
                    joint = i + 1;
                    found[joint] = true;
                    rest_joints[joint] = glm::vec3(rest[3]);
                }
            }
            model.joints.push_back(joint);
            model.inverse_bind.push_back(bone.inverse_bind);
            rest_turn.push_back(glm::normalize(glm::quat_cast(turned)));
        }
        for (std::size_t joint = kWrist; joint < kHandJointCount; ++joint) {
            if (!found[joint])
                return std::nullopt;
        }
        rest_joints[0] = (rest_joints[kWrist] + rest_joints[kFingers[1] + 1]) * 0.5f;
        const HandPose rest_pose = handPoseFromJoints(rest_joints);
        for (std::size_t bone = 0; bone < model.joints.size(); ++bone)
            model.turn.push_back(glm::normalize(glm::inverse(rest_pose[model.joints[bone]].orientation) * rest_turn[bone]));

        for (const HandModel::Vertex& vertex : mesh.vertices) {
            for (const std::uint16_t bone : vertex.bones) {
                if (bone >= model.joints.size())
                    return std::nullopt;
            }
        }
        for (const std::uint32_t index : mesh.indices) {
            if (index >= mesh.vertices.size())
                return std::nullopt;
        }
        model.vertices = std::move(mesh.vertices);
        model.indices = std::move(mesh.indices);
        return model;
    }

    // Appends the triangles of the model, its bones moved to the joints of a
    // pose from handPoseFromJoints(), steadied or not. `scratch` only saves an
    // allocation from one call to the next.
    inline void appendSkinnedHand(std::vector<HandVertex>& out, const HandModel& model, const HandPose& pose,
                                  std::vector<HandVertex>& scratch) {
        std::vector<glm::mat4> bones(model.joints.size());
        for (std::size_t bone = 0; bone < bones.size(); ++bone) {
            const HandJointPose& joint = pose[model.joints[bone]];
            glm::mat4 placed = glm::mat4_cast(glm::normalize(joint.orientation * model.turn[bone]));
            placed[3] = glm::vec4(joint.position, 1.0f);
            bones[bone] = placed * model.inverse_bind[bone];
        }
        scratch.resize(model.vertices.size());
        for (std::size_t i = 0; i < scratch.size(); ++i) {
            const HandModel::Vertex& vertex = model.vertices[i];
            glm::mat4 blended(0.0f);
            for (std::size_t part = 0; part < 4; ++part) {
                if (vertex.weights[part] > 0.0f)
                    blended += bones[vertex.bones[part]] * vertex.weights[part];
            }
            scratch[i].position = glm::vec3(blended * glm::vec4(vertex.position, 1.0f));
            const glm::vec3 normal = glm::mat3(blended) * vertex.normal;
            const float length = glm::length(normal);
            scratch[i].normal = length > 1e-6f ? normal / length : glm::vec3(0.0f, 1.0f, 0.0f);
        }
        out.reserve(out.size() + model.indices.size());
        for (const std::uint32_t index : model.indices)
            out.push_back(scratch[index]);
    }

} // namespace lfs::vis::vr

/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later */

#pragma once

#include "vr_hands.hpp"
#include "vr_math.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <string>
#include <string_view>
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
        struct Vertex {
            glm::vec3 position{0.0f};
            glm::vec3 normal{0.0f};
            std::array<std::uint16_t, 4> bones{};
            std::array<float, 4> weights{};
        };
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

        // Elements of a glTF accessor inside the binary chunk.
        struct Accessor {
            const std::uint8_t* data = nullptr;
            std::size_t stride = 0;
            std::size_t count = 0;
            int component = 0; // 5121 unsigned byte, 5123 unsigned short, 5125 unsigned int, 5126 float
            bool normalized = false;

            [[nodiscard]] double number(const std::size_t element, const std::size_t part) const {
                const std::uint8_t* const at = data + element * stride;
                switch (component) {
                case 5121:
                    return at[part];
                case 5123: {
                    std::uint16_t value;
                    std::memcpy(&value, at + part * 2, 2);
                    return value;
                }
                case 5125: {
                    std::uint32_t value;
                    std::memcpy(&value, at + part * 4, 4);
                    return value;
                }
                default: {
                    float value;
                    std::memcpy(&value, at + part * 4, 4);
                    return value;
                }
                }
            }

            // Integers that stand for 0..1 are scaled back to it.
            [[nodiscard]] float real(const std::size_t element, const std::size_t part) const {
                const double value = number(element, part);
                if (!normalized || component == 5126)
                    return static_cast<float>(value);
                return static_cast<float>(value / (component == 5121 ? 255.0 : component == 5123 ? 65535.0
                                                                                                 : 4294967295.0));
            }
        };

        [[nodiscard]] inline std::optional<Accessor> accessor(const nlohmann::json& document,
                                                              const std::span<const std::uint8_t> binary,
                                                              const std::size_t index, const std::string_view type,
                                                              const std::size_t parts) {
            const auto& described = document.at("accessors").at(index);
            if (described.at("type").get<std::string>() != type || !described.contains("bufferView"))
                return std::nullopt;
            const auto& view = document.at("bufferViews").at(described.at("bufferView").get<std::size_t>());
            Accessor result;
            result.component = described.at("componentType").get<int>();
            result.normalized = described.value("normalized", false);
            result.count = described.at("count").get<std::size_t>();
            const std::size_t width = result.component == 5121 ? 1 : result.component == 5123 ? 2
                                                                                              : 4;
            if (result.component != 5121 && result.component != 5123 && result.component != 5125 && result.component != 5126)
                return std::nullopt;
            const std::size_t element = width * parts;
            result.stride = view.value("byteStride", std::size_t{0});
            if (result.stride == 0)
                result.stride = element;
            const std::size_t offset = view.value("byteOffset", std::size_t{0}) + described.value("byteOffset", std::size_t{0});
            if (result.count == 0 || result.stride < element ||
                offset + (result.count - 1) * result.stride + element > binary.size())
                return std::nullopt;
            result.data = binary.data() + offset;
            return result;
        }
    } // namespace hand_detail

    // Reads a hand model from a binary glTF file (.glb): the first mesh that
    // has a skin, with that skin. Nothing is returned for a file that is not
    // one, or that lacks a bone for any of the hand joints.
    [[nodiscard]] inline std::optional<HandModel> loadHandModel(const std::span<const std::uint8_t> file) {
        using namespace hand_detail;
        const auto word = [&](const std::size_t offset) {
            std::uint32_t value = 0;
            if (offset + 4 <= file.size())
                std::memcpy(&value, file.data() + offset, 4);
            return value;
        };
        constexpr std::uint32_t kGltf = 0x46546C67, kJson = 0x4E4F534A, kBinary = 0x004E4942;
        if (word(0) != kGltf || word(4) != 2 || word(16) != kJson)
            return std::nullopt;
        const std::size_t json_size = word(12);
        const std::size_t binary_header = 20 + json_size;
        if (binary_header + 8 > file.size() || word(binary_header + 4) != kBinary)
            return std::nullopt;
        const std::size_t binary_size = word(binary_header);
        if (binary_header + 8 + binary_size > file.size())
            return std::nullopt;
        const std::span<const std::uint8_t> binary = file.subspan(binary_header + 8, binary_size);

        try {
            const auto document = nlohmann::json::parse(file.data() + 20, file.data() + 20 + json_size);
            const auto& nodes = document.at("nodes");
            const nlohmann::json* skinned = nullptr;
            for (const auto& node : nodes) {
                if (node.contains("mesh") && node.contains("skin")) {
                    skinned = &node;
                    break;
                }
            }
            if (!skinned)
                return std::nullopt;
            const auto& skin = document.at("skins").at(skinned->at("skin").get<std::size_t>());
            const auto& primitive =
                document.at("meshes").at(skinned->at("mesh").get<std::size_t>()).at("primitives").at(0);
            if (primitive.value("mode", 4) != 4)
                return std::nullopt;
            const auto& attributes = primitive.at("attributes");
            const auto read = [&](const nlohmann::json& index, const std::string_view type, const std::size_t parts) {
                return accessor(document, binary, index.get<std::size_t>(), type, parts);
            };
            const auto positions = read(attributes.at("POSITION"), "VEC3", 3);
            const auto normals = read(attributes.at("NORMAL"), "VEC3", 3);
            const auto bones = read(attributes.at("JOINTS_0"), "VEC4", 4);
            const auto weights = read(attributes.at("WEIGHTS_0"), "VEC4", 4);
            const auto indices = read(primitive.at("indices"), "SCALAR", 1);
            const auto binds = read(skin.at("inverseBindMatrices"), "MAT4", 16);
            const auto& bone_nodes = skin.at("joints");
            if (!positions || !normals || !bones || !weights || !indices || !binds ||
                normals->count != positions->count || bones->count != positions->count ||
                weights->count != positions->count || binds->count != bone_nodes.size() || indices->count % 3 != 0)
                return std::nullopt;

            HandModel model;
            // Where each joint is, and how its bone is turned, in the rest pose.
            HandJoints rest_joints{};
            std::array<bool, kHandJointCount> found{};
            std::vector<glm::quat> rest_turn;
            for (std::size_t bone = 0; bone < bone_nodes.size(); ++bone) {
                const std::string name = nodes.at(bone_nodes[bone].get<std::size_t>()).value("name", std::string{});
                glm::mat4 matrix;
                for (int column = 0; column < 4; ++column) {
                    for (int row = 0; row < 4; ++row)
                        matrix[column][row] = binds->real(bone, static_cast<std::size_t>(column * 4 + row));
                }
                const glm::mat4 rest = glm::inverse(matrix);
                glm::mat3 turned(rest);
                for (int column = 0; column < 3; ++column)
                    turned[column] = direction(turned[column], glm::vec3(0.0f));
                // A bone that is no joint, a root for instance, follows the wrist.
                std::size_t joint = kWrist;
                for (std::size_t i = 0; i < kBoneNames.size(); ++i) {
                    if (name == kBoneNames[i]) {
                        joint = i + 1;
                        found[joint] = true;
                        rest_joints[joint] = glm::vec3(rest[3]);
                    }
                }
                model.joints.push_back(joint);
                model.inverse_bind.push_back(matrix);
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

            model.vertices.resize(positions->count);
            for (std::size_t i = 0; i < positions->count; ++i) {
                auto& vertex = model.vertices[i];
                float total = 0.0f;
                for (std::size_t part = 0; part < 3; ++part) {
                    vertex.position[static_cast<int>(part)] = positions->real(i, part);
                    vertex.normal[static_cast<int>(part)] = normals->real(i, part);
                }
                for (std::size_t part = 0; part < 4; ++part) {
                    const double bone = bones->number(i, part);
                    if (!(bone >= 0.0 && bone < static_cast<double>(model.joints.size())))
                        return std::nullopt;
                    vertex.bones[part] = static_cast<std::uint16_t>(bone);
                    vertex.weights[part] = weights->real(i, part);
                    total += vertex.weights[part];
                }
                if (!(total > 1e-6f))
                    return std::nullopt;
                for (float& weight : vertex.weights)
                    weight /= total;
            }
            model.indices.resize(indices->count);
            for (std::size_t i = 0; i < indices->count; ++i) {
                const double index = indices->number(i, 0);
                if (!(index >= 0.0 && index < static_cast<double>(model.vertices.size())))
                    return std::nullopt;
                model.indices[i] = static_cast<std::uint32_t>(index);
            }
            return model;
        } catch (const nlohmann::json::exception&) {
            return std::nullopt;
        }
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

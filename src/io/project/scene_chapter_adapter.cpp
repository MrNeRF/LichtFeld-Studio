/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 */

#include "io/scene_chapter_adapter.hpp"

#include "core/path_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <functional>
#include <glm/gtc/type_ptr.hpp>
#include <ranges>
#include <span>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace lfs::io::project {

    namespace {

        lfs::Error adapter_error(const lfs::ErrorCode code, std::string message,
                                 std::string detail,
                                 const std::optional<lfs::core::Uuid> node = {}) {
            lfs::SmallFields fields;
            if (node) {
                fields.add("node_uuid", node->to_string());
            }
            return lfs::make_error(lfs::ErrorInit{
                .code = code,
                .domain = lfs::ErrorDomain::IO,
                .severity = lfs::Severity::Error,
                .retryability = lfs::Retryability::NotRetryable,
                .operation_id = {},
                .user_message = std::move(message),
                .detail = std::move(detail),
                .detection = LFS_SOURCE_SITE_CURRENT(),
                .fields = std::move(fields),
                .native = std::nullopt,
            });
        }

        template <typename T>
        lfs::Result<T> fail(const lfs::ErrorCode code, std::string message,
                            std::string detail,
                            const std::optional<lfs::core::Uuid> node = {}) {
            if constexpr (std::same_as<T, void>) {
                return lfs::Result<void>::failure(adapter_error(
                    code, std::move(message), std::move(detail), node));
            } else {
                return adapter_error(
                    code, std::move(message), std::move(detail), node);
            }
        }

        std::optional<std::string_view> node_type_name(const lfs::core::NodeType type) {
            using enum lfs::core::NodeType;
            switch (type) {
            case SPLAT:
                return "splat";
            case POINTCLOUD:
                return "pointcloud";
            case GROUP:
                return "group";
            case CROPBOX:
                return "cropbox";
            case ELLIPSOID:
                return "ellipsoid";
            case DATASET:
                return "dataset";
            case CAMERA_GROUP:
                return "camera_group";
            case CAMERA:
                return "camera";
            case IMAGE_GROUP:
                return "image_group";
            case IMAGE:
                return "image";
            case MESH:
                return "mesh";
            case PLY_SEQUENCE:
                return "ply_sequence";
            case KEYFRAME_GROUP:
            case KEYFRAME:
                return std::nullopt;
            }
            return std::nullopt;
        }

        std::optional<lfs::core::NodeType> parse_node_type(
            const std::string_view value) {
            using enum lfs::core::NodeType;
            if (value == "splat") {
                return SPLAT;
            }
            if (value == "pointcloud") {
                return POINTCLOUD;
            }
            if (value == "group") {
                return GROUP;
            }
            if (value == "cropbox") {
                return CROPBOX;
            }
            if (value == "ellipsoid") {
                return ELLIPSOID;
            }
            if (value == "dataset") {
                return DATASET;
            }
            if (value == "camera_group") {
                return CAMERA_GROUP;
            }
            if (value == "camera") {
                return CAMERA;
            }
            if (value == "image_group") {
                return IMAGE_GROUP;
            }
            if (value == "image") {
                return IMAGE;
            }
            if (value == "mesh") {
                return MESH;
            }
            if (value == "ply_sequence") {
                return PLY_SEQUENCE;
            }
            return std::nullopt;
        }

        template <std::size_t N>
        std::vector<float> tensor_values(const lfs::core::Tensor& value,
                                         const std::string_view field,
                                         const lfs::core::Uuid& node_uuid) {
            assert(value.is_valid());
            assert(value.dtype() == lfs::core::DataType::Float32);
            assert(value.numel() == N);
            if (!value.is_valid() ||
                value.dtype() != lfs::core::DataType::Float32 ||
                value.numel() != N) {
                throw std::logic_error(std::format(
                    "Camera node {} {} must be float32 with {} values",
                    node_uuid.to_string(), field, N));
            }
            const auto cpu = value.cpu().contiguous();
            const float* data = cpu.ptr<float>();
            return std::vector<float>(data, data + N);
        }

        std::vector<float> tensor_values_bounded(
            const lfs::core::Tensor& value, const std::string_view field,
            const lfs::core::Uuid& node_uuid) {
            assert(value.is_valid());
            assert(value.dtype() == lfs::core::DataType::Float32);
            assert(value.ndim() == 1);
            if (!value.is_valid() ||
                value.dtype() != lfs::core::DataType::Float32 ||
                value.ndim() != 1 || value.numel() > 16) {
                throw std::logic_error(std::format(
                    "Camera node {} {} must be float32 [0..16]",
                    node_uuid.to_string(), field));
            }
            const auto cpu = value.cpu().contiguous();
            const float* data = cpu.ptr<float>();
            return std::vector<float>(data, data + cpu.numel());
        }

        CameraRecord capture_camera(const lfs::core::Camera& camera,
                                    const lfs::core::Uuid& node_uuid) {
            const auto rotation_values =
                tensor_values<9>(camera.R(), "R", node_uuid);
            const auto translation_values =
                tensor_values<3>(camera.T(), "T", node_uuid);
            CameraRecord result;
            result.uid = camera.uid();
            result.camera_id = camera.camera_id();
            std::ranges::copy(rotation_values, result.rotation.begin());
            std::ranges::copy(translation_values, result.translation.begin());
            result.focal_x = camera.focal_x();
            result.focal_y = camera.focal_y();
            result.center_x = camera.center_x();
            result.center_y = camera.center_y();
            result.radial_distortion = tensor_values_bounded(
                camera.radial_distortion(), "radial_distortion", node_uuid);
            result.tangential_distortion = tensor_values_bounded(
                camera.tangential_distortion(), "tangential_distortion", node_uuid);
            result.camera_model_type =
                static_cast<std::int32_t>(camera.camera_model_type());
            result.camera_width = camera.camera_width();
            result.camera_height = camera.camera_height();
            result.image_width = camera.image_width();
            result.image_height = camera.image_height();
            result.image_name = camera.image_name();
            result.image_path = lfs::core::path_to_utf8(camera.image_path());
            result.mask_path = lfs::core::path_to_utf8(camera.mask_path());
            result.depth_path = lfs::core::path_to_utf8(camera.depth_path());
            result.normal_path = lfs::core::path_to_utf8(camera.normal_path());
            result.has_alpha = camera.has_alpha();
            result.split =
                camera.split() == lfs::core::CameraSplit::Train ? "train" : "eval";
            return result;
        }

        lfs::Result<void> validate_binding(
            const lfs::core::SceneNode& node, const PayloadBinding& binding,
            const bool training_model) {
            if (binding.instance_uuid.is_nil() || binding.fourcc.size() != 4 ||
                binding.source_kind.empty()) {
                return fail<void>(
                    lfs::ErrorCode::InvalidArgument,
                    "A scene payload binding is invalid.",
                    "Payload binding has a null UUID, invalid fourcc, or empty source kind",
                    node.uuid);
            }
            if (binding.source_kind == "rad") {
                if (binding.fourcc != "REFS" || !binding.reference_uuid ||
                    *binding.reference_uuid != binding.instance_uuid) {
                    return fail<void>(
                        lfs::ErrorCode::FailedPrecondition,
                        "A live RAD node cannot be embedded.",
                        "RAD binding must be an external REFS instance", node.uuid);
                }
                if (node.payload_diverged) {
                    return fail<void>(
                        lfs::ErrorCode::FailedPrecondition,
                        "An edited live RAD node cannot be saved.",
                        "Bake the RAD node to an embedded resident splat before saving",
                        node.uuid);
                }
                return {};
            }
            if (training_model) {
                if (binding.fourcc != "CKPT") {
                    return fail<void>(
                        lfs::ErrorCode::FailedPrecondition,
                        "The training-model payload binding is invalid.",
                        "Training model state must bind to CKPT, never SPLT", node.uuid);
                }
                return {};
            }
            if (node.type == lfs::core::NodeType::SPLAT &&
                (binding.source_kind == "ply" || binding.source_kind == "spz" ||
                 binding.source_kind == "sog" ||
                 binding.source_kind == "generated" ||
                 binding.source_kind == "baked_rad") &&
                (binding.fourcc != "SPLT" ||
                 binding.instance_uuid != node.uuid)) {
                return fail<void>(
                    lfs::ErrorCode::FailedPrecondition,
                    "An imported splat node must be embedded.",
                    "PLY/SPZ/SOG/generated/baked RAD nodes require node-UUID SPLT binding",
                    node.uuid);
            }
            if (node.type == lfs::core::NodeType::POINTCLOUD &&
                (binding.fourcc != "PCLD" || binding.instance_uuid != node.uuid)) {
                return fail<void>(
                    lfs::ErrorCode::FailedPrecondition,
                    "A point-cloud node has the wrong payload binding.",
                    "Point clouds require a node-UUID PCLD binding", node.uuid);
            }
            if (node.type == lfs::core::NodeType::MESH &&
                (binding.fourcc != "MESH" || binding.instance_uuid != node.uuid)) {
                return fail<void>(
                    lfs::ErrorCode::FailedPrecondition,
                    "A mesh node has the wrong payload binding.",
                    "Meshes require a node-UUID MESH binding", node.uuid);
            }
            return {};
        }

        lfs::Result<SceneNodeRecord> capture_node(
            const lfs::core::SceneNode& node, const std::uint32_t child_order,
            const ScenePayloadBindings& payload_bindings,
            const lfs::core::Uuid& training_uuid) {
            const auto type = node_type_name(node.type);
            if (!type) {
                return fail<SceneNodeRecord>(
                    lfs::ErrorCode::InvalidArgument,
                    "A generated scene node cannot be written to SCNG.",
                    "KEYFRAME and KEYFRAME_GROUP nodes are regenerated from SEQR",
                    node.uuid);
            }
            SceneNodeRecord result{
                .uuid = node.uuid,
                .type = std::string(*type),
                .name = node.name,
                .parent_uuid = std::nullopt,
                .child_order = child_order,
                .visible = node.visible.get(),
                .locked = node.locked.get(),
                .training_enabled = node.training_enabled,
                .payload_diverged = node.payload_diverged,
                .georef_pose = std::nullopt,
                .payload = std::nullopt,
                .cropbox = std::nullopt,
                .ellipsoid = std::nullopt,
                .camera = std::nullopt,
            };
            std::copy_n(glm::value_ptr(node.local_transform.get()), 16,
                        result.local_transform.begin());
            if (node.georef_pose) {
                result.georef_pose = GeorefPose{
                    .rotation =
                        {node.georef_pose->rotation.w, node.georef_pose->rotation.x,
                         node.georef_pose->rotation.y, node.georef_pose->rotation.z},
                    .translation =
                        {node.georef_pose->translation.x,
                         node.georef_pose->translation.y,
                         node.georef_pose->translation.z},
                };
            }
            if (node.cropbox) {
                result.cropbox = CropBoxRecord{
                    .min = {node.cropbox->min.x, node.cropbox->min.y, node.cropbox->min.z},
                    .max = {node.cropbox->max.x, node.cropbox->max.y, node.cropbox->max.z},
                    .inverse = node.cropbox->inverse,
                    .enabled = node.cropbox->enabled,
                    .color = {node.cropbox->color.x, node.cropbox->color.y,
                              node.cropbox->color.z},
                    .line_width = node.cropbox->line_width,
                };
            }
            if (node.ellipsoid) {
                result.ellipsoid = EllipsoidRecord{
                    .radii = {node.ellipsoid->radii.x, node.ellipsoid->radii.y,
                              node.ellipsoid->radii.z},
                    .inverse = node.ellipsoid->inverse,
                    .enabled = node.ellipsoid->enabled,
                    .color = {node.ellipsoid->color.x, node.ellipsoid->color.y,
                              node.ellipsoid->color.z},
                    .line_width = node.ellipsoid->line_width,
                };
            }
            if (node.camera) {
                if (node.camera->has_in_memory_mask()) {
                    return fail<SceneNodeRecord>(
                        lfs::ErrorCode::FailedPrecondition,
                        "A camera with an in-memory mask cannot be saved yet.",
                        "Matrix U3 is unresolved; refusing to silently drop the mask",
                        node.uuid);
                }
                try {
                    result.camera = capture_camera(*node.camera, node.uuid);
                } catch (const std::exception& error) {
                    // LFS-CENSUS-OK(empty-catch): normalize legacy tensor exceptions at the chapter boundary.
                    return fail<SceneNodeRecord>(
                        lfs::ErrorCode::ContractViolation,
                        "A camera has an invalid tensor schema.",
                        error.what(), node.uuid);
                }
            }

            const bool requires_binding =
                node.type == lfs::core::NodeType::SPLAT ||
                node.type == lfs::core::NodeType::POINTCLOUD ||
                node.type == lfs::core::NodeType::MESH;
            const auto binding = payload_bindings.find(node.uuid);
            if (requires_binding && binding == payload_bindings.end()) {
                return fail<SceneNodeRecord>(
                    lfs::ErrorCode::FailedPrecondition,
                    "A geometry scene node has no project payload binding.",
                    std::format("{} node '{}' is missing its SPLT/PCLD/MESH/CKPT/REFS binding",
                                *type, node.name),
                    node.uuid);
            }
            if (binding != payload_bindings.end()) {
                const bool is_training = node.uuid == training_uuid;
                if (auto valid = validate_binding(node, binding->second, is_training);
                    !valid) {
                    return std::move(valid).error();
                }
                result.payload = binding->second;
            }
            return result;
        }

        lfs::core::Tensor float_tensor(const std::span<const float> values,
                                       const lfs::core::TensorShape& shape) {
            return lfs::core::Tensor::from_vector(
                std::vector<float>(values.begin(), values.end()), shape,
                lfs::core::Device::CPU);
        }

        std::shared_ptr<lfs::core::Camera> hydrate_camera(
            const CameraRecord& value) {
            auto camera = std::make_shared<lfs::core::Camera>(
                float_tensor(value.rotation, {3, 3}),
                float_tensor(value.translation, {3}),
                value.focal_x, value.focal_y, value.center_x, value.center_y,
                float_tensor(value.radial_distortion,
                             {value.radial_distortion.size()}),
                float_tensor(value.tangential_distortion,
                             {value.tangential_distortion.size()}),
                static_cast<lfs::core::CameraModelType>(value.camera_model_type),
                value.image_name, lfs::core::utf8_to_path(value.image_path),
                lfs::core::utf8_to_path(value.mask_path), value.camera_width,
                value.camera_height, value.uid, value.camera_id,
                lfs::core::utf8_to_path(value.depth_path),
                lfs::core::utf8_to_path(value.normal_path));
            camera->set_image_dimensions(value.image_width, value.image_height);
            camera->set_has_alpha(value.has_alpha);
            camera->set_split(value.split == "train" ? lfs::core::CameraSplit::Train
                                                     : lfs::core::CameraSplit::Eval);
            return camera;
        }

        struct StagedNode {
            SceneNodeRecord record;
            lfs::core::Scene::RestoreNodeDesc desc;
        };

    } // namespace

    lfs::Result<CapturedSceneGraphState> capture_scene_graph_state(
        const lfs::core::Scene& scene,
        const ScenePayloadBindings& payload_bindings,
        const std::span<const lfs::core::Uuid> omit_node_uuids) {
        CapturedSceneGraphState result;
        const lfs::core::Uuid training_uuid = scene.getTrainingModelNodeUuid();
        std::unordered_set<lfs::core::Uuid> omitted(
            omit_node_uuids.begin(), omit_node_uuids.end());
        result.training_model_uuid =
            training_uuid.is_nil() || omitted.contains(training_uuid)
                ? std::optional<lfs::core::Uuid>{}
                : std::optional<lfs::core::Uuid>{training_uuid};

        const auto all_nodes = scene.getNodes();
        result.nodes.reserve(all_nodes.size());
        std::unordered_set<lfs::core::NodeId> excluded;
        for (const lfs::core::SceneNode* node : all_nodes) {
            if (node->type == lfs::core::NodeType::KEYFRAME ||
                node->type == lfs::core::NodeType::KEYFRAME_GROUP ||
                omitted.contains(node->uuid)) {
                excluded.insert(node->id);
            }
        }
        const auto persisted = [&](const lfs::core::SceneNode* node) {
            return node != nullptr && !excluded.contains(node->id);
        };

        std::function<lfs::Result<void>(const lfs::core::SceneNode&,
                                        std::optional<lfs::core::Uuid>,
                                        std::uint32_t)>
            visit;
        visit = [&](const lfs::core::SceneNode& node,
                    const std::optional<lfs::core::Uuid> parent_uuid,
                    const std::uint32_t child_order) -> lfs::Result<void> {
            auto captured =
                capture_node(node, child_order, payload_bindings, training_uuid);
            if (!captured) {
                return lfs::Result<void>::failure(std::move(captured).error());
            }
            captured->parent_uuid = parent_uuid;
            result.nodes.push_back(std::move(*captured));
            std::uint32_t saved_order = 0;
            for (const lfs::core::NodeId child_id : node.children) {
                const lfs::core::SceneNode* child = scene.getNodeById(child_id);
                // Explicit null first so static analysis sees getNodeById checked
                // (persisted() also rejects null).
                if (!child || !persisted(child)) {
                    continue;
                }
                if (auto status = visit(*child, node.uuid, saved_order++); !status) {
                    return status;
                }
            }
            return {};
        };

        std::uint32_t root_order = 0;
        for (const lfs::core::SceneNode* node : all_nodes) {
            if (!persisted(node) || node->parent_id != lfs::core::NULL_NODE) {
                continue;
            }
            if (auto status = visit(*node, std::nullopt, root_order++); !status) {
                return std::move(status).error();
            }
        }
        return result;
    }

    lfs::Result<SceneGraphChapter>
    materialize_scene_graph_chapter(
        CapturedSceneGraphState state) {
        SceneGraphChapter result;
        if (auto status = result.set_training_model_uuid(
                state.training_model_uuid);
            !status) {
            return std::move(status).error();
        }
        for (const auto& node : state.nodes) {
            if (auto status = result.upsert_node(node);
                !status) {
                return std::move(status).error();
            }
        }
        if (auto valid = result.validate_hierarchy(); !valid) {
            return std::move(valid).error();
        }
        return result;
    }

    lfs::Result<SceneGraphChapter> capture_scene_graph(
        const lfs::core::Scene& scene,
        const ScenePayloadBindings& payload_bindings,
        const std::span<const lfs::core::Uuid> omit_node_uuids) {
        auto state = capture_scene_graph_state(
            scene, payload_bindings, omit_node_uuids);
        if (!state) {
            return std::move(state).error();
        }
        return materialize_scene_graph_chapter(
            std::move(*state));
    }

    namespace {

        lfs::Result<void> populate_scene_stage(
            const SceneGraphChapter& chapter, lfs::core::Scene& scene,
            const ScenePayloadResolver& resolver,
            const bool defer_geometry_payloads) {
            auto nodes = chapter.nodes();
            if (!nodes) {
                return lfs::Result<void>::failure(std::move(nodes).error());
            }
            if (auto valid = chapter.validate_hierarchy(); !valid) {
                return valid;
            }
            auto training_uuid = chapter.training_model_uuid();
            if (!training_uuid) {
                return lfs::Result<void>::failure(std::move(training_uuid).error());
            }

            for (const SceneNodeRecord& record : *nodes) {
                if (*training_uuid && record.uuid == **training_uuid &&
                    record.type != "splat") {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "The training-model node is not a splat.",
                        std::format(
                            "SCNG training-model UUID {} has type '{}'",
                            record.uuid.to_string(), record.type),
                        record.uuid);
                }
            }

            std::vector<StagedNode> staged;
            staged.reserve(nodes->size());
            for (const SceneNodeRecord& record : *nodes) {
                const auto type = parse_node_type(record.type);
                if (!type) {
                    return fail<void>(
                        lfs::ErrorCode::Unsupported,
                        "This project contains an unsupported scene node type.",
                        std::format("SCNG node {} has unknown type '{}'",
                                    record.uuid.to_string(), record.type),
                        record.uuid);
                }
                const bool is_cropbox =
                    *type == lfs::core::NodeType::CROPBOX;
                const bool is_ellipsoid =
                    *type == lfs::core::NodeType::ELLIPSOID;
                const bool is_camera =
                    *type == lfs::core::NodeType::CAMERA;
                const bool is_geometry =
                    *type == lfs::core::NodeType::SPLAT ||
                    *type == lfs::core::NodeType::POINTCLOUD ||
                    *type == lfs::core::NodeType::MESH;
                if (record.cropbox.has_value() != is_cropbox ||
                    record.ellipsoid.has_value() != is_ellipsoid ||
                    record.camera.has_value() != is_camera ||
                    record.payload.has_value() != is_geometry) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "A scene node has mismatched type-specific state.",
                        std::format(
                            "SCNG node {} type '{}' has an absent or extraneous "
                            "payload/cropbox/ellipsoid/camera record",
                            record.uuid.to_string(), record.type),
                        record.uuid);
                }
                if (record.camera &&
                    (record.camera->camera_model_type <
                         static_cast<std::int32_t>(
                             lfs::core::CameraModelType::PINHOLE) ||
                     record.camera->camera_model_type >
                         static_cast<std::int32_t>(
                             lfs::core::CameraModelType::THIN_PRISM_FISHEYE))) {
                    return fail<void>(
                        lfs::ErrorCode::DataLoss,
                        "A saved camera model is unsupported.",
                        std::format("SCNG camera model {} is outside the v1 enum",
                                    record.camera->camera_model_type),
                        record.uuid);
                }
                lfs::core::Scene::RestoreNodeDesc desc{
                    .uuid = record.uuid,
                    .type = *type,
                    .name = record.name,
                    .parent = lfs::core::NULL_NODE,
                    .gaussian_count = 0,
                    .local_transform = glm::make_mat4(record.local_transform.data()),
                    .visible = record.visible,
                    .locked = record.locked,
                    .training_enabled = record.training_enabled,
                    .payload_diverged = record.payload_diverged,
                    .payload_hydration =
                        is_geometry
                            ? (defer_geometry_payloads
                                   ? lfs::core::PayloadHydrationState::Unloaded
                                   : lfs::core::PayloadHydrationState::Loaded)
                            : lfs::core::PayloadHydrationState::NotApplicable,
                    .georef_pose = std::nullopt,
                    .model = nullptr,
                    .point_cloud = nullptr,
                    .mesh = nullptr,
                    .cropbox = nullptr,
                    .ellipsoid = nullptr,
                    .camera = nullptr,
                    .keyframe = nullptr,
                };
                if (record.georef_pose) {
                    desc.georef_pose = lfs::core::GeoreferencePose{
                        .rotation =
                            glm::dquat(record.georef_pose->rotation[0],
                                       record.georef_pose->rotation[1],
                                       record.georef_pose->rotation[2],
                                       record.georef_pose->rotation[3]),
                        .translation =
                            glm::dvec3(record.georef_pose->translation[0],
                                       record.georef_pose->translation[1],
                                       record.georef_pose->translation[2]),
                    };
                }
                if (record.cropbox) {
                    desc.cropbox = std::make_unique<lfs::core::CropBoxData>(
                        lfs::core::CropBoxData{
                            .min = glm::vec3(record.cropbox->min[0],
                                             record.cropbox->min[1],
                                             record.cropbox->min[2]),
                            .max = glm::vec3(record.cropbox->max[0],
                                             record.cropbox->max[1],
                                             record.cropbox->max[2]),
                            .inverse = record.cropbox->inverse,
                            .enabled = record.cropbox->enabled,
                            .color = glm::vec3(record.cropbox->color[0],
                                               record.cropbox->color[1],
                                               record.cropbox->color[2]),
                            .line_width = record.cropbox->line_width,
                            .flash_intensity = 0.0f,
                        });
                }
                if (record.ellipsoid) {
                    desc.ellipsoid = std::make_unique<lfs::core::EllipsoidData>(
                        lfs::core::EllipsoidData{
                            .radii = glm::vec3(record.ellipsoid->radii[0],
                                               record.ellipsoid->radii[1],
                                               record.ellipsoid->radii[2]),
                            .inverse = record.ellipsoid->inverse,
                            .enabled = record.ellipsoid->enabled,
                            .color = glm::vec3(record.ellipsoid->color[0],
                                               record.ellipsoid->color[1],
                                               record.ellipsoid->color[2]),
                            .line_width = record.ellipsoid->line_width,
                            .flash_intensity = 0.0f,
                        });
                }
                if (record.camera) {
                    try {
                        desc.camera = hydrate_camera(*record.camera);
                    } catch (const std::exception& error) {
                        // LFS-CENSUS-OK(empty-catch): reject malformed camera tensors before mutating the scene.
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "A saved camera could not be constructed.", error.what(),
                            record.uuid);
                    }
                }
                if (defer_geometry_payloads && is_geometry) {
                    staged.push_back(StagedNode{record, std::move(desc)});
                    continue;
                }
                if (*type == lfs::core::NodeType::SPLAT) {
                    if (!record.payload || !resolver.splat) {
                        return fail<void>(
                            lfs::ErrorCode::FailedPrecondition,
                            "A saved splat payload cannot be resolved.",
                            "SPLAT/CKPT/REFS binding or resolver is missing", record.uuid);
                    }
                    auto payload = resolver.splat(*record.payload);
                    if (!payload) {
                        return lfs::Result<void>::failure(std::move(payload).error());
                    }
                    if (!*payload) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "A saved splat payload resolved to null.",
                            "SPLT/CKPT/REFS resolver returned a null model",
                            record.uuid);
                    }
                    desc.gaussian_count = (*payload)->size();
                    desc.model = std::move(*payload);
                } else if (*type == lfs::core::NodeType::POINTCLOUD) {
                    if (!record.payload || !resolver.point_cloud) {
                        return fail<void>(
                            lfs::ErrorCode::FailedPrecondition,
                            "A saved point-cloud payload cannot be resolved.",
                            "PCLD binding or resolver is missing", record.uuid);
                    }
                    auto payload = resolver.point_cloud(*record.payload);
                    if (!payload) {
                        return lfs::Result<void>::failure(std::move(payload).error());
                    }
                    if (!*payload) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "A saved point-cloud payload resolved to null.",
                            "PCLD resolver returned a null point cloud",
                            record.uuid);
                    }
                    desc.point_cloud = std::move(*payload);
                } else if (*type == lfs::core::NodeType::MESH) {
                    if (!record.payload || !resolver.mesh) {
                        return fail<void>(
                            lfs::ErrorCode::FailedPrecondition,
                            "A saved mesh payload cannot be resolved.",
                            "MESH binding or resolver is missing", record.uuid);
                    }
                    auto payload = resolver.mesh(*record.payload);
                    if (!payload) {
                        return lfs::Result<void>::failure(std::move(payload).error());
                    }
                    if (!*payload) {
                        return fail<void>(
                            lfs::ErrorCode::DataLoss,
                            "A saved mesh payload resolved to null.",
                            "MESH resolver returned a null mesh",
                            record.uuid);
                    }
                    desc.mesh = std::move(*payload);
                }
                staged.push_back(StagedNode{record, std::move(desc)});
            }

            // Contiguity was already checked by validate_hierarchy. Restore in
            // parent-first order with siblings sorted by child_order so a foreign
            // writer that shuffles the DOM array but keeps child_order is correct.
            std::unordered_map<lfs::core::Uuid, std::vector<std::size_t>>
                children_by_parent;
            std::vector<std::size_t> roots;
            roots.reserve(staged.size());
            children_by_parent.reserve(staged.size());
            for (std::size_t index = 0; index < staged.size(); ++index) {
                if (staged[index].record.parent_uuid) {
                    children_by_parent[*staged[index].record.parent_uuid]
                        .push_back(index);
                } else {
                    roots.push_back(index);
                }
            }
            const auto by_child_order =
                [&](const std::size_t lhs, const std::size_t rhs) {
                    return staged[lhs].record.child_order <
                           staged[rhs].record.child_order;
                };
            std::ranges::sort(roots, by_child_order);
            for (auto& [parent, children] : children_by_parent) {
                (void)parent;
                std::ranges::sort(children, by_child_order);
            }

            std::vector<std::size_t> restore_order;
            restore_order.reserve(staged.size());
            std::function<void(std::size_t)> append_subtree;
            append_subtree = [&](const std::size_t index) {
                restore_order.push_back(index);
                const auto children =
                    children_by_parent.find(staged[index].record.uuid);
                if (children == children_by_parent.end()) {
                    return;
                }
                for (const std::size_t child : children->second) {
                    append_subtree(child);
                }
            };
            for (const std::size_t root : roots) {
                append_subtree(root);
            }
            if (restore_order.size() != staged.size()) {
                return fail<void>(
                    lfs::ErrorCode::Internal,
                    "The staged scene hierarchy could not be ordered.",
                    "SCNG restore order is incomplete after child_order sort");
            }

            std::unordered_map<lfs::core::Uuid, lfs::core::NodeId> ids;
            ids.reserve(staged.size());
            for (const std::size_t index : restore_order) {
                StagedNode& node = staged[index];
                if (node.record.parent_uuid) {
                    const auto parent = ids.find(*node.record.parent_uuid);
                    if (parent == ids.end()) {
                        return fail<void>(
                            lfs::ErrorCode::Internal,
                            "The staged scene hierarchy lost its parent.",
                            "SCNG was validated parent-first but staged parent is absent",
                            node.record.uuid);
                    }
                    node.desc.parent = parent->second;
                }
                const lfs::core::NodeId id =
                    scene.restoreNodeWithUuid(std::move(node.desc));
                if (id == lfs::core::NULL_NODE) {
                    return fail<void>(
                        lfs::ErrorCode::Internal,
                        "A validated scene node could not be restored.",
                        std::format("Scene rejected SCNG node '{}'", node.record.name),
                        node.record.uuid);
                }
                ids.emplace(node.record.uuid, id);
            }
            if (*training_uuid) {
                scene.setTrainingModelNode(**training_uuid);
                if (scene.getTrainingModelNodeUuid() != **training_uuid) {
                    return fail<void>(
                        lfs::ErrorCode::Internal,
                        "The training-model node could not be designated.",
                        std::format("Scene rejected training-model UUID {}",
                                    (**training_uuid).to_string()),
                        **training_uuid);
                }
            }
            return {};
        }

    } // namespace

    lfs::Result<std::unique_ptr<lfs::core::Scene>>
    stage_scene_graph(
        const SceneGraphChapter& chapter,
        lfs::core::Scene& target,
        const ScenePayloadResolver& resolver) {
        auto staged =
            lfs::core::Scene::createRestoreStage(target);
        if (auto populated =
                populate_scene_stage(
                    chapter, *staged, resolver, false);
            !populated) {
            return std::move(populated).error();
        }
        return staged;
    }

    lfs::Result<std::unique_ptr<lfs::core::Scene>>
    stage_scene_shell(
        const SceneGraphChapter& chapter,
        lfs::core::Scene& target) {
        auto staged =
            lfs::core::Scene::createRestoreStage(target);
        if (auto populated =
                populate_scene_stage(
                    chapter, *staged, {}, true);
            !populated) {
            return std::move(populated).error();
        }
        return staged;
    }

    lfs::Result<void> hydrate_scene_graph(
        const SceneGraphChapter& chapter,
        lfs::core::Scene& scene,
        const ScenePayloadResolver& resolver) {
        auto staged =
            stage_scene_graph(chapter, scene, resolver);
        if (!staged) {
            return lfs::Result<void>::failure(
                std::move(staged).error());
        }
        scene.commitRestoreStage(std::move(*staged));
        return {};
    }

} // namespace lfs::io::project

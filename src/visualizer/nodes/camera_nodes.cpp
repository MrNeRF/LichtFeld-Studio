/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "visualizer/nodes/camera_nodes.hpp"
#include "core/nodes/builtin.hpp"
#include "core/scene.hpp"
#include "core/tensor_spatial.hpp"
#include <charconv>
#include <glm/gtc/matrix_inverse.hpp>
#include <numbers>

namespace lfs::vis {
    namespace {
        using namespace lfs::nodes;
        using core::Tensor;
        EvalHost& requireHost(NodeContext& context) {
            auto* host = context.host();
            if (!host || host->cameras().empty())
                throw NodeError("This node requires dataset cameras in the scene");
            return *host;
        }
        const EvaluationCamera& chosenCamera(NodeContext& context, EvalHost& host) {
            const auto name = context.properties().value("camera", std::string("0"));
            const auto cameras = host.cameras();
            for (const auto& camera : cameras)
                if (camera.name == name)
                    return camera;
            size_t index = 0;
            const auto parsed = std::from_chars(name.data(), name.data() + name.size(), index);
            if (parsed.ec == std::errc{} && parsed.ptr == name.data() + name.size() && index < cameras.size())
                return cameras[index];
            throw NodeError("The selected dataset camera no longer exists");
        }
        Tensor worldPositions(const FieldContext& domain, FieldMemo& memo, const glm::mat4& world) {
            const auto p = memo.evaluate(position_field(), domain);
            std::vector<float> linear(9), offset(3);
            for (int i = 0; i < 3; ++i) {
                offset[i] = world[3][i];
                for (int j = 0; j < 3; ++j)
                    linear[i * 3 + j] = world[i][j];
            }
            return p.matmul(Tensor::from_vector(linear, {3, 3}, domain.device())) + Tensor::from_vector(offset, {3}, domain.device());
        }
        SocketDecl output(const char* name, std::string_view type) { return {name, name, std::string(type)}; }
    } // namespace

    std::vector<lfs::nodes::EvaluationCamera> captureNodeCameras(const core::Scene& scene) {
        std::vector<lfs::nodes::EvaluationCamera> result;
        for (const auto& camera : scene.getAllCameras()) {
            // Camera metadata is tiny and captured once on the viewer thread.
            // Worker callbacks never read the live scene or camera tensors.
            const auto r = camera->R().cpu().contiguous(), t = camera->T().cpu().contiguous();
            const auto* rotation = r.ptr<float>();
            const auto* translation = t.ptr<float>();
            glm::mat4 view(1.0f);
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col)
                    view[col][row] = rotation[row * 3 + col];
                view[3][row] = translation[row];
            }
            if (const auto world = scene.getCameraSceneTransformByUid(camera->uid()))
                view = view * glm::inverse(*world);
            result.push_back({camera->image_name(), view, camera->focal_x(), camera->focal_y(), camera->center_x(), camera->center_y(), camera->camera_width(), camera->camera_height()});
        }
        return result;
    }

    void registerCameraNodes(lfs::nodes::NodeTypeRegistry& registry) {
        using namespace lfs::nodes;
        NodeTypeInfo info;
        info.id = "lfs.camera_info";
        info.category = "Input";
        info.uses_host = true;
        info.properties = {{"camera", "Camera", PropertyKind::String, "0"}};
        info.outputs = {output("Position", VECTOR_SOCKET), output("Direction", VECTOR_SOCKET), output("Up", VECTOR_SOCKET), output("FOV", FLOAT_SOCKET), output("Image Size", VECTOR_SOCKET)};
        info.evaluate = [](NodeContext& context) {
            auto& host = requireHost(context);
            const auto& camera = chosenCamera(context, host);
            const auto pose = glm::inverse(camera.world_to_camera);
            context.set_output("Position", glm::vec3(pose[3]));
            context.set_output("Direction", glm::normalize(glm::vec3(pose[2])));
            context.set_output("Up", glm::normalize(-glm::vec3(pose[1])));
            context.set_output("FOV", glm::degrees(2 * std::atan(float(camera.height) / (2 * camera.focal_y))));
            context.set_output("Image Size", glm::vec3(camera.width, camera.height, 0));
        };
        set_builtin_node_text(info);
        registry.register_type(std::move(info));

        NodeTypeInfo coverage;
        coverage.id = "lfs.camera_coverage";
        coverage.category = "Selection";
        coverage.uses_host = true;
        coverage.inputs = {SocketDecl{"Max Distance", "Max Distance", std::string(FLOAT_SOCKET), 0.0f}.minimum(0).step_size(0.1)};
        coverage.outputs = {output("Count", INT_SOCKET), output("Ratio", FLOAT_SOCKET)};
        coverage.evaluate = [](NodeContext& context) {
            auto& host = requireHost(context);
            const auto cameras = host.cameras();
            const auto world = host.object_to_world();
            const auto value = context.input("Max Distance").get_if<float>();
            const float maximum = value ? std::max(0.0f, *value) : 0;
            std::vector<float> packed;
            for (const auto& camera : cameras) {
                const auto pose = glm::inverse(camera.world_to_camera);
                for (int row = 0; row < 3; ++row)
                    for (int col = 0; col < 4; ++col) {
                        const float v = camera.world_to_camera[col][row];
                        packed.push_back(row == 0 ? (v * camera.focal_x + camera.world_to_camera[col][2] * camera.center_x) / camera.width : row == 1 ? (v * camera.focal_y + camera.world_to_camera[col][2] * camera.center_y) / camera.height
                                                                                                                                                      : v);
                    }
                packed.insert(packed.end(), {pose[3].x, pose[3].y, pose[3].z, 0});
            }
            const size_t count = cameras.size();
            const auto counts = Field(std::string(INT_SOCKET), [packed = std::move(packed), count, world, maximum](const FieldContext& domain, FieldMemo& memo) {
                return core::camera_frustum_counts(worldPositions(domain, memo, world), Tensor::from_vector(packed, {count, 16}, domain.device()), maximum);
            });
            context.set_output("Count", counts);
            context.set_output("Ratio", Field(std::string(FLOAT_SOCKET), [counts, count](const FieldContext& domain, FieldMemo& memo) {
                                   return memo.evaluate(counts, domain).to(core::DataType::Float32) / float(count);
                               }));
        };
        set_builtin_node_text(coverage);
        registry.register_type(std::move(coverage));

        NodeTypeInfo distance;
        distance.id = "lfs.view_distance";
        distance.category = "Input";
        distance.uses_host = true;
        distance.properties = {{"camera", "Camera", PropertyKind::String, "0"},
                               {"source", "Source", PropertyKind::Enum, "dataset", {"dataset", "captured"}},
                               {"captured_transform", "Captured Transform", PropertyKind::Data, nlohmann::json::array()}};
        distance.outputs = {output("Distance", FLOAT_SOCKET)};
        distance.evaluate = [](NodeContext& context) {
            auto& host = requireHost(context);
            glm::vec3 position;
            if (context.properties().value("source", std::string("dataset")) == "captured") {
                const auto saved = context.properties().value("captured_transform", nlohmann::json::array());
                if (!saved.is_array() || saved.size() != 16)
                    throw NodeError("Capture a viewport camera before using View Distance");
                position = {saved[12].get<float>(), saved[13].get<float>(), saved[14].get<float>()};
            } else
                position = glm::vec3(glm::inverse(chosenCamera(context, host).world_to_camera)[3]);
            const auto world = host.object_to_world();
            context.set_output("Distance", Field(std::string(FLOAT_SOCKET), [position, world](const FieldContext& domain, FieldMemo& memo) {
                                   const auto delta = worldPositions(domain, memo, world) - Tensor::from_vector({position.x, position.y, position.z}, {3}, domain.device());
                                   return (delta * delta).sum(1).sqrt();
                               }));
        };
        set_builtin_node_text(distance);
        registry.register_type(std::move(distance));
    }
} // namespace lfs::vis

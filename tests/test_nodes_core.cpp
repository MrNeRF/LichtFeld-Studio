/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/base64.hpp"
#include "core/nodes/nodes.hpp"
#include "core/splat_data_transform.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_fused.hpp"
#include "core/tensor_procedural.hpp"
#include "core/tensor_random.hpp"
#include "core/tensor_spatial.hpp"
#include "io/formats/ply.hpp"
#include "visualizer/nodes/camera_nodes.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <string_view>

namespace {

    using namespace lfs::nodes;
    using lfs::core::Device;
    using lfs::core::GpuBackend;
    using lfs::core::Tensor;
    using lfs::core::TensorShape;

    struct Target {
        const char* name;
        std::optional<GpuBackend> backend;
    };

    std::vector<Target> test_targets() {
        std::vector<Target> targets{{"CPU", std::nullopt},
                                    {"Metal", GpuBackend::Metal},
                                    {"Vulkan", GpuBackend::Vulkan}};
#if LFS_HAS_CUDA
        targets.push_back({"CUDA", GpuBackend::CUDA});
#endif
        return targets;
    }

    class NodesCore : public testing::TestWithParam<Target> {
    protected:
        void SetUp() override {
            if (GetParam().backend) {
                if (!lfs::core::gpu_backend_available(*GetParam().backend))
                    GTEST_SKIP() << GetParam().name << " backend unavailable";
                scope_.emplace(*GetParam().backend);
            }
            register_builtin_nodes(registry_);
        }

        Device device() const {
            return GetParam().backend ? Device::GPU : Device::CPU;
        }

        EvalResult evaluate(const NodeTree& tree, EvalInputs inputs, EvalHost* host = nullptr,
                            EvalCache* cache = nullptr, const EvalControl& control = {}) const {
            inputs.device = device();
            return lfs::nodes::evaluate(tree, std::move(inputs), host, cache, control);
        }

        Tensor tensor(std::vector<float> values, TensorShape shape) const {
            Tensor result = Tensor::from_vector(values, std::move(shape), Device::CPU);
            return device() == Device::CPU ? result : result.to(device());
        }

        Tensor ints(std::vector<int> values, TensorShape shape) const {
            Tensor result = Tensor::from_vector(values, std::move(shape), Device::CPU);
            return device() == Device::CPU ? result : result.to(device());
        }

        template <typename T>
        static std::vector<T> host(const Tensor& value) {
            Tensor cpu = value.device() == Device::CPU ? value : value.cpu();
            cpu = cpu.contiguous();
            std::vector<T> result(cpu.numel());
            if (!result.empty())
                std::memcpy(result.data(), cpu.data_ptr(), result.size() * sizeof(T));
            return result;
        }

        void expect_finite(const Geometry& geometry) const {
            const auto tensor_is_finite = [&](const Tensor& value, std::string_view name) {
                if (!value.is_valid())
                    return;
                for (const float element : host<float>(value.to(lfs::core::DataType::Float32)))
                    EXPECT_TRUE(std::isfinite(element)) << name << " contains " << element;
            };
            if (geometry.splats) {
                tensor_is_finite(geometry.splats->means, "splats.means");
                tensor_is_finite(geometry.splats->sh0, "splats.sh0");
                tensor_is_finite(geometry.splats->shN, "splats.shN");
                tensor_is_finite(geometry.splats->scaling, "splats.scaling");
                tensor_is_finite(geometry.splats->rotation, "splats.rotation");
                tensor_is_finite(geometry.splats->opacity, "splats.opacity");
                for (const auto& [name, value] : geometry.splats->attributes)
                    tensor_is_finite(value, "splats." + name);
            }
            if (geometry.points) {
                tensor_is_finite(geometry.points->positions, "points.positions");
                tensor_is_finite(geometry.points->colors, "points.colors");
                for (const auto& [name, value] : geometry.points->attributes)
                    tensor_is_finite(value, "points." + name);
            }
            if (geometry.mesh && geometry.mesh->mesh) {
                const auto& mesh = *geometry.mesh->mesh;
                tensor_is_finite(mesh.vertices, "mesh.vertices");
                tensor_is_finite(mesh.indices, "mesh.indices");
                tensor_is_finite(mesh.normals, "mesh.normals");
                tensor_is_finite(mesh.tangents, "mesh.tangents");
                tensor_is_finite(mesh.texcoords, "mesh.texcoords");
                tensor_is_finite(mesh.colors, "mesh.colors");
            }
        }

        Geometry splats(int degree = 1) const {
            const std::size_t coeff = static_cast<std::size_t>((degree + 1) * (degree + 1) - 1);
            SplatsComponent value;
            value.means = tensor({0, 0, 0, 1, 0, 0, 3, 0, 0}, {3, 3});
            value.sh0 = tensor({0, 0, 0, 0.1f, 0.2f, 0.3f, -0.1f, 0.1f, 0.2f}, {3, 3});
            std::vector<float> shn(3 * coeff * 3);
            for (std::size_t index = 0; index < shn.size(); ++index)
                shn[index] = 0.01f * static_cast<float>(index + 1);
            value.shN = tensor(std::move(shn), {3, coeff, 3});
            value.scaling = tensor(std::vector<float>(9, std::log(0.1f)), {3, 3});
            value.rotation = tensor({1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, {3, 4});
            value.opacity = tensor({0, 1, -1}, {3});
            value.sh_degree = degree;
            value.attributes["weight"] = tensor({10, 20, 30}, {3});
            return Geometry{std::move(value), std::nullopt, std::nullopt};
        }

        EvalResult single(std::string_view type_id, Geometry geometry,
                          std::function<void(Node&)> configure = {}) {
            NodeTree tree(registry_);
            Node& node = tree.add_node(std::string(type_id));
            if (configure)
                configure(node);
            EXPECT_TRUE(tree.add_link({tree.input_node().name, "Geometry", node.name, "Geometry"}));
            EXPECT_TRUE(tree.add_link({node.name, "Geometry", tree.output_node().name, "Geometry"}));
            return evaluate(tree, {std::move(geometry), {}, 1});
        }

        Tensor field_result(std::string_view id, std::string_view socket, std::string_view socket_type,
                            Geometry geometry, std::function<void(Node&)> configure = {}, EvalHost* eval_host = nullptr,
                            std::optional<Device> execution_device = {}, const bool round_trip = false) {
            Tensor value;
            NodeTypeInfo capture;
            capture.id = "test.capture";
            capture.inputs = {{"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)},
                              {"Value", "Value", std::string(socket_type), {}, {}, {}, {}, true}};
            capture.outputs = {{"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)}};
            capture.evaluate = [&](NodeContext& context) {
                const auto source = *context.input("Geometry").get_if<Geometry>();
                FieldContext domain{Domain::Splat, &*source.splats, nullptr, nullptr, 100};
                value = context.evaluate_field("Value", domain, socket_type);
                context.set_output("Geometry", source);
            };
            registry_.unregister_type(capture.id);
            registry_.register_type(std::move(capture));
            NodeTree tree(registry_);
            auto& node = tree.add_node(std::string(id), "Field");
            if (configure)
                configure(node);
            tree.add_node("test.capture", "Capture");
            EXPECT_TRUE(tree.add_link({"Field", std::string(socket), "Capture", "Value"}));
            EXPECT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Capture", "Geometry"}));
            EXPECT_TRUE(tree.add_link({"Capture", "Geometry", tree.output_node().name, "Geometry"}));
            if (round_trip)
                tree = NodeTree::from_json(tree.to_json(), registry_);
            const auto result = lfs::nodes::evaluate(tree, {geometry, {}, 1, execution_device.value_or(device())}, eval_host);
            if (!result.ok)
                for (const auto& [name, message] : result.errors)
                    std::cerr << name << ": " << message << '\n';
            EXPECT_TRUE(result.ok) << (result.errors.empty() ? "" : result.errors.begin()->second);
            registry_.unregister_type("test.capture");
            return value;
        }

        std::shared_ptr<lfs::core::MeshData> torus(int segments = 16, int sides = 8) const {
            std::vector<float> vertices;
            std::vector<int> faces;
            for (int ring = 0; ring < segments; ++ring) {
                const float u = 2 * std::numbers::pi_v<float> * ring / segments;
                for (int side = 0; side < sides; ++side) {
                    const float v = 2 * std::numbers::pi_v<float> * side / sides;
                    const float radius = 1.5f + 0.5f * std::cos(v);
                    vertices.insert(vertices.end(),
                                    {radius * std::cos(u), radius * std::sin(u), 0.5f * std::sin(v)});
                    const int a = ring * sides + side;
                    const int b = ((ring + 1) % segments) * sides + side;
                    const int c = ((ring + 1) % segments) * sides + (side + 1) % sides;
                    const int d = ring * sides + (side + 1) % sides;
                    faces.insert(faces.end(), {a, b, c, a, c, d});
                }
            }
            return std::make_shared<lfs::core::MeshData>(
                tensor(vertices, {static_cast<size_t>(segments * sides), 3}),
                ints(faces, {static_cast<size_t>(segments * sides * 2), 3}));
        }

        NodeTypeRegistry registry_;
        std::optional<lfs::core::GpuBackendScope> scope_;
    };

    TEST(NodesCoreMetadata, RegistriesExposeFrozenTypes) {
        SocketTypeRegistry sockets;
        NodeTypeRegistry nodes;
        register_builtin_nodes(nodes);
        EXPECT_EQ(sockets.list().size(), 8u);
        EXPECT_GE(nodes.list().size(), 45u);
        EXPECT_FALSE(nodes.find("lfs.object_info"));
        EXPECT_TRUE(nodes.find("lfs.mesh_to_splats"));
    }

    TEST(NodesCoreMetadata, BuiltinHelpIsCompleteWithoutUiInitialisation) {
        NodeTypeRegistry registry;
        register_builtin_nodes(registry);
        for (const auto& node : registry.list()) {
            SCOPED_TRACE(node->id);
            EXPECT_FALSE(node->description.empty());
            const auto characters = std::ranges::count_if(node->description, [](unsigned char c) {
                return (c & 0xc0) != 0x80;
            });
            EXPECT_LE(characters, 110);
            for (const auto* banned : {"log-scale", "geometric mean", "field", "tensor", "domain"})
                EXPECT_EQ(node->description.find(banned), std::string::npos);
            const auto lines = std::ranges::count(node->help, '\n') + 1;
            EXPECT_GE(lines, 2);
            EXPECT_LE(lines, 4);
            for (const auto& socket : node->inputs)
                EXPECT_FALSE(socket.description.empty()) << socket.identifier;
            for (const auto& socket : node->outputs)
                EXPECT_FALSE(socket.description.empty()) << socket.identifier;
            for (const auto& property : node->properties)
                EXPECT_FALSE(property.description.empty()) << property.identifier;
        }
        EXPECT_EQ(registry.find("lfs.scale_clamp")->description,
                  "Shortens needle-shaped Gaussians that show up as streaks when you move away from the capture path.");
    }

    TEST(NodesCoreMetadata, JsonRoundTripPreservesMissingNodeAndToleratesUnknownKeys) {
        NodeTypeRegistry registry;
        register_builtin_nodes(registry);
        NodeTree source(registry, "Round trip");
        auto json = source.to_json();
        json["unknown_tree_key"] = 42;
        json["nodes"][0]["type_id"] = "missing.plugin_node";
        json["nodes"][0]["plugin_payload"] = {1, 2, 3};
        json["nodes"][1]["unknown_node_key"] = true;
        NodeTree loaded = NodeTree::from_json(json, registry);
        const auto output = loaded.to_json();
        EXPECT_FALSE(output.contains("unknown_tree_key"));
        EXPECT_EQ(output["nodes"][0]["type_id"], "missing.plugin_node");
        EXPECT_EQ(output["nodes"][0]["plugin_payload"], nlohmann::json({1, 2, 3}));
        EXPECT_FALSE(output["nodes"][1].contains("unknown_node_key"));
    }

    TEST(NodesCoreMetadata, CycleRejectedAndSingleInputLinkReplaced) {
        NodeTypeRegistry registry;
        register_builtin_nodes(registry);
        NodeTree tree(registry);
        Node& a = tree.add_node("lfs.transform_geometry", "A");
        Node& b = tree.add_node("lfs.transform_geometry", "B");
        EXPECT_TRUE(tree.add_link({tree.input_node().name, "Geometry", a.name, "Geometry"}));
        EXPECT_TRUE(tree.add_link({a.name, "Geometry", b.name, "Geometry"}));
        EXPECT_TRUE(tree.add_link({b.name, "Geometry", tree.output_node().name, "Geometry"}));
        std::string error;
        EXPECT_FALSE(tree.add_link({b.name, "Geometry", a.name, "Geometry"}, &error));
        EXPECT_EQ(error, "Node graph contains a cycle");
        EXPECT_TRUE(tree.validate().empty());
        EXPECT_EQ(std::ranges::count_if(tree.links,
                                        [&](const Link& link) {
                                            return link.to_node == tree.output_node().name &&
                                                   link.to_socket == "Geometry";
                                        }),
                  1);
    }

    TEST_P(NodesCore, FieldMemoisationComputesExpressionOnce) {
        int calls = 0;
        Field field(std::string(FLOAT_SOCKET), [&](const FieldContext& context, FieldMemo&) {
            ++calls;
            return Tensor::ones({context.size()}, context.device());
        });
        Geometry geometry = splats();
        FieldContext context{Domain::Splat, &*geometry.splats, nullptr, nullptr, 55};
        FieldMemo memo;
        (void)field.evaluate(context, memo);
        (void)field.evaluate(context, memo);
        EXPECT_EQ(calls, 1);
    }

    TEST_P(NodesCore, ImplicitBoolToSelectionAndStructuralPropagation) {
        NodeTree tree(registry_);
        Node& flag = tree.add_node("lfs.boolean", "Flag");
        flag.input_values["Value"] = true;
        Node& remove = tree.add_node("lfs.delete_geometry", "Delete");
        EXPECT_TRUE(tree.add_link({tree.input_node().name, "Geometry", remove.name, "Geometry"}));
        EXPECT_TRUE(tree.add_link({flag.name, "Value", remove.name, "Selection"}));
        EXPECT_TRUE(tree.add_link({remove.name, "Geometry", tree.output_node().name, "Geometry"}));
        const auto result = evaluate(tree, {splats(), {}, 1});
        ASSERT_TRUE(result.ok) << result.errors.begin()->second;
        ASSERT_TRUE(result.geometry.splats);
        EXPECT_EQ(result.geometry.splats->means.shape()[0], 0u);
        EXPECT_EQ(result.geometry.splats->attributes.at("weight").shape()[0], 0u);
    }

    TEST_P(NodesCore, SetColourInvertAndShDegreeObeySHRules) {
        auto set = single("lfs.set_colour", splats(), [](Node& node) {
            node.input_values["Colour"] = glm::vec3(0.25f, 0.5f, 0.75f);
        });
        ASSERT_TRUE(set.ok);
        EXPECT_EQ(host<float>(set.geometry.splats->shN),
                  std::vector<float>(set.geometry.splats->shN.numel(), 0.0f));

        auto flat_set = single("lfs.set_colour", splats(0), [](Node& node) {
            node.input_values["Colour"] = glm::vec3(0.8f, 0.2f, 0.1f);
        });
        ASSERT_TRUE(flat_set.ok) << (flat_set.errors.empty() ? "" : flat_set.errors.begin()->second);
        EXPECT_EQ(flat_set.geometry.splats->shN.shape(), TensorShape({3, 0, 3}));
        EXPECT_EQ(flat_set.geometry.splats->shN.numel(), 0u);

        auto staged_model = splat_data_from_geometry(splats(3));
        ASSERT_TRUE(staged_model);
        staged_model->set_active_sh_degree(0);
        auto staged = geometry_from_splat_data(*staged_model);
        EXPECT_EQ(staged.splats->shN.shape(), TensorShape({3, 0, 3}));
        auto staged_set = single("lfs.set_colour", std::move(staged));
        ASSERT_TRUE(staged_set.ok) << (staged_set.errors.empty() ? "" : staged_set.errors.begin()->second);
        EXPECT_EQ(staged_set.geometry.splats->shN.shape(), TensorShape({3, 0, 3}));

        Geometry original = splats();
        auto inverted = single("lfs.invert_colour", original);
        ASSERT_TRUE(inverted.ok);
        const auto before = host<float>(original.splats->shN);
        const auto after = host<float>(inverted.geometry.splats->shN);
        ASSERT_EQ(before.size(), after.size());
        for (std::size_t index = 0; index < before.size(); ++index)
            EXPECT_NEAR(after[index], -before[index], 1e-6f);

        auto degree = single("lfs.set_sh_degree", splats(1), [](Node& node) {
            node.input_values["Degree"] = std::int64_t(3);
        });
        ASSERT_TRUE(degree.ok);
        EXPECT_EQ(degree.geometry.splats->shN.shape(), TensorShape({3, 15, 3}));
    }

    TEST_P(NodesCore, SelectionBlendEndpointsPreserveRawAttributesExactly) {
        auto input = splats(1);
        input.splats->scaling = tensor({-3, 0, 3, -3, 0, 3, -3, 0, 3}, {3, 3});
        using Configure = std::function<void(Node&)>;
        const std::vector<std::pair<std::string, Configure>> writers = {
            {"lfs.set_position", [](Node& node) { node.input_values["Offset"] = glm::vec3(1, 2, 3); }},
            {"lfs.set_colour", [](Node& node) { node.input_values["Colour"] = glm::vec3(0.9f, 0.1f, 0.2f); }},
            {"lfs.set_opacity", [](Node& node) { node.input_values["Opacity"] = 0.25f; }},
            {"lfs.set_scale", [](Node& node) { node.input_values["Scale"] = glm::vec3(0.2f, 0.3f, 0.4f); }},
            {"lfs.sharpen", [](Node& node) { node.input_values["Amount"] = 0.5f; }},
            {"lfs.scale_clamp", [](Node& node) { node.input_values["Max Aspect"] = 2.0f; }},
            {"lfs.colour_correct", [](Node& node) { node.input_values["Exposure"] = 2.0f; }},
            {"lfs.recolour", [](Node& node) {
                 node.input_values["Colour"] = glm::vec3(0.8f, 0.1f, 0.3f);
                 node.properties["keep_shading"] = false;
             }},
            {"lfs.invert_colour", [](Node&) {}}};

        const auto row = [&](const Tensor& value, size_t index) {
            return host<float>(value.slice(0, index, index + 1).contiguous());
        };
        // A NaN weight (e.g. a negative base raised to a fraction) must act as unselected.
        for (const float unselected : {0.0f, std::numeric_limits<float>::quiet_NaN()}) {
            for (const auto& [type, configure] : writers) {
                SCOPED_TRACE(type + " unselected=" + std::to_string(unselected));
                input.splats->attributes["selection"] = tensor({unselected, 1, 0.5f}, {3});
                NodeTree tree(registry_);
                auto& attribute = tree.add_node("lfs.named_attribute", "Selection");
                attribute.input_values["Name"] = std::string("selection");
                auto& writer = tree.add_node(type, "Writer");
                configure(writer);
                ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Writer", "Geometry"}));
                ASSERT_TRUE(tree.add_link({"Selection", "Attribute", "Writer", "Selection"}));
                ASSERT_TRUE(tree.add_link({"Writer", "Geometry", tree.output_node().name, "Geometry"}));
                const auto result = evaluate(tree, {input, {}, 1});
                ASSERT_TRUE(result.ok) << (result.errors.empty() ? "no error text"
                                                                 : result.errors.begin()->second);
                ASSERT_TRUE(result.geometry.splats);
                const auto& before = *input.splats;
                const auto& after = *result.geometry.splats;
                EXPECT_EQ(row(after.means, 0), row(before.means, 0));
                EXPECT_EQ(row(after.sh0, 0), row(before.sh0, 0));
                EXPECT_EQ(row(after.shN, 0), row(before.shN, 0));
                EXPECT_EQ(row(after.scaling, 0), row(before.scaling, 0));
                EXPECT_EQ(row(after.rotation, 0), row(before.rotation, 0));
                EXPECT_EQ(row(after.opacity, 0), row(before.opacity, 0));
                for (const auto& [name, value] : before.attributes)
                    if (name != "selection") // The NaN weight itself never compares equal.
                        EXPECT_EQ(row(after.attributes.at(name), 0), row(value, 0));

                const bool selected_changed = row(after.means, 1) != row(before.means, 1) ||
                                              row(after.sh0, 1) != row(before.sh0, 1) ||
                                              row(after.shN, 1) != row(before.shN, 1) ||
                                              row(after.scaling, 1) != row(before.scaling, 1) ||
                                              row(after.rotation, 1) != row(before.rotation, 1) ||
                                              row(after.opacity, 1) != row(before.opacity, 1);
                EXPECT_TRUE(selected_changed);
            }
        }
    }

    // Scale Clamp shortens the longest axis to Max Aspect times the middle one and leaves splats that
    // already comply untouched.
    TEST_P(NodesCore, ScaleClampBoundsLongestOverMiddleAndKeepsCompliantSplats) {
        constexpr std::size_t count = 4096;
        std::mt19937 random(5);
        std::uniform_real_distribution<float> log_scale(-6.0f, 2.0f);
        std::vector<float> values(count * 3);
        for (auto& value : values)
            value = log_scale(random);
        // Splats exactly at the limit, which rounding in the middle axis must not clip.
        const float limit = std::log(16.0f);
        for (std::size_t row = 0; row < 64; ++row) {
            const float middle = -6.2713494f + 0.01f * static_cast<float>(row);
            values[row * 3] = middle;
            values[row * 3 + 1] = middle + limit;
            values[row * 3 + 2] = middle - 1.3950546f;
        }
        SplatsComponent component;
        component.means = tensor(std::vector<float>(count * 3, 0.0f), {count, 3});
        component.sh0 = tensor(std::vector<float>(count * 3, 0.0f), {count, 3});
        component.shN = tensor(std::vector<float>(count * 9, 0.0f), {count, 3, 3});
        component.scaling = tensor(values, {count, 3});
        std::vector<float> identity(count * 4, 0.0f);
        for (std::size_t row = 0; row < count; ++row)
            identity[row * 4] = 1.0f;
        component.rotation = tensor(std::move(identity), {count, 4});
        component.opacity = tensor(std::vector<float>(count, 0.0f), {count});
        component.sh_degree = 1;
        Geometry input{std::move(component), std::nullopt, std::nullopt};
        const auto clamped = single("lfs.scale_clamp", input, [](Node& node) { node.input_values["Max Aspect"] = 16.0f; });
        ASSERT_TRUE(clamped.ok);
        const auto after = host<float>(clamped.geometry.splats->scaling);
        for (std::size_t row = 0; row < count; ++row) {
            std::array<float, 3> before_row{values[row * 3], values[row * 3 + 1], values[row * 3 + 2]};
            std::array<float, 3> after_row{after[row * 3], after[row * 3 + 1], after[row * 3 + 2]};
            std::ranges::sort(before_row);
            std::ranges::sort(after_row);
            ASSERT_LE(after_row[2] - after_row[1], limit + 1e-5f) << "row " << row;
            if (before_row[2] - before_row[1] <= limit)
                for (std::size_t axis = 0; axis < 3; ++axis)
                    ASSERT_EQ(after[row * 3 + axis], values[row * 3 + axis]) << "row " << row;
        }
    }

    TEST_P(NodesCore, ColourCorrectGammaTouchesOnlyDCAndAffineTouchesEverySHCoefficient) {
        Geometry input = splats(1);
        const auto original_shn = host<float>(input.splats->shN);
        auto gamma = single("lfs.colour_correct", input, [](Node& node) {
            node.input_values["Gamma"] = 2.0f;
        });
        ASSERT_TRUE(gamma.ok);
        EXPECT_EQ(host<float>(gamma.geometry.splats->shN), original_shn);
        auto exposure = single("lfs.colour_correct", input, [](Node& node) {
            node.input_values["Exposure"] = 1.0f;
        });
        ASSERT_TRUE(exposure.ok);
        const auto exposed = host<float>(exposure.geometry.splats->shN);
        for (std::size_t index = 0; index < exposed.size(); ++index)
            EXPECT_NEAR(exposed[index], original_shn[index] * 2.0f, 2e-5f);
    }

    TEST_P(NodesCore, ColourCorrectLevelsMidpointAndThreeWayFollowSHRules) {
        constexpr float c0 = 0.28209479177387814f;
        auto geometry = splats(1);
        geometry.splats->sh0 =
            (tensor({0.25f, 0.25f, 0.25f, 0.5f, 0.5f, 0.5f, 0.75f, 0.75f, 0.75f}, {3, 3}) -
             0.5f) /
            c0;
        const auto original_shn = host<float>(geometry.splats->shN);
        const auto levels = single("lfs.colour_correct", geometry, [](Node& node) {
            node.input_values["Black Point"] = 0.25f;
            node.input_values["White Point"] = 0.75f;
        });
        ASSERT_TRUE(levels.ok);
        const auto level_base = host<float>(levels.geometry.splats->sh0 * c0 + 0.5f);
        const std::vector<float> expected_levels{0, 0, 0, 0.5f, 0.5f, 0.5f, 1, 1, 1};
        for (size_t index = 0; index < expected_levels.size(); ++index)
            EXPECT_NEAR(level_base[index], expected_levels[index], 1e-5f);
        const auto level_shn = host<float>(levels.geometry.splats->shN);
        for (size_t index = 0; index < level_shn.size(); ++index)
            EXPECT_NEAR(level_shn[index], original_shn[index] * 2, 2e-5f);

        geometry.splats->sh0 =
            (tensor({-0.25f, -0.25f, -0.25f, 0.25f, 0.25f, 0.25f, 1, 1, 1}, {3, 3}) - 0.5f) /
            c0;
        const auto midpoint = single("lfs.colour_correct", geometry, [](Node& node) {
            node.input_values["Midpoint"] = 0.25f;
        });
        ASSERT_TRUE(midpoint.ok);
        const auto midpoint_base = host<float>(midpoint.geometry.splats->sh0 * c0 + 0.5f);
        const std::vector<float> expected_midpoint{-0.25f, -0.25f, -0.25f, 0.5f, 0.5f, 0.5f, 1, 1, 1};
        for (size_t index = 0; index < expected_midpoint.size(); ++index)
            EXPECT_NEAR(midpoint_base[index], expected_midpoint[index], 2e-5f);
        EXPECT_EQ(host<float>(midpoint.geometry.splats->shN), original_shn);

        geometry.splats->sh0 =
            (tensor({0, 0, 0, 0.5f, 0.5f, 0.5f, 1, 1, 1}, {3, 3}) - 0.5f) / c0;
        const auto graded = single("lfs.colour_correct", geometry, [](Node& node) {
            node.input_values["Shadows"] = glm::vec3(1, 0, 0);
            node.input_values["Midtones"] = glm::vec3(0, 1, 0);
            node.input_values["Highlights"] = glm::vec3(0, 0, 1);
        });
        ASSERT_TRUE(graded.ok);
        const auto graded_base = host<float>(graded.geometry.splats->sh0 * c0 + 0.5f);
        const std::vector<float> expected_graded{0.35f, 0, 0, 0.5f, 0.85f, 0.5f, 1, 1, 1.35f};
        for (size_t index = 0; index < expected_graded.size(); ++index)
            EXPECT_NEAR(graded_base[index], expected_graded[index], 2e-5f);
        EXPECT_EQ(host<float>(graded.geometry.splats->shN), original_shn);
    }

    TEST_P(NodesCore, ColourCorrectAutoRangeUsesBaseLumaPercentiles) {
        constexpr float c0 = 0.28209479177387814f;
        auto geometry = splats(0);
        constexpr size_t count = 101;
        std::vector<float> positions(count * 3, 0);
        std::vector<float> colours;
        colours.reserve(count * 3);
        for (size_t index = 0; index < count; ++index) {
            const float value = static_cast<float>(index) / 100;
            colours.insert(colours.end(), {value, value, value});
        }
        geometry.splats->means = tensor(std::move(positions), {count, 3});
        geometry.splats->sh0 = (tensor(std::move(colours), {count, 3}) - 0.5f) / c0;
        geometry.splats->shN = Tensor::zeros({count, 0, 3}, device());
        geometry.splats->scaling = Tensor::zeros({count, 3}, device());
        geometry.splats->rotation = Tensor::zeros({count, 4}, device());
        geometry.splats->opacity = Tensor::zeros({count}, device());
        geometry.splats->attributes.clear();
        const auto result = single("lfs.colour_correct", geometry, [](Node& node) {
            node.properties["auto_range"] = true;
        });
        ASSERT_TRUE(result.ok);
        const auto base = host<float>(result.geometry.splats->sh0 * c0 + 0.5f);
        EXPECT_NEAR(base[3], 0, 2e-5f);
        EXPECT_NEAR(base[99 * 3], 1, 2e-5f);
    }

    TEST_P(NodesCore, ColourCorrectHueShiftPreservesRec709Luma) {
        constexpr float c0 = 0.28209479177387814f;
        auto geometry = splats(0);
        geometry.splats->sh0 = (tensor({1, 0, 0, 0, 1, 0, 0, 0, 1}, {3, 3}) - 0.5f) / c0;
        const auto result = single("lfs.colour_correct", geometry, [](Node& node) {
            node.input_values["Hue Shift"] = 90.0f;
        });
        ASSERT_TRUE(result.ok);
        const auto before = host<float>(geometry.splats->sh0 * c0 + 0.5f);
        const auto after = host<float>(result.geometry.splats->sh0 * c0 + 0.5f);
        for (size_t row = 0; row < before.size(); row += 3) {
            const auto luma = [](const std::vector<float>& colour, size_t index) {
                return colour[index] * 0.2126f + colour[index + 1] * 0.7152f +
                       colour[index + 2] * 0.0722f;
            };
            EXPECT_NEAR(luma(after, row), luma(before, row), 1e-4f);
        }
    }

    TEST_P(NodesCore, ColourCorrectParameterExtremesRemainFiniteOutsideDisplayRange) {
        constexpr float c0 = 0.28209479177387814f;
        auto geometry = splats(1);
        geometry.splats->sh0 =
            (tensor({-1, 0, 3, 3, -1, 1, 0.5f, 2, -0.5f}, {3, 3}) - 0.5f) / c0;
        using Configure = std::function<void(Node&)>;
        const std::vector<std::pair<std::string, Configure>> cases = {
            {"selection min", [](Node& node) { node.input_values["Selection"] = 0.0f; }},
            {"selection max", [](Node& node) { node.input_values["Selection"] = 1.0f; }},
            {"exposure min", [](Node& node) { node.input_values["Exposure"] = -10.0f; }},
            {"exposure max", [](Node& node) { node.input_values["Exposure"] = 10.0f; }},
            {"black min", [](Node& node) { node.input_values["Black Point"] = -2.0f; }},
            {"black max", [](Node& node) { node.input_values["Black Point"] = 2.0f; }},
            {"white min", [](Node& node) { node.input_values["White Point"] = -2.0f; }},
            {"white max", [](Node& node) { node.input_values["White Point"] = 2.0f; }},
            {"levels forward", [](Node& node) {
                 node.input_values["Black Point"] = -2.0f;
                 node.input_values["White Point"] = 2.0f;
             }},
            {"levels reversed", [](Node& node) {
                 node.input_values["Black Point"] = 2.0f;
                 node.input_values["White Point"] = -2.0f;
             }},
            {"midpoint min", [](Node& node) { node.input_values["Midpoint"] = 0.01f; }},
            {"midpoint max", [](Node& node) { node.input_values["Midpoint"] = 0.99f; }},
            {"contrast min", [](Node& node) { node.input_values["Contrast"] = 0.0f; }},
            {"contrast max", [](Node& node) { node.input_values["Contrast"] = 2.0f; }},
            {"saturation min", [](Node& node) { node.input_values["Saturation"] = 0.0f; }},
            {"saturation max", [](Node& node) { node.input_values["Saturation"] = 2.0f; }},
            {"hue min", [](Node& node) { node.input_values["Hue Shift"] = -360.0f; }},
            {"hue max", [](Node& node) { node.input_values["Hue Shift"] = 360.0f; }},
            {"temperature min", [](Node& node) { node.input_values["Temperature"] = -1.0f; }},
            {"temperature max", [](Node& node) { node.input_values["Temperature"] = 1.0f; }},
            {"tint min", [](Node& node) { node.input_values["Tint"] = -1.0f; }},
            {"tint max", [](Node& node) { node.input_values["Tint"] = 1.0f; }},
            {"shadows min", [](Node& node) { node.input_values["Shadows"] = glm::vec3(-1); }},
            {"shadows max", [](Node& node) { node.input_values["Shadows"] = glm::vec3(1); }},
            {"midtones min", [](Node& node) { node.input_values["Midtones"] = glm::vec3(-1); }},
            {"midtones max", [](Node& node) { node.input_values["Midtones"] = glm::vec3(1); }},
            {"highlights min", [](Node& node) { node.input_values["Highlights"] = glm::vec3(-1); }},
            {"highlights max", [](Node& node) { node.input_values["Highlights"] = glm::vec3(1); }},
            {"gamma min", [](Node& node) { node.input_values["Gamma"] = 0.001f; }},
            {"gamma max", [](Node& node) { node.input_values["Gamma"] = 2.0f; }},
            {"fuzzer overflow case", [](Node& node) {
                 node.input_values["Exposure"] = 9.0f;
                 node.input_values["Midpoint"] = 0.99f;
                 node.input_values["Gamma"] = 0.15f;
                 node.input_values["Shadows"] = glm::vec3(-1);
                 node.input_values["Midtones"] = glm::vec3(1);
                 node.input_values["Highlights"] = glm::vec3(-1);
             }}};
        for (const auto& [name, configure] : cases) {
            SCOPED_TRACE(name);
            const auto result = single("lfs.colour_correct", geometry, configure);
            ASSERT_TRUE(result.ok) << (result.errors.empty() ? "no error text"
                                                             : result.errors.begin()->second);
            expect_finite(result.geometry);
        }
    }

    TEST_P(NodesCore, RecolourPreservesShadingAndFadesViewDependentColour) {
        constexpr float c0 = 0.28209479177387814f;
        auto geometry = splats(1);
        geometry.splats->sh0 =
            (tensor({1, 0, 0, 0, 1, 0, 0, 0, 1}, {3, 3}) - 0.5f) / c0;
        const auto shaded = single("lfs.recolour", geometry, [](Node& node) {
            node.input_values["Colour"] = glm::vec3(0.5f);
            node.properties["fade_view_dependent"] = false;
        });
        ASSERT_TRUE(shaded.ok);
        const auto base = host<float>(shaded.geometry.splats->sh0 * c0 + 0.5f);
        const float expected_luma[] = {0.2126f, 0.7152f, 0.0722f};
        for (size_t row = 0; row < 3; ++row)
            for (size_t channel = 0; channel < 3; ++channel)
                EXPECT_NEAR(base[row * 3 + channel], expected_luma[row], 2e-5f);

        const auto original_shn = host<float>(geometry.splats->shN);
        const auto mixed = single("lfs.recolour", geometry, [](Node& node) {
            node.input_values["Colour"] = glm::vec3(0, 0, 1);
            node.input_values["Weight"] = 0.5f;
            node.input_values["Selection"] = 0.5f;
            node.properties["keep_shading"] = false;
        });
        ASSERT_TRUE(mixed.ok);
        const auto mixed_base = host<float>(mixed.geometry.splats->sh0 * c0 + 0.5f);
        const auto original_base = host<float>(geometry.splats->sh0 * c0 + 0.5f);
        for (size_t index = 0; index < mixed_base.size(); ++index) {
            const float target = index % 3 == 2 ? 1 : 0;
            EXPECT_NEAR(mixed_base[index], original_base[index] * 0.75f + target * 0.25f, 2e-5f);
        }
        const auto mixed_shn = host<float>(mixed.geometry.splats->shN);
        for (size_t index = 0; index < mixed_shn.size(); ++index)
            EXPECT_NEAR(mixed_shn[index], original_shn[index] * 0.75f, 2e-5f);
    }

    TEST_P(NodesCore, DecimateRanksOnlySelectedSplatsAndKeepsUnselected) {
        auto geometry = splats(0);
        geometry.splats->means = tensor({0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0}, {5, 3});
        geometry.splats->sh0 = Tensor::zeros({5, 3}, device());
        geometry.splats->shN = Tensor::zeros({5, 0, 3}, device());
        geometry.splats->scaling =
            tensor({0, 0, 0, std::log(2.0f), 0, 0, std::log(3.0f), 0, 0,
                    std::log(100.0f), 0, 0, std::log(100.0f), 0, 0},
                   {5, 3});
        geometry.splats->rotation = Tensor::zeros({5, 4}, device());
        geometry.splats->opacity = tensor({std::log(9.0f), 0, -std::log(9.0f), 0, 0}, {5});
        geometry.splats->attributes = {{"selected", tensor({1, 1, 1, 0, 0}, {5})}};
        NodeTree tree(registry_);
        auto& attribute = tree.add_node("lfs.named_attribute", "Selected");
        attribute.input_values["Name"] = std::string("selected");
        tree.add_node("lfs.decimate", "Decimate");
        tree.find_node("Decimate")->input_values["Keep Fraction"] = 0.5f;
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Decimate", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Selected", "Attribute", "Decimate", "Selection"}));
        ASSERT_TRUE(tree.add_link({"Decimate", "Geometry", tree.output_node().name, "Geometry"}));
        const auto result = evaluate(tree, {geometry, {}, 1});
        ASSERT_TRUE(result.ok) << (result.errors.empty() ? "" : result.errors.begin()->second);
        EXPECT_EQ(host<float>(result.geometry.splats->means),
                  (std::vector<float>{1, 0, 0, 3, 0, 0, 4, 0, 0}));
        EXPECT_EQ(host<float>(result.geometry.splats->attributes.at("selected")),
                  (std::vector<float>{1, 0, 0}));
    }

    TEST_P(NodesCore, JoinPadsMixedSHAndOffsetsMeshIndices) {
        Geometry a = splats(0), b = splats(2);
        auto mesh_a = std::make_shared<lfs::core::MeshData>(tensor({0, 0, 0, 1, 0, 0, 0, 1, 0}, {3, 3}),
                                                            ints({0, 1, 2}, {1, 3}));
        auto mesh_b = std::make_shared<lfs::core::MeshData>(tensor({0, 0, 1, 1, 0, 1, 0, 1, 1}, {3, 3}),
                                                            ints({0, 1, 2}, {1, 3}));
        a.mesh = MeshComponent{mesh_a};
        b.mesh = MeshComponent{mesh_b};
        auto constant_type = [&](std::string id, Geometry value) {
            NodeTypeInfo info;
            info.id = id;
            info.label = id;
            info.category = "Test";
            info.outputs.push_back({"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)});
            info.evaluate = [value](NodeContext& context) {
                context.set_output("Geometry", value);
            };
            registry_.register_type(std::move(info));
        };
        constant_type("test.a", a);
        constant_type("test.b", b);
        NodeTree tree(registry_);
        tree.add_node("test.a", "A");
        tree.add_node("test.b", "B");
        tree.add_node("lfs.join_geometry", "Join");
        EXPECT_TRUE(tree.add_link({"A", "Geometry", "Join", "Geometry"}));
        EXPECT_TRUE(tree.add_link({"B", "Geometry", "Join", "Geometry"}));
        EXPECT_TRUE(tree.add_link({"Join", "Geometry", tree.output_node().name, "Geometry"}));
        auto result = evaluate(tree, {{}, {}, 1});
        ASSERT_TRUE(result.ok);
        EXPECT_EQ(result.geometry.splats->sh_degree, 2);
        EXPECT_EQ(result.geometry.splats->shN.shape(), TensorShape({6, 8, 3}));
        EXPECT_EQ(host<int>(result.geometry.mesh->mesh->indices), (std::vector<int>{0, 1, 2, 3, 4, 5}));
        EXPECT_EQ(result.geometry.mesh->mesh->materials.size(), 2u);
        ASSERT_EQ(result.geometry.mesh->mesh->submeshes.size(), 2u);
        EXPECT_EQ(result.geometry.mesh->mesh->submeshes[1].material_index, 1u);
        EXPECT_EQ(result.geometry.mesh->mesh->submeshes[1].start_index, 3u);
    }

    TEST_P(NodesCore, MeshAttributesSurviveDeviceTransferAndJoin) {
        auto mesh_a = std::make_shared<lfs::core::MeshData>(tensor({0, 0, 0, 1, 0, 0, 0, 1, 0}, {3, 3}).cpu(),
                                                            ints({0, 1, 2}, {1, 3}).cpu());
        auto mesh_b = std::make_shared<lfs::core::MeshData>(tensor({0, 0, 1, 1, 0, 1, 0, 1, 1}, {3, 3}).cpu(),
                                                            ints({0, 1, 2}, {1, 3}).cpu());
        Geometry a, b;
        a.mesh = MeshComponent{mesh_a};
        a.mesh->attributes["a"] = Tensor::full({3}, 7.0f, Device::CPU);
        b.mesh = MeshComponent{mesh_b};
        NodeTypeInfo info;
        info.id = "test.mesh_b";
        info.outputs.push_back({"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)});
        info.evaluate = [b](NodeContext& context) { context.set_output("Geometry", b); };
        registry_.unregister_type(info.id);
        ASSERT_TRUE(registry_.register_type(std::move(info)));
        NodeTree tree(registry_);
        tree.add_node("test.mesh_b", "B");
        tree.add_node("lfs.join_geometry", "Join");
        Node& read = tree.add_node("lfs.named_attribute", "Read");
        read.input_values["Name"] = std::string("a");
        Node& copy = tree.add_node("lfs.store_named_attribute", "Copy");
        copy.input_values["Name"] = std::string("copy");
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Join", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"B", "Geometry", "Join", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Join", "Geometry", "Copy", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Read", "Attribute", "Copy", "Value"}));
        ASSERT_TRUE(tree.add_link({"Copy", "Geometry", tree.output_node().name, "Geometry"}));
        const auto result = evaluate(tree, {a, {}, 1, device()});
        registry_.unregister_type("test.mesh_b");
        ASSERT_TRUE(result.ok) << (result.errors.empty() ? "" : result.errors.begin()->second);
        ASSERT_TRUE(result.geometry.mesh);
        const auto& attributes = result.geometry.mesh->attributes;
        ASSERT_TRUE(attributes.contains("a"));
        EXPECT_EQ(attributes.at("a").device(), device());
        EXPECT_EQ(host<float>(attributes.at("a")), (std::vector<float>{7, 7, 7, 0, 0, 0}));
        EXPECT_EQ(host<float>(attributes.at("copy")), (std::vector<float>{7, 7, 7, 0, 0, 0}));
    }

    TEST_P(NodesCore, MeshTransformsCarryNormalsAndTangents) {
        auto source = std::make_shared<lfs::core::MeshData>(tensor({0, 0, 0, 1, 0, 0, 0, 1, 0}, {3, 3}),
                                                            ints({0, 1, 2}, {1, 3}));
        const float diagonal = std::sqrt(0.5f);
        source->normals = tensor({0, 0, 1, 0, 0, 1, diagonal, diagonal, 0}, {3, 3});
        source->tangents = tensor({1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, -1}, {3, 4});
        const auto close = [](const std::vector<float>& actual, const std::vector<float>& expected) {
            ASSERT_EQ(actual.size(), expected.size());
            for (size_t i = 0; i < actual.size(); ++i)
                EXPECT_NEAR(actual[i], expected[i], 1e-5f) << "element " << i;
        };

        // A quarter turn about Y takes +Z to +X and +X to -Z.
        const auto turned = transform_mesh(*source, rotation_matrix(glm::vec3(0, 90, 0)));
        close(host<float>(turned->normals), {1, 0, 0, 1, 0, 0, 0, diagonal, -diagonal});
        close(host<float>(turned->tangents), {0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, -1});

        // A stretching mirror: normals by the inverse transpose, tangent handedness flipped.
        const auto mirrored = transform_mesh(*source, glm::scale(glm::mat4(1), glm::vec3(-2, 1, 1)));
        const float x = -0.5f / std::sqrt(1.25f), y = 1.0f / std::sqrt(1.25f);
        close(host<float>(mirrored->vertices), {0, 0, 0, -2, 0, 0, 0, 1, 0});
        close(host<float>(mirrored->normals), {0, 0, 1, 0, 0, 1, x, y, 0});
        close(host<float>(mirrored->tangents), {-1, 0, 0, -1, -1, 0, 0, -1, -1, 0, 0, 1});

        // Flattening keeps normals finite.
        const auto flat = transform_mesh(*source, glm::scale(glm::mat4(1), glm::vec3(1, 1, 0)));
        for (const float value : host<float>(flat->normals))
            EXPECT_TRUE(std::isfinite(value));
    }

    TEST_P(NodesCore, TriangleRayIndexFindsPointsInsideAClosedMesh) {
        const auto mesh = torus(48, 24);
        const lfs::core::TriangleRayIndex index(mesh->vertices, mesh->indices);
        std::mt19937 random(5);
        std::uniform_real_distribution<float> across(-2.2f, 2.2f), along(-0.7f, 0.7f);
        std::vector<float> points;
        std::vector<bool> expected;
        // Away from the faceted surface, the smooth torus decides inside.
        while (expected.size() < 20000) {
            const float x = across(random), y = across(random), z = along(random);
            const float ring = std::sqrt(x * x + y * y) - 1.5f;
            const float distance = std::sqrt(ring * ring + z * z);
            if (std::abs(distance - 0.5f) < 0.03f)
                continue;
            points.insert(points.end(), {x, y, z});
            expected.push_back(distance < 0.5f);
        }
        points.insert(points.end(), {std::numeric_limits<float>::quiet_NaN(), 0, 0, 1e30f, 0, 0});
        expected.insert(expected.end(), {false, false});
        const auto queries = tensor(points, {expected.size(), 3});
        const auto inside = index.odd_crossings(queries);
        EXPECT_EQ(inside.device(), device());
        EXPECT_EQ(inside.cpu().to_vector_bool(), expected);
        // Batches answer like one query.
        const auto first = index.odd_crossings(queries.slice(0, 0, 777));
        EXPECT_EQ(first.cpu().to_vector_bool(), std::vector<bool>(expected.begin(), expected.begin() + 777));

        const lfs::core::TriangleRayIndex empty(mesh->vertices, ints({}, {0, 3}));
        EXPECT_EQ(empty.odd_crossings(queries).cpu().to_vector_bool(), std::vector<bool>(expected.size(), false));
    }

    TEST_P(NodesCore, DeleteSeparateAndStoredSelectionCarryAttributes) {
        NodeTree tree(registry_);
        Node &stored = tree.add_node("lfs.stored_selection"),
             &separate = tree.add_node("lfs.separate_geometry");
        Tensor mask = Tensor::from_vector(std::vector<bool>{true, false, true}, {3}, Device::CPU);
        if (device() == Device::GPU)
            mask = mask.to(device());
        set_stored_selection(stored, mask);
        EXPECT_TRUE(tree.add_link({tree.input_node().name, "Geometry", separate.name, "Geometry"}));
        EXPECT_TRUE(tree.add_link({stored.name, "Selection", separate.name, "Selection"}));
        EXPECT_TRUE(tree.add_link({separate.name, "Selection", tree.output_node().name, "Geometry"}));
        auto result = evaluate(tree, {splats(), {}, 1});
        ASSERT_TRUE(result.ok);
        EXPECT_EQ(result.geometry.splats->means.shape()[0], 2u);
        EXPECT_EQ(host<float>(result.geometry.splats->attributes.at("weight")), (std::vector<float>{10, 30}));
    }

    TEST_P(NodesCore, PointsSplatsRoundTripAndMeshToPoints) {
        Geometry points;
        points.points = PointsComponent{tensor({0, 0, 0, 1, 0, 0}, {2, 3}),
                                        tensor({1, 0, 0, 0, 1, 0}, {2, 3}),
                                        {{"id", tensor({4, 5}, {2})}}};
        auto converted = single("lfs.points_to_splats", points, [](Node& node) {
            node.input_values["Radius"] = 0.2f;
            node.input_values["Opacity"] = 0.8f;
        });
        ASSERT_TRUE(converted.ok);
        ASSERT_TRUE(converted.geometry.splats);
        EXPECT_EQ(converted.geometry.splats->sh_degree, 0);
        auto back = single("lfs.splats_to_points", converted.geometry);
        ASSERT_TRUE(back.ok);
        EXPECT_EQ(host<float>(back.geometry.points->colors), host<float>(points.points->colors));
        EXPECT_EQ(host<float>(back.geometry.points->attributes.at("id")), (std::vector<float>{4, 5}));

        auto mesh = std::make_shared<lfs::core::MeshData>(tensor({0, 0, 0, 1, 0, 0, 0, 1, 0}, {3, 3}),
                                                          ints({0, 1, 2}, {1, 3}));
        auto mesh_points = single("lfs.mesh_to_points", geometry_from_mesh(mesh));
        ASSERT_TRUE(mesh_points.ok);
        EXPECT_EQ(mesh_points.geometry.points->positions.shape()[0], 3u);
    }

    TEST_P(NodesCore, CorePayloadConversionsDropDeletedRows) {
        Geometry original = splats(1);
        auto data = splat_data_from_geometry(original);
        ASSERT_NE(data, nullptr);
        Tensor deleted = Tensor::from_vector(std::vector<bool>{false, true, false}, {3}, Device::CPU);
        if (device() == Device::GPU)
            deleted = deleted.to(device());
        data->deleted() = std::move(deleted);
        Geometry filtered = geometry_from_splat_data(*data);
        ASSERT_TRUE(filtered.splats);
        EXPECT_EQ(filtered.splats->means.shape()[0], 2u);
        EXPECT_EQ(filtered.splats->shN.shape(), TensorShape({2, 3, 3}));
        auto restored = splat_data_from_geometry(filtered);
        ASSERT_NE(restored, nullptr);
        EXPECT_EQ(restored->shN_canonical().shape(), TensorShape({2, 3, 3}));
    }

    TEST_P(NodesCore, InsideMeshAndNeighbourCountProduceFields) {
        const std::vector<float> vertices = {-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1,
                                             -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1};
        const std::vector<int> faces = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                                        1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7};
        auto mesh = std::make_shared<lfs::core::MeshData>(tensor(vertices, {8, 3}), ints(faces, {12, 3}));
        Geometry mixed = splats();
        mixed.splats->means = tensor({0, 0, 0, 2, 0, 0, 0.5f, 0, 0}, {3, 3});
        mixed.mesh = MeshComponent{mesh};
        NodeTree tree(registry_);
        tree.add_node("lfs.inside_mesh", "Inside");
        tree.add_node("lfs.delete_geometry", "Delete");
        EXPECT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Inside", "Mesh"}));
        EXPECT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Delete", "Geometry"}));
        EXPECT_TRUE(tree.add_link({"Inside", "Selection", "Delete", "Selection"}));
        EXPECT_TRUE(tree.add_link({"Delete", "Geometry", tree.output_node().name, "Geometry"}));
        auto inside_result = evaluate(tree, {mixed, {}, 1});
        ASSERT_TRUE(inside_result.ok);
        EXPECT_EQ(inside_result.geometry.splats->means.shape()[0], 1u);

        NodeTree neighbours(registry_);
        neighbours.add_node("lfs.neighbour_count", "Count");
        neighbours.add_node("lfs.compare", "Compare");
        neighbours.add_node("lfs.delete_geometry", "Delete");
        neighbours.find_node("Count")->input_values["Radius"] = 1.1f;
        neighbours.find_node("Compare")->input_values["B"] = 1.0f;
        neighbours.find_node("Compare")->properties["operation"] = "greater_equal";
        EXPECT_TRUE(neighbours.add_link({"Count", "Count", "Compare", "A"}));
        EXPECT_TRUE(neighbours.add_link({neighbours.input_node().name, "Geometry", "Delete", "Geometry"}));
        EXPECT_TRUE(neighbours.add_link({"Compare", "Result", "Delete", "Selection"}));
        EXPECT_TRUE(neighbours.add_link({"Delete", "Geometry", neighbours.output_node().name, "Geometry"}));
        auto neighbour_result = evaluate(neighbours, {splats(), {}, 1});
        ASSERT_TRUE(neighbour_result.ok);
        EXPECT_EQ(neighbour_result.geometry.splats->means.shape()[0], 1u);
    }

    TEST_P(NodesCore, MeshFaceDeletionRemovesFacesUsingSelectedVertex) {
        auto mesh = std::make_shared<lfs::core::MeshData>(
            tensor({0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0}, {4, 3}), ints({0, 1, 2, 0, 2, 3}, {2, 3}));
        NodeTree tree(registry_);
        tree.add_node("lfs.index", "Index");
        tree.add_node("lfs.compare", "Compare");
        tree.add_node("lfs.delete_geometry", "Delete");
        tree.find_node("Compare")->input_values["B"] = 0.0f;
        tree.find_node("Compare")->properties["operation"] = "equal";
        EXPECT_TRUE(tree.add_link({"Index", "Index", "Compare", "A"}));
        EXPECT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Delete", "Geometry"}));
        EXPECT_TRUE(tree.add_link({"Compare", "Result", "Delete", "Selection"}));
        EXPECT_TRUE(tree.add_link({"Delete", "Geometry", tree.output_node().name, "Geometry"}));
        auto result = evaluate(tree, {geometry_from_mesh(mesh), {}, 1});
        ASSERT_TRUE(result.ok);
        EXPECT_EQ(result.geometry.mesh->mesh->face_count(), 0);
    }

    TEST_P(NodesCore, CacheSkipsUpstreamAndMutePassesGeometryThrough) {
        int calls = 0;
        NodeTypeInfo info;
        info.id = "test.counter";
        info.label = "Counter";
        info.category = "Test";
        info.inputs.push_back({"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)});
        info.outputs.push_back({"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)});
        info.evaluate = [&](NodeContext& context) {
            ++calls;
            context.set_output("Geometry", context.input("Geometry"));
        };
        ASSERT_TRUE(registry_.register_type(std::move(info)));
        NodeTree tree(registry_);
        Node &counter = tree.add_node("test.counter"), &opacity = tree.add_node("lfs.set_opacity");
        EXPECT_TRUE(tree.add_link({tree.input_node().name, "Geometry", counter.name, "Geometry"}));
        EXPECT_TRUE(tree.add_link({counter.name, "Geometry", opacity.name, "Geometry"}));
        EXPECT_TRUE(tree.add_link({opacity.name, "Geometry", tree.output_node().name, "Geometry"}));
        EvalCache cache;
        ASSERT_TRUE(evaluate(tree, {splats(), {}, 7}, nullptr, &cache).ok);
        opacity.input_values["Opacity"] = 0.25f;
        ASSERT_TRUE(evaluate(tree, {splats(), {}, 7}, nullptr, &cache).ok);
        EXPECT_EQ(calls, 1);
        opacity.muted = true;
        auto muted = evaluate(tree, {splats(), {}, 7}, nullptr, &cache);
        ASSERT_TRUE(muted.ok);
        EXPECT_EQ(calls, 1);
        EXPECT_EQ(host<float>(muted.geometry.splats->opacity), (std::vector<float>{0, 1, -1}));
    }

    TEST_P(NodesCore, SceneTimeSecondsFramesAndCachedClockChanges) {
        NodeTree tree(registry_);
        tree.add_node("lfs.scene_time", "Clock");
        tree.add_node("lfs.combine_xyz", "Vector");
        tree.add_node("lfs.set_position", "Position");
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Position", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Clock", "Seconds", "Vector", "X"}));
        ASSERT_TRUE(tree.add_link({"Clock", "Frame", "Vector", "Y"}));
        ASSERT_TRUE(tree.add_link({"Vector", "Vector", "Position", "Position"}));
        ASSERT_TRUE(tree.add_link({"Position", "Geometry", tree.output_node().name, "Geometry"}));
        EvalCache cache;
        const auto geometry = splats();
        for (float seconds : {0.0f, 0.125f, 1.0f, 3.5f, 0.0f}) {
            const auto result = evaluate(tree, {.geometry = geometry, .seconds = seconds, .frames_per_second = 30}, nullptr, &cache);
            ASSERT_TRUE(result.ok);
            const auto positions = host<float>(result.geometry.splats->means);
            for (size_t row = 0; row < positions.size() / 3; ++row) {
                EXPECT_FLOAT_EQ(positions[3 * row], seconds);
                EXPECT_FLOAT_EQ(positions[3 * row + 1], seconds * 30);
                EXPECT_FLOAT_EQ(positions[3 * row + 2], 0);
            }
        }
    }

    TEST_P(NodesCore, MissingTypeFailsWithoutLosingInputGeometry) {
        NodeTree tree(registry_);
        auto json = tree.to_json();
        json["nodes"][0]["type_id"] = "missing.node";
        NodeTree loaded = NodeTree::from_json(json, registry_);
        auto result = evaluate(loaded, {splats(), {}, 1});
        EXPECT_FALSE(result.ok);
        EXPECT_FALSE(result.errors.empty());
        EXPECT_EQ(result.geometry.splats->means.shape()[0], 3u);
    }

    TEST_P(NodesCore, EveryRegisteredBuiltinEvaluatesOnSyntheticGeometry) {
        Geometry input = splats();
        input.points = PointsComponent{tensor({0, 0, 0, 1, 0, 0, 0, 1, 0}, {3, 3}),
                                       tensor({1, 0, 0, 0, 1, 0, 0, 0, 1}, {3, 3}),
                                       {}};
        auto mesh = std::make_shared<lfs::core::MeshData>(
            tensor({-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1},
                   {8, 3}),
            ints({0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4,
                  1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7},
                 {12, 3}));
        input.mesh = MeshComponent{mesh};

        for (const auto& builtin : registry_.list()) {
            if (!builtin->id.starts_with("lfs.") || builtin->id == "lfs.group_input" ||
                builtin->id == "lfs.group_output" || builtin->id == "lfs.group" ||
                builtin->category == "Layout")
                continue;
            SCOPED_TRACE(builtin->id);
            ASSERT_TRUE(builtin->evaluate);
            ASSERT_FALSE(builtin->outputs.empty());
            NodeTree tree(registry_);
            tree.add_node(builtin->id, "Builtin");
            Node* node = tree.find_node("Builtin");
            ASSERT_NE(node, nullptr);
            if (builtin->id == "lfs.named_attribute")
                node->input_values["Name"] = std::string("weight");
            if (builtin->id == "lfs.simplify")
                node->input_values["Ratio"] = 1.0f;
            if (builtin->id == "lfs.geometry_proximity")
                node->input_values["Target"] = input;
            if (builtin->id == "lfs.instance_on_points")
                node->input_values["Instance"] = input;

            const SocketDecl& output = builtin->outputs.front();
            if (output.type == GEOMETRY_SOCKET) {
                const auto geometry_input =
                    std::ranges::find_if(builtin->inputs, [](const SocketDecl& socket) {
                        return socket.type == GEOMETRY_SOCKET;
                    });
                if (geometry_input != builtin->inputs.end())
                    ASSERT_TRUE(tree.add_link(
                        {tree.input_node().name, "Geometry", "Builtin", geometry_input->identifier}));
                ASSERT_TRUE(
                    tree.add_link({"Builtin", output.identifier, tree.output_node().name, "Geometry"}));
            } else {
                tree.add_node("lfs.delete_geometry", "Consumer");
                ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Consumer", "Geometry"}));
                if (builtin->id == "lfs.inside_mesh")
                    ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Builtin", "Mesh"}));
                ASSERT_TRUE(tree.add_link({"Builtin", output.identifier, "Consumer", "Selection"}));
                ASSERT_TRUE(tree.add_link({"Consumer", "Geometry", tree.output_node().name, "Geometry"}));
            }
            Geometry evaluation_input =
                builtin->id == "lfs.inside_mesh" || output.type == GEOMETRY_SOCKET ? input : splats();
            const auto result = evaluate(tree, {evaluation_input, {}, 91});
            if (!result.ok)
                for (const auto& [name, message] : result.errors)
                    std::cerr << name << ": " << message << '\n';
            EXPECT_TRUE(result.ok) << (result.errors.empty() ? "no error text"
                                                             : result.errors.begin()->second);
            if (result.ok)
                expect_finite(result.geometry);
        }
    }

    TEST_P(NodesCore, EveryGeometryBuiltinPassesEmptyGeometryCleanly) {
        const auto empty_splats = [&](const int degree) {
            const size_t rest = static_cast<size_t>((degree + 1) * (degree + 1) - 1);
            return SplatsComponent{Tensor::empty({0, 3}, device()),
                                   Tensor::empty({0, 3}, device()),
                                   Tensor::empty({0, rest, 3}, device()),
                                   Tensor::empty({0, 3}, device()),
                                   Tensor::empty({0, 4}, device()),
                                   Tensor::empty({0}, device()),
                                   degree,
                                   1,
                                   {{"weight", Tensor::empty({0}, device())}}};
        };
        const PointsComponent empty_points{Tensor::empty({0, 3}, device()), Tensor::empty({0, 3}, device()), {{"weight", Tensor::empty({0}, device())}}};
        MeshComponent empty_mesh{std::make_shared<lfs::core::MeshData>(Tensor::empty({0, 3}, device()),
                                                                       Tensor::empty({0, 3}, device(), lfs::core::DataType::Int32))};
        empty_mesh.attributes["weight"] = Tensor::empty({0}, device());
        std::vector<std::pair<std::string, Geometry>> cases;
        for (int degree = 0; degree <= 3; ++degree)
            cases.emplace_back("splats degree " + std::to_string(degree), Geometry{empty_splats(degree), std::nullopt, std::nullopt});
        cases.emplace_back("points", Geometry{std::nullopt, empty_points, std::nullopt});
        cases.emplace_back("mesh", Geometry{std::nullopt, std::nullopt, empty_mesh});
        cases.emplace_back("mixed", Geometry{empty_splats(2), empty_points, empty_mesh});
        // Instance on Points takes splats as its instance whatever its points are.
        NodeTypeInfo instance_type;
        instance_type.id = "test.empty_splats";
        instance_type.outputs.push_back({"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)});
        instance_type.evaluate = [instance = Geometry{empty_splats(1), std::nullopt, std::nullopt}](NodeContext& context) {
            context.set_output("Geometry", instance);
        };
        registry_.unregister_type(instance_type.id);
        ASSERT_TRUE(registry_.register_type(std::move(instance_type)));

        for (const auto& [label, empty] : cases) {
            SCOPED_TRACE(label);
            for (const auto& builtin : registry_.list()) {
                if (!builtin->id.starts_with("lfs.") || builtin->id == "lfs.group_input" ||
                    builtin->id == "lfs.group_output")
                    continue;
                const auto input = std::ranges::find(builtin->inputs, GEOMETRY_SOCKET, &SocketDecl::type);
                const auto output = std::ranges::find(builtin->outputs, GEOMETRY_SOCKET, &SocketDecl::type);
                if (input == builtin->inputs.end() || output == builtin->outputs.end())
                    continue;
                SCOPED_TRACE(builtin->id);
                NodeTree tree(registry_);
                tree.add_node(builtin->id, "Builtin");
                ASSERT_TRUE(tree.add_link(
                    {tree.input_node().name, "Geometry", "Builtin", input->identifier}));
                if (builtin->id == "lfs.instance_on_points") {
                    tree.add_node("test.empty_splats", "Instance");
                    ASSERT_TRUE(tree.add_link({"Instance", "Geometry", "Builtin", "Instance"}));
                }
                ASSERT_TRUE(tree.add_link(
                    {"Builtin", output->identifier, tree.output_node().name, "Geometry"}));
                const auto result = evaluate(tree, {empty, {}, 1});
                ASSERT_TRUE(result.ok) << (result.errors.empty() ? "no error text"
                                                                 : result.errors.begin()->second);
                expect_finite(result.geometry);
                if (result.geometry.splats)
                    EXPECT_EQ(result.geometry.splats->means.shape()[0], 0u);
                if (result.geometry.points)
                    EXPECT_EQ(result.geometry.points->positions.shape()[0], 0u);
                if (result.geometry.mesh && result.geometry.mesh->mesh)
                    EXPECT_EQ(result.geometry.mesh->mesh->vertex_count(), 0);
            }
        }
        registry_.unregister_type("test.empty_splats");
    }

    TEST(NodesCoreMetadata, OnlyAttributeNodesKeepElements) {
        NodeTypeRegistry registry;
        register_builtin_nodes(registry);
        for (const auto* id : {"lfs.set_colour", "lfs.transform_geometry", "lfs.colour_correct", "lfs.group_input", "lfs.group_output",
                               "lfs.reroute", "lfs.rgb_curves", "lfs.store_named_attribute"})
            EXPECT_TRUE(registry.find(id)->keeps_elements) << id;
        for (const auto* id : {"lfs.join_geometry", "lfs.separate_geometry", "lfs.delete_geometry",
                               "lfs.remove_floaters", "lfs.object_info", "lfs.splats_to_points"})
            if (const auto type = registry.find(id))
                EXPECT_FALSE(type->keeps_elements) << id;
    }

    TEST(NodesCoreMetadata, Base64RoundTripAndRejectsMalformedData) {
        const std::vector<std::uint8_t> bytes{0, 1, 127, 128, 254, 255, 11};
        EXPECT_EQ(lfs::core::base64_decode(lfs::core::base64_encode(bytes)), bytes);
        EXPECT_TRUE(lfs::core::base64_decode("").empty());
        EXPECT_EQ(lfs::core::base64_decode("Zg=="), std::vector<std::uint8_t>{'f'});
        EXPECT_THROW(lfs::core::base64_decode("abc"), std::invalid_argument);
        EXPECT_THROW(lfs::core::base64_decode("Z!=="), std::invalid_argument);
        EXPECT_THROW(lfs::core::base64_decode("Zh=="), std::invalid_argument);
        EXPECT_THROW(lfs::core::base64_decode("Zg=a"), std::invalid_argument);
    }

    TEST(NodesCoreMetadata, DeclarationsAreCompleteAndIdentifiersAreExact) {
        NodeTypeRegistry registry;
        register_builtin_nodes(registry);
        for (const auto& type : registry.list()) {
            SCOPED_TRACE(type->id);
            EXPECT_NE(type->description, type->label);
            EXPECT_TRUE(type->description.ends_with("."));
            for (const auto& property : type->properties) {
                EXPECT_EQ(property.identifier.find(' '), std::string::npos);
                if (property.kind == PropertyKind::Enum) {
                    EXPECT_FALSE(property.items.empty());
                    EXPECT_NE(std::ranges::find(property.items, property.default_value.get<std::string>()),
                              property.items.end());
                    for (const auto& item : property.items)
                        EXPECT_EQ(item.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_"),
                                  std::string::npos);
                }
            }
            for (const auto& socket : type->inputs)
                if (socket.type == FLOAT_SOCKET || socket.type == INT_SOCKET ||
                    socket.type == VECTOR_SOCKET || socket.type == COLOUR_SOCKET)
                    EXPECT_TRUE(socket.step.has_value()) << socket.identifier;
        }
        NodeTree tree(registry);
        tree.add_node("lfs.transform_geometry", "Transform");
        EXPECT_FALSE(tree.add_link({tree.input_node().name, "geometry", "Transform", "Geometry"}));
        EXPECT_FALSE(tree.add_link({tree.input_node().name, "Geometry", "Transform", "geometry"}));
    }

    TEST(NodesCoreMetadata, UsefulSocketDefaultsAndRangesAreDeclaredLocally) {
        NodeTypeRegistry registry;
        register_builtin_nodes(registry);
        const auto socket = [&](std::string_view type_id, std::string_view identifier) -> const SocketDecl& {
            const auto type = registry.find(type_id);
            EXPECT_NE(type, nullptr);
            const auto found = std::ranges::find(type->inputs, identifier, &SocketDecl::identifier);
            EXPECT_NE(found, type->inputs.end());
            return *found;
        };
        const auto property =
            [&](std::string_view type_id, std::string_view identifier) -> const PropertyDecl& {
            const auto type = registry.find(type_id);
            EXPECT_NE(type, nullptr);
            const auto found = std::ranges::find(type->properties, identifier, &PropertyDecl::identifier);
            EXPECT_NE(found, type->properties.end());
            return *found;
        };
        EXPECT_FLOAT_EQ(*socket("lfs.sharpen", "Amount").default_value.get_if<float>(), 0.25f);
        EXPECT_EQ(socket("lfs.sharpen", "Amount").min, 0.0);
        EXPECT_EQ(socket("lfs.sharpen", "Amount").max, 0.95);
        EXPECT_FLOAT_EQ(*socket("lfs.remove_floaters", "Min Opacity").default_value.get_if<float>(), 0.02f);
        EXPECT_FLOAT_EQ(*socket("lfs.remove_floaters", "Isolation Radius").default_value.get_if<float>(),
                        3.0f);
        EXPECT_FLOAT_EQ(*socket("lfs.scale_clamp", "Max Aspect").default_value.get_if<float>(), 16.0f);
        EXPECT_EQ(socket("lfs.scale_clamp", "Max Aspect").min, 1.0);
        EXPECT_EQ(socket("lfs.colour_correct", "Midpoint").min, 0.01);
        EXPECT_EQ(socket("lfs.colour_correct", "Midpoint").max, 0.99);
        EXPECT_EQ(socket("lfs.recolour", "Weight").min, 0.0);
        EXPECT_EQ(socket("lfs.recolour", "Weight").max, 1.0);
        EXPECT_EQ(socket("lfs.decimate", "Keep Fraction").min, 0.001);
        EXPECT_EQ(socket("lfs.decimate", "Keep Fraction").max, 1.0);
        EXPECT_FALSE(property("lfs.colour_correct", "auto_range").default_value.get<bool>());
        EXPECT_FALSE(property("lfs.scale_clamp", "include_flat").default_value.get<bool>());
        EXPECT_TRUE(property("lfs.recolour", "keep_shading").default_value.get<bool>());
        EXPECT_TRUE(property("lfs.recolour", "fade_view_dependent").default_value.get<bool>());
        EXPECT_FLOAT_EQ(*socket("lfs.colour_key", "Tolerance").default_value.get_if<float>(), 0.1f);
        EXPECT_FLOAT_EQ(*socket("lfs.points_to_splats", "Opacity").default_value.get_if<float>(), 0.9f);
    }

    TEST_P(NodesCore, SingleValueInputsRejectAttributeFieldsAndClampConstants) {
        NodeTree tree(registry_);
        tree.add_node("lfs.position", "Position");
        tree.add_node("lfs.transform_geometry", "Transform");
        ASSERT_TRUE(tree.add_link({"Position", "Position", "Transform", "Translation"}));
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Transform", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Transform", "Geometry", tree.output_node().name, "Geometry"}));
        auto rejected = evaluate(tree, {splats(), {}, 1});
        EXPECT_FALSE(rejected.ok);
        ASSERT_TRUE(rejected.errors.contains("Transform"));
        EXPECT_NE(rejected.errors.at("Transform").find("single value"), std::string::npos);

        tree.remove_link({"Position", "Position", "Transform", "Translation"});
        tree.add_node("lfs.vector_math", "Constant");
        tree.find_node("Constant")->input_values["A"] = glm::vec3(1, 2, 3);
        ASSERT_TRUE(tree.add_link({"Constant", "Vector", "Transform", "Translation"}));
        auto accepted = evaluate(tree, {splats(), {}, 1});
        ASSERT_TRUE(accepted.ok);
        auto positions = host<float>(accepted.geometry.splats->means);
        EXPECT_NEAR(positions[0], 1, 1e-5f);
        EXPECT_NEAR(positions[1], 2, 1e-5f);
        EXPECT_NEAR(positions[2], 3, 1e-5f);
        auto clamped = single("lfs.set_sh_degree", splats(), [](Node& node) {
            node.input_values["Degree"] = std::int64_t(99);
        });
        ASSERT_TRUE(clamped.ok);
        EXPECT_EQ(clamped.geometry.splats->sh_degree, 3);
    }

    TEST_P(NodesCore, EulerXYZAppliesXFirstForTransformAndSelection) {
        auto geometry = splats(0);
        geometry.splats->means = tensor({1, 0, 0, 1, 0, 0, 1, 0, 0}, {3, 3});
        geometry.points = PointsComponent{geometry.splats->means, Tensor::ones({3, 3}, device()), {}};
        const auto transformed = single("lfs.transform_geometry", geometry, [](Node& node) {
            node.input_values["Rotation"] = glm::vec3(90, 90, 0);
        });
        ASSERT_TRUE(transformed.ok);
        for (const auto& positions :
             {transformed.geometry.splats->means, transformed.geometry.points->positions}) {
            const auto values = host<float>(positions);
            EXPECT_NEAR(values[0], 0, 1e-5f);
            EXPECT_NEAR(values[1], 0, 1e-5f);
            EXPECT_NEAR(values[2], -1, 1e-5f);
        }
        geometry.splats->means = tensor({0, 0, -1, 1, 0, 0, 0, 1, 0}, {3, 3});
        for (const auto& id : {"lfs.box_selection", "lfs.ellipsoid_selection"}) {
            const auto selected = field_result(id, "Selection", FLOAT_SOCKET, geometry, [id](Node& node) {
                node.input_values["Rotation"] = glm::vec3(90, 90, 0);
                node.input_values[id == std::string("lfs.box_selection") ? "Size" : "Radii"] =
                    glm::vec3(4, 0.2f, 0.2f);
            });
            EXPECT_EQ(host<float>(selected), (std::vector<float>{1, 0, 0}));
        }
    }

    TEST_P(NodesCore, PaintSelectionSamplesOverlapEraseAndMatchBruteForceAabb) {
        auto geometry = splats(0);
        geometry.splats->means = tensor({0.0f, 0.0f, 0.0f,
                                         0.5f, 0.0f, 0.0f,
                                         0.75f, 0.0f, 0.0f,
                                         1.25f, 0.0f, 0.0f,
                                         10.0f, 10.0f, 10.0f,
                                         -20.0f, 4.0f, 7.0f},
                                        {6, 3});
        const nlohmann::json strokes = {
            {{0.0f, 0.0f, 0.0f, 1.0f, 1.0f}, {1.0f, 0.0f, 0.0f, 1.0f, 0.6f}},
            {{0.5f, 0.0f, 0.0f, 0.25f, 0.0f}},
        };
        const auto selected = field_result(
            "lfs.paint_selection", "Selection", FLOAT_SOCKET, geometry, [&](Node& node) {
                node.properties["data"] = strokes;
                node.input_values["Softness"] = 1.0f;
            });
        const auto actual = host<float>(selected);

        const std::vector<glm::vec3> positions{{0, 0, 0}, {0.5f, 0, 0}, {0.75f, 0, 0}, {1.25f, 0, 0}, {10, 10, 10}, {-20, 4, 7}};
        std::vector<float> brute;
        for (const auto& position : positions) {
            float selected = 0.0f;
            for (const auto& stroke : strokes) {
                float paint = 0.0f;
                float erase = 0.0f;
                for (const auto& sample : stroke) {
                    const glm::vec3 centre(sample[0].get<float>(), sample[1].get<float>(), sample[2].get<float>());
                    const float weight = std::clamp(1.0f - glm::distance(position, centre) / sample[3].get<float>(), 0.0f, 1.0f);
                    if (sample[4].get<float>() == 0.0f)
                        erase = std::max(erase, weight);
                    else
                        paint = std::max(paint, sample[4].get<float>() * weight);
                }
                selected = std::min(std::max(selected, paint), 1.0f - erase);
            }
            brute.push_back(selected);
        }
        ASSERT_EQ(actual.size(), brute.size());
        for (std::size_t index = 0; index < brute.size(); ++index)
            EXPECT_NEAR(actual[index], brute[index], 1e-5f) << index;
        EXPECT_GT(actual[2], 0.0f);       // overlapping paint samples take their maximum
        EXPECT_FLOAT_EQ(actual[1], 0.0f); // the erase sample removes its centre
        EXPECT_FLOAT_EQ(actual[4], 0.0f); // rejected by the strokes' AABB
    }

    TEST_P(NodesCore, PaintSelectionStrokesApplyInOrder) {
        auto geometry = splats(0);
        geometry.splats->means = tensor({0, 0, 0, 3, 0, 0}, {2, 3});
        const nlohmann::json paint = {{0.0f, 0.0f, 0.0f, 1.0f, 1.0f}, {3.0f, 0.0f, 0.0f, 1.0f, 1.0f}};
        const nlohmann::json erase = {{0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, {3.0f, 0.0f, 0.0f, 1.0f, 0.0f}};
        const nlohmann::json repaint = {{0.0f, 0.0f, 0.0f, 1.0f, 1.0f}};
        const auto selection = [&](const nlohmann::json& strokes) {
            return host<float>(field_result("lfs.paint_selection", "Selection", FLOAT_SOCKET, geometry,
                                            [&](Node& node) {
                                                node.properties["data"] = strokes;
                                                node.input_values["Softness"] = 0.0f;
                                            }));
        };
        EXPECT_EQ(selection(nlohmann::json::array({paint})), (std::vector<float>{1.0f, 1.0f}));
        EXPECT_EQ(selection(nlohmann::json::array({paint, erase})), (std::vector<float>{0.0f, 0.0f}));
        // Painting over an erased patch selects it again; only the repainted point returns.
        EXPECT_EQ(selection(nlohmann::json::array({paint, erase, repaint})), (std::vector<float>{1.0f, 0.0f}));
        // Erasing first and painting later is not undone by the earlier erase.
        EXPECT_EQ(selection(nlohmann::json::array({erase, paint})), (std::vector<float>{1.0f, 1.0f}));
    }

    TEST_P(NodesCore, PaintSelectionChunksMatchBruteForceAcrossDistantStrokes) {
        // A long stroke spans several sample chunks; the distant stroke widens the strokes' bounds
        // over every point, so only the per-chunk culling keeps the work local.
        nlohmann::json strokes = nlohmann::json::array();
        nlohmann::json line = nlohmann::json::array();
        for (int index = 0; index < 150; ++index)
            line.push_back({0.1f * static_cast<float>(index), 0.05f * static_cast<float>(index % 7), 0.0f, 0.3f,
                            index % 50 == 25 ? 0.0f : 1.0f});
        strokes.push_back(line);
        strokes.push_back({{40.0f, 40.0f, 0.0f, 2.0f, 1.0f}, {41.0f, 40.0f, 0.0f, 2.0f, 0.0f}});
        std::vector<glm::vec3> positions;
        std::vector<float> flat;
        for (int x = -2; x < 44; ++x)
            for (int y = -1; y < 42; y += 3) {
                const glm::vec3 position(0.37f * static_cast<float>(x), 0.11f * static_cast<float>(y), 0.0f);
                positions.push_back(position);
                flat.insert(flat.end(), {position.x, position.y, position.z});
            }
        positions.push_back({40.5f, 40.0f, 0.0f});
        flat.insert(flat.end(), {40.5f, 40.0f, 0.0f});
        auto geometry = splats(0);
        geometry.splats->means = tensor(flat, {positions.size(), 3});
        const auto actual = host<float>(field_result(
            "lfs.paint_selection", "Selection", FLOAT_SOCKET, geometry, [&](Node& node) {
                node.properties["data"] = strokes;
                node.input_values["Softness"] = 0.5f;
            }));
        ASSERT_EQ(actual.size(), positions.size());
        std::size_t selected_count = 0;
        for (std::size_t point = 0; point < positions.size(); ++point) {
            float selected = 0.0f;
            for (const auto& stroke : strokes) {
                float paint = 0.0f;
                float erase = 0.0f;
                for (const auto& sample : stroke) {
                    const glm::vec3 centre(sample[0].get<float>(), sample[1].get<float>(), sample[2].get<float>());
                    const float distance = glm::distance(positions[point], centre) / sample[3].get<float>();
                    const float weight = std::clamp((1.0f - distance) / 0.5f, 0.0f, 1.0f);
                    if (sample[4].get<float>() == 0.0f)
                        erase = std::max(erase, weight);
                    else
                        paint = std::max(paint, sample[4].get<float>() * weight);
                }
                selected = std::min(std::max(selected, paint), 1.0f - erase);
            }
            EXPECT_NEAR(actual[point], selected, 1e-5f) << point;
            selected_count += selected > 0.0f;
        }
        EXPECT_GT(selected_count, 20u);
    }

    TEST_P(NodesCore, CancellationRaisedInsideANodeStopsWithoutAnError) {
        bool stop = false;
        NodeTypeInfo info;
        info.id = "test.cancel";
        info.inputs.push_back({"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)});
        info.outputs.push_back({"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)});
        info.evaluate = [&](NodeContext& context) {
            stop = true;
            throw_if_evaluation_cancelled();
            context.set_output("Geometry", context.input("Geometry"));
        };
        ASSERT_TRUE(registry_.register_type(std::move(info)));
        NodeTree tree(registry_);
        const Node& node = tree.add_node("test.cancel");
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", node.name, "Geometry"}));
        ASSERT_TRUE(tree.add_link({node.name, "Geometry", tree.output_node().name, "Geometry"}));
        EvalCache cache;
        EvalControl control;
        control.cancelled = [&] { return stop; };
        const auto result = evaluate(tree, {splats(), {}, 7}, nullptr, &cache, control);
        EXPECT_TRUE(result.cancelled);
        EXPECT_FALSE(result.ok);
        EXPECT_TRUE(result.errors.empty());
        EXPECT_FALSE(cache.nodes.contains(node.name));
        // Outside an evaluation there is nothing to cancel.
        EXPECT_NO_THROW(throw_if_evaluation_cancelled());
    }

    TEST_P(NodesCore, ConsumersKeepTheSelectionTheyEvaluated) {
        NodeTree tree(registry_);
        Node& box = tree.add_node("lfs.box_selection");
        box.input_values["Centre"] = glm::vec3(1, 0, 0);
        box.input_values["Size"] = glm::vec3(0.5f);
        const Node& opacity = tree.add_node("lfs.set_opacity");
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", opacity.name, "Geometry"}));
        ASSERT_TRUE(tree.add_link({box.name, "Selection", opacity.name, "Selection"}));
        ASSERT_TRUE(tree.add_link({opacity.name, "Geometry", tree.output_node().name, "Geometry"}));
        EvalCache cache;
        ASSERT_TRUE(evaluate(tree, {splats(), {}, 7}, nullptr, &cache).ok);
        const auto& consumer = cache.nodes.at(opacity.name);
        ASSERT_TRUE(consumer.selection);
        EXPECT_EQ(consumer.selection->mask.cpu().to_vector_bool(), (std::vector<bool>{false, true, false}));
    }

    TEST_P(NodesCore, ConsumersRecordOnlySelectionsEvaluatedOnTheirInput) {
        NodeTypeInfo info;
        info.id = "test.select_moved";
        info.inputs = {{"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)},
                       {"Selection", "Selection", std::string(FLOAT_SOCKET), 1.0f, {}, {}, {}, true}};
        info.outputs = {{"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)}};
        info.evaluate = [](NodeContext& context) {
            auto geometry = *context.input("Geometry").get_if<Geometry>();
            geometry.splats->means = geometry.splats->means + 1.0f;
            const auto domain = field_context(*geometry.splats);
            context.record_selection(domain, context.evaluate_field("Selection", domain).ge(0.5f));
            context.set_output("Geometry", geometry);
        };
        registry_.unregister_type(info.id);
        ASSERT_TRUE(registry_.register_type(std::move(info)));
        NodeTree tree(registry_);
        Node& box = tree.add_node("lfs.box_selection");
        box.input_values["Centre"] = glm::vec3(1, 0, 0);
        box.input_values["Size"] = glm::vec3(0.5f);
        const Node& node = tree.add_node("test.select_moved");
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", node.name, "Geometry"}));
        ASSERT_TRUE(tree.add_link({box.name, "Selection", node.name, "Selection"}));
        ASSERT_TRUE(tree.add_link({node.name, "Geometry", tree.output_node().name, "Geometry"}));
        EvalCache cache;
        ASSERT_TRUE(evaluate(tree, {splats(), {}, 7}, nullptr, &cache).ok);
        // The mask covers moved positions, so it cannot stand in for a preview on the input.
        EXPECT_FALSE(cache.nodes.at(node.name).selection);
        registry_.unregister_type("test.select_moved");
    }

    TEST_P(NodesCore, SharedFieldsReadAttributesAsTheyAreAfterAnOverwrite) {
        NodeTree tree(registry_);
        // Node references do not survive later add_node calls; keep names.
        const auto store = [&](const std::string& node_name, const std::string& name, const std::optional<float> value) {
            Node& node = tree.add_node("lfs.store_named_attribute", node_name);
            node.input_values["Name"] = name;
            if (value)
                node.input_values["Value"] = *value;
            return node_name;
        };
        const auto first = store("First", "a", 1.0f);
        const auto before = store("Before", "b", std::nullopt);
        const auto overwrite = store("Overwrite", "a", 2.0f);
        const auto after = store("After", "c", std::nullopt);
        tree.add_node("lfs.named_attribute", "Read").input_values["Name"] = std::string("a");
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", first, "Geometry"}));
        ASSERT_TRUE(tree.add_link({first, "Geometry", before, "Geometry"}));
        ASSERT_TRUE(tree.add_link({before, "Geometry", overwrite, "Geometry"}));
        ASSERT_TRUE(tree.add_link({overwrite, "Geometry", after, "Geometry"}));
        ASSERT_TRUE(tree.add_link({after, "Geometry", tree.output_node().name, "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Read", "Attribute", before, "Value"}));
        ASSERT_TRUE(tree.add_link({"Read", "Attribute", after, "Value"}));
        // Every copy of a mesh component shares its mesh, so a memo keyed on the mesh alone serves the
        // first read of "a" to the second.
        const auto mesh = torus(4, 3);
        const auto result = evaluate(tree, {Geometry{std::nullopt, std::nullopt, MeshComponent{mesh}}, {}, 7, device()});
        ASSERT_TRUE(result.ok) << (result.errors.empty() ? "" : result.errors.begin()->second);
        ASSERT_TRUE(result.geometry.mesh);
        const auto& attributes = result.geometry.mesh->attributes;
        const auto vertices = static_cast<size_t>(mesh->vertex_count());
        EXPECT_EQ(host<float>(attributes.at("b")), std::vector<float>(vertices, 1.0f));
        EXPECT_EQ(host<float>(attributes.at("c")), std::vector<float>(vertices, 2.0f));
    }

    TEST_P(NodesCore, PaintSelectionSoftnessEndpointsAndInvert) {
        auto geometry = splats(0);
        geometry.splats->means = tensor({0, 0, 0, 0.75f, 0, 0, 1.1f, 0, 0}, {3, 3});
        const auto configure = [](Node& node, const float softness, const bool invert) {
            node.properties["data"] = {{{0.0f, 0.0f, 0.0f, 1.0f, 1.0f}}};
            node.properties["invert"] = invert;
            node.input_values["Softness"] = softness;
        };
        const auto hard = field_result("lfs.paint_selection", "Selection", FLOAT_SOCKET, geometry,
                                       [&](Node& node) { configure(node, 0.0f, false); });
        EXPECT_EQ(host<float>(hard), (std::vector<float>{1.0f, 1.0f, 0.0f}));
        const auto soft = field_result("lfs.paint_selection", "Selection", FLOAT_SOCKET, geometry,
                                       [&](Node& node) { configure(node, 1.0f, false); });
        const auto soft_values = host<float>(soft);
        EXPECT_NEAR(soft_values[0], 1.0f, 1e-5f);
        EXPECT_NEAR(soft_values[1], 0.25f, 1e-5f);
        EXPECT_NEAR(soft_values[2], 0.0f, 1e-5f);
        const auto inverted = field_result("lfs.paint_selection", "Selection", FLOAT_SOCKET, geometry,
                                           [&](Node& node) { configure(node, 1.0f, true); });
        const auto inverted_values = host<float>(inverted);
        for (std::size_t index = 0; index < soft_values.size(); ++index)
            EXPECT_NEAR(inverted_values[index], 1.0f - soft_values[index], 1e-5f);
    }

    TEST_P(NodesCore, ColourCorrectKnownSaturationAndWhiteBalanceMatrices) {
        auto geometry = splats(3);
        auto grey = single("lfs.colour_correct", geometry, [](Node& node) {
            node.input_values["Saturation"] = 0.0f;
        });
        ASSERT_TRUE(grey.ok);
        constexpr float c0 = 0.28209479177387814f;
        for (const auto& pair :
             {std::pair{geometry.splats->sh0 * c0 + 0.5f, grey.geometry.splats->sh0 * c0 + 0.5f},
              std::pair{geometry.splats->shN, grey.geometry.splats->shN}}) {
            const auto original = host<float>(pair.first);
            const auto corrected = host<float>(pair.second);
            for (size_t row = 0; row < original.size(); row += 3) {
                const float luminance =
                    original[row] * 0.2126f + original[row + 1] * 0.7152f + original[row + 2] * 0.0722f;
                for (size_t channel = 0; channel < 3; ++channel)
                    EXPECT_NEAR(corrected[row + channel], luminance, 2e-5f);
            }
        }
        auto balanced = single("lfs.colour_correct", geometry, [](Node& node) {
            node.input_values["Temperature"] = 0.25f;
            node.input_values["Tint"] = -0.5f;
        });
        ASSERT_TRUE(balanced.ok);
        const auto original = host<float>(geometry.splats->shN);
        const auto corrected = host<float>(balanced.geometry.splats->shN);
        const float gains[] = {1.25f, 0.5f, 0.75f};
        for (size_t index = 0; index < original.size(); ++index)
            EXPECT_NEAR(corrected[index], original[index] * gains[index % 3], 2e-5f);
    }

    TEST_P(NodesCore, ColourCorrectIdentityPreservesNegativeBaseAndShBitExactly) {
        auto geometry = splats(3);
        geometry.splats->sh0 = tensor({-4, -3, -2, -1, 0, 1, 2, 3, 4}, {3, 3});
        const auto sh0 = host<float>(geometry.splats->sh0);
        const auto shn = host<float>(geometry.splats->shN);
        const auto result = single("lfs.colour_correct", geometry);
        ASSERT_TRUE(result.ok);
        EXPECT_EQ(host<float>(result.geometry.splats->sh0), sh0);
        EXPECT_EQ(host<float>(result.geometry.splats->shN), shn);
    }

    TEST_P(NodesCore, ScaleClampBlendsNeedleWithoutChangingShortAxes) {
        auto geometry = splats();
        geometry.splats->scaling = tensor({-3, 0, 3, -3, 0, 3, -3, 0, 3}, {3, 3});
        auto result = single("lfs.scale_clamp", geometry, [](Node& node) {
            node.input_values["Max Aspect"] = std::exp(1.0f);
            node.input_values["Selection"] = 0.5f;
        });
        ASSERT_TRUE(result.ok);
        const auto values = host<float>(result.geometry.splats->scaling);
        for (size_t index = 0; index < values.size(); index += 3) {
            EXPECT_FLOAT_EQ(values[index], -3);
            EXPECT_NEAR(values[index + 1], 0, 1e-5f);
            EXPECT_NEAR(values[index + 2], 2, 1e-5f);
        }
    }

    TEST_P(NodesCore, ScaleClampShortensNeedlesAndPreservesFlatSplatsByDefault) {
        const float log4 = std::log(4.0f);
        const float log16 = std::log(16.0f);
        const float log64 = std::log(64.0f);
        auto geometry = splats();
        geometry.splats->scaling = tensor({log64, 0, 0, log64, log64, 0, log4, 0, 0}, {3, 3});
        const auto before = host<float>(geometry.splats->scaling);

        const auto result = single("lfs.scale_clamp", geometry);
        ASSERT_TRUE(result.ok);
        const auto values = host<float>(result.geometry.splats->scaling);
        EXPECT_NEAR(values[0], log16, 1e-5f);
        EXPECT_FLOAT_EQ(values[1], before[1]);
        EXPECT_FLOAT_EQ(values[2], before[2]);
        EXPECT_EQ(std::vector<float>(values.begin() + 3, values.end()),
                  std::vector<float>(before.begin() + 3, before.end()));

        const auto with_flat = single("lfs.scale_clamp", geometry, [](Node& node) {
            node.properties["include_flat"] = true;
        });
        ASSERT_TRUE(with_flat.ok);
        const auto flat_values = host<float>(with_flat.geometry.splats->scaling);
        EXPECT_LE(flat_values[3] - flat_values[5], log16 + 1e-5f);
    }

    TEST_P(NodesCore, ScaleClampHandlesEmptyInput) {
        Geometry empty;
        empty.splats = SplatsComponent{Tensor::empty({0, 3}, device()),
                                       Tensor::empty({0, 3}, device()),
                                       Tensor::empty({0, 3, 3}, device()),
                                       Tensor::empty({0, 3}, device()),
                                       Tensor::empty({0, 4}, device()),
                                       Tensor::empty({0}, device()),
                                       1};
        const auto result = single("lfs.scale_clamp", empty);
        ASSERT_TRUE(result.ok);
        ASSERT_TRUE(result.geometry.splats);
        EXPECT_EQ(result.geometry.splats->scaling.shape(), TensorShape({0, 3}));
    }

    TEST_P(NodesCore, ScaleClampCollapsedAxesRespectSelectionsAndNonzeroAspect) {
        const float collapsed = -std::numeric_limits<float>::infinity();
        for (bool flat : {false, true}) {
            for (float weight : {0.0f, 0.25f, 1.0f}) {
                for (int shift = 0; shift < 3; ++shift) {
                    auto geometry = splats();
                    std::vector<float> original{collapsed, collapsed, collapsed, 6, collapsed, collapsed, 6, 0, collapsed};
                    for (size_t row = 0; row < original.size(); row += 3)
                        std::rotate(original.begin() + row, original.begin() + row + shift, original.begin() + row + 3);
                    geometry.splats->scaling = tensor(original, {3, 3});
                    const auto result = single("lfs.scale_clamp", geometry, [&](Node& node) {
                        node.input_values["Max Aspect"] = std::exp(2.0f);
                        node.input_values["Selection"] = weight;
                        node.properties["include_flat"] = flat;
                    });
                    ASSERT_TRUE(result.ok);
                    const auto actual = host<float>(result.geometry.splats->scaling);
                    for (size_t i = 0; i < original.size(); ++i) {
                        if (original[i] == collapsed) {
                            EXPECT_EQ(actual[i], collapsed) << i;
                        } else {
                            const float target = i < 6 ? original[i] : flat ? std::clamp(original[i], 2.0f, 4.0f)
                                                                            : std::min(original[i], 2.0f);
                            EXPECT_NEAR(actual[i], original[i] * (1 - weight) + target * weight, 1e-5f) << i;
                        }
                    }
                }
            }
        }
    }

    TEST_P(NodesCore, SharpenSeed10CoverageNearSaturationMatchesDoubleReference) {
        auto geometry = splats();
        const std::vector<float> logits{-1.4096522f, -100.0f, 100.0f};
        geometry.splats->opacity = tensor(logits, {3});
        geometry.splats->scaling = tensor({-80, 80, -std::numeric_limits<float>::infinity(), 1, 2, 3, -3, -2, -1}, {3, 3});
        NodeTree tree(registry_);
        auto& boolean = tree.add_node("lfs.boolean_math", "Boolean");
        boolean.properties["operation"] = "and";
        boolean.input_values["A"] = true;
        boolean.input_values["B"] = false;
        auto& distance = tree.add_node("lfs.distance", "Distance");
        distance.properties["mode"] = "line";
        distance.input_values["Vector"] = glm::vec3(-0.6279172757530982f, 0.626565338433462f, 1.8062866772833788f);
        distance.input_values["Point"] = glm::vec3(-1.9800320780380165f, 1.6760289130573076f, -0.9195863934260489f);
        distance.input_values["Direction"] = glm::vec3(1.4300321781511167f, -0.579142019187024f, -1.1000398726153522f);
        distance.input_values["Normal"] = glm::vec3(-1.6785224615751129f, 0.4252425189733855f, 0.3258174906958726f);
        tree.add_node("lfs.set_scale", "Scale");
        auto& sharpen = tree.add_node("lfs.sharpen", "Sharpen");
        constexpr float amount = 0.8037107154414453f;
        sharpen.input_values["Amount"] = amount;
        sharpen.properties["keep_coverage"] = true;
        tree.add_node("lfs.splats_to_points", "Points");
        auto& clumps = tree.add_node("lfs.remove_clumps", "Clumps");
        clumps.input_values["Selection"] = 0.17230541194853133f;
        clumps.input_values["Radius"] = 1.0796938448451159f;
        clumps.input_values["Min Size"] = std::int64_t(2);
        clumps.properties["delete"] = false;
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Scale", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Boolean", "Result", "Scale", "Selection"}));
        ASSERT_TRUE(tree.add_link({"Distance", "Distance", "Scale", "Scale"}));
        ASSERT_TRUE(tree.add_link({"Scale", "Geometry", "Sharpen", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Distance", "Distance", "Sharpen", "Selection"}));
        ASSERT_TRUE(tree.add_link({"Sharpen", "Geometry", "Points", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Points", "Geometry", "Clumps", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Clumps", "Geometry", tree.output_node().name, "Geometry"}));
        const auto result = evaluate(tree, {geometry, {}, 1});
        ASSERT_TRUE(result.ok);
        const auto actual = host<float>(result.geometry.splats->opacity);
        for (size_t row = 0; row < logits.size(); ++row) {
            const double alpha = std::clamp((1.0 / (1.0 + std::exp(-double(logits[row])))) / double(1.0f - amount),
                                            double(1e-6f), double(1.0f - 1e-6f));
            EXPECT_NEAR(actual[row], std::log(alpha / (1.0 - alpha)), 1e-5) << "row=" << row;
        }
    }

    TEST_P(NodesCore, CollapsedScalesSharpenAndSetScaleSelectionEndpoints) {
        const float collapsed = -std::numeric_limits<float>::infinity();
        auto geometry = splats();
        const std::vector<float> original{collapsed, collapsed, collapsed, 2, collapsed, collapsed, 2, 0, collapsed};
        geometry.splats->scaling = tensor(original, {3, 3});
        for (float weight : {0.0f, 0.5f, 1.0f}) {
            for (float amount : {0.0f, 0.25f, 0.95f}) {
                const auto result = single("lfs.sharpen", geometry, [&](Node& node) {
                    node.input_values["Selection"] = weight;
                    node.input_values["Amount"] = amount;
                });
                ASSERT_TRUE(result.ok);
                const auto actual = host<float>(result.geometry.splats->scaling);
                for (size_t i = 0; i < original.size(); ++i) {
                    if (original[i] == collapsed)
                        EXPECT_EQ(actual[i], collapsed);
                    else
                        EXPECT_NEAR(actual[i], original[i] + weight * std::log(1 - amount), 1e-5f);
                }
                for (float opacity : host<float>(result.geometry.splats->opacity))
                    EXPECT_TRUE(std::isfinite(opacity));
            }
            const auto result = single("lfs.set_scale", geometry, [&](Node& node) {
                node.input_values["Selection"] = weight;
                node.input_values["Scale"] = glm::vec3(2, 0, -1);
            });
            ASSERT_TRUE(result.ok);
            const auto actual = host<float>(result.geometry.splats->scaling);
            for (size_t i = 0; i < original.size(); ++i) {
                const float target = i % 3 == 0 ? std::log(2.0f) : collapsed;
                if (weight == 0)
                    EXPECT_EQ(actual[i], original[i]);
                else if (weight == 1)
                    EXPECT_EQ(actual[i], target);
                else if (target == collapsed || original[i] == collapsed)
                    EXPECT_EQ(actual[i], collapsed);
                else
                    EXPECT_NEAR(actual[i], original[i] * (1 - weight) + target * weight, 1e-5f);
            }
        }
    }

    TEST_P(NodesCore, SimplifyPreservesCollapsedCovarianceAndSingletons) {
        const float collapsed = -std::numeric_limits<float>::infinity();
        for (int rank = 0; rank < 3; ++rank) {
            auto geometry = splats(3);
            geometry.splats->means = tensor({2, 4, 6, 2, 4, 6, 2, 4, 6}, {3, 3});
            geometry.splats->opacity = Tensor::zeros({3}, device());
            std::vector<float> scales(9, collapsed);
            for (size_t i = 0; i < scales.size(); ++i)
                if (int(i % 3) < rank)
                    scales[i] = std::log(0.1f);
            geometry.splats->scaling = tensor(scales, {3, 3});
            const auto result = single("lfs.simplify", geometry, [](Node& node) { node.input_values["Ratio"] = 0.25f; });
            ASSERT_TRUE(result.ok) << (result.errors.empty() ? "" : result.errors.begin()->second);
            ASSERT_EQ(result.geometry.splats->means.size(0), 1u);
            const auto actual = host<float>(result.geometry.splats->scaling);
            EXPECT_EQ(std::count(actual.begin(), actual.end(), collapsed), 3 - rank);
            for (float value : actual)
                if (value != collapsed)
                    EXPECT_NEAR(value, std::log(0.1f), 1e-5f);
            const auto means = host<float>(result.geometry.splats->means);
            for (size_t axis = 0; axis < 3; ++axis)
                EXPECT_NEAR(means[axis], 2.0f * (axis + 1), 1e-5f);
            auto activated = result.geometry;
            activated.splats->scaling = activated.splats->scaling.exp();
            expect_finite(activated);
        }
        // One merged pair and one singleton: neither may acquire thickness.
        auto geometry = splats();
        geometry.splats->means = tensor({1, 0, 0, 1, 0, 0, 100, 0, 0}, {3, 3});
        geometry.splats->opacity = Tensor::zeros({3}, device());
        geometry.splats->scaling = tensor(std::vector<float>(9, collapsed), {3, 3});
        const auto result = single("lfs.simplify", geometry, [](Node& node) { node.input_values["Ratio"] = 0.5f; });
        ASSERT_TRUE(result.ok);
        ASSERT_EQ(result.geometry.splats->means.size(0), 2u);
        EXPECT_EQ(host<float>(result.geometry.splats->scaling), std::vector<float>(6, collapsed));
    }

    TEST_P(NodesCore, SimplifyRotatedCollapsedSplatsDoNotGainThickness) {
        const float collapsed = -std::numeric_limits<float>::infinity();
        auto geometry = splats(3);
        geometry.splats->means = tensor({2, 4, 6, 2, 4, 6, 2, 4, 6}, {3, 3});
        geometry.splats->opacity = Tensor::zeros({3}, device());
        geometry.splats->rotation = tensor({0.8f, 0, 0.6f, 0, 0.8f, 0, 0.6f, 0, 0.8f, 0, 0.6f, 0}, {3, 4});
        geometry.splats->scaling = tensor({std::log(0.1f), collapsed, collapsed, std::log(0.1f), collapsed, collapsed, std::log(0.1f), collapsed, collapsed}, {3, 3});
        const auto result = single("lfs.simplify", geometry, [](Node& node) { node.input_values["Ratio"] = 0.25f; });
        ASSERT_TRUE(result.ok);
        const auto actual = host<float>(result.geometry.splats->scaling);
        EXPECT_EQ(std::count(actual.begin(), actual.end(), collapsed), 2);
        for (float value : actual)
            if (value != collapsed)
                EXPECT_NEAR(value, std::log(0.1f), 1e-5f);
        // Different centres add a genuine second covariance direction. The
        // null-space guard must not indiscriminately copy the input zero mask.
        geometry.splats->means = tensor({0, 0, 0, 0, 1, 0, 0, 2, 0}, {3, 3});
        const auto spread = single("lfs.simplify", geometry, [](Node& node) { node.input_values["Ratio"] = 0.25f; });
        ASSERT_TRUE(spread.ok);
        const auto spread_scales = host<float>(spread.geometry.splats->scaling);
        EXPECT_EQ(std::count(spread_scales.begin(), spread_scales.end(), collapsed), 1);
        for (float value : spread_scales)
            EXPECT_TRUE(value == collapsed || std::isfinite(value));
    }

    TEST_P(NodesCore, InsideMeshHandlesNonConvexTorusAndOutsideBounds) {
        auto geometry = splats();
        geometry.splats->means = tensor({1.5f, 0, 0, 0, 0, 0, 20, 0, 0}, {3, 3});
        const auto selected =
            field_result("lfs.inside_mesh", "Selection", BOOL_SOCKET, geometry, [&](Node& node) {
                node.input_values["Mesh"] = geometry_from_mesh(torus());
            });
        EXPECT_EQ(host<std::uint8_t>(selected), (std::vector<std::uint8_t>{1, 0, 0}));
    }

    TEST_P(NodesCore, NeighbourCountsAndIsolationExcludeSelf) {
        auto geometry = splats();
        geometry.splats->means = tensor({0, 0, 0, 0.1f, 0, 0, 1, 0, 0}, {3, 3});
        const auto counts =
            field_result("lfs.neighbour_count", "Count", INT_SOCKET, geometry, [](Node& node) {
                node.input_values["Radius"] = 0.2f;
            });
        EXPECT_EQ(host<int>(counts), (std::vector<int>{1, 1, 0}));
        auto exact = single("lfs.remove_floaters", geometry, [](Node& node) {
            node.input_values["Isolation Radius"] = 0.2f;
            node.input_values["Min Neighbours"] = std::int64_t(1);
            node.properties["relative_to_size"] = false;
        });
        ASSERT_TRUE(exact.ok);
        EXPECT_EQ(exact.geometry.splats->means.shape()[0], 2u);
        EXPECT_EQ(host<float>(exact.geometry.splats->attributes.at("weight")), (std::vector<float>{10, 20}));
        auto two_required = single("lfs.remove_floaters", geometry, [](Node& node) {
            node.input_values["Isolation Radius"] = 0.2f;
            node.input_values["Min Neighbours"] = std::int64_t(2);
            node.properties["relative_to_size"] = false;
        });
        ASSERT_TRUE(two_required.ok);
        EXPECT_EQ(two_required.geometry.splats->means.shape()[0], 0u);
    }

    // Fails if one collapsed splat (zero size) sets the radius range for all others: relative mode then counted
    // no neighbours anywhere and Remove Floaters deleted every splat.
    TEST_P(NodesCore, RemoveFloatersIgnoresACollapsedSplatInRelativeMode) {
        auto geometry = splats();
        geometry.splats->means = tensor({0, 0, 0, 0.01f, 0, 0, 10, 0, 0}, {3, 3});
        const float small = std::log(0.1f), collapsed = -std::numeric_limits<float>::infinity();
        geometry.splats->scaling = tensor({small, small, small, small, small, small, collapsed, collapsed, collapsed}, {3, 3});
        const auto result = single("lfs.remove_floaters", geometry, [](Node& node) {
            node.input_values["Min Opacity"] = 0.0f;
        });
        ASSERT_TRUE(result.ok);
        EXPECT_EQ(host<float>(result.geometry.splats->attributes.at("weight")), (std::vector<float>{10, 20}));
    }

    // Fails if relative mode measures point clouds by their (absent) size instead of in scene units.
    TEST_P(NodesCore, RelativeNeighbourCountUsesSceneUnitsOnPoints) {
        Tensor value;
        NodeTypeInfo capture;
        capture.id = "test.capture_points";
        capture.inputs = {{"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)},
                          {"Value", "Value", std::string(INT_SOCKET), {}, {}, {}, {}, true}};
        capture.outputs = {{"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)}};
        capture.evaluate = [&](NodeContext& context) {
            const auto source = *context.input("Geometry").get_if<Geometry>();
            const FieldContext domain{Domain::Point, nullptr, &*source.points, nullptr, 101};
            value = context.evaluate_field("Value", domain, INT_SOCKET);
            context.set_output("Geometry", source);
        };
        registry_.unregister_type(capture.id);
        registry_.register_type(std::move(capture));
        PointsComponent points;
        points.positions = tensor({0, 0, 0, 0.5f, 0, 0}, {2, 3});
        points.colors = tensor({1, 1, 1, 1, 1, 1}, {2, 3});
        NodeTree tree(registry_);
        auto& count = tree.add_node("lfs.neighbour_count", "Count");
        count.input_values["Radius"] = 1.0f;
        count.properties["relative_to_size"] = true;
        tree.add_node("test.capture_points", "Capture");
        ASSERT_TRUE(tree.add_link({"Count", "Count", "Capture", "Value"}));
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Capture", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Capture", "Geometry", tree.output_node().name, "Geometry"}));
        const auto result = lfs::nodes::evaluate(tree, {Geometry{std::nullopt, std::move(points), std::nullopt}, {}, 1, device()});
        registry_.unregister_type("test.capture_points");
        ASSERT_TRUE(result.ok) << (result.errors.empty() ? "" : result.errors.begin()->second);
        EXPECT_EQ(host<int>(value), (std::vector<int>{1, 1}));
    }

    TEST_P(NodesCore, RadiusNeighborCountsMatchBruteForceWithCollisionsAndMasks) {
        constexpr size_t count = 257;
        std::mt19937 random(91);
        std::uniform_int_distribution<int> coordinate(-12, 12);
        std::vector<float> xyz(count * 3), refs(count), query(count);
        for (size_t i = 0; i < count; ++i) {
            for (size_t axis = 0; axis < 3; ++axis)
                xyz[3 * i + axis] = coordinate(random) * 0.25f;
            refs[i] = i % 3 == 0 ? 7 : 0;
            query[i] = i % 5 == 0 ? 1 : 0;
        }
        std::copy_n(xyz.begin(), 3, xyz.begin() + 3);
        xyz[6] = std::numeric_limits<float>::infinity();
        const auto points = tensor(xyz, {count, 3});
        const auto references = tensor(refs, {count}).to(lfs::core::DataType::UInt8);
        const auto queries = tensor(query, {count}).gt(0);
        for (const float radius : {0.01f, 1.0f, 20.0f}) {
            for (const int limit : {1, 3, 1000}) {
                std::vector<int> expected(count), masked(count);
                for (size_t i = 0; i < count; ++i) {
                    for (size_t j = 0; j < count; ++j) {
                        float distance = 0;
                        for (size_t axis = 0; axis < 3; ++axis) {
                            const float delta = xyz[3 * i + axis] - xyz[3 * j + axis];
                            distance += delta * delta;
                        }
                        if (i != j && refs[j] && distance <= radius * radius)
                            expected[i] = std::min(limit, expected[i] + 1);
                    }
                    masked[i] = query[i] ? expected[i] : 0;
                }
                const auto actual = lfs::core::radius_neighbor_counts(points, references, radius, limit);
                EXPECT_EQ(actual.dtype(), lfs::core::DataType::Int32);
                EXPECT_EQ(actual.device(), device());
                EXPECT_EQ(host<int>(actual), expected);
                EXPECT_EQ(host<int>(lfs::core::radius_neighbor_counts(points, references.gt(0), radius, limit, &queries)), masked);
            }
        }
        // Cell indices clamp far from the origin; coincident points there still count each other.
        const auto far = tensor({1e9f, 0, 0, 1e9f, 0, 0, 1e9f, 0, 0}, {3, 3});
        EXPECT_EQ(host<int>(lfs::core::radius_neighbor_counts(far, Tensor::full_bool({3}, true, device()), 1.0f, 2)),
                  (std::vector<int>{2, 2, 2}));
        EXPECT_THROW(lfs::core::radius_neighbor_counts(points, references, 1.0f, 0), std::exception);
        const auto empty = lfs::core::radius_neighbor_counts(Tensor::empty({0, 3}, device()), Tensor::full_bool({0}, true, device()), 1.0f, 3);
        EXPECT_EQ(empty.numel(), 0u);
    }

    // Fails if components are cut short (the old label propagation stopped after 64 rounds, splitting long chains),
    // if a point outside the selection still bridges two components, or if a backend labels differently.
    // Fails if the indexed spacing query differs from the full one at the same points, or skips duplicates.
    TEST_P(NodesCore, PerPointRadiusCountsMatchBruteForce) {
        // A dense cluster, a sparse cloud and a far field with radii from a thousandth to tens of units, so
        // tree boxes are skipped, counted whole and opened, around queries inside and outside the references.
        std::mt19937 random(9133);
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        constexpr size_t count = 4000;
        std::vector<float> xyz(count * 3), radii(count), references(count), queries(count);
        for (size_t i = 0; i < count; ++i) {
            const float pick = unit(random);
            const float spread = pick < 0.5f ? 0.05f : pick < 0.8f ? 2.0f
                                                                   : 10.0f;
            const float offset = pick < 0.8f ? 0.0f : 40.0f;
            xyz[i * 3] = offset + spread * normal(random);
            xyz[i * 3 + 1] = spread * normal(random);
            xyz[i * 3 + 2] = spread * normal(random);
            radii[i] = std::exp(std::log(1e-3f) + unit(random) * (std::log(50.0f) - std::log(1e-3f)));
            references[i] = unit(random) < 0.8f ? 1.0f : 0.0f;
            queries[i] = unit(random) < 0.7f ? 1.0f : 0.0f;
        }
        std::copy_n(xyz.begin() + 3, 3, xyz.begin() + 6);
        xyz[30] = std::numeric_limits<float>::quiet_NaN();
        radii[5] = 0.0f;
        radii[6] = -1.0f;
        radii[7] = std::numeric_limits<float>::quiet_NaN();
        radii[8] = std::numeric_limits<float>::infinity();
        radii[9] = 1e-30f;
        const auto finite = [&](size_t i) {
            return std::isfinite(xyz[i * 3]) && std::isfinite(xyz[i * 3 + 1]) && std::isfinite(xyz[i * 3 + 2]);
        };
        // The library's inclusive test, each operation rounded on its own.
        const auto within = [](const float* a, const float* b, const float radius) {
            volatile float x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
            volatile float limit = radius * radius;
            if (limit < std::numeric_limits<float>::min() || !std::isfinite(limit)) {
                x = x / radius;
                y = y / radius;
                z = z / radius;
                limit = 1.0f;
            }
            volatile float xx = x * x, yy = y * y, zz = z * z;
            volatile float partial = xx + yy;
            volatile float total = partial + zz;
            return total <= limit;
        };
        const auto brute = [&](const bool masked_references) {
            std::vector<int> counts(count, 0);
            for (size_t i = 0; i < count; ++i) {
                if (!finite(i) || !(radii[i] > 0.0f) || !std::isfinite(radii[i]))
                    continue;
                for (size_t j = 0; j < count; ++j)
                    if (j != i && (!masked_references || references[j] != 0.0f) && finite(j) &&
                        within(&xyz[i * 3], &xyz[j * 3], radii[i]))
                        ++counts[i];
            }
            return counts;
        };
        const auto points = tensor(xyz, {count, 3});
        const auto radius_tensor = tensor(radii, {count});
        const auto query_mask = tensor(queries, {count}).to(lfs::core::DataType::Bool);
        for (const bool masked_references : {false, true}) {
            const auto counts = brute(masked_references);
            const auto reference_mask = masked_references ? tensor(references, {count}).to(lfs::core::DataType::Bool)
                                                          : Tensor::full_bool({count}, true, device());
            for (const int32_t max_count : {1, 3, 7, 1 << 20}) {
                SCOPED_TRACE(std::format("masked references {} max_count {}", masked_references, max_count));
                std::vector<int> expected(count), expected_queried(count);
                for (size_t i = 0; i < count; ++i) {
                    expected[i] = std::min(counts[i], max_count);
                    expected_queried[i] = queries[i] != 0.0f ? expected[i] : 0;
                }
                const auto all = lfs::core::radius_neighbor_counts(points, reference_mask, radius_tensor, max_count);
                EXPECT_EQ(all.device(), device());
                EXPECT_EQ(all.dtype(), lfs::core::DataType::Int32);
                EXPECT_EQ(host<int>(all), expected);
                EXPECT_EQ(host<int>(lfs::core::radius_neighbor_counts(points, reference_mask, radius_tensor, max_count,
                                                                      &query_mask)),
                          expected_queried);
            }
        }
        // No usable reference at all.
        const auto none = Tensor::zeros({count}, device(), lfs::core::DataType::Bool);
        EXPECT_EQ(host<int>(lfs::core::radius_neighbor_counts(points, none, radius_tensor, 4)), std::vector<int>(count, 0));
    }

    TEST_P(NodesCore, MutualRadiusComponentsMatchBruteForce) {
        // Clusters of different density with radii from a fraction of their spacing to far beyond it, so tree
        // boxes are skipped for their largest radius as well as for the query's.
        std::mt19937 random(4417);
        std::uniform_real_distribution<float> unit(0.0f, 1.0f);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        constexpr size_t count = 3000;
        std::vector<float> xyz(count * 3), radii(count);
        for (size_t i = 0; i < count; ++i) {
            const float pick = unit(random);
            const float spread = pick < 0.6f ? 0.05f : pick < 0.9f ? 1.0f
                                                                   : 8.0f;
            for (int axis = 0; axis < 3; ++axis)
                xyz[i * 3 + axis] = (axis == 0 && pick >= 0.9f ? 20.0f : 0.0f) + spread * normal(random);
            radii[i] = std::exp(std::log(2e-3f) + unit(random) * (std::log(6.0f) - std::log(2e-3f)));
        }
        std::copy_n(xyz.begin() + 3, 3, xyz.begin() + 9);
        xyz[30] = std::numeric_limits<float>::infinity();
        radii[4] = 0.0f;
        radii[5] = -1.0f;
        radii[6] = std::numeric_limits<float>::quiet_NaN();
        const auto within = [](const float* a, const float* b, const float radius) {
            volatile float x = a[0] - b[0], y = a[1] - b[1], z = a[2] - b[2];
            volatile float limit = radius * radius;
            if (limit < std::numeric_limits<float>::min() || !std::isfinite(limit)) {
                x = x / radius;
                y = y / radius;
                z = z / radius;
                limit = 1.0f;
            }
            volatile float xx = x * x, yy = y * y, zz = z * z;
            volatile float partial = xx + yy;
            volatile float total = partial + zz;
            return total <= limit;
        };
        const auto usable = [&](size_t i) {
            return std::isfinite(xyz[i * 3]) && std::isfinite(xyz[i * 3 + 1]) && std::isfinite(xyz[i * 3 + 2]) &&
                   radii[i] > 0.0f && std::isfinite(radii[i]);
        };
        std::vector<int> parent(count);
        std::iota(parent.begin(), parent.end(), 0);
        const auto root = [&](int x) {
            while (parent[x] != x)
                x = parent[x] = parent[parent[x]];
            return x;
        };
        std::vector<int> degree(count);
        for (size_t i = 0; i < count; ++i)
            for (size_t j = i + 1; j < count; ++j)
                if (usable(i) && usable(j) && within(&xyz[i * 3], &xyz[j * 3], radii[i]) &&
                    within(&xyz[i * 3], &xyz[j * 3], radii[j])) {
                    const int a = root(int(i)), b = root(int(j));
                    parent[std::max(a, b)] = std::min(a, b);
                    ++degree[i];
                    ++degree[j];
                }
        // Walks hand out neighbours in batches of at most 64, so dense points take several.
        EXPECT_GT(*std::max_element(degree.begin(), degree.end()), 256);
        std::vector<int> expected(count);
        for (size_t i = 0; i < count; ++i)
            expected[i] = root(int(i));
        const auto labels = lfs::core::mutual_radius_components(tensor(xyz, {count, 3}), tensor(radii, {count}));
        EXPECT_EQ(labels.device(), device());
        EXPECT_EQ(labels.dtype(), lfs::core::DataType::Int32);
        EXPECT_EQ(host<int>(labels), expected);
        EXPECT_GT(std::set<int>(expected.begin(), expected.end()).size(), 10u);
        EXPECT_LT(std::set<int>(expected.begin(), expected.end()).size(), count / 2);
    }

    TEST_P(NodesCore, RadiusConnectedComponentsMatchBruteForce) {
        const auto reference = [](const std::vector<float>& xyz, const std::vector<bool>& selected, const float radius) {
            const size_t count = xyz.size() / 3;
            std::vector<int> parent(count);
            std::iota(parent.begin(), parent.end(), 0);
            const auto root = [&](int x) {
                while (parent[x] != x)
                    x = parent[x] = parent[parent[x]];
                return x;
            };
            for (size_t i = 0; i < count; ++i) {
                if (!selected[i] || !std::isfinite(xyz[i * 3]) || !std::isfinite(xyz[i * 3 + 1]) || !std::isfinite(xyz[i * 3 + 2]))
                    continue;
                for (size_t j = i + 1; j < count; ++j) {
                    if (!selected[j])
                        continue;
                    float squared = 0;
                    for (int axis = 0; axis < 3; ++axis) {
                        const float delta = xyz[i * 3 + axis] - xyz[j * 3 + axis];
                        squared += delta * delta;
                    }
                    if (squared <= radius * radius) {
                        const int a = root(int(i)), b = root(int(j));
                        parent[std::max(a, b)] = std::min(a, b);
                    }
                }
            }
            std::vector<int> labels(count);
            for (size_t i = 0; i < count; ++i)
                labels[i] = root(int(i));
            return labels;
        };
        const auto check = [&](const std::vector<float>& xyz, const std::vector<bool>& selected, const float radius) {
            const size_t count = xyz.size() / 3;
            const auto points = tensor(xyz, {count, 3});
            const bool all = std::ranges::all_of(selected, [](bool value) { return value; });
            std::vector<float> mask(selected.begin(), selected.end());
            const auto actual = all ? lfs::core::radius_connected_components(points, radius)
                                    : lfs::core::radius_connected_components(points, radius, tensor(mask, {count}).to(lfs::core::DataType::Bool));
            EXPECT_EQ(actual.device(), device());
            EXPECT_EQ(actual.dtype(), lfs::core::DataType::Int32);
            EXPECT_EQ(host<int>(actual), reference(xyz, selected, radius));
        };
        std::mt19937 random(5171);
        std::uniform_real_distribution<float> coordinate(-2.0f, 2.0f);
        constexpr size_t count = 1500;
        std::vector<float> xyz(count * 3);
        for (auto& value : xyz)
            value = coordinate(random);
        std::copy_n(xyz.begin(), 3, xyz.begin() + 3);
        xyz[30] = std::numeric_limits<float>::quiet_NaN();
        std::vector<bool> every(count, true), some(count);
        for (size_t i = 0; i < count; ++i)
            some[i] = i % 7 != 0;
        for (const float radius : {0.05f, 0.3f, 0.6f}) {
            check(xyz, every, radius);
            check(xyz, some, radius);
        }
        // A chain far longer than any fixed number of propagation rounds, cut once by the selection.
        constexpr size_t chain = 5000;
        std::vector<float> line(chain * 3, 0.0f);
        for (size_t i = 0; i < chain; ++i)
            line[i * 3] = 0.009f * static_cast<float>(chain - 1 - i);
        std::vector<bool> cut(chain, true);
        cut[2500] = false;
        check(line, std::vector<bool>(chain, true), 0.01f);
        check(line, cut, 0.01f);
        EXPECT_EQ(lfs::core::radius_connected_components(Tensor::empty({0, 3}, device()), 1.0f).numel(), 0u);
    }

    TEST_P(NodesCore, RadiusNeighborMinMatchesBruteForceForFloatAndInt) {
        constexpr size_t count = 193;
        std::mt19937 random(417);
        std::uniform_real_distribution<float> coordinate(-3.0f, 3.0f);
        std::uniform_int_distribution<int> integer(-50, 50);
        std::vector<float> xyz(count * 3), floats(count);
        std::vector<int> integers(count);
        for (size_t i = 0; i < count; ++i) {
            for (int axis = 0; axis < 3; ++axis)
                xyz[i * 3 + axis] = coordinate(random);
            floats[i] = coordinate(random);
            integers[i] = integer(random);
        }
        // Exercise coincident points and hash collisions.
        std::copy_n(xyz.begin(), 3, xyz.begin() + 3);
        const auto points = tensor(xyz, {count, 3});
        for (const float radius : {0.05f, 0.75f, 20.0f}) {
            std::vector<float> expected_float = floats;
            std::vector<int> expected_int = integers;
            for (size_t i = 0; i < count; ++i) {
                for (size_t j = 0; j < count; ++j) {
                    float squared = 0;
                    for (int axis = 0; axis < 3; ++axis) {
                        const float delta = xyz[i * 3 + axis] - xyz[j * 3 + axis];
                        squared += delta * delta;
                    }
                    if (squared <= radius * radius) {
                        expected_float[i] = std::min(expected_float[i], floats[j]);
                        expected_int[i] = std::min(expected_int[i], integers[j]);
                    }
                }
            }
            const auto actual_float = lfs::core::radius_neighbor_min(points, tensor(floats, {count}), radius);
            const auto actual_int = lfs::core::radius_neighbor_min(points, ints(integers, {count}), radius);
            EXPECT_EQ(actual_float.device(), device());
            EXPECT_EQ(actual_float.dtype(), lfs::core::DataType::Float32);
            EXPECT_EQ(host<float>(actual_float), expected_float);
            EXPECT_EQ(host<int>(actual_int), expected_int);
        }
        EXPECT_EQ(lfs::core::radius_neighbor_min(Tensor::empty({0, 3}, device()),
                                                 Tensor::empty({0}, device(), lfs::core::DataType::Int32), 1.0f)
                      .numel(),
                  0u);

        std::vector<float> local_radii(count);
        for (size_t i = 0; i < count; ++i)
            local_radii[i] = std::ldexp(0.75f, int(i % 6) - 3);
        auto expected = integers;
        for (size_t i = 0; i < count; ++i)
            for (size_t j = 0; j < count; ++j) {
                float squared = 0;
                for (int axis = 0; axis < 3; ++axis) {
                    const float d = xyz[3 * i + axis] - xyz[3 * j + axis];
                    squared += d * d;
                }
                const float radius = std::min(local_radii[i], local_radii[j]);
                if (squared <= radius * radius)
                    expected[i] = std::min(expected[i], integers[j]);
            }
        const auto radii = tensor(local_radii, {count});
        const auto values = ints(integers, {count});
        auto actual = values;
        for (int octave = -3; octave <= 2; ++octave)
            actual = actual.minimum(lfs::core::radius_neighbor_min(points, values, std::ldexp(1.0f, octave), &radii));
        EXPECT_EQ(host<int>(actual), expected) << "Octaves must include every symmetric local-radius edge";
    }

    TEST_P(NodesCore, CurvesRampDistanceGradientAndNoiseEvaluateOnBackend) {
        auto geometry = splats();
        const auto curve = field_result("lfs.float_curve", "Value", FLOAT_SOCKET, geometry, [](Node& node) {
            node.input_values["Value"] = 0.5f;
            node.properties["points"] = {{0.0f, 0.0f}, {0.5f, 0.8f}, {1.0f, 1.0f}};
        });
        for (float value : host<float>(curve))
            EXPECT_NEAR(value, 0.8f, 1e-6f);

        const auto ramp = field_result("lfs.colour_ramp", "Colour", COLOUR_SOCKET, geometry, [](Node& node) {
            node.input_values["Fac"] = 0.25f;
            node.properties["stops"] = {{0.0f, 1.0f, 0.0f, 0.0f, 0.25f},
                                        {0.5f, 0.0f, 0.0f, 1.0f, 0.75f},
                                        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}};
        });
        const auto ramp_values = host<float>(ramp);
        ASSERT_EQ(ramp_values.size(), 9u);
        EXPECT_NEAR(ramp_values[0], 0.5f, 1e-6f);
        EXPECT_NEAR(ramp_values[1], 0.0f, 1e-6f);
        EXPECT_NEAR(ramp_values[2], 0.5f, 1e-6f);

        const auto distance = field_result("lfs.distance", "Distance", FLOAT_SOCKET, geometry, [](Node& node) {
            node.input_values["Vector"] = glm::vec3(3, 4, 0);
            node.input_values["Point"] = glm::vec3(0);
        });
        for (float value : host<float>(distance))
            EXPECT_NEAR(value, 5.0f, 1e-6f);

        const auto gradient = field_result("lfs.gradient_texture", "Fac", FLOAT_SOCKET, geometry, [](Node& node) {
            node.input_values["Vector"] = glm::vec3(2, 4, 8);
            node.properties["type"] = "diagonal";
        });
        for (float value : host<float>(gradient))
            EXPECT_NEAR(value, 3.0f, 1e-6f);

        const auto noise = field_result("lfs.noise_texture", "Fac", FLOAT_SOCKET, geometry, [](Node& node) {
            node.input_values["Vector"] = glm::vec3(0.13f, -0.27f, 0.91f);
            node.input_values["Detail"] = 2.5f;
            node.input_values["Seed"] = 7.0f;
        });
        EXPECT_EQ(noise.device(), device());
        for (float value : host<float>(noise)) {
            EXPECT_TRUE(std::isfinite(value));
            EXPECT_GE(value, 0.0f);
            EXPECT_LE(value, 1.0f);
        }
    }

    TEST_P(NodesCore, StoreNamedAttributeBlendsAndRemoveClumpsFindsComponents) {
        auto geometry = splats();
        const auto stored = single("lfs.store_named_attribute", geometry, [](Node& node) {
            node.input_values["Name"] = std::string("temperature");
            node.input_values["Value"] = 4.0f;
            node.input_values["Selection"] = 0.25f;
        });
        ASSERT_TRUE(stored.ok) << (stored.errors.empty() ? "" : stored.errors.begin()->second);
        EXPECT_EQ(host<float>(stored.geometry.splats->attributes.at("temperature")),
                  (std::vector<float>{1, 1, 1}));

        geometry.splats->means = tensor({0, 0, 0, 0.1f, 0, 0, 0.2f, 0, 0, 10, 0, 0}, {4, 3});
        geometry.splats->sh0 = tensor(std::vector<float>(12), {4, 3});
        geometry.splats->shN = tensor(std::vector<float>(36), {4, 3, 3});
        geometry.splats->scaling = tensor(std::vector<float>(12, std::log(0.1f)), {4, 3});
        geometry.splats->rotation = tensor({1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, {4, 4});
        geometry.splats->opacity = tensor(std::vector<float>(4), {4});
        geometry.splats->attributes["weight"] = tensor({1, 2, 3, 4}, {4});
        const auto cleaned = single("lfs.remove_clumps", geometry, [](Node& node) {
            node.input_values["Radius"] = 2.5f;
            node.input_values["Min Size"] = std::int64_t(2);
        });
        ASSERT_TRUE(cleaned.ok) << (cleaned.errors.empty() ? "" : cleaned.errors.begin()->second);
        ASSERT_TRUE(cleaned.geometry.splats);
        EXPECT_EQ(cleaned.geometry.splats->means.shape()[0], 3u);
        EXPECT_EQ(host<float>(cleaned.geometry.splats->attributes.at("weight")),
                  (std::vector<float>{1, 2, 3}));
    }

    TEST_P(NodesCore, RemoveClumpsPreservesTenfoldSurfaceDensityAndRejectsIsolatedClusters) {
        std::vector<float> positions;
        for (int patch = 0; patch < 2; ++patch) {
            const float step = patch ? 0.0316227766f : 0.01f;
            for (int y = 0; y < 20; ++y)
                for (int x = 0; x < 20; ++x)
                    positions.insert(positions.end(), {patch * 0.22f + x * step, y * step, 0.0f});
        }
        const auto surface = positions;
        for (int cluster = 0; cluster < 4; ++cluster)
            for (int point = 0; point < 8; ++point)
                positions.insert(positions.end(), {cluster * 0.5f + (point & 1) * 0.002f,
                                                   ((point >> 1) & 1) * 0.002f,
                                                   2.0f + ((point >> 2) & 1) * 0.002f});
        Geometry geometry;
        geometry.points = PointsComponent{tensor(positions, {positions.size() / 3, 3}),
                                          tensor(std::vector<float>(positions.size()), {positions.size() / 3, 3}),
                                          {}};
        const auto result = single("lfs.remove_clumps", geometry);
        ASSERT_TRUE(result.ok) << (result.errors.empty() ? "" : result.errors.begin()->second);
        ASSERT_TRUE(result.geometry.points);
        EXPECT_EQ(host<float>(result.geometry.points->positions), surface);
    }

    TEST_P(NodesCore, RemoveClumpsFullSceneAcceptance) {
        const auto* directory = std::getenv("LFS_NODE_SCENES_DIR");
        if (!directory)
            GTEST_SKIP() << "Set LFS_NODE_SCENES_DIR to the full garden/bicycle PLY fixture directory";
        for (const auto& [name, limit, expected] : {
                 std::tuple{"garden", 0.01, 992505u}, std::tuple{"bicycle", 0.02, 983348u}}) {
            const auto loaded = lfs::io::load_ply(std::filesystem::path(directory) / (std::string(name) + ".ply"));
            ASSERT_TRUE(loaded) << name;
            auto geometry = geometry_from_splat_data(loaded->value);
            geometry.splats->means = geometry.splats->means.to(device());
            const auto count = geometry.splats->means.shape()[0];
            ASSERT_EQ(count, 1'000'000u) << "The acceptance fixtures are the full 1M-splat captures";
            for (int run = 0; run < 2; ++run) {
                const auto begin = std::chrono::steady_clock::now();
                const auto result = single("lfs.remove_clumps", geometry);
                ASSERT_TRUE(result.ok);
                const auto kept = host<float>(result.geometry.splats->means).size() / 3;
                const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
                std::cout << "CLUMPS " << GetParam().name << ' ' << name << " run=" << run
                          << " kept=" << kept << " removed=" << (count - kept)
                          << " percent=" << (100.0 * (count - kept) / count) << " seconds=" << seconds << '\n';
                EXPECT_EQ(kept, expected) << "CPU/Metal/Vulkan must retain identical element counts";
                EXPECT_LE(double(count - kept) / count, limit);
            }
        }
    }

    TEST_P(NodesCore, MergeByDistanceAveragesPointsRemapsMeshAndKeepsOpaqueSplat) {
        Geometry points_geometry;
        points_geometry.points = PointsComponent{
            tensor({0, 0, 0, 0.1f, 0, 0, 2, 0, 0}, {3, 3}),
            tensor({1, 0, 0, 0, 0, 1, 0, 1, 0}, {3, 3}),
            {}};
        const auto points = single("lfs.merge_by_distance", std::move(points_geometry), [](Node& node) {
            node.input_values["Distance"] = 0.2f;
        });
        ASSERT_TRUE(points.ok) << (points.errors.empty() ? "" : points.errors.begin()->second);
        ASSERT_TRUE(points.geometry.points);
        EXPECT_EQ(points.geometry.points->positions.shape(), TensorShape({2, 3}));
        EXPECT_EQ(host<float>(points.geometry.points->positions),
                  (std::vector<float>{0.05f, 0, 0, 2, 0, 0}));
        EXPECT_EQ(host<float>(points.geometry.points->colors),
                  (std::vector<float>{0.5f, 0, 0.5f, 0, 1, 0}));

        auto mesh = std::make_shared<lfs::core::MeshData>(
            tensor({0, 0, 0, 0.1f, 0, 0, 1, 0, 0, 0, 1, 0}, {4, 3}),
            ints({0, 1, 2, 0, 2, 3}, {2, 3}));
        mesh->colors = tensor({1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1}, {4, 4});
        const auto merged_mesh = single("lfs.merge_by_distance", geometry_from_mesh(mesh), [](Node& node) {
            node.input_values["Distance"] = 0.2f;
        });
        ASSERT_TRUE(merged_mesh.ok) << (merged_mesh.errors.empty() ? "" : merged_mesh.errors.begin()->second);
        ASSERT_TRUE(merged_mesh.geometry.mesh && merged_mesh.geometry.mesh->mesh);
        EXPECT_EQ(merged_mesh.geometry.mesh->mesh->vertices.shape(), TensorShape({3, 3}));
        EXPECT_EQ(merged_mesh.geometry.mesh->mesh->indices.shape(), TensorShape({1, 3}));
        EXPECT_EQ(host<int>(merged_mesh.geometry.mesh->mesh->indices), (std::vector<int>{0, 1, 2}));

        auto splat_geometry = splats(0);
        splat_geometry.splats->means = tensor({0, 0, 0, 0.1f, 0, 0, 2, 0, 0}, {3, 3});
        splat_geometry.splats->opacity = tensor({-2, 3, 0}, {3});
        const auto merged_splats = single("lfs.merge_by_distance", std::move(splat_geometry), [](Node& node) {
            node.input_values["Distance"] = 0.2f;
        });
        ASSERT_TRUE(merged_splats.ok) << (merged_splats.errors.empty() ? "" : merged_splats.errors.begin()->second);
        ASSERT_TRUE(merged_splats.geometry.splats);
        EXPECT_EQ(merged_splats.geometry.splats->means.shape()[0], 2u);
        EXPECT_EQ(host<float>(merged_splats.geometry.splats->opacity), (std::vector<float>{3, 0}));
    }

    TEST_P(NodesCore, GeometryProximityMatchesBruteForceAcrossComponents) {
        auto geometry = splats();
        std::mt19937 random(819);
        std::uniform_real_distribution<float> coord(-3, 3);
        std::vector<float> q(291), t(153);
        std::generate(q.begin(), q.end(), [&] { return coord(random); });
        std::generate(t.begin(), t.end(), [&] { return coord(random); });
        q[0] = 100;
        q[1] = -80;
        q[2] = 20; // exact fallback across a large empty gap
        geometry.splats->means = tensor(q, {97, 3});
        Geometry target;
        target.points = PointsComponent{tensor(t, {51, 3}), Tensor::ones({51, 3}, device()), {}};
        const auto positions = host<float>(field_result("lfs.geometry_proximity", "Position", VECTOR_SOCKET, geometry, [&](Node& node) { node.input_values["Target"] = target; }));
        const auto distances = host<float>(field_result("lfs.geometry_proximity", "Distance", FLOAT_SOCKET, geometry, [&](Node& node) { node.input_values["Target"] = target; }));
        for (size_t i = 0; i < 97; ++i) {
            float best = INFINITY;
            size_t index = 0;
            for (size_t j = 0; j < 51; ++j) {
                float d = 0;
                for (int a = 0; a < 3; ++a)
                    d += std::pow(q[3 * i + a] - t[3 * j + a], 2);
                if (d < best) {
                    best = d;
                    index = j;
                }
            }
            EXPECT_NEAR(distances[i], std::sqrt(best), 2e-5f);
            for (int a = 0; a < 3; ++a)
                EXPECT_FLOAT_EQ(positions[3 * i + a], t[3 * index + a]);
        }
        const auto ties = lfs::core::nearest_point_indices(tensor({0, 0, 0}, {1, 3}), tensor({1, 0, 0, -1, 0, 0}, {2, 3}));
        EXPECT_EQ(host<int>(ties), (std::vector<int>{0}));
        EXPECT_EQ(host<int>(lfs::core::nearest_point_indices(tensor({0, 0, 0}, {1, 3}), Tensor::empty({0, 3}, device()))), (std::vector<int>{-1}));
        target = geometry_from_mesh(std::make_shared<lfs::core::MeshData>(tensor(t, {51, 3}), ints({0, 1, 2}, {1, 3})));
        EXPECT_EQ(host<float>(field_result("lfs.geometry_proximity", "Position", VECTOR_SOCKET, geometry, [&](Node& node) { node.input_values["Target"] = target; })), positions);
        target = splats();
        target.splats->means = tensor(t, {51, 3});
        EXPECT_EQ(host<float>(field_result("lfs.geometry_proximity", "Position", VECTOR_SOCKET, geometry, [&](Node& node) { node.input_values["Target"] = target; })), positions);
    }

    TEST_P(NodesCore, InstanceOnPointsMatchesTransformIncludingShAndSelection) {
        auto source = splats(3);
        Geometry anchors;
        anchors.points = PointsComponent{tensor({2, 3, 4, -1, 4, 2}, {2, 3}), Tensor::ones({2, 3}, device()), {}};
        NodeTree tree(registry_);
        auto& node = tree.add_node("lfs.instance_on_points", "Scatter");
        node.input_values["Points"] = anchors;
        node.input_values["Instance"] = source;
        node.input_values["Rotation"] = glm::vec3(23, -31, 47);
        node.input_values["Scale"] = glm::vec3(0.7f, 1.2f, 1.8f);
        tree.add_link({node.name, "Geometry", tree.output_node().name, "Geometry"});
        const auto result = evaluate(tree, {{}, {}, 1, device()});
        ASSERT_TRUE(result.ok) << (result.errors.empty() ? "" : result.errors.begin()->second);
        ASSERT_TRUE(result.geometry.splats);
        // CPU evaluation stays on the CPU; GPU evaluation stays on its backend.
        EXPECT_EQ(result.geometry.splats->means.device(), device());
        EXPECT_EQ(result.geometry.splats->shN.device(), device());
        EXPECT_EQ(result.geometry.splats->means.size(0), 6u);
        const glm::vec3 positions[] = {glm::vec3(2, 3, 4), glm::vec3(-1, 4, 2)};
        const auto rotation = glm::rotate(glm::mat4(1), glm::radians(47.0f), glm::vec3(0, 0, 1)) * glm::rotate(glm::mat4(1), glm::radians(-31.0f), glm::vec3(0, 1, 0)) * glm::rotate(glm::mat4(1), glm::radians(23.0f), glm::vec3(1, 0, 0));
        for (int a = 0; a < 2; ++a) {
            auto reference = splat_data_from_geometry(source);
            lfs::core::transform(*reference, glm::translate(glm::mat4(1), positions[a]) * rotation * glm::scale(glm::mat4(1), glm::vec3(0.7f, 1.2f, 1.8f)));
            const auto expected = geometry_from_splat_data(*reference);
            for (const auto& pair : {std::pair{result.geometry.splats->means, expected.splats->means}, std::pair{result.geometry.splats->shN, expected.splats->shN}, std::pair{result.geometry.splats->scaling, expected.splats->scaling}, std::pair{result.geometry.splats->rotation, expected.splats->rotation}}) {
                const auto actual = host<float>(pair.first.slice(0, a * 3, a * 3 + 3)), want = host<float>(pair.second);
                ASSERT_EQ(actual.size(), want.size());
                for (size_t i = 0; i < want.size(); ++i)
                    EXPECT_NEAR(actual[i], want[i], 6e-5f) << i;
            }
        }
        node.input_values["Selection"] = 0.49f;
        const auto empty = evaluate(tree, {{}, {}, 2});
        ASSERT_TRUE(empty.ok);
        EXPECT_EQ(empty.geometry.splats->means.size(0), 0u);
        node.input_values["Selection"] = 1.0f;
        node.input_values["Scale"] = 2.0f;
        ASSERT_TRUE(evaluate(tree, {{}, {}, 3}).ok); // scalar broadcasts to vector
        // Check the cap before allocating the repeated splats.
        anchors.points->positions = Tensor::zeros({1, 3}, device()).expand({20'000'000, 3});
        anchors.points->colors = Tensor::ones({1, 3}, device()).expand({20'000'000, 3});
        node.input_values["Points"] = anchors;
        const auto capped = evaluate(tree, {{}, {}, 4});
        EXPECT_FALSE(capped.ok);
        EXPECT_TRUE(std::ranges::any_of(capped.errors, [](const auto& entry) { return entry.second.find("50 million") != std::string::npos; }));
    }

    TEST_P(NodesCore, InstanceOnPointsKeepsShBeyondTheDegreeAndMatchesTheCpu) {
        // Degree 1 with eight stored coefficients: the five beyond band 1 pass through unchanged, and a
        // collapsed scale axis leaves every coefficient alone, on every device as on the CPU.
        for (const glm::vec3 size : {glm::vec3(0.6f, 1.3f, 0.9f), glm::vec3(0.0f, 1.0f, 1.0f)}) {
            SCOPED_TRACE(size.x);
            const auto make = [&](const Device target) {
                auto source = splats(1);
                std::vector<float> sh(3 * 8 * 3);
                for (size_t i = 0; i < sh.size(); ++i)
                    sh[i] = 0.01f * float(i % 29) - 0.1f;
                source.splats->shN = Tensor::from_vector(sh, {3, 8, 3}, Device::CPU).to(target);
                source.splats->sh_degree = 1;
                for (auto* value : {&source.splats->means, &source.splats->sh0, &source.splats->scaling,
                                    &source.splats->rotation, &source.splats->opacity})
                    *value = value->to(target);
                for (auto& [_, value] : source.splats->attributes)
                    value = value.to(target);
                Geometry anchors;
                anchors.points = PointsComponent{Tensor::from_vector(std::vector<float>{2, 3, 4, -1, 4, 2, 0, 0, 1}, {3, 3}, Device::CPU).to(target),
                                                 Tensor::ones({3, 3}, target),
                                                 {}};
                NodeTree tree(registry_);
                auto& node = tree.add_node("lfs.instance_on_points", "Scatter");
                node.input_values["Points"] = anchors;
                node.input_values["Instance"] = source;
                node.input_values["Rotation"] = glm::vec3(23, -31, 47);
                node.input_values["Scale"] = size;
                tree.add_link({node.name, "Geometry", tree.output_node().name, "Geometry"});
                const auto result = evaluate(tree, {{}, {}, 1, target});
                EXPECT_TRUE(result.ok) << (result.errors.empty() ? "" : result.errors.begin()->second);
                return std::pair{result.geometry.splats, sh};
            };
            const auto [actual, sh] = make(device());
            const auto [expected, unused] = make(Device::CPU);
            (void)unused;
            ASSERT_TRUE(actual && expected);
            const auto shn = host<float>(actual->shN);
            ASSERT_EQ(shn.size(), 3u * sh.size());
            for (size_t copy = 0; copy < 3; ++copy)
                for (size_t row = 0; row < 3; ++row)
                    for (size_t k = 3; k < 8; ++k)
                        for (size_t c = 0; c < 3; ++c)
                            EXPECT_EQ(shn[((copy * 3 + row) * 8 + k) * 3 + c], sh[(row * 8 + k) * 3 + c]);
            if (size.x == 0)
                EXPECT_EQ(shn, host<float>(expected->shN));
            for (const auto& pair : {std::pair{actual->means, expected->means}, std::pair{actual->shN, expected->shN},
                                     std::pair{actual->scaling, expected->scaling}, std::pair{actual->rotation, expected->rotation}}) {
                const auto got = host<float>(pair.first), want = host<float>(pair.second);
                ASSERT_EQ(got.size(), want.size());
                for (size_t i = 0; i < want.size(); ++i)
                    if (std::isfinite(want[i]) || std::isfinite(got[i]))
                        EXPECT_NEAR(got[i], want[i], 2e-5f) << i;
            }
        }
    }

    TEST_P(NodesCore, InstanceOnPointsSupportsReflectionsAndCollapsedAxes) {
        for (const auto size : {glm::vec3(-1, 0.5f, 2), glm::vec3(0, 1, 2)}) {
            const auto source = splats(3);
            NodeTree tree(registry_);
            auto& node = tree.add_node("lfs.instance_on_points");
            Geometry anchors = geometry_from_mesh(std::make_shared<lfs::core::MeshData>(tensor({2, 3, 4}, {1, 3}), ints({}, {0, 3})));
            node.input_values["Points"] = anchors;
            node.input_values["Instance"] = source;
            node.input_values["Scale"] = size;
            node.input_values["Rotation"] = glm::vec3(0, 0, 30);
            tree.add_link({node.name, "Geometry", tree.output_node().name, "Geometry"});
            const auto result = evaluate(tree, {{}, {}, 1});
            ASSERT_TRUE(result.ok);
            auto reference = splat_data_from_geometry(source);
            lfs::core::transform(*reference, glm::translate(glm::mat4(1), glm::vec3(2, 3, 4)) * glm::rotate(glm::mat4(1), glm::radians(30.0f), glm::vec3(0, 0, 1)) * glm::scale(glm::mat4(1), size));
            const auto expected = geometry_from_splat_data(*reference);
            for (const auto& pair : {std::pair{result.geometry.splats->means, expected.splats->means}, std::pair{result.geometry.splats->shN, expected.splats->shN}, std::pair{result.geometry.splats->scaling, expected.splats->scaling}}) {
                const auto actual = host<float>(pair.first), want = host<float>(pair.second);
                ASSERT_EQ(actual.size(), want.size());
                for (size_t i = 0; i < want.size(); ++i)
                    EXPECT_NEAR(actual[i], want[i], 6e-5f) << i;
            }
        }
    }

    TEST_P(NodesCore, CollapsedScalesSurviveTransformsMergeAndInstances) {
        const float collapsed = -std::numeric_limits<float>::infinity();
        auto source = splats(3);
        source.splats->scaling = tensor({collapsed, collapsed, collapsed, 0, collapsed, collapsed, 0, 1, collapsed}, {3, 3});
        const auto assert_valid = [&](const Geometry& result) {
            for (float value : host<float>(result.splats->scaling))
                EXPECT_TRUE(std::isfinite(value) || value == collapsed) << value;
            auto activated = result;
            activated.splats->scaling = activated.splats->scaling.exp();
            expect_finite(activated);
        };
        for (float factor : {-2.0f, 0.0f, 2.0f}) {
            const auto result = single("lfs.transform_geometry", source, [&](Node& node) {
                node.input_values["Scale"] = factor;
                node.input_values["Rotation"] = glm::vec3(13, -29, 37);
            });
            ASSERT_TRUE(result.ok);
            assert_valid(result.geometry);
            const auto scales = host<float>(result.geometry.splats->scaling);
            EXPECT_EQ(std::count(scales.begin(), scales.end(), collapsed), factor == 0 ? 9 : 6);
        }
        auto coincident = source;
        coincident.splats->means = Tensor::zeros({3, 3}, device());
        const auto merged = single("lfs.merge_by_distance", coincident);
        ASSERT_TRUE(merged.ok);
        assert_valid(merged.geometry);
        EXPECT_EQ(host<float>(merged.geometry.splats->scaling), (std::vector<float>{0, collapsed, collapsed}));

        for (const auto size : {glm::vec3(0), glm::vec3(0, 1, 2), glm::vec3(-1, 0, 2), glm::vec3(-1, 0.5f, 2), glm::vec3(-2)}) {
            NodeTree tree(registry_);
            auto& node = tree.add_node("lfs.instance_on_points");
            Geometry anchors;
            anchors.points = PointsComponent{tensor({2, 3, 4}, {1, 3}), Tensor::ones({1, 3}, device()), {}};
            node.input_values["Points"] = anchors;
            node.input_values["Instance"] = source;
            node.input_values["Scale"] = size;
            node.input_values["Rotation"] = glm::vec3(13, -29, 37);
            ASSERT_TRUE(tree.add_link({node.name, "Geometry", tree.output_node().name, "Geometry"}));
            const auto result = evaluate(tree, {{}, {}, 1});
            ASSERT_TRUE(result.ok);
            assert_valid(result.geometry);
            const auto scales = host<float>(result.geometry.splats->scaling);
            const auto rotations = host<float>(result.geometry.splats->rotation);
            const auto means = host<float>(result.geometry.splats->means);
            const glm::mat3 linear = glm::mat3(glm::rotate(glm::mat4(1), glm::radians(37.0f), glm::vec3(0, 0, 1)) *
                                               glm::rotate(glm::mat4(1), glm::radians(-29.0f), glm::vec3(0, 1, 0)) *
                                               glm::rotate(glm::mat4(1), glm::radians(13.0f), glm::vec3(1, 0, 0)) * glm::scale(glm::mat4(1), size));
            for (size_t row = 0; row < 3; ++row) {
                glm::mat3 covariance(0), diagonal(0);
                const auto q = glm::normalize(glm::quat(rotations[row * 4], rotations[row * 4 + 1], rotations[row * 4 + 2], rotations[row * 4 + 3]));
                const auto rotation = glm::mat3_cast(q);
                for (int axis = 0; axis < 3; ++axis) {
                    diagonal[axis][axis] = std::exp(2 * scales[row * 3 + axis]);
                    covariance[axis][axis] = row > 0 && axis == 0 ? 1.0f : row == 2 && axis == 1 ? std::exp(2.0f)
                                                                                                 : 0.0f;
                }
                const auto expected = linear * covariance * glm::transpose(linear);
                const auto actual = rotation * diagonal * glm::transpose(rotation);
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 3; ++r)
                        EXPECT_NEAR(actual[c][r], expected[c][r], 1e-4f);
                const auto position = linear * glm::vec3(row == 0 ? 0 : row == 1 ? 1
                                                                                 : 3,
                                                         0, 0) +
                                      glm::vec3(2, 3, 4);
                for (int axis = 0; axis < 3; ++axis)
                    EXPECT_NEAR(means[row * 3 + axis], position[axis], 1e-5f);
                EXPECT_GE(std::count(scales.begin() + row * 3, scales.begin() + row * 3 + 3, collapsed), 3 - int(row));
            }
        }
    }

    TEST_P(NodesCore, Seed5ConstantRampZeroInstancesRecolourScaleClampRegression) {
        auto source = splats(3);
        auto& s = *source.splats;
        const auto indices = ints(std::vector<int>(16, 0), {16});
        s.means = s.means.index_select(0, indices);
        s.sh0 = s.sh0.index_select(0, indices);
        s.shN = s.shN.index_select(0, indices);
        s.scaling = s.scaling.index_select(0, indices);
        s.rotation = s.rotation.index_select(0, indices);
        s.opacity = s.opacity.index_select(0, indices);
        s.attributes.clear();
        Geometry anchors;
        anchors.points = PointsComponent{Tensor::zeros({8, 3}, device()), Tensor::ones({8, 3}, device()), {}};
        NodeTree tree(registry_);
        auto& ramp = tree.add_node("lfs.colour_ramp", "Ramp");
        ramp.input_values["Fac"] = -0.17210110884051355f;
        ramp.properties["interpolation"] = "constant";
        auto& instance = tree.add_node("lfs.instance_on_points", "Instance");
        instance.input_values["Instance"] = source;
        instance.input_values["Points"] = anchors;
        instance.input_values["Selection"] = 0.5438486464309299f;
        instance.input_values["Rotation"] = glm::vec3(-1.4020075016210316f, -1.126018474514678f, -1.218905424765976f);
        auto& colour = tree.add_node("lfs.recolour", "Colour");
        colour.input_values["Selection"] = 2.0f;
        colour.input_values["Colour"] = glm::vec3(0.64235388642241f, 0.10659810068128317f, 0.91770056721701f);
        colour.input_values["Weight"] = 0.9588916500839504f;
        colour.properties["keep_shading"] = false;
        colour.properties["fade_view_dependent"] = false;
        auto& clamp = tree.add_node("lfs.scale_clamp", "Clamp");
        clamp.input_values["Max Aspect"] = 1.5084207228853708f;
        clamp.properties["include_flat"] = false;
        ASSERT_TRUE(tree.add_link({"Ramp", "Colour", "Instance", "Scale"}));
        ASSERT_TRUE(tree.add_link({"Instance", "Geometry", "Colour", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Colour", "Geometry", "Clamp", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Ramp", "Alpha", "Clamp", "Selection"}));
        ASSERT_TRUE(tree.add_link({"Clamp", "Geometry", tree.output_node().name, "Geometry"}));
        const auto result = evaluate(tree, {{}, {}, 1});
        ASSERT_TRUE(result.ok);
        ASSERT_EQ(result.geometry.splats->means.size(0), 128u);
        EXPECT_EQ(host<float>(result.geometry.splats->scaling), std::vector<float>(384, -std::numeric_limits<float>::infinity()));
        auto activated = result.geometry;
        activated.splats->scaling = activated.splats->scaling.exp();
        expect_finite(activated);
    }

    TEST_P(NodesCore, CameraNodesMatchSyntheticRingAndCapturedDistance) {
        struct Host final : EvalHost {
            std::vector<EvaluationCamera> views;
            glm::mat4 world{1};
            uint64_t generation() const override { return 1; }
            std::optional<Geometry> object_geometry(std::string_view, TransformSpace) override { return {}; }
            std::span<const EvaluationCamera> cameras() const override { return views; }
            glm::mat4 object_to_world() const override { return world; }
        } camera_host;
        for (int i = 0; i < 8; ++i) {
            const float angle = i * 2 * std::numbers::pi_v<float> / 8;
            const glm::vec3 eye(5 * std::cos(angle), 1, 5 * std::sin(angle));
            const auto view = glm::scale(glm::mat4(1), glm::vec3(1, -1, -1)) * glm::lookAt(eye, glm::vec3(0), glm::vec3(0, 1, 0));
            camera_host.views.push_back({std::to_string(i), view, 320, 320, 320, 240, 640, 480});
        }
        lfs::vis::registerCameraNodes(registry_);
        auto geometry = splats();
        geometry.splats->means = tensor({0, 0, 0, 2, 0, 0, 8, 0, 0}, {3, 3});
        camera_host.world = glm::translate(glm::mat4(1), glm::vec3(0.2f, -0.1f, 0.3f));
        for (float maximum : {0.0f, 4.5f, 6.0f}) {
            const auto counts = host<int>(field_result("lfs.camera_coverage", "Count", INT_SOCKET, geometry, [&](Node& node) { node.input_values["Max Distance"] = maximum; }, &camera_host));
            const auto ratios = host<float>(field_result("lfs.camera_coverage", "Ratio", FLOAT_SOCKET, geometry, [&](Node& node) { node.input_values["Max Distance"] = maximum; }, &camera_host));
            for (int i = 0; i < 3; ++i) {
                const glm::vec3 local = i == 0 ? glm::vec3(0) : i == 1 ? glm::vec3(2, 0, 0)
                                                                       : glm::vec3(8, 0, 0);
                const auto p = camera_host.world * glm::vec4(local, 1);
                int expected = 0;
                for (const auto& c : camera_host.views) {
                    const auto v = c.world_to_camera * p;
                    const float x = c.focal_x * v.x + c.center_x * v.z, y = c.focal_y * v.y + c.center_y * v.z;
                    const auto eye = glm::inverse(c.world_to_camera)[3];
                    expected += v.z > 1e-6f && x >= 0 && y >= 0 && x <= c.width * v.z && y <= c.height * v.z && (maximum == 0 || glm::length(glm::vec3(p - eye)) <= maximum);
                }
                EXPECT_EQ(counts[i], expected);
                EXPECT_NEAR(ratios[i], expected / 8.0f, 1e-6f);
            }
        }
        const auto position = host<float>(field_result("lfs.camera_info", "Position", VECTOR_SOCKET, geometry, {}, &camera_host));
        EXPECT_NEAR(position[0], 5, 1e-5);
        EXPECT_NEAR(position[1], 1, 1e-5);
        const auto fov = host<float>(field_result("lfs.camera_info", "FOV", FLOAT_SOCKET, geometry, {}, &camera_host));
        EXPECT_NEAR(fov[0], glm::degrees(2 * std::atan(480.0f / 640)), 1e-5);
        const auto image_size = host<float>(field_result("lfs.camera_info", "Image Size", VECTOR_SOCKET, geometry, {}, &camera_host));
        EXPECT_EQ(image_size[0], 640);
        EXPECT_EQ(image_size[1], 480);
        EXPECT_EQ(image_size[2], 0);
        const auto direction = host<float>(field_result("lfs.camera_info", "Direction", VECTOR_SOCKET, geometry, {}, &camera_host));
        const auto wanted = glm::normalize(glm::vec3(-5, -1, 0));
        for (int axis = 0; axis < 3; ++axis)
            EXPECT_NEAR(direction[axis], wanted[axis], 1e-5);
        const auto distance = host<float>(field_result("lfs.view_distance", "Distance", FLOAT_SOCKET, geometry, [](Node& node) {node.properties["source"]="captured";node.properties["captured_transform"]={1,0,0,0,0,1,0,0,0,0,1,0,0,0,3,1}; }, &camera_host));
        EXPECT_NEAR(distance[0], glm::length(glm::vec3(0.2f, -0.1f, -2.7f)), 1e-5);
        camera_host.views.clear();
        NodeTree tree(registry_);
        auto& camera = tree.add_node("lfs.view_distance");
        auto& consumer = tree.add_node("lfs.delete_geometry");
        tree.add_link({camera.name, "Distance", consumer.name, "Selection"});
        tree.add_link({tree.input_node().name, "Geometry", consumer.name, "Geometry"});
        tree.add_link({consumer.name, "Geometry", tree.output_node().name, "Geometry"});
        const auto missing = evaluate(tree, {geometry, {}, 2}, &camera_host);
        EXPECT_FALSE(missing.ok);
        EXPECT_TRUE(std::ranges::any_of(missing.errors, [](const auto& entry) { return entry.second.find("requires dataset cameras") != std::string::npos; }));
    }

    TEST_P(NodesCore, RandomHashMatchesIntegerReferenceIncludingStridesAndHighBits) {
        const std::vector<int> indices{0, 1, 2, 255, 256, 16777216, 16777217,
                                       std::numeric_limits<int>::max(), std::numeric_limits<int>::min(), -1};
        std::vector<int> interleaved;
        for (const int index : indices) {
            interleaved.push_back(index);
            interleaved.push_back(999);
        }
        const auto input = ints(interleaved, {indices.size(), 2}).slice(1, 0, 1).squeeze(1);
        for (const uint32_t seed : {0u, 17u, 0x80000000u, 0xffffffffu}) {
            std::vector<float> reference;
            for (const int index : indices) {
                uint32_t hash = static_cast<uint32_t>(index) ^ seed;
                hash = (hash ^ (hash >> 16u)) * 0x7feb352du;
                hash = (hash ^ (hash >> 15u)) * 0x846ca68bu;
                hash ^= hash >> 16u;
                reference.push_back(static_cast<float>(hash >> 8u) * 0x1p-24f);
            }
            const auto result = lfs::core::random_from_indices(input, seed);
            EXPECT_EQ(result.device(), device());
            EXPECT_EQ(host<float>(result), reference) << "seed=" << seed;
            for (const float value : reference) {
                EXPECT_GE(value, 0);
                EXPECT_LT(value, 1);
            }
        }
        const auto empty = lfs::core::random_from_indices(ints({}, {0}), 17);
        EXPECT_EQ(empty.numel(), 0);
        EXPECT_EQ(empty.device(), device());
    }

    TEST_P(NodesCore, RandomValueMatchesCpuBitExactlyForSeedsRangesFieldsAndConversions) {
        auto geometry = splats();
        constexpr size_t count = 4099;
        geometry.splats->means = Tensor::zeros({count, 3}, device());
        std::vector<float> minimum(count), maximum(count);
        for (size_t i = 0; i < count; ++i) {
            minimum[i] = float(i % 17) * 0.31f - 2;
            maximum[i] = float(i % 11) * -0.27f + 1;
        }
        geometry.splats->attributes["min"] = tensor(minimum, {count});
        geometry.splats->attributes["max"] = tensor(maximum, {count});
        float error = 0;
        for (const int seed : {0, 17, -1, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()}) {
            for (int range = 0; range < 5; ++range) {
                const auto configure = [=](Node& node) {
                    node.properties["seed"] = seed;
                    if (range == 1) {
                        node.input_values["Min"] = 0.678698060057878f;
                        node.input_values["Max"] = -1.0708173877834182f;
                    } else if (range == 2) {
                        node.input_values["Min"] = 1.88136409995301f;
                        node.input_values["Max"] = -0.9553689656533741f;
                    } else if (range == 3) {
                        node.input_values["Min"] = -3.5f;
                        node.input_values["Max"] = -3.5f;
                    } else if (range == 4) {
                        node.input_values["Min"] = named_attribute_field("min", std::string(FLOAT_SOCKET));
                        node.input_values["Max"] = named_attribute_field("max", std::string(FLOAT_SOCKET));
                    }
                };
                for (const auto type : {FLOAT_SOCKET, INT_SOCKET, BOOL_SOCKET, VECTOR_SOCKET, COLOUR_SOCKET}) {
                    const auto reference = host<float>(field_result("lfs.random_value", "Value", type, geometry, configure, nullptr, Device::CPU).to(lfs::core::DataType::Float32));
                    const auto actual = host<float>(field_result("lfs.random_value", "Value", type, geometry, configure).to(lfs::core::DataType::Float32));
                    ASSERT_EQ(actual.size(), reference.size());
                    EXPECT_EQ(actual, reference) << "seed=" << seed << " range=" << range << " type=" << type;
                    EXPECT_EQ(std::memcmp(actual.data(), reference.data(), actual.size() * sizeof(float)), 0)
                        << "seed=" << seed << " range=" << range << " type=" << type;
                    for (size_t i = 0; i < actual.size(); ++i)
                        error = std::max(error, std::abs(actual[i] - reference[i]));
                }
            }
        }
        EXPECT_EQ(error, 0);
        std::cout << "Random Value CPU/" << GetParam().name << " max error " << error << '\n';
    }

    TEST_P(NodesCore, RandomValueFuzzCleanupAndExtremeColourRegressions) {
        // Exact node parameters from review seed 0 graphs 65 and 142. Use a
        // small synthetic payload so this regression does not require garden.ply.
        auto geometry = splats(3);
        auto& s = *geometry.splats;
        constexpr size_t count = 257;
        std::vector<int> repeated(count);
        for (size_t i = 0; i < count; ++i)
            repeated[i] = int(i % 3);
        const auto indices = ints(repeated, {count});
        for (auto* value : {&s.means, &s.sh0, &s.shN, &s.scaling, &s.rotation, &s.opacity})
            *value = value->index_select(0, indices);
        std::vector<float> positions(count * 3, 0);
        for (size_t i = 0; i < count; ++i)
            positions[i * 3] = float(i) * 0.2f; // All points isolated at the failing query radius.
        s.means = tensor(positions, {count, 3});
        s.attributes.clear();
        for (const bool colour_chain : {false, true}) {
            NodeTree tree(registry_);
            const auto input = tree.input_node().name, output = tree.output_node().name;
            auto& random = tree.add_node("lfs.random_value", "Random");
            random.input_values["Min"] = colour_chain ? 1.88136409995301f : 0.678698060057878f;
            random.input_values["Max"] = colour_chain ? -0.9553689656533741f : -1.0708173877834182f;
            if (colour_chain) {
                tree.add_node("lfs.gradient_texture", "Gradient").properties["type"] = "diagonal";
                auto& correct = tree.add_node("lfs.colour_correct", "Correct");
                correct.properties["auto_range"] = true;
                correct.input_values["Selection"] = 0.42511368531503624f;
                correct.input_values["Exposure"] = 3.008884881414131f;
                correct.input_values["Black Point"] = 0.7842575205379938f;
                correct.input_values["White Point"] = -0.7270517910720553f;
                correct.input_values["Midpoint"] = 0.99f;
                correct.input_values["Contrast"] = 2.0f;
                correct.input_values["Saturation"] = 0.4100576812013792f;
                correct.input_values["Hue Shift"] = 0.7835067443254515f;
                correct.input_values["Temperature"] = 0.27517091222663903f;
                correct.input_values["Tint"] = 0.6303857412858538f;
                correct.input_values["Shadows"] = glm::vec3(0.1563354897528767f, 0.6490276122149595f, 0.49411055450869856f);
                correct.input_values["Midtones"] = glm::vec3(0.7519442597880299f, 0.4858927178470527f, 0.621337391930684f);
                correct.input_values["Highlights"] = glm::vec3(0.0030512087616476613f, 0.8435756150132411f, 0.724649520722652f);
                correct.input_values["Gamma"] = 1.8927186892537895f;
                tree.add_node("lfs.invert_colour", "Invert").input_values["Selection"] = 0.9897881324650221f;
                auto& recolour = tree.add_node("lfs.recolour", "Recolour");
                recolour.input_values["Colour"] = glm::vec3(0.3500935021857404f, 0.4818020824426007f, 0.41463360322620446f);
                recolour.input_values["Weight"] = 0.20930645811603366f;
                recolour.properties["keep_shading"] = true;
                recolour.properties["fade_view_dependent"] = false;
                ASSERT_TRUE(tree.add_link({"Random", "Value", "Gradient", "Vector"}));
                ASSERT_TRUE(tree.add_link({input, "Geometry", "Correct", "Geometry"}));
                ASSERT_TRUE(tree.add_link({"Correct", "Geometry", "Invert", "Geometry"}));
                ASSERT_TRUE(tree.add_link({"Invert", "Geometry", "Recolour", "Geometry"}));
                ASSERT_TRUE(tree.add_link({"Gradient", "Fac", "Recolour", "Selection"}));
                ASSERT_TRUE(tree.add_link({"Recolour", "Geometry", output, "Geometry"}));
            } else {
                auto& vector = tree.add_node("lfs.vector_math", "Vector");
                vector.properties["operation"] = "add";
                vector.input_values["A"] = glm::vec3(1.1266080730316999f, -0.536056806377653f, 1.1699958894348854f);
                vector.input_values["Scale"] = 1.0663404053152297f;
                tree.add_node("lfs.set_position", "Move");
                auto& cleanup = tree.add_node("lfs.remove_floaters", "Cleanup");
                cleanup.input_values["Min Opacity"] = 0.0f;
                cleanup.input_values["Max Size"] = 0.9548304191655907f;
                cleanup.input_values["Isolation Radius"] = 0.6885203373941402f;
                cleanup.input_values["Min Neighbours"] = 2;
                cleanup.properties["preview"] = true;
                cleanup.properties["relative_to_size"] = true;
                ASSERT_TRUE(tree.add_link({"Random", "Value", "Vector", "B"}));
                ASSERT_TRUE(tree.add_link({input, "Geometry", "Move", "Geometry"}));
                ASSERT_TRUE(tree.add_link({"Random", "Value", "Move", "Position"}));
                ASSERT_TRUE(tree.add_link({"Random", "Value", "Move", "Offset"}));
                ASSERT_TRUE(tree.add_link({"Vector", "Value", "Move", "Selection"}));
                ASSERT_TRUE(tree.add_link({"Vector", "Vector", "Cleanup", "Selection"}));
                ASSERT_TRUE(tree.add_link({"Move", "Geometry", "Cleanup", "Geometry"}));
                ASSERT_TRUE(tree.add_link({"Cleanup", "Geometry", output, "Geometry"}));
            }
            const auto reference = lfs::nodes::evaluate(tree, {geometry, {}, 1, Device::CPU});
            const auto actual = evaluate(tree, {geometry, {}, 1});
            ASSERT_TRUE(reference.ok);
            ASSERT_TRUE(actual.ok);
            expect_finite(actual.geometry);
            const auto& a = *actual.geometry.splats;
            const auto& b = *reference.geometry.splats;
            if (!colour_chain) {
                EXPECT_GT(a.means.shape()[0], 0);
                EXPECT_LT(a.means.shape()[0], count);
            }
            for (const auto& pair : {std::pair{a.means, b.means}, {a.sh0, b.sh0}, {a.shN, b.shN}, {a.scaling, b.scaling}, {a.rotation, b.rotation}, {a.opacity, b.opacity}}) {
                ASSERT_EQ(pair.first.shape(), pair.second.shape());
                const auto values = host<float>(pair.first), expected = host<float>(pair.second);
                for (size_t i = 0; i < values.size(); ++i)
                    EXPECT_NEAR(values[i], expected[i], 2e-5f * (1 + std::abs(expected[i]))) << "colour_chain=" << colour_chain << " i=" << i;
            }
        }
    }

    TEST_P(NodesCore, SharpenZeroIsBitExactIncludingMergeWinners) {
        auto geometry = splats(3);
        geometry.splats->opacity = tensor({25.0f, -25.0f, 0.1234567f}, {3});
        NodeTree tree(registry_);
        const auto input = tree.input_node().name, output = tree.output_node().name;
        auto& sharpen = tree.add_node("lfs.sharpen", "Sharpen");
        sharpen.input_values["Amount"] = 0.0f;
        sharpen.input_values["Selection"] = 0.5100295992547904f;
        sharpen.properties["keep_coverage"] = true;
        ASSERT_TRUE(tree.add_link({input, "Geometry", "Sharpen", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Sharpen", "Geometry", output, "Geometry"}));
        const auto check = [&](const Geometry& result) {
            const auto& a = *result.splats;
            const auto& b = *geometry.splats;
            for (const auto& pair : {std::pair{a.means, b.means}, {a.sh0, b.sh0}, {a.shN, b.shN}, {a.scaling, b.scaling}, {a.rotation, b.rotation}, {a.opacity, b.opacity}})
                EXPECT_EQ(host<float>(pair.first), host<float>(pair.second));
        };
        auto result = evaluate(tree, {geometry, {}, 1});
        ASSERT_TRUE(result.ok);
        check(result.geometry);
        tree.add_node("lfs.join_geometry", "Join");
        tree.add_node("lfs.merge_by_distance", "Merge").input_values["Distance"] = 0.870758021072702f;
        ASSERT_TRUE(tree.add_link({"Sharpen", "Geometry", "Join", "Geometry"}));
        ASSERT_TRUE(tree.add_link({input, "Geometry", "Join", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Join", "Geometry", "Merge", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Merge", "Geometry", output, "Geometry"}));
        result = evaluate(tree, {geometry, {}, 1});
        ASSERT_TRUE(result.ok);
        check(result.geometry);
    }

    TEST_P(NodesCore, UniformReflectionPreservesAxesAndShParity) {
        auto geometry = splats(3);
        geometry.splats->scaling = tensor({-1, -1, -1, -3, -1, -1, -2, -2, -4}, {3, 3});
        geometry.splats->rotation = tensor({0.5f, 0.5f, 0.5f, 0.5f, 0, 1, 0, 0, 1, 0, 0, 0}, {3, 4});
        const auto run = [&](float scale, Device target) {
            NodeTree tree(registry_);
            auto& transform = tree.add_node("lfs.transform_geometry", "Transform");
            transform.input_values["Translation"] = glm::vec3(-1.95850298327221f, 0.01029500065166955f, 0.6724207119411831f);
            transform.input_values["Rotation"] = glm::vec3(1.6567954333253958f, -0.13583290700924433f, -0.831743214379475f);
            transform.input_values["Scale"] = scale;
            EXPECT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Transform", "Geometry"}));
            EXPECT_TRUE(tree.add_link({"Transform", "Geometry", tree.output_node().name, "Geometry"}));
            auto result = lfs::nodes::evaluate(tree, {geometry, {}, 1, target});
            EXPECT_TRUE(result.ok);
            return result.geometry;
        };
        const auto positive = run(2.0f, Device::CPU);
        const auto reference = run(-2.0f, Device::CPU);
        const auto actual = run(-2.0f, device());
        expect_finite(actual);
        EXPECT_EQ(host<float>(positive.splats->rotation), host<float>(reference.splats->rotation));
        EXPECT_EQ(host<float>(positive.splats->scaling), host<float>(reference.splats->scaling));
        const auto quaternion = host<float>(actual.splats->rotation), expected = host<float>(reference.splats->rotation);
        for (size_t i = 0; i < quaternion.size(); ++i)
            EXPECT_NEAR(quaternion[i], expected[i], 1e-6f);
        const auto sh = host<float>(actual.splats->shN), original = host<float>(positive.splats->shN);
        for (size_t i = 0; i < sh.size(); ++i) {
            const size_t coefficient = (i / 3) % 15;
            const float sign = coefficient < 3 || coefficient >= 8 ? -1 : 1;
            EXPECT_NEAR(sh[i], sign * original[i], 2e-5f);
        }
    }

    TEST_P(NodesCore, NoiseMatchesCpuBitExactly) {
        auto geometry = splats();
        std::mt19937 random(173);
        std::uniform_real_distribution<float> values(-4, 4);
        std::vector<float> positions(771);
        std::generate(positions.begin(), positions.end(), [&] { return values(random); });
        geometry.splats->means = tensor(positions, {257, 3});
        const auto configure = [](Node& node) {node.input_values["Scale"]=1.7f;node.input_values["Detail"]=3.4f;node.input_values["Roughness"]=0.63f;node.input_values["Distortion"]=0.27f;node.input_values["Seed"]=9.0f; };
        const auto reference = host<float>(field_result("lfs.noise_texture", "Fac", FLOAT_SOCKET, geometry, configure, nullptr, Device::CPU));
        const auto actual = host<float>(field_result("lfs.noise_texture", "Fac", FLOAT_SOCKET, geometry, configure));
        ASSERT_EQ(reference.size(), actual.size());
        float error = 0;
        for (size_t i = 0; i < actual.size(); ++i)
            error = std::max(error, std::abs(reference[i] - actual[i]));
        EXPECT_EQ(error, 0.0f) << GetParam().name;
        std::cout << "Noise CPU/" << GetParam().name << " max error " << error << '\n';
        // Independent NumPy gradient-table/hash/fBM reference, stored as exact
        // Float32 bits so the CPU implementation cannot silently change the stream.
        geometry.splats->means = tensor({0, 0, 0, .125f, -.25f, .75f, 1.25f, 2.5f, -.125f,
                                         -17.2f, 3.8f, 42.1f, 3.14159f, -2.71828f, 1.41421f,
                                         65536.125f, -65536.25f, 3.75f},
                                        {6, 3});
        const auto golden = host<float>(field_result("lfs.noise_texture", "Fac", FLOAT_SOCKET, geometry, configure));
        const std::array<uint32_t, 6> bits{0x3f042a67u, 0x3f06d9aau, 0x3f1f88a8u, 0x3f12f920u, 0x3f022d98u, 0x3ef5e40du};
        ASSERT_EQ(golden.size(), bits.size());
        for (size_t i = 0; i < bits.size(); ++i)
            EXPECT_EQ(std::bit_cast<uint32_t>(golden[i]), bits[i]) << i;
    }

    // Fails if serializing a graph drops an unconnected input's runtime default: Noise then samples an empty
    // vector instead of the position, and the worker serializes graphs for every evaluation.
    TEST_P(NodesCore, NoiseKeepsItsPositionDefaultThroughSerialization) {
        auto geometry = splats();
        geometry.splats->means = tensor({0, 0, 0, .123f, .27f, .38f, 1.5f, -2.25f, .75f}, {3, 3});
        const auto direct = host<float>(field_result("lfs.noise_texture", "Fac", FLOAT_SOCKET, geometry));
        const auto loaded = host<float>(field_result("lfs.noise_texture", "Fac", FLOAT_SOCKET, geometry, {}, nullptr, {}, true));
        EXPECT_EQ(loaded, direct);
        EXPECT_NE(direct[0], direct[1]);
    }

    TEST_P(NodesCore, FusedPreciseDivisionMatchesCpu) {
        if (device() == Device::CPU)
            GTEST_SKIP() << "Fused kernel requires GPU";
        namespace fused = lfs::core::fused;
        using lfs::core::DataType;
        std::mt19937 random(192);
        std::uniform_real_distribution<float> value(.1f, 16);
        std::vector<float> numerator(1025), denominator(1025), expected(1025);
        for (size_t i = 0; i < expected.size(); ++i) {
            numerator[i] = value(random);
            denominator[i] = value(random);
            expected[i] = numerator[i] / denominator[i];
        }
        fused::Builder builder(1);
        const auto a = builder.input(DataType::Float32, 1), b = builder.input(DataType::Float32, 1);
        builder.output(fused::precise_divide(a.load(), b.load()), DataType::Float32);
        const fused::Kernel kernel(builder);
        const auto actual = host<float>(kernel({expected.size()}, {tensor(numerator, {expected.size()}), tensor(denominator, {expected.size()})})[0]);
        EXPECT_EQ(actual, expected);
    }

    // Fails if precise division is not correctly rounded across the float range: subnormal inputs and results, the
    // overflow boundary, signed zeros, infinities and NaN must match the CPU bit for bit.
    TEST_P(NodesCore, FusedPreciseDivisionIsCorrectlyRoundedAcrossTheFloatRange) {
        if (device() == Device::CPU)
            GTEST_SKIP() << "Fused kernel requires GPU";
        namespace fused = lfs::core::fused;
        using lfs::core::DataType;
        const auto bits = [](std::uint32_t value) { return std::bit_cast<float>(value); };
        constexpr float inf = std::numeric_limits<float>::infinity();
        const std::vector<float> special{0.0f, -0.0f, 1.0f, -1.0f, 3.0f, 0.1f, 7.0f, 1e-38f, 3e38f, inf, -inf,
                                         std::numeric_limits<float>::quiet_NaN(), bits(1), bits(0x00400001u),
                                         bits(0x007fffffu), bits(0x00800000u), bits(0x7f7fffffu)};
        std::vector<float> numerator, denominator;
        for (const float a : special) {
            for (const float b : special) {
                numerator.push_back(a);
                denominator.push_back(b);
            }
        }
        std::mt19937 random(2766);
        std::uniform_int_distribution<std::uint32_t> any;
        for (int i = 0; i < 1 << 16; ++i) {
            numerator.push_back(bits(any(random)));
            denominator.push_back(bits(any(random)));
        }
        // Quotient exponents around the subnormal range and the overflow boundary, where rounding shifts and carries.
        std::uniform_int_distribution<std::uint32_t> fraction(0, 0x7fffffu), sign(0, 1);
        for (const int exponent : {-160, -150, -149, -140, -127, -126, -125, 126, 127, 128}) {
            for (int i = 0; i < 4096; ++i) {
                const int ey = 1 + int(any(random) % 200);
                const int ex = std::clamp(ey + exponent, 0, 254);
                numerator.push_back(bits(sign(random) << 31 | std::uint32_t(ex) << 23 | fraction(random)));
                denominator.push_back(bits(sign(random) << 31 | std::uint32_t(ey) << 23 | fraction(random)));
            }
        }
        fused::Builder builder(1);
        const auto a = builder.input(DataType::Float32, 1), b = builder.input(DataType::Float32, 1);
        builder.output(fused::precise_divide(a.load(), b.load()), DataType::Float32);
        const fused::Kernel kernel(builder);
        const size_t n = numerator.size();
        const auto actual = host<float>(kernel({n}, {tensor(numerator, {n}), tensor(denominator, {n})})[0]);
        size_t mismatches = 0;
        for (size_t i = 0; i < n; ++i) {
            const float expected = numerator[i] / denominator[i];
            const bool same = std::isnan(expected) ? std::isnan(actual[i])
                                                   : std::bit_cast<std::uint32_t>(actual[i]) == std::bit_cast<std::uint32_t>(expected);
            if (!same && ++mismatches <= 8) {
                ADD_FAILURE() << std::hex << std::bit_cast<std::uint32_t>(numerator[i]) << " / "
                              << std::bit_cast<std::uint32_t>(denominator[i]) << ": "
                              << std::bit_cast<std::uint32_t>(actual[i]) << " vs " << std::bit_cast<std::uint32_t>(expected);
            }
        }
        EXPECT_EQ(mismatches, 0u) << "of " << n;
    }

    TEST_P(NodesCore, FusedNoiseVaryingFieldsStridesAndFractionalOctavesMatchCpuExactly) {
        constexpr size_t n = 257;
        std::mt19937 random(825);
        std::uniform_real_distribution<float> value(-10, 10);
        std::vector<float> positions(n * 6), scale(n), detail(n), roughness(n), distortion(n), seed(n);
        std::generate(positions.begin(), positions.end(), [&] { return value(random); });
        for (size_t i = 0; i < n; ++i) {
            scale[i] = value(random);
            detail[i] = float(i % 65) * .25f - .5f;
            roughness[i] = float(i % 13) * .1f - .1f;
            distortion[i] = i % 3 ? .5f : 0;
            seed[i] = value(random);
        }
        auto p = tensor(positions, {n, 6}).slice(1, 0, 3);
        const auto s = tensor(scale, {n}), d = tensor(detail, {n}), r = tensor(roughness, {n}), w = tensor(distortion, {n}), k = tensor(seed, {n});
        const auto expected = host<float>(lfs::core::procedural_noise(p.cpu(), s.cpu(), d.cpu(), r.cpu(), w.cpu(), k.cpu()));
        const auto actual = host<float>(lfs::core::procedural_noise(p, s, d, r, w, k));
        ASSERT_EQ(actual.size(), n);
        for (size_t i = 0; i < n; ++i) {
            EXPECT_EQ(actual[i], expected[i]) << "element " << i;
            EXPECT_GE(actual[i], 0);
            EXPECT_LE(actual[i], 1);
            if (detail[i] <= 0)
                EXPECT_EQ(actual[i], 0);
        }
        const auto empty = Tensor::empty({0}, device());
        EXPECT_EQ(lfs::core::procedural_noise(Tensor::empty({0, 3}, device()), empty, empty, empty, empty, empty).numel(), 0);
    }

    TEST_P(NodesCore, FusedCurvesSearchControlPointsAndReuseKernels) {
        using lfs::core::CurveKnot;
        // Same-depth counts exercise cache reuse; >64 words exercises device tables.
        for (const size_t count : {2, 3, 4, 17, 23, 257}) {
            std::vector<CurveKnot> knots;
            std::vector<float> values{-0.125f};
            for (size_t i = 0; i < count; ++i) {
                const float x = float(i) / float(count - 1);
                knots.push_back({x, x * x, 2 * x});
                values.push_back(x);
                if (i + 1 < count)
                    values.push_back((float(i) + .5f) / float(count - 1));
            }
            values.push_back(1.125f);
            for (bool clamped : {false, true}) {
                const auto result = host<float>(lfs::core::interpolate_curve(tensor(values, {values.size()}), knots, clamped));
                for (size_t i = 0; i < values.size(); ++i) {
                    const float x = clamped ? std::clamp(values[i], 0.0f, 1.0f) : values[i];
                    EXPECT_NEAR(result[i], x * x, 2e-5f) << count << ":" << i;
                }
                const auto exact = host<float>(lfs::core::interpolate_curve(tensor(values, {values.size()}).slice(0, 1, 2), knots, clamped));
                EXPECT_EQ(exact[0], knots[0][1]);
            }
        }
    }

    TEST_P(NodesCore, FusedRampDuplicateStopsAllModesAndAlpha) {
        using lfs::core::ColourStop;
        using lfs::core::RampInterpolation;
        const std::vector<ColourStop> stops{{0, 0, .2f, .5f, .1f}, {.5f, .4f, .6f, .8f, .3f}, {.5f, .8f, .7f, .4f, .7f}, {1, 1, .9f, .2f, 1}};
        const auto values = tensor({-.17f, 0, .125f, .5f, .625f, 1, 2}, {7});
        for (auto mode : {RampInterpolation::Constant, RampInterpolation::Linear, RampInterpolation::Ease}) {
            for (bool alpha : {false, true}) {
                const auto actual = host<float>(lfs::core::interpolate_colour_ramp(values, stops, mode, alpha));
                const auto reference = host<float>(lfs::core::interpolate_colour_ramp(values.cpu(), stops, mode, alpha));
                ASSERT_EQ(actual.size(), reference.size());
                for (size_t i = 0; i < actual.size(); ++i)
                    EXPECT_NEAR(actual[i], reference[i], 1e-7f) << i;
                const int channels = alpha ? 1 : 3;
                EXPECT_EQ(actual[0], stops.front()[alpha ? 4 : 1]);
                EXPECT_EQ(actual[3 * channels], stops[2][alpha ? 4 : 1]);
                const auto single = host<float>(lfs::core::interpolate_colour_ramp(values, {stops.data(), 1}, mode, alpha));
                for (size_t i = 0; i < single.size(); ++i)
                    EXPECT_EQ(single[i], stops[0][alpha ? 4 : i % 3 + 1]);
            }
        }
    }

    TEST_P(NodesCore, FusedRgbCurvesPreserveBlendEndpointsAndHigherSh) {
        auto geometry = splats();
        geometry.splats->sh0 = tensor({-.5f, 0, 2, 1e20f, -1e20f, 0, .3f, .5f, 1}, {3, 3});
        for (float weight : {0.0f, .25f, 1.0f}) {
            const auto result = single("lfs.rgb_curves", geometry, [weight](Node& node) {
                node.input_values["Selection"] = weight;
                node.properties["combined"] = {{0, 0}, {1, 1}};
                node.properties["r"] = {{0, .25}, {1, .75}};
                node.properties["g"] = {{0, .25}, {1, .75}};
                node.properties["b"] = {{0, .25}, {1, .75}};
            });
            ASSERT_TRUE(result.ok);
            const auto source = host<float>(geometry.splats->sh0), actual = host<float>(result.geometry.splats->sh0);
            constexpr float c0 = 0.28209479177387814f;
            for (size_t i = 0; i < actual.size(); ++i) {
                const float rgb = source[i] * c0 + .5f, mapped = .25f + .5f * std::clamp(rgb, 0.0f, 1.0f);
                const float blended = weight == 0 ? rgb : weight == 1 ? mapped
                                                                      : rgb * (1 - weight) + mapped * weight;
                const float expected = (blended - .5f) / c0;
                EXPECT_NEAR(actual[i], expected, std::max(1e-6f, std::abs(expected) * 1e-6f));
            }
            EXPECT_EQ(host<float>(result.geometry.splats->shN), host<float>(geometry.splats->shN));
        }
    }

    TEST_P(NodesCore, PointNeighbourSpacingMatchesBruteForceWithinItsCells) {
        std::mt19937 random(17);
        std::normal_distribution<float> spread(0.0f, 0.02f);
        std::uniform_real_distribution<float> anywhere(-1.0f, 1.0f);
        std::vector<float> xyz;
        // Dense clusters, sparse points between them, duplicates and a non-finite point.
        for (int cluster = 0; cluster < 20; ++cluster) {
            const float cx = anywhere(random), cy = anywhere(random), cz = anywhere(random);
            for (int i = 0; i < 120; ++i)
                xyz.insert(xyz.end(), {cx + spread(random), cy + spread(random), cz + spread(random)});
        }
        for (int i = 0; i < 400; ++i)
            xyz.insert(xyz.end(), {anywhere(random), anywhere(random), anywhere(random)});
        xyz.insert(xyz.end(), {xyz[0], xyz[1], xyz[2], std::numeric_limits<float>::quiet_NaN(), 0, 0});
        const size_t count = xyz.size() / 3;
        const float width = 0.05f;
        const auto cell = [&](const float value) {
            return static_cast<int>(std::clamp(std::floor(value / width), -268435456.0f, 268435456.0f));
        };
        std::vector<float> expected(count, 0.0f);
        for (size_t i = 0; i < count; ++i) {
            const float* p = &xyz[i * 3];
            if (!std::isfinite(p[0]) || !std::isfinite(p[1]) || !std::isfinite(p[2]))
                continue;
            std::vector<float> near, ring;
            for (size_t j = 0; j < count; ++j) {
                const float* q = &xyz[j * 3];
                if (j == i || !std::isfinite(q[0]) || !std::isfinite(q[1]) || !std::isfinite(q[2]))
                    continue;
                int reach = 0;
                for (int axis = 0; axis < 3; ++axis)
                    reach = std::max(reach, std::abs(cell(q[axis]) - cell(p[axis])));
                const float dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2];
                const float distance = dx * dx + dy * dy + dz * dz;
                if (reach <= 1)
                    near.push_back(distance);
                else if (reach == 2)
                    ring.push_back(distance);
            }
            auto& candidates = near;
            if (near.size() < 3)
                candidates.insert(candidates.end(), ring.begin(), ring.end());
            std::ranges::sort(candidates);
            const size_t found = std::min<size_t>(3, candidates.size());
            float sum = 0;
            for (size_t k = 0; k < found; ++k)
                sum += std::sqrt(candidates[k]);
            expected[i] = found ? sum / static_cast<float>(found) : width * 4;
        }
        const auto actual = host<float>(lfs::core::point_neighbor_spacing(tensor(xyz, {count, 3}), width));
        ASSERT_EQ(actual.size(), count);
        for (size_t i = 0; i < count; ++i)
            ASSERT_NEAR(actual[i], expected[i], 1e-6f + 1e-5f * expected[i]) << "point " << i;
    }

    TEST_P(NodesCore, PointNeighbourSpacingExpandsAndHandlesEmptyInputs) {
        const auto points = tensor({0, 0, 0, 1.8f, 0, 0, 1.9f, 0, 0, 2, 0, 0}, {4, 3});
        const auto spacing = lfs::core::point_neighbor_spacing(points, 1.0f);
        const auto values = host<float>(spacing);
        EXPECT_EQ(spacing.device(), device());
        ASSERT_EQ(values.size(), 4);
        EXPECT_NEAR(values[0], 1.9f, 1e-5f);
        EXPECT_NEAR(values[1], 0.7f, 1e-5f);
        EXPECT_NEAR(values[2], 0.7f, 1e-5f);
        EXPECT_NEAR(values[3], 2.3f / 3, 1e-5f);
        EXPECT_EQ(lfs::core::point_neighbor_spacing(Tensor::empty({0, 3}, device()), 1.0f).numel(), 0);
        EXPECT_FLOAT_EQ(host<float>(lfs::core::point_neighbor_spacing(tensor({0, 0, 0}, {1, 3}), 1.0f))[0], 4.0f);
    }

    TEST_P(NodesCore, ScaleClampGuaranteesNeedleAspectOnRandomLogScales) {
        std::mt19937 random(27);
        std::uniform_real_distribution<float> value(-15, 10);
        for (int sample = 0; sample < 16; ++sample) {
            auto geometry = splats();
            std::vector<float> scales(9);
            std::generate(scales.begin(), scales.end(), [&] { return value(random); });
            geometry.splats->scaling = tensor(scales, {3, 3});
            const auto result = single("lfs.scale_clamp", geometry, [](Node& node) { node.input_values["Max Aspect"] = 4.0f; });
            ASSERT_TRUE(result.ok);
            const auto output = result.geometry.splats->scaling;
            const auto largest = output.max(1);
            const auto middle = output.sum(1) - largest - output.min(1);
            EXPECT_LE((largest - middle).exp().max().item<float>(), 4.00001f);
        }
    }

    TEST_P(NodesCore, MeshToSplatsSamplesSphereWithRadialNormalsAndDensity) {
        constexpr int rings = 32;
        constexpr int sides = 64;
        std::vector<float> vertices;
        std::vector<int> faces;
        for (int ring = 0; ring <= rings; ++ring) {
            const float latitude = std::numbers::pi_v<float> * ring / rings;
            for (int side = 0; side < sides; ++side) {
                const float longitude = 2 * std::numbers::pi_v<float> * side / sides;
                vertices.insert(vertices.end(), {std::sin(latitude) * std::cos(longitude),
                                                 std::sin(latitude) * std::sin(longitude), std::cos(latitude)});
                if (ring < rings) {
                    const int a = ring * sides + side;
                    const int b = ring * sides + (side + 1) % sides;
                    faces.insert(faces.end(), {a, b, a + sides, b, b + sides, a + sides});
                }
            }
        }
        auto mesh = std::make_shared<lfs::core::MeshData>(tensor(vertices, {vertices.size() / 3, 3}), ints(faces, {faces.size() / 3, 3}));
        mesh->normals = mesh->vertices;
        const auto geometry = geometry_from_mesh(mesh);
        const auto settings = [](Node& node) {
            node.input_values["Density"] = 40.0f;
            node.properties["seed"] = 91;
        };
        const auto result = single("lfs.mesh_to_splats", geometry, settings);
        ASSERT_TRUE(result.ok);
        ASSERT_TRUE(result.geometry.splats);
        const auto& splats = *result.geometry.splats;
        EXPECT_NEAR(splats.means.shape()[0], 4 * std::numbers::pi * 40, 3);
        const auto positions = host<float>(splats.means);
        const auto rotations = host<float>(splats.rotation);
        for (size_t i = 0; i < positions.size() / 3; ++i) {
            const glm::vec3 p{positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]};
            EXPECT_NEAR(glm::length(p), 1.0f, 0.003f);
            const float w = rotations[4 * i];
            const float x = rotations[4 * i + 1];
            const float y = rotations[4 * i + 2];
            const float z = rotations[4 * i + 3];
            const glm::vec3 normal{2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)};
            EXPECT_GT(glm::dot(normal, glm::normalize(p)), 0.9999f);
        }
        EXPECT_NEAR((splats.scaling.slice(1, 0, 1) - splats.scaling.slice(1, 2, 3)).exp().mean().item<float>(), 10, 1e-4f);
        EXPECT_NEAR(splats.opacity.sigmoid().mean().item<float>(), 0.95f, 1e-5f);
        EXPECT_EQ(splats.sh_degree, 0);
        const auto repeated = single("lfs.mesh_to_splats", geometry, settings);
        ASSERT_TRUE(repeated.ok);
        EXPECT_EQ(host<float>(repeated.geometry.splats->means), positions);
        const auto limited = single("lfs.mesh_to_splats", geometry, [](Node& node) { node.input_values["Max Count"] = std::int64_t(31); });
        ASSERT_TRUE(limited.ok);
        EXPECT_EQ(limited.geometry.splats->means.shape()[0], 31u);
    }

    TEST_P(NodesCore, MeshToSplatsPlacesTheSameSamplesOnEveryBackend) {
        const auto mesh = torus(24, 12);
        NodeTree tree(registry_);
        Node& node = tree.add_node("lfs.mesh_to_splats");
        node.input_values["Density"] = 200.0f;
        node.properties["seed"] = 1234567;
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", node.name, "Geometry"}));
        ASSERT_TRUE(tree.add_link({node.name, "Geometry", tree.output_node().name, "Geometry"}));
        const auto here = evaluate(tree, {geometry_from_mesh(mesh), {}, 1});
        const auto cpu = lfs::nodes::evaluate(tree, {geometry_from_mesh(mesh), {}, 1, Device::CPU});
        ASSERT_TRUE(here.ok && cpu.ok);
        ASSERT_TRUE(here.geometry.splats && cpu.geometry.splats);
        EXPECT_EQ(here.geometry.splats->means.device(), device());
        const auto actual = host<float>(here.geometry.splats->means);
        const auto expected = host<float>(cpu.geometry.splats->means);
        ASSERT_EQ(actual.size(), expected.size());
        ASSERT_GT(actual.size(), 3 * 4000u);
        size_t mismatches = 0;
        for (size_t i = 0; i < actual.size(); ++i)
            mismatches += std::abs(actual[i] - expected[i]) > 1e-5f;
        EXPECT_EQ(mismatches, 0u);
    }

    TEST_P(NodesCore, MeshToSplatsUsesVertexMaterialAndTextureColours) {
        auto mesh = std::make_shared<lfs::core::MeshData>(
            tensor({0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 0, 0, 4, 0, 0, 3, 1, 0}, {6, 3}),
            ints({0, 1, 2, 3, 4, 5}, {2, 3}));
        mesh->materials.resize(2);
        mesh->materials[0].base_color = {1, 0, 0, 1};
        mesh->materials[1].base_color = {0, 1, 0, 1};
        mesh->submeshes = {{0, 3, 0}, {3, 3, 1}};
        const auto run = [&] { return single("lfs.mesh_to_splats", geometry_from_mesh(mesh), [](Node& node) { node.input_values["Density"] = 64.0f; }); };
        auto result = run();
        ASSERT_TRUE(result.ok);
        auto colours = host<float>(result.geometry.splats->sh0 * 0.28209479177387814f + 0.5f);
        const auto positions = host<float>(result.geometry.splats->means);
        for (size_t i = 0; i < colours.size() / 3; ++i) {
            EXPECT_NEAR(colours[3 * i], positions[3 * i] < 2 ? 1 : 0, 1e-6f);
            EXPECT_NEAR(colours[3 * i + 1], positions[3 * i] < 2 ? 0 : 1, 1e-6f);
            EXPECT_NEAR(colours[3 * i + 2], 0, 1e-6f);
        }
        mesh->colors = tensor({0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
                               0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1},
                              {6, 4});
        result = run();
        ASSERT_TRUE(result.ok);
        EXPECT_NEAR((result.geometry.splats->sh0.slice(1, 2, 3) * 0.28209479177387814f + 0.5f).min().item<float>(), 1, 1e-5f);
        mesh->colors = {};
        mesh->materials[0].base_color = {1, 1, 1, 1};
        mesh->materials[1].base_color = {1, 1, 1, 1};
        mesh->materials[0].albedo_tex = mesh->materials[1].albedo_tex = 1;
        mesh->texture_images = {{{255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255}, 2, 2, 4}};
        mesh->texcoords = Tensor::full({6, 2}, 0.25f, device());
        result = run();
        ASSERT_TRUE(result.ok);
        EXPECT_NEAR((result.geometry.splats->sh0.slice(1, 0, 1) * 0.28209479177387814f + 0.5f).min().item<float>(), 1, 1e-5f);
        EXPECT_NEAR((result.geometry.splats->sh0.slice(1, 2, 3) * 0.28209479177387814f + 0.5f).max().item<float>(), 0, 1e-5f);
    }

    TEST_P(NodesCore, DenseCellNeighbourSpacingIsIndependentOfHashInsertionOrder) {
        std::vector<float> positions;
        for (int index = 0; index < 256; ++index)
            positions.insert(positions.end(), {index / 512.0f, 0.0f, 0.0f});
        const auto points = tensor(positions, {256, 3});
        const auto expected = lfs::core::point_neighbor_spacing(points.to(Device::CPU), 1.0f).to_vector();
        for (int repeat = 0; repeat < 5; ++repeat) {
            const auto actual = lfs::core::point_neighbor_spacing(points, 1.0f).to_vector();
            ASSERT_EQ(actual.size(), expected.size());
            EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
        }
    }

    TEST_P(NodesCore, PointsToSplatsAutoRadiusUsesThreeNearestNeighbours) {
        std::vector<float> points;
        for (int z = 0; z < 3; ++z)
            for (int y = 0; y < 3; ++y)
                for (int x = 0; x < 3; ++x)
                    points.insert(points.end(), {float(x), float(y), float(z)});
        Geometry geometry;
        geometry.points = PointsComponent{tensor(points, {27, 3}), Tensor::ones({27, 3}, device()), {}};
        const auto result = single("lfs.points_to_splats", geometry);
        ASSERT_TRUE(result.ok);
        for (const float radius : host<float>(result.geometry.splats->scaling.exp()))
            EXPECT_NEAR(radius, 0.5f, 1e-5f);
        points.insert(points.end(), {1000, 1000, 1000});
        geometry.points = PointsComponent{tensor(points, {28, 3}), Tensor::ones({28, 3}, device()), {}};
        const auto clamped = single("lfs.points_to_splats", geometry);
        ASSERT_TRUE(clamped.ok);
        EXPECT_NEAR(clamped.geometry.splats->scaling.exp().max().item<float>(), 2.0f, 1e-5f);
    }

    TEST_P(NodesCore, PointsToSplatsLargeCloudKeepsLocalRadiusVariation) {
        constexpr size_t count = 300001;
        Geometry geometry;
        geometry.points = PointsComponent{
            Tensor::uniform({count, 3}, -1, 1, device(), lfs::core::DataType::Float32, 17),
            Tensor::ones({count, 3}, device()),
            {}};
        const auto result = single("lfs.points_to_splats", geometry);
        ASSERT_TRUE(result.ok);
        ASSERT_EQ(result.geometry.splats->means.shape()[0], count);
        const auto scales = result.geometry.splats->scaling;
        EXPECT_LT(scales.min().item<float>(), scales.max().item<float>());
        EXPECT_TRUE(std::isfinite(scales.min().item<float>()));
    }

    TEST_P(NodesCore, SeededSamplingDoesNotAdvanceGlobalRandomSequence) {
        Tensor::manual_seed(71);
        const auto expected = host<float>(Tensor::rand({10}, device()));
        Tensor::manual_seed(71);
        const auto weights = Tensor::ones({10}, device());
        const auto uniform = Tensor::uniform({10}, 0, 1, device(), lfs::core::DataType::Float32, 13);
        const auto samples = Tensor::multinomial(weights, 30, true, 29);
        EXPECT_EQ(host<float>(uniform), host<float>(Tensor::uniform({10}, 0, 1, device(), lfs::core::DataType::Float32, 13)));
        EXPECT_EQ(host<int64_t>(samples), host<int64_t>(Tensor::multinomial(weights, 30, true, 29)));
        EXPECT_EQ(host<float>(Tensor::rand({10}, device())), expected);
    }

    TEST_P(NodesCore, FloaterPreviewKeepsOnlyUnchangedCandidates) {
        const auto geometry = splats();
        const auto result = single("lfs.remove_floaters", geometry, [](Node& node) {
            node.input_values["Isolation Radius"] = 0.0f;
            node.input_values["Min Opacity"] = 0.4f;
            node.properties["preview"] = true;
        });
        ASSERT_TRUE(result.ok);
        ASSERT_EQ(result.geometry.splats->means.shape()[0], 1u);
        const auto& candidate = *result.geometry.splats;
        const auto& original = *geometry.splats;
        for (const auto pair : {std::pair{&candidate.means, &original.means}, {&candidate.sh0, &original.sh0}, {&candidate.shN, &original.shN}, {&candidate.opacity, &original.opacity}, {&candidate.scaling, &original.scaling}, {&candidate.rotation, &original.rotation}})
            EXPECT_EQ(host<float>(*pair.first), host<float>(pair.second->slice(0, 2, 3)));
        EXPECT_EQ(host<float>(candidate.attributes.at("weight")), (std::vector<float>{30}));
    }

    TEST_P(NodesCore, ExactNeighboursKeepCoincidentPointsButExcludeTheSameIndex) {
        const auto positions = tensor({0, 0, 0, 0, 0, 0, 2, 0, 0, 8, 0, 0}, {4, 3});
        const auto references = tensor({1, 1, 1, 1}, {4}).gt(0);
        EXPECT_EQ(host<std::uint8_t>(lfs::core::radius_neighbors(positions, references, 1.0f, true)),
                  (std::vector<std::uint8_t>{1, 1, 0, 0}));
        const auto first = tensor({1, 0, 0, 0}, {4}).gt(0);
        EXPECT_EQ(host<std::uint8_t>(lfs::core::radius_neighbors(positions, first, 1.0f, true)),
                  (std::vector<std::uint8_t>{0, 1, 0, 0}));
        const auto queries = tensor({1, 0, 1, 0}, {4}).gt(0);
        EXPECT_EQ(host<std::uint8_t>(lfs::core::radius_neighbors(positions, references, 1.0f, true, &queries)),
                  (std::vector<std::uint8_t>{1, 0, 0, 0}));
    }

    TEST_P(NodesCore, RelativeFloaterRadiusKeepsProportionalBackgroundAndRemovesSmallOutlier) {
        auto geometry = splats();
        geometry.splats->means =
            tensor({0, 0, 0, 0.1f, 0, 0, 10, 0, 0, 30, 0, 0, 100, 0, 0}, {5, 3});
        geometry.splats->sh0 = Tensor::zeros({5, 3}, device());
        geometry.splats->shN = Tensor::zeros({5, 3, 3}, device());
        geometry.splats->scaling = tensor({std::log(0.1f), std::log(0.1f), std::log(0.1f),
                                           std::log(0.1f), std::log(0.1f), std::log(0.1f),
                                           std::log(10.0f), std::log(10.0f), std::log(10.0f),
                                           std::log(10.0f), std::log(10.0f), std::log(10.0f),
                                           std::log(0.1f), std::log(0.1f), std::log(0.1f)},
                                          {5, 3});
        geometry.splats->rotation =
            tensor({1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0}, {5, 4});
        geometry.splats->opacity = Tensor::zeros({5}, device());
        geometry.splats->attributes.clear();
        const auto result = single("lfs.remove_floaters", geometry);
        ASSERT_TRUE(result.ok);
        ASSERT_EQ(result.geometry.splats->means.shape()[0], 4u);
        const auto positions = host<float>(result.geometry.splats->means);
        EXPECT_EQ(positions, (std::vector<float>{0, 0, 0, 0.1f, 0, 0, 10, 0, 0, 30, 0, 0}));
    }

    TEST_P(NodesCore, HostGenerationInvalidatesOnlyHostDependentOutputs) {
        struct Host : EvalHost {
            std::uint64_t revision = 1;
            Geometry geometry;
            std::uint64_t generation() const override {
                return revision;
            }
            std::optional<Geometry> object_geometry(std::string_view, TransformSpace) override {
                return geometry;
            }
        } host;
        host.geometry = splats();
        int calls = 0;
        NodeTypeInfo type;
        type.id = "test.host";
        type.uses_host = true;
        type.outputs = {{"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)}};
        type.evaluate = [&](NodeContext& context) {
            ++calls;
            context.set_output("Geometry",
                               *context.host()->object_geometry("object", TransformSpace::Original));
        };
        registry_.register_type(std::move(type));
        NodeTree tree(registry_);
        tree.add_node("test.host", "Host");
        ASSERT_TRUE(tree.add_link({"Host", "Geometry", tree.output_node().name, "Geometry"}));
        EvalCache cache;
        ASSERT_TRUE(evaluate(tree, {{}, {}, 1}, &host, &cache).ok);
        ASSERT_TRUE(evaluate(tree, {{}, {}, 1}, &host, &cache).ok);
        EXPECT_EQ(calls, 1);
        ++host.revision;
        ASSERT_TRUE(evaluate(tree, {{}, {}, 1}, &host, &cache).ok);
        EXPECT_EQ(calls, 2);
        const auto missing = evaluate(tree, {{}, {}, 1}, nullptr, &cache);
        EXPECT_FALSE(missing.ok);
        EXPECT_TRUE(missing.errors.contains("Host"));
    }

    TEST_P(NodesCore, InvalidPowerReportsNodeErrorInsteadOfPublishingNonFiniteOpacity) {
        NodeTree tree(registry_);
        auto& math = tree.add_node("lfs.math", "Power");
        math.properties["operation"] = "power";
        tree.add_node("lfs.set_opacity", "Opacity");
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Opacity", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Power", "Value", "Opacity", "Opacity"}));
        ASSERT_TRUE(tree.add_link({"Opacity", "Geometry", tree.output_node().name, "Geometry"}));
        for (const auto [base, exponent] : {std::pair{-1.0689604f, -0.848964f}, {0.0f, -1.0f}, {1e30f, 2.0f}}) {
            SCOPED_TRACE(std::format("base={} exponent={}", base, exponent));
            tree.find_node("Power")->input_values["A"] = base;
            tree.find_node("Power")->input_values["B"] = exponent;
            const auto result = evaluate(tree, {.geometry = splats()});
            EXPECT_FALSE(result.ok);
            ASSERT_TRUE(result.errors.contains("Opacity"));
            EXPECT_NE(result.errors.at("Opacity").find("Set Opacity received 3 non-finite values"), std::string::npos);
            tree.find_node("Opacity")->input_values["Selection"] = 0.0f;
            const auto unselected = evaluate(tree, {.geometry = splats()});
            ASSERT_TRUE(unselected.ok);
            EXPECT_EQ(host<float>(unselected.geometry.splats->opacity), (std::vector<float>{0, 1, -1}));
            tree.find_node("Opacity")->input_values["Selection"] = 1.0f;
        }
        const auto valid = field_result("lfs.math", "Value", FLOAT_SOCKET, splats(), [](Node& node) {
            node.properties["operation"] = "power";
            node.input_values["A"] = -2.0f;
            node.input_values["B"] = 3.0f;
        });
        EXPECT_EQ(host<float>(valid), (std::vector<float>{-8, -8, -8}));
    }

    TEST_P(NodesCore, ArithmeticOperationsMatchExpectedValues) {
        const std::vector<std::pair<std::string, float>> expected = {{"add", 2.75f},
                                                                     {"subtract", -1.25f},
                                                                     {"multiply", 1.5f},
                                                                     {"divide", 0.375f},
                                                                     {"power", 0.5625f},
                                                                     {"minimum", 0.75f},
                                                                     {"maximum", 2},
                                                                     {"absolute", 0.75f},
                                                                     {"sqrt", std::sqrt(0.75f)},
                                                                     {"floor", 0},
                                                                     {"fraction", 0.75f},
                                                                     {"sine", std::sin(0.75f)},
                                                                     {"cosine", std::cos(0.75f)},
                                                                     {"greater_than", 0},
                                                                     {"less_than", 1},
                                                                     {"clamp", 0.75f}};
        for (const auto& [operation, expected_value] : expected) {
            SCOPED_TRACE(operation);
            auto value = field_result("lfs.math", "Value", FLOAT_SOCKET, splats(), [&](Node& node) {
                node.properties["operation"] = operation;
                node.input_values["A"] = 0.75f;
                node.input_values["B"] = 2.0f;
            });
            ASSERT_TRUE(value.is_valid());
            for (const auto scalar : host<float>(value))
                EXPECT_NEAR(scalar, expected_value, 1e-5f);
        }
        auto zero = field_result("lfs.math", "Value", FLOAT_SOCKET, splats(), [](Node& node) {
            node.properties["operation"] = "divide";
            node.input_values["A"] = 3.0f;
        });
        EXPECT_EQ(host<float>(zero), (std::vector<float>{0, 0, 0}));
        auto fraction = field_result("lfs.math", "Value", FLOAT_SOCKET, splats(), [](Node& node) {
            node.properties["operation"] = "fraction";
            node.input_values["A"] = -0.25f;
        });
        EXPECT_EQ(host<float>(fraction), (std::vector<float>{0.75f, 0.75f, 0.75f}));
        auto mapped = field_result("lfs.map_range", "Result", FLOAT_SOCKET, splats(), [](Node& node) {
            node.input_values["Value"] = 3.0f;
            node.input_values["From Max"] = 4.0f;
            node.input_values["To Min"] = -2.0f;
            node.input_values["To Max"] = 2.0f;
        });
        EXPECT_EQ(host<float>(mapped), (std::vector<float>{1, 1, 1}));
    }

    TEST_P(NodesCore, VectorCompareAndBooleanOperationsMatchExpectedValues) {
        const std::vector<std::pair<std::string, glm::vec3>> vectors = {
            {"add", {3, 2, 3}},
            {"subtract", {-1, 2, 1}},
            {"multiply", {2, 0, 2}},
            {"scale", {2, 4, 4}},
            {"normalise", {1.0f / 3, 2.0f / 3, 2.0f / 3}}};
        const auto configure = [](Node& node) {
            node.input_values["A"] = glm::vec3(1, 2, 2);
            node.input_values["B"] = glm::vec3(2, 0, 1);
            node.input_values["Scale"] = 2.0f;
        };
        for (const auto& [operation, expected] : vectors) {
            SCOPED_TRACE(operation);
            auto value = field_result("lfs.vector_math", "Vector", VECTOR_SOCKET, splats(), [&](Node& node) {
                configure(node);
                node.properties["operation"] = operation;
            });
            ASSERT_TRUE(value.is_valid());
            const auto values = host<float>(value);
            for (size_t index = 0; index < values.size(); ++index)
                EXPECT_NEAR(values[index], expected[index % 3], 1e-5f);
        }
        for (const auto& [operation, expected] :
             std::vector<std::pair<std::string, float>>{{"length", 3},
                                                        {"distance", std::sqrt(6.0f)},
                                                        {"dot", 4}}) {
            auto value = field_result("lfs.vector_math", "Value", FLOAT_SOCKET, splats(), [&](Node& node) {
                configure(node);
                node.properties["operation"] = operation;
            });
            ASSERT_TRUE(value.is_valid());
            for (const auto scalar : host<float>(value))
                EXPECT_NEAR(scalar, expected, 1e-5f);
        }
        for (const auto& [operation, expected] :
             std::vector<std::pair<std::string, bool>>{{"less_than", false},
                                                       {"less_equal", true},
                                                       {"greater_than", false},
                                                       {"greater_equal", true},
                                                       {"equal", true},
                                                       {"not_equal", false}}) {
            auto value = field_result("lfs.compare", "Result", BOOL_SOCKET, splats(), [&](Node& node) {
                node.properties["operation"] = operation;
                node.input_values["A"] = 1.0f;
                node.input_values["B"] = 1.0f;
            });
            ASSERT_TRUE(value.is_valid());
            for (const auto scalar : host<std::uint8_t>(value))
                EXPECT_EQ(scalar != 0, expected);
        }
        for (const auto& [operation, expected] : std::vector<std::pair<std::string, bool>>{{"and", false},
                                                                                           {"or", true},
                                                                                           {"not", false},
                                                                                           {"xor", true}}) {
            auto value = field_result("lfs.boolean_math", "Result", BOOL_SOCKET, splats(), [&](Node& node) {
                node.properties["operation"] = operation;
                node.input_values["A"] = true;
                node.input_values["B"] = false;
            });
            ASSERT_TRUE(value.is_valid());
            for (const auto scalar : host<std::uint8_t>(value))
                EXPECT_EQ(scalar != 0, expected);
        }
    }

    TEST_P(NodesCore, ColourFieldsAndRandomHashStayOnTheirDevice) {
        auto geometry = splats();
        geometry.splats->sh0 = (tensor({1, 0, 0, 0, 1, 0, 0, 0, 1}, {3, 3}) - 0.5f) / 0.28209479177387814f;
        auto hsv = field_result("lfs.hsv_range", "Selection", FLOAT_SOCKET, geometry, [](Node& node) {
            node.input_values["Hue Range"] = 0.05f;
        });
        EXPECT_EQ(host<float>(hsv), (std::vector<float>{1, 0, 0}));
        auto key = field_result("lfs.colour_key", "Selection", FLOAT_SOCKET, geometry, [](Node& node) {
            node.input_values["Colour"] = glm::vec4(1, 0, 0, 1);
        });
        EXPECT_EQ(host<float>(key), (std::vector<float>{1, 0, 0}));
        auto colour = field_result("lfs.combine_colour", "Colour", COLOUR_SOCKET, geometry, [](Node& node) {
            node.properties["mode"] = "hsv";
            node.input_values["R"] = 1.0f / 3;
            node.input_values["G"] = 1.0f;
            node.input_values["B"] = 0.75f;
        });
        ASSERT_TRUE(colour.is_valid());
        const auto channels = host<float>(colour);
        for (size_t row = 0; row < channels.size(); row += 3) {
            EXPECT_NEAR(channels[row], 0, 1e-6f);
            EXPECT_NEAR(channels[row + 1], 0.75f, 1e-6f);
            EXPECT_NEAR(channels[row + 2], 0, 1e-6f);
        }
        auto hue = field_result("lfs.separate_colour", "R", FLOAT_SOCKET, geometry, [](Node& node) {
            node.properties["mode"] = "hsv";
            node.input_values["Colour"] = glm::vec3(0, 0.75f, 0);
        });
        for (const auto value : host<float>(hue))
            EXPECT_NEAR(value, 1.0f / 3, 1e-6f);
        const auto seeded = [](Node& node) {
            node.properties["seed"] = 17;
        };
        auto random_a = field_result("lfs.random_value", "Value", FLOAT_SOCKET, geometry, seeded);
        auto random_b = field_result("lfs.random_value", "Value", FLOAT_SOCKET, geometry, seeded);
        EXPECT_EQ(random_a.device(), device());
        EXPECT_EQ(host<float>(random_a), host<float>(random_b));
        for (const auto value : host<float>(random_a)) {
            EXPECT_GE(value, 0);
            EXPECT_LT(value, 1);
        }
    }

    TEST_P(NodesCore, ColourSelectionsAndHsvUseClampedDisplayedColour) {
        constexpr float c0 = 0.28209479177387814f;
        auto geometry = splats(0);
        geometry.splats->sh0 =
            (tensor({1.2f, 0.4f, -0.2f, -0.2f, 0.8f, 0.4f, 0.2f, 0.4f, 0.6f}, {3, 3}) -
             0.5f) /
            c0;

        const auto hsv = field_result("lfs.hsv_range", "Selection", FLOAT_SOCKET, geometry, [](Node& node) {
            node.input_values["Hue"] = 1.0f / 15.0f;
            node.input_values["Hue Range"] = 0.01f;
            node.input_values["Saturation Min"] = 0.99f;
            node.input_values["Saturation Max"] = 1.0f;
            node.input_values["Value Min"] = 0.9f;
        });
        EXPECT_EQ(host<float>(hsv), (std::vector<float>{1, 0, 0}));

        const auto keyed = field_result("lfs.colour_key", "Selection", FLOAT_SOCKET, geometry, [](Node& node) {
            node.input_values["Colour"] = glm::vec3(1.0f, 0.4f, 0.0f);
            node.input_values["Tolerance"] = 1e-4f;
        });
        EXPECT_EQ(host<float>(keyed), (std::vector<float>{1, 0, 0}));

        const auto channel_value = [&](std::string_view output) {
            return field_result("lfs.separate_colour", output, FLOAT_SOCKET, geometry, [](Node& node) {
                node.properties["mode"] = "hsv";
                node.input_values["Colour"] = glm::vec3(-0.2f, 0.8f, 0.4f);
            });
        };
        for (const float value : host<float>(channel_value("R")))
            EXPECT_NEAR(value, 5.0f / 12.0f, 1e-6f);
        for (const float value : host<float>(channel_value("G")))
            EXPECT_FLOAT_EQ(value, 1.0f);
        for (const float value : host<float>(channel_value("B")))
            EXPECT_NEAR(value, 0.8f, 1e-6f);
    }

    TEST_P(NodesCore, SharedFieldInDiamondIsComputedOnce) {
        int calls = 0;
        NodeTypeInfo source;
        source.id = "test.field_counter";
        source.outputs = {{"Value", "Value", std::string(FLOAT_SOCKET)}};
        source.evaluate = [&](NodeContext& context) {
            context.set_output("Value",
                               Field(std::string(FLOAT_SOCKET), [&](const FieldContext& domain, FieldMemo&) {
                                   ++calls;
                                   return Tensor::full({domain.size()}, 0.25f, domain.device());
                               }));
        };
        registry_.register_type(std::move(source));
        NodeTree tree(registry_);
        tree.add_node("test.field_counter", "Counter");
        tree.add_node("lfs.math", "Math");
        tree.add_node("lfs.set_opacity", "Set");
        ASSERT_TRUE(tree.add_link({"Counter", "Value", "Math", "A"}));
        ASSERT_TRUE(tree.add_link({"Counter", "Value", "Math", "B"}));
        ASSERT_TRUE(tree.add_link({"Math", "Value", "Set", "Opacity"}));
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Set", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Set", "Geometry", tree.output_node().name, "Geometry"}));
        const auto result = evaluate(tree, {splats(), {}, 1});
        ASSERT_TRUE(result.ok);
        EXPECT_EQ(calls, 1);
        for (const auto value : host<float>(result.geometry.splats->opacity.sigmoid()))
            EXPECT_NEAR(value, 0.5f, 1e-6f);
    }

    TEST_P(NodesCore, StoredSelectionCachesDecodedMaskAndFollowsTensorBackend) {
        NodeTree tree(registry_);
        auto& node = tree.add_node("lfs.stored_selection", "Stored");
        set_stored_selection(node,
                             Tensor::from_vector(std::vector<bool>{true, false, true}, {3}, Device::CPU));
        node.properties["invert"] = true;
        const auto json = tree.to_json();
        auto loaded = NodeTree::from_json(json, registry_);
        const auto field = stored_selection_field(*loaded.find_node("Stored"));
        auto geometry = splats();
        FieldContext context{Domain::Splat, &*geometry.splats, nullptr, nullptr,
                             geometry.splats->means.debug_id()};
        for (const auto backend : {GpuBackend::Metal, GpuBackend::Vulkan, GpuBackend::CUDA}) {
            if (!lfs::core::gpu_backend_available(backend))
                continue;
            lfs::core::GpuBackendScope other_scope(backend);
            FieldMemo memo;
            const auto value = field.evaluate(context, memo);
            EXPECT_EQ(value.device(), geometry.splats->means.device());
            EXPECT_EQ(lfs::core::gpu_backend_of(value), lfs::core::gpu_backend_of(geometry.splats->means));
            EXPECT_EQ(host<std::uint8_t>(value), (std::vector<std::uint8_t>{0, 1, 0}));
            auto other_geometry = splats();
            FieldContext other_context{Domain::Splat, &*other_geometry.splats, nullptr, nullptr,
                                       other_geometry.splats->means.debug_id()};
            FieldMemo other_memo;
            const auto other_value = field.evaluate(other_context, other_memo);
            EXPECT_EQ(lfs::core::gpu_backend_of(other_value),
                      lfs::core::gpu_backend_of(other_geometry.splats->means));
            EXPECT_EQ(host<std::uint8_t>(other_value), (std::vector<std::uint8_t>{0, 1, 0}));
        }
    }

    TEST_P(NodesCore, StoredSelectionRejectsChangedElementCount) {
        constexpr size_t captured_count = 935'592;
        constexpr size_t received_count = 467'796;
        NodeTree tree(registry_);
        auto& stored = tree.add_node("lfs.stored_selection", "Stored");
        set_stored_selection(stored, Tensor::zeros({captured_count}, Device::CPU,
                                                   lfs::core::DataType::Bool));
        tree.add_node("lfs.delete_geometry", "Delete");
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Delete", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Stored", "Selection", "Delete", "Selection"}));
        ASSERT_TRUE(tree.add_link({"Delete", "Geometry", tree.output_node().name, "Geometry"}));
        auto geometry = splats(0);
        geometry.splats->means = Tensor::zeros({received_count, 3}, device());
        const auto result = evaluate(tree, {std::move(geometry), {}, 1});
        EXPECT_FALSE(result.ok);
        ASSERT_TRUE(result.errors.contains("Stored"));
        EXPECT_EQ(result.errors.at("Stored"),
                  "Stored selection was captured on 935,592 splats but receives 467,796 — a node or modifier before it "
                  "changes the count; recapture or move it before that change.");
    }

    class NodesCoreScale : public NodesCore {};

    TEST_P(NodesCoreScale, TexturesAndCurvesStayInteractive) {
        if (device() == Device::CPU)
            GTEST_SKIP() << "Performance regression requires GPU";
        constexpr size_t count = 1'000'000;
        auto geometry = splats(0);
        geometry.splats->means = Tensor::uniform({count, 3}, -5, 5, device(), lfs::core::DataType::Float32, 31);
        geometry.splats->sh0 = Tensor::uniform({count, 3}, -1, 1, device(), lfs::core::DataType::Float32, 17);
        const auto measure = [&](const char* label, double budget, auto action) {
            action().sum().template item<float>(); // compile and warm the exact field/kernel layout
            const auto start = std::chrono::steady_clock::now();
            action().sum().template item<float>();
            const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            std::cout << "[NodesTexturePerformance " << GetParam().name << "] " << label << " " << ms << " ms\n";
            if (const char* perf = std::getenv("LFS_NODES_PERF"); perf && std::string_view(perf) == "1")
                EXPECT_LT(ms, budget) << label;
        };
        for (float detail : {0.0f, 2.5f, 8.0f}) {
            const auto label = "Noise Texture 1M detail=" + std::to_string(detail) + " distortion=0.5";
            measure(label.c_str(), 15, [&] {
                return field_result("lfs.noise_texture", "Fac", FLOAT_SOCKET, geometry, [detail](Node& node) {
                    node.input_values["Detail"] = detail;
                    node.input_values["Distortion"] = .5f;
                });
            });
        }
        measure("Gradient Texture radial 1M", 5, [&] {
            return field_result("lfs.gradient_texture", "Fac", FLOAT_SOCKET, geometry, [](Node& node) {
                node.properties["type"] = "radial";
                node.input_values["Vector"] = position_field();
            });
        });
        measure("Colour Ramp 1M", 5, [&] {
            return field_result("lfs.colour_ramp", "Colour", COLOUR_SOCKET, geometry, [](Node& node) {
                node.properties["stops"] = {{0, 0, 0, 0, 1}, {.25, .8, .3, .1, 1}, {.6, .2, .8, .1, 1}, {1, 1, 1, 1, 1}};
            });
        });
        measure("Float Curve 1M", 5, [&] {
            return field_result("lfs.float_curve", "Value", FLOAT_SOCKET, geometry, [](Node& node) {
                node.properties["points"] = {{0, 0}, {.25, .4}, {.6, .5}, {1, 1}};
            });
        });
        measure("RGB Curves 1M", 5, [&] {
            auto result = single("lfs.rgb_curves", geometry, [](Node& node) {
                node.properties["combined"] = {{0, 0}, {.25, .4}, {.6, .5}, {1, 1}};
            });
            EXPECT_TRUE(result.ok);
            return result.geometry.splats->sh0;
        });
    }

    TEST_P(NodesCoreScale, MillionSplatsColourSelectionCleanupAndTorus) {
        if (device() == Device::CPU)
            GTEST_SKIP() << "Scale test requires a GPU backend";
        constexpr size_t count = 1'000'000;
        using lfs::core::DataType;
        auto positions = Tensor::rand({count, 3}, device()) * 5 - 2.5f;
        // Guaranteed four-neighbour clusters inside and outside the torus keep
        // this pipeline test deterministic now that isolation counts are exact.
        positions.slice(0, 0, 5).copy_from(tensor({1.5f, 0, 0}, {1, 3}).expand({5, 3}));
        positions.slice(0, 5, 10).copy_from(Tensor::zeros({5, 3}, device()));
        positions.slice(0, count - 1, count).copy_from(Tensor::full({1, 3}, 20, device()));
        Geometry geometry;
        geometry.splats = SplatsComponent{
            positions,
            (Tensor::rand({count, 3}, device()) - 0.5f) / 0.28209479177387814f,
            Tensor::rand({count, 15, 3}, device()) * 0.1f,
            Tensor::full({count, 3}, std::log(0.01f), device()),
            Tensor::cat({Tensor::ones({count, 1}, device()), Tensor::zeros({count, 3}, device())}, 1),
            Tensor::rand({count}, device()).clamp(1e-5f, 1 - 1e-5f).logit(),
            3,
            1,
            {}};
        NodeTree tree(registry_);
        geometry.splats->opacity.slice(0, 0, 10).copy_from(Tensor::zeros({10}, device()));
        tree.add_node("lfs.colour_correct", "Correct");
        tree.find_node("Correct")->input_values["Exposure"] = 0.2f;
        tree.find_node("Correct")->input_values["Saturation"] = 0.8f;
        tree.add_node("lfs.hsv_range", "HSV");
        tree.find_node("HSV")->input_values["Hue Range"] = 0.12f;
        tree.add_node("lfs.set_colour", "SetColour");
        tree.find_node("SetColour")->input_values["Colour"] = glm::vec3(0.2f, 0.8f, 0.4f);
        tree.add_node("lfs.remove_floaters", "Isolation1");
        tree.find_node("Isolation1")->input_values["Isolation Radius"] = 0.015f;
        tree.find_node("Isolation1")->input_values["Min Opacity"] = 0.1f;
        tree.find_node("Isolation1")->properties["relative_to_size"] = false;
        tree.add_node("lfs.remove_floaters", "Isolation4");
        tree.find_node("Isolation4")->input_values["Isolation Radius"] = 0.015f;
        tree.find_node("Isolation4")->input_values["Min Neighbours"] = std::int64_t(4);
        tree.find_node("Isolation4")->properties["relative_to_size"] = false;
        tree.add_node("lfs.inside_mesh", "Torus");
        tree.find_node("Torus")->input_values["Mesh"] = geometry_from_mesh(torus());
        tree.add_node("lfs.delete_geometry", "Delete");
        std::string previous = tree.input_node().name;
        for (const auto* name : {"Correct", "SetColour", "Isolation1", "Isolation4", "Delete"}) {
            ASSERT_TRUE(tree.add_link({previous, "Geometry", name, "Geometry"}));
            previous = name;
        }
        ASSERT_TRUE(tree.add_link({"HSV", "Selection", "SetColour", "Selection"}));
        ASSERT_TRUE(tree.add_link({"Torus", "Selection", "Delete", "Selection"}));
        ASSERT_TRUE(tree.add_link({"Delete", "Geometry", tree.output_node().name, "Geometry"}));
        EvalCache cache;
        const auto start = std::chrono::steady_clock::now();
        const auto result = evaluate(tree, {geometry, {}, 1}, nullptr, &cache);
        ASSERT_TRUE(result.ok) << (result.errors.empty() ? "" : result.errors.begin()->second);
        const auto& output = *result.geometry.splats;
        EXPECT_FALSE(output.means.has_nan());
        EXPECT_FALSE(output.sh0.has_nan());
        EXPECT_FALSE(output.shN.has_nan());
        EXPECT_FALSE(output.scaling.has_nan());
        const auto after_one =
            cache.nodes.at("Isolation1").outputs.at("Geometry").get_if<Geometry>()->splats->means.shape()[0];
        const auto after_four =
            cache.nodes.at("Isolation4").outputs.at("Geometry").get_if<Geometry>()->splats->means.shape()[0];
        EXPECT_LT(after_one, count);
        EXPECT_GT(after_one, 0u);
        EXPECT_LT(after_four, after_one);
        EXPECT_GT(after_four, 0u);
        EXPECT_LT(output.means.shape()[0], after_four);
        EXPECT_GT(output.means.shape()[0], 0u);
        EXPECT_EQ(output.shN.shape()[1], 15u);
        std::cout << "[NodesCoreScale " << GetParam().name << "] counts " << count << " -> " << after_one
                  << " -> " << after_four << " -> " << output.means.shape()[0] << '\n';
        for (const auto& [name, time] : result.time_ms)
            std::cout << "[NodesCoreScale " << GetParam().name << "] " << name << " " << time << " ms\n";
        const auto elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        std::cout << "[NodesCoreScale " << GetParam().name << "] synchronized total " << elapsed << " ms\n";
    }

    TEST_P(NodesCore, CpuMeshUploadsOnceAndJoinsGpuSplats) {
        if (device() == Device::CPU)
            GTEST_SKIP() << "GPU upload contract";
        auto mesh = std::make_shared<lfs::core::MeshData>();
        mesh->vertices = Tensor::from_vector({0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f}, {3, 3}, Device::CPU);
        mesh->indices = Tensor::from_vector({0, 1, 2}, {1, 3}, Device::CPU);
        mesh->texcoords = Tensor::zeros({3, 2}, Device::CPU);
        mesh->materials.emplace_back();
        mesh->materials[0].albedo_tex = 1;
        mesh->texture_images.push_back({{255, 128, 0}, 1, 1, 3});
        GeometryDeviceCache uploads;
        const auto first = uploads.convert(geometry_from_mesh(mesh), Device::GPU);
        const auto second = uploads.convert(geometry_from_mesh(mesh), Device::GPU);
        EXPECT_EQ(first.mesh->mesh, second.mesh->mesh);
        EXPECT_EQ(first.mesh->textures[0].data_ptr(), second.mesh->textures[0].data_ptr());
        EXPECT_EQ(first.mesh->mesh->vertices.device(), Device::GPU);
        EXPECT_EQ(first.mesh->textures[0].device(), Device::GPU);
        mesh->mark_dirty();
        const auto third = uploads.convert(geometry_from_mesh(mesh), Device::GPU);
        EXPECT_NE(first.mesh->mesh, third.mesh->mesh);
        EXPECT_EQ(mesh->vertices.device(), Device::CPU);
        const auto snapshot = mesh->readOnlySnapshot();
        EXPECT_EQ(snapshot->id(), mesh->id());
        EXPECT_EQ(snapshot->generation(), mesh->generation());
        NodeTypeInfo source;
        source.id = "test.cpu_mesh";
        source.outputs = {{"Geometry", "Geometry", std::string(GEOMETRY_SOCKET)}};
        source.evaluate = [mesh](NodeContext& context) { context.set_output("Geometry", geometry_from_mesh(mesh)); };
        registry_.register_type(std::move(source));
        NodeTree tree(registry_);
        tree.add_node("test.cpu_mesh", "Mesh");
        tree.add_node("lfs.mesh_to_splats", "Sample").input_values["Max Count"] = int64_t(12);
        tree.add_node("lfs.join_geometry", "Join");
        ASSERT_TRUE(tree.add_link({"Mesh", "Geometry", "Sample", "Geometry"}));
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Join", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Sample", "Geometry", "Join", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Join", "Geometry", tree.output_node().name, "Geometry"}));
        const auto result = lfs::nodes::evaluate(tree, {.geometry = splats()});
        ASSERT_TRUE(result.ok);
        EXPECT_EQ(result.geometry.splats->means.shape()[0], 15);
        EXPECT_EQ(result.geometry.splats->means.device(), Device::GPU);
        EXPECT_EQ(lfs::core::gpu_backend_of(result.geometry.splats->means), GetParam().backend);
        NodeTree points(registry_);
        points.add_node("lfs.mesh_to_points", "Points");
        ASSERT_TRUE(points.add_link({points.input_node().name, "Geometry", "Points", "Geometry"}));
        ASSERT_TRUE(points.add_link({"Points", "Geometry", points.output_node().name, "Geometry"}));
        const auto gpu_points = lfs::nodes::evaluate(points, {.geometry = geometry_from_mesh(mesh)});
        ASSERT_TRUE(gpu_points.ok);
        EXPECT_EQ(gpu_points.geometry.points->positions.device(), Device::GPU);

        Geometry cloud;
        cloud.points = PointsComponent{Tensor::zeros({3, 3}, Device::CPU),
                                       Tensor::full({3, 3}, 255, Device::CPU, lfs::core::DataType::UInt8),
                                       {}};
        NodeTree conversion(registry_);
        conversion.add_node("lfs.points_to_splats", "Convert").input_values["Radius"] = 0.1f;
        ASSERT_TRUE(conversion.add_link({conversion.input_node().name, "Geometry", "Convert", "Geometry"}));
        ASSERT_TRUE(conversion.add_link({"Convert", "Geometry", conversion.output_node().name, "Geometry"}));
        EvalCache cache;
        const auto converted = lfs::nodes::evaluate(conversion, {.geometry = cloud, .geometry_generation = 1}, nullptr, &cache);
        ASSERT_TRUE(converted.ok);
        EXPECT_EQ(converted.geometry.splats->means.device(), Device::GPU);
        EXPECT_NEAR((converted.geometry.splats->sh0 * 0.28209479177387814f + 0.5f).mean().item<float>(), 1.0f, 1e-5f);
        const auto repeated = lfs::nodes::evaluate(conversion, {.geometry = cloud, .geometry_generation = 1}, nullptr, &cache);
        ASSERT_TRUE(repeated.ok);
        EXPECT_TRUE(repeated.nodes.at(conversion.input_node().name).cached);
        EXPECT_EQ(converted.geometry.splats->means.data_ptr(), repeated.geometry.splats->means.data_ptr());
    }

    TEST_P(NodesCoreScale, SpatialOperationsStayInteractive) {
        if (device() == Device::CPU)
            GTEST_SKIP() << "Performance regression requires GPU";
        using lfs::core::DataType;
        constexpr size_t count = 1'000'000;
        const auto index = Tensor::arange(static_cast<float>(count)).to(device());
        const auto cluster = (index / 16).floor();
        const auto x = cluster - (cluster / 50).floor() * 50;
        const auto y = (cluster / 50).floor() - (cluster / 2500).floor() * 50;
        const auto z = (cluster / 2500).floor();
        Geometry geometry;
        geometry.splats = SplatsComponent{
            Tensor::stack({x, y, z}, 1) + Tensor::uniform({count, 3}, -0.01f, 0.01f, device(), DataType::Float32, 31),
            Tensor::zeros({count, 3}, device()),
            Tensor::zeros({count, 0, 3}, device()),
            Tensor::uniform({count, 1}, -11, 0, device(), DataType::Float32, 17).expand({static_cast<int>(count), 3}).contiguous(),
            Tensor::cat({Tensor::ones({count, 1}, device()), Tensor::zeros({count, 3}, device())}, 1),
            Tensor::ones({count}, device()),
            0,
            1,
            {}};
        geometry.splats->means.sum().item<float>();
        const auto measure = [&](const char* label, auto action) {
            const auto start = std::chrono::steady_clock::now();
            action();
            const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            std::cout << "[NodesSpatialPerformance " << GetParam().name << "] " << label << " " << elapsed << " ms\n";
            // Shared CI runners have no stable GPU time; budgets are opt-in like LFS_DECIMATE_PERF.
            if (const char* perf = std::getenv("LFS_NODES_PERF"); perf && std::string_view(perf) == "1")
                EXPECT_LT(elapsed, 2000.0) << label;
        };
        measure("Remove Floaters relative 1M", [&] {
            const auto result = single("lfs.remove_floaters", geometry, [](Node& node) { node.input_values["Min Neighbours"] = int64_t(3); });
            ASSERT_TRUE(result.ok);
            result.geometry.splats->means.sum().item<float>();
        });
        measure("Neighbour Count relative 1M", [&] {
            const auto value = field_result("lfs.neighbour_count", "Count", INT_SOCKET, geometry, [](Node& node) {
                node.input_values["Radius"] = 3.0f;
                node.properties["relative_to_size"] = true;
            });
            value.sum().item<int>();
        });
        for (const size_t points : {140000u, 200000u, 1000000u}) {
            Geometry cloud;
            cloud.points = PointsComponent{geometry.splats->means.slice(0, 0, points), Tensor::ones({points, 3}, device()), {}};
            const std::string label = "Points to Splats auto " + std::to_string(points);
            measure(label.c_str(), [&] {
                const auto result = single("lfs.points_to_splats", cloud);
                ASSERT_TRUE(result.ok);
                result.geometry.splats->scaling.sum().item<float>();
            });
        }
    }

    TEST_P(NodesCore, NestedGroupsEvaluateGeometryBitwise) {
        NodeTree inner(registry_, "Inner");
        NodeTree middle(registry_, "Middle");
        NodeTree outer(registry_, "Outer");
        const TreeResolver resolver = [&](std::string_view uuid) -> const NodeTree* {
            for (const auto* tree : {&inner, &middle, &outer})
                if (tree->uuid == uuid)
                    return tree;
            return nullptr;
        };
        for (const auto pair : {std::pair{&middle, &inner}, {&outer, &middle}}) {
            auto& owner = *pair.first;
            owner.add_node("lfs.group", "Instance").properties["tree"] = pair.second->uuid;
            ASSERT_TRUE(owner.add_link({owner.input_node().name, "Geometry", "Instance", "Geometry"}, nullptr, resolver));
            ASSERT_TRUE(owner.add_link({"Instance", "Geometry", owner.output_node().name, "Geometry"}, nullptr, resolver));
        }
        const auto geometry = splats();
        const auto result = evaluate(outer, {.geometry = geometry, .device = device(), .tree_resolver = resolver});
        ASSERT_TRUE(result.ok);
        const auto actual = result.geometry.splats->shN.to_vector();
        const auto expected = geometry.splats->shN.to_vector();
        ASSERT_EQ(actual.size(), expected.size());
        EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float)), 0);
        EXPECT_TRUE(result.nodes.contains("Instance/Instance/" + inner.output_node().name));
    }

    // Fails if a group's cached contents survive an edit upstream of the group or inside a group it contains:
    // interface geometry hashed only by type, and a group key covered only its own graph.
    TEST_P(NodesCore, GroupCachesFollowUpstreamAndNestedEdits) {
        NodeTree inner(registry_, "Inner");
        NodeTree middle(registry_, "Middle");
        NodeTree outer(registry_, "Outer");
        const TreeResolver resolver = [&](std::string_view uuid) -> const NodeTree* {
            for (const auto* tree : {&inner, &middle, &outer})
                if (tree->uuid == uuid)
                    return tree;
            return nullptr;
        };
        inner.add_node("lfs.transform_geometry", "Move").input_values["Translation"] = glm::vec3(0, 0, 0);
        ASSERT_TRUE(inner.add_link({inner.input_node().name, "Geometry", "Move", "Geometry"}));
        ASSERT_TRUE(inner.add_link({"Move", "Geometry", inner.output_node().name, "Geometry"}));
        middle.add_node("lfs.group", "Inner").properties["tree"] = inner.uuid;
        ASSERT_TRUE(middle.add_link({middle.input_node().name, "Geometry", "Inner", "Geometry"}, nullptr, resolver));
        ASSERT_TRUE(middle.add_link({"Inner", "Geometry", middle.output_node().name, "Geometry"}, nullptr, resolver));
        outer.add_node("lfs.transform_geometry", "Shift").input_values["Translation"] = glm::vec3(1, 0, 0);
        outer.add_node("lfs.group", "Middle").properties["tree"] = middle.uuid;
        ASSERT_TRUE(outer.add_link({outer.input_node().name, "Geometry", "Shift", "Geometry"}));
        ASSERT_TRUE(outer.add_link({"Shift", "Geometry", "Middle", "Geometry"}, nullptr, resolver));
        ASSERT_TRUE(outer.add_link({"Middle", "Geometry", outer.output_node().name, "Geometry"}, nullptr, resolver));
        EvalCache cache;
        const auto x = [&] {
            const auto result = evaluate(outer, {.geometry = splats(), .device = device(), .tree_resolver = resolver}, nullptr, &cache);
            EXPECT_TRUE(result.ok);
            return host<float>(result.geometry.splats->means)[0];
        };
        EXPECT_FLOAT_EQ(x(), 1.0f);
        outer.find_node("Shift")->input_values["Translation"] = glm::vec3(2, 0, 0);
        EXPECT_FLOAT_EQ(x(), 2.0f);
        inner.find_node("Move")->input_values["Translation"] = glm::vec3(3, 0, 0);
        EXPECT_FLOAT_EQ(x(), 5.0f);
    }

    // Fails if a muted group still runs its contents, or if nested graphs ignore cancellation.
    TEST_P(NodesCore, GroupsBypassWhenMutedAndStopWhenCancelled) {
        NodeTree inner(registry_, "Inner");
        NodeTree outer(registry_, "Outer");
        const TreeResolver resolver = [&](std::string_view uuid) -> const NodeTree* {
            return uuid == inner.uuid ? &inner : uuid == outer.uuid ? &outer
                                                                    : nullptr;
        };
        std::string previous = inner.input_node().name;
        for (int step = 0; step < 5; ++step) {
            const std::string name = "Move " + std::to_string(step);
            inner.add_node("lfs.transform_geometry", name).input_values["Translation"] = glm::vec3(1, 0, 0);
            ASSERT_TRUE(inner.add_link({previous, "Geometry", name, "Geometry"}));
            previous = name;
        }
        ASSERT_TRUE(inner.add_link({previous, "Geometry", inner.output_node().name, "Geometry"}));
        outer.add_node("lfs.group", "Group").properties["tree"] = inner.uuid;
        ASSERT_TRUE(outer.add_link({outer.input_node().name, "Geometry", "Group", "Geometry"}, nullptr, resolver));
        ASSERT_TRUE(outer.add_link({"Group", "Geometry", outer.output_node().name, "Geometry"}, nullptr, resolver));
        const auto run = [&](const EvalControl& control) {
            return evaluate(outer, {.geometry = splats(), .device = device(), .tree_resolver = resolver}, nullptr, nullptr, control);
        };
        const auto moved = run({});
        ASSERT_TRUE(moved.ok);
        EXPECT_FLOAT_EQ(host<float>(moved.geometry.splats->means)[0], 5.0f);
        outer.find_node("Group")->muted = true;
        const auto muted = run({});
        ASSERT_TRUE(muted.ok);
        EXPECT_FLOAT_EQ(host<float>(muted.geometry.splats->means)[0], 0.0f);
        outer.find_node("Group")->muted = false;
        int checks = 0;
        const auto cancelled = run({.cancelled = [&] { return ++checks > 4; }});
        EXPECT_TRUE(cancelled.cancelled);
        EXPECT_FALSE(cancelled.nodes.contains("Group/Move 4"));
    }

    // Fails if one cache serves stale geometry after a mute toggle, after a cancelled run, or once the graphs
    // are reloaded from JSON.
    TEST_P(NodesCore, GroupCachesStayCorrectAcrossMutingCancellationAndReload) {
        auto inner = std::make_unique<NodeTree>(registry_, "Inner");
        auto middle = std::make_unique<NodeTree>(registry_, "Middle");
        auto outer = std::make_unique<NodeTree>(registry_, "Outer");
        const TreeResolver resolver = [&](std::string_view uuid) -> const NodeTree* {
            for (const auto* tree : {inner.get(), middle.get(), outer.get()})
                if (tree->uuid == uuid)
                    return tree;
            return nullptr;
        };
        std::string previous = inner->input_node().name;
        for (int step = 0; step < 3; ++step) {
            const std::string name = "Move " + std::to_string(step);
            inner->add_node("lfs.transform_geometry", name).input_values["Translation"] = glm::vec3(1, 0, 0);
            ASSERT_TRUE(inner->add_link({previous, "Geometry", name, "Geometry"}));
            previous = name;
        }
        ASSERT_TRUE(inner->add_link({previous, "Geometry", inner->output_node().name, "Geometry"}));
        middle->add_node("lfs.group", "Inner").properties["tree"] = inner->uuid;
        ASSERT_TRUE(middle->add_link({middle->input_node().name, "Geometry", "Inner", "Geometry"}, nullptr, resolver));
        ASSERT_TRUE(middle->add_link({"Inner", "Geometry", middle->output_node().name, "Geometry"}, nullptr, resolver));
        outer->add_node("lfs.group", "Middle").properties["tree"] = middle->uuid;
        ASSERT_TRUE(outer->add_link({outer->input_node().name, "Geometry", "Middle", "Geometry"}, nullptr, resolver));
        ASSERT_TRUE(outer->add_link({"Middle", "Geometry", outer->output_node().name, "Geometry"}, nullptr, resolver));
        EvalCache cache;
        const auto run = [&](const EvalControl& control = {}) {
            return evaluate(*outer, {.geometry = splats(), .device = device(), .tree_resolver = resolver}, nullptr, &cache,
                            control);
        };
        const auto x = [&] {
            const auto result = run();
            EXPECT_TRUE(result.ok);
            return result.ok ? host<float>(result.geometry.splats->means)[0] : -1.0f;
        };
        EXPECT_FLOAT_EQ(x(), 3.0f);
        middle->find_node("Inner")->muted = true;
        EXPECT_FLOAT_EQ(x(), 0.0f);
        middle->find_node("Inner")->muted = false;
        EXPECT_FLOAT_EQ(x(), 3.0f);

        // A run cancelled inside the innermost graph leaves nothing that a later run reuses as complete.
        inner->find_node("Move 2")->input_values["Translation"] = glm::vec3(5, 0, 0);
        int checks = 0;
        EXPECT_TRUE(run({.cancelled = [&] { return ++checks > 2; }}).cancelled);
        EXPECT_FLOAT_EQ(x(), 7.0f);
        const auto unchanged = run();
        ASSERT_TRUE(unchanged.ok);
        EXPECT_TRUE(unchanged.nodes.at("Middle").cached);

        // The reloaded graphs evaluate like the originals, and edits to them reach the result.
        inner = std::make_unique<NodeTree>(NodeTree::from_json(inner->to_json(), registry_));
        middle = std::make_unique<NodeTree>(NodeTree::from_json(middle->to_json(), registry_));
        outer = std::make_unique<NodeTree>(NodeTree::from_json(outer->to_json(), registry_));
        EXPECT_FLOAT_EQ(x(), 7.0f);
        inner->find_node("Move 0")->input_values["Translation"] = glm::vec3(-1, 0, 0);
        EXPECT_FLOAT_EQ(x(), 5.0f);
    }

    TEST(NodesGraphEditing, ForcedJsonGroupCycleReportsNamedNodeError) {
        NodeTypeRegistry registry;
        register_builtin_nodes(registry);
        NodeTree a(registry, "A"), b(registry, "B");
        a.add_node("lfs.group", "Into B").properties["tree"] = b.uuid;
        b.add_node("lfs.group", "Into A").properties["tree"] = a.uuid;
        // JSON can come from outside the validated assignment path.
        auto json = a.to_json();
        json["links"] = nlohmann::json::array({{{"from_node", "Into B"}, {"from_socket", "Geometry"}, {"to_node", a.output_node().name}, {"to_socket", "Geometry"}}});
        a = NodeTree::from_json(json, registry);
        const TreeResolver resolver = [&](std::string_view uuid) -> const NodeTree* {
            return uuid == a.uuid ? &a : uuid == b.uuid ? &b
                                                        : nullptr;
        };
        const auto result = evaluate(a, {.tree_resolver = resolver});
        ASSERT_FALSE(result.ok);
        ASSERT_TRUE(result.errors.contains("Into B"));
        EXPECT_NE(result.errors.at("Into B").find("Group cycle"), std::string::npos);
        EXPECT_NE(result.errors.at("Into B").find("A"), std::string::npos);
        EXPECT_NE(result.errors.at("Into B").find("B"), std::string::npos);
    }

    TEST_P(NodesCore, MissingGroupReportsNodeErrorWithoutMutatingOtherBranches) {
        NodeTree tree(registry_);
        tree.add_node("lfs.group", "Missing").properties["tree"] = "deleted";
        const auto unchanged = tree.to_json();
        auto result = evaluate(tree, {.geometry = splats(), .device = device()});
        EXPECT_TRUE(result.ok); // Unreachable missing instances do not affect the output.
        EXPECT_EQ(tree.to_json(), unchanged);
        tree.links.push_back({"Missing", "Geometry", tree.output_node().name, "Geometry"});
        result = evaluate(tree, {.geometry = splats(), .device = device()});
        EXPECT_FALSE(result.ok);
        EXPECT_EQ(result.errors.at("Missing"), "Missing graph");
        EXPECT_TRUE(result.nodes.contains(tree.input_node().name));
    }

    TEST_P(NodesCore, ReroutePassesGeometryAndLazyFieldsBitwise) {
        NodeTree tree(registry_);
        tree.add_node("lfs.reroute", "Geometry Route");
        tree.add_node("lfs.reroute", "Field Route");
        tree.add_node("lfs.position", "Position");
        tree.add_node("lfs.set_position", "Set Position");
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Geometry Route", "Input"}));
        ASSERT_TRUE(tree.add_link({"Geometry Route", "Output", "Set Position", "Geometry"}));
        ASSERT_TRUE(tree.add_link({"Position", "Position", "Field Route", "Input"}));
        ASSERT_TRUE(tree.add_link({"Field Route", "Output", "Set Position", "Position"}));
        ASSERT_TRUE(tree.add_link({"Set Position", "Geometry", tree.output_node().name, "Geometry"}));
        const auto input = splats();
        const auto result = evaluate(tree, {.geometry = input, .device = device()});
        ASSERT_TRUE(result.ok);
        for (const auto pair : {std::pair{input.splats->means, result.geometry.splats->means},
                                {input.splats->shN, result.geometry.splats->shN}}) {
            const auto a = pair.first.to_vector(), b = pair.second.to_vector();
            ASSERT_EQ(a.size(), b.size());
            EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), 0);
        }
    }

    TEST(NodesGraphEditing, RerouteResolvesUpstreamTypeAndRejectsFloatToGeometry) {
        NodeTypeRegistry registry;
        register_builtin_nodes(registry);
        NodeTree tree(registry);
        tree.add_node("lfs.value", "Value");
        tree.add_node("lfs.reroute", "First");
        tree.add_node("lfs.reroute", "Second");
        EXPECT_EQ(effective_outputs(tree, *tree.find_node("First")).front().type, ANY_SOCKET);
        ASSERT_TRUE(tree.add_link({"Value", "Value", "First", "Input"}));
        ASSERT_TRUE(tree.add_link({"First", "Output", "Second", "Input"}));
        EXPECT_EQ(effective_outputs(tree, *tree.find_node("Second")).front().type, FLOAT_SOCKET);
        std::string error;
        EXPECT_FALSE(tree.add_link({"Second", "Output", tree.output_node().name, "Geometry"}, &error));
        EXPECT_FALSE(error.empty());
    }

    TEST_P(NodesCore, LayoutCardsHaveNoEvaluationStatusOrTime) {
        NodeTree tree(registry_);
        tree.add_node("lfs.frame", "Frame");
        tree.add_node("lfs.note", "Note");
        tree.add_node("lfs.reroute", "Route");
        ASSERT_TRUE(tree.add_link({tree.input_node().name, "Geometry", "Route", "Input"}));
        ASSERT_TRUE(tree.add_link({"Route", "Output", tree.output_node().name, "Geometry"}));
        const auto result = evaluate(tree, {.geometry = splats(), .device = device()});
        ASSERT_TRUE(result.ok);
        for (const auto* name : {"Frame", "Note", "Route"}) {
            EXPECT_FALSE(result.nodes.contains(name));
            EXPECT_FALSE(result.time_ms.contains(name));
            EXPECT_FALSE(result.errors.contains(name));
        }
    }

    TEST(NodesGraphEditing, GroupInterfaceDefaultsAndNumericLimitsReachEffectiveInputs) {
        NodeTypeRegistry registry;
        register_builtin_nodes(registry);
        NodeTree inner(registry), outer(registry);
        inner.group_interface.inputs.push_back({"Amount", "Strength", std::string(FLOAT_SOCKET), 0.25f, 0.0, 1.0, 0.1});
        inner.group_interface.outputs.push_back({"Value", "Value", std::string(FLOAT_SOCKET), 0.0f});
        ASSERT_TRUE(inner.add_link({inner.input_node().name, "Amount", inner.output_node().name, "Value"}));
        outer.add_node("lfs.group", "Group").properties["tree"] = inner.uuid;
        outer.group_interface.outputs.push_back({"Value", "Value", std::string(FLOAT_SOCKET), 0.0f});
        const TreeResolver resolver = [&](std::string_view uuid) -> const NodeTree* { return uuid == inner.uuid ? &inner : nullptr; };
        ASSERT_TRUE(outer.add_link({"Group", "Value", outer.output_node().name, "Value"}, nullptr, resolver));
        const auto inputs = effective_inputs(outer, *outer.find_node("Group"), resolver);
        EXPECT_EQ(inputs.back().label, "Strength");
        EXPECT_EQ(inputs.back().min, 0.0);
        EXPECT_EQ(inputs.back().max, 1.0);
        EXPECT_EQ(inputs.back().step, 0.1);
        auto result = evaluate(outer, {.tree_resolver = resolver});
        ASSERT_TRUE(result.ok);
        EXPECT_EQ(*result.output_values.at("Value").get_if<float>(), 0.25f);
        outer.find_node("Group")->input_values["Amount"] = 9.0f;
        result = evaluate(outer, {.tree_resolver = resolver});
        ASSERT_TRUE(result.ok);
        EXPECT_EQ(*result.output_values.at("Value").get_if<float>(), 1.0f);
        outer.find_node("Group")->input_values["Amount"] = -9.0f;
        result = evaluate(outer, {.tree_resolver = resolver});
        EXPECT_EQ(*result.output_values.at("Value").get_if<float>(), 0.0f);
    }

    TEST(NodesGraphEditing, GroupsReroutesMissingGraphsAndCycles) {
        NodeTypeRegistry registry;
        register_builtin_nodes(registry);
        NodeTree inner(registry, "Inner");
        NodeTree outer(registry, "Outer");
        ASSERT_TRUE(outer.remove_link({outer.input_node().name, "Geometry",
                                       outer.output_node().name, "Geometry"}));
        auto& group = outer.add_node("lfs.group", "Instance");
        group.properties["tree"] = inner.uuid;
        const TreeResolver resolver = [&](const std::string_view uuid) -> const NodeTree* {
            if (uuid == inner.uuid)
                return &inner;
            if (uuid == outer.uuid)
                return &outer;
            return nullptr;
        };
        ASSERT_TRUE(outer.add_link({outer.input_node().name, "Geometry", group.name, "Geometry"},
                                   nullptr, resolver));
        ASSERT_TRUE(outer.add_link({group.name, "Geometry", outer.output_node().name, "Geometry"},
                                   nullptr, resolver));
        auto result = lfs::nodes::evaluate(outer, {.tree_resolver = resolver});
        EXPECT_TRUE(result.ok);

        const auto second = outer.links.back();
        ASSERT_TRUE(outer.remove_link(second));
        auto& reroute = outer.add_node("lfs.reroute", "Route");
        ASSERT_TRUE(outer.add_link({group.name, "Geometry", reroute.name, "Input"}, nullptr,
                                   resolver));
        ASSERT_TRUE(outer.add_link({reroute.name, "Output", outer.output_node().name, "Geometry"},
                                   nullptr, resolver));
        EXPECT_EQ(effective_outputs(outer, reroute, resolver).front().type, GEOMETRY_SOCKET);
        EXPECT_TRUE(lfs::nodes::evaluate(outer, {.tree_resolver = resolver}).ok);

        group.properties["tree"] = "missing";
        result = lfs::nodes::evaluate(outer, {.tree_resolver = resolver});
        EXPECT_FALSE(result.ok);
        EXPECT_NE(result.errors.at(group.name).find("Missing graph"), std::string::npos);

        group.properties["tree"] = inner.uuid;
        auto& back = inner.add_node("lfs.group", "Back");
        back.properties["tree"] = outer.uuid;
        std::string cycle;
        EXPECT_TRUE(group_reference_would_cycle(outer, inner.uuid, resolver, &cycle));
        result = lfs::nodes::evaluate(outer, {.tree_resolver = resolver});
        EXPECT_FALSE(result.ok);
        EXPECT_NE(result.errors.at(group.name).find("Group cycle"), std::string::npos);
    }

    INSTANTIATE_TEST_SUITE_P(Backends, NodesCoreScale, testing::ValuesIn(test_targets()),
                             [](const auto& info) {
                                 return std::string(info.param.name);
                             });

    INSTANTIATE_TEST_SUITE_P(Backends, NodesCore, testing::ValuesIn(test_targets()), [](const auto& info) {
        return std::string(info.param.name);
    });

} // namespace

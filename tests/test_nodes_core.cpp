/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "core/base64.hpp"
#include "core/nodes/nodes.hpp"
#include "core/tensor_backend.hpp"
#include "core/tensor_spatial.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <iostream>
#include <numbers>
#include <optional>
#include <random>

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
                            Geometry geometry, std::function<void(Node&)> configure = {}) {
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
            const auto result = evaluate(tree, {geometry, {}, 1});
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
        TreeTypeRegistry trees;
        SocketTypeRegistry sockets;
        NodeTypeRegistry nodes;
        register_builtin_nodes(nodes);
        ASSERT_EQ(trees.list().size(), 1u);
        EXPECT_EQ(trees.list()[0].id, "lfs.geometry");
        EXPECT_EQ(sockets.list().size(), 7u);
        EXPECT_GE(nodes.list().size(), 45u);
        EXPECT_FALSE(nodes.find("lfs.object_info"));
        EXPECT_TRUE(nodes.find("lfs.mesh_to_splats"));
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
        const std::vector<float> expected_midpoint{-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f, 1, 1, 1};
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

    TEST_P(NodesCore, CorePayloadConversionsDropDeletedRowsAndNormalizeU8Colours) {
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

        Tensor colors = Tensor::empty({2, 3}, Device::CPU, lfs::core::DataType::UInt8);
        const std::uint8_t bytes[] = {0, 127, 255, 255, 64, 0};
        std::memcpy(colors.data_ptr(), bytes, sizeof(bytes));
        if (device() == Device::GPU)
            colors = colors.to(device());
        lfs::core::PointCloud cloud(tensor({0, 0, 0, 1, 0, 0}, {2, 3}), colors);
        Geometry points = geometry_from_point_cloud(cloud);
        const auto normalized = host<float>(points.points->colors);
        EXPECT_FLOAT_EQ(normalized[0], 0.0f);
        EXPECT_NEAR(normalized[1], 127.0f / 255.0f, 1e-6f);
        EXPECT_FLOAT_EQ(normalized[2], 1.0f);
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
                builtin->id == "lfs.group_output")
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
            EXPECT_TRUE(result.ok) << (result.errors.empty() ? "no error text"
                                                             : result.errors.begin()->second);
        }
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
        EXPECT_FLOAT_EQ(*socket("lfs.scale_clamp", "Max Aspect").default_value.get_if<float>(), 10.0f);
        EXPECT_EQ(socket("lfs.colour_correct", "Midpoint").min, 0.01);
        EXPECT_EQ(socket("lfs.colour_correct", "Midpoint").max, 0.99);
        EXPECT_EQ(socket("lfs.recolour", "Weight").min, 0.0);
        EXPECT_EQ(socket("lfs.recolour", "Weight").max, 1.0);
        EXPECT_EQ(socket("lfs.decimate", "Keep Fraction").min, 0.001);
        EXPECT_EQ(socket("lfs.decimate", "Keep Fraction").max, 1.0);
        EXPECT_FALSE(property("lfs.colour_correct", "auto_range").default_value.get<bool>());
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

    TEST_P(NodesCore, ScaleClampUsesMidrangeAndSelectionWeight) {
        auto geometry = splats();
        geometry.splats->scaling = tensor({-3, 0, 3, -3, 0, 3, -3, 0, 3}, {3, 3});
        auto result = single("lfs.scale_clamp", geometry, [](Node& node) {
            node.input_values["Max Aspect"] = std::exp(1.0f);
            node.input_values["Selection"] = 0.5f;
        });
        ASSERT_TRUE(result.ok);
        const auto values = host<float>(result.geometry.splats->scaling);
        for (size_t index = 0; index < values.size(); index += 3) {
            EXPECT_NEAR(values[index], -1.75f, 1e-5f);
            EXPECT_NEAR(values[index + 1], 0, 1e-5f);
            EXPECT_NEAR(values[index + 2], 1.75f, 1e-5f);
        }
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
        EXPECT_THROW(lfs::core::radius_neighbor_counts(points, references, 1.0f, 0), std::exception);
        const auto empty = lfs::core::radius_neighbor_counts(Tensor::empty({0, 3}, device()), Tensor::full_bool({0}, true, device()), 1.0f, 3);
        EXPECT_EQ(empty.numel(), 0u);
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

    TEST_P(NodesCore, ScaleClampGuaranteesAspectOnRandomLogScales) {
        std::mt19937 random(27);
        std::uniform_real_distribution<float> value(-15, 10);
        for (int sample = 0; sample < 16; ++sample) {
            auto geometry = splats();
            std::vector<float> scales(9);
            std::generate(scales.begin(), scales.end(), [&] { return value(random); });
            geometry.splats->scaling = tensor(scales, {3, 3});
            const auto result = single("lfs.scale_clamp", geometry, [](Node& node) { node.input_values["Max Aspect"] = 4.0f; });
            ASSERT_TRUE(result.ok);
            EXPECT_LE((result.geometry.splats->scaling.max(1) - result.geometry.splats->scaling.min(1)).exp().max().item<float>(), 4.00001f);
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

    class NodesCoreScale : public NodesCore {};

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

    INSTANTIATE_TEST_SUITE_P(Backends, NodesCoreScale, testing::ValuesIn(test_targets()),
                             [](const auto& info) {
                                 return std::string(info.param.name);
                             });

    INSTANTIATE_TEST_SUITE_P(Backends, NodesCore, testing::ValuesIn(test_targets()), [](const auto& info) {
        return std::string(info.param.name);
    });

} // namespace

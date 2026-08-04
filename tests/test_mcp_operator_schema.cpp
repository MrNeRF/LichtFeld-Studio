/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */

#include "app/mcp_operator_schema.hpp"
#include "core/property_registry.hpp"
#include "visualizer/operator/operator_properties.hpp"

#include <glm/glm.hpp>
#include <gtest/gtest.h>

namespace {

    struct OperatorArgsCleanup {
        std::string id;
        ~OperatorArgsCleanup() {
            lfs::core::prop::PropertyRegistry::instance().unregister_operator_args(id);
        }
    };

    nlohmann::json input_schema_json(const lfs::mcp::McpToolInputSchema& schema) {
        nlohmann::json result{
            {"type", schema.type},
            {"properties", schema.properties},
        };
        if (!schema.required.empty()) {
            result["required"] = schema.required;
        }
        return result;
    }

} // namespace

TEST(McpOperatorSchemaTest, EmitsExactJsonForEveryPropType) {
    using namespace lfs::core::prop;

    std::vector<PropertyMeta> properties;
    properties.push_back(operator_arg("bool_value", "", PropType::Bool));

    auto int_value = operator_arg("int_value", "", PropType::Int);
    int_value.min_value = -5.0;
    int_value.max_value = 5.0;
    properties.push_back(std::move(int_value));

    properties.push_back(operator_arg("float_value", "", PropType::Float));
    auto text = operator_arg("text", "Text value", PropType::String);
    text.name = "Display-only text name";
    properties.push_back(std::move(text));
    properties.push_back(arg_enum(
        "mode", "",
        {
            {.name = "Replace", .identifier = "replace", .value = 0},
            {.name = "Add", .identifier = "add", .value = 1},
        }));
    properties.push_back(operator_arg("size_value", "", PropType::SizeT));
    properties.push_back(operator_arg("vec2", "", PropType::Vec2));

    auto vec3 = operator_arg("vec3", "", PropType::Vec3);
    vec3.min_value = 0.0;
    vec3.max_value = 1.0;
    properties.push_back(std::move(vec3));

    properties.push_back(operator_arg("vec4", "", PropType::Vec4));
    properties.push_back(operator_arg("quat", "", PropType::Quat));
    properties.push_back(operator_arg("mat4", "", PropType::Mat4));
    properties.push_back(operator_arg("color3", "", PropType::Color3));
    properties.push_back(operator_arg("color4", "", PropType::Color4));
    properties.push_back(arg_tensor("tensor", ""));
    properties.push_back(arg_float_vector("float_vector", "", 3));
    properties.push_back(arg_int_vector("int_vector", "", 4));

    auto& registry = PropertyRegistry::instance();
    registry.register_operator_args("test.all_types", std::move(properties));
    OperatorArgsCleanup cleanup{"test.all_types"};

    const auto actual = input_schema_json(
        lfs::app::build_operator_input_schema("test.all_types", {"text"}));
    const auto expected = nlohmann::json::parse(R"json(
        {
          "type": "object",
          "required": ["text"],
          "properties": {
            "bool_value": {"type": "boolean"},
            "int_value": {"type": "integer", "minimum": -5.0, "maximum": 5.0},
            "float_value": {"type": "number"},
            "text": {"type": "string", "description": "Text value"},
            "mode": {"type": "string", "enum": ["replace", "add"]},
            "size_value": {"type": "integer"},
            "vec2": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
            "vec3": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "minimum": 0.0, "maximum": 1.0},
            "vec4": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            "quat": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            "mat4": {"type": "array", "items": {"type": "number"}, "minItems": 16, "maxItems": 16},
            "color3": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "color4": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            "tensor": {"type": "array"},
            "float_vector": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            "int_vector": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
          }
        }
    )json");

    EXPECT_EQ(actual, expected);
}

TEST(McpOperatorSchemaTest, FiltersUnflaggedPropertyMetas) {
    using namespace lfs::core::prop;

    auto hidden = operator_arg("hidden", "", PropType::String);
    hidden.flags = PROP_NONE;
    PropertyRegistry::instance().register_group(PropertyGroup{
        .id = "operator.test.flag_filter",
        .name = "test.flag_filter",
        .properties = {
            arg_string("visible", ""),
            std::move(hidden),
        },
    });
    OperatorArgsCleanup cleanup{"test.flag_filter"};

    const auto actual = input_schema_json(
        lfs::app::build_operator_input_schema("test.flag_filter", {}));
    const auto expected = nlohmann::json::parse(R"json(
        {
          "type": "object",
          "properties": {
            "visible": {"type": "string"}
          }
        }
    )json");

    EXPECT_EQ(actual, expected);
}

TEST(McpOperatorSchemaTest, PreservesDeserializationErrorsAndVectorStorage) {
    using namespace lfs::core::prop;

    auto& registry = PropertyRegistry::instance();
    registry.register_operator_args(
        "test.deserialize",
        {
            arg_bool("enabled", ""),
            arg_string("label", ""),
            arg_float("scale", ""),
            arg_int("count", ""),
            arg_float_vector("point", "", 3),
            arg_float_vector("weights", "", 2),
            arg_int_vector("indices", "", 2),
            arg_tensor("tensor", ""),
        });
    OperatorArgsCleanup cleanup{"test.deserialize"};

    lfs::vis::op::OperatorProperties props;
    auto result = lfs::app::populate_operator_props(
        nlohmann::json::array(), "test.deserialize", {}, props);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Tool arguments must be a JSON object");

    result = lfs::app::populate_operator_props(
        nlohmann::json::object(), "test.deserialize", {"label"}, props);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Field 'label' must be provided");

    result = lfs::app::populate_operator_props(
        nlohmann::json{{"enabled", 1}}, "test.deserialize", {}, props);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Field 'enabled' must be a boolean");

    result = lfs::app::populate_operator_props(
        nlohmann::json{{"label", 3}}, "test.deserialize", {}, props);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Field 'label' must be a string");

    result = lfs::app::populate_operator_props(
        nlohmann::json{{"scale", "wide"}}, "test.deserialize", {}, props);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Field 'scale' must be a number");

    result = lfs::app::populate_operator_props(
        nlohmann::json{{"count", 1.5}}, "test.deserialize", {}, props);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Field 'count' must be an integer");

    result = lfs::app::populate_operator_props(
        nlohmann::json{{"point", 7}}, "test.deserialize", {}, props);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Field 'point' must be an array");

    result = lfs::app::populate_operator_props(
        nlohmann::json{{"point", nlohmann::json::array({1.0, 2.0})}},
        "test.deserialize", {}, props);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Field 'point' must have exactly 3 entries");

    result = lfs::app::populate_operator_props(
        nlohmann::json{{"weights", nlohmann::json::array({1.0, "bad"})}},
        "test.deserialize", {}, props);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Field 'weights' must contain only numbers");

    result = lfs::app::populate_operator_props(
        nlohmann::json{{"indices", nlohmann::json::array({1, 2.0})}},
        "test.deserialize", {}, props);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Field 'indices' must contain only integers");

    result = lfs::app::populate_operator_props(
        nlohmann::json{{"tensor", nlohmann::json::array()}},
        "test.deserialize", {}, props);
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error(), "Field 'tensor' is not supported through MCP yet");

    result = lfs::app::populate_operator_props(
        nlohmann::json{{"point", nlohmann::json::array({1.0, 2.0, 3.0})},
                       {"weights", nlohmann::json::array({0.25, 0.75})},
                       {"indices", nlohmann::json::array({4, 7})}},
        "test.deserialize", {}, props);
    ASSERT_TRUE(result.has_value());
    const auto point = props.get<glm::vec3>("point");
    const auto weights = props.get<std::vector<float>>("weights");
    const auto indices = props.get<std::vector<int>>("indices");
    ASSERT_TRUE(point.has_value());
    ASSERT_TRUE(weights.has_value());
    ASSERT_TRUE(indices.has_value());
    EXPECT_EQ(*point, glm::vec3(1.0f, 2.0f, 3.0f));
    EXPECT_EQ(*weights, (std::vector<float>{0.25f, 0.75f}));
    EXPECT_EQ(*indices, (std::vector<int>{4, 7}));
}

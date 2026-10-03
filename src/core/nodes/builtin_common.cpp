/* SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
 * SPDX-License-Identifier: GPL-3.0-or-later */
#include "builtin_common.hpp"

namespace lfs::nodes::builtin {
    std::string property_string(const NodeContext& context, std::string_view name, std::string fallback) {
        const auto found = context.properties().find(std::string(name));
        if (found != context.properties().end() && found->is_string())
            return found->get<std::string>();
        return fallback;
    }

    float property_float(const NodeContext& context, std::string_view name, float fallback) {
        const auto found = context.properties().find(std::string(name));
        if (found != context.properties().end() && found->is_number())
            return found->get<float>();
        return fallback;
    }

    int property_int(const NodeContext& context, std::string_view name, int fallback) {
        return static_cast<int>(property_float(context, name, static_cast<float>(fallback)));
    }

    bool property_bool(const NodeContext& context, std::string_view name, bool fallback) {
        const auto found = context.properties().find(std::string(name));
        if (found != context.properties().end() && found->is_boolean())
            return found->get<bool>();
        return fallback;
    }

    float input_float(const NodeContext& context, std::string_view name, float fallback) {
        if (const auto* value = context.input(name).get_if<float>())
            return *value;
        if (const auto* value = context.input(name).get_if<std::int64_t>())
            return static_cast<float>(*value);
        return fallback;
    }

    int input_int(const NodeContext& context, std::string_view name, int fallback) {
        return static_cast<int>(input_float(context, name, static_cast<float>(fallback)));
    }

    Geometry geometry_input(const NodeContext& context, std::string_view name) {
        if (const auto* value = context.input(name).get_if<Geometry>())
            return *value;
        return {};
    }

    FieldContext field_context(const SplatsComponent& component) {
        return {Domain::Splat, &component, nullptr, nullptr,
                static_cast<std::uint64_t>(component.means.debug_id()) ^ (component.sh0.debug_id() << 8U) ^
                    (component.scaling.debug_id() << 16U) ^ (component.opacity.debug_id() << 24U) ^
                    (component.shN.debug_id() << 32U)};
    }

    FieldContext field_context(const PointsComponent& component) {
        return {Domain::Point, nullptr, &component, nullptr,
                static_cast<std::uint64_t>(component.positions.debug_id()) ^
                    (component.colors.debug_id() << 32U)};
    }

    FieldContext field_context(const MeshComponent& component) {
        return {Domain::Vertex, nullptr, nullptr, &component, component.mesh ? component.mesh->id() : 0};
    }

    core::Tensor selection(const NodeContext& context, std::string_view socket, const FieldContext& domain,
                           bool structural) {
        core::Tensor value = context.evaluate_field(socket, domain, FLOAT_SOCKET);
        return structural ? value.ge(0.5f) : value.clamp(0.0f, 1.0f);
    }

    core::Tensor blend(const core::Tensor& old_value, const core::Tensor& new_value, core::Tensor weight) {
        while (weight.ndim() < old_value.ndim())
            weight = weight.unsqueeze(-1);
        return old_value * (weight.neg() + 1.0f) + new_value * weight;
    }

    template <typename Component>
    AttributeMap filter_attributes(const Component& component, const core::Tensor& keep) {
        AttributeMap result;
        for (const auto& [name, value] : component.attributes)
            result.emplace(name, value.index_select(0, keep).contiguous());
        return result;
    }

    SplatsComponent filter_splats(const SplatsComponent& source, const core::Tensor& mask) {
        const Tensor keep =
            mask.dtype() == DataType::Bool ? mask.nonzero().reshape({-1}).to(DataType::Int32) : mask;
        return {source.means.index_select(0, keep).contiguous(),
                source.sh0.index_select(0, keep).contiguous(),
                source.shN.index_select(0, keep).contiguous(),
                source.scaling.index_select(0, keep).contiguous(),
                source.rotation.index_select(0, keep).contiguous(),
                source.opacity.index_select(0, keep).contiguous(),
                source.sh_degree,
                source.scene_scale,
                filter_attributes(source, keep)};
    }

    PointsComponent filter_points(const PointsComponent& source, const core::Tensor& mask) {
        const Tensor keep =
            mask.dtype() == DataType::Bool ? mask.nonzero().reshape({-1}).to(DataType::Int32) : mask;
        return {source.positions.index_select(0, keep).contiguous(),
                source.colors.index_select(0, keep).contiguous(), filter_attributes(source, keep)};
    }

    std::shared_ptr<core::MeshData> copy_mesh(const core::MeshData& source, core::Tensor vertices,
                                              core::Tensor indices) {
        auto mesh = std::make_shared<core::MeshData>();
        mesh->vertices = std::move(vertices);
        mesh->indices = std::move(indices);
        mesh->normals = source.normals;
        mesh->tangents = source.tangents;
        mesh->texcoords = source.texcoords;
        mesh->colors = source.colors;
        mesh->materials = source.materials;
        mesh->submeshes = source.submeshes;
        mesh->texture_images = source.texture_images;
        return mesh;
    }

    glm::vec3 input_vector(const NodeContext& context, std::string_view name) {
        if (const auto* value = context.input(name).get_if<glm::vec3>())
            return *value;
        if (const auto* value = context.input(name).get_if<glm::vec4>())
            return glm::vec3(*value);
        return glm::vec3(0);
    }

    Tensor vector_tensor(glm::vec3 value, Device device) {
        return Tensor::from_vector({value.x, value.y, value.z}, {1, 3}, Device::CPU).to(device);
    }

    Tensor matrix_tensor(const glm::mat3& value, Device device) {
        // GLM columns become rows: row-vector tensor multiplication uses A transposed.
        std::vector<float> rows;
        for (int column = 0; column < 3; ++column)
            for (int row = 0; row < 3; ++row)
                rows.push_back(value[column][row]);
        return Tensor::from_vector(rows, {3, 3}, Device::CPU).to(device);
    }

    glm::mat4 rotation_matrix(glm::vec3 degrees) {
        const auto angle = glm::radians(degrees);
        const auto x = glm::rotate(glm::mat4(1), angle.x, glm::vec3(1, 0, 0));
        const auto y = glm::rotate(glm::mat4(1), angle.y, glm::vec3(0, 1, 0));
        const auto z = glm::rotate(glm::mat4(1), angle.z, glm::vec3(0, 0, 1));
        return z * y * x;
    }

    Tensor safe_divide(const Tensor& a, const Tensor& b) {
        return Tensor::where(b.eq(0), Tensor::zeros_like(a), a / b);
    }

    Tensor channel(const Tensor& value, int axis) {
        return value.slice(1, axis, axis + 1).squeeze(1);
    }

    Field operation(std::string_view type, std::vector<Field> inputs,
                    std::function<Tensor(const std::vector<Tensor>&)> evaluate) {
        const bool dependent = std::ranges::any_of(inputs, &Field::context_dependent);
        return Field(
            std::string(type),
            [inputs = std::move(inputs), evaluate = std::move(evaluate)](const FieldContext& context,
                                                                         FieldMemo& memo) {
                std::vector<Tensor> values;
                values.reserve(inputs.size());
                for (const auto& input : inputs)
                    values.push_back(input.evaluate(context, memo));
                return evaluate(values);
            },
            dependent);
    }

    Tensor rgb_to_hsv(const Tensor& rgb) {
        const auto maximum = rgb.max(1);
        const auto delta = maximum - rgb.min(1);
        const auto r = channel(rgb, 0);
        const auto g = channel(rgb, 1);
        const auto b = channel(rgb, 2);
        const auto red = safe_divide(g - b, delta);
        const auto green = safe_divide(b - r, delta) + 2;
        const auto blue = safe_divide(r - g, delta) + 4;
        auto hue = Tensor::where(maximum.eq(r), red, Tensor::where(maximum.eq(g), green, blue)) / 6;
        hue = hue - hue.floor();
        hue = Tensor::where(delta.eq(0), Tensor::zeros_like(hue), hue);
        return Tensor::stack({hue, safe_divide(delta, maximum), maximum}, 1);
    }

    Tensor hsv_to_rgb(const Tensor& hsv) {
        const auto h = channel(hsv, 0);
        const auto s = channel(hsv, 1).clamp(0, 1);
        const auto v = channel(hsv, 2);
        const auto ramp = [&](float offset) {
            const auto phase = h + offset;
            return ((phase - phase.floor()) * 6 - 3).abs().sub(1).clamp(0, 1);
        };
        const auto red = v * (s.neg() + 1 + s * ramp(0));
        const auto green = v * (s.neg() + 1 + s * ramp(2.0f / 3));
        const auto blue = v * (s.neg() + 1 + s * ramp(1.0f / 3));
        return Tensor::stack({red, green, blue}, 1);
    }

    SocketDecl in(std::string id, std::string type, Value value, bool field, bool multi) {
        return {id, id, std::move(type), std::move(value), {}, {}, {}, field, multi, false};
    }

    SocketDecl out(std::string id, std::string type) {
        auto socket = in(id, std::move(type), {}, false, false);
        socket.hide_value = true;
        return socket;
    }

    PropertyDecl prop(std::string id, PropertyKind kind, nlohmann::json value,
                      std::vector<std::string> items) {
        return {id, id, kind, std::move(value), std::move(items), {}, {}};
    }

    std::vector<SocketDecl> geometry_inputs(std::vector<SocketDecl> extra) {
        extra.insert(extra.begin(), in("Geometry", std::string(GEOMETRY_SOCKET)));
        return extra;
    }

    NodeTypeInfo type(std::string id, std::string label, std::string category, std::string description,
                      std::vector<SocketDecl> inputs, std::vector<SocketDecl> outputs,
                      std::function<void(NodeContext&)> evaluate, std::vector<PropertyDecl> properties) {
        NodeTypeInfo result;
        result.id = std::move(id);
        result.label = std::move(label);
        result.category = std::move(category);
        result.description = std::move(description);
        result.inputs = std::move(inputs);
        result.outputs = std::move(outputs);
        result.properties = std::move(properties);
        result.evaluate = std::move(evaluate);
        return result;
    }

    void register_type(NodeTypeRegistry& registry, NodeTypeInfo info) {
        registry.register_type(std::move(info));
    }

} // namespace lfs::nodes::builtin
